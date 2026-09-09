"""
Adapter for the CUNQA backend circuit.

This module implements the ``Circuit`` interface for CUNQA circuits.

Local operations are translated one by one, exactly as the abstract
:class:`~netqmpi.sdk.circuit.Circuit` dispatch expects. Collective ones
cannot: CUNQA's telegate helpers (:func:`cunqa.qc_protocols.cat_entangler`
and :func:`cunqa.qc_protocols.cat_disentangler`) write instructions into
*every* participating circuit in a single call, so they can only run once
all the ranks are known and each of them has been translated up to the
matching call. :func:`translate_group` performs that joint pass, walking
every rank's operation stream and stopping at the collective calls to
expand them in one go.

The rooted transfers (``qscatter``, ``qgather``) sit in between: they are
collective for the user, since every rank has to call them, but each rank's
half is a sequence of ordinary teledata blocks that CUNQA pairs up by tag
at run time, so they need no joint expansion.
"""
import os, sys
sys.path.append(os.getenv("HOME"))

from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from cunqa.circuit import CunqaCircuit
from cunqa.qc_protocols import qsend, qrecv, cat_entangler, cat_disentangler

from netqmpi.sdk.circuit import Circuit
from netqmpi.sdk.operations import (
    Operation,
    Gate, ControlledGate, ClassicalControlledGate,
    Measure, Reset, Barrier,
    OperationContainer,
    CollectiveOperation,
    QSend, QRecv, QScatter, QGather, Expose, Unexpose,
)
from netqmpi.runtime.adapters.cunqa.cunqa_communicator import CunqaCommunicator

#: Name of the classical register holding the correction bits of the
#: distributed protocols, kept apart from the user's own register.
PROTOCOL_CLREG = "netqmpi_protocol"


class CunqaCircuitAdapter(Circuit):
    """
    Circuit adapter for the CUNQA backend.

    This class wraps a ``CunqaCircuit`` instance and exposes the common
    interface defined by the abstract ``Circuit`` base class.
    """

    def __init__(
        self,
        num_qubits: int,
        num_clbits: int,
        comm: CunqaCommunicator
    ):
        """
        Initialize the CUNQA circuit adapter.

        The underlying ``CunqaCircuit`` starts with no communication qubit
        and no protocol classical bits: how many are needed only becomes
        known once the application has been traced, so they are added by
        :meth:`prepare` right before the instructions are emitted.

        Args:
            num_qubits: Number of qubits in the circuit.
            num_clbits: Number of classical bits in the circuit.
            comm: Communicator associated with the circuit.
        """
        super().__init__(num_qubits, num_clbits, comm)
        self._cunqa_circuit = CunqaCircuit(
            (num_qubits, 0), num_clbits, id=f"rank_{self._comm.rank}")
        self._protocol_clreg: Optional[str] = None
        self._prepared: bool = False

    # ------------------------------------------------------------------
    # Backend resources
    # ------------------------------------------------------------------

    @property
    def cunqa_circuit(self) -> CunqaCircuit:
        """
        Return the underlying CUNQA circuit.

        Returns:
            The wrapped ``CunqaCircuit``.
        """
        return self._cunqa_circuit

    def prepare(self) -> None:
        """
        Reserve on the CUNQA circuit the resources the trace asked for.

        Adds the communication qubits and the protocol classical register
        sized by the trace. Must run before any instruction is emitted,
        since the protocol register has to sit after the user's own bits.

        Calling it twice is a no-op: a second call would append a further
        register and quietly renumber the protocol bits.
        """
        if self._prepared:
            return
        self._prepared = True

        if self.num_comm_qubits:
            self._cunqa_circuit.add_comm_qubits(self.num_comm_qubits)
        if self.num_protocol_clbits:
            self._protocol_clreg = self._cunqa_circuit.add_cl_register(
                PROTOCOL_CLREG, self.num_protocol_clbits)

    def _q(self, qubit: int) -> int:
        """
        Map a NetQMPI qubit index onto its CUNQA counterpart.

        Args:
            qubit: Data qubit index, or a communication qubit as returned
                by :meth:`~netqmpi.sdk.circuit.Circuit.comm_qubit`.

        Returns:
            The qubit index in the CUNQA circuit.
        """
        if qubit < self.num_qubits:
            return qubit
        return self._cunqa_circuit.comm_qubits[qubit - self.num_qubits]

    def _cl(self, clbit: int) -> int:
        """
        Map a protocol classical bit onto its CUNQA counterpart.

        Args:
            clbit: Protocol classical bit reserved during the trace.

        Returns:
            The classical bit index in the CUNQA circuit.
        """
        return self._cunqa_circuit.classical_regs[self._protocol_clreg][clbit]

    def comm_qubit_of(self, slot: int) -> int:
        """
        Return the CUNQA index of one of this circuit's comm-qubit slots.

        Args:
            slot: Communication-qubit slot reserved by a protocol.

        Returns:
            The communication qubit index in the CUNQA circuit.
        """
        return self._cunqa_circuit.comm_qubits[slot]

    # ------------------------------------------------------------------
    # Circuit abstract interface
    # ------------------------------------------------------------------

    def _translate_gate(self, op: Gate):
        """
        Translate a single-qubit unitary gate into a CUNQA instruction.

        Args:
            op: Gate operation to translate.

        Raises:
            NotImplementedError: If the gate has no CUNQA counterpart here.
        """
        qubits = [self._q(q) for q in op.qubits]

        gate_map = {
            # No params
            "H":    lambda: self._cunqa_circuit.h(*qubits),
            "X":    lambda: self._cunqa_circuit.x(*qubits),
            "Y":    lambda: self._cunqa_circuit.y(*qubits),
            "Z":    lambda: self._cunqa_circuit.z(*qubits),
            "S":    lambda: self._cunqa_circuit.s(*qubits),
            "SDG":  lambda: self._cunqa_circuit.sdg(*qubits),
            "T":    lambda: self._cunqa_circuit.t(*qubits),
            "TDG":  lambda: self._cunqa_circuit.tdg(*qubits),
            "SWAP": lambda: self._cunqa_circuit.swap(*qubits),

            # One param
            "RX": lambda: self._cunqa_circuit.rx(op.params[0], *qubits),
            "RY": lambda: self._cunqa_circuit.ry(op.params[0], *qubits),
            "RZ": lambda: self._cunqa_circuit.rz(op.params[0], *qubits),
            "P":  lambda: self._cunqa_circuit.p(op.params[0], *qubits),
        }

        if op.name not in gate_map:
            raise NotImplementedError(
                f"Gate '{op.name}' is not implemented for the CUNQA backend.")
        gate_map[op.name]()

    def _translate_controlled_gate(self, op: ControlledGate):
        """
        Translate a controlled quantum gate into a CUNQA instruction.

        Args:
            op: Controlled gate operation to translate.

        Raises:
            NotImplementedError: If the controlled gate has no CUNQA
                counterpart here.
        """
        if len(op.targets) != 1:
            raise NotImplementedError(
                "Only single-target controlled gates are supported by the "
                "CUNQA backend.")

        target = op.targets[0]
        name = target.name
        qubits = [self._q(q) for q in op.controls] + [self._q(q) for q in target.qubits]
        n_controls = len(op.controls)

        if n_controls == 1:
            gate_map = {
                # No params
                "X":    lambda: self._cunqa_circuit.cx(*qubits),
                "Y":    lambda: self._cunqa_circuit.cy(*qubits),
                "Z":    lambda: self._cunqa_circuit.cz(*qubits),
                "H":    lambda: self._cunqa_circuit.ch(*qubits),
                "S":    lambda: self._cunqa_circuit.cs(*qubits),
                "SDG":  lambda: self._cunqa_circuit.csdg(*qubits),
                "T":    lambda: self._cunqa_circuit.ct(*qubits),
                "SX":   lambda: self._cunqa_circuit.csx(*qubits),
                "SWAP": lambda: self._cunqa_circuit.cswap(*qubits),

                # One param
                "RX": lambda: self._cunqa_circuit.crx(target.params[0], *qubits),
                "RY": lambda: self._cunqa_circuit.cry(target.params[0], *qubits),
                "RZ": lambda: self._cunqa_circuit.crz(target.params[0], *qubits),
                "P":  lambda: self._cunqa_circuit.cp(target.params[0], *qubits),
            }
        elif n_controls == 2:
            gate_map = {
                "X": lambda: self._cunqa_circuit.ccx(*qubits),
                "Y": lambda: self._cunqa_circuit.ccy(*qubits),
                "Z": lambda: self._cunqa_circuit.ccz(*qubits),
            }
        else:
            gate_map = {
                "X": lambda: self._cunqa_circuit.mcx(*qubits),
                "Y": lambda: self._cunqa_circuit.mcy(*qubits),
                "Z": lambda: self._cunqa_circuit.mcz(*qubits),
            }

        if name not in gate_map:
            raise NotImplementedError(
                f"Controlled gate '{name}' with {n_controls} control(s) is not "
                f"implemented for the CUNQA backend.")
        gate_map[name]()

    def _translate_classical_controlled_gate(self, op: ClassicalControlledGate):
        """
        Translate a classically controlled gate into a CUNQA instruction.

        Args:
            op: Classically controlled gate operation to translate.
        """
        self._cunqa_circuit.cif(op.cbits)
        for target in op.targets:
            self._translate_gate(target)
        self._cunqa_circuit.endcif()

    def _translate_measure(self, op: Measure):
        """
        Translate a measurement operation into a CUNQA instruction.

        Args:
            op: Measurement operation to translate.
        """
        self._cunqa_circuit.measure(self._q(op.qubit), op.cbit)

    def _translate_reset(self, op: Reset):
        """
        Translate a reset operation into a CUNQA instruction.

        Args:
            op: Reset operation to translate.
        """
        self._cunqa_circuit.reset(self._q(op.qubit))

    def _translate_barrier(self, op: Barrier):
        """
        Translate a barrier operation into a CUNQA instruction.

        Args:
            op: Barrier operation to translate.

        Raises:
            NotImplementedError: Always, because this operation is not yet supported.
        """
        raise NotImplementedError("Barrier is not implemented for the CUNQA backend.")

    def _translate_operation_container(self, op: OperationContainer):
        """
        Translate an operation container by recursively translating its children.

        Children are dispatched one nesting level at a time rather than
        flattened, so a nested block that means more than its leaves — a
        :class:`QScatter`, say — is still translated as the block it is.

        Args:
            op: Operation container to translate.
        """

        for child in op.children:
            self.translate(child)

    def _translate_qsend(self, op: QSend):
        """
        Translate a quantum send operation into a CUNQA instruction.

        Args:
            op: Quantum send operation to translate.
        """
        qsend(
            self._cunqa_circuit,
            self._q(op.qubits[0]),
            self.comm_qubit_of(op.comm_slot),
            [self._cl(c) for c in op.clbits],
            recving_circuit=f"rank_{op.dest_rank}",
            tag=op.tag,
        )

    def _translate_qrecv(self, op: QRecv):
        """
        Translate a quantum receive operation into a CUNQA instruction.

        Args:
            op: Quantum receive operation to translate.
        """
        qrecv(
            self._cunqa_circuit,
            self._q(op.qubits[0]),
            self.comm_qubit_of(op.comm_slot),
            [self._cl(c) for c in op.clbits],
            control_circuit=f"rank_{op.src_rank}",
            tag=op.tag,
        )

    def _translate_qscatter(self, op: QScatter):
        """
        Translate a quantum scatter operation into CUNQA instructions.

        A scatter is this rank's half of a rooted exchange: the teledata
        blocks it holds — sends on the root, receives elsewhere — which is
        what the container is translated into, one after the other.

        Args:
            op: Quantum scatter operation to translate.
        """
        self._translate_operation_container(op)

    def _translate_qgather(self, op: QGather):
        """
        Translate a quantum gather operation into CUNQA instructions.

        Args:
            op: Quantum gather operation to translate.
        """
        self._translate_operation_container(op)

    def _translate_expose(self, op: Expose):
        """
        Reject a per-rank translation of an expose operation.

        Args:
            op: Expose operation to translate.

        Raises:
            RuntimeError: Always. ``cat_entangler`` writes into every
                participating circuit at once, so the expansion belongs to
                :func:`translate_group`, not to a single-rank translation.
        """
        raise RuntimeError(
            f"{op!r} is collective and must be expanded by translate_group(), "
            f"which needs the circuits of all the participating ranks.")

    def _translate_unexpose(self, op: Unexpose):
        """
        Reject a per-rank translation of an unexpose operation.

        Args:
            op: Unexpose operation to translate.

        Raises:
            RuntimeError: Always, for the same reason as :meth:`_translate_expose`.
        """
        raise RuntimeError(
            f"{op!r} is collective and must be expanded by translate_group(), "
            f"which needs the circuits of all the participating ranks.")

    # Dispatch table: maps each Operation type to its translation method.
    # ClassicalControlledGate and ControlledGate must appear before Gate
    # because both are subclasses of Operation but not of Gate.
    _DISPATCH: dict = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._DISPATCH = {}

    def _build_dispatch(self):
        """
        Build the dispatch table for operation translation.

        Returns:
            A mapping from operation types to translation methods.
        """
        return {
            ClassicalControlledGate: self._translate_classical_controlled_gate,
            ControlledGate:          self._translate_controlled_gate,
            Gate:                    self._translate_gate,
            Measure:                 self._translate_measure,
            Reset:                   self._translate_reset,
            Barrier:                 self._translate_barrier,
            OperationContainer:      self._translate_operation_container,
            QSend:                   self._translate_qsend,
            QRecv:                   self._translate_qrecv,
            QScatter:                self._translate_qscatter,
            QGather:                 self._translate_qgather,
            Expose:                  self._translate_expose,
            Unexpose:                self._translate_unexpose,
        }

    def translate(self, op: Operation) -> Any:
        """
        Dispatch an operation to its corresponding translation method.

        Args:
            op: Operation to translate.

        Returns:
            The translated CUNQA instruction or instructions.

        Raises:
            TypeError: If the operation type is unknown.
        """
        super().translate(op)
        return self._cunqa_circuit


# ----------------------------------------------------------------------
# Joint translation of a group of ranks
# ----------------------------------------------------------------------

def _emit_expose(
    records: Dict[int, Expose],
    adapters: Dict[int, CunqaCircuitAdapter],
) -> None:
    """
    Expand one expose window into a CUNQA cat-entangler.

    The GHZ state is spread over one communication qubit per participant,
    and the root's control is teleported onto all of them, so each receiver
    can drive local gates with it.

    Args:
        records: Each participant's expose record, keyed by rank.
        adapters: Circuit adapter of every rank in the group.
    """
    order = records[next(iter(records))].ranks       # root first
    root = order[0]

    cat_entangler(
        [adapters[r].cunqa_circuit for r in order],
        adapters[root]._q(records[root].data_qubit),
        [adapters[r].comm_qubit_of(records[r].comm_slot) for r in order],
        [adapters[r]._cl(records[r].clbits[0]) for r in order],
        tag=records[root].tag,
    )


def _emit_unexpose(
    records: Dict[int, Unexpose],
    adapters: Dict[int, CunqaCircuitAdapter],
) -> None:
    """
    Expand one unexpose into a CUNQA cat-disentangler.

    Every receiver measures its communication qubit out of the GHZ state
    and ships the outcome back, and the root turns those outcomes into the
    phase correction its data qubit needs.

    Args:
        records: Each participant's unexpose record, keyed by rank.
        adapters: Circuit adapter of every rank in the group.
    """
    order = records[next(iter(records))].ranks       # root first
    root, receivers = order[0], order[1:]

    cat_disentangler(
        [adapters[r].cunqa_circuit for r in order],
        adapters[root]._q(records[root].data_qubit),
        [adapters[r].comm_qubit_of(records[r].comm_slot) for r in receivers],
        [adapters[root]._cl(c) for c in records[root].clbits],
        [adapters[r]._cl(records[r].clbits[0]) for r in receivers],
    )


def idle_circuit(index: int) -> CunqaCircuit:
    """
    Build the trivial circuit sent to a vQPU that no rank is using.

    CUNQA runs one executor per family of vQPUs, and every round that
    executor waits for a circuit from *each* vQPU of the family before it
    runs anything: a vQPU left out does not sit idle, it holds up the whole
    family. A run with fewer ranks than the family has vQPUs therefore
    submits this circuit to each of the spare ones, which does nothing, is
    over immediately, and whose counts are discarded.

    Args:
        index: Position of the spare vQPU, used to give the circuit an id
            of its own.

    Returns:
        A one-qubit circuit holding a single measurement.
    """
    circuit = CunqaCircuit((1, 0), 1, id=f"netqmpi_idle_{index}")
    circuit.measure(0, 0)
    return circuit


def check_transfers(adapters: Dict[int, CunqaCircuitAdapter]) -> None:
    """
    Check that every point-to-point transfer of the group has both halves.

    A ``qsend`` and its ``qrecv`` are paired at run time by the tag both
    sides derive from the ranks involved, so a transfer whose other half
    was never traced is not an error CUNQA can report: the vQPU that made
    the call simply waits for a partner that never comes, and the run hangs
    with nothing to show for it. Reading the group's own records is enough
    to see it coming, and to say which rank is left waiting for what.

    Only the backend-agnostic records are read, so any backend that pairs
    its transfers by tag can use this as it stands.

    Args:
        adapters: Circuit adapter of every rank, keyed by rank.

    Raises:
        RuntimeError: If a ``qsend`` has no matching ``qrecv``, or a
            ``qrecv`` no matching ``qsend``.
    """
    ranks = sorted(adapters)
    sends: Dict[str, Tuple[int, QSend]] = {}
    recvs: Dict[str, Tuple[int, QRecv]] = {}
    for rank in ranks:
        for op in adapters[rank].ops.flatten():
            if isinstance(op, QSend):
                sends[op.tag] = (rank, op)
            elif isinstance(op, QRecv):
                recvs[op.tag] = (rank, op)

    # How many transfers each ordered pair of ranks traced, so a mismatch
    # can be reported as the count it is rather than as one lone qubit.
    sent = Counter((rank, op.dest_rank) for rank, op in sends.values())
    received = Counter((op.src_rank, rank) for rank, op in recvs.values())

    def outside(rank: int) -> str:
        return (f"rank {rank} is not one of the {len(ranks)} ranks of this run "
                f"{ranks}")

    def plural(count: int, call: str) -> str:
        return f"{count} {call}" + ("" if count == 1 else "s")

    def tally(source: int, dest: int) -> str:
        return (f"rank {source} traced {plural(sent[(source, dest)], 'qsend')} to "
                f"rank {dest}, and rank {dest} traced "
                f"{plural(received[(source, dest)], 'qrecv')} from rank {source}")

    problems: List[str] = []
    for tag in sorted(sends):
        if tag in recvs:
            continue
        rank, op = sends[tag]
        if op.dest_rank not in adapters:
            problems.append(
                f"rank {rank} sends qubit {op.qubits[0]} to rank {op.dest_rank}, "
                f"but {outside(op.dest_rank)}")
        else:
            problems.append(
                f"rank {rank} sends qubit {op.qubits[0]} to rank {op.dest_rank}, "
                f"which never receives it: {tally(rank, op.dest_rank)}")
    for tag in sorted(recvs):
        if tag in sends:
            continue
        rank, op = recvs[tag]
        if op.src_rank not in adapters:
            problems.append(
                f"rank {rank} waits on qubit {op.qubits[0]} for a transfer from "
                f"rank {op.src_rank}, but {outside(op.src_rank)}")
        else:
            problems.append(
                f"rank {rank} waits on qubit {op.qubits[0]} for a transfer from "
                f"rank {op.src_rank}, which never sends it: "
                f"{tally(op.src_rank, rank)}")

    if problems:
        listing = "\n  - ".join(problems)
        raise RuntimeError(
            "Every qsend needs a qrecv on the destination rank to pair with, "
            "and the run would wait for the missing half forever:\n  - "
            + listing)


def _rank_list(ranks: List[int]) -> str:
    """
    Spell a list of ranks the way a sentence would.

    Args:
        ranks: Ranks to name.

    Returns:
        ``"rank 7"``, ``"ranks 7 and 9"``, ``"ranks 7, 9 and 11"``.
    """
    names = [str(r) for r in sorted(ranks)]
    plural = "rank" if len(names) == 1 else "ranks"
    if len(names) == 1:
        listing = names[0]
    else:
        listing = ", ".join(names[:-1]) + " and " + names[-1]
    return f"{plural} {listing}"


def _describe_collective(op: CollectiveOperation) -> str:
    """
    Spell a collective record the way the user wrote the call.

    Args:
        op: Record of an ``expose`` or ``unexpose``.

    Returns:
        The call as it appears in the program, receivers and root.
    """
    name = "expose" if isinstance(op, Expose) else "unexpose"
    return f"{name}(ranks={op.ranks[1:]}, root={op.root})"


def _deadlock_report(blocked: Dict[int, CollectiveOperation]) -> str:
    """
    Explain which ranks reached a collective call and which did not.

    Every rank of the group is, at this point, either sitting on a
    collective or done with its block, so for each call still waiting it is
    known exactly who is missing and what they are doing instead.

    Args:
        blocked: Collective each blocked rank is waiting on, keyed by rank.

    Returns:
        The message of the deadlock error.
    """
    # Group the waiting ranks by the call they are waiting on, so a
    # collective half of whose participants arrived is reported once.
    calls: Dict[Tuple[Any, ...], CollectiveOperation] = {}
    for rank in sorted(blocked):
        op = blocked[rank]
        calls.setdefault((type(op).__name__, op.tag, tuple(op.ranks)), op)

    lines: List[str] = []
    never_called = False        # a participant ended its block without the call
    called_something_else = False   # a participant is stuck on a different call
    for op in calls.values():
        reached = [r for r in op.ranks
                   if r in blocked and op.matches(blocked[r])]
        missing = [r for r in op.ranks if r not in reached]
        lines.append(f"  {_describe_collective(op)}")
        lines.append(
            "      called by:  "
            + ", ".join(f"rank {r}" for r in sorted(reached)))
        for rank in sorted(missing):
            if rank in blocked:
                called_something_else = True
                doing = f"is waiting on {_describe_collective(blocked[rank])}"
            else:
                never_called = True
                doing = "finished its block without calling it"
            lines.append(f"      missing:    rank {rank}, which {doing}")

    # Point at the mistake that actually fits what the ranks are doing.
    if never_called:
        lines.append(
            "  Every rank a collective names has to call it, just as every rank "
            "of an MPI_Bcast calls it.")
    if called_something_else:
        lines.append(
            "  These ranks are waiting on different calls: the participants of a "
            "collective have to reach theirs in the same order, and open and "
            "close nested windows in the same order too.")

    return (
        "Deadlock while translating the group: a collective call is waiting "
        "for ranks that never reached it.\n"
        + "\n".join(lines)
    )


def translate_group(adapters: Dict[int, CunqaCircuitAdapter]) -> List[CunqaCircuit]:
    """
    Translate the circuits of a whole group of ranks into CUNQA circuits.

    Every rank's operation stream is drained until it reaches a collective
    call. Once all the participants of a collective are waiting on it, the
    call is expanded into all of their circuits at once and they resume.
    This mirrors what the ranks would do if they really ran side by side,
    while keeping each circuit's instructions in program order.

    Args:
        adapters: Circuit adapter of every rank, keyed by rank.

    Returns:
        The translated CUNQA circuits, ordered by rank.

    Raises:
        RuntimeError: If a collective names a rank outside the group, or if
            the ranks block on collectives that never match — the trace
            equivalent of a deadlock.
    """
    check_transfers(adapters)

    ranks = sorted(adapters)
    streams = {r: list(adapters[r].ops.flatten()) for r in ranks}
    cursors = {r: 0 for r in ranks}

    for adapter in adapters.values():
        adapter.prepare()

    def pending(rank: int) -> Optional[CollectiveOperation]:
        """Return the collective operation *rank* is blocked on, if any."""
        stream, cursor = streams[rank], cursors[rank]
        if cursor < len(stream) and isinstance(stream[cursor], CollectiveOperation):
            return stream[cursor]
        return None

    while True:
        # Every rank runs ahead on its own until it hits a collective.
        for rank in ranks:
            stream = streams[rank]
            while cursors[rank] < len(stream) and not isinstance(
                stream[cursors[rank]], CollectiveOperation
            ):
                adapters[rank].translate(stream[cursors[rank]])
                cursors[rank] += 1

        blocked = {r: pending(r) for r in ranks if pending(r) is not None}
        if not blocked:
            break

        # A collective is ready when all of its participants sit on it.
        ready = None
        for rank in sorted(blocked):
            op = blocked[rank]
            missing = [r for r in op.ranks
                       if r not in blocked or not op.matches(blocked[r])]
            if not missing:
                ready = op
                break
            outside = [r for r in op.ranks if r not in adapters]
            if outside:
                raise RuntimeError(
                    f"rank {rank} called {_describe_collective(op)}, which names "
                    f"{_rank_list(outside)}, but this run has {len(ranks)} ranks, "
                    f"numbered 0 to {ranks[-1]}. A collective can only name ranks "
                    f"taking part in the run, so nothing can ever match that call.")

        if ready is None:
            raise RuntimeError(_deadlock_report(blocked))

        records = {r: blocked[r] for r in ready.ranks}
        if isinstance(ready, Expose):
            _emit_expose(records, adapters)
        else:
            _emit_unexpose(records, adapters)
        for rank in ready.ranks:
            cursors[rank] += 1

    return [adapters[r].cunqa_circuit for r in ranks]
