"""
Circuit adapter for Qiskit AerSimulator.

Translates SDK operations into Qiskit gates appended directly to a
shared global QuantumCircuit owned by the executor.  Every local qubit
index is shifted by the rank's offset before being written to the global
register.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from qiskit import QuantumCircuit  # type: ignore[import-not-found]

from netqmpi.sdk.circuit import Circuit
from netqmpi.sdk.operations import (
    Operation,
    Gate, ControlledGate, ClassicalControlledGate,
    Measure, Reset, Barrier,
    OperationContainer,
    QSend, QRecv, QScatter, QGather, Expose, Unexpose,
    CollectiveOperation,
)

if TYPE_CHECKING:
    from netqmpi.runtime.adapters.aer.aer_communicator import AerCommunicator


class AerCircuitAdapter(Circuit):
    """
    Circuit adapter that writes operations into a shared global QuantumCircuit.

    Each rank owns a contiguous slice ``[qubit_offset, qubit_offset + num_qubits)``
    of the global qubit register and the analogous slice of the classical
    register.  All translate methods map local indices to global indices
    before appending gates.

    A rank's slice is not sized or placed until every rank has finished
    tracing: the ranks may ask for registers of different widths — a
    ``qscatter`` root holds one qubit per receiver while the receivers hold
    one each — so where a slice starts cannot be known from the rank index
    alone. :meth:`assign_slice` fills the offsets in once the layout is
    settled.

    A *communication* qubit belongs to no slice at all. It addresses a
    control another rank has lent through an open ``expose`` window, and
    :meth:`_global` resolves it to that rank's data qubit for as long as the
    window is open.
    """

    def __init__(
        self,
        num_qubits: int,
        num_clbits: int,
        comm: "AerCommunicator",
    ) -> None:
        """
        Initialize the AerCircuitAdapter.

        Args:
            num_qubits: Number of qubits for this rank's circuit slice.
            num_clbits: Number of classical bits for this rank's circuit slice.
            comm: Communicator owning this rank.
        """
        super().__init__(num_qubits, num_clbits, comm)
        self._global_circuit: Optional["QuantumCircuit"] = None
        self._offset = 0
        self._clbit_offset = 0
        self._config = comm._config
        # Communication-qubit slots currently holding a control lent by
        # another rank, mapped to the global qubit that control lives on.
        # Filled by an expose window and emptied by the matching unexpose.
        self._borrowed: Dict[int, int] = {}

    def assign_slice(self, global_circuit: "QuantumCircuit",
                     qubit_offset: int, clbit_offset: int) -> None:
        """
        Place this rank's slice in the global circuit.

        Called once every rank has finished tracing, so that the widths each
        of them asked for are all known and the slices can be laid out
        without overlapping.

        Args:
            global_circuit: The circuit shared by every rank.
            qubit_offset: Global index where this rank's qubits start.
            clbit_offset: Global index where this rank's classical bits start.
        """
        self._global_circuit = global_circuit
        self._offset = qubit_offset
        self._clbit_offset = clbit_offset

    # ------------------------------------------------------------------
    # Translation methods
    # ------------------------------------------------------------------

    def _global(self, qubit: int) -> int:
        """
        Map a NetQMPI qubit index onto its index in the global circuit.

        Data qubits sit in this rank's own slice. A communication qubit is
        not a qubit of this rank at all: it addresses a control another rank
        has lent through an open expose window, so it resolves to that
        rank's data qubit.

        Args:
            qubit: Data qubit index, or a communication qubit as returned by
                :meth:`~netqmpi.sdk.circuit.Circuit.comm_qubit`.

        Returns:
            The qubit index in the global circuit.

        Raises:
            RuntimeError: If a communication qubit is used outside the expose
                window that lent it.
        """
        if qubit < self._num_qubits:
            return qubit + self._offset

        slot = qubit - self._num_qubits
        borrowed = self._borrowed.get(slot)
        if borrowed is None:
            raise RuntimeError(
                f"rank {self._comm.rank} used communication qubit {qubit} "
                f"(slot {slot}) with no expose window open on it.")
        return borrowed

    # ------------------------------------------------------------------
    # Telegate windows, opened and closed by the joint pass
    # ------------------------------------------------------------------

    def lend_control(self, slot: int, control: int) -> None:
        """
        Point a communication-qubit slot at the control another rank lent.

        Args:
            slot: Communication-qubit slot reserved by the expose.
            control: Global index of the root's exposed data qubit.
        """
        self._borrowed[slot] = control

    def release_control(self, slot: int) -> None:
        """
        Close a slot opened by :meth:`lend_control`.

        Args:
            slot: Communication-qubit slot the window reserved.
        """
        self._borrowed.pop(slot, None)

    # ------------------------------------------------------------------
    # Translation methods
    # ------------------------------------------------------------------

    def _translate_gate(self, op: Gate) -> None:
        """
        Translate a single-qubit (or two-qubit SWAP) gate.

        Args:
            op: Gate operation to translate.
        """
        q = self._global(op.qubits[0])
        gate_map = {
            "H":    lambda: self._global_circuit.h(q),
            "X":    lambda: self._global_circuit.x(q),
            "Y":    lambda: self._global_circuit.y(q),
            "Z":    lambda: self._global_circuit.z(q),
            "S":    lambda: self._global_circuit.s(q),
            "SDG":  lambda: self._global_circuit.sdg(q),
            "T":    lambda: self._global_circuit.t(q),
            "TDG":  lambda: self._global_circuit.tdg(q),
            "RX":   lambda: self._global_circuit.rx(op.params[0], q),
            "RY":   lambda: self._global_circuit.ry(op.params[0], q),
            "RZ":   lambda: self._global_circuit.rz(op.params[0], q),
            "SWAP": lambda: self._global_circuit.swap(
                self._global(op.qubits[0]),
                self._global(op.qubits[1]),
            ),
        }
        if op.name in gate_map:
            gate_map[op.name]()

    def _translate_controlled_gate(self, op: ControlledGate) -> None:
        """
        Translate a controlled gate (CX, CZ, CRZ, CCX).

        Args:
            op: Controlled gate operation to translate.
        """
        target_name = op.targets[0].name
        ctrl = [self._global(c) for c in op.controls]
        tgt = [self._global(q) for q in op.targets[0].qubits]

        if target_name == "X":
            if len(ctrl) == 1:
                self._global_circuit.cx(ctrl[0], tgt[0])
            elif len(ctrl) == 2:
                self._global_circuit.ccx(ctrl[0], ctrl[1], tgt[0])
        elif target_name == "Z" and len(ctrl) == 1:
            self._global_circuit.cz(ctrl[0], tgt[0])
        elif target_name == "RZ" and len(ctrl) == 1:
            self._global_circuit.crz(op.targets[0].params[0], ctrl[0], tgt[0])

    def _translate_classical_controlled_gate(self, op: ClassicalControlledGate) -> None:
        """
        Translate a classically controlled gate.

        Args:
            op: Classically controlled gate operation.

        Raises:
            NotImplementedError: Always; not yet supported for this backend.
        """
        raise NotImplementedError(
            "ClassicalControlledGate is not yet implemented for the Aer backend."
        )

    def _translate_measure(self, op: Measure) -> None:
        """
        Translate a measurement into a global-circuit instruction.

        Args:
            op: Measurement operation to translate.
        """
        self._global_circuit.measure(
            self._global(op.qubits[0]),
            op.cbit + self._clbit_offset,
        )

    def _translate_reset(self, op: Reset) -> None:
        """
        Translate a reset into a global-circuit instruction.

        Args:
            op: Reset operation to translate.
        """
        self._global_circuit.reset(self._global(op.qubits[0]))

    def _translate_barrier(self, op: Barrier) -> None:
        """
        Translate a barrier across all global qubits owned by this rank.

        Args:
            op: Barrier operation to translate.
        """
        if op.qubits:
            global_qubits = [self._global(q) for q in op.qubits]
        else:
            global_qubits = list(range(self._offset, self._offset + self._num_qubits))
        self._global_circuit.barrier(global_qubits)

    def _translate_operation_container(self, op: OperationContainer) -> None:
        """
        Translate an operation container by translating each leaf operation.

        Args:
            op: Operation container to translate.
        """
        for child in op.children:
            self.translate(child)

    def _translate_qsend(self, op: QSend) -> None:
        """
        Reject a transfer reached outside the joint translation pass.

        A transfer needs both of its halves to be emitted: the sender says
        which qubit leaves, the receiver says where it lands, and the pair
        also fixes *when* it happens relative to the other ranks' gates.
        Translating one rank's stream on its own can supply none of that,
        so rather than guess — which is what this adapter used to do, with
        silently wrong results — it refuses. :func:`translate_group` is the
        supported entry point.

        Args:
            op: Quantum send operation to translate.

        Raises:
            RuntimeError: Always.
        """
        raise RuntimeError(
            f"qsend (tag {op.tag}) cannot be translated on its own: the Aer "
            f"adapter needs every rank's stream at once to pair a transfer "
            f"with its qrecv and to order it against the other ranks' gates. "
            f"Use translate_group().")

    def _translate_qrecv(self, op: QRecv) -> None:
        """
        Reject a transfer reached outside the joint translation pass.

        See :meth:`_translate_qsend`.

        Args:
            op: Quantum receive operation to translate.

        Raises:
            RuntimeError: Always.
        """
        raise RuntimeError(
            f"qrecv (tag {op.tag}) cannot be translated on its own: the Aer "
            f"adapter needs every rank's stream at once to pair a transfer "
            f"with its qsend and to order it against the other ranks' gates. "
            f"Use translate_group().")

    # ------------------------------------------------------------------
    # Transfers, emitted by the joint pass
    # ------------------------------------------------------------------

    @property
    def qubit_offset(self) -> int:
        """Global index where this rank's slice of the register starts."""
        return self._offset

    @property
    def clbit_offset(self) -> int:
        """Global index where this rank's classical bits start."""
        return self._clbit_offset

    def emit_transfer(self, op: QSend, source: int, target: int) -> None:
        """
        Move one qubit from a sender's slot to a receiver's slot.

        In ``swap`` mode the move is an unphysical SWAP straight across the
        global register: no entanglement is consumed and no noise is
        introduced, which is what makes this backend a *correctness*
        reference rather than a model of a network.

        Args:
            op: The send half of the transfer, for error reporting.
            source: Global index of the sender's qubit.
            target: Global index of the receiver's qubit.

        Raises:
            NotImplementedError: When ``transfer_mode`` is ``"teleport"``.
        """
        if self._config.transfer_mode != "swap":
            raise NotImplementedError(
                f"transfer_mode={self._config.transfer_mode!r} is not "
                f"implemented for the Aer backend; use 'swap'.")
        self._global_circuit.swap(source, target)

    def _translate_qscatter(self, op: QScatter) -> None:
        """
        Translate a quantum scatter operation.

        Args:
            op: Quantum scatter operation to translate.

        Raises:
            NotImplementedError: Always; not yet implemented for this backend.
        """
        raise NotImplementedError(
            "QScatter is not yet implemented for the Aer backend."
        )

    def _translate_qgather(self, op: QGather) -> None:
        """
        Translate a quantum gather operation.

        Args:
            op: Quantum gather operation to translate.

        Raises:
            NotImplementedError: Always; not yet implemented for this backend.
        """
        raise NotImplementedError(
            "QGather is not yet implemented for the Aer backend."
        )

    def _translate_expose(self, op: Expose) -> None:
        """
        Reject a telegate window reached outside the joint translation pass.

        An expose is collective: the root says which qubit it lends and every
        receiver says which slot it lends into, and none of that is knowable
        from one rank's stream. :func:`translate_group` is the supported
        entry point.

        Args:
            op: Expose operation to translate.

        Raises:
            RuntimeError: Always.
        """
        raise RuntimeError(
            f"expose (tag {op.tag}) cannot be translated on its own: the Aer "
            f"adapter needs every participant's stream at once to open a "
            f"telegate window. Use translate_group().")

    def _translate_unexpose(self, op: Unexpose) -> None:
        """
        Reject a telegate window reached outside the joint translation pass.

        See :meth:`_translate_expose`.

        Args:
            op: Unexpose operation to translate.

        Raises:
            RuntimeError: Always.
        """
        raise RuntimeError(
            f"unexpose (tag {op.tag}) cannot be translated on its own: the "
            f"Aer adapter needs every participant's stream at once to close a "
            f"telegate window. Use translate_group().")

    # ------------------------------------------------------------------
    # Dispatch table (mirrors the pattern in CunqaCircuitAdapter)
    # ------------------------------------------------------------------

    _DISPATCH: dict = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._DISPATCH = {}

    def _build_dispatch(self):
        """
        Build the dispatch table mapping operation types to translation methods.

        Returns:
            A dict from operation type to the corresponding translate method.
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
        Dispatch an operation and return the shared global QuantumCircuit.

        Args:
            op: Operation to translate.

        Returns:
            The shared global QuantumCircuit after appending the operation.

        Raises:
            TypeError: If the operation type is unknown.
        """
        super().translate(op)
        return self._global_circuit


# ----------------------------------------------------------------------
# Joint translation
# ----------------------------------------------------------------------

#: Operations a rank cannot emit on its own: they need a partner rank to be
#: sitting on the matching call before anything can be written out.
BLOCKING = (QSend, QRecv, CollectiveOperation)


def _transfer_partners(blocked: Dict[int, Operation]
                       ) -> Optional[Tuple[str, Tuple[int, QSend], Tuple[int, QRecv]]]:
    """
    Find a transfer whose two halves are both waiting.

    Args:
        blocked: The operation each rank is stopped on, keyed by rank.

    Returns:
        ``(tag, (source_rank, qsend), (target_rank, qrecv))`` for the first
        matched transfer, or ``None`` when nothing can be paired.
    """
    sends: Dict[str, Tuple[int, QSend]] = {}
    recvs: Dict[str, Tuple[int, QRecv]] = {}
    for rank, op in blocked.items():
        if isinstance(op, QSend):
            sends[op.tag] = (rank, op)
        elif isinstance(op, QRecv):
            recvs[op.tag] = (rank, op)

    for tag in sorted(sends):
        if tag in recvs:
            return tag, sends[tag], recvs[tag]
    return None


def _ready_collective(blocked: Dict[int, Operation]
                      ) -> Optional[CollectiveOperation]:
    """
    Find a collective every one of its participants is waiting on.

    Args:
        blocked: The operation each rank is stopped on, keyed by rank.

    Returns:
        The first collective whose whole group has arrived, or ``None``.
    """
    for rank in sorted(blocked):
        op = blocked[rank]
        if not isinstance(op, CollectiveOperation):
            continue
        if all(other in blocked and op.matches(blocked[other])
               for other in op.ranks):
            return op
    return None


def _open_window(adapters: Dict[int, "AerCircuitAdapter"],
                 blocked: Dict[int, Operation], op: Expose) -> None:
    """
    Open a telegate window: lend the root's control to every receiver.

    A real backend shares the control through a GHZ state — CUNQA's
    cat-entangler — so that each receiver holds a computational-basis copy
    it can use as a local control, and hands it back untouched. Aer runs
    every rank inside one circuit, so the copy is unnecessary: a receiver's
    controlled gate is emitted against the root's own qubit directly.

    That is exact rather than approximate. A telegate reads its control only
    in the computational basis, which is precisely why the cat-entangled
    copy can stand in for the original; running it the other way round is
    equally faithful. It is also unphysical in the same way this adapter's
    ``qsend`` is: no entanglement is consumed and no correction is sent,
    which keeps Aer a correctness reference rather than a model of a
    network.

    Args:
        adapters: Circuit adapter of every rank, keyed by rank.
        blocked: The operation each rank is stopped on, keyed by rank.
        op: Any participant's record of the window being opened.
    """
    root_record = blocked[op.root]
    control = adapters[op.root].qubit_offset + root_record.data_qubit

    for receiver in op.ranks[1:]:
        adapters[receiver].lend_control(blocked[receiver].comm_slot, control)


def _close_window(adapters: Dict[int, "AerCircuitAdapter"],
                  blocked: Dict[int, Operation], op: Unexpose) -> None:
    """
    Close a telegate window, giving the root its control back.

    Args:
        adapters: Circuit adapter of every rank, keyed by rank.
        blocked: The operation each rank is stopped on, keyed by rank.
        op: Any participant's record of the window being closed.
    """
    for receiver in op.ranks[1:]:
        adapters[receiver].release_control(blocked[receiver].comm_slot)


def _deadlock_error(blocked: Dict[int, Operation], ranks: List[int]) -> RuntimeError:
    """
    Describe a set of calls that can never pair up.

    Args:
        blocked: The operation each rank is stopped on, keyed by rank.
        ranks: Every rank of the group.

    Returns:
        An error naming what each stuck rank is waiting for.
    """
    lines = []
    for rank in sorted(blocked):
        op = blocked[rank]
        if isinstance(op, QSend):
            lines.append(f"  rank {rank} is sending qubits {op.qubits} to "
                         f"rank {op.dest_rank} (tag {op.tag})")
        elif isinstance(op, QRecv):
            lines.append(f"  rank {rank} is receiving qubits {op.qubits} from "
                         f"rank {op.src_rank} (tag {op.tag})")
        elif isinstance(op, CollectiveOperation):
            waiting = [r for r in op.ranks
                       if r not in blocked or not op.matches(blocked[r])]
            lines.append(f"  rank {rank} is in {type(op).__name__.lower()} "
                         f"over ranks {op.ranks} (tag {op.tag}), still "
                         f"waiting for {waiting}")
        else:
            lines.append(f"  rank {rank} is blocked on {type(op).__name__}, "
                         f"which the Aer adapter cannot pair")
    return RuntimeError(
        "The ranks blocked on calls that never match, so the program cannot "
        "be ordered:\n" + "\n".join(lines) +
        f"\nEvery qsend needs a qrecv on the destination rank, and every "
        f"expose and unexpose has to be reached by all of its participants, "
        f"among the {len(ranks)} ranks of this run {ranks}.")


def translate_group(adapters: Dict[int, "AerCircuitAdapter"]) -> None:
    """
    Translate the circuits of a whole group of ranks into the global circuit.

    Aer runs every rank inside a single ``QuantumCircuit``, so the order in
    which instructions are appended *is* the order in which they execute.
    Translating one rank fully and then the next therefore only works when
    the program's cross-rank dependencies happen to follow rank order: a
    chain 0 -> 1 -> 2 survives it, while a control that returns to rank 0
    between hops does not, and the run then produces a wrong answer with no
    error raised.

    This pass instead interleaves the ranks the way they would really run.
    Each rank advances through its own operations until it reaches something
    it cannot emit alone — a transfer, or a collective — and that call is
    expanded once every rank it involves is waiting on it, after which they
    all resume. The result is an emission order that respects every
    dependency the program expressed.

    Pairing a ``qsend`` with its ``qrecv`` also supplies what a single-rank
    pass cannot: the receiver's own qubit index. The transfer moves the
    state from the sender's slot to the slot the *receiver* asked for,
    instead of assuming both sides chose the same local index. An
    ``expose`` likewise needs the root's record for the qubit being lent and
    each receiver's for the slot it is lent into.

    Args:
        adapters: Circuit adapter of every rank, keyed by rank.

    Raises:
        RuntimeError: If the ranks block on calls that never match — the
            trace-time equivalent of a deadlock — or if a matched transfer
            disagrees on how many qubits it moves.
    """
    ranks = sorted(adapters)
    streams = {rank: list(adapters[rank].ops.flatten()) for rank in ranks}
    cursors = {rank: 0 for rank in ranks}

    while True:
        # Every rank runs ahead on its own until it hits something it
        # cannot emit without a partner.
        for rank in ranks:
            stream = streams[rank]
            while cursors[rank] < len(stream) and not isinstance(
                stream[cursors[rank]], BLOCKING
            ):
                adapters[rank].translate(stream[cursors[rank]])
                cursors[rank] += 1

        blocked = {rank: streams[rank][cursors[rank]]
                   for rank in ranks if cursors[rank] < len(streams[rank])}
        if not blocked:
            return

        transfer = _transfer_partners(blocked)
        if transfer is not None:
            _emit_transfer(adapters, *transfer)
            cursors[transfer[1][0]] += 1
            cursors[transfer[2][0]] += 1
            continue

        collective = _ready_collective(blocked)
        if collective is None:
            raise _deadlock_error(blocked, ranks)

        if isinstance(collective, Expose):
            _open_window(adapters, blocked, collective)
        elif isinstance(collective, Unexpose):
            _close_window(adapters, blocked, collective)
        else:
            raise RuntimeError(
                f"{type(collective).__name__} is not implemented for the Aer "
                f"backend.")

        for rank in collective.ranks:
            cursors[rank] += 1


def _emit_transfer(adapters: Dict[int, "AerCircuitAdapter"], tag: str,
                   source: Tuple[int, QSend], target: Tuple[int, QRecv]) -> None:
    """
    Move the qubits of one matched transfer across the global register.

    Args:
        adapters: Circuit adapter of every rank, keyed by rank.
        tag: Identifier the two halves share, for error reporting.
        source: The sending rank and its ``qsend`` record.
        target: The receiving rank and its ``qrecv`` record.

    Raises:
        RuntimeError: If the two halves disagree on how many qubits move.
    """
    source_rank, send_op = source
    target_rank, recv_op = target

    if len(send_op.qubits) != len(recv_op.qubits):
        raise RuntimeError(
            f"Transfer {tag} moves {len(send_op.qubits)} qubit(s) out of "
            f"rank {source_rank} but rank {target_rank} receives "
            f"{len(recv_op.qubits)}.")

    sender, receiver = adapters[source_rank], adapters[target_rank]
    for out_qubit, in_qubit in zip(send_op.qubits, recv_op.qubits):
        sender.emit_transfer(
            send_op,
            out_qubit + sender.qubit_offset,
            in_qubit + receiver.qubit_offset,
        )
