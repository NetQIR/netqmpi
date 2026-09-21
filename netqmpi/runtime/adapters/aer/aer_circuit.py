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

    For qsend, the destination offset within the same circuit group is
    computed as ``group_base + dest_rank * num_qubits``, which remains
    valid regardless of how many circuit groups exist.
    """

    def __init__(
        self,
        num_qubits: int,
        num_clbits: int,
        comm: "AerCommunicator",
        global_circuit: "QuantumCircuit",
        qubit_offset: int,
        clbit_offset: int,
        group_base: int,
    ) -> None:
        """
        Initialize the AerCircuitAdapter.

        Args:
            num_qubits: Number of qubits for this rank's circuit slice.
            num_clbits: Number of classical bits for this rank's circuit slice.
            comm: Communicator owning this rank.
            global_circuit: Shared QuantumCircuit for all ranks.
            qubit_offset: Global qubit index where this rank's slice starts.
            clbit_offset: Global clbit index where this rank's slice starts.
            group_base: Global qubit index where this circuit group starts
                (used to compute qsend destination offsets).
        """
        super().__init__(num_qubits, num_clbits, comm)
        self._global_circuit = global_circuit
        self._offset = qubit_offset
        self._clbit_offset = clbit_offset
        self._group_base = group_base
        self._config = comm._config

    # ------------------------------------------------------------------
    # Translation methods
    # ------------------------------------------------------------------

    def _translate_gate(self, op: Gate) -> None:
        """
        Translate a single-qubit (or two-qubit SWAP) gate.

        Args:
            op: Gate operation to translate.
        """
        q = op.qubits[0] + self._offset
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
                op.qubits[0] + self._offset,
                op.qubits[1] + self._offset,
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
        ctrl = [c + self._offset for c in op.controls]
        tgt = [q + self._offset for q in op.targets[0].qubits]

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
            op.qubits[0] + self._offset,
            op.cbit + self._clbit_offset,
        )

    def _translate_reset(self, op: Reset) -> None:
        """
        Translate a reset into a global-circuit instruction.

        Args:
            op: Reset operation to translate.
        """
        self._global_circuit.reset(op.qubits[0] + self._offset)

    def _translate_barrier(self, op: Barrier) -> None:
        """
        Translate a barrier across all global qubits owned by this rank.

        Args:
            op: Barrier operation to translate.
        """
        if op.qubits:
            global_qubits = [q + self._offset for q in op.qubits]
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
        Translate an expose operation.

        Args:
            op: Expose operation to translate.

        Raises:
            NotImplementedError: Always; not yet implemented for this backend.
        """
        raise NotImplementedError(
            "Expose is not yet implemented for the Aer backend."
        )

    def _translate_unexpose(self, op: Unexpose) -> None:
        """
        Translate an unexpose operation.

        Args:
            op: Unexpose operation to translate.

        Raises:
            NotImplementedError: Always; not yet implemented for this backend.
        """
        raise NotImplementedError(
            "Unexpose is not yet implemented for the Aer backend."
        )

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


def _deadlock_error(blocked: Dict[int, Operation], ranks: List[int]) -> RuntimeError:
    """
    Describe a set of transfers that can never pair up.

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
        else:
            lines.append(f"  rank {rank} is blocked on {type(op).__name__}, "
                         f"which the Aer adapter cannot pair")
    return RuntimeError(
        "The ranks blocked on transfers that never match, so the program "
        "cannot be ordered:\n" + "\n".join(lines) +
        f"\nEvery qsend needs a qrecv on the destination rank, and vice "
        f"versa, among the {len(ranks)} ranks of this run {ranks}.")


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
    Each rank advances through its own operations until it reaches a
    transfer; once both halves of a transfer are waiting, the qubit is moved
    and both ranks resume. The result is an emission order that respects
    every dependency the program expressed.

    Pairing a ``qsend`` with its ``qrecv`` also supplies what a single-rank
    pass cannot: the receiver's own qubit index. The transfer moves the
    state from the sender's slot to the slot the *receiver* asked for,
    instead of assuming both sides chose the same local index.

    Args:
        adapters: Circuit adapter of every rank, keyed by rank.

    Raises:
        RuntimeError: If the ranks block on transfers that never match — the
            trace-time equivalent of a deadlock — or if a matched pair
            disagrees on how many qubits it moves.
    """
    ranks = sorted(adapters)
    streams = {rank: list(adapters[rank].ops.flatten()) for rank in ranks}
    cursors = {rank: 0 for rank in ranks}

    while True:
        # Every rank runs ahead on its own until it hits a transfer.
        for rank in ranks:
            stream = streams[rank]
            while cursors[rank] < len(stream) and not isinstance(
                stream[cursors[rank]], (QSend, QRecv)
            ):
                adapters[rank].translate(stream[cursors[rank]])
                cursors[rank] += 1

        blocked = {rank: streams[rank][cursors[rank]]
                   for rank in ranks if cursors[rank] < len(streams[rank])}
        if not blocked:
            return

        matched = _transfer_partners(blocked)
        if matched is None:
            raise _deadlock_error(blocked, ranks)

        tag, (source_rank, send_op), (target_rank, recv_op) = matched

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

        cursors[source_rank] += 1
        cursors[target_rank] += 1
