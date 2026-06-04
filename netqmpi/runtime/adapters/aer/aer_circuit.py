"""
Circuit adapter for Qiskit AerSimulator.

Translates SDK operations into Qiskit gates appended directly to a
shared global QuantumCircuit owned by the executor.  Every local qubit
index is shifted by the rank's offset before being written to the global
register.
"""
from __future__ import annotations

from typing import Any, TYPE_CHECKING

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
        for child in op.flatten():
            self.translate(child)

    def _translate_qsend(self, op: QSend) -> None:
        """
        Translate a quantum send into the global circuit.

        In ``swap`` mode, inserts a SWAP gate between the source qubit slot
        and the matching slot on the destination rank within the same circuit
        group.  The destination offset is ``group_base + dest_rank * num_qubits``,
        which remains correct across multiple circuit groups.

        Args:
            op: Quantum send operation to translate.

        Raises:
            NotImplementedError: When ``transfer_mode`` is ``"teleport"``.
        """
        if self._config.transfer_mode == "swap":
            dest_offset = self._group_base + op.dest_rank * self._num_qubits
            for q in op.qubits:
                self._global_circuit.swap(q + self._offset, q + dest_offset)
        else:
            raise NotImplementedError(
                "teleport mode is not yet implemented; use transfer_mode='swap'"
            )

    def _translate_qrecv(self, op: QRecv) -> None:
        """
        Translate a quantum receive (no-op for the monolithic circuit model).

        After a ``qsend`` SWAP the transferred state is already in the
        destination slot, so the receiver needs no circuit action.

        Args:
            op: Quantum receive operation to translate.
        """

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
