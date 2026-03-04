"""
Circuit adapter for the NetQASM backend.

Implements the Circuit interface to work with NetQASM circuits.
"""
from typing import List, Any

from netqmpi.sdk.core.circuit import Circuit
from netqmpi.sdk.core.operations import (
    Operation,
    Gate, ControlledGate, ClassicalControlledGate,
    Measure, Reset, Barrier,
    OperationContainer,
    QSend, QRecv, QScatter, QGather, Expose, Unexpose,
)


class NetQASMCircuitAdapter(Circuit):
    """
    Adapter that implements Circuit for the NetQASM backend.
    
    This adapter encapsulates a NetQASM circuit and provides
    the common interface defined in the abstract Circuit class.
    """
    
    def __init__(self, num_qubits: int, num_clbits: int):
        """
        Initialize the NetQASM circuit.
        
        Args:
            num_qubits: Number of qubits in the circuit.
            num_clbits: Number of classical bits in the circuit.
        """
        super().__init__(num_qubits, num_clbits)
        # TODO: Initialize the underlying NetQASM circuit
        self._netqasm_circuit = None  # This would be the actual NetQASM object
    
    def operations_supported(self) -> List[str]:
        """
        Returns the operations supported by this NetQASM circuit.
        
        Returns:
            List of available quantum operations.
        """
        # TODO: Implement actual NetQASM operations list
        return [
            'H', 'X', 'Y', 'Z', 'S', 'T', 'K',
            'CNOT', 'CZ', 'CPHASE',
            'RX', 'RY', 'RZ',
            'measure', 'reset'
        ]

    def _translate_gate(self, op: Gate):
        """Translate a single unitary gate (H, X, RZ, …) into a NetQASM instruction."""
        raise NotImplementedError(f"Gate '{op.name}' is not yet implemented for the NetQASM backend.")

    def _translate_controlled_gate(self, op: ControlledGate):
        """Translate a qubit-controlled gate (e.g. CNOT, CZ) into NetQASM instructions."""
        raise NotImplementedError("ControlledGate is not yet implemented for the NetQASM backend.")

    def _translate_classical_controlled_gate(self, op: ClassicalControlledGate):
        """Translate a classically-controlled gate (conditioned on cbit values) into NetQASM instructions."""
        raise NotImplementedError("ClassicalControlledGate is not yet implemented for the NetQASM backend.")

    def _translate_measure(self, op: Measure):
        """Translate a measurement into a NetQASM measure instruction."""
        raise NotImplementedError("Measure is not yet implemented for the NetQASM backend.")

    def _translate_reset(self, op: Reset):
        """Translate a qubit reset into a NetQASM init instruction."""
        raise NotImplementedError("Reset is not yet implemented for the NetQASM backend.")

    def _translate_barrier(self, op: Barrier):
        """Translate a barrier (no-op for NetQASM; used to prevent re-ordering)."""
        raise NotImplementedError("Barrier is not yet implemented for the NetQASM backend.")

    def _translate_operation_container(self, op: OperationContainer):
        """Recursively translate a composite OperationContainer by flattening its children."""
        raise NotImplementedError("OperationContainer translation is not yet implemented for the NetQASM backend.")

    def _translate_qsend(self, op: QSend):
        """Translate a QSend into the NetQASM teleportation-based send protocol."""
        raise NotImplementedError("QSend is not yet implemented for the NetQASM backend.")

    def _translate_qrecv(self, op: QRecv):
        """Translate a QRecv into the NetQASM teleportation-based receive protocol."""
        raise NotImplementedError("QRecv is not yet implemented for the NetQASM backend.")

    def _translate_qscatter(self, op: QScatter):
        """Translate a QScatter into individual NetQASM send operations from the sender rank."""
        raise NotImplementedError("QScatter is not yet implemented for the NetQASM backend.")

    def _translate_qgather(self, op: QGather):
        """Translate a QGather into individual NetQASM receive operations at the receiver rank."""
        raise NotImplementedError("QGather is not yet implemented for the NetQASM backend.")

    def _translate_expose(self, op: Expose):
        """Translate an Expose into a NetQASM GHZ-based telegate exposure."""
        raise NotImplementedError("Expose is not yet implemented for the NetQASM backend.")

    def _translate_unexpose(self, op: Unexpose):
        """Translate an Unexpose into the NetQASM GHZ measurement and classical corrections."""
        raise NotImplementedError("Unexpose is not yet implemented for the NetQASM backend.")

    # Dispatch table: maps each Operation type to its translation method.
    # ClassicalControlledGate and ControlledGate must appear before Gate
    # because both are subclasses of Operation but not of Gate.
    _DISPATCH: dict = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._DISPATCH = {}

    def _build_dispatch(self):
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
        Dispatches a generic Operation to the appropriate private translation method
        via a type-keyed dispatch table.

        Args:
            op: The generic Operation to translate.

        Returns:
            The corresponding NetQASM instruction(s).

        Raises:
            TypeError: If the operation type is unknown.
        """
        if not self._DISPATCH:
            self._DISPATCH = self._build_dispatch()

        handler = self._DISPATCH.get(type(op))
        if handler is None:
            # Walk the MRO to support subclasses not registered explicitly.
            handler = next(
                (self._DISPATCH[t] for t in type(op).__mro__ if t in self._DISPATCH),
                None,
            )
        if handler is None:
            raise TypeError(f"Unknown operation type: {type(op).__name__}")
        return handler(op)


    def build(self) -> Any:
        pass

    # These methods will delegate to the underlying NetQASM circuit
