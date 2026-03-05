"""
Circuit adapter for the Cunqa backend.

Implements the Circuit interface to work with Cunqa circuits.
"""
import os, sys
sys.path.append(os.getenv("HOME"))

from typing import List, Any

from cunqa.circuit.core import CunqaCircuit

from netqmpi.sdk.core.circuit import Circuit
from netqmpi.sdk.core.operations import (
    Operation,
    Gate, ControlledGate, ClassicalControlledGate,
    Measure, Reset, Barrier,
    OperationContainer,
    QSend, QRecv, QScatter, QGather, Expose, Unexpose,
)

class CunqaCircuitAdapter(Circuit):
    """
    Adapter that implements Circuit for the Cunqa backend.
    
    This adapter encapsulates a Cunqa circuit and provides
    the common interface defined in the abstract Circuit class.
    """
    
    def __init__(self, num_qubits: int, num_clbits: int, rank):
        """
        Initialize the Cunqa circuit.
        
        Args:
            num_qubits: Number of qubits in the circuit.
            num_clbits: Number of classical bits in the circuit.
        """
        super().__init__(num_qubits, num_clbits)
        # TODO: Initialize the underlying Cunqa circuit
        self._cunqa_circuit = CunqaCircuit(num_qubits, num_clbits, id=f"rank_{rank}")
    
    def _translate_gate(self, op: Gate):
        """Translate a single unitary gate (H, X, RZ, …) into a CUNQA instruction."""
        
        gate_map = {
            # 1 qubit
            "H":   lambda: self._cunqa_circuit.h(*op.qubits),
            "X":   lambda: self._cunqa_circuit.x(*op.qubits),
            "Y":   lambda: self._cunqa_circuit.y(*op.qubits),
            "Z":   lambda: self._cunqa_circuit.z(*op.qubits),
            "S":   lambda: self._cunqa_circuit.s(*op.qubits),
            "SDG": lambda: self._cunqa_circuit.sdg(*op.qubits),
            "T":   lambda: self._cunqa_circuit.t(*op.qubits),
            "TDG": lambda: self._cunqa_circuit.tdg(*op.qubits),

            # 1 qubit
            "RX": lambda: self._cunqa_circuit.rx(op.params[0], *op.qubits),
            "RY": lambda: self._cunqa_circuit.ry(op.params[0], *op.qubits),
            "RZ": lambda: self._cunqa_circuit.rz(op.params[0], *op.qubits),
        }

        if op.name in gate_map:
            gate_map[op.name]()

    def _translate_controlled_gate(self, op: ControlledGate):
        """Translate a qubit-controlled gate (e.g. CNOT, CZ) into CUNQA instructions."""
        
        gate_2q = {
            # 2 qubits
            "CX":   lambda: self._cunqa_circuit.cx(*op.qubits),
            "CZ":   lambda: self._cunqa_circuit.cz(*op.qubits),
            "SWAP": lambda: self._cunqa_circuit.swap(*op.qubits),
            
            # 2 qubits
            "CRZ": lambda: self._cunqa_circuit.crz(op.params[0], *op.qubits),
        }

        if op.name in gate_2q:
            gate_2q[op.name]()
            

    def _translate_classical_controlled_gate(self, op: ClassicalControlledGate):
        """Translate a classically-controlled gate (conditioned on cbit values) into CUNQA instructions."""
        raise NotImplementedError("ClassicalControlledGate is not yet implemented for the CUNQA backend.")

    def _translate_measure(self, op: Measure):
        """Translate a measurement into a CUNQA measure instruction."""
        self._cunqa_circuit.measure(op.qubits[0], op.cbit)

    def _translate_reset(self, op: Reset):
        """Translate a qubit reset into a CUNQA init instruction."""
        self._cunqa_circuit.reset(op.qubits[0]) # TODO: Mirar si resetea con varios también

    def _translate_barrier(self, op: Barrier):
        """Translate a barrier (no-op for CUNQA; used to prevent re-ordering)."""
        raise NotImplementedError("Barrier is not implemented for the CUNQA backend.")

    def _translate_operation_container(self, op: OperationContainer):
        """Recursively translate a composite OperationContainer by flattening its children."""
        
        for child in op.flatten():
            self.translate(child)

    def _translate_qsend(self, op: QSend):
        """Translate a QSend into the CUNQA teleportation-based send protocol."""
        self._cunqa_circuit.qsend(op.qubits[0], f"rank_{op.dest_rank}")

    def _translate_qrecv(self, op: QRecv):
        """Translate a QRecv into the CUNQA teleportation-based receive protocol."""
        self._cunqa_circuit.qrecv(op.qubits[0], f"rank_{op.src_rank}")

    def _translate_qscatter(self, op: QScatter):
        """Translate a QScatter into individual CUNQA send operations from the sender rank."""
        raise NotImplementedError("QScatter is not yet implemented for the CUNQA backend.")

    def _translate_qgather(self, op: QGather):
        """Translate a QGather into individual CUNQA receive operations at the receiver rank."""
        raise NotImplementedError("QGather is not yet implemented for the CUNQA backend.")

    def _translate_expose(self, op: Expose):
        """Translate an Expose into a CUNQA GHZ-based telegate exposure."""
        raise NotImplementedError("Expose is not yet implemented for the CUNQA backend.")

    def _translate_unexpose(self, op: Unexpose):
        """Translate an Unexpose into the CUNQA GHZ measurement and classical corrections."""
        raise NotImplementedError("Unexpose is not yet implemented for the CUNQA backend.")

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
            The corresponding CUNQA instruction(s).

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
    
    # TODO: Add specific methods for applying gates, measurements, etc.
    # These methods will delegate to the underlying Cunqa circuit
