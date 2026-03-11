"""
Adapter for the CUNQA backend circuit.

This module implements the ``Circuit`` interface for CUNQA circuits.
"""
import os, sys
sys.path.append(os.getenv("HOME"))

from typing import Any

from cunqa.circuit.core import CunqaCircuit

from netqmpi.sdk.circuit import Circuit
from netqmpi.sdk.operations import (
    Operation,
    Gate, ControlledGate, ClassicalControlledGate,
    Measure, Reset, Barrier,
    OperationContainer,
    QSend, QRecv, QScatter, QGather, Expose, Unexpose,
)
from netqmpi.runtime.adapters.cunqa.cunqa_communicator import CunqaCommunicator

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

        Args:
            num_qubits: Number of qubits in the circuit.
            num_clbits: Number of classical bits in the circuit.
            comm: Communicator associated with the circuit.
        """
        super().__init__(num_qubits, num_clbits, comm)
        # TODO: Initialize the underlying Cunqa circuit
        self._cunqa_circuit = CunqaCircuit(num_qubits, num_clbits, id=f"rank_{self._comm.rank}")
    
    def _translate_gate(self, op: Gate):
        """
        Translate a single-qubit unitary gate into a CUNQA instruction.

        Args:
            op: Gate operation to translate.
        """
        
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
        """
        Translate a controlled quantum gate into a CUNQA instruction.

        Args:
            op: Controlled gate operation to translate.
        """
        
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
        """
        Translate a classically controlled gate into a CUNQA instruction.

        Args:
            op: Classically controlled gate operation to translate.

        Raises:
            NotImplementedError: Always, because this operation is not yet supported.
        """
        raise NotImplementedError("ClassicalControlledGate is not yet implemented for the CUNQA backend.")

    def _translate_measure(self, op: Measure):
        """
        Translate a measurement operation into a CUNQA instruction.

        Args:
            op: Measurement operation to translate.
        """
        self._cunqa_circuit.measure(op.qubits[0], op.cbit)

    def _translate_reset(self, op: Reset):
        """
        Translate a reset operation into a CUNQA instruction.

        Args:
            op: Reset operation to translate.
        """
        self._cunqa_circuit.reset(op.qubits[0]) # TODO: Mirar si resetea con varios también

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

        Args:
            op: Operation container to translate.
        """
        
        for child in op.flatten():
            self.translate(child)

    def _translate_qsend(self, op: QSend):
        """
        Translate a quantum send operation into a CUNQA instruction.

        Args:
            op: Quantum send operation to translate.
        """
        self._cunqa_circuit.qsend(op.qubits[0], f"rank_{op.dest_rank}")

    def _translate_qrecv(self, op: QRecv):
        """
        Translate a quantum receive operation into a CUNQA instruction.

        Args:
            op: Quantum receive operation to translate.
        """
        self._cunqa_circuit.qrecv(op.qubits[0], f"rank_{op.src_rank}")

    def _translate_qscatter(self, op: QScatter):
        """
        Translate a quantum scatter operation into CUNQA instructions.

        Args:
            op: Quantum scatter operation to translate.

        Raises:
            NotImplementedError: Always, because this operation is not yet supported.
        """
        raise NotImplementedError("QScatter is not yet implemented for the CUNQA backend.")

    def _translate_qgather(self, op: QGather):
        """
        Translate a quantum gather operation into CUNQA instructions.

        Args:
            op: Quantum gather operation to translate.

        Raises:
            NotImplementedError: Always, because this operation is not yet supported.
        """
        raise NotImplementedError("QGather is not yet implemented for the CUNQA backend.")

    def _translate_expose(self, op: Expose):
        """
        Translate an expose operation into a CUNQA instruction.

        Args:
            op: Expose operation to translate.

        Raises:
            NotImplementedError: Always, because this operation is not yet supported.
        """
        raise NotImplementedError("Expose is not yet implemented for the CUNQA backend.")

    def _translate_unexpose(self, op: Unexpose):
        """
        Translate an unexpose operation into a CUNQA instruction.

        Args:
            op: Unexpose operation to translate.

        Raises:
            NotImplementedError: Always, because this operation is not yet supported.
        """
        raise NotImplementedError("Unexpose is not yet implemented for the CUNQA backend.")

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

    def build(self):
        pass