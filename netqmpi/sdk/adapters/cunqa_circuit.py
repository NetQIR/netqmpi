"""
Circuit adapter for the Cunqa backend.

Implements the Circuit interface to work with Cunqa circuits.
"""
from typing import List
import os, sys
sys.path.append(os.getenv("HOME"))

from cunqa.circuit.core import CunqaCircuit

from netqmpi.sdk.core.circuit import Circuit

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
    
    def H(self, q):
        self._cunqa_circuit.h(q)
        
    def measure(self, q, c):
        self._cunqa_circuit.measure(q, c)
        
    def qsend(self, q, rank):
        self._cunqa_circuit.qsend(q, f"rank_{rank}")
    
    def qrecv(self, q, rank):
        self._cunqa_circuit.qrecv(q, f"rank_{rank}")
    
    def operations_supported(self) -> List[str]:
        """
        Returns the operations supported by this Cunqa circuit.
        
        Returns:
            List of available quantum operations.
        """
        # TODO: Implement actual Cunqa operations list
        return [
            'H', 'X', 'Y', 'Z', 'S', 'T', 'Sdg', 'Tdg',
            'CNOT', 'CZ', 'SWAP',
            'RX', 'RY', 'RZ', 'U3',
            'measure', 'reset', 'barrier'
        ]
    
    # TODO: Add specific methods for applying gates, measurements, etc.
    # These methods will delegate to the underlying Cunqa circuit
