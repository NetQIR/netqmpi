"""
Base abstraction for quantum circuits.

Defines the contract that all circuit adapters must follow.
"""
from abc import ABC, abstractmethod
from typing import List


class Circuit(ABC):
    """
    Abstract class representing a quantum circuit.
    
    Defines the common interface that all quantum circuit adapters
    must implement, regardless of the backend used.
    
    Attributes:
        num_qubits (int): Number of qubits in the circuit.
        num_clbits (int): Number of classical bits in the circuit.
    """
    
    def __init__(self, num_qubits: int, num_clbits: int):
        """
        Initialize the quantum circuit.
        
        Args:
            num_qubits: Number of qubits in the circuit.
            num_clbits: Number of classical bits in the circuit.
        """
        self._num_qubits = num_qubits
        self._num_clbits = num_clbits
    
    @property
    def num_qubits(self) -> int:
        """Returns the number of qubits in the circuit."""
        return self._num_qubits
    
    @property
    def num_clbits(self) -> int:
        """Returns the number of classical bits in the circuit."""
        return self._num_clbits
    
    @abstractmethod
    def operations_supported(self) -> List[str]:
        """
        Returns the list of operations supported by this circuit.
        
        Returns:
            List of supported quantum operation names.
            Examples: ['H', 'CNOT', 'X', 'Y', 'Z', 'T', 'S', 'measure', 'reset']
        """
        pass
