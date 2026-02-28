"""
Base abstraction for quantum circuit executors.

Defines the contract that all quantum backend adapters must follow.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, List


class Executor(ABC):
    """
    Abstract class representing a quantum circuit executor.
    
    Implements the Factory Method pattern for circuit creation
    and defines the interface that all backend adapters must follow.
    
    Attributes:
        size (int): Number of nodes or resources available in the executor.
        config (Dict[str, Any]): Executor-specific configuration.
    """
    
    def __init__(self, size: int, config: Dict[str, Any] = None):
        """
        Initialize the executor.
        
        Args:
            size: Number of available nodes or resources.
            config: Dictionary with backend-specific configuration.
        """
        self._size = size
        self._config = config or {}
    
    @property
    def size(self) -> int:
        """Returns the size (number of nodes/resources) of the executor."""
        return self._size
    
    @property
    def config(self) -> Dict[str, Any]:
        """Returns the executor configuration."""
        return self._config
    
    @abstractmethod
    def operations_supported(self) -> List[str]:
        """
        Returns the list of operations supported by this executor.
        
        Returns:
            List of supported quantum operation names.
            Examples: ['H', 'CNOT', 'X', 'Y', 'Z', 'T', 'S', 'measure']
        """
        pass
    
    @abstractmethod
    def create_circuit(self, num_qubits: int, num_clbits: int):
        """
        Factory Method: Creates a backend-specific quantum circuit.
        
        Args:
            num_qubits: Number of qubits in the circuit.
            num_clbits: Number of classical bits in the circuit.
            
        Returns:
            Backend-specific Circuit instance.
        """
        pass
