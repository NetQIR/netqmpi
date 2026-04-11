"""
High-level MPI-style communicator.

This module defines the backend-agnostic communicator interface exposed
to user application code through :attr:`Environment.comm`. Concrete
backend implementations are injected by the runtime or executor layer.

No backend-specific package (such as ``netqasm`` or ``cunqa``) is
imported here.
"""
from __future__ import annotations

from typing import Any, List, Dict
from abc import ABC, abstractmethod

from netqmpi.sdk.circuit import Circuit
from netqmpi.runtime.run_config import RunConfig

class QMPICommunicator(ABC):
    """
    Backend-agnostic facade for rank-based communication.

    This class exposes the communication interface required by user code
    and by :class:`~netqmpi.sdk.circuit.Circuit`, while delegating the
    backend-specific behavior to concrete subclasses.

    It provides:

    - ``rank`` and ``size`` properties.
    - Context-manager support for connection lifecycle handling.
    - Utility helpers for rank naming and neighbor traversal.
    """

    def __init__(self, rank: int, size: int) -> None:
        """
        Initialize the communicator.

        Args:
            rank: Numeric index of the current rank.
            size: Total number of ranks in the communicator.
        """
        self._rank = rank
        self._size = size
        self.circuits: List[Circuit] = []
        self.results: Dict = {}

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def rank(self) -> int:
        """
        Return the numeric index of the current rank.

        Returns:
            The current rank.
        """
        return self._rank

    @property
    def size(self) -> int:
        """
        Return the total number of ranks in the communicator.

        Returns:
            The communicator size.
        """
        return self._size

    # ------------------------------------------------------------------
    # Context manager (wraps the backend connection lifecycle)
    # ------------------------------------------------------------------

    @abstractmethod
    def __enter__(self) -> Any:
        """
        Enter the communicator context.

        Returns:
            A backend-specific object or the communicator itself.
        """
        pass

    @abstractmethod
    def __exit__(self, exc_type, exc_val, exc_tb) -> Any:
        """
        Exit the communicator context.

        Args:
            exc_type: Exception type, if one was raised.
            exc_val: Exception instance, if one was raised.
            exc_tb: Traceback, if one was raised.

        Returns:
            A backend-specific result from the context manager exit.
        """
        pass


    # ------------------------------------------------------------------
    # Quantum operations
    # ------------------------------------------------------------------

    def qsend(self, circuit, qubits: List[int], dest_rank: int):
        """
        Send a qubit to the destination rank using teleportation.
        """
        circuit.qsend(qubits, dest_rank)

    def qrecv(self, circuit, qubits: List[int], src_rank: int) -> List[int]:
        """
        Receive a qubit from the source rank using teleportation.
        """
        return circuit.qrecv(qubits, src_rank)

    def qscatter(self, qubits: List[int], rank_sender: int) -> List[int]:
        pass

    def qgather(self, qubits: List[int], rank_recv: int) -> List[int]:
        pass

    def expose(self, qubits: List[int], rank: int = 0):
        """
        Expose qubits to the network.
        :param qubits: List of qubits to expose.
        :param rank: Exposer rank
        """
        pass
    def unexpose(self, rank: int = 0):
        """
        Unexpose qubits from the network.
        :param rank: Exposer rank
        :return: None
        """
        pass


    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def get_rank_name(self, rank: int) -> str:
        """
        Return the canonical string name for a rank.

        Args:
            rank: Numeric rank identifier.

        Returns:
            The canonical rank name.
        """
        return f"rank_{rank}"

    def get_next_rank(self, rank: int) -> int:
        """
        Return the next rank in cyclic order.

        Args:
            rank: Reference rank.

        Returns:
            The next rank modulo the communicator size.
        """
        return (rank + 1) % self._size

    def get_prev_rank(self, rank: int) -> int:
        """
        Return the previous rank in cyclic order.

        Args:
            rank: Reference rank.

        Returns:
            The previous rank modulo the communicator size.
        """
        return (rank - 1) % self._size