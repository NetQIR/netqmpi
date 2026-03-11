"""
High-level MPI-style communicator.

This module defines the backend-agnostic communicator interface exposed
to user application code through :attr:`Environment.comm`. Concrete
backend implementations are injected by the runtime or executor layer.

No backend-specific package (such as ``netqasm`` or ``cunqa``) is
imported here.
"""
from __future__ import annotations

from typing import Any
from abc import ABC, abstractmethod

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