"""
Abstract base class for backend-specific communicators.

Every quantum backend must provide a concrete subclass that manages
the low-level resources required for inter-rank communication:
connections, EPR sockets, classical sockets, etc.

This module deliberately imports nothing from any concrete backend
(``netqasm``, ``cunqa``, …).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List


class BaseCommunicator(ABC):
    """
    Backend-agnostic contract for a single-rank communicator.

    A concrete implementation (e.g. ``NetQASMCommunicator``) is created
    by the runtime and injected into :class:`QMPICommunicator`.
    """

    def __init__(self, rank: int, size: int) -> None:
        self._rank = rank
        self._size = size

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def rank(self) -> int:
        return self._rank

    @property
    def size(self) -> int:
        return self._size

    # ------------------------------------------------------------------
    # Context manager (wraps the backend connection lifecycle)
    # ------------------------------------------------------------------

    @abstractmethod
    def __enter__(self) -> BaseCommunicator:
        pass

    @abstractmethod
    def __exit__(self, exc_type, exc_val, exc_tb) -> Any:
        pass

    # ------------------------------------------------------------------
    # Connection-level helpers
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def connection(self) -> Any:
        """Return the underlying backend connection object."""

    @abstractmethod
    def flush(self) -> None:
        """Flush pending operations on the backend connection."""

    @abstractmethod
    def create_qubit(self) -> Any:
        """Allocate a fresh qubit on this connection."""

    # ------------------------------------------------------------------
    # Classical / EPR socket access
    # ------------------------------------------------------------------

    @abstractmethod
    def get_socket(self, my_rank: int, other_rank: int) -> Any:
        """Return (or lazily create) a classical socket to *other_rank*."""

    @abstractmethod
    def get_epr_socket(self, my_rank: int, other_rank: int) -> Any:
        """Return the EPR socket connecting *my_rank* ↔ *other_rank*."""

    @abstractmethod
    def get_epr_sockets_list(self) -> List[Any]:
        """Return all EPR sockets for this rank as a flat list."""

    # ------------------------------------------------------------------
    # Collective helpers
    # ------------------------------------------------------------------

    @abstractmethod
    def create_ghz(self) -> Any:
        """Create a GHZ state across all ranks and return the local qubit."""

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def get_rank_name(self, rank: int) -> str:
        return f"rank_{rank}"

    def get_next_rank(self, rank: int) -> int:
        return (rank + 1) % self._size

    def get_prev_rank(self, rank: int) -> int:
        return (rank - 1) % self._size
