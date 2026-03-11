"""
High-level MPI-style communicator — **backend-agnostic**.

:class:`QMPICommunicator` is the object that user application code sees
(through :attr:`Environment.comm`).  It delegates every backend-specific
operation to a concrete :class:`~netqmpi.sdk.communicator.base.BaseCommunicator`
that is injected at construction time by the runtime/executor layer.

**No** concrete backend (``netqasm``, ``cunqa``, …) is imported here.
"""
from __future__ import annotations

from typing import Any
from abc import ABC, abstractmethod

class QMPICommunicator(ABC):
    """
    Thin facade over a backend-specific :class:`BaseCommunicator`.

    Exposes only the operations that user-level code and
    :class:`~netqmpi.sdk.circuit.Circuit` need:

    * ``rank`` / ``size`` properties.
    * ``connection``, ``flush()``, ``create_qubit()`` — connection helpers.
    * ``get_socket()``, ``get_epr_socket()``, ``get_epr_sockets_list()``
      — socket access.
    * ``create_ghz()`` — collective helper.
    * Context-manager protocol (``with comm: …``).
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
    def __enter__(self) -> Any:
        pass

    @abstractmethod
    def __exit__(self, exc_type, exc_val, exc_tb) -> Any:
        pass

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def get_rank_name(self, rank: int) -> str:
        return f"rank_{rank}"

    def get_next_rank(self, rank: int) -> int:
        return (rank + 1) % self._size

    def get_prev_rank(self, rank: int) -> int:
        return (rank - 1) % self._size