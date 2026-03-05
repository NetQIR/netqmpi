"""
High-level MPI-style communicator — **backend-agnostic**.

:class:`QMPICommunicator` is the object that user application code sees
(through :attr:`Environment.comm`).  It delegates every backend-specific
operation to a concrete :class:`~netqmpi.sdk.communicator.base.BaseCommunicator`
that is injected at construction time by the runtime/executor layer.

**No** concrete backend (``netqasm``, ``cunqa``, …) is imported here.
"""
from __future__ import annotations

from typing import Any, List

from netqmpi.sdk.communicator.base import BaseCommunicator


class QMPICommunicator:
    """
    Thin facade over a backend-specific :class:`BaseCommunicator`.

    Exposes only the operations that user-level code and
    :class:`~netqmpi.sdk.core.circuit.Circuit` need:

    * ``rank`` / ``size`` properties.
    * ``connection``, ``flush()``, ``create_qubit()`` — connection helpers.
    * ``get_socket()``, ``get_epr_socket()``, ``get_epr_sockets_list()``
      — socket access.
    * ``create_ghz()`` — collective helper.
    * Context-manager protocol (``with comm: …``).
    """

    def __init__(self, backend: BaseCommunicator) -> None:
        self._backend = backend

    # ------------------------------------------------------------------
    # Properties (proxied from the backend)
    # ------------------------------------------------------------------

    @property
    def rank(self) -> int:
        return self._backend.rank

    @property
    def size(self) -> int:
        return self._backend.size

    @property
    def connection(self) -> Any:
        """The live backend connection (e.g. ``NetQASMConnection``)."""
        return self._backend.connection

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self):
        self._backend.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return self._backend.__exit__(exc_type, exc_val, exc_tb)

    # ------------------------------------------------------------------
    # Connection helpers
    # ------------------------------------------------------------------

    def flush(self) -> None:
        """Flush pending operations on the backend connection."""
        self._backend.flush()

    def create_qubit(self) -> Any:
        """Allocate a fresh qubit on this connection."""
        return self._backend.create_qubit()

    # ------------------------------------------------------------------
    # Socket access
    # ------------------------------------------------------------------

    def get_socket(self, my_rank: int, other_rank: int) -> Any:
        return self._backend.get_socket(my_rank, other_rank)

    def get_epr_socket(self, my_rank: int, other_rank: int) -> Any:
        return self._backend.get_epr_socket(my_rank, other_rank)

    def get_epr_sockets_list(self) -> List[Any]:
        return self._backend.get_epr_sockets_list()

    # ------------------------------------------------------------------
    # Collective helpers
    # ------------------------------------------------------------------

    def create_ghz(self) -> Any:
        return self._backend.create_ghz()

    # ------------------------------------------------------------------
    # Utility (convenience wrappers)
    # ------------------------------------------------------------------

    def get_rank(self) -> int:
        return self.rank

    def get_size(self) -> int:
        return self.size

    def get_next_rank(self, rank: int) -> int:
        return self._backend.get_next_rank(rank)

    def get_prev_rank(self, rank: int) -> int:
        return self._backend.get_prev_rank(rank)