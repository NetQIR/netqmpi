"""
High-level MPI-style communicator.

This module defines the backend-agnostic communicator interface exposed
to user application code through :attr:`Environment.comm`. Concrete
backend implementations are injected by the runtime or executor layer.

No backend-specific package (such as ``netqasm`` or ``cunqa``) is
imported here.
"""
from __future__ import annotations

from typing import Any, List, Dict, Optional
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

    def qsend(self, circuit: Circuit, qubits: List[int], dest_rank: int) -> None:
        """
        Send a qubit to the destination rank using teleportation.

        Args:
            circuit: Circuit holding the qubits to send.
            qubits: Local qubit indices to send.
            dest_rank: Destination rank.
        """
        circuit.qsend(qubits, dest_rank)

    def qrecv(self, circuit: Circuit, qubits: List[int], src_rank: int) -> None:
        """
        Receive a qubit from the source rank using teleportation.

        Args:
            circuit: Circuit receiving the qubits.
            qubits: Local qubit indices that will hold the incoming state.
            src_rank: Source rank.
        """
        circuit.qrecv(qubits, src_rank)

    def qscatter(self, circuit: Circuit, qubits: List[int], root: int) -> List[int]:
        """
        Scatter the qubits of the root across every rank.

        Collective call, like ``MPI_Scatter``: every rank of the
        communicator has to reach it. The root passes its whole buffer,
        split into one chunk per rank in rank order; every other rank
        passes the local qubits its chunk lands on. The transfers move the
        qubits, so the root is left holding only its own chunk.

        Args:
            circuit: Circuit of the calling rank.
            qubits: The whole buffer on the root, this rank's landing
                qubits elsewhere.
            root: Rank whose buffer is scattered.

        Returns:
            The local qubits holding this rank's chunk.
        """
        return circuit.qscatter(qubits, root)

    def qgather(self, circuit: Circuit, qubits: List[int], root: int) -> List[int]:
        """
        Gather the qubits of every rank into the root.

        Collective call, like ``MPI_Gather``, and the mirror image of
        :meth:`qscatter`: the root passes the whole buffer the chunks land
        on, every other rank the qubits it contributes. Here too the
        qubits are moved, so the contributors are left with theirs back in
        ``|0⟩``.

        Args:
            circuit: Circuit of the calling rank.
            qubits: The whole buffer on the root, this rank's contribution
                elsewhere.
            root: Rank the qubits are gathered into.

        Returns:
            The whole buffer on the root, this rank's contribution elsewhere.
        """
        return circuit.qgather(qubits, root)

    def expose(
        self,
        circuit: Circuit,
        qubit: Optional[int],
        ranks: List[int],
        root: Optional[int] = None,
    ) -> Optional[int]:
        """
        Open a telegate window sharing a control qubit across ranks.

        Collective call: every rank in ``[root] + ranks`` must reach it.
        The root lends the state of ``qubit`` to the other participants,
        each of which gets back the index of a local communication qubit
        carrying that control until the matching :meth:`unexpose`.

        Args:
            circuit: Circuit of the calling rank.
            qubit: Data qubit to expose. Read on the root only.
            ranks: Ranks the qubit is exposed to.
            root: Rank exposing its qubit. Defaults to the calling rank.

        Returns:
            The qubit index this rank must use as control, or ``None`` if
            it does not take part in the window.
        """
        return circuit.expose(qubit, ranks, root=root)

    def unexpose(
        self,
        circuit: Circuit,
        ranks: List[int],
        root: Optional[int] = None,
    ) -> None:
        """
        Close the telegate window opened by the matching :meth:`expose`.

        Args:
            circuit: Circuit of the calling rank.
            ranks: Ranks the qubit was exposed to.
            root: Rank that exposed its qubit. Defaults to the calling rank.
        """
        circuit.unexpose(ranks, root=root)


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