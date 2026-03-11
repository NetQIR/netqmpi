"""
Inter-rank communication primitives as first-class Operations.

Each class encodes the *intent* of a distributed quantum operation.
The concrete backend adapter is responsible for implementing the
protocol (e.g. teleportation, GHZ) inside ``Circuit.translate(op)``.

All classes inherit from :class:`~netqmpi.sdk.operations.Operation`,
so they flow through :class:`~netqmpi.sdk.operations.OperationContainer`
and ``flatten()`` exactly like any gate or measurement.
"""
from __future__ import annotations
from typing import List

from netqmpi.sdk.operations.operation import Operation


class QSend(Operation):
    """
    Send local qubits to a remote rank.

    The protocol (e.g. teleportation) is chosen by the backend adapter.

    Attributes:
        qubits    (List[int]): Local qubit indices to send (consumed).
        dest_rank (int):       Destination rank.
    """

    def __init__(self, qubits: List[int], dest_rank: int) -> None:
        """
        Args:
            qubits:    Local qubit indices to send.
            dest_rank: Rank of the receiving process.

        Raises:
            ValueError: If *qubits* is empty or *dest_rank* is negative.
        """
        if not qubits:
            raise ValueError("qubits must be a non-empty list.")
        if dest_rank < 0:
            raise ValueError("dest_rank must be a non-negative integer.")
        super().__init__(qubits)
        self._dest_rank = dest_rank

    @property
    def dest_rank(self) -> int:
        """Destination rank."""
        return self._dest_rank

    def __repr__(self) -> str:
        return f"QSend(qubits={self._qubits}, dest_rank={self._dest_rank})"

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, QSend)
            and self._qubits == other._qubits
            and self._dest_rank == other._dest_rank
        )

    def __hash__(self) -> int:
        return hash(("QSend", tuple(self._qubits), self._dest_rank))


class QRecv(Operation):
    """
    Receive qubits from a remote rank into local qubit slots.

    Attributes:
        qubits   (List[int]): Local qubit indices where the state will land.
        src_rank (int):       Source rank.
    """

    def __init__(self, qubits: List[int], src_rank: int) -> None:
        """
        Args:
            qubits:   Local qubit indices to receive into.
                      ``len(qubits)`` determines how many qubits are expected.
            src_rank: Rank of the sending process.

        Raises:
            ValueError: If *qubits* is empty or *src_rank* is negative.
        """
        if not qubits:
            raise ValueError("qubits must be a non-empty list.")
        if src_rank < 0:
            raise ValueError("src_rank must be a non-negative integer.")
        super().__init__(qubits)
        self._src_rank = src_rank

    @property
    def src_rank(self) -> int:
        """Source rank."""
        return self._src_rank

    @property
    def n_qubits(self) -> int:
        """Number of qubits to receive."""
        return len(self._qubits)

    def __repr__(self) -> str:
        return f"QRecv(qubits={self._qubits}, src_rank={self._src_rank})"

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, QRecv)
            and self._qubits == other._qubits
            and self._src_rank == other._src_rank
        )

    def __hash__(self) -> int:
        return hash(("QRecv", tuple(self._qubits), self._src_rank))


class QScatter(Operation):
    """
    Scatter qubits from *sender_rank* across all ranks.

    The adapter splits ``qubits`` into chunks and teleports each chunk
    to its target rank.

    Attributes:
        qubits      (List[int]): Local qubit indices being scattered.
        sender_rank (int):       Rank that owns the full qubit list.
    """

    def __init__(self, qubits: List[int], sender_rank: int) -> None:
        """
        Args:
            qubits:      Local qubit indices to scatter.
            sender_rank: Rank of the scattering process.

        Raises:
            ValueError: If *qubits* is empty or *sender_rank* is negative.
        """
        if not qubits:
            raise ValueError("qubits must be a non-empty list.")
        if sender_rank < 0:
            raise ValueError("sender_rank must be a non-negative integer.")
        super().__init__(qubits)
        self._sender_rank = sender_rank

    @property
    def sender_rank(self) -> int:
        """Rank that scatters the qubits."""
        return self._sender_rank

    def __repr__(self) -> str:
        return f"QScatter(qubits={self._qubits}, sender_rank={self._sender_rank})"

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, QScatter)
            and self._qubits == other._qubits
            and self._sender_rank == other._sender_rank
        )

    def __hash__(self) -> int:
        return hash(("QScatter", tuple(self._qubits), self._sender_rank))


class QGather(Operation):
    """
    Gather qubits from all ranks into *recv_rank*.

    Attributes:
        qubits    (List[int]): Local qubit indices being contributed.
        recv_rank (int):       Rank that will hold all gathered qubits.
    """

    def __init__(self, qubits: List[int], recv_rank: int) -> None:
        """
        Args:
            qubits:    Local qubit indices to contribute to the gather.
            recv_rank: Rank of the gathering process.

        Raises:
            ValueError: If *qubits* is empty or *recv_rank* is negative.
        """
        if not qubits:
            raise ValueError("qubits must be a non-empty list.")
        if recv_rank < 0:
            raise ValueError("recv_rank must be a non-negative integer.")
        super().__init__(qubits)
        self._recv_rank = recv_rank

    @property
    def recv_rank(self) -> int:
        """Rank that gathers the qubits."""
        return self._recv_rank

    def __repr__(self) -> str:
        return f"QGather(qubits={self._qubits}, recv_rank={self._recv_rank})"

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, QGather)
            and self._qubits == other._qubits
            and self._recv_rank == other._recv_rank
        )

    def __hash__(self) -> int:
        return hash(("QGather", tuple(self._qubits), self._recv_rank))


class Expose(Operation):
    """
    Expose local qubits to the network via a shared GHZ state (telegate).

    After this operation, the exposed qubits can be operated on remotely
    by other ranks.  Must be paired with an :class:`Unexpose`.

    Attributes:
        qubits (List[int]): Local qubit indices to expose.
        rank   (int):       Rank that acts as the exposer (default: 0).
    """

    def __init__(self, qubits: List[int], rank: int = 0) -> None:
        """
        Args:
            qubits: Local qubit indices to expose.
            rank:   Exposer rank (default: 0).

        Raises:
            ValueError: If *qubits* is empty or *rank* is negative.
        """
        if not qubits:
            raise ValueError("qubits must be a non-empty list.")
        if rank < 0:
            raise ValueError("rank must be a non-negative integer.")
        super().__init__(qubits)
        self._rank = rank

    @property
    def rank(self) -> int:
        """Exposer rank."""
        return self._rank

    def __repr__(self) -> str:
        return f"Expose(qubits={self._qubits}, rank={self._rank})"

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, Expose)
            and self._qubits == other._qubits
            and self._rank == other._rank
        )

    def __hash__(self) -> int:
        return hash(("Expose", tuple(self._qubits), self._rank))


class Unexpose(Operation):
    """
    Terminate a previously opened :class:`Expose` window.

    Does not act on any qubit directly; the adapter handles the
    GHZ measurement and classical corrections.

    Attributes:
        rank (int): Exposer rank that opened the :class:`Expose` (default: 0).
    """

    def __init__(self, rank: int = 0) -> None:
        """
        Args:
            rank: Exposer rank (default: 0).

        Raises:
            ValueError: If *rank* is negative.
        """
        if rank < 0:
            raise ValueError("rank must be a non-negative integer.")
        super().__init__([])
        self._rank = rank

    @property
    def rank(self) -> int:
        """Exposer rank."""
        return self._rank

    def __repr__(self) -> str:
        return f"Unexpose(rank={self._rank})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Unexpose) and self._rank == other._rank

    def __hash__(self) -> int:
        return hash(("Unexpose", self._rank))
