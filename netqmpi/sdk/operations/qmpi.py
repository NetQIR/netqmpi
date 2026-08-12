"""
Inter-rank communication primitives as first-class Operations.

Each class encodes the *intent* of a distributed quantum operation.
The concrete backend adapter is responsible for implementing the
protocol (e.g. teleportation, GHZ) inside ``Circuit.translate(op)``.

All classes inherit from :class:`~netqmpi.sdk.operations.Operation`,
so they flow through :class:`~netqmpi.sdk.operations.OperationContainer`
and ``flatten()`` exactly like any gate or measurement.

Three families of primitives live here:

- *Point-to-point* operations (:class:`QSend`, :class:`QRecv`), which
  every rank can translate on its own because the backend emits an
  independent instruction block on each side.
- *Rooted transfers* (:class:`RootedTransfer` subclasses such as
  :class:`QScatter` and :class:`QGather`), which every rank must call but
  which expand, on each of them, into the point-to-point transfers above.
  They are containers holding those transfers, so a backend that can send
  and receive a qubit gets them for free.
- *Collective* operations (:class:`CollectiveOperation` subclasses such
  as :class:`Expose` and :class:`Unexpose`), whose backend expansion
  writes instructions into **every** participating circuit at once and
  therefore can only be emitted when all participants have reached the
  matching call.  Each participant carries the resources it contributes
  to the protocol (a communication-qubit slot, protocol classical bits)
  plus a ``tag`` that is identical across ranks, so the runtime can pair
  the calls up without any trace-time communication.
"""
from __future__ import annotations
from typing import List, Optional

from netqmpi.sdk.operations.container import OperationContainer
from netqmpi.sdk.operations.operation import Operation


class CollectiveOperation(Operation):
    """
    Base class for operations that must be expanded jointly by all ranks.

    A collective operation is recorded independently by every
    participating rank, but the backend can only translate it once all
    participants are sitting on the matching call.  Two records match
    when they have the same type and the same :attr:`tag`.

    Attributes:
        rank   (int):       Rank whose circuit holds this record.
        ranks  (List[int]): Participating ranks, in protocol order.
        tag    (str):       Identifier shared by every participant.
    """

    def __init__(
        self,
        qubits: List[int],
        rank: int,
        ranks: List[int],
        tag: str,
    ) -> None:
        """
        Args:
            qubits: Local qubit indices the record acts on.
            rank:   Rank owning this record.
            ranks:  Participating ranks, in protocol order.
            tag:    Identifier shared by every participant.

        Raises:
            ValueError: If *ranks* is empty or *tag* is not a string.
        """
        if not isinstance(ranks, list) or not ranks:
            raise ValueError("ranks must be a non-empty list of integers.")
        if any(not isinstance(r, int) or r < 0 for r in ranks):
            raise ValueError("ranks must contain non-negative integers.")
        if not isinstance(tag, str) or not tag:
            raise ValueError("tag must be a non-empty string.")
        super().__init__(qubits)
        self._rank = rank
        self._ranks = list(ranks)
        self._tag = tag

    @property
    def rank(self) -> int:
        """Rank owning this record."""
        return self._rank

    @property
    def ranks(self) -> List[int]:
        """Participating ranks, in protocol order."""
        return list(self._ranks)

    @property
    def tag(self) -> str:
        """Identifier shared by every participant."""
        return self._tag

    def matches(self, other: object) -> bool:
        """
        Report whether *other* is this rank's counterpart of the same call.

        Args:
            other: Candidate record held by another rank.

        Returns:
            ``True`` if both records belong to the same collective call.
        """
        return (
            type(other) is type(self)
            and other.tag == self._tag           # type: ignore[attr-defined]
            and other.ranks == self._ranks       # type: ignore[attr-defined]
        )


class QSend(Operation):
    """
    Send local qubits to a remote rank.

    The protocol (e.g. teleportation) is chosen by the backend adapter.

    Attributes:
        qubits    (List[int]): Local qubit indices to send (consumed).
        dest_rank (int):       Destination rank.
        comm_slot (int):       Communication-qubit slot reserved locally.
        clbits    (List[int]): Protocol classical bits reserved locally.
        tag       (str):       Identifier shared with the matching :class:`QRecv`.
    """

    def __init__(
        self,
        qubits: List[int],
        dest_rank: int,
        comm_slot: Optional[int] = None,
        clbits: Optional[List[int]] = None,
        tag: Optional[str] = None,
    ) -> None:
        """
        Args:
            qubits:    Local qubit indices to send.
            dest_rank: Rank of the receiving process.
            comm_slot: Communication-qubit slot reserved for the transfer.
            clbits:    Protocol classical bits reserved for the transfer.
            tag:       Identifier shared with the matching :class:`QRecv`.

        Raises:
            ValueError: If *qubits* is empty or *dest_rank* is negative.
        """
        if not qubits:
            raise ValueError("qubits must be a non-empty list.")
        if dest_rank < 0:
            raise ValueError("dest_rank must be a non-negative integer.")
        super().__init__(qubits)
        self._dest_rank = dest_rank
        self._comm_slot = comm_slot
        self._clbits = list(clbits) if clbits is not None else []
        self._tag = tag

    @property
    def dest_rank(self) -> int:
        """Destination rank."""
        return self._dest_rank

    @property
    def comm_slot(self) -> Optional[int]:
        """Communication-qubit slot reserved for the transfer."""
        return self._comm_slot

    @property
    def clbits(self) -> List[int]:
        """Protocol classical bits reserved for the transfer."""
        return list(self._clbits)

    @property
    def tag(self) -> Optional[str]:
        """Identifier shared with the matching :class:`QRecv`."""
        return self._tag

    def __repr__(self) -> str:
        return f"QSend(qubits={self._qubits}, dest_rank={self._dest_rank}, tag={self._tag})"

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, QSend)
            and self._qubits == other._qubits
            and self._dest_rank == other._dest_rank
            and self._tag == other._tag
        )

    def __hash__(self) -> int:
        return hash(("QSend", tuple(self._qubits), self._dest_rank, self._tag))


class QRecv(Operation):
    """
    Receive qubits from a remote rank into local qubit slots.

    Attributes:
        qubits    (List[int]): Local qubit indices where the state will land.
        src_rank  (int):       Source rank.
        comm_slot (int):       Communication-qubit slot reserved locally.
        clbits    (List[int]): Protocol classical bits reserved locally.
        tag       (str):       Identifier shared with the matching :class:`QSend`.
    """

    def __init__(
        self,
        qubits: List[int],
        src_rank: int,
        comm_slot: Optional[int] = None,
        clbits: Optional[List[int]] = None,
        tag: Optional[str] = None,
    ) -> None:
        """
        Args:
            qubits:    Local qubit indices to receive into.
                       ``len(qubits)`` determines how many qubits are expected.
            src_rank:  Rank of the sending process.
            comm_slot: Communication-qubit slot reserved for the transfer.
            clbits:    Protocol classical bits reserved for the transfer.
            tag:       Identifier shared with the matching :class:`QSend`.

        Raises:
            ValueError: If *qubits* is empty or *src_rank* is negative.
        """
        if not qubits:
            raise ValueError("qubits must be a non-empty list.")
        if src_rank < 0:
            raise ValueError("src_rank must be a non-negative integer.")
        super().__init__(qubits)
        self._src_rank = src_rank
        self._comm_slot = comm_slot
        self._clbits = list(clbits) if clbits is not None else []
        self._tag = tag

    @property
    def src_rank(self) -> int:
        """Source rank."""
        return self._src_rank

    @property
    def n_qubits(self) -> int:
        """Number of qubits to receive."""
        return len(self._qubits)

    @property
    def comm_slot(self) -> Optional[int]:
        """Communication-qubit slot reserved for the transfer."""
        return self._comm_slot

    @property
    def clbits(self) -> List[int]:
        """Protocol classical bits reserved for the transfer."""
        return list(self._clbits)

    @property
    def tag(self) -> Optional[str]:
        """Identifier shared with the matching :class:`QSend`."""
        return self._tag

    def __repr__(self) -> str:
        return f"QRecv(qubits={self._qubits}, src_rank={self._src_rank}, tag={self._tag})"

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, QRecv)
            and self._qubits == other._qubits
            and self._src_rank == other._src_rank
            and self._tag == other._tag
        )

    def __hash__(self) -> int:
        return hash(("QRecv", tuple(self._qubits), self._src_rank, self._tag))


class RootedTransfer(OperationContainer):
    """
    Base class for the rooted collectives built out of teledata.

    :class:`QScatter` and :class:`QGather` move qubits between one root
    and every other rank. Because a quantum state cannot be copied, they
    can only be built out of transfers that *consume* the source qubit:
    each of them expands into one :class:`QSend` per qubit leaving this
    rank and one :class:`QRecv` per qubit arriving, and the record keeps
    those children so the backends translate the collective through the
    very same point-to-point path they already implement.

    These records are deliberately **not** :class:`CollectiveOperation`
    instances. A collective in that sense is one whose backend expansion
    writes into every participating circuit at once, and so has to wait
    for all the ranks; here each side is an ordinary point-to-point
    transfer that the runtime pairs up by tag, so every rank can be
    translated on its own.

    Attributes:
        rank   (int):       Rank owning this record.
        root   (int):       Rank the qubits are scattered from / gathered into.
        ranks  (List[int]): Participating ranks, in rank order.
        qubits (List[int]): Local qubits taking part: the whole buffer on
                            the root, this rank's chunk elsewhere.
    """

    def __init__(
        self,
        rank: int,
        root: int,
        ranks: List[int],
        qubits: List[int],
    ) -> None:
        """
        Args:
            rank:   Rank owning this record.
            root:   Rank the qubits are scattered from / gathered into.
            ranks:  Participating ranks, in rank order.
            qubits: Local qubits this rank contributes or receives.

        Raises:
            ValueError: If the participant list is empty, if it does not
                contain both *rank* and *root*, or if *qubits* is empty.
        """
        if not isinstance(ranks, list) or not ranks:
            raise ValueError("ranks must be a non-empty list of integers.")
        if any(not isinstance(r, int) or r < 0 for r in ranks):
            raise ValueError("ranks must contain non-negative integers.")
        if root not in ranks:
            raise ValueError(f"the root ({root}) must be one of the ranks {ranks}.")
        if rank not in ranks:
            raise ValueError(f"rank {rank} does not take part in {ranks}.")
        if not isinstance(qubits, list) or not qubits:
            raise ValueError("qubits must be a non-empty list.")
        super().__init__()
        self._qubits = list(qubits)
        self._rank = rank
        self._root = root
        self._ranks = list(ranks)

    @property
    def qubits(self) -> List[int]:
        """Local qubits taking part in the transfer."""
        return list(self._qubits)

    @property
    def rank(self) -> int:
        """Rank owning this record."""
        return self._rank

    @property
    def root(self) -> int:
        """Rank the qubits are scattered from or gathered into."""
        return self._root

    @property
    def ranks(self) -> List[int]:
        """Participating ranks, in rank order."""
        return list(self._ranks)

    @property
    def is_root(self) -> bool:
        """Whether the rank owning this record is the root."""
        return self._rank == self._root

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(rank={self._rank}, root={self._root}, "
            f"ranks={self._ranks}, qubits={self._qubits})"
        )

    def __eq__(self, other: object) -> bool:
        return (
            type(other) is type(self)
            and self._rank == other._rank        # type: ignore[attr-defined]
            and self._root == other._root        # type: ignore[attr-defined]
            and self._ranks == other._ranks      # type: ignore[attr-defined]
            and self._qubits == other._qubits    # type: ignore[attr-defined]
        )

    def __hash__(self) -> int:
        return hash((type(self).__name__, self._rank, self._root,
                     tuple(self._ranks), tuple(self._qubits)))


class QScatter(RootedTransfer):
    """
    Scatter the qubits held by the root across every rank.

    This is ``MPI_Scatter`` with qubits instead of bytes: the root's
    buffer is split into one chunk per rank, in rank order, and rank *r*
    ends up holding chunk *r*. Handing a qubit over means *moving* it, so
    every chunk but the root's own is teleported away and the root is
    left with those qubits back in ``|0⟩``: after the call the data it
    scattered lives only on the receivers.

    Attributes:
        rank   (int):       Rank owning this record.
        root   (int):       Rank whose buffer is scattered.
        ranks  (List[int]): Participating ranks, in rank order.
        qubits (List[int]): Local qubits taking part: the whole buffer on
                            the root, this rank's chunk elsewhere.
    """

    @property
    def sender_rank(self) -> int:
        """Rank that scatters the qubits."""
        return self._root


class QGather(RootedTransfer):
    """
    Gather the qubits of every rank into the root.

    The mirror image of :class:`QScatter`, and ``MPI_Gather`` with qubits
    instead of bytes: rank *r* contributes its chunk, and the root ends up
    holding all of them in rank order. Here too the transfer moves the
    qubits, so once the call is over the contributors are left with theirs
    back in ``|0⟩`` and only the root holds the data.

    Attributes:
        rank   (int):       Rank owning this record.
        root   (int):       Rank the qubits are gathered into.
        ranks  (List[int]): Participating ranks, in rank order.
        qubits (List[int]): Local qubits taking part: the whole buffer on
                            the root, this rank's chunk elsewhere.
    """

    @property
    def recv_rank(self) -> int:
        """Rank that gathers the qubits."""
        return self._root


class Expose(CollectiveOperation):
    """
    Open a telegate window sharing a control qubit across ranks.

    The *root* rank lends the state of one of its data qubits to every
    other participant, which receives it on a local communication qubit
    and can then apply locally-controlled gates with it.  The backend
    realises this with a shared GHZ state (cat-entangler); the window is
    closed by the matching :class:`Unexpose`.

    Every participant records its own ``Expose``, holding only the
    resources it contributes: one communication-qubit slot and the
    protocol classical bits used for the corrections (``len(ranks) - 1``
    bits on the root, one bit on each receiver).

    Attributes:
        rank       (int):       Rank owning this record.
        root       (int):       Rank that exposes its data qubit.
        ranks      (List[int]): Participants, root first.
        data_qubit (int):       Exposed data qubit (root only, else ``None``).
        comm_slot  (int):       Local communication-qubit slot.
        clbits     (List[int]): Local protocol classical bits.
        tag        (str):       Identifier shared by every participant.
    """

    def __init__(
        self,
        rank: int,
        root: int,
        ranks: List[int],
        tag: str,
        comm_slot: int,
        clbits: List[int],
        data_qubit: Optional[int] = None,
    ) -> None:
        """
        Args:
            rank:       Rank owning this record.
            root:       Rank exposing its data qubit.
            ranks:      Participants, root first.
            tag:        Identifier shared by every participant.
            comm_slot:  Local communication-qubit slot.
            clbits:     Local protocol classical bits.
            data_qubit: Exposed data qubit, on the root only.

        Raises:
            ValueError: If the participant list is inconsistent with *root*,
                or if the root does not provide a data qubit.
        """
        if not ranks or ranks[0] != root:
            raise ValueError("ranks must list the root first.")
        if len(ranks) < 2:
            raise ValueError("expose needs at least one rank besides the root.")
        if rank == root and data_qubit is None:
            raise ValueError("the root must provide the data qubit to expose.")
        super().__init__([] if data_qubit is None else [data_qubit], rank, ranks, tag)
        self._root = root
        self._comm_slot = comm_slot
        self._clbits = list(clbits)
        self._data_qubit = data_qubit

    @property
    def root(self) -> int:
        """Rank that exposes its data qubit."""
        return self._root

    @property
    def receivers(self) -> List[int]:
        """Ranks that receive the exposed control qubit."""
        return list(self._ranks[1:])

    @property
    def data_qubit(self) -> Optional[int]:
        """Exposed data qubit, or ``None`` outside the root."""
        return self._data_qubit

    @property
    def comm_slot(self) -> int:
        """Local communication-qubit slot used by the protocol."""
        return self._comm_slot

    @property
    def clbits(self) -> List[int]:
        """Local protocol classical bits used by the protocol."""
        return list(self._clbits)

    def __repr__(self) -> str:
        return (
            f"Expose(rank={self._rank}, root={self._root}, ranks={self._ranks}, "
            f"data_qubit={self._data_qubit}, comm_slot={self._comm_slot}, "
            f"clbits={self._clbits}, tag='{self._tag}')"
        )

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, Expose)
            and self._rank == other._rank
            and self._ranks == other._ranks
            and self._tag == other._tag
        )

    def __hash__(self) -> int:
        return hash(("Expose", self._rank, tuple(self._ranks), self._tag))


class Unexpose(CollectiveOperation):
    """
    Close a telegate window opened by :class:`Expose`.

    Carries the very same resources as the :class:`Expose` it closes, so
    the backend can emit the cat-disentangler (comm-qubit measurement and
    the phase correction on the root's data qubit) without re-deriving
    them.

    Attributes:
        rank       (int):       Rank owning this record.
        root       (int):       Rank that exposed its data qubit.
        ranks      (List[int]): Participants, root first.
        data_qubit (int):       Exposed data qubit (root only, else ``None``).
        comm_slot  (int):       Local communication-qubit slot.
        clbits     (List[int]): Local protocol classical bits.
        tag        (str):       Identifier shared by every participant.
    """

    def __init__(
        self,
        rank: int,
        root: int,
        ranks: List[int],
        tag: str,
        comm_slot: int,
        clbits: List[int],
        data_qubit: Optional[int] = None,
    ) -> None:
        """
        Args:
            rank:       Rank owning this record.
            root:       Rank that exposed its data qubit.
            ranks:      Participants, root first.
            tag:        Identifier shared by every participant.
            comm_slot:  Local communication-qubit slot.
            clbits:     Local protocol classical bits.
            data_qubit: Exposed data qubit, on the root only.
        """
        super().__init__([] if data_qubit is None else [data_qubit], rank, ranks, tag)
        self._root = root
        self._comm_slot = comm_slot
        self._clbits = list(clbits)
        self._data_qubit = data_qubit

    @classmethod
    def closing(cls, expose: Expose) -> Unexpose:
        """
        Build the record that closes a given :class:`Expose`.

        Args:
            expose: The expose record opened by this rank.

        Returns:
            An :class:`Unexpose` carrying the same protocol resources.
        """
        return cls(
            rank=expose.rank,
            root=expose.root,
            ranks=expose.ranks,
            tag=expose.tag,
            comm_slot=expose.comm_slot,
            clbits=expose.clbits,
            data_qubit=expose.data_qubit,
        )

    @property
    def root(self) -> int:
        """Rank that exposed its data qubit."""
        return self._root

    @property
    def receivers(self) -> List[int]:
        """Ranks that received the exposed control qubit."""
        return list(self._ranks[1:])

    @property
    def data_qubit(self) -> Optional[int]:
        """Exposed data qubit, or ``None`` outside the root."""
        return self._data_qubit

    @property
    def comm_slot(self) -> int:
        """Local communication-qubit slot used by the protocol."""
        return self._comm_slot

    @property
    def clbits(self) -> List[int]:
        """Local protocol classical bits used by the protocol."""
        return list(self._clbits)

    def __repr__(self) -> str:
        return (
            f"Unexpose(rank={self._rank}, root={self._root}, ranks={self._ranks}, "
            f"tag='{self._tag}')"
        )

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, Unexpose)
            and self._rank == other._rank
            and self._ranks == other._ranks
            and self._tag == other._tag
        )

    def __hash__(self) -> int:
        return hash(("Unexpose", self._rank, tuple(self._ranks), self._tag))
