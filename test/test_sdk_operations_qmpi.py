"""
The records a cross-rank call leaves behind.

A distributed call is traced independently on each rank: nobody talks to
anybody while the program is being recorded. What makes the halves fit
back together afterwards is carried *inside* the records themselves — the
``tag`` both sides derive from data they already share, the participant
list, and the communication resources each rank contributes. These tests
pin that contract:

- the point-to-point pair :class:`QSend` / :class:`QRecv`;
- the rooted collectives :class:`QScatter` / :class:`QGather`, which are
  containers of the transfers they expand into, so a backend that can move
  one qubit gets them for free;
- the telegate window :class:`Expose` / :class:`Unexpose`, whose
  :meth:`~netqmpi.sdk.operations.qmpi.CollectiveOperation.matches` is what
  a runtime uses to decide that every participant has arrived.
"""
from __future__ import annotations

import pytest

from netqmpi.sdk.operations import (
    CollectiveOperation,
    Expose,
    Gate,
    OperationContainer,
    QGather,
    QRecv,
    QScatter,
    QSend,
    RootedTransfer,
    Unexpose,
)


# ----------------------------------------------------------------------
# QSend / QRecv
# ----------------------------------------------------------------------

def test_qsend_carries_its_destination_and_resources():
    """A send names where the qubit goes and what the protocol borrows."""
    op = QSend([0], dest_rank=1, comm_slot=0, clbits=[0, 1], tag="teledata_0_1_0")
    assert op.qubits == [0]
    assert op.dest_rank == 1
    assert op.comm_slot == 0
    assert op.clbits == [0, 1]
    assert op.tag == "teledata_0_1_0"


def test_qrecv_counts_the_qubits_it_expects():
    """The receiver's buffer length is how many qubits are due to arrive."""
    op = QRecv([2, 3], src_rank=0)
    assert op.n_qubits == 2
    assert op.src_rank == 0
    assert op.comm_slot is None       # unreserved outside a circuit
    assert op.clbits == []


def test_transfer_resources_are_copies():
    """An adapter cannot rewrite the classical bits a transfer reserved."""
    op = QSend([0], 1, clbits=[0, 1])
    op.clbits.append(9)
    assert op.clbits == [0, 1]


@pytest.mark.parametrize("factory", [QSend, QRecv])
@pytest.mark.parametrize("qubits, rank, match", [
    ([], 1, "non-empty"),
    ([0], -1, "non-negative"),
])
def test_transfer_validation(factory, qubits, rank, match):
    """A transfer of nothing, or with a rank that cannot exist."""
    with pytest.raises(ValueError, match=match):
        factory(qubits, rank)


def test_transfers_compare_by_qubits_peer_and_tag():
    """The tag is part of the identity: it is what pairs the two halves."""
    assert QSend([0], 1, tag="t") == QSend([0], 1, tag="t")
    assert QSend([0], 1, tag="t") != QSend([0], 1, tag="other")
    assert QSend([0], 1, tag="t") != QSend([0], 2, tag="t")
    assert QSend([0], 1, tag="t") != QRecv([0], 1, tag="t")
    assert len({QSend([0], 1, tag="t"), QSend([0], 1, tag="t")}) == 1


@pytest.mark.parametrize("op, expected", [
    (QSend([0], 1, tag="t"), "QSend(qubits=[0], dest_rank=1, tag=t)"),
    (QRecv([0], 1, tag="t"), "QRecv(qubits=[0], src_rank=1, tag=t)"),
])
def test_transfer_repr(op, expected):
    """Both halves print the peer they are waiting for."""
    assert repr(op) == expected


# ----------------------------------------------------------------------
# Rooted collectives
# ----------------------------------------------------------------------

def test_rooted_transfer_is_a_container_of_point_to_point_moves():
    """A scatter *is* its sends: a backend translates the children."""
    record = QScatter(rank=0, root=0, ranks=[0, 1, 2], qubits=[0, 1])
    record.add(QSend([0], 1, tag="teledata_0_1_0"))
    record.add(QSend([1], 2, tag="teledata_0_2_0"))

    assert isinstance(record, OperationContainer)
    assert len(record) == 2
    assert [type(child) for child in record.children] == [QSend, QSend]
    assert [op.dest_rank for op in record.flatten()] == [1, 2]


def test_rooted_transfer_reports_its_role():
    """Whether this rank is the root decides which half it traced."""
    root = QGather(rank=0, root=0, ranks=[0, 1], qubits=[0, 1])
    leaf = QGather(rank=1, root=0, ranks=[0, 1], qubits=[0])
    assert root.is_root and not leaf.is_root
    assert root.qubits == [0, 1] and leaf.qubits == [0]
    assert root.ranks == [0, 1]


def test_scatter_and_gather_name_the_rank_that_holds_the_buffer():
    """Both aliases point at the root, from each collective's viewpoint."""
    assert QScatter(rank=1, root=0, ranks=[0, 1], qubits=[0]).sender_rank == 0
    assert QGather(rank=1, root=0, ranks=[0, 1], qubits=[0]).recv_rank == 0


@pytest.mark.parametrize("kwargs, match", [
    (dict(rank=0, root=0, ranks=[], qubits=[0]), "non-empty list"),
    (dict(rank=0, root=0, ranks=[0, -1], qubits=[0]), "non-negative"),
    (dict(rank=0, root=0, ranks=[0, "1"], qubits=[0]), "non-negative"),
    (dict(rank=0, root=5, ranks=[0, 1], qubits=[0]), "must be one of the ranks"),
    (dict(rank=7, root=0, ranks=[0, 1], qubits=[0]), "does not take part"),
    (dict(rank=0, root=0, ranks=[0, 1], qubits=[]), "non-empty"),
])
def test_rooted_transfer_validation(kwargs, match):
    """A rooted collective with an inconsistent participant list."""
    with pytest.raises(ValueError, match=match):
        QScatter(**kwargs)


def test_rooted_transfers_of_different_kinds_never_compare_equal():
    """A scatter and a gather over the same ranks are opposite calls."""
    scatter = QScatter(rank=0, root=0, ranks=[0, 1], qubits=[0])
    gather = QGather(rank=0, root=0, ranks=[0, 1], qubits=[0])
    assert scatter != gather
    assert scatter == QScatter(rank=0, root=0, ranks=[0, 1], qubits=[0])
    assert len({scatter, gather}) == 2
    assert "QScatter(rank=0, root=0" in repr(scatter)


def test_rooted_transfers_are_not_collectives_in_the_blocking_sense():
    """
    They expand into ordinary transfers, so no rank has to wait to emit.

    This is not a detail: the Aer and CUNQA joint passes treat
    :class:`CollectiveOperation` records as barriers that only open once
    every participant has arrived, and a scatter must *not* be one.
    """
    assert not issubclass(RootedTransfer, CollectiveOperation)
    assert issubclass(RootedTransfer, OperationContainer)


# ----------------------------------------------------------------------
# Telegate window
# ----------------------------------------------------------------------

def make_window(rank, root=0, ranks=(0, 1), tag="expose_0_1_0", data_qubit=None):
    """Build one participant's record of a telegate window."""
    return Expose(rank=rank, root=root, ranks=list(ranks), tag=tag,
                  comm_slot=0, clbits=[0],
                  data_qubit=data_qubit if rank == root else None)


def test_expose_root_lends_a_data_qubit_and_receivers_lend_a_slot():
    """The root's record names the qubit; a receiver's names no qubit."""
    root = make_window(0, data_qubit=1)
    receiver = make_window(1)

    assert root.data_qubit == 1 and root.qubits == [1]
    assert receiver.data_qubit is None and receiver.qubits == []
    assert root.root == receiver.root == 0
    assert root.receivers == [1]
    assert root.comm_slot == 0 and root.clbits == [0]


@pytest.mark.parametrize("kwargs, match", [
    (dict(rank=0, root=0, ranks=[1, 0], tag="t", comm_slot=0, clbits=[],
          data_qubit=0), "list the root first"),
    (dict(rank=0, root=0, ranks=[0], tag="t", comm_slot=0, clbits=[],
          data_qubit=0), "at least one rank besides"),
    (dict(rank=0, root=0, ranks=[0, 1], tag="t", comm_slot=0, clbits=[]),
     "must provide the data qubit"),
    (dict(rank=0, root=0, ranks=[0, 1], tag="", comm_slot=0, clbits=[],
          data_qubit=0), "tag must be"),
    (dict(rank=0, root=0, ranks=[0, -1], tag="t", comm_slot=0, clbits=[],
          data_qubit=0), "non-negative"),
])
def test_expose_validation(kwargs, match):
    """A window nobody could open: no receiver, no root qubit, no tag."""
    with pytest.raises(ValueError, match=match):
        Expose(**kwargs)


def test_matches_pairs_the_records_of_one_call():
    """Same kind, same tag, same participants — that is one call."""
    root = make_window(0, data_qubit=1)
    receiver = make_window(1)

    assert root.matches(receiver) and receiver.matches(root)
    assert not root.matches(make_window(1, tag="expose_0_1_1"))
    assert not root.matches(make_window(1, ranks=(0, 1, 2)))
    assert not root.matches(Unexpose.closing(receiver))     # opposite ends


def test_unexpose_closing_reuses_the_windows_resources():
    """Closing a window must give back exactly what opening it took."""
    opened = Expose(rank=0, root=0, ranks=[0, 1, 2], tag="expose_0_1_2_0",
                    comm_slot=3, clbits=[4, 5], data_qubit=1)
    closed = Unexpose.closing(opened)

    assert isinstance(closed, Unexpose)
    assert (closed.rank, closed.root, closed.ranks) == (0, 0, [0, 1, 2])
    assert (closed.comm_slot, closed.clbits) == (3, [4, 5])
    assert closed.data_qubit == 1
    assert closed.tag == opened.tag
    assert closed.receivers == [1, 2]
    assert "Unexpose(rank=0, root=0" in repr(closed)


def test_window_records_compare_by_rank_participants_and_tag():
    """Two ranks' records of the same window are different records."""
    assert make_window(0, data_qubit=1) != make_window(1)
    assert make_window(1) == make_window(1)
    assert len({make_window(1), make_window(1), make_window(0, data_qubit=1)}) == 2


def test_collective_accessors_hand_out_copies():
    """The participant list cannot be edited through a record."""
    window = make_window(0, data_qubit=1)
    window.ranks.append(9)
    window.clbits.append(9)
    assert window.ranks == [0, 1]
    assert window.clbits == [0]
