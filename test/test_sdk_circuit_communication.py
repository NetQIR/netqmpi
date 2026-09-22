"""
The communication primitives, as the SDK records them.

Nothing is exchanged while a NetQMPI program is traced: each rank runs its
own ``main()`` and writes its own record, and only later does a runtime
have to decide that *this* send is the partner of *that* receive. What
makes that possible is agreement reached without communication — both
sides derive the same ``tag`` from the same public facts (who is calling,
who is on the other end, and how many such calls have happened so far),
and both run the same allocator over their own trace.

These tests are written from that angle. They trace the same program on
every rank and check that the records fit together: the tags pair up, the
chunks of a rooted collective add up, the resources a protocol borrows are
given back, and a telegate window hands out a control that stops being
addressable the moment it is closed.

Everything here is backend-free — no simulator is involved, and nothing is
executed.
"""
from __future__ import annotations

import pytest

from netqmpi.sdk.operations import (
    Expose,
    QGather,
    QRecv,
    QScatter,
    QSend,
    Unexpose,
)


def only(circuit, kind):
    """Return the operations of one type traced by a circuit."""
    return [op for op in circuit if isinstance(op, kind)]


# ----------------------------------------------------------------------
# qsend / qrecv
# ----------------------------------------------------------------------

def test_a_send_records_one_transfer_per_qubit(make_circuit):
    """Each qubit moves in its own protocol block."""
    circuit = make_circuit(num_qubits=3, num_clbits=1, rank=0, size=2)
    assert circuit.qsend([0, 1, 2], 1) is circuit

    sends = only(circuit, QSend)
    assert [op.qubits for op in sends] == [[0], [1], [2]]
    assert {op.dest_rank for op in sends} == {1}


def test_a_transfer_reserves_a_comm_qubit_and_two_classical_bits(make_circuit):
    """Teledata needs one EPR half locally and two correction bits."""
    circuit = make_circuit(num_qubits=1, num_clbits=1, rank=0, size=2)
    circuit.qsend([0], 1)

    send = only(circuit, QSend)[0]
    assert send.comm_slot == 0
    assert send.clbits == [0, 1]
    assert circuit.num_comm_qubits == 1
    assert circuit.num_protocol_clbits == 2


def test_sequential_transfers_reuse_the_same_resources(make_circuit):
    """
    The block is over before the next one starts, so one slot serves all.

    This is what the backend is asked to provide, so a program that moves
    a hundred qubits one after another must still fit on a node with a
    single communication qubit.
    """
    circuit = make_circuit(num_qubits=1, num_clbits=1, rank=0, size=2)
    for _ in range(100):
        circuit.qsend([0], 1)

    assert circuit.num_comm_qubits == 1
    assert circuit.num_protocol_clbits == 2
    assert {op.comm_slot for op in only(circuit, QSend)} == {0}


def test_the_two_halves_of_a_transfer_agree_on_the_tag(make_group):
    """
    Neither rank asks the other: both derive the same name.

    The tag is built from the ordered pair ``(source, destination)`` and a
    per-pair counter, so the *n*-th send from 0 to 1 is named exactly like
    the *n*-th receive on 1 from 0.
    """
    group = make_group(2, num_qubits=1, num_clbits=1)
    group[0].qsend([0], 1)
    group[1].qrecv([0], 0)

    assert only(group[0], QSend)[0].tag == only(group[1], QRecv)[0].tag == "teledata_0_1_0"


def test_repeated_transfers_between_the_same_pair_are_numbered(make_group):
    """Three sends and three receives pair up one for one, in order."""
    group = make_group(2, num_qubits=1, num_clbits=1)
    for _ in range(3):
        group[0].qsend([0], 1)
        group[1].qrecv([0], 0)

    assert [op.tag for op in only(group[0], QSend)] == [
        "teledata_0_1_0", "teledata_0_1_1", "teledata_0_1_2"]
    assert [op.tag for op in only(group[1], QRecv)] == [
        "teledata_0_1_0", "teledata_0_1_1", "teledata_0_1_2"]


def test_each_direction_and_each_peer_is_counted_separately(make_circuit):
    """A send to rank 1 must never collide with one to rank 2."""
    circuit = make_circuit(num_qubits=1, num_clbits=1, rank=0, size=3)
    circuit.qsend([0], 1)
    circuit.qsend([0], 2)
    circuit.qsend([0], 1)
    circuit.qrecv([0], 1)

    assert [op.tag for op in only(circuit, QSend)] == [
        "teledata_0_1_0", "teledata_0_2_0", "teledata_0_1_1"]
    # A receive from rank 1 is a transfer 1 -> 0: its own counter.
    assert only(circuit, QRecv)[0].tag == "teledata_1_0_0"


def test_a_receive_lands_where_the_receiver_asks(make_circuit):
    """The local index is the receiver's choice, not the sender's."""
    circuit = make_circuit(num_qubits=3, num_clbits=1, rank=1, size=2)
    circuit.qrecv([2], 0)
    assert only(circuit, QRecv)[0].qubits == [2]


@pytest.mark.parametrize("method", ["qsend", "qrecv"])
def test_transfers_validate_their_qubits(method, make_circuit):
    """An index the circuit does not have is refused at the call."""
    circuit = make_circuit(num_qubits=2, num_clbits=1, rank=0, size=2)
    with pytest.raises(IndexError, match=r"out of range"):
        getattr(circuit, method)([0, 9], 1)
    assert list(circuit) == []


# ----------------------------------------------------------------------
# qscatter
# ----------------------------------------------------------------------

def test_the_root_of_a_scatter_gives_its_whole_buffer_away(make_circuit):
    """One chunk per *other* rank, in rank order, and nothing kept."""
    root = make_circuit(num_qubits=2, num_clbits=2, rank=0, size=3)
    kept = root.qscatter([0, 1], root=0)

    assert kept == [], "unlike MPI_Scatter, the root keeps no chunk"

    # One block in the program, two transfers inside it.
    record = root.ops.children[0]
    assert isinstance(record, QScatter)
    assert record.root == 0 and record.ranks == [0, 1, 2]
    assert [(op.qubits, op.dest_rank) for op in only(root, QSend)] == [
        ([0], 1), ([1], 2)]


def test_a_scatter_splits_the_buffer_into_equal_chunks(make_circuit):
    """Four qubits over two receivers is two each, in rank order."""
    root = make_circuit(num_qubits=4, num_clbits=4, rank=0, size=3)
    root.qscatter([0, 1, 2, 3], root=0)

    assert [(op.qubits, op.dest_rank) for op in only(root, QSend)] == [
        ([0], 1), ([1], 1), ([2], 2), ([3], 2)]


def test_a_receiver_names_the_qubits_its_chunk_lands_on(make_circuit):
    """Off the root the call records receives and returns those qubits."""
    receiver = make_circuit(num_qubits=2, num_clbits=2, rank=2, size=3)
    landed = receiver.qscatter([0, 1], root=0)

    assert landed == [0, 1]
    assert [(op.qubits, op.src_rank) for op in only(receiver, QRecv)] == [
        ([0], 0), ([1], 0)]


def test_the_scattered_chunks_pair_up_across_the_ranks(make_group):
    """Every send the root traced is the receive a receiver traced."""
    group = make_group(3, num_qubits=2, num_clbits=2)
    group[0].qscatter([0, 1], root=0)
    group[1].qscatter([0], root=0)
    group[2].qscatter([0], root=0)

    sent = [op.tag for op in only(group[0], QSend)]
    received = [op.tag for rank in (1, 2) for op in only(group[rank], QRecv)]
    assert sent == received == ["teledata_0_1_0", "teledata_0_2_0"]


@pytest.mark.parametrize("size, qubits, root, error, match", [
    (3, [0, 1], 5, ValueError, "not a rank of the communicator"),
    (3, [0, 1], -1, ValueError, "not a rank of the communicator"),
    (1, [0], 0, ValueError, "at least one rank besides the root"),
    (3, [], 0, ValueError, "non-empty list of qubits"),
    (3, 0, 0, ValueError, "non-empty list of qubits"),
    (3, [0, 1, 2], 0, ValueError, "do not split evenly"),
    (3, [0, 9], 0, IndexError, "not a data qubit"),
])
def test_scatter_validation(size, qubits, root, error, match, make_circuit):
    """Everything a root can get wrong about its buffer, caught at the call."""
    circuit = make_circuit(num_qubits=3, num_clbits=3, rank=0, size=size)
    with pytest.raises(error, match=match):
        circuit.qscatter(qubits, root=root)


# ----------------------------------------------------------------------
# qgather
# ----------------------------------------------------------------------

def test_the_root_of_a_gather_collects_from_every_other_rank(make_circuit):
    """Its own chunk is already in place, so it only receives the rest."""
    root = make_circuit(num_qubits=3, num_clbits=3, rank=0, size=3)
    buffer = root.qgather([0, 1, 2], root=0)

    assert buffer == [0, 1, 2]
    assert [(op.qubits, op.src_rank) for op in only(root, QRecv)] == [
        ([1], 1), ([2], 2)]
    assert only(root, QSend) == []


def test_a_contributor_hands_its_chunk_over(make_circuit):
    """Off the root the qubits are sent and left behind in |0>."""
    leaf = make_circuit(num_qubits=1, num_clbits=1, rank=2, size=3)
    contributed = leaf.qgather([0], root=0)

    assert contributed == [0]
    assert [(op.qubits, op.dest_rank) for op in only(leaf, QSend)] == [([0], 0)]


def test_a_gather_places_each_rank_chunk_at_its_own_offset(make_circuit):
    """With two qubits per rank the root's slots line up in rank order."""
    root = make_circuit(num_qubits=6, num_clbits=6, rank=0, size=3)
    root.qgather([0, 1, 2, 3, 4, 5], root=0)

    assert [(op.qubits, op.src_rank) for op in only(root, QRecv)] == [
        ([2], 1), ([3], 1), ([4], 2), ([5], 2)]


def test_the_gathered_chunks_pair_up_across_the_ranks(make_group):
    """The mirror image of the scatter pairing, tags and all."""
    group = make_group(3, num_qubits=3, num_clbits=3)
    group[0].qgather([0, 1, 2], root=0)
    group[1].qgather([0], root=0)
    group[2].qgather([0], root=0)

    received = [op.tag for op in only(group[0], QRecv)]
    sent = [op.tag for rank in (1, 2) for op in only(group[rank], QSend)]
    assert received == sent == ["teledata_1_0_0", "teledata_2_0_0"]


def test_a_gather_record_knows_its_shape(make_circuit):
    """The record names the collective it belongs to, for the adapters."""
    root = make_circuit(num_qubits=2, num_clbits=2, rank=0, size=2)
    root.qgather([0, 1], root=0)

    record = [op for op in root.ops.children if isinstance(op, QGather)][0]
    assert record.is_root and record.recv_rank == 0
    assert record.ranks == [0, 1] and record.qubits == [0, 1]


@pytest.mark.parametrize("size, qubits, root, error, match", [
    (3, [0, 1], 9, ValueError, "not a rank of the communicator"),
    (3, [], 0, ValueError, "non-empty list of qubits"),
    (3, [0, 1], 0, ValueError, "do not split evenly"),
    (3, [0, 9], 0, IndexError, "not a data qubit"),
])
def test_gather_validation(size, qubits, root, error, match, make_circuit):
    """A gather whose buffer cannot hold one chunk per rank."""
    circuit = make_circuit(num_qubits=3, num_clbits=3, rank=0, size=size)
    with pytest.raises(error, match=match):
        circuit.qgather(qubits, root=root)


# ----------------------------------------------------------------------
# expose / unexpose
# ----------------------------------------------------------------------

def test_the_root_keeps_addressing_its_own_qubit(make_circuit):
    """Lending a control does not move it: the root's index is unchanged."""
    root = make_circuit(num_qubits=1, num_clbits=1, rank=0, size=2)
    handle = root.expose(0, [1])

    assert handle == 0
    window = only(root, Expose)[0]
    assert window.root == 0 and window.ranks == [0, 1]
    assert window.data_qubit == 0


def test_a_receiver_gets_a_communication_qubit_to_control_with(make_circuit):
    """The handle is addressable by the whole gate API, like a data qubit."""
    receiver = make_circuit(num_qubits=2, num_clbits=1, rank=1, size=2)
    handle = receiver.expose(None, [1], root=0)

    assert handle == receiver.comm_qubit(0) == 2
    assert receiver.num_comm_qubits == 1
    receiver.cx(handle, 0)                  # accepted while the window is open


def test_every_participant_names_the_window_the_same_way(make_group):
    """Again derived, not exchanged: same group, same tag, same order."""
    group = make_group(3, num_qubits=1, num_clbits=1)
    for rank in range(3):
        group[rank].expose(0 if rank == 2 else None, [0, 1], root=2)

    tags = {only(group[rank], Expose)[0].tag for rank in range(3)}
    assert tags == {"expose_2_0_1_0"}


def test_the_root_collects_one_correction_bit_per_receiver(make_circuit):
    """Its share of the protocol grows with the group; a receiver's does not."""
    root = make_circuit(num_qubits=1, num_clbits=1, rank=0, size=4)
    root.expose(0, [1, 2, 3])
    assert only(root, Expose)[0].clbits == [0, 1, 2]

    receiver = make_circuit(num_qubits=1, num_clbits=1, rank=2, size=4)
    receiver.expose(None, [1, 2, 3], root=0)
    assert only(receiver, Expose)[0].clbits == [0]


def test_the_participant_list_is_normalised_the_same_way_everywhere(make_circuit):
    """Root first, duplicates dropped, and the root filtered out of ranks."""
    circuit = make_circuit(num_qubits=1, num_clbits=1, rank=0, size=3)
    circuit.expose(0, [1, 2, 1, 0])
    assert only(circuit, Expose)[0].ranks == [0, 1, 2]


def test_a_rank_outside_the_group_records_nothing(make_circuit):
    """A window it does not take part in must not reserve its resources."""
    outsider = make_circuit(num_qubits=1, num_clbits=1, rank=3, size=4)
    assert outsider.expose(None, [1], root=0) is None
    assert outsider.unexpose([1], root=0) is outsider
    assert list(outsider) == []
    assert outsider.num_comm_qubits == 0


def test_closing_a_window_gives_the_resources_back(make_circuit):
    """The slot returns to the pool, so the next window reuses it."""
    circuit = make_circuit(num_qubits=1, num_clbits=1, rank=1, size=2)
    first = circuit.expose(None, [1], root=0)
    circuit.unexpose([1], root=0)
    second = circuit.expose(None, [1], root=0)

    assert first == second
    assert circuit.num_comm_qubits == 1
    assert [type(op).__name__ for op in circuit] == ["Expose", "Unexpose", "Expose"]


def test_the_closing_record_carries_the_windows_resources(make_circuit):
    """``unexpose`` must undo exactly what ``expose`` set up."""
    circuit = make_circuit(num_qubits=1, num_clbits=1, rank=0, size=2)
    circuit.expose(0, [1])
    circuit.unexpose([1])

    opened, closed = only(circuit, Expose)[0], only(circuit, Unexpose)[0]
    assert closed.tag == opened.tag
    assert closed.comm_slot == opened.comm_slot
    assert closed.clbits == opened.clbits
    assert closed.data_qubit == opened.data_qubit


def test_a_lent_control_stops_being_addressable_when_the_window_closes(make_circuit):
    """
    Using it afterwards is a bug, and one the SDK can see while tracing.

    The index stays *in range* — the pool still reports the slot — so the
    check is on the window, not on the width, and the message says so.
    """
    circuit = make_circuit(num_qubits=1, num_clbits=1, rank=1, size=2)
    handle = circuit.expose(None, [1], root=0)
    circuit.cx(handle, 0)
    circuit.unexpose([1], root=0)

    with pytest.raises(IndexError, match="window is already closed"):
        circuit.cx(handle, 0)


def test_nested_windows_unwind_innermost_first(make_circuit):
    """Two windows over the same group behave like scopes."""
    circuit = make_circuit(num_qubits=1, num_clbits=1, rank=1, size=2)
    outer = circuit.expose(None, [1], root=0)
    inner = circuit.expose(None, [1], root=0)

    assert outer != inner
    assert circuit.num_comm_qubits == 2, "both are held at once"

    circuit.unexpose([1], root=0)
    closings = only(circuit, Unexpose)
    assert closings[0].tag == only(circuit, Expose)[1].tag, "the inner one"

    circuit.unexpose([1], root=0)
    assert only(circuit, Unexpose)[1].tag == only(circuit, Expose)[0].tag


def test_windows_over_different_groups_are_tracked_apart(make_circuit):
    """
    Closing one must not close the other.

    Example 5 does exactly this: rank 2's window over ``[0, 1]`` stays open
    across the whole transform while rank 1's window over ``[0]`` opens and
    closes inside it.
    """
    circuit = make_circuit(num_qubits=1, num_clbits=1, rank=0, size=3)
    wide = circuit.expose(None, [0, 1], root=2)
    narrow = circuit.expose(None, [0], root=1)

    circuit.unexpose([0], root=1)
    circuit.cx(wide, 0)                     # still open

    with pytest.raises(IndexError, match="window is already closed"):
        circuit.cx(narrow, 0)

    circuit.unexpose([0, 1], root=2)
    with pytest.raises(IndexError, match="window is already closed"):
        circuit.cx(wide, 0)


def test_a_transfer_inside_a_window_borrows_a_second_slot(make_circuit):
    """
    Resources held at once cannot overlap, whatever holds them.

    A telegate window keeps its communication qubit for as long as it is
    open, so a transfer made meanwhile has to be given another one — the
    node needs two.
    """
    circuit = make_circuit(num_qubits=2, num_clbits=1, rank=1, size=2)
    circuit.expose(None, [1], root=0)
    circuit.qsend([0], 0)

    assert only(circuit, QSend)[0].comm_slot == 1
    assert circuit.num_comm_qubits == 2

    circuit.unexpose([1], root=0)
    circuit.qsend([0], 0)
    assert only(circuit, QSend)[1].comm_slot == 0, "slot 0 is free again"
    assert circuit.num_comm_qubits == 2


def test_closing_a_window_that_was_never_opened_is_refused(make_circuit):
    """Named with the arguments it was called with, so it can be found."""
    circuit = make_circuit(num_qubits=1, num_clbits=1, rank=0, size=2)
    with pytest.raises(RuntimeError, match=r"unexpose\(ranks=\[1\], root=0\)"):
        circuit.unexpose([1])


def test_closing_the_wrong_group_does_not_match_an_open_window(make_circuit):
    """A window is keyed by its participants, not by being the latest one."""
    circuit = make_circuit(num_qubits=1, num_clbits=1, rank=0, size=3)
    circuit.expose(0, [1])
    with pytest.raises(RuntimeError, match="without a matching open expose"):
        circuit.unexpose([2])


@pytest.mark.parametrize("qubit, ranks, error, match", [
    (0, [], ValueError, "non-empty list"),
    (0, 1, ValueError, "non-empty list"),
    (0, [0], ValueError, "at least one rank besides the root"),
    (9, [1], IndexError, "not a data qubit"),
])
def test_expose_validation(qubit, ranks, error, match, make_circuit):
    """A window with no receiver, or a root lending something it has not got."""
    circuit = make_circuit(num_qubits=2, num_clbits=1, rank=0, size=3)
    with pytest.raises(error, match=match):
        circuit.expose(qubit, ranks)


def test_the_root_defaults_to_the_calling_rank(make_circuit):
    """``expose(q, [1])`` on rank 0 means "rank 0 lends q to rank 1"."""
    circuit = make_circuit(num_qubits=1, num_clbits=1, rank=0, size=2)
    circuit.expose(0, [1])
    assert only(circuit, Expose)[0].root == 0
