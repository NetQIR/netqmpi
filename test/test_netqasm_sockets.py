"""
Regression tests for how the NetQASM adapter uses classical sockets.

These need NetQASM but not NetSquid: the classical sockets SquidASM gives a
program are NetQASM's own ``ThreadSocket``, which only talks to a hub shared
by the threads of one process. Without SquidASM installed the adapter is
imported against NetQASM's ``debug`` simulator, which hands out the very same
socket class.

**The deadlock (N1).** ``cascade`` with two or more qubits per rank never
finished on SquidASM, even with two ranks: the receiver was left waiting in
``socket_hub._wait_for_remote`` for a sender whose program had already
ended. The adapter opened a fresh socket for every transfer, and every
socket between the same two ranks has the same hub key. The hub marks a
socket as open in ``_open_sockets`` and leaves a marker for its peer in
``_remote_sockets``; closing a socket removes *its peer's* marker. When the
sender ran ahead — its second transfer done, its second socket opened and
closed — while the receiver was still inside the first transfer, the
receiver closing its first socket erased the marker the sender's second
socket had left. The receiver's second socket then found neither an open
peer nor a marker, and waited forever.

Transfers that alternate direction, as in ``ghz``, never let one side run
ahead like that, which is why they were not affected.
"""
from __future__ import annotations

import os
import threading

import pytest

pytest.importorskip("netqasm", reason="the NetQASM backend needs NetQASM")
try:
    import squidasm  # noqa: F401
except ImportError:
    # Only decides what netqasm.sdk.external resolves to. The integration
    # tests that would run SquidASM are skipped without it anyway.
    os.environ.setdefault("NETQASM_SIMULATOR", "debug")

from netqasm.sdk.classical_communication.message import StructuredMessage  # noqa: E402
from netqasm.sdk.classical_communication.thread_socket.socket import (  # noqa: E402
    ThreadSocket,
)
from netqasm.sdk.classical_communication.thread_socket.socket_hub import (  # noqa: E402
    reset_socket_hub,
)

from netqmpi.runtime.adapters.netqasm import (  # noqa: E402
    NetQASMCommunicator, NetQASMRunConfig,
)

#: Long enough for any healthy exchange here, which takes milliseconds.
PATIENCE = 5.0


@pytest.fixture(autouse=True)
def fresh_hub():
    """Every test starts and ends with an empty socket hub and no run state."""
    reset_socket_hub()
    NetQASMCommunicator.reset_run()
    yield
    reset_socket_hub()
    NetQASMCommunicator.reset_run()


def race_ahead(open_sender_socket, open_receiver_socket, drop_sender_socket,
               drop_receiver_socket):
    """
    Two transfers from rank 0 to rank 1, with the sender running ahead.

    The sender completes both of its transfers — open a socket, send the
    corrections, let go of it — while the receiver is still inside its first
    one, which is what happens on SquidASM when the second EPR pair is ready
    before the receiver has consumed the first.

    Args:
        open_sender_socket: Returns the socket for the sender's next transfer.
        open_receiver_socket: Returns the socket for the receiver's next
            transfer.
        drop_sender_socket: Called when the sender is done with a transfer.
        drop_receiver_socket: Called when the receiver is done with a transfer.

    Returns:
        The corrections the receiver got, or ``None`` if it was still waiting
        after :data:`PATIENCE` seconds.
    """
    sender_done = threading.Event()
    received = []

    def sender():
        for shot in range(2):
            socket = open_sender_socket()
            socket.send_structured(StructuredMessage("Corrections", (shot, shot)))
            del socket
            drop_sender_socket()
        sender_done.set()

    def receiver():
        socket = open_receiver_socket()
        sender_done.wait()
        received.append(socket.recv_structured().payload)
        del socket
        drop_receiver_socket()

        socket = open_receiver_socket()
        received.append(socket.recv_structured().payload)
        del socket
        drop_receiver_socket()

    threads = [threading.Thread(target=sender, daemon=True),
               threading.Thread(target=receiver, daemon=True)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(PATIENCE)
    if any(thread.is_alive() for thread in threads):
        return None
    return received


def test_a_socket_per_transfer_strands_the_receiver():
    """
    The backend behaviour behind N1, reproduced with NetQASM's own sockets.

    This pins down the mechanism rather than the fix: if it ever starts
    passing the other way, NetQASM changed how its hub pairs sockets.
    """
    outcome = race_ahead(
        open_sender_socket=lambda: ThreadSocket("rank_0", "rank_1"),
        open_receiver_socket=lambda: ThreadSocket("rank_1", "rank_0"),
        drop_sender_socket=lambda: None,
        drop_receiver_socket=lambda: None,
    )
    assert outcome is None, "the receiver was expected to be stranded"


def communicators(size: int):
    """Build one communicator per rank, as the executor does."""
    config = NetQASMRunConfig()
    return [NetQASMCommunicator(rank, size, config) for rank in range(size)]


def test_consecutive_transfers_on_one_pair_complete():
    """Two transfers in the same direction, sender ahead, both arrive."""
    sender, receiver = communicators(2)
    outcome = race_ahead(
        open_sender_socket=lambda: sender.get_socket(0, 1),
        open_receiver_socket=lambda: receiver.get_socket(1, 0),
        drop_sender_socket=lambda: None,
        drop_receiver_socket=lambda: None,
    )
    assert outcome == [[0, 0], [1, 1]]


def test_a_round_reuses_one_socket_per_peer():
    """Within a round every transfer to a peer goes through the same socket."""
    sender, receiver = communicators(2)
    opened = threading.Thread(target=receiver.get_socket, args=(1, 0),
                              daemon=True)
    opened.start()
    first = sender.get_socket(0, 1)
    opened.join(PATIENCE)
    assert sender.get_socket(0, 1) is first


def test_sockets_do_not_outlive_their_round():
    """
    A new round opens new sockets, and the old ones are gone from the hub.

    SquidASM resets the hub between rounds, so a socket kept from one round
    to the next is no longer connected ("Socket is not connected so cannot
    send"); and one released only later would, when collected, erase the
    hub entries of the new round's socket with the same key.
    """
    sender, receiver = communicators(2)
    opened = threading.Thread(target=receiver.get_socket, args=(1, 0),
                              daemon=True)
    opened.start()
    first = sender.get_socket(0, 1)
    opened.join(PATIENCE)
    assert first.connected
    del first

    sender.close_sockets()
    receiver.close_sockets()

    from netqasm.sdk.classical_communication.thread_socket.socket_hub import (
        _socket_hub,
    )
    assert not _socket_hub._open_sockets
    assert not _socket_hub._remote_sockets

    opened = threading.Thread(target=receiver.get_socket, args=(1, 0),
                              daemon=True)
    opened.start()
    second = sender.get_socket(0, 1)
    opened.join(PATIENCE)
    assert second.connected
