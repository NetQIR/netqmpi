"""
The two objects a user program actually holds.

A NetQMPI ``main()`` is handed an
:class:`~netqmpi.sdk.environment.Environment` and reads two things off it:
``env.comm``, the rank's
:class:`~netqmpi.sdk.communicator.QMPICommunicator`, and
``env.create_circuit(...)``. Everything else — which backend is running,
how the ranks were started, where the results come from — is behind those
two calls, and that is precisely what makes the same script run on four
backends unchanged.

So what is worth testing here is the *facade*: that the communicator
answers rank and size, that its neighbour helpers wrap around the ring,
that each communication primitive forwards to the circuit with the
arguments reordered the way the circuit expects, and that a circuit
created through the environment is registered on the communicator — which
is how a runtime later finds every circuit a rank traced.
"""
from __future__ import annotations

import pytest

from netqmpi.sdk.circuit import Circuit
from netqmpi.sdk.communicator import QMPICommunicator
from netqmpi.sdk.environment import Environment
from netqmpi.sdk.operations import Expose, QRecv, QSend, Unexpose

from conftest import RecordingCircuit, StubCommunicator, StubExecutor


# ----------------------------------------------------------------------
# QMPICommunicator
# ----------------------------------------------------------------------

def test_the_communicator_interface_must_be_implemented_by_a_backend():
    """Only the context hooks are abstract; everything else is shared."""
    assert QMPICommunicator.__abstractmethods__ == frozenset({"__enter__", "__exit__"})
    with pytest.raises(TypeError, match="abstract"):
        QMPICommunicator(0, 2)


def test_rank_and_size_are_what_the_runtime_assigned():
    """The SPMD pair every program branches on."""
    comm = StubCommunicator(rank=2, size=5)
    assert comm.rank == 2 and comm.size == 5
    assert comm.circuits == [] and comm.results == {}


def test_neighbours_wrap_around_the_ring():
    """``2_round_robin`` relies on this closing the loop."""
    comm = StubCommunicator(rank=0, size=3)
    assert [comm.get_next_rank(r) for r in range(3)] == [1, 2, 0]
    assert [comm.get_prev_rank(r) for r in range(3)] == [2, 0, 1]


def test_a_single_rank_is_its_own_neighbour():
    """The degenerate ring still answers, rather than dividing by zero."""
    comm = StubCommunicator(rank=0, size=1)
    assert comm.get_next_rank(0) == comm.get_prev_rank(0) == 0


def test_ranks_have_canonical_names():
    """Backends address nodes by name; the mapping is fixed here."""
    comm = StubCommunicator(rank=0, size=4)
    assert [comm.get_rank_name(r) for r in range(4)] == [
        "rank_0", "rank_1", "rank_2", "rank_3"]


def test_the_block_is_a_context_manager():
    """``with comm:`` is where a backend opens and closes the run."""
    comm = StubCommunicator(rank=0, size=2)
    with comm as entered:
        assert entered is comm
    assert (comm.entered, comm.exited) == (1, 1)


# ----------------------------------------------------------------------
# Delegation to the circuit
# ----------------------------------------------------------------------

@pytest.fixture
def rank_zero():
    """A rank-0 communicator and a circuit of its own, wired together."""
    comm = StubCommunicator(rank=0, size=3)
    circuit = RecordingCircuit(3, 3, comm)
    comm.circuits.append(circuit)
    return comm, circuit


def test_qsend_and_qrecv_forward_to_the_circuit(rank_zero):
    """``comm.qsend(circuit, ...)`` is ``circuit.qsend(...)``."""
    comm, circuit = rank_zero
    comm.qsend(circuit, [0], 1)
    comm.qrecv(circuit, [1], 2)

    assert [(type(op).__name__, op.qubits) for op in circuit] == [
        ("QSend", [0]), ("QRecv", [1])]
    assert [op for op in circuit if isinstance(op, QSend)][0].dest_rank == 1
    assert [op for op in circuit if isinstance(op, QRecv)][0].src_rank == 2


def test_the_rooted_collectives_return_what_the_circuit_returns(rank_zero):
    """The chunk a rank ends up holding comes back through the facade."""
    comm, circuit = rank_zero
    assert comm.qscatter(circuit, [0, 1], root=0) == []          # root keeps none

    receiver_comm = StubCommunicator(rank=1, size=3)
    receiver = RecordingCircuit(1, 1, receiver_comm)
    assert receiver_comm.qscatter(receiver, [0], root=0) == [0]
    assert receiver_comm.qgather(receiver, [0], root=0) == [0]


def test_expose_forwards_the_root_and_returns_the_handle(rank_zero):
    """The root argument is keyword-only on the circuit; the facade maps it."""
    comm, circuit = rank_zero
    handle = comm.expose(circuit, 0, [1, 2])

    assert handle == 0                                  # the root's own qubit
    window = [op for op in circuit if isinstance(op, Expose)][0]
    assert window.root == 0 and window.ranks == [0, 1, 2]

    comm.unexpose(circuit, [1, 2])
    assert any(isinstance(op, Unexpose) for op in circuit)


def test_expose_on_a_remote_root_hands_back_a_communication_qubit(rank_zero):
    """A receiver passes no qubit of its own and gets a borrowed one."""
    comm = StubCommunicator(rank=1, size=3)
    circuit = RecordingCircuit(2, 2, comm)
    handle = comm.expose(circuit, None, [1, 2], root=0)
    assert handle == circuit.comm_qubit(0)


# ----------------------------------------------------------------------
# Environment
# ----------------------------------------------------------------------

def test_the_environment_exposes_the_rank_communicator():
    """``env.comm`` is the only way a program reaches its rank."""
    comm = StubCommunicator(rank=1, size=2)
    env = Environment(comm, StubExecutor())
    assert env.comm is comm


def test_create_circuit_goes_through_the_backend_factory():
    """
    The environment never builds a circuit itself.

    That indirection is the whole point: the same ``create_circuit`` call
    yields a Qiskit-backed adapter under ``--aer`` and a NetQASM one under
    ``--netqasm``, and the program cannot tell.
    """
    comm = StubCommunicator(rank=0, size=2)
    executor = StubExecutor()
    env = Environment(comm, executor)

    circuit = env.create_circuit(num_qubits=3, num_clbits=2)

    assert isinstance(circuit, Circuit)
    assert executor.created == [(3, 2)]
    assert circuit.num_qubits == 3 and circuit.num_clbits == 2
    assert circuit.comm is comm


def test_every_circuit_a_rank_creates_is_registered_on_its_communicator():
    """
    This list is how a runtime finds the program at the end of the block.

    The Aer adapter pairs the *i*-th circuit of every rank into one
    distributed program, so both the membership and the order matter.
    """
    comm = StubCommunicator(rank=0, size=2)
    env = Environment(comm, StubExecutor())

    first = env.create_circuit(1, 1)
    second = env.create_circuit(2, 2)

    assert comm.circuits == [first, second]
