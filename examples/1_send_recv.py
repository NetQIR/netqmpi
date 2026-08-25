"""
Teleporting a qubit from one rank to another.

The most basic thing two ranks can do with a qubit: rank 0 prepares one in
superposition and hands it to rank 1 with ``qsend``/``qrecv``. Nothing in
the program mentions entanglement, Bell measurements or the classical
corrections they need — the backend fills in the teledata protocol behind
those two calls.

Handing a qubit over *moves* it, since a quantum state cannot be copied:
by the time the call returns, rank 0's qubit is back in ``|0⟩`` and only
rank 1 holds the state. That is why rank 0 has nothing left to measure and
contributes no counts.

Rank 1 measures the ``|+⟩`` it received in the computational basis, so it
reads 0 and 1 about equally often. What the run shows is that the transfer
happened at all; it takes a state like ``|1⟩`` to tell a correct
teleportation from a broken one.

This file uses SDK abstractions only, so the same code runs on any backend.

Run with::

    netqmpi -n 2 --cunqa 1_send_recv.py
"""
from netqmpi.sdk.environment import Environment


def main(env: Environment = None):
    comm = env.comm
    rank = comm.rank

    next_rank = comm.get_next_rank(rank)
    previous_rank = comm.get_prev_rank(rank)

    # Only what is inside the block takes part in the distributed program.
    with comm:
        circuit = env.create_circuit(num_qubits=1, num_clbits=1)

        if rank == 0:
            circuit.h(0)                            # prepare |+>
            comm.qsend(circuit, [0], next_rank)     # and give it away
        else:
            comm.qrecv(circuit, [0], previous_rank)
            circuit.measure(0, 0)

    # The circuits of all the ranks are submitted together when the last
    # rank leaves the block, so only that rank sees the results here — and
    # it sees every rank's counts, keyed by rank.
    if comm.results:
        for other, counts in comm.results.items():
            print(f"rank_{other}: {counts}")
