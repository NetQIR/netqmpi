"""
Passing one qubit down a chain of ranks.

Rank 0 prepares a qubit in superposition and teleports it to rank 1, which
teleports it to rank 2, and so on until the last rank, which measures it.
The state is relayed from neighbour to neighbour rather than broadcast:
every hop is a ``qsend``/``qrecv`` pair, so at any moment exactly one rank
holds the qubit and the ranks it already left are back in ``|0⟩``.

Each rank names its neighbours with ``get_next_rank`` and
``get_prev_rank``, the way an MPI program walks a ring, so the same code
works for any number of ranks from two upwards.

The qubit is measured only at the end of the chain, so the last rank reads
0 or 1 about equally often — the ``|+⟩`` rank 0 prepared, having survived
every hop — and no other rank contributes counts.

Run with::

    netqmpi -n 3 --cunqa 2_round_robin.py
"""
from netqmpi.sdk.environment import Environment


def main(env: Environment = None):
    comm = env.comm
    rank = comm.rank
    size = comm.size

    next_rank = comm.get_next_rank(rank)
    previous_rank = comm.get_prev_rank(rank)

    with comm:
        circuit = env.create_circuit(num_qubits=1, num_clbits=1)

        if rank == 0:
            circuit.h(0)                                # prepare |+>
            comm.qsend(circuit, [0], next_rank)
        else:
            comm.qrecv(circuit, [0], previous_rank)

            # Everyone but the end of the chain passes it straight on.
            if rank != size - 1:
                comm.qsend(circuit, [0], next_rank)
            else:
                circuit.measure(0, 0)

    # Only the last rank to leave the block sees the results, and it sees
    # every rank's counts, keyed by rank.
    if comm.results:
        for other, counts in comm.results.items():
            print(f"rank_{other}: {counts}")
