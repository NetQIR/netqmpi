"""Two ranks each lend a qubit to the other at the same time.

Both ranks do call ``expose`` here, which is what makes this different from
lonely_expose.py: each of them opens a window of its own and waits for its
neighbour to join it, while the neighbour is waiting for the opposite. It
is the collective equivalent of two processes that both post a receive.

Layer: CUNQA adapter, joint translation.   Raised by: translate_group
Expected: the deadlock report showing each rank as the one caller of its
own window and as the missing participant of the other's, closing with the
advice about reaching collectives in the same order.

    netqmpi -n 2 --cunqa crossed_expose.py
"""
from netqmpi.sdk.environment import Environment

RANKS = 2


def main(env: Environment = None):
    comm = env.comm
    rank = comm.rank
    neighbour = 1 - rank

    with comm:
        circuit = env.create_circuit(num_qubits=1, num_clbits=1)

        # Each rank is the root of its own window and a participant of the
        # other's -- but only ever calls the first, so neither completes.
        comm.expose(circuit, 0, [neighbour], root=rank)
        comm.unexpose(circuit, [neighbour], root=rank)

        circuit.measure(0, 0)
