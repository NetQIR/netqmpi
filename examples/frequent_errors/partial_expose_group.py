"""A three-rank window that one of its three participants never joins.

Ranks 0 and 1 open a window whose participant list names rank 2 as well,
but rank 2 goes straight to its measurement. Unlike lonely_expose.py, the
window is not waiting on everybody: it has two of its three participants,
so the report names who is already there and who is not.

Layer: CUNQA adapter, joint translation.   Raised by: translate_group
Expected: the deadlock report listing ranks 0 and 1 as the callers and
rank 2 as the participant that finished its block without calling it.

    netqmpi -n 3 --cunqa partial_expose_group.py
"""
from netqmpi.sdk.environment import Environment

RANKS = 3
ROOT = 0


def main(env: Environment = None):
    comm = env.comm
    rank = comm.rank

    with comm:
        circuit = env.create_circuit(num_qubits=1, num_clbits=1)

        if rank != 2:                  # rank 2 never reaches the window
            control = comm.expose(circuit, 0, [1, 2], root=ROOT)
            if rank == 1:
                circuit.cs(control, 0)
            comm.unexpose(circuit, [1, 2], root=ROOT)

        circuit.measure(0, 0)
