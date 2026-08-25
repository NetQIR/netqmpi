"""One rank opens a telegate window its peer never joins.

Collective calls must be reached by every participant. Here only the root
calls expose, so the group can never be completed.

Layer: CUNQA adapter, joint translation.   Raised by: translate_group
Expected: the deadlock report listing what each rank is waiting for.

    netqmpi -n 2 --cunqa lonely_expose.py
"""
from netqmpi.sdk.environment import Environment

RANKS = 2


def main(env: Environment = None):
    comm = env.comm
    rank = comm.rank

    with comm:
        circuit = env.create_circuit(num_qubits=1, num_clbits=1)

        if rank == 1:                  # only the root calls it
            comm.expose(circuit, 0, [0], root=1)
            comm.unexpose(circuit, [0], root=1)

        circuit.measure(0, 0)
