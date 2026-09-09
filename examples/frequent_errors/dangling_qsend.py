"""Sends a qubit nobody receives.

Point-to-point transfers are paired by tag at run time, so a send with no
matching recv is not detected while tracing.

Layer: CUNQA adapter, joint translation.   Raised by: check_transfers
Expected: an error naming the rank left holding the qubit, and how many
qsend and qrecv calls each side traced.

    netqmpi -n 2 --cunqa dangling_qsend.py
"""
from netqmpi.sdk.environment import Environment

RANKS = 2


def main(env: Environment = None):
    comm = env.comm
    rank = comm.rank

    with comm:
        circuit = env.create_circuit(num_qubits=1, num_clbits=1)
        if rank == 0:
            circuit.h(0)
            comm.qsend(circuit, [0], 1)     # rank 1 never calls qrecv
        else:
            circuit.measure(0, 0)
