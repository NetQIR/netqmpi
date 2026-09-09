"""Contributes a chunk of a different size than the root reserved.

The root lays out one qubit per rank, but rank 1 hands over two. Nothing
at trace time compares the two sides, so the mismatch only shows up when
the transfers fail to pair up.

Layer: CUNQA adapter, joint translation.   Raised by: check_transfers
Expected: the mismatch reported as the count it is -- two qsends against
one qrecv -- rather than a deadlock once the circuits run.

    netqmpi -n 3 --cunqa gather_disagreeing_chunks.py
"""
from netqmpi.sdk.environment import Environment

RANKS = 3
ROOT = 0


def main(env: Environment = None):
    comm = env.comm
    rank = comm.rank

    with comm:
        if rank == ROOT:
            circuit = env.create_circuit(num_qubits=3, num_clbits=3)
            comm.qgather(circuit, [0, 1, 2], root=ROOT)
            circuit.measure_all()
        elif rank == 1:
            circuit = env.create_circuit(num_qubits=2, num_clbits=2)
            comm.qgather(circuit, [0, 1], root=ROOT)    # two, not one
            circuit.measure_all()
        else:
            circuit = env.create_circuit(num_qubits=1, num_clbits=1)
            comm.qgather(circuit, [0], root=ROOT)
            circuit.measure(0, 0)
