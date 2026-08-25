"""Scatters a buffer that does not split evenly among the ranks.

Layer: SDK, while tracing.        Raised by: Circuit._rooted_chunk
Expected: ValueError with the buffer size and the number of ranks.

    netqmpi -n 3 --cunqa scatter_uneven_buffer.py
"""
from netqmpi.sdk.environment import Environment

RANKS = 3
ROOT = 0


def main(env: Environment = None):
    comm = env.comm
    rank = comm.rank

    with comm:
        if rank == ROOT:
            circuit = env.create_circuit(num_qubits=4, num_clbits=4)
            comm.qscatter(circuit, [0, 1, 2, 3], root=ROOT)   # 4 qubits, 3 ranks
            circuit.measure_all()
        else:
            circuit = env.create_circuit(num_qubits=1, num_clbits=1)
            comm.qscatter(circuit, [0], root=ROOT)
            circuit.measure(0, 0)
