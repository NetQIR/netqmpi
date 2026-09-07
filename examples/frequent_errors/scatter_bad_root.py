"""Names a root that is not a rank of the communicator.

Layer: SDK, while tracing.        Raised by: Circuit._rooted_chunk
Expected: ValueError naming the root and the valid rank range.

    netqmpi -n 3 --cunqa scatter_bad_root.py
"""
from netqmpi.sdk.environment import Environment

RANKS = 3


def main(env: Environment = None):
    comm = env.comm

    with comm:
        circuit = env.create_circuit(num_qubits=1, num_clbits=1)
        comm.qscatter(circuit, [0], root=7)     # there is no rank 7
        circuit.measure(0, 0)
