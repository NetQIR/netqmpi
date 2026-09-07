"""One rank builds two circuits while the others build one.

Circuits are paired across ranks by creation order, so the groups cannot
be formed.

Layer: CUNQA communicator, at the end of the run.
Raised by: CunqaCommunicator._execute_session
Expected: RuntimeError listing how many circuits each rank built.

    netqmpi -n 2 --cunqa mismatched_circuit_counts.py
"""
from netqmpi.sdk.environment import Environment

RANKS = 2


def main(env: Environment = None):
    comm = env.comm
    rank = comm.rank

    with comm:
        circuit = env.create_circuit(num_qubits=1, num_clbits=1)
        circuit.h(0)
        circuit.measure(0, 0)

        if rank == 0:
            extra = env.create_circuit(num_qubits=1, num_clbits=1)
            extra.x(0)
            extra.measure(0, 0)
