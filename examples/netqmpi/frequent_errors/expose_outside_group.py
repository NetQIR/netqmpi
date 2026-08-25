"""Exposes a qubit to a rank that is not in the run.

Layer: CUNQA adapter, joint translation.   Raised by: translate_group
Expected: an error naming the rank that is outside the group.

    netqmpi -n 2 --cunqa expose_outside_group.py
"""
from netqmpi.sdk.environment import Environment

RANKS = 2


def main(env: Environment = None):
    comm = env.comm

    with comm:
        circuit = env.create_circuit(num_qubits=1, num_clbits=1)
        comm.expose(circuit, 0, [7], root=0)    # there is no rank 7
        comm.unexpose(circuit, [7], root=0)
        circuit.measure(0, 0)
