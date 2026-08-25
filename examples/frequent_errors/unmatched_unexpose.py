"""Closes a telegate window that was never opened.

Layer: SDK, while tracing.        Raised by: Circuit.unexpose
Expected: RuntimeError naming the rank and the arguments it called with.

    netqmpi -n 2 --cunqa unmatched_unexpose.py
"""
from netqmpi.sdk.environment import Environment

RANKS = 2


def main(env: Environment = None):
    comm = env.comm

    with comm:
        circuit = env.create_circuit(num_qubits=1, num_clbits=1)
        comm.unexpose(circuit, [0], root=1)     # no matching expose
        circuit.measure(0, 0)
