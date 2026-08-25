"""Measures every qubit into a register that is too small.

Layer: SDK, while tracing.        Raised by: Circuit.measure_all
Expected: ValueError comparing the two sizes.

    netqmpi -n 2 --cunqa not_enough_clbits.py
"""
from netqmpi.sdk.environment import Environment

RANKS = 2


def main(env: Environment = None):
    comm = env.comm

    with comm:
        circuit = env.create_circuit(num_qubits=3, num_clbits=1)
        circuit.h(0)
        circuit.measure_all()     # 3 qubits, 1 clbit
