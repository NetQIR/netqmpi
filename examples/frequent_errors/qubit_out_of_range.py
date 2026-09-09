"""Addresses a qubit the circuit does not have.

Layer: SDK, while tracing.        Raised by: Circuit._check_qubit
Expected: IndexError naming the offending index and the valid range.

    netqmpi -n 2 --cunqa qubit_out_of_range.py
"""
from netqmpi.sdk.environment import Environment

RANKS = 2


def main(env: Environment = None):
    comm = env.comm

    with comm:
        circuit = env.create_circuit(num_qubits=2, num_clbits=2)
        circuit.h(0)
        circuit.cx(0, 5)          # only qubits 0 and 1 exist
        circuit.measure_all()
