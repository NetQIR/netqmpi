"""A script without the main() entry point NetQMPI expects.

Layer: CLI, before anything runs.        Raised by: netqmpi.helpers.load_main
Expected: ValueError naming the file and the missing function.

    netqmpi -n 2 --cunqa no_main.py
"""
from netqmpi.sdk.environment import Environment

RANKS = 2


def run(env: Environment = None):      # not called main()
    with env.comm:
        circuit = env.create_circuit(num_qubits=1, num_clbits=1)
        circuit.h(0)
        circuit.measure(0, 0)
