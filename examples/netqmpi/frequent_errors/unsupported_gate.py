"""Uses an operation the CUNQA backend does not implement.

The trace accepts it; only the backend translation rejects it, so the
error surfaces at the end of the block instead of at the offending line.

Layer: CUNQA adapter, joint translation.
Raised by: CunqaCircuitAdapter._translate_barrier
Expected: NotImplementedError naming the operation and the backend.

    netqmpi -n 2 --cunqa unsupported_gate.py
"""
from netqmpi.sdk.environment import Environment

RANKS = 2


def main(env: Environment = None):
    comm = env.comm

    with comm:
        circuit = env.create_circuit(num_qubits=1, num_clbits=1)
        circuit.h(0)
        circuit.barrier()          # no CUNQA counterpart
        circuit.measure(0, 0)
