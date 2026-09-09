"""Asks for more qubits than the vQPU was raised with.

The vQPUs are raised from a definition file that fixes how many data and
communication qubits each of them has (4 and 4 in the file the examples
use). A circuit that needs more cannot be placed on one.

Layer: CUNQA, when the circuits are submitted (needs vQPUs).
Expected: CUNQA refuses the job with "Not enough data qubits in the QPU
for the circuit." -- which says neither how many were needed, nor how many
there are, nor which rank asked.

    netqmpi -n 2 --cunqa exceeds_qpu_qubits.py
"""
from netqmpi.sdk.environment import Environment

RANKS = 2


def main(env: Environment = None):
    comm = env.comm
    rank = comm.rank

    with comm:
        circuit = env.create_circuit(num_qubits=9, num_clbits=9)
        circuit.h(0)
        for q in range(1, 9):
            circuit.cx(0, q)
        if rank == 0:
            comm.qsend(circuit, [0], 1)
        else:
            comm.qrecv(circuit, [0], 0)
        circuit.measure_all()
