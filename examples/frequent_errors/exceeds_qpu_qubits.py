"""Asks for more qubits than the vQPU was raised with.

The executor gives each rank a fixed slice of the merged register, sized
by the backend definition passed to qraise (4 data + 4 comm qubits in the
file the adapter currently hardcodes). A circuit that needs more spills
into the next rank's qubits.

Layer: CUNQA runtime (needs vQPUs).
Expected: today, no error at all -- the counts are simply wrong.

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
