"""Uses an exposed control after the window that lent it was closed.

Layer: SDK, while tracing.        Raised by: Circuit._check_qubit
Expected: IndexError saying the communication qubit's window is closed.

    netqmpi -n 2 --cunqa closed_expose_window.py
"""
import numpy as np
from netqmpi.sdk.environment import Environment

RANKS = 2


def main(env: Environment = None):
    comm = env.comm
    rank = comm.rank

    with comm:
        circuit = env.create_circuit(num_qubits=1, num_clbits=1)
        control = comm.expose(circuit, 0, [0], root=1)
        comm.unexpose(circuit, [0], root=1)

        if rank == 0:
            circuit.cp(control, 0, np.pi / 2)   # the control is gone
        circuit.measure(0, 0)
