"""
Distributed 3-qubit QFT built out of telegates.

Each rank holds one qubit of the transform. The controlled rotations that
cross ranks are not moved qubit by qubit: the control is *exposed*, that
is, shared with the ranks that need it through a GHZ state, so they can
apply the rotation locally and give the control back untouched.

``expose`` and ``unexpose`` are collective, like an ``MPI_Bcast``: every
rank listed in the window has to call them, and the root is the rank
lending its qubit. On the root the call returns its own data qubit, on the
receivers the communication qubit now carrying that control.

Run with::

    netqmpi -n 3 --cunqa qft_expose.py
"""
import numpy as np
from netqmpi.sdk.environment import Environment


def main(env: Environment = None):
    comm = env.comm
    rank = comm.rank

    with comm:
        circuit = env.create_circuit(num_qubits=1, num_clbits=1)

        # Rank 2 lends its qubit to ranks 0 and 1: both need it as the
        # control of a rotation. The window stays open until the very end.
        control_2 = comm.expose(circuit, 0, [0, 1], root=2)
        # Rank 1 lends its qubit to rank 0 for the CS gate.
        control_1 = comm.expose(circuit, 0, [0], root=1)

        if rank == 0:
            circuit.h(0)
            circuit.cp(control_1, 0, np.pi/2) # CS gate
            circuit.cp(control_2, 0, np.pi/4) # CT gate

        # Rank 1 is done being a control, so it can move on to its own H.
        comm.unexpose(circuit, [0], root=1)

        if rank == 1:
            circuit.h(0)
            circuit.cp(control_2, 0, np.pi/2) # CS gate

        comm.unexpose(circuit, [0, 1], root=2)

        if rank == 2:
            circuit.h(0)

        circuit.measure(0, 0)

    # The circuits of all the ranks are submitted together when the last
    # rank leaves the block, so only that rank sees the results here — and
    # it sees every rank's counts, keyed by rank.
    if comm.results:
        for other, counts in comm.results.items():
            print(f"rank_{other}: {counts}")
