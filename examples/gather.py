"""
Distributed gather: every rank hands its qubit to one of them.

``qgather`` is the quantum ``MPI_Gather`` and the mirror image of
``qscatter``: collective as well, with the root passing the whole buffer
the chunks are to land on — its own contribution already sitting at
position ``root`` — and every other rank passing the qubits it gives away.

The transfer moves the qubits rather than copying them, so a contributor
does not keep what it gathered: its slot is back in ``|0⟩`` once the call
returns, and the data lives on the root alone.

Run with::

    netqmpi -n 3 --cunqa gather.py
"""
from netqmpi.sdk.environment import Environment

ROOT = 0


def main(env: Environment = None):
    comm = env.comm
    rank, size = comm.rank, comm.size

    with comm:
        if rank == ROOT:
            # One slot per rank, rank r landing on qubit r. Only the root's
            # own slot carries anything before the call; the rest are the
            # empty slots the incoming states are reconstructed on.
            circuit = env.create_circuit(num_qubits=size, num_clbits=size)
            circuit.x(ROOT)
            comm.qgather(circuit, list(range(size)), root=ROOT)
            circuit.measure_all()
        else:
            circuit = env.create_circuit(num_qubits=1, num_clbits=1)
            circuit.x(0)                         # this rank's contribution
            comm.qgather(circuit, [0], root=ROOT)
            circuit.measure(0, 0)

    # The root should read a 1 on every slot — its own and the ones it
    # gathered — while the contributors read the 0 left behind by the
    # transfer.
    if comm.results:
        for other, counts in comm.results.items():
            print(f"rank_{other}: {counts}")
