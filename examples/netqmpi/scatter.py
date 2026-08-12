"""
Distributed scatter: one rank hands a qubit to each of the others.

``qscatter`` is the quantum ``MPI_Scatter``: collective, so every rank has
to call it, and rooted, so one rank passes the whole buffer and the others
the qubits their chunk is to land on. The buffer is split in rank order,
one chunk per rank.

The catch a classical scatter does not have is that qubits cannot be
copied. The chunks are teleported, which *consumes* them: after the call
the root holds only its own chunk, and the qubits it scattered are back in
``|0⟩``. That is what the measurements below show — the root prepared each
qubit in a different state, and the correlations come out on the ranks the
states moved to, not on the root.

Run with::

    netqmpi -n 3 --cunqa scatter.py
"""
from netqmpi.sdk.environment import Environment

ROOT = 0


def main(env: Environment = None):
    comm = env.comm
    rank, size = comm.rank, comm.size

    with comm:
        if rank == ROOT:
            # One qubit per rank: rank r gets qubit r. Every qubit but the
            # root's own leaves the circuit, so its slot ends up in |0⟩ and
            # the root measures 0 there whatever it prepared.
            circuit = env.create_circuit(num_qubits=size, num_clbits=size)
            for q in range(size):
                circuit.x(q)                     # |1⟩ on every qubit
            mine = comm.qscatter(circuit, list(range(size)), root=ROOT)
            circuit.measure_all()
        else:
            circuit = env.create_circuit(num_qubits=1, num_clbits=1)
            mine = comm.qscatter(circuit, [0], root=ROOT)
            circuit.measure(mine[0], 0)

    # Every rank but the root should read a 1: that is the |1⟩ the root
    # prepared, now living on the receiver. The root reads 1 on its own
    # chunk alone, 0 on every slot it gave away.
    if comm.results:
        for other, counts in comm.results.items():
            print(f"rank_{other}: {counts}")
