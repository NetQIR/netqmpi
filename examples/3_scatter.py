"""
Distributed scatter: one rank hands a qubit to each of the others.

``qscatter`` is collective, so every rank has to call it, and rooted, so
one rank passes the whole buffer and the others the qubits their chunk is
to land on. The buffer is split in rank order, one chunk per rank *other*
than the root.

The catch a classical scatter does not have is that qubits cannot be
copied. The chunks are teleported, which *consumes* them: the root gives
its buffer away in full and keeps nothing, so its qubits are back in
``|0⟩`` when the call returns, and the states it prepared are alive only
on the ranks they moved to. That is what the measurements below show.

Run with::

    netqmpi -n 3 --cunqa 3_scatter.py
"""
from netqmpi.sdk.environment import Environment

ROOT = 0


def main(env: Environment = None):
    comm = env.comm
    rank, size = comm.rank, comm.size

    with comm:
        if rank == ROOT:
            # One qubit for each of the other ranks. All of them leave the
            # circuit, so the root measures 0 everywhere whatever it
            # prepared.
            circuit = env.create_circuit(num_qubits=size - 1, num_clbits=size - 1)
            for q in range(size - 1):
                circuit.x(q)                     # |1⟩ on every qubit
            comm.qscatter(circuit, list(range(size - 1)), root=ROOT)
            circuit.measure_all()
        else:
            circuit = env.create_circuit(num_qubits=1, num_clbits=1)
            mine = comm.qscatter(circuit, [0], root=ROOT)
            circuit.measure(mine[0], 0)

    # Every rank but the root should read a 1: that is the |1⟩ the root
    # prepared, now living on the receiver. The root reads 0 everywhere,
    # having given its whole buffer away.
    if comm.results:
        for other, counts in comm.results.items():
            print(f"rank_{other}: {counts}")
