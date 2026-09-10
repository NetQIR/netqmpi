"""
Teleportation cascade — a *point-to-point teledata* workload.

Rank 0 prepares ``q`` qubits in ``|+>`` and hands them down the whole
chain of ranks with ``qsend``/``qrecv``: 0 -> 1 -> ... -> size-1. The last
rank rotates back to the X basis with a Hadamard and measures.

This is the teledata counterpart of :mod:`qft`: no control is ever shared,
every transfer *moves* a state, and the cost grows as O(q * (size - 1))
EPR pairs — linear in the number of ranks rather than quadratic.

Because a teleported qubit leaves its source in ``|0>``, the noise-free
outcome is all-zeros on **every** rank: rank 0 gave its qubits away,
the intermediate ranks received and forwarded them, and the last rank
maps the ``|+>`` it holds back to ``|0>``. A single, uniform success
criterion for the whole run.

Parameters (environment variables, read at import):
    NQB_QUBITS_PER_RANK: qubits cascaded down the chain (default 1).

Backend-agnostic: SDK abstractions only, runs unchanged on any backend.
"""
import os

from netqmpi.sdk.environment import Environment

QUBITS_PER_RANK = int(os.environ.get("NQB_QUBITS_PER_RANK", "1"))


def main(env: Environment = None):
    comm = env.comm
    rank = comm.rank
    size = comm.size
    q = QUBITS_PER_RANK

    qubits = list(range(q))

    with comm:
        circuit = env.create_circuit(num_qubits=q, num_clbits=q)

        if rank == 0:
            for i in qubits:
                circuit.h(i)                       # prepare |+>
            if size > 1:
                comm.qsend(circuit, qubits, 1)     # and give them away
        else:
            comm.qrecv(circuit, qubits, rank - 1)
            if rank < size - 1:
                comm.qsend(circuit, qubits, rank + 1)
            else:
                for i in qubits:
                    circuit.h(i)                   # rotate back to the X basis

        # Every rank ends in |0...0>: the senders gave their state away and
        # the last one rotated it back. Noise-free outcome: all zeros.
        for i in qubits:
            circuit.measure(i, i)
