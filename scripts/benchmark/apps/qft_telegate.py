"""
Distributed QFT echo, telegate variant — CUNQA only.

Every rank holds ``q`` qubits of an ``N = size * q`` qubit register. The
program applies the QFT and then its inverse, so the net operation is the
identity: an input basis state ``|x>`` comes back as ``|x>`` and every
rank reads its own slice of ``x`` deterministically. That makes the
noise-free outcome exact and turns any deviation into a fidelity signal
sensitive to *every* controlled rotation — unlike a bare QFT on a basis
state, whose Z-basis histogram is uniform no matter what the rotations do.

The controlled rotations that cross ranks are telegates: the control is
``expose``d (shared through a GHZ state, CUNQA's cat-entangler) so the
holders of the targets apply the rotation locally and hand the control
back untouched. Communication therefore grows as O(N^2) in gates and
O(N - q) in expose windows per half, which is what makes this the
communication-heaviest probe of the suite.

The final bit-reversal swaps of a textbook QFT are omitted: they cancel
between the transform and its inverse, and emitting them would add
cross-rank qubit moves that say nothing about the rotations under test.

Parameters (environment variables, read at import):
    NQB_QUBITS_PER_RANK: qubits held by each rank (default 1).
    NQB_INPUT: integer input state ``x`` (default 1), taken modulo 2^N.

Backend support: ``expose``/``unexpose`` are implemented **only by the
CUNQA adapter** today (Aer, Qoala and NetQASM raise ``NotImplementedError``),
so this variant is not portable. Its portable twin :mod:`qft` performs the
same transform over ``qsend``/``qrecv`` and runs on every backend; running
the two on CUNQA is what isolates the cost of telegate against teledata.
"""
import os

from netqmpi.sdk.environment import Environment

QUBITS_PER_RANK = int(os.environ.get("NQB_QUBITS_PER_RANK", "1"))
INPUT_STATE = int(os.environ.get("NQB_INPUT", "0"))

#: pi as a literal keeps the app free of a numpy import, which not every
#: backend environment is guaranteed to carry.
PI = 3.141592653589793


def _angle(distance: int, inverse: bool) -> float:
    """
    Return the controlled-phase angle for two qubits ``distance`` apart.

    Args:
        distance: Difference between the control and target global indices.
        inverse: Whether the angle belongs to the inverse transform.

    Returns:
        The rotation angle in radians, negated for the inverse transform.
    """
    theta = PI / (2 ** distance)
    return -theta if inverse else theta


def _half(comm, circuit, rank, size, q, inverse):
    """
    Emit one half of the echo: the QFT, or its inverse.

    The transform is ordered by *control* rather than by target, which is
    equivalent (the controlled-phase gates are diagonal, so they commute
    with each other and with the Hadamards on untouched qubits) and lets
    each control be exposed once for all of its targets instead of once
    per gate.

    Args:
        comm: The rank communicator.
        circuit: The circuit being built.
        rank: Index of the calling rank.
        size: Number of ranks.
        q: Qubits held by each rank.
        inverse: Whether to emit the inverse transform.
    """
    n = size * q
    controls = range(n - 1, -1, -1) if inverse else range(n)

    for k in controls:
        root, local_k = divmod(k, q)

        # In the inverse transform the Hadamard comes before the rotations.
        if inverse and rank == root:
            circuit.h(local_k)

        # Every rank below the control's owner holds only targets j < k.
        receivers = list(range(root))
        handle = None
        if receivers:
            # Collective: every rank calls it, non-participants get None.
            handle = comm.expose(circuit, local_k, receivers, root=root)

        if rank == root:
            # Targets living on the control's own rank need no telegate.
            for local_j in range(local_k):
                circuit.cp(local_k, local_j, _angle(local_k - local_j, inverse))
        elif rank in receivers:
            for local_j in range(q):
                distance = k - (rank * q + local_j)
                circuit.cp(handle, local_j, _angle(distance, inverse))

        if receivers:
            comm.unexpose(circuit, receivers, root=root)

        if not inverse and rank == root:
            circuit.h(local_k)


def main(env: Environment = None):
    comm = env.comm
    rank = comm.rank
    size = comm.size
    q = QUBITS_PER_RANK
    n = size * q

    # This rank's slice of the input, least-significant qubit first.
    x = INPUT_STATE % (2 ** n)
    my_bits = [(x >> (rank * q + i)) & 1 for i in range(q)]

    with comm:
        circuit = env.create_circuit(num_qubits=q, num_clbits=q)

        for i, bit in enumerate(my_bits):
            if bit:
                circuit.x(i)

        _half(comm, circuit, rank, size, q, inverse=False)
        _half(comm, circuit, rank, size, q, inverse=True)

        for i in range(q):
            circuit.measure(i, i)
