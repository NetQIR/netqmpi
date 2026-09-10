"""
Distributed QFT echo — an *all-to-all teledata* workload.

Every rank holds ``q`` qubits of an ``N = size * q`` qubit register. The
program applies the QFT and then its inverse, so the net operation is the
identity: an input basis state ``|x>`` comes back as ``|x>`` and every rank
reads its own slice of ``x`` deterministically. That makes the noise-free
outcome exact and turns any deviation into a fidelity signal sensitive to
*every* controlled rotation — unlike a bare QFT on a basis state, whose
Z-basis histogram is uniform no matter what the rotations do.

Controlled rotations that cross ranks are resolved with **teledata**: the
control qubit itself travels to the rank holding the targets (``qsend``),
the rotations are applied locally, and the qubit travels back
(``qrecv``). Each rank keeps one scratch qubit as the landing slot for a
visiting control, so a rank's register is ``q + 1`` qubits wide.

This is the portable half of the QFT pair. Its twin
:mod:`qft_telegate` performs the same transform by *sharing* the control
with ``expose``/``unexpose`` instead of moving it; that variant only runs
on CUNQA, so comparing the two there is what isolates the cost of telegate
against teledata.

Cost: ``2 * q * size * (size - 1)`` transfers, quadratic in the number of
ranks — the communication-heaviest probe of the suite.

The final bit-reversal swaps of a textbook QFT are omitted: they cancel
between the transform and its inverse, and emitting them would add
cross-rank qubit moves that say nothing about the rotations under test.

Parameters (environment variables, read at import):
    NQB_QUBITS_PER_RANK: qubits held by each rank (default 1).
    NQB_INPUT: integer input state ``x`` (default 0), taken modulo 2^N.
        The default of 0 is what the fidelity runs use: the echo is
        sensitive to every rotation whatever the input, and an all-zero
        expected outcome makes the success criterion independent of each
        backend's bit ordering. Set it non-zero for a correctness check.

Backend-agnostic: SDK abstractions only, runs unchanged on any backend.
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

    The transform is ordered by *control* rather than by target. That is
    equivalent — the controlled-phase gates are diagonal, so they commute
    with each other and with the Hadamards on qubits they do not touch —
    and it lets each control make a single tour of the ranks that hold its
    targets instead of one round trip per gate.

    Args:
        comm: The rank communicator.
        circuit: The circuit being built.
        rank: Index of the calling rank.
        size: Number of ranks.
        q: Qubits held by each rank.
        inverse: Whether to emit the inverse transform.
    """
    n = size * q
    scratch = q                      # landing slot for a visiting control
    controls = range(n - 1, -1, -1) if inverse else range(n)

    for k in controls:
        root, local_k = divmod(k, q)

        # In the inverse transform the Hadamard comes before the rotations.
        if inverse and rank == root:
            circuit.h(local_k)

        # Targets on the control's own rank need no transfer at all.
        if rank == root:
            for local_j in range(local_k):
                circuit.cp(local_k, local_j, _angle(local_k - local_j, inverse))

        # Every rank below the control's owner holds only targets j < k, so
        # the control tours them one by one. It rides in the scratch slot on
        # both sides of every hop: the Aer adapter transfers a qubit to the
        # *same* local index on the destination and ignores the index the
        # receiver names, so sender and receiver must agree on it for the
        # program to mean the same thing on every backend.
        if root and rank == root:
            circuit.swap(local_k, scratch)

        for host in range(root):
            if rank == root:
                comm.qsend(circuit, [scratch], host)
                comm.qrecv(circuit, [scratch], host)
            elif rank == host:
                comm.qrecv(circuit, [scratch], root)
                for local_j in range(q):
                    distance = k - (rank * q + local_j)
                    circuit.cp(scratch, local_j, _angle(distance, inverse))
                comm.qsend(circuit, [scratch], root)

        if root and rank == root:
            circuit.swap(local_k, scratch)

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
        # q data qubits plus one scratch slot for visiting controls.
        circuit = env.create_circuit(num_qubits=q + 1, num_clbits=q)

        for i, bit in enumerate(my_bits):
            if bit:
                circuit.x(i)

        _half(comm, circuit, rank, size, q, inverse=False)
        _half(comm, circuit, rank, size, q, inverse=True)

        for i in range(q):
            circuit.measure(i, i)
