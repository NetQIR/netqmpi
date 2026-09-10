"""
Distributed GHZ echo — a *one-to-many teledata star* workload.

Rank 0 puts its first qubit in ``|+>`` and then takes it on a tour of every
other rank (``qsend`` out, ``qrecv`` back), so each host can apply a local
CNOT against it. That fans the entanglement out from a single control into
a GHZ state over the whole register: the broadcast pattern of an
``MPI_Bcast``, with every transfer touching rank 0.

A GHZ state cannot be scored from per-rank histograms — every rank's
marginal is 50/50 no matter how good the run was — so the program then
*undoes* the entanglement with a second, identical tour (CNOT is its own
inverse) and a final Hadamard. The register returns to ``|0...0>`` and the
noise-free outcome is all-zeros on every rank, the same criterion the
other probes use.

Cost: ``4 * (size - 1)`` transfers, linear in the number of ranks like
:mod:`cascade` but with a different topology — a star centred on rank 0
rather than a chain between neighbours, so the load concentrates on one
node instead of spreading along the line.

Parameters (environment variables, read at import):
    NQB_QUBITS_PER_RANK: qubits held by each rank (default 1).

Backend-agnostic: SDK abstractions only, runs unchanged on any backend.
"""
import os

from netqmpi.sdk.environment import Environment

QUBITS_PER_RANK = int(os.environ.get("NQB_QUBITS_PER_RANK", "1"))


def _fan_out(comm, circuit, rank, size, q):
    """
    Take rank 0's control on one tour of the ranks, CNOT-ing as it goes.

    Applied twice this is the identity: CNOT is self-inverse and the tour
    repeats in the same order, so the second pass disentangles what the
    first one built.

    Args:
        comm: The rank communicator.
        circuit: The circuit being built.
        rank: Index of the calling rank.
        size: Number of ranks.
        q: Qubits held by each rank.
    """
    scratch = q                      # landing slot for the visiting control

    if rank == 0:
        for target in range(1, q):
            circuit.cx(0, target)
        # The control rides in the scratch slot for the whole tour: the Aer
        # adapter transfers a qubit to the *same* local index on the
        # destination and ignores the index the receiver names, so sender and
        # receiver must agree on it for the program to mean the same thing on
        # every backend.
        circuit.swap(0, scratch)

    for host in range(1, size):
        if rank == 0:
            comm.qsend(circuit, [scratch], host)
            comm.qrecv(circuit, [scratch], host)
        elif rank == host:
            comm.qrecv(circuit, [scratch], 0)
            for target in range(q):
                circuit.cx(scratch, target)
            comm.qsend(circuit, [scratch], 0)

    if rank == 0:
        circuit.swap(0, scratch)


def main(env: Environment = None):
    comm = env.comm
    rank = comm.rank
    size = comm.size
    q = QUBITS_PER_RANK

    with comm:
        # q data qubits plus one scratch slot for the visiting control.
        circuit = env.create_circuit(num_qubits=q + 1, num_clbits=q)

        if rank == 0:
            circuit.h(0)

        _fan_out(comm, circuit, rank, size, q)   # build the GHZ
        _fan_out(comm, circuit, rank, size, q)   # and undo it

        if rank == 0:
            circuit.h(0)

        # Back to |0...0>: noise-free outcome is all zeros on every rank.
        for i in range(q):
            circuit.measure(i, i)
