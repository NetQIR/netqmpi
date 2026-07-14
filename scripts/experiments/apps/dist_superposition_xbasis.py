"""
Distributed-superposition fidelity probe (X-basis readout).

Rank 0 prepares |+> with an H gate and teleports the qubit to rank 1 with
``qsend``. Rank 1 receives it with ``qrecv``, rotates back to the X basis with a
final H, and measures. With a perfect qdevice the teleported |+> is mapped to
|0>, so the noise-free expected outcome is a deterministic ``0``. Under memory
(T1/T2) or gate (depolarising) noise, P(outcome == 0) drops from 1 towards 0.5,
which makes this a valid fidelity probe (unlike measuring |+> directly in Z,
which is 50/50 regardless of noise).

Backend-agnostic: identical to ``examples/netqmpi/send_recv.py`` except for the
extra H before the measurement. Runs unchanged on any NetQMPI backend.
"""
from netqmpi.sdk.environment import Environment


def main(env: Environment = None):
    comm = env.comm
    rank = comm.rank

    next_rank = comm.get_next_rank(rank)
    previous_rank = comm.get_prev_rank(rank)

    with comm:
        if rank == 0:
            circuit = env.create_circuit(num_qubits=1, num_clbits=1)
            circuit.h(0)                          # prepare |+>
            comm.qsend(circuit, [0], next_rank)   # teleport to rank 1
        else:
            circuit = env.create_circuit(num_qubits=1, num_clbits=1)
            comm.qrecv(circuit, [0], previous_rank)
            circuit.h(0)                          # rotate to X basis
            circuit.measure(0, 0)                 # noise-free expected: 0

    results = comm.results

    if rank != 0:
        print(f"measure: {results}")
    else:
        print("teleportation complete")
