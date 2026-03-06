from typing import Optional

from netqmpi.sdk.core.environment import Environment

def print_info(message, rank):
    """
    Print the message with the rank number.
    """
    print(f"rank_{rank}: {message}")

def main(env: Environment = None): # type: ignore
    comm = env.comm
    rank = comm.get_rank()
    next_rank = comm.get_next_rank(rank)
    previous_rank = comm.get_prev_rank(rank)

    circuit = env.create_circuit(num_qubits=1, num_clbits=1)

    with comm:
        if rank == 0:
            circuit.h(0)
            print_info(f"start to teleport a qubit to rank_{next_rank}", rank)
            circuit.qsend([0], next_rank)
        else:
            circuit.qrecv([0], previous_rank)
            circuit.measure(0, 0)

        result = circuit.build()

        if rank != 0:
            print_info(f"measure: {result['results'][0]}", rank)
        else:
            print_info("teleportation complete", rank)

if __name__ == "__main__":
    main()