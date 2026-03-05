from netqmpi.sdk.core.environment import Environment


def print_info(message, rank):
    """
    Print the message with the rank number.
    """
    print(f"rank_{rank}: {message}")

def main(env: Environment = None):
    comm = env.comm
    rank = comm.get_rank()
    size = comm.get_size()
    next_rank = comm.get_next_rank(rank)
    previous_rank = comm.get_prev_rank(rank)

    with comm:
        if rank == 0:
            circuit = env.create_circuit(num_qubits=1, num_clbits=0)
            circuit.h(0)
            print_info(f"start to teleport a qubit to rank_{next_rank}", rank)
            circuit.qsend([0], next_rank)
            circuit.build()
        else:
            circuit = env.create_circuit(num_qubits=1, num_clbits=1)
            print_info(f"starting to receive a qubit from {previous_rank}", rank)
            circuit.qrecv([0], previous_rank)

            # Send to next rank only if not the last rank
            if rank != size - 1:
                print_info(f"received a qubit from rank_{previous_rank}", rank)
                circuit.qsend([0], next_rank)
                print_info(f"sent a qubit to rank_{next_rank}", rank)
                circuit.build()
            else:
                circuit.measure(0, 0)
                result = circuit.build()
                comm.flush()
                print_info(f"measurement {result['results'][0]}", rank)

if __name__ == "__main__":
    main()