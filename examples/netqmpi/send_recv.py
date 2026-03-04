from netqmpi.sdk.core.environment import Environment

def print_info(message, rank):
    """
    Print the message with the rank number.
    """
    print(f"rank_{rank}: {message}")

def main(env: Environment = None):
    comm = env.comm
    rank = comm.get_rank()
    next_rank = comm.get_next_rank(rank)
    previous_rank = comm.get_prev_rank(rank)

    with comm:
        if rank == 0:
            # Create a qubit |+> to teleport
            q = comm.create_qubit()
            q.H()
            print_info(f"start to teleport a qubit to rank_{next_rank}", rank)

            comm.qsend([q], next_rank)
        else:
            [qubit_recv] = comm.qrecv(previous_rank)
            measurement = qubit_recv.measure()
            comm.flush()
            print_info(f"measure: {measurement}", rank)

if __name__ == "__main__":
    main()