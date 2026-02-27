def print_info(message, rank):
    """
    Print the message with the rank number.
    """
    print(f"rank_{rank}: {message}")

def main(comm=None):
    rank = comm.get_rank()
    size = comm.get_size()
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
            print_info(f"starting to receive a qubit from {previous_rank}", rank)
            [qubit_recv] = comm.qrecv(previous_rank)

            # Send to next rank only if not the last rank
            if rank != size - 1:
                print_info(f"received a qubit from rank_{previous_rank}", rank)
                comm.qsend([qubit_recv], next_rank)
                print_info(f"sent a qubit to rank_{next_rank}", rank)
            else:
                # Measure the qubit
                measurement = qubit_recv.measure()
                comm.flush()
                print_info(f"measurement {measurement}", rank)

if __name__ == "__main__":
    main()