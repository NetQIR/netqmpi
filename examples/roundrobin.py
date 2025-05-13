from netqasm.sdk import Qubit
from netqmpi.sdk.communicator import QMPICommunicator

def print_info(message, rank):
    """
    Print the message with the rank number.
    """
    print(f"rank_{rank}: {message}")

def main(app_config=None, rank=0, size=1):
    COMM_WORLD = QMPICommunicator(rank, size, app_config)

    next_rank = COMM_WORLD.get_next_rank(rank)
    previous_rank = COMM_WORLD.get_prev_rank(rank)

    with COMM_WORLD.connection:
        if rank == 0:
            # Create a qubit |+> to teleport
            q = Qubit(COMM_WORLD.connection)
            q.H()
            print_info(f"start to teleport a qubit to rank_{next_rank}", rank)

            COMM_WORLD.qsend(q, next_rank)

        else:
            print_info(f"starting to receive a qubit from {previous_rank}", rank)
            qubit_recv = COMM_WORLD.qrecv(previous_rank)

            # Send to next rank only if not the last rank
            if rank != size - 1:
                print_info(f"received a qubit from rank_{previous_rank}", rank)
                COMM_WORLD.qsend(qubit_recv, next_rank)
                print_info(f"sent a qubit to rank_{next_rank}", rank)
            else:
                # Measure the qubit
                measurement = qubit_recv.measure()
                COMM_WORLD.connection.flush()
                print_info(f"measurement {measurement}", rank)

if __name__ == "__main__":
    main()