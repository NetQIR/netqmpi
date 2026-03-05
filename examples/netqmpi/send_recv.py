from netqmpi.sdk.communicator.communicator import QMPICommunicator
from netqmpi.sdk.core.environment import Environment

def print_info(message, rank):
    """
    Print the message with the rank number.
    """
    print(f"rank_{rank}: {message}")

def main(env: Environment = None, comm: QMPICommunicator = None):
    rank = comm.get_rank()
    next_rank = comm.get_next_rank(rank)
    previous_rank = comm.get_prev_rank(rank)

    with comm:
        if rank == 0:
            # Create a qubit |+> to teleport
            circuit = env.create_circuit(1, 0, rank)
            circuit.h(0)
            print_info(f"start to teleport a qubit to rank_{next_rank}", rank)

            comm.qsend(circuit, [0], next_rank)
        else:
            circuit = env.create_circuit(1, 1, rank)
            comm.qrecv(circuit, [0], previous_rank)
            circuit.measure(0, 0)

if __name__ == "__main__":
    main()