from netqmpi.sdk.communicator.communicator import QMPICommunicator

def print_info(message, rank):
    """
    Print the message with the rank number.
    """
    print(f"rank_{rank}: {message}")

def main(comm : QMPICommunicator = None):
    rank = comm.get_rank()
    size = comm.get_size()
    ROOT_RANK = 0

    with comm:
        # Create a qubit |++++> to gather between the nodes
        local_qubits = [comm.create_qubit() for _ in range(size)]
        for q in local_qubits:
            q.H()

        full_qubits = comm.qgather(local_qubits, ROOT_RANK)
        comm.flush()

        if rank == ROOT_RANK:
            # Measure the qubits
            binary_code = []
            for q in full_qubits:
                value = q.measure()
                comm.flush()
                print_info(f"{q}-{value}", rank)

if __name__ == "__main__":
    main()