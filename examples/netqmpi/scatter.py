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
        qubits = []

        if rank == ROOT_RANK:
            # Create a qubit |++++> to scatter between the nodes
            qubits = [comm.create_qubit() for _ in range(size)]
            for q in qubits:
                q.H()

        comm.flush()
        local_qubits = comm.qscatter(qubits, ROOT_RANK)

        # Measure the qubits
        binary_code = [q.measure() for q in local_qubits]
        comm.flush()

        print_info(f"my binary result is {binary_code}", rank)

if __name__ == "__main__":
    main()