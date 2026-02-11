from netqmpi.sdk.communicator import QMPICommunicator

def print_info(message, rank):
    """
    Print the message with the rank number.
    """
    print(f"rank_{rank}: {message}")

def main(app_config=None, rank=0, size=1):
    COMM_WORLD = QMPICommunicator(rank, size, app_config)
    ROOT_RANK = 0

    with COMM_WORLD:
        qubits = []

        if rank == ROOT_RANK:
            # Create a qubit |++++> to scatter between the nodes
            qubits = [COMM_WORLD.create_qubit() for _ in range(size)]
            for q in qubits:
                q.H()

        COMM_WORLD.flush()
        local_qubits = COMM_WORLD.qscatter(qubits, ROOT_RANK)

        # Measure the qubits
        binary_code = [q.measure() for q in local_qubits]
        COMM_WORLD.flush()

        print_info(f"my binary result is {binary_code}", rank)

if __name__ == "__main__":
    main()