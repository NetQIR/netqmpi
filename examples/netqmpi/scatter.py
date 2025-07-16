from netqasm.sdk import Qubit
from netqmpi.sdk.communicator import QMPICommunicator

def print_info(message, rank):
    """
    Print the message with the rank number.
    """
    print(f"rank_{rank}: {message}")

def main(app_config=None, rank=0, size=1):
    COMM_WORLD = QMPICommunicator(rank, size, app_config)
    ROOT_RANK = 0

    with COMM_WORLD.connection:
        qubits = []

        if rank == ROOT_RANK:
            # Create a qubit |++++> to scatter between the nodes
            qubits = [Qubit(COMM_WORLD.connection) for _ in range(size)]
            for q in qubits:
                q.H()

        COMM_WORLD.connection.flush()
        local_qubits = COMM_WORLD.qscatter(qubits, ROOT_RANK)

        # Measure the qubits
        binary_code = [q.measure() for q in local_qubits]
        COMM_WORLD.connection.flush()

        print_info(f"my binary result is {binary_code}", rank)

if __name__ == "__main__":
    main()