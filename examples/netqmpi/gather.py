from netqasm.sdk import Qubit
from netqmpi.sdk.communicator import QMPICommunicator

def print_info(message, rank):
    """
    Print the message with the rank number.
    """
    print(f"rank_{rank}: {message}")

def main(app_config=None, rank=0, size=1):
    COMM_WORLD = QMPICommunicator(rank, size, app_config)

    my_rank = COMM_WORLD.get_rank()
    ROOT_RANK = 0

    with COMM_WORLD.connection:
        # Create a qubit |++++> to gather between the nodes
        local_qubits = [Qubit(COMM_WORLD.connection) for _ in range(size)]
        for q in local_qubits:
            q.H()

        full_qubits = COMM_WORLD.qgather(local_qubits, ROOT_RANK)
        COMM_WORLD.connection.flush()

        if my_rank == ROOT_RANK:
            # Measure the qubits
            binary_code = []
            for q in full_qubits:
                value = q.measure()
                COMM_WORLD.connection.flush()
                print_info(f"{q}-{value}", rank)

if __name__ == "__main__":
    main()