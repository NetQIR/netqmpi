from netqmpi.sdk.communicator.communicator import QMPICommunicator
from netqmpi.sdk.core.circuit import Circuit

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
        circuit = Circuit(size, size)
        for q in range(size):
            circuit.H(q)

        full_qubits = comm.qgather(circuit, [], ROOT_RANK)
        comm.flush()

        if rank == ROOT_RANK:
            # Measure the qubits
            for q in full_qubits:
                value = q.measure()
                comm.flush()
                print_info(f"{q}-{value}", rank)

if __name__ == "__main__":
    main()