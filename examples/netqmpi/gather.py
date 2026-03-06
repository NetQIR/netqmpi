from netqmpi.sdk.core.environment import Environment

def print_info(message, rank):
    """
    Print the message with the rank number.
    """
    print(f"rank_{rank}: {message}")

def main(env: Environment = None): # type: ignore
    comm = env.comm
    rank = comm.get_rank()
    size = comm.get_size()
    ROOT_RANK = 0

    with comm:
        if rank == ROOT_RANK:
            # Root allocates size qubits (one per rank), applies H, gathers
            circuit = env.create_circuit(num_qubits=size, num_clbits=size)
            for i in range(size):
                circuit.h(i)
            circuit.qgather(list(range(size)), ROOT_RANK)
            circuit.measure_all()
            result = circuit.build()
            comm.flush()

            for i, val in enumerate(result['results']):
                print_info(f"qubit {i} -> {val}", rank)
        else:
            # Non-root ranks allocate 1 qubit, H, then send via gather
            circuit = env.create_circuit(num_qubits=1, num_clbits=0)
            circuit.h(0)
            circuit.qgather([0], ROOT_RANK)
            circuit.build()

if __name__ == "__main__":
    main()