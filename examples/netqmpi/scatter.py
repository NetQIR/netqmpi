from netqmpi.sdk.core.environment import Environment

def print_info(message, rank):
    """
    Print the message with the rank number.
    """
    print(f"rank_{rank}: {message}")

def main(env: Environment = None):
    comm = env.comm
    rank = comm.get_rank()
    size = comm.get_size()
    ROOT_RANK = 0

    with comm:
        if rank == ROOT_RANK:
            circuit = env.create_circuit(num_qubits=size, num_clbits=size)
            for i in range(size):
                circuit.h(i)
            circuit.qscatter(list(range(size)), ROOT_RANK)
        else:
            circuit = env.create_circuit(num_qubits=1, num_clbits=1)
            circuit.qscatter([0], ROOT_RANK)

        circuit.measure_all()
        result = circuit.build()
        comm.flush()

        print_info(f"my binary result is {result['results']}", rank)

if __name__ == "__main__":
    main()