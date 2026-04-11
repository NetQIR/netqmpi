from netqmpi.sdk.environment import Environment

def print_info(message, rank):
    """
    Print the message with the rank number.
    """
    print(f"rank_{rank}: {message}")

def main(env: Environment = None):
    comm = env.comm
    rank = comm.rank
    
    next_rank = comm.get_next_rank(rank)
    previous_rank = comm.get_prev_rank(rank)

    with comm: # Sólo se ejecuta lo que hay en el with
        if rank == 0:
            circuit = env.create_circuit(num_qubits=1, num_clbits=0)
            circuit.h(0)
            comm.qsend(circuit, [0], next_rank)
        else:
            circuit = env.create_circuit(num_qubits=1, num_clbits=1)
            comm.qrecv(circuit, [0], previous_rank)
            circuit.measure(0, 0)

    results = comm.results # Estos son los resultados de la ejecución

    if rank != 0:
        print(f"measure: {results}") # Que automaticamente imprima el rank desde donde se imprime
    else:
        print("teleportation complete")