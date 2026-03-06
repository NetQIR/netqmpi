from netqmpi.sdk.core.environment import Environment

def main(env: Environment = None): # type: ignore
    comm = env.comm
    rank = comm.get_rank()
    ROOT_RANK = 0

    with comm:
        circuit = env.create_circuit(num_qubits=1, num_clbits=1)

        if rank == ROOT_RANK:
            circuit.h(0)

        with circuit.expose([0], rank=ROOT_RANK):
            if rank == 1:
                # Operate on the exposed qubit from another rank
                circuit.cx(0, 0)  # example: identity CNOT (both same qubit)

        circuit.measure(0, 0)
        result = circuit.build()
        comm.flush()

        print(f"rank_{rank} measured {result['results'][0]}")