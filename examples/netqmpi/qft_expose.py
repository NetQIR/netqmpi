from netqmpi.sdk.communicator.communicator import QMPICommunicator

def main(comm : QMPICommunicator = None):
    rank = comm.get_rank()
    ROOT_RANK = 0

    with comm:
        qubit_exposed = []

        my_qubit = comm.create_qubit()

        if rank == ROOT_RANK:
            qubit_exposed.append(my_qubit)
            my_qubit.H()

        comm.expose(qubit_exposed, 0)

        if rank == 1:
            qubit_exposed[0].cnot(my_qubit)

        comm.unexpose(0)

        measure = my_qubit.measure()

        comm.flush()

        print(f"rank_{rank} measured {measure}")