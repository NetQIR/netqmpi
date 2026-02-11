from netqmpi.sdk.communicator import QMPICommunicator

def main(app_config = None, rank=0, size=1):
    COMM_WORLD = QMPICommunicator(rank, size, app_config)
    ROOT_RANK = 0

    with COMM_WORLD:
        qubit_exposed = []

        my_qubit = COMM_WORLD.create_qubit()

        if rank == ROOT_RANK:
            qubit_exposed.append(my_qubit)
            my_qubit.H()

        COMM_WORLD.expose(qubit_exposed, 0)

        if rank == 1:
            qubit_exposed[0].cnot(my_qubit)

        COMM_WORLD.unexpose(0)

        measure = my_qubit.measure()

        COMM_WORLD.flush()

        print(f"rank_{rank} measured {measure}")