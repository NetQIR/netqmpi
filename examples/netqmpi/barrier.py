from netqmpi.sdk.communicator import QMPICommunicator

def print_info(message, rank):
    """
    Print the message with the rank number.
    """
    print(f"rank_{rank}: {message}")

def main(app_config=None, rank=0, size=1):
    COMM_WORLD = QMPICommunicator(rank, size, app_config)

    with COMM_WORLD:
        # Each rank does some initial work
        print_info(f"Starting phase 1", rank)
        
        # Create a qubit and perform some operation
        q = COMM_WORLD.create_qubit()
        if rank % 2 == 0:
            q.H()  # Even ranks apply Hadamard
        else:
            q.X()  # Odd ranks apply X gate
        
        print_info(f"Completed phase 1, waiting at barrier", rank)
        
        # Synchronization point - all ranks must reach here
        COMM_WORLD.barrier()
        
        print_info(f"Passed barrier, starting phase 2", rank)
        
        # Phase 2: All ranks have completed phase 1
        # Perform additional operations knowing all ranks are synchronized
        q.H()
        measurement = q.measure()
        
        COMM_WORLD.flush()
        print_info(f"Phase 2 complete, measurement: {measurement}", rank)

if __name__ == "__main__":
    main()
