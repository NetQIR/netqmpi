from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List

import numpy as np
from netqasm.sdk import Qubit
from netqasm.sdk.classical_communication.message import StructuredMessage

if TYPE_CHECKING:
    from netqmpi.sdk.communicator import QMPICommunicator

def list_split(lst: List, n: int) -> List[List]:
    """
    Split a list into n chunks.
    """
    avg = len(lst) // n
    rem = len(lst) % n
    chunks = []
    start = 0

    for i in range(n):
        end = start + avg + (1 if i < rem else 0)
        chunks.append(lst[start:end])
        start = end

    return chunks

class CollectiveComm(ABC):
    @staticmethod
    @abstractmethod
    def qscatter(communicator: "QMPICommunicator", qubits: List[Qubit], rank_sender: int = 0) -> List[Qubit]:
        """
        Scatter qubits from the sender rank to all other ranks.
        """
        pass

    @staticmethod
    @abstractmethod
    def qgather(communicator: "QMPICommunicator", qubits: List[Qubit], rank_recv: int = 0) -> List[Qubit]:
        """
        Gather qubits from all ranks to the sender rank.
        """
        pass

    @staticmethod
    @abstractmethod
    def expose(communicator: "QMPICommunicator", qubits: List[Qubit], rank: int = 0) -> None:
        """
        Expose qubits to the specified rank for the rest of the communicator. It's important to note that the user can't
        call another communication primitive before calling the "unexpose" method.
        """
        pass

    @staticmethod
    @abstractmethod
    def unexpose(communicator: "QMPICommunicator", rank: int = 0) -> None:
        """
        Unexpose qubits from the specified rank.
        """
        pass

class CollectiveCommTeledata(CollectiveComm):
    @staticmethod
    def qscatter(communicator: "QMPICommunicator", qubits: List[Qubit], rank_sender: int = 0) -> List[Qubit]:
        # Get the rank of the current process
        rank = communicator.get_rank()
        size = communicator.get_size()

        if rank == rank_sender:
            # If the current rank is the sender, send the qubits to all other ranks
            # Split the qubits into chunks for each rank
            qubits_per_rank = list_split(qubits, size)

            # Send the qubits to each rank
            for i in range(size):
                if i != rank_sender:
                    communicator.qsend(qubits_per_rank[i], i)

            return qubits_per_rank[rank_sender]

        else:
            # Receive the qubits from the sender
            # If the current rank is not the sender, receive the qubits
            qubits = communicator.qrecv(rank_sender)

            return qubits

    @staticmethod
    def qgather(communicator: "QMPICommunicator", qubits: List[Qubit], rank_recv: int = 0) -> List[Qubit]:
        # Get the rank of the current process
        rank = communicator.get_rank()
        size = communicator.get_size()

        if rank == rank_recv:
            # If the current rank is the receiver, receive the qubits from all other ranks
            # Create a list to store the received qubits
            received_qubits = [[]] * size

            # Receive the qubits from each rank
            for i in range(size):
                if i != rank_recv:
                    print(f"[{rank}] Recibiendo qubits de {i}")
                    received_qubits[i] = communicator.qrecv(i)
                else:
                    # If the current rank is the receiver, add its own qubits to the list
                    received_qubits[i] = qubits

            # Flatten the list of received qubits in a orderly manner
            flatten_recv_qubits = [qubit for sublist in received_qubits for qubit in sublist]

            return flatten_recv_qubits

        else:
            print(f"[{rank}] Enviando mis qubits a rank {rank_recv}")
            # Send the qubits to the receiver
            communicator.qsend(qubits, rank_recv)

            return qubits

    @staticmethod
    def expose(communicator: "QMPICommunicator", qubits: List[Qubit], rank: int = 0) -> None:
        """
        Expose qubits to the specified rank for the rest of the communicator.
        """
        # This method is a placeholder for future implementation
        raise NotImplementedError("Expose method is only available in the CollectiveCommTelegate class.")

    @staticmethod
    def unexpose(communicator: "QMPICommunicator", rank: int = 0) -> None:
        """
        Unexpose qubits from the specified rank.
        """
        # This method is a placeholder for future implementation
        raise NotImplementedError("Unexpose method is only available in the CollectiveCommTelegate class.")

class CollectiveCommTelegate(CollectiveCommTeledata):
    @staticmethod
    def expose(communicator: "QMPICommunicator", qubits: List[Qubit], rank: int = 0) -> None:
        """
        Expose qubits to the specified rank for the rest of the communicator.
        """
        # TODO: Although a list of qubits is received, only the first one will be exposed.
        communicator.qubits_exposed.extend(qubits)

        # Create GHZ state across all ranks
        communicator.ghz_qubit = communicator.create_ghz()

        if rank == communicator.rank:
            # If the current rank is the exposer, expose the qubit.
            communicator.qubits_exposed[0].cnot(communicator.ghz_qubit)

            # Measure the ghz qubit
            measure = communicator.ghz_qubit.measure()

            # Send the measurement result to all the communicator ranks
            for rnk in range(communicator.size):
                if rnk != communicator.rank:
                    socket = communicator.get_socket(communicator.rank, rnk)
                    socket.send_structured(StructuredMessage("Expose", (measure,)))
                    communicator.connection.flush()

        else:
            measure = communicator.get_socket(communicator.rank, rank).recv_structured().payload[0]
            bit = int(measure)

            # If bit is 1, apply an X gate to the GHZ qubit
            bit and communicator.ghz_qubit.X()
            qubits.insert(0, communicator.ghz_qubit)

    @staticmethod
    def unexpose(communicator: "QMPICommunicator", rank: int = 0) -> None:
        """
        Unexpose qubits from the specified rank.
        """

        # Remove the exposed qubits from the communicator
        communicator.qubits_exposed.clear()

        bits = []

        if rank == communicator.rank:
            # If the current rank is the exposer, recv the measurements of the GHZ qubits
            for other_rank in range(communicator.size):
                if other_rank != communicator.rank:
                    socket = communicator.get_socket(communicator.rank, other_rank)
                    bits.append(int(socket.recv_structured().payload[0]))
                    communicator.connection.flush()

            # Compute AND of all bits
            np.bitwise_and.reduce(bits) and communicator.ghz_qubit.Z()

        else:
            # Apply Hadamard gate to the GHZ qubit
            communicator.ghz_qubit.H()
            # Send the measurement result to the exposer
            socket = communicator.get_socket(communicator.rank, rank)
            measure = communicator.ghz_qubit.measure()

            communicator.connection.flush()
            socket.send_structured(StructuredMessage("Unexpose", (measure,)))

        # Flush the connection
        communicator.connection.flush()