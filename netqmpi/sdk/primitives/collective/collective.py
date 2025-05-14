from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List

import numpy as np
from netqasm.sdk import Qubit

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
        pass