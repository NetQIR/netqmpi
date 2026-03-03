from typing import List

from netqmpi.sdk.core.circuit import Circuit
from netqmpi.sdk.primitives.collective.collective import CollectiveCommTeledata, CollectiveCommTelegate
from netqmpi.sdk.primitives.p2p import P2PCommTeledata

class QMPICommunicator:
    def __init__(self, rank, size, executor, app_config = None):
        self.rank = rank
        self.size = size
        self.app_config = app_config
        self.executor = executor

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def get_rank(self) -> int:
        return self.rank

    def get_next_rank(self, rank: int) -> int:
        return (rank + 1) % self.size

    def get_prev_rank(self, rank: int) -> int:
        return (rank - 1) % self.size

    def get_size(self) -> int:
        return self.size

    def qsend(self, circuit, qubits: List[int], dest_rank: int):
        """
        Send a qubit to the destination rank using teleportation.
        """
        circuit.qsend(qubits, dest_rank)

    def qrecv(self, circuit, qubits: List[int], src_rank: int) -> List[int]:
        """
        Receive a qubit from the source rank using teleportation.
        """
        return circuit.qrecv(qubits, src_rank)

    def qscatter(self, qubits: List[int], rank_sender: int) -> List[int]:
        return CollectiveCommTeledata.qscatter(self, qubits, rank_sender)

    def qgather(self, qubits: List[int], rank_recv: int) -> List[int]:
        return CollectiveCommTeledata.qgather(self, qubits, rank_recv)

    def expose(self, qubits: List[int], rank: int = 0):
        """
        Expose qubits to the network.
        :param qubits: List of qubits to expose.
        :param rank: Exposer rank
        """
        CollectiveCommTelegate.expose(self, qubits, rank)
    def unexpose(self, rank: int = 0):
        """
        Unexpose qubits from the network.
        :param rank: Exposer rank
        :return: None
        """

        CollectiveCommTelegate.unexpose(self, rank)