from typing import List

from netqmpi.sdk.primitives.collective.collective import CollectiveCommTeledata, CollectiveCommTelegate
from netqmpi.sdk.primitives.p2p import P2PCommTeledata

class QMPICommunicator:
    def __init__(self, rank, size, executor, app_config):
        self.rank = rank
        self.size = size
        self.app_config = app_config
        self.executor = executor

    def __enter__(self):
        self.executor.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return self.connection.__exit__(exc_type, exc_val, exc_tb)

    def get_rank(self) -> int:
        return self.rank

    def get_next_rank(self, rank: int) -> int:
        return (rank + 1) % self.size

    def get_prev_rank(self, rank: int) -> int:
        return (rank - 1) % self.size

    def __get_rank_name(self, rank: int) -> str:
        return f"rank_{rank}"

    def get_size(self) -> int:
        return self.size

    def qsend(self, qubits: List[Qubit], dest_rank: int):
        """
        Send a qubit to the destination rank using teleportation.
        """
        P2PCommTeledata.qsend(self, qubits, dest_rank)

    def qrecv(self, src_rank: int) -> List[Qubit]:
        """
        Receive a qubit from the source rank using teleportation.
        """
        return P2PCommTeledata.qrecv(self, src_rank)

    def qscatter(self, qubits: List[Qubit], rank_sender: int) -> List[Qubit]:
        return CollectiveCommTeledata.qscatter(self, qubits, rank_sender)

    def qgather(self, qubits: List[Qubit], rank_recv: int) -> List[Qubit]:
        return CollectiveCommTeledata.qgather(self, qubits, rank_recv)

    def expose(self, qubits: List[Qubit], rank: int = 0):
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