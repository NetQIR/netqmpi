from typing import List

import numpy as np
from netqasm.sdk import EPRSocket, Qubit, set_qubit_state
from netqasm.sdk.classical_communication.message import StructuredMessage
from netqasm.sdk.external import Socket
from netqasm.sdk.external import NetQASMConnection

from netqmpi.sdk.primitives.collective.collective import CollectiveCommTeledata
from netqmpi.sdk.primitives.p2p import P2PCommTeledata


class QMPICommunicator:
    def __init__(self, rank, size, app_config):
        self.rank = rank
        self.size = size
        self.app_config = app_config
        self.epr_sockets = {}
        self.sockets = {}

        for i in range(size):
            self.epr_sockets[self.__get_rank_name(i)] = {}
            self.sockets[self.__get_rank_name(i)] = {}

        # Create the combination of EPR Sockets for the rank that call the function
        self.epr_sockets[self.__get_rank_name(rank)] = {}

        for i in range(size):
            if i != rank:
                self.epr_sockets[self.__get_rank_name(rank)][self.__get_rank_name(i)] = EPRSocket(self.__get_rank_name(i))

        # Get the epr_sockets for this rank as a list
        self.epr_sockets_list = list(self.epr_sockets[self.__get_rank_name(rank)].values())

        self.connection = NetQASMConnection(
            app_name=app_config.app_name, log_config=app_config.log_config, epr_sockets=self.epr_sockets_list
        )

    def get_epr_sockets_list(self) -> List[EPRSocket]:
        return self.epr_sockets_list

    def get_rank(self) -> int:
        return self.rank

    def get_next_rank(self, rank: int) -> int:
        return (rank + 1) % self.size

    def get_prev_rank(self, rank: int) -> int:
        return (rank - 1) % self.size

    def __get_rank_name(self, rank: int) -> str:
        return f"rank_{rank}"

    def get_socket(self, my_rank: int, other_rank: int) -> Socket:
        # Get the dictionary of EPR sockets and sockets for the given rank
        my_sockets = self.sockets[self.__get_rank_name(my_rank)]

        # Check if the EPR socket already exists
        if self.__get_rank_name(other_rank) not in my_sockets:
            # Create a new EPR socket and add it to the dictionary
            my_sockets[self.__get_rank_name(other_rank)] = Socket(self.__get_rank_name(my_rank),
                                                                  self.__get_rank_name(other_rank))

        return my_sockets[self.__get_rank_name(other_rank)]

    def get_epr_socket(self, my_rank, other_rank) -> EPRSocket:
        # Get the dictionary of EPR sockets and sockets for the given rank
        my_eprs = self.epr_sockets[self.__get_rank_name(my_rank)]

        # Check if the EPR socket already exists
        if self.__get_rank_name(other_rank) not in my_eprs:
            # Create a new EPR socket and add it to the dictionary
            my_eprs[self.__get_rank_name(other_rank)] = EPRSocket(self.__get_rank_name(other_rank))

        return my_eprs[self.__get_rank_name(other_rank)]

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
