from abc import ABC, abstractmethod
from typing import List, TYPE_CHECKING

from netqasm.sdk.classical_communication.message import StructuredMessage
from netqasm.sdk.qubit import Qubit

if TYPE_CHECKING:
    from netqmpi.sdk.communicator import QMPICommunicator

class P2PComm(ABC):
    @staticmethod
    @abstractmethod
    def qsend(communicator: "QMPICommunicator", qubits: List[Qubit], dest_rank: int) -> None:
        """
        Send a list of qubits to the destination rank using teleportation.
        """
        pass

    @staticmethod
    @abstractmethod
    def qrecv(communicator: "QMPICommunicator", src_rank: int, expected_qubits: int) -> List[Qubit]:
        """
        Receive a qubit from the source rank using teleportation.

        """
        pass

class P2PCommTeledata(P2PComm):
    @staticmethod
    def qsend(communicator: "QMPICommunicator", qubits: List[Qubit], dest_rank: int) -> None:
        """
        Send data using the specified communication type.

        Args:
            qubit: The data to be sent.
            dest_rank: The destination rank.
            comm: The communicator instance.
        """
        epr_socket = communicator.get_epr_socket(communicator.rank, dest_rank)
        socket = communicator.get_socket(communicator.rank, dest_rank)

        for qubit in qubits:
            # Create EPR pairs
            epr = epr_socket.create_keep()[0]

            # Teleport
            qubit.cnot(epr)
            qubit.H()
            m1 = qubit.measure()
            m2 = epr.measure()

            socket.send_structured(StructuredMessage("Corrections", (m1, m2)))

    @staticmethod
    def qrecv(communicator: "QMPICommunicator", src_rank: int, expected_qubits: int = 1) -> List[Qubit]:
        """
                Receive data using the specified communication type.

                Args:
                    src_rank: The source rank.
                    expected_qubits: The expected number of qubits to receive.

                Returns:
                    The qubit received.
                """
        epr_socket = communicator.get_epr_socket(communicator.rank, src_rank)
        socket = communicator.get_socket(communicator.rank, src_rank)

        qubits = []

        for i in range(expected_qubits):
            epr = epr_socket.recv_keep()[0]

            communicator.connection.flush()

            # Get the corrections
            m1, m2 = socket.recv_structured().payload

            if m2 == 1:
                epr.X()
            if m1 == 1:
                epr.Z()

            communicator.connection.flush()

            # Create a new qubit to return
            q = Qubit(communicator.connection)

            # Swap the state of the qubit with the EPR pair
            epr.cnot(q)
            q.cnot(epr)
            epr.cnot(q)

            qubits.append(q)

        return qubits