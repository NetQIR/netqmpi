"""
Environment object injected into every user ``main()`` function.

This class acts as the bridge between the runtime layer, which sets up
the multi-process execution environment and selects the backend, and the
SDK layer, which provides the user-facing programming API.

By depending only on this class, user applications remain fully
backend-agnostic and do not need to import any concrete executor or
backend adapter.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from netqmpi.sdk.communicator import QMPICommunicator
from netqmpi.sdk.circuit import Circuit

if TYPE_CHECKING:
    from netqmpi.runtime import Executor


class Environment:
    """
    Runtime context injected into every user ``main()`` function.

    This class encapsulates two responsibilities:

    - Communication, through the
      :class:`~netqmpi.sdk.communicator.QMPICommunicator` exposed by the
      :attr:`comm` property.
    - Circuit creation, through :meth:`create_circuit`, which delegates
      to the backend-specific
      :class:`~netqmpi.runtime.executor.Executor` selected by the
      runtime.

    Example:
        def main(env: Environment = None):
            rank = env.comm.rank
            circuit = env.create_circuit(num_qubits=2, num_clbits=1)

            with env.comm:
                ...

    Args:
        comm: Communicator associated with this rank.
        executor: Executor responsible for creating backend-specific
            circuit instances.
    """

    def __init__(self, comm: QMPICommunicator, executor: Executor) -> None:
        """
        Initialize the environment.

        Args:
            comm: Communicator associated with this rank.
            executor: Executor responsible for creating backend-specific
                circuit instances.
        """
        self._comm = comm
        self._executor = executor

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def comm(self) -> QMPICommunicator:
        """
        Return the communicator associated with this rank.

        Returns:
            The rank communicator.
        """
        return self._comm

    def create_circuit(self, num_qubits: int, num_clbits: int) -> Circuit:
        """
        Create a backend-specific quantum circuit.

        This method delegates circuit creation to the underlying
        :class:`~netqmpi.runtime.executor.Executor`, allowing user code
        to remain backend-agnostic.

        Args:
            num_qubits: Number of qubits in the circuit.
            num_clbits: Number of classical bits in the circuit.

        Returns:
            A backend-specific :class:`~netqmpi.sdk.circuit.Circuit`
            instance ready to receive quantum operations.
        """
        return self._executor.create_circuit(num_qubits, num_clbits, comm=self.comm)