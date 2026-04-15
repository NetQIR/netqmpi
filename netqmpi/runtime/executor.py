"""
Base abstraction for quantum circuit executors.

This module defines the contract that all quantum backend adapters must
implement, including circuit creation, application construction, and
application execution.

It belongs to the runtime layer, so imports from backend-specific
packages such as ``netqasm`` or ``cunqa`` must not appear here.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, Optional

from netqmpi.runtime.run_config import RunConfig
from netqmpi.sdk.circuit import Circuit

if TYPE_CHECKING:
    from netqmpi.sdk.communicator import QMPICommunicator


class Executor(ABC):
    """
    Abstract base class for quantum backend executors.

    This interface combines three responsibilities into a single adapter
    contract:

    1. Circuit factory through :meth:`create_circuit`.
    2. Application builder through :meth:`build_apps`.
    3. Application runner through :meth:`run`.

    Attributes:
        size: Number of nodes or resources managed by the executor.
        config: Backend-specific configuration dictionary.
    """

    def __init__(self, size: int, config: RunConfig) -> None:
        """
        Initialize the executor.

        Args:
            size: Number of available nodes or resources.
            config: Backend-specific configuration dictionary.
        """
        self._size = size
        self._config = config

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        """
        Return the number of nodes or resources managed by this executor.

        Returns:
            The executor size.
        """
        return self._size

    @property
    def config(self) -> Dict[str, Any]:
        """
        Return the backend-specific configuration.

        Returns:
            The configuration dictionary.
        """
        return self._config

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def create_circuit(
        self,
        num_qubits: int,
        num_clbits: int,
        comm: QMPICommunicator,
    ) -> Circuit:
        """
        Create a backend-specific quantum circuit.

        Args:
            num_qubits: Number of qubits in the circuit.
            num_clbits: Number of classical bits in the circuit.
            comm: Communicator associated with the circuit.

        Returns:
            A backend-specific :class:`~netqmpi.sdk.circuit.Circuit`.
        """

    @abstractmethod
    def build_apps(self, file: str, size: int) -> Any:
        """
        Load a script and build the rank-specific application instances.

        Each rank receives an injected
        :class:`~netqmpi.sdk.environment.Environment` exposing this
        executor's :meth:`create_circuit` factory.

        Args:
            file: Path to the NetQMPI Python script. It must contain a
                ``main(env=None)`` function.
            size: Number of parallel quantum nodes to simulate.

        Returns:
            A backend-specific application instance ready to be passed to
            :meth:`run`.
        """

    @abstractmethod
    def run(self, apps: Any) -> None:
        """
        Execute an application instance with the given configuration.

        Args:
            app_instance: Object returned by :meth:`build_apps`.
            config: Simulation or execution parameters. Backend adapters
                may accept a subclass of :class:`RunConfig` with
                additional backend-specific fields.
        """