"""
Base abstraction for quantum circuit executors.

Defines the full contract that all quantum backend adapters must follow:

- Circuit factory (:meth:`create_circuit`) — Factory Method pattern.
- Application builder (:meth:`build_app`) — wires up N rank processes from a
  user script.
- Application runner (:meth:`run`) — executes the built application on the
  backend simulator or hardware.

Belongs to the RUNTIME layer. No imports from ``netqasm`` or ``cunqa`` are
allowed here; those live exclusively in the adapter packages.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from netqmpi.runtime.run_config import RunConfig
from netqmpi.sdk.core.circuit import Circuit

if TYPE_CHECKING:
    from netqmpi.sdk.core.environment import Environment


class Executor(ABC):
    """
    Abstract class representing a quantum backend executor.

    Combines three responsibilities into a single adapter contract:

    1. **Circuit factory** — :meth:`create_circuit` produces a
       backend-specific :class:`~netqmpi.sdk.core.circuit.Circuit`.
    2. **Application builder** — :meth:`build_app` loads a user script and
       wires up *N* rank processes, each receiving an injected
       :class:`~netqmpi.sdk.core.environment.Environment`.
    3. **Application runner** — :meth:`run` drives the built application
       through the backend simulator or hardware.

    Attributes:
        size (int):             Number of nodes/resources in this executor.
        config (Dict[str, Any]): Backend-specific configuration.
    """

    def __init__(self, size: int, config: Optional[Dict[str, Any]] = None) -> None:
        """
        Args:
            size:   Number of available nodes or resources.
            config: Dictionary with backend-specific configuration.
        """
        self._size = size
        self._config = config or {}

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        """Returns the number of nodes/resources of this executor."""
        return self._size

    @property
    def config(self) -> Dict[str, Any]:
        """Returns the backend-specific configuration."""
        return self._config

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def operations_supported(self) -> List[str]:
        """
        Returns the list of quantum operations supported by this backend.

        Returns:
            List of gate/operation names, e.g. ``['H', 'CNOT', 'measure']``.
        """

    @abstractmethod
    def create_circuit(
        self,
        num_qubits: int,
        num_clbits: int,
        environment: Optional[Environment] = None,
    ) -> Circuit:
        """
        Factory Method: creates a backend-specific quantum circuit.

        Args:
            num_qubits:  Number of qubits in the circuit.
            num_clbits:  Number of classical bits in the circuit.
            environment: The :class:`~netqmpi.sdk.core.environment.Environment`
                         to bind to the circuit.

        Returns:
            A backend-specific :class:`~netqmpi.sdk.core.circuit.Circuit`.
        """

    @abstractmethod
    def build_app(
        self,
        file: str,
        num_processes: int,
        argv_file: Optional[str] = None,
        roles_cfg_file: str = "roles.yaml",
    ) -> Any:
        """
        Load *file* and wire up *num_processes* rank instances.

        Each rank process receives an injected
        :class:`~netqmpi.sdk.core.environment.Environment` that exposes
        this executor's :meth:`create_circuit` factory.

        Args:
            file:           Path to the NetQMPI ``.py`` script.  Must contain
                            a ``main(env=None)`` function.
            num_processes:  Number of parallel quantum nodes to simulate.
            argv_file:      Optional YAML file with per-rank argument values.
            roles_cfg_file: Path to the roles configuration YAML file.

        Returns:
            A backend-specific application instance ready to be passed to
            :meth:`run`.
        """

    @abstractmethod
    def run(self, app_instance: Any, config: RunConfig) -> None:
        """
        Execute *app_instance* with the given *config*.

        Args:
            app_instance: The object returned by :meth:`build_app`.
            config:       Simulation/execution parameters.  Backend adapters
                          may accept a subclass of :class:`RunConfig` that
                          carries additional backend-specific fields.
        """
