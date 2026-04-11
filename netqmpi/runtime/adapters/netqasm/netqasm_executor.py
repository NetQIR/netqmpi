"""
NetQASM backend adapter.

This module implements the full
:class:`~netqmpi.runtime.executor.Executor` contract for the NetQASM
simulator, including circuit creation, application construction, and
simulation execution.

This is the only file in the NetQASM adapter layer allowed to import
from ``netqasm.*``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Callable

from netqasm.runtime.app_config import AppConfig
from netqasm.util.yaml import load_yaml
from netqasm.runtime.settings import Formalism

from netqmpi.runtime.executor import Executor
from netqmpi.runtime.run_config import RunConfig
from netqmpi.runtime.adapters.netqasm import NetQASMCommunicator, NetQASMCircuitAdapter
from netqmpi.sdk.circuit import Circuit
from netqmpi.sdk.environment import Environment
from netqmpi.helpers import load_main

# ---------------------------------------------------------------------------
# Extended run configuration for the NetQASM backend
# ---------------------------------------------------------------------------

@dataclass
class NetQASMRunConfig(RunConfig):
    """
    Extension of :class:`~netqmpi.runtime.run_config.RunConfig` with
    NetQASM-specific simulation parameters.

    Attributes:
        formalism: Quantum state formalism to use in the simulation.
        network_config: Network configuration describing the simulated
            topology. If ``None``, the default topology is used.
        log_cfg: NetQASM log configuration controlling per-rank
            instruction logging.
    """

    formalism: Formalism = field(default_factory=lambda: Formalism.KET)
    enable_logging: bool = True
    hardware: str = "generic"
    post_function: Optional[Callable] = None
    network_config: Optional[Any] = None
    log_cfg: Optional[Any] = None
    argv = None
    roles: str = "roles.yaml"

# ---------------------------------------------------------------------------
# Concrete Executor
# ---------------------------------------------------------------------------

class NetQASMExecutorAdapter(Executor):
    """
    Executor implementation for the NetQASM backend.

    This adapter handles circuit creation, application construction, and
    simulation execution for the NetQASM runtime.
    """

    def __init__(self, size: int, config: NetQASMRunConfig = None) -> None:
        """
        Initialize the NetQASM executor adapter.

        Args:
            size: Number of available NetQASM nodes.
            self._config: NetQASM-specific configuration dictionary.
        """
        
        super().__init__(size, config or NetQASMRunConfig())

    # ------------------------------------------------------------------
    # Executor interface — circuit factory
    # ------------------------------------------------------------------

    def create_circuit(
        self,
        num_qubits: int,
        num_clbits: int,
        comm: NetQASMCommunicator,
    ) -> Circuit:
        """
        Create a NetQASM circuit adapter.

        Args:
            num_qubits: Number of qubits in the circuit.
            num_clbits: Number of classical bits in the circuit.
            comm: Communicator bound to the circuit.

        Returns:
            A :class:`~netqmpi.runtime.adapters.netqasm.NetQASMCircuitAdapter`
            instance.
        """
        return NetQASMCircuitAdapter(num_qubits, num_clbits, comm=comm)

    # ------------------------------------------------------------------
    # Executor interface — application builder
    # ------------------------------------------------------------------

    def _make_environment_injector(self, main_func, rank: int, size: int):
        """
        Wrap ``main_func`` to inject an :class:`Environment`.

        The returned wrapper is invoked by the NetQASM runtime with an
        ``app_config`` object. It builds the corresponding
        :class:`NetQASMCommunicator` and :class:`Environment`, then calls
        the original function.

        Args:
            main_func: User entry-point function.
            rank: Rank assigned to the wrapped program.
            size: Total number of ranks.

        Returns:
            A wrapped callable compatible with the NetQASM runtime.
        """
        def wrapped_main():
            env = Environment(NetQASMCommunicator(rank, size, self._config), self)
            main_func(env=env)
            
        return wrapped_main

    def build_apps(self, file: str, size: int) -> Any:
        """
        Load a file and build a NetQASM application instance.

        The resulting application instance contains one program per rank,
        each wrapping the user ``main`` function with an injected
        :class:`~netqmpi.sdk.environment.Environment`.

        Args:
            file: Path to the NetQMPI Python file.
            size: Number of parallel quantum nodes.
            argv_file: Optional YAML file containing per-rank input
                arguments.
            roles_cfg_file: Path to the roles configuration file.

        Returns:
            A :class:`~netqasm.runtime.application.ApplicationInstance`
            ready to be passed to :meth:`run`.

        Raises:
            ValueError: If ``file`` is ``None`` or does not point to a
                Python file.
        """
        if file is None:
            raise ValueError("file must be provided")
        if not file.endswith(".py"):
            raise ValueError("file must be a .py file")

        argv: dict = load_yaml(self._config.argv) if self._config.argv is not None else {}
        main_func = load_main(file)

        apps = []
        for rank in range(size):
            wrapped_main = self._make_environment_injector(main_func, rank, size)
            apps.append(wrapped_main)        
    
        return apps
    
    # ------------------------------------------------------------------
    # Executor interface — application runner
    # ------------------------------------------------------------------

    def run(self, apps: Any) -> None:
        """
        Run an application instance through the NetQASM simulator.

        Args:
            app_instance: Application instance returned by
                :meth:`build_apps`.
        """
        for app in apps:
            app()