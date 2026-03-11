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

import importlib
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from netqasm.runtime import env as netqasm_env
from netqasm.runtime.application import (
    Application, ApplicationInstance, Program, network_cfg_from_path,
)
from netqasm.runtime.interface.config import NetworkConfig
from netqasm.runtime.process_logs import create_app_instr_logs, make_last_log
from netqasm.runtime.settings import Formalism, Simulator, set_simulator
from netqasm.sdk.config import LogConfig
from netqasm.util.yaml import load_yaml

from netqmpi.runtime.executor import Executor
from netqmpi.runtime.run_config import RunConfig
from netqmpi.runtime.adapters.netqasm import NetQASMCommunicator, NetQASMCircuitAdapter

from netqmpi.sdk import QMPICommunicator
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
    network_config: Optional[NetworkConfig] = None
    log_cfg: Optional[LogConfig] = None

# ---------------------------------------------------------------------------
# Concrete Executor
# ---------------------------------------------------------------------------

class NetQASMExecutorAdapter(Executor):
    """
    Executor implementation for the NetQASM backend.

    This adapter handles circuit creation, application construction, and
    simulation execution for the NetQASM runtime.
    """

    def __init__(self, size: int, config: Dict[str, Any] = None) -> None:
        """
        Initialize the NetQASM executor adapter.

        Args:
            size: Number of available NetQASM nodes.
            config: NetQASM-specific configuration dictionary.
        """
        super().__init__(size, config)

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
        def wrapped_main(app_config=None):
            env = Environment(NetQASMCommunicator(rank, size, app_config), self)
            main_func(env=env)
            
        return wrapped_main

    def build_apps(
        self,
        file: str,
        size: int,
        argv_file: Optional[str] = None,
        roles_cfg_file: str = "roles.yaml",
    ) -> ApplicationInstance:
        """
        Load a script and build a NetQASM application instance.

        The resulting application instance contains one program per rank,
        each wrapping the user ``main`` function with an injected
        :class:`~netqmpi.sdk.environment.Environment`.

        Args:
            file: Path to the NetQMPI Python script.
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

        argv: dict = load_yaml(argv_file) if argv_file is not None else {}
        main_func = load_main(file)

        programs: list = []
        argv_per_rank: dict = {}

        for rank in range(size):
            rank_name = f"rank_{rank}"
            
            wrapped_main = self._make_environment_injector(main_func, rank, size)
            programs.append(Program(party=rank_name, entry=wrapped_main, args=[], results=[]))
            argv_per_rank[rank_name] = argv.copy()

        roles = netqasm_env.load_roles_config(roles_cfg_file)
        if roles is None:
            roles = {prog.party: prog.party for prog in programs}

        app = Application(programs=programs, metadata=None)
        return ApplicationInstance(
            app=app,
            program_inputs=argv_per_rank,
            network=None,
            party_alloc=roles,
            logging_cfg=None,
        )

    # ------------------------------------------------------------------
    # Executor interface — application runner
    # ------------------------------------------------------------------

    def run(self, app_instance: ApplicationInstance, config: RunConfig) -> None:
        """
        Run an application instance through the NetQASM simulator.

        Args:
            app_instance: Application instance returned by
                :meth:`build_apps`.
            config: Simulation parameters. A
                :class:`NetQASMRunConfig` can be provided to control
                NetQASM-specific options such as ``formalism`` or
                ``log_cfg``.
        """
        simulator = os.environ.get("NETQASM_SIMULATOR", Simulator.NETSQUID.value)
        set_simulator(simulator)

        simulate_application = importlib.import_module("netqasm.sdk.external").simulate_application

        formalism = getattr(config, "formalism", Formalism.KET)
        log_cfg = config.log_cfg
        network_config = config.network_config

        if network_config is not None:
            network_config = network_cfg_from_path(".", network_config)

        simulate_application(
            app_instance=app_instance,
            num_rounds=config.num_rounds,
            network_cfg=network_config,
            formalism=formalism,
            post_function=config.post_function,
            log_cfg=log_cfg,
            use_app_config=config.use_app_config,
            enable_logging=config.enable_logging,
            hardware=config.hardware,
        )

        if config.enable_logging and log_cfg is not None:
            create_app_instr_logs(log_cfg.log_subroutines_dir)
            make_last_log(log_cfg.log_subroutines_dir)