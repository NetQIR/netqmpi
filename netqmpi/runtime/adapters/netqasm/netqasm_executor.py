"""
NetQASM backend adapter.

Implements the full :class:`~netqmpi.runtime.executor.Executor` contract for
the NetQASM simulator:

- :meth:`create_circuit`  — Factory Method for NetQASM circuits.
- :meth:`build_app`       — loads a user script and wires up N rank processes.
- :meth:`run`             — drives the NetQASM simulator.

This is the only file in the NetQASM adapter layer that is allowed to import
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
from netqmpi.sdk.adapters.netqasm.netqasm_circuit import NetQASMCircuitAdapter
from netqmpi.sdk.communicator.communicator import QMPICommunicator
from netqmpi.sdk.core.circuit import Circuit
from netqmpi.sdk.core.environment import Environment
from netqmpi.runtime.external import import_module_from_path


# ---------------------------------------------------------------------------
# Extended run configuration for the NetQASM backend
# ---------------------------------------------------------------------------

@dataclass
class NetQASMRunConfig(RunConfig):
    """
    :class:`~netqmpi.runtime.run_config.RunConfig` extended with fields
    specific to the NetQASM simulator.

    Attributes:
        formalism:      Quantum state formalism (KET, DENSITY, STABILIZER, …).
        network_config: NetQASM
                        :class:`~netqasm.runtime.interface.config.NetworkConfig`
                        describing the simulated network topology.  ``None``
                        uses the default topology for the script directory.
        log_cfg:        NetQASM :class:`~netqasm.sdk.config.LogConfig`
                        controlling per-rank instruction logging.
    """

    formalism: Formalism = field(default_factory=lambda: Formalism.KET)
    network_config: Optional[NetworkConfig] = None
    log_cfg: Optional[LogConfig] = None


# ---------------------------------------------------------------------------
# Private helper
# ---------------------------------------------------------------------------

def _make_environment_injector(main_func, rank: int, size: int, executor: Executor):
    """Wrap *main_func* to inject a :class:`~netqmpi.sdk.core.environment.Environment`.

    The returned wrapper is called by the NetQASM runtime with ``app_config``
    and builds both the :class:`~netqmpi.sdk.communicator.QMPICommunicator`
    and the :class:`~netqmpi.sdk.core.environment.Environment` before
    forwarding to the original function.
    """
    def wrapper(app_config=None):
        comm = QMPICommunicator(rank, size, app_config)
        environment = Environment(comm, executor)
        return main_func(env=environment)
    return wrapper


# ---------------------------------------------------------------------------
# Concrete Executor
# ---------------------------------------------------------------------------

class NetQASMExecutorAdapter(Executor):
    """
    :class:`~netqmpi.runtime.executor.Executor` implementation for the
    NetQASM backend.

    Handles circuit creation, script loading, process wiring, and simulator
    execution in a single cohesive adapter.
    """

    def __init__(self, size: int, config: Dict[str, Any] = None) -> None:
        """
        Args:
            size:   Number of available NetQASM nodes.
            config: NetQASM-specific configuration dictionary.
        """
        super().__init__(size, config)

    # ------------------------------------------------------------------
    # Executor interface — circuit factory
    # ------------------------------------------------------------------

    def operations_supported(self) -> List[str]:
        """Returns the quantum operations available in NetQASM."""
        return [
            'H', 'X', 'Y', 'Z', 'S', 'T', 'K',
            'CNOT', 'CZ', 'CPHASE',
            'RX', 'RY', 'RZ',
            'measure', 'reset',
        ]

    def create_circuit(
        self,
        num_qubits: int,
        num_clbits: int,
        environment: Optional[Environment] = None,
    ) -> Circuit:
        """
        Factory Method: creates a :class:`~netqmpi.sdk.adapters.netqasm.NetQASMCircuitAdapter`.

        Args:
            num_qubits:  Number of qubits in the circuit.
            num_clbits:  Number of classical bits in the circuit.
            environment: The :class:`~netqmpi.sdk.core.environment.Environment`
                         to bind to the circuit.

        Returns:
            :class:`~netqmpi.sdk.adapters.netqasm.NetQASMCircuitAdapter` instance.
        """
        return NetQASMCircuitAdapter(num_qubits, num_clbits, environment=environment)

    # ------------------------------------------------------------------
    # Executor interface — application builder
    # ------------------------------------------------------------------

    def build_app(
        self,
        file: str,
        num_processes: int,
        argv_file: Optional[str] = None,
        roles_cfg_file: str = "roles.yaml",
    ) -> ApplicationInstance:
        """
        Load *file* and create a
        :class:`~netqasm.runtime.application.ApplicationInstance` with
        *num_processes* rank entries, each wrapping ``main`` with an injected
        :class:`~netqmpi.sdk.core.environment.Environment` backed by *self*.

        Args:
            file:           Path to the NetQMPI ``.py`` script.
            num_processes:  Number of parallel quantum nodes.
            argv_file:      Optional YAML file with per-rank argument values.
            roles_cfg_file: Path to the roles configuration file.

        Returns:
            :class:`~netqasm.runtime.application.ApplicationInstance` ready
            to pass to :meth:`run`.

        Raises:
            ValueError: If *file* is ``None``, not a ``.py`` file, or has no
                        ``main`` function.
        """
        if file is None:
            raise ValueError("file must be provided")
        if not file.endswith(".py"):
            raise ValueError("file must be a .py file")

        argv: dict = load_yaml(argv_file) if argv_file is not None else {}

        programs: list = []
        argv_per_rank: dict = {}

        for rank_int in range(num_processes):
            rank_name = f"rank_{rank_int}"
            prog_module = import_module_from_path(file)
            main_func = getattr(prog_module, "main", None)
            if main_func is None:
                raise ValueError(f"main function not found in {file}")

            wrapped_main = _make_environment_injector(main_func, rank_int, num_processes, self)
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
        Run *app_instance* through the NetQASM simulator.

        Args:
            app_instance: Object returned by :meth:`build_app`.
            config:       Simulation parameters.  Pass a
                          :class:`NetQASMRunConfig` to control NetQASM-specific
                          options such as *formalism* or *log_cfg*.
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
