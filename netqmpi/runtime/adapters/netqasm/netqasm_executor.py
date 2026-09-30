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

from netqmpi.runtime.adapters.netqasm._compat import require_major

from netqmpi.runtime.adapters.netqasm.netqasm_communicator import (
    NetQASMCommunicator,
)
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
        netqasm_major: Major NetQASM release this run expects, 2 by default.
            The ``--netqasm1.0`` flag sets it to 1. It selects an
            environment rather than an adapter: the API this backend uses is
            the same in both releases, so the value is checked against what
            is installed and the run stops early if they disagree.
        shots: Number of times the program is simulated. Overrides the
            generic default of 1024, which is wrong by two orders of
            magnitude for this backend: SquidASM simulates the whole network
            once per shot, at roughly a second each on a two-rank program,
            so the generic default would take a quarter of an hour and look
            like a hang. Raise it with ``--shots`` when the statistics
            matter more than the wait.
        formalism: Quantum state formalism to use in the simulation.
        enable_logging: Whether SquidASM writes its instruction and
            communication logs. Off by default: the logs cost disk I/O inside
            the simulation, and SquidASM names their directory to the second,
            so runs sharing a working directory used to collide on it with
            ``FileExistsError``. When on, each run logs to a directory of its
            own under ``./log`` unless ``log_cfg.log_dir`` names one.
        hardware: SquidASM node hardware, ``"generic"`` or ``"nv"``.
        network_config: Path to a NetQASM network YAML describing the
            simulated topology. If ``None``, a fully connected network is
            built from the fields below.
        num_qubits: Qubits per node in the built network. SquidASM's own
            default is 5, which is what this keeps. A run whose ranks need
            more at once is refused before it starts.
        t1: Amplitude-damping time of every qubit, in ns. 0 disables it.
        t2: Dephasing time of every qubit, in ns. 0 disables it.
        gate_fidelity: Fidelity of every gate, in ``[0, 1]``.
        link_fidelity: Fidelity of the EPR pairs every link delivers, in
            ``[0, 1]``. 1 means a noiseless link.
        link_noise: SquidASM noise model applied when ``link_fidelity`` is
            below 1: ``"Depolarise"``, ``"DiscreteDepolarise"`` or
            ``"Bitflip"``.
        epr_setup_timeout: Wall-clock seconds a node waits for a peer to set
            up an EPR socket. SquidASM hard-codes 5, which runs with a dozen
            ranks or more exceed on a loaded machine.
        poll_interval: Seconds SquidASM's main thread sleeps between checks
            on the program threads, instead of spinning.
        log_cfg: NetQASM log configuration controlling per-rank
            instruction logging.

    With the defaults the simulated network is **ideal**: no decoherence
    (``t1 = t2 = 0``), perfect gates and perfect links, so every fidelity it
    reports is 1 unless the program itself is wrong.
    """

    netqasm_major: int = 2
    shots: int = 50
    formalism: Formalism = field(default_factory=lambda: Formalism.KET)
    enable_logging: bool = False
    hardware: str = "generic"
    post_function: Optional[Callable] = None
    network_config: Optional[Any] = None
    num_qubits: int = 5
    t1: float = 0.0
    t2: float = 0.0
    gate_fidelity: float = 1.0
    link_fidelity: float = 1.0
    link_noise: str = "Depolarise"
    epr_setup_timeout: float = 60.0
    poll_interval: float = 0.001
    log_cfg: Optional[Any] = None
    argv = None
    roles: str = "roles.yaml"

    def __post_init__(self) -> None:
        if self.num_qubits < 1:
            raise ValueError(f"num_qubits must be at least 1, got {self.num_qubits}.")
        if self.t1 < 0 or self.t2 < 0:
            raise ValueError(f"t1 and t2 must not be negative, got {self.t1}, {self.t2}.")
        for name in ("gate_fidelity", "link_fidelity"):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}.")
        if self.epr_setup_timeout <= 0:
            raise ValueError(
                f"epr_setup_timeout must be positive, got {self.epr_setup_timeout}.")

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
            config: NetQASM-specific configuration.

        Raises:
            RuntimeError: If the installed NetQASM is not the major release
                the configuration asks for.
        """
        config = config or NetQASMRunConfig()
        require_major(config.netqasm_major)
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
        # A previous run that failed part-way leaves its programs on the
        # class; starting from them would silently mix two runs together.
        NetQASMCommunicator.reset_run()

        for app in apps:
            app()