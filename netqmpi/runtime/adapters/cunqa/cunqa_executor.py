"""
Executor adapter for the CUNQA backend.

This module provides an implementation of the ``Executor`` interface for
running applications with the CUNQA backend.
"""
import os, sys
sys.path.append(os.getenv("HOME"))

from typing import Dict, Any
from dataclasses import dataclass

from cunqa.qpu import qraise, get_QPUs, qdrop

from netqmpi.runtime.executor import Executor
from netqmpi.sdk.environment import Environment
from netqmpi.runtime.adapters.cunqa.cunqa_circuit import CunqaCircuitAdapter
from netqmpi.runtime.adapters.cunqa.cunqa_communicator import CunqaCommunicator
from netqmpi.runtime.run_config import RunConfig
from netqmpi.helpers import load_main  


@dataclass
class CunqaRunConfig(RunConfig):
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

    simulator: str = "Aer"
    time: str = "00:10:00"
    

class CunqaExecutorAdapter(Executor):
    """
    Executor adapter for the CUNQA backend.

    This adapter enables execution through CUNQA while conforming to the
    common interface defined by :class:`Executor`.
    """
    
    def __init__(self, size: int, config: CunqaRunConfig = None):
        """
        Initialize the CUNQA executor adapter.

        Args:
            size: Number of available CUNQA nodes.
            config: Backend-specific configuration parameters.
        """
        super().__init__(size, config or CunqaRunConfig())
        self._family: str = None
    
    def create_circuit(
        self, 
        num_qubits: int, 
        num_clbits: int, 
        comm: CunqaCommunicator
    ) -> CunqaCircuitAdapter:

        """
        Create a CUNQA circuit adapter.

        Args:
            num_qubits: Number of qubits in the circuit.
            num_clbits: Number of classical bits in the circuit.
            comm: Communicator associated with the circuit.

        Returns:
            A ``CunqaCircuitAdapter`` instance.
        """
        return CunqaCircuitAdapter(num_qubits, num_clbits, comm)

    def build_apps(self, file: str, size: int) -> Any:
        """
        Build one application wrapper per rank from the provided file.

        Args:
            file: Path to the file containing the main entry point.
            size: Number of ranks to instantiate.

        Returns:
            A collection of wrapped application callables, one per rank.
        """
        main_func = load_main(file)

        try:
            self._family = qraise(size, "00:10:00", simulator="Aer", co_located=True, quantum_comm=True)
            qpus  = get_QPUs(co_located = True, family = self._family)
        except Exception as error:
            raise error

        apps = []
        for rank, qpu in enumerate(qpus):
            env = Environment(CunqaCommunicator(rank, size, qpu, self._config), self)
            wrapped_main = lambda env=env: main_func(env=env)
            apps.append(wrapped_main)
        
        return apps
        
    def run(self, apps: Any) -> None:
        """
        Execute the provided applications on the CUNQA backend.

        The applications are first invoked to build their circuits. Then the
        required QPUs are raised, the circuits are translated and submitted,
        and the execution results are gathered.

        Args:
            apps: Applications to execute.
            configuration: Execution configuration for the backend.

        Returns:
            The gathered execution results.

        Raises:
            Exception: Propagates any exception raised during backend setup,
                circuit execution, result gathering, or resource cleanup.
        """
        for app in apps:
            app()
            
        qdrop(self._family)