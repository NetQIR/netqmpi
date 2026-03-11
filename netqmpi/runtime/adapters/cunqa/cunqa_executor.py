"""
Executor adapter for the CUNQA backend.

This module provides an implementation of the ``Executor`` interface for
running applications with the CUNQA backend.
"""
import os, sys
sys.path.append(os.getenv("HOME"))

from typing import Dict, Any, List, Optional, TYPE_CHECKING

from cunqa.qpu import qraise, get_QPUs, run, qdrop
from cunqa.qjob import gather

from netqmpi.runtime.executor import Executor
from netqmpi.sdk.environment import Environment
from netqmpi.runtime.adapters.cunqa.cunqa_circuit import CunqaCircuitAdapter
from netqmpi.runtime.adapters.cunqa.cunqa_communicator import CunqaCommunicator
from netqmpi.helpers import load_main  

class CunqaExecutorAdapter(Executor):
    """
    Executor adapter for the CUNQA backend.

    This adapter enables execution through CUNQA while conforming to the
    common interface defined by :class:`Executor`.
    """
    
    def __init__(self, size: int, config: Dict[str, Any] = None):
        """
        Initialize the CUNQA executor adapter.

        Args:
            size: Number of available CUNQA nodes.
            config: Backend-specific configuration parameters.
        """
        super().__init__(size, config)
        self._circuits = []
    
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
        circuit = CunqaCircuitAdapter(num_qubits, num_clbits, comm)
        self._circuits.append(circuit)
        return circuit

    def build_apps(self, script: str, size: int) -> Any:
        """
        Build one application wrapper per rank from the provided script.

        Args:
            script: Path to the script containing the main entry point.
            size: Number of ranks to instantiate.

        Returns:
            A collection of wrapped application callables, one per rank.
        """
        main_func = load_main(script)

        apps = []
        for rank in range(size):
            env = Environment(CunqaCommunicator(rank, size), self)
            wrapped_main = lambda env=env: main_func(env=env)
            apps.append(wrapped_main)
        
        return apps
        
    def run(self, apps: Any, configuration: Any) -> None:
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

        try:
            family = qraise(2, "00:10:00", simulator="Aer", co_located=True, quantum_comm=True)
        except Exception as error:
            raise error

        try:
            qpus  = get_QPUs(co_located = True, family = family)
            
            for circuit in self._circuits:
                circuit.translate(circuit.ops)
                print(circuit._cunqa_circuit.instructions)
            qjobs = run([circuit._cunqa_circuit for circuit in self._circuits], qpus, shots = 1024) # non-blocking call
            results = gather(qjobs)
            for result in results:
                print(result)
            qdrop(family)
            return results
        except Exception as error:
            qdrop(family)
            raise error