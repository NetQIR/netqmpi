"""
Executor adapter for the Cunqa backend.

Implements the Executor interface to work with Cunqa.
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
    Adapter that implements Executor for the Cunqa backend.
    
    This adapter allows using Cunqa as an execution backend
    following the common interface defined in the Executor class.
    """
    
    def __init__(self, size: int, config: Dict[str, Any] = None):
        """
        Initialize the Cunqa executor.
        
        Args:
            size: Number of available Cunqa nodes.
            config: Cunqa-specific configuration.
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
        Factory Method: Creates a Cunqa circuit.
        
        Args:
            num_qubits: Number of qubits in the circuit.
            num_clbits: Number of classical bits in the circuit.
            
        Returns:
            CunqaCircuitAdapter instance.
        """
        circuit = CunqaCircuitAdapter(num_qubits, num_clbits, comm)
        self._circuits.append(circuit)
        return circuit

    def build_apps(self, script: str, size: int) -> Any:
        main_func = load_main(script)

        apps = []
        for rank in range(size):
            env = Environment(CunqaCommunicator(rank, size), self)
            wrapped_main = lambda env=env: main_func(env=env)
            apps.append(wrapped_main)
        
        return apps
        
    def run(self, apps: Any, configuration: Any) -> None:
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