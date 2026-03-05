"""
Executor adapter for the Cunqa backend.

Implements the Executor interface to work with Cunqa.
"""
import os, sys
sys.path.append(os.getenv("HOME"))

from typing import Dict, Any, List, Optional

from cunqa.qpu import qraise, get_QPUs, run, qdrop
from cunqa.qjob import gather

from netqmpi.runtime.executor import Executor
from netqmpi.sdk.core.environment import Environment
from netqmpi.sdk.adapters.cunqa_circuit import CunqaCircuitAdapter
from netqmpi.sdk.communicator import QMPICommunicator
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
        
        self.env = Environment(self.create_circuit)
    
    def operations_supported(self) -> List[str]:
        """
        Returns the operations supported by Cunqa.
        
        Returns:
            List of quantum operations available in Cunqa.
        """
        # TODO: Implement actual Cunqa operations list
        return [
            'H', 'X', 'Y', 'Z', 'S', 'T', 'Sdg', 'Tdg',
            'CNOT', 'CZ', 'SWAP',
            'RX', 'RY', 'RZ', 'U3',
            'measure', 'reset', 'barrier'
        ]
    
    def create_circuit(self, num_qubits: int, num_clbits: int, rank: int) -> CunqaCircuitAdapter:
        """
        Factory Method: Creates a Cunqa circuit.
        
        Args:
            num_qubits: Number of qubits in the circuit.
            num_clbits: Number of classical bits in the circuit.
            
        Returns:
            CunqaCircuitAdapter instance.
        """
        return CunqaCircuitAdapter(num_qubits, num_clbits, rank)

    def build_app(self, script: str, num_processes: int) -> Any:
        main_func = load_main(script)

        def wrapped_main(rank, app_config=None):
            return main_func(env=self.env, comm=QMPICommunicator(rank, num_processes, self))
        
        return wrapped_main
        
    def load_network_cfg(self, app_dir: str, user_network_cfg: Optional[Any]) -> Any:
        pass

    def run(self, app: Any, configuration: Any) -> None:
        for i in range(self.size):
            app(i)

        try:
            family = qraise(2, "00:10:00", simulator="Aer", co_located=True, quantum_comm=True)
        except Exception as error:
            raise error

        try:
            qpus  = get_QPUs(co_located = True, family = family)
            
            for circuit in self.env.circuits:
                circuit.translate(circuit.ops)
                print(f"ID: {circuit._cunqa_circuit.id}\n\t{circuit._cunqa_circuit.instructions}")
            qjobs = run([circuit._cunqa_circuit for circuit in self.env.circuits], qpus, shots = 1024) # non-blocking call
            results = gather(qjobs)
            #results = "hola"
            qdrop(family)
            return results
        except Exception as error:
            qdrop(family)
            raise error

    def postprocess_logs(self, configuration: Any) -> None:
        pass