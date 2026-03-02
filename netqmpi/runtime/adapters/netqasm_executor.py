"""
Executor adapter for the NetQASM backend.

Implements the Executor interface to work with NetQASM.
"""
from typing import Dict, Any, Optional

from netqmpi.runtime.executor import Executor
from netqmpi.sdk.adapters.netqasm_circuit import NetQASMCircuitAdapter
from netqmpi.sdk.communicator import QMPICommunicator
from netqmpi.helpers import load_main


from netqasm.runtime import env
from netqasm.runtime.application import ApplicationInstance, Program, Application
from netqasm.sdk.external import simulate_application
from netqasm.sdk.external import NetQASMConnection

class NetQASMExecutorAdapter(Executor):
    """
    Adapter that implements Executor for the NetQASM backend.
    
    This adapter allows using NetQASM as an execution backend
    following the common interface defined in the Executor class.
    """
    
    def __init__(self, size: int, config: Dict[str, Any] = None):
        """
        Initialize the NetQASM executor.
        
        Args:
            size: Number of available NetQASM nodes.
            config: NetQASM-specific configuration.
        """
        super().__init__(size, config)
        
        self.connection = NetQASMConnection(
            app_name=config.app_name, log_config=config.log, epr_sockets=self.epr_sockets_list
        )
        # TODO: Initialize NetQASM-specific resources
    
    def connect():
        pass
    
    def create_circuit(self, num_qubits: int, num_clbits: int) -> 'NetQASMCircuitAdapter':
        """
        Factory Method: Creates a NetQASM circuit.
        
        Args:
            num_qubits: Number of qubits in the circuit.
            num_clbits: Number of classical bits in the circuit.
            
        Returns:
            NetQASMCircuitAdapter instance.
        """
        return NetQASMCircuitAdapter(num_qubits, num_clbits)
    
    def build_app(self, script: str, num_processes: int) -> Any:
        if script is None:
            raise ValueError("script must be provided")
        if not script.endswith(".py"):
            raise ValueError("script must be a .py script")

        main_func = load_main(script)
        if main_func is None:
                raise ValueError(f"main function not found in {script}")

        programs = []
        for rank in range(num_processes):
            def wrapped_main(app_config=None):
                comm = QMPICommunicator(rank, num_processes, app_config)
                return main_func(comm=comm)
            
            prog = Program(party=rank, entry=wrapped_main, args=[], results=[])
            programs += [prog]

        roles = env.load_roles_config("roles.yaml") # TODO: This is hardcoded
        roles = (
            {prog.party: prog.party for prog in programs}
            if roles is None
            else roles
        )

        app = Application(programs = programs, metadata = None)
        app_instance = ApplicationInstance(
            app = app,
            program_inputs={}, # TODO: This is hardcoded for now
            network=None,
            party_alloc=roles,
            logging_cfg=None
        )

        return app_instance

    def load_network_cfg(self, app_dir: str, user_network_cfg: Optional[Any]) -> Any:
        # TODO: Hago la configuración
        return {}

    def run(self, *, app_instance: Any, configuration: Any) -> None:
        simulate_application(
            app_instance=app_instance,
            num_rounds=configuration.num_rounds,
            network_cfg=configuration.network_config,
            formalism=configuration.formalism,
            post_function=configuration.post_function,
            log_cfg=configuration.log_cfg,
            use_app_config=configuration.use_app_config,
            enable_logging=configuration.enable_logging,
            hardware=configuration.hardware,
        )

    def postprocess_logs(self, configuration: Any) -> None:
        pass
    