import argparse
import importlib
import subprocess
import sys
import os
import time
from typing import Optional, Callable

import netqasm.logging.glob
from netqasm.logging.glob import set_log_level
from netqasm.runtime.application import network_cfg_from_path
from netqasm.runtime.interface.config import QuantumHardware, NetworkConfig
from netqasm.runtime.process_logs import create_app_instr_logs, make_last_log
from netqasm.runtime.settings import Simulator, Formalism, set_simulator
from netqasm.sdk.config import LogConfig

from netqmpi.sdk.external import app_instance_from_file


class NetQASMConfig:
    def __init__(self, network_config=None, post_function=None):
        self.network_config: Optional[NetworkConfig] = network_config
        self.post_function: Optional[Callable] = post_function
        self.formalism: Formalism = Formalism.KET
        self.num_rounds : int = 1
        self.log_cfg : Optional[LogConfig] = None
        self.use_app_config : bool = True
        self.enable_logging : bool = True
        self.hardware : str = "generic"


def simulate(script: str, num_procs: int = 1, script_args=None, configuration: NetQASMConfig = NetQASMConfig(),
             timer=None):
    """
    Simulate the execution of a NetQMPI Python script using NetQASM
    """

    if script_args is None:
        script_args = []
    simulator = os.environ.get("NETQASM_SIMULATOR", Simulator.NETSQUID.value)
    set_simulator(simulator)

    simulate_application = importlib.import_module("netqasm.sdk.external").simulate_application
    app_dir = "."

    app_instance = app_instance_from_file(script,
                                          num_processes=num_procs)  # TODO pendiente ver que hacer con los script_args
    configuration.network_cfg = network_cfg_from_path(app_dir, configuration.network_config)

    if timer:
        start = time.perf_counter()

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

    if configuration.enable_logging:
        if configuration.log_cfg is not None:
            create_app_instr_logs(configuration.log_cfg.log_subroutines_dir)
            make_last_log(configuration.log_cfg.log_subroutines_dir)

    if timer:
        print(f"finished simulation in {round(time.perf_counter() - start, 2)} seconds")


def main():
    parser = argparse.ArgumentParser(
        description="Run a NetQMPI Python code."
    )
    parser.add_argument("-np", "--num-procs", type=int, required=True,
                        help="Number of parallel processes")
    parser.add_argument("script", type=str,
                        help="Path to the NetQMPI Python script to be executed")
    parser.add_argument("script_args", nargs=argparse.REMAINDER,
                        help="Additional arguments to pass to the script")

    args = parser.parse_args()

    if not os.path.isfile(args.script):
        print(f"Error: File {args.script} does not exist.", file=sys.stderr)
        sys.exit(1)

    if args.num_procs < 1:
        print("Number of processes must be at least 1.", file=sys.stderr)
        sys.exit(1)

    simulate(
        script=args.script,
        num_procs=args.num_procs,
        script_args=args.script_args,
    )

if __name__ == "__main__":
    main()