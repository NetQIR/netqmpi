import argparse
import importlib
import subprocess
import sys
import os
import time
from typing import Optional, Callable

from netqmpi.runtime.executor import Executor

from dataclasses import dataclass
from typing import Optional, Callable, Any

@dataclass
class SimulationConfig:
    network_config: Optional[Any] = None
    post_function: Optional[Callable] = None
    num_rounds: int = 1
    enable_logging: bool = True
    log_cfg: Optional[Any] = None
    hardware: str = "generic"
    formalism: Optional[Any] = None      # el backend decide si lo usa
    simulator_name: Optional[str] = None # o backend-specific

def simulate(
    script: str, 
    num_procs: int, 
    backend: Executor, 
    script_args,
    configuration: SimulationConfig = SimulationConfig(),
    timer=None
):
    if script_args is None:
        script_args = []

    app_dir = "."
    app_instance = backend.build_app(script, num_procs, script_args)
    configuration.network_config = backend.load_network_cfg(app_dir, configuration.network_config)

    if timer:
        start = time.perf_counter()

    backend.run(app_instance=app_instance, configuration=configuration)

    if configuration.enable_logging:
        backend.postprocess_logs(configuration)

    if timer:
        print(f"finished simulation in {round(time.perf_counter() - start, 2)} seconds")

def main():
    parser = argparse.ArgumentParser(description="Run a NetQMPI Python code.")
    
    parser.add_argument(
        "-np", "--num-procs", 
        type=int, 
        required=True,
        help="Number of parallel processes"
    )
    
    parser.add_argument(
        "script", 
        type=str,
        help="Path to the NetQMPI Python script to be executed"
    )
    
    parser.add_argument(
        "script_args", 
        nargs=argparse.REMAINDER,
        help="Additional arguments to pass to the script"
    )

    backend_group = parser.add_mutually_exclusive_group()

    backend_group.add_argument(
        "--netqasm",
        action="store_true",
        help="Use NetQASM backend"
    )

    backend_group.add_argument(
        "--cunqa",
        action="store_true",
        help="Use CunQA backend"
    )

    args = parser.parse_args()

    if not os.path.isfile(args.script):
        parser.error(f"Error: File {args.script} does not exist.", file=sys.stderr)
    if args.num_procs < 1:
        parser.error("Number of processes must be at least 1")

    simulate(
        script=args.script,
        num_procs=args.num_procs,
        script_args=args.script_args,
    )

if __name__ == "__main__":
    main()