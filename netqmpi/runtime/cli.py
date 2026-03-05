import argparse
import os
import time
from typing import Optional, Callable

from netqmpi.runtime.executor import Executor
from netqmpi.runtime.adapters.cunqa_executor import CunqaExecutorAdapter
from netqmpi.runtime.adapters.netqasm_executor import NetQASMExecutorAdapter

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
    executor: Executor, 
    script_args,
    configuration: SimulationConfig = SimulationConfig(),
    timer=None
):
    if script_args is None:
        script_args = []

    app_dir = "."
    app = executor.build_app(script, num_procs)
    configuration.network_config = executor.load_network_cfg(app_dir, configuration.network_config)

    if timer:
        start = time.perf_counter()

    results = executor.run(app=app, configuration=configuration)

    for result in results:
        print(result.counts)

    if configuration.enable_logging:
        executor.postprocess_logs(configuration)

    if timer:
        print(f"finished simulation in {round(time.perf_counter() - start, 2)} seconds")

def main():
    
    print("Hola!!!!!!")
    import sys
    print("ARGV:", sys.argv)
    
    parser = argparse.ArgumentParser(description="Run a NetQMPI Python code.")
    
    parser.add_argument(
        "-np", "--num-procs", 
        type=int, 
        required=True,
        help="Number of parallel processes"
    )
    
    parser.add_argument("script", type=str, help="Path to the NetQMPI Python script to be executed")

    backend_group = parser.add_mutually_exclusive_group()
    backend_group.add_argument("--netqasm", action="store_true", help="Use NetQASM backend")
    backend_group.add_argument("--cunqa", action="store_true", help="Use CunQA backend")

    #parser.add_argument("script_args", nargs=argparse.REMAINDER, help="Script configuration options.")

    args = parser.parse_args()

    if not os.path.isfile(args.script):
        parser.error(f"Error: File {args.script} does not exist.", file=sys.stderr)
    if args.num_procs < 1:
        parser.error("Number of processes must be at least 1")
    
    print(args)
    
    if args.netqasm:
        executor = NetQASMExecutorAdapter(args.num_procs)
    elif args.cunqa:
        executor = CunqaExecutorAdapter(args.num_procs)
    else:
        # si quieres un default cuando no se pasa ninguna
        print("No backend flag; using default (NetQASM)")
        executor = NetQASMExecutorAdapter(args.num_procs)    
    
    print("Hola!!!!!!")
    
    simulate(
        script=args.script,
        num_procs=args.num_procs,
        executor=executor,
        script_args=None,
    )

if __name__ == "__main__":
    main()