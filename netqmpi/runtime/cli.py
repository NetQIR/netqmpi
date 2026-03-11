"""
NetQMPI command-line entry point.

This module is intentionally free of any backend-specific imports
(``netqasm``, ``cunqa``, ...).  All simulator logic is delegated to an
:class:`~netqmpi.runtime.executor.Executor` implementation that lives
in the corresponding adapter package.
"""

import time, argparse
from typing import Optional

from netqmpi.runtime import Executor
from netqmpi.runtime.adapters.cunqa import CunqaExecutorAdapter
from netqmpi.runtime.adapters.netqasm import NetQASMExecutorAdapter
from netqmpi.runtime.run_config import RunConfig

def simulate(
    script: str,
    num_procs: int = 1,
    executor: Optional[Executor] = None,
    config: Optional[RunConfig] = None,
    timer: bool = False,
) -> None:
    """
    Build and run a NetQMPI script using the given backend *executor*.

    Args:
        script:    Path to the NetQMPI Python script.
        num_procs: Number of parallel quantum nodes.
        executor:  Backend executor.  Defaults to
                   :class:`~netqmpi.runtime.adapters.netqasm.NetQASMExecutorAdapter`
                   when ``None``.
        config:    Simulation parameters.  Defaults to :class:`RunConfig`
                   (all default values) when ``None``.
        timer:     If ``True``, prints the wall-clock time of the run.
    """
    if executor is None:
        executor = NetQASMExecutorAdapter(size=num_procs)

    if config is None:
        config = RunConfig()

    if timer:
        start = time.perf_counter()

    apps_instance = executor.build_apps(script, size=num_procs)
    executor.run(apps_instance, config)

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
    
    parser.add_argument("script", type=str, help="Path to the NetQMPI Python script to be executed")

    backend_group = parser.add_mutually_exclusive_group()
    backend_group.add_argument("--netqasm", action="store_true", help="Use NetQASM backend")
    backend_group.add_argument("--cunqa", action="store_true", help="Use CunQA backend")

    # TODO: Turn ON and OFF the timer
    # TODO: Get specific configurations

    args = parser.parse_args()

    if args.num_procs < 1:
        parser.error("Number of processes must be at least 1")
    
    if args.netqasm:
        executor = NetQASMExecutorAdapter(args.num_procs)
    elif args.cunqa:
        executor = CunqaExecutorAdapter(args.num_procs)
    else:
        print("No backend flag; using default (NetQASM)")
        executor = NetQASMExecutorAdapter(args.num_procs)    
    
    simulate(
        script=args.script,
        num_procs=args.num_procs,
        executor=executor
    )

if __name__ == "__main__":
    main()