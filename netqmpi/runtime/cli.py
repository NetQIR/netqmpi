"""
NetQMPI command-line entry point.

This module is intentionally free of any backend-specific imports
(``netqasm``, ``cunqa``, ...).  All simulator logic is delegated to an
:class:`~netqmpi.runtime.executor.Executor` implementation that lives
in the corresponding adapter package.
"""
import argparse
import os
import sys
import time
from typing import Optional

from netqmpi.runtime.executor import Executor
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
                   :class:`~netqmpi.sdk.adapters.netqasm.NetQASMExecutorAdapter`
                   when ``None``.
        config:    Simulation parameters.  Defaults to :class:`RunConfig`
                   (all default values) when ``None``.
        timer:     If ``True``, prints the wall-clock time of the run.
    """
    if executor is None:
        from netqmpi.sdk.adapters.netqasm.netqasm_executor import NetQASMExecutorAdapter
        executor = NetQASMExecutorAdapter(size=num_procs)

    if config is None:
        config = RunConfig()

    if timer:
        start = time.perf_counter()

    app_instance = executor.build_app(script, num_processes=num_procs)
    executor.run(app_instance, config)

    if timer:
        print(f"finished simulation in {round(time.perf_counter() - start, 2)} seconds")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a NetQMPI Python script."
    )
    parser.add_argument(
        "-np", "--num-procs",
        type=int,
        required=True,
        help="Number of parallel processes",
    )
    parser.add_argument(
        "script",
        type=str,
        help="Path to the NetQMPI Python script to be executed",
    )
    parser.add_argument(
        "script_args",
        nargs=argparse.REMAINDER,
        help="Additional arguments to pass to the script",
    )

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
    )


if __name__ == "__main__":
    main()