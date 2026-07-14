"""
NetQMPI command-line entry point.

This module is intentionally free of backend-specific imports from the
core runtime logic. All simulator-specific behavior is delegated to an
:class:`~netqmpi.runtime.executor.Executor` implementation provided by
the corresponding adapter package.
"""

import time, argparse
from typing import Optional, Type, TypeVar

from netqmpi.runtime import Executor
from netqmpi.runtime.run_config import RunConfig, read_config_block

_ConfigT = TypeVar("_ConfigT", bound=RunConfig)


def _build_config(config_cls: Type[_ConfigT], backend: str, args) -> _ConfigT:
    """
    Build a backend config from the ``--config`` YAML file and CLI overrides.

    Reads the generic settings plus the ``backend`` block from the config file
    (if given), then applies ``--shots`` as an explicit override so a quick
    command-line run can bump the shot count without editing the file.

    Args:
        config_cls: The backend's :class:`RunConfig` subclass.
        backend: Backend name, used to select its block in the config file.
        args: Parsed CLI arguments (uses ``args.config`` and ``args.shots``).

    Returns:
        A populated ``config_cls`` instance.
    """
    if args.config:
        config = config_cls.from_dict(read_config_block(args.config, backend))
    else:
        config = config_cls()
    if args.shots is not None:
        config.shots = args.shots
    return config

def simulate(
    script: str,
    num_procs: int = 1,
    executor: Optional[Executor] = None,
    config: Optional[RunConfig] = None,
    timer: bool = False,
) -> None:
    """
    Build and run a NetQMPI script using the given backend executor.

    Args:
        script: Path to the NetQMPI Python script.
        num_procs: Number of parallel quantum nodes.
        executor: Backend executor to use. If ``None``, a
            :class:`~netqmpi.runtime.adapters.netqasm.NetQASMExecutorAdapter`
            is used by default.
        config: Simulation parameters. If ``None``, a default
            :class:`RunConfig` instance is used.
        timer: If ``True``, print the wall-clock execution time.
    """
    if executor is None:
        executor = NetQASMExecutorAdapter(size=num_procs)

    if timer:
        start = time.perf_counter()

    apps_instance = executor.build_apps(script, size=num_procs)
    executor.run(apps_instance)

    if timer:
        print(f"finished simulation in {round(time.perf_counter() - start, 2)} seconds")

def main():
    """
    Parse command-line arguments and execute the requested NetQMPI script.

    The selected backend adapter is instantiated from the provided flags
    and passed to :func:`simulate`.
    """
    
    parser = argparse.ArgumentParser(description="Run a NetQMPI Python code.")
    
    parser.add_argument(
        "-n", "--num-procs", 
        type=int, 
        required=True,
        help="Number of parallel processes"
    )
    
    parser.add_argument("script", type=str, help="Path to the NetQMPI Python script to be executed")

    backend_group = parser.add_mutually_exclusive_group()
    backend_group.add_argument("--netqasm", action="store_true", help="Use NetQASM backend")
    backend_group.add_argument("--cunqa", action="store_true", help="Use CUNQA backend")
    backend_group.add_argument("--aer", action="store_true", help="Use Qiskit AerSimulator backend")
    backend_group.add_argument("--qoala", action="store_true", help="Use Qoala backend (simulation only)")

    parser.add_argument(
        "--shots",
        type=int,
        help="Number of shots (overrides the value in --config, if any)",
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to a YAML config file with generic settings and an optional "
             "per-backend block (e.g. a 'qoala:' block with link_fidelity and a "
             "'hardware' qdevice section). Replaces per-backend parameter flags.",
    )

    # TODO: Turn ON and OFF the timer

    args = parser.parse_args()

    if args.num_procs < 1:
        parser.error("Number of processes must be at least 1")

    try:
        if args.netqasm:
            from netqmpi.runtime.adapters.netqasm import NetQASMExecutorAdapter, NetQASMRunConfig

            config = _build_config(NetQASMRunConfig, "netqasm", args)
            executor = NetQASMExecutorAdapter(args.num_procs, config=config)
        elif args.cunqa:
            from netqmpi.runtime.adapters.cunqa import CunqaExecutorAdapter, CunqaRunConfig

            config = _build_config(CunqaRunConfig, "cunqa", args)
            executor = CunqaExecutorAdapter(args.num_procs, config=config)
        elif args.aer:
            from netqmpi.runtime.adapters.aer import AerExecutorAdapter, AerSimulatorConfig

            config = _build_config(AerSimulatorConfig, "aer", args)
            executor = AerExecutorAdapter(args.num_procs, config=config)
        elif args.qoala:
            from netqmpi.runtime.adapters.qoala import QoalaExecutorAdapter, QoalaRunConfig

            config = _build_config(QoalaRunConfig, "qoala", args)
            executor = QoalaExecutorAdapter(args.num_procs, config=config)
        else:
            from netqmpi.runtime.adapters.netqasm import NetQASMExecutorAdapter, NetQASMRunConfig

            print("No backend flag; using default (NetQASM)")
            config = _build_config(NetQASMRunConfig, "netqasm", args)
            executor = NetQASMExecutorAdapter(args.num_procs, config=config)
    except (ValueError, OSError) as error:
        parser.error(str(error))

    simulate(
        script=args.script,
        num_procs=args.num_procs,
        executor=executor
    )

if __name__ == "__main__":
    main()