"""
NetQMPI command-line entry point.

This module is intentionally free of backend-specific imports from the
core runtime logic. All simulator-specific behavior is delegated to an
:class:`~netqmpi.runtime.executor.Executor` implementation provided by
the corresponding adapter package.
"""

import time, argparse
import dataclasses
import json
import os
import sys
from typing import Any, Dict, Mapping, Optional, Type, TypeVar

from netqmpi.runtime import Executor
from netqmpi.runtime.run_config import RunConfig, read_config_block

_ConfigT = TypeVar("_ConfigT", bound=RunConfig)


def _read_plugin_config(path: str) -> Dict[str, Any]:
    """
    Read the YAML mapping ``--plugin-config`` names.

    Args:
        path: Path to the YAML file.

    Returns:
        The mapping; an empty file is an empty mapping.

    Raises:
        ValueError: If the file holds something other than a mapping.
    """
    import yaml  # local import: only needed when the flag is used

    with open(path) as fh:
        data = yaml.safe_load(fh)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(
            f"--plugin-config {path} must contain a mapping of parameters, "
            f"got {type(data).__name__}.")
    return data


def _build_config(config_cls: Type[_ConfigT], backend: str, args,
                  **fixed: Any) -> _ConfigT:
    """
    Build a backend config from the ``--config`` YAML file and CLI overrides.

    Reads the generic settings plus the ``backend`` block from the config file
    (if given), then lays the command-line overrides on top — ``--shots``,
    ``--plugin`` and ``--plugin-config`` — so a quick run does not need the
    file edited. Everything goes through ``config_cls.from_dict`` in one go,
    so a config that applies a plugin's ``run_defaults`` there knows exactly
    which keys the user set, and its validation sees the final values.

    Args:
        config_cls: The backend's :class:`RunConfig` subclass.
        backend: Backend name, used to select its block in the config file
            and to name it in errors.
        args: Parsed CLI arguments (uses ``config``, ``shots``, ``plugin``
            and ``plugin_config``).
        **fixed: Fields the selected flag decides, applied last.

    Returns:
        A populated ``config_cls`` instance.

    Raises:
        ValueError: If ``--plugin`` or ``--plugin-config`` is given for a
            backend whose config has no ``plugin`` key, or the config is
            invalid.
    """
    data = read_config_block(args.config, backend) if args.config else {}
    overrides: Dict[str, Any] = {}
    if args.shots is not None:
        overrides["shots"] = args.shots

    for flag, value in (("--plugin", args.plugin),
                        ("--plugin-config", args.plugin_config)):
        if value is not None and "plugin" not in {
                f.name for f in dataclasses.fields(config_cls)}:
            # Ignoring it would run a plain simulation that the user takes
            # for one of the plugin's hardware.
            raise ValueError(
                f"{flag} is not supported by the {backend} backend, which "
                f"has no simulation plugins; the run would ignore it. "
                f"Plugins are a feature of the NetQASM backend (--netqasm, "
                f"--netqasm1.0).")
    if args.plugin is not None:
        overrides["plugin"] = args.plugin
    if args.plugin_config is not None:
        overrides["plugin_config"] = _read_plugin_config(args.plugin_config)

    return config_cls.from_dict({**data, **overrides, **fixed})

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
        # Imported here, like the backend branches of main(), so this module
        # stays free of backend imports.
        from netqmpi.runtime.adapters.netqasm import NetQASMExecutorAdapter

        executor = NetQASMExecutorAdapter(size=num_procs)

    if timer:
        start = time.perf_counter()

    apps_instance = executor.build_apps(script, size=num_procs)
    executor.run(apps_instance)

    if timer:
        print(f"finished simulation in {round(time.perf_counter() - start, 2)} seconds")

class _ListPluginsAction(argparse.Action):
    """``--list-plugins``: print the installed plugins, then exit."""

    def __init__(self, option_strings, dest=argparse.SUPPRESS,
                 default=argparse.SUPPRESS, help=None):
        super().__init__(option_strings, dest=dest, default=default, nargs=0,
                         help=help)

    def __call__(self, parser, namespace, values, option_string=None):
        from netqmpi import plugins

        found = plugins.installed()
        if not found:
            print(f"No NetQMPI plugins are installed (no distribution "
                  f"registers an entry point in the '{plugins.GROUP}' group).")
        else:
            name_width = max(len(plugin.name) for plugin in found)
            target_width = max(len(plugin.target) for plugin in found)
            for plugin in found:
                print(f"{plugin.name:<{name_width}}  "
                      f"{plugin.target:<{target_width}}  "
                      f"{plugin.distribution} {plugin.version}")
        parser.exit()


def _jsonable(value: Any) -> Any:
    """Turn a value ``json`` cannot write (a NumPy number, say) into one it can."""
    for method in ("tolist", "item"):
        convert = getattr(value, method, None)
        if callable(convert):
            return convert()
    return str(value)


def write_metrics(path: str, metrics: Mapping[str, Any]) -> None:
    """
    Write the metrics of a run to a JSON file, for ``--metrics``.

    Only the metrics are written, not the configuration of the run. A
    backend that reports none gets an empty object and a note on stderr,
    so that a sweep over backends always finds a file to read.

    Args:
        path: File to write; overwritten if it exists.
        metrics: What :meth:`~netqmpi.runtime.executor.Executor.metrics`
            returned.
    """
    with open(path, "w", encoding="utf-8") as stream:
        json.dump(metrics, stream, indent=2, default=_jsonable)
        stream.write("\n")
    if not metrics:
        print(f"netqmpi: this backend reported no metrics; wrote an empty "
              f"object to {path}", file=sys.stderr)


class _VersionAction(argparse.Action):
    """``--version``: print release, git tag and commit, then exit.

    Unlike argparse's own ``version`` action this asks git only when the flag
    is given, not every time the parser is built.
    """

    def __init__(self, option_strings, dest=argparse.SUPPRESS,
                 default=argparse.SUPPRESS, help=None):
        super().__init__(option_strings, dest=dest, default=default, nargs=0,
                         help=help)

    def __call__(self, parser, namespace, values, option_string=None):
        from netqmpi.version import describe

        print(describe())
        parser.exit()


def main():
    """
    Parse command-line arguments and execute the requested NetQMPI script.

    The selected backend adapter is instantiated from the provided flags
    and passed to :func:`simulate`.
    """
    
    parser = argparse.ArgumentParser(description="Run a NetQMPI Python code.")

    parser.add_argument(
        "--version", action=_VersionAction,
        help="Show the NetQMPI release, git tag and commit, and exit")

    parser.add_argument(
        "--list-plugins", action=_ListPluginsAction,
        help="List the installed simulation plugins (name, target, "
             "distribution and version), and exit")
    
    parser.add_argument(
        "-n", "--num-procs", 
        type=int, 
        required=True,
        help="Number of parallel processes"
    )
    
    parser.add_argument("script", type=str, help="Path to the NetQMPI Python script to be executed")

    backend_group = parser.add_mutually_exclusive_group()
    backend_group.add_argument(
        "--netqasm", "--squidasm", dest="netqasm", action="store_true",
        help="Use the NetQASM/SquidASM backend against NetQASM 2.x")
    backend_group.add_argument(
        "--netqasm1.0", dest="netqasm1", action="store_true",
        help="Use the NetQASM/SquidASM backend against the legacy NetQASM 1.x")
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

    parser.add_argument(
        "--plugin",
        type=str,
        default=None,
        help="Simulation plugin: an installed plugin's name (see "
             "--list-plugins) or 'module:callable'. Overrides the config file. "
             "NetQASM backend only.",
    )

    parser.add_argument(
        "--plugin-config",
        type=str,
        default=None,
        metavar="FILE",
        help="YAML file with the plugin's parameters (its plugin_config). "
             "Overrides the config file. NetQASM backend only.",
    )

    parser.add_argument(
        "--metrics",
        metavar="FILE",
        default=None,
        help="Write the metrics of the run (simulated time per shot, what a "
             "plugin reports) to FILE as JSON. A backend that reports no "
             "metrics writes an empty object.",
    )

    # TODO: Turn ON and OFF the timer

    args = parser.parse_args()

    if args.num_procs < 1:
        parser.error("Number of processes must be at least 1")
    if args.metrics is not None:
        # Checked now, not after a run that may take hours.
        directory = os.path.dirname(os.path.abspath(args.metrics))
        if not os.path.isdir(directory):
            parser.error(f"--metrics: directory {directory!r} does not exist")

    try:
        if args.netqasm or args.netqasm1:
            from netqmpi.runtime.adapters.netqasm import NetQASMExecutorAdapter, NetQASMRunConfig

            # Both flags drive the same adapter: the API it uses is the same
            # in either release. What they pick is the environment, and the
            # executor stops the run if the installed NetQASM is not it.
            config = _build_config(NetQASMRunConfig, "netqasm", args,
                                   netqasm_major=1 if args.netqasm1 else 2)
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
    except (ValueError, OSError, RuntimeError) as error:
        parser.error(str(error))

    simulate(
        script=args.script,
        num_procs=args.num_procs,
        executor=executor
    )

    if args.metrics is not None:
        write_metrics(args.metrics, executor.metrics())

if __name__ == "__main__":
    main()