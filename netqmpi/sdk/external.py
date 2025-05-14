import importlib
import os.path
import sys
import importlib.util
from typing import Optional

from netqasm.runtime import env
from netqasm.runtime.application import ApplicationInstance, Program, Application
from netqasm.util.yaml import load_yaml

def import_module_from_path(path):
    module_name = os.path.splitext(os.path.basename(path))[0]

    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None:
        raise ImportError(f"Could not load spec from {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

def app_instance_from_file(file: str = None, num_processes: int = 2,
                           argv_file: str = None,
                           roles_cfg_file: str = 'roles.yaml') -> ApplicationInstance:
    """

    Create an Application Instance from a single file repeated num_processes times.
    :param roles_cfg_file: path to the roles configuration file.
    :param argv_file: file with the arguments values for the application.
    :param num_processes: number of processes to create.
    :param file: the NetQASM file to load.
    :return: ApplicationInstance
    """

    if file is None:
        raise ValueError("file must be provided")
    if not file.endswith(".py"):
        raise ValueError("file must be a .py file")

    program_files = {}

    for i in range(num_processes):
        program_files[f"rank_{i}"] = file

    programs = []
    argv_per_rank = {}  # Normally the values is the same for all ranks and, by default, append the rank and the size

    argv = {}
    if argv_file is not None:
        argv = load_yaml(argv_file)

    for rank, program_file in program_files.items():
        current_argv = argv.copy()

        prog_module = import_module_from_path(program_file)

        main_func = getattr(prog_module, "main")
        if main_func is None:
            raise ValueError(f"main function not found in {program_file}")

        prog = Program(party=rank, entry=main_func, args=[], results=[])
        programs += [prog]
        current_argv["rank"] = int(rank.split("_")[-1])
        current_argv["size"] = num_processes
        argv_per_rank[rank] = current_argv

    roles = env.load_roles_config(roles_cfg_file)
    roles = (
        {prog.party: prog.party for prog in programs}
        if roles is None
        else roles
    )

    app = Application(programs = programs, metadata = None)
    app_instance = ApplicationInstance(
        app = app,
        program_inputs=argv_per_rank,
        network=None,
        party_alloc=roles,
        logging_cfg=None
    )

    return app_instance
