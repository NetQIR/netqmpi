"""
Backend-agnostic utilities for loading user scripts at runtime.

This module intentionally contains no backend-specific imports.  All
backend-specific wiring (environment injection, ApplicationInstance
creation, etc.) lives in the adapter packages
(``netqmpi.sdk.adapters.*``).
"""
import importlib.util
import os.path
import sys


def import_module_from_path(path: str):
    """
    Dynamically import a Python file as a module.

    The module is registered in ``sys.modules`` under its filename stem so
    that subsequent imports of the same path return the cached module.

    Args:
        path: Absolute or relative path to the ``.py`` file.

    Returns:
        The loaded module object.

    Raises:
        ImportError:  If the module spec cannot be created from *path*.
        FileNotFoundError: If *path* does not exist.
    """
    module_name = os.path.splitext(os.path.basename(path))[0]

    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None:
        raise ImportError(f"Could not load spec from {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module
