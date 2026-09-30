import os
import sys
import types
from typing import Dict, Tuple

#: Compiled app scripts, keyed by ``(absolute path, mtime_ns, size)``.
_CODE_CACHE: Dict[Tuple[str, int, int], types.CodeType] = {}


def _compiled(path):
    """
    Return the compiled code of a script, compiling it only when it changed.

    Reading and compiling the script is what ``runpy.run_path`` spends its
    time on, and on a shared file system (BeeGFS) that came to 7-10 ms per
    run against 0.4 ms on a local disk. A ``stat`` is enough to tell whether
    the cached code is still the script's.

    Args:
        path: Path of the script.

    Returns:
        The script's code object.
    """
    stat = os.stat(path)
    key = (os.path.abspath(path), stat.st_mtime_ns, stat.st_size)
    code = _CODE_CACHE.get(key)
    if code is None:
        with open(path, "rb") as fh:
            source = fh.read()
        code = compile(source, path, "exec", dont_inherit=True)
        _CODE_CACHE[key] = code
    return code


def _run_path(path):
    """
    Run a script the way ``runpy.run_path`` does, from cached code.

    The module body still runs on every call — scripts may read their
    parameters from the environment at import, as the benchmark apps do —
    only compiling it is skipped.

    Args:
        path: Path of the script.

    Returns:
        The script's global namespace after running it.
    """
    code = _compiled(path)
    run_name = "<run_path>"
    module = types.ModuleType(run_name)
    module.__dict__.update(__file__=path, __cached__=None, __loader__=None,
                           __package__="", __spec__=None)

    saved_module = sys.modules.get(run_name)
    saved_argv0 = sys.argv[0] if sys.argv else None
    sys.modules[run_name] = module
    if sys.argv:
        sys.argv[0] = path
    try:
        exec(code, module.__dict__)
    finally:
        if saved_module is None:
            sys.modules.pop(run_name, None)
        else:
            sys.modules[run_name] = saved_module
        if sys.argv:
            sys.argv[0] = saved_argv0
    # Like runpy, hand back a copy: the temporary module's namespace may be
    # cleared once it is dropped.
    return dict(module.__dict__)


def load_main(path):
    if path is None:
        raise ValueError("script must be provided")
    if not path.endswith(".py"):
        raise ValueError("script must be a .py script")

    namespace = _run_path(path)

    if "main" not in namespace:
        raise ValueError(f"{path} does not define a main() function")

    if namespace["main"] is None:
            raise ValueError(f"main function not found in {path}")

    return namespace["main"]
