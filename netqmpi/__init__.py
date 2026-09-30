from netqmpi import version
from netqmpi.version import describe, get_build_info

__all__ = ["__version__", "describe", "get_build_info", "version"]


def __getattr__(name: str):
    # Resolved on first use, so that importing NetQMPI never waits on git.
    if name == "__version__":
        return get_build_info().version
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
