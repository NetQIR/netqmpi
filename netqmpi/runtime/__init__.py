"""
NetQMPI runtime package.

This package is responsible for setting up the multi-process execution
environment and exporting the abstract building blocks shared by all
backend adapters.
"""
from netqmpi.runtime.executor import Executor
from netqmpi.runtime.run_config import RunConfig

__all__ = ['Executor', 'RunConfig']
