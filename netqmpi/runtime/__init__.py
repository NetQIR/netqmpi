"""
NetQMPI runtime package.

Responsible for setting up the N-process execution environment.
Exports the abstract building blocks used by all backend adapters:

- :class:`~netqmpi.runtime.executor.Executor`   – abstract contract for circuit
                                                   creation, app build, and run.
- :class:`~netqmpi.runtime.run_config.RunConfig` – backend-agnostic run config
"""
from netqmpi.runtime.executor import Executor
from netqmpi.runtime.run_config import RunConfig

__all__ = ['Executor', 'RunConfig']
