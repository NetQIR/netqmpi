"""
NetQMPI core module.

Contains the base abstractions (Executor and Circuit) that define
the common interface for all quantum backends.
"""
from netqmpi.runtime.executor import Executor
from netqmpi.sdk.core.circuit import Circuit

__all__ = ['Executor', 'Circuit']
