"""
Backend adapters for NetQMPI.

This module contains the backend-specific adapters
(NetQASM, Cunqa, etc.) that implement the interfaces defined in core.
"""
# Facilitate direct imports from netqmpi.sdk.adapters
from netqmpi.runtime.adapters.netqasm_executor import NetQASMExecutorAdapter
from netqmpi.runtime.adapters.cunqa_executor import CunqaExecutorAdapter

__all__ = [
    'NetQASMExecutorAdapter',
    'CunqaExecutorAdapter'
]
