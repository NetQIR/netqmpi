"""
Backend adapters for NetQMPI.

This module contains the backend-specific adapters
(NetQASM, Cunqa, etc.) that implement the interfaces defined in core.
"""
# Facilitate direct imports from netqmpi.sdk.adapters
from netqmpi.sdk.adapters.netqasm import NetQASMExecutorAdapter, NetQASMCircuitAdapter
from netqmpi.sdk.adapters.cunqa import CunqaExecutorAdapter, CunqaCircuitAdapter

__all__ = [
    'NetQASMExecutorAdapter',
    'NetQASMCircuitAdapter',
    'CunqaExecutorAdapter',
    'CunqaCircuitAdapter',
]
