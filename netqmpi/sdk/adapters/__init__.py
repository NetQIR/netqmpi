"""
Backend adapters for NetQMPI.

This module contains the backend-specific adapters
(NetQASM, Cunqa, etc.) that implement the interfaces defined in core.
"""
# Facilitate direct imports from netqmpi.sdk.adapters
from netqmpi.sdk.adapters.netqasm import NetQASMCircuitAdapter
from netqmpi.sdk.adapters.cunqa import CunqaCircuitAdapter

__all__ = [
    'NetQASMCircuitAdapter',
    'CunqaCircuitAdapter',
]
