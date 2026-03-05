"""
Backend adapters for NetQMPI.

This module contains the backend-specific adapters
(NetQASM, Cunqa, etc.) that implement the interfaces defined in core.
"""
# Facilitate direct imports from netqmpi.sdk.adapters
from netqmpi.runtime.adapters.netqasm.netqasm_executor import NetQASMExecutorAdapter
from netqmpi.runtime.adapters.netqasm.netqasm_communicator import NetQASMCommunicator
from netqmpi.runtime.adapters.cunqa.cunqa_executor import CunqaExecutorAdapter
from netqmpi.runtime.adapters.cunqa.cunqa_communicator import CunqaCommunicator

__all__ = [
    'NetQASMExecutorAdapter',
    'NetQASMCommunicator',    
    'CunqaExecutorAdapter',
    'CunqaCommunicator'
]
