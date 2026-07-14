"""Qoala runtime adapters for NetQMPI (simulation only).

This package exposes the Qoala-specific runtime adapter classes: the
communicator, circuit adapter, executor adapter and run configuration.

Qoala is a NetSquid-based simulator; this backend has no real-hardware
execution path and must not be considered on par with a physical deployment.
"""
from netqmpi.runtime.adapters.qoala.qoala_communicator import QoalaCommunicator
from netqmpi.runtime.adapters.qoala.qoala_circuit import QoalaCircuitAdapter
from netqmpi.runtime.adapters.qoala.qoala_executor import (
    QoalaExecutorAdapter,
    QoalaRunConfig,
    QoalaQDeviceConfig,
)

__all__ = [
    'QoalaCommunicator',
    'QoalaCircuitAdapter',
    'QoalaExecutorAdapter',
    'QoalaRunConfig',
    'QoalaQDeviceConfig',
]
