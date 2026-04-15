"""
NetQMPI SDK core module.

Contains the user-facing abstractions that define the programming
interface for distributed quantum applications:

- :class:`~netqmpi.sdk.environment.Environment` – runtime context
  injected into every ``main()`` function.
- :class:`~netqmpi.sdk.circuit.Circuit` – abstract quantum circuit.
"""
from netqmpi.sdk.circuit import Circuit
from netqmpi.sdk.communicator import QMPICommunicator
from netqmpi.sdk.environment import Environment

__all__ = [
  'Circuit',
  'QMPICommunicator',
  'Environment'
]
