"""
NetQMPI SDK core module.

Contains the user-facing abstractions that define the programming
interface for distributed quantum applications:

- :class:`~netqmpi.sdk.core.environment.Environment` – runtime context
  injected into every ``main()`` function.
- :class:`~netqmpi.sdk.core.circuit.Circuit` – abstract quantum circuit.
"""
from netqmpi.sdk.core.environment import Environment
from netqmpi.sdk.core.circuit import Circuit

__all__ = ['Environment', 'Circuit']
