"""
Environment – the object injected into every user ``main()`` function.

Acts as the bridge between the RUNTIME layer (which sets up the N-process
execution environment and chooses the backend) and the SDK layer (which
provides the user-facing programming API).

By depending only on this class, user application files remain completely
backend-agnostic: they never import a concrete executor or backend adapter.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from netqmpi.sdk.communicator.communicator import QMPICommunicator
from netqmpi.sdk.core.circuit import Circuit

if TYPE_CHECKING:
    pass


class Environment:
    """
    Runtime context injected into every user ``main()`` function.

    Encapsulates two responsibilities:

    * **Communication** – exposes the
      :class:`~netqmpi.sdk.communicator.QMPICommunicator` for this rank via
      the :attr:`comm` property.
    * **Circuit creation** – exposes :meth:`create_circuit`, a factory method
      that delegates to the backend
      :class:`~netqmpi.runtime.executor.Executor` chosen by the runtime, so
      that user code never has to reference a backend-specific class.

    Example usage inside a user application::

        def main(env: Environment = None):
            rank = env.comm.get_rank()
            circuit = env.create_circuit(num_qubits=2, num_clbits=1)

            with env.comm:
                ...

    Args:
        comm: The :class:`~netqmpi.sdk.communicator.QMPICommunicator` for
              this rank, created by the runtime.
        executor: Any object that implements
                  ``create_circuit(num_qubits, num_clbits)``.  In practice
                  this will be a concrete
                  :class:`~netqmpi.runtime.executor.Executor` subclass
                  instantiated by the runtime.
    """

    def __init__(self, comm: QMPICommunicator, executor) -> None:
        self._comm = comm
        self._executor = executor

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def comm(self) -> QMPICommunicator:
        """The :class:`~netqmpi.sdk.communicator.QMPICommunicator` for this rank."""
        return self._comm

    def create_circuit(self, num_qubits: int, num_clbits: int) -> Circuit:
        """
        Factory Method – creates a backend-specific quantum circuit.

        Delegates to the underlying
        :class:`~netqmpi.runtime.executor.Executor` so that user code
        remains backend-agnostic.

        Args:
            num_qubits: Number of qubits in the circuit.
            num_clbits: Number of classical bits in the circuit.

        Returns:
            A backend-specific :class:`~netqmpi.sdk.core.circuit.Circuit`
            instance ready to receive quantum operations.
        """
        return self._executor.create_circuit(num_qubits, num_clbits)
