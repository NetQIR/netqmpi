"""
Concrete :class:`~netqmpi.sdk.communicator.QMPICommunicator` for the Qoala
backend (simulation only).

Following the NetQASM adapter's model, all ranks run in the same process and the
joint NetSquid simulation is deferred until every rank has left its ``with comm``
block. Each rank compiles its circuit to a ``.iqoala`` program on ``__exit__``
and registers it; when the last rank registers, the executor builds the Qoala
network and runs one simulation for all ranks at once.

This module imports no ``qoala`` package: program text is produced by
:class:`~netqmpi.runtime.adapters.qoala.qoala_circuit.QoalaCircuitAdapter` and
the simulation is driven by
:class:`~netqmpi.runtime.adapters.qoala.qoala_executor.QoalaExecutorAdapter`.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Tuple

from netqmpi.sdk import QMPICommunicator

if TYPE_CHECKING:
    from netqmpi.runtime.adapters.qoala.qoala_circuit import QoalaProgramSpec
    from netqmpi.runtime.adapters.qoala.qoala_executor import QoalaExecutorAdapter


class QoalaCommunicator(QMPICommunicator):
    """
    Qoala-backed communicator for a single rank.

    The communicator only orchestrates: it does not execute quantum operations
    (the circuit is deferred and compiled to a Qoala program). It collects one
    program per rank and, once all ``size`` ranks are ready, asks the executor
    to run the joint simulation and stores the per-rank measurement histogram in
    :attr:`results`.
    """

    # rank -> (compiled program spec, communicator instance)
    _registry: Dict[int, Tuple["QoalaProgramSpec", "QoalaCommunicator"]] = {}

    def __init__(self, rank: int, size: int, config: Any, executor: "QoalaExecutorAdapter") -> None:
        """
        Initialize the Qoala communicator.

        Args:
            rank: Numeric index of the current rank.
            size: Total number of ranks in the communicator.
            config: Backend configuration (``QoalaRunConfig``).
            executor: Executor that owns the shared simulation driver.
        """
        super().__init__(rank, size)
        self._config = config
        self._executor = executor

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "QoalaCommunicator":
        """Enter the communicator context."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """
        Compile this rank's circuit and, when all ranks are ready, simulate.

        Args:
            exc_type: Exception type, if one was raised.
            exc_val: Exception instance, if one was raised.
            exc_tb: Traceback, if one was raised.
        """
        # Never swallow an exception raised inside the ``with`` block.
        if exc_type is not None:
            QoalaCommunicator._registry.clear()
            return None

        if len(self.circuits) != 1:
            raise NotImplementedError(
                "The Qoala backend currently supports exactly one circuit per rank "
                f"(rank {self.rank} created {len(self.circuits)})."
            )

        spec = self.circuits[0].build_program()
        QoalaCommunicator._registry[self.rank] = (spec, self)

        if len(QoalaCommunicator._registry) == self.size:
            registry = dict(QoalaCommunicator._registry)
            QoalaCommunicator._registry.clear()
            results = self._executor.run_simulation(registry)
            for rank, comm in ((r, c) for r, (_, c) in registry.items()):
                comm.results = results.get(rank, {})

        return None
