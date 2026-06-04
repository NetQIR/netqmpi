"""
Communicator adapter for Qiskit AerSimulator.

Manages the context lifecycle for a single rank.  The global
QuantumCircuit is owned by :class:`AerExecutorAdapter`; this class
coordinates the barrier synchronization that ensures all ranks have
finished building their circuits before the simulation runs, and that
all ranks receive results before any of them continue past the
``with env.comm:`` block.
"""
from __future__ import annotations

import threading
from typing import TYPE_CHECKING, List, Optional

from netqmpi.sdk.communicator import QMPICommunicator
from netqmpi.runtime.adapters.aer.aer_run_config import AerSimulatorConfig

if TYPE_CHECKING:
    from netqmpi.runtime.adapters.aer.aer_executor import AerExecutorAdapter


class AerCommunicator(QMPICommunicator):
    """
    AerSimulator-backed communicator for a single rank.

    All N ranks run concurrently in separate threads.  ``__exit__`` uses
    a :class:`threading.Barrier` to synchronise them:

    1. Every rank finishes building its circuit ops and reaches the barrier.
    2. One designated thread translates all circuits (in rank order, so
       gate ordering in the global circuit is deterministic) and runs the
       simulation.
    3. All threads are released with results available and continue past
       the ``with env.comm:`` block simultaneously.

    The barrier and class-level communicator list are reset after the last
    rank exits so the adapter is reusable within the same process.

    Args:
        rank: Numeric index of the current rank.
        size: Total number of ranks.
        config: AerSimulator-specific configuration.
        executor: Executor that owns the global QuantumCircuit.
    """

    # All AerCommunicator instances for the current run (rank-ordered).
    communicators: List["AerCommunicator"] = []
    # Set by AerExecutorAdapter.build_apps after all communicators are created.
    _barrier: Optional[threading.Barrier] = None

    def __init__(
        self,
        rank: int,
        size: int,
        config: AerSimulatorConfig,
        executor: "AerExecutorAdapter",
    ) -> None:
        """
        Initialize the communicator.

        Args:
            rank: Numeric index of the current rank.
            size: Total number of ranks in the communicator.
            config: AerSimulator-specific configuration.
            executor: Executor that owns the global QuantumCircuit.
        """
        super().__init__(rank, size)
        self._config = config
        self._executor = executor
        AerCommunicator.communicators.append(self)

    def __enter__(self) -> "AerCommunicator":
        """
        Enter the communicator context.

        Returns:
            The communicator instance.
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """
        Synchronise all ranks, run the simulation, and broadcast results.

        Three-phase barrier protocol:

        * **Phase 1** – all ranks wait until every rank has finished
          appending operations to its circuit.
        * **Phase 2** – one designated thread (party 0) translates all
          circuits into the global QuantumCircuit in rank order and
          submits the simulation.  All other threads block here.
        * **Phase 3** – all threads are released once results are
          available; the designated thread resets class-level state for
          the next run.

        After this method returns, ``env.comm.results`` is populated for
        every rank.

        Args:
            exc_type: Exception type, if one was raised.
            exc_val: Exception instance, if one was raised.
            exc_tb: Traceback, if one was raised.
        """
        # Phase 1: wait for every rank to finish building its circuit.
        party_id = AerCommunicator._barrier.wait()

        # Phase 2: one thread translates all circuits in rank order and
        # runs the simulation.  Rank order is deterministic because
        # build_apps creates communicators 0..size-1 in sequence.
        if party_id == 0:
            for comm in AerCommunicator.communicators:
                for circuit in comm.circuits:
                    circuit.translate(circuit.ops)
            self._executor._run_simulation()

        # Phase 3: all threads block until the simulation is done, then
        # the designated thread resets shared state.
        AerCommunicator._barrier.wait()

        if party_id == 0:
            self._executor._reset()
            AerCommunicator.communicators = []
            AerCommunicator._barrier = None

        return None
