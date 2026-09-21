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
    2. One designated thread translates all the ranks' circuits jointly —
       interleaving them so that transfers pair up and cross-rank
       dependencies survive — and runs the simulation.
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
    # Raised by the designated thread, re-raised by the executor once every
    # rank has been released. Without it a failure there — an unmatched
    # transfer, an unsupported gate — would leave the other ranks waiting on
    # a barrier that nobody will ever reach, and the process would hang
    # instead of reporting what went wrong.
    _error: Optional[BaseException] = None

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
        * **Phase 2** – one designated thread (party 0) translates every
          rank's circuits *together* into the global QuantumCircuit and
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

        # Phase 2: one thread translates every rank's circuits together and
        # runs the simulation. Whatever happens it must reach the next
        # barrier, or the other ranks wait for it forever.
        if party_id == 0:
            try:
                self._translate_all()
                self._executor._run_simulation()
            except BaseException as error:        # noqa: BLE001 - re-raised below
                AerCommunicator._error = error

        # Phase 3: all threads block until the simulation is done, then
        # the designated thread resets shared state.
        AerCommunicator._barrier.wait()

        if party_id == 0:
            self._executor._reset()
            AerCommunicator.communicators = []
            AerCommunicator._barrier = None

        return None

    # ------------------------------------------------------------------
    # Translation
    # ------------------------------------------------------------------

    def _translate_all(self) -> None:
        """
        Translate every rank's circuits into the global circuit, jointly.

        The circuits are paired by creation order: the *i*-th circuit of
        every rank belongs to the same distributed program, so they are
        translated together. Translating them one rank at a time would put
        all of rank 0's gates before any of rank 1's, which silently
        reorders any dependency that does not happen to follow rank order.

        Raises:
            RuntimeError: If the ranks did not create the same number of
                circuits, since their circuits could not then be paired.
        """
        from netqmpi.runtime.adapters.aer.aer_circuit import translate_group

        communicators = {comm.rank: comm
                         for comm in AerCommunicator.communicators}
        ranks = sorted(communicators)

        counts = {len(communicators[rank].circuits) for rank in ranks}
        if len(counts) > 1:
            per_rank = {rank: len(communicators[rank].circuits) for rank in ranks}
            raise RuntimeError(
                "Every rank must create the same number of circuits so that "
                f"they can be paired into distributed programs, got {per_rank}.")

        for index in range(counts.pop() if counts else 0):
            translate_group({rank: communicators[rank].circuits[index]
                             for rank in ranks})
