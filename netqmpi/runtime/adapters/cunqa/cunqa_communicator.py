"""
Concrete :class:`~netqmpi.sdk.communicator.base.BaseCommunicator` backed
by the CUNQA runtime.

This communicator adapts the backend-specific communication layer to the
backend-agnostic :class:`QMPICommunicator` interface.

CUNQA takes the whole distributed program at once — every rank's circuit
is submitted together and the vQPUs resolve the communication directives
between them at run time. The ranks therefore do not execute as they are
traced: each of them records its circuits, and the last one to leave its
``with comm:`` block triggers the joint translation and the actual run.
"""
from __future__ import annotations

from typing import Any, Dict, List

from netqmpi.sdk import QMPICommunicator
from netqmpi.runtime.run_config import RunConfig

from cunqa.qpu import QPU, run
from cunqa.qjob import gather


class CunqaSession:
    """
    State shared by every rank of a single NetQMPI run.

    Holds what only makes sense for the program as a whole: the vQPUs, the
    run configuration, the communicators that have already finished
    tracing, and the results once they are back.

    The results of a distributed program are the results of *every* rank —
    a single rank's counts say nothing about the correlations the program
    was written to produce — so they are kept here, whole, and shared with
    all the communicators rather than split among them.

    Attributes:
        size: Number of ranks in the run.
        qpus: vQPUs backing the ranks, ordered by rank.
        config: Run configuration shared by every rank.
        communicators: Communicator of each rank, keyed by rank.
        finished: Ranks that are done tracing.
        results: Counts of every rank, keyed by rank, once the run is over.
    """

    def __init__(self, size: int, qpus: List[QPU], config: RunConfig) -> None:
        """
        Initialize the session.

        Args:
            size: Number of ranks in the run.
            qpus: vQPUs backing the ranks, ordered by rank.
            config: Run configuration shared by every rank.
        """
        self.size = size
        self.qpus = qpus
        self.config = config
        self.communicators: Dict[int, CunqaCommunicator] = {}
        self.finished: set = set()
        self.results: Dict[int, Any] = {}


class CunqaCommunicator(QMPICommunicator):
    """
    CUNQA-backed communicator for a single rank.

    This class provides the communicator implementation used by the CUNQA
    backend and is injected into :class:`~netqmpi.sdk.communicator.QMPICommunicator`.

    Args:
        rank: Numeric index of the current rank.
        size: Total number of ranks in the communicator.
        qpu: vQPU backing this rank.
        config: Run configuration.
        session: State shared with the other ranks of the run.
    """

    def __init__(
        self,
        rank: int,
        size: int,
        qpu: QPU,
        config: RunConfig,
        session: CunqaSession = None,
    ) -> None:
        """
        Initialize the communicator.

        Args:
            rank: Numeric index of the current rank.
            size: Total number of ranks in the communicator.
            qpu: vQPU backing this rank.
            config: Run configuration.
            session: State shared with the other ranks of the run. When
                omitted, a single-rank session is created.
        """
        super().__init__(rank, size)
        self._config = config
        self._session = session or CunqaSession(size, [qpu], config)
        self._session.communicators[rank] = self

    @property
    def session(self) -> CunqaSession:
        """
        Return the state shared with the other ranks of the run.

        Returns:
            The session this communicator belongs to.
        """
        return self._session

    def __enter__(self) -> None:
        """
        Enter the runtime context for the communicator.

        Returns:
            None.
        """
        return None

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """
        Exit the runtime context for the communicator.

        Marks this rank as done tracing. The last rank to leave its block
        translates every rank's circuits together and runs them, because
        the distributed protocols only exist once all the participants are
        known.

        Args:
            exc_type: Exception type, if an exception was raised.
            exc_val: Exception instance, if an exception was raised.
            exc_tb: Traceback, if an exception was raised.

        Returns:
            None.
        """
        if exc_type is not None:
            return None

        self._session.finished.add(self.rank)
        if len(self._session.finished) == self._session.size:
            self._execute_session()

        return None

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def _execute_session(self) -> None:
        """
        Translate and run every rank's circuits, then publish the results.

        Circuits are grouped by creation order: the *i*-th circuit of every
        rank belongs to the same distributed program and is translated and
        submitted together with its peers.

        The counts of every rank are stored whole in the session and shared
        with all the communicators, so reading ``comm.results`` from any
        rank gives the complete picture of the run rather than that rank's
        slice of it.

        Raises:
            RuntimeError: If the ranks did not create the same number of
                circuits, since their circuits could not then be paired.
        """
        # Local import: the circuit adapter imports this module.
        from netqmpi.runtime.adapters.cunqa.cunqa_circuit import translate_group

        session = self._session
        communicators = session.communicators
        ranks = sorted(communicators)

        counts = {len(communicators[r].circuits) for r in ranks}
        if len(counts) > 1:
            per_rank = {r: len(communicators[r].circuits) for r in ranks}
            raise RuntimeError(
                "Every rank must create the same number of circuits so that "
                f"they can be paired into distributed programs, got {per_rank}.")

        per_group: Dict[int, List] = {r: [] for r in ranks}
        for index in range(counts.pop() if counts else 0):
            group = {r: communicators[r].circuits[index] for r in ranks}
            cunqa_circuits = translate_group(group)

            qjobs = run(cunqa_circuits, session.qpus, shots=session.config.shots)
            for rank, result in zip(ranks, gather(qjobs)):
                per_group[rank].append(result.counts)

        # A rank that built a single circuit gets its counts directly; one
        # that built several gets the list, in creation order.
        session.results = {
            rank: (group_counts[0] if len(group_counts) == 1 else group_counts)
            for rank, group_counts in per_group.items()
        }
        for rank in ranks:
            communicators[rank].results = session.results
