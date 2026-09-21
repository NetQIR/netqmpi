"""
Executor adapter for Qiskit AerSimulator.

This module provides the :class:`AerExecutorAdapter` implementation of
the :class:`~netqmpi.runtime.executor.Executor` interface for running
NetQMPI applications on Qiskit's AerSimulator backend.
"""
from __future__ import annotations

import threading
from typing import Any, List, Tuple

# Imported at module scope, not lazily inside the methods that use them.
# This package is only ever imported once a run has chosen the Aer backend
# (the CLI defers it to its ``--aer`` branch), so there is nothing to gain
# by deferring further — and deferring actively misleads: importing Qiskit
# takes the best part of a second, and paying for it inside
# ``create_circuit`` charged it to the user's trace, where a profiler reads
# it as the cost of building the circuit rather than of loading a library.
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister  # type: ignore[import-not-found]
from qiskit_aer import AerSimulator  # type: ignore[import-not-found]

from netqmpi.runtime.executor import Executor
from netqmpi.sdk.environment import Environment
from netqmpi.runtime.adapters.aer.aer_circuit import AerCircuitAdapter
from netqmpi.runtime.adapters.aer.aer_communicator import AerCommunicator
from netqmpi.runtime.adapters.aer.aer_run_config import AerSimulatorConfig
from netqmpi.helpers import load_main


class AerExecutorAdapter(Executor):
    """
    Executor adapter that runs NetQMPI apps on Qiskit's AerSimulator.

    Owns the single global QuantumCircuit every rank writes into. It is
    built by :meth:`lay_out` once the ranks have finished tracing, because
    only then are the widths they each asked for known; a rank's slice is
    sized to its own request rather than to a width assumed common to all.

    :meth:`run` launches every rank in a separate thread.
    :meth:`build_apps` installs a :class:`threading.Barrier` on
    :class:`AerCommunicator` so that ``__exit__`` can synchronise all
    threads before and after the simulation.
    """

    def __init__(self, size: int, config: AerSimulatorConfig = None) -> None:
        """
        Initialize the AerSimulator executor adapter.

        Args:
            size: Number of parallel ranks to simulate.
            config: AerSimulator-specific configuration.  Defaults to
                :class:`AerSimulatorConfig` with its built-in defaults.
        """
        _config = config or AerSimulatorConfig()
        super().__init__(size, _config)
        # Re-narrow the type so the checker knows we have AerSimulatorConfig.
        self._config: AerSimulatorConfig = _config
        self._global_circuit = None
        self._qubit_count: int = 0
        self._clbit_count: int = 0
        # Protects global-circuit mutations when ranks call create_circuit concurrently.
        self._lock = threading.Lock()

    def create_circuit(
        self,
        num_qubits: int,
        num_clbits: int,
        comm: AerCommunicator,
    ) -> AerCircuitAdapter:
        """
        Create an AerCircuitAdapter for one rank.

        No space is reserved here. The ranks may ask for registers of
        different widths — a ``qscatter`` root holds one qubit per receiver
        while each receiver holds one — so a rank's slice cannot be placed
        from its rank index and its own width alone; doing that overlapped
        the slices and silently corrupted the program. :meth:`lay_out`
        places them all once every rank has finished tracing.

        Thread-safe: multiple ranks may call this simultaneously.

        Args:
            num_qubits: Number of qubits for this rank's circuit slice.
            num_clbits: Number of classical bits for this rank's circuit slice.
            comm: Communicator associated with this rank.

        Returns:
            An :class:`AerCircuitAdapter` whose slice is placed later.
        """
        return AerCircuitAdapter(num_qubits, num_clbits, comm)

    def lay_out(self, groups: List[List[AerCircuitAdapter]]) -> None:
        """
        Build the global circuit and give every rank its slice.

        Slices are laid out group by group and, within a group, in rank
        order, so the bit layout of the resulting histogram is deterministic
        and independent of the order in which the rank threads happened to
        reach :meth:`create_circuit`.

        Args:
            groups: The circuits of each distributed program, rank-ordered
                within each group.
        """
        with self._lock:
            self._global_circuit = QuantumCircuit(0, 0)
            self._qubit_count = 0
            self._clbit_count = 0

            for index, group in enumerate(groups):
                for rank, adapter in enumerate(group):
                    if adapter.num_qubits:
                        self._global_circuit.add_register(
                            QuantumRegister(adapter.num_qubits,
                                            f"q{index}_r{rank}"))
                    if adapter.num_clbits:
                        self._global_circuit.add_register(
                            ClassicalRegister(adapter.num_clbits,
                                              f"c{index}_r{rank}"))
                    adapter.assign_slice(self._global_circuit,
                                         self._qubit_count, self._clbit_count)
                    self._qubit_count += adapter.num_qubits
                    self._clbit_count += adapter.num_clbits

    def build_apps(self, file: str, size: int) -> List[Any]:
        """
        Build one callable wrapper per rank and install the sync barrier.

        Creates all :class:`AerCommunicator` instances and then installs
        a :class:`threading.Barrier` on the class so that every rank's
        ``__exit__`` can synchronise before the simulation runs.

        Args:
            file: Path to the NetQMPI Python script defining ``main()``.
            size: Number of ranks to instantiate.

        Returns:
            A list of zero-argument callables, one per rank.
        """
        main_func = load_main(file)
        apps = []
        for rank in range(size):
            comm = AerCommunicator(rank, size, self._config, self)
            env = Environment(comm, self)
            wrapped_main = lambda env=env: main_func(env=env)
            apps.append(wrapped_main)
        # Install the barrier after all communicators exist so __exit__ can use it.
        AerCommunicator._barrier = threading.Barrier(size)
        return apps

    def run(self, apps: List[Any]) -> None:
        """
        Launch every rank in a separate thread and wait for all to finish.

        Running ranks concurrently is required so that the
        :class:`threading.Barrier` in ``AerCommunicator.__exit__`` can
        synchronise them: all N threads must reach the barrier for any of
        them to proceed past it.

        Args:
            apps: List of callables returned by :meth:`build_apps`.

        Raises:
            Exception: Whatever the designated thread raised while
                translating or simulating, re-raised here once every rank
                has been released.
        """
        AerCommunicator._error = None

        threads = [threading.Thread(target=app) for app in apps]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # A failure inside the designated thread is captured there rather
        # than raised, so that every rank still clears the barrier; this is
        # where it becomes the caller's problem.
        error, AerCommunicator._error = AerCommunicator._error, None
        if error is not None:
            raise error

    # ------------------------------------------------------------------
    # Internal helpers called by AerCommunicator.__exit__
    # ------------------------------------------------------------------

    def _run_simulation(self) -> None:
        """
        Submit the global circuit to AerSimulator and broadcast counts.

        Called by the designated thread inside ``AerCommunicator.__exit__``
        after all ranks have finished building their circuits.
        """
        run_kwargs: dict = {"shots": self._config.shots}
        if self._config.seed_simulator is not None:
            run_kwargs["seed_simulator"] = self._config.seed_simulator

        simulator = AerSimulator()
        job = simulator.run(self._global_circuit, **run_kwargs)
        counts = job.result().get_counts()

        for comm in AerCommunicator.communicators:
            comm.results = counts

    def _reset(self) -> None:
        """
        Reset executor state for the next run.

        Called by the designated thread inside ``AerCommunicator.__exit__``
        after all ranks have received their results.
        """
        self._global_circuit = None
        self._qubit_count = 0
        self._clbit_count = 0
