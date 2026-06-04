"""
Executor adapter for Qiskit AerSimulator.

This module provides the :class:`AerExecutorAdapter` implementation of
the :class:`~netqmpi.runtime.executor.Executor` interface for running
NetQMPI applications on Qiskit's AerSimulator backend.
"""
from __future__ import annotations

import threading
from typing import Any, List, Tuple

from netqmpi.runtime.executor import Executor
from netqmpi.sdk.environment import Environment
from netqmpi.runtime.adapters.aer.aer_circuit import AerCircuitAdapter
from netqmpi.runtime.adapters.aer.aer_communicator import AerCommunicator
from netqmpi.runtime.adapters.aer.aer_run_config import AerSimulatorConfig
from netqmpi.helpers import load_main


class AerExecutorAdapter(Executor):
    """
    Executor adapter that runs NetQMPI apps on Qiskit's AerSimulator.

    Owns a single global QuantumCircuit that grows with every
    :meth:`create_circuit` call.  Each call allocates a new contiguous
    *circuit group* of ``size * num_qubits`` qubits (one slice per rank),
    so any number of circuits can be created per rank while the SWAP-based
    qsend offsets remain consistent.

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
        # Each entry: (num_qubits, num_clbits, qubit_base, clbit_base)
        self._circuit_groups: List[Tuple[int, int, int, int]] = []
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
        Create an AerCircuitAdapter and extend the global circuit if needed.

        Thread-safe: multiple ranks may call this simultaneously.  The
        first rank to request a given circuit group allocates
        ``size * num_qubits`` new qubits for that group; subsequent ranks
        reuse the already-allocated group.

        Args:
            num_qubits: Number of qubits for this rank's circuit slice.
            num_clbits: Number of classical bits for this rank's circuit slice.
            comm: Communicator associated with this rank.

        Returns:
            An :class:`AerCircuitAdapter` writing into the global circuit at
            the correct qubit/clbit offset for this rank and circuit group.
        """
        with self._lock:
            if self._global_circuit is None:
                from qiskit import QuantumCircuit  # type: ignore[import-not-found]
                self._global_circuit = QuantumCircuit(0, 0)

            # len(comm.circuits) is the index of the circuit being created:
            # Environment.create_circuit() appends AFTER this method returns.
            circuit_index = len(comm.circuits)

            if circuit_index >= len(self._circuit_groups):
                # First rank for this group: allocate slots for all ranks.
                from qiskit import QuantumRegister, ClassicalRegister  # type: ignore[import-not-found]
                qr = QuantumRegister(self._size * num_qubits, f'qr{circuit_index}')
                cr = ClassicalRegister(self._size * num_clbits, f'cr{circuit_index}')
                self._global_circuit.add_register(qr)
                self._global_circuit.add_register(cr)
                self._circuit_groups.append(
                    (num_qubits, num_clbits, self._qubit_count, self._clbit_count)
                )
                self._qubit_count += self._size * num_qubits
                self._clbit_count += self._size * num_clbits

            _, _, qubit_base, clbit_base = self._circuit_groups[circuit_index]
            qubit_offset = qubit_base + comm._rank * num_qubits
            clbit_offset = clbit_base + comm._rank * num_clbits

            return AerCircuitAdapter(
                num_qubits, num_clbits, comm,
                self._global_circuit, qubit_offset, clbit_offset, qubit_base,
            )

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
        """
        threads = [threading.Thread(target=app) for app in apps]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    # ------------------------------------------------------------------
    # Internal helpers called by AerCommunicator.__exit__
    # ------------------------------------------------------------------

    def _run_simulation(self) -> None:
        """
        Submit the global circuit to AerSimulator and broadcast counts.

        Called by the designated thread inside ``AerCommunicator.__exit__``
        after all ranks have finished building their circuits.
        """
        from qiskit_aer import AerSimulator  # type: ignore[import-not-found]

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
        self._circuit_groups = []
        self._qubit_count = 0
        self._clbit_count = 0
