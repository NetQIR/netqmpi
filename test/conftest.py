"""
Shared test doubles for the backend-independent part of the suite.

Most of NetQMPI can be tested without any simulator at all. The SDK layer
*records* what a program does — gates, transfers, collectives — into an
:class:`~netqmpi.sdk.operations.container.OperationContainer`, and only the
runtime layer turns that record into backend instructions. Everything the
SDK decides on its own (index validation, resource accounting, the tags
that pair a call with its counterpart on another rank, the chunking of a
rooted collective) is therefore observable from the trace alone.

The doubles below supply the two collaborators the SDK needs:

- :class:`StubCommunicator`, a concrete
  :class:`~netqmpi.sdk.communicator.QMPICommunicator` that knows only its
  rank and size and counts context entries.
- :class:`RecordingCircuit`, a concrete
  :class:`~netqmpi.sdk.circuit.Circuit` whose translation hooks record the
  operation they were handed instead of emitting anything, which is what
  makes the dispatch table observable.
- :class:`StubExecutor`, an :class:`~netqmpi.runtime.executor.Executor`
  that hands out :class:`RecordingCircuit` instances.

Tests that need a real simulator live in the backend files and guard
themselves with ``importorskip``.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import pytest

from netqmpi.runtime.executor import Executor
from netqmpi.runtime.run_config import RunConfig
from netqmpi.sdk.circuit import Circuit
from netqmpi.sdk.communicator import QMPICommunicator


class StubCommunicator(QMPICommunicator):
    """
    Communicator that only knows its rank and size.

    :class:`~netqmpi.sdk.communicator.QMPICommunicator` is abstract solely
    because of its context-manager hooks; every primitive it exposes is
    concrete and delegates to the circuit. Filling those two hooks in is
    therefore enough to exercise the whole facade.

    Attributes:
        entered: How many times the block was entered.
        exited: How many times the block was left.
    """

    def __init__(self, rank: int = 0, size: int = 2) -> None:
        super().__init__(rank, size)
        self.entered = 0
        self.exited = 0

    def __enter__(self) -> "StubCommunicator":
        self.entered += 1
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.exited += 1
        return None


class RecordingCircuit(Circuit):
    """
    Circuit whose translation hooks record instead of emitting.

    Each hook appends ``(kind, op)`` to :attr:`translated`, so a test can
    assert *which* hook the dispatcher chose for an operation — the one
    thing the dispatch table is responsible for. Containers recurse into
    their children the way a real adapter does, so a rooted collective
    still reaches the hooks of the transfers it expands into.

    Attributes:
        translated: The ``(kind, operation)`` pairs seen so far.
    """

    def __init__(self, num_qubits: int, num_clbits: int,
                 comm: QMPICommunicator) -> None:
        super().__init__(num_qubits, num_clbits, comm)
        self.translated: List[Tuple[str, Any]] = []

    def _record(self, kind: str, op: Any) -> str:
        self.translated.append((kind, op))
        return kind

    @property
    def kinds(self) -> List[str]:
        """The hook names reached so far, in order."""
        return [kind for kind, _ in self.translated]

    # -- the thirteen hooks a backend adapter must fill in ---------------

    def _translate_gate(self, op):
        return self._record("gate", op)

    def _translate_controlled_gate(self, op):
        return self._record("controlled_gate", op)

    def _translate_classical_controlled_gate(self, op):
        return self._record("classical_controlled_gate", op)

    def _translate_measure(self, op):
        return self._record("measure", op)

    def _translate_reset(self, op):
        return self._record("reset", op)

    def _translate_barrier(self, op):
        return self._record("barrier", op)

    def _translate_operation_container(self, op):
        self._record("container", op)
        for child in op.children:
            self.translate(child)
        return "container"

    def _translate_qsend(self, op):
        return self._record("qsend", op)

    def _translate_qrecv(self, op):
        return self._record("qrecv", op)

    def _translate_qscatter(self, op):
        return self._record("qscatter", op)

    def _translate_qgather(self, op):
        return self._record("qgather", op)

    def _translate_expose(self, op):
        return self._record("expose", op)

    def _translate_unexpose(self, op):
        return self._record("unexpose", op)


class StubExecutor(Executor):
    """
    Executor that hands out :class:`RecordingCircuit` instances.

    Attributes:
        created: The ``(num_qubits, num_clbits)`` pairs asked for.
        built: The ``(file, size)`` pairs :meth:`build_apps` was called with.
        ran: The app lists :meth:`run` was called with.
    """

    def __init__(self, size: int = 2, config: Optional[RunConfig] = None) -> None:
        super().__init__(size, config or RunConfig())
        self.created: List[Tuple[int, int]] = []
        self.built: List[Tuple[str, int]] = []
        self.ran: List[Any] = []

    def create_circuit(self, num_qubits: int, num_clbits: int,
                       comm: QMPICommunicator) -> RecordingCircuit:
        self.created.append((num_qubits, num_clbits))
        return RecordingCircuit(num_qubits, num_clbits, comm)

    def build_apps(self, file: str, size: int) -> List[Any]:
        self.built.append((file, size))
        return [lambda: None for _ in range(size)]

    def run(self, apps: Any) -> None:
        self.ran.append(apps)


@pytest.fixture
def make_circuit():
    """
    Return a factory for a traced-but-not-executed circuit.

    Returns:
        ``factory(num_qubits=2, num_clbits=2, rank=0, size=2)``, which
        builds a :class:`RecordingCircuit` on a fresh
        :class:`StubCommunicator` and registers it there the way
        :meth:`~netqmpi.sdk.environment.Environment.create_circuit` would.
    """
    def factory(num_qubits: int = 2, num_clbits: int = 2,
                rank: int = 0, size: int = 2,
                comm: Optional[QMPICommunicator] = None) -> RecordingCircuit:
        comm = comm if comm is not None else StubCommunicator(rank, size)
        circuit = RecordingCircuit(num_qubits, num_clbits, comm)
        comm.circuits.append(circuit)
        return circuit

    return factory


@pytest.fixture
def make_group(make_circuit):
    """
    Return a factory for one circuit per rank of the same program.

    Collectives are only meaningful across ranks: every participant traces
    its own record of the same call, and the records have to agree. This
    builds the whole set at once.

    Returns:
        ``factory(size, num_qubits=2, num_clbits=2)``, a dict keyed by rank.
    """
    def factory(size: int, num_qubits: int = 2,
                num_clbits: int = 2) -> Dict[int, RecordingCircuit]:
        return {rank: make_circuit(num_qubits, num_clbits, rank=rank, size=size)
                for rank in range(size)}

    return factory
