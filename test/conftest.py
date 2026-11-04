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

The :func:`install_plugins` fixture installs simulation plugins the way a
plugin package does, through ``netqmpi.plugins`` entry points, without pip.

Tests that need a real simulator live in the backend files and guard
themselves with ``importorskip``.
"""
from __future__ import annotations

import sys
import textwrap
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


#: The module of the dummy plugin package :func:`install_plugins` installs.
#: Every kind of plugin NetQMPI has to tell apart lives in it.
DUMMY_PLUGIN_MODULE = "netqmpi_dummy_plugin"

DUMMY_PLUGIN_SOURCE = """
    import contextlib

    CALLS = []

    class Device:
        \"\"\"A plugin with every optional member of the contract.\"\"\"
        run_defaults = {"num_qubits": 3, "shots": 7}
        backends = ("netqasm",)

        def configure(self, plugin_config, run_config):
            CALLS.append(("configure", dict(plugin_config),
                          run_config.num_qubits, run_config.shots))
            if plugin_config.get("bad"):
                raise ValueError("dummy_device: 'bad' is not a parameter")
            # Neither change may reach the run: both are copies.
            run_config.num_qubits = 99
            plugin_config["tampered"] = True

        def __call__(self, plugin_config):
            CALLS.append(("enter", dict(plugin_config)))
            return contextlib.nullcontext()

    device = Device()

    def plain(plugin_config):
        \"\"\"A plugin with none of the optional members.\"\"\"
        CALLS.append(("enter", dict(plugin_config)))
        return contextlib.nullcontext()

    class ForQoala:
        backends = ("qoala",)

        def __call__(self, plugin_config):
            return contextlib.nullcontext()

    for_qoala = ForQoala()

    class TypoInDefaults:
        run_defaults = {"num_qbits": 3}

        def __call__(self, plugin_config):
            return contextlib.nullcontext()

    typo_in_defaults = TypoInDefaults()
"""

#: Entry points of the dummy distribution: name -> ``module:attribute``.
DUMMY_ENTRY_POINTS = {
    "dummy_device": f"{DUMMY_PLUGIN_MODULE}:device",
    "dummy_plain": f"{DUMMY_PLUGIN_MODULE}:plain",
    "dummy_qoala": f"{DUMMY_PLUGIN_MODULE}:for_qoala",
    "dummy_typo": f"{DUMMY_PLUGIN_MODULE}:typo_in_defaults",
    "dummy_missing": f"{DUMMY_PLUGIN_MODULE}:no_such_attribute",
}


def write_distribution(root, name: str, version: str,
                       entry_points: Dict[str, str],
                       modules: Optional[Dict[str, str]] = None) -> None:
    """
    Lay out an installed distribution under ``root``, as pip would.

    ``importlib.metadata`` finds a distribution by its ``.dist-info``
    directory on ``sys.path``; this writes one with the given
    ``netqmpi.plugins`` entry points, plus the modules they point to.

    Args:
        root: Directory that is (or will be) on ``sys.path``.
        name: Distribution name, e.g. ``"netqmpi-dummy"``.
        version: Distribution version.
        entry_points: Entry-point name to ``module:attribute``.
        modules: Module name to source, written next to the metadata.
    """
    info = root / f"{name.replace('-', '_')}-{version}.dist-info"
    info.mkdir(parents=True)
    (info / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n")
    lines = "".join(f"{key} = {value}\n" for key, value in entry_points.items())
    (info / "entry_points.txt").write_text(f"[netqmpi.plugins]\n{lines}")
    for module, source in (modules or {}).items():
        (root / f"{module}.py").write_text(textwrap.dedent(source))


@pytest.fixture
def install_plugins(tmp_path, monkeypatch):
    """
    Install the dummy plugin distribution ``netqmpi-dummy`` 1.2.3.

    It goes into a directory of its own on ``sys.path``, standing in for
    ``site-packages``, so the plugins are found from any working directory
    and never from the test's own.

    Returns:
        ``install(root=None, name=..., version=..., entry_points=..., modules=...)``
        to install further distributions; called with no arguments it
        installs nothing more and returns the dummy plugin module's name.
    """
    site = tmp_path / "site-packages"
    site.mkdir()
    write_distribution(site, "netqmpi-dummy", "1.2.3", DUMMY_ENTRY_POINTS,
                       {DUMMY_PLUGIN_MODULE: DUMMY_PLUGIN_SOURCE})
    monkeypatch.syspath_prepend(str(site))

    def install(**distribution):
        if distribution:
            write_distribution(site, **distribution)
        return DUMMY_PLUGIN_MODULE

    yield install
    # The module records calls; the next test must import a fresh one.
    sys.modules.pop(DUMMY_PLUGIN_MODULE, None)
