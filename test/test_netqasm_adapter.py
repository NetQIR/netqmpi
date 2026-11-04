"""
Regression tests for the NetQASM adapter that do not need SquidASM.

Everything the adapter decides before handing the run to SquidASM — which
network to simulate on, whether the program fits in it, which EPR sockets
each rank opens, where logs go, how SquidASM is patched, what happens when
the simulation fails — is exercised here with ``simulate_application``
replaced by a double. What SquidASM then does with it is the business of the
``integration`` tests in ``test_netqasm_backend.py``.
"""
from __future__ import annotations

import os
import sys
import textwrap
import threading
import types
from pathlib import Path

import pytest

pytest.importorskip("netqasm", reason="the NetQASM backend needs NetQASM")
try:
    import squidasm  # noqa: F401
except ImportError:
    os.environ.setdefault("NETQASM_SIMULATOR", "debug")

import netqasm.sdk.external  # noqa: E402
from netqasm.runtime.interface.config import (  # noqa: E402
    NoiseType, QuantumHardware, default_network_config,
)

from netqmpi.runtime.adapters.netqasm import (  # noqa: E402
    NetQASMCommunicator, NetQASMExecutorAdapter, NetQASMRunConfig, _simulation,
)
from netqmpi.runtime.adapters.netqasm._compat import INSTALLED_MAJOR  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
APPS = Path(__file__).resolve().parent / "apps"


class FakeSimulation:
    """
    Stands in for ``simulate_application`` and records what it was given.

    Attributes:
        calls: The keyword arguments of every call.
        epr_peers: For each call, the ranks each rank opened EPR sockets to.
        peaks: For each call, the most qubits each rank holds at once.
        error: Raised by the call when set.
    """

    def __init__(self, error: BaseException = None) -> None:
        self.calls = []
        self.epr_peers = []
        self.peaks = []
        self.error = error

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        self.epr_peers.append({
            comm.rank: sorted(int(s.remote_app_name.split("_")[1])
                              for s in comm._epr_sockets_list)
            for comm in NetQASMCommunicator.communicators
        })
        self.peaks.append({
            comm.rank: max(circuit.peak_qubits() for circuit in comm.circuits)
            for comm in NetQASMCommunicator.communicators
        })
        if self.error is not None:
            raise self.error
        return []


@pytest.fixture
def simulation(monkeypatch):
    """Replace SquidASM's entry point, and record calls to stop its backend."""
    fake = FakeSimulation()
    monkeypatch.setattr(netqasm.sdk.external, "simulate_application", fake,
                        raising=False)
    fake.stopped = 0

    def stop_backend():
        fake.stopped += 1

    monkeypatch.setattr(_simulation, "stop_backend", stop_backend)
    NetQASMCommunicator.reset_run()
    yield fake
    NetQASMCommunicator.reset_run()


def run(app: Path, ranks: int, monkeypatch, **config):
    """
    Run a NetQMPI app through the adapter, up to the simulator.

    Args:
        app: Path of the app to run.
        ranks: Number of ranks.
        monkeypatch: The pytest fixture, for the app's environment variables.
        **config: Fields of :class:`NetQASMRunConfig` to override.
    """
    settings = NetQASMRunConfig(**config)
    settings.netqasm_major = INSTALLED_MAJOR
    executor = NetQASMExecutorAdapter(ranks, settings)
    executor.run(executor.build_apps(str(app), ranks))


def load_app(name: str, monkeypatch, qubits: int) -> Path:
    """Point the benchmark apps at ``qubits`` per rank before they load."""
    monkeypatch.setenv("NQB_QUBITS_PER_RANK", str(qubits))
    return APPS / f"{name}.py"


# ----------------------------------------------------------------------
# N3 — network size, noise, capacity
# ----------------------------------------------------------------------

def test_built_network_matches_squidasms_default():
    """With the default config the network is SquidASM's own default."""
    names = ["rank_0", "rank_1", "rank_2"]
    built = _simulation.build_network(names, NetQASMRunConfig())
    reference = default_network_config(names, QuantumHardware.Generic)
    assert built == reference


def test_network_size_and_noise_come_from_the_config():
    """num_qubits and the noise fields reach every node and link."""
    config = NetQASMRunConfig(num_qubits=9, t1=1e6, t2=5e5, gate_fidelity=0.99,
                              link_fidelity=0.9, link_noise="Bitflip")
    network = _simulation.build_network(["rank_0", "rank_1"], config)
    for node in network.nodes:
        assert len(node.qubits) == 9
        assert {(q.t1, q.t2) for q in node.qubits} == {(1e6, 5e5)}
        assert node.gate_fidelity == 0.99
    assert {(link.noise_type, link.fidelity) for link in network.links} == {
        (NoiseType.Bitflip, 0.9)}


@pytest.mark.parametrize("field, value", [
    ("num_qubits", 0), ("t1", -1.0), ("gate_fidelity", 1.5),
    ("link_fidelity", -0.1), ("epr_setup_timeout", 0),
])
def test_nonsense_config_is_refused(field, value):
    with pytest.raises(ValueError, match=field):
        NetQASMRunConfig(**{field: value})


@pytest.mark.parametrize("name, qubits, peak", [
    # cascade: q data qubits, plus the EPR half while one of them moves.
    ("cascade", 1, 2), ("cascade", 2, 3), ("cascade", 6, 7),
    # ghz: q data qubits and the scratch slot, plus the EPR half.
    ("ghz", 1, 3), ("ghz", 2, 4), ("ghz", 4, 6),
])
def test_peak_qubits_follow_the_program(name, qubits, peak, simulation,
                                        monkeypatch):
    """The peak a rank holds is what the adapter really allocates."""
    run(load_app(name, monkeypatch, qubits), 3, monkeypatch, num_qubits=64)
    assert max(simulation.peaks[0].values()) == peak, simulation.peaks


def test_a_program_too_big_for_its_nodes_is_refused(simulation, monkeypatch):
    """cascade with 6 qubits needs 7 at once; the default node has 5."""
    with pytest.raises(ValueError, match=r"rank 0 needs 7 qubits at once.*has 5"):
        run(load_app("cascade", monkeypatch, 6), 2, monkeypatch)
    assert simulation.calls == []


def test_a_bigger_network_lets_it_run(simulation, monkeypatch):
    run(load_app("cascade", monkeypatch, 6), 2, monkeypatch, num_qubits=7)
    network = simulation.calls[0]["network_cfg"]
    assert {len(node.qubits) for node in network.nodes} == {7}


# ----------------------------------------------------------------------
# N2 / C1 — EPR sockets only towards peers
# ----------------------------------------------------------------------

def test_epr_sockets_follow_the_traced_peers(simulation, monkeypatch):
    """A chain opens EPR sockets to its neighbours, a star to its centre."""
    run(load_app("cascade", monkeypatch, 1), 5, monkeypatch)
    assert simulation.epr_peers[0] == {0: [1], 1: [0, 2], 2: [1, 3],
                                       3: [2, 4], 4: [3]}

    run(load_app("ghz", monkeypatch, 1), 5, monkeypatch)
    assert simulation.epr_peers[1] == {0: [1, 2, 3, 4], 1: [0], 2: [0],
                                       3: [0], 4: [0]}


# ----------------------------------------------------------------------
# N4 — logging
# ----------------------------------------------------------------------

def test_logging_is_off_by_default():
    assert NetQASMRunConfig().enable_logging is False


def test_concurrent_runs_get_their_own_log_directory(tmp_path, monkeypatch):
    """Two runs starting in the same second must not share a log dir."""
    monkeypatch.chdir(tmp_path)
    config = NetQASMRunConfig(enable_logging=True)
    first = _simulation.log_config(config)
    second = _simulation.log_config(config)
    assert first.log_dir != second.log_dir
    assert Path(first.log_dir).parent == tmp_path / "log"
    assert config.log_cfg is None          # the caller's config is untouched


def test_an_explicit_log_dir_is_respected(tmp_path):
    from netqasm.sdk.config import LogConfig
    config = NetQASMRunConfig(enable_logging=True,
                              log_cfg=LogConfig(log_dir=str(tmp_path)))
    assert _simulation.log_config(config).log_dir == str(tmp_path)


def test_logging_reaches_squidasm_only_when_asked(simulation, monkeypatch,
                                                  tmp_path):
    monkeypatch.chdir(tmp_path)
    run(load_app("cascade", monkeypatch, 1), 2, monkeypatch)
    assert simulation.calls[0]["enable_logging"] is False
    assert not (tmp_path / "log").exists()


# ----------------------------------------------------------------------
# C1 — SquidASM workarounds
# ----------------------------------------------------------------------

@pytest.fixture
def fake_squidasm(monkeypatch):
    """Install just enough of SquidASM for the patches to find."""
    seen = {}

    class NetworkStack:
        def setup_epr_socket(self, epr_socket_id, remote_node_id,
                             remote_epr_socket_id, timeout=5.0):
            seen["timeout"] = timeout
            yield "set up"

    def as_completed(futures, names=None, sleep_time=0):
        seen["sleep_time"] = sleep_time
        return iter(())

    netstack = types.ModuleType("squidasm.nqasm.netstack")
    netstack.NetworkStack = NetworkStack
    runtime_mgr = types.ModuleType("squidasm.run.multithread.runtime_mgr")
    runtime_mgr.as_completed = as_completed
    multithread = types.ModuleType("squidasm.run.multithread")
    multithread.runtime_mgr = runtime_mgr
    for name, module in {
        "squidasm": types.ModuleType("squidasm"),
        "squidasm.nqasm": types.ModuleType("squidasm.nqasm"),
        "squidasm.nqasm.netstack": netstack,
        "squidasm.run": types.ModuleType("squidasm.run"),
        "squidasm.run.multithread": multithread,
        "squidasm.run.multithread.runtime_mgr": runtime_mgr,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    return NetworkStack, runtime_mgr, seen


def test_epr_setup_timeout_is_raised_while_simulating(fake_squidasm):
    NetworkStack, runtime_mgr, seen = fake_squidasm
    original_setup = NetworkStack.setup_epr_socket
    original_as_completed = runtime_mgr.as_completed

    with _simulation.squidasm_patches(epr_setup_timeout=60.0, poll_interval=0.01):
        assert list(NetworkStack().setup_epr_socket(0, 1, 0)) == ["set up"]
        assert seen["timeout"] == 60.0
        # A caller asking for longer still gets it.
        list(NetworkStack().setup_epr_socket(0, 1, 0, timeout=120.0))
        assert seen["timeout"] == 120.0
        list(runtime_mgr.as_completed([]))
        assert seen["sleep_time"] == 0.01

    assert NetworkStack.setup_epr_socket is original_setup
    assert runtime_mgr.as_completed is original_as_completed


def test_the_configured_timeout_reaches_squidasm(simulation, fake_squidasm,
                                                 monkeypatch):
    NetworkStack, _, seen = fake_squidasm

    def simulate(**kwargs):
        list(NetworkStack().setup_epr_socket(0, 1, 0))
        return []

    monkeypatch.setattr(netqasm.sdk.external, "simulate_application", simulate,
                        raising=False)
    run(load_app("cascade", monkeypatch, 1), 2, monkeypatch,
        epr_setup_timeout=42.0)
    assert seen["timeout"] == 42.0


# ----------------------------------------------------------------------
# G1 — failures
# ----------------------------------------------------------------------

RAISES_WHILE_TRACING = """
    from netqmpi.sdk.environment import Environment

    def main(env: Environment = None):
        comm = env.comm
        with comm:
            circuit = env.create_circuit(num_qubits=1, num_clbits=1)
            if comm.rank == 1:
                raise NotImplementedError("rank 1 gave up")
            circuit.measure(0, 0)
"""


def test_a_rank_failing_while_tracing_fails_the_run(simulation, monkeypatch,
                                                    tmp_path):
    app = tmp_path / "app.py"
    app.write_text(textwrap.dedent(RAISES_WHILE_TRACING))
    with pytest.raises(NotImplementedError, match="rank 1 gave up"):
        run(app, 4, monkeypatch)
    assert simulation.calls == []
    assert NetQASMCommunicator.netqasm_circuits == []
    assert NetQASMCommunicator.communicators == []


@pytest.mark.parametrize("ranks", [2, 8])
def test_a_failed_simulation_stops_the_backend(simulation, monkeypatch, ranks):
    """A rank that raises inside SquidASM fails the run and frees the backend."""
    simulation.error = NotImplementedError("rank 1 gave up")
    with pytest.raises(NotImplementedError, match="rank 1 gave up"):
        run(load_app("cascade", monkeypatch, 1), ranks, monkeypatch)
    assert simulation.stopped == 1
    assert NetQASMCommunicator.netqasm_circuits == []
    assert NetQASMCommunicator.communicators == []

    # And the next run in the same process starts clean.
    simulation.error = None
    run(load_app("cascade", monkeypatch, 1), 2, monkeypatch)
    assert len(simulation.calls) == 2


def test_stop_backend_stops_whatever_squidasm_left_running(monkeypatch):
    stopped = threading.Event()

    class Backend:
        def stop_backend(self):
            stopped.set()

    glob = types.ModuleType("squidasm.sim.glob")
    glob.get_running_backend = lambda block=True: Backend()
    for name, module in {"squidasm": types.ModuleType("squidasm"),
                         "squidasm.sim": types.ModuleType("squidasm.sim"),
                         "squidasm.sim.glob": glob}.items():
        monkeypatch.setitem(sys.modules, name, module)

    from netqmpi.runtime.adapters.netqasm._simulation import stop_backend
    stop_backend()
    assert stopped.is_set()


# ----------------------------------------------------------------------
# Telegate windows: expose / unexpose
# ----------------------------------------------------------------------

FREQUENT_ERRORS = REPO_ROOT / "examples" / "frequent_errors"


def write_app(tmp_path: Path, source: str) -> Path:
    """Write an app module to ``tmp_path`` and return its path."""
    app = tmp_path / "app.py"
    app.write_text(textwrap.dedent(source))
    return app


#: Rank 0 lends its qubit to every other rank, each of which uses it as the
#: control of a local CNOT. Each receiver holds two live data qubits when
#: the window opens; the root holds one.
WIDE_WINDOW = """
    from netqmpi.sdk.environment import Environment

    def main(env: Environment = None):
        comm, rank, size = env.comm, env.comm.rank, env.comm.size
        receivers = list(range(1, size))
        with comm:
            circuit = env.create_circuit(num_qubits=2, num_clbits=2)
            if rank == 0:
                circuit.x(0)
            else:
                circuit.x(0)
                circuit.x(1)
            handle = comm.expose(circuit, 0, receivers, root=0)
            if rank != 0:
                circuit.cx(handle, 0)
                circuit.cx(handle, 1)
            comm.unexpose(circuit, receivers, root=0)
            circuit.measure(0, 0)
            circuit.measure(1, 1)
"""

#: Two windows over different groups, one open across the other: the shape
#: of ``examples/5_qft_expose.py``, with CNOTs instead of controlled phases.
NESTED_WINDOWS = """
    from netqmpi.sdk.environment import Environment

    def main(env: Environment = None):
        comm, rank = env.comm, env.comm.rank
        with comm:
            circuit = env.create_circuit(num_qubits=1, num_clbits=1)
            outer = comm.expose(circuit, 0, [0, 1], root=2)
            inner = comm.expose(circuit, 0, [0], root=1)
            if rank == 0:
                circuit.cx(inner, 0)
                circuit.cx(outer, 0)
            comm.unexpose(circuit, [0], root=1)
            if rank == 1:
                circuit.cx(outer, 0)
            comm.unexpose(circuit, [0, 1], root=2)
            circuit.measure(0, 0)
"""

#: ``examples/frequent_errors/partial_expose_group.py`` with a CNOT instead
#: of a controlled-S, which NetQASM cannot express and would report first.
PARTIAL_GROUP = """
    from netqmpi.sdk.environment import Environment

    def main(env: Environment = None):
        comm, rank = env.comm, env.comm.rank
        with comm:
            circuit = env.create_circuit(num_qubits=1, num_clbits=1)
            if rank != 2:
                control = comm.expose(circuit, 0, [1, 2], root=0)
                if rank == 1:
                    circuit.cx(control, 0)
                comm.unexpose(circuit, [1, 2], root=0)
            circuit.measure(0, 0)
"""


@pytest.mark.parametrize("ranks", [2, 3, 5])
def test_a_telegate_window_reaches_the_simulation(ranks, simulation,
                                                  monkeypatch, tmp_path):
    """expose and unexpose translate instead of raising NotImplementedError."""
    run(write_app(tmp_path, WIDE_WINDOW), ranks, monkeypatch)
    assert len(simulation.calls) == 1


def test_epr_sockets_follow_a_telegate_star(simulation, monkeypatch, tmp_path):
    """The root talks to every receiver; a receiver only to the root."""
    run(write_app(tmp_path, WIDE_WINDOW), 4, monkeypatch)
    assert simulation.epr_peers[0] == {0: [1, 2, 3], 1: [0], 2: [0], 3: [0]}


def test_epr_sockets_follow_nested_windows(simulation, monkeypatch, tmp_path):
    """Each window adds its own root-receiver pairs, and only those."""
    run(write_app(tmp_path, NESTED_WINDOWS), 3, monkeypatch)
    assert simulation.epr_peers[0] == {0: [1, 2], 1: [0, 2], 2: [0, 1]}


def test_peak_qubits_count_the_window(simulation, monkeypatch, tmp_path):
    """
    A receiver holds its comm qubit for the whole window, the root one EPR
    half at a time.

    The root holds its data qubit plus the EPR half it is measuring: 2. A
    receiver holds two data qubits plus the comm qubit: 3, not 2.
    """
    run(write_app(tmp_path, WIDE_WINDOW), 3, monkeypatch)
    assert simulation.peaks[0] == {0: 2, 1: 3, 2: 3}


def test_peak_qubits_of_nested_windows(simulation, monkeypatch, tmp_path):
    """Rank 0 receives two controls at once: its data qubit plus two comm."""
    run(write_app(tmp_path, NESTED_WINDOWS), 3, monkeypatch)
    # Rank 1 is a receiver of the outer window (comm qubit) and root of the
    # inner one (data qubit plus a transient EPR half).
    assert simulation.peaks[0] == {0: 3, 1: 3, 2: 2}


def test_a_window_that_does_not_fit_is_refused(simulation, monkeypatch,
                                               tmp_path):
    """The comm qubit counts: two live data qubits do not leave room for it."""
    with pytest.raises(ValueError,
                       match=r"rank 1 needs 3 qubits at once, node 'rank_1' has 2"):
        run(write_app(tmp_path, WIDE_WINDOW), 3, monkeypatch, num_qubits=2)
    assert simulation.calls == []


def test_a_lonely_expose_is_reported(simulation, monkeypatch):
    """Only the root reaches the window: a deadlock report, not a hang."""
    with pytest.raises(RuntimeError, match="Deadlock") as error:
        run(FREQUENT_ERRORS / "lonely_expose.py", 2, monkeypatch)
    message = str(error.value)
    assert "expose(ranks=[0], root=1)" in message
    assert "called by:  rank 1" in message
    assert "missing:    rank 0, which finished its block without calling it" in message
    assert simulation.calls == []
    assert NetQASMCommunicator.communicators == []


def test_crossed_windows_are_reported(simulation, monkeypatch):
    """Each rank waits on its own window: named as caller and as missing."""
    with pytest.raises(RuntimeError, match="Deadlock") as error:
        run(FREQUENT_ERRORS / "crossed_expose.py", 2, monkeypatch)
    message = str(error.value)
    assert "missing:    rank 1, which is waiting on expose(ranks=[0], root=1)" in message
    assert "missing:    rank 0, which is waiting on expose(ranks=[1], root=0)" in message
    assert "same order" in message
    assert simulation.calls == []


def test_a_partial_group_is_reported(simulation, monkeypatch, tmp_path):
    """Two of three participants arrive; the third is named as missing."""
    with pytest.raises(RuntimeError, match="Deadlock") as error:
        run(write_app(tmp_path, PARTIAL_GROUP), 3, monkeypatch)
    message = str(error.value)
    assert "called by:  rank 0, rank 1" in message
    assert "missing:    rank 2, which finished its block without calling it" in message
    assert simulation.calls == []


def test_a_window_naming_a_rank_outside_the_run_is_reported(simulation,
                                                            monkeypatch):
    """Rank 7 cannot answer in a 2-rank run; say so before opening sockets."""
    with pytest.raises(RuntimeError, match="names rank 7") as error:
        run(FREQUENT_ERRORS / "expose_outside_group.py", 2, monkeypatch)
    assert "this run has 2 ranks, numbered 0 to 1" in str(error.value)
    assert simulation.calls == []


def test_a_rejected_window_leaves_the_adapter_reusable(simulation, monkeypatch,
                                                       tmp_path):
    """After a refused run, the next one in the same process starts clean."""
    with pytest.raises(RuntimeError, match="Deadlock"):
        run(FREQUENT_ERRORS / "crossed_expose.py", 2, monkeypatch)
    run(write_app(tmp_path, NESTED_WINDOWS), 3, monkeypatch)
    assert len(simulation.calls) == 1


def test_the_partial_group_example_reports_its_gate_first(simulation,
                                                          monkeypatch):
    """
    Its controlled-S is refused while rank 1 translates, before the check.

    Documented in examples/frequent_errors/README.md as the one expose
    example whose NetQASM error differs from CUNQA's.
    """
    with pytest.raises(NotImplementedError,
                       match="Controlled-S is not implemented for the NetQASM"):
        run(FREQUENT_ERRORS / "partial_expose_group.py", 3, monkeypatch)
    assert simulation.calls == []


class _Comm:
    """Just the rank, for the adapter's error messages."""
    rank = 1


class _Socket:
    """Hands out one structured message."""

    def __init__(self, header, payload):
        from netqasm.sdk.classical_communication.message import StructuredMessage
        self.message = StructuredMessage(header, payload)

    def recv_structured(self):
        return self.message


def _bare_adapter():
    from netqmpi.runtime.adapters.netqasm import NetQASMCircuitAdapter
    adapter = NetQASMCircuitAdapter.__new__(NetQASMCircuitAdapter)
    adapter._comm = _Comm()
    adapter._num_qubits = 1
    adapter._qubits = [None, None]           # one data qubit, one comm slot
    return adapter


def test_a_telegate_message_for_another_call_is_refused():
    """A correction from a different call raises instead of being applied."""
    adapter = _bare_adapter()
    assert adapter._receive(_Socket("expose t", 1), "expose t", 0) == 1
    with pytest.raises(RuntimeError, match=r"rank 1 expected the 'expose t' "
                                           r"outcome from rank 0 but received "
                                           r"'Corrections'"):
        adapter._receive(_Socket("Corrections", [0, 1]), "expose t", 0)


def test_an_empty_comm_qubit_is_not_silently_reallocated():
    """A comm slot with no control raises rather than becoming a fresh |0>."""
    adapter = _bare_adapter()
    with pytest.raises(RuntimeError,
                       match=r"communication qubit 1 \(slot 0\) holds no "
                             r"exposed control"):
        adapter._qubit(1)
