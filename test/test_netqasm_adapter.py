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
APPS = REPO_ROOT / "scripts" / "benchmark" / "apps"


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
