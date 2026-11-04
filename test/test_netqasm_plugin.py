"""
The ``plugin`` hook of the NetQASM backend, its example plugins, and
``formalism`` read by name.

A plugin is how a program brings hardware or links SquidASM cannot describe
on its own: NetQMPI imports it, hands it ``plugin_config`` and keeps the
context manager it returns open for the length of the simulation.

Installed plugins are selected by short name through entry points; the
``install_plugins`` fixture of ``conftest.py`` installs a dummy distribution
that has every optional member of the contract (``run_defaults``,
``configure``, ``backends``) to exercise them here against the real
``NetQASMRunConfig`` and command line.
"""
from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path

import pytest

pytest.importorskip("squidasm", reason="the NetQASM backend needs SquidASM")
pytest.importorskip("netsquid", reason="the NetQASM backend needs NetSquid")

from netqasm.runtime.settings import Formalism  # noqa: E402

from netqmpi.runtime.adapters.netqasm import (  # noqa: E402
    NetQASMExecutorAdapter, NetQASMRunConfig,
)
from netqmpi.runtime.adapters.netqasm import _simulation  # noqa: E402
from netqmpi.runtime.adapters.netqasm._compat import INSTALLED_MAJOR  # noqa: E402
from netqmpi.runtime import cli  # noqa: E402
from netqmpi.runtime.run_config import read_config_block  # noqa: E402

#: Records what SquidASM asks of it while the plugin is active.
PLUGIN = """
    import contextlib

    from squidasm.sim.network import network

    CALLS = []

    @contextlib.contextmanager
    def wrap(config):
        CALLS.append(("enter", dict(config)))
        original = network.QDevice

        class Recording(original):
            def __init__(self, *args, num_qubits, **kwargs):
                CALLS.append(("qdevice", num_qubits))
                super().__init__(*args, num_qubits=num_qubits, **kwargs)

        network.QDevice = Recording
        try:
            yield
        finally:
            network.QDevice = original
            CALLS.append("exit")
"""

APP = """
    def main(env):
        comm = env.comm
        with comm:
            circuit = env.create_circuit(num_qubits=1, num_clbits=1)
            circuit.x(0)
            circuit.measure(0, 0)
"""


def test_the_formalism_is_read_by_name(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("netqasm:\n  formalism: dm\n")

    config = NetQASMRunConfig.from_dict(read_config_block(str(path), "netqasm"))

    assert config.formalism is Formalism.DM


def test_an_unknown_formalism_is_refused():
    with pytest.raises(ValueError, match="formalism"):
        NetQASMRunConfig(formalism="rho")


@pytest.mark.parametrize("spec", [":wrap", "module:"])
def test_a_malformed_plugin_is_refused(spec):
    with pytest.raises(ValueError, match="module:callable"):
        NetQASMRunConfig(plugin=spec)


def test_a_name_without_a_colon_is_looked_up_among_installed_plugins():
    with pytest.raises(ValueError, match="Unknown plugin 'module_only'"):
        NetQASMRunConfig(plugin="module_only")


def test_a_plugin_that_does_not_import_is_refused(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="Cannot import"):
        NetQASMRunConfig(plugin="no_such_plugin:wrap")


def test_plugin_config_must_be_a_mapping(tmp_path, monkeypatch):
    (tmp_path / "noop_plugin.py").write_text(
        "import contextlib\nwrap = lambda config: contextlib.nullcontext()\n")
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="mapping"):
        NetQASMRunConfig(plugin="noop_plugin:wrap", plugin_config=[0.1, 0.2])


def test_plugin_config_needs_a_plugin():
    with pytest.raises(ValueError, match="no plugin"):
        NetQASMRunConfig(plugin_config={"anything": 1})


@pytest.mark.integration
def test_the_plugin_is_active_during_the_simulation(tmp_path, monkeypatch):
    # The plugin is found in the working directory, as `python -m` would.
    (tmp_path / "recording_plugin.py").write_text(textwrap.dedent(PLUGIN))
    (tmp_path / "app.py").write_text(textwrap.dedent(APP))
    monkeypatch.chdir(tmp_path)

    config = NetQASMRunConfig(shots=2, num_qubits=3, netqasm_major=INSTALLED_MAJOR,
                              plugin="recording_plugin:wrap",
                              plugin_config={"device": "anything"})
    executor = NetQASMExecutorAdapter(1, config)
    executor.run(executor.build_apps(str(tmp_path / "app.py"), 1))

    import recording_plugin
    assert recording_plugin.CALLS == [("enter", {"device": "anything"}),
                                      ("qdevice", 3), "exit"]
    # It yields None, so it reports nothing; NetQMPI's own metrics remain.
    metrics = executor.metrics()
    assert "plugin" not in metrics
    assert len(metrics["netqmpi"]["sim_time_ns"]) == 2


# ---------------------------------------------------------------------------
# Metrics: what a plugin reports, and the simulated time of each shot
# ---------------------------------------------------------------------------

#: Counts the pairs every link delivers, and reports them either through
#: report() or as a mapping it fills, as ``plugin_config["style"]`` says.
#: "fail" breaks the simulation instead, to show nothing is reported then.
COUNTING_PLUGIN = """
    import contextlib

    from squidasm.sim.network import network

    STATS = []

    class LinkStats:
        def __init__(self):
            self.deliveries = 0
            self.reports = 0

        def report(self):
            self.reports += 1
            return {"epr_pairs": self.deliveries}

    def counting(config):
        return _counting(config.get("style", "report"))

    @contextlib.contextmanager
    def _counting(style):
        stats = LinkStats()
        STATS.append(stats)
        filled = {"epr_pairs": 0}
        original = network.NetSquidNetwork.__dict__["_create_link_distributor"]

        def create(self, link, state_delay=1000):
            if style == "fail":
                raise RuntimeError("link down")
            distributor = original(self, link, state_delay)
            add_delivery = distributor.add_delivery

            def counted(*args, **kwargs):
                stats.deliveries += 1
                filled["epr_pairs"] += 1
                return add_delivery(*args, **kwargs)

            distributor.add_delivery = counted
            return distributor

        network.NetSquidNetwork._create_link_distributor = create
        try:
            yield filled if style == "mapping" else stats
        finally:
            network.NetSquidNetwork._create_link_distributor = original
"""

#: Teleport |1> from rank 0 to rank 1: one pair per shot.
TELEPORT_ONE = """
    def main(env):
        comm = env.comm
        with comm:
            circuit = env.create_circuit(num_qubits=1, num_clbits=1)
            if comm.rank == 0:
                circuit.x(0)
                comm.qsend(circuit, [0], 1)
            else:
                comm.qrecv(circuit, [0], 0)
                circuit.measure(0, 0)
"""


class _Report:
    def __init__(self, report):
        self._report = report

    def report(self):
        return self._report()


def test_without_a_plugin_nothing_is_yielded():
    with _simulation.plugin_context(NetQASMRunConfig()) as yielded:
        assert yielded is None


def test_a_plugin_yielding_something_else_is_refused_before_simulating(
        tmp_path, monkeypatch):
    (tmp_path / "odd_plugin.py").write_text(
        "import contextlib\nwrap = lambda config: contextlib.nullcontext('stats')\n")
    monkeypatch.chdir(tmp_path)
    config = NetQASMRunConfig(plugin="odd_plugin:wrap")
    with pytest.raises(TypeError, match=r"'odd_plugin:wrap' yielded a str"):
        with _simulation.plugin_context(config):
            pytest.fail("the simulation must not start")


def test_plugin_metrics_copies_a_mapping_and_calls_report():
    filled = {"epr_pairs": 3}
    assert _simulation.plugin_metrics("p:q", None) is None
    copied = _simulation.plugin_metrics("p:q", filled)
    assert copied == filled and copied is not filled
    assert _simulation.plugin_metrics("p:q", _Report(lambda: {"a": 1})) == {"a": 1}


def test_an_error_in_report_names_the_plugin():
    def broken():
        raise KeyError("pair_rate")

    with pytest.raises(RuntimeError, match=r"report\(\) of plugin 'p:q' raised "
                                           r"KeyError") as error:
        _simulation.plugin_metrics("p:q", _Report(broken))
    assert isinstance(error.value.__cause__, KeyError)


def test_a_report_that_is_not_a_mapping_is_refused():
    with pytest.raises(TypeError, match=r"report\(\) of plugin 'p:q' returned a list"):
        _simulation.plugin_metrics("p:q", _Report(lambda: [1, 2]))


def test_a_shot_lasts_from_the_last_rank_ready_to_the_last_one_done():
    rank_0 = [(100.0, 1100.0), (2000.0, 2500.0)]
    rank_1 = [(150.0, 1300.0), (1900.0, 2400.0)]
    assert _simulation.shot_times([rank_0, rank_1]) == [1150.0, 500.0]
    # A program with no complete record (a shot that failed) gives no times.
    assert _simulation.shot_times([rank_0, rank_1[:1]]) == []
    assert _simulation.shot_times([]) == []


def _run_counting(tmp_path, monkeypatch, style, shots=3):
    """Run TELEPORT_ONE on two ranks with the counting plugin."""
    (tmp_path / "counting_plugin.py").write_text(textwrap.dedent(COUNTING_PLUGIN))
    (tmp_path / "app.py").write_text(textwrap.dedent(TELEPORT_ONE))
    monkeypatch.chdir(tmp_path)

    from netqmpi.sdk.environment import Environment
    comms = []
    original = Environment.__init__

    def capture(self, comm, executor):
        original(self, comm, executor)
        comms.append(comm)

    monkeypatch.setattr(Environment, "__init__", capture)
    config = NetQASMRunConfig(shots=shots, netqasm_major=INSTALLED_MAJOR,
                              plugin="counting_plugin:counting",
                              plugin_config={"style": style})
    executor = NetQASMExecutorAdapter(2, config)
    executor.run(executor.build_apps(str(tmp_path / "app.py"), 2))
    import counting_plugin
    return executor, comms, counting_plugin.STATS[-1]


@pytest.mark.integration
@pytest.mark.parametrize("style", ["report", "mapping"])
def test_what_the_plugin_yields_becomes_its_metrics(tmp_path, monkeypatch, style):
    executor, comms, stats = _run_counting(tmp_path, monkeypatch, style)

    last = comms[-1]
    assert last.results == {"1": 3}             # the outcomes are untouched
    assert last.metrics["plugin"] == {"epr_pairs": 3}
    assert stats.reports == (1 if style == "report" else 0)
    # The same metrics on every rank, and for the caller of run().
    assert all(comm.metrics == last.metrics for comm in comms)
    assert executor.metrics() == last.metrics

    # One simulated time per shot, each at least the 1 us SquidASM's links
    # take to deliver a pair.
    times = last.metrics["netqmpi"]["sim_time_ns"]
    assert len(times) == 3 and all(t >= 1000.0 for t in times), times


@pytest.mark.integration
def test_a_failed_simulation_reports_its_error_and_no_metrics(tmp_path, monkeypatch):
    with pytest.raises(RuntimeError, match="link down"):
        _run_counting(tmp_path, monkeypatch, "fail")

    import counting_plugin
    assert counting_plugin.STATS[-1].reports == 0


@pytest.mark.integration
def test_the_metrics_flag_writes_the_plugins_report(tmp_path, monkeypatch):
    import json

    from netqmpi.runtime import cli

    (tmp_path / "counting_plugin.py").write_text(textwrap.dedent(COUNTING_PLUGIN))
    (tmp_path / "app.py").write_text(textwrap.dedent(TELEPORT_ONE))
    (tmp_path / "config.yaml").write_text(
        "shots: 2\nnetqasm:\n  plugin: counting_plugin:counting\n")
    monkeypatch.chdir(tmp_path)
    flag = "--netqasm" if INSTALLED_MAJOR >= 2 else "--netqasm1.0"
    monkeypatch.setattr("sys.argv", ["netqmpi", "-n", "2", "app.py", flag,
                                     "--config", "config.yaml",
                                     "--metrics", "metrics.json"])

    cli.main()

    metrics = json.loads((tmp_path / "metrics.json").read_text())
    assert metrics["plugin"] == {"epr_pairs": 2}
    assert len(metrics["netqmpi"]["sim_time_ns"]) == 2


# ---------------------------------------------------------------------------
# Installed plugins: short names, run_defaults, configure, backends
# ---------------------------------------------------------------------------

def _args(config=None, shots=None, plugin=None, plugin_config=None):
    return argparse.Namespace(config=config, shots=shots, plugin=plugin,
                              plugin_config=plugin_config)


def test_a_short_name_in_the_config_file_selects_the_installed_plugin(
        install_plugins, tmp_path, monkeypatch):
    install_plugins()
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "config.yaml"
    path.write_text("netqasm:\n  plugin: dummy_plain\n")

    config = NetQASMRunConfig.from_dict(read_config_block(str(path), "netqasm"))

    assert config.plugin == "dummy_plain"


def test_run_defaults_apply_where_nothing_else_sets_the_key(install_plugins):
    install_plugins()
    config = NetQASMRunConfig.from_dict({"plugin": "dummy_device"})
    assert (config.num_qubits, config.shots) == (3, 7)
    assert config.formalism is Formalism.KET, "not in run_defaults: dataclass default"


def test_the_config_file_wins_over_run_defaults(install_plugins, tmp_path):
    # Top-level (generic) keys count as the config file too.
    install_plugins()
    path = tmp_path / "config.yaml"
    path.write_text("shots: 9\nnetqasm:\n  plugin: dummy_device\n"
                    "  num_qubits: 4\n")

    config = cli._build_config(NetQASMRunConfig, "netqasm",
                               _args(config=str(path)))

    assert (config.num_qubits, config.shots) == (4, 9)


def test_the_command_line_wins_over_the_config_file_and_run_defaults(
        install_plugins, tmp_path):
    install_plugins()
    path = tmp_path / "config.yaml"
    path.write_text("shots: 9\n")

    config = cli._build_config(
        NetQASMRunConfig, "netqasm",
        _args(config=str(path), shots=11, plugin="dummy_device"))

    assert (config.num_qubits, config.shots) == (3, 11)


def test_run_defaults_with_a_typo_blame_the_plugin(install_plugins):
    install_plugins()
    with pytest.raises(ValueError, match="dummy_typo.*num_qbits"):
        NetQASMRunConfig.from_dict({"plugin": "dummy_typo"})


def test_configure_gets_the_final_config_and_cannot_change_it(install_plugins,
                                                              tmp_path):
    module = install_plugins()
    params = tmp_path / "params.yaml"
    params.write_text("a: 1\n")

    config = cli._build_config(
        NetQASMRunConfig, "netqasm",
        _args(shots=11, plugin="dummy_device", plugin_config=str(params)),
        netqasm_major=INSTALLED_MAJOR)

    assert sys.modules[module].CALLS == [("configure", {"a": 1}, 3, 11)]
    assert config.num_qubits == 3 and config.plugin_config == {"a": 1}


def test_a_plugin_for_another_backend_is_refused(install_plugins):
    install_plugins()
    with pytest.raises(ValueError, match="dummy_qoala.*netqasm"):
        NetQASMRunConfig(plugin="dummy_qoala")


def _netqasm_flag():
    return "--netqasm" if INSTALLED_MAJOR == 2 else "--netqasm1.0"


def test_configure_errors_surface_before_any_rank_is_traced(
        install_plugins, tmp_path, monkeypatch, capsys):
    install_plugins()
    (tmp_path / "app.py").write_text(textwrap.dedent(APP))
    params = tmp_path / "params.yaml"
    params.write_text("bad: true\n")
    simulated = []
    monkeypatch.setattr(cli, "simulate", lambda **kwargs: simulated.append(kwargs))
    monkeypatch.setattr("sys.argv", [
        "netqmpi", "-n", "1", str(tmp_path / "app.py"), _netqasm_flag(),
        "--plugin", "dummy_device", "--plugin-config", str(params)])

    with pytest.raises(SystemExit) as exit_info:
        cli.main()

    assert exit_info.value.code == 2
    assert "dummy_device: 'bad' is not a parameter" in capsys.readouterr().err
    assert simulated == [], "the program was handed on to be traced"


@pytest.mark.integration
def test_an_installed_plugin_runs_by_short_name_from_the_command_line(
        install_plugins, tmp_path, monkeypatch):
    module = install_plugins()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "app.py").write_text(textwrap.dedent(APP))
    params = tmp_path / "params.yaml"
    params.write_text("device: anything\n")
    monkeypatch.chdir(elsewhere)
    monkeypatch.setattr("sys.argv", [
        "netqmpi", "-n", "1", "app.py", _netqasm_flag(), "--shots", "2",
        "--plugin", "dummy_device", "--plugin-config", str(params)])

    cli.main()

    assert sys.modules[module].CALLS == [
        ("configure", {"device": "anything"}, 3, 2),
        ("enter", {"device": "anything"}),
    ]


# ---------------------------------------------------------------------------
# The example plugins of examples/plugins/
# ---------------------------------------------------------------------------

EXAMPLES = Path(__file__).resolve().parents[1] / "examples" / "plugins"


def _run_example(name, plugin_config, shots, capsys):
    """Run an example's app.py with its plugin, on one rank, and return its output."""
    config = NetQASMRunConfig(shots=shots, netqasm_major=INSTALLED_MAJOR,
                              plugin=f"{name}:{name}", plugin_config=plugin_config)
    executor = NetQASMExecutorAdapter(1, config)
    executor.run(executor.build_apps(str(EXAMPLES / name / "app.py"), 1))
    return capsys.readouterr().out


@pytest.fixture
def in_example(monkeypatch):
    """
    Run from an example's directory and return its plugin module.

    The module is loaded the way NetQMPI loads it, from the working
    directory, so the tests exercise the same path as a real run.
    """
    def enter(name):
        monkeypatch.chdir(EXAMPLES / name)
        _simulation.load_plugin(f"{name}:{name}")
        return sys.modules[name]
    return enter


def test_readout_error_refuses_an_unknown_key(in_example):
    readout_error = in_example("readout_error")
    with pytest.raises(ValueError, match="unknown keys"):
        readout_error.readout_error({"prob_error": 0.1})


def test_readout_error_refuses_a_typo_when_the_config_is_built(in_example):
    # Through its `configure`, before any rank is traced.
    in_example("readout_error")
    with pytest.raises(ValueError, match="unknown keys"):
        NetQASMRunConfig(plugin="readout_error:readout_error",
                         plugin_config={"prob_error": 0.1})


def test_readout_error_refuses_a_value_that_is_not_a_probability(in_example):
    readout_error = in_example("readout_error")
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        readout_error.readout_error({"prob_error_1": 1.5})


def test_readout_error_restores_squidasm_on_exit(in_example):
    from squidasm.nqasm.qnodeos import SubroutineHandler
    readout_error = in_example("readout_error")
    before = SubroutineHandler.__dict__["_get_executor_class"]
    with readout_error.readout_error({"prob_error_0": 0.5}):
        assert SubroutineHandler.__dict__["_get_executor_class"] is not before
    assert SubroutineHandler.__dict__["_get_executor_class"] is before


@pytest.mark.integration
def test_readout_error_flips_every_outcome_at_probability_one(in_example, capsys):
    # The app measures |0> on qubit 0 and |1> on qubit 1, so without error
    # every shot reads "10"; with both probabilities at 1, every shot "01".
    in_example("readout_error")
    output = _run_example("readout_error",
                          {"prob_error_0": 1.0, "prob_error_1": 1.0}, 4, capsys)
    assert "rank 0: {'01': 4}" in output, output
    # Its metrics: two measurements a shot, all of them misread.
    assert "plugin: 8 of 8 measurements misread" in output, output


def test_gate_noise_refuses_an_unknown_key(in_example):
    gate_noise = in_example("gate_noise")
    with pytest.raises(ValueError, match="unknown keys"):
        gate_noise.gate_noise({"cnot_depolar_prob": 0.1})


def test_gate_noise_refuses_a_typo_when_the_config_is_built(in_example):
    in_example("gate_noise")
    with pytest.raises(ValueError, match="unknown keys"):
        NetQASMRunConfig(plugin="gate_noise:gate_noise",
                         plugin_config={"cnot_depolar_prob": 0.1})


@pytest.mark.parametrize("config", [{"two_qubit_depolar_prob": -0.1},
                                    {"single_qubit_depolar_prob": 2},
                                    {"two_qubit_duration": -1}])
def test_gate_noise_refuses_an_invalid_value(in_example, config):
    gate_noise = in_example("gate_noise")
    with pytest.raises(ValueError, match="gate_noise"):
        gate_noise.gate_noise(config)


def test_gate_noise_builds_the_processor_it_describes(in_example):
    from netsquid.components import instructions as instr
    gate_noise = in_example("gate_noise")
    params = gate_noise.GateNoise.from_dict({"two_qubit_depolar_prob": 0.2,
                                             "two_qubit_duration": 300.0,
                                             "single_qubit_duration": 50.0})
    qdevice = gate_noise.build_qdevice(params, name="node", num_qubits=3,
                                       gate_fidelity=0.5)   # ignored by design

    assert qdevice.num_positions == 3
    gates = {g.instruction: g for g in qdevice.get_physical_instructions()}
    assert gates[instr.INSTR_CNOT].duration == 300.0
    assert gates[instr.INSTR_H].duration == 50.0
    assert gates[instr.INSTR_CNOT].quantum_noise_model.depolar_rate == 0.2
    assert gates[instr.INSTR_H].quantum_noise_model is None   # none asked for


@pytest.mark.integration
def test_gate_noise_without_noise_keeps_the_bell_pair_exact(in_example, capsys):
    in_example("gate_noise")
    output = _run_example("gate_noise", {}, 10, capsys)
    assert "disagree in 0.0% of 10 shots" in output, output


@pytest.mark.integration
def test_gate_noise_on_the_cnot_breaks_the_correlation(in_example, capsys):
    # A CNOT that fully depolarises both qubits makes the two bits disagree
    # in half of the shots; none disagreeing in 40 has probability 2^-40.
    in_example("gate_noise")
    output = _run_example("gate_noise", {"two_qubit_depolar_prob": 1.0}, 40, capsys)
    assert "disagree in 0.0%" not in output and "disagree in" in output, output
