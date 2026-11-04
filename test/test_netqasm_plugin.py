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
