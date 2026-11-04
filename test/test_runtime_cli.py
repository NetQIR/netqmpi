"""
The command line and the script loader behind it.

``netqmpi -n 3 app.py --aer`` does three things before any quantum work
starts: it picks a backend adapter, builds that backend's typed config out
of ``--config`` plus ``--shots``, and loads ``main()`` out of the user's
script. All three are ordinary Python and worth testing as such — a wrong
backend flag or a silently ignored ``--shots`` is a class of bug no
simulator will ever catch for you.

Nothing here runs a simulation: the executor is replaced by a double that
records what it was handed, and :func:`~netqmpi.runtime.cli.simulate` is
intercepted where the argument parsing hands over to it.
"""
from __future__ import annotations

import argparse
import textwrap
from dataclasses import dataclass
from typing import Any, Dict, Optional

import pytest

from netqmpi.helpers import load_main
from netqmpi.runtime import cli
from netqmpi.runtime.run_config import RunConfig

from conftest import StubExecutor

APP = """
    from netqmpi.sdk.environment import Environment

    def main(env: Environment = None):
        return env
"""


def write_app(tmp_path, source: str = APP, name: str = "app.py") -> str:
    """Write a NetQMPI script and return its path."""
    path = tmp_path / name
    path.write_text(textwrap.dedent(source))
    return str(path)


# ----------------------------------------------------------------------
# load_main
# ----------------------------------------------------------------------

def test_the_entry_point_is_loaded_out_of_the_script(tmp_path):
    """A NetQMPI program is a module with a ``main(env)`` in it."""
    main = load_main(write_app(tmp_path))
    assert callable(main)
    assert main(env="sentinel") == "sentinel"


def test_the_script_runs_when_it_is_loaded(tmp_path):
    """
    Module-level code executes at load time, before any rank starts.

    Worth knowing when a program does its setup at import: it happens once
    per rank, because every rank loads the file for itself.
    """
    path = write_app(tmp_path, """
        MARKER = []
        MARKER.append('imported')

        def main(env=None):
            return MARKER
    """)
    assert load_main(path)() == ["imported"]


def test_a_script_without_main_is_named(tmp_path):
    """The most common first mistake, reported before anything runs."""
    path = write_app(tmp_path, "def run(env=None):\n    pass\n")
    with pytest.raises(ValueError, match="does not define a main"):
        load_main(path)


def test_a_main_that_is_not_a_function_is_refused(tmp_path):
    """``main = None`` would otherwise fail much deeper, on the call."""
    path = write_app(tmp_path, "main = None\n")
    with pytest.raises(ValueError, match="main function not found"):
        load_main(path)


@pytest.mark.parametrize("path, match", [
    (None, "script must be provided"),
    ("app.txt", "must be a .py script"),
    ("", "must be a .py script"),
])
def test_load_main_validates_the_path(path, match):
    """Checked before the file is even opened."""
    with pytest.raises(ValueError, match=match):
        load_main(path)


# ----------------------------------------------------------------------
# _build_config
# ----------------------------------------------------------------------

def args(config=None, shots=None, plugin=None, plugin_config=None):
    """The slice of the parsed CLI arguments the config builder reads."""
    return argparse.Namespace(config=config, shots=shots, plugin=plugin,
                              plugin_config=plugin_config)


def test_without_a_config_file_the_defaults_apply():
    """``netqmpi -n 2 app.py --aer`` is a valid, fully defaulted run."""
    config = cli._build_config(RunConfig, "aer", args())
    assert config.shots == 1024


def test_the_config_file_fills_the_backends_block(tmp_path):
    """The generic block and the backend's own are merged, then typed."""
    path = tmp_path / "config.yaml"
    path.write_text("shots: 32\naer:\n  shots: 8\n")

    assert cli._build_config(RunConfig, "aer", args(config=str(path))).shots == 8
    assert cli._build_config(RunConfig, "qoala", args(config=str(path))).shots == 32


def test_shots_on_the_command_line_wins_over_the_file(tmp_path):
    """A quick run must not need the file edited."""
    path = tmp_path / "config.yaml"
    path.write_text("shots: 32\n")

    config = cli._build_config(RunConfig, "aer", args(config=str(path), shots=5))
    assert config.shots == 5


def test_an_unset_shots_flag_leaves_the_file_alone():
    """``--shots`` is an override, not a default that clobbers the config."""
    assert cli._build_config(RunConfig, "aer", args(shots=None)).shots == 1024


# ----------------------------------------------------------------------
# simulate
# ----------------------------------------------------------------------

def test_simulate_builds_the_apps_and_runs_them(tmp_path):
    """The two-step contract every executor implements."""
    executor = StubExecutor(size=3)
    script = write_app(tmp_path)

    cli.simulate(script=script, num_procs=3, executor=executor)

    assert executor.built == [(script, 3)]
    assert len(executor.ran) == 1
    assert len(executor.ran[0]) == 3, "one app per rank"


def test_the_timer_reports_the_wall_clock(tmp_path, capsys):
    """Off by default, and a single line when asked for."""
    executor = StubExecutor(size=1)
    cli.simulate(write_app(tmp_path), 1, executor=executor, timer=True)
    assert "finished simulation in" in capsys.readouterr().out

    cli.simulate(write_app(tmp_path), 1, executor=StubExecutor(size=1))
    assert capsys.readouterr().out == ""


def test_simulate_without_an_executor_is_currently_broken(tmp_path):
    """
    The documented default backend is unreachable.

    :func:`~netqmpi.runtime.cli.simulate` falls back to
    ``NetQASMExecutorAdapter`` when no executor is given, but ``cli.py``
    never imports that name — the CLI's own ``--netqasm`` branch imports it
    locally, inside :func:`~netqmpi.runtime.cli.main`. Calling
    ``simulate()`` from Python with no executor therefore raises
    ``NameError`` instead of running anything.

    This test pins the current behaviour rather than the intended one, so
    that it is visible instead of being discovered from an unhelpful
    traceback. Fixing the import is what should make it change.
    """
    with pytest.raises(NameError, match="NetQASMExecutorAdapter"):
        cli.simulate(write_app(tmp_path), num_procs=1)


# ----------------------------------------------------------------------
# Argument parsing and backend selection
# ----------------------------------------------------------------------

@dataclass
class FakeNetQASMConfig(RunConfig):
    """The fields of ``NetQASMRunConfig`` the CLI itself sets."""
    netqasm_major: int = 2
    plugin: Optional[str] = None
    plugin_config: Optional[Dict[str, Any]] = None


class FakeExecutor:
    """Stands in for a backend adapter, so no simulator has to be installed."""

    def __init__(self, size, config=None):
        self.size = size
        self.config = config


#: What each backend flag must select, and the block of the config file it
#: reads. Keeping all four in one table is the point: the flags are the
#: whole user-facing difference between the backends, and three of them
#: cannot be installed side by side in one environment.
BACKENDS = {
    "netqasm": ("--netqasm", "NetQASMExecutorAdapter", "NetQASMRunConfig"),
    "cunqa": ("--cunqa", "CunqaExecutorAdapter", "CunqaRunConfig"),
    "aer": ("--aer", "AerExecutorAdapter", "AerSimulatorConfig"),
    "qoala": ("--qoala", "QoalaExecutorAdapter", "QoalaRunConfig"),
}


@pytest.fixture
def fake_backends(monkeypatch):
    """
    Put a stub adapter package in place of each of the four backends.

    The CLI imports its adapter *inside* the branch that selected it, so a
    stub module registered under that name is what the branch picks up.
    This keeps the selection logic testable in an environment where — as
    is normal here — NetQASM, CUNQA, Aer and Qoala cannot all be installed
    at once.
    """
    import sys
    import types

    for backend, (_, executor_name, config_name) in BACKENDS.items():
        module = types.ModuleType(f"netqmpi.runtime.adapters.{backend}")
        setattr(module, executor_name, FakeExecutor)
        setattr(module, config_name,
                FakeNetQASMConfig if backend == "netqasm" else RunConfig)
        monkeypatch.setitem(sys.modules, f"netqmpi.runtime.adapters.{backend}", module)


@pytest.fixture
def captured_run(monkeypatch):
    """
    Run :func:`~netqmpi.runtime.cli.main` without simulating anything.

    Returns:
        ``invoke(*argv)``, which returns the keyword arguments
        :func:`~netqmpi.runtime.cli.simulate` was called with.
    """
    calls = []
    monkeypatch.setattr(cli, "simulate", lambda **kwargs: calls.append(kwargs))

    def invoke(*argv):
        monkeypatch.setattr("sys.argv", ["netqmpi", *argv])
        cli.main()
        return calls[-1]

    return invoke


@pytest.mark.parametrize("backend", sorted(BACKENDS))
def test_each_flag_selects_its_own_backend(backend, tmp_path, fake_backends,
                                           captured_run):
    """One flag, one adapter, and the script and rank count passed through."""
    flag = BACKENDS[backend][0]
    script = write_app(tmp_path)

    call = captured_run("-n", "3", script, flag)

    assert isinstance(call["executor"], FakeExecutor)
    assert call["executor"].size == 3
    assert call["script"] == script and call["num_procs"] == 3


def test_no_flag_falls_back_to_netqasm(tmp_path, fake_backends, captured_run,
                                       capsys):
    """The default is announced rather than assumed silently."""
    captured_run("-n", "2", write_app(tmp_path))
    assert "No backend flag; using default (NetQASM)" in capsys.readouterr().out


def test_the_netqasm_release_flags_drive_the_same_adapter(tmp_path,
                                                          fake_backends,
                                                          captured_run):
    """
    ``--netqasm`` and ``--netqasm1.0`` differ only in the release they ask for.

    NetQASM 1.x and 2.x cannot share an environment, so the flag does not
    pick an adapter — it states which installation the run expects, and the
    executor refuses to start if that is not what is there.
    """
    script = write_app(tmp_path)

    two = captured_run("-n", "2", script, "--netqasm")["executor"]
    one = captured_run("-n", "2", script, "--netqasm1.0")["executor"]

    assert two.config.netqasm_major == 2
    assert one.config.netqasm_major == 1


def test_shots_and_config_reach_the_selected_backend(tmp_path, fake_backends,
                                                     captured_run):
    """``--config`` and ``--shots`` combine exactly as ``_build_config`` says."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text("shots: 12\naer:\n  shots: 8\n")

    from_file = captured_run("-n", "2", write_app(tmp_path), "--aer",
                             "--config", str(config_file))
    assert from_file["executor"].config.shots == 8

    overridden = captured_run("-n", "2", write_app(tmp_path), "--aer",
                              "--config", str(config_file), "--shots", "64")
    assert overridden["executor"].config.shots == 64


def test_the_real_aer_adapter_is_what_the_flag_selects(tmp_path, captured_run):
    """
    The same check once more, against the genuine adapter.

    The table above proves the branching; this proves the names in it are
    the ones the package actually exports.
    """
    pytest.importorskip("qiskit", reason="the Aer backend needs Qiskit")
    from netqmpi.runtime.adapters.aer import AerExecutorAdapter, AerSimulatorConfig

    call = captured_run("-n", "2", write_app(tmp_path), "--aer", "--shots", "64")

    assert isinstance(call["executor"], AerExecutorAdapter)
    assert isinstance(call["executor"].config, AerSimulatorConfig)
    assert call["executor"].config.shots == 64


@pytest.mark.parametrize("argv", [
    ("app.py",),                                # no -n
    ("-n", "0", "app.py"),                      # no ranks to run
    ("-n", "-2", "app.py"),
    ("-n", "2"),                                # no script
    ("-n", "2", "app.py", "--aer", "--cunqa"),  # two backends at once
    ("-n", "2", "app.py", "--netqasm", "--netqasm1.0"),
])
def test_unusable_command_lines_are_refused(argv, monkeypatch):
    """argparse reports these as usage errors, before any backend is imported."""
    monkeypatch.setattr("sys.argv", ["netqmpi", *argv])
    with pytest.raises(SystemExit) as exit_info:
        cli.main()
    assert exit_info.value.code == 2


def test_a_broken_config_file_is_a_usage_error(tmp_path, fake_backends,
                                               monkeypatch, capsys):
    """
    Reported as a usage error, not as a traceback out of the YAML reader.

    The CLI catches ``ValueError``, ``OSError`` and ``RuntimeError`` around
    the whole backend selection precisely so a mistyped key, a missing file
    or an environment the backend refuses reads like a command-line
    mistake, which is what it is.
    """
    config_file = tmp_path / "config.yaml"
    config_file.write_text("aer:\n  seed: 3\n")      # not a field of the config
    monkeypatch.setattr("sys.argv", ["netqmpi", "-n", "2", write_app(tmp_path),
                                     "--aer", "--config", str(config_file)])

    with pytest.raises(SystemExit) as exit_info:
        cli.main()
    assert exit_info.value.code == 2
    assert "Unknown config keys" in capsys.readouterr().err


def test_a_missing_config_file_is_a_usage_error(tmp_path, fake_backends,
                                                monkeypatch, capsys):
    """The other half of the same guard: the file is not there at all."""
    monkeypatch.setattr("sys.argv", ["netqmpi", "-n", "2", write_app(tmp_path),
                                     "--aer", "--config", str(tmp_path / "absent.yaml")])

    with pytest.raises(SystemExit) as exit_info:
        cli.main()
    assert exit_info.value.code == 2
    assert "absent.yaml" in capsys.readouterr().err


# ----------------------------------------------------------------------
# Simulation plugins on the command line
# ----------------------------------------------------------------------

def test_squidasm_is_an_alias_of_netqasm(tmp_path, fake_backends, captured_run):
    """The simulator users think of is SquidASM; the flag says the same."""
    call = captured_run("-n", "2", write_app(tmp_path), "--squidasm")
    assert isinstance(call["executor"].config, FakeNetQASMConfig)
    assert call["executor"].config.netqasm_major == 2


def test_plugin_flags_win_over_the_config_file(tmp_path, fake_backends,
                                               captured_run):
    """``--plugin`` and ``--plugin-config`` override the file, as ``--shots`` does."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text("netqasm:\n  plugin: from_file\n"
                           "  plugin_config:\n    x: 1\n")
    plugin_config = tmp_path / "params.yaml"
    plugin_config.write_text("y: 2\n")
    script = write_app(tmp_path)

    from_file = captured_run("-n", "2", script, "--netqasm",
                             "--config", str(config_file))["executor"].config
    assert (from_file.plugin, from_file.plugin_config) == ("from_file", {"x": 1})

    overridden = captured_run("-n", "2", script, "--netqasm",
                              "--config", str(config_file),
                              "--plugin", "from_cli",
                              "--plugin-config", str(plugin_config))
    config = overridden["executor"].config
    assert (config.plugin, config.plugin_config) == ("from_cli", {"y": 2})


def test_an_empty_plugin_config_file_is_an_empty_mapping(tmp_path,
                                                         fake_backends,
                                                         captured_run):
    """Given on purpose, so it is not the same as no ``--plugin-config`` at all."""
    plugin_config = tmp_path / "params.yaml"
    plugin_config.write_text("")
    call = captured_run("-n", "1", write_app(tmp_path), "--netqasm",
                        "--plugin", "p", "--plugin-config", str(plugin_config))
    assert call["executor"].config.plugin_config == {}


def test_a_plugin_config_file_that_is_not_a_mapping_is_a_usage_error(
        tmp_path, fake_backends, monkeypatch, capsys):
    plugin_config = tmp_path / "params.yaml"
    plugin_config.write_text("- 0.1\n- 0.2\n")
    monkeypatch.setattr("sys.argv", ["netqmpi", "-n", "1", write_app(tmp_path),
                                     "--netqasm", "--plugin", "p",
                                     "--plugin-config", str(plugin_config)])
    with pytest.raises(SystemExit) as exit_info:
        cli.main()
    assert exit_info.value.code == 2
    err = capsys.readouterr().err
    assert "params.yaml" in err and "mapping" in err


@pytest.mark.parametrize("backend", ["aer", "cunqa", "qoala"])
@pytest.mark.parametrize("flag", ["--plugin", "--plugin-config"])
def test_plugin_flags_are_refused_by_a_backend_without_plugins(
        backend, flag, tmp_path, fake_backends, monkeypatch, capsys):
    """
    ``--aer --plugin nv_2026`` would otherwise run an ideal simulation.

    The user would believe they had simulated the plugin's hardware, so the
    flag is refused, naming the backend, instead of being ignored.
    """
    plugin_config = tmp_path / "params.yaml"
    plugin_config.write_text("x: 1\n")
    value = "nv_2026" if flag == "--plugin" else str(plugin_config)
    monkeypatch.setattr("sys.argv", ["netqmpi", "-n", "2", write_app(tmp_path),
                                     BACKENDS[backend][0], flag, value])

    with pytest.raises(SystemExit) as exit_info:
        cli.main()
    assert exit_info.value.code == 2
    err = capsys.readouterr().err
    assert f"{flag} is not supported by the {backend} backend" in err


def test_list_plugins_names_each_plugin_its_target_and_distribution(
        install_plugins, monkeypatch, capsys):
    """Works without ``-n`` or a script, like ``--version``."""
    module = install_plugins()
    monkeypatch.setattr("sys.argv", ["netqmpi", "--list-plugins"])

    with pytest.raises(SystemExit) as exit_info:
        cli.main()
    assert exit_info.value.code == 0
    lines = capsys.readouterr().out.splitlines()
    device = [line for line in lines if line.split()[:1] == ["dummy_device"]]
    assert len(device) == 1, lines
    assert f"{module}:device" in device[0]
    assert "netqmpi-dummy 1.2.3" in device[0]


def test_list_plugins_says_when_none_is_installed(monkeypatch, capsys):
    from netqmpi import plugins
    monkeypatch.setattr(plugins, "installed", lambda: [])
    monkeypatch.setattr("sys.argv", ["netqmpi", "--list-plugins"])

    with pytest.raises(SystemExit) as exit_info:
        cli.main()
    assert exit_info.value.code == 0
    assert "No NetQMPI plugins are installed" in capsys.readouterr().out


# ----------------------------------------------------------------------
# --metrics
# ----------------------------------------------------------------------

class _Number:
    """A value ``json`` cannot write as it is, like a NumPy integer."""

    def item(self):
        return 7


def test_metrics_are_written_as_json(tmp_path, fake_backends, captured_run,
                                     monkeypatch, capsys):
    """Only the metrics go to the file, and values json cannot write are converted."""
    import json

    reported = {"netqmpi": {"sim_time_ns": [1500.0, 1600.0]},
                "plugin": {"epr_pairs": _Number()}}
    monkeypatch.setattr(FakeExecutor, "metrics", lambda self: reported,
                        raising=False)
    path = tmp_path / "metrics.json"

    captured_run("-n", "2", write_app(tmp_path), "--netqasm",
                 "--metrics", str(path))

    assert json.loads(path.read_text()) == {
        "netqmpi": {"sim_time_ns": [1500.0, 1600.0]},
        "plugin": {"epr_pairs": 7}}
    assert capsys.readouterr().err == ""


def test_a_backend_without_metrics_writes_an_empty_object(tmp_path, fake_backends,
                                                          captured_run,
                                                          monkeypatch, capsys):
    """The file is always written, so a sweep over backends finds it; stderr says why it is empty."""
    import json

    monkeypatch.setattr(FakeExecutor, "metrics", lambda self: {}, raising=False)
    path = tmp_path / "metrics.json"

    captured_run("-n", "2", write_app(tmp_path), "--aer", "--metrics", str(path))

    assert json.loads(path.read_text()) == {}
    assert "reported no metrics" in capsys.readouterr().err


def test_without_the_flag_no_metrics_are_asked_for(tmp_path, fake_backends,
                                                   captured_run, monkeypatch):
    """An executor that has no metrics() at all still runs as before."""
    monkeypatch.delattr(FakeExecutor, "metrics", raising=False)
    call = captured_run("-n", "2", write_app(tmp_path), "--aer")
    assert isinstance(call["executor"], FakeExecutor)


def test_metrics_into_a_missing_directory_is_a_usage_error(tmp_path, fake_backends,
                                                           monkeypatch, capsys):
    """Refused before the run, not discovered after it."""
    target = tmp_path / "absent" / "metrics.json"
    monkeypatch.setattr("sys.argv", ["netqmpi", "-n", "2", write_app(tmp_path),
                                     "--aer", "--metrics", str(target)])

    with pytest.raises(SystemExit) as exit_info:
        cli.main()
    assert exit_info.value.code == 2
    err = capsys.readouterr().err
    assert "--metrics" in err and "absent" in err
