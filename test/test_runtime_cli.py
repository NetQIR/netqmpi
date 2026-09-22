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

def args(config=None, shots=None):
    """The slice of the parsed CLI arguments the config builder reads."""
    return argparse.Namespace(config=config, shots=shots)


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
        setattr(module, config_name, RunConfig)
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
