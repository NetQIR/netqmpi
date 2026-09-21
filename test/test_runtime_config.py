"""
Run configuration: the YAML file and the dataclasses it fills in.

NetQMPI keeps one configuration file for every backend rather than a
growing wall of command-line flags. The file mixes *generic* settings at
the top level with an optional per-backend block, and only the block of
the backend being run is merged in — so a single file can carry the
NetSquid link fidelity, the Aer seed and the CUNQA shot count side by
side, and switching ``--netqasm`` for ``--aer`` changes which half of it
applies.

Two properties make that safe, and both are tested here: an unknown key is
an error rather than a silent no-op (a typo in a config file is otherwise
invisible, and the run just quietly uses the default), and a backend block
never leaks into another backend's config.
"""
from __future__ import annotations

import dataclasses
import textwrap

import pytest

from netqmpi.runtime.run_config import KNOWN_BACKENDS, RunConfig, read_config_block


@pytest.fixture(scope="module")
def aer_config():
    """
    The Aer backend's config class, or a skip where Qiskit is missing.

    The class itself is a plain dataclass, but its package imports Qiskit
    at module scope, so reaching it needs the backend installed.
    """
    pytest.importorskip("qiskit", reason="the Aer backend needs Qiskit")
    from netqmpi.runtime.adapters.aer import AerSimulatorConfig
    return AerSimulatorConfig


def write(tmp_path, text: str) -> str:
    """Write a YAML config file and return its path."""
    path = tmp_path / "config.yaml"
    path.write_text(textwrap.dedent(text))
    return str(path)


# ----------------------------------------------------------------------
# RunConfig
# ----------------------------------------------------------------------

def test_the_generic_config_carries_a_shot_count():
    """Shots are the one setting every backend understands."""
    assert RunConfig().shots == 1024
    assert RunConfig(shots=10).shots == 10


def test_from_dict_fills_the_declared_fields():
    """The bridge between the parsed YAML and the typed config."""
    assert RunConfig.from_dict({"shots": 7}).shots == 7
    assert RunConfig.from_dict({}).shots == 1024


def test_an_unknown_key_is_reported_rather_than_ignored():
    """A typo in a config file must not be silently dropped."""
    with pytest.raises(ValueError, match=r"Unknown config keys for RunConfig: \['shot'\]"):
        RunConfig.from_dict({"shot": 7})


def test_the_error_lists_every_unknown_key_at_once():
    """Sorted, so fixing a file is one pass rather than one key per run."""
    with pytest.raises(ValueError, match=r"\['alpha', 'beta'\]"):
        RunConfig.from_dict({"beta": 1, "alpha": 2, "shots": 3})


def test_a_backend_config_extends_the_generic_one(aer_config):
    """Backend fields are added by subclassing, not by a free-form dict."""
    assert issubclass(aer_config, RunConfig)

    config = aer_config.from_dict(
        {"shots": 64, "transfer_mode": "swap", "seed_simulator": 11})
    assert (config.shots, config.transfer_mode, config.seed_simulator) == (
        64, "swap", 11)

    names = {field.name for field in dataclasses.fields(aer_config)}
    assert names == {"shots", "transfer_mode", "seed_simulator"}


def test_a_backend_config_rejects_another_backends_key(aer_config):
    """``seed`` is not ``seed_simulator``, and the run says so."""
    with pytest.raises(ValueError, match="Unknown config keys for AerSimulatorConfig"):
        aer_config.from_dict({"seed": 11})


def test_the_aer_defaults_are_the_documented_ones(aer_config):
    """The reference backend moves qubits with a SWAP and seeds nothing."""
    config = aer_config()
    assert config.transfer_mode == "swap"
    assert config.seed_simulator is None


# ----------------------------------------------------------------------
# read_config_block
# ----------------------------------------------------------------------

def test_generic_settings_reach_every_backend(tmp_path):
    """Top-level keys are shared by all of them."""
    path = write(tmp_path, "shots: 500\n")
    for backend in KNOWN_BACKENDS:
        assert read_config_block(path, backend) == {"shots": 500}


def test_only_the_selected_backends_block_is_merged(tmp_path):
    """One file, four backends, and no cross-contamination."""
    path = write(tmp_path, """
        shots: 1000
        aer:
          seed_simulator: 7
        qoala:
          link_fidelity: 0.8
    """)

    assert read_config_block(path, "aer") == {"shots": 1000, "seed_simulator": 7}
    assert read_config_block(path, "qoala") == {"shots": 1000, "link_fidelity": 0.8}
    assert read_config_block(path, "netqasm") == {"shots": 1000}


def test_a_backend_block_overrides_the_generic_value(tmp_path):
    """The more specific setting wins, as a reader would expect."""
    path = write(tmp_path, """
        shots: 1000
        aer:
          shots: 16
    """)
    assert read_config_block(path, "aer") == {"shots": 16}


def test_a_nested_block_is_passed_through_whole(tmp_path):
    """A qdevice section is the backend's business, not the reader's."""
    path = write(tmp_path, """
        qoala:
          hardware:
            t1: 0
            t2: 0
    """)
    assert read_config_block(path, "qoala") == {"hardware": {"t1": 0, "t2": 0}}


def test_an_empty_file_is_an_empty_config(tmp_path):
    """Nothing configured is not an error: every field has a default."""
    assert read_config_block(write(tmp_path, ""), "aer") == {}


def test_an_empty_backend_block_is_an_empty_config(tmp_path):
    """A block with no keys under it parses as ``None``; it must not break."""
    path = write(tmp_path, """
        shots: 4
        aer:
    """)
    assert read_config_block(path, "aer") == {"shots": 4}


def test_a_file_that_is_not_a_mapping_is_refused(tmp_path):
    """A list at the top level names the file in the error."""
    path = write(tmp_path, "- shots\n- 10\n")
    with pytest.raises(ValueError, match="must contain a mapping at top level"):
        read_config_block(path, "aer")


def test_a_backend_block_that_is_not_a_mapping_is_refused(tmp_path):
    """``aer: 7`` is a mistake worth naming."""
    path = write(tmp_path, "aer: 7\n")
    with pytest.raises(ValueError, match="'aer' block .* must be a mapping"):
        read_config_block(path, "aer")


def test_a_missing_file_fails_where_it_is_read(tmp_path):
    """An OSError the CLI turns into a usage error, not a traceback."""
    with pytest.raises(OSError):
        read_config_block(str(tmp_path / "absent.yaml"), "aer")


def test_the_known_backends_match_the_cli_flags():
    """A backend the reader does not know would have its block treated as generic."""
    assert set(KNOWN_BACKENDS) == {"netqasm", "cunqa", "aer", "qoala"}
