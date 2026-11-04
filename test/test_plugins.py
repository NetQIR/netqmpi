"""
How NetQMPI finds a simulation plugin and reads its optional members.

A plugin package registers its plugins in the ``netqmpi.plugins``
entry-point group, and a run names one by that short name; ``module:callable``
keeps working for a plugin that is not installed. None of this depends on a
backend, so nothing here needs a simulator: the plugins come from a dummy
distribution the ``install_plugins`` fixture lays out on ``sys.path``.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any, Dict, Optional

import pytest

from netqmpi import plugins


def test_installed_plugins_are_listed_with_their_distribution(install_plugins):
    module = install_plugins()

    found = {plugin.name: plugin for plugin in plugins.installed()}

    assert found["dummy_device"] == plugins.InstalledPlugin(
        name="dummy_device", target=f"{module}:device",
        distribution="netqmpi-dummy", version="1.2.3")
    assert {"dummy_plain", "dummy_qoala"} <= set(found)


def test_a_short_name_loads_the_installed_plugin_from_any_directory(
        install_plugins, tmp_path, monkeypatch):
    module = install_plugins()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    plugin = plugins.load("dummy_device")

    assert plugin is sys.modules[module].device


def test_an_unknown_short_name_lists_the_installed_plugins(install_plugins):
    install_plugins()
    with pytest.raises(ValueError) as error:
        plugins.load("nv_2027")
    message = str(error.value)
    assert "Unknown plugin 'nv_2027'" in message
    assert "dummy_device" in message and "dummy_plain" in message


def test_an_unknown_short_name_says_when_nothing_is_installed(monkeypatch):
    monkeypatch.setattr(plugins, "installed", lambda: [])
    with pytest.raises(ValueError, match="no plugins are installed"):
        plugins.load("nv_2027")


def test_an_entry_point_whose_target_is_missing_is_named(install_plugins):
    install_plugins()
    with pytest.raises(ValueError, match="dummy_missing.*netqmpi-dummy 1.2.3"):
        plugins.load("dummy_missing")


def test_a_short_name_two_distributions_provide_is_ambiguous(install_plugins):
    module = install_plugins()
    install_plugins(name="netqmpi-rival", version="0.1",
                    entry_points={"dummy_plain": f"{module}:plain"})
    with pytest.raises(ValueError) as error:
        plugins.load("dummy_plain")
    assert "netqmpi-dummy" in str(error.value)
    assert "netqmpi-rival" in str(error.value)


def test_module_callable_is_still_imported_from_the_working_directory(
        tmp_path, monkeypatch):
    (tmp_path / "local_plugin_29.py").write_text(
        "import contextlib\nwrap = lambda config: contextlib.nullcontext()\n")
    monkeypatch.chdir(tmp_path)
    try:
        assert callable(plugins.load("local_plugin_29:wrap"))
    finally:
        sys.modules.pop("local_plugin_29", None)


@pytest.mark.parametrize("spec", [":wrap", "module:"])
def test_a_malformed_module_callable_is_refused(spec):
    with pytest.raises(ValueError, match="module:callable"):
        plugins.load(spec)


# ---------------------------------------------------------------------------
# The optional members: run_defaults, backends, configure
# ---------------------------------------------------------------------------

@dataclass
class Config:
    """A run config with a plugin, as a backend that supports them has."""
    shots: int = 1
    num_qubits: int = 5
    plugin: Optional[str] = None
    plugin_config: Optional[Dict[str, Any]] = None


def test_run_defaults_are_empty_for_a_plugin_without_them(install_plugins):
    install_plugins()
    assert plugins.run_defaults(plugins.load("dummy_plain"), "dummy_plain",
                                Config) == {}


def test_run_defaults_are_read_from_the_plugin(install_plugins):
    install_plugins()
    defaults = plugins.run_defaults(plugins.load("dummy_device"),
                                    "dummy_device", Config)
    assert defaults == {"num_qubits": 3, "shots": 7}


def test_run_defaults_with_an_unknown_key_blame_the_plugin(install_plugins):
    install_plugins()
    with pytest.raises(ValueError) as error:
        plugins.run_defaults(plugins.load("dummy_typo"), "dummy_typo", Config)
    assert "dummy_typo" in str(error.value) and "num_qbits" in str(error.value)


@pytest.mark.parametrize("defaults", [{"plugin": "other"},
                                      {"plugin_config": {}},
                                      ["num_qubits"]])
def test_run_defaults_cannot_name_a_plugin_or_be_anything_but_a_mapping(defaults):
    class Plugin:
        run_defaults = defaults

        def __call__(self, plugin_config):
            raise AssertionError("not called")

    with pytest.raises(ValueError, match="run_defaults"):
        plugins.run_defaults(Plugin(), "p", Config)


def test_a_plugin_for_another_backend_is_refused(install_plugins):
    install_plugins()
    with pytest.raises(ValueError, match="dummy_qoala.*qoala.*netqasm"):
        plugins.check_backend(plugins.load("dummy_qoala"), "dummy_qoala",
                              "netqasm")


def test_a_plugin_that_names_no_backend_is_accepted_by_any(install_plugins):
    install_plugins()
    plugins.check_backend(plugins.load("dummy_plain"), "dummy_plain", "netqasm")
    plugins.check_backend(plugins.load("dummy_device"), "dummy_device", "netqasm")


def test_configure_sees_copies_of_the_final_config(install_plugins):
    module = install_plugins()
    config = Config(shots=4, num_qubits=6, plugin="dummy_device",
                    plugin_config={"a": 1})

    plugins.configure(plugins.load("dummy_device"), config)

    assert sys.modules[module].CALLS == [("configure", {"a": 1}, 6, 4)]
    assert config.num_qubits == 6, "configure changed the run config"
    assert config.plugin_config == {"a": 1}, "configure changed plugin_config"


def test_configure_is_optional(install_plugins):
    install_plugins()
    plugins.configure(plugins.load("dummy_plain"), Config(plugin="dummy_plain"))
