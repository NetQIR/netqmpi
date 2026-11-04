"""
Simulation plugins: what NetQMPI expects of one, and how it finds it.

A plugin customises how a backend simulates the network: a hardware model, a
link model, readout error. NetQMPI does not interpret it. It finds the
plugin, hands it ``plugin_config`` unread, and keeps the context manager the
plugin returns open for the length of the simulation. Which backends accept
plugins is up to each backend's run config, which has a ``plugin`` key when
it does (today only the NetQASM backend's).

A plugin is named in one of two ways:

* By a **short name**, without a colon, which is looked up in the
  ``netqmpi.plugins`` entry-point group. This is how an installed plugin
  package is used::

      # pyproject.toml of the plugin package
      [project.entry-points."netqmpi.plugins"]
      nv_2026 = "netqmpi_colorcenters:nv_2026"

* As ``"module:callable"``, with the module imported the way ``python -m``
  would import it: the working directory first.

Nothing here imports a backend, so ``netqmpi --list-plugins`` works whatever
simulator is installed.
"""
from __future__ import annotations

import copy
import importlib
import os
import sys
from dataclasses import dataclass, fields
from typing import (
    Any, Callable, Collection, ContextManager, Dict, List, Mapping, Optional,
)

try:  # pragma: no cover - depends on the Python version
    from typing import Protocol
except ImportError:  # pragma: no cover - Python < 3.8
    Protocol = object  # type: ignore[assignment,misc]

#: The entry-point group plugin packages register their plugins in.
GROUP = "netqmpi.plugins"

#: Run-config keys ``run_defaults`` may not set: they say which plugin runs.
_NOT_DEFAULTS = ("plugin", "plugin_config")


class Plugin(Protocol):
    """
    What NetQMPI requires of a plugin, as a :class:`typing.Protocol`.

    Only ``__call__`` is required, so a plain function is a plugin. NetQMPI
    reads three further members with ``getattr``, each optional:

    ``run_defaults``
        A mapping of run-config keys to the values the plugin needs, for
        example ``{"formalism": "dm", "num_qubits": 3}``. They take the place
        of the backend's own defaults; anything the config file or the
        command line sets still wins. They are applied when the config is
        built from a file or the command line, through the run config's
        ``from_dict``.
    ``configure(plugin_config, run_config)``
        Called once when the run config is built, before any rank is traced,
        with copies of ``plugin_config`` and of the final run config. It
        validates them and raises :class:`ValueError` on a bad value or an
        incompatible setting; whatever it changes is discarded.
    ``backends``
        The backends the plugin is written for, e.g. ``("netqasm",)``. A
        backend not in it refuses the plugin. Without it, any backend that
        accepts plugins does.
    """

    def __call__(self, plugin_config: Mapping[str, Any]) -> ContextManager[Any]:
        """Return the context manager entered around the simulation."""
        ...


@dataclass(frozen=True)
class InstalledPlugin:
    """
    A plugin registered in the ``netqmpi.plugins`` entry-point group.

    Attributes:
        name: The short name a run selects it by.
        target: The ``module:attribute`` the name points to.
        distribution: The installed distribution that registers it.
        version: That distribution's version.
    """
    name: str
    target: str
    distribution: str
    version: str


def installed() -> List[InstalledPlugin]:
    """
    List the plugins installed through entry points, sorted by name.

    Reads each installed distribution's entry points rather than calling
    ``importlib.metadata.entry_points``, whose signature changed in Python
    3.10; this works the same from 3.8 on and also gives the distribution
    that provides each plugin. A distribution found twice on ``sys.path``
    counts once, as the first copy, which is the one Python imports.

    Returns:
        One entry per plugin and distribution.
    """
    from importlib import metadata

    seen = set()
    found = []
    for dist in metadata.distributions():
        name = dist.metadata["Name"] or "?"
        key = name.lower().replace("_", "-")
        if key in seen:
            continue
        seen.add(key)
        for entry_point in dist.entry_points:
            if entry_point.group == GROUP:
                found.append(InstalledPlugin(entry_point.name, entry_point.value,
                                             name, dist.version))
    return sorted(found, key=lambda plugin: (plugin.name, plugin.distribution))


def load(spec: str) -> Callable[[Mapping[str, Any]], ContextManager[Any]]:
    """
    Find the plugin a run config names.

    Args:
        spec: A short name registered in the ``netqmpi.plugins``
            entry-point group, or ``"package.module:callable"``.

    Returns:
        The plugin: a callable that takes ``plugin_config``.

    Raises:
        ValueError: If no installed plugin has the short name, two
            distributions provide it, ``spec`` is a malformed
            ``module:callable``, or what it names cannot be imported or is
            not callable.
    """
    if ":" not in spec:
        return _load_installed(spec)
    return _load_module_callable(spec)


def _load_installed(name: str) -> Callable[..., Any]:
    """Import the plugin an entry point named ``name`` points to."""
    every = installed()
    matches = [plugin for plugin in every if plugin.name == name]
    if not matches:
        names = sorted({plugin.name for plugin in every})
        available = (f"installed plugins: {', '.join(names)}" if names
                     else "no plugins are installed")
        raise ValueError(
            f"Unknown plugin {name!r}: {available}. Name an installed plugin "
            f"by its entry-point name (see 'netqmpi --list-plugins') or a "
            f"local one as 'module:callable'.")
    if len(matches) > 1:
        providers = ", ".join(f"{plugin.distribution} {plugin.version} "
                              f"({plugin.target})" for plugin in matches)
        raise ValueError(
            f"Plugin {name!r} is provided by more than one installed "
            f"distribution: {providers}. Name the one to use as "
            f"'module:callable'.")

    plugin = matches[0]
    origin = f"{plugin.target}, from {plugin.distribution} {plugin.version}"
    # An entry point reads "module:attribute", optionally with "[extras]".
    module_name, _, attribute = plugin.target.split("[")[0].partition(":")
    try:
        module = importlib.import_module(module_name.strip())
    except ImportError as error:
        raise ValueError(
            f"Cannot import plugin {name!r} ({origin}): {error}") from error
    target = module
    for part in attribute.strip().split("."):
        target = getattr(target, part, None)
    if not callable(target):
        raise ValueError(f"Plugin {name!r} ({origin}) is not a callable.")
    return target


def _load_module_callable(spec: str) -> Callable[..., Any]:
    """Import ``module:callable``, from the working directory first."""
    module_name, _, attribute = spec.partition(":")
    if not module_name or not attribute:
        raise ValueError(f"plugin must read 'module:callable', got {spec!r}.")

    cwd = os.getcwd()
    added = cwd not in sys.path
    if added:
        sys.path.insert(0, cwd)
    try:
        module = importlib.import_module(module_name)
    except ImportError as error:
        raise ValueError(f"Cannot import plugin module {module_name!r}: {error}") from error
    finally:
        if added:
            sys.path.remove(cwd)

    factory = getattr(module, attribute, None)
    if not callable(factory):
        raise ValueError(f"Plugin module {module_name!r} has no callable {attribute!r}.")
    return factory


def run_defaults(plugin: Any, spec: str, config_cls: type) -> Dict[str, Any]:
    """
    Read and check the run-config defaults a plugin asks for.

    Args:
        plugin: The loaded plugin.
        spec: The name the run config gave it, for error messages.
        config_cls: The backend's run-config dataclass.

    Returns:
        A copy of ``plugin.run_defaults``, or ``{}`` if it has none.

    Raises:
        ValueError: If ``run_defaults`` is not a mapping, names a key the
            run config does not have, or sets ``plugin``/``plugin_config``.
            The message names the plugin: the user's config is not at fault.
    """
    defaults = getattr(plugin, "run_defaults", None)
    if defaults is None:
        return {}
    if not isinstance(defaults, Mapping):
        raise ValueError(
            f"Plugin {spec!r}: run_defaults must be a mapping of run-config "
            f"keys, got {type(defaults).__name__}.")
    known = {f.name for f in fields(config_cls)} - set(_NOT_DEFAULTS)
    unknown = sorted(set(defaults) - known)
    if unknown:
        raise ValueError(
            f"Plugin {spec!r}: run_defaults sets keys {config_cls.__name__} "
            f"does not accept as defaults: {unknown}.")
    return dict(defaults)


def check_backend(plugin: Any, spec: str, backend: str) -> None:
    """
    Refuse a plugin whose ``backends`` does not include ``backend``.

    Args:
        plugin: The loaded plugin.
        spec: The name the run config gave it, for error messages.
        backend: The backend about to run it, e.g. ``"netqasm"``.

    Raises:
        ValueError: If the plugin declares backends and ``backend`` is not
            one of them.
    """
    backends: Optional[Collection[str]] = getattr(plugin, "backends", None)
    if backends is None:
        return
    if isinstance(backends, str):
        backends = (backends,)
    if backend not in backends:
        raise ValueError(
            f"Plugin {spec!r} is written for the {', '.join(backends)} "
            f"backend(s), not {backend}.")


def configure(plugin: Any, run_config: Any) -> None:
    """
    Let a plugin check the run config it will run under.

    The plugin's ``configure``, if it has one, is called with a deep copy
    of ``plugin_config`` and a copy of ``run_config`` that carries another,
    so it can read both but change neither. The run config itself is copied
    shallowly: it may hold objects, such as ``post_function``, that cannot
    be deep-copied.

    Args:
        plugin: The loaded plugin.
        run_config: The final run config, with ``plugin_config``.

    Raises:
        Exception: Whatever the plugin's ``configure`` raises;
            :class:`ValueError` is the documented choice.
    """
    hook = getattr(plugin, "configure", None)
    if hook is None:
        return
    view = copy.copy(run_config)
    view.plugin_config = copy.deepcopy(run_config.plugin_config)
    hook(copy.deepcopy(dict(run_config.plugin_config or {})), view)
