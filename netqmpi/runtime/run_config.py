"""
Backend-agnostic configuration for running a NetQMPI application.

This module defines a runtime configuration object that uses only
primitive Python types, ensuring that the runtime layer remains fully
decoupled from any specific backend (e.g. NetQASM, CUNQA, Qoala).

Backend adapters may subclass :class:`RunConfig` to introduce additional
fields required by their simulator or hardware. All configuration is
loaded from a single YAML file (see :func:`read_config_block`) instead of
per-backend command-line flags, so the CLI stays small as backends grow.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Dict, Type, TypeVar

# Backend names that may appear as top-level blocks in a NetQMPI config file.
# Keys not matching one of these are treated as generic (shared) settings.
KNOWN_BACKENDS = ("netqasm", "cunqa", "aer", "qoala")

_T = TypeVar("_T", bound="RunConfig")


@dataclass
class RunConfig:
    """
    Configuration for a single NetQMPI simulation run.

    All fields rely on plain Python types. Backend-specific parameters
    (such as NetQASM formalism or Qoala qdevice noise) should be provided
    by subclasses defined in the corresponding adapter packages.

    Attributes:
        shots: Number of times the simulation is repeated.
    """
    shots: int = 1024

    @classmethod
    def from_dict(cls: Type[_T], data: Dict[str, Any]) -> _T:
        """
        Build a config from a plain dict, mapping keys to dataclass fields.

        Unknown keys raise :class:`ValueError` so typos in a YAML config
        surface immediately instead of being silently ignored. Subclasses
        with nested/structured fields (e.g. a qdevice block) should override
        this to translate those keys before delegating here.

        Args:
            data: Mapping of field names to values.

        Returns:
            A config instance of ``cls``.
        """
        field_names = {f.name for f in dataclasses.fields(cls)}
        unknown = set(data) - field_names
        if unknown:
            raise ValueError(
                f"Unknown config keys for {cls.__name__}: {sorted(unknown)}"
            )
        return cls(**{k: v for k, v in data.items() if k in field_names})


def read_config_block(path: str, backend: str) -> Dict[str, Any]:
    """
    Read a NetQMPI YAML config file and return the settings for one backend.

    The file mixes generic (shared) settings at the top level with optional
    per-backend blocks keyed by backend name. Only the block for ``backend``
    is merged on top of the generic settings; blocks for other backends are
    ignored. Example::

        shots: 1000
        seed: 7
        qoala:
          link_fidelity: 0.8
          hardware:
            t1: 0

    Reading this with ``backend="qoala"`` yields
    ``{"shots": 1000, "seed": 7, "link_fidelity": 0.8, "hardware": {...}}``.

    Args:
        path: Path to the YAML config file.
        backend: Backend whose block should be merged in.

    Returns:
        The merged settings dict, ready for ``RunConfig.from_dict``.

    Raises:
        ValueError: If the file or the backend block is not a mapping.
    """
    import yaml  # local import: only needed when a config file is used

    with open(path) as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file {path} must contain a mapping at top level.")

    generic = {k: v for k, v in data.items() if k not in KNOWN_BACKENDS}
    specific = data.get(backend) or {}
    if not isinstance(specific, dict):
        raise ValueError(f"'{backend}' block in {path} must be a mapping.")

    return {**generic, **specific}
