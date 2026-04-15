"""
Backend-agnostic configuration for running a NetQMPI application.

This module defines a runtime configuration object that uses only
primitive Python types, ensuring that the runtime layer remains fully
decoupled from any specific backend (e.g. NetQASM, CUNQA).

Backend adapters may subclass :class:`RunConfig` to introduce additional
fields required by their simulator or hardware.
"""
from __future__ import annotations

from dataclasses import dataclass

@dataclass
class RunConfig:
    """
    Configuration for a single NetQMPI simulation run.

    All fields rely on plain Python types. Backend-specific parameters
    (such as NetQASM formalism or CUNQA topology descriptions) should be
    provided by subclasses defined in the corresponding adapter packages.

    Attributes:
        name: Name of the app running.
        shots: Number of times the simulation is repeated.
    """
    shots: int = 1024