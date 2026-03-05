"""
Backend-agnostic configuration for running a NetQMPI application.

Uses only primitive Python types so that the runtime layer stays completely
decoupled from any specific backend (NetQASM, Cunqa, …).

Backend adapters may subclass :class:`RunConfig` to add fields that are
specific to their underlying simulator or hardware.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional


@dataclass
class RunConfig:
    """
    Configuration for a single NetQMPI simulation run.

    All fields use plain Python types. Backend-specific configuration
    (e.g. NetQASM formalism, Cunqa topology) is handled by subclasses
    defined in the corresponding adapter packages.

    Attributes:
        num_rounds:      Number of times to repeat the simulation.
        enable_logging:  Whether to write execution logs to disk.
        use_app_config:  Whether to pass the backend app-config to each process.
        hardware:        Hardware model identifier (e.g. ``"generic"``).
        post_function:   Optional callable invoked after each simulation round.
        network_config:  Opaque backend-specific network topology description.
        log_cfg:         Opaque backend-specific logging configuration.
    """

    num_rounds: int = 1
    enable_logging: bool = True
    use_app_config: bool = True
    hardware: str = "generic"
    post_function: Optional[Callable] = None
    network_config: Optional[Any] = None
    log_cfg: Optional[Any] = None
