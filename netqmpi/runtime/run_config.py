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
from typing import Any, Callable, Optional


@dataclass
class RunConfig:
    """
    Configuration for a single NetQMPI simulation run.

    All fields rely on plain Python types. Backend-specific parameters
    (such as NetQASM formalism or CUNQA topology descriptions) should be
    provided by subclasses defined in the corresponding adapter packages.

    Attributes:
        num_rounds: Number of times the simulation is repeated.
        enable_logging: Whether execution logs are written to disk.
        use_app_config: Whether the backend application configuration is
            passed to each process.
        hardware: Identifier of the hardware model to simulate.
        post_function: Optional callable invoked after each simulation round.
        network_config: Backend-specific network topology description.
        log_cfg: Backend-specific logging configuration.
    """

    num_rounds: int = 1
    enable_logging: bool = True
    use_app_config: bool = True
    hardware: str = "generic"
    post_function: Optional[Callable] = None
    network_config: Optional[Any] = None
    log_cfg: Optional[Any] = None