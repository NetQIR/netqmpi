"""
Backend-specific configuration for Qiskit AerSimulator runs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from netqmpi.runtime.run_config import RunConfig


@dataclass
class AerSimulatorConfig(RunConfig):
    """
    Extension of :class:`~netqmpi.runtime.run_config.RunConfig` with
    Qiskit AerSimulator-specific fields.

    Attributes:
        shots: Number of simulation shots.
        transfer_mode: Qubit transfer protocol for qsend/qrecv.
            ``"swap"`` (default) inserts an unphysical SWAP gate between
            the source and destination qubit slots — produces shallower
            circuits and is easier to debug, but does not model a real
            quantum-network transfer.
            ``"teleport"`` implements a physically realistic teleportation
            circuit using mid-circuit measurement and classical feedforward,
            which requires AerSimulator dynamic-circuits support.
        seed_simulator: Optional RNG seed for reproducible simulations.
    """

    shots: int = 1024
    transfer_mode: str = "swap"        # "swap" | "teleport"
    seed_simulator: Optional[int] = None
