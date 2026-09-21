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
        transfer_mode: Qubit transfer protocol for qsend/qrecv. Only
            ``"swap"`` exists today; ``"teleport"`` is accepted by the
            dataclass but raises :class:`NotImplementedError` at
            translation time.

            ``"swap"`` moves the state with a SWAP straight across the
            global register. That is unphysical — no entanglement is
            consumed, no classical correction is sent, and nothing can go
            wrong — which is exactly what makes this backend useful as a
            **correctness reference**: a wrong answer here is a bug in the
            translation, never decoherence. It is not a model of a quantum
            network, and fidelities measured on it say nothing about one.

            Implementing ``"teleport"`` is only worth doing together with
            an Aer noise model. On a noiseless simulator a teleportation
            circuit returns exactly what the SWAP returns, just with more
            gates and a pair of ancillas per transfer, so on its own it
            would add cost without adding information.
        seed_simulator: Optional RNG seed for reproducible simulations.
    """

    shots: int = 1024
    transfer_mode: str = "swap"        # "swap" | "teleport"
    seed_simulator: Optional[int] = None
