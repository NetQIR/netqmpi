"""
A NetQMPI plugin that gives one- and two-qubit gates different noise.

SquidASM's generic node applies a single ``gate_fidelity`` to every gate, and
fixes gate durations at 1 µs for one-qubit gates and 500 µs for two-qubit
gates. On real processors two-qubit gates are typically an order of
magnitude noisier than one-qubit gates, and durations depend on the
platform. This plugin sets both separately:

    netqasm:
      plugin: gate_noise:gate_noise
      plugin_config:
        single_qubit_depolar_prob: 0.001
        two_qubit_depolar_prob: 0.02
        single_qubit_duration: 50.0      # ns
        two_qubit_duration: 300.0        # ns
        init_duration: 1000.0            # ns

Each gate is followed by depolarising noise with the given probability on
every qubit it acts on. Memory noise (``t1``, ``t2`` in the ``netqasm``
block) and the number of qubits per node (``num_qubits``) still come from
NetQMPI's own configuration, so ``gate_fidelity`` is the only key the plugin
replaces.

Extension point: the *device*. SquidASM builds the quantum processor of every
generic node by calling ``squidasm.sim.network.network.QDevice``; the plugin
puts a factory of its own under that name for the duration of the run.
SquidASM's ``QDevice`` cannot simply be given other instructions, because its
constructor overwrites the noise model of every instruction with the one
``gate_fidelity`` dictates, so the factory builds a NetSquid
``QuantumProcessor`` directly.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, fields
from typing import Any, Iterator, List, Mapping, Optional

from netsquid.components import instructions as instr
from netsquid.components.models.qerrormodels import DepolarNoiseModel
from netsquid.components.qprocessor import PhysicalInstruction, QuantumProcessor
from squidasm.sim.network import network

#: The one-qubit gates SquidASM's generic node offers.
SINGLE_QUBIT_GATES = (
    instr.INSTR_X, instr.INSTR_Y, instr.INSTR_Z, instr.INSTR_H, instr.INSTR_K,
    instr.INSTR_S, instr.INSTR_T,
    instr.INSTR_ROT_X, instr.INSTR_ROT_Y, instr.INSTR_ROT_Z,
)

#: The two-qubit gates SquidASM's generic node offers.
TWO_QUBIT_GATES = (instr.INSTR_CNOT, instr.INSTR_CZ)


@dataclass(frozen=True)
class GateNoise:
    """
    The ``plugin_config`` of this plugin. Defaults are SquidASM's own.

    Attributes:
        single_qubit_depolar_prob: Depolarising probability after a
            one-qubit gate.
        two_qubit_depolar_prob: Depolarising probability, on each qubit,
            after a two-qubit gate.
        single_qubit_duration: Duration of a one-qubit gate, in ns.
        two_qubit_duration: Duration of a two-qubit gate, in ns.
        init_duration: Duration of a qubit initialisation, in ns.
    """

    single_qubit_depolar_prob: float = 0.0
    two_qubit_depolar_prob: float = 0.0
    single_qubit_duration: float = 1e3
    two_qubit_duration: float = 5e5
    init_duration: float = 1e5

    @classmethod
    def from_dict(cls, config: Mapping[str, Any]) -> "GateNoise":
        """
        Build the parameters from ``plugin_config``, checking every value.

        Raises:
            ValueError: On an unknown key, a probability outside [0, 1] or a
                negative duration.
        """
        names = {f.name for f in fields(cls)}
        unknown = set(config) - names
        if unknown:
            raise ValueError(f"gate_noise: unknown keys {sorted(unknown)}; "
                             f"expected some of {sorted(names)}.")
        params = cls(**{key: float(value) for key, value in config.items()})
        for name in ("single_qubit_depolar_prob", "two_qubit_depolar_prob"):
            if not 0.0 <= getattr(params, name) <= 1.0:
                raise ValueError(f"gate_noise: {name} must be in [0, 1].")
        for name in ("single_qubit_duration", "two_qubit_duration", "init_duration"):
            if getattr(params, name) < 0.0:
                raise ValueError(f"gate_noise: {name} cannot be negative.")
        return params

    def instructions(self) -> List[PhysicalInstruction]:
        """The physical instructions of a node with this noise."""

        def gate(instruction, duration, prob):
            noise = DepolarNoiseModel(prob, time_independent=True) if prob else None
            return PhysicalInstruction(instruction, duration=duration,
                                       quantum_noise_model=noise)

        return (
            [PhysicalInstruction(instr.INSTR_INIT, duration=self.init_duration)]
            + [gate(g, self.single_qubit_duration, self.single_qubit_depolar_prob)
               for g in SINGLE_QUBIT_GATES]
            + [gate(g, self.two_qubit_duration, self.two_qubit_depolar_prob)
               for g in TWO_QUBIT_GATES]
        )


def gate_noise(config: Mapping[str, Any]):
    """
    The plugin entry point, named in the config file.

    Args:
        config: ``plugin_config``, with the fields of :class:`GateNoise`.

    Returns:
        A context manager under which SquidASM builds nodes with this noise.

    Raises:
        ValueError: On a key or value :meth:`GateNoise.from_dict` refuses.
    """
    return _gate_noise(GateNoise.from_dict(config))


# Optional members of the plugin contract (netqmpi.plugins.Plugin), which a
# plain function can carry as attributes: ``configure`` checks plugin_config
# when the config is built, before any rank is traced, and ``backends``
# refuses the plugin on any backend but the one whose internals it replaces.
gate_noise.configure = lambda plugin_config, run_config: GateNoise.from_dict(plugin_config)
gate_noise.backends = ("netqasm",)


def build_qdevice(params: GateNoise, name: str = "QDevice", num_qubits: int = 5,
                  mem_fidelities: Optional[list] = None,
                  **_: Any) -> QuantumProcessor:
    """
    Build one node's processor.

    Takes the keyword arguments SquidASM passes to ``QDevice``. Of those,
    ``gate_fidelity`` is ignored, since this plugin replaces it, and
    ``mem_fidelities`` carries the T1/T2 memory noise of the ``netqasm``
    block.

    Args:
        params: Gate noise and durations.
        name: Processor name.
        num_qubits: Memory positions, from ``num_qubits`` in the config.
        mem_fidelities: Memory noise model of each position.

    Returns:
        The processor.
    """
    return QuantumProcessor(name, num_positions=num_qubits,
                            phys_instructions=params.instructions(),
                            memory_noise_models=mem_fidelities)


@contextmanager
def _gate_noise(params: GateNoise) -> Iterator[None]:
    """Make SquidASM build every generic node with ``params``."""
    original = network.QDevice

    def qdevice(**kwargs: Any) -> QuantumProcessor:
        return build_qdevice(params, **kwargs)

    network.QDevice = qdevice
    try:
        yield
    finally:
        network.QDevice = original
