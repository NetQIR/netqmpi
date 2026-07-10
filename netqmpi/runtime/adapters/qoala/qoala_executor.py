"""
Executor adapter for the Qoala backend (simulation only).

Qoala is a NetSquid-based *simulator* of Qoala-spec quantum-internet nodes;
there is no real-hardware execution path, so this backend is explicitly
simulation-only and must not be treated as on par with a physical deployment.

Like the NetQASM adapter, the whole N-node network runs inside a *single* Python
process as one NetSquid discrete-event simulation (one ``ProcNode`` context per
rank), not as N operating-system processes. Each rank compiles its circuit to a
``.iqoala`` program; when all ranks are ready, :meth:`run_simulation` builds the
Qoala network, submits one batch (of ``shots`` iterations) per node, pairs the
remote PIDs for entanglement, runs the simulation and returns a per-rank
measurement histogram.

This is the only module in the Qoala adapter allowed to import ``qoala.*`` /
``netsquid``. Those imports are performed lazily inside :meth:`run_simulation`
so that merely selecting the backend (and building apps) does not pull in the
heavy NetSquid runtime until a simulation actually runs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from netqmpi.runtime.executor import Executor
from netqmpi.runtime.run_config import RunConfig
from netqmpi.sdk.environment import Environment
from netqmpi.runtime.adapters.qoala.qoala_circuit import QoalaCircuitAdapter, QoalaProgramSpec
from netqmpi.runtime.adapters.qoala.qoala_communicator import QoalaCommunicator
from netqmpi.helpers import load_main


# Generic single/two-qubit instruction sets exposed by every node's qdevice.
# INIT and MEASURE are split out from the "gate" instructions so their duration
# (init_time / measure_time) can be configured independently, matching the
# SquidASM qdevice_cfg surface. Qoala's uniform-topology factories bundle them
# all under a single duration, which is why the topology is built by hand here.
_INIT_INSTRUCTIONS = ["INSTR_INIT"]
_MEASURE_INSTRUCTIONS = ["INSTR_MEASURE", "INSTR_MEASURE_INSTANT"]
_GATE_INSTRUCTIONS = [
    "INSTR_ROT_X", "INSTR_ROT_Y", "INSTR_ROT_Z",
    "INSTR_X", "INSTR_Y", "INSTR_Z", "INSTR_H",
]
_TWO_GATE_INSTRUCTIONS = ["INSTR_CNOT", "INSTR_CZ"]


@dataclass
class QoalaQDeviceConfig:
    """
    Hardware parameters of a single node's quantum device (the Qoala analogue of
    SquidASM's ``qdevice_cfg``). Applied uniformly to every node.

    Durations are in nanoseconds. ``T1 == T2 == 0`` means "no memory noise"
    (Qoala's convention for a perfect qubit); depolarising probabilities of 0
    mean noiseless gates. The defaults therefore describe a perfect qdevice.

    Note (documented limitation): ``init_time`` and ``measure_time`` are exposed
    independently by building the topology per-instruction. Depolarising noise is
    applied to the single-/two-qubit *gates* only; ``INSTR_INIT`` and
    ``INSTR_MEASURE`` carry their own duration but no depolarising error, so there
    is no separate readout-flip model here.

    Attributes:
        t1: Amplitude-damping time (ns). 0 disables amplitude damping.
        t2: Dephasing time (ns). 0 disables dephasing. Requires ``t2 <= 2 * t1``
            when both are non-zero.
        single_qubit_gate_time: Duration (ns) of single-qubit gates.
        two_qubit_gate_time: Duration (ns) of two-qubit gates.
        init_time: Duration (ns) of qubit initialization.
        measure_time: Duration (ns) of measurement.
        single_qubit_gate_depolar_prob: Depolarising probability of single-qubit gates.
        two_qubit_gate_depolar_prob: Depolarising probability of two-qubit gates.
    """

    t1: float = 0.0
    t2: float = 0.0
    single_qubit_gate_time: float = 5e3
    two_qubit_gate_time: float = 200e3
    init_time: float = 5e3
    measure_time: float = 5e3
    single_qubit_gate_depolar_prob: float = 0.0
    two_qubit_gate_depolar_prob: float = 0.0

    _FIELDS = (
        "t1", "t2", "single_qubit_gate_time", "two_qubit_gate_time",
        "init_time", "measure_time",
        "single_qubit_gate_depolar_prob", "two_qubit_gate_depolar_prob",
    )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QoalaQDeviceConfig":
        """Build a config from a plain dict, rejecting unknown keys."""
        unknown = set(data) - set(cls._FIELDS)
        if unknown:
            raise ValueError(f"Unknown qdevice config keys: {sorted(unknown)}")
        return cls(**{k: data[k] for k in cls._FIELDS if k in data})

    @classmethod
    def from_yaml(cls, path: str) -> "QoalaQDeviceConfig":
        """Load qdevice hardware parameters from a YAML file."""
        import yaml  # local import: only needed when a hw config file is used
        with open(path) as fh:
            data = yaml.safe_load(fh) or {}
        return cls.from_dict(data)


@dataclass
class QoalaRunConfig(RunConfig):
    """
    Extension of :class:`~netqmpi.runtime.run_config.RunConfig` with
    Qoala/NetSquid simulation parameters.

    Attributes:
        num_qubits_per_node: Physical qubits exposed by every node. If ``None``,
            it is inferred from the compiled circuits (user qubits plus a
            teleportation scratch slot).
        link_duration: EPR-pair generation time (ns) for the perfect links.
        qnos_instr_time: Duration (ns) of a single quantum-processor instruction.
        hw_config: Per-node qdevice hardware parameters. If ``None``, a perfect
            qdevice (no memory/gate noise) is used.
        seed: Optional NetSquid random seed for reproducible runs.
    """

    num_qubits_per_node: Optional[int] = None
    link_duration: float = 1000.0
    qnos_instr_time: float = 1000.0
    hw_config: Optional[QoalaQDeviceConfig] = None
    seed: Optional[int] = None


class QoalaExecutorAdapter(Executor):
    """
    Executor adapter for the Qoala simulator.

    Handles circuit creation and per-rank application construction, and owns the
    shared simulation driver invoked once all ranks have compiled their program.
    """

    def __init__(self, size: int, config: QoalaRunConfig = None) -> None:
        """
        Initialize the Qoala executor adapter.

        Args:
            size: Number of parallel quantum nodes to simulate.
            config: Qoala-specific configuration. Defaults to
                :class:`QoalaRunConfig` with its built-in defaults.
        """
        _config = config or QoalaRunConfig()
        super().__init__(size, _config)
        self._config: QoalaRunConfig = _config

    # ------------------------------------------------------------------
    # Executor interface — circuit factory
    # ------------------------------------------------------------------

    def create_circuit(
        self,
        num_qubits: int,
        num_clbits: int,
        comm: QoalaCommunicator,
    ) -> QoalaCircuitAdapter:
        """
        Create a Qoala circuit adapter.

        Args:
            num_qubits: Number of qubits in the circuit.
            num_clbits: Number of classical bits in the circuit.
            comm: Communicator associated with the circuit.

        Returns:
            A :class:`QoalaCircuitAdapter` instance.
        """
        return QoalaCircuitAdapter(num_qubits, num_clbits, comm)

    # ------------------------------------------------------------------
    # Executor interface — application builder
    # ------------------------------------------------------------------

    def build_apps(self, file: str, size: int) -> List[Any]:
        """
        Build one callable wrapper per rank from the provided script.

        Args:
            file: Path to the NetQMPI Python script defining ``main()``.
            size: Number of ranks to instantiate.

        Returns:
            A list of zero-argument callables, one per rank.
        """
        main_func = load_main(file)
        apps: List[Any] = []
        for rank in range(size):
            comm = QoalaCommunicator(rank, size, self._config, self)
            env = Environment(comm, self)
            apps.append(lambda env=env: main_func(env=env))
        return apps

    # ------------------------------------------------------------------
    # Executor interface — application runner
    # ------------------------------------------------------------------

    def run(self, apps: List[Any]) -> None:
        """
        Run every rank's ``main``. The joint simulation is triggered by the last
        rank leaving its ``with comm`` block (see :meth:`run_simulation`).

        Args:
            apps: Callables returned by :meth:`build_apps`.
        """
        for app in apps:
            app()

    # ------------------------------------------------------------------
    # Simulation driver (invoked by QoalaCommunicator once all ranks ready)
    # ------------------------------------------------------------------

    def run_simulation(
        self, registry: Dict[int, Tuple[QoalaProgramSpec, QoalaCommunicator]]
    ) -> Dict[int, Dict[str, int]]:
        """
        Build the Qoala network and run one simulation for all ranks.

        Args:
            registry: Mapping ``rank -> (program spec, communicator)`` gathered
                as each rank left its ``with comm`` block.

        Returns:
            Mapping ``rank -> {bitstring: count}`` with the measurement
            histogram for each rank (empty for ranks that measure nothing).
        """
        import netsquid as ns
        from qoala.lang.ehi import UnitModule
        from qoala.lang.parse import QoalaParser
        from qoala.runtime.config import (
            ClassicalConnectionConfig,
            LatenciesConfig,
            NtfConfig,
            ProcNodeConfig,
            ProcNodeNetworkConfig,
        )
        from qoala.runtime.program import ProgramInput
        from qoala.sim.build import build_network_from_config
        from qoala.util.runner import create_batch

        ns.sim_reset()
        ns.set_qstate_formalism(ns.QFormalism.DM)
        if self._config.seed is not None:
            ns.set_random_state(seed=self._config.seed)

        ranks = sorted(registry.keys())
        num_iterations = max(1, int(self._config.shots))

        num_qubits = self._config.num_qubits_per_node
        if num_qubits is None:
            num_qubits = max(spec.num_qubits for spec, _ in registry.values())
        num_qubits = max(num_qubits, 1)

        # Uniform qdevice topology shared by every node (perfect unless a
        # QoalaQDeviceConfig with noise is supplied).
        topology = self._build_topology(num_qubits)

        # One ProcNode per rank; node_id == rank so remote_id templates are trivial.
        nodes = [
            ProcNodeConfig(
                node_name=self._rank_name(r),
                node_id=r,
                topology=topology,
                latencies=LatenciesConfig(qnos_instr_time=self._config.qnos_instr_time),
                ntf=NtfConfig.from_cls_name("GenericNtf"),
                determ_sched=True,
            )
            for r in ranks
        ]

        network_cfg = ProcNodeNetworkConfig.from_nodes_perfect_links(
            nodes=nodes, link_duration=self._config.link_duration
        )
        # Perfect links cover EPR generation only; classical channels are separate.
        network_cfg.cconns = [
            ClassicalConnectionConfig.from_nodes(i, j, 1e9)
            for i in ranks
            for j in ranks
            if i < j
        ]

        network = build_network_from_config(network_cfg)

        programs = {r: QoalaParser(registry[r][0].iqoala_text).parse() for r in ranks}
        inputs = {
            r: [ProgramInput(dict(registry[r][0].program_input)) for _ in range(num_iterations)]
            for r in ranks
        }

        batches = {}
        for r in ranks:
            procnode = network.nodes[self._rank_name(r)]
            unit_module = UnitModule.from_full_ehi(procnode.memmgr.get_ehi())
            batches[r] = procnode.submit_batch(
                create_batch(programs[r], unit_module, inputs[r], num_iterations)
            )

        for r in ranks:
            procnode = network.nodes[self._rank_name(r)]
            remote_pids = {
                batches[o].batch_id: [p.pid for p in batches[o].instances]
                for o in ranks
                if o != r
            }
            procnode.initialize_processes(remote_pids, linear=True)

        network.start()
        ns.sim_run()

        results: Dict[int, Dict[str, int]] = {}
        for r in ranks:
            procnode = network.nodes[self._rank_name(r)]
            batch_result = procnode.scheduler.get_batch_results()[0]
            results[r] = self._build_counts(batch_result.results, registry[r][0].outputs)
        return results

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _rank_name(rank: int) -> str:
        """Canonical node name for a rank (matches ``QMPICommunicator``)."""
        return f"rank_{rank}"

    def _build_topology(self, num_qubits: int):
        """
        Build the per-node qdevice topology (``TopologyConfig``).

        With no ``hw_config`` this returns the default perfect topology. With a
        :class:`QoalaQDeviceConfig` it builds the topology by hand so that
        ``init_time``/``measure_time`` are independent from the gate time and the
        depolarising probabilities are applied verbatim (not via a fidelity
        conversion). The link stays perfect (set elsewhere), isolating noise to
        the qdevice.

        Args:
            num_qubits: Number of physical qubits per node.

        Returns:
            A Qoala ``TopologyConfig``.
        """
        from qoala.runtime.config import (
            GateConfig,
            MultiGateConfig,
            QubitConfig,
            QubitIdConfig,
            SingleGateConfig,
            TopologyConfig,
        )

        hw = self._config.hw_config
        if hw is None:
            return TopologyConfig.perfect_config_uniform_default_params(num_qubits)

        def single_gate_configs():
            cfgs = [
                GateConfig.perfect_config(name=n, duration=int(hw.init_time))
                for n in _INIT_INSTRUCTIONS
            ]
            cfgs += [
                GateConfig.perfect_config(name=n, duration=int(hw.measure_time))
                for n in _MEASURE_INSTRUCTIONS
            ]
            cfgs += [
                GateConfig.with_depolar_prob(
                    name=n,
                    duration=int(hw.single_qubit_gate_time),
                    depolar_prob=hw.single_qubit_gate_depolar_prob,
                )
                for n in _GATE_INSTRUCTIONS
            ]
            return cfgs

        qubits = [
            QubitIdConfig(
                qubit_id=i,
                qubit_config=QubitConfig.t1t2_config(
                    is_communication=True, T1=int(hw.t1), T2=int(hw.t2)
                ),
            )
            for i in range(num_qubits)
        ]

        single_gates = [
            SingleGateConfig(qubit_id=i, gate_configs=single_gate_configs())
            for i in range(num_qubits)
        ]

        multi_gates = [
            MultiGateConfig(
                qubit_ids=[i, j],
                gate_configs=[
                    GateConfig.with_depolar_prob(
                        name=n,
                        duration=int(hw.two_qubit_gate_time),
                        depolar_prob=hw.two_qubit_gate_depolar_prob,
                    )
                    for n in _TWO_GATE_INSTRUCTIONS
                ],
            )
            for i in range(num_qubits)
            for j in range(num_qubits)
            if i != j
        ]

        return TopologyConfig(
            qubits=qubits, single_gates=single_gates, multi_gates=multi_gates
        )

    @staticmethod
    def _build_counts(
        program_results: List[Any], outputs: List[Tuple[int, str]]
    ) -> Dict[str, int]:
        """
        Aggregate per-iteration program results into a measurement histogram.

        Args:
            program_results: One ``ProgramResult`` per simulation iteration.
            outputs: ``(clbit_index, host_var_name)`` pairs returned by the rank.

        Returns:
            ``{bitstring: count}`` ordered by ascending classical-bit index; an
            empty dict if the rank returns no measurements.
        """
        if not outputs:
            return {}
        ordered_vars = [var for _, var in sorted(outputs, key=lambda t: t[0])]
        counts: Dict[str, int] = {}
        for result in program_results:
            bits = "".join(str(int(result.values[var])) for var in ordered_vars)
            counts[bits] = counts.get(bits, 0) + 1
        return counts
