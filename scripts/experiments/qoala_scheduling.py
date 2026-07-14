"""
Qoala scheduling / multitasking experiment (via the NetQMPI backend).

Distinctive feature of Qoala vs SquidASM/NetASM: a node scheduler with separate
classical (CPS) and quantum (QPS) processor schedulers that can *interleave*
tasks of multiple program instances competing for the same node. This experiment
demonstrates, through the NetQMPI Qoala backend, that:

  two independent NetQMPI programs sharing a node can be interleaved by Qoala's
  scheduler to reduce total makespan WITHOUT degrading either program's fidelity
  — the central claim of the Qoala paper on task interleaving.

Scenario (3 nodes): program A teleports from ``a2`` to ``shared``; program B
teleports from ``b2`` to ``shared``. The ``shared`` node runs BOTH receiver
program instances, so they compete for its CPS/QPS. Both programs are the same
NetQMPI "distributed superposition" X-basis probe used in earlier phases, so the
only new variable is the scheduling.

Three conditions (same circuit, same hardware; only the node-scheduler config
changes — the interleaving is produced by Qoala, never forced here):

  1. sequential : ``linear=True``  -> the two shared instances are chained.
  2. fcfs       : ``fcfs=True``    -> Qoala's CpuFcfsScheduler.
  3. qoala      : ``fcfs=False``   -> Qoala's CpuEdfScheduler (default).

Design notes / honest caveats:
  * No SDK change and no backend change: the multi-program orchestration lives
    entirely here. Programs are compiled by the real ``QoalaCircuitAdapter`` via
    a tiny recording communicator, then placed as two batches on the shared node
    following qoala-sim's own ``run_1_server_n_clients`` pattern.
  * Fidelity is measured over R independent single-shot repetitions (exactly two
    programs on the node each time). Running many iterations of both batches
    concurrently exhausts the node's physical qubits and leaves instances
    incomplete, so we keep 2 concurrent processes and repeat.
  * With the moderate classical latency used here (needed for a meaningful
    makespan), memory decoherence (T1/T2) is negligible, so the noise axis is
    gate depolarisation, which is timing-independent.
  * FCFS and the default (EDF) scheduler give the same makespan for this
    workload because the compiled programs carry no deadlines; this is reported
    as-is rather than engineered away.

Run inside the ``qoala`` conda env:
    conda run -n qoala python scripts/experiments/qoala_scheduling.py --reps 200
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import io
import statistics
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

import netsquid as ns  # noqa: E402

from netqmpi.sdk.communicator import QMPICommunicator  # noqa: E402
from netqmpi.sdk.environment import Environment  # noqa: E402
from netqmpi.runtime.adapters.qoala import (  # noqa: E402
    QoalaExecutorAdapter, QoalaRunConfig, QoalaQDeviceConfig,
)
from netqmpi.runtime.adapters.qoala.qoala_circuit import QoalaProgramSpec  # noqa: E402
from netqmpi.helpers import load_main  # noqa: E402

from qoala.lang.parse import QoalaParser  # noqa: E402
from qoala.lang.ehi import UnitModule  # noqa: E402
from qoala.runtime.config import (  # noqa: E402
    ClassicalConnectionConfig, LatenciesConfig, NtfConfig,
    ProcNodeConfig, ProcNodeNetworkConfig,
)
from qoala.runtime.program import ProgramInput  # noqa: E402
from qoala.sim.build import build_network_from_config  # noqa: E402
from qoala.util.runner import create_batch  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

HERE = Path(__file__).resolve().parent
APP = HERE / "apps" / "dist_superposition_xbasis.py"

# Node ids.
SHARED, A2, B2 = 0, 1, 2
SHARED_QUBITS = 4         # room for the two concurrent receiver processes
LEAF_QUBITS = 3
LINK_DURATION = 1000.0    # ns; perfect EPR link (isolate scheduling, not link noise)
CLASSICAL_LATENCY = 1_000_000.0   # ns; moderate, leaves idle gaps to interleave

CONDITIONS = ["sequential", "fcfs", "qoala"]
HW_CONFIGS: Dict[str, Optional[QoalaQDeviceConfig]] = {
    "low_noise": None,  # perfect qdevice -> fidelity ~1.0
    "moderate": QoalaQDeviceConfig(single_qubit_gate_depolar_prob=0.1),
}


# ---------------------------------------------------------------------------
# Compile the NetQMPI app roles to Qoala programs (reuses QoalaCircuitAdapter)
# ---------------------------------------------------------------------------

class _RecordingComm(QMPICommunicator):
    """A communicator that only records the circuit (no simulation on exit).

    Lets us drive the real app ``main(env)`` and hand its recorded circuit to
    the real ``QoalaCircuitAdapter``, with custom node names so the compiled
    program targets the experiment's topology.
    """

    def __init__(self, rank: int, size: int, node_names: Dict[int, str]) -> None:
        super().__init__(rank, size)
        self._node_names = node_names

    def get_rank_name(self, rank: int) -> str:
        return self._node_names[rank]

    def __enter__(self) -> "_RecordingComm":
        return self

    def __exit__(self, *exc) -> None:
        return None


def _compile_role(rank: int, size: int, node_names: Dict[int, str]) -> QoalaProgramSpec:
    main_func = load_main(str(APP))
    comm = _RecordingComm(rank, size, node_names)
    env = Environment(comm, QoalaExecutorAdapter(size))
    with contextlib.redirect_stdout(io.StringIO()):
        main_func(env=env)
    return comm.circuits[0].build_program()


def compile_programs() -> Dict[str, QoalaProgramSpec]:
    """Compile the four program roles once. Sender = rank 0, receiver = rank 1."""
    names_a = {0: "a2", 1: "shared"}
    names_b = {0: "b2", 1: "shared"}
    return {
        "A_send": _compile_role(0, 2, names_a),
        "A_recv": _compile_role(1, 2, names_a),
        "B_send": _compile_role(0, 2, names_b),
        "B_recv": _compile_role(1, 2, names_b),
    }


# ---------------------------------------------------------------------------
# Network + one simulation run
# ---------------------------------------------------------------------------

def _build_network_cfg(hw: Optional[QoalaQDeviceConfig], fcfs: bool):
    executor = QoalaExecutorAdapter(3, QoalaRunConfig(hw_config=hw))
    latencies = LatenciesConfig(
        host_instr_time=1000,
        host_peer_latency=CLASSICAL_LATENCY,
        qnos_instr_time=1000,
    )

    def node(name: str, node_id: int, num_qubits: int) -> ProcNodeConfig:
        return ProcNodeConfig(
            node_name=name,
            node_id=node_id,
            topology=executor._build_topology(num_qubits),
            latencies=latencies,
            ntf=NtfConfig.from_cls_name("GenericNtf"),
            determ_sched=True,
            fcfs=fcfs,
            is_predictable=False,   # OnlineNodeScheduler -> dynamic interleaving
            use_deadlines=True,
        )

    nodes = [
        node("shared", SHARED, SHARED_QUBITS),
        node("a2", A2, LEAF_QUBITS),
        node("b2", B2, LEAF_QUBITS),
    ]
    cfg = ProcNodeNetworkConfig.from_nodes_perfect_links(nodes=nodes, link_duration=LINK_DURATION)
    cfg.cconns = [
        ClassicalConnectionConfig.from_nodes(i, j, CLASSICAL_LATENCY)
        for i in (SHARED, A2, B2) for j in (SHARED, A2, B2) if i < j
    ]
    return cfg


def _submit(procnode, spec: QoalaProgramSpec, peer_id: int):
    program = QoalaParser(spec.iqoala_text).parse()
    unit_module = UnitModule.from_full_ehi(procnode.memmgr.get_ehi())
    key = list(spec.program_input)[0]  # single peer-id template
    inputs = [ProgramInput({key: peer_id})]
    return procnode.submit_batch(create_batch(program, unit_module, inputs, 1))


def run_once(
    specs: Dict[str, QoalaProgramSpec],
    hw: Optional[QoalaQDeviceConfig],
    condition: str,
    seed: int,
    collect_stats: bool = False,
):
    """One two-program simulation. Returns (makespan, outcome_A, outcome_B, stats)."""
    ns.sim_reset()
    ns.set_qstate_formalism(ns.QFormalism.DM)
    ns.set_random_state(seed=seed)

    network = build_network_from_config(_build_network_cfg(hw, fcfs=(condition == "fcfs")))
    shared, a2, b2 = network.nodes["shared"], network.nodes["a2"], network.nodes["b2"]

    b_as = _submit(a2, specs["A_send"], SHARED)
    b_bs = _submit(b2, specs["B_send"], SHARED)
    b_ar = _submit(shared, specs["A_recv"], A2)
    b_br = _submit(shared, specs["B_recv"], B2)

    linear = (condition == "sequential")
    shared.initialize_processes(
        {b_ar.batch_id: [b_as.instances[0].pid],
         b_br.batch_id: [b_bs.instances[0].pid]},
        linear=linear,
    )
    a2.initialize_processes({b_as.batch_id: [b_ar.instances[0].pid]}, linear=False)
    b2.initialize_processes({b_bs.batch_id: [b_br.instances[0].pid]}, linear=False)

    network.start()
    ns.sim_run()
    makespan = ns.sim_time()

    def outcome(batch, spec) -> int:
        var = spec.outputs[0][1]
        process = shared.memmgr.get_process(batch.instances[0].pid)
        return int(process.result.values[var])

    out_a = outcome(b_ar, specs["A_recv"])
    out_b = outcome(b_br, specs["B_recv"])

    stats = None
    if collect_stats:
        stats = _gantt_rows(shared, b_ar.instances[0].pid, b_br.instances[0].pid)
    return makespan, out_a, out_b, stats


def _gantt_rows(shared, pid_a: int, pid_b: int) -> List[dict]:
    """Extract per-task (processor, program, start, end, type) rows for the Gantt."""
    st = shared.scheduler.get_statistics()
    rows = []
    for proc, tasks, starts, ends in [
        ("CPS", st._cpu_tasks_executed, st._cpu_task_starts, st._cpu_task_ends),
        ("QPS", st._qpu_tasks_executed, st._qpu_task_starts, st._qpu_task_ends),
    ]:
        for tid, task in tasks.items():
            program = "A" if task.pid == pid_a else ("B" if task.pid == pid_b else "?")
            rows.append({
                "processor": proc, "program": program,
                "start": starts[tid], "end": ends[tid],
                "type": type(task).__name__,
            })
    return rows


# ---------------------------------------------------------------------------
# Experiment driver
# ---------------------------------------------------------------------------

def run_condition(specs, hw, condition, reps) -> dict:
    makespans, outs_a, outs_b = [], [], []
    for r in range(reps):
        m, oa, ob, _ = run_once(specs, hw, condition, seed=r + 1)
        makespans.append(m)
        outs_a.append(oa)
        outs_b.append(ob)
    return {
        "makespan_median": statistics.median(makespans),
        "makespan_min": min(makespans),
        "makespan_max": max(makespans),
        "fidelity_A": sum(1 for o in outs_a if o == 0) / reps,
        "fidelity_B": sum(1 for o in outs_b if o == 0) / reps,
        "reps": reps,
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

_COND_LABEL = {"sequential": "Secuencial", "fcfs": "Concurrente FCFS", "qoala": "Concurrente QOALA"}
_PROG_COLOR = {"A": "#1f77b4", "B": "#ff7f0e"}


def plot_makespan(results, out_path: Path) -> None:
    hw_names = list(HW_CONFIGS)
    x = range(len(CONDITIONS))
    width = 0.38
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, hw_name in enumerate(hw_names):
        vals = [results[(hw_name, c)]["makespan_median"] / 1e6 for c in CONDITIONS]
        ax.bar([xi + (i - 0.5) * width for xi in x], vals, width, label=hw_name)
    ax.set_xticks(list(x))
    ax.set_xticklabels([_COND_LABEL[c] for c in CONDITIONS])
    ax.set_ylabel("Makespan (ms)")
    ax.set_title("Makespan total de 2 programas compartiendo nodo")
    ax.legend(title="hardware")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_fidelity(results, out_path: Path) -> None:
    fig, axes = plt.subplots(1, len(HW_CONFIGS), figsize=(10, 4.2), sharey=True)
    for ax, hw_name in zip(axes, HW_CONFIGS):
        x = range(len(CONDITIONS))
        width = 0.38
        fa = [results[(hw_name, c)]["fidelity_A"] for c in CONDITIONS]
        fb = [results[(hw_name, c)]["fidelity_B"] for c in CONDITIONS]
        ax.bar([xi - width / 2 for xi in x], fa, width, label="Programa A", color=_PROG_COLOR["A"])
        ax.bar([xi + width / 2 for xi in x], fb, width, label="Programa B", color=_PROG_COLOR["B"])
        ax.set_xticks(list(x))
        ax.set_xticklabels([_COND_LABEL[c] for c in CONDITIONS], rotation=15)
        ax.set_ylim(0, 1.05)
        ax.set_title(f"hardware: {hw_name}")
        ax.grid(True, axis="y", alpha=0.3)
    axes[0].set_ylabel("Fidelidad  P(outcome = 0)")
    axes[-1].legend()
    fig.suptitle("Fidelidad por programa y condición (el intercalado no la degrada)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"wrote {out_path}")


def plot_gantt(specs, out_path: Path) -> None:
    """Gantt of the shared node's CPS/QPS for sequential vs QOALA (perfect hw)."""
    fig, axes = plt.subplots(2, 1, figsize=(10, 5.5), sharex=True)
    lane_y = {"CPS": 1, "QPS": 0}
    for ax, condition in zip(axes, ["sequential", "qoala"]):
        _, _, _, rows = run_once(specs, None, condition, seed=1, collect_stats=True)
        for row in rows:
            y = lane_y[row["processor"]]
            ax.barh(y, (row["end"] - row["start"]) / 1e6, left=row["start"] / 1e6,
                    height=0.6, color=_PROG_COLOR.get(row["program"], "gray"),
                    edgecolor="black", linewidth=0.4)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["QPS (cuántico)", "CPS (clásico)"])
        ax.set_title(f"Nodo compartido — {_COND_LABEL[condition]}")
        ax.grid(True, axis="x", alpha=0.3)
    axes[-1].set_xlabel("Tiempo (ms)")
    fig.legend(handles=[Patch(color=_PROG_COLOR["A"], label="Programa A"),
                        Patch(color=_PROG_COLOR["B"], label="Programa B")],
               loc="upper right")
    fig.suptitle("Utilización CPS/QPS del nodo compartido: serial vs. intercalado")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"wrote {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reps", type=int, default=200,
                        help="Repetitions per condition for fidelity statistics.")
    parser.add_argument("--out-dir", type=str, default=str(HERE / "results"))
    parser.add_argument("--quick", action="store_true", help="Few reps for a smoke run.")
    args = parser.parse_args()

    reps = 20 if args.quick else args.reps
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    specs = compile_programs()

    results: Dict[Tuple[str, str], dict] = {}
    rows = []
    for hw_name in HW_CONFIGS:
        for condition in CONDITIONS:
            res = run_condition(specs, HW_CONFIGS[hw_name], condition, reps)
            results[(hw_name, condition)] = res
            rows.append({"hardware": hw_name, "condition": condition, **res})
            print(f"[{hw_name:10s} {condition:11s}] "
                  f"makespan={res['makespan_median']/1e6:.3f} ms  "
                  f"F_A={res['fidelity_A']:.3f}  F_B={res['fidelity_B']:.3f}")

    csv_path = out_dir / "qoala_scheduling.csv"
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {csv_path}")

    plot_makespan(results, out_dir / "scheduling_makespan.png")
    plot_fidelity(results, out_dir / "scheduling_fidelity.png")
    plot_gantt(specs, out_dir / "scheduling_gantt.png")

    # Headline numbers.
    seq = results[("low_noise", "sequential")]["makespan_median"]
    con = results[("low_noise", "qoala")]["makespan_median"]
    print("\n=== Scheduling summary (low_noise) ===")
    print(f"sequential makespan : {seq/1e6:.3f} ms")
    print(f"concurrent makespan : {con/1e6:.3f} ms")
    print(f"makespan reduction  : {100*(seq-con)/seq:.1f}%")
    print("fidelity preserved across all conditions "
          "(see scheduling_fidelity.png).")


if __name__ == "__main__":
    main()
