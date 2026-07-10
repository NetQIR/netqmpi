"""
Qoala qdevice hardware sweep: does NetQMPI propagate NetSquid/Qoala hardware
parameters faithfully through the CircuitAdapter translation?

The probe is a distributed-superposition teleportation with X-basis readout
(see ``apps/dist_superposition_xbasis.py``): rank 0 prepares |+>, teleports it to
rank 1, which rotates back to the X basis and measures. The noise-free expected
outcome is a deterministic 0, so fidelity is estimated as ``P(outcome == 0)``.

For every grid point we run the SAME circuit two ways with the SAME qdevice
hardware and a perfect entanglement link:

* **NetQMPI-via-Qoala**: the real ``QoalaExecutorAdapter`` running the NetQMPI
  app (the translation path under test).
* **native Qoala**: hand-written ``.iqoala`` programs (``native/``) run through
  qoala-sim's own runner (independent control).

The maximum |fidelity_netqmpi - fidelity_native| across the grid is the
"translation fidelity" of the adapter; it should stay within the statistical
error implied by the number of shots.

Run inside the ``qoala`` conda env:
    conda run -n qoala python scripts/experiments/qoala_hw_sweep.py --shots 1000
"""
from __future__ import annotations

import argparse
import ast
import contextlib
import csv
import io
import math
import re
import warnings
from pathlib import Path
from typing import Dict, List, Optional

# NetSquid emits scheduling precision warnings when the simulation clock is huge
# (T1/T2 up to 1e12 ns plus the ~1e9 ns classical latency) relative to gate
# intervals. They are benign and do not affect the measured fidelities.
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from netqmpi.runtime.adapters.qoala import (  # noqa: E402
    QoalaExecutorAdapter,
    QoalaRunConfig,
    QoalaQDeviceConfig,
)

HERE = Path(__file__).resolve().parent
APP = HERE / "apps" / "dist_superposition_xbasis.py"
NATIVE_ALICE = HERE / "native" / "dist_superposition_alice.iqoala"
NATIVE_BOB = HERE / "native" / "dist_superposition_bob.iqoala"

NUM_QUBITS = 2          # user qubit + teleportation scratch slot
LINK_DURATION = 1000.0  # ns; perfect link


# ---------------------------------------------------------------------------
# Hardware grid
# ---------------------------------------------------------------------------

def base_hw(**overrides) -> QoalaQDeviceConfig:
    """A perfect qdevice, optionally overriding individual parameters."""
    return QoalaQDeviceConfig(**overrides)


# T1 == T2 swept over orders of magnitude (ns), gates noiseless.
# The fidelity knee sits near the total teleportation-protocol duration
# (dominated by the ~1e9 ns classical correction latency, not the gate times),
# so the grid spans 1e12 (near-perfect memory) down to 1e8 (fully decohered).
T_SWEEP = [1e12, 3e11, 1e11, 3e10, 1e10, 3e9, 1e9, 3e8, 1e8]
# Single-qubit gate depolarising probability from 0 to 0.5, memory perfect.
DEPOLAR_SWEEP = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]


# ---------------------------------------------------------------------------
# Fidelity estimators (both paths, same hardware)
# ---------------------------------------------------------------------------

def fidelity_netqmpi(hw: QoalaQDeviceConfig, shots: int, seed: Optional[int]) -> float:
    """P(outcome == 0) from the real NetQMPI -> Qoala executor path."""
    config = QoalaRunConfig(
        shots=shots, hw_config=hw, seed=seed,
        num_qubits_per_node=NUM_QUBITS, link_duration=LINK_DURATION,
    )
    executor = QoalaExecutorAdapter(2, config=config)
    apps = executor.build_apps(str(APP), size=2)

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        executor.run(apps)

    match = re.search(r"measure:\s*(\{.*\})", buf.getvalue())
    if not match:
        raise RuntimeError(f"no measurement in NetQMPI output:\n{buf.getvalue()}")
    counts: Dict[str, int] = ast.literal_eval(match.group(1))
    total = sum(counts.values())
    return counts.get("0", 0) / total


def fidelity_native(hw: QoalaQDeviceConfig, shots: int, seed: Optional[int]) -> float:
    """P(outcome == 0) from hand-written native Qoala programs (control)."""
    from qoala.lang.parse import QoalaParser
    from qoala.runtime.config import (
        ClassicalConnectionConfig,
        LatenciesConfig,
        NtfConfig,
        ProcNodeConfig,
        ProcNodeNetworkConfig,
    )
    from qoala.runtime.program import ProgramInput
    from qoala.util.runner import run_n_node_app

    # Reuse the executor's topology builder so BOTH paths get identical hardware.
    topology = QoalaExecutorAdapter(
        2, QoalaRunConfig(hw_config=hw, num_qubits_per_node=NUM_QUBITS)
    )._build_topology(NUM_QUBITS)

    def node_cfg(name: str, node_id: int) -> ProcNodeConfig:
        return ProcNodeConfig(
            node_name=name,
            node_id=node_id,
            topology=topology,
            latencies=LatenciesConfig(qnos_instr_time=1000),
            ntf=NtfConfig.from_cls_name("GenericNtf"),
            determ_sched=True,
        )

    nodes = [node_cfg("rank_0", 0), node_cfg("rank_1", 1)]
    network_cfg = ProcNodeNetworkConfig.from_nodes_perfect_links(
        nodes=nodes, link_duration=LINK_DURATION
    )
    network_cfg.cconns = [ClassicalConnectionConfig.from_nodes(0, 1, 1e9)]

    programs = {
        "rank_0": QoalaParser(NATIVE_ALICE.read_text()).parse(),
        "rank_1": QoalaParser(NATIVE_BOB.read_text()).parse(),
    }
    program_inputs = {
        "rank_0": ProgramInput({"peer_1_id": 1}),
        "rank_1": ProgramInput({"peer_0_id": 0}),
    }

    result = run_n_node_app(
        num_iterations=shots,
        programs=programs,
        program_inputs=program_inputs,
        network_cfg=network_cfg,
        linear=True,
    )
    outcomes = [r.values["outcome"] for r in result.batch_results["rank_1"].results]
    return sum(1 for o in outcomes if int(o) == 0) / len(outcomes)


def stderr(fid: float, shots: int) -> float:
    """Standard error of a proportion estimated from ``shots`` samples."""
    return math.sqrt(max(fid * (1.0 - fid), 0.0) / shots)


# ---------------------------------------------------------------------------
# Sweep driver
# ---------------------------------------------------------------------------

def run_sweep(name: str, param: str, values: List[float], make_hw, shots: int,
              seed: Optional[int]) -> List[dict]:
    rows = []
    for v in values:
        hw = make_hw(v)
        f_net = fidelity_netqmpi(hw, shots, seed)
        f_nat = fidelity_native(hw, shots, seed)
        row = {
            "sweep": name,
            "param": param,
            "value": v,
            "fidelity_netqmpi": f_net,
            "fidelity_native": f_nat,
            "abs_diff": abs(f_net - f_nat),
            "stderr_netqmpi": stderr(f_net, shots),
            "stderr_native": stderr(f_nat, shots),
            "shots": shots,
        }
        rows.append(row)
        print(f"[{name}] {param}={v:<10g}  "
              f"F_netqmpi={f_net:.3f}  F_native={f_nat:.3f}  |Δ|={row['abs_diff']:.3f}")
    return rows


def plot_sweep(rows: List[dict], xlabel: str, title: str, out_path: Path,
               logx: bool = False) -> None:
    xs = [r["value"] for r in rows]
    f_net = [r["fidelity_netqmpi"] for r in rows]
    f_nat = [r["fidelity_native"] for r in rows]
    e_net = [r["stderr_netqmpi"] for r in rows]
    e_nat = [r["stderr_native"] for r in rows]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(xs, f_net, yerr=e_net, marker="o", capsize=3, label="NetQMPI → Qoala")
    ax.errorbar(xs, f_nat, yerr=e_nat, marker="s", capsize=3, label="Qoala nativo")
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Fidelidad  P(outcome = 0)")
    ax.set_ylim(0.4, 1.02)
    ax.axhline(0.5, ls="--", lw=0.8, color="gray", label="límite despolarizado (0.5)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shots", type=int, default=1000, help="Shots per grid point.")
    parser.add_argument("--seed", type=int, default=None,
                        help="NetSquid seed for the NetQMPI path (native path reseeds internally).")
    parser.add_argument("--out-dir", type=str, default=str(HERE / "results"),
                        help="Directory for CSV and plots.")
    parser.add_argument("--quick", action="store_true",
                        help="Small grid / few shots for a fast smoke run.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    shots = 100 if args.quick else args.shots
    t_values = [1e8, 1e6, 1e4] if args.quick else T_SWEEP
    d_values = [0.0, 0.25, 0.5] if args.quick else DEPOLAR_SWEEP

    t_rows = run_sweep(
        "T1T2", "t1t2_ns", t_values,
        lambda v: base_hw(t1=v, t2=v), shots, args.seed,
    )
    d_rows = run_sweep(
        "depolar", "single_qubit_gate_depolar_prob", d_values,
        lambda v: base_hw(single_qubit_gate_depolar_prob=v), shots, args.seed,
    )

    all_rows = t_rows + d_rows
    csv_path = out_dir / "qoala_hw_sweep.csv"
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"wrote {csv_path}")

    plot_sweep(t_rows, "T1 = T2 (ns)",
               "Fidelidad de teleportación vs. T1/T2 (puertas perfectas)",
               out_dir / "fidelity_vs_t1t2.png", logx=True)
    plot_sweep(d_rows, "single_qubit_gate_depolar_prob",
               "Fidelidad de teleportación vs. depolarización de puerta 1q (memoria perfecta)",
               out_dir / "fidelity_vs_depolar.png", logx=False)

    max_diff = max(r["abs_diff"] for r in all_rows)
    worst = max(all_rows, key=lambda r: r["abs_diff"])
    # 3-sigma band from combined statistical error at the worst point.
    band = 3.0 * math.hypot(worst["stderr_netqmpi"], worst["stderr_native"])
    print("\n=== Translation-fidelity summary ===")
    print(f"shots per point: {shots}")
    print(f"max |F_netqmpi - F_native| = {max_diff:.4f} "
          f"at {worst['param']}={worst['value']:g}")
    print(f"3σ statistical band at that point ≈ {band:.4f}")
    verdict = "WITHIN" if max_diff <= band else "OUTSIDE"
    print(f"=> max discrepancy is {verdict} the expected statistical error.")


if __name__ == "__main__":
    main()
