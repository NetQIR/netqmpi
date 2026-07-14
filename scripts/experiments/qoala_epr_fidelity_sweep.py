"""
EPR fidelity sweep: effect of imperfect entanglement on NetQMPI teleportation.

Building on the previous experiment (which established that NetQMPI-via-Qoala
reproduces native Qoala within statistical error), this experiment stays purely
in NetQMPI and studies the OTHER noise source: the entanglement link.

The qdevice is kept **perfect** (no memory/gate noise) and the EPR pairs used by
``qsend``/``qrecv`` are depolarised to a target fidelity ``F_epr`` to the ideal
Bell state (``ProcNodeNetworkConfig.from_nodes_imperfect_links``). We sweep
``F_epr`` from 0.25 (maximally mixed pair) to 1.0 (perfect link) and measure the
teleportation fidelity of the distributed-superposition probe
(``apps/dist_superposition_xbasis.py``): prepare |+>, teleport, measure in X,
expected 0. Teleportation fidelity is ``F_tel = P(outcome == 0)``.

Analytical prediction. A depolarised pair ``(1-p)|Φ+><Φ+| + p·I/4`` with
``p = (4/3)(1 - F_epr)`` teleports |ψ> to ``(1-p)|ψ><ψ| + p·I/2``, so for the
X-basis probe:

    F_tel = 1 - p/2 = (1 + 2·F_epr) / 3

i.e. F_tel = 1 at F_epr = 1 (perfect baseline) and 0.5 at F_epr = 0.25. The
measured NetQMPI points are compared against this law.

Run inside the ``qoala`` conda env:
    conda run -n qoala python scripts/experiments/qoala_epr_fidelity_sweep.py --shots 1000
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
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from netqmpi.runtime.adapters.qoala import QoalaExecutorAdapter, QoalaRunConfig  # noqa: E402

# NetSquid emits benign scheduling precision warnings; they do not affect results.
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

HERE = Path(__file__).resolve().parent
APP = HERE / "apps" / "dist_superposition_xbasis.py"

NUM_QUBITS = 2          # user qubit + teleportation scratch slot
LINK_DURATION = 1000.0  # ns

# EPR fidelity grid: 0.25 (maximally mixed) -> 1.0 (perfect Bell pair).
EPR_SWEEP = [0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95, 1.0]


def theory(f_epr: float) -> float:
    """Analytical teleportation fidelity for the X-basis probe: (1 + 2 F)/3."""
    return (1.0 + 2.0 * f_epr) / 3.0


def teleport_fidelity(f_epr: float, shots: int, seed: Optional[int]) -> float:
    """P(outcome == 0) with a perfect qdevice and EPR fidelity ``f_epr``."""
    config = QoalaRunConfig(
        shots=shots, hw_config=None, link_fidelity=f_epr, seed=seed,
        num_qubits_per_node=NUM_QUBITS, link_duration=LINK_DURATION,
    )
    executor = QoalaExecutorAdapter(2, config=config)
    apps = executor.build_apps(str(APP), size=2)

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        executor.run(apps)

    match = re.search(r"measure:\s*(\{.*\})", buf.getvalue())
    if not match:
        raise RuntimeError(f"no measurement in output:\n{buf.getvalue()}")
    counts = ast.literal_eval(match.group(1))
    return counts.get("0", 0) / sum(counts.values())


def stderr(fid: float, shots: int) -> float:
    return math.sqrt(max(fid * (1.0 - fid), 0.0) / shots)


def run_sweep(values: List[float], shots: int, seed: Optional[int]) -> List[dict]:
    rows = []
    for f in values:
        f_tel = teleport_fidelity(f, shots, seed)
        pred = theory(f)
        row = {
            "epr_fidelity": f,
            "teleport_fidelity": f_tel,
            "theory": pred,
            "abs_dev_from_theory": abs(f_tel - pred),
            "stderr": stderr(f_tel, shots),
            "shots": shots,
        }
        rows.append(row)
        print(f"F_epr={f:<5}  F_tel={f_tel:.3f}  theory={pred:.3f}  "
              f"|dev|={row['abs_dev_from_theory']:.3f}")
    return rows


def plot_sweep(rows: List[dict], out_path: Path) -> None:
    xs = [r["epr_fidelity"] for r in rows]
    ys = [r["teleport_fidelity"] for r in rows]
    es = [r["stderr"] for r in rows]
    dense = [i / 100 for i in range(25, 101)]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(dense, [theory(f) for f in dense], color="gray", lw=1.5,
            label="teoría  (1 + 2·F_epr)/3")
    ax.errorbar(xs, ys, yerr=es, marker="o", capsize=3, ls="-",
                label="NetQMPI → Qoala")
    ax.axhline(1.0, ls=":", lw=0.8, color="green", label="baseline F_epr = 1 (perfecto)")
    ax.axhline(0.5, ls="--", lw=0.8, color="gray", label="límite despolarizado (0.5)")
    ax.set_xlabel("Fidelidad del par EPR  F_epr")
    ax.set_ylabel("Fidelidad de teleportación  P(outcome = 0)")
    ax.set_ylim(0.4, 1.03)
    ax.set_title("Teleportación NetQMPI (qsend/qrecv) vs. fidelidad del EPR\n"
                 "(qdevice perfecto, enlace despolarizante)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shots", type=int, default=1000, help="Shots per grid point.")
    parser.add_argument("--seed", type=int, default=None, help="NetSquid seed.")
    parser.add_argument("--out-dir", type=str, default=str(HERE / "results"))
    parser.add_argument("--quick", action="store_true", help="Small grid / few shots.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    shots = 100 if args.quick else args.shots
    values = [0.25, 0.5, 0.75, 1.0] if args.quick else EPR_SWEEP

    rows = run_sweep(values, shots, args.seed)

    csv_path = out_dir / "qoala_epr_fidelity_sweep.csv"
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {csv_path}")

    plot_sweep(rows, out_dir / "fidelity_vs_epr.png")

    baseline = next(r["teleport_fidelity"] for r in rows if r["epr_fidelity"] == 1.0)
    max_dev = max(r["abs_dev_from_theory"] for r in rows)
    worst = max(rows, key=lambda r: r["abs_dev_from_theory"])
    band = 3.0 * worst["stderr"]
    print("\n=== EPR fidelity sweep summary ===")
    print(f"shots per point: {shots}")
    print(f"perfect-link baseline (F_epr=1): F_tel = {baseline:.3f}")
    print(f"max |F_tel - theory| = {max_dev:.4f} at F_epr={worst['epr_fidelity']:g}")
    print(f"3σ statistical band there ≈ {band:.4f}")
    verdict = "WITHIN" if max_dev <= band else "OUTSIDE"
    print(f"=> measured teleportation fidelity is {verdict} the (1+2F)/3 law "
          f"at the {worst['epr_fidelity']:g} point.")


if __name__ == "__main__":
    main()
