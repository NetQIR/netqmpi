"""
Aggregate the raw benchmark records, fit the cost model, and plot it.

The measurements answer three questions, and this script produces one
artefact for each.

**What does NetQMPI itself cost?** The trace and the translation are the
abstraction; everything else belongs to the platform. Both are expected to
be affine in the number of recorded operations — a fixed price to enter the
machinery plus a price per operation — so the model fitted here is

    t_netqmpi(G) = alpha + beta * G

with ``G`` the operations the trace recorded, ``alpha`` in milliseconds and
``beta`` in microseconds per operation. The same shape is fitted to the
Python-side peak memory. Because both coefficients are *constants* while a
simulator's cost grows with the width of the register, the model can be
solved for the point where the abstraction stops mattering: the crossover
where NetQMPI falls below 1% of the run.

**What does the backend cost?** Simulation time is fitted as an exponential
in the total number of qubits, ``t_backend = a * exp(b * Q)``, by regressing
its logarithm — the statevector behaviour that makes the crossover exist.

**Does the same program mean the same thing everywhere?** The portability
matrix and the fidelity table come straight from the records: every
``(app, backend)`` cell reports whether the run completed, refused, or
completed with the wrong answer.

Run it in any environment that has numpy and matplotlib::

    python scripts/benchmark/analyze.py [--raw results/raw.jsonl]
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402

BENCH_DIR = Path(__file__).resolve().parent
RESULTS = BENCH_DIR / "results"

#: Order used everywhere a backend is shown, and their plot colours.
BACKENDS = ["aer", "cunqa", "qoala", "netqasm"]
COLOURS = {"aer": "#4C72B0", "cunqa": "#DD8452",
           "qoala": "#55A868", "netqasm": "#C44E52"}

#: Phases of a run, innermost cost first, as stacked in the breakdown plot.
PHASES = [
    ("t_trace", "trace (SDK)", "#8172B3"),
    ("t_translate", "translate (adapter)", "#937860"),
    ("t_import", "import", "#DA8BC3"),
    ("t_setup", "setup", "#8C8C8C"),
    ("t_backend", "backend", "#CCB974"),
]

#: Overhead share below which the abstraction is considered irrelevant.
CROSSOVER_TARGET = 0.01

#: Backends that simulate without a noise model, so anything short of a
#: perfect echo is a translation bug rather than decoherence. Aer runs an
#: ideal statevector and its swap-based transfer inserts no noise at all;
#: CUNQA's Munich simulator likewise runs ideal. Qoala and NetQASM drive
#: NetSquid with real hardware models, where F < 1 is physics, not a defect.
NOISELESS = frozenset({"aer", "cunqa"})


# ----------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------

def load(path: Path) -> List[Dict[str, Any]]:
    """
    Read the JSONL produced by the runner.

    Args:
        path: Path to the raw records.

    Returns:
        Every record, in file order.

    Raises:
        SystemExit: If the file does not exist.
    """
    if not path.is_file():
        raise SystemExit(f"No raw records at {path}. Run run_all.sh first.")
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def timing_records(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Return the successful timing runs, excluding the memory passes.

    ``tracemalloc`` roughly doubles the cost of a trace, so a record taken
    with it on is unusable for timing and is kept only for its memory
    figure.

    Args:
        records: All records.

    Returns:
        The records fit to be timed.
    """
    return [r for r in records
            if r.get("status") == "ok" and not r.get("memory_pass")]


def by_config(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Collapse the repetitions of each configuration to their median.

    Fitting against every repetition lets a configuration that happened to
    run more times pull the regression, and it charges the fit with
    rep-to-rep jitter that the model is not trying to explain. One point per
    configuration is what the model is actually about.

    Args:
        records: Timing records.

    Returns:
        One record per configuration, with the timing fields replaced by
        their median over the repetitions.
    """
    groups: Dict[Tuple, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[(record["backend"], record["app"], record["ranks"],
                record["qubits_per_rank"], record["shots"])].append(record)

    fields = ["t_import", "t_setup", "t_trace", "t_translate", "t_backend",
              "t_total", "t_netqmpi", "overhead_fraction"]
    collapsed = []
    for group in groups.values():
        merged = dict(group[0])
        for field in fields:
            merged[field] = float(np.median([r[field] for r in group]))
        merged["reps_merged"] = len(group)
        collapsed.append(merged)
    return collapsed


def warm(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Return the steady-state repetitions.

    The first repetition of a process pays for imports and for whatever
    each library initialises lazily on first use; it is reported separately
    as the cold-start cost rather than mixed into the model.

    Args:
        records: Timing records.

    Returns:
        Records from repetitions after the first, or every record when a
        configuration only ever ran once.
    """
    later = [r for r in records if r.get("rep", 0) > 0]
    return later or list(records)


# ----------------------------------------------------------------------
# Model fitting
# ----------------------------------------------------------------------

def fit_affine(x: Sequence[float], y: Sequence[float]) -> Optional[Dict[str, float]]:
    """
    Least-squares fit of ``y = alpha + beta * x``.

    Args:
        x: Independent variable.
        y: Dependent variable.

    Returns:
        A mapping with ``alpha``, ``beta``, ``r2`` and ``n``, or ``None``
        when there are too few distinct points to fit.
    """
    x_array, y_array = np.asarray(x, float), np.asarray(y, float)
    if x_array.size < 3 or np.unique(x_array).size < 2:
        return None

    design = np.vstack([np.ones_like(x_array), x_array]).T
    (alpha, beta), *_ = np.linalg.lstsq(design, y_array, rcond=None)

    residual = y_array - (alpha + beta * x_array)
    total = y_array - y_array.mean()
    r2 = 1.0 - float(residual @ residual) / float(total @ total) if total.any() else 1.0
    return {"alpha": float(alpha), "beta": float(beta), "r2": r2,
            "n": int(x_array.size)}


def fit_split(local: Sequence[float], comm: Sequence[float],
              y: Sequence[float]) -> Optional[Dict[str, float]]:
    """
    Least-squares fit of ``y = alpha + beta * G_local + gamma * G_comm``.

    Treating every recorded operation as equally expensive is the obvious
    first model and a poor one: a local gate translates into one native
    instruction, whereas ``qsend`` expands into a whole teleportation
    protocol — entanglement, a Bell measurement and its corrections. Giving
    the communication primitives their own coefficient separates the two,
    and the ratio ``gamma / beta`` is a directly useful number: how many
    local gates a single act of communication costs to translate.

    Args:
        local: Local operations per configuration.
        comm: Communication primitives per configuration.
        y: Measured cost.

    Returns:
        A mapping with ``alpha``, ``beta``, ``gamma``, ``r2`` and ``n``, or
        ``None`` when the two regressors cannot be told apart.
    """
    local_array = np.asarray(local, float)
    comm_array = np.asarray(comm, float)
    y_array = np.asarray(y, float)
    if local_array.size < 4:
        return None

    design = np.vstack([np.ones_like(local_array), local_array, comm_array]).T
    if np.linalg.matrix_rank(design) < 3:
        return None

    (alpha, beta, gamma), *_ = np.linalg.lstsq(design, y_array, rcond=None)
    residual = y_array - design @ np.array([alpha, beta, gamma])
    total = y_array - y_array.mean()
    r2 = 1.0 - float(residual @ residual) / float(total @ total) if total.any() else 1.0
    return {"alpha": float(alpha), "beta": float(beta), "gamma": float(gamma),
            "r2": r2, "n": int(local_array.size)}


def fit_exponential(x: Sequence[float], y: Sequence[float]) -> Optional[Dict[str, float]]:
    """
    Least-squares fit of ``y = a * exp(b * x)`` through ``log y``.

    Args:
        x: Independent variable.
        y: Dependent variable; non-positive values are dropped.

    Returns:
        A mapping with ``a``, ``b``, ``r2`` and ``n``, or ``None`` when
        there are too few usable points.
    """
    x_array, y_array = np.asarray(x, float), np.asarray(y, float)
    keep = y_array > 0
    x_array, y_array = x_array[keep], y_array[keep]
    if x_array.size < 3 or np.unique(x_array).size < 2:
        return None

    fit = fit_affine(x_array, np.log(y_array))
    if fit is None:
        return None
    return {"a": float(math.exp(fit["alpha"])), "b": fit["beta"],
            "r2": fit["r2"], "n": fit["n"]}


def crossover_ops(overhead: Dict[str, float],
                  backend_seconds_per_op: float) -> Optional[float]:
    """
    Solve for the workload at which NetQMPI drops below the target share.

    With ``t_netqmpi = alpha + beta * G`` and a backend cost that grows at
    ``gamma`` per operation, the share falls below ``target`` once

        alpha + beta * G <= target * (alpha + beta * G + gamma * G)

    Args:
        overhead: The affine fit of the NetQMPI cost.
        backend_seconds_per_op: Slope of the backend cost in the same units.

    Returns:
        The number of operations at the crossover, or ``None`` when the
        backend never outgrows the abstraction.
    """
    alpha, beta = overhead["alpha"], overhead["beta"]
    target = CROSSOVER_TARGET
    denominator = target * backend_seconds_per_op - (1 - target) * beta
    if denominator <= 0:
        return None
    return (1 - target) * alpha / denominator


def build_model(records: Sequence[Dict[str, Any]],
                all_records: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Fit every model the report needs, per backend and pooled.

    Args:
        records: Successful timing records, used for every timing fit.
        all_records: Every record, including the memory passes — those are
            excluded from the timing set because ``tracemalloc`` distorts it,
            but they are the only ones carrying a Python peak.

    Returns:
        A nested mapping of fitted coefficients, keyed by backend with a
        ``"pooled"`` entry covering all of them.
    """
    model: Dict[str, Any] = {}
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[record["backend"]].append(record)
    groups["pooled"] = list(records)

    for name, group in groups.items():
        steady = by_config(warm(group))
        if not steady:
            continue

        ops = [r["ops_total"] for r in steady]
        entry: Dict[str, Any] = {
            "runs": len(steady),
            "t_netqmpi_vs_ops": fit_affine(ops, [r["t_netqmpi"] for r in steady]),
            "t_trace_vs_ops": fit_affine(ops, [r["t_trace"] for r in steady]),
            "t_translate_vs_ops": fit_affine(
                ops, [r["t_translate"] for r in steady]),
            "t_backend_vs_ops": fit_affine(ops, [r["t_backend"] for r in steady]),
            "t_netqmpi_split": fit_split(
                [r["ops_local"] for r in steady],
                [r["ops_comm"] for r in steady],
                [r["t_netqmpi"] for r in steady]),
            "t_backend_vs_qubits": fit_exponential(
                [r["ranks"] * r["qubits_per_rank"] for r in steady],
                [r["t_backend"] for r in steady]),
        }

        cold = [r for r in group if r.get("rep", 0) == 0]
        if cold:
            entry["cold_start_import_s"] = float(
                np.median([r["t_import"] for r in cold]))
            entry["cold_start_setup_s"] = float(
                np.median([r["t_setup"] for r in cold]))

        memory = by_config([r for r in all_records
                            if r.get("status") == "ok" and r.get("py_peak_bytes")
                            and (name == "pooled" or r["backend"] == name)])
        if memory:
            entry["py_peak_vs_ops"] = fit_affine(
                [r["ops_total"] for r in memory],
                [r["py_peak_bytes"] for r in memory])

        overhead = entry["t_netqmpi_vs_ops"]
        backend_fit = entry["t_backend_vs_ops"]
        if overhead and backend_fit and backend_fit["beta"] > 0:
            entry["crossover_ops_1pct"] = crossover_ops(
                overhead, backend_fit["beta"])

        model[name] = entry

    return model


# ----------------------------------------------------------------------
# Tables
# ----------------------------------------------------------------------

def portability_table(records: Sequence[Dict[str, Any]]) -> str:
    """
    Render the app-by-backend support matrix as Markdown.

    A cell reports the **worst** outcome seen for that pair, not the best: a
    program that is right on two ranks and wrong on four is not portable,
    and taking the best would hide exactly the defects this table exists to
    surface.

    Whether a low fidelity counts as a defect depends on the backend. On a
    noiseless one (:data:`NOISELESS`) anything short of a perfect echo is a
    translation bug. On Qoala and NetQASM, which drive NetSquid with real
    hardware models, it is decoherence, so the figure is reported without a
    verdict attached.

    Args:
        records: All records.

    Returns:
        A Markdown table.
    """
    unsupported: Dict[Tuple[str, str], str] = {}
    fidelities: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    other: Dict[Tuple[str, str], str] = {}

    for record in records:
        key = (record["app"], record["backend"])
        status = record.get("status")
        if status == "ok":
            value = record.get("fidelity")
            if value is not None and not math.isnan(value):
                fidelities[key].append(float(value))
        elif status == "unsupported":
            unsupported[key] = record.get("error", "not implemented")
        else:
            other.setdefault(key, status or "error")

    apps = sorted({app for app, _ in
                   list(unsupported) + list(fidelities) + list(other)})

    lines = ["| app | " + " | ".join(BACKENDS) + " |",
             "|---|" + "---|" * len(BACKENDS)]
    for app in apps:
        row = [app]
        for backend in BACKENDS:
            key = (app, backend)
            if key in unsupported:
                row.append("n/i")
            elif key in fidelities:
                worst, best = min(fidelities[key]), max(fidelities[key])
                if worst > 0.99:
                    row.append("OK")
                elif backend in NOISELESS:
                    row.append(f"**WRONG** {worst:.2f}–{best:.2f}")
                else:
                    row.append(f"F={worst:.2f}–{best:.2f}")
            elif key in other:
                row.append(other[key])
            else:
                row.append("–")
        lines.append("| " + " | ".join(row) + " |")

    lines.append("")
    lines.append("`OK` = echo exact on every configuration tried. "
                 "`n/i` = the adapter raises `NotImplementedError`. "
                 "`WRONG` = completed but returned the wrong answer on a "
                 "backend that has no noise model. A bare `F=` range is "
                 "decoherence on a backend that models hardware.")
    return "\n".join(lines)


def overhead_table(records: Sequence[Dict[str, Any]]) -> str:
    """
    Render the per-backend overhead summary as Markdown.

    Both regimes are reported because neither is the whole truth. *Cold* is
    the first repetition in a process: what a user actually pays to run a
    program once, imports included. *Warm* is the steady state, which is
    what the model is fitted on and what a repeated invocation inside one
    process would cost.

    Args:
        records: Successful timing records.

    Returns:
        A Markdown table.
    """
    lines = ["| backend | configs | cold total | warm total | warm NetQMPI | "
             "warm share | median ops |",
             "|---|---|---|---|---|---|---|"]

    for backend in BACKENDS:
        group = [r for r in records if r["backend"] == backend]
        if not group:
            continue
        steady = by_config(warm(group))
        cold = by_config([r for r in group if r.get("rep", 0) == 0])
        if not steady:
            continue
        lines.append(
            "| {} | {} | {:.3f} s | {:.4f} s | {:.2f} ms | {:.2f}% | {:.0f} |".format(
                backend, len(steady),
                float(np.median([r["t_total"] for r in cold])) if cold else float("nan"),
                float(np.median([r["t_total"] for r in steady])),
                float(np.median([r["t_netqmpi"] for r in steady])) * 1e3,
                float(np.median([r["overhead_fraction"] for r in steady])) * 100,
                float(np.median([r["ops_total"] for r in steady]))))

    return "\n".join(lines)


def model_table(model: Dict[str, Any]) -> str:
    """
    Render the fitted cost model as Markdown.

    Args:
        model: Output of :func:`build_model`.

    Returns:
        A Markdown table.
    """
    lines = ["| backend | alpha (ms) | beta (us/op) | R^2 | configs |",
             "|---|---|---|---|---|"]
    thin = False
    for name in BACKENDS + ["pooled"]:
        fit = (model.get(name) or {}).get("t_netqmpi_vs_ops")
        if not fit:
            continue
        # Two free parameters: with barely more points than that, R^2 is a
        # statement about the arithmetic rather than about the model.
        mark = " †" if fit["n"] < 5 else ""
        thin = thin or bool(mark)
        lines.append("| {} | {:.3f} | {:.2f} | {:.3f}{} | {} |".format(
            name, fit["alpha"] * 1e3, fit["beta"] * 1e6, fit["r2"], mark,
            fit["n"]))
    if thin:
        lines += ["", "† fitted on fewer than 5 configurations against 2 free "
                      "parameters: the coefficients are indicative, the R^2 is "
                      "not evidence."]

    lines += ["", "Splitting local gates from communication primitives, "
                  "`t = alpha + beta*G_local + gamma*G_comm`:", "",
              "| backend | alpha (ms) | beta (us/local op) | "
              "gamma (us/comm op) | gamma/beta | R^2 |",
              "|---|---|---|---|---|---|"]
    for name in BACKENDS + ["pooled"]:
        fit = (model.get(name) or {}).get("t_netqmpi_split")
        if not fit:
            continue
        ratio = fit["gamma"] / fit["beta"] if fit["beta"] else float("nan")
        lines.append("| {} | {:.3f} | {:.2f} | {:.2f} | {:.1f}x | {:.3f} |".format(
            name, fit["alpha"] * 1e3, fit["beta"] * 1e6,
            fit["gamma"] * 1e6, ratio, fit["r2"]))
    return "\n".join(lines)


# ----------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------

def plot_phase_breakdown(records: Sequence[Dict[str, Any]], path: Path) -> None:
    """
    Stacked bars of where the wall-clock goes, per backend and app.

    Args:
        records: Successful timing records.
        path: Destination PNG.
    """
    labels, stacks = [], []
    for backend in BACKENDS:
        for app in sorted({r["app"] for r in records if r["backend"] == backend}):
            group = warm([r for r in records
                          if r["backend"] == backend and r["app"] == app])
            if not group:
                continue
            labels.append(f"{backend}\n{app}")
            stacks.append([float(np.median([r[key] for r in group]))
                           for key, _, _ in PHASES])

    if not stacks:
        return

    stacks_array = np.asarray(stacks)
    figure, axes = plt.subplots(figsize=(max(8, len(labels) * 1.1), 5))
    bottom = np.zeros(len(labels))
    for index, (_, label, colour) in enumerate(PHASES):
        axes.bar(labels, stacks_array[:, index], bottom=bottom,
                 label=label, color=colour)
        bottom += stacks_array[:, index]

    axes.set_ylabel("wall-clock (s)")
    axes.set_yscale("log")
    axes.set_title("Where the time goes: NetQMPI phases vs backend execution")
    axes.legend(fontsize=8, ncol=2)
    axes.tick_params(axis="x", labelsize=7)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def plot_overhead_model(records: Sequence[Dict[str, Any]],
                        model: Dict[str, Any], path: Path) -> None:
    """
    NetQMPI cost against workload size, with the fitted affine model.

    Args:
        records: Successful timing records.
        model: Output of :func:`build_model`.
        path: Destination PNG.
    """
    figure, axes = plt.subplots(figsize=(7, 5))

    for backend in BACKENDS:
        group = warm([r for r in records if r["backend"] == backend])
        if not group:
            continue
        axes.scatter([r["ops_total"] for r in group],
                     [r["t_netqmpi"] * 1e3 for r in group],
                     s=22, alpha=0.7, label=backend, color=COLOURS[backend])

    fit = (model.get("pooled") or {}).get("t_netqmpi_vs_ops")
    if fit:
        ops = np.array([r["ops_total"] for r in records], float)
        grid = np.linspace(0, ops.max() * 1.05, 100)
        axes.plot(grid, (fit["alpha"] + fit["beta"] * grid) * 1e3, "k--",
                  linewidth=1.5,
                  label=(r"$t = {:.2f}\,\mathrm{{ms}} + {:.2f}\,\mu s \times G$"
                         "\n" r"($R^2={:.3f}$)").format(
                             fit["alpha"] * 1e3, fit["beta"] * 1e6, fit["r2"]))

    axes.set_xlabel("operations recorded by the trace, $G$")
    axes.set_ylabel("NetQMPI time (trace + translate), ms")
    axes.set_title("The cost of the abstraction is affine in the workload")
    axes.legend(fontsize=8)
    axes.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def plot_overhead_share(records: Sequence[Dict[str, Any]], path: Path) -> None:
    """
    NetQMPI's share of the run against the total, showing the 1% line.

    Args:
        records: Successful timing records.
        path: Destination PNG.
    """
    figure, axes = plt.subplots(figsize=(7, 5))

    for backend in BACKENDS:
        group = warm([r for r in records if r["backend"] == backend])
        if not group:
            continue
        axes.scatter([r["t_total"] for r in group],
                     [r["overhead_fraction"] * 100 for r in group],
                     s=26, alpha=0.75, label=backend, color=COLOURS[backend])

    axes.axhline(CROSSOVER_TARGET * 100, color="k", linestyle="--", linewidth=1)
    axes.text(0.02, CROSSOVER_TARGET * 100 * 1.15, "1% of the run",
              transform=axes.get_yaxis_transform(), fontsize=8)
    axes.set_xscale("log")
    axes.set_yscale("log")
    axes.set_xlabel("total wall-clock of the run (s)")
    axes.set_ylabel("NetQMPI share (%)")
    axes.set_title("The abstraction stops mattering as the simulation grows")
    axes.legend(fontsize=8)
    axes.grid(alpha=0.3, which="both")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def plot_memory(records: Sequence[Dict[str, Any]],
                model: Dict[str, Any], path: Path) -> None:
    """
    Python-side peak memory against workload size, with the fitted model.

    Args:
        records: All records; only the memory passes carry a peak.
        model: Output of :func:`build_model`.
        path: Destination PNG.
    """
    group = [r for r in records
             if r.get("status") == "ok" and r.get("py_peak_bytes")]
    if not group:
        return

    figure, axes = plt.subplots(figsize=(7, 5))
    for backend in BACKENDS:
        subset = [r for r in group if r["backend"] == backend]
        if not subset:
            continue
        axes.scatter([r["ops_total"] for r in subset],
                     [r["py_peak_bytes"] / 1024 for r in subset],
                     s=24, alpha=0.75, label=backend, color=COLOURS[backend])

    fit = (model.get("pooled") or {}).get("py_peak_vs_ops")
    if fit:
        ops = np.array([r["ops_total"] for r in group], float)
        grid = np.linspace(0, ops.max() * 1.05, 100)
        axes.plot(grid, (fit["alpha"] + fit["beta"] * grid) / 1024, "k--",
                  linewidth=1.5,
                  label=(r"$M = {:.0f}\,\mathrm{{KiB}} + {:.0f}\,\mathrm{{B}}"
                         r"\times G$" "\n" r"($R^2={:.3f}$)").format(
                             fit["alpha"] / 1024, fit["beta"], fit["r2"]))

    axes.set_xlabel("operations recorded by the trace, $G$")
    axes.set_ylabel("Python peak allocation (KiB)")
    axes.set_title("Memory held by the operation stream is affine too")
    axes.legend(fontsize=8)
    axes.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def plot_fidelity(records: Sequence[Dict[str, Any]], path: Path) -> None:
    """
    Fidelity of every app on every backend that ran it.

    Args:
        records: All records.
        path: Destination PNG.
    """
    group = [r for r in records
             if r.get("status") == "ok" and r.get("fidelity") is not None
             and not math.isnan(r["fidelity"])]
    if not group:
        return

    apps = sorted({r["app"] for r in group})
    figure, axes = plt.subplots(figsize=(max(7, len(apps) * 1.8), 4.5))
    width = 0.8 / max(1, len(BACKENDS))

    for index, backend in enumerate(BACKENDS):
        heights, positions = [], []
        for app_index, app in enumerate(apps):
            subset = [r["fidelity"] for r in group
                      if r["backend"] == backend and r["app"] == app]
            if not subset:
                continue
            heights.append(float(np.median(subset)))
            positions.append(app_index + index * width - 0.4 + width / 2)
        if heights:
            axes.bar(positions, heights, width=width,
                     label=backend, color=COLOURS[backend])

    axes.axhline(1.0, color="k", linestyle=":", linewidth=1)
    axes.set_xticks(range(len(apps)))
    axes.set_xticklabels(apps)
    axes.set_ylim(0, 1.15)
    axes.set_ylabel("fidelity  P(all-zero echo)")
    axes.set_title("Same program, four backends: does it still mean the same?")
    axes.legend(fontsize=8)
    axes.grid(alpha=0.3, axis="y")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", default=str(RESULTS / "raw.jsonl"))
    parser.add_argument("--out-dir", default=str(RESULTS))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = load(Path(args.raw))
    timings = timing_records(records)
    if not timings:
        raise SystemExit("No successful timing runs in the raw records.")

    model = build_model(timings, records)
    (out_dir / "model.json").write_text(json.dumps(model, indent=2))

    plot_phase_breakdown(timings, out_dir / "phase_breakdown.png")
    plot_overhead_model(timings, model, out_dir / "overhead_model.png")
    plot_overhead_share(timings, out_dir / "overhead_share.png")
    plot_memory(records, model, out_dir / "memory_model.png")
    plot_fidelity(records, out_dir / "fidelity.png")

    report = "\n\n".join([
        "## Portability matrix\n\n" + portability_table(records),
        "## Overhead by backend\n\n" + overhead_table(timings),
        "## Fitted cost model  t_netqmpi = alpha + beta * G\n\n" + model_table(model),
    ])

    pooled = model.get("pooled", {})
    crossover = pooled.get("crossover_ops_1pct")
    if crossover:
        report += (f"\n\nPooled crossover: NetQMPI falls below "
                   f"{CROSSOVER_TARGET:.0%} of the run at about "
                   f"{crossover:,.0f} recorded operations.\n")

    (out_dir / "report.md").write_text(report + "\n")
    print(report)
    print(f"\nwrote {out_dir}/report.md, model.json and 5 plots")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
