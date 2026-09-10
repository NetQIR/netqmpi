"""
Run one NetQMPI configuration under the profiler and emit a JSON record.

One process handles exactly one ``(backend, app, ranks, qubits, shots)``
point. That is deliberate: peak RSS is a process-wide, monotonic figure, and
the cost of importing a backend's simulator only happens once per process,
so sharing a process between configurations would smear both. The
orchestrator (:file:`run_all.sh`) loops over the grid and each backend runs
in the environment that can import it.

Every run produces a record whatever happens. A backend that does not
implement one of the primitives an app uses is data — it is what the
portability matrix is built from — not a crash, so the unsupported
operation is caught and recorded with ``status="unsupported"``. Two of the
backends run their ranks in threads and a failure there leaves the main
thread waiting on a barrier forever, which is why a watchdog is armed for
every run.

Usage::

    python run_benchmark.py --backend aer --app qft --ranks 3 --qubits 1 \
        --shots 1024 --reps 3 --out results/raw.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

BENCH_DIR = Path(__file__).resolve().parent
REPO_ROOT = BENCH_DIR.parent.parent
for path in (str(BENCH_DIR), str(REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)

import metrics                                          # noqa: E402
from profiler import (                                  # noqa: E402
    INTERLEAVED_BACKENDS, MemoryPass, Profiler, peak_rss_bytes,
)

#: Apps live next to this script and are addressed by name.
APPS_DIR = BENCH_DIR / "apps"

#: Exception raised by an adapter for a primitive it has not implemented.
#: Recorded rather than propagated, so the run still yields a data point.
UNSUPPORTED = NotImplementedError

#: Last exception seen in a worker thread, so a hang caused by a rank dying
#: behind a barrier can be reported for what it really was.
_thread_error: Dict[str, Optional[BaseException]] = {"exc": None}


def _install_thread_excepthook() -> None:
    """Record exceptions raised inside rank threads instead of losing them."""
    previous = threading.excepthook

    def hook(args):
        _thread_error["exc"] = args.exc_value
        previous(args)

    threading.excepthook = hook


# ----------------------------------------------------------------------
# Backends
# ----------------------------------------------------------------------

def _build_executor(backend: str, ranks: int, args) -> Tuple[Any, float]:
    """
    Import a backend adapter and build its executor.

    Args:
        backend: Backend name.
        ranks: Number of ranks the run needs.
        args: Parsed command-line arguments.

    Returns:
        A pair ``(executor, import_seconds)``.

    Raises:
        ValueError: If the backend name is unknown.
    """
    start = time.perf_counter()

    if backend == "aer":
        from netqmpi.runtime.adapters.aer import AerExecutorAdapter, AerSimulatorConfig
        import_seconds = time.perf_counter() - start
        config = AerSimulatorConfig()
        config.shots = args.shots
        config.transfer_mode = args.transfer_mode
        return AerExecutorAdapter(ranks, config), import_seconds

    if backend == "cunqa":
        from netqmpi.runtime.adapters.cunqa import CunqaExecutorAdapter, CunqaRunConfig
        import_seconds = time.perf_counter() - start
        config = CunqaRunConfig()
        config.shots = args.shots
        config.qraise = args.cunqa_qraise
        if args.cunqa_family:
            config.family = args.cunqa_family
        return CunqaExecutorAdapter(ranks, config), import_seconds

    if backend == "qoala":
        from netqmpi.runtime.adapters.qoala import QoalaExecutorAdapter, QoalaRunConfig
        import_seconds = time.perf_counter() - start
        config = QoalaRunConfig()
        config.shots = args.shots
        return QoalaExecutorAdapter(ranks, config), import_seconds

    if backend == "netqasm":
        from netqmpi.runtime.adapters.netqasm import (
            NetQASMExecutorAdapter, NetQASMRunConfig,
        )
        import_seconds = time.perf_counter() - start
        config = NetQASMRunConfig()
        config.shots = args.shots
        return NetQASMExecutorAdapter(ranks, config), import_seconds

    raise ValueError(f"Unknown backend: {backend}")


# ----------------------------------------------------------------------
# One repetition
# ----------------------------------------------------------------------

def _run_once(backend: str, app_path: str, ranks: int, args,
              memory: bool) -> Dict[str, Any]:
    """
    Execute one repetition and return its measurements.

    A fresh executor is built for every repetition so that the ``setup``
    phase measures resource acquisition each time rather than reusing what
    a previous repetition left warm.

    Args:
        backend: Backend name.
        app_path: Path to the NetQMPI app to run.
        ranks: Number of ranks.
        args: Parsed command-line arguments.
        memory: Whether to take a :mod:`tracemalloc` peak, which roughly
            doubles the cost of the trace and so is never combined with a
            timing measurement.

    Returns:
        A mapping of phase timings, counters and quality metrics.
    """
    from netqmpi.sdk.environment import Environment

    envs: List[Any] = []
    original_init = Environment.__init__

    def capture(self, comm, executor):
        original_init(self, comm, executor)
        envs.append(self)

    Environment.__init__ = capture
    profiler = Profiler(backend)

    try:
        with MemoryPass(memory) as memory_pass:
            executor, import_seconds = _build_executor(backend, ranks, args)
            profiler.times["import"] = import_seconds

            setup_start = time.perf_counter()
            apps = executor.build_apps(app_path, ranks)
            profiler.times["setup"] = time.perf_counter() - setup_start

            profiler.install()
            if envs:
                # Everything the backend does happens inside the
                # communicator's __exit__; timing it is what separates the
                # trace from the barrier waits of a threaded backend.
                profiler.hook_communicator(type(envs[0].comm))
            try:
                run_start = time.perf_counter()
                executor.run([profiler.wrap_app(app) for app in apps]
                             if isinstance(apps, list) else apps)
                profiler.finish_run(time.perf_counter() - run_start)
            finally:
                profiler.uninstall()
    finally:
        Environment.__init__ = original_init

    envs.sort(key=lambda env: env.comm.rank)
    histograms = metrics.per_rank_histograms(envs, backend)
    overall, per_rank = metrics.fidelity(histograms)

    record: Dict[str, Any] = {
        "status": "ok",
        "t_import": profiler.times["import"],
        "t_setup": profiler.times["setup"],
        "t_trace": profiler.times["trace"],
        "t_translate": profiler.times["translate"],
        "t_backend": profiler.times["backend"],
        "t_sync": profiler.times["sync"],
        "t_total": profiler.times["total"],
        "t_netqmpi": profiler.netqmpi_seconds,
        "overhead_fraction": profiler.overhead_fraction,
        "translate_calls": profiler.counts["translate_calls"],
        "fidelity": overall,
        "fidelity_per_rank": per_rank,
        "fidelity_joint": metrics.joint_fidelity(envs, backend),
        "shots_observed": sum(sum(h.values()) for h in histograms.values()),
        "rss_peak_bytes": peak_rss_bytes(),
        "py_peak_bytes": memory_pass.peak_bytes,
        "translate_isolated": backend not in INTERLEAVED_BACKENDS,
    }
    record.update(metrics.count_operations(envs))
    return record


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------

def _base_record(args, rep: int) -> Dict[str, Any]:
    """
    Return the identifying fields shared by every record of this run.

    Args:
        args: Parsed command-line arguments.
        rep: Repetition index.

    Returns:
        A mapping describing the configuration.
    """
    return {
        "backend": args.backend,
        "app": args.app,
        "ranks": args.ranks,
        "qubits_per_rank": args.qubits,
        "shots": args.shots,
        "transfer_mode": args.transfer_mode if args.backend == "aer" else None,
        "rep": rep,
        "timestamp": time.time(),
    }


def _emit(record: Dict[str, Any], out: Optional[str]) -> None:
    """
    Append one record to the output file, and echo a summary line.

    Args:
        record: The record to write.
        out: Destination file, or ``None`` to only echo.
    """
    line = json.dumps(record)
    if out:
        with open(out, "a") as handle:
            handle.write(line + "\n")
            handle.flush()

    summary = (f"{record['backend']:>8} {record['app']:>12} "
               f"n={record['ranks']} q={record['qubits_per_rank']} "
               f"rep={record['rep']} {record['status']}")
    if record["status"] == "ok":
        summary += (f" total={record['t_total']:.3f}s "
                    f"netqmpi={record['t_netqmpi']*1000:.1f}ms "
                    f"({record['overhead_fraction']*100:.1f}%) "
                    f"F={record['fidelity']:.4f} "
                    f"ops={record['ops_total']}")
    else:
        summary += f" :: {record.get('error', '')[:90]}"
    print(summary, flush=True)


def _arm_watchdog(seconds: float, record: Dict[str, Any], out: Optional[str]) -> None:
    """
    Kill the process if a run hangs, recording why it did.

    The Aer and NetQASM adapters run their ranks in threads behind a
    barrier; when one rank dies the others wait for it forever, so a plain
    exception handler never sees it. The watchdog reports the thread's
    exception when there was one, which is what turns an unimplemented
    primitive into a portability data point rather than a lost run.

    Args:
        seconds: Wall-clock budget for the whole run.
        record: Identifying fields to attach to the failure record.
        out: Destination file for the record.
    """
    def fire(exc: Optional[BaseException], reason: str):
        failed = dict(record)
        if isinstance(exc, UNSUPPORTED):
            failed["status"] = "unsupported"
            failed["error"] = str(exc)
        else:
            failed["status"] = "timeout"
            failed["error"] = reason + (f"; thread error: {exc}" if exc else "")
        _emit(failed, out)
        os._exit(3)

    def watch():
        # Poll rather than wait out the whole budget: once a rank thread has
        # died there is nothing left to wait for, and burning the full
        # timeout on every unsupported primitive would dominate the sweep.
        deadline = time.perf_counter() + seconds
        while time.perf_counter() < deadline:
            exc = _thread_error["exc"]
            if exc is not None:
                # Give the surviving ranks a moment to unwind on their own;
                # if they do, the main path reports it and this never fires.
                time.sleep(2.0)
                fire(_thread_error["exc"], "rank thread died")
            time.sleep(0.2)
        fire(_thread_error["exc"], f"exceeded {seconds}s")

    thread = threading.Thread(target=watch, daemon=True)
    thread.start()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", required=True,
                        choices=["aer", "cunqa", "qoala", "netqasm"])
    parser.add_argument("--app", required=True,
                        help="App name under apps/ (qft, cascade, ghz, qft_telegate)")
    parser.add_argument("--ranks", type=int, required=True)
    parser.add_argument("--qubits", type=int, default=1,
                        help="Qubits held by each rank")
    parser.add_argument("--shots", type=int, default=1024)
    parser.add_argument("--reps", type=int, default=3,
                        help="Timing repetitions")
    parser.add_argument("--memory", action="store_true",
                        help="Add a separate tracemalloc pass after the timing reps")
    parser.add_argument("--input", type=int, default=0,
                        help="QFT input state; 0 keeps the all-zero criterion")
    parser.add_argument("--transfer-mode", default="swap",
                        choices=["swap", "teleport"],
                        help="Aer qsend implementation")
    parser.add_argument("--cunqa-qraise", action="store_true",
                        help="Let the adapter raise and drop the vQPUs itself")
    parser.add_argument("--cunqa-family", default=None,
                        help="Attach to this already-raised vQPU family")
    parser.add_argument("--timeout", type=float, default=900.0,
                        help="Wall-clock budget for the whole invocation")
    parser.add_argument("--out", default=None, help="JSONL file to append to")
    args = parser.parse_args()

    app_path = APPS_DIR / f"{args.app}.py"
    if not app_path.is_file():
        parser.error(f"No such app: {app_path}")

    # Apps read their shape from the environment: the executors load a
    # module by path and call main(env), so there is no argument to pass.
    os.environ["NQB_QUBITS_PER_RANK"] = str(args.qubits)
    os.environ["NQB_INPUT"] = str(args.input)

    _install_thread_excepthook()
    _arm_watchdog(args.timeout, _base_record(args, -1), args.out)

    passes = [(rep, False) for rep in range(args.reps)]
    if args.memory:
        passes.append((args.reps, True))

    exit_code = 0
    for rep, memory in passes:
        record = _base_record(args, rep)
        record["memory_pass"] = memory
        try:
            record.update(_run_once(args.backend, str(app_path), args.ranks,
                                    args, memory))
        except UNSUPPORTED as error:
            record["status"] = "unsupported"
            record["error"] = str(error)
            exit_code = 0            # an expected, informative outcome
        except Exception as error:   # noqa: BLE001 - any failure is a data point
            record["status"] = "error"
            record["error"] = f"{type(error).__name__}: {error}"
            record["traceback"] = traceback.format_exc()[-2000:]
            exit_code = 1
        _emit(record, args.out)

        if record["status"] != "ok":
            break                    # the rest of the reps would fail alike

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
