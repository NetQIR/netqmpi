"""
Portability matrix: the benchmark apps, on every backend, at a few sizes (G2).

None of the failures the multi-backend benchmark found — N1 (NetQASM
deadlock with two qubits per rank), N2 (NetQASM ``ghz`` from 8 ranks) and C1
(SquidASM's EPR set-up timeout from 10 ranks) — shows up with one qubit per
rank and at most three ranks, which is all the suite used to try. This runs
``q in {1, 2}`` qubits per rank at ``n in {2, 4, 8, 12}`` ranks, with a
couple of shots, and checks the one thing every app promises: all-zeros on
every rank.

Each case runs in a child process with a deadline. A case that hangs fails
instead of stalling the session, and the child dumps every thread's stack
shortly before its deadline, so the failure says *where* it was stuck.

Slow: select with ``-m slow``, skip with ``-m "not slow"``. The NetQASM cases
are also ``integration``.
"""
from __future__ import annotations

import ast
import importlib.util
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
APPS = REPO_ROOT / "scripts" / "benchmark" / "apps"

QUBITS = [1, 2]
RANKS = [2, 4, 8, 12]

#: Seconds a case may take before it counts as hung, per backend.
DEADLINE = {"aer": 120, "qoala": 600, "netqasm": 900}

#: Shots per case. Enough to see a wrong answer on a deterministic program.
SHOTS = {"aer": 16, "qoala": 2, "netqasm": 2}

#: Largest register Aer is asked to simulate with a statevector. The QFT
#: probes are not Clifford, so their cost doubles with every qubit.
AER_MAX_STATEVECTOR = 20

CHILD = """
    import faulthandler, sys
    faulthandler.dump_traceback_later(float(sys.argv[5]), exit=False)

    backend, app, ranks, shots = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])

    from netqmpi.sdk.environment import Environment
    captured = []
    original = Environment.__init__
    def capture(self, comm, executor):
        original(self, comm, executor)
        captured.append(self)
    Environment.__init__ = capture

    if backend == "aer":
        from netqmpi.runtime.adapters.aer import AerExecutorAdapter as Executor, AerSimulatorConfig as Config
        config = Config(shots=shots)
    elif backend == "qoala":
        from netqmpi.runtime.adapters.qoala import QoalaExecutorAdapter as Executor, QoalaRunConfig as Config
        config = Config(shots=shots)
    else:
        from netqmpi.runtime.adapters.netqasm import NetQASMExecutorAdapter as Executor, NetQASMRunConfig as Config
        from netqmpi.runtime.adapters.netqasm._compat import INSTALLED_MAJOR
        config = Config(shots=shots)
        config.netqasm_major = INSTALLED_MAJOR

    executor = Executor(ranks, config)
    executor.run(executor.build_apps(app, ranks))
    faulthandler.cancel_dump_traceback_later()
    print("RESULTS", repr({env.comm.rank: dict(env.comm.results) for env in captured}))
"""


def available(backend: str) -> bool:
    """Whether the simulator a backend needs is installed."""
    needs = {"aer": ["qiskit_aer"], "qoala": ["qoala", "netsquid"],
             "netqasm": ["squidasm", "netsquid"]}[backend]
    return all(importlib.util.find_spec(name) is not None for name in needs)


def cases():
    """Every (backend, app, ranks, qubits) the matrix covers."""
    portable = {"aer": ["cascade", "ghz", "qft", "qft_telegate"],
                "qoala": ["cascade", "ghz"],
                "netqasm": ["cascade", "ghz"]}
    for backend, apps in portable.items():
        marks = [pytest.mark.slow]
        if backend == "netqasm":
            marks.append(pytest.mark.integration)
        if not available(backend):
            marks.append(pytest.mark.skip(reason=f"{backend} is not installed"))
        for app in apps:
            for ranks in RANKS:
                for qubits in QUBITS:
                    if backend == "aer" and app.startswith("qft"):
                        width = ranks * (qubits + (app == "qft"))
                        if width > AER_MAX_STATEVECTOR:
                            continue
                    yield pytest.param(backend, app, ranks, qubits, marks=marks,
                                       id=f"{backend}-{app}-n{ranks}-q{qubits}")


def all_zeros(histogram) -> bool:
    return all(set(key.replace(" ", "")) <= {"0"} for key in histogram)


@pytest.mark.parametrize("backend, app, ranks, qubits", list(cases()))
def test_app_returns_all_zeros(backend, app, ranks, qubits, tmp_path):
    driver = tmp_path / "driver.py"
    driver.write_text(textwrap.dedent(CHILD))
    deadline = DEADLINE[backend]
    env = dict(os.environ, PYTHONPATH=str(REPO_ROOT),
               NQB_QUBITS_PER_RANK=str(qubits))
    try:
        child = subprocess.run(
            [sys.executable, str(driver), backend, str(APPS / f"{app}.py"),
             str(ranks), str(SHOTS[backend]), str(deadline - 15)],
            cwd=tmp_path, env=env, capture_output=True, text=True,
            timeout=deadline)
    except subprocess.TimeoutExpired as expired:
        stacks = (expired.stderr or b"")
        if isinstance(stacks, bytes):
            stacks = stacks.decode(errors="replace")
        pytest.fail(f"hung for {deadline}s; thread stacks:\n{stacks[-20000:]}")

    output = child.stdout + child.stderr
    assert child.returncode == 0, output[-20000:]
    line = next(l for l in child.stdout.splitlines() if l.startswith("RESULTS "))
    results = ast.literal_eval(line[len("RESULTS "):])

    assert set(results) == set(range(ranks)), results
    for rank, histogram in results.items():
        assert sum(histogram.values()) == SHOTS[backend], (rank, histogram)
        assert all_zeros(histogram), (rank, histogram)
