"""
Regression tests: a rank that fails must fail the run, not hang it.

The Aer executor runs one thread per rank and lines them up on a
:class:`threading.Barrier`. A rank that raised anywhere but in the thread
that happened to be designated to run the simulation never reached that
barrier, so every other rank waited on it forever: the multi-backend
benchmark needed a watchdog and ``os._exit`` to get out of such runs.

Each scenario runs in a child process with a timeout, so that a regression
shows up as a failed test instead of a test session that never ends.
"""
from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

pytest.importorskip("qiskit", reason="the Aer backend needs Qiskit")
pytest.importorskip("qiskit_aer", reason="the Aer backend needs qiskit-aer")

REPO_ROOT = Path(__file__).resolve().parents[1]

#: How long a failing run may take before it counts as hung. Importing
#: Qiskit alone takes a second or two; the failure itself should be instant.
TIMEOUT = 60

#: Rank 1 raises before it ever enters its communicator block, so it never
#: reaches the barrier the other ranks are waiting on.
RAISES_BEFORE_THE_BLOCK = """
    from netqmpi.sdk.environment import Environment

    def main(env: Environment = None):
        comm = env.comm
        if comm.rank == 1:
            raise NotImplementedError("rank 1 gave up")
        with comm:
            circuit = env.create_circuit(num_qubits=1, num_clbits=1)
            circuit.measure(0, 0)
"""

#: Rank 1 raises while tracing, inside its communicator block.
RAISES_INSIDE_THE_BLOCK = """
    from netqmpi.sdk.environment import Environment

    def main(env: Environment = None):
        comm = env.comm
        with comm:
            circuit = env.create_circuit(num_qubits=1, num_clbits=1)
            if comm.rank == 1:
                raise NotImplementedError("rank 1 gave up")
            circuit.measure(0, 0)
"""

#: Rank 1 raises after the joint simulation, on its way out.
RAISES_AFTER_THE_BLOCK = """
    from netqmpi.sdk.environment import Environment

    def main(env: Environment = None):
        comm = env.comm
        with comm:
            circuit = env.create_circuit(num_qubits=1, num_clbits=1)
            circuit.measure(0, 0)
        if comm.rank == 1:
            raise NotImplementedError("rank 1 gave up")
"""

#: Rank 1 returns without ever entering its communicator block.
RETURNS_WITHOUT_THE_BLOCK = """
    from netqmpi.sdk.environment import Environment

    def main(env: Environment = None):
        comm = env.comm
        if comm.rank == 1:
            return
        with comm:
            circuit = env.create_circuit(num_qubits=1, num_clbits=1)
            circuit.measure(0, 0)
"""

#: What the child process does: run the app, report the exception it got,
#: then check that nothing is left behind that would keep the process alive.
DRIVER = """
    import sys, threading
    from netqmpi.runtime.adapters.aer import AerExecutorAdapter, AerSimulatorConfig

    ranks = int(sys.argv[2])
    executor = AerExecutorAdapter(ranks, AerSimulatorConfig(shots=4))
    try:
        executor.run(executor.build_apps(sys.argv[1], ranks))
    except BaseException as error:
        print(f"RAISED {type(error).__name__}: {error}")
    else:
        print("RETURNED")

    # Qiskit Aer leaves an idle ThreadPoolExecutor worker behind even after a
    # clean run; the interpreter joins it on exit, so it cannot hang anything.
    alive = [t.name for t in threading.enumerate()
             if t is not threading.main_thread() and not t.daemon
             and not t.name.startswith("ThreadPoolExecutor")]
    print(f"ALIVE {alive}")

    # A second run in the same process must not inherit the first one's state.
    executor = AerExecutorAdapter(2, AerSimulatorConfig(shots=4))
    executor.run(executor.build_apps(sys.argv[3], 2))
    print("SECOND RUN OK")
"""

HEALTHY = """
    from netqmpi.sdk.environment import Environment

    def main(env: Environment = None):
        with env.comm:
            circuit = env.create_circuit(num_qubits=1, num_clbits=1)
            circuit.measure(0, 0)
"""


def run_in_child(source: str, ranks: int, tmp_path: Path) -> str:
    """
    Run an app on the Aer backend in a child process.

    Args:
        source: Body of the app module, which must define ``main(env)``.
        ranks: Number of ranks to run.
        tmp_path: Directory to write the app and the driver to.

    Returns:
        What the child printed.

    Raises:
        AssertionError: If the child does not finish within :data:`TIMEOUT`.
    """
    app = tmp_path / "app.py"
    app.write_text(textwrap.dedent(source))
    healthy = tmp_path / "healthy.py"
    healthy.write_text(textwrap.dedent(HEALTHY))
    driver = tmp_path / "driver.py"
    driver.write_text(textwrap.dedent(DRIVER))

    try:
        child = subprocess.run(
            [sys.executable, str(driver), str(app), str(ranks), str(healthy)],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=TIMEOUT,
            env={"PYTHONPATH": str(REPO_ROOT), "PATH": "/usr/bin:/bin"},
        )
    except subprocess.TimeoutExpired:
        raise AssertionError(
            f"the run hung: still going after {TIMEOUT}s with {ranks} ranks")
    return child.stdout + child.stderr


@pytest.mark.parametrize("ranks", [2, 8])
@pytest.mark.parametrize("source", [
    RAISES_BEFORE_THE_BLOCK, RAISES_INSIDE_THE_BLOCK, RAISES_AFTER_THE_BLOCK,
], ids=["before", "inside", "after"])
def test_a_failing_rank_fails_the_run(source, ranks, tmp_path):
    """The rank's own exception comes out of ``executor.run()``, promptly."""
    output = run_in_child(source, ranks, tmp_path)
    assert "RAISED NotImplementedError: rank 1 gave up" in output, output
    assert "ALIVE []" in output, output
    assert "SECOND RUN OK" in output, output


@pytest.mark.parametrize("ranks", [2, 8])
def test_a_rank_that_skips_the_block_is_named(ranks, tmp_path):
    """A rank that never synchronises is reported instead of waited for."""
    output = run_in_child(RETURNS_WITHOUT_THE_BLOCK, ranks, tmp_path)
    assert "RAISED RuntimeError" in output, output
    assert "rank 1" in output, output
    assert "ALIVE []" in output, output
    assert "SECOND RUN OK" in output, output
