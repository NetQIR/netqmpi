"""
Regression tests for the NetQASM/SquidASM adapter.

These drive a real NetSquid simulation, which costs seconds per shot, so
they are marked ``integration`` and can be skipped with
``pytest -m "not integration"``. Shot counts are kept to the minimum that
still distinguishes a working transfer from a broken one.

The adapter did not run at all before these fixes: an empty
``program_inputs`` made SquidASM raise ``KeyError`` on the first party it
tried to start. What the tests below pin down is everything that was found
behind that: qubits re-allocated on every recursive ``translate`` call,
corrections sent as unresolved futures, a sent qubit left dead in its slot,
a shot count that was ignored, and gates that were silently dropped instead
of reported.
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

pytest.importorskip("squidasm", reason="the NetQASM backend needs SquidASM")
pytest.importorskip("netsquid", reason="the NetQASM backend needs NetSquid")

from netqmpi.runtime.adapters.netqasm import (  # noqa: E402
    NetQASMExecutorAdapter, NetQASMRunConfig,
)
from netqmpi.runtime.adapters.netqasm._compat import (  # noqa: E402
    INSTALLED_MAJOR, USE_ONLY_SHARED_INSTRUCTIONS,
)
from netqmpi.sdk.environment import Environment  # noqa: E402

pytestmark = pytest.mark.integration

#: Enough shots to tell a deterministic outcome from a coin flip without
#: paying for more NetSquid runs than the point needs.
SHOTS = 3


def run_app(source: str, ranks: int, tmp_path: Path, shots: int = SHOTS):
    """
    Run a NetQMPI program on the NetQASM backend.

    Args:
        source: Body of the app module, which must define ``main(env)``.
        ranks: Number of ranks to run.
        tmp_path: Directory to write the app to.
        shots: Number of simulated repetitions.

    Returns:
        One histogram per rank, keyed by rank.
    """
    app = tmp_path / "app.py"
    app.write_text(textwrap.dedent(source))

    captured = []
    original = Environment.__init__

    def capture(self, comm, executor):
        original(self, comm, executor)
        captured.append(self)

    Environment.__init__ = capture
    try:
        config = NetQASMRunConfig()
        config.shots = shots
        # Whichever release is installed is the one being exercised.
        config.netqasm_major = INSTALLED_MAJOR
        executor = NetQASMExecutorAdapter(ranks, config)
        executor.run(executor.build_apps(str(app), ranks))
    finally:
        Environment.__init__ = original

    return {env.comm.rank: dict(env.comm.results)
            for env in sorted(captured, key=lambda e: e.comm.rank)}


#: Teleport |1>. A working transfer gives the receiver a 1 every time; a
#: broken one gives |0>, or a coin flip if the corrections go astray.
TELEPORT_ONE = """
    from netqmpi.sdk.environment import Environment

    def main(env: Environment = None):
        comm, rank = env.comm, env.comm.rank
        with comm:
            circuit = env.create_circuit(num_qubits=1, num_clbits=1)
            if rank == 0:
                circuit.x(0)
                comm.qsend(circuit, [0], 1)
            else:
                comm.qrecv(circuit, [0], 0)
                circuit.measure(0, 0)
"""

#: The sender measures the slot it gave away. The SDK promises a sent qubit
#: is left in |0>; the adapter used to leave the measured — and therefore
#: freed — qubit in place, so touching the slot again aborted the run.
SENDER_KEEPS_A_SLOT = """
    from netqmpi.sdk.environment import Environment

    def main(env: Environment = None):
        comm, rank = env.comm, env.comm.rank
        with comm:
            circuit = env.create_circuit(num_qubits=1, num_clbits=1)
            if rank == 0:
                circuit.x(0)
                comm.qsend(circuit, [0], 1)
                circuit.measure(0, 0)       # the slot must still be usable
            else:
                comm.qrecv(circuit, [0], 0)
                circuit.measure(0, 0)
"""

#: A local CNOT. ControlledGate has no ``name`` attribute, so the adapter
#: raised AttributeError on every controlled gate and none had ever run.
LOCAL_CNOT = """
    from netqmpi.sdk.environment import Environment

    def main(env: Environment = None):
        comm, rank = env.comm, env.comm.rank
        with comm:
            circuit = env.create_circuit(num_qubits=2, num_clbits=2)
            if rank == 0:
                circuit.x(0)
                circuit.cx(0, 1)            # both qubits end up at |1>
            circuit.measure(0, 0)
            circuit.measure(1, 1)
"""

#: A local SWAP. NetQASM 2.x builds one itself; on 1.x the adapter has to
#: assemble it from three CNOTs, and it used to emit a single CNOT, which is
#: a different gate. Either way the excitation must end up on qubit 1.
LOCAL_SWAP = """
    from netqmpi.sdk.environment import Environment

    def main(env: Environment = None):
        comm, rank = env.comm, env.comm.rank
        with comm:
            circuit = env.create_circuit(num_qubits=2, num_clbits=2)
            if rank == 0:
                circuit.x(0)
                circuit.swap(0, 1)
            circuit.measure(0, 0)
            circuit.measure(1, 1)
"""

#: A controlled-phase, which NetQASM has no instruction for. It used to be
#: dropped from the circuit without a word.
UNSUPPORTED_GATE = """
    from netqmpi.sdk.environment import Environment

    def main(env: Environment = None):
        comm, rank = env.comm, env.comm.rank
        with comm:
            circuit = env.create_circuit(num_qubits=2, num_clbits=2)
            if rank == 0:
                circuit.cp(0, 1, 0.5)
            circuit.measure(0, 0)
"""


def test_teleports_a_known_state(tmp_path):
    """A transferred |1> arrives as |1>, on every shot."""
    results = run_app(TELEPORT_ONE, 2, tmp_path)
    assert results[1] == {"1": SHOTS}, results


def test_sender_slot_stays_usable(tmp_path):
    """After qsend the sender's slot is a live qubit in |0>."""
    results = run_app(SENDER_KEEPS_A_SLOT, 2, tmp_path)
    assert results[0] == {"0": SHOTS}, results
    assert results[1] == {"1": SHOTS}, results


def test_controlled_gates_reach_the_backend(tmp_path):
    """A CNOT is emitted rather than raising on a missing attribute."""
    results = run_app(LOCAL_CNOT, 2, tmp_path)
    assert set(results[0]) == {"11"}, results


def test_shots_are_actually_repeated(tmp_path):
    """Asking for N shots returns N samples, not one."""
    results = run_app(TELEPORT_ONE, 2, tmp_path, shots=4)
    assert sum(results[1].values()) == 4, results


def test_swap_moves_the_state(tmp_path):
    """
    A swap exchanges the two qubits, on either NetQASM release.

    The adapter uses the native instruction where there is one and three
    CNOTs where there is not, so this is the same expectation held against
    two different translations.
    """
    results = run_app(LOCAL_SWAP, 2, tmp_path)
    # Bits are ordered most significant first, so the excitation having
    # moved from qubit 0 to qubit 1 reads as "10".
    assert results[0] == {"10": SHOTS}, results


def test_swap_is_assembled_from_three_cnots():
    """
    A swap is emitted as three CNOTs, never as a native swap instruction.

    NetQASM 2.3 offers ``Qubit.swap``; 2.0 — the newest release SquidASM
    accepts — does not, and calling it against a SquidASM that cannot
    execute it *hangs* the simulation rather than failing. This asserts the
    instructions actually emitted, so switching to the native call would
    fail here instead of somewhere in a NetSquid event loop.
    """
    from netqmpi.runtime.adapters.netqasm.netqasm_circuit import (
        NetQASMCircuitAdapter,
    )

    emitted = []

    class StubQubit:
        def __init__(self, name):
            self.name = name

        def cnot(self, target):
            emitted.append(("cnot", self.name, target.name))

        def swap(self, target):
            emitted.append(("swap", self.name, target.name))

    adapter = NetQASMCircuitAdapter.__new__(NetQASMCircuitAdapter)
    adapter._qubits = [StubQubit("a"), StubQubit("b")]

    NetQASMCircuitAdapter._swap(adapter, 0, 1)

    assert emitted == [
        ("cnot", "a", "b"),
        ("cnot", "b", "a"),
        ("cnot", "a", "b"),
    ], emitted
    assert USE_ONLY_SHARED_INSTRUCTIONS is True


def test_the_other_major_is_refused():
    """
    Asking for the release that is not installed stops the run early.

    Both flags drive this one adapter, so the only thing that can go wrong
    is running it from the wrong environment. Saying so here beats failing
    several layers into SquidASM with a missing attribute.
    """
    other = 1 if INSTALLED_MAJOR >= 2 else 2
    config = NetQASMRunConfig()
    config.netqasm_major = other
    with pytest.raises(RuntimeError, match=f"NetQASM {other}"):
        NetQASMExecutorAdapter(2, config)


def test_unsupported_gate_is_reported(tmp_path):
    """A gate NetQASM cannot express must be refused, not dropped."""
    with pytest.raises(NotImplementedError, match="Controlled-P"):
        run_app(UNSUPPORTED_GATE, 2, tmp_path, shots=1)


# ----------------------------------------------------------------------
# Runs that used to hang. Each goes to a child process with a deadline, so a
# regression fails the test instead of stalling the session.
# ----------------------------------------------------------------------

#: Runs an app on the NetQASM backend and reports how it ended, then checks
#: that nothing it leaves behind would keep the process alive.
CHILD = """
    import sys, threading, time
    from netqmpi.runtime.adapters.netqasm import (
        NetQASMExecutorAdapter, NetQASMRunConfig, NetQASMCommunicator)
    from netqmpi.runtime.adapters.netqasm._compat import INSTALLED_MAJOR
    from netqmpi.sdk.environment import Environment

    captured = []
    original = Environment.__init__
    def capture(self, comm, executor):
        original(self, comm, executor)
        captured.append(self)
    Environment.__init__ = capture

    app, ranks, shots = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    config = NetQASMRunConfig(shots=shots)
    config.netqasm_major = INSTALLED_MAJOR
    executor = NetQASMExecutorAdapter(ranks, config)
    start = time.perf_counter()
    try:
        executor.run(executor.build_apps(app, ranks))
    except BaseException as error:
        print(f"RAISED {type(error).__name__}: {error}")
    else:
        results = {env.comm.rank: dict(env.comm.results) for env in captured}
        print(f"RESULTS {results!r}")
    print(f"SECONDS {time.perf_counter() - start:.1f}")
    alive = [t.name for t in threading.enumerate()
             if t is not threading.main_thread() and not t.daemon]
    print(f"ALIVE {alive}")
"""


def run_in_child(source: str, ranks: int, tmp_path: Path, shots: int = 1,
                 timeout: float = 120.0, env: dict = None) -> str:
    """
    Run an app on the NetQASM backend in a child process.

    Args:
        source: Body of the app module, or a path to an existing app.
        ranks: Number of ranks.
        tmp_path: Directory for the app and the driver.
        shots: Number of simulated repetitions.
        timeout: Seconds after which the run counts as hung.
        env: Extra environment variables for the child.

    Returns:
        What the child printed.
    """
    import os
    import subprocess
    import sys

    if isinstance(source, Path):
        app = source
    else:
        app = tmp_path / "app.py"
        app.write_text(textwrap.dedent(source))
    driver = tmp_path / "driver.py"
    driver.write_text(textwrap.dedent(CHILD))

    repo = Path(__file__).resolve().parents[1]
    child_env = dict(os.environ, PYTHONPATH=str(repo), **(env or {}))
    try:
        child = subprocess.run(
            [sys.executable, str(driver), str(app), str(ranks), str(shots)],
            cwd=tmp_path, capture_output=True, text=True, timeout=timeout,
            env=child_env)
    except subprocess.TimeoutExpired:
        raise AssertionError(f"the run hung: still going after {timeout}s")
    return child.stdout + child.stderr


#: Two transfers in a row from rank 0 to rank 1: |1> then |1>.
TWO_SENDS_IN_A_ROW = """
    from netqmpi.sdk.environment import Environment

    def main(env: Environment = None):
        comm, rank = env.comm, env.comm.rank
        with comm:
            circuit = env.create_circuit(num_qubits=2, num_clbits=2)
            if rank == 0:
                circuit.x(0)
                circuit.x(1)
                comm.qsend(circuit, [0], 1)
                comm.qsend(circuit, [1], 1)
            else:
                comm.qrecv(circuit, [0], 0)
                comm.qrecv(circuit, [1], 0)
            circuit.measure(0, 0)
            circuit.measure(1, 1)
"""


def test_two_sends_in_a_row_arrive(tmp_path):
    """
    N1: consecutive transfers in the same direction used to deadlock.

    See ``test_netqasm_sockets.py`` for the mechanism.
    """
    output = run_in_child(TWO_SENDS_IN_A_ROW, 2, tmp_path, shots=2, timeout=30)
    assert "RESULTS {0: {'00': 2}, 1: {'11': 2}}" in output, output


def test_cascade_with_two_qubits_per_rank_finishes(tmp_path):
    """N1 as the benchmark met it: cascade, n=2, q=2, one shot, under 30 s."""
    app = Path(__file__).resolve().parent / "apps/cascade.py"
    output = run_in_child(app, 2, tmp_path, shots=1, timeout=30,
                          env={"NQB_QUBITS_PER_RANK": "2"})
    assert "RESULTS {0: {'00': 1}, 1: {'00': 1}}" in output, output


#: A chain in which rank 1 fails *during the simulation*: SDG is refused only
#: when the operation runs, inside a SquidASM program thread, while the
#: ranks after it wait for a qubit that will never come.
FAILS_MID_SIMULATION = """
    from netqmpi.sdk.environment import Environment

    def main(env: Environment = None):
        comm, rank, size = env.comm, env.comm.rank, env.comm.size
        with comm:
            circuit = env.create_circuit(num_qubits=1, num_clbits=1)
            if rank > 0:
                comm.qrecv(circuit, [0], rank - 1)
            if rank == 1:
                circuit.sdg(0)
            if rank < size - 1:
                comm.qsend(circuit, [0], rank + 1)
            circuit.measure(0, 0)
"""


@pytest.mark.parametrize("ranks", [2, 8])
def test_a_rank_failing_mid_simulation_fails_the_run(ranks, tmp_path):
    """G1: the rank's exception comes out promptly, and SquidASM is stopped."""
    output = run_in_child(FAILS_MID_SIMULATION, ranks, tmp_path, timeout=120)
    assert "RAISED NotImplementedError: SDG" in output, output
    assert "ALIVE []" in output, output


def test_a_program_too_big_for_the_default_network_is_refused(tmp_path):
    """N3: six qubits cascaded need seven at once; nodes have five."""
    app = Path(__file__).resolve().parent / "apps/cascade.py"
    output = run_in_child(app, 2, tmp_path, timeout=60,
                          env={"NQB_QUBITS_PER_RANK": "6"})
    assert "RAISED ValueError" in output and "needs 7 qubits" in output, output


# ----------------------------------------------------------------------
# Telegate windows. Every expectation is deterministic, and is what the Aer
# backend gives for the same program (see test_aer_primitives.py).
# ----------------------------------------------------------------------

#: A remote CNOT: rank 0 lends its control, rank 1 holds the target.
REMOTE_CNOT = """
    from netqmpi.sdk.environment import Environment

    def main(env: Environment = None):
        comm, rank = env.comm, env.comm.rank
        with comm:
            circuit = env.create_circuit(num_qubits=1, num_clbits=1)
            if rank == 0 and {control}:
                circuit.x(0)
            if rank == 1 and {target}:
                circuit.x(0)
            handle = comm.expose(circuit, 0, [1], root=0)
            if rank == 1:
                circuit.cx(handle, 0)
            comm.unexpose(circuit, [1], root=0)
            circuit.measure(0, 0)
"""

#: One root, two receivers in one window: a CNOT on rank 1, and on rank 2 a
#: CZ between Hadamards, which is the same CNOT.
ONE_ROOT_TWO_RECEIVERS = """
    from netqmpi.sdk.environment import Environment

    def main(env: Environment = None):
        comm, rank = env.comm, env.comm.rank
        with comm:
            circuit = env.create_circuit(num_qubits=1, num_clbits=1)
            if rank == 0 and {payload}:
                circuit.x(0)
            handle = comm.expose(circuit, 0, [1, 2], root=0)
            if rank == 1:
                circuit.cx(handle, 0)
            elif rank == 2:
                circuit.h(0)
                circuit.cz(handle, 0)
                circuit.h(0)
            comm.unexpose(circuit, [1, 2], root=0)
            circuit.measure(0, 0)
"""

#: The root lends |+> and takes it back through a Hadamard. The unexpose
#: correction is all that keeps the root's answer deterministic: without
#: its Z the root reads a coin flip. With ``kick`` set the receiver's target
#: is |1>, so its CZ kicks a phase back and the root must read 1 instead.
PHASE_COMES_BACK = """
    from netqmpi.sdk.environment import Environment

    def main(env: Environment = None):
        comm, rank = env.comm, env.comm.rank
        with comm:
            circuit = env.create_circuit(num_qubits=1, num_clbits=1)
            if rank == 0:
                circuit.h(0)
            if rank == 1 and {kick}:
                circuit.x(0)
            handle = comm.expose(circuit, 0, [1], root=0)
            if rank == 1:
                circuit.cz(handle, 0)
            comm.unexpose(circuit, [1], root=0)
            if rank == 0:
                circuit.h(0)
            circuit.measure(0, 0)
"""

#: Two windows over different groups, one open across the other: the shape
#: of examples/5_qft_expose.py, with CNOTs instead of controlled phases.
NESTED_WINDOWS = """
    from netqmpi.sdk.environment import Environment

    def main(env: Environment = None):
        comm, rank = env.comm, env.comm.rank
        with comm:
            circuit = env.create_circuit(num_qubits=1, num_clbits=1)
            if rank == 2:
                circuit.x(0)                       # only rank 2 lends a |1>
            outer = comm.expose(circuit, 0, [0, 1], root=2)
            inner = comm.expose(circuit, 0, [0], root=1)
            if rank == 0:
                circuit.cx(inner, 0)               # control is |0>: no-op
                circuit.cx(outer, 0)               # control is |1>: flips
            comm.unexpose(circuit, [0], root=1)
            if rank == 1:
                circuit.cx(outer, 0)               # still open: flips
            comm.unexpose(circuit, [0, 1], root=2)
            circuit.measure(0, 0)
"""

#: The same window opened four times on one comm slot, with a superposed
#: control and two targets. Each window entangles or disentangles the
#: ranks, so every one of them has to be opened and closed correctly for
#: the echo to come back to all zeros.
WINDOW_REOPENED = """
    from netqmpi.sdk.environment import Environment

    def main(env: Environment = None):
        comm, rank = env.comm, env.comm.rank
        with comm:
            circuit = env.create_circuit(num_qubits=2, num_clbits=2)
            if rank == 0:
                circuit.h(0)
            for _ in range(4):
                handle = comm.expose(circuit, 0, [1], root=0)
                if rank == 1:
                    circuit.cx(handle, 0)
                    circuit.cx(handle, 1)
                comm.unexpose(circuit, [1], root=0)
            if rank == 0:
                circuit.h(0)
            circuit.measure(0, 0)
            circuit.measure(1, 1)
"""


@pytest.mark.parametrize("control, target", [(0, 0), (0, 1), (1, 0), (1, 1)])
def test_a_remote_cnot_follows_the_truth_table(control, target, tmp_path):
    """Rank 0's control flips rank 1's target, and comes back untouched."""
    source = REMOTE_CNOT.format(control=control, target=target)
    results = run_app(source, 2, tmp_path, shots=2)
    assert results[0] == {str(control): 2}, results
    assert results[1] == {str(target ^ control): 2}, results


@pytest.mark.parametrize("payload", [0, 1])
def test_one_window_serves_two_receivers(payload, tmp_path):
    """Both receivers read the control, through a cx and through a cz."""
    results = run_app(ONE_ROOT_TWO_RECEIVERS.format(payload=payload), 3,
                      tmp_path, shots=2)
    assert results == {0: {str(payload): 2}, 1: {str(payload): 2},
                       2: {str(payload): 2}}, results


@pytest.mark.parametrize("kick", [0, 1])
def test_unexpose_gives_the_root_its_phase_back(kick, tmp_path):
    """The cat-disentangler's Z correction keeps the root coherent."""
    results = run_app(PHASE_COMES_BACK.format(kick=kick), 2, tmp_path,
                      shots=4)
    assert results[0] == {str(kick): 4}, results
    assert results[1] == {str(kick): 4}, results


def test_windows_over_different_groups_nest(tmp_path):
    """Both receivers see rank 2's control as 1 and rank 1's as 0."""
    results = run_app(NESTED_WINDOWS, 3, tmp_path, shots=2)
    assert results == {0: {"1": 2}, 1: {"1": 2}, 2: {"1": 2}}, results


def test_a_window_can_be_opened_again(tmp_path):
    """Four windows in a row on one slot: an echo back to all zeros."""
    results = run_app(WINDOW_REOPENED, 2, tmp_path, shots=4)
    assert results == {0: {"00": 4}, 1: {"00": 4}}, results
