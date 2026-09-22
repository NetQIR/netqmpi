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
