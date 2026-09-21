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


def test_unsupported_gate_is_reported(tmp_path):
    """A gate NetQASM cannot express must be refused, not dropped."""
    with pytest.raises(NotImplementedError, match="Controlled-P"):
        run_app(UNSUPPORTED_GATE, 2, tmp_path, shots=1)
