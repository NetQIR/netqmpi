"""
Regression tests for the Qoala adapter.

Qoala drives a NetSquid simulation, but unlike the SquidASM tests these are
cheap — a two-rank run costs a little over a second — so they are not marked
``integration`` and run with the rest of the suite. Environments without
NetSquid skip the file outright.

What they pin down is the gate table. The adapter translated a controlled
gate by comparing the *target gate's* name against ``"RX"`` and ``"RZ"``,
neither of which the SDK ever emits: ``cx`` records a controlled
``Gate('X')`` and ``cz`` a controlled ``Gate('Z')``, so both of the gates
the table meant to support were unreachable. Worse, ``crz(theta)`` *does*
record a controlled ``Gate('RZ')``, so it fell into the ``"RZ"`` branch and
came out as a plain controlled-Z whatever the angle was. ``SWAP``, which
the SDK records as a plain two-qubit ``Gate``, never reached the table at
all.
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

pytest.importorskip("qoala", reason="the Qoala backend needs qoala-sim")
pytest.importorskip("netsquid", reason="the Qoala backend needs NetSquid")

from netqmpi.runtime.adapters.qoala import (  # noqa: E402
    QoalaExecutorAdapter, QoalaRunConfig,
)
from netqmpi.sdk.environment import Environment  # noqa: E402

SHOTS = 20


def run_app(source: str, ranks: int, tmp_path: Path, shots: int = SHOTS,
            **config):
    """
    Run a NetQMPI program on the Qoala backend.

    Args:
        source: Body of the app module, which must define ``main(env)``.
        ranks: Number of ranks to run.
        tmp_path: Directory to write the app to.
        shots: Number of simulated repetitions.
        **config: Other :class:`QoalaRunConfig` fields to set.

    Returns:
        One histogram per rank, keyed by rank. Bits are ordered by ascending
        classical-bit index, which is how the adapter assembles them.
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
        config = QoalaRunConfig(**config)
        config.shots = shots
        executor = QoalaExecutorAdapter(ranks, config)
        executor.run(executor.build_apps(str(app), ranks))
    finally:
        Environment.__init__ = original

    return {env.comm.rank: dict(env.comm.results)
            for env in sorted(captured, key=lambda e: e.comm.rank)}


#: A local CNOT: |1> on the control flips the target, so both read 1.
LOCAL_CNOT = """
    from netqmpi.sdk.environment import Environment

    def main(env: Environment = None):
        comm = env.comm
        with comm:
            circuit = env.create_circuit(num_qubits=2, num_clbits=2)
            if comm.rank == 0:
                circuit.x(0)
                circuit.cx(0, 1)
            circuit.measure(0, 0)
            circuit.measure(1, 1)
"""

#: A local CZ. It leaves the computational basis alone, so the point is
#: simply that it translates instead of raising on a name it never matches.
LOCAL_CZ = """
    from netqmpi.sdk.environment import Environment

    def main(env: Environment = None):
        comm = env.comm
        with comm:
            circuit = env.create_circuit(num_qubits=2, num_clbits=2)
            if comm.rank == 0:
                circuit.x(0)
                circuit.x(1)
                circuit.cz(0, 1)
            circuit.measure(0, 0)
            circuit.measure(1, 1)
"""

#: A local SWAP moves the excitation from qubit 0 to qubit 1.
LOCAL_SWAP = """
    from netqmpi.sdk.environment import Environment

    def main(env: Environment = None):
        comm = env.comm
        with comm:
            circuit = env.create_circuit(num_qubits=2, num_clbits=2)
            if comm.rank == 0:
                circuit.x(0)
                circuit.swap(0, 1)
            circuit.measure(0, 0)
            circuit.measure(1, 1)
"""

#: A controlled rotation, which NetQASM's cphase cannot express.
LOCAL_CRZ = """
    from netqmpi.sdk.environment import Environment

    def main(env: Environment = None):
        comm = env.comm
        with comm:
            circuit = env.create_circuit(num_qubits=2, num_clbits=2)
            if comm.rank == 0:
                circuit.crz(0.5, 0, 1)
            circuit.measure(0, 0)
            circuit.measure(1, 1)
"""

#: Teleport |1>: the receiver must read 1 on every shot.
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

#: A qubit swapped into a scratch slot, sent round-trip, and swapped back.
#: This is the shape ``apps/ghz.py`` uses, and the smallest program found
#: that the adapter mis-compiles.
SWAP_AROUND_A_ROUND_TRIP = """
    from netqmpi.sdk.environment import Environment

    def main(env: Environment = None):
        comm = env.comm
        with comm:
            circuit = env.create_circuit(num_qubits=2, num_clbits=1)
            if comm.rank == 0:
                circuit.x(0)
                circuit.swap(0, 1)
                comm.qsend(circuit, [1], 1)
                comm.qrecv(circuit, [1], 1)
                circuit.swap(0, 1)
                circuit.measure(0, 0)
            else:
                comm.qrecv(circuit, [1], 0)
                comm.qsend(circuit, [1], 0)
                circuit.measure(0, 0)
"""

#: A qubit sent away and immediately received back on the same rank.
TELEPORT_ROUND_TRIP = """
    from netqmpi.sdk.environment import Environment

    def main(env: Environment = None):
        comm = env.comm
        with comm:
            circuit = env.create_circuit(num_qubits=1, num_clbits=1)
            if comm.rank == 0:
                circuit.x(0)
                comm.qsend(circuit, [0], 1)
                comm.qrecv(circuit, [0], 1)
                circuit.measure(0, 0)
            else:
                comm.qrecv(circuit, [0], 0)
                comm.qsend(circuit, [0], 0)
                circuit.measure(0, 0)
"""

#: A telegate window, which the adapter does not implement.
TELEGATE = """
    from netqmpi.sdk.environment import Environment

    def main(env: Environment = None):
        comm = env.comm
        with comm:
            circuit = env.create_circuit(num_qubits=1, num_clbits=1)
            handle = comm.expose(circuit, 0, [1], root=0)
            if comm.rank == 1:
                circuit.x(handle)
            comm.unexpose(circuit, [1], root=0)
            circuit.measure(0, 0)
"""


def test_cnot_reaches_the_backend(tmp_path):
    """``cx`` translates to a cnot instead of raising on an unmatched name."""
    results = run_app(LOCAL_CNOT, 2, tmp_path)
    assert results[0] == {"11": SHOTS}, results


def test_cz_reaches_the_backend(tmp_path):
    """``cz`` translates to a cphase; both qubits stay excited."""
    results = run_app(LOCAL_CZ, 2, tmp_path)
    assert results[0] == {"11": SHOTS}, results


def test_swap_moves_the_state(tmp_path):
    """``swap`` is emitted as three CNOTs, not dropped as an unknown gate."""
    results = run_app(LOCAL_SWAP, 2, tmp_path)
    # Bits are ordered by classical-bit index, so the excitation having moved
    # to qubit 1 reads as "01".
    assert results[0] == {"01": SHOTS}, results


def test_controlled_rotation_is_refused(tmp_path):
    """A controlled rotation must be refused rather than become a CZ."""
    with pytest.raises(NotImplementedError, match="Controlled-RZ"):
        run_app(LOCAL_CRZ, 2, tmp_path, shots=1)


def test_teleports_a_known_state(tmp_path):
    """A transferred |1> arrives as |1>, on every shot."""
    results = run_app(TELEPORT_ONE, 2, tmp_path)
    assert results[1] == {"1": SHOTS}, results


def test_round_trip_keeps_its_state(tmp_path):
    """A qubit sent away and received back is still the one that left."""
    results = run_app(TELEPORT_ROUND_TRIP, 2, tmp_path, shots=10)
    assert results[0] == {"1": 10}, results


def test_swap_around_a_round_trip(tmp_path):
    """
    A local gate on both sides of a round trip must survive the simulation.

    This is the smallest reproduction of what stopped ``apps/ghz.py`` running
    on Qoala (Q1). The round trip alone was fine (see
    :func:`test_round_trip_keeps_its_state`) and so were the swaps on their
    own; with a swap on each side, rank 0 holds a qubit across the trip, and
    shots simulated concurrently on shared memory starved each other of
    qubits until the simulation ran out of events. ``_build_counts`` then
    raised ``KeyError`` on the variable that never came back. See
    ``test_qoala_shots.py`` for the mechanism.
    """
    results = run_app(SWAP_AROUND_A_ROUND_TRIP, 2, tmp_path, shots=10)
    assert results[0] == {"1": 10}, results


@pytest.mark.parametrize("ranks", [2, 3, 4])
def test_ghz_completes_with_fidelity_one(ranks, tmp_path, monkeypatch):
    """Q1: ``apps/ghz.py`` finishes and reads all-zeros on every rank."""
    monkeypatch.setenv("NQB_QUBITS_PER_RANK", "1")
    app = Path(__file__).resolve().parents[1] / "scripts/benchmark/apps/ghz.py"
    results = run_app(app.read_text(), ranks, tmp_path, shots=5)
    assert results == {rank: {"0": 5} for rank in range(ranks)}, results


def test_shots_sharing_a_simulation_still_finish(tmp_path):
    """Several shots per simulation get enough memory not to starve."""
    results = run_app(SWAP_AROUND_A_ROUND_TRIP, 2, tmp_path, shots=10,
                      concurrent_shots=5)
    assert results[0] == {"1": 10}, results


#: A fair coin on rank 0.
COIN = """
    from netqmpi.sdk.environment import Environment

    def main(env: Environment = None):
        comm = env.comm
        with comm:
            circuit = env.create_circuit(num_qubits=1, num_clbits=1)
            if comm.rank == 0:
                circuit.h(0)
            circuit.measure(0, 0)
"""


def test_shots_simulated_separately_are_independent(tmp_path):
    """
    One simulation per shot must not replay the same random numbers.

    If resetting NetSquid between shots also reset its random state, every
    shot would draw the same outcome and a coin would always land the same
    way.
    """
    results = run_app(COIN, 2, tmp_path, shots=40)
    assert set(results[0]) == {"0", "1"}, results


def test_telegate_is_refused(tmp_path):
    """``expose`` is not implemented, and says so."""
    with pytest.raises(NotImplementedError, match="Expose"):
        run_app(TELEGATE, 2, tmp_path, shots=1)
