"""
Every NetQMPI primitive, executed end to end on the Aer backend.

The files around this one check what a program *records*. This one checks
what it *does*: each test writes a small NetQMPI application, runs it on
Qiskit Aer with several ranks, and asserts on the histogram that comes
back.

Aer is the right backend for that. It moves a qubit with a SWAP straight
across one global register instead of teleporting it — no entanglement is
consumed, no correction is sent, nothing decoheres — so a wrong answer
here is a bug in NetQMPI, never noise. It is a correctness reference, not
a model of a network.

Two habits keep the assertions sharp:

- **Deterministic payloads.** Where possible a program prepares ``|1>``
  and the test demands *every* shot come back the same. A probabilistic
  claim can be satisfied by accident; ``{'1': 512}`` cannot.
- **Echoes.** Where a superposition is needed, the program undoes what it
  did, so the right answer is all-zeros and any misordering shows up as a
  wrong result rather than a subtly different distribution.

Reading the histograms
----------------------
Aer keeps every rank's classical bits in one register, so a raw key spans
the whole run. :func:`run_app` splits it back into one histogram per rank,
with the bits in ascending classical-bit order — ``"01"`` means
``cbit 0 = 0, cbit 1 = 1`` — which is the order the program wrote them in
rather than Qiskit's most-significant-first printing.
"""
from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Dict

import pytest

pytest.importorskip("qiskit", reason="the Aer backend needs Qiskit")
pytest.importorskip("qiskit_aer", reason="the Aer backend needs qiskit-aer")

from netqmpi.runtime.adapters.aer import (  # noqa: E402
    AerExecutorAdapter, AerSimulatorConfig,
)
from netqmpi.sdk.environment import Environment  # noqa: E402

#: Enough shots that an even split is unmistakable and a deterministic
#: outcome is not a fluke, while a run still costs milliseconds.
SHOTS = 512

#: Fixed so the balanced checks cannot flake.
SEED = 20260921

#: Half-width allowed around an even split: ~4 sigma at this shot count.
BALANCE_TOLERANCE = 0.06


class Results(dict):
    """
    One histogram per rank, plus the joint histogram they came from.

    Attributes:
        joint: The histogram over every rank's classical bits at once,
            which is the only place a *correlation* between ranks is
            visible.
    """

    joint: Dict[str, int] = {}


def run_app(source: str, ranks: int, tmp_path: Path, **config_overrides) -> Results:
    """
    Run a NetQMPI program on Aer and split the results per rank.

    Args:
        source: Body of the app module, which must define ``main(env)``.
        ranks: Number of ranks to run.
        tmp_path: Directory to write the app to.
        **config_overrides: Fields to set on the
            :class:`~netqmpi.runtime.adapters.aer.AerSimulatorConfig`.

    Returns:
        A :class:`Results` mapping each rank to its own histogram.
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
        config = AerSimulatorConfig(shots=SHOTS, seed_simulator=SEED)
        for field, value in config_overrides.items():
            setattr(config, field, value)
        executor = AerExecutorAdapter(ranks, config)
        executor.run(executor.build_apps(str(app), ranks))
    finally:
        Environment.__init__ = original

    captured.sort(key=lambda env: env.comm.rank)
    joint = dict(captured[0].comm.results or {})

    results = Results({env.comm.rank: {} for env in captured})
    widths = {env.comm.rank: sum(c.num_clbits for c in env.comm.circuits)
              for env in captured}
    for key, count in joint.items():
        bits = ascending(key)
        offset = 0
        for env in captured:
            rank = env.comm.rank
            slice_ = bits[offset:offset + widths[rank]]
            results[rank][slice_] = results[rank].get(slice_, 0) + count
            offset += widths[rank]

    results.joint = joint
    return results


def ascending(key: str) -> str:
    """
    Return a histogram key as bits in ascending classical-bit order.

    Args:
        key: A raw Qiskit histogram key, registers separated by spaces and
            printed most-significant first.

    Returns:
        The same bits with index *i* holding classical bit *i*.
    """
    return key.replace(" ", "")[::-1]


def assert_exact(histogram, expected: str) -> None:
    """Assert every shot returned the same given outcome."""
    assert histogram == {expected: SHOTS}, (
        f"expected {{{expected!r}: {SHOTS}}}, got {histogram}")


def assert_balanced(histogram) -> None:
    """Assert one bit came back about evenly split between 0 and 1."""
    total = sum(histogram.values())
    assert set(histogram) == {"0", "1"}, f"expected a single bit, got {histogram}"
    share = histogram["1"] / total
    assert abs(share - 0.5) < BALANCE_TOLERANCE, (
        f"expected an even split, got P(1) = {share:.3f} from {histogram}")


# ======================================================================
# Local circuits: does each gate of the fluent API do what it says?
# ======================================================================

LOCAL = """
    import numpy as np
    from netqmpi.sdk.environment import Environment

    def main(env: Environment = None):
        with env.comm:
            circuit = env.create_circuit(num_qubits={qubits}, num_clbits={qubits})
            {body}
            circuit.measure_all()
"""

#: ``(id, body, qubits, expected)``. Every entry is deterministic: the
#: program either prepares a basis state or undoes its own superposition,
#: so the expectation is a single bit string that must come back on every
#: shot. The bits read in ascending qubit order.
LOCAL_PROGRAMS = [
    ("x", "circuit.x(0)", 1, "1"),
    ("x_twice", "circuit.x(0); circuit.x(0)", 1, "0"),
    ("y", "circuit.y(0)", 1, "1"),
    ("h_twice", "circuit.h(0); circuit.h(0)", 1, "0"),
    ("z_flips_the_x_basis", "circuit.h(0); circuit.z(0); circuit.h(0)", 1, "1"),
    ("s_squared_is_z", "circuit.h(0); circuit.s(0); circuit.s(0); circuit.h(0)", 1, "1"),
    ("sdg_undoes_s", "circuit.h(0); circuit.s(0); circuit.sdg(0); circuit.h(0)", 1, "0"),
    ("t_to_the_fourth_is_z",
     "circuit.h(0); circuit.t(0); circuit.t(0); circuit.t(0); circuit.t(0); circuit.h(0)",
     1, "1"),
    ("tdg_undoes_t", "circuit.h(0); circuit.t(0); circuit.tdg(0); circuit.h(0)", 1, "0"),
    ("rx_pi_is_a_flip", "circuit.rx(np.pi, 0)", 1, "1"),
    ("ry_pi_is_a_flip", "circuit.ry(np.pi, 0)", 1, "1"),
    ("rz_pi_in_the_x_basis", "circuit.h(0); circuit.rz(np.pi, 0); circuit.h(0)", 1, "1"),
    ("rx_full_turn_comes_back", "circuit.rx(2 * np.pi, 0)", 1, "0"),
    ("swap_moves_the_payload", "circuit.x(0); circuit.swap(0, 1)", 2, "01"),
    ("cx_fires_on_one", "circuit.x(0); circuit.cx(0, 1)", 2, "11"),
    ("cx_is_idle_on_zero", "circuit.cx(0, 1)", 2, "00"),
    ("cz_flips_the_x_basis",
     "circuit.x(0); circuit.h(1); circuit.cz(0, 1); circuit.h(1)", 2, "11"),
    ("cz_is_idle_on_zero", "circuit.h(1); circuit.cz(0, 1); circuit.h(1)", 2, "00"),
    ("crz_fires_on_one",
     "circuit.x(0); circuit.h(1); circuit.crz(np.pi, 0, 1); circuit.h(1)", 2, "11"),
    ("crz_is_idle_on_zero",
     "circuit.h(1); circuit.crz(np.pi, 0, 1); circuit.h(1)", 2, "00"),
    ("ccx_needs_both_controls",
     "circuit.x(0); circuit.x(1); circuit.ccx(0, 1, 2)", 3, "111"),
    ("ccx_is_idle_on_one_control", "circuit.x(0); circuit.ccx(0, 1, 2)", 3, "100"),
    ("reset_clears_a_qubit", "circuit.x(0); circuit.reset(0)", 1, "0"),
    ("reset_clears_a_superposition", "circuit.h(0); circuit.reset(0)", 1, "0"),
    ("a_barrier_changes_nothing",
     "circuit.x(0); circuit.barrier(); circuit.barrier([0])", 1, "1"),
    ("bell_state_is_correlated_not_random",
     "circuit.h(0); circuit.cx(0, 1); circuit.cx(0, 1); circuit.h(0)", 2, "00"),
]


@pytest.mark.parametrize("body, qubits, expected",
                         [row[1:] for row in LOCAL_PROGRAMS],
                         ids=[row[0] for row in LOCAL_PROGRAMS])
def test_local_gate_semantics(body, qubits, expected, tmp_path):
    """Each gate of the fluent API, run and read back."""
    results = run_app(LOCAL.format(body=body, qubits=qubits), 1, tmp_path)
    assert_exact(results[0], expected)


def test_a_superposition_really_is_one(tmp_path):
    """
    The deterministic tests above cannot tell H from X, so this one does.

    H must produce an even split, not a flipped bit, and that is the
    property every echo elsewhere is built on.
    """
    results = run_app(LOCAL.format(body="circuit.h(0)", qubits=1), 1, tmp_path)
    assert_balanced(results[0])


def test_measure_writes_where_the_program_asks(tmp_path):
    """A qubit can be measured into any classical bit, not just its own."""
    source = """
        from netqmpi.sdk.environment import Environment

        def main(env: Environment = None):
            with env.comm:
                circuit = env.create_circuit(num_qubits=2, num_clbits=2)
                circuit.x(0)
                circuit.measure(0, 1)      # qubit 0 into classical bit 1
                circuit.measure(1, 0)
    """
    results = run_app(source, 1, tmp_path)
    assert_exact(results[0], "01")


# ======================================================================
# Point-to-point transfers
# ======================================================================

TELEPORT = """
    from netqmpi.sdk.environment import Environment

    def main(env: Environment = None):
        comm, rank = env.comm, env.comm.rank
        with comm:
            circuit = env.create_circuit(num_qubits=1, num_clbits=1)
            if rank == 0:
                circuit.x(0)                    # payload |1>
                comm.qsend(circuit, [0], 1)
            else:
                comm.qrecv(circuit, [0], 0)
            circuit.measure(0, 0)
"""

RELAY = """
    from netqmpi.sdk.environment import Environment

    def main(env: Environment = None):
        comm, rank, size = env.comm, env.comm.rank, env.comm.size
        with comm:
            circuit = env.create_circuit(num_qubits=1, num_clbits=1)
            if rank == 0:
                circuit.x(0)
            else:
                comm.qrecv(circuit, [0], rank - 1)
            if rank < size - 1:
                comm.qsend(circuit, [0], rank + 1)
            circuit.measure(0, 0)
"""

BULK = """
    from netqmpi.sdk.environment import Environment

    def main(env: Environment = None):
        comm, rank = env.comm, env.comm.rank
        with comm:
            circuit = env.create_circuit(num_qubits=3, num_clbits=3)
            if rank == 0:
                circuit.x(0); circuit.x(2)      # |101>
                comm.qsend(circuit, [0, 1, 2], 1)
            else:
                comm.qrecv(circuit, [0, 1, 2], 0)
            circuit.measure_all()
"""

ROUND_TRIP = """
    from netqmpi.sdk.environment import Environment

    def main(env: Environment = None):
        comm, rank = env.comm, env.comm.rank
        with comm:
            circuit = env.create_circuit(num_qubits=1, num_clbits=1)
            if rank == 0:
                circuit.h(0)                    # a state worth preserving
                comm.qsend(circuit, [0], 1)
                comm.qrecv(circuit, [0], 1)
                circuit.h(0)                    # undo it: |0> if intact
            else:
                comm.qrecv(circuit, [0], 0)
                comm.qsend(circuit, [0], 0)
            circuit.measure(0, 0)
"""


def test_a_qubit_moves_to_the_rank_that_receives_it(tmp_path):
    """The sender is left with |0>: a transfer moves, it does not copy."""
    results = run_app(TELEPORT, 2, tmp_path)
    assert_exact(results[0], "0")
    assert_exact(results[1], "1")


@pytest.mark.parametrize("ranks", [2, 3, 5])
def test_a_payload_survives_a_relay_down_the_chain(ranks, tmp_path):
    """Each hop moves it on, and only the last rank still holds it."""
    results = run_app(RELAY, ranks, tmp_path)
    for rank in range(ranks - 1):
        assert_exact(results[rank], "0")
    assert_exact(results[ranks - 1], "1")


def test_several_qubits_move_in_one_call(tmp_path):
    """``qsend([0, 1, 2], …)`` keeps the order of the buffer."""
    results = run_app(BULK, 2, tmp_path)
    assert_exact(results[0], "000")
    assert_exact(results[1], "101")


def test_a_superposition_survives_a_round_trip(tmp_path):
    """
    An echo, so the claim is about the *state* and not about a bit.

    |+> goes to rank 1 and comes back; undoing the H must give |0> on
    every shot, which it only does if the state made the trip intact.
    """
    results = run_app(ROUND_TRIP, 2, tmp_path)
    assert_exact(results[0], "0")


def test_the_ranks_may_ask_for_registers_of_different_widths(tmp_path):
    """
    A rank's slice is sized to its own request, not to a common width.

    Getting this wrong overlapped the ranks' slices in the global circuit
    and silently corrupted the program, which is why the layout waits for
    every rank to finish tracing.
    """
    source = """
        from netqmpi.sdk.environment import Environment

        def main(env: Environment = None):
            comm, rank = env.comm, env.comm.rank
            with comm:
                if rank == 0:
                    circuit = env.create_circuit(num_qubits=4, num_clbits=4)
                    circuit.x(3)
                    comm.qsend(circuit, [3], 1)
                    circuit.measure_all()
                else:
                    circuit = env.create_circuit(num_qubits=1, num_clbits=1)
                    comm.qrecv(circuit, [0], 0)
                    circuit.measure(0, 0)
    """
    results = run_app(source, 2, tmp_path)
    assert_exact(results[0], "0000")
    assert_exact(results[1], "1")


# ======================================================================
# Rooted collectives
# ======================================================================

SCATTER = """
    from netqmpi.sdk.environment import Environment

    def main(env: Environment = None):
        comm, rank, size = env.comm, env.comm.rank, env.comm.size
        with comm:
            if rank == 0:
                circuit = env.create_circuit(num_qubits=size - 1, num_clbits=size - 1)
                for q in range(size - 1):
                    circuit.x(q)
                comm.qscatter(circuit, list(range(size - 1)), root=0)
                circuit.measure_all()
            else:
                circuit = env.create_circuit(num_qubits=1, num_clbits=1)
                mine = comm.qscatter(circuit, [0], root=0)
                circuit.measure(mine[0], 0)
"""

SCATTER_CHUNKS = """
    from netqmpi.sdk.environment import Environment

    def main(env: Environment = None):
        comm, rank = env.comm, env.comm.rank
        with comm:
            if rank == 0:
                # Two qubits per receiver: |10> for rank 1, |01> for rank 2.
                circuit = env.create_circuit(num_qubits=4, num_clbits=4)
                circuit.x(0); circuit.x(3)
                comm.qscatter(circuit, [0, 1, 2, 3], root=0)
                circuit.measure_all()
            else:
                circuit = env.create_circuit(num_qubits=2, num_clbits=2)
                mine = comm.qscatter(circuit, [0, 1], root=0)
                for i, q in enumerate(mine):
                    circuit.measure(q, i)
"""

GATHER = """
    from netqmpi.sdk.environment import Environment

    def main(env: Environment = None):
        comm, rank, size = env.comm, env.comm.rank, env.comm.size
        with comm:
            if rank == 0:
                circuit = env.create_circuit(num_qubits=size, num_clbits=size)
                circuit.x(0)                       # the root's own chunk
                comm.qgather(circuit, list(range(size)), root=0)
                circuit.measure_all()
            else:
                circuit = env.create_circuit(num_qubits=1, num_clbits=1)
                circuit.x(0)
                comm.qgather(circuit, [0], root=0)
                circuit.measure(0, 0)
"""

SCATTER_THEN_GATHER = """
    from netqmpi.sdk.environment import Environment

    def main(env: Environment = None):
        comm, rank, size = env.comm, env.comm.rank, env.comm.size
        with comm:
            if rank == 0:
                circuit = env.create_circuit(num_qubits=size, num_clbits=size)
                circuit.x(1); circuit.x(2)         # one payload per receiver
                comm.qscatter(circuit, [1, 2], root=0)
                comm.qgather(circuit, [0, 1, 2], root=0)
                circuit.measure_all()
            else:
                circuit = env.create_circuit(num_qubits=1, num_clbits=1)
                mine = comm.qscatter(circuit, [0], root=0)
                circuit.x(mine[0])                 # flip it while it is here
                comm.qgather(circuit, [0], root=0)
                circuit.measure(0, 0)
"""


def test_a_scatter_hands_the_root_buffer_out(tmp_path):
    """Every receiver reads the |1> the root prepared; the root keeps none."""
    results = run_app(SCATTER, 3, tmp_path)
    assert_exact(results[0], "00")
    assert_exact(results[1], "1")
    assert_exact(results[2], "1")


def test_a_scatter_splits_the_buffer_in_rank_order(tmp_path):
    """Chunks larger than one qubit land in the order the root laid them."""
    results = run_app(SCATTER_CHUNKS, 3, tmp_path)
    assert_exact(results[0], "0000")
    assert_exact(results[1], "10")
    assert_exact(results[2], "01")


def test_a_gather_collects_every_rank_chunk(tmp_path):
    """The root reads a 1 on every slot; the contributors are left at 0."""
    results = run_app(GATHER, 3, tmp_path)
    assert_exact(results[0], "111")
    assert_exact(results[1], "0")
    assert_exact(results[2], "0")


def test_a_scatter_and_a_gather_compose(tmp_path):
    """
    Out and back: what the receivers did to their chunk comes home.

    The root scatters two qubits in ``|1>``, each receiver flips its own,
    and the gather brings them back — so the root must read them as ``0``
    while its own untouched chunk stays ``0`` too.
    """
    results = run_app(SCATTER_THEN_GATHER, 3, tmp_path)
    assert_exact(results[0], "000")
    assert_exact(results[1], "0")
    assert_exact(results[2], "0")


# ======================================================================
# Telegate windows
# ======================================================================

REMOTE_CONTROL = """
    from netqmpi.sdk.environment import Environment

    def main(env: Environment = None):
        comm, rank, size = env.comm, env.comm.rank, env.comm.size
        receivers = list(range(1, size))
        with comm:
            circuit = env.create_circuit(num_qubits=1, num_clbits=1)
            if rank == 0 and {payload}:
                circuit.x(0)
            handle = comm.expose(circuit, 0, receivers, root=0)
            if rank in receivers:
                circuit.cx(handle, 0)
            comm.unexpose(circuit, receivers, root=0)
            circuit.measure(0, 0)
"""

ENTANGLING_WINDOW = """
    from netqmpi.sdk.environment import Environment

    def main(env: Environment = None):
        comm, rank = env.comm, env.comm.rank
        with comm:
            circuit = env.create_circuit(num_qubits=1, num_clbits=1)
            if rank == 0:
                circuit.h(0)                       # control in superposition
            handle = comm.expose(circuit, 0, [1], root=0)
            if rank == 1:
                circuit.cx(handle, 0)              # entangles the two ranks
            comm.unexpose(circuit, [1], root=0)
            circuit.measure(0, 0)
"""

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

WINDOW_REOPENED = """
    from netqmpi.sdk.environment import Environment

    def main(env: Environment = None):
        comm, rank = env.comm, env.comm.rank
        with comm:
            circuit = env.create_circuit(num_qubits=1, num_clbits=1)
            if rank == 0:
                circuit.x(0)
            for _ in range(3):
                handle = comm.expose(circuit, 0, [1], root=0)
                if rank == 1:
                    circuit.cx(handle, 0)          # flips on every window
                comm.unexpose(circuit, [1], root=0)
            circuit.measure(0, 0)
"""


@pytest.mark.parametrize("payload, expected", [(0, "0"), (1, "1")])
@pytest.mark.parametrize("ranks", [2, 3, 4])
def test_an_exposed_control_drives_every_receivers_gate(payload, expected,
                                                        ranks, tmp_path):
    """One window serves the whole group, and the control is read correctly."""
    results = run_app(REMOTE_CONTROL.format(payload=payload), ranks, tmp_path)
    assert_exact(results[0], str(payload))       # the root keeps its state
    for rank in range(1, ranks):
        assert_exact(results[rank], expected)


def test_a_window_can_entangle_the_ranks_it_spans(tmp_path):
    """
    The control is lent, not measured, so superposition survives it.

    Rank 0 lends ``|+>`` and rank 1 uses it as a control: the two ranks
    end up in a Bell state, which shows as perfect correlation in the
    joint histogram and an even split in each rank's own.
    """
    results = run_app(ENTANGLING_WINDOW, 2, tmp_path)

    assert_balanced(results[0])
    assert_balanced(results[1])

    correlated = {key: count for key, count in results.joint.items()
                  if ascending(key)[0] == ascending(key)[1]}
    assert sum(correlated.values()) == SHOTS, (
        f"the ranks must agree on every shot, got {results.joint}")


def test_windows_over_different_groups_nest(tmp_path):
    """
    The shape ``5_qft_expose`` uses: one window open across another.

    Rank 2 lends its ``|1>`` to ranks 0 and 1 for the whole block, while
    rank 1's own window over rank 0 opens and closes inside it. Both
    receivers must see rank 2's control as ``1`` and rank 1's as ``0``.
    """
    results = run_app(NESTED_WINDOWS, 3, tmp_path)
    assert_exact(results[0], "1")       # flipped once, by the outer control
    assert_exact(results[1], "1")       # flipped by the outer control
    assert_exact(results[2], "1")       # its own payload, given back intact


def test_a_window_can_be_opened_again_after_it_closes(tmp_path):
    """Three windows in a row reuse the slot and each one fires."""
    results = run_app(WINDOW_REOPENED, 2, tmp_path)
    assert_exact(results[0], "1")
    assert_exact(results[1], "1")       # flipped three times


# ======================================================================
# Several programs in one run
# ======================================================================

def test_a_rank_may_run_more_than_one_distributed_program(tmp_path):
    """
    Circuits are paired across ranks by creation order.

    The *i*-th circuit of every rank is one distributed program, so two
    circuits per rank are two independent programs — and a transfer in the
    second must not be paired with one in the first.
    """
    source = """
        from netqmpi.sdk.environment import Environment

        def main(env: Environment = None):
            comm, rank = env.comm, env.comm.rank
            with comm:
                first = env.create_circuit(num_qubits=1, num_clbits=1)
                second = env.create_circuit(num_qubits=1, num_clbits=1)

                if rank == 0:
                    first.x(0)
                    comm.qsend(first, [0], 1)
                    comm.qrecv(second, [0], 1)
                else:
                    comm.qrecv(first, [0], 0)
                    second.x(0)
                    comm.qsend(second, [0], 0)

                first.measure(0, 0)
                second.measure(0, 0)
    """
    results = run_app(source, 2, tmp_path)

    # The layout is program-major — program 0's ranks, then program 1's —
    # so the bits read: rank 0 and rank 1 of the first program, then of the
    # second. The payload of each program sits on the rank that received
    # it, and nowhere else.
    assert len(results.joint) == 1, results.joint
    bits = ascending(next(iter(results.joint)))
    assert bits == "0110", f"expected one payload on each receiver, got {bits}"


# ======================================================================
# What the backend refuses, and how it says so
# ======================================================================

DANGLING_SEND = """
    from netqmpi.sdk.environment import Environment

    def main(env: Environment = None):
        comm, rank = env.comm, env.comm.rank
        with comm:
            circuit = env.create_circuit(num_qubits=1, num_clbits=1)
            if rank == 0:
                circuit.x(0)
                comm.qsend(circuit, [0], 1)        # nobody receives it
            circuit.measure(0, 0)
"""

CROSSED_WINDOWS = """
    from netqmpi.sdk.environment import Environment

    def main(env: Environment = None):
        comm, rank = env.comm, env.comm.rank
        other = 1 - rank
        with comm:
            circuit = env.create_circuit(num_qubits=1, num_clbits=1)
            # Each rank opens a window of its own and waits for the other,
            # which is waiting for the opposite: the collective equivalent
            # of two processes that both post a receive.
            handle = comm.expose(circuit, 0, [other], root=rank)
            comm.unexpose(circuit, [other], root=rank)
            circuit.measure(0, 0)
"""

LONELY_WINDOW = """
    from netqmpi.sdk.environment import Environment

    def main(env: Environment = None):
        comm, rank = env.comm, env.comm.rank
        with comm:
            circuit = env.create_circuit(num_qubits=1, num_clbits=1)
            if rank == 0:
                comm.expose(circuit, 0, [1], root=0)   # rank 1 never joins
                comm.unexpose(circuit, [1], root=0)
            circuit.measure(0, 0)
"""

MISMATCHED_COUNTS = """
    from netqmpi.sdk.environment import Environment

    def main(env: Environment = None):
        comm, rank = env.comm, env.comm.rank
        with comm:
            circuit = env.create_circuit(num_qubits=1, num_clbits=1)
            if rank == 0:
                env.create_circuit(num_qubits=1, num_clbits=1)
            circuit.measure(0, 0)
"""

CLASSICAL_CONTROL = """
    from netqmpi.sdk.environment import Environment
    from netqmpi.sdk.operations import ClassicalControlledGate, Gate

    def main(env: Environment = None):
        with env.comm:
            circuit = env.create_circuit(num_qubits=2, num_clbits=2)
            circuit.measure(0, 0)
            circuit._add(ClassicalControlledGate([0], [Gate('X', [1])]))
            circuit.measure(1, 1)
"""


def test_a_send_nobody_receives_is_reported_not_hung(tmp_path):
    """
    A deadlock at trace time is named, with what each rank was waiting for.

    Reporting it is the whole point: the alternative is a run that hangs,
    or worse, one that quietly drops the transfer.
    """
    with pytest.raises(RuntimeError, match="never match") as error:
        run_app(DANGLING_SEND, 2, tmp_path)

    message = str(error.value)
    assert "rank 0 is sending qubits [0] to rank 1" in message
    assert "Every qsend needs a qrecv" in message


def test_two_windows_waiting_for_each_other_are_reported(tmp_path):
    """Each rank is named both as a caller and as the missing participant."""
    with pytest.raises(RuntimeError, match="never match") as error:
        run_app(CROSSED_WINDOWS, 2, tmp_path)

    message = str(error.value)
    assert "rank 0 is in expose over ranks [0, 1]" in message
    assert "rank 1 is in expose over ranks [1, 0]" in message
    assert "still waiting for [1]" in message


def test_a_window_its_group_never_joins_is_reported(tmp_path):
    """A collective reached by one rank only cannot be completed."""
    with pytest.raises(RuntimeError, match="never match") as error:
        run_app(LONELY_WINDOW, 2, tmp_path)
    assert "still waiting for [1]" in str(error.value)


def test_ranks_must_create_the_same_number_of_circuits(tmp_path):
    """Otherwise the circuits cannot be paired into programs at all."""
    with pytest.raises(RuntimeError, match="same number of circuits") as error:
        run_app(MISMATCHED_COUNTS, 2, tmp_path)
    assert "{0: 2, 1: 1}" in str(error.value)


def test_an_unsupported_operation_is_named_with_its_backend(tmp_path):
    """
    Not every operation the SDK can record is implemented everywhere.

    A classically controlled gate is one of them, and the refusal names
    both the operation and the backend rather than dropping it.
    """
    with pytest.raises(NotImplementedError,
                       match="ClassicalControlledGate is not yet implemented"):
        run_app(CLASSICAL_CONTROL, 1, tmp_path)


def test_the_teleport_transfer_mode_is_refused_for_now(tmp_path):
    """
    The config accepts it; the adapter says plainly that it is not built.

    On a noiseless simulator a teleportation circuit returns exactly what
    the SWAP returns, so it would cost gates and ancillas without adding
    information — but silently running a SWAP for it would be a lie.
    """
    with pytest.raises(NotImplementedError, match="transfer_mode='teleport'"):
        run_app(TELEPORT, 2, tmp_path, transfer_mode="teleport")


def test_the_shot_count_is_honoured(tmp_path):
    """``--shots`` has to reach the simulator, not just the config object."""
    results = run_app(TELEPORT, 2, tmp_path, shots=64)
    assert sum(results.joint.values()) == 64


def test_seeding_makes_a_run_reproducible(tmp_path):
    """The same seed and the same program give the same histogram."""
    first = run_app(LOCAL.format(body="circuit.h(0)", qubits=1), 1, tmp_path)
    second = run_app(LOCAL.format(body="circuit.h(0)", qubits=1), 1, tmp_path)
    assert first.joint == second.joint


# ----------------------------------------------------------------------
# Adapter-level guards, reached without a run
# ----------------------------------------------------------------------

def test_the_rooted_collectives_are_never_translated_as_blocks():
    """
    They flatten into their transfers, so these hooks stay unreachable.

    The scatter and gather tests above pass *because* the container
    flattens to the ``QSend``/``QRecv`` pairs the adapter does implement.
    Reaching the hook itself would mean the flattening had stopped
    happening, so it refuses rather than emitting a plausible nothing.
    """
    from netqmpi.runtime.adapters.aer.aer_circuit import AerCircuitAdapter
    from netqmpi.sdk.operations import QGather, QScatter

    scatter = QScatter(rank=0, root=0, ranks=[0, 1], qubits=[0])
    gather = QGather(rank=0, root=0, ranks=[0, 1], qubits=[0])

    with pytest.raises(NotImplementedError, match="QScatter"):
        AerCircuitAdapter.__dict__["_translate_qscatter"](None, scatter)
    with pytest.raises(NotImplementedError, match="QGather"):
        AerCircuitAdapter.__dict__["_translate_qgather"](None, gather)


def test_a_transfer_whose_halves_disagree_is_refused(tmp_path):
    """
    The two sides must move the same number of qubits.

    The SDK cannot produce this — it records one transfer per qubit — so
    the guard is reached directly, as a backend written against the
    records would reach it.
    """
    from netqmpi.runtime.adapters.aer.aer_circuit import _emit_transfer
    from netqmpi.sdk.operations import QRecv, QSend

    with pytest.raises(RuntimeError, match="moves 2 qubit"):
        _emit_transfer({}, "tag",
                       (0, QSend([0, 1], 1, tag="tag")),
                       (1, QRecv([0], 0, tag="tag")))


def test_a_communication_qubit_outside_its_window_is_refused():
    """
    The adapter's own guard behind the trace-time one.

    A closed window is normally caught while tracing, where the traceback
    still points at the user's line; this is the backstop for a record
    that reached translation anyway.
    """
    from netqmpi.runtime.adapters.aer.aer_circuit import AerCircuitAdapter

    from conftest import StubCommunicator

    comm = StubCommunicator(rank=0, size=2)
    comm._config = AerSimulatorConfig()
    adapter = AerCircuitAdapter(1, 1, comm)

    with pytest.raises(RuntimeError, match="no expose window open"):
        adapter._global(adapter.comm_qubit(0))


# ----------------------------------------------------------------------
# Known gaps
# ----------------------------------------------------------------------

CONTROLLED_PHASE = """
    import numpy as np
    from netqmpi.sdk.environment import Environment

    def main(env: Environment = None):
        with env.comm:
            circuit = env.create_circuit(num_qubits=2, num_clbits=2)
            circuit.x(0)                       # control |1>
            circuit.h(1)
            circuit.{call}                     # a pi phase, in two halves
            circuit.{call}
            circuit.h(1)
            circuit.measure_all()
"""


@pytest.mark.parametrize("call", [
    "cs(0, 1)",
    "cp(0, 1, np.pi / 2)",
])
def test_controlled_phase_gates_reach_the_simulator(call, tmp_path):
    """
    Two quarter turns of controlled phase make a controlled-Z.

    With the control in ``|1>`` and the target in ``|+>``, that flips the
    target, so a working ``cs``/``cp`` reads ``11`` and a dropped one
    reads ``10``.
    """
    results = run_app(CONTROLLED_PHASE.format(call=call), 1, tmp_path)
    assert_exact(results[0], "11")
