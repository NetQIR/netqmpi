"""
``comm.counts``: every rank's histogram, keyed by rank, on every backend.

``comm.results`` keeps each backend's native shape; ``comm.counts`` is the
portable one the examples print. These tests cover the SDK default and the
Aer adapter, which is the only backend that has to work it out (it runs all
ranks in one circuit, so it splits the joint histogram by register).
"""
from __future__ import annotations

import textwrap

import pytest

from netqmpi.sdk.communicator import QMPICommunicator
from netqmpi.sdk.environment import Environment


def test_counts_start_empty():
    class Dummy(QMPICommunicator):
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return None

        def qsend(self, *args, **kwargs):
            raise NotImplementedError

        def qrecv(self, *args, **kwargs):
            raise NotImplementedError

    try:
        comm = Dummy(0, 1)
    except TypeError as error:          # more abstract methods than stubbed
        pytest.skip(f"cannot instantiate a bare communicator: {error}")
    assert comm.counts == {}


aer_executor = pytest.importorskip(
    "netqmpi.runtime.adapters.aer.aer_executor",
    reason="the Aer backend needs Qiskit and qiskit-aer")


# ----------------------------------------------------------------------
# split_by_register
# ----------------------------------------------------------------------

def test_split_reverses_qiskit_register_order():
    # Qiskit prints the last register first: 'c0_r1 c0_r0' reads '1 00'.
    split = aer_executor.split_by_register(
        {"1 00": 3, "0 01": 1}, ["c0_r0", "c0_r1"])
    assert split == {"c0_r0": {"00": 3, "01": 1}, "c0_r1": {"1": 3, "0": 1}}


def test_split_adds_up_marginals():
    split = aer_executor.split_by_register(
        {"0 0": 2, "0 1": 5, "1 1": 7}, ["a", "b"])
    assert split["a"] == {"0": 2, "1": 12}
    assert split["b"] == {"0": 7, "1": 7}


def test_split_rejects_a_mismatched_layout():
    with pytest.raises(RuntimeError, match="registers"):
        aer_executor.split_by_register({"0 0 0": 1}, ["a", "b"])


# ----------------------------------------------------------------------
# End to end on Aer
# ----------------------------------------------------------------------

SHOTS = 256


def run(source: str, ranks: int, tmp_path):
    """Run an app on Aer and return every rank's communicator, by rank."""
    from netqmpi.runtime.adapters.aer import AerExecutorAdapter, AerSimulatorConfig

    app = tmp_path / "app.py"
    app.write_text(textwrap.dedent(source))

    captured = []
    original = Environment.__init__

    def capture(self, comm, executor):
        original(self, comm, executor)
        captured.append(self)

    Environment.__init__ = capture
    try:
        config = AerSimulatorConfig()
        config.shots = SHOTS
        executor = AerExecutorAdapter(ranks, config)
        executor.run(executor.build_apps(str(app), ranks))
    finally:
        Environment.__init__ = original

    return [env.comm for env in sorted(captured, key=lambda e: e.comm.rank)]


def test_every_rank_gets_the_same_rank_keyed_counts(tmp_path):
    comms = run("""
        def main(env=None):
            comm = env.comm
            with comm:
                circuit = env.create_circuit(num_qubits=comm.rank + 1,
                                             num_clbits=comm.rank + 1)
                circuit.x(comm.rank)
                circuit.measure_all()
    """, 3, tmp_path)

    expected = {0: {"1": SHOTS}, 1: {"10": SHOTS}, 2: {"100": SHOTS}}
    for comm in comms:
        assert comm.counts == expected
    # results keeps Aer's native joint histogram.
    assert comms[0].results == {"100 10 1": SHOTS}


def test_several_circuits_per_rank_give_a_list(tmp_path):
    # The global circuit is laid out program by program (c0_r0, c0_r1,
    # c1_r0, c1_r1), so slicing by position per rank would mix them up.
    comms = run("""
        def main(env=None):
            comm = env.comm
            with comm:
                first = env.create_circuit(num_qubits=1, num_clbits=1)
                second = env.create_circuit(num_qubits=2, num_clbits=2)
                if comm.rank == 0:
                    first.x(0)
                else:
                    second.x(1)
                first.measure(0, 0)
                second.measure_all()
    """, 2, tmp_path)

    assert comms[-1].counts == {
        0: [{"1": SHOTS}, {"00": SHOTS}],
        1: [{"0": SHOTS}, {"10": SHOTS}],
    }


def test_a_rank_without_classical_bits_reports_nothing(tmp_path):
    comms = run("""
        def main(env=None):
            comm = env.comm
            with comm:
                if comm.rank == 0:
                    circuit = env.create_circuit(num_qubits=1, num_clbits=1)
                    circuit.x(0)
                    circuit.measure(0, 0)
                else:
                    env.create_circuit(num_qubits=1, num_clbits=0)
    """, 2, tmp_path)

    assert comms[-1].counts == {0: {"1": SHOTS}, 1: {}}
