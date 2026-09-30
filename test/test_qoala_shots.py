"""
Regression tests for how the Qoala adapter runs its shots (Q1).

``apps/ghz.py`` always failed on Qoala with ``KeyError: 'm_8'`` (``m_16`` at
three ranks, ``m_24`` at four). The generated program is not at fault:
``m_<8(n-1)>`` is rank 0's final measurement, and the ``.iqoala`` text
measures it and returns it (checked below). The program simply never got
to its last block.

The adapter submitted every shot as one batch, and Qoala runs the instances
of a batch concurrently on nodes whose physical qubits they share. ``ghz``
keeps its control qubit allocated while the tour is away; one shot holding
it, and another holding the rest of the node's memory, each waited for
memory the other would only free later. Qoala's scheduler waits for memory
to be freed rather than failing, NetSquid stops when no event is left, and
the results came back with programs unfinished.

These tests need no simulator. The ones that run Qoala itself are in
``test_qoala_backend.py``.
"""
from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from netqmpi.runtime.adapters.qoala.qoala_circuit import QoalaCircuitAdapter
from netqmpi.runtime.adapters.qoala.qoala_executor import (
    QoalaExecutorAdapter, QoalaRunConfig,
)

from conftest import StubCommunicator

GHZ = Path(__file__).resolve().parents[1] / "scripts/benchmark/apps/ghz.py"


def compile_ghz(ranks: int, monkeypatch):
    """Compile ``apps/ghz.py`` (one qubit per rank) for every rank."""
    monkeypatch.setenv("NQB_QUBITS_PER_RANK", "1")
    main = runpy.run_path(str(GHZ))["main"]
    specs = {}
    for rank in range(ranks):
        comm = StubCommunicator(rank, ranks)

        def create_circuit(num_qubits, num_clbits, comm=comm):
            circuit = QoalaCircuitAdapter(num_qubits, num_clbits, comm)
            comm.circuits.append(circuit)
            return circuit

        main(env=SimpleNamespace(comm=comm, create_circuit=create_circuit))
        specs[rank] = comm.circuits[0].build_program()
    return specs


@pytest.mark.parametrize("ranks", [2, 3, 4])
def test_the_missing_variable_is_the_final_measurement(ranks, monkeypatch):
    """``m_<8(n-1)>`` is measured and returned: nothing is lost in compiling."""
    spec = compile_ghz(ranks, monkeypatch)[0]
    var = f"m_{8 * (ranks - 1)}"
    assert spec.outputs == [(0, var)]
    assert f"returns: {var}" in spec.iqoala_text
    assert f"return_result({var})" in spec.iqoala_text


def test_an_unfinished_program_is_reported_as_a_deadlock():
    """A result without its returned variable is a deadlock, and says so."""
    finished = SimpleNamespace(values={"m_8": 0})
    unfinished = SimpleNamespace(values={"m_0": 1, "m_1": 0})
    with pytest.raises(RuntimeError, match=r"rank 0's program unfinished.*m_8.*deadlock"):
        QoalaExecutorAdapter._build_counts([finished, unfinished],
                                           [(0, "m_8")], rank=0)


def test_one_shot_per_simulation_by_default():
    assert QoalaRunConfig().concurrent_shots == 1
    assert QoalaExecutorAdapter.plan_simulations(5, 1) == [1, 1, 1, 1, 1]


@pytest.mark.parametrize("shots, concurrent, plan", [
    (10, 4, [4, 4, 2]), (3, 8, [3]), (0, 1, [1]), (1, 1, [1]),
])
def test_shots_are_split_into_simulations(shots, concurrent, plan):
    assert QoalaExecutorAdapter.plan_simulations(shots, concurrent) == plan


def test_shots_sharing_a_node_each_get_all_they_need():
    """With k shots per simulation, each node holds k shots' worth of qubits."""
    assert QoalaExecutorAdapter.qubits_per_node(3, 1) == 3
    assert QoalaExecutorAdapter.qubits_per_node(3, 4) == 12


def test_concurrent_shots_must_be_positive():
    with pytest.raises(ValueError, match="concurrent_shots"):
        QoalaRunConfig(concurrent_shots=0)


class FakeResults:
    """Results for a fake simulation: every shot reads all-zero."""

    def __init__(self, outputs):
        self.results = [SimpleNamespace(values={var: 0 for _, var in outputs})]


def test_each_simulation_runs_its_share_of_the_shots(monkeypatch):
    """The executor runs the plan, seeds each simulation, and pools results."""
    executor = QoalaExecutorAdapter(2, QoalaRunConfig(shots=5, concurrent_shots=2,
                                                      seed=7))
    spec = SimpleNamespace(num_qubits=3, outputs=[(0, "m_8")], iqoala_text="")
    registry = {0: (spec, None), 1: (spec, None)}
    calls = []

    def simulate(ranks, registry, programs, topology, shots, seed):
        calls.append((shots, seed, topology))
        return {r: [SimpleNamespace(values={"m_8": 0})] * shots for r in ranks}

    parser = SimpleNamespace(QoalaParser=lambda text: SimpleNamespace(parse=lambda: text))
    monkeypatch.setitem(sys.modules, "qoala", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "qoala.lang", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "qoala.lang.parse", parser)
    monkeypatch.setattr(executor, "_simulate", simulate)
    monkeypatch.setattr(executor, "_build_topology", lambda n: f"{n} qubits")

    counts = executor.run_simulation(registry)

    assert [(shots, seed) for shots, seed, _ in calls] == [(2, 7), (2, 8), (1, 9)]
    assert {topology for _, _, topology in calls} == {"6 qubits"}
    assert counts == {0: {"0": 5}, 1: {"0": 5}}
