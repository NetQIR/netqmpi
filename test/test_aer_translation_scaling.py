"""
Regression test for the cost of the Aer adapter's joint translation pass.

:func:`~netqmpi.runtime.adapters.aer.aer_circuit.translate_group` used to
re-scan every rank after every transfer it emitted, rebuilding the set of
blocked calls from scratch each time. The work per operation therefore grew
with the number of ranks: in the multi-backend benchmark a cascade cost
~15 µs/op to translate up to 32 ranks and 67.5 µs/op at 256, with the same
register width split differently (256 ranks x 1 qubit against 64 x 4)
costing 67.5 against 26.1 µs/op.

Timing that is fragile, so the test counts work instead: every step of the
pass that looks at a pending operation asks what kind it is, and those
``isinstance`` checks are counted per traced operation. A pass that only
touches the ranks a call released does a bounded number of them per
operation whatever the number of ranks; the old pass did O(ranks).

No simulation runs: the circuits are traced by hand and translated into an
empty global circuit, which is all the pass needs.
"""
from __future__ import annotations

import builtins

import pytest

pytest.importorskip("qiskit", reason="the Aer backend needs Qiskit")
pytest.importorskip("qiskit_aer", reason="the Aer backend needs qiskit-aer")

from netqmpi.runtime.adapters.aer import aer_circuit  # noqa: E402
from netqmpi.runtime.adapters.aer import (  # noqa: E402
    AerExecutorAdapter, AerSimulatorConfig,
)
from netqmpi.runtime.adapters.aer.aer_circuit import AerCircuitAdapter  # noqa: E402

from conftest import StubCommunicator  # noqa: E402


def traced_cascade(size: int, qubits: int = 1):
    """
    Trace ``scripts/benchmark/apps/cascade.py`` on every rank, without running it.

    Args:
        size: Number of ranks.
        qubits: Qubits cascaded down the chain.

    Returns:
        The rank-ordered circuit adapters, laid out in a global circuit.
    """
    config = AerSimulatorConfig()
    group = []
    for rank in range(size):
        comm = StubCommunicator(rank, size)
        comm._config = config
        circuit = AerCircuitAdapter(qubits, qubits, comm)
        comm.circuits.append(circuit)
        chain = list(range(qubits))
        if rank == 0:
            for i in chain:
                circuit.h(i)
            comm.qsend(circuit, chain, 1)
        else:
            comm.qrecv(circuit, chain, rank - 1)
            if rank < size - 1:
                comm.qsend(circuit, chain, rank + 1)
            else:
                for i in chain:
                    circuit.h(i)
        for i in chain:
            circuit.measure(i, i)
        group.append(circuit)

    AerExecutorAdapter(size, config).lay_out([group])
    return group


def checks_per_operation(size: int, monkeypatch) -> float:
    """
    Translate a traced cascade and count the pass's type checks per operation.

    Args:
        size: Number of ranks.
        monkeypatch: The pytest fixture, used to shadow ``isinstance`` in the
            adapter module only.

    Returns:
        ``isinstance`` calls made by the pass, per traced operation.
    """
    group = traced_cascade(size)
    operations = sum(len(list(circuit.ops.flatten())) for circuit in group)

    calls = 0

    def counting_isinstance(obj, cls):
        nonlocal calls
        calls += 1
        return builtins.isinstance(obj, cls)

    monkeypatch.setattr(aer_circuit, "isinstance", counting_isinstance,
                        raising=False)
    try:
        aer_circuit.translate_group(dict(enumerate(group)))
    finally:
        monkeypatch.delattr(aer_circuit, "isinstance")

    return calls / operations


def test_translation_work_per_operation_does_not_grow_with_ranks(monkeypatch):
    """The pass does the same work per operation at 256 ranks as at 8."""
    small = checks_per_operation(8, monkeypatch)
    large = checks_per_operation(256, monkeypatch)
    assert large <= 1.5 * small, (
        f"{large:.1f} type checks per operation at 256 ranks against "
        f"{small:.1f} at 8: the pass is scanning ranks it did not release")


@pytest.mark.parametrize("size", [2, 8, 32])
def test_worklist_emits_every_transfer_in_chain_order(size):
    """Each hop of the chain is emitted once, in the order the chain runs."""
    group = traced_cascade(size)
    aer_circuit.translate_group(dict(enumerate(group)))

    circuit = group[0]._global_circuit
    swaps = [tuple(circuit.find_bit(q).index for q in instruction.qubits)
             for instruction in circuit.data
             if instruction.operation.name == "swap"]
    assert swaps == [(rank, rank + 1) for rank in range(size - 1)]


# ----------------------------------------------------------------------
# A2 — the simulator's method and threads come from the config
# ----------------------------------------------------------------------

def test_simulator_method_and_threads_come_from_the_config(monkeypatch):
    """``method`` and ``max_parallel_threads`` reach AerSimulator."""
    from netqmpi.runtime.adapters.aer import aer_executor

    created = []

    class RecordingSimulator(aer_executor.AerSimulator):
        def __init__(self, *args, **kwargs):
            created.append(kwargs)
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(aer_executor, "AerSimulator", RecordingSimulator)

    config = AerSimulatorConfig(shots=8, method="stabilizer",
                                max_parallel_threads=1)
    group = traced_cascade(3)
    executor = AerExecutorAdapter(3, config)
    executor.lay_out([group])
    aer_circuit.translate_group(dict(enumerate(group)))
    executor._global_circuit = group[0]._global_circuit
    executor._run_simulation()

    assert created == [{"method": "stabilizer", "max_parallel_threads": 1}]


def test_default_simulator_settings_are_aers_own():
    config = AerSimulatorConfig()
    assert (config.method, config.max_parallel_threads) == ("automatic", 0)
