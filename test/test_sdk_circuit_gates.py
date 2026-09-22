"""
The fluent circuit API, and what it records.

:class:`~netqmpi.sdk.circuit.Circuit` is the only class a user program
talks to directly. It is not a simulator and it computes nothing: every
call appends an operation to the circuit's container and returns ``self``,
so the whole program is a *record* that backends translate later. Three
properties of that recording are worth holding still:

- **What each call records.** ``cx`` must record a controlled ``X`` and
  ``crz(theta)`` a controlled ``RZ`` carrying the angle, because the
  adapters dispatch on exactly that and a gate recorded under the wrong
  shape is one an adapter quietly drops.
- **What it refuses.** Out-of-range qubits and classical bits are caught
  at the offending line, where the traceback still points into the user's
  own code, rather than in a backend later on.
- **How it dispatches.** :meth:`~netqmpi.sdk.circuit.Circuit.translate`
  routes an operation to the hook its type maps to, walking the MRO for
  subclasses an adapter never registered.
"""
from __future__ import annotations

import math

import pytest

from netqmpi.sdk.circuit import Circuit
from netqmpi.sdk.operations import (
    Barrier,
    ClassicalControlledGate,
    ControlledGate,
    Gate,
    Measure,
    Operation,
    OperationContainer,
    QRecv,
    QSend,
    Reset,
)

THETA = math.pi / 3


# ----------------------------------------------------------------------
# What each call records
# ----------------------------------------------------------------------

#: One row per gate of the fluent API: how it is called, and the operation
#: it has to leave in the circuit.
GATE_TABLE = [
    ("h", (0,), Gate("H", [0])),
    ("x", (0,), Gate("X", [0])),
    ("y", (0,), Gate("Y", [0])),
    ("z", (0,), Gate("Z", [0])),
    ("s", (0,), Gate("S", [0])),
    ("sdg", (0,), Gate("SDG", [0])),
    ("t", (0,), Gate("T", [0])),
    ("tdg", (0,), Gate("TDG", [0])),
    ("rx", (THETA, 0), Gate("RX", [0], [THETA])),
    ("ry", (THETA, 0), Gate("RY", [0], [THETA])),
    ("rz", (THETA, 0), Gate("RZ", [0], [THETA])),
    ("swap", (0, 1), Gate("SWAP", [0, 1])),
    ("cx", (0, 1), ControlledGate([0], [Gate("X", [1])])),
    ("cz", (0, 1), ControlledGate([0], [Gate("Z", [1])])),
    ("cs", (0, 1), ControlledGate([0], [Gate("S", [1])])),
    ("ct", (0, 1), ControlledGate([0], [Gate("T", [1])])),
    ("cp", (0, 1, THETA), ControlledGate([0], [Gate("P", [1], [THETA])])),
    ("crz", (THETA, 0, 1), ControlledGate([0], [Gate("RZ", [1], [THETA])])),
    ("ccx", (0, 1, 2), ControlledGate([0, 1], [Gate("X", [2])])),
    ("reset", (0,), Reset(0)),
    ("barrier", ((0, 1),), Barrier([0, 1])),
    ("barrier", (), Barrier()),
]


@pytest.mark.parametrize("method, args, expected",
                         GATE_TABLE, ids=[row[0] for row in GATE_TABLE])
def test_every_call_records_the_operation_it_promises(method, args, expected,
                                                      make_circuit):
    """The record is the contract between the SDK and every adapter."""
    circuit = make_circuit(num_qubits=3, num_clbits=3)
    args = tuple(list(a) if isinstance(a, tuple) else a for a in args)

    result = getattr(circuit, method)(*args)

    assert result is circuit, "the fluent API must return the circuit"
    assert circuit.ops.children == [expected]


def test_measure_pairs_the_indices_it_was_given(make_circuit):
    """A qubit may be measured into any classical bit, not just its own."""
    circuit = make_circuit(num_qubits=3, num_clbits=3)
    circuit.measure(2, 0)
    assert circuit.ops.children == [Measure(2, 0)]


def test_calls_chain_in_program_order(make_circuit):
    """Chained calls land in the container in the order they were written."""
    circuit = make_circuit(num_qubits=2, num_clbits=2)
    circuit.h(0).cx(0, 1).measure(0, 0).measure(1, 1)

    assert [type(op).__name__ for op in circuit] == [
        "Gate", "ControlledGate", "Measure", "Measure"]
    assert len(circuit) == 4


def test_measure_all_covers_every_qubit(make_circuit):
    """Each qubit goes to the classical bit of the same index."""
    circuit = make_circuit(num_qubits=3, num_clbits=3)
    assert circuit.measure_all() is circuit
    assert list(circuit) == [Measure(0, 0), Measure(1, 1), Measure(2, 2)]


def test_measure_all_needs_a_wide_enough_register(make_circuit):
    """Refused up front, with both sizes named."""
    circuit = make_circuit(num_qubits=3, num_clbits=2)
    with pytest.raises(ValueError, match="2 clbits < 3 qubits"):
        circuit.measure_all()


def test_a_wider_classical_register_is_fine(make_circuit):
    """More bits than qubits is not an error; the extras stay untouched."""
    circuit = make_circuit(num_qubits=1, num_clbits=4)
    circuit.measure_all()
    assert list(circuit) == [Measure(0, 0)]


# ----------------------------------------------------------------------
# Properties
# ----------------------------------------------------------------------

def test_circuit_reports_the_widths_it_was_asked_for(make_circuit):
    """Widths are fixed at creation; the protocol pools start empty."""
    circuit = make_circuit(num_qubits=3, num_clbits=2, rank=1, size=4)

    assert circuit.num_qubits == 3
    assert circuit.num_clbits == 2
    assert circuit.num_comm_qubits == 0
    assert circuit.num_protocol_clbits == 0
    assert circuit.comm.rank == 1 and circuit.comm.size == 4
    assert isinstance(circuit.ops, OperationContainer)


def test_comm_qubits_are_addressed_right_after_the_data_ones(make_circuit):
    """That is what lets a lent control be used like any local qubit."""
    circuit = make_circuit(num_qubits=3, num_clbits=1)
    assert circuit.comm_qubit(0) == 3
    assert circuit.comm_qubit(2) == 5


def test_len_counts_blocks_while_iteration_yields_leaves(make_circuit):
    """
    ``len`` is the number of top-level entries, ``iter`` the leaves.

    A transfer is one entry either way here, but the distinction is what
    keeps a rooted collective — one entry, many leaves — readable.
    """
    circuit = make_circuit(num_qubits=2, num_clbits=2, rank=0, size=2)
    circuit.h(0)
    circuit.qsend([0], 1)

    assert len(circuit) == 2
    assert [type(op).__name__ for op in circuit] == ["Gate", "QSend"]


# ----------------------------------------------------------------------
# Index validation
# ----------------------------------------------------------------------

@pytest.mark.parametrize("method, args", [
    ("h", (5,)), ("x", (-1,)), ("rx", (THETA, 5)),
    ("cx", (0, 5)), ("cx", (5, 0)), ("ccx", (0, 1, 5)),
    ("swap", (0, 9)), ("reset", (7,)), ("measure", (5, 0)),
    ("barrier", ([0, 5],)),
])
def test_a_qubit_the_circuit_does_not_have_is_refused(method, args, make_circuit):
    """Caught at the offending line, with the valid range named."""
    circuit = make_circuit(num_qubits=3, num_clbits=3)
    with pytest.raises(IndexError, match=r"out of range \[0, 3\)"):
        getattr(circuit, method)(*args)


@pytest.mark.parametrize("cbit", [3, 99, -1])
def test_a_classical_bit_the_circuit_does_not_have_is_refused(cbit, make_circuit):
    """The same guard on the classical side."""
    circuit = make_circuit(num_qubits=3, num_clbits=3)
    with pytest.raises(IndexError, match=r"Classical bit index .* out of range"):
        circuit.measure(0, cbit)


def test_a_communication_qubit_is_not_addressable_without_a_window(make_circuit):
    """
    Only ``expose`` produces one, so until then the range stops at the data.

    The message names the split — so many data qubits, so many
    communication ones — because that is the question a user hitting it is
    actually asking.
    """
    circuit = make_circuit(num_qubits=2, num_clbits=2)
    with pytest.raises(IndexError, match=r"\(2 data \+ 0 comm qubits\)"):
        circuit.h(circuit.comm_qubit(0))


def test_nothing_is_recorded_when_a_call_is_refused(make_circuit):
    """A rejected call must leave the program exactly as it was."""
    circuit = make_circuit(num_qubits=2, num_clbits=2)
    circuit.h(0)
    with pytest.raises(IndexError):
        circuit.cx(0, 7)
    assert list(circuit) == [Gate("H", [0])]


# ----------------------------------------------------------------------
# Dispatch
# ----------------------------------------------------------------------

def test_translate_routes_each_operation_to_its_own_hook(make_circuit):
    """The dispatch table is how an adapter's thirteen hooks get used."""
    circuit = make_circuit(num_qubits=2, num_clbits=2, rank=0, size=2)

    for op in [Gate("H", [0]),
               ControlledGate([0], [Gate("X", [1])]),
               ClassicalControlledGate([0], [Gate("X", [1])]),
               Measure(0, 0), Reset(0), Barrier([0]),
               QSend([0], 1, tag="a"), QRecv([0], 1, tag="b"),
               OperationContainer()]:
        circuit.translate(op)

    assert circuit.kinds == [
        "gate", "controlled_gate", "classical_controlled_gate",
        "measure", "reset", "barrier", "qsend", "qrecv", "container",
    ]


def test_controlled_gates_are_matched_before_plain_ones(make_circuit):
    """
    Order in the table is load-bearing.

    ``ControlledGate`` and ``ClassicalControlledGate`` are siblings of
    ``Gate``, not subclasses, but an adapter that got the table order
    wrong would route them through the single-qubit hook and emit the
    target gate with its control dropped.
    """
    circuit = make_circuit(num_qubits=2, num_clbits=2)
    circuit.translate(ControlledGate([0], [Gate("X", [1])]))
    assert circuit.kinds == ["controlled_gate"]


def test_an_unregistered_subclass_falls_back_to_its_base(make_circuit):
    """A backend may specialise an operation without touching the table."""
    class TaggedGate(Gate):
        pass

    circuit = make_circuit(num_qubits=2, num_clbits=2)
    circuit.translate(TaggedGate("H", [0]))
    assert circuit.kinds == ["gate"]


def test_an_unknown_operation_is_named_rather_than_ignored(make_circuit):
    """Silently skipping it would produce a wrong answer with no error."""
    class Mystery(Operation):
        def __repr__(self):
            return "Mystery()"

    circuit = make_circuit(num_qubits=1, num_clbits=1)
    with pytest.raises(TypeError, match="Unknown operation type: Mystery"):
        circuit.translate(Mystery([0]))


def test_a_container_reaches_the_hooks_of_its_children(make_circuit):
    """Blocks are translated by recursion, which is how collectives work."""
    circuit = make_circuit(num_qubits=2, num_clbits=2)
    block = OperationContainer().add(Gate("H", [0])).add(Measure(0, 0))

    circuit.translate(block)
    assert circuit.kinds == ["container", "gate", "measure"]


def test_every_hook_of_the_contract_is_abstract():
    """
    A partial adapter must fail at import, not at the operation it forgot.

    :class:`~netqmpi.sdk.circuit.Circuit` declares thirteen translation
    hooks; leaving one out has to be a ``TypeError`` when the class is
    instantiated rather than a silent gap in a backend.
    """
    assert len(Circuit.__abstractmethods__) == 13

    class Incomplete(Circuit):
        def _translate_gate(self, op): pass

    with pytest.raises(TypeError, match="abstract"):
        Incomplete(1, 1, None)
