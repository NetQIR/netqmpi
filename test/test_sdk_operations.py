"""
The operation objects a traced program is made of.

Every NetQMPI program is recorded as a tree of
:class:`~netqmpi.sdk.operations.Operation` instances before any backend
sees it, so these classes are the vocabulary the whole library shares.
What matters about them is exactly three things, and this file pins all
three:

- **Validation.** A malformed operation must be refused where it is built,
  not carried into an adapter that will fail in the backend's own words.
- **Accessors.** ``qubits``, ``params``, ``controls``, ``cbits`` and the
  rest hand out *copies*: an adapter that sorts a gate's qubit list must
  not be able to rewrite the program it is translating.
- **Value semantics.** ``__eq__`` / ``__hash__`` make operations
  comparable, which is what lets a test say "this is the program I traced"
  instead of picking the record apart field by field.
"""
from __future__ import annotations

import pytest

from netqmpi.sdk.operations import (
    Barrier,
    ClassicalControlledGate,
    ControlledGate,
    Gate,
    Measure,
    Operation,
    Reset,
)


# ----------------------------------------------------------------------
# Operation — the shared base
# ----------------------------------------------------------------------

def test_operation_is_abstract():
    """The base class describes an intent; it is never instantiated."""
    with pytest.raises(TypeError):
        Operation([0])


@pytest.mark.parametrize("qubits", ["0", 0, (0, 1), [0, "1"], [0, 1.0], None])
def test_operation_refuses_anything_but_a_list_of_ints(qubits):
    """A qubit index is an int, and a qubit list is a list of them."""
    with pytest.raises(TypeError, match="list of integers"):
        Gate("H", qubits)


def test_qubits_accessor_hands_out_a_copy():
    """Mutating what an accessor returned must not touch the operation."""
    gate = Gate("H", [0, 1])
    gate.qubits.append(99)
    assert gate.qubits == [0, 1]


# ----------------------------------------------------------------------
# Gate
# ----------------------------------------------------------------------

def test_gate_normalises_its_name():
    """Names are compared upper-case, so that is how they are stored."""
    assert Gate("h", [0]).name == "H"
    assert Gate("rz", [0], [0.5]).name == "RZ"


@pytest.mark.parametrize("name", ["", "   ", None, 7])
def test_gate_refuses_an_empty_name(name):
    """An unnamed gate cannot be dispatched by any adapter."""
    with pytest.raises(ValueError, match="non-empty string"):
        Gate(name, [0])


def test_gate_params_default_to_empty_and_are_copied():
    """A gate with no angle carries no parameters, and they stay its own."""
    plain = Gate("X", [0])
    assert plain.params == []

    rotation = Gate("RX", [0], [0.25])
    rotation.params.append(9.0)
    assert rotation.params == [0.25]


def test_gate_equality_covers_name_qubits_and_params():
    """Two gates are the same gate only if all three agree."""
    assert Gate("RX", [0], [0.5]) == Gate("rx", [0], [0.5])
    assert Gate("RX", [0], [0.5]) != Gate("RX", [1], [0.5])
    assert Gate("RX", [0], [0.5]) != Gate("RX", [0], [0.6])
    assert Gate("RX", [0], [0.5]) != Gate("RY", [0], [0.5])
    assert Gate("H", [0]) != "H"


def test_gate_is_hashable_by_value():
    """Equal gates collapse in a set; different ones do not."""
    assert len({Gate("H", [0]), Gate("h", [0]), Gate("H", [1])}) == 2


def test_gate_repr_shows_params_only_when_there_are_any():
    """The repr is what a failing assertion prints, so it stays readable."""
    assert repr(Gate("H", [0])) == "Gate(name='H', qubits=[0])"
    assert "params=[0.5]" in repr(Gate("RZ", [0], [0.5]))


# ----------------------------------------------------------------------
# ControlledGate
# ----------------------------------------------------------------------

def test_controlled_gate_spans_controls_and_targets():
    """The qubit list is what the operation touches: controls first."""
    op = ControlledGate([0, 1], [Gate("X", [2])])
    assert op.controls == [0, 1]
    assert op.qubits == [0, 1, 2]
    assert op.targets == [Gate("X", [2])]


def test_controlled_gate_accessors_hand_out_copies():
    """Neither the controls nor the targets can be edited from outside."""
    op = ControlledGate([0], [Gate("X", [1])])
    op.controls.append(9)
    op.targets.append(Gate("Y", [3]))
    assert op.controls == [0]
    assert len(op.targets) == 1


@pytest.mark.parametrize("controls, targets, error, match", [
    ([], [Gate("X", [1])], ValueError, "controls must be"),
    (0, [Gate("X", [1])], ValueError, "controls must be"),
    ([0], [], ValueError, "targets must be"),
    ([0], [Measure(1, 0)], TypeError, "must be a Gate"),
    ([0], ["X"], TypeError, "must be a Gate"),
])
def test_controlled_gate_validation(controls, targets, error, match):
    """A control with nothing to control, or a target that is not a gate."""
    with pytest.raises(error, match=match):
        ControlledGate(controls, targets)


def test_controlled_gate_equality():
    """Same controls and same targets, or not the same operation."""
    assert ControlledGate([0], [Gate("X", [1])]) == ControlledGate([0], [Gate("X", [1])])
    assert ControlledGate([0], [Gate("X", [1])]) != ControlledGate([1], [Gate("X", [0])])
    assert ControlledGate([0], [Gate("X", [1])]) != ControlledGate([0], [Gate("Z", [1])])


# ----------------------------------------------------------------------
# ClassicalControlledGate
# ----------------------------------------------------------------------

def test_classical_controlled_gate_keeps_cbits_out_of_the_qubit_list():
    """Its conditions are classical bits, so they are not qubits."""
    op = ClassicalControlledGate([0, 1], [Gate("X", [2])])
    assert op.cbits == [0, 1]
    assert op.qubits == [2]


@pytest.mark.parametrize("cbits, targets, error, match", [
    ([], [Gate("X", [0])], ValueError, "cbits must be"),
    (["a"], [Gate("X", [0])], TypeError, "must be an integer"),
    ([0], [], ValueError, "targets must be"),
    ([0], [Reset(1)], TypeError, "must be a Gate"),
])
def test_classical_controlled_gate_validation(cbits, targets, error, match):
    """The same contract as its quantum cousin, on classical conditions."""
    with pytest.raises(error, match=match):
        ClassicalControlledGate(cbits, targets)


def test_classical_controlled_gate_equality_and_repr():
    """Value semantics, and a repr naming both halves."""
    op = ClassicalControlledGate([0], [Gate("X", [1])])
    assert op == ClassicalControlledGate([0], [Gate("X", [1])])
    assert op != ClassicalControlledGate([1], [Gate("X", [1])])
    assert "cbits=[0]" in repr(op)


# ----------------------------------------------------------------------
# Measure / Reset / Barrier
# ----------------------------------------------------------------------

def test_measure_pairs_a_qubit_with_a_classical_bit():
    """The two indices are independent: qubit 2 may land on cbit 0."""
    op = Measure(2, 0)
    assert (op.qubit, op.cbit, op.qubits) == (2, 0, [2])


def test_measure_equality_covers_the_classical_bit():
    """Same qubit into a different bit is a different measurement."""
    assert Measure(0, 0) == Measure(0, 0)
    assert Measure(0, 0) != Measure(0, 1)
    assert Measure(0, 0) != Measure(1, 0)
    assert len({Measure(0, 0), Measure(0, 0), Measure(0, 1)}) == 2


def test_reset_targets_one_qubit():
    """Reset names a single qubit and compares by it."""
    assert Reset(1).qubit == 1
    assert Reset(1) == Reset(1)
    assert Reset(1) != Reset(0)
    assert repr(Reset(1)) == "Reset(qubit=1)"


def test_barrier_defaults_to_the_whole_circuit():
    """An empty qubit list is the documented 'everything' barrier."""
    assert Barrier().qubits == []
    assert Barrier([0, 1]).qubits == [0, 1]
    assert Barrier([0]) != Barrier([1])
    assert len({Barrier(), Barrier([])}) == 1
