"""
The composite that holds a traced program.

:class:`~netqmpi.sdk.operations.container.OperationContainer` is the shape
of every NetQMPI program: a tree whose leaves are operations and whose
branches are blocks — a ``qscatter`` is itself a container of the
transfers it expands into. Two traversals come out of that tree and they
mean different things, which is the distinction these tests guard:

``children``
    Direct entries, nesting preserved. An adapter dispatching on operation
    type reads this, so it still sees *a scatter* rather than a handful of
    loose sends.

``flatten``
    Every leaf, depth-first, in insertion order. A runtime that has to
    interleave the ranks reads this, because it needs the linear sequence
    of things that actually happen.
"""
from __future__ import annotations

import pytest

from netqmpi.sdk.operations import (
    Gate,
    Measure,
    Operation,
    OperationContainer,
    QSend,
    QScatter,
)


@pytest.fixture
def nested():
    """
    A container holding a leaf, a nested block, and another leaf.

    Returns:
        ``(root, inner)`` so a test can reach either level.
    """
    root = OperationContainer()
    inner = OperationContainer()
    inner.add(Gate("X", [1])).add(Gate("Y", [2]))

    root.add(Gate("H", [0]))
    root.add(inner)
    root.add(Measure(0, 0))
    return root, inner


def test_a_container_is_itself_an_operation():
    """Nesting works because a block is the same kind of thing as a leaf."""
    assert isinstance(OperationContainer(), Operation)
    assert issubclass(QScatter, OperationContainer)


def test_add_returns_self_so_calls_chain():
    """The recording API is fluent all the way down."""
    ops = OperationContainer()
    assert ops.add(Gate("H", [0])).add(Measure(0, 0)) is ops
    assert len(ops) == 2


@pytest.mark.parametrize("value", ["H", 0, None, ["H"]])
def test_add_refuses_anything_but_an_operation(value):
    """Junk in the tree would only surface much later, in an adapter."""
    with pytest.raises(TypeError, match="Expected an Operation"):
        OperationContainer().add(value)


def test_children_keep_the_nesting(nested):
    """A block comes out whole, not dissolved into its contents."""
    root, inner = nested
    assert root.children == [Gate("H", [0]), inner, Measure(0, 0)]
    assert len(root) == 3               # direct entries, not leaves


def test_flatten_yields_every_leaf_depth_first(nested):
    """The linear order is what a runtime replays."""
    root, _ = nested
    assert list(root.flatten()) == [
        Gate("H", [0]), Gate("X", [1]), Gate("Y", [2]), Measure(0, 0),
    ]


def test_iterating_a_container_flattens_it(nested):
    """``for op in container`` is the leaf traversal, not the direct one."""
    root, _ = nested
    assert list(root) == list(root.flatten())


def test_children_accessor_hands_out_a_copy(nested):
    """Appending to what ``children`` returned must not extend the tree."""
    root, _ = nested
    root.children.append(Gate("Z", [0]))
    assert len(root) == 3


def test_qubits_are_the_union_in_first_touch_order(nested):
    """
    The qubit set of a block is derived, never stored.

    Order matters here because it is insertion order, not sorted order: a
    block reports the qubits in the order the program first touched them.
    """
    root, _ = nested
    assert root.qubits == [0, 1, 2]

    repeated = OperationContainer()
    repeated.add(Gate("H", [2])).add(Gate("X", [0])).add(Gate("Y", [2]))
    assert repeated.qubits == [2, 0]        # deduplicated, first touch wins


def test_an_empty_container_is_empty_everywhere():
    """No children, no leaves, no qubits — and it still reprs."""
    ops = OperationContainer()
    assert len(ops) == 0
    assert list(ops.flatten()) == []
    assert ops.qubits == []
    assert repr(ops) == "OperationContainer(children=0)"


def test_deeply_nested_blocks_flatten_in_program_order():
    """Three levels deep, the leaf order is still the order traced."""
    level3 = OperationContainer().add(Gate("T", [3]))
    level2 = OperationContainer().add(Gate("S", [2])).add(level3)
    level1 = OperationContainer().add(Gate("H", [1])).add(level2)

    assert [op.name for op in level1.flatten()] == ["H", "S", "T"]
    assert level1.qubits == [1, 2, 3]


def test_a_rooted_collective_flattens_into_its_transfers():
    """This is how a backend gets scatter for free from qsend."""
    scatter = QScatter(rank=0, root=0, ranks=[0, 1, 2], qubits=[0, 1])
    scatter.add(QSend([0], 1, tag="a")).add(QSend([1], 2, tag="b"))

    outer = OperationContainer().add(Gate("X", [0])).add(scatter)
    assert [type(op).__name__ for op in outer.flatten()] == ["Gate", "QSend", "QSend"]
    assert [type(op).__name__ for op in outer.children] == ["Gate", "QScatter"]
