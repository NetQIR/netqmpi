"""
Composite container for quantum operations.
"""
from __future__ import annotations
from typing import Iterator, List, Union

from netqmpi.sdk.operations.operation import Operation


class OperationContainer(Operation):
    """
    Composite container for quantum operations.

    Implements the Composite pattern: it can hold both leaf
    :class:`~netqmpi.sdk.operations.Operation` instances and nested
    :class:`OperationContainer` objects, allowing circuits to be built
    hierarchically.

    :meth:`flatten` produces a depth-first iterator over every leaf
    :class:`~netqmpi.sdk.operations.Operation` in insertion order.

    Example::

        ops = OperationContainer()
        ops.add(Gate('H', [0])).add(Measure(0, 0))

        sub = OperationContainer()
        sub.add(Gate('X', [1]))
        ops.add_circuit(sub)

        for op in ops.flatten():
            print(op)
    """

    def __init__(self) -> None:
        # Pass an empty qubit list; the real qubit set is derived from children.
        super().__init__([])
        self._children: List[Union[Operation, OperationContainer]] = []

    # ------------------------------------------------------------------
    # Properties (override)
    # ------------------------------------------------------------------

    @property
    def qubits(self) -> List[int]:
        """Union of all qubit indices across children, in insertion order."""
        seen: list[int] = []
        for q in (q for child in self._children for q in child.qubits):
            if q not in seen:
                seen.append(q)
        return seen

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add(self, operation: Operation) -> OperationContainer:
        """
        Append a single leaf :class:`~netqmpi.sdk.operations.Operation`.

        Args:
            operation: The operation to add.

        Returns:
            *self*, enabling method chaining.

        Raises:
            TypeError: If *operation* is not an
                :class:`~netqmpi.sdk.operations.Operation` instance.
        """
        if not isinstance(operation, Operation):
            raise TypeError(
                f"Expected an Operation instance, got {type(operation).__name__}."
            )
        self._children.append(operation)
        return self

    # ------------------------------------------------------------------
    # Traversal
    # ------------------------------------------------------------------

    def flatten(self) -> Iterator[Operation]:
        """
        Depth-first iterator over all leaf operations.

        Yields:
            Each :class:`~netqmpi.sdk.operations.Operation` in the
            order they were added, recursing into nested containers.
        """
        for child in self._children:
            if isinstance(child, OperationContainer):
                yield from child.flatten()
            else:
                yield child

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        """Number of direct children (leaves + sub-containers, not flattened)."""
        return len(self._children)

    def __iter__(self) -> Iterator[Operation]:
        """Iterating over the container is equivalent to calling :meth:`flatten`."""
        return self.flatten()

    def __repr__(self) -> str:
        return f"OperationContainer(children={len(self._children)})"
