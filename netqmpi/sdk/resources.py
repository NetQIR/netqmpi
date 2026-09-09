"""
Reusable index pools for the resources a distributed protocol borrows.

Communication qubits and the classical bits used by the correction rounds
of teledata/telegate are scarce, backend-managed resources: they are
taken when a protocol block opens and given back when it closes, so two
blocks that never overlap in time can share the same physical resource.

This module provides the tiny allocator both the circuit layer and the
backend adapters rely on to agree, without any inter-rank communication,
on which slot each protocol block uses.
"""
from __future__ import annotations

from typing import List


class IndexPool:
    """
    Allocator of small non-negative indices with reuse.

    Indices are handed out from a free list first and only then from a
    fresh counter, which keeps the total footprint at the maximum number
    of *simultaneously* held indices rather than the total number of
    acquisitions.

    Example::

        pool = IndexPool()
        a = pool.acquire(2)   # [0, 1]
        pool.release(a)
        b = pool.acquire(1)   # [0]  -- reused
        pool.size             # 2
    """

    def __init__(self) -> None:
        """Initialize an empty pool."""
        self._free: List[int] = []
        self._next: int = 0

    @property
    def size(self) -> int:
        """
        Return how many distinct indices the pool ever handed out.

        Returns:
            The high-water mark of the allocator.
        """
        return self._next

    def acquire(self, count: int = 1) -> List[int]:
        """
        Reserve ``count`` indices.

        Args:
            count: Number of indices to reserve.

        Returns:
            The reserved indices, in ascending order.

        Raises:
            ValueError: If *count* is not strictly positive.
        """
        if count < 1:
            raise ValueError("count must be strictly positive.")
        taken: List[int] = []
        for _ in range(count):
            if self._free:
                taken.append(self._free.pop(0))
            else:
                taken.append(self._next)
                self._next += 1
        return sorted(taken)

    def release(self, indices: List[int]) -> None:
        """
        Give indices back to the pool so later blocks can reuse them.

        Args:
            indices: Indices previously returned by :meth:`acquire`.
        """
        for index in indices:
            if index not in self._free:
                self._free.append(index)
        self._free.sort()
