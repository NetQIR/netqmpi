"""
The allocator behind communication qubits and protocol classical bits.

A teledata transfer borrows one communication qubit and two classical bits
for as long as it lasts, and a telegate window holds its communication
qubit until ``unexpose``. Those are scarce, backend-provided resources, so
:class:`~netqmpi.sdk.resources.IndexPool` hands out the *smallest* indices
that are free rather than fresh ones: a program with a hundred sequential
transfers must ask the backend for one communication qubit, not a hundred.

The reuse also has to be deterministic. Every rank runs the same allocator
over its own trace with no communication, and both sides of a transfer
must agree on which slot it uses, so "which index comes back" is part of
the contract and not an implementation detail.
"""
from __future__ import annotations

import pytest

from netqmpi.sdk.resources import IndexPool


def test_a_fresh_pool_has_handed_out_nothing():
    """``size`` is what the backend must reserve, so it starts at zero."""
    assert IndexPool().size == 0


def test_indices_are_handed_out_from_zero_upwards():
    """Fresh acquisitions count up, and a single one needs no argument."""
    pool = IndexPool()
    assert pool.acquire() == [0]
    assert pool.acquire(2) == [1, 2]
    assert pool.size == 3


def test_released_indices_are_reused_before_fresh_ones():
    """This is the whole point: sequential blocks share one slot."""
    pool = IndexPool()
    first = pool.acquire()
    pool.release(first)
    assert pool.acquire() == first
    assert pool.size == 1


def test_size_is_the_high_water_mark_not_the_total():
    """A thousand transfers in a row still cost one communication qubit."""
    pool = IndexPool()
    for _ in range(1000):
        pool.release(pool.acquire())
    assert pool.size == 1


def test_overlapping_holders_each_get_their_own():
    """Two windows open at once cannot share a slot."""
    pool = IndexPool()
    outer = pool.acquire()
    inner = pool.acquire()
    assert outer != inner
    assert pool.size == 2

    pool.release(inner)
    pool.release(outer)
    assert pool.acquire(2) == [0, 1]        # both back, none added
    assert pool.size == 2


def test_acquisitions_come_back_sorted():
    """
    Reuse must not scramble the order.

    After a release the free list is consulted first, so a mixed
    acquisition draws partly from it and partly from the counter; the
    result is still ascending, which is what the callers assume when they
    map ``clbits[0]``/``clbits[1]`` onto the two correction bits.
    """
    pool = IndexPool()
    pool.acquire(3)                  # 0, 1, 2
    pool.release([1])
    assert pool.acquire(2) == [1, 3]


@pytest.mark.parametrize("count", [0, -1, -5])
def test_acquiring_nothing_is_refused(count):
    """A block that reserves no resource is a bug in the caller."""
    with pytest.raises(ValueError, match="strictly positive"):
        IndexPool().acquire(count)


def test_releasing_twice_does_not_duplicate_a_slot():
    """A double release must not hand the same index to two holders."""
    pool = IndexPool()
    index = pool.acquire()
    pool.release(index)
    pool.release(index)
    assert pool.acquire(2) == [0, 1]        # 0 came back once, 1 is fresh


def test_releasing_a_set_keeps_the_pool_ordered():
    """Freed indices come back smallest first, whatever order they arrive."""
    pool = IndexPool()
    pool.acquire(4)
    pool.release([3, 0, 2])
    assert pool.acquire(3) == [0, 2, 3]
