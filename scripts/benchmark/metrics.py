"""
Workload counters and the cross-backend fidelity metric.

Two jobs. First, describing the *size* of what a run asked NetQMPI to do,
which is what the overhead model is regressed against: how many operations
the trace recorded, split into local gates and communication primitives.
Second, turning each backend's very differently shaped result object into
one comparable number.

The fidelity metric
-------------------
Every probe in :mod:`apps` is an *echo*: it performs a transform and then
undoes it, so the noise-free outcome is all-zeros on every rank. Fidelity
is the probability of observing exactly that.

That choice is deliberate. An all-zero expectation is independent of how a
backend orders the bits inside a histogram key, which removes an entire
class of silent comparison bugs; and it is a criterion every rank can be
scored on separately, which matters because **CUNQA reports one histogram
per rank — marginals, not the joint distribution**. From marginals the true
joint success probability cannot be reconstructed, so the portable metric
is the product of the per-rank success probabilities. Where a backend does
expose the joint distribution (Aer keeps every rank's classical bits in one
register) the joint figure is reported alongside it, and the two agree when
rank errors are independent.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from netqmpi.sdk.operations import (
    Barrier, ClassicalControlledGate, ControlledGate, Expose, Gate, Measure,
    QGather, QRecv, QScatter, QSend, Reset, Unexpose,
)

#: Operation classes counted separately, most specific first: ``Gate`` is a
#: base of the controlled variants, so a plain ``isinstance`` ladder has to
#: test the subclasses before it.
_OP_KINDS: List[Tuple[str, type]] = [
    ("classical_controlled", ClassicalControlledGate),
    ("controlled_gate", ControlledGate),
    ("measure", Measure),
    ("reset", Reset),
    ("barrier", Barrier),
    ("qsend", QSend),
    ("qrecv", QRecv),
    ("qscatter", QScatter),
    ("qgather", QGather),
    ("expose", Expose),
    ("unexpose", Unexpose),
    ("gate", Gate),
]

#: Counter keys that are communication rather than local computation.
COMM_KINDS = ("qsend", "qrecv", "qscatter", "qgather", "expose", "unexpose")


def count_operations(envs: List[Any]) -> Dict[str, int]:
    """
    Count the operations every rank recorded during the trace.

    Args:
        envs: The per-rank :class:`~netqmpi.sdk.environment.Environment`
            objects of the run, in rank order.

    Returns:
        A mapping with one key per operation kind, plus ``ops_total``
        (every recorded operation), ``ops_comm`` (the communication
        primitives) and ``ops_local`` (the rest).
    """
    counts: Dict[str, int] = {name: 0 for name, _ in _OP_KINDS}

    for env in envs:
        for circuit in env.comm.circuits:
            for op in circuit.ops.flatten():
                for name, kind in _OP_KINDS:
                    if isinstance(op, kind):
                        counts[name] += 1
                        break

    counts["ops_total"] = sum(counts[name] for name, _ in _OP_KINDS)
    counts["ops_comm"] = sum(counts[name] for name in COMM_KINDS)
    counts["ops_local"] = counts["ops_total"] - counts["ops_comm"]
    return counts


def _clean(key: Any) -> str:
    """
    Return a histogram key as a bare bit string.

    Backends decorate their keys differently — CUNQA separates registers
    with spaces, Qiskit does the same when a circuit has several — and none
    of that carries meaning for an all-zero criterion.

    Args:
        key: Raw histogram key.

    Returns:
        The key with whitespace removed.
    """
    return str(key).replace(" ", "")


def _all_zero(key: Any) -> bool:
    """
    Return whether a histogram key is the all-zeros outcome.

    Args:
        key: Raw histogram key.

    Returns:
        ``True`` if every bit of the key is ``0``.
    """
    bits = _clean(key)
    return bool(bits) and set(bits) == {"0"}


def per_rank_histograms(envs: List[Any], backend: str) -> Dict[int, Dict[str, int]]:
    """
    Normalise a run's results into one histogram per rank.

    The four backends hand back three different shapes: CUNQA gives every
    communicator the whole ``{rank: counts}`` picture, Qoala and NetQASM
    give each communicator its own histogram, and Aer gives every
    communicator the *same* histogram over the classical bits of all ranks
    at once.

    Args:
        envs: The per-rank environments of the run, in rank order.
        backend: Backend name.

    Returns:
        A mapping from rank to that rank's histogram. Empty when the run
        produced no results at all.
    """
    if not envs:
        return {}

    if backend == "aer":
        return _split_joint(envs)

    # CUNQA publishes the complete picture, keyed by rank, on every
    # communicator; the last rank to leave its block is the one that has it.
    for env in envs:
        results = env.comm.results
        if results and all(isinstance(key, int) for key in results):
            return {rank: dict(counts) for rank, counts in results.items()}

    return {env.comm.rank: dict(env.comm.results or {}) for env in envs}


def _split_joint(envs: List[Any]) -> Dict[int, Dict[str, int]]:
    """
    Split Aer's single global histogram into one histogram per rank.

    Aer runs every rank inside one ``QuantumCircuit``, so a histogram key
    covers all the ranks' classical bits at once. Qiskit prints them
    most-significant first, so the key is reversed before slicing to index
    it by classical bit number.

    Args:
        envs: The per-rank environments of the run, in rank order.

    Returns:
        A mapping from rank to that rank's marginal histogram.
    """
    joint = envs[0].comm.results or {}
    per_rank: Dict[int, Dict[str, int]] = {env.comm.rank: {} for env in envs}

    widths = {env.comm.rank: sum(c.num_clbits for c in env.comm.circuits)
              for env in envs}

    for key, count in joint.items():
        bits = _clean(key)[::-1]          # bits[i] is classical bit i
        offset = 0
        for env in envs:
            rank = env.comm.rank
            slice_ = bits[offset:offset + widths[rank]]
            per_rank[rank][slice_] = per_rank[rank].get(slice_, 0) + count
            offset += widths[rank]

    return per_rank


def joint_fidelity(envs: List[Any], backend: str) -> Optional[float]:
    """
    Return the joint probability of the all-zeros outcome, if observable.

    Only meaningful for a backend that keeps every rank's classical bits in
    one histogram; from per-rank marginals the joint cannot be recovered.

    Args:
        envs: The per-rank environments of the run, in rank order.
        backend: Backend name.

    Returns:
        The joint success probability, or ``None`` when the backend only
        exposes marginals.
    """
    if backend != "aer" or not envs:
        return None

    joint = envs[0].comm.results or {}
    total = sum(joint.values())
    if not total:
        return None
    hits = sum(count for key, count in joint.items() if _all_zero(key))
    return hits / total


def fidelity(histograms: Dict[int, Dict[str, int]]) -> Tuple[float, Dict[int, float]]:
    """
    Score a run against its noise-free all-zeros expectation.

    Args:
        histograms: One histogram per rank, as returned by
            :func:`per_rank_histograms`.

    Returns:
        A pair ``(overall, per_rank)``. ``overall`` is the product of the
        per-rank success probabilities; ranks that measured nothing are
        skipped rather than scored zero, since a rank with no classical
        output cannot fail. ``nan`` when no rank measured anything.
    """
    per_rank: Dict[int, float] = {}

    for rank, histogram in sorted(histograms.items()):
        total = sum(histogram.values())
        if not total:
            continue
        hits = sum(count for key, count in histogram.items() if _all_zero(key))
        per_rank[rank] = hits / total

    if not per_rank:
        return float("nan"), {}

    overall = 1.0
    for value in per_rank.values():
        overall *= value
    return overall, per_rank
