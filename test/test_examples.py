"""
The shipped examples, run as tests against their expected output.

``examples/1`` … ``examples/5`` are the documented introduction to NetQMPI,
so what they print is a specification: each one's docstring states what the
run should show. These tests hold the backends to it.

Where the reference comes from
------------------------------
Examples 1–3 were run on **CUNQA** (2048 shots, vQPUs raised by the run) and
the expectations below are what it produced:

======================  ======================================================
``1_send_recv`` -n 2    ``{0: {}, 1: {'0': 1000, '1': 1048}}``
``2_round_robin`` -n 3  ranks 0 and 1 empty, ``rank 2: {'0': 1002, '1': 1046}``
``3_scatter`` -n 3      ``rank 0: {'00': 2048}``, ranks 1 and 2 ``{'1': 2048}``
======================  ======================================================

Examples 4 and 5 **cannot be run on CUNQA as deployed in the local
container**: the default vQPU definition is too narrow for them, and they
fail in ``cunqa.qpu.run`` with "Not enough data qubits in the QPU" and "Not
enough comm qubits in the QPU" respectively. Widening it needs a vQPU
definition file passed to ``qraise -b``, which is a property of the
deployment rather than of NetQMPI. Their expectations here are therefore
taken from their own docstrings, which state the intended result exactly.

A rank that never measures
--------------------------
The two backends say so differently: CUNQA reports no counts at all for
that rank, while Aer keeps every rank's classical bits in one register and
so reports them as zeros. :func:`check_silent` accepts either, since they
mean the same thing.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

pytest.importorskip("qiskit", reason="the Aer backend needs Qiskit")
pytest.importorskip("qiskit_aer", reason="the Aer backend needs qiskit-aer")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts" / "benchmark"))

import metrics  # noqa: E402

from netqmpi.runtime.adapters.aer import (  # noqa: E402
    AerExecutorAdapter, AerSimulatorConfig,
)
from netqmpi.sdk.environment import Environment  # noqa: E402

EXAMPLES = REPO_ROOT / "examples"
SHOTS = 2048
SEED = 20260921

#: Half-width allowed around an even split. With 2048 shots the standard
#: error of a proportion is ~0.011, so this is roughly 4 sigma: wide enough
#: never to flake, narrow enough that a broken transfer (which collapses the
#: split to 0 or 1) cannot slip through.
BALANCE_TOLERANCE = 0.05


def run_example(name: str, ranks: int):
    """
    Run one shipped example on the Aer backend.

    Args:
        name: File name under ``examples/``, without the extension.
        ranks: Number of ranks, as the example's docstring prescribes.

    Returns:
        One histogram per rank, keyed by rank.
    """
    captured = []
    original = Environment.__init__

    def capture(self, comm, executor):
        original(self, comm, executor)
        captured.append(self)

    Environment.__init__ = capture
    try:
        config = AerSimulatorConfig()
        config.shots = SHOTS
        config.seed_simulator = SEED
        executor = AerExecutorAdapter(ranks, config)
        executor.run(executor.build_apps(str(EXAMPLES / f"{name}.py"), ranks))
    finally:
        Environment.__init__ = original

    captured.sort(key=lambda env: env.comm.rank)
    return metrics.per_rank_histograms(captured, "aer")


def check_silent(histogram) -> None:
    """
    Assert a rank contributed nothing: no counts, or only zeros.

    Args:
        histogram: That rank's histogram.
    """
    assert all(set(key.replace(" ", "")) == {"0"} for key in histogram), (
        f"expected an unmeasured rank, got {histogram}")


def check_exact(histogram, expected: str) -> None:
    """
    Assert every shot returned the same, given outcome.

    Args:
        histogram: That rank's histogram.
        expected: The bit string every shot must produce.
    """
    observed = {key.replace(" ", ""): value for key, value in histogram.items()}
    assert observed == {expected: SHOTS}, (
        f"expected {{{expected!r}: {SHOTS}}}, got {observed}")


def check_balanced(histogram) -> None:
    """
    Assert a single bit came back about evenly split between 0 and 1.

    Args:
        histogram: That rank's histogram.
    """
    counts = {key.replace(" ", ""): value for key, value in histogram.items()}
    total = sum(counts.values())
    assert total == SHOTS, f"expected {SHOTS} shots, got {counts}"
    assert set(counts) == {"0", "1"}, f"expected a single bit, got {counts}"
    share = counts["1"] / total
    assert abs(share - 0.5) < BALANCE_TOLERANCE, (
        f"expected an even split, got P(1) = {share:.3f} from {counts}")


# ----------------------------------------------------------------------
# 1 — teleport one qubit from rank 0 to rank 1
# ----------------------------------------------------------------------

def test_send_recv():
    """Rank 1 reads the |+> it was given; rank 0 kept nothing to measure."""
    results = run_example("1_send_recv", 2)
    check_silent(results[0])
    check_balanced(results[1])


# ----------------------------------------------------------------------
# 2 — relay one qubit down a chain
# ----------------------------------------------------------------------

@pytest.mark.parametrize("ranks", [2, 3, 4])
def test_round_robin(ranks):
    """Only the end of the chain measures, and it reads an even split."""
    results = run_example("2_round_robin", ranks)
    for rank in range(ranks - 1):
        check_silent(results[rank])
    check_balanced(results[ranks - 1])


# ----------------------------------------------------------------------
# 3 — the root scatters its whole buffer away
# ----------------------------------------------------------------------

def test_scatter():
    """Every receiver reads the |1> the root prepared; the root keeps none."""
    results = run_example("3_scatter", 3)
    check_exact(results[0], "00")       # gave the whole buffer away
    check_exact(results[1], "1")
    check_exact(results[2], "1")


# ----------------------------------------------------------------------
# 4 — every rank hands its qubit to the root
# ----------------------------------------------------------------------

def test_gather():
    """The root reads a 1 on every slot; the contributors are left at 0."""
    results = run_example("4_gather", 3)
    check_exact(results[0], "111")      # its own plus the two it gathered
    check_exact(results[1], "0")
    check_exact(results[2], "0")


# ----------------------------------------------------------------------
# 5 — a 3-qubit QFT built out of telegates
# ----------------------------------------------------------------------

@pytest.mark.xfail(raises=NotImplementedError, strict=True,
                   reason="the Aer adapter does not implement expose/unexpose")
def test_qft_expose():
    """
    The QFT of ``|000>`` is ``|+++>``, so every rank reads an even split.

    Marked ``xfail(strict=True)``: it is expected to fail today because the
    adapter refuses ``expose``, and it will start failing *loudly* the day
    that is implemented — which is exactly when this expectation wants
    checking.
    """
    results = run_example("5_qft_expose", 3)
    for rank in range(3):
        check_balanced(results[rank])
