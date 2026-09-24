"""
Count the communication instructions the CUNQA adapter actually emits.

The profiler counts NetQMPI primitives as they are traced; this script counts
what reaches CUNQA after translation. Each probe is built and translated
exactly as a benchmark run would do it, but the submission is intercepted, so
nothing is simulated and the counts are exact.

Three quantities are recorded per probe and rank count:

- ``send``: classical messages, one per ``send`` instruction.
- ``gen_ent``: entanglement-generation calls. Every participant of an
  entangled resource issues one, so a Bell pair is two calls and a GHZ state
  over ``m`` circuits is ``m`` calls.
- ``total``: every instruction of every rank's circuit.

Run it inside the CUNQA container, from the repository root::

    docker run --rm -v "$PWD":/work -w /work netqmpi:cunqa bash -lc \\
        'export PYTHONPATH=/work:$PYTHONPATH; \\
         python3 scripts/benchmark/count_emitted.py'

The vQPUs are raised exactly as the benchmark raises them (quantum
communication, Munich simulator, co-located mode, CUNQA's default vQPU
definition).
"""
import collections
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from cunqa.qpu import qdrop, qraise  # noqa: E402

import netqmpi.runtime.adapters.cunqa.cunqa_communicator as cunqa_comm  # noqa: E402
from netqmpi.runtime.adapters.cunqa import (  # noqa: E402
    CunqaExecutorAdapter,
    CunqaRunConfig,
)

APPS = ("cascade", "ghz", "qft", "qft_telegate")
RANKS = (2, 3, 4, 5)
OUT = REPO / "scripts" / "benchmark" / "results" / "cunqa_emitted.json"

_captured = []


class _Result:
    """Stand-in for a CUNQA result; the probes never read its counts."""

    counts = {"0": 1}


def _intercept_run(circuits, qpus, shots=None):
    """Record the translated circuits instead of submitting them."""
    _captured.append(list(circuits))
    return [None] * len(circuits)


def main() -> None:
    """Translate every probe and write the instruction counts."""
    cunqa_comm.run = _intercept_run
    cunqa_comm.gather = lambda jobs: [_Result() for _ in jobs]
    os.environ["NQB_QUBITS_PER_RANK"] = "1"

    family = qraise(max(RANKS), "00:20:00", quantum_comm=True,
                    simulator="Munich", co_located=True)
    counts = {}
    try:
        for app in APPS:
            for n in RANKS:
                _captured.clear()
                config = CunqaRunConfig()
                config.family = family
                config.shots = 1
                executor = CunqaExecutorAdapter(n, config)
                executor.run(executor.build_apps(
                    str(REPO / "scripts" / "benchmark" / "apps" / f"{app}.py"), n))

                names = collections.Counter()
                for circuit in _captured[0][:n]:      # the fillers are not ranks
                    for instruction in circuit.instructions:
                        names[instruction["name"]] += 1
                counts[f"{app}/{n}"] = {"send": names["send"],
                                        "gen_ent": names["gen_ent"],
                                        "total": sum(names.values())}
    finally:
        qdrop(family)

    OUT.write_text(json.dumps(counts, indent=2) + "\n")
    print(f"wrote {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()
