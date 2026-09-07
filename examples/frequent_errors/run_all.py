"""Run every failing example and show what NetQMPI reports.

Each example is traced and translated exactly as ``netqmpi --cunqa`` would,
with only the submission to the vQPUs stubbed out, so the harness needs
CUNQA importable but no qraise and no SLURM allocation. Examples whose
failure only happens once the circuits actually run therefore come out
clean here -- which is itself part of the picture.

    python run_all.py            # every example
    python run_all.py scatter    # only those whose name contains "scatter"
"""
import os
import runpy
import sys
import traceback
from pathlib import Path

sys.path.append(os.getenv("HOME", ""))      # CUNQA lives there, as in the adapter

from netqmpi.helpers import load_main
from netqmpi.runtime.run_config import RunConfig
from netqmpi.runtime.adapters.cunqa import cunqa_communicator as cunqa
from netqmpi.runtime.adapters.cunqa import CunqaCircuitAdapter
from netqmpi.sdk.environment import Environment

HERE = Path(__file__).resolve().parent
RULE = "=" * 78


class StubResult:
    """Stands in for a CUNQA job result: the run itself is not simulated."""

    def __init__(self, circuit):
        self.counts = {"<not executed>": 0}


def stub_run(circuits, qpus, shots=None):
    return [StubResult(c) for c in circuits]


class StubExecutor:
    """Just the circuit factory: no qraise, no vQPUs."""

    def create_circuit(self, num_qubits, num_clbits, comm):
        return CunqaCircuitAdapter(num_qubits, num_clbits, comm)


def run_example(path: Path) -> str:
    """Run one example. Returns 'raised' or 'clean'."""
    namespace = runpy.run_path(str(path))
    ranks = namespace.get("RANKS", 2)
    summary = (namespace.get("__doc__") or "").strip().splitlines()[0]

    print(f"\n{RULE}\n{path.name} -- {summary}")
    print(f"$ netqmpi -n {ranks} --cunqa {path.name}\n{'-' * 78}")

    try:
        main = load_main(str(path))
        config = RunConfig(shots=100)
        session = cunqa.CunqaSession(ranks, [None] * ranks, config)
        for rank in range(ranks):
            comm = cunqa.CunqaCommunicator(rank, ranks, None, config, session)
            main(env=Environment(comm, StubExecutor()))
    except Exception:
        traceback.print_exc(file=sys.stdout)
        return "raised"

    print("no error reported: this one only fails once the circuits run on vQPUs.")
    return "clean"


def main() -> None:
    cunqa.run = stub_run
    cunqa.gather = lambda qjobs: qjobs

    pattern = sys.argv[1] if len(sys.argv) > 1 else ""
    examples = sorted(p for p in HERE.glob("*.py")
                      if p.name != Path(__file__).name and pattern in p.name)

    outcomes = {}
    for path in examples:
        outcomes[path.name] = run_example(path)

    print(f"\n{RULE}\n{len(examples)} examples: "
          f"{sum(o == 'raised' for o in outcomes.values())} reported an error, "
          f"{sum(o == 'clean' for o in outcomes.values())} passed translation silently.")
    for name, outcome in outcomes.items():
        print(f"  {'raised ' if outcome == 'raised' else 'SILENT '} {name}")


if __name__ == "__main__":
    main()
