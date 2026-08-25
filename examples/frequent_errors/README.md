# Failing examples

A corpus of programs that are *meant* to break, one failure mode each, to see
what NetQMPI tells the user today and to have something to test an error
interface against. Every file runs with the CUNQA backend:

    netqmpi -n <RANKS> --cunqa <example>.py

The rank count each one needs is in its docstring and in its `RANKS` constant.

## Running them without vQPUs

    python run_all.py            # every example
    python run_all.py scatter    # only those whose name contains "scatter"

`run_all.py` traces and translates each example exactly as the CUNQA backend
would, stubbing only the submission to the vQPUs, so it needs CUNQA importable
but no `qraise` and no SLURM allocation. Examples that only fail once the
circuits actually run come out clean there, and the harness says so.

## What fails, and where it is caught

| Example | What is wrong | Caught | What the user gets today |
|---|---|---|---|
| `qubit_out_of_range.py` | Gate on a qubit the circuit does not have | trace | `IndexError: Qubit index 5 out of range [0, 2) (2 data + 0 comm qubits).` |
| `not_enough_clbits.py` | `measure_all` with too few clbits | trace | `ValueError: Not enough classical bits to measure all qubits (1 clbits < 3 qubits).` |
| `closed_expose_window.py` | Exposed control used after `unexpose` | trace | `IndexError: Qubit index 1 is a communication qubit whose expose window is already closed.` |
| `unmatched_unexpose.py` | `unexpose` with no matching `expose` | trace | `RuntimeError: rank 0 called unexpose(ranks=[0], root=1) without a matching open expose window.` |
| `scatter_uneven_buffer.py` | Root buffer does not split evenly | trace | `ValueError: the root of a qscatter must hold one chunk per rank: 4 qubits do not split evenly among 3 ranks.` |
| `scatter_bad_root.py` | Root is not a rank of the communicator | trace | `ValueError: qscatter root 7 is not a rank of the communicator [0, 3).` |
| `no_main.py` | Script has no `main()` | load | `ValueError: <path> does not define a main() function` |
| `lonely_expose.py` | Only one rank joins a collective | translation | `RuntimeError: Deadlock while translating the group: ... {1: "Expose(rank=1, root=1, ranks=[1, 0], data_qubit=0, comm_slot=0, clbits=[0], tag='expose_1_0_0')"}` |
| `expose_outside_group.py` | Collective names a rank not in the run | translation | `RuntimeError: rank 0 issued Expose(...) naming ranks outside the group [0, 1].` |
| `mismatched_circuit_counts.py` | Ranks build different numbers of circuits | translation | `RuntimeError: Every rank must create the same number of circuits ... got {0: 2, 1: 1}.` |
| `unsupported_gate.py` | Operation the backend cannot express | translation | `NotImplementedError: Barrier is not implemented for the CUNQA backend.` |
| `dangling_qsend.py` | `qsend` with no matching `qrecv` | run | CUNQA's `Deadlock: every pending quantum task is blocked (circular SEND/RECV or unsatisfiable entanglement).` |
| `gather_disagreeing_chunks.py` | Ranks disagree on the chunk size | run | the same CUNQA deadlock, naming no NetQMPI call |
| `exceeds_qpu_qubits.py` | Circuit needs more qubits than the vQPU has | run | CUNQA's `ValueError: Not enough data qubits in the QPU for the circuit.`, naming no rank and no numbers |

The three tiers matter more than the individual messages:

- **trace** -- raised at the offending line while the rank's `main()` runs, so
  the traceback points into the user's own code. These are the good ones.
- **translation** -- raised from `__exit__` of the *last* rank to leave its
  `with comm:` block, because that is when the ranks' circuits are translated
  together. The traceback anchors at `with comm:`, never at the call that
  caused it, and the rank it is attributed to need not be the guilty one.
- **run** -- raised inside CUNQA, in CUNQA's vocabulary (quantum tasks, SEND /
  RECV), with nothing tying it back to a `qsend`, a `qscatter` or a line of
  user code.
