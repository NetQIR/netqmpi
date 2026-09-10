# Troubleshooting

## The three error tiers

Where an error is raised matters more than its wording, because it determines
how much the traceback tells you.

::::{grid} 1 1 3 3
:gutter: 2

:::{grid-item-card} Trace time
Raised at the offending line while your `main()` runs. The traceback points into
your own code. **These are the good ones.**
:::

:::{grid-item-card} Translation time
Raised from `__exit__` of the *last* rank to leave its block, since that is when
the ranks' circuits are translated together. The traceback anchors at
`with comm:`, and the rank blamed need not be the guilty one.
:::

:::{grid-item-card} Run time
Raised inside the backend, in the backend's own vocabulary, with nothing tying
it back to a line of your code.
:::

::::

## Error catalogue

The repository ships
[`examples/frequent_errors/`](https://github.com/NetQIR/net-qmpi/tree/main/examples/frequent_errors):
a corpus of programs that are *meant* to fail, one failure mode each. Run them
all with

```bash
python examples/frequent_errors/run_all.py            # every example
python examples/frequent_errors/run_all.py scatter    # only matching ones
```

`run_all.py` traces and translates each example exactly as the CUNQA backend
would, stubbing only the submission to the vQPUs — so it needs CUNQA importable,
but no `qraise` and no SLURM allocation.

### Caught at trace time

| Symptom | What is wrong |
|---|---|
| `IndexError: Qubit index 5 out of range [0, 2) (2 data + 0 comm qubits).` | Gate on a qubit the circuit does not have |
| `ValueError: Not enough classical bits to measure all qubits (1 clbits < 3 qubits).` | `measure_all()` with too few classical bits |
| `IndexError: Qubit index 1 is a communication qubit whose expose window is already closed.` | Exposed control used after `unexpose` |
| `RuntimeError: rank 0 called unexpose(ranks=[0], root=1) without a matching open expose window.` | `unexpose` with no matching `expose` |
| `ValueError: the root of a qscatter must hold one chunk per receiving rank: 3 qubits do not split evenly among 2 ranks.` | Root buffer does not split evenly |
| `ValueError: qscatter root 7 is not a rank of the communicator [0, 3).` | Root is not a rank of the communicator |
| `ValueError: <path> does not define a main() function` | Script has no `main()` (raised at load) |

### Caught at translation time

| Symptom | What is wrong |
|---|---|
| `RuntimeError: Deadlock ...` followed by `expose(...) / called by: rank 1 / missing: rank 0, which finished its block without calling it` | Only some ranks joined a collective |
| The same report with `missing: rank 1, which is waiting on expose(ranks=[0], root=1)` | Each rank opened its own window and waits on the other |
| `RuntimeError: rank 0 called expose(ranks=[7], root=0), which names rank 7, but this run has 2 ranks, numbered 0 to 1.` | A collective names a rank outside the run |
| `RuntimeError: Every rank must create the same number of circuits ... got {0: 2, 1: 1}.` | Ranks built different numbers of circuits |
| `NotImplementedError: Barrier is not implemented for the CUNQA backend.` | Operation the backend cannot express |
| `RuntimeError: ... rank 0 sends qubit 0 to rank 1, which never receives it: rank 0 traced 1 qsend to rank 1, and rank 1 traced 0 qrecvs from rank 0` | `qsend` with no matching `qrecv` |
| The same check as a count mismatch | Ranks disagree on the chunk size of a collective |

### Caught only at run time

| Symptom | What is wrong |
|---|---|
| CUNQA's `ValueError: Not enough data qubits in the QPU for the circuit.`, naming no rank and no numbers | The circuit needs more qubits than the vQPU has — resize the [vQPU definition](../backends/cunqa.md#sizing-the-vqpus) |

## Common situations

### The run hangs and never finishes

Two very different causes:

**An oversized vQPU register on CUNQA.** The executor simulates the whole family
in one register spanning every qubit each vQPU declares, whether the circuits use
it or not. With the `Aer` simulator that register is a dense statevector,
reinitialised once per shot, so a generous vQPU definition turns into a run that
never seems to finish. It is waiting on the simulator, not deadlocked. Use the
default `Munich` simulator, or size the definition down — see
[Sizing the vQPUs](../backends/cunqa.md#sizing-the-vqpus).

**An error inside the CUNQA executor.** The executor has no error path back to
the client, so any exception it raises leaves the vQPUs waiting and the program
stuck in `future.get()`.

### A gate seems to do nothing

Some adapters ignore gate names they do not know instead of raising. A `swap` on
NetQASM, or a `cs`/`ct`/`cp` on Aer, is dropped without warning. Check the
[backend support matrix](circuits.md#backend-support-matrix) before assuming the
physics is wrong.

### `comm.results` is empty

Expected on most ranks. Circuits are submitted jointly when the *last* rank
leaves its `with comm:` block, so results land on that rank and — depending on
the backend — possibly not on the others. Guard with `if comm.results:`; see
{ref}`Reading results <reading-results>`.

It is also empty on any rank that measured nothing. After a `qsend` the sender's
qubit is back in `|0⟩`, so a rank that gave everything away contributes no
counts.

### A collective deadlocks

Every rank of the participant group must reach every collective, in the same
order. The usual culprit is a collective inside a branch:

```python
if rank == 0:                            # WRONG — only rank 0 reaches it
    comm.qscatter(circuit, [0], root=0)
```

Branch on what each rank *passes*, not on whether it calls:

```python
if rank == ROOT:                         # right — every rank calls
    comm.qscatter(circuit, list(range(size - 1)), root=ROOT)
else:
    comm.qscatter(circuit, [0], root=ROOT)
```

### `AttributeError: 'ControlledGate' object has no attribute 'name'`

Raised by the NetQASM adapter for any controlled gate. Controlled gates are not
usable on that backend as it stands; use CUNQA or Aer.

### CUNQA says there are no vQPUs

The run stops before building anything and tells you what to raise. Either raise
them yourself once and reuse the allocation across runs, or ask NetQMPI to raise
them with `qraise: true` in the `cunqa` block:

```bash
qraise -n 3 -t 00:10:00 --quantum_comm --co-located    # once
netqmpi -n 3 examples/3_scatter.py --cunqa             # as often as you like
```
