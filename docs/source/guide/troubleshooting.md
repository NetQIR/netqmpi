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
[`examples/frequent_errors/`](https://github.com/NetQIR/netqmpi/tree/main/examples/frequent_errors):
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
| `ValueError: The simulated network is too small for this program: rank 1 needs 3 qubits at once, node 'rank_1' has 2` | NetQASM: a rank's live data qubits, EPR half and open windows' communication qubits do not fit in its node — raise `num_qubits` |

The collective rows read the same on CUNQA and NetQASM, which both check the
order of the `expose`/`unexpose` calls across ranks before running anything.
Aer reports the same mistakes in its own words: `RuntimeError: The ranks blocked
on calls that never match`, followed by what each rank is waiting for.

### Caught only at run time

| Symptom | What is wrong |
|---|---|
| CUNQA's `ValueError: Not enough data qubits in the QPU for the circuit.`, naming no rank and no numbers | The circuit needs more qubits than the vQPU has — resize the [vQPU definition](../backends/cunqa.md#sizing-the-vqpus) |
| NetQASM: `RuntimeError: rank 1 expected the 'unexpose expose_...' outcome from rank 0 but received ...` | Two ranks reached their transfers and telegate windows in different orders; the correction is refused rather than applied to the wrong call |
| NetQASM: `RuntimeError: rank 1: communication qubit 1 (slot 0) holds no exposed control ...` | An exposed control was measured before its window closed |
| NetQASM: `TypeError: Plugin '<module:callable>' yielded a <type>; ...`, before the simulation | The plugin's context manager yields something other than `None`, a mapping or an object with `report()` — see [Reporting metrics](../backends/netqasm-plugins.md#reporting-metrics) |
| NetQASM: `RuntimeError: The simulation succeeded, but report() of plugin '<module:callable>' raised ...` | The plugin's `report()` failed; the chained traceback is the plugin's |

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

**A rank that failed, on Aer or NetQASM (0.3.1 and earlier).** Both run their ranks in
threads that wait for each other; a rank that raised left the others waiting
forever. Now the first failure ends the run with that rank's exception.

**NetQASM with two transfers in a row between the same ranks (0.3.1 and earlier).**
See [classical sockets](../backends/netqasm.md#classical-sockets).

**Finding out where it is stuck.** Have Python dump every thread's stack after a
while:

```python
import faulthandler
faulthandler.dump_traceback_later(60, exit=False)
```

The portability matrix (`pytest -m slow`) does this for every case it runs.

### A gate seems to do nothing

An operation an adapter cannot express should raise `NotImplementedError`,
naming the operation and the backend. If a gate seems to do nothing, check the
[backend support matrix](circuits.md#backend-support-matrix) before assuming the
physics is wrong, and report it as a bug if the adapter dropped it silently.

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

Raised by the NetQASM adapter for any controlled gate in 0.3.1 and earlier. It
now emits `cx` and `cz`, locally and from an exposed control, and raises
`NotImplementedError` for every other controlled gate; use CUNQA or Aer for
those.

### CUNQA says there are no vQPUs

The run stops before building anything and tells you what to raise. Either raise
them yourself once and reuse the allocation across runs, or ask NetQMPI to raise
them with `qraise: true` in the `cunqa` block:

```bash
qraise -n 3 -t 00:10:00 --quantum_comm --co-located    # once
netqmpi -n 3 examples/3_scatter.py --cunqa             # as often as you like
```
