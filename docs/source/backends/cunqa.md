# CUNQA backend

```bash
netqmpi -n <N> app.py --cunqa [--shots N] [--config cunqa.yaml]
```

{bdg-primary}`reference backend`

CUNQA emulates distributed quantum computing on HPC infrastructure through
**virtual QPUs** (vQPUs), provisioned via the job scheduler. It is NetQMPI's
reference backend: the only one that implements every communication primitive,
and the one the shipped examples are written against.

- **Package:** [`cunqa`](https://github.com/CESGA-Quantum-Spain/cunqa)
- **Requires:** an HPC environment with SLURM
- **Adapter:** {mod}`netqmpi.runtime.adapters.cunqa`

## What it supports

| Primitive | Status |
|---|---|
| `qsend` / `qrecv` | ✅ via `cunqa.qc_protocols.qsend` / `qrecv` |
| `qscatter` / `qgather` | ✅ expanded into their constituent transfers |
| `expose` / `unexpose` | ✅ via `cat_entangler` / `cat_disentangler` |
| `reset` | ✅ |
| `barrier` | ❌ `NotImplementedError` |
| Classically controlled gates | ❌ `NotImplementedError` |
| Several circuits per rank | ✅ the *i*-th circuit of every rank forms one program |

Gates: `H` `X` `Y` `Z` `S` `SDG` `T` `TDG` `SWAP` `RX` `RY` `RZ` `P`; controlled
versions with one control (`X` `Y` `Z` `H` `S` `SDG` `T` `SX` `SWAP` `RX` `RY`
`RZ` `P`), two controls (`X` `Y` `Z`, as `ccx`/`ccy`/`ccz`) and more (`mcx`,
`mcy`, `mcz`). Anything else raises `NotImplementedError` naming the gate — this
adapter rejects unknown gates rather than dropping them.

## Which vQPUs the run uses

A run needs **one vQPU per rank**. By default it expects them to be *already
raised*, so one allocation can serve many runs:

```bash
qraise -n 3 -t 00:10:00 --quantum_comm --co-located    # once
netqmpi -n 3 examples/3_scatter.py --cunqa             # as often as you like
```

### Spare vQPUs are harmless — but not idle

The family may hold **more** vQPUs than the run needs: three raised, `-n 2` run.
The extras cost the program nothing, but they cannot simply be left out.

CUNQA runs a single executor per family, and it starts a round only once *every*
vQPU of that family has submitted something. A vQPU left out would not sit idle
— it would hang the run. NetQMPI therefore hands each spare one a trivial
circuit ({func}`~netqmpi.runtime.adapters.cunqa.cunqa_circuit.idle_circuit`) and
discards its counts.

### A run cannot spread across families

Each family is executed on its own. If vQPUs of several families are up, name the
one to use:

```yaml
cunqa:
  family: my_family
```

Otherwise the run stops and lists what it found:

```text
RuntimeError: Found vQPUs of more than one family running: 'a' (3 vQPUs),
'b' (2 vQPUs). A run cannot spread across families, because each of them is
executed on its own, so name the one to use with 'family: <name>' in the cunqa
block of the config file.
```

If there are **no** vQPUs up at all, the run stops before building anything and
says exactly what to raise:

```text
RuntimeError: This run needs 3 vQPUs but found 0 already raised. Raise them
before running, for instance with 'qraise -n 3 -t 00:10:00 --quantum_comm
--co-located', or add 'qraise: true' to the cunqa block of the config file to
have NetQMPI raise and drop them for you.
```

## Letting NetQMPI raise them

Ask for it in the `cunqa` block, and the vQPUs are raised for the run and dropped
afterwards. `backend` is the vQPU definition file they are raised with, which is
what fixes the qubit budget of the run:

```yaml
# cunqa.yaml
shots: 1024
cunqa:
  qraise: true                          # raise for this run, drop after it
  backend: examples/cunqa_backend.json  # vQPU definition (qubit budget)
  time: "00:10:00"                      # SLURM reservation
  simulator: Munich
```

```bash
netqmpi -n 3 examples/3_scatter.py --cunqa --config cunqa.yaml
```

Only vQPUs *this run* raised are dropped, so a family you raised beforehand stays
up for your next run. A failure during setup drops them too, so a broken run
never leaves a family behind.

`family` picks which raised vQPUs to attach to, or names the family to raise, and
`co_located` has to match how they were raised. `backend`, `time` and `simulator`
only mean anything when `qraise: true`, so setting them while attaching is
**reported rather than silently ignored** — see
[the cunqa block](../guide/configuration.md#cunqa-block).

## Sizing the vQPUs

:::{admonition} This is the single most common cause of a run that never finishes
:class: danger

The executor simulates the whole family in **one register**, spanning every qubit
each vQPU declares — whether the circuits use it or not. The cost of a run is
therefore set by `num_qubits` × the number of ranks, **not** by the circuits.

With a statevector simulator that register is 2^N amplitudes:

| vQPU definition | `-n 2` | `-n 3` | `-n 4` | `-n 5` |
|---|---|---|---|---|
| `[4, 4]` — 8 qubits each | 1 MiB | 256 MiB | 64 GiB | 16 TiB |
| `[3, 2]` — 5 qubits each | 16 KiB | 512 KiB | 16 MiB | 512 MiB |
:::

That is why `simulator` matters:

**`Munich`** (NetQMPI's default)
: Decision diagrams keep a mostly-idle register small, so oversized vQPUs go
  unnoticed.

**`Aer`** (CUNQA's own default)
: Allocates the dense statevector and reinitialises it once per shot. The same
  program on generous vQPUs turns into a run that never seems to finish — it is
  waiting on the simulator, not deadlocked.

[`examples/cunqa_backend.json`](https://github.com/NetQIR/net-qmpi/tree/main/examples)
is sized for the examples and runs on either. It fits them all up to three ranks;
`4_gather.py`'s root holds one qubit per rank, so a four-rank run of it wants
`[4, 2]`.

If the circuit needs more qubits than the vQPU has, the error comes from CUNQA
itself, at run time, naming no rank and no numbers:

```text
ValueError: Not enough data qubits in the QPU for the circuit.
```

## How the adapter works

### Joint translation

CUNQA is the reason the `CircuitAdapter` contract has a group-translation escape
hatch. Its telegate helpers — `cat_entangler` and `cat_disentangler` — write
instructions into **every** participating circuit in a single call, so they can
only run once all the ranks are known and each has been translated *up to* the
matching call.

{func}`~netqmpi.runtime.adapters.cunqa.cunqa_circuit.translate_group` performs
that joint pass:

1. Every rank's operation stream is drained on its own until it reaches a
   collective.
2. Once all the participants of a collective are waiting on it, the call is
   expanded into all of their circuits at once, and they resume.

This mirrors what the ranks would do if they really ran side by side, while
keeping each circuit's instructions in program order. It is also what turns a
deadlock into a *report* instead of a hang: if the ranks block on collectives
that never match, `translate_group` says which rank is missing and what it was
doing instead.

The rooted transfers sit in between. `qscatter` and `qgather` are collective for
the user — every rank has to call them — but each rank's half is a plain sequence
of transfers, so the adapter just translates the container's children.

The per-rank `_translate_expose` / `_translate_unexpose` hooks therefore raise
deliberately:

```text
RuntimeError: Expose(...) is collective and must be expanded by translate_group(),
which needs the circuits of all the participating ranks.
```

### Pre-flight checks

Before anything is submitted, `translate_group` calls `check_transfers`, which
verifies that every `qsend` of the group has its matching `qrecv`. Transfers used
to be paired by tag at run time, so a `qsend` nobody received could only make
CUNQA hang. Now it reports itself in NetQMPI's own terms:

```text
RuntimeError: rank 0 sends qubit 0 to rank 1, which never receives it:
rank 0 traced 1 qsend to rank 1, and rank 1 traced 0 qrecvs from rank 0
```

### Resource layout

{meth}`CunqaCircuitAdapter.prepare` reserves, on the CUNQA circuit, the
communication qubits and the protocol classical register the trace asked for. The
protocol register has to sit **after** the user's own bits, which is why
`prepare` must run before any instruction is emitted.

## Known limitation: no error path back

CUNQA cannot *tell* you when something goes wrong at run time. The executor has
no error path back to the client, so any exception it raises leaves the vQPUs
waiting and the program stuck in `future.get()`.

In practice this means: if a `--cunqa` run hangs with no output, suspect either
an oversized register (above) or an exception inside the executor — not a bug in
your collectives, which are checked before submission.

## API

- {class}`~netqmpi.runtime.adapters.cunqa.cunqa_executor.CunqaExecutorAdapter`
- {class}`~netqmpi.runtime.adapters.cunqa.cunqa_executor.CunqaRunConfig`
- {class}`~netqmpi.runtime.adapters.cunqa.cunqa_circuit.CunqaCircuitAdapter`
- {func}`~netqmpi.runtime.adapters.cunqa.cunqa_circuit.translate_group`
- {class}`~netqmpi.runtime.adapters.cunqa.cunqa_communicator.CunqaCommunicator`
- {class}`~netqmpi.runtime.adapters.cunqa.cunqa_communicator.CunqaSession`

Full reference: {ref}`CUNQA adapter <cunqa>`.
