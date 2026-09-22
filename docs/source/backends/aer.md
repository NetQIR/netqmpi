# Qiskit Aer backend

```bash
netqmpi -n <N> app.py --aer [--shots N]
```

The Aer backend maps the whole distributed program onto **one monolithic
`QuantumCircuit`** and simulates it with `AerSimulator`. It models no network at
all — there is no entanglement distribution, no latency, no decoherence between
nodes — which makes it the wrong tool for studying network physics and the right
one for checking that a program's *logic* is correct.

It is the lightest backend and the only one with no special installation
requirements.

- **Packages:** `qiskit`, `qiskit-aer`
- **Adapter:** {mod}`netqmpi.runtime.adapters.aer`

## Installation

```bash
pip install qiskit qiskit-aer
```

## What it supports

| Primitive | Status |
|---|---|
| `qsend` / `qrecv` | ✅ SWAP-based (see below) |
| `qscatter` / `qgather` | ✅ via the `qsend`/`qrecv` blocks the SDK records them as |
| `expose` / `unexpose` | ✅ the control is shared in place (see below) |
| `reset` | ✅ |
| `barrier` | ✅ |
| Classically controlled gates | ❌ `NotImplementedError` |

Gates: `H` `X` `Y` `Z` `S` `SDG` `T` `TDG` `SWAP` `RX` `RY` `RZ`; controlled
`X` (one or two controls), `Z`, `RZ`, and the phase family `P` / `S` / `T`
(`cp`, `cs`, `ct`).

:::{note}
A gate the adapter does not know now raises `NotImplementedError`. It used to
be **dropped without warning**, which was worse than it sounds: `cs`, `ct` and
`cp` record a controlled `S`/`T`/`P`, none of which the table handled, so a QFT
written with `cp` lost every one of its rotations and still produced a
plausible histogram. An echo probe could not see it either — a transform whose
rotations have all vanished, followed by its inverse, is still the identity.
Both the missing gates and the silent skip are fixed.
:::

## Qubit transfer: `swap` versus `teleport`

`transfer_mode` selects how `qsend`/`qrecv` are realised.

**`"swap"`** (default)
: Inserts a SWAP gate between the source qubit slot and the matching slot on the
  destination rank, within the same circuit group. Produces shallower circuits
  and is easier to debug, but is **unphysical**: it does not model a real
  quantum-network transfer. `qrecv` is a no-op, since the state is already in
  the destination slot.

**`"teleport"`**
: **Not implemented** — it raises `NotImplementedError`. Worth doing only
  alongside an Aer noise model: on a noiseless simulator a teleportation
  circuit returns exactly what the SWAP returns, with more gates and two
  ancillas per transfer, so on its own it would add cost without adding
  information.

The receiver chooses where an incoming qubit lands. `qrecv` used to be a no-op
and `qsend` swapped to the *same* local index on the destination, ignoring the
index the receiver asked for — so a program where the two sides named different
indices meant different things on different backends. Pairing each `qsend` with
its `qrecv` supplies the missing half.

```yaml
aer:
  transfer_mode: swap        # the only working mode today
  seed_simulator: 7
```

## Execution model

All *N* ranks run concurrently in separate **threads**, synchronised by a
`threading.Barrier` in `AerCommunicator.__exit__`:

1. **Phase 1** — every rank finishes appending operations and reaches the
   barrier.
2. **Phase 2** — one designated thread translates every rank's circuits
   *together* and submits the simulation. Every other thread blocks.
3. **Phase 3** — all threads are released with results available and continue
   past the `with env.comm:` block simultaneously.

The barrier and the class-level communicator list are reset after the last rank
exits, so the adapter is reusable within the same process. A failure inside the
designated thread is captured, the barrier is released, and the executor
re-raises it; previously it left the other ranks waiting on a barrier nobody
would reach and the process hung instead of reporting anything.

### Why the ranks are translated together

Aer runs every rank inside one `QuantumCircuit`, so the order instructions are
appended in **is** the order they execute. Translating one rank fully before
starting the next therefore only works when a program's cross-rank dependencies
happen to follow rank order. A chain 0 → 1 → 2 survived it; a control that
returns to rank 0 between hops did not, and the run returned a plausible-looking
histogram **with no error at all**.

{func}`~netqmpi.runtime.adapters.aer.aer_circuit.translate_group` instead
interleaves the ranks the way they would really run: each advances through its
own operations until it reaches something it cannot emit alone — a transfer, or
a collective — and that call is expanded once every rank it involves is waiting
on it. Transfers that never pair up are reported as the deadlock they are,
naming the ranks and the tag.

### Telegate windows

`expose` shares a control without moving it. A real backend does that through a
GHZ state (CUNQA's cat-entangler) so each receiver holds a computational-basis
copy to use as a local control. Aer runs every rank inside one circuit, so the
copy is unnecessary: a receiver's controlled gate is emitted against the root's
own qubit directly.

That is exact rather than approximate — a telegate reads its control only in the
computational basis, which is precisely why the cat-entangled copy can stand in
for the original — and unphysical in the same way `qsend` is, which keeps this
backend a correctness reference rather than a model of a network.

### Register layout

A rank's slice is sized and placed only once every rank has finished tracing,
because the ranks need not ask for the same width: a `qscatter` root holds one
qubit per receiver while each receiver holds one. The circuit used to be sized
from the *first* rank to call `create_circuit` while each rank's offset was
computed from its *own* width, so the slices overlapped as soon as the widths
differed — `3_scatter` returned the wrong answer and `4_gather` died with
"duplicate qubit arguments". Slices are now laid out group by group and, within
a group, in rank order, which also makes the histogram's bit layout
deterministic instead of depending on which thread got there first.

## Results

`comm.results` is a **single counts dictionary for the whole global circuit** —
every rank's classical bits concatenated — broadcast identically to every rank:

```python
{'01': 517, '00': 507}
```

It is *not* keyed by rank, unlike CUNQA. Which slice of the bitstring belongs to
which rank follows from the classical-bit offsets, in rank order.

## Configuration

| Key | Default | Meaning |
|---|---|---|
| `shots` | `1024` | Number of simulation shots |
| `transfer_mode` | `"swap"` | `"swap"` or `"teleport"` |
| `seed_simulator` | `None` | RNG seed, for reproducible runs |

## API

- {class}`~netqmpi.runtime.adapters.aer.aer_executor.AerExecutorAdapter`
- {class}`~netqmpi.runtime.adapters.aer.aer_run_config.AerSimulatorConfig`
- {class}`~netqmpi.runtime.adapters.aer.aer_circuit.AerCircuitAdapter`
- {class}`~netqmpi.runtime.adapters.aer.aer_communicator.AerCommunicator`
