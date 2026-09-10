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
| `qsend` / `qrecv` | ⚠️ SWAP-based (see below) |
| `qscatter` / `qgather` | ❌ `NotImplementedError` |
| `expose` / `unexpose` | ❌ `NotImplementedError` |
| `reset` | ✅ |
| `barrier` | ✅ |
| Classically controlled gates | ❌ `NotImplementedError` |

Gates: `H` `X` `Y` `Z` `S` `SDG` `T` `TDG` `SWAP` `RX` `RY` `RZ`; controlled `X`
(one or two controls), `Z` and `RZ`.

:::{caution}
Gate names the adapter does not know are **dropped without warning** — no
instruction is emitted and no error is raised. In particular `cs`, `ct` and `cp`
record a controlled `S`/`T`/`P`, none of which the adapter handles, so a QFT
written with `cp` silently loses its rotations on this backend.
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
: A physically realistic teleportation circuit using mid-circuit measurement and
  classical feedforward. **Not yet implemented** — it raises
  `NotImplementedError`.

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
2. **Phase 2** — one designated thread translates all circuits *in rank order*,
   so gate ordering in the global circuit is deterministic, and submits the
   simulation. Every other thread blocks.
3. **Phase 3** — all threads are released with results available and continue
   past the `with env.comm:` block simultaneously.

The barrier and the class-level communicator list are reset after the last rank
exits, so the adapter is reusable within the same process.

Rank *r* owns qubits `[r·num_qubits, (r+1)·num_qubits)` of the global circuit;
that offset is what `qsend`'s SWAP targets.

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
