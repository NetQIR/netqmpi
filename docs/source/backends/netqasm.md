# NetQASM / SquidASM backend

```bash
netqmpi -n <N> app.py --netqasm      # NetQASM 2.x
netqmpi -n <N> app.py --netqasm1.0   # legacy NetQASM 1.x
```

The NetQASM backend targets **low-level quantum-network simulation**: EPR
sockets, NetQASM subroutines and a simulated network topology, executed on
NetSquid through SquidASM. It is the backend NetQMPI was originally built on,
and the **default** when no flag is given.

- **Packages:** `squidasm`, `netsquid`, `netqasm` **2.x** (or **1.x** with
  `--netqasm1.0`)
- **Adapter:** {mod}`netqmpi.runtime.adapters.netqasm`

## Installation

NetSquid requires a free account; see the
[NetQASM installation docs](https://netqasm.readthedocs.io/en/stable/installation.html).
SquidASM comes from the same private index.

```bash
conda create -n netqasm2 python=3.11 pip -y
conda activate netqasm2
export PIP_EXTRA_INDEX_URL='https://<user>:<url-encoded-pwd>@pypi.netsquid.org'
pip install "squidasm>=0.13" "netqasm>=2,<3"
```

The exact set this was verified against is pinned in
`environments/netqasm2-requirements.txt`.

:::{tip}
Pin `squidasm>=0.13`. PyPI carries a placeholder package also called
`squidasm`, at version `0.0.1` with no dependencies, and pip will happily
install that instead of the real one from the private index. A version floor
is what forces it to look in the right place.
:::

### Which NetQASM, and why two flags

`--netqasm` targets NetQASM **2.x**, `--netqasm1.0` the older **1.x**. Both
drive the *same* adapter — the API this backend uses is unchanged between the
releases and SquidASM accepts either — so the flag selects an **environment**,
and the run stops immediately, naming both versions, if the installed one is
not the one asked for.

Three boundaries are worth knowing:

- **NetQASM 2.x needs Python ≥ 3.9** (PEP 585 generics), so it cannot be
  dropped into a 3.8 environment built for 1.x.
- **SquidASM caps at NetQASM 2.0.0.** Version 0.13.6 declares
  `netqasm<=2.0.0,>=1.0.0`. NetQASM ships 2.1, 2.2 and 2.3 as well, but no
  SquidASM release supports them — 2.3 is what [Qoala](qoala.md) builds on,
  which is a different simulator.
- **SquidASM and Qoala still cannot share an environment**, though not because
  of NetQASM: they need incompatible majors of `netsquid-magic` (16.x and 14.x
  respectively).

The adapter therefore stays inside the instruction set both releases share.
`Qubit.swap` exists in 2.3 but not in 2.0, and calling it against a SquidASM
that does not implement it *hangs* the simulation rather than failing, so a
swap is still assembled from three CNOTs.

## What it supports

| Primitive | Status |
|---|---|
| `qsend` / `qrecv` | ✅ teleportation over EPR sockets |
| `qscatter` | ✅ via the teledata blocks the SDK records it as |
| `qgather` | ⚠️ aborts — see below |
| `expose` / `unexpose` | ❌ `NotImplementedError` |
| `reset` | ❌ `NotImplementedError` |
| `barrier` | ❌ `NotImplementedError` |
| Classically controlled gates | ❌ `NotImplementedError` |

Gates: `H` `X` `Y` `Z` `S` `T` `SWAP`, plus `RX` `RY` `RZ`; controlled `X` and
`Z`. `SDG`, `TDG` and the controlled phase family raise — NetQASM has no
controlled-rotation instruction.

:::{caution}
`qgather` still aborts. `3_scatter` reproduces CUNQA's reference output
exactly, but `4_gather` — where the root receives several qubits into one
register — dies inside NetQASM's own teardown:

```text
_free_physical_qubit: self._used_physical_qubit_addresses.remove(...)
KeyError: 0
```

a physical qubit freed twice. It is the same family as the allocation race
fixed below, reached by a pattern the fix does not cover.
:::

:::{admonition} What was fixed
:class: note

This backend did not run at all until recently: `program_inputs` was built
empty, so SquidASM raised `KeyError: 'rank_0'` on the first party it tried to
start. Behind that first error were eight more problems, all now fixed:

- **`translate()` re-allocated the whole register on every call.** It recurses
  into an `OperationContainer` through itself, so each nested operation got a
  brand-new register and abandoned the one earlier operations were written
  against.
- **Corrections were sent unresolved.** `qsend` put the futures of the two Bell
  measurements on the socket without flushing first, so the receiver got
  objects rather than bits.
- **A sent qubit was left dead in its slot.** Measuring frees a qubit in
  NetQASM, but the SDK promises the slot holds `|0⟩` after a `qsend`; touching
  it again aborted the run. Slots are now filled **lazily**, which also removed
  a filler qubit `qrecv` had to free and with it a race that aborted roughly
  one run in four.
- **`shots` was ignored.** `num_rounds` was pinned at 1, so every run returned
  a single sample and a 50/50 outcome came back as a certainty.
- **No controlled gate had ever worked.** The adapter read `op.name` on a
  `ControlledGate`, which has no such attribute, so `cx` and `cz` raised
  `AttributeError`. `SWAP`, which the SDK records as a two-qubit `Gate`, sat in
  the controlled-gate table where nothing would look for it, and was
  implemented as a single CNOT besides.
- **Unknown gates were dropped in silence**, emitting a circuit without the
  gate and a plausible histogram for a program that never ran. They now raise.
- **Results were counted per measurement, not per shot**, so a rank with two
  classical bits reported twice as many single-bit outcomes as it ran shots.
  A shot is now one bit string, like every other backend returns.
- **Failures did not end the process.** Translation now happens *before* the
  simulator starts, so an unsupported gate is reported at once instead of
  killing a program thread and leaving the run waiting for it forever.
:::

## Configuration

The `netqasm` block accepts the fields of
{class}`~netqmpi.runtime.adapters.netqasm.netqasm_executor.NetQASMRunConfig`:

| Key | Default | Meaning |
|---|---|---|
| `shots` | `50` | Simulated repetitions — see the note below |
| `netqasm_major` | `2` | NetQASM release the run expects; `--netqasm1.0` sets `1` |
| `formalism` | `Formalism.KET` | Quantum state formalism |
| `enable_logging` | `true` | Per-rank instruction logging |
| `hardware` | `"generic"` | Hardware model name |
| `network_config` | `None` | Simulated topology; default when unset |
| `log_cfg` | `None` | NetQASM log configuration |
| `roles` | `"roles.yaml"` | Roles configuration file |
| `post_function` | `None` | Function invoked after the simulation |

:::{important}
`shots` defaults to **50** here, not to the generic 1024. SquidASM simulates
the whole network once per shot, at roughly a second per shot on a two-rank
program, so the generic default would take over a quarter of an hour and print
nothing until it finished — indistinguishable from a hang. Raise it with
`--shots` when the statistics matter more than the wait.
:::

Several of these hold Python objects rather than scalars and cannot be expressed
in YAML. Build a `NetQASMRunConfig` in Python and call
{func}`~netqmpi.runtime.cli.simulate` directly instead — see
[Programmatic use](../guide/cli.md#programmatic-use).

### Choosing the simulator

The underlying simulator is read from the `NETQASM_SIMULATOR` environment
variable, defaulting to NetSquid:

```bash
NETQASM_SIMULATOR=netsquid netqmpi -n 2 app.py --netqasm
```

## Execution model

All ranks run in the same process. Each rank's `__exit__` wraps its circuit in a
NetQASM `Program` and registers it; when the last rank registers, the adapter
assembles an `ApplicationInstance` and hands it to
`netqasm.sdk.external.simulate_application`, which runs one simulation for all
ranks at once.

Rank *r* is the party named `rank_r`. The party-to-node allocation is read from
the `roles` file if present, and otherwise defaults to the identity mapping.

Repetitions go through SquidASM's own `num_rounds`. Making that work needed two
things: classical sockets are no longer cached — the communicator outlives the
network they were opened on, and a second round found the cached socket closed
("Socket is not connected so cannot send") — and each shot starts from empty
qubit slots, or a qubit the program never measured would survive the round and
be reused dead by the next one.

## Results

`comm.results` is a histogram of **this rank's own** measurement outcomes, one
bit string per shot, most significant bit first, with unmeasured bits reading
`0`:

```python
{'0': 517, '1': 507}
```

Unlike CUNQA, it is not keyed by rank and does not carry the other ranks' counts.

## Logging

With `enable_logging` on, NetQASM writes per-rank instruction logs and a network
log under `log/<timestamp>/`:

```text
log/20260305-102400/
├── network_log.yaml
├── rank_0_instrs.yaml
├── rank_1_instrs.yaml
├── subroutines_rank_0.pkl
└── results.yaml
```

These record the NetQASM subroutines each rank actually ran, which is the most
direct way to see what a `qsend` expanded into.

## API

- {class}`~netqmpi.runtime.adapters.netqasm.netqasm_executor.NetQASMExecutorAdapter`
- {class}`~netqmpi.runtime.adapters.netqasm.netqasm_executor.NetQASMRunConfig`
- {class}`~netqmpi.runtime.adapters.netqasm.netqasm_circuit.NetQASMCircuitAdapter`
- {class}`~netqmpi.runtime.adapters.netqasm.netqasm_communicator.NetQASMCommunicator`
