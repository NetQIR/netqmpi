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
| `enable_logging` | `false` | Per-rank instruction logging — see [Logging](#logging) |
| `hardware` | `"generic"` | Node hardware, `"generic"` or `"nv"` |
| `network_config` | `None` | Path to a network YAML; when unset the network is built from the keys below |
| `num_qubits` | `5` | Qubits per node (SquidASM's own default) |
| `t1`, `t2` | `0`, `0` | Qubit amplitude-damping and dephasing times, in ns; 0 disables them |
| `gate_fidelity` | `1.0` | Fidelity of every gate |
| `link_fidelity` | `1.0` | Fidelity of the EPR pairs every link delivers |
| `link_noise` | `"Depolarise"` | Link noise model used when `link_fidelity < 1`: `Depolarise`, `DiscreteDepolarise` or `Bitflip` |
| `epr_setup_timeout` | `60.0` | Wall-clock seconds a node waits for a peer to set up an EPR socket (SquidASM hard-codes 5) |
| `poll_interval` | `0.001` | Seconds SquidASM's main thread sleeps between checks on the program threads |
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

**With the defaults the simulation is ideal**: five perfect qubits per node,
perfect gates and noiseless links, exactly SquidASM's default network. Noise has
to be asked for:

```yaml
netqasm:
  num_qubits: 8
  t1: 1.0e9
  t2: 5.0e8
  gate_fidelity: 0.99
  link_fidelity: 0.95
```

Before simulating, the adapter checks that every rank fits in its node — its
live data qubits plus the EPR half it holds while transferring — and refuses the
run otherwise, naming the rank, what it needs and what the node has:

```text
ValueError: The simulated network is too small for this program:
  rank 0 needs 7 qubits at once, node 'rank_0' has 5
```

`cascade` with six qubits per rank, for instance, needs seven.

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

Repetitions go through SquidASM's own `num_rounds`, and each shot starts from
empty qubit slots, or a qubit the program never measured would survive the round
and be reused dead by the next one.

Each rank opens EPR sockets only towards its **peers** — the ranks its traced
program sends to or receives from. A chain therefore sets up O(n) EPR circuits
instead of O(n²), which matters because each set-up runs under SquidASM's
wall-clock timeout (see [scale limits](#scale-limits-and-the-squidasm-workarounds)).

### Classical sockets

A rank has **one classical socket per peer and per round**: opened by the first
transfer with that peer in the round, reused by every later one, and dropped when
the round ends.

Both alternatives fail, for reasons inside NetQASM's thread-socket hub:

- *One per transfer* (0.3.1) deadlocks with two transfers in a row in the same
  direction. All sockets between two ranks share one hub key, and closing a
  socket erases the marker its peer's socket left in the hub. A sender that ran
  ahead — its second socket opened and closed while the receiver was still in
  the first transfer — lost that marker when the receiver closed its first
  socket, and the receiver's second socket then waited forever for a peer that
  was gone. This is why `cascade` never finished with q ≥ 2, while `ghz`, whose
  transfers alternate direction, did.
- *One for the whole run* fails from the second round: SquidASM resets the hub
  between rounds, and the kept socket reports "Socket is not connected so cannot
  send".

`test/test_netqasm_sockets.py` reproduces the hub behaviour with NetQASM's own
sockets and needs no NetSquid.

### Scale limits and the SquidASM workarounds

**EPR-socket set-up timeout.** SquidASM waits for the remote node to install its
rules with a timeout of 5 s measured in **wall-clock** time, not simulated time
(`squidasm/nqasm/netstack.py`, `_wait_for_remote_node`), and exposes no way to
change it. With many ranks in threads, or on a loaded machine, runs failed with
`TimeoutError: Remote node did not initialize the correct rules` — 1 in 16 runs at
10 ranks, 3 in 7 at 12, all of them from 16. The adapter patches the default for
the duration of each simulation to `epr_setup_timeout` (60 s) and restores it
afterwards. This is a workaround for a SquidASM limitation and should be reported
upstream.

**Busy wait.** SquidASM waits for the program threads with
`as_completed(..., sleep_time=0)`, a loop that never sleeps and competes for the
GIL with the threads it waits on. The adapter gives it a `poll_interval` sleep,
again only while its own simulation runs.

**Blocking polls.** NetQASM's thread sockets poll every 0.1 s when a peer has
not connected yet or a message has not arrived, so every transfer whose partner
is not already waiting costs up to a tenth of a second of wall-clock time. Long
serial chains of transfers (`ghz` makes 4(n−1)) are slow for that reason alone.

**When a rank fails.** An exception in any rank's program — while tracing, or
inside a SquidASM thread — ends the run with that exception. SquidASM itself
would leave its NetSquid thread running, keeping the process alive and making
the next simulation in the same process fail with "Already a backend running";
the adapter stops it. Program threads blocked inside NetQASM on the failed run
cannot be interrupted; they are daemon threads and do not keep the process
alive.

## Results

`comm.results` is a histogram of **this rank's own** measurement outcomes, one
bit string per shot, most significant bit first, with unmeasured bits reading
`0`:

```python
{'0': 517, '1': 507}
```

Unlike CUNQA, it is not keyed by rank and does not carry the other ranks' counts.

## Logging

Logging is **off by default**. SquidASM writes its logs to
`./log/<YYYYmmdd-HHMMSS>` and names that directory to the second, so two runs in
the same working directory starting within the same second failed with
`FileExistsError`; the logs also add disk I/O inside the simulation being
timed.

With `enable_logging: true`, each run gets a directory of its own under `./log`
(unless `log_cfg.log_dir` names one), and NetQASM writes per-rank instruction
logs and a network log there:

```text
log/20260305-102400-k2j3x9/20260305-102400/
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
