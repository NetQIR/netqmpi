# NetQASM / SquidASM backend

```bash
netqmpi -n <N> app.py --netqasm
```

The NetQASM backend targets **low-level quantum-network simulation**: EPR
sockets, NetQASM subroutines and a simulated network topology, executed on
NetSquid through SquidASM. It is the backend NetQMPI was originally built on,
and the **default** when no flag is given.

- **Packages:** `squidasm`, `netsquid`, `netqasm` **1.x**
- **Adapter:** {mod}`netqmpi.runtime.adapters.netqasm`

## Installation

```bash
pip install squidasm --extra-index-url https://<user>:<pwd>@pypi.netsquid.org
```

NetSquid requires a free account; see the
[NetQASM installation docs](https://netqasm.readthedocs.io/en/stable/installation.html).

:::{warning}
This backend uses `netqasm` **1.x**, which is incompatible with the **2.x** that
[Qoala](qoala.md) requires. The two must live in separate environments.
:::

## What it supports

| Primitive | Status |
|---|---|
| `qsend` / `qrecv` | ✅ teleportation over EPR sockets |
| `qscatter` / `qgather` | ❌ |
| `expose` / `unexpose` | ❌ |
| `reset` | ❌ `NotImplementedError` |
| `barrier` | ❌ `NotImplementedError` |
| Classically controlled gates | ❌ `NotImplementedError` |

Gates: `H` `X` `Y` `Z` `S` `T`, plus `RX` `RY` `RZ`. `SDG` and `TDG` raise.

:::{admonition} Current adapter limitations
:class: caution

Three rough edges are worth knowing before you pick this backend:

**Controlled gates do not work.** The adapter looks up `op.name` on a
`ControlledGate`, which has no such attribute, so every `cx`, `cz`, `ccx`, `cs`,
`ct`, `cp` and `crz` raises:

```text
AttributeError: 'ControlledGate' object has no attribute 'name'
```

**Rotation angles are not faithful.** `rx`/`ry`/`rz` are emitted as
`rot_X(n=round(theta), d=16)`, i.e. an angle of `round(theta)·π/2¹⁶`. A `rz(π/2)`
becomes a rotation of roughly 10⁻⁴ radians.

**`swap` is dropped silently.** It is not in the adapter's gate map, and unknown
names are ignored rather than rejected, so no instruction is emitted and no error
is raised.
:::

## Configuration

The `netqasm` block accepts the fields of
{class}`~netqmpi.runtime.adapters.netqasm.netqasm_executor.NetQASMRunConfig`:

| Key | Default | Meaning |
|---|---|---|
| `formalism` | `Formalism.KET` | Quantum state formalism |
| `enable_logging` | `true` | Per-rank instruction logging |
| `hardware` | `"generic"` | Hardware model name |
| `network_config` | `None` | Simulated topology; default when unset |
| `log_cfg` | `None` | NetQASM log configuration |
| `roles` | `"roles.yaml"` | Roles configuration file |
| `post_function` | `None` | Function invoked after the simulation |

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

## Results

`comm.results` is a histogram of **this rank's own** measurement outcomes,
accumulated across flushes:

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
