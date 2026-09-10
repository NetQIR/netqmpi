# Backends

The same, unmodified application runs on every backend; you choose one with a
command-line flag. What changes is *what is being modelled* — a quantum network,
an HPC emulation, a plain circuit — and how faithfully.

## Choosing one

| Backend | Flag | What it targets | Key dependencies |
|---|---|---|---|
| **[CUNQA](cunqa.md)** {bdg-primary}`reference` | `--cunqa` | HPC emulation of DQC through virtual QPUs (vQPUs) | [`cunqa`](https://github.com/CESGA-Quantum-Spain/cunqa) (HPC / SLURM environment) |
| **[NetQASM / SquidASM](netqasm.md)** | `--netqasm` | Low-level quantum-network simulation (EPR sockets, NetQASM routines) | [`squidasm`](https://github.com/QuTech-Delft/squidasm), [`netsquid`](https://netsquid.org), `netqasm` **1.x** |
| **[Qiskit Aer](aer.md)** | `--aer` | Shot-based circuit simulation | `qiskit`, `qiskit-aer` |
| **[Qoala](qoala.md)** | `--qoala` | Quantum-internet **node execution environment**, with task scheduling and multitasking — simulation only | [`qoala`](https://github.com/QuTech-Delft/qoala-sim), [`netsquid`](https://netsquid.org), `netqasm` **2.x**, Python 3.10–3.12 |

::::{grid} 1 1 2 2
:gutter: 2

:::{grid-item-card} Start here
**CUNQA** is the reference backend. It is the only one implementing the whole
primitive set — transfers, rooted collectives and the telegate window — and the
one the shipped examples are written against.
:::

:::{grid-item-card} No cluster at hand?
**Aer** is the lightest option and the only one with no special installation
requirements, but it supports point-to-point transfers only, via an unphysical
SWAP.
:::

::::

## Feature support at a glance

| | CUNQA | NetQASM | Aer | Qoala |
|---|:--:|:--:|:--:|:--:|
| `qsend` / `qrecv` | ✅ | ✅ | ⚠️ SWAP-based | ✅ |
| `qscatter` / `qgather` | ✅ | ❌ | ❌ | ❌ |
| `expose` / `unexpose` | ✅ | ❌ | ❌ | ❌ |
| Controlled gates | ✅ | ❌ | partial | ❌ |
| `reset` | ✅ | ❌ | ✅ | ❌ |
| `barrier` | ❌ | ❌ | ✅ | ❌ |
| Several circuits per rank | ✅ | ✅ | ✅ | ❌ |
| Hardware noise model | via vQPU definition | via `formalism` | ❌ | ✅ full qdevice |
| Runs without special hardware | ❌ needs SLURM | ✅ | ✅ | ✅ |

The per-gate breakdown, including which operations are *silently ignored* rather
than rejected, is in the
[gate support matrix](../guide/circuits.md#backend-support-matrix).

(comm-results-shape)=
## The shape of `comm.results`

The one place the abstraction is not watertight: each backend reports results
differently.

| Backend | Shape | Populated on |
|---|---|---|
| CUNQA | `{rank: {bitstring: count}}` — every rank's counts | every rank |
| Aer | `{bitstring: count}` for the *whole* global circuit, all ranks' classical bits concatenated | every rank, identically |
| Qoala | `{bitstring: count}` for this rank alone | every rank |
| NetQASM | `{outcome: count}` for this rank's own measurements | the rank itself |

## Environment constraints

:::{admonition} NetQASM 1.x and Qoala's 2.x cannot share an environment
:class: warning

The `--netqasm` and `--qoala` backends depend on mutually incompatible major
versions of `netqasm`, so they must live in **separate** environments (two conda
envs, for instance). CUNQA and Aer have no such constraint.

This is also why the CLI imports backends lazily: only the one you select is
ever imported.
:::

:::{admonition} NetSquid needs a free account
:class: note

Both NetQASM and Qoala depend on [NetSquid](https://netsquid.org), installed
from a private index:

```bash
pip install netsquid --extra-index-url https://<user>:<pwd>@pypi.netsquid.org
```
:::

## How a backend is put together

Every backend provides the same three Runtime components, and **adding one never
requires touching the SDK**:

```{mermaid}
graph LR
    subgraph SDK["SDK — backend-agnostic"]
        E[Environment]
        C[Circuit]
        Q[QMPICommunicator]
    end
    subgraph RT["Runtime — backend-specific"]
        X[Executor]
        A[CircuitAdapter]
        M[Communicator]
    end
    E -->|create_circuit| X
    C -->|recorded ops| A
    Q -->|on exit| M
    A --> B[(Backend)]
    M --> B
    X --> B
```

See [Architecture](../development/architecture.md) and
[Writing a backend](../development/writing-a-backend.md).

```{toctree}
:hidden:

cunqa
netqasm
aer
qoala
```
