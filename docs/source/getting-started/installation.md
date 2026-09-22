# Installation

NetQMPI has a small core and a set of optional backends. The core is what your
application code depends on; each backend lives behind a **lazy import**, so you
only need to install the dependencies of the backend you actually run.

## Core package

```bash
pip install netqmpi
```

The core has a single runtime dependency, `pyyaml` (used to parse
[`--config`](../guide/configuration.md) files). Installing it gives you the
`netqmpi` command-line entry point and the whole SDK.

To work from a checkout instead:

```bash
git clone https://github.com/NetQIR/net-qmpi.git
cd net-qmpi
pip install -e .
```

## Backends

Pick the backend that matches what you want to do — see
[Backends](../backends/index.md) for the comparison — and install its
dependencies.

:::::{tab-set}

::::{tab-item} CUNQA
:sync: cunqa

HPC emulation of DQC through virtual QPUs. This is the **reference backend**,
and what the shipped examples are written against.

On a cluster: install and configure
[CUNQA](https://github.com/CESGA-Quantum-Spain/cunqa) (it provisions vQPUs
through the job scheduler), then install `netqmpi` in the same environment.

On an ordinary computer, use the container — it packs a single-node SLURM,
CUNQA and NetQMPI together, so your machine stands in for the HPC environment:

```bash
docker pull jvazquezperez/cunqa_netqmpi
docker run --rm -it -p 8888:8888 jvazquezperez/cunqa_netqmpi
```

SLURM and a Jupyter server on port 8888 (token `cunqa`) come up, and you get a
shell. It runs the real backend, so it also runs out of room like one: expect a
ceiling of five or six ranks on a single machine.

See the [CUNQA backend page](../backends/cunqa.md) for mounting your own
checkout, raising vQPUs and sizing them.
::::

::::{tab-item} NetQASM / SquidASM
:sync: netqasm

Low-level quantum-network simulation. NetSquid and SquidASM come from a
private index that needs a free account; see the
[NetQASM installation docs](https://netqasm.readthedocs.io/en/stable/installation.html).

```bash
conda create -n netqasm2 python=3.11 pip -y
conda activate netqasm2
export PIP_EXTRA_INDEX_URL='https://<user>:<url-encoded-pwd>@pypi.netsquid.org'
pip install "squidasm>=0.13" "netqasm>=2,<3"
```

This pulls in `netsquid` and `netqasm` **2.x** — used by `--netqasm`. The
verified set is pinned in `environments/netqasm2-requirements.txt`.

For the legacy `--netqasm1.0` path, build a second environment on Python 3.8
with `netqasm` **1.x** instead; 2.x requires Python 3.9 or newer.

:::{tip}
Pin `squidasm>=0.13`. PyPI carries a placeholder package of the same name at
`0.0.1` with no dependencies, and pip installs that in preference to the real
one unless a version floor sends it to the private index.
:::
::::

::::{tab-item} Qiskit Aer
:sync: aer

Shot-based circuit simulation. The lightest option, and the only one with no
special installation requirements.

```bash
pip install qiskit qiskit-aer
```
::::

::::{tab-item} Qoala
:sync: qoala

Quantum-internet node execution environment, with task scheduling and
multitasking. **Simulation only.**

```bash
conda create -n qoala python=3.11 -y
conda activate qoala
pip install netsquid --extra-index-url https://<user>:<pwd>@pypi.netsquid.org
pip install qoala   --extra-index-url https://<user>:<pwd>@pypi.netsquid.org
pip install netqmpi
```

Qoala requires Python 3.10–3.12 and `netqasm` **2.x**.
::::

:::::

## Two things that will bite you

:::{admonition} NetSquid needs an account
:class: important

The NetQASM and Qoala backends both depend on [NetSquid](https://netsquid.org),
which is distributed from a private package index and requires a free account:

```bash
pip install netsquid --extra-index-url https://<user>:<pwd>@pypi.netsquid.org
```

Substitute your NetSquid credentials for `<user>` and `<pwd>`.
:::

:::{admonition} Three environments, and what actually separates them
:class: warning

| Environment | Holds | Serves |
|---|---|---|
| `netqasm2` | SquidASM + `netqasm` 2.x + `netsquid-magic` 16.x, Python ≥ 3.9 | `--netqasm` |
| `squidasm` | SquidASM + `netqasm` 1.x, Python 3.8 | `--netqasm1.0` |
| `qoala` | qoala-sim + `netqasm` 2.3 + `netsquid-magic` 14.x | `--qoala` |

SquidASM and Qoala cannot share an environment, but **not** because of the
NetQASM version — both run on 2.x. They need incompatible majors of
`netsquid-magic` (16.x against 14.x), and installing one over the other leaves
the displaced backend unable to build a link layer.

SquidASM also caps at NetQASM **2.0.0** (`squidasm` 0.13.6 declares
`netqasm<=2.0.0`), while qoala-sim builds on 2.3. And NetQASM 2.x needs Python
3.9 or newer, which is what keeps the legacy 1.x environment on 3.8.

CUNQA and Aer have no such constraints and can share an environment with any of
them.
:::

## Verifying the installation

```bash
netqmpi --help
```

The CLI itself imports no backend, so this works as soon as the core package is
installed. To check a backend end to end, run one of the shipped examples:

```bash
netqmpi -n 2 examples/1_send_recv.py --aer --shots 1024
```

## Python version

`setup.py` declares `python_requires='>=3.6'`, but the effective floor is set by
the backend you choose:

| Backend | Effective Python requirement |
|---|---|
| Qiskit Aer | Whatever `qiskit` supports |
| NetQASM / SquidASM | Set by the `netsquid` / `squidasm` wheels |
| CUNQA | Set by the cluster's CUNQA build |
| Qoala | **3.10 – 3.12** |
