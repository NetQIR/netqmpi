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

::::{tab-set}

:::{tab-item} CUNQA
:sync: cunqa

HPC emulation of DQC through virtual QPUs. This is the **reference backend** —
the only one implementing every communication primitive.

Install and configure [CUNQA](https://github.com/CESGA-Quantum-Spain/cunqa) on your HPC
cluster (it provisions vQPUs via the job scheduler), then install `netqmpi` in
the same environment. CUNQA needs a working SLURM allocation, so it cannot be
installed on a laptop.

See the [CUNQA backend page](../backends/cunqa.md) for how to raise vQPUs and
size them.
:::

:::{tab-item} NetQASM / SquidASM
:sync: netqasm

Low-level quantum-network simulation.

```bash
pip install squidasm --extra-index-url https://<user>:<pwd>@pypi.netsquid.org
```

This pulls in `netsquid` and `netqasm` **1.x**. See the
[NetQASM installation docs](https://netqasm.readthedocs.io/en/stable/installation.html).
:::

:::{tab-item} Qiskit Aer
:sync: aer

Shot-based circuit simulation. The lightest option, and the only one with no
special installation requirements.

```bash
pip install qiskit qiskit-aer
```
:::

:::{tab-item} Qoala
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
:::

::::

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

:::{admonition} NetQASM 1.x and 2.x cannot coexist
:class: warning

The NetQASM/SquidASM backend uses `netqasm` **1.x**; Qoala uses `netqasm`
**2.x**. The two are mutually incompatible, so the `--netqasm` and `--qoala`
backends must live in **separate environments** (two conda envs, for instance).

CUNQA and Aer have no such constraint and can share an environment with either.
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
