# NetQMPI

**NetQMPI** brings the classical **MPI** (Message Passing Interface) programming
model to **Distributed Quantum Computing (DQC)**. Following a Single-Program,
Multiple-Data (SPMD) paradigm, you write *one* script that runs across *N*
quantum nodes and coordinates them through message-passing primitives —
including quantum-aware ones such as {meth}`~netqmpi.sdk.communicator.QMPICommunicator.qsend`,
{meth}`~netqmpi.sdk.communicator.QMPICommunicator.qrecv` and quantum
collectives — without manually orchestrating entanglement, teleportation or
classical messaging.

NetQMPI is **backend-agnostic**: the same, unmodified application runs on a
quantum-network simulator, an HPC emulator or a circuit simulator, simply by
selecting a backend on the command line.

```python
from netqmpi.sdk.environment import Environment

def main(env: Environment = None):
    comm = env.comm

    with comm:
        circuit = env.create_circuit(num_qubits=1, num_clbits=1)

        if comm.rank == 0:
            circuit.h(0)                                    # prepare |+>
            comm.qsend(circuit, [0], comm.get_next_rank(0))  # and give it away
        else:
            comm.qrecv(circuit, [0], comm.get_prev_rank(comm.rank))
            circuit.measure(0, 0)
```

```bash
netqmpi -n 2 app.py --cunqa      # HPC vQPU emulation      (reference backend)
netqmpi -n 2 app.py --netqasm    # quantum-network simulation
netqmpi -n 2 app.py --aer        # circuit simulation
netqmpi -n 2 app.py --qoala      # node execution environment
```

[CUNQA](https://github.com/CESGA-Quantum-Spain/cunqa) is the **reference backend**: it is the only one that implements the whole
primitive set — point-to-point transfers, the rooted collectives and the telegate
window — and it is what the shipped examples are written against.

---

## Where to start

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} {octicon}`rocket` Installation
:link: getting-started/installation
:link-type: doc

Install the core package and the backend you intend to use.
:::

:::{grid-item-card} {octicon}`zap` Quick start
:link: getting-started/quickstart
:link-type: doc

Write, run and read the results of your first distributed program.
:::

:::{grid-item-card} {octicon}`book` Core concepts
:link: getting-started/concepts
:link-type: doc

Ranks, the communicator block, circuits, and how a program is executed.
:::

:::{grid-item-card} {octicon}`server` Backends
:link: backends/index
:link-type: doc

What each backend targets, what it supports, and how to configure it.
:::

:::{grid-item-card} {octicon}`arrow-switch` Communication primitives
:link: guide/communication
:link-type: doc

`qsend`/`qrecv`, `qscatter`/`qgather` and `expose`/`unexpose` in detail.
:::

:::{grid-item-card} {octicon}`code` API reference
:link: api/index
:link-type: doc

Every public class, method and configuration field.
:::

::::

---

## Documentation contents

```{toctree}
:maxdepth: 2
:caption: Getting started

getting-started/installation
getting-started/quickstart
getting-started/concepts
```

```{toctree}
:maxdepth: 2
:caption: User guide

guide/programming-model
guide/circuits
guide/communication
guide/cli
guide/configuration
guide/troubleshooting
```

```{toctree}
:maxdepth: 2
:caption: Backends

backends/index
backends/cunqa
backends/netqasm
backends/aer
backends/qoala
```

```{toctree}
:maxdepth: 2
:caption: Development

development/architecture
development/writing-a-backend
```

```{toctree}
:maxdepth: 2
:caption: Reference

api/index
```

---

## Citing NetQMPI

If you use NetQMPI in your research, please cite:

> **NetQMPI: An MPI-Inspired Library for Programming Distributed Quantum
> Applications Over Quantum Networks Using NetQASM SDK.**
> F. Javier Cardama, Jorge Vázquez-Pérez, Tomás F. Pena, Andrés Gómez.
> *IEEE Access*, Vol. 14, 2026, pp. 125459-125475.
> DOI: [10.1109/ACCESS.2026.3723566](https://doi.org/10.1109/ACCESS.2026.3723566)

> **Emulating NetQMPI applications with CUNQA: A Decoupled Architecture for HPC
> Environments.**
> Jorge Vázquez-Pérez, F. Javier Cardama, Tomás F. Pena, Andrés Gómez.
> *Proceedings of the IEEE International Conference on Distributed Computer
> Systems (ICDCS 2026).*
> DOI: [10.1109/ICDCSW72724.2026.00045](https://doi.org/10.1109/ICDCSW72724.2026.00045)


> **NetQIR: An Extension of QIR for Distributed Quantum Computing.**
> F. Javier Cardama, Jorge Vázquez-Pérez, C. Piñeiro, T. F. Pena, J. C. Pichel,
> Andrés Gómez.
> *Future Generation Computer Systems*, Vol. 174, 2026, Article 107989.
> DOI: [10.1016/j.future.2025.107989](https://doi.org/10.1016/j.future.2025.107989)
