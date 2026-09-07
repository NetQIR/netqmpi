# Qoala backend adapter (simulation only)

This package implements the NetQMPI backend contract on top of
[qoala-sim](https://github.com/QuTech-Delft/qoala-sim), the NetSquid-based
simulator for the Qoala execution-environment specification
(paper: arXiv:2502.17296).

> **Simulation only.** Qoala has no real-hardware execution path. This backend
> is a simulator and must not be considered on par with a physical quantum
> network deployment.

## Pieces

| File | Class | Role |
|---|---|---|
| `qoala_executor.py` | `QoalaExecutorAdapter`, `QoalaRunConfig` | Builds N `ProcNode` contexts in one NetSquid simulation and runs the joint app. Only module that imports `qoala.*`/`netsquid`. |
| `qoala_circuit.py` | `QoalaCircuitAdapter`, `QoalaProgramSpec` | Pure-Python compiler from NetQMPI operations to `.iqoala` program text. |
| `qoala_communicator.py` | `QoalaCommunicator` | Collects one program per rank and triggers the joint simulation once all ranks are ready. |

## Execution model

All ranks run in a **single Python process**; the whole N-node network is one
NetSquid discrete-event simulation (one `ProcNode` per rank), not N OS
processes. Each rank compiles its circuit to an `.iqoala` program; when the last
rank leaves its `with comm` block, the executor builds the network, submits one
batch of `shots` iterations per node, pairs remote PIDs for entanglement, runs
the simulation and returns a `{bitstring: count}` histogram per rank.

## Operation mapping (v1 scope)

Implemented: local gates (`h`, `x`, `y`, `z`, `s`, `t`, `rx`, `ry`, `rz`,
`cx`, `cz`), `measure`, and `qsend`/`qrecv` via teleportation
(EPR request + Bell-state measurement + classical corrections, mirroring the
Qoala `teleport` example).

Not implemented yet (raise `NotImplementedError`, as in the CUNQA adapter):
`qscatter`, `qgather`, `expose`, `unexpose`, `reset`, `barrier`,
classically-controlled gates.

See `docs/design/qoala-backend.md` for the full operation-to-operation mapping.

## Environment requirement (important)

Qoala requires **Python 3.10–3.12** and **`netqasm >= 2.0`**, which are
incompatible with the `squidasm`/NetQASM-1.0 environment used by the `--netqasm`
backend. This backend therefore lives in its own conda environment; `--netqasm`
and `--qoala` cannot run in the same interpreter.

Create the environment (needs a NetSquid account):

```sh
conda create -n qoala python=3.11 -y
conda run -n qoala pip install --extra-index-url https://<user>:<pwd>@pypi.netsquid.org netsquid
conda run -n qoala pip install -e /path/to/qoala-sim --extra-index-url https://<user>:<pwd>@pypi.netsquid.org
conda run -n qoala pip install -e /path/to/NetQMPI
```

## Run the distributed-superposition example

The same `app.py` runs unchanged on every backend:

```sh
conda run -n qoala   netqmpi -n 2 examples/1_send_recv.py --qoala --shots 20
conda run -n squidasm netqmpi -n 2 examples/1_send_recv.py --netqasm
```
