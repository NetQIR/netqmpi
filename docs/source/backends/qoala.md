# Qoala backend

```bash
netqmpi -n <N> app.py --qoala [--shots N] [--config qoala.yaml]
```

Qoala models the **software/hardware architecture of a quantum-internet node**:
task scheduling, multitasking between programs, and a configurable qdevice, all
on top of NetSquid. Where NetQASM simulates the network, Qoala simulates what
happens *inside a node* while the network runs.

:::{admonition} Simulation only
:class: important

Qoala is a NetSquid-based simulator. This backend has no real-hardware execution
path and must not be considered on par with a physical deployment.
:::

- **Packages:** `qoala`, `netsquid`, `netqasm` **2.x**
- **Python:** 3.10 – 3.12
- **Adapter:** {mod}`netqmpi.runtime.adapters.qoala`

## Installation

Qoala needs its **own environment**, because its `netqasm` 2.x is incompatible
with the 1.x used by the [NetQASM backend](netqasm.md):

```bash
conda create -n qoala python=3.11 -y
conda activate qoala
pip install netsquid --extra-index-url https://<user>:<pwd>@pypi.netsquid.org
pip install qoala   --extra-index-url https://<user>:<pwd>@pypi.netsquid.org
pip install netqmpi
```

## What it supports

| Primitive | Status |
|---|---|
| `qsend` / `qrecv` | ✅ full teleportation (EPR + BSM + corrections) |
| `qscatter` / `qgather` | ❌ `NotImplementedError` |
| `expose` / `unexpose` | ❌ `NotImplementedError` |
| `reset` | ❌ `NotImplementedError` |
| `barrier` | ❌ `NotImplementedError` |
| Classically controlled gates | ❌ `NotImplementedError` |
| Several circuits per rank | ❌ exactly one circuit per rank |

Gates: `H` `X` `Y` `Z` directly; `S` `SDG` `T` `TDG` as fixed `rot_z` rotations;
`RX` `RY` `RZ` **discretised to multiples of π/16**.

:::{caution}
Controlled gates are effectively unavailable. The adapter maps controlled-`RX`
to `cnot` and controlled-`RZ` to `cphase`, but `cx()` records a controlled-`X`
and `cz()` a controlled-`Z`, so neither matches and both raise. `crz()` does map
to `cphase`, but discards its angle.
:::

More than one circuit per rank is rejected explicitly:

```text
NotImplementedError: The Qoala backend currently supports exactly one circuit
per rank (rank 0 created 2).
```

## Configuration

This is the backend with the richest noise model — it is the reason to choose it.

```yaml
# qoala.yaml
shots: 1000
seed: 7
qoala:
  link_fidelity: 0.8          # EPR-pair fidelity in [0.25, 1.0]
  link_duration: 1000.0       # EPR generation time (ns)
  qnos_instr_time: 1000.0     # one quantum-processor instruction (ns)
  hardware:                   # qdevice noise model; omit for a perfect device
    t1: 0
    t2: 0
    single_qubit_gate_time: 5e3
    two_qubit_gate_time: 200e3
    single_qubit_gate_depolar_prob: 0.1
    two_qubit_gate_depolar_prob: 0.0
```

`link_fidelity` below 1.0 uses a depolarising link with
`prob_max_mixed = (4/3)(1 - link_fidelity)`; a value outside `[0.25, 1.0]` is
rejected when the config is built. `t1 == t2 == 0` means "no memory noise", so
the defaults describe a **perfect** qdevice.

The full field list is in
[the qoala block](../guide/configuration.md#qoala-block).

:::{note}
Depolarising noise applies to the **gates** only. `INSTR_INIT` and
`INSTR_MEASURE` carry their own duration but no depolarising error, so there is
no separate readout-flip model.
:::

## Execution model

The adapter compiles each rank's circuit to **`.iqoala` program text** rather
than driving an API. Local gates and measurements are buffered and flushed into
`QL` blocks; a `qsend` or `qrecv` closes the current block and emits the
teleportation blocks around it.

All ranks run in the same process, and the joint NetSquid simulation is deferred
until every rank has left its `with comm:` block. Each rank compiles and
registers its program on `__exit__`; when the last rank registers, the executor
builds the Qoala network and runs one simulation for all ranks at once.

The communicator module imports **no** `qoala` package: program text is produced
by `QoalaCircuitAdapter` and the simulation is driven by `QoalaExecutorAdapter`,
which imports `qoala` and `netsquid` lazily inside its own methods.

## Results

`comm.results` is a `{bitstring: count}` histogram for **this rank alone**,
ordered by ascending classical-bit index. A rank that returns no measurements
gets an empty dict.

## Validation experiments

The repository ships three experiments validating this backend, under
[`scripts/experiments/`](https://github.com/NetQIR/net-qmpi/tree/main/scripts/experiments):

**Hardware-parameter propagation**
: Checks that qdevice noise (T1/T2, gate times, depolarisation) is propagated
  faithfully through NetQMPI's translation, by comparing against a hand-written
  native Qoala control program with the same qdevice and a perfect link.

**EPR fidelity sweep**
: With a perfect qdevice, sweeps the EPR-pair fidelity of `qsend`/`qrecv` and
  compares against the perfect link.

**Scheduling / multitasking**
: Two NetQMPI programs sharing a node. Qoala's scheduler reduces the makespan by
  18.9% without degrading fidelity, with a CPS/QPS Gantt chart of the
  interleaving.

The fidelity probe used throughout is a distributed superposition teleported and
read out **in the X basis**: measuring `|+⟩` directly in Z gives 50/50 even
without noise, so it is useless as a fidelity measure. Measuring in the state's
own basis makes fidelity run from 1.0 (noiseless) to 0.5 (fully depolarised),
sensitive to T1, T2 and depolarisation.

## API

- {class}`~netqmpi.runtime.adapters.qoala.qoala_executor.QoalaExecutorAdapter`
- {class}`~netqmpi.runtime.adapters.qoala.qoala_executor.QoalaRunConfig`
- {class}`~netqmpi.runtime.adapters.qoala.qoala_executor.QoalaQDeviceConfig`
- {class}`~netqmpi.runtime.adapters.qoala.qoala_circuit.QoalaCircuitAdapter`
- {class}`~netqmpi.runtime.adapters.qoala.qoala_communicator.QoalaCommunicator`
