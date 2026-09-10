# Quick start

This page takes you from an empty file to a running distributed quantum program.

## 1. Write the program

A NetQMPI application is a plain Python file that defines a **`main(env)`**
function. The same file runs on every rank — this is the SPMD model — and each
rank tells itself apart by its `rank` number.

```{code-block} python
:caption: app.py
:linenos:

from netqmpi.sdk.environment import Environment


def main(env: Environment = None):
    comm = env.comm
    rank = comm.rank

    next_rank = comm.get_next_rank(rank)
    previous_rank = comm.get_prev_rank(rank)

    # Only what is inside the block takes part in the distributed program.
    with comm:
        circuit = env.create_circuit(num_qubits=1, num_clbits=1)

        if rank == 0:
            circuit.h(0)                          # prepare |+>
            comm.qsend(circuit, [0], next_rank)   # and give it away
        else:
            comm.qrecv(circuit, [0], previous_rank)
            circuit.measure(0, 0)

    # The circuits of all ranks are submitted together when the last rank
    # leaves the block, so only that rank sees results here — and it sees
    # every rank's counts, keyed by rank.
    if comm.results:
        for other, counts in comm.results.items():
            print(f"rank_{other}: {counts}")
```

Three things are worth noticing:

`main(env)` is the entry point
: NetQMPI loads your script and calls this function once per rank, injecting an
  {class}`~netqmpi.sdk.environment.Environment`. A script without `main()` is
  rejected before anything runs.

Nothing mentions entanglement
: {meth}`~netqmpi.sdk.communicator.QMPICommunicator.qsend` and
  {meth}`~netqmpi.sdk.communicator.QMPICommunicator.qrecv` are all you write.
  The EPR-pair generation, the Bell measurement and the classical corrections
  that make up the teleportation protocol are filled in by the backend adapter.

The program is backend-agnostic
: It imports only from `netqmpi.sdk`. No backend package appears anywhere, which
  is what lets the same file run on all four backends.

## 2. Run it

The launcher is MPI-like: pick the number of nodes with `-n` and the backend
with a flag.

```bash
netqmpi -n 2 app.py --aer --shots 1024      # circuit simulation
netqmpi -n 2 app.py --netqasm               # quantum-network simulation
netqmpi -n 2 app.py --cunqa --shots 1024    # HPC vQPU emulation
netqmpi -n 2 app.py --qoala --shots 100     # node execution environment
```

Start with `--aer` if you just want to see it work: it is the only backend with
no special installation requirements.

## 3. Read the results

Output is a **histogram of measurement outcomes per rank**, gathered after all
ranks have left the `with comm:` block:

```text
rank_0: {'0': 1024}
rank_1: {'0': 517, '1': 507}
```

Rank 1 measures the `|+⟩` it received in the computational basis, so it reads 0
and 1 about equally often. Rank 0 reads `0` because handing a qubit over
**moves** it: a quantum state cannot be copied, so once `qsend` returns, rank 0's
qubit is back in `|0⟩`.

:::{admonition} Only some ranks see `comm.results`
:class: note

Circuits are submitted jointly when the *last* rank leaves its `with comm:`
block, so `comm.results` is populated on that rank and — depending on the
backend — possibly not on the others. The `if comm.results:` guard above is the
idiomatic way to print once. See {ref}`Reading results <reading-results>`.
:::

## What to read next

- [Core concepts](concepts.md) — ranks, the communicator block, and the
  execution model.
- [Communication primitives](../guide/communication.md) — moving qubits with
  `qscatter`/`qgather`, sharing a control with `expose`/`unexpose`.
- [Backends](../backends/index.md) — which one to use, and what each supports.

## Shipped examples

The repository ships runnable programs under
[`examples/`](https://github.com/NetQIR/net-qmpi/tree/main/examples):

| Example | What it shows |
|---|---|
| `1_send_recv.py` | Distributed superposition, teleported between two ranks |
| `2_round_robin.py` | A qubit passed around a ring of ranks |
| `3_scatter.py` | `qscatter`: the root hands one qubit to each other rank |
| `4_gather.py` | `qgather`: every rank hands its qubit to the root |
| `5_qft_expose.py` | A full 3-rank QFT built out of telegates |

There is also
[`examples/frequent_errors/`](https://github.com/NetQIR/net-qmpi/tree/main/examples/frequent_errors),
a corpus of programs that are *meant* to fail — one failure mode each — with a
`run_all.py` that reports what NetQMPI says about every one of them without
needing any vQPU. It is documented in [Troubleshooting](../guide/troubleshooting.md).
