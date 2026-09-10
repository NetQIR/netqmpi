# Communication primitives

NetQMPI has three families of quantum communication primitive. All of them are
methods on the {class}`~netqmpi.sdk.communicator.QMPICommunicator` and take the
circuit as their first argument.

| Family | Calls | Semantics | MPI analogue |
|---|---|---|---|
| Point-to-point | `qsend` / `qrecv` | Moves qubits between two ranks | `MPI_Send` / `MPI_Recv` |
| Rooted collectives | `qscatter` / `qgather` | Moves qubits between a root and the rest | `MPI_Scatter` / `MPI_Gather` |
| Telegate window | `expose` / `unexpose` | *Lends* a control qubit, no move | `MPI_Bcast`, roughly |

:::{admonition} All the collectives are barriers
:class: warning

`qscatter`, `qgather`, `expose` and `unexpose` must be reached by **every**
participating rank, in the **same order**. A rank that skips one, or reaches
them in a different order, deadlocks the run — NetQMPI detects the common cases
at translation time and names the missing rank.
:::

## `qsend` / `qrecv` — moving a qubit

```python
comm.qsend(circuit, qubits, dest_rank)
comm.qrecv(circuit, qubits, src_rank)
```

The sender names the local qubits to hand over; the receiver names the local
qubits the incoming states land on. The backend fills in the teledata protocol
— EPR-pair generation, Bell measurement, classical corrections — behind those
two calls.

```python
with comm:
    circuit = env.create_circuit(num_qubits=1, num_clbits=1)

    if rank == 0:
        circuit.h(0)                            # prepare |+>
        comm.qsend(circuit, [0], 1)             # and give it away
    else:
        comm.qrecv(circuit, [0], 0)
        circuit.measure(0, 0)
```

Each qubit is transferred by its own protocol block, which borrows **one
communication qubit and two protocol classical bits** and returns them
immediately, so a sequence of transfers reuses the same slot.

:::{admonition} Sends and receives must pair up
:class: important

A `qsend` with no matching `qrecv` is caught at translation time on CUNQA:

```text
RuntimeError: rank 0 sends qubit 0 to rank 1, which never receives it:
rank 0 traced 1 qsend to rank 1, and rank 1 traced 0 qrecvs from rank 0
```

The qubits a transfer lands on must be in `|0⟩` when the call is reached.
Whatever they held is destroyed, not saved.
:::

## `qscatter` / `qgather` — moving a buffer

Both are **collective** (every rank calls them) and **rooted** (one rank passes
the whole buffer, the others pass their own chunk). Both return the local qubits
holding this rank's share.

### `qscatter`

```python
mine = comm.qscatter(circuit, qubits, root)
```

The root's buffer is split into one chunk per rank **other than the root**, in
rank order.

:::{admonition} This is where `qscatter` parts company with `MPI_Scatter`
:class: caution

Because qubits cannot be copied, the chunks are *moved*. The root keeps
**nothing**: scatter two qubits over two other ranks and the root ends
empty-handed, its slots back in `|0⟩`. The call returns an empty list on the
root for exactly that reason.
:::

```python
with comm:
    if rank == ROOT:
        # One qubit for each of the *other* ranks; the root gives them all away.
        circuit = env.create_circuit(num_qubits=size - 1, num_clbits=size - 1)
        for q in range(size - 1):
            circuit.x(q)                                    # |1> on every qubit
        comm.qscatter(circuit, list(range(size - 1)), root=ROOT)
        circuit.measure_all()                               # reads 0 everywhere
    else:
        circuit = env.create_circuit(num_qubits=1, num_clbits=1)
        mine = comm.qscatter(circuit, [0], root=ROOT)       # lands on qubit 0
        circuit.measure(mine[0], 0)                         # reads 1
```

### `qgather`

```python
whole = comm.qgather(circuit, qubits, root)
```

The mirror image: the root passes the whole buffer the chunks land on — with its
own contribution already sitting at position `root` — and every other rank
passes the qubits it contributes. The contributors are left with `|0⟩`.

```python
with comm:
    if rank == ROOT:
        circuit = env.create_circuit(num_qubits=size, num_clbits=size)
        circuit.x(ROOT)                                     # its own contribution
        comm.qgather(circuit, list(range(size)), root=ROOT)
        circuit.measure_all()                               # reads 1 everywhere
    else:
        circuit = env.create_circuit(num_qubits=1, num_clbits=1)
        circuit.x(0)
        comm.qgather(circuit, [0], root=ROOT)
        circuit.measure(0, 0)                               # reads 0
```

### Rules and errors

The root's buffer must split evenly among the holders — the other ranks for a
scatter, every rank for a gather — and only data qubits can be moved:

```text
ValueError: the root of a qscatter must hold one chunk per receiving rank:
            3 qubits do not split evenly among 2 ranks.
ValueError: qscatter root 7 is not a rank of the communicator [0, 3).
ValueError: qscatter needs at least one rank besides the root (0).
IndexError: Qubit index 3 is not a data qubit (expected [0, 2)).
```

A whole scatter costs the same resources as a single `qsend`: each transfer
borrows one communication qubit and two protocol classical bits and gives them
straight back.

## `expose` / `unexpose` — sharing a control

Moving a qubit is not always what a distributed algorithm needs. When several
ranks only want to apply gates *controlled* by a remote qubit — the crossing
rotations of a QFT, for instance — the qubit can stay where it is and be **lent**
to them through a shared GHZ state. This is a *telegate*.

```python
control = comm.expose(circuit, qubit, ranks, root=None)
comm.unexpose(circuit, ranks, root=None)
```

Both are collective across `[root] + ranks`. `root` names the rank lending the
qubit and defaults to the calling rank. The `qubit` argument is read on the root
only; the other participants may pass anything.

The call returns **the index each rank must use as control** — the root's own
data qubit on the root, a freshly reserved communication qubit on every
receiver — so the gate is written exactly like a local one:

```python
with comm:
    circuit = env.create_circuit(num_qubits=1, num_clbits=1)

    # Rank 1 lends its qubit 0 to rank 0, which drives a CS with it.
    control = comm.expose(circuit, 0, [0], root=1)

    if rank == 0:
        circuit.h(0)
        circuit.cs(control, 0)

    comm.unexpose(circuit, [0], root=1)   # the control goes back untouched
```

Unlike a transfer, the root **keeps** its state: it gets the qubit back untouched
when the window closes.

### Nested windows

Windows unwind like scopes — `unexpose` pairs with the innermost matching
`expose` for the same participant group. `5_qft_expose.py` builds a full 3-rank
QFT that way, with rank 2's window spanning rank 1's:

```python
control_2 = comm.expose(circuit, 0, [0, 1], root=2)   # outer window
control_1 = comm.expose(circuit, 0, [0], root=1)      # inner window

if rank == 0:
    circuit.h(0)
    circuit.cp(control_1, 0, np.pi/2)   # CS
    circuit.cp(control_2, 0, np.pi/4)   # CT

comm.unexpose(circuit, [0], root=1)     # closes the inner one

if rank == 1:
    circuit.h(0)
    circuit.cp(control_2, 0, np.pi/2)   # CS

comm.unexpose(circuit, [0, 1], root=2)  # closes the outer one
```

Resources are held for as long as the window is open, so **overlapping windows do
add up** — unlike transfers, which give their slot back immediately. Windows that
do not overlap reuse the same communication qubit.

### Rules and errors

```text
RuntimeError: rank 0 called unexpose(ranks=[0], root=1) without a matching open expose window.
IndexError:   Qubit index 1 is a communication qubit whose expose window is already closed.
ValueError:   expose needs at least one rank besides the root (0).
RuntimeError: rank 0 called expose(ranks=[7], root=0), which names rank 7, but this
              run has 2 ranks, numbered 0 to 1.
```

Using an exposed control after its window has closed is caught at trace time,
pointing straight at the offending line. A window that only some ranks open is
caught at translation time, as a deadlock report naming who is missing — see
[Troubleshooting](troubleshooting.md).

## Backend support

Only **CUNQA** implements all three families today.

| Primitive | CUNQA | NetQASM | Aer | Qoala |
|---|:--:|:--:|:--:|:--:|
| `qsend` / `qrecv` | ✅ | ✅ | ⚠️ SWAP-based | ✅ |
| `qscatter` / `qgather` | ✅ | ❌ | ❌ | ❌ |
| `expose` / `unexpose` | ✅ | ❌ | ❌ | ❌ |

On CUNQA the telegate window is expanded by
{func}`~netqmpi.runtime.adapters.cunqa.cunqa_circuit.translate_group`, because
CUNQA's `cat_entangler` / `cat_disentangler` helpers write into *every*
participating circuit in a single call. That is why the per-rank
`_translate_expose` hook of the CUNQA adapter deliberately raises instead of
emitting anything.
