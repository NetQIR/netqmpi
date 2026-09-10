# Core concepts

Five ideas are enough to read and write any NetQMPI program.

## Ranks and the SPMD model

You write **one** program. NetQMPI runs it on *N* quantum nodes, and each
instance — a **rank** — distinguishes itself by its number:

```python
rank = comm.rank    # 0 .. size-1, this node's index
size = comm.size    # how many nodes are in the run
```

`-n 3` on the command line means three ranks, numbered 0, 1 and 2. Branching on
`rank` is how a single file expresses different behaviour per node, exactly as in
classical MPI.

For ring topologies, the communicator provides cyclic neighbour helpers:

```python
comm.get_next_rank(rank)    # (rank + 1) % size
comm.get_prev_rank(rank)    # (rank - 1) % size
```

## The environment

Every `main()` receives an {class}`~netqmpi.sdk.environment.Environment`. It is
the *only* object connecting your code to the runtime, and it has exactly two
jobs:

```python
def main(env: Environment = None):
    comm = env.comm                                       # communication
    circuit = env.create_circuit(num_qubits=2, num_clbits=2)  # circuit creation
```

{attr}`~netqmpi.sdk.environment.Environment.comm` is this rank's
{class}`~netqmpi.sdk.communicator.QMPICommunicator`, and
{meth}`~netqmpi.sdk.environment.Environment.create_circuit` is a factory that
returns a **backend-specific** circuit behind a backend-agnostic interface. That
indirection is the whole reason your application never imports a backend.

## The communicator block

```python
with comm:
    # everything here is part of the distributed program
```

The `with comm:` block delimits the distributed program. Circuit operations
recorded inside it are **traced, not executed**: they accumulate in an operation
container. When the block exits, the backend takes over — it translates the
recorded operations into its native instructions and runs them.

This is why results are not available until *after* the block, and why the ranks
synchronise there: the circuits of all ranks are submitted together when the last
rank leaves its block.

:::{admonition} Collectives must be reached by every rank
:class: warning

Because tracing is per-rank but execution is joint, a collective —
`qscatter`, `qgather`, `expose`, `unexpose` — that only some ranks reach is a
deadlock. NetQMPI detects the common cases at translation time and reports which
rank is missing; see [Troubleshooting](../guide/troubleshooting.md).
:::

## Circuits, qubits and classical bits

A circuit is created with a number of **data qubits** and **classical bits**:

```python
circuit = env.create_circuit(num_qubits=2, num_clbits=2)
circuit.h(0).cx(0, 1).measure(0, 0)     # fluent: each call returns the circuit
```

Under the hood there are two more resource pools you do not size yourself:

**Communication qubits**
: Reserved by the runtime for distributed protocols. They are addressed *right
  after* the data qubits, and are only ever handed to you by
  {meth}`~netqmpi.sdk.circuit.Circuit.expose`, which returns an index you can
  pass to any gate exactly like a local qubit.

**Protocol classical bits**
: Carry the correction outcomes of teledata/telegate. They are *additional* to
  the `num_clbits` you requested, so a protocol never clobbers your own
  measurements.

Both are borrowed when a protocol block opens and given back when it closes, so
non-overlapping protocols reuse the same physical resource. The allocator is
{class}`~netqmpi.sdk.resources.IndexPool`, and the totals a backend has to
provide are exposed as
{attr}`~netqmpi.sdk.circuit.Circuit.num_comm_qubits` and
{attr}`~netqmpi.sdk.circuit.Circuit.num_protocol_clbits`.

## Qubits move, they do not copy

This is the one place where the MPI analogy breaks, and it explains most
surprising results. The no-cloning theorem means a quantum `send` is a **move**:

```python
circuit.h(0)                        # rank 0 prepares |+>
comm.qsend(circuit, [0], 1)         # ... and no longer has it
                                    # rank 0's qubit 0 is back in |0>
```

The consequences show up throughout the API:

- After a `qsend`, the sender's qubit is in `|0⟩`.
- After a `qscatter`, the **root keeps nothing** — unlike `MPI_Scatter`, its
  buffer is split among the *other* ranks only.
- After a `qgather`, the contributors are left with `|0⟩`.
- The qubits a transfer lands on must already be in `|0⟩`; whatever they held is
  destroyed, not saved.

The exception is {meth}`~netqmpi.sdk.circuit.Circuit.expose`, which *lends* a
qubit instead of moving it: the root keeps its state and gets it back untouched
when the window closes. See [Communication primitives](../guide/communication.md).
