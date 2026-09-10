# Architecture

Earlier versions of NetQMPI were implemented directly on top of the NetQASM SDK,
which tied programs to a single execution stack. NetQMPI has since been
restructured into a **decoupled architecture** that strictly separates *what* a
distributed quantum program does from *how* and *where* it runs.

```{image} ../_static/netqmpi-architecture-overview.png
:alt: NetQMPI architecture overview
:width: 800px
:align: center
```

## Two layers, one boundary

### SDK — user-facing

Backend-agnostic abstractions. Application code depends on **these and nothing
else**.

{class}`~netqmpi.sdk.environment.Environment`
: The local node context, and a factory for circuits. Injected into every
  `main()`.

{class}`~netqmpi.sdk.circuit.Circuit`
: A fluent gate + `qsend`/`qrecv` API that **records** operations into an
  {class}`~netqmpi.sdk.operations.container.OperationContainer` rather than
  executing them.

{class}`~netqmpi.sdk.communicator.QMPICommunicator`
: Rank/size and the communication primitives, plus the context manager that
  delimits the distributed program.

### Runtime — execution-facing

Selects and drives a concrete backend through the **Adapter pattern + dependency
injection**.

{class}`~netqmpi.runtime.executor.Executor`
: Bootstraps the processes, discovers resources, and injects a backend-specific
  communicator into each node's `Environment`.

`CircuitAdapter` (a `Circuit` subclass)
: Translates the recorded operations into native backend instructions.

`Communicator` (a `QMPICommunicator` subclass)
: Maps ranks and communication onto the platform's resources, and triggers
  execution on context exit.

Because the boundary is strict, **the same `app.py` runs on any backend by
switching a flag** — no changes to application logic.

## The lifecycle of a run

```{mermaid}
sequenceDiagram
    autonumber
    participant CLI as netqmpi CLI
    participant Ex as Executor
    participant App as your main()
    participant Ci as Circuit
    participant Ad as CircuitAdapter
    participant B as Backend

    CLI->>Ex: instantiate (flag + RunConfig)
    CLI->>Ex: build_apps(script, size)
    Ex->>Ex: load_main(script), acquire resources
    Ex->>App: one Environment per rank
    loop per rank
        App->>Ci: env.create_circuit(...)
        App->>Ci: gates and primitives (recorded)
        App->>Ex: leaves `with comm:`
    end
    Note over Ex,B: the last rank triggers it
    Ex->>Ad: translate recorded operations
    Ad->>B: native instructions
    B-->>App: counts in comm.results
```

## Design patterns in play

**Command** — {class}`~netqmpi.sdk.operations.operation.Operation`
: Each operation encapsulates everything needed to describe one quantum action,
  independently of any backend. That is what makes recording-then-translating
  possible.

**Composite** — {class}`~netqmpi.sdk.operations.container.OperationContainer`
: Holds both leaf operations and nested containers, so a block that means more
  than its parts — a `QScatter` expanding into individual transfers — survives as
  the block it is. `flatten()` walks the leaves; `children` preserves the
  nesting.

**Adapter + dependency injection** — the whole Runtime layer
: The SDK never names a backend. The executor injects a concrete communicator
  into the `Environment`, and `create_circuit` returns a concrete adapter behind
  the abstract `Circuit` interface.

**Object pool** — {class}`~netqmpi.sdk.resources.IndexPool`
: Communication qubits and protocol classical bits are borrowed and returned, so
  the footprint is the maximum held *simultaneously* rather than the total ever
  acquired.

## How collectives stay consistent without communication

The ranks trace independently — there is no inter-rank communication at trace
time — yet each has to name a collective exactly as its peers do, so the sides of
a call can be paired at translation time.

They manage it by deriving the name from data every participant already knows:
the kind of call, the ranks involved, and a per-key counter.

```python
tag = self._next_tag("teledata", (self._comm.rank, dest_rank))
# -> "teledata_0_1_0", then "teledata_0_1_1", ...
```

Since every rank increments the same counter for the same key in the same order,
the *n*-th `qsend` from rank 0 to rank 1 gets the same tag on both sides. The
same mechanism names expose windows, with the group `(root, *receivers)` as the
key. This is why collectives must be reached **in the same order** on every rank:
the counters would otherwise drift apart.

## Resource allocation

Both pools hand out indices from a free list first and only then from a fresh
counter:

```python
pool = IndexPool()
a = pool.acquire(2)   # [0, 1]
pool.release(a)
b = pool.acquire(1)   # [0]  -- reused
pool.size             # 2    -- high-water mark
```

A teledata block acquires one communication qubit and two protocol classical bits
and releases them immediately, so sequential transfers reuse one slot. An expose
window holds its slot until the matching `unexpose`, so nested windows stack.

The backend reads
{attr}`~netqmpi.sdk.circuit.Circuit.num_comm_qubits` and
{attr}`~netqmpi.sdk.circuit.Circuit.num_protocol_clbits` at translation time,
when the trace is complete, and reserves exactly that much.

## Where the abstraction leaks

Three places, all documented rather than hidden:

1. **`comm.results` differs per backend** — keyed by rank on CUNQA, a single
   global histogram on Aer, this rank's own on Qoala and NetQASM. See
   {ref}`the comparison <comm-results-shape>`.
2. **Collectives that a backend expands jointly** need the circuits of all ranks
   at once. `CircuitAdapter` therefore allows a group-translation entry point
   alongside the per-operation hooks — CUNQA's `translate_group`.
3. **Not every backend implements every operation.** The SDK accepts everything
   at trace time; unsupported operations surface at translation time. See the
   [support matrix](../guide/circuits.md#backend-support-matrix).
