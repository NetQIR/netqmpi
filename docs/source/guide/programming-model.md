# The programming model

## Anatomy of a NetQMPI application

```python
from netqmpi.sdk.environment import Environment     # 1. the only import you need


def main(env: Environment = None):                  # 2. the entry point
    comm = env.comm
    rank, size = comm.rank, comm.size               # 3. who am I

    with comm:                                      # 4. the distributed block
        circuit = env.create_circuit(2, 2)          # 5. circuits
        ...                                         #    gates and primitives

    if rank == size - 1:                            # 6. results, after the block
        print(comm.counts)
```

### 1. Imports

Application code imports from `netqmpi.sdk` and nothing else. No backend package
appears anywhere, which is what makes the same file run on every backend. If you
find yourself importing `qiskit`, `netqasm` or `cunqa` in an application, the
program has stopped being portable.

### 2. The entry point

NetQMPI loads your script with `runpy` and looks for a callable named `main`. A
script without one is rejected before anything is built:

```text
ValueError: app.py does not define a main() function
```

The `env: Environment = None` default is a convention: it documents the injected
type and lets the file be imported without running.

### 3. Rank and size

The same code runs on every node; `rank` is what makes each behave differently.
This is the SPMD paradigm inherited from classical MPI.

### 4. The `with comm:` block

Inside the block, circuit calls are **traced**, not executed. Each call appends
an {class}`~netqmpi.sdk.operations.operation.Operation` to the circuit's
{class}`~netqmpi.sdk.operations.container.OperationContainer`. The container is
a Composite: nested blocks (a `qscatter`, say) are sub-containers holding the
individual transfers they expand into.

When the block exits, the communicator's `__exit__` takes over: the operations
are translated into the backend's native instructions and executed. The ranks
rendezvous there — circuits are submitted jointly once the last rank has left.

### 5. Circuits

{meth}`~netqmpi.sdk.environment.Environment.create_circuit` returns a
backend-specific subclass of {class}`~netqmpi.sdk.circuit.Circuit` behind a
uniform interface, and registers it with the communicator so the runtime can
find it at translation time. Most ranks create exactly one circuit; creating
several is supported by the CUNQA backend, where the *i*-th circuit of every
rank forms one distributed program.

:::{admonition} Every rank must create the same number of circuits
:class: warning

Circuits are paired across ranks by creation order. If the ranks disagree, the
run fails at translation time:

```text
RuntimeError: Every rank must create the same number of circuits so that they
can be paired into distributed programs, got {0: 2, 1: 1}.
```

The Qoala backend currently supports exactly **one** circuit per rank.
:::

(reading-results)=
### 6. Reading results

`comm.counts` is populated when the joint execution finishes, and has the
same shape on every backend: every rank's histogram, keyed by rank.

```python
{0: {'0': 1024}, 1: {'0': 517, '1': 507}}
```

A rank that built several circuits gets a list of histograms, in creation
order. The run ends when the *last* rank leaves its `with comm:` block, so that
rank always has `comm.counts`; whether the others do depends on whether the
backend runs ranks concurrently (Aer) or one after another (CUNQA, Qoala,
NetQASM). The portable idiom, used by every shipped example, prints from the
last rank:

```python
if rank == comm.size - 1:
    for other, counts in comm.counts.items():
        print(f"rank {other}: {counts}")
```

`comm.results` holds the same outcomes in **each backend's native shape**. Use
it when you need what only one backend offers, such as the correlations between
ranks in Aer's joint histogram:

| Backend | Shape of `comm.results` | Populated on |
|---|---|---|
| **CUNQA** | `{rank: {bitstring: count}}` — every rank's counts | every rank |
| **Aer** | `{bitstring: count}` for the *whole* global circuit (all ranks' classical bits concatenated) | every rank, identically |
| **Qoala** | `{bitstring: count}` for this rank alone | every rank |
| **NetQASM** | `{outcome: count}` for this rank's own measurements | the rank itself |

`comm.metrics`, next to it, holds what the run measured besides the outcomes,
grouped by who measured it. It is empty unless the backend fills it, which
today only the NetQASM backend does (see [Metrics](../backends/netqasm.md#metrics)).

## Tracing versus execution

Understanding *when* things happen explains most of NetQMPI's error messages.

```{mermaid}
sequenceDiagram
    participant U as Your main()
    participant C as Circuit (SDK)
    participant A as CircuitAdapter
    participant B as Backend

    Note over U,C: inside `with comm:`
    U->>C: circuit.h(0)
    C->>C: record Gate('H', [0])
    U->>C: comm.qsend(circuit, [0], 1)
    C->>C: record QSend + reserve comm qubit/clbits

    Note over U,B: on `with` exit — the last rank triggers it
    C->>A: translate(ops)
    A->>B: native instructions
    B-->>C: counts
```

There are three moments at which a program can fail, and they differ sharply in
how helpful the error is:

**Trace time**
: Raised at the offending line while your `main()` runs, so the traceback points
  into your own code. Out-of-range qubits, too few classical bits, a mismatched
  `unexpose`. These are the good ones.

**Translation time**
: Raised from `__exit__` of the *last* rank to leave its block, because that is
  when the ranks' circuits are translated together. The traceback anchors at
  `with comm:`, never at the call that caused it, and the rank it is attributed
  to need not be the guilty one. Deadlocked collectives and unmatched transfers
  land here.

**Run time**
: Raised inside the backend, in the backend's own vocabulary, with nothing tying
  it back to a line of your code.

[Troubleshooting](troubleshooting.md) catalogues what falls into each tier.

## Resource accounting

You size the data qubits and classical bits; the runtime sizes everything else.

```python
circuit = env.create_circuit(num_qubits=2, num_clbits=2)

circuit.num_qubits            # 2   — what you asked for
circuit.num_clbits            # 2   — what you asked for
circuit.num_comm_qubits       # communication qubits the protocols need
circuit.num_protocol_clbits   # classical bits the corrections need
```

The last two are **high-water marks**: the maximum number held *at the same
time*, not the total ever used. A `qsend` borrows one communication qubit and
two protocol classical bits and gives them straight back, so ten sequential
`qsend`s still cost one communication qubit — and so does a whole `qscatter`.
An `expose` window, by contrast, holds its slot until the matching `unexpose`,
so overlapping windows do add up.

Both values are only final once the circuit has been fully traced, which is why
backends read them at translation time. The allocator behind them is
{class}`~netqmpi.sdk.resources.IndexPool`.
