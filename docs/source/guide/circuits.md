# Circuits and gates

A circuit is created through the environment, never constructed directly:

```python
circuit = env.create_circuit(num_qubits=2, num_clbits=2)
```

What comes back is a **backend-specific** subclass of
{class}`~netqmpi.sdk.circuit.Circuit` behind a uniform interface. Every gate
method returns the circuit, so calls chain:

```python
circuit.h(0).cx(0, 1).measure(0, 0).measure(1, 1)
```

## Qubit indexing

Indices span two ranges:

```text
 0 .. num_qubits-1                     data qubits   (the ones you asked for)
 num_qubits .. num_qubits+num_comm-1   communication qubits (runtime-owned)
```

Both are accepted by the gate API, so a control borrowed from a remote rank
through {meth}`~netqmpi.sdk.circuit.Circuit.expose` is used exactly like a local
qubit. You never compute a communication index yourself — `expose` returns it,
and {meth}`~netqmpi.sdk.circuit.Circuit.comm_qubit` converts a slot to an index
if you need it.

Indices are validated as you build:

```text
IndexError: Qubit index 5 out of range [0, 2) (2 data + 0 comm qubits).
IndexError: Qubit index 1 is a communication qubit whose expose window is already closed.
IndexError: Qubit index 3 is not a data qubit (expected [0, 2)).
```

The last one comes from operations that only accept data qubits: the rooted
collectives, and the qubit a root exposes. Communication qubits belong to the
protocols and are gone by the time the block is over, so they cannot be moved.

## Gate reference

### Single-qubit gates

| Method | Recorded as | Description |
|---|---|---|
| {meth}`~netqmpi.sdk.circuit.Circuit.h` | `Gate('H')` | Hadamard |
| {meth}`~netqmpi.sdk.circuit.Circuit.x` | `Gate('X')` | Pauli-X |
| {meth}`~netqmpi.sdk.circuit.Circuit.y` | `Gate('Y')` | Pauli-Y |
| {meth}`~netqmpi.sdk.circuit.Circuit.z` | `Gate('Z')` | Pauli-Z |
| {meth}`~netqmpi.sdk.circuit.Circuit.s` | `Gate('S')` | Phase, √Z |
| {meth}`~netqmpi.sdk.circuit.Circuit.sdg` | `Gate('SDG')` | S† |
| {meth}`~netqmpi.sdk.circuit.Circuit.t` | `Gate('T')` | π/8 phase |
| {meth}`~netqmpi.sdk.circuit.Circuit.tdg` | `Gate('TDG')` | T† |

### Parametric single-qubit gates

| Method | Recorded as | Description |
|---|---|---|
| {meth}`~netqmpi.sdk.circuit.Circuit.rx` `(theta, qubit)` | `Gate('RX', params=[theta])` | Rotation about X |
| {meth}`~netqmpi.sdk.circuit.Circuit.ry` `(theta, qubit)` | `Gate('RY', params=[theta])` | Rotation about Y |
| {meth}`~netqmpi.sdk.circuit.Circuit.rz` `(theta, qubit)` | `Gate('RZ', params=[theta])` | Rotation about Z |

Angles are in radians. Note the argument order: the angle comes **first**.

### Two- and three-qubit gates

| Method | Recorded as | Description |
|---|---|---|
| {meth}`~netqmpi.sdk.circuit.Circuit.cx` `(control, target)` | `ControlledGate([c], [Gate('X')])` | CNOT |
| {meth}`~netqmpi.sdk.circuit.Circuit.cz` `(control, target)` | `ControlledGate([c], [Gate('Z')])` | Controlled-Z |
| {meth}`~netqmpi.sdk.circuit.Circuit.cs` `(control, target)` | `ControlledGate([c], [Gate('S')])` | Controlled-S |
| {meth}`~netqmpi.sdk.circuit.Circuit.ct` `(control, target)` | `ControlledGate([c], [Gate('T')])` | Controlled-T |
| {meth}`~netqmpi.sdk.circuit.Circuit.cp` `(control, target, theta)` | `ControlledGate([c], [Gate('P', params=[theta])])` | Controlled phase |
| {meth}`~netqmpi.sdk.circuit.Circuit.crz` `(theta, control, target)` | `ControlledGate([c], [Gate('RZ', params=[theta])])` | Controlled-RZ |
| {meth}`~netqmpi.sdk.circuit.Circuit.swap` `(qubit1, qubit2)` | `Gate('SWAP')` | SWAP |
| {meth}`~netqmpi.sdk.circuit.Circuit.ccx` `(c1, c2, target)` | `ControlledGate([c1, c2], [Gate('X')])` | Toffoli |

`cp` generalises `cs` (θ = π/2) and `ct` (θ = π/4), which is what the crossing
rotations of a QFT are made of — see
[`5_qft_expose.py`](https://github.com/NetQIR/net-qmpi/blob/main/examples/5_qft_expose.py).

:::{note}
`cp` takes its angle **last** (`cp(control, target, theta)`) while `crz` takes it
**first** (`crz(theta, control, target)`), matching `rx`/`ry`/`rz`. The
inconsistency is in the API as it stands.
:::

### Non-unitary operations

| Method | Description |
|---|---|
| {meth}`~netqmpi.sdk.circuit.Circuit.measure` `(qubit, cbit)` | Measure one qubit into one classical bit |
| {meth}`~netqmpi.sdk.circuit.Circuit.measure_all` `()` | Measure qubit *i* into classical bit *i*, for every data qubit |
| {meth}`~netqmpi.sdk.circuit.Circuit.reset` `(qubit)` | Reset a qubit to `|0⟩` |
| {meth}`~netqmpi.sdk.circuit.Circuit.barrier` `(qubits=None)` | Insert a barrier; `None` means the whole circuit |

`measure_all` refuses to run if the circuit is short of classical bits:

```text
ValueError: Not enough classical bits to measure all qubits (1 clbits < 3 qubits).
```

## Backend support matrix

Not every backend implements every operation. The SDK accepts all of them at
trace time; a backend that cannot express one reports it at translation time.

Legend: **✅** supported · **❌** raises `NotImplementedError` ·
**⚠️** see the note · **∅** *silently ignored* — no instruction is emitted and
no error is raised.

| Operation | CUNQA | NetQASM | Aer | Qoala |
|---|:--:|:--:|:--:|:--:|
| `h` `x` `y` `z` `s` `t` | ✅ | ✅ | ✅ | ✅ |
| `sdg` `tdg` | ✅ | ❌ | ✅ | ✅ |
| `rx` `ry` `rz` | ✅ | ⚠️ [^nqrot] | ✅ | ⚠️ [^qrot] |
| `swap` | ✅ | ∅ [^nqswap] | ✅ | ❌ |
| `cx` `cz` | ✅ | ❌ [^nqctrl] | ✅ | ❌ [^qctrl] |
| `cs` `ct` `cp` | ✅ | ❌ [^nqctrl] | ∅ [^aerctrl] | ❌ [^qctrl] |
| `crz` | ✅ | ❌ [^nqctrl] | ✅ | ⚠️ [^qctrl] |
| `ccx` | ✅ | ❌ [^nqctrl] | ✅ | ❌ |
| `measure` `measure_all` | ✅ | ✅ | ✅ | ✅ |
| `reset` | ✅ | ❌ | ✅ | ❌ |
| `barrier` | ❌ | ❌ | ✅ | ❌ |
| Classically controlled gates | ❌ | ❌ | ❌ | ❌ |
| `qsend` / `qrecv` | ✅ | ✅ | ⚠️ [^aerswap] | ✅ |
| `qscatter` / `qgather` | ✅ | ❌ | ❌ | ❌ |
| `expose` / `unexpose` | ✅ | ❌ | ❌ | ❌ |

[^nqrot]: The NetQASM adapter emits `rot_X(n=round(theta), d=16)`, i.e. an angle
    of `round(theta)·π/2¹⁶`. Rotation angles are therefore **not** faithfully
    reproduced on this backend.

[^qrot]: Discretised to multiples of π/16 (`rot_* Q n 4`). Angles that are not a
    multiple of π/16 are rounded to the nearest one.

[^nqswap]: `swap` is not in the NetQASM adapter's single-qubit gate map, and the
    adapter ignores names it does not know instead of raising, so the gate is
    dropped without warning.

[^nqctrl]: The NetQASM adapter looks up `op.name` on a `ControlledGate`, which has
    no such attribute, so **every** controlled gate raises
    `AttributeError: 'ControlledGate' object has no attribute 'name'`.

[^aerctrl]: The Aer adapter handles controlled `X`, `Z` and `RZ`. A controlled
    `S`, `T` or `P` matches no branch and is dropped without warning.

[^qctrl]: The Qoala adapter maps controlled-`RX` to `cnot` and controlled-`RZ` to
    `cphase`. Since `cx` records a controlled-`X` and `cz` a controlled-`Z`,
    neither matches; `crz` maps to `cphase`, discarding its angle.

[^aerswap]: Supported in the default `transfer_mode="swap"`, which inserts an
    unphysical SWAP rather than modelling a real transfer.
    `transfer_mode="teleport"` raises `NotImplementedError`.

:::{admonition} If in doubt, use CUNQA
:class: tip

CUNQA is the only backend that implements the whole primitive set, and the one
the shipped examples are written against. Start there, then port to the
simulator that matches the physics you want to study.
:::

## Operations under the hood

Every call appends an {class}`~netqmpi.sdk.operations.operation.Operation` — a
Command object — to the circuit's
{class}`~netqmpi.sdk.operations.container.OperationContainer`. The container is
a Composite, so a block that means more than its parts (a `qscatter`, which
expands into individual transfers) stays a sub-container rather than being
flattened away.

```python
for op in circuit:              # depth-first over leaf operations
    print(op)
# Gate(H, qubits=[0])
# QSend(qubits=[0], dest_rank=1, ...)

len(circuit)                    # top-level entries, not flattened
circuit.ops.children            # direct children, nesting preserved
```

This is the representation a backend adapter consumes; see
[Writing a backend](../development/writing-a-backend.md).
