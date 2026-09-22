# The test suite

NetQMPI ships a test suite of **340-odd tests** under
[`test/`](https://github.com/NetQIR/net-qmpi/tree/main/test). It is built around
one observation about the architecture: because the SDK only *records* a
program and the runtime only *translates* the record, most of the library can
be tested with no simulator at all — and the part that cannot is pinned on the
one backend that installs anywhere.

```bash
pytest                          # everything installed backends allow
pytest -m "not integration"     # skip the slow NetSquid runs (~30 s)
pytest test/test_sdk_circuit_communication.py -q      # one file
pytest -k expose                                      # one topic
```

Configuration lives in [`pytest.ini`](https://github.com/NetQIR/net-qmpi/blob/main/pytest.ini):
`testpaths = test`, the repository root on `pythonpath`, and a single marker,
`integration`.

:::{admonition} Nothing has to be installed to run most of it
:class: tip

The SDK and runtime tests import no backend whatsoever, so they run in a bare
checkout with only `pytest` and `pyyaml`. Backend files begin with
`pytest.importorskip`, so a missing simulator **skips** that file instead of
failing the run — which matters here, since NetQASM 1.x, NetQASM 2.x, CUNQA and
Qoala cannot all live in one environment.
:::

## The two layers

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item-card} Trace-level tests
*No backend. Milliseconds.*

Run a program's `main()` against test doubles and assert on the
{class}`~netqmpi.sdk.operations.container.OperationContainer` it produced: which
operations, in which order, with which tags and which borrowed resources.

This is where validation, resource accounting and the cross-rank agreement of
tags are pinned.
:::

:::{grid-item-card} Execution-level tests
*Aer (or NetQASM/Qoala). Seconds.*

Write a small application, run it on several ranks, and assert on the
histogram. This is where a primitive's *meaning* is checked: that a `qsend`
really moves the state, that an exposed control really drives the receiver's
gate.
:::

::::

## What each file covers

| File | Layer | Needs | Tests | Pins |
|---|---|---|---|---|
| `test_sdk_operations.py` | trace | — | 35 | The operation vocabulary: validation, copied accessors, value semantics |
| `test_sdk_operations_qmpi.py` | trace | — | 31 | The cross-rank records: transfers, rooted collectives, telegate windows, `matches` |
| `test_sdk_container.py` | trace | — | 14 | The composite: `children` vs `flatten`, nesting, derived qubit sets |
| `test_sdk_resources.py` | trace | — | 11 | The index pool: reuse, high-water mark, ordering |
| `test_sdk_circuit_gates.py` | trace | — | 51 | The fluent gate API, index validation, the dispatch table |
| `test_sdk_circuit_communication.py` | trace | — | 48 | `qsend`/`qrecv`, `qscatter`/`qgather`, `expose`/`unexpose` as recorded |
| `test_sdk_communicator_environment.py` | trace | — | 13 | The user-facing facade: rank, size, neighbours, circuit registration |
| `test_runtime_config.py` | runtime | — | 17 | `RunConfig`, the YAML reader, per-backend blocks |
| `test_runtime_cli.py` | runtime | — | 30 | `load_main`, backend selection, `--shots`/`--config`, usage errors |
| `test_aer_primitives.py` | execution | Qiskit Aer | 62 | Every gate and every primitive, run end to end |
| `test_aer_transfers.py` | execution | Qiskit Aer | 13 | Cross-rank ordering regressions in the joint translation pass |
| `test_examples.py` | execution | Qiskit Aer | 7 | The five shipped examples, against their documented output |
| `test_netqasm_backend.py` | execution | SquidASM + NetSquid | 8 | The NetQASM adapter (marked `integration`) |
| `test_qoala_backend.py` | execution | qoala-sim + NetSquid | 9 | The Qoala adapter's gate table |

### Trace-level files

`test_sdk_operations.py` — every operation class the SDK can record. A
malformed operation is refused where it is built (an unnamed gate, a control
with nothing to control, a qubit list that is not a list of integers), accessors
hand out copies so an adapter cannot rewrite the program it is translating, and
`__eq__`/`__hash__` give operations value semantics, which is what lets the
other files assert *"this is the program I traced"* in one line.

`test_sdk_operations_qmpi.py` — the records a cross-rank call leaves behind.
Nothing is exchanged while a program is traced, so these classes carry
everything the runtime needs to fit the halves back together: the `tag`, the
participant list, and the resources each rank contributes. It also pins one
structural fact — {class}`~netqmpi.sdk.operations.qmpi.RootedTransfer` is
deliberately **not** a `CollectiveOperation`, because a scatter expands into
ordinary transfers and must not make the ranks wait.

`test_sdk_container.py` — the two traversals and the difference between them:
`children` keeps the nesting (an adapter dispatching on type still sees *a
scatter*), `flatten` yields every leaf depth-first (a runtime interleaving the
ranks needs the linear order).

`test_sdk_resources.py` — {class}`~netqmpi.sdk.resources.IndexPool`, the
allocator behind communication qubits and protocol classical bits. A hundred
sequential transfers must cost the backend *one* communication qubit; two
windows open at once must cost two. Reuse also has to be deterministic, since
every rank runs the allocator over its own trace and both sides must agree.

`test_sdk_circuit_gates.py` — what each fluent call records (`cx` a controlled
`X`, `crz(theta)` a controlled `RZ` carrying the angle), what it refuses
(out-of-range qubits and classical bits, caught at the offending line), and how
{meth}`~netqmpi.sdk.circuit.Circuit.translate` dispatches — including that
controlled gates are matched *before* plain ones, and that an unknown operation
raises instead of being skipped.

`test_sdk_circuit_communication.py` — the primitives as the SDK records them,
traced on every rank of a run at once so the records can be checked against each
other:

- both halves of a transfer derive the same tag (`teledata_0_1_0`) without
  talking, and each peer and direction is counted separately;
- a scatter splits the root's buffer in rank order and the root keeps nothing;
  a gather is its mirror image;
- an `expose` hands the root its own qubit back and each receiver a
  communication qubit, one correction bit per receiver on the root;
- that communication qubit stops being addressable the moment the window
  closes, nested windows unwind innermost first, and a transfer made while a
  window is open borrows a *second* slot.

`test_sdk_communicator_environment.py` — the two objects a user program
actually holds. Rank, size, ring neighbours, the `with comm:` block, the
forwarding of each primitive to the circuit, and the fact that every circuit
created through the environment is registered on the communicator — which is
how a runtime later finds the program.

### Runtime files

`test_runtime_config.py` — one configuration file serves every backend:
generic settings at the top level, an optional per-backend block, and only the
selected backend's block merged in. Two properties make that safe and both are
tested: an unknown key is an error rather than a silent default, and a backend
block never leaks into another backend's config.

`test_runtime_cli.py` — everything `netqmpi -n 3 app.py --aer` does before any
quantum work starts. `load_main` and its error messages; `_build_config` and the
precedence of `--shots` over `--config`; and backend selection for **all** flags
— including `--netqasm1.0` — by registering stub adapter modules under the
adapter package names, so the branching is testable in an environment where the
four backends cannot coexist. Usage errors (`-n 0`, two backend flags, a
mistyped config key) must exit with status 2 and a message, not a traceback.

### Execution files

`test_aer_primitives.py` is the broad end-to-end file. Aer is the right backend
for it: it moves a qubit with a SWAP straight across one global register, so no
entanglement is consumed and nothing decoheres — a wrong answer is a bug in
NetQMPI, never noise. It covers

- **every gate of the fluent API**, one parametrised case each: `x`, `y`, `z`,
  `h`, `s`/`sdg`, `t`/`tdg`, `rx`/`ry`/`rz`, `swap`, `cx`, `cz`, `crz`, `ccx`,
  `reset`, `barrier`, `measure` into an arbitrary classical bit, `measure_all`;
- **transfers**: a move (the sender is left in `|0⟩`), a relay down a chain of
  2, 3 and 5 ranks, several qubits in one call, a superposition surviving a
  round trip, and ranks asking for registers of *different* widths;
- **rooted collectives**: scatter, scatter with multi-qubit chunks, gather, and
  a scatter/gather round trip;
- **telegate windows**: a control driving every receiver's gate in groups of 2,
  3 and 4 ranks, a window that *entangles* the ranks it spans, nested windows
  over different groups (the `5_qft_expose` shape), and a window reopened three
  times;
- **several distributed programs in one run**, paired across ranks by creation
  order;
- **what the backend refuses**: a send nobody receives, two windows waiting for
  each other, a window its group never joins, ranks that created different
  numbers of circuits, an unimplemented operation, `transfer_mode="teleport"` —
  each checked for the *content* of the message, not just the exception type.

## Conventions

**Deterministic payloads.** Wherever possible a program prepares `|1⟩` and the
test demands every shot come back the same. `{'1': 512}` cannot happen by
accident; a probability can.

**Echoes.** Where a superposition is needed, the program undoes what it did, so
the right answer is all-zeros and a misordering shows up as a wrong result
rather than as a subtly different distribution. This is how the transfer
round-trip and the cross-rank ordering regressions are written.

**Bit order.** Aer keeps every rank's classical bits in one register, so a raw
histogram key spans the whole run. `run_app` splits it into one histogram per
rank with the bits in **ascending classical-bit order** — `"01"` means
`cbit 0 = 0, cbit 1 = 1` — which is the order the program wrote them in rather
than Qiskit's most-significant-first printing. The joint key stays available as
`results.joint`, which is the only place a *correlation* between ranks is
visible.

**Doubles.** [`test/conftest.py`](https://github.com/NetQIR/net-qmpi/blob/main/test/conftest.py)
supplies what the SDK needs and nothing more:

| Double | Stands in for | Used for |
|---|---|---|
| `StubCommunicator` | {class}`~netqmpi.sdk.communicator.QMPICommunicator` | A rank that knows its number and counts block entries |
| `RecordingCircuit` | {class}`~netqmpi.sdk.circuit.Circuit` | A circuit whose thirteen translation hooks record instead of emitting, which is what makes the dispatch observable |
| `StubExecutor` | {class}`~netqmpi.runtime.executor.Executor` | A backend that hands out recording circuits |
| `make_circuit` | fixture | One traced circuit on a fresh communicator |
| `make_group` | fixture | One circuit per rank of the same program, for checking records against each other |

## Known gaps the suite pins

Two tests are expected to fail today. They use `xfail(strict=True)`, so the day
the underlying issue is fixed the suite turns **red** and the marker has to be
removed — the tests are reminders, not silence.

:::{admonition} Aer drops the controlled-phase family
:class: warning

`test_aer_primitives.py::test_controlled_phase_gates_reach_the_simulator`

The Aer adapter's controlled-gate table handles `X`, `Z` and `RZ` only.
{meth}`~netqmpi.sdk.circuit.Circuit.cs`, {meth}`~netqmpi.sdk.circuit.Circuit.ct`
and {meth}`~netqmpi.sdk.circuit.Circuit.cp` record a controlled `S`, `T` or `P`,
none of which the table names, and `_translate_controlled_gate` ends without
emitting anything — **no error, no gate**. CUNQA raises `NotImplementedError`
for a gate it does not know; Aer stays quiet.

It matters beyond the gate table: `examples/5_qft_expose.py` builds its rotations
out of `cp`, so on Aer that example runs with every rotation missing. It still
passes its own test, because on the all-zero input every control is `|0⟩` and the
rotations would be no-ops anyway — which is exactly why this gap needs a test of
its own.
:::

:::{admonition} `simulate()` has no reachable default backend
:class: warning

`test_runtime_cli.py::test_simulate_without_an_executor_is_currently_broken`

{func}`~netqmpi.runtime.cli.simulate` falls back to `NetQASMExecutorAdapter`
when it is called with no executor, but `cli.py` never imports that name — the
`--netqasm` branch imports it locally, inside `main()`. Calling `simulate()`
from Python without an executor therefore raises `NameError`. The CLI itself is
unaffected, since it always passes one. That test pins the current behaviour
rather than the intended one, so it is visible instead of being discovered from
an unhelpful traceback.
:::

## Adding a test

**A question about what a program records** goes in a trace-level file and needs
no backend:

```python
def test_a_send_records_one_transfer_per_qubit(make_circuit):
    circuit = make_circuit(num_qubits=3, num_clbits=1, rank=0, size=2)
    circuit.qsend([0, 1, 2], 1)

    sends = [op for op in circuit if isinstance(op, QSend)]
    assert [op.qubits for op in sends] == [[0], [1], [2]]
```

**A question about what a program does** goes in `test_aer_primitives.py`. Write
the application as a string, run it, assert on the histogram:

```python
def test_a_qubit_moves_to_the_rank_that_receives_it(tmp_path):
    results = run_app(TELEPORT, 2, tmp_path)
    assert_exact(results[0], "0")      # the sender kept nothing
    assert_exact(results[1], "1")
```

**A new backend** gets its own file, guarded with `importorskip`, and marked
`integration` if a run costs seconds rather than milliseconds. The programs in
`test_aer_primitives.py` are written against the SDK alone, so they port to any
backend that implements the primitives they use — which is the point of the
architecture, and the cheapest way to hold a new adapter to the same standard.

:::{seealso}
[`examples/frequent_errors/`](https://github.com/NetQIR/net-qmpi/tree/main/examples/frequent_errors)
is the companion corpus: programs that are *meant* to fail, one failure mode
each, with the layer that catches them and the message they produce. See
[Troubleshooting](../guide/troubleshooting.md).
:::
