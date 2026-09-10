# Writing a new backend

Adding a backend **never requires touching the SDK**. You provide three Runtime
components under `netqmpi/runtime/adapters/<backend>/`, mirroring the existing
`cunqa/`, `netqasm/`, `aer/` and `qoala/` packages, and register a flag.

```text
netqmpi/runtime/adapters/mybackend/
├── __init__.py                 # public exports
├── mybackend_executor.py       # Executor + RunConfig subclass
├── mybackend_circuit.py        # CircuitAdapter
└── mybackend_communicator.py   # Communicator
```

## 1. The circuit adapter

Subclass {class}`~netqmpi.sdk.circuit.Circuit` and implement the `_translate_*`
hooks. Each maps one recorded operation onto your backend's native instructions.

```python
from netqmpi.sdk.circuit import Circuit
from netqmpi.sdk.operations import (
    Gate, ControlledGate, ClassicalControlledGate,
    Measure, Reset, Barrier, OperationContainer,
    QSend, QRecv, QScatter, QGather, Expose, Unexpose,
)


class MyCircuitAdapter(Circuit):

    def _translate_gate(self, op: Gate):
        gate_map = {
            "H": lambda: self._native.h(op.qubits[0]),
            "X": lambda: self._native.x(op.qubits[0]),
            "RX": lambda: self._native.rx(op.params[0], op.qubits[0]),
        }
        if op.name not in gate_map:
            raise NotImplementedError(
                f"Gate '{op.name}' is not implemented for the MyBackend backend.")
        gate_map[op.name]()

    def _translate_operation_container(self, op: OperationContainer):
        for child in op.children:      # one nesting level at a time
            self.translate(child)

    ...
```

The full set of hooks: `_translate_gate`, `_translate_controlled_gate`,
`_translate_classical_controlled_gate`, `_translate_measure`, `_translate_reset`,
`_translate_barrier`, `_translate_operation_container`, `_translate_qsend`,
`_translate_qrecv`, `_translate_qscatter`, `_translate_qgather`,
`_translate_expose`, `_translate_unexpose`.

Dispatch is by operation type, walking the MRO for subclasses that are not
registered explicitly, so you never write the `isinstance` chain yourself.

:::{admonition} Raise, do not ignore
:class: important

For operations you have not implemented, **raise `NotImplementedError` naming the
operation and the backend**:

```python
raise NotImplementedError("Barrier is not implemented for the MyBackend backend.")
```

The `if name in gate_map:` pattern with no `else` — as the NetQASM and Aer
adapters use — drops unsupported gates without a word, and produces results that
are silently wrong. It is the single worst failure mode a backend can have.
:::

### Iterating the container

Translate `OperationContainer` children **one nesting level at a time** rather
than flattening. A `QScatter` that has been flattened is indistinguishable from a
sequence of transfers, and a backend that needs to treat the block as a block —
because it has a native scatter, say — has lost the information.

### Collectives a backend expands jointly

Operations deriving from
{class}`~netqmpi.sdk.operations.qmpi.CollectiveOperation` (`Expose`, `Unexpose`)
are the exception to per-rank translation. If your backend expands them into all
the participating circuits at once — as CUNQA's `cat_entangler` does — write a
group-translation function that walks every rank's stream and stops each of them
at the matching collective:

```python
def translate_group(adapters: Dict[int, MyCircuitAdapter]) -> List[NativeCircuit]:
    ranks = sorted(adapters)
    streams = {r: list(adapters[r].ops.flatten()) for r in ranks}
    cursors = {r: 0 for r in ranks}

    while True:
        # Every rank runs ahead on its own until it hits a collective.
        for rank in ranks:
            while (cursors[rank] < len(streams[rank])
                   and not isinstance(streams[rank][cursors[rank]], CollectiveOperation)):
                adapters[rank].translate(streams[rank][cursors[rank]])
                cursors[rank] += 1

        # A collective is ready when all of its participants sit on it.
        ...
```

Then make the per-rank hooks raise, so the joint path cannot be bypassed by
accident. See
{func}`~netqmpi.runtime.adapters.cunqa.cunqa_circuit.translate_group` for the
worked version, including the deadlock report it produces when ranks block on
collectives that never match.

### Reading the resource budget

At translation time the trace is complete, so the pools are final:

```python
circuit.num_qubits            # user data qubits
circuit.num_comm_qubits       # communication qubits to reserve
circuit.num_clbits            # user classical bits
circuit.num_protocol_clbits   # protocol classical bits to reserve
```

Reserve the protocol classical register **after** the user's own bits, so a
protocol never clobbers a user measurement. `comm_qubit(slot)` converts a slot
index to a circuit-wide qubit index.

## 2. The executor

Subclass {class}`~netqmpi.runtime.executor.Executor` and implement three methods.

```python
from netqmpi.runtime.executor import Executor
from netqmpi.runtime.run_config import RunConfig
from netqmpi.sdk.environment import Environment
from netqmpi.helpers import load_main


@dataclass
class MyRunConfig(RunConfig):
    shots: int = 1024
    my_parameter: float = 0.5


class MyExecutorAdapter(Executor):

    def __init__(self, size: int, config: MyRunConfig = None):
        super().__init__(size, config or MyRunConfig())

    def create_circuit(self, num_qubits, num_clbits, comm) -> MyCircuitAdapter:
        return MyCircuitAdapter(num_qubits, num_clbits, comm)

    def build_apps(self, file: str, size: int):
        main_func = load_main(file)          # validates main() exists
        apps = []
        for rank in range(size):
            comm = MyCommunicator(rank, size, self._config, executor=self)
            env = Environment(comm, self)
            apps.append(lambda env=env: main_func(env=env))
        return apps

    def run(self, apps) -> None:
        for app in apps:
            app()
```

`build_apps` is where resources are acquired. Acquire them inside a `try` and
release them on failure, so a broken setup never leaves an allocation behind —
CUNQA's `_drop_raised_qpus` is the model.

## 3. The communicator

Subclass {class}`~netqmpi.sdk.communicator.QMPICommunicator` and implement the
context manager. `__exit__` is where execution is triggered.

```python
class MyCommunicator(QMPICommunicator):

    _registry: Dict[int, "MyCommunicator"] = {}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:            # never swallow a user exception
            MyCommunicator._registry.clear()
            return None

        MyCommunicator._registry[self.rank] = self

        # The last rank to leave triggers the joint run.
        if len(MyCommunicator._registry) == self.size:
            registry = dict(MyCommunicator._registry)
            MyCommunicator._registry.clear()
            results = self._executor.run_simulation(registry)
            for rank, comm in registry.items():
                comm.results = results.get(rank, {})
        return None
```

Two rules worth stating explicitly:

**Never swallow an exception raised inside the `with` block.** Return `None`, not
`True`, and clear any shared state so the next run starts clean.

**Reset class-level state after the last rank exits.** Adapters are reusable
within the same process — the test suite and the experiment drivers rely on it.

## 4. Register the flag

Add a branch to {func}`netqmpi.runtime.cli.main`, with the import **inside** the
branch so users without your backend's dependencies are unaffected:

```python
backend_group.add_argument("--mybackend", action="store_true",
                           help="Use MyBackend backend")

...

elif args.mybackend:
    from netqmpi.runtime.adapters.mybackend import MyExecutorAdapter, MyRunConfig

    config = _build_config(MyRunConfig, "mybackend", args)
    executor = MyExecutorAdapter(args.num_procs, config=config)
```

Add the name to `KNOWN_BACKENDS` in {mod}`netqmpi.runtime.run_config` so a
`mybackend:` block in a config file is recognised as a backend block rather than
merged into the generic settings.

## Checklist

- [ ] Every `_translate_*` hook either implemented or raising `NotImplementedError`
- [ ] Unknown gate names raise rather than being dropped
- [ ] `OperationContainer` children translated one nesting level at a time
- [ ] Communication qubits and protocol classical bits reserved from the pools
- [ ] Protocol classical register placed after the user's own bits
- [ ] `__exit__` does not swallow exceptions and resets shared state
- [ ] Resources released on a failed setup
- [ ] Backend imported lazily in the CLI, and added to `KNOWN_BACKENDS`
- [ ] `examples/1_send_recv.py` runs unmodified
