# API reference

Generated from the docstrings in the package itself, so it never drifts away
from the code.

```{toctree}
:maxdepth: 2

sdk
runtime
adapters
```

## Where to look

**Writing an application?** Everything you need is in the
[SDK](sdk.rst) — {class}`~netqmpi.sdk.environment.Environment`,
{class}`~netqmpi.sdk.circuit.Circuit` and
{class}`~netqmpi.sdk.communicator.QMPICommunicator`.

**Configuring a run?** See [Runtime](runtime.rst) for
{class}`~netqmpi.runtime.run_config.RunConfig` and the CLI, and
[Adapters](adapters.rst) for each backend's config subclass.

**Writing a backend?** [Runtime](runtime.rst) has the
{class}`~netqmpi.runtime.executor.Executor` contract; [Adapters](adapters.rst)
has four worked implementations.

## Package layout

```text
netqmpi/
├── helpers.py                  load_main(): locate a script's main()
├── sdk/                        user-facing, backend-agnostic
│   ├── environment.py          Environment
│   ├── circuit.py              Circuit (fluent API + translation hooks)
│   ├── communicator.py         QMPICommunicator
│   ├── resources.py            IndexPool
│   └── operations/             the Command/Composite operation model
│       ├── operation.py        Operation
│       ├── gate.py             Gate, ControlledGate, ClassicalControlledGate
│       ├── non_unitary.py      Measure, Reset, Barrier
│       ├── container.py        OperationContainer
│       └── qmpi.py             QSend, QRecv, QScatter, QGather, Expose, Unexpose
└── runtime/                    execution-facing
    ├── cli.py                  netqmpi entry point, simulate()
    ├── executor.py             Executor contract
    ├── run_config.py           RunConfig, read_config_block()
    └── adapters/
        ├── cunqa/              reference backend
        ├── netqasm/
        ├── aer/
        └── qoala/
```
