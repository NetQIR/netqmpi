# Simulation plugins (NetQASM backend)

A **plugin** changes how SquidASM simulates the network, without changing
NetQMPI or the program that runs on it. The program stays the same SPMD script;
the command line or the configuration file names a plugin and passes it its
parameters, and NetQMPI keeps the plugin active while SquidASM simulates.

A plugin is an add-on: install its package, then select it by name.

```bash
pip install netqmpi-colorcenters
netqmpi -n 3 app.py --netqasm --plugin nv_2026 --plugin-config nv.yaml
```

or, equivalently, in the configuration file:

```yaml
netqasm:
  plugin: nv_2026            # an installed plugin, or module:callable
  plugin_config:             # handed to the plugin as is
    ...
```

Plugins are a feature of the [NetQASM backend](netqasm.md), with both
`--netqasm` (alias `--squidasm`) and `--netqasm1.0`. Any other backend refuses
`--plugin` and `--plugin-config` rather than ignoring them, since ignoring them
would run an ideal simulation that looks like the plugin's hardware:

```text
netqmpi: error: --plugin is not supported by the aer backend, which has no simulation plugins; the run would ignore it. ...
```

## When a plugin is needed

The `netqasm` block of the configuration describes the simulated hardware with
a handful of keys (see [Configuration](netqasm.md#configuration)):

- the number of qubits per node;
- memory decoherence, through `t1` and `t2`;
- **one** `gate_fidelity` for every gate;
- one `link_fidelity` and one of three noise models for every link;
- generic or NV hardware.

That is all SquidASM's network description can express. Anything else needs
code, for example:

- separate noise and durations for one- and two-qubit gates;
- readout error;
- another native gate set, or other decompositions of the gates a program uses;
- links with another physical model, such as heralded links with a success
  probability per attempt.

A plugin supplies that code. Because it lives outside NetQMPI, a model of a
particular platform (colour-centre processors with heralded double-click links,
for instance) can be developed, versioned and shared on its own, and every
NetQMPI program can run on it by changing its configuration file.

## Using a plugin

`plugin` names the plugin, and `plugin_config` is an arbitrary mapping that
NetQMPI hands to it unread:

| Key | Default | Command line | Meaning |
|---|---|---|---|
| `plugin` | `None` | `--plugin NAME` | An installed plugin's short name, or `"module:callable"` |
| `plugin_config` | `None` | `--plugin-config FILE` (a YAML mapping) | Parameters of the plugin; NetQMPI does not read them |

The command-line flags take precedence over the configuration file, as
`--shots` does. `plugin_config` without a `plugin` is an error.

### Naming a plugin

A name **without a colon** is a short name, looked up among the installed
plugins: those that a package registers in the `netqmpi.plugins` entry-point
group (see [Packaging](#packaging-a-plugin)). It works from any directory.
`netqmpi --list-plugins` shows what is installed, with the target each name
points to and the distribution and version that provide it:

```text
$ netqmpi --list-plugins
gate_noise     gate_noise:gate_noise        netqmpi-example-gate-noise 0.1.0
readout_error  readout_error:readout_error  netqmpi-example-readout-error 0.1.0
```

An unknown short name is an error that lists the installed plugins.

A name **with a colon** is `"module:callable"`. The module is imported the way
`python -m` would import it: **the working directory first**, then the installed
packages. A plugin can therefore also be a single file next to the program, as
in the [examples](#examples), or a module of an installed package that
registers no entry point, named with its full dotted path:

```yaml
netqasm:
  plugin: my_platform.netqmpi_plugin:simulation
```

A configuration that names a plugin which cannot be found, that gives
`plugin_config` without a `plugin`, or that the plugin itself refuses, is
refused before the program is traced (see [Errors](#errors)).

## Writing a plugin

### The contract

A plugin is a callable that takes `plugin_config`, a `dict`, and returns a
**context manager**. NetQMPI enters the context manager just before the
simulation and exits it right after:

```python
from contextlib import contextmanager


def my_plugin(config):
    """The entry point that `plugin:` names."""
    params = check(config)          # refuse unknown keys and bad values here
    return _active(params)


@contextmanager
def _active(params):
    original = ...                  # what SquidASM uses now
    ...                             # replace it with your own
    try:
        yield                       # SquidASM simulates here
    finally:
        ...                         # put the original back
```

Between entering and exiting, SquidASM builds the network and simulates every
shot of the run. Whatever the plugin has replaced is what SquidASM uses to do
so. What the context manager yields is how the plugin reports what it observed;
`None` reports nothing (see [Reporting metrics](#reporting-metrics)).

### Optional members

The object the plugin's name points to may also carry three attributes, which
NetQMPI reads with `getattr`, so a plugin without them keeps working. A class
instance is the natural form; a plain function can carry them as attributes too,
as the [examples](#examples) do.

```python
class NV2026:
    backends = ("netqasm",)
    run_defaults = {"formalism": "dm", "num_qubits": 3}

    def configure(self, plugin_config, run_config):
        """Called once when the config is built, before tracing. Raises on bad values."""
        check(plugin_config)
        if run_config.num_qubits < 2:
            raise ValueError("nv_2026: an NV node needs num_qubits >= 2")

    def __call__(self, plugin_config):
        """Unchanged: returns the context manager entered around the simulation."""
        return _active(check(plugin_config))


nv_2026 = NV2026()
```

`run_defaults`
: Run-config keys the plugin's hardware needs, such as a density-matrix
  `formalism` or the `num_qubits` of the device that was characterised. They
  replace the backend's own defaults, and only those:
  **dataclass default < plugin `run_defaults` < configuration file < command line**.
  A key the user sets, at the top level or in the `netqasm` block of the file or
  on the command line, still wins, so check it in `configure` if the plugin
  cannot run with any other value. `run_defaults` may not set `plugin` or
  `plugin_config`, and a key `NetQASMRunConfig` does not have is reported as the
  plugin's mistake. They apply when the configuration is built from a file or
  the command line ({meth}`NetQASMRunConfig.from_dict
  <netqmpi.runtime.adapters.netqasm.netqasm_executor.NetQASMRunConfig.from_dict>`);
  a `NetQASMRunConfig(...)` built in Python takes its arguments as given.

`configure(plugin_config, run_config)`
: Called once when the configuration is built, after NetQMPI has validated it
  and **before any rank is traced**, with the final run configuration, command
  line included. It receives copies, so it can read both but change neither.
  Raise `ValueError` on a bad parameter or an incompatible setting; the command
  line reports it as a usage error.

`backends`
: The backends the plugin is written for, e.g. `("netqasm",)`. Any other backend
  refuses the plugin, so that a SquidASM plugin is not half-applied by a backend
  that gains plugins later. Without it, any backend that accepts plugins does.

The contract is written down as the {class}`netqmpi.plugins.Plugin` protocol;
a plugin does not need to inherit from it.

### Lifecycle

| When | What NetQMPI does |
|---|---|
| The configuration is built | Finds the plugin, so a wrong name stops the run before any circuit is traced; merges its `run_defaults`; checks its `backends`; calls its `configure` |
| The last rank leaves its `with comm:` block | Calls the plugin with `plugin_config` and enters the context manager it returns |
| Inside the context | SquidASM builds the network and simulates all the shots of the run |
| The simulation returns or raises | Exits the context manager |
| The simulation returned | Reads what the context manager yielded into `comm.metrics["plugin"]` (see [Reporting metrics](#reporting-metrics)) |

The plugin is called once per run, and its context manager is entered once,
around all the shots, after every rank has been traced. Validate the parameters
in `configure`, so that a typo in the configuration is reported before the
program runs, and again when called, before replacing anything, for callers
that build a run configuration without going through `configure`.

### Extension points

SquidASM has no API for plugging in a device, an executor or a link model of
your own, so a plugin replaces, for the duration of the run, the names SquidASM
builds them from. These are the ones a plugin usually needs, in SquidASM
0.13:

| Replace | To change | SquidASM calls it with |
|---|---|---|
| `squidasm.sim.network.network.QDevice` | The quantum processor of every generic node: its instructions, their noise and duration, its memory | Keywords `name`, `num_qubits`, `gate_fidelity`, `mem_fidelities` |
| `squidasm.nqasm.qnodeos.SubroutineHandler._get_executor_class` | How NetQASM instructions become operations on the processor: gate decompositions, measurement | `flavour`; a classmethod |
| `squidasm.sim.network.network.NetSquidNetwork._create_link_distributor` | The pairs a link delivers: their state, fidelity and timing | `link`, a `netqasm.runtime.interface.config.Link`; returns a `MagicDistributor` |
| `squidasm.sim.network.network.MagicNetworkLayerProtocol` | The link layer: which memory positions receive the halves of a pair | `nodes`, `magic_distributor`, `translation_unit`, `path`, `network` |

For example, a plugin that slows every link down to one pair per
`pair_interval` nanoseconds, so that qubits decohere while they wait for their
pair:

```python
from contextlib import contextmanager

from squidasm.sim.network import network


def slow_links(config):
    return _slow_links(float(config.get("pair_interval", 50_000.0)))


@contextmanager
def _slow_links(pair_interval):
    original = network.NetSquidNetwork.__dict__["_create_link_distributor"]

    def distributor(self, link, state_delay=None):
        return original(self, link, state_delay=pair_interval)

    network.NetSquidNetwork._create_link_distributor = distributor
    try:
        yield
    finally:
        network.NetSquidNetwork._create_link_distributor = original
```

### Reporting metrics

When a plugin is used to characterise nodes and links, the histogram in
`comm.results` is rarely what matters: the number of pairs a link delivered,
their fidelity or how long the program took are. A plugin reports such
quantities through **what its context manager yields**:

- `None`, as plugins that report nothing do: nothing is added.
- A **mapping** that the plugin fills during the run.
- An object with a **`report()`** method returning a mapping, which NetQMPI
  calls once, after the simulation.

```python
@contextmanager
def _active(params):
    stats = LinkStats()          # filled by the distributor during the run
    ...                          # replace what SquidASM uses
    try:
        yield stats              # -> comm.metrics["plugin"] = stats.report()
    finally:
        ...                      # put the original back
```

Anything else is refused when the context is entered, before the simulation.
The mapping is read, or `report()` called, only once the simulation has
returned and the context has exited: a failed run reports its error, never
metrics. An exception in `report()` fails the run with an error naming the
plugin.

The metrics are a mapping on the communicator, `comm.metrics`, grouped by who
produced them:

| Entry | Holds |
|---|---|
| `comm.metrics["plugin"]` | A copy of what the plugin reported |
| `comm.metrics["netqmpi"]["sim_time_ns"]` | NetQMPI's own: the simulated time of each shot, in ns, in shot order |

`comm.metrics` exists on every backend and is empty unless the backend fills
it. On this backend it is filled at the same moment as `comm.results`, when the
last rank leaves `with comm:`, so only the last rank sees it inside `main`.
Every rank's communicator then holds the same metrics, which describe the run
rather than a rank. `--metrics FILE` writes them to a JSON file (see
[Command-line interface](../guide/cli.md)), and `executor.metrics()` returns
them to a caller in Python.

**Key names.** So that the reports of different plugins can be compared, use
these names when a plugin measures the quantity they describe. None of them is
required:

| Key | Meaning |
|---|---|
| `sim_time_ns` | Simulated time, in ns |
| `epr_pairs` | EPR pairs delivered |
| `epr_attempts` | Attempts made to deliver them |
| `pair_fidelity` | Fidelity of the delivered pairs |

**The simulated time of a shot** runs from the moment the last rank has its
connection and EPR sockets ready to the moment the last rank finishes, its
final instructions flushed. It starts there, rather than at the start of
SquidASM's round, because SquidASM sets the sockets up in wall-clock time while
the simulated clock runs on. Local work that a rank does before the last rank is
ready is therefore not counted. The value still depends on how the
operating system schedules the threads that SquidASM runs the ranks in, one per
rank, because the simulated clock does not wait for them. Expect shots of the
same program to differ, by a factor of three in this case:
`examples/2_round_robin.py` on four ranks gave shots between 1.0 and 3.5 ms
over three runs of four shots. Compare distributions over many shots rather than single
values.

### Guidelines

1. **Validate `plugin_config` in `configure` and in the callable**, before
   anything is replaced, and refuse unknown keys. NetQMPI does not read
   `plugin_config`, so the plugin is the only place where a mistake in it can be
   caught.
2. **Restore everything in `finally`**, so that a failed simulation does not
   leave SquidASM modified for the next run in the same process. Read a method
   to restore from the class's `__dict__`
   (`SubroutineHandler.__dict__["_get_executor_class"]`): a classmethod read
   through the class comes back bound, and putting that back would not restore
   the original.
3. **Replace the name SquidASM looks up.** SquidASM builds nodes by calling
   `QDevice` from the module `squidasm.sim.network.network`, so that is the
   attribute to replace; binding a name in the plugin's own module changes
   nothing.
4. **Respect what NetQMPI still configures.** A replacement for `QDevice`
   receives `num_qubits` and `mem_fidelities` from the `netqasm` block
   (`num_qubits`, `t1`, `t2`) and must honour them. NetQMPI checks that every
   rank fits in `num_qubits` before simulating, so a device with fewer
   positions would fail in the middle of the run instead.
5. **Draw random numbers from NetSquid** (`netsquid.util.simtools.get_random_state()`),
   so that seeding NetSquid makes a run with the plugin reproducible.
6. **Pin the SquidASM version.** These are SquidASM internals rather than a
   public API. The examples are tested with SquidASM 0.13.4 (NetQASM 1.0) and
   0.13.6 (NetQASM 2.0).
7. **Package it**, so it can be installed and selected by name (see below).

### Packaging a plugin

A plugin used beyond one directory is best installed. Its package registers each
plugin under a short name in the `netqmpi.plugins` entry-point group:

```toml
# pyproject.toml of the plugin package
[project]
name = "netqmpi-colorcenters"
dependencies = ["netqmpi", "squidasm>=0.13,<0.14"]

[project.entry-points."netqmpi.plugins"]
nv_2026  = "netqmpi_colorcenters:nv_2026"
snv_2026 = "netqmpi_colorcenters:snv_2026"
```

Once it is installed, `--plugin nv_2026` or `plugin: nv_2026` selects it from
any directory, and `netqmpi --list-plugins` shows it. NetQMPI reads the entry
points with `importlib.metadata`, the same way on every Python it supports, 3.8
(NetQASM 1.x) included. Two installed distributions that register the same name
make the name ambiguous; the error names both, and `module:callable` still
selects either.

## Examples

`examples/plugins/` contains two complete plugins, each with a program, a
configuration file, the plugin's parameters alone (`params.yaml`) and a
`pyproject.toml` that makes it an installable plugin package. Run them from
their own directory, so that the plugin is imported from there:

```bash
cd examples/plugins/readout_error
netqmpi -n 2 app.py --netqasm --config config.yaml
```

or install them and select them by name from anywhere:

```bash
pip install ./examples/plugins/readout_error
netqmpi -n 2 app.py --netqasm --plugin readout_error --plugin-config params.yaml
```

Both are plain functions that carry `configure`, which runs the same checks as
the function, and `backends = ("netqasm",)` as attributes.

### Readout error: replacing the executor

SquidASM measures without error. `readout_error` makes every measurement report
the wrong bit with a probability that depends on the outcome: `prob_error_0`
that a 0 reads as 1, and `prob_error_1` that a 1 reads as 0. It returns a
subclass of SquidASM's own executor from `_get_executor_class`, whose
measurement flips the outcome after the qubit has been measured. The
post-measurement state is therefore the true one and only the reported bit is
wrong, which is what a readout error is.

The error applies to every measurement the simulated nodes make, including the
Bell measurement that `qsend` performs, so it also degrades teleportation.

```{literalinclude} ../../../examples/plugins/readout_error/readout_error.py
:language: python
:start-at: "def readout_error"
```

Its program measures a qubit in `|0⟩` and a qubit in `|1⟩`, so any outcome other
than `10` is a readout error:

```text
$ netqmpi -n 2 app.py --netqasm --config config.yaml
rank 1: {'00': 36, '10': 157, '11': 6, '01': 1}
  |0> read as 1 in 3.5% of 200 shots
  |1> read as 0 in 18.5% of 200 shots
  plugin: 85 of 800 measurements misread
```

against `prob_error_0: 0.05` and `prob_error_1: 0.20`; without the plugin every
shot reads `10`.

The plugin also [reports a metric](#reporting-metrics): it yields a mapping
that it fills as it measures, so the program reads in `comm.metrics["plugin"]`
how many measurements it made, every rank's included (two ranks, two qubits,
200 shots), and how many it misread:

```{literalinclude} ../../../examples/plugins/readout_error/app.py
:language: python
:start-at: "# What the plugin itself counted"
```

`--metrics metrics.json` writes the same numbers, with the simulated time of
each shot, to a file.

### Noise per gate type: replacing the device

SquidASM's generic node applies one `gate_fidelity` to every gate and fixes the
durations of gates. `gate_noise` sets the depolarising probability and the
duration of one- and two-qubit gates separately. It replaces `QDevice` with a
factory that builds a NetSquid `QuantumProcessor` directly: SquidASM's own
`QDevice` cannot simply be given other instructions, because its constructor
overwrites the noise model of every instruction with the one `gate_fidelity`
dictates.

```{literalinclude} ../../../examples/plugins/gate_noise/gate_noise.py
:language: python
:start-at: "def build_qdevice"
```

`num_qubits` and the memory noise (`t1`, `t2`) still come from the `netqasm`
block, and only `gate_fidelity` is replaced. The program prepares a Bell pair on
every rank and counts the shots in which the two bits disagree, which only noise
can cause:

```text
$ netqmpi -n 2 app.py --netqasm --config config.yaml
rank 1: {'11': 80, '00': 81, '10': 20, '01': 19}
  the two bits disagree in 19.5% of 200 shots
```

With the CNOT depolarising each qubit with probability 0.2, the bits disagree
in 0.5 × (1 − 0.8²) = 18 % of the shots in expectation.

## Testing a plugin

A plugin can be tested at two levels, as `test/test_netqasm_plugin.py` does for
both examples:

- **Its callable, without simulating:** that it refuses unknown keys and bad
  values, that the objects it builds have the parameters it was given, and that
  it restores what it replaced when its context exits.
- **Its optional members:** that `run_defaults` names only run-config keys,
  which `NetQASMRunConfig.from_dict({"plugin": ...})` checks, and that
  `configure` refuses what it should.
- **Its registration:** that the entry point resolves, through
  `netqmpi.plugins.load("short_name")` once the package is installed. The
  NetQMPI tests do this without pip, by writing a `.dist-info` directory with an
  `entry_points.txt` on `sys.path` (the `install_plugins` fixture of
  `test/conftest.py`).
- **A run:** build a `NetQASMRunConfig` with the plugin and run a small program
  through the NetQASM executor:

  ```python
  from netqmpi.runtime.adapters.netqasm import NetQASMExecutorAdapter, NetQASMRunConfig

  config = NetQASMRunConfig(shots=4, plugin="readout_error:readout_error",
                            plugin_config={"prob_error_0": 1.0, "prob_error_1": 1.0})
  executor = NetQASMExecutorAdapter(1, config)
  executor.run(executor.build_apps("app.py", 1))
  ```

  Choosing parameters at their limits, such as probabilities of 0 or 1, makes
  the expected outcome deterministic, so the test does not depend on
  statistics. After the run, `executor.metrics()["plugin"]` is what the
  plugin reported, so its metrics can be checked the same way.

## Errors

| Message | Cause |
|---|---|
| `Unknown plugin '...': installed plugins: ...` | A short name no installed package registers |
| `Plugin '...' is provided by more than one installed distribution` | Two packages register the same short name |
| `Cannot import plugin '...' (target, from distribution version)` | An installed entry point whose module does not import |
| `Plugin '...' (...) is not a callable` | An installed entry point whose target is missing or not callable |
| `plugin must read 'module:callable'` | `plugin` has a colon but lacks the module or the callable |
| `Cannot import plugin module ...` | The module is neither in the working directory nor installed |
| `Plugin module ... has no callable ...` | The module has no callable by that name |
| `Plugin '...': run_defaults sets keys ... does not accept` | The plugin's `run_defaults` names a key the run config lacks |
| `Plugin '...' is written for the ... backend(s), not netqasm` | The plugin's `backends` excludes the NetQASM backend |
| `plugin_config is given but no plugin to hand it to` | `plugin_config` without `plugin` |
| `plugin_config must be a mapping of parameters` | `plugin_config` is a list or a single value |
| `--plugin-config FILE must contain a mapping of parameters` | The file of `--plugin-config` is a list or a single value |
| `--plugin is not supported by the ... backend` | `--plugin` or `--plugin-config` with `--aer`, `--cunqa` or `--qoala` |
| `Plugin ... yielded a ...; a plugin's context manager must yield None, a mapping of metrics, or an object with a report() method returning one` | The context manager yields something NetQMPI cannot read metrics from; raised before the simulation |
| `The simulation succeeded, but report() of plugin ... raised ...` | The plugin's `report()` failed after a successful run; its error is chained |
| `report() of plugin ... returned a ..., not a mapping of metrics` | `report()` returned a list or a single value |

Errors in `plugin_config` itself are the plugin's to report, from its
`configure` before the program is traced, or when NetQMPI calls it just before
the simulation if it has no `configure`.
