# Command-line interface

NetQMPI installs a single console entry point, `netqmpi`, defined in
{mod}`netqmpi.runtime.cli`.

```bash
netqmpi -n <NUM_NODES> <script.py> [--cunqa | --netqasm | --netqasm1.0 | --aer | --qoala]
        [--shots N] [--config config.yaml] [--metrics FILE]
        [--plugin NAME] [--plugin-config params.yaml]
netqmpi --list-plugins
netqmpi --version
```

## Arguments

`-n`, `--num-procs` *(required)*
: Number of parallel quantum nodes — the communicator size. Must be at least 1.

`script` *(required, positional)*
: Path to the NetQMPI Python script. It must be a `.py` file defining a
  `main()` function; both conditions are checked before anything is built.

`--cunqa` / `--netqasm` / `--netqasm1.0` / `--aer` / `--qoala`
: Backend selection. Mutually exclusive. `--squidasm` is an alias of
  `--netqasm`. If none is given, NetQMPI falls back to NetQASM and says so:

  ```text
  No backend flag; using default (NetQASM)
  ```

`--shots N`
: Number of shots. Overrides the value in `--config`, if any, so a quick run can
  bump the shot count without editing the file. Defaults to 1024, except on
  NetQASM, where it is 50 (see [the NetQASM backend](../backends/netqasm.md#configuration)).

`--config PATH`
: Path to a YAML file with generic settings and an optional per-backend block.
  This replaces per-backend command-line flags — see
  [Configuration](configuration.md).

`--plugin NAME`
: [Simulation plugin](../backends/netqasm-plugins.md) to run with: the short
  name of an installed plugin or `module:callable`. Overrides `plugin` in
  `--config`. NetQASM backend only: any other backend refuses it with a usage
  error rather than running without it.

`--plugin-config FILE`
: YAML file with the plugin's parameters, a mapping handed to it as
  `plugin_config`. Overrides `plugin_config` in `--config`. NetQASM backend
  only, like `--plugin`.

`--list-plugins`
: Print each installed plugin — its name, the `module:attribute` it points to,
  and the distribution and version that provide it — and exit.

`--metrics FILE`
: Write the metrics of the run, what `comm.metrics` holds, to `FILE` as JSON
  once the run has finished. Only the metrics are written, not the
  configuration. Today only the [NetQASM backend](../backends/netqasm.md)
  reports metrics: the simulated time of each shot and what a
  [plugin](../backends/netqasm-plugins.md#reporting-metrics) reports. The other
  backends write an empty object, `{}`, and say so on stderr, so a sweep over
  backends always finds a file to read. A directory that does not exist is a
  usage error, reported before the run. Nothing is written if the run fails.

## Examples

```bash
# The reference backend, attaching to already-raised vQPUs
netqmpi -n 3 examples/3_scatter.py --cunqa --shots 1024

# ... with a hardware/vQPU configuration file
netqmpi -n 3 examples/5_qft_expose.py --cunqa --config cunqa.yaml

# Quantum-network simulation
netqmpi -n 2 examples/1_send_recv.py --netqasm

# Circuit simulation, no special setup needed
netqmpi -n 2 examples/1_send_recv.py --aer --shots 1024

# Node execution environment, with a noise model
netqmpi -n 2 examples/1_send_recv.py --qoala --config qoala.yaml

# Quantum-network simulation on the hardware model of an installed plugin
netqmpi -n 3 app.py --netqasm --plugin nv_2026 --plugin-config nv.yaml
```

## Which NetQMPI is this?

```bash
netqmpi --version
# NetQMPI 0.3.1 (tag v0.3.1 + 1 commit, commit 662436b0544560689612d7546eab9089a423f783)
```

It prints the release declared in `setup.py`, the latest git tag, how many
commits the code is past it, the full commit hash, and whether the working tree
had uncommitted changes. From Python, `netqmpi.__version__` is the release and
`netqmpi.get_build_info()` returns every field, `as_dict()` included, to store
next to results.

An installed wheel or sdist reports the tag and commit it was **built** from:
`setup.py` freezes them into `netqmpi/_build_info.py` in the build (never in the
sources). A checkout — plain or installed with `pip install -e .` — asks git
when first asked, so it always describes the code that is checked out now.

## Lazy backend imports

The CLI imports **no** backend at module level. The adapter package is imported
only inside the branch its flag selects, so a user with just `qiskit-aer`
installed can run `--aer` without CUNQA, NetQASM or Qoala being importable.

This is also what lets the `--netqasm` and `--qoala` backends coexist in the
codebase despite their incompatible `netqasm` versions: only one of them is ever
imported in a given run.

## Programmatic use

{func}`~netqmpi.runtime.cli.simulate` is the same path the CLI takes, and can be
called directly when you want to drive runs from Python — a parameter sweep, for
instance:

```python
from netqmpi.runtime.cli import simulate
from netqmpi.runtime.adapters.cunqa import CunqaExecutorAdapter, CunqaRunConfig

config = CunqaRunConfig(shots=4096, family="my_family")
executor = CunqaExecutorAdapter(size=3, config=config)

simulate(script="app.py", num_procs=3, executor=executor, timer=True)
```

`timer=True` prints the wall-clock execution time:

```text
finished simulation in 12.47 seconds
```

After the run, `executor.metrics()` returns the metrics `--metrics` would
write: an empty dict on a backend that reports none.

The experiment drivers under
[`scripts/experiments/`](https://github.com/NetQIR/netqmpi/tree/main/scripts/experiments)
use this entry point to sweep hardware parameters across many runs.
