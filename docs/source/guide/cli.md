# Command-line interface

NetQMPI installs a single console entry point, `netqmpi`, defined in
{mod}`netqmpi.runtime.cli`.

```bash
netqmpi -n <NUM_NODES> <script.py> [--cunqa | --netqasm | --aer | --qoala]
        [--shots N] [--config config.yaml]
```

## Arguments

`-n`, `--num-procs` *(required)*
: Number of parallel quantum nodes — the communicator size. Must be at least 1.

`script` *(required, positional)*
: Path to the NetQMPI Python script. It must be a `.py` file defining a
  `main()` function; both conditions are checked before anything is built.

`--cunqa` / `--netqasm` / `--aer` / `--qoala`
: Backend selection. Mutually exclusive. If none is given, NetQMPI falls back to
  NetQASM and says so:

  ```text
  No backend flag; using default (NetQASM)
  ```

`--shots N`
: Number of shots. Overrides the value in `--config`, if any, so a quick run can
  bump the shot count without editing the file. Defaults to 1024.

`--config PATH`
: Path to a YAML file with generic settings and an optional per-backend block.
  This replaces per-backend command-line flags — see
  [Configuration](configuration.md).

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
```

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

The experiment drivers under
[`scripts/experiments/`](https://github.com/NetQIR/net-qmpi/tree/main/scripts/experiments)
use this entry point to sweep hardware parameters across many runs.
