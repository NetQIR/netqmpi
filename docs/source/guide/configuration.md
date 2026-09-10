# Configuration files

Backend-specific parameters are passed through a **single YAML file** with
`--config`, rather than a proliferation of per-backend command-line flags:

```bash
netqmpi -n 3 app.py --cunqa --config run.yaml
```

## File layout

The file mixes **generic settings** at the top level with optional **per-backend
blocks** keyed by backend name. Only the block matching the selected backend is
read; blocks for other backends are ignored, so one file can carry the settings
for every backend you use.

```yaml
# run.yaml
shots: 1000              # generic: applies whatever the backend

cunqa:                   # read only with --cunqa
  qraise: true
  backend: examples/cunqa_backend.json
  simulator: Munich

qoala:                   # read only with --qoala
  link_fidelity: 0.8
  hardware:
    t1: 0
    single_qubit_gate_depolar_prob: 0.1
```

The recognised backend block names are `cunqa`, `netqasm`, `aer` and `qoala`
(`KNOWN_BACKENDS` in {mod}`netqmpi.runtime.run_config`). Any other top-level key
is treated as a generic setting and merged into every backend's config.

The merge is performed by {func}`~netqmpi.runtime.run_config.read_config_block`:
generic settings first, then the selected backend's block on top. Reading the
file above with `--qoala` therefore yields
`{"shots": 1000, "link_fidelity": 0.8, "hardware": {...}}`.

## Typos are errors, not silence

Unknown keys are rejected rather than ignored, so a mistyped field surfaces
immediately:

```text
ValueError: Unknown config keys for CunqaRunConfig: ['simulater']
```

## Generic settings

| Key | Type | Default | Meaning |
|---|---|---|---|
| `shots` | int | `1024` | Number of times the program is repeated |

`--shots` on the command line overrides whatever the file says, so a quick run
can bump the shot count without editing the file.

## `cunqa` block

Configures which virtual QPUs the run uses, and whether NetQMPI raises them
itself. See the [CUNQA backend page](../backends/cunqa.md) for the full story.

| Key | Type | Default | Meaning |
|---|---|---|---|
| `qraise` | bool | `false` | Raise the vQPUs for this run and drop them afterwards. When `false`, attach to vQPUs that are already up. |
| `backend` | str | `None` | Path to the vQPU definition file the vQPUs are raised with — this is what fixes the qubit budget. **Raise-mode only.** |
| `simulator` | str | `"Munich"` | Simulator backing each vQPU. **Raise-mode only.** |
| `time` | str | `"00:10:00"` | SLURM wall-clock reservation, as `D-HH:MM:SS` or `HH:MM:SS`. **Raise-mode only.** |
| `family` | str | `None` | Which raised family to attach to, or the name of the family to raise. |
| `co_located` | bool | `true` | Whether the vQPUs are reachable from other nodes (*co-located* mode) rather than only from the node they run on (*hpc* mode). Must match how they were raised. |

:::{admonition} Raise-only settings are reported, not ignored
:class: important

Setting `backend`, `simulator` or `time` while attaching to running vQPUs is an
error, not a silent no-op:

```text
ValueError: simulator, time only apply when NetQMPI raises the vQPUs itself.
Either add 'qraise: true' to the cunqa block of the config file, or drop those
settings and raise the vQPUs yourself with qraise before running.
```
:::

## `qoala` block

| Key | Type | Default | Meaning |
|---|---|---|---|
| `num_qubits_per_node` | int | `None` | Physical qubits per node. Inferred from the compiled circuits when unset. |
| `link_duration` | float | `1000.0` | EPR-pair generation time, in ns |
| `qnos_instr_time` | float | `1000.0` | Duration of one quantum-processor instruction, in ns |
| `link_fidelity` | float | `1.0` | EPR-pair fidelity to the ideal Bell state, in `[0.25, 1.0]`. Below 1.0 uses a depolarising link with `prob_max_mixed = (4/3)(1 - link_fidelity)`. |
| `seed` | int | `None` | NetSquid random seed, for reproducible runs |
| `hardware` | mapping | `None` | qdevice noise model; omit for a perfect device |

A `link_fidelity` outside `[0.25, 1.0]` is rejected when the config is built.

### `qoala.hardware` — the qdevice

Maps to {class}`~netqmpi.runtime.adapters.qoala.qoala_executor.QoalaQDeviceConfig`
and is applied uniformly to every node. Durations are in nanoseconds; `t1 == t2
== 0` means "no memory noise", and zero depolarising probabilities mean noiseless
gates, so the defaults describe a **perfect** qdevice.

| Key | Default | Meaning |
|---|---|---|
| `t1` | `0.0` | Amplitude-damping time (ns); 0 disables it |
| `t2` | `0.0` | Dephasing time (ns); 0 disables it. Requires `t2 <= 2 * t1` when both are non-zero |
| `single_qubit_gate_time` | `5e3` | Duration of single-qubit gates (ns) |
| `two_qubit_gate_time` | `200e3` | Duration of two-qubit gates (ns) |
| `init_time` | `5e3` | Duration of qubit initialization (ns) |
| `measure_time` | `5e3` | Duration of measurement (ns) |
| `single_qubit_gate_depolar_prob` | `0.0` | Depolarising probability of single-qubit gates |
| `two_qubit_gate_depolar_prob` | `0.0` | Depolarising probability of two-qubit gates |

:::{note}
Depolarising noise is applied to the **gates** only. `INSTR_INIT` and
`INSTR_MEASURE` carry their own duration but no depolarising error, so there is
no separate readout-flip model.
:::

## `aer` block

| Key | Type | Default | Meaning |
|---|---|---|---|
| `transfer_mode` | str | `"swap"` | Qubit transfer protocol for `qsend`/`qrecv`. `"swap"` inserts an unphysical SWAP between the source and destination slots — shallower and easier to debug, but not a real network transfer. `"teleport"` is **not yet implemented** and raises. |
| `seed_simulator` | int | `None` | RNG seed, for reproducible simulations |

## `netqasm` block

| Key | Type | Default | Meaning |
|---|---|---|---|
| `formalism` | `Formalism` | `Formalism.KET` | Quantum state formalism used by the simulation |
| `enable_logging` | bool | `true` | Per-rank instruction logging |
| `hardware` | str | `"generic"` | Hardware model name |
| `network_config` | Any | `None` | Simulated network topology; default topology when unset |
| `log_cfg` | Any | `None` | NetQASM log configuration |
| `roles` | str | `"roles.yaml"` | Roles configuration file |
| `post_function` | callable | `None` | Function invoked after the simulation |

Several of these — `formalism`, `network_config`, `log_cfg`, `post_function` —
hold objects rather than scalars and cannot be expressed in YAML. Set them by
constructing a
{class}`~netqmpi.runtime.adapters.netqasm.netqasm_executor.NetQASMRunConfig` in
Python and driving the run through
{func}`~netqmpi.runtime.cli.simulate`, as shown in
[Programmatic use](cli.md#programmatic-use).

## Defining your own

A backend config is a dataclass extending
{class}`~netqmpi.runtime.run_config.RunConfig`:

```python
from dataclasses import dataclass
from typing import Optional
from netqmpi.runtime.run_config import RunConfig


@dataclass
class MyBackendConfig(RunConfig):
    """Backend-specific configuration."""

    shots: int = 1024
    my_parameter: float = 0.5
    seed: Optional[int] = None
```

{meth}`~netqmpi.runtime.run_config.RunConfig.from_dict` maps YAML keys onto the
dataclass fields and rejects unknown ones. Override it if your config has a
nested block to translate, as `QoalaRunConfig` does for `hardware`.
