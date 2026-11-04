# Simulation plugins for the NetQASM backend

Two complete plugins, each with a program and a configuration file. A plugin
changes how SquidASM simulates the network without changing NetQMPI or the
program: the configuration file names it, and NetQMPI keeps it active while
SquidASM simulates. The
[plugin guide](https://netqir.github.io/netqmpi/backends/netqasm-plugins.html)
describes the contract, the parts of SquidASM a plugin can replace, and how to
test one.

| Directory | Plugin | Replaces in SquidASM | Shows |
|---|---|---|---|
| [`readout_error/`](readout_error) | `readout_error:readout_error` | The executor (`SubroutineHandler._get_executor_class`) | Measurements report the wrong bit with probabilities `prob_error_0` and `prob_error_1` |
| [`gate_noise/`](gate_noise) | `gate_noise:gate_noise` | The device (`squidasm.sim.network.network.QDevice`) | One- and two-qubit gates with their own depolarising noise and duration |

Run each from its own directory, which is where its plugin is imported from:

```bash
cd readout_error
netqmpi -n 2 app.py --netqasm --config config.yaml   # with the plugin
netqmpi -n 2 app.py --netqasm                        # without it, to compare
```

or install it as a plugin package and select it by name from any directory:

```bash
pip install ./readout_error
netqmpi --list-plugins
netqmpi -n 2 readout_error/app.py --netqasm --plugin readout_error \
        --plugin-config readout_error/params.yaml
```

Each directory holds:

- the plugin, a single module whose entry point the config file names, with
  the optional `configure` (its checks, run before any rank is traced) and
  `backends`;
- `config.yaml`, with the `plugin` and its `plugin_config`;
- `params.yaml`, the same `plugin_config` alone, for `--plugin-config`;
- `pyproject.toml`, which registers the plugin in the `netqmpi.plugins`
  entry-point group when the directory is installed;
- `app.py`, a NetQMPI program whose output shows what the plugin changes.

Both need SquidASM 0.13 and run with NetQASM 2.x (`--netqasm`) or 1.x
(`--netqasm1.0`). Their tests are in `test/test_netqasm_plugin.py`.
