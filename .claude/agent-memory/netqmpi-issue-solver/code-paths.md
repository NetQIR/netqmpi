---
name: code-paths
description: Non-obvious NetQMPI code paths and test techniques (config building, plugin discovery, entry-point tests without pip), verified 2026-11-04
metadata:
  type: reference
---

- **Config building** (`netqmpi/runtime/cli.py` `_build_config`, since #29): YAML
  block + CLI overrides (`--shots`, `--plugin`, `--plugin-config`) + flag-fixed
  fields (`netqasm_major=`) are merged into one dict and passed to a single
  `config_cls.from_dict`, so `__post_init__` sees final values. Do not go back to
  setting attributes after construction: validation and plugin `configure` would
  miss them.
- **Plugin support is detected structurally**: a config class with a `plugin`
  dataclass field accepts `--plugin`; others refuse it (usage error, exit 2).
  Plugin discovery and the optional-member helpers live in `netqmpi/plugins.py`
  (backend-agnostic); `_simulation.load_plugin` is only an alias.
- **Entry points across Python 3.8 to 3.12**: iterate `importlib.metadata.distributions()`
  and each `dist.entry_points` (filter on `.group`). Same API everywhere and it
  gives the distribution name and version; `entry_points(group=...)` is 3.10+ only.
- **Testing entry points without pip**: write `<name>-<ver>.dist-info/{METADATA,entry_points.txt}`
  plus the module into a tmp dir and `monkeypatch.syspath_prepend` it
  (`install_plugins` fixture in `test/conftest.py`). Pop the module from
  `sys.modules` on teardown, or the next test reuses it.
- Example plugins in `examples/plugins/*` are imported by tests via `chdir`; a
  module-level attribute added to their entry function (`fn.configure = ...`) is
  how a plain-function plugin opts into the optional members.
