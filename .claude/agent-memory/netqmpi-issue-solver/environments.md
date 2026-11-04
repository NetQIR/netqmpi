---
name: environments
description: How to run each NetQMPI backend's tests on this machine (conda envs, Python versions, docs venv, CUNQA container caveats), verified 2026-11-04
metadata:
  type: reference
---

Verified 2026-11-04 (issue #29) by importing each package with the env's own interpreter.

- **`conda run -n <env>` is unreliable here**: the shell profile prepends
  `miniconda3/envs/netqasm2/bin` to `PATH`, so every `conda run` resolved to the
  netqasm2 Python. Call `~/miniconda3/envs/<env>/bin/python -m pytest ...` (or
  `.../bin/netqmpi`) directly. Check with `python -c "import sys; print(sys.executable)"`.
- Inventory (differs from the agent file's table):
  - `qiskit-1-3`: Python 3.10, netqmpi editable, qiskit-aer, pytest; **no netqasm**.
  - `squidasm`: **Python 3.8**, NetQASM 1.0.0, squidasm 0.13.4, netsquid 1.1.7,
    qiskit-aer, pytest. Use it for `--netqasm1.0` and Python 3.8 compatibility.
  - `netqasm2`: Python 3.11, NetQASM 2.0, squidasm 0.13.6, **netqmpi editable
    installed**, pytest; no Qiskit.
  - `qoala`: Python 3.11, qoala-sim + netsquid; netqmpi not installed, but
    `pytest.ini` (`pythonpath = .`) makes the repo importable, so
    `envs/qoala/bin/python -m pytest <repo>/test/test_qoala_*.py` works.
  - `quantum`: Python 3.10, qiskit-aer only; **no pytest, no pyyaml**. Not usable for tests.
- Docs: no conda env has Sphinx; the git-ignored `.venv-docs/` (Sphinx 8.1.3)
  builds them: `.venv-docs/bin/python -m sphinx -b html -W --keep-going docs/source <out>`.
- CUNQA container `netqmpi-issue-dev`: pass `-e PYTHONDONTWRITEBYTECODE=1` and
  `-p no:cacheprovider`. Root-owned `cpython-312` `__pycache__` files from earlier
  runs already sit in `/work` (git-ignored). Other agent worktrees exist under
  `.claude/worktrees/`, so the container may be in use by parallel work.
- `pip install --no-deps --no-build-isolation --no-index --target <scratch> <dir>`
  installs a local package without touching an env, but leaves `build/` (not
  git-ignored) and `*.egg-info` in the source dir; delete them afterwards.
