"""
End-to-end test for the Qoala backend (simulation only).

Reuses the "distributed superposition" example
(``examples/netqmpi/send_recv.py``): rank 0 prepares an ``H`` superposition and
teleports it to rank 1 with ``qsend``/``qrecv``; rank 1 measures. The same
``app.py`` is used unchanged for every backend.

This test only runs where the Qoala stack is installed (Python 3.10-3.12 with
``netqasm >= 2.0``). It is skipped otherwise, so it does not disturb the
NetQASM/``squidasm`` environment. Run it inside the ``qoala`` conda env:

    conda run -n qoala pytest test/test_qoala_backend.py -m integration
"""
import ast
import re
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("qoala", reason="Qoala backend requires the qoala-sim package.")

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = REPO_ROOT / "examples" / "netqmpi" / "send_recv.py"


@pytest.mark.integration
def test_qoala_distributed_superposition():
    """The H-then-teleport example runs on Qoala and yields a valid histogram."""
    shots = 20
    proc = subprocess.run(
        [sys.executable, "-m", "netqmpi.runtime.cli",
         "-n", "2", str(EXAMPLE), "--qoala", "--shots", str(shots)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=600,
    )

    assert proc.returncode == 0, f"CLI failed:\n{proc.stdout}\n{proc.stderr}"
    # Rank 0 (sender) completes the teleportation.
    assert "teleportation complete" in proc.stdout, proc.stdout

    # Rank 1 (receiver) prints its measurement histogram.
    match = re.search(r"measure:\s*(\{.*\})", proc.stdout)
    assert match, f"no measurement histogram in output:\n{proc.stdout}"

    counts = ast.literal_eval(match.group(1))
    assert set(counts).issubset({"0", "1"}), counts
    assert sum(counts.values()) == shots, counts


def _receiver_p0(hw, shots):
    """Run the X-basis probe in-process and return P(outcome == 0)."""
    import contextlib
    import io

    from netqmpi.runtime.adapters.qoala import QoalaExecutorAdapter, QoalaRunConfig

    app = REPO_ROOT / "scripts" / "experiments" / "apps" / "dist_superposition_xbasis.py"
    config = QoalaRunConfig(shots=shots, hw_config=hw, seed=1)
    executor = QoalaExecutorAdapter(2, config=config)
    apps = executor.build_apps(str(app), size=2)

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        executor.run(apps)
    counts = ast.literal_eval(re.search(r"measure:\s*(\{.*\})", buf.getvalue()).group(1))
    return counts.get("0", 0) / sum(counts.values())


@pytest.mark.integration
def test_qoala_hardware_params_propagate():
    """qdevice noise must flow through the CircuitAdapter, not be dropped.

    Perfect hardware teleports |+> to a deterministic X-basis 0 (fidelity 1);
    a fully depolarising single-qubit gate collapses it to ~0.5.
    """
    from netqmpi.runtime.adapters.qoala import QoalaQDeviceConfig

    perfect = _receiver_p0(QoalaQDeviceConfig(), shots=200)
    assert perfect == 1.0, perfect

    depolarised = _receiver_p0(
        QoalaQDeviceConfig(single_qubit_gate_depolar_prob=0.5), shots=400
    )
    assert 0.4 <= depolarised <= 0.6, depolarised


@pytest.mark.integration
def test_qoala_config_file(tmp_path):
    """The unified --config YAML feeds backend params (here, EPR fidelity).

    Uses the X-basis probe (noise-free fidelity 1.0): a maximally mixed link
    (fidelity 0.25) collapses the teleported |+> to ~0.5, proving the config
    value reaches the backend.
    """
    probe = REPO_ROOT / "scripts" / "experiments" / "apps" / "dist_superposition_xbasis.py"
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text("qoala:\n  link_fidelity: 0.25\n")

    shots = 400
    proc = subprocess.run(
        [sys.executable, "-m", "netqmpi.runtime.cli",
         "-n", "2", str(probe), "--qoala", "--config", str(cfg), "--shots", str(shots)],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert proc.returncode == 0, f"CLI failed:\n{proc.stdout}\n{proc.stderr}"

    counts = ast.literal_eval(re.search(r"measure:\s*(\{.*\})", proc.stdout).group(1))
    assert sum(counts.values()) == shots, counts
    p0 = counts.get("0", 0) / shots
    assert 0.4 <= p0 <= 0.6, p0  # ~0.5, not the noise-free 1.0
