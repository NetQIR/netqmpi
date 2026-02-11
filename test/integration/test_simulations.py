"""
Integration tests for NetQMPI examples using the SquidASM simulator.

These tests run full quantum simulations and require:
- SquidASM installed (conda activate squidasm)
- NetSquid available as backend

Run only integration tests:  pytest -m integration
Skip integration tests:      pytest -m "not integration"
"""
import os
import pytest

# Skip entire module if SquidASM is not available
squidasm = pytest.importorskip("squidasm", reason="SquidASM required for integration tests")

from netqmpi.runtime.cli import simulate, NetQASMConfig

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXAMPLES_DIR = os.path.join(PROJECT_ROOT, "examples", "netqmpi")


def _make_config() -> NetQASMConfig:
    """Create a NetQASMConfig suitable for testing (logging disabled)."""
    config = NetQASMConfig()
    config.enable_logging = False
    return config


@pytest.mark.integration
class TestSendRecv:
    """Integration tests for point-to-point qubit teleportation."""

    def test_two_nodes_completes(self):
        """send_recv with 2 processes should complete without errors."""
        script = os.path.join(EXAMPLES_DIR, "send_recv.py")
        simulate(script=script, num_procs=2, configuration=_make_config())


# @pytest.mark.integration
# class TestScatter:
#     """Integration tests for quantum scatter operation."""

#     def test_scatter_two_nodes(self):
#         """scatter with 2 processes should complete without errors."""
#         script = os.path.join(EXAMPLES_DIR, "scatter.py")
#         simulate(script=script, num_procs=2, configuration=_make_config())

#     def test_scatter_three_nodes(self):
#         """scatter with 3 processes should complete without errors."""
#         script = os.path.join(EXAMPLES_DIR, "scatter.py")
#         simulate(script=script, num_procs=3, configuration=_make_config())


# @pytest.mark.integration
# class TestGather:
#     """Integration tests for quantum gather operation."""

#     def test_gather_two_nodes(self):
#         """gather with 2 processes should complete without errors."""
#         script = os.path.join(EXAMPLES_DIR, "gather.py")
#         simulate(script=script, num_procs=2, configuration=_make_config())


@pytest.mark.integration
class TestRoundRobin:
    """Integration tests for round-robin qubit forwarding."""

    def test_roundrobin_two_nodes(self):
        """roundrobin with 2 processes should complete without errors."""
        script = os.path.join(EXAMPLES_DIR, "roundrobin.py")
        simulate(script=script, num_procs=2, configuration=_make_config())

    def test_roundrobin_three_nodes(self):
        """roundrobin with 3 processes should complete without errors."""
        script = os.path.join(EXAMPLES_DIR, "roundrobin.py")
        simulate(script=script, num_procs=3, configuration=_make_config())
