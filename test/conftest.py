import os
import pytest
from unittest.mock import MagicMock, patch


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXAMPLES_DIR = os.path.join(PROJECT_ROOT, "examples", "netqmpi")


@pytest.fixture
def mock_app_config():
    """Create a mock app_config matching NetQASM's ApplicationConfig interface."""
    config = MagicMock()
    config.app_name = "rank_0"
    config.log_config = None
    return config


@pytest.fixture
def netqasm_patches():
    """Patch all NetQASM dependencies used by QMPICommunicator for the duration of a test.

    Yields a dict with keys: 'NetQASMConnection', 'EPRSocket', 'Socket', 'Qubit'
    so tests can assert on calls to the mocked classes.
    """
    with patch('netqmpi.sdk.communicator.communicator.NetQASMConnection') as mock_conn, \
         patch('netqmpi.sdk.communicator.communicator.EPRSocket') as mock_epr, \
         patch('netqmpi.sdk.communicator.communicator.Socket') as mock_socket, \
         patch('netqmpi.sdk.communicator.communicator.Qubit') as mock_qubit:
        # Make EPRSocket return a unique mock per call so sockets are distinguishable
        mock_epr.side_effect = lambda name: MagicMock(name=f"EPRSocket_{name}")

        yield {
            'NetQASMConnection': mock_conn,
            'EPRSocket': mock_epr,
            'Socket': mock_socket,
            'Qubit': mock_qubit,
        }


@pytest.fixture
def make_communicator(netqasm_patches):
    """Factory fixture to create a QMPICommunicator with mocked NetQASM deps.

    Patches remain active for the entire test, so methods like create_qubit()
    and get_socket() will also use the mocked classes.
    """
    from netqmpi.sdk.communicator import QMPICommunicator

    def _make(rank=0, size=3):
        config = MagicMock()
        config.app_name = f"rank_{rank}"
        config.log_config = None
        return QMPICommunicator(rank=rank, size=size, app_config=config)

    return _make


@pytest.fixture
def examples_dir():
    """Return absolute path to the examples/netqmpi directory."""
    return EXAMPLES_DIR
