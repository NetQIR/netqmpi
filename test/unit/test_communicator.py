import pytest
from unittest.mock import MagicMock, patch


class TestCommunicatorInit:
    """Tests for QMPICommunicator.__init__ setup."""

    def test_stores_rank_and_size(self, make_communicator):
        comm = make_communicator(rank=1, size=4)
        assert comm.rank == 1
        assert comm.size == 4

    def test_creates_epr_sockets_for_other_ranks(self, netqasm_patches, make_communicator):
        comm = make_communicator(rank=0, size=3)
        # EPRSocket called for rank_1 and rank_2 (not self)
        assert netqasm_patches['EPRSocket'].call_count == 2

    def test_creates_connection_with_app_name(self, netqasm_patches, make_communicator):
        comm = make_communicator(rank=0, size=2)
        call_kwargs = netqasm_patches['NetQASMConnection'].call_args
        assert call_kwargs.kwargs['app_name'] == "rank_0"

    def test_single_process_no_epr_sockets(self, netqasm_patches, make_communicator):
        comm = make_communicator(rank=0, size=1)
        assert netqasm_patches['EPRSocket'].call_count == 0
        assert len(comm.epr_sockets_list) == 0

    def test_initializes_empty_exposed_qubits(self, make_communicator):
        comm = make_communicator(rank=0, size=2)
        assert comm.qubits_exposed == []

    def test_ghz_qubit_initially_none(self, make_communicator):
        comm = make_communicator(rank=0, size=2)
        assert comm.ghz_qubit is None


class TestRankArithmetic:
    """Tests for get_next_rank / get_prev_rank navigation."""

    def test_next_rank_increments(self, make_communicator):
        comm = make_communicator(rank=0, size=4)
        assert comm.get_next_rank(0) == 1
        assert comm.get_next_rank(1) == 2
        assert comm.get_next_rank(2) == 3

    def test_next_rank_wraps_around(self, make_communicator):
        comm = make_communicator(rank=0, size=4)
        assert comm.get_next_rank(3) == 0

    def test_prev_rank_decrements(self, make_communicator):
        comm = make_communicator(rank=0, size=4)
        assert comm.get_prev_rank(3) == 2
        assert comm.get_prev_rank(2) == 1

    def test_prev_rank_wraps_around(self, make_communicator):
        comm = make_communicator(rank=0, size=4)
        assert comm.get_prev_rank(0) == 3

    def test_next_and_prev_are_inverses(self, make_communicator):
        comm = make_communicator(rank=0, size=5)
        for r in range(5):
            assert comm.get_prev_rank(comm.get_next_rank(r)) == r
            assert comm.get_next_rank(comm.get_prev_rank(r)) == r

    def test_two_processes_toggle(self, make_communicator):
        comm = make_communicator(rank=0, size=2)
        assert comm.get_next_rank(0) == 1
        assert comm.get_next_rank(1) == 0
        assert comm.get_prev_rank(0) == 1
        assert comm.get_prev_rank(1) == 0

    @pytest.mark.parametrize("size", [1, 2, 5, 10])
    def test_next_rank_always_in_range(self, make_communicator, size):
        comm = make_communicator(rank=0, size=size)
        for r in range(size):
            nxt = comm.get_next_rank(r)
            assert 0 <= nxt < size

    @pytest.mark.parametrize("size", [1, 2, 5, 10])
    def test_prev_rank_always_in_range(self, make_communicator, size):
        comm = make_communicator(rank=0, size=size)
        for r in range(size):
            prv = comm.get_prev_rank(r)
            assert 0 <= prv < size


class TestContextManager:
    """Tests for __enter__ / __exit__ protocol."""

    def test_enter_returns_self(self, make_communicator):
        comm = make_communicator(rank=0, size=2)
        result = comm.__enter__()
        assert result is comm

    def test_enter_delegates_to_connection(self, make_communicator):
        comm = make_communicator(rank=0, size=2)
        comm.__enter__()
        comm.connection.__enter__.assert_called_once()

    def test_exit_delegates_to_connection(self, make_communicator):
        comm = make_communicator(rank=0, size=2)
        comm.__exit__(None, None, None)
        comm.connection.__exit__.assert_called_once_with(None, None, None)

    def test_exit_propagates_exception_info(self, make_communicator):
        comm = make_communicator(rank=0, size=2)
        exc_type, exc_val, exc_tb = ValueError, ValueError("test"), None
        comm.__exit__(exc_type, exc_val, exc_tb)
        comm.connection.__exit__.assert_called_once_with(exc_type, exc_val, exc_tb)

    def test_usable_as_with_statement(self, make_communicator):
        comm = make_communicator(rank=0, size=2)
        with comm as c:
            assert c is comm


class TestFlush:
    """Tests for the flush() wrapper."""

    def test_flush_delegates_to_connection(self, make_communicator):
        comm = make_communicator(rank=0, size=2)
        comm.flush()
        comm.connection.flush.assert_called_once()

    def test_multiple_flushes(self, make_communicator):
        comm = make_communicator(rank=0, size=2)
        for _ in range(3):
            comm.flush()
        assert comm.connection.flush.call_count == 3


class TestCreateQubit:
    """Tests for the create_qubit() factory."""

    def test_creates_qubit_with_own_connection(self, netqasm_patches, make_communicator):
        comm = make_communicator(rank=0, size=2)
        comm.create_qubit()
        netqasm_patches['Qubit'].assert_called_once_with(comm.connection)

    def test_returns_the_created_qubit(self, netqasm_patches, make_communicator):
        comm = make_communicator(rank=0, size=2)
        q = comm.create_qubit()
        assert q is netqasm_patches['Qubit'].return_value

    def test_multiple_qubits_are_independent_calls(self, netqasm_patches, make_communicator):
        comm = make_communicator(rank=0, size=2)
        q1 = comm.create_qubit()
        q2 = comm.create_qubit()
        assert netqasm_patches['Qubit'].call_count == 2


class TestSocketManagement:
    """Tests for get_socket lazy creation and caching."""

    def test_get_socket_creates_on_first_call(self, netqasm_patches, make_communicator):
        comm = make_communicator(rank=0, size=3)
        s = comm.get_socket(0, 1)
        netqasm_patches['Socket'].assert_called_once_with("rank_0", "rank_1")

    def test_get_socket_caches_on_subsequent_calls(self, netqasm_patches, make_communicator):
        comm = make_communicator(rank=0, size=3)
        s1 = comm.get_socket(0, 1)
        s2 = comm.get_socket(0, 1)
        netqasm_patches['Socket'].assert_called_once()
        assert s1 is s2

    def test_different_peers_get_different_sockets(self, netqasm_patches, make_communicator):
        comm = make_communicator(rank=0, size=3)
        s1 = comm.get_socket(0, 1)
        s2 = comm.get_socket(0, 2)
        assert netqasm_patches['Socket'].call_count == 2


class TestEPRSocketLookup:
    """Tests for get_epr_socket."""

    def test_returns_existing_epr_socket(self, make_communicator):
        comm = make_communicator(rank=0, size=3)
        # rank 0 should have EPR sockets for rank 1 and rank 2
        epr = comm.get_epr_socket(0, 1)
        assert epr is not None

    def test_raises_for_nonexistent_pair(self, make_communicator):
        comm = make_communicator(rank=0, size=3)
        # rank 1's EPR sockets were not initialized (only rank 0's were)
        with pytest.raises(Exception, match="does not exist"):
            comm.get_epr_socket(1, 2)


class TestGetters:
    """Tests for simple getter methods."""

    def test_get_rank(self, make_communicator):
        comm = make_communicator(rank=2, size=5)
        assert comm.get_rank() == 2

    def test_get_size(self, make_communicator):
        comm = make_communicator(rank=0, size=7)
        assert comm.get_size() == 7

    def test_get_epr_sockets_list_length(self, make_communicator):
        comm = make_communicator(rank=0, size=4)
        # Should have EPR sockets for ranks 1, 2, 3
        assert len(comm.get_epr_sockets_list()) == 3

    def test_get_epr_sockets_list_empty_for_single_process(self, make_communicator):
        comm = make_communicator(rank=0, size=1)
        assert comm.get_epr_sockets_list() == []
