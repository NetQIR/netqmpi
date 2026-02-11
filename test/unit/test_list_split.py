import pytest

from netqmpi.sdk.primitives.collective.collective import list_split


class TestListSplitBasic:
    """Core behavior of list_split."""

    def test_even_split(self):
        assert list_split([1, 2, 3, 4], 2) == [[1, 2], [3, 4]]

    def test_uneven_split_remainder_goes_to_first_chunks(self):
        assert list_split([1, 2, 3, 4, 5], 2) == [[1, 2, 3], [4, 5]]

    def test_split_into_single_chunk(self):
        assert list_split([1, 2, 3], 1) == [[1, 2, 3]]

    def test_split_equal_to_length(self):
        assert list_split([1, 2, 3], 3) == [[1], [2], [3]]

    def test_more_chunks_than_elements(self):
        assert list_split([1, 2], 4) == [[1], [2], [], []]

    def test_empty_list(self):
        assert list_split([], 3) == [[], [], []]

    def test_single_element_multiple_chunks(self):
        assert list_split([42], 3) == [[42], [], []]


class TestListSplitInvariants:
    """Properties that must always hold regardless of inputs."""

    @pytest.mark.parametrize("n_items,n_chunks", [
        (0, 1), (1, 1), (5, 3), (10, 4), (100, 7), (3, 10),
    ])
    def test_produces_correct_number_of_chunks(self, n_items, n_chunks):
        result = list_split(list(range(n_items)), n_chunks)
        assert len(result) == n_chunks

    @pytest.mark.parametrize("n_items,n_chunks", [
        (5, 2), (10, 3), (7, 4), (1, 5), (0, 3),
    ])
    def test_all_elements_preserved(self, n_items, n_chunks):
        data = list(range(n_items))
        result = list_split(data, n_chunks)
        flat = [item for chunk in result for item in chunk]
        assert flat == data

    @pytest.mark.parametrize("n_items,n_chunks", [
        (10, 3), (11, 4), (7, 3), (100, 7),
    ])
    def test_chunk_sizes_differ_by_at_most_one(self, n_items, n_chunks):
        result = list_split(list(range(n_items)), n_chunks)
        sizes = [len(c) for c in result]
        assert max(sizes) - min(sizes) <= 1

    def test_preserves_original_order(self):
        data = list(range(20))
        result = list_split(data, 6)
        flat = [item for chunk in result for item in chunk]
        assert flat == data

    @pytest.mark.parametrize("n_items,n_chunks", [
        (5, 3), (10, 3), (7, 2),
    ])
    def test_larger_chunks_come_first(self, n_items, n_chunks):
        """Remainder elements are distributed to the first chunks."""
        result = list_split(list(range(n_items)), n_chunks)
        sizes = [len(c) for c in result]
        # Sizes should be non-increasing
        assert sizes == sorted(sizes, reverse=True)
