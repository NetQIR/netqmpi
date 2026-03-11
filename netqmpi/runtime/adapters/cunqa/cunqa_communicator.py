"""
Concrete :class:`~netqmpi.sdk.communicator.base.BaseCommunicator` backed
by the CUNQA runtime.

This communicator adapts the backend-specific communication layer to the
backend-agnostic :class:`QMPICommunicator` interface.
"""
from __future__ import annotations

from netqmpi.sdk import QMPICommunicator

class CunqaCommunicator(QMPICommunicator):
    """
    CUNQA-backed communicator for a single rank.

    This class provides the communicator implementation used by the CUNQA
    backend and is injected into :class:`~netqmpi.sdk.communicator.QMPICommunicator`.

    Args:
        rank: Numeric index of the current rank.
        size: Total number of ranks in the communicator.
    """

    def __init__(self, rank: int, size: int) -> None:
        """
        Initialize the communicator.

        Args:
            rank: Numeric index of the current rank.
            size: Total number of ranks in the communicator.
        """
        super().__init__(rank, size)

    def __enter__(self) -> None:
        """
        Enter the runtime context for the communicator.

        Returns:
            None.
        """
        return None

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """
        Exit the runtime context for the communicator.

        Args:
            exc_type: Exception type, if an exception was raised.
            exc_val: Exception instance, if an exception was raised.
            exc_tb: Traceback, if an exception was raised.

        Returns:
            None.
        """
        return None