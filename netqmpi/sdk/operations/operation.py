"""
Abstract base for all quantum operations (Command pattern).
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List


class Operation(ABC):
    """
    Abstract base class for all quantum operations.

    Follows the Command pattern: each subclass encapsulates all the
    information needed to describe a single quantum action, keeping it
    independent of any backend.

    Attributes:
        qubits (List[int]): Qubit indices this operation acts on.
    """

    def __init__(self, qubits: List[int]) -> None:
        """
        Args:
            qubits: Qubit indices this operation acts on.

        Raises:
            TypeError: If *qubits* is not a list of integers.
        """
        if not isinstance(qubits, list) or not all(
            isinstance(q, int) for q in qubits
        ):
            raise TypeError("qubits must be a list of integers.")
        self._qubits: List[int] = list(qubits)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def qubits(self) -> List[int]:
        """Returns a copy of the qubit indices this operation acts on."""
        return list(self._qubits)

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    @abstractmethod
    def __repr__(self) -> str:  # pragma: no cover
        pass

    def __eq__(self, other: object) -> bool:
        return isinstance(other, self.__class__) and self._qubits == other._qubits

    def __hash__(self) -> int:
        return hash((self.__class__.__name__, tuple(self._qubits)))
