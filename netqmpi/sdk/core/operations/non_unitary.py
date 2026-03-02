"""
Non-unitary quantum operations: Measure, Reset, Barrier.
"""
from __future__ import annotations
from typing import List, Optional

from netqmpi.sdk.core.operations.gate import Gate
from netqmpi.sdk.core.operations.operation import Operation


class Measure(Operation):
    """
    Measurement — collapses a qubit and stores the outcome in a classical bit.

    Attributes:
        qubit (int): Qubit index to measure.
        cbit  (int): Classical bit index that receives the result.
    """

    def __init__(self, qubit: int, cbit: int) -> None:
        """
        Args:
            qubit: Qubit index to measure.
            cbit:  Classical bit index for the result.
        """
        super().__init__([qubit])
        self._cbit: int = cbit

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def qubit(self) -> int:
        """Qubit index."""
        return self._qubits[0]

    @property
    def cbit(self) -> int:
        """Classical bit index."""
        return self._cbit

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"Measure(qubit={self.qubit}, cbit={self._cbit})"

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, Measure)
            and self._qubits == other._qubits
            and self._cbit == other._cbit
        )

    def __hash__(self) -> int:
        return hash(("Measure", self.qubit, self._cbit))


class Reset(Operation):
    """
    Reset — unconditionally sets a qubit back to |0⟩.

    Attributes:
        qubit (int): Qubit index to reset.
    """

    def __init__(self, qubit: int) -> None:
        """
        Args:
            qubit: Qubit index to reset.
        """
        super().__init__([qubit])

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def qubit(self) -> int:
        """Qubit index."""
        return self._qubits[0]

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"Reset(qubit={self.qubit})"

    def __hash__(self) -> int:
        return hash(("Reset", self.qubit))


class Barrier(Operation):
    """
    Barrier — prevents the compiler from re-ordering operations across it.

    An empty qubit list means the barrier spans the whole circuit.

    Attributes:
        qubits (List[int]): Qubits the barrier spans.
    """

    def __init__(self, qubits: Optional[List[int]] = None) -> None:
        """
        Args:
            qubits: Qubits the barrier spans.  Defaults to ``[]``
                    (full-circuit barrier).
        """
        super().__init__(qubits if qubits is not None else [])

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"Barrier(qubits={self._qubits})"

    def __hash__(self) -> int:
        return hash(("Barrier", tuple(self._qubits)))
