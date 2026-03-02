"""
Unitary gate operations (Command pattern).
"""
from __future__ import annotations
from typing import List, Optional

from netqmpi.sdk.core.operations.operation import Operation


class Gate(Operation):
    """
    Generic unitary single gate (H, X, RZ, U3, …).

    Attributes:
        name   (str):         Standard gate name, normalised to upper-case.
        qubits (List[int]):   Qubits the gate acts on.
        params (List[float]): Optional rotation / angle parameters.
    """

    def __init__(
        self,
        name: str,
        qubits: List[int],
        params: Optional[List[float]] = None,
    ) -> None:
        """
        Args:
            name:   Gate identifier (e.g. ``'H'``, ``'X'``, ``'RZ'``).
            qubits: Qubit indices.
            params: Rotation angles or gate parameters (default: ``[]``).

        Raises:
            ValueError: If *name* is empty.
        """
        super().__init__(qubits)
        if not isinstance(name, str) or not name.strip():
            raise ValueError("name must be a non-empty string.")
        self._name: str = name.upper()
        self._params: List[float] = list(params) if params is not None else []

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        """Gate name in upper-case."""
        return self._name

    @property
    def params(self) -> List[float]:
        """Returns a copy of the gate parameters."""
        return list(self._params)

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        params_str = f", params={self._params}" if self._params else ""
        return f"Gate(name='{self._name}', qubits={self._qubits}{params_str})"

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, Gate)
            and self._name == other._name
            and self._qubits == other._qubits
            and self._params == other._params
        )

    def __hash__(self) -> int:
        return hash((self._name, tuple(self._qubits), tuple(self._params)))


class ControlledGate(Operation):
    """
    Generic controlled gate.

    Wraps one or more target :class:`Gate` objects behind a set of control
    qubits.  The overall qubit list is ``controls + all target qubits``.

    Attributes:
        controls (List[int]):  Control qubit indices.
        targets  (List[Gate]): Gates applied when all controls are |1⟩.
    """

    def __init__(self, controls: List[int], targets: List[Gate]) -> None:
        """
        Args:
            controls: Control qubit indices.
            targets:  :class:`Gate` instances to apply conditionally.

        Raises:
            ValueError: If *controls* or *targets* are empty.
            TypeError:  If any element of *targets* is not a :class:`Gate`.
        """
        if not isinstance(controls, list) or not controls:
            raise ValueError("controls must be a non-empty list of integers.")
        if not isinstance(targets, list) or not targets:
            raise ValueError("targets must be a non-empty list of Gate instances.")
        if not all(isinstance(t, Gate) for t in targets):
            raise TypeError("Every element in targets must be a Gate instance.")

        target_qubits = [q for gate in targets for q in gate.qubits]
        super().__init__(controls + target_qubits)
        self._controls: List[int] = list(controls)
        self._targets: List[Gate] = list(targets)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def controls(self) -> List[int]:
        """Control qubit indices."""
        return list(self._controls)

    @property
    def targets(self) -> List[Gate]:
        """Target gates applied under control."""
        return list(self._targets)

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"ControlledGate(controls={self._controls}, targets={self._targets})"
        )

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, ControlledGate)
            and self._controls == other._controls
            and self._targets == other._targets
        )

    def __hash__(self) -> int:
        return hash((tuple(self._controls), tuple(self._targets)))


class ClassicalControlledGate(Operation):
    """
    Gate conditioned on the value of one or more classical bits.

    Unlike :class:`ControlledGate` (whose controls are qubits),
    here the controls are *classical bit* indices — typically the
    results of prior measurements.  The gate fires when **all** listed
    cbits equal 1.

    The ``qubits`` property returns only the target qubits (the cbits are
    classical and therefore not part of the quantum register).

    Attributes:
        cbits   (List[int]):  Classical bit indices used as conditions.
        targets (List[Gate]): Gates applied when all conditions are satisfied.
    """

    def __init__(self, cbits: List[int], targets: List[Gate]) -> None:
        """
        Args:
            cbits:   Classical bit indices acting as conditions.
            targets: :class:`Gate` instances to apply when all cbits are 1.

        Raises:
            ValueError: If *cbits* or *targets* are empty.
            TypeError:  If any element of *targets* is not a :class:`Gate`.
        """
        if not isinstance(cbits, list) or not cbits:
            raise ValueError("cbits must be a non-empty list of integers.")
        if not all(isinstance(c, int) for c in cbits):
            raise TypeError("Every element in cbits must be an integer.")
        if not isinstance(targets, list) or not targets:
            raise ValueError("targets must be a non-empty list of Gate instances.")
        if not all(isinstance(t, Gate) for t in targets):
            raise TypeError("Every element in targets must be a Gate instance.")

        target_qubits = [q for gate in targets for q in gate.qubits]
        super().__init__(target_qubits)
        self._cbits: List[int] = list(cbits)
        self._targets: List[Gate] = list(targets)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def cbits(self) -> List[int]:
        """Classical bit indices used as conditions."""
        return list(self._cbits)

    @property
    def targets(self) -> List[Gate]:
        """Target gates applied when all conditions are satisfied."""
        return list(self._targets)

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"ClassicalControlledGate(cbits={self._cbits}, targets={self._targets})"
        )

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, ClassicalControlledGate)
            and self._cbits == other._cbits
            and self._targets == other._targets
        )

    def __hash__(self) -> int:
        return hash((tuple(self._cbits), tuple(self._targets)))
