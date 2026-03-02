"""
Base abstraction for quantum circuits.

Defines the contract that all circuit adapters must follow.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Iterator, List, Optional

from netqmpi.sdk.core.operations.qmpi import (
    Expose, QGather, QRecv, QScatter, QSend, Unexpose,
)
from netqmpi.sdk.core.operations.container import OperationContainer
from netqmpi.sdk.core.operations.gate import ControlledGate, Gate
from netqmpi.sdk.core.operations.non_unitary import Barrier, Measure, Reset
from netqmpi.sdk.core.operations.operation import Operation


class _ExposeContext:
    """
    Context manager returned by :meth:`Circuit.expose`.

    On entry  → appends :class:`~netqmpi.sdk.core.operations.Expose` to the circuit.
    On exit   → appends the matching :class:`~netqmpi.sdk.core.operations.Unexpose`
                automatically, even if the body raises an exception.

    Usage::

        with circuit.expose([0, 1], rank=0):
            circuit.h(0).cx(0, 1)
        # Unexpose(rank=0) has been added here
    """

    def __init__(self, circuit: "Circuit", qubits: List[int], rank: int) -> None:
        self._circuit = circuit
        self._qubits = qubits
        self._rank = rank

    def __enter__(self) -> "Circuit":
        """Appends Expose and returns the circuit for further chaining."""
        self._circuit._add(Expose(self._qubits, self._rank))
        return self._circuit

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Appends Unexpose regardless of whether the body raised."""
        self._circuit._add(Unexpose(self._rank))
        return False  # never suppress exceptions


class Circuit(ABC):
    """
    Abstract class representing a quantum circuit.

    Provides:
    - An :class:`~netqmpi.sdk.core.operations.OperationContainer` that
      stores operations following the Composite pattern.
    - A fluent gate API (``h``, ``cx``, ``rx``, ``measure``, …) that appends
      operations to the container and returns *self* for chaining.
    - Abstract hooks :meth:`translate` and :meth:`build` that each backend
      adapter must implement to convert the generic operations into
      backend-native instructions.

    Attributes:
        num_qubits (int): Number of qubits in the circuit.
        num_clbits (int): Number of classical bits in the circuit.
    """

    def __init__(self, num_qubits: int, num_clbits: int) -> None:
        """
        Args:
            num_qubits: Number of qubits in the circuit.
            num_clbits: Number of classical bits in the circuit.
        """
        self._num_qubits = num_qubits
        self._num_clbits = num_clbits
        self._ops = OperationContainer()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def num_qubits(self) -> int:
        """Returns the number of qubits in the circuit."""
        return self._num_qubits

    @property
    def num_clbits(self) -> int:
        """Returns the number of classical bits in the circuit."""
        return self._num_clbits

    @property
    def ops(self) -> OperationContainer:
        """Root :class:`~netqmpi.sdk.core.operations.OperationContainer`."""
        return self._ops

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def operations_supported(self) -> List[str]:
        """
        Returns the list of operations supported by this circuit.

        Returns:
            List of supported quantum operation names,
            e.g. ``['H', 'CX', 'RZ', 'measure', 'reset']``.
        """

    @abstractmethod
    def translate(self, op: Operation) -> Any:
        """
        Translate a generic :class:`~netqmpi.sdk.core.operations.Operation`
        into a backend-native instruction.

        Args:
            op: The operation to translate.

        Returns:
            A backend-specific object (gate call, instruction, …).
        """

    @abstractmethod
    def build(self) -> Any:
        """
        Materialise the circuit for the backend by translating every
        operation in :attr:`ops` via :meth:`translate`.

        Returns:
            A backend-native circuit object ready for execution.
        """

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_qubit(self, qubit: int) -> None:
        if not (0 <= qubit < self._num_qubits):
            raise IndexError(
                f"Qubit index {qubit} out of range [0, {self._num_qubits}).")

    def _check_cbit(self, cbit: int) -> None:
        if not (0 <= cbit < self._num_clbits):
            raise IndexError(
                f"Classical bit index {cbit} out of range [0, {self._num_clbits}).")

    def _add(self, op: Operation) -> Circuit:
        """Append *op* and return *self* for chaining."""
        self._ops.add(op)
        return self

    # ------------------------------------------------------------------
    # Fluent gate API — single-qubit gates
    # ------------------------------------------------------------------

    def h(self, qubit: int) -> Circuit:
        """Hadamard gate on *qubit*."""
        self._check_qubit(qubit)
        return self._add(Gate('H', [qubit]))

    def x(self, qubit: int) -> Circuit:
        """Pauli-X (NOT) gate on *qubit*."""
        self._check_qubit(qubit)
        return self._add(Gate('X', [qubit]))

    def y(self, qubit: int) -> Circuit:
        """Pauli-Y gate on *qubit*."""
        self._check_qubit(qubit)
        return self._add(Gate('Y', [qubit]))

    def z(self, qubit: int) -> Circuit:
        """Pauli-Z gate on *qubit*."""
        self._check_qubit(qubit)
        return self._add(Gate('Z', [qubit]))

    def s(self, qubit: int) -> Circuit:
        """S (phase) gate on *qubit*."""
        self._check_qubit(qubit)
        return self._add(Gate('S', [qubit]))

    def sdg(self, qubit: int) -> Circuit:
        """S† (conjugate phase) gate on *qubit*."""
        self._check_qubit(qubit)
        return self._add(Gate('SDG', [qubit]))

    def t(self, qubit: int) -> Circuit:
        """T gate on *qubit*."""
        self._check_qubit(qubit)
        return self._add(Gate('T', [qubit]))

    def tdg(self, qubit: int) -> Circuit:
        """T† (conjugate T) gate on *qubit*."""
        self._check_qubit(qubit)
        return self._add(Gate('TDG', [qubit]))

    # ------------------------------------------------------------------
    # Fluent gate API — parametric single-qubit gates
    # ------------------------------------------------------------------

    def rx(self, theta: float, qubit: int) -> Circuit:
        """Rotation around X axis by *theta* radians on *qubit*."""
        self._check_qubit(qubit)
        return self._add(Gate('RX', [qubit], [theta]))

    def ry(self, theta: float, qubit: int) -> Circuit:
        """Rotation around Y axis by *theta* radians on *qubit*."""
        self._check_qubit(qubit)
        return self._add(Gate('RY', [qubit], [theta]))

    def rz(self, theta: float, qubit: int) -> Circuit:
        """Rotation around Z axis by *theta* radians on *qubit*."""
        self._check_qubit(qubit)
        return self._add(Gate('RZ', [qubit], [theta]))

    # ------------------------------------------------------------------
    # Fluent gate API — two-qubit gates
    # ------------------------------------------------------------------

    def cx(self, control: int, target: int) -> Circuit:
        """CNOT gate: *control* → *target*."""
        self._check_qubit(control)
        self._check_qubit(target)
        return self._add(ControlledGate([control], [Gate('X', [target])]))

    def cz(self, control: int, target: int) -> Circuit:
        """CZ gate: *control* → *target*."""
        self._check_qubit(control)
        self._check_qubit(target)
        return self._add(ControlledGate([control], [Gate('Z', [target])]))

    def swap(self, qubit1: int, qubit2: int) -> Circuit:
        """SWAP gate between *qubit1* and *qubit2*."""
        self._check_qubit(qubit1)
        self._check_qubit(qubit2)
        return self._add(Gate('SWAP', [qubit1, qubit2]))

    def crz(self, theta: float, control: int, target: int) -> Circuit:
        """Controlled-RZ gate."""
        self._check_qubit(control)
        self._check_qubit(target)
        return self._add(ControlledGate([control], [Gate('RZ', [target], [theta])]))

    # ------------------------------------------------------------------
    # Fluent gate API — three-qubit gates
    # ------------------------------------------------------------------

    def ccx(self, control1: int, control2: int, target: int) -> Circuit:
        """Toffoli (CCX) gate."""
        self._check_qubit(control1)
        self._check_qubit(control2)
        self._check_qubit(target)
        return self._add(ControlledGate([control1, control2], [Gate('X', [target])]))

    # ------------------------------------------------------------------
    # Fluent API — non-unitary operations
    # ------------------------------------------------------------------

    def measure(self, qubit: int, cbit: int) -> Circuit:
        """Measure *qubit* into classical bit *cbit*."""
        self._check_qubit(qubit)
        self._check_cbit(cbit)
        return self._add(Measure(qubit, cbit))

    def measure_all(self) -> Circuit:
        """Measure every qubit into the classical bit of the same index."""
        if self._num_clbits < self._num_qubits:
            raise ValueError(
                "Not enough classical bits to measure all qubits "
                f"({self._num_clbits} clbits < {self._num_qubits} qubits)."
            )
        for i in range(self._num_qubits):
            self._add(Measure(i, i))
        return self

    def reset(self, qubit: int) -> Circuit:
        """Reset *qubit* to |0⟩."""
        self._check_qubit(qubit)
        return self._add(Reset(qubit))

    def barrier(self, qubits: Optional[List[int]] = None) -> Circuit:
        """
        Insert a barrier.

        Args:
            qubits: Qubits to barrier.  ``None`` (default) = full-circuit barrier.
        """
        if qubits is not None:
            for q in qubits:
                self._check_qubit(q)
        return self._add(Barrier(qubits))

    # ------------------------------------------------------------------
    # Fluent API — inter-rank communication primitives
    # ------------------------------------------------------------------

    def qsend(self, qubits: List[int], dest_rank: int) -> Circuit:
        """
        Send *qubits* to *dest_rank*.

        The adapter chooses the transfer protocol (e.g. teleportation).
        """
        for q in qubits:
            self._check_qubit(q)
        return self._add(QSend(qubits, dest_rank))

    def qrecv(self, qubits: List[int], src_rank: int) -> Circuit:
        """
        Receive qubits from *src_rank* into local qubit slots *qubits*.

        ``len(qubits)`` determines how many qubits are expected.
        """
        for q in qubits:
            self._check_qubit(q)
        return self._add(QRecv(qubits, src_rank))

    def qscatter(self, qubits: List[int], sender_rank: int) -> Circuit:
        """Scatter *qubits* from *sender_rank* across all ranks."""
        for q in qubits:
            self._check_qubit(q)
        return self._add(QScatter(qubits, sender_rank))

    def qgather(self, qubits: List[int], recv_rank: int) -> Circuit:
        """Contribute *qubits* to a gather into *recv_rank*."""
        for q in qubits:
            self._check_qubit(q)
        return self._add(QGather(qubits, recv_rank))

    def expose(self, qubits: List[int], rank: int = 0) -> _ExposeContext:
        """
        Expose *qubits* to the network via a shared GHZ state.

        Returns an :class:`_ExposeContext` that can be used as a
        ``with`` statement.  The matching :class:`Unexpose` is appended
        automatically when the ``with`` block exits::

            with circuit.expose([0], rank=0):
                circuit.h(0)
            # Unexpose(rank=0) inserted here

        Args:
            qubits: Local qubit indices to expose.
            rank:   Exposer rank (default: 0).
        """
        for q in qubits:
            self._check_qubit(q)
        return _ExposeContext(self, qubits, rank)

    def unexpose(self, rank: int = 0) -> Circuit:
        """Manually close an expose window for *rank* (use :meth:`expose` as a
        context manager instead when possible)."""
        return self._add(Unexpose(rank))

    # ------------------------------------------------------------------
    # Iteration helper
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[Operation]:
        """Iterates over all leaf operations in the circuit."""
        return self._ops.flatten()

    def __len__(self) -> int:
        """Number of top-level entries in the operation container."""
        return len(self._ops)
