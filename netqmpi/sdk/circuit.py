"""
Base abstraction for quantum circuits.

This module defines the contract that all circuit adapters must follow.
It provides a backend-agnostic circuit representation based on generic
operations and exposes the abstract hooks required by concrete backend
implementations.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Iterator, List, Optional

from netqmpi.sdk.operations.qmpi import (
    Expose, QGather, QRecv, QScatter, QSend, Unexpose,
)
from netqmpi.sdk.operations.container import OperationContainer
from netqmpi.sdk.operations.gate import ControlledGate, Gate
from netqmpi.sdk.operations.non_unitary import Barrier, Measure, Reset
from netqmpi.sdk.operations.operation import Operation

if TYPE_CHECKING:
    from netqmpi.sdk import QMPICommunicator


class _ExposeContext:
    """
    Context manager returned by :meth:`Circuit.expose`.

    On entry, it appends :class:`~netqmpi.sdk.operations.qmpi.Expose` to
    the circuit. On exit, it automatically appends the matching
    :class:`~netqmpi.sdk.operations.qmpi.Unexpose`, even if an exception
    is raised inside the context.

    Example:
        with circuit.expose([0, 1], rank=0):
            circuit.h(0).cx(0, 1)
    """

    def __init__(self, circuit: "Circuit", qubits: List[int], rank: int) -> None:
        """
        Initialize the expose context manager.

        Args:
            circuit: Circuit associated with the expose operation.
            qubits: Local qubit indices to expose.
            rank: Rank acting as the exposer.
        """
        self._circuit = circuit
        self._qubits = qubits
        self._rank = rank

    def __enter__(self) -> "Circuit":
        """
        Enter the expose context and return the circuit.

        Returns:
            The circuit instance, allowing method chaining inside the
            context.
        """
        self._circuit._add(Expose(self._qubits, self._rank))
        return self._circuit

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """
        Exit the expose context and append the matching unexpose operation.

        Args:
            exc_type: Exception type, if one was raised.
            exc_val: Exception instance, if one was raised.
            exc_tb: Traceback, if one was raised.

        Returns:
            ``False`` so that exceptions, if any, are not suppressed.
        """
        self._circuit._add(Unexpose(self._rank))
        return False  # never suppress exceptions


class Circuit(ABC):
    """
    Abstract base class representing a quantum circuit.

    This class provides:

    - An :class:`~netqmpi.sdk.operations.container.OperationContainer`
      storing operations according to the Composite pattern.
    - A fluent gate API (``h``, ``cx``, ``rx``, ``measure``, etc.) that
      appends operations to the container and returns ``self`` for
      chaining.
    - Abstract hooks :meth:`translate` and :meth:`build` that concrete
      backend adapters must implement.

    Attributes:
        num_qubits: Number of qubits in the circuit.
        num_clbits: Number of classical bits in the circuit.
    """

    def __init__(
        self,
        num_qubits: int,
        num_clbits: int,
        comm: QMPICommunicator,
    ) -> None:
        """
        Initialize the circuit.

        Args:
            num_qubits: Number of qubits in the circuit.
            num_clbits: Number of classical bits in the circuit.
            comm: Communicator associated with the circuit.
        """
        self._num_qubits = num_qubits
        self._num_clbits = num_clbits
        self._comm = comm
        self._ops = OperationContainer()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def num_qubits(self) -> int:
        """
        Return the number of qubits in the circuit.

        Returns:
            The number of qubits.
        """
        return self._num_qubits

    @property
    def num_clbits(self) -> int:
        """
        Return the number of classical bits in the circuit.

        Returns:
            The number of classical bits.
        """
        return self._num_clbits

    @property
    def ops(self) -> OperationContainer:
        """
        Return the root operation container.

        Returns:
            The operation container storing the circuit operations.
        """
        return self._ops

    @property
    def comm(self) -> QMPICommunicator:
        """
        Return the communicator associated with the circuit.

        Returns:
            The circuit communicator.
        """
        return self._comm

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def translate(self, op: Operation) -> Any:
        """
        Translate a generic operation into a backend-native instruction.

        Args:
            op: Operation to translate.

        Returns:
            A backend-specific object representing the translated
            operation.
        """

    @abstractmethod
    def build(self) -> Any:
        """
        Materialize the circuit for the target backend.

        Returns:
            A backend-native circuit object ready for execution.
        """

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _check_qubit(self, qubit: int) -> None:
        """
        Validate a qubit index.

        Args:
            qubit: Qubit index to validate.

        Raises:
            IndexError: If the qubit index is out of range.
        """
        if not (0 <= qubit < self._num_qubits):
            raise IndexError(
                f"Qubit index {qubit} out of range [0, {self._num_qubits}).")

    def _check_cbit(self, cbit: int) -> None:
        """
        Validate a classical bit index.

        Args:
            cbit: Classical bit index to validate.

        Raises:
            IndexError: If the classical bit index is out of range.
        """
        if not (0 <= cbit < self._num_clbits):
            raise IndexError(
                f"Classical bit index {cbit} out of range [0, {self._num_clbits}).")

    def _add(self, op: Operation) -> Circuit:
        """
        Append an operation to the circuit.

        Args:
            op: Operation to append.

        Returns:
            The current circuit instance.
        """
        self._ops.add(op)
        return self

    # ------------------------------------------------------------------
    # Fluent gate API — single-qubit gates
    # ------------------------------------------------------------------

    def h(self, qubit: int) -> Circuit:
        """
        Apply a Hadamard gate to a qubit.

        Args:
            qubit: Target qubit index.

        Returns:
            The current circuit instance.
        """
        self._check_qubit(qubit)
        return self._add(Gate('H', [qubit]))

    def x(self, qubit: int) -> Circuit:
        """
        Apply a Pauli-X gate to a qubit.

        Args:
            qubit: Target qubit index.

        Returns:
            The current circuit instance.
        """
        self._check_qubit(qubit)
        return self._add(Gate('X', [qubit]))

    def y(self, qubit: int) -> Circuit:
        """
        Apply a Pauli-Y gate to a qubit.

        Args:
            qubit: Target qubit index.

        Returns:
            The current circuit instance.
        """
        self._check_qubit(qubit)
        return self._add(Gate('Y', [qubit]))

    def z(self, qubit: int) -> Circuit:
        """
        Apply a Pauli-Z gate to a qubit.

        Args:
            qubit: Target qubit index.

        Returns:
            The current circuit instance.
        """
        self._check_qubit(qubit)
        return self._add(Gate('Z', [qubit]))

    def s(self, qubit: int) -> Circuit:
        """
        Apply an S gate to a qubit.

        Args:
            qubit: Target qubit index.

        Returns:
            The current circuit instance.
        """
        self._check_qubit(qubit)
        return self._add(Gate('S', [qubit]))

    def sdg(self, qubit: int) -> Circuit:
        """
        Apply an S-dagger gate to a qubit.

        Args:
            qubit: Target qubit index.

        Returns:
            The current circuit instance.
        """
        self._check_qubit(qubit)
        return self._add(Gate('SDG', [qubit]))

    def t(self, qubit: int) -> Circuit:
        """
        Apply a T gate to a qubit.

        Args:
            qubit: Target qubit index.

        Returns:
            The current circuit instance.
        """
        self._check_qubit(qubit)
        return self._add(Gate('T', [qubit]))

    def tdg(self, qubit: int) -> Circuit:
        """
        Apply a T-dagger gate to a qubit.

        Args:
            qubit: Target qubit index.

        Returns:
            The current circuit instance.
        """
        self._check_qubit(qubit)
        return self._add(Gate('TDG', [qubit]))

    # ------------------------------------------------------------------
    # Fluent gate API — parametric single-qubit gates
    # ------------------------------------------------------------------

    def rx(self, theta: float, qubit: int) -> Circuit:
        """
        Apply an X-axis rotation to a qubit.

        Args:
            theta: Rotation angle in radians.
            qubit: Target qubit index.

        Returns:
            The current circuit instance.
        """
        self._check_qubit(qubit)
        return self._add(Gate('RX', [qubit], [theta]))

    def ry(self, theta: float, qubit: int) -> Circuit:
        """
        Apply a Y-axis rotation to a qubit.

        Args:
            theta: Rotation angle in radians.
            qubit: Target qubit index.

        Returns:
            The current circuit instance.
        """
        self._check_qubit(qubit)
        return self._add(Gate('RY', [qubit], [theta]))

    def rz(self, theta: float, qubit: int) -> Circuit:
        """
        Apply a Z-axis rotation to a qubit.

        Args:
            theta: Rotation angle in radians.
            qubit: Target qubit index.

        Returns:
            The current circuit instance.
        """
        self._check_qubit(qubit)
        return self._add(Gate('RZ', [qubit], [theta]))

    # ------------------------------------------------------------------
    # Fluent gate API — two-qubit gates
    # ------------------------------------------------------------------

    def cx(self, control: int, target: int) -> Circuit:
        """
        Apply a controlled-X gate.

        Args:
            control: Control qubit index.
            target: Target qubit index.

        Returns:
            The current circuit instance.
        """
        self._check_qubit(control)
        self._check_qubit(target)
        return self._add(ControlledGate([control], [Gate('X', [target])]))

    def cz(self, control: int, target: int) -> Circuit:
        """
        Apply a controlled-Z gate.

        Args:
            control: Control qubit index.
            target: Target qubit index.

        Returns:
            The current circuit instance.
        """
        self._check_qubit(control)
        self._check_qubit(target)
        return self._add(ControlledGate([control], [Gate('Z', [target])]))

    def swap(self, qubit1: int, qubit2: int) -> Circuit:
        """
        Apply a SWAP gate between two qubits.

        Args:
            qubit1: First qubit index.
            qubit2: Second qubit index.

        Returns:
            The current circuit instance.
        """
        self._check_qubit(qubit1)
        self._check_qubit(qubit2)
        return self._add(Gate('SWAP', [qubit1, qubit2]))

    def crz(self, theta: float, control: int, target: int) -> Circuit:
        """
        Apply a controlled-RZ gate.

        Args:
            theta: Rotation angle in radians.
            control: Control qubit index.
            target: Target qubit index.

        Returns:
            The current circuit instance.
        """
        self._check_qubit(control)
        self._check_qubit(target)
        return self._add(ControlledGate([control], [Gate('RZ', [target], [theta])]))

    # ------------------------------------------------------------------
    # Fluent gate API — three-qubit gates
    # ------------------------------------------------------------------

    def ccx(self, control1: int, control2: int, target: int) -> Circuit:
        """
        Apply a Toffoli gate.

        Args:
            control1: First control qubit index.
            control2: Second control qubit index.
            target: Target qubit index.

        Returns:
            The current circuit instance.
        """
        self._check_qubit(control1)
        self._check_qubit(control2)
        self._check_qubit(target)
        return self._add(ControlledGate([control1, control2], [Gate('X', [target])]))

    # ------------------------------------------------------------------
    # Fluent API — non-unitary operations
    # ------------------------------------------------------------------

    def measure(self, qubit: int, cbit: int) -> Circuit:
        """
        Measure a qubit into a classical bit.

        Args:
            qubit: Measured qubit index.
            cbit: Destination classical bit index.

        Returns:
            The current circuit instance.
        """
        self._check_qubit(qubit)
        self._check_cbit(cbit)
        return self._add(Measure(qubit, cbit))

    def measure_all(self) -> Circuit:
        """
        Measure every qubit into the classical bit of the same index.

        Returns:
            The current circuit instance.

        Raises:
            ValueError: If there are fewer classical bits than qubits.
        """
        if self._num_clbits < self._num_qubits:
            raise ValueError(
                "Not enough classical bits to measure all qubits "
                f"({self._num_clbits} clbits < {self._num_qubits} qubits)."
            )
        for i in range(self._num_qubits):
            self._add(Measure(i, i))
        return self

    def reset(self, qubit: int) -> Circuit:
        """
        Reset a qubit to the ``|0⟩`` state.

        Args:
            qubit: Target qubit index.

        Returns:
            The current circuit instance.
        """
        self._check_qubit(qubit)
        return self._add(Reset(qubit))

    def barrier(self, qubits: Optional[List[int]] = None) -> Circuit:
        """
        Insert a barrier.

        Args:
            qubits: Qubits to include in the barrier. If ``None``, the
                barrier applies to the full circuit.

        Returns:
            The current circuit instance.
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
        Send qubits to another rank.

        The backend adapter decides the concrete transfer protocol.

        Args:
            qubits: Local qubit indices to send.
            dest_rank: Destination rank.

        Returns:
            The current circuit instance.
        """
        for q in qubits:
            self._check_qubit(q)
        return self._add(QSend(qubits, dest_rank))

    def qrecv(self, qubits: List[int], src_rank: int) -> Circuit:
        """
        Receive qubits from another rank into local qubit slots.

        Args:
            qubits: Local qubit indices that will receive the incoming qubits.
            src_rank: Source rank.

        Returns:
            The current circuit instance.
        """
        for q in qubits:
            self._check_qubit(q)
        return self._add(QRecv(qubits, src_rank))

    def qscatter(self, qubits: List[int], sender_rank: int) -> Circuit:
        """
        Scatter qubits from one rank across all ranks.

        Args:
            qubits: Qubits involved in the scatter operation.
            sender_rank: Rank acting as the sender.

        Returns:
            The current circuit instance.
        """
        for q in qubits:
            self._check_qubit(q)
        return self._add(QScatter(qubits, sender_rank))

    def qgather(self, qubits: List[int], recv_rank: int) -> Circuit:
        """
        Contribute qubits to a gather operation.

        Args:
            qubits: Qubits contributed to the gather.
            recv_rank: Rank receiving the gathered qubits.

        Returns:
            The current circuit instance.
        """
        for q in qubits:
            self._check_qubit(q)
        return self._add(QGather(qubits, recv_rank))

    def expose(self, qubits: List[int], rank: int = 0) -> _ExposeContext:
        """
        Expose qubits to the network through a shared GHZ state.

        Args:
            qubits: Local qubit indices to expose.
            rank: Exposer rank.

        Returns:
            A context manager that automatically appends the matching
            :class:`Unexpose` operation when the context exits.
        """
        for q in qubits:
            self._check_qubit(q)
        return _ExposeContext(self, qubits, rank)

    def unexpose(self, rank: int = 0) -> Circuit:
        """
        Manually close an expose window.

        Args:
            rank: Exposer rank associated with the expose operation.

        Returns:
            The current circuit instance.
        """
        return self._add(Unexpose(rank))

    # ------------------------------------------------------------------
    # Iteration helper
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[Operation]:
        """
        Iterate over all leaf operations in the circuit.

        Returns:
            An iterator over the flattened circuit operations.
        """
        return self._ops.flatten()

    def __len__(self) -> int:
        """
        Return the number of top-level entries in the operation container.

        Returns:
            The number of top-level stored operations.
        """
        return len(self._ops)