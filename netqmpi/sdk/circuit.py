"""
Base abstraction for quantum circuits.

This module defines the contract that all circuit adapters must follow.
It provides a backend-agnostic circuit representation based on generic
operations and exposes the abstract hooks required by concrete backend
implementations.

Qubit indices span two ranges. Indices below :attr:`Circuit.num_qubits`
address the data qubits the user asked for; indices from there on address
the *communication qubits* the runtime reserves for distributed
protocols, and are only ever produced by :meth:`Circuit.expose`. Both
ranges are accepted by the gate API, so a control qubit borrowed from a
remote rank is used exactly like a local one.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional, Tuple

from netqmpi.sdk.operations import (
    Operation,
    Gate, ControlledGate, ClassicalControlledGate,
    Measure, Reset, Barrier,
    OperationContainer,
    QSend, QRecv, QScatter, QGather, Expose, Unexpose,
)
from netqmpi.sdk.resources import IndexPool

if TYPE_CHECKING:
    from netqmpi.sdk import QMPICommunicator

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

        # Resources borrowed by distributed protocols. Slots are reserved
        # when a protocol block opens and returned when it closes, so the
        # backend only has to provide as many as are held at once.
        self._comm_pool = IndexPool()
        self._protocol_clbit_pool = IndexPool()

        # Expose windows still open, keyed by participant group so that
        # unexpose() can pair with the innermost matching expose().
        self._open_exposures: Dict[Tuple[int, ...], List[Expose]] = {}
        # Communication-qubit slots the user may currently address.
        self._held_comm_slots: set = set()
        # Per-group counters feeding the tags that let every rank pair its
        # own record with the ones traced by the other participants.
        self._collective_counters: Dict[Tuple[Any, ...], int] = {}

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
    def num_comm_qubits(self) -> int:
        """
        Return how many communication qubits the circuit needs.

        This is the largest number of communication qubits held at the
        same time by the distributed protocols traced so far, and it is
        only final once the circuit has been fully traced.

        Returns:
            The number of communication qubits to reserve on the backend.
        """
        return self._comm_pool.size

    @property
    def num_protocol_clbits(self) -> int:
        """
        Return how many classical bits the distributed protocols need.

        These bits carry the correction outcomes of teledata/telegate and
        are additional to the :attr:`num_clbits` requested by the user, so
        a protocol never clobbers a user measurement.

        Returns:
            The number of protocol classical bits to reserve on the backend.
        """
        return self._protocol_clbit_pool.size

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
    def _translate_gate(self, op: Gate):
        """
        Translate a single-qubit unitary gate into a backend instruction.

        Args:
            op: Gate operation to translate.
        """

    @abstractmethod
    def _translate_controlled_gate(self, op: ControlledGate):
        """
        Translate a controlled quantum gate into a backend instruction.

        Args:
            op: Controlled gate operation to translate.
        """
            
    @abstractmethod
    def _translate_classical_controlled_gate(self, op: ClassicalControlledGate):
        """
        Translate a classically controlled gate into a backend instruction.

        Args:
            op: Classically controlled gate operation to translate.

        Raises:
            NotImplementedError: Always, because this operation is not yet supported.
        """

    @abstractmethod
    def _translate_measure(self, op: Measure):
        """
        Translate a measurement operation into a backend instruction.

        Args:
            op: Measurement operation to translate.
        """

    @abstractmethod
    def _translate_reset(self, op: Reset):
        """
        Translate a reset operation into a backend instruction.

        Args:
            op: Reset operation to translate.
        """

    @abstractmethod
    def _translate_barrier(self, op: Barrier):
        """
        Translate a barrier operation into a backend instruction.

        Args:
            op: Barrier operation to translate.

        Raises:
            NotImplementedError: Always, because this operation is not yet supported.
        """

    @abstractmethod
    def _translate_operation_container(self, op: OperationContainer):
        """
        Translate an operation container by recursively translating its children.

        Args:
            op: Operation container to translate.
        """

    @abstractmethod
    def _translate_qsend(self, op: QSend):
        """
        Translate a quantum send operation into a backend instruction.

        Args:
            op: Quantum send operation to translate.
        """

    @abstractmethod
    def _translate_qrecv(self, op: QRecv):
        """
        Translate a quantum receive operation into a backend instruction.

        Args:
            op: Quantum receive operation to translate.
        """

    @abstractmethod
    def _translate_qscatter(self, op: QScatter):
        """
        Translate a quantum scatter operation into backend instructions.

        The record is a container holding the point-to-point transfers the
        scatter expands into, so an adapter that already translates
        :class:`QSend` and :class:`QRecv` only has to translate those
        children.

        Args:
            op: Quantum scatter operation to translate.
        """

    @abstractmethod
    def _translate_qgather(self, op: QGather):
        """
        Translate a quantum gather operation into backend instructions.

        As with :meth:`_translate_qscatter`, the record holds the transfers
        the gather expands into.

        Args:
            op: Quantum gather operation to translate.
        """

    @abstractmethod
    def _translate_expose(self, op: Expose):
        """
        Translate an expose operation into a backend instruction.

        Args:
            op: Expose operation to translate.

        Raises:
            NotImplementedError: Always, because this operation is not yet supported.
        """

    @abstractmethod
    def _translate_unexpose(self, op: Unexpose):
        """
        Translate an unexpose operation into a backend instruction.

        Args:
            op: Unexpose operation to translate.

        Raises:
            NotImplementedError: Always, because this operation is not yet supported.
        """

    # Dispatch table: maps each Operation type to its translation method.
    # ClassicalControlledGate and ControlledGate must appear before Gate
    # because both are subclasses of Operation but not of Gate.
    _DISPATCH: dict = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._DISPATCH = {}

    def _build_dispatch(self):
        """
        Build the dispatch table for operation translation.

        Returns:
            A mapping from operation types to translation methods.
        """
        return {
            ClassicalControlledGate: self._translate_classical_controlled_gate,
            ControlledGate:          self._translate_controlled_gate,
            Gate:                    self._translate_gate,
            Measure:                 self._translate_measure,
            Reset:                   self._translate_reset,
            Barrier:                 self._translate_barrier,
            OperationContainer:      self._translate_operation_container,
            QSend:                   self._translate_qsend,
            QRecv:                   self._translate_qrecv,
            QScatter:                self._translate_qscatter,
            QGather:                 self._translate_qgather,
            Expose:                  self._translate_expose,
            Unexpose:                self._translate_unexpose,
        }

    def translate(self, op: Operation) -> Any:
        """
        Dispatch an operation to its corresponding translation method.

        Args:
            op: Operation to translate.

        Returns:
            The translated backend instruction or instructions.

        Raises:
            TypeError: If the operation type is unknown.
        """
        if not self._DISPATCH:
            self._DISPATCH = self._build_dispatch()

        handler = self._DISPATCH.get(type(op))
        if handler is None:
            # Walk the MRO to support subclasses not registered explicitly.
            handler = next(
                (self._DISPATCH[t] for t in type(op).__mro__ if t in self._DISPATCH),
                None,
            )
        if handler is None:
            raise TypeError(f"Unknown operation type: {type(op).__name__}")
        
        return handler(op)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def comm_qubit(self, slot: int) -> int:
        """
        Return the circuit-wide index of a communication-qubit slot.

        Communication qubits are addressed right after the data qubits,
        so the value returned here can be handed to any gate of the fluent
        API just like a data qubit index.

        Args:
            slot: Communication-qubit slot reserved by a protocol.

        Returns:
            The qubit index addressing that slot.
        """
        return self._num_qubits + slot

    def _check_qubit(self, qubit: int) -> None:
        """
        Validate a qubit index, data or communication.

        Args:
            qubit: Qubit index to validate.

        Raises:
            IndexError: If the qubit index addresses neither a data qubit
                nor a communication qubit currently reserved.
        """
        limit = self._num_qubits + self._comm_pool.size
        if not (0 <= qubit < limit):
            raise IndexError(
                f"Qubit index {qubit} out of range [0, {limit}) "
                f"({self._num_qubits} data + {self._comm_pool.size} comm qubits).")
        if qubit >= self._num_qubits and qubit - self._num_qubits not in self._held_comm_slots:
            raise IndexError(
                f"Qubit index {qubit} is a communication qubit whose expose "
                f"window is already closed.")

    def _check_data_qubit(self, qubit: int) -> None:
        """
        Validate an index that must address a data qubit.

        Args:
            qubit: Qubit index to validate.

        Raises:
            IndexError: If the qubit index is not a data qubit.
        """
        if not (0 <= qubit < self._num_qubits):
            raise IndexError(
                f"Qubit index {qubit} is not a data qubit "
                f"(expected [0, {self._num_qubits})).")

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
    
    def cs(self, control: int, target: int) -> Circuit:
        """
        Apply a controlled-S gate.

        Args:
            control: Control qubit index.
            target: Target qubit index.

        Returns:
            The current circuit instance.
        """
        self._check_qubit(control)
        self._check_qubit(target)
        return self._add(ControlledGate([control], [Gate('S', [target])]))

    def ct(self, control: int, target: int) -> Circuit:
        """
        Apply a controlled-T gate.

        Args:
            control: Control qubit index.
            target: Target qubit index.

        Returns:
            The current circuit instance.
        """
        self._check_qubit(control)
        self._check_qubit(target)
        return self._add(ControlledGate([control], [Gate('T', [target])]))
    
    def cp(self, control: int, target: int, beta: float) -> Circuit:
        """
        Apply a controlled-P gate.

        Args:
            control: Control qubit index.
            target: Target qubit index.
            beta: angle applied by the P gate.

        Returns:
            The current circuit instance.
        """
        self._check_qubit(control)
        self._check_qubit(target)
        return self._add(ControlledGate([control], [Gate('P', [target], [beta])]))

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

    def cp(self, control: int, target: int, theta: float) -> Circuit:
        """
        Apply a controlled phase gate.

        Generalises :meth:`cs` (``theta = pi/2``) and :meth:`ct`
        (``theta = pi/4``), which is what the rotations of a QFT are made
        of.

        Args:
            control: Control qubit index.
            target: Target qubit index.
            theta: Phase angle in radians.

        Returns:
            The current circuit instance.
        """
        self._check_qubit(control)
        self._check_qubit(target)
        return self._add(ControlledGate([control], [Gate('P', [target], [theta])]))

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

    def _next_tag(self, prefix: str, key: Tuple[Any, ...]) -> str:
        """
        Build the next tag identifying a collective or point-to-point call.

        Tags are derived from data every participant already knows — the
        kind of call and the ranks involved — plus a per-key counter, so
        each rank names a call exactly like its peers do without any
        trace-time communication.

        Args:
            prefix: Short name of the protocol the tag belongs to.
            key: Participant-derived key shared by all sides of the call.

        Returns:
            The tag for this call.
        """
        counter_key = (prefix, *key)
        index = self._collective_counters.get(counter_key, 0)
        self._collective_counters[counter_key] = index + 1
        return f"{prefix}_" + "_".join(str(k) for k in key) + f"_{index}"

    def _teledata_send(self, qubit: int, dest_rank: int) -> QSend:
        """
        Build the record of one qubit leaving this rank.

        The protocol block borrows one communication qubit and two
        protocol classical bits, and gives them back straight away: the
        block is over by the time the next instruction runs, so a later
        transfer can reuse the very same resources.

        Args:
            qubit: Local qubit index to send.
            dest_rank: Destination rank.

        Returns:
            The send record, resources and tag already reserved.
        """
        slot = self._comm_pool.acquire()[0]
        clbits = self._protocol_clbit_pool.acquire(2)
        tag = self._next_tag("teledata", (self._comm.rank, dest_rank))
        op = QSend([qubit], dest_rank, comm_slot=slot, clbits=clbits, tag=tag)
        self._comm_pool.release([slot])
        self._protocol_clbit_pool.release(clbits)
        return op

    def _teledata_recv(self, qubit: int, src_rank: int) -> QRecv:
        """
        Build the record of one qubit arriving at this rank.

        Args:
            qubit: Local qubit index the incoming state will land on.
            src_rank: Source rank.

        Returns:
            The receive record, resources and tag already reserved.
        """
        slot = self._comm_pool.acquire()[0]
        clbits = self._protocol_clbit_pool.acquire(2)
        tag = self._next_tag("teledata", (src_rank, self._comm.rank))
        op = QRecv([qubit], src_rank, comm_slot=slot, clbits=clbits, tag=tag)
        self._comm_pool.release([slot])
        self._protocol_clbit_pool.release(clbits)
        return op

    def qsend(self, qubits: List[int], dest_rank: int) -> Circuit:
        """
        Send qubits to another rank.

        The backend adapter decides the concrete transfer protocol; each
        qubit is transferred by its own protocol block, which borrows one
        communication qubit and two protocol classical bits for as long as
        the transfer lasts.

        Args:
            qubits: Local qubit indices to send.
            dest_rank: Destination rank.

        Returns:
            The current circuit instance.
        """
        for q in qubits:
            self._check_qubit(q)
        for q in qubits:
            self._add(self._teledata_send(q, dest_rank))
        return self

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
        for q in qubits:
            self._add(self._teledata_recv(q, src_rank))
        return self

    def _rooted_chunk(
        self,
        qubits: List[int],
        root: int,
        what: str,
    ) -> Tuple[int, List[int], List[List[int]]]:
        """
        Validate a rooted transfer and split the root's buffer.

        Args:
            qubits: Local qubits the caller passed to the collective.
            root: Rank the qubits are scattered from or gathered into.
            what: Name of the collective, used in the error messages.

        Returns:
            A tuple ``(root, ranks, chunks)``, where ``chunks`` holds one
            list of local qubit indices per rank on the root and is empty
            elsewhere.

        Raises:
            IndexError: If a qubit index is not a data qubit of this rank.
            ValueError: If the root is not a rank of the communicator, if
                the buffer is empty, or if the root's buffer does not
                split evenly among the ranks.
        """
        size = self._comm.size
        if not (0 <= root < size):
            raise ValueError(
                f"{what} root {root} is not a rank of the communicator "
                f"[0, {size}).")
        if not isinstance(qubits, list) or not qubits:
            raise ValueError(f"{what} needs a non-empty list of qubits.")
        for q in qubits:
            # Only data qubits can be moved: the communication ones belong
            # to the protocols and are gone by the time the block is over.
            self._check_data_qubit(q)

        ranks = list(range(size))
        if self._comm.rank != root:
            return root, ranks, []

        if len(qubits) % size:
            raise ValueError(
                f"the root of a {what} must hold one chunk per rank: "
                f"{len(qubits)} qubits do not split evenly among {size} ranks.")
        step = len(qubits) // size
        return root, ranks, [qubits[r * step:(r + 1) * step] for r in ranks]

    def qscatter(self, qubits: List[int], root: int) -> List[int]:
        """
        Scatter the qubits of the root across every rank.

        Collective call, like ``MPI_Scatter``: every rank of the
        communicator must reach it. The root passes its whole buffer,
        which is split into one chunk per rank in rank order, and every
        other rank passes the local qubits its chunk is to land on — as
        many as the root reserved for it.

        Qubits are *moved*, not copied: the chunks leaving the root are
        teleported away, so once the call is over the root only holds its
        own chunk and the qubits it scattered are back in ``|0⟩``. The
        qubits a chunk lands on must be in ``|0⟩`` when the call is
        reached, as they must be for a plain :meth:`qrecv`: whatever they
        held is not saved anywhere, it is destroyed by the transfer.

        Args:
            qubits: The whole buffer on the root, this rank's landing
                qubits on every other rank.
            root: Rank whose buffer is scattered.

        Returns:
            The local qubits holding this rank's chunk.

        Raises:
            IndexError: If a qubit index is not a data qubit of this rank.
            ValueError: If the root is not a rank of the communicator, if
                the buffer is empty, or if the root's buffer does not
                split evenly among the ranks.
        """
        rank = self._comm.rank
        root, ranks, chunks = self._rooted_chunk(qubits, root, "qscatter")

        record = QScatter(rank=rank, root=root, ranks=ranks, qubits=qubits)
        if rank == root:
            for other in ranks:
                if other == root:
                    continue
                for q in chunks[other]:
                    record.add(self._teledata_send(q, other))
        else:
            for q in qubits:
                record.add(self._teledata_recv(q, root))

        self._add(record)
        return list(chunks[root]) if rank == root else list(qubits)

    def qgather(self, qubits: List[int], root: int) -> List[int]:
        """
        Gather the qubits of every rank into the root.

        Collective call, like ``MPI_Gather``, and the mirror image of
        :meth:`qscatter`: the root passes the whole buffer the chunks are
        to land on — its own chunk, at position ``root``, already holding
        its contribution — and every other rank passes the local qubits it
        contributes.

        Qubits are *moved* here as well, so once the call is over the
        contributors are left with theirs back in ``|0⟩`` and only the root
        holds the gathered data. The slots the root gathers into — every
        one of its buffer but its own chunk — must be in ``|0⟩`` when the
        call is reached, exactly as for a plain :meth:`qrecv`.

        Args:
            qubits: The whole buffer on the root, this rank's contribution
                on every other rank.
            root: Rank the qubits are gathered into.

        Returns:
            The whole buffer on the root, this rank's contribution
            elsewhere.

        Raises:
            IndexError: If a qubit index is not a data qubit of this rank.
            ValueError: If the root is not a rank of the communicator, if
                the buffer is empty, or if the root's buffer does not
                split evenly among the ranks.
        """
        rank = self._comm.rank
        root, ranks, chunks = self._rooted_chunk(qubits, root, "qgather")

        record = QGather(rank=rank, root=root, ranks=ranks, qubits=qubits)
        if rank == root:
            for other in ranks:
                if other == root:
                    continue
                for q in chunks[other]:
                    record.add(self._teledata_recv(q, other))
        else:
            for q in qubits:
                record.add(self._teledata_send(q, root))

        self._add(record)
        return list(qubits)

    @staticmethod
    def _expose_group(ranks: List[int], root: int) -> List[int]:
        """
        Normalise the participant list of an expose window.

        The root always comes first, because the telegate protocol treats
        it asymmetrically, and the receivers keep the order the caller
        gave so that every rank builds the very same list.

        Args:
            ranks: Ranks the qubit is exposed to.
            root: Rank exposing its qubit.

        Returns:
            The participants, root first.

        Raises:
            ValueError: If *ranks* is not a list, or lists no receiver.
        """
        if not isinstance(ranks, list) or not ranks:
            raise ValueError("ranks must be a non-empty list of integers.")
        receivers = list(dict.fromkeys(r for r in ranks if r != root))
        if not receivers:
            raise ValueError(
                f"expose needs at least one rank besides the root ({root}).")
        return [root, *receivers]

    def expose(
        self,
        qubit: Optional[int],
        ranks: List[int],
        root: Optional[int] = None,
    ) -> Optional[int]:
        """
        Open a telegate window sharing a control qubit across ranks.

        This is a *collective* call: every rank in ``[root] + ranks`` must
        reach it, exactly as they all reach an ``MPI_Bcast``. The ``root``
        lends the state of ``qubit`` to the other participants, which each
        receive it on a communication qubit of their own and can then use
        it as a local control until the matching :meth:`unexpose`.

        Args:
            qubit: Data qubit to expose. Read on the root only; the other
                participants may pass ``None``.
            ranks: Ranks the qubit is exposed to.
            root: Rank exposing its qubit. Defaults to the calling rank.

        Returns:
            The qubit index to use as control on this rank — ``qubit``
            itself on the root, the freshly reserved communication qubit
            on every receiver — or ``None`` if this rank does not take part.

        Raises:
            IndexError: If the root exposes something other than a data qubit.
            ValueError: If the participant list is empty or names no receiver.
        """
        rank = self._comm.rank
        root = rank if root is None else root
        group = self._expose_group(ranks, root)
        if rank not in group:
            return None

        if rank == root:
            self._check_data_qubit(qubit)

        tag = self._next_tag("expose", (root, *group[1:]))
        slot = self._comm_pool.acquire()[0]
        # The root collects one correction bit per receiver at the end of
        # the window; each receiver only ever handles its own.
        clbits = self._protocol_clbit_pool.acquire(
            len(group) - 1 if rank == root else 1)

        op = Expose(
            rank=rank,
            root=root,
            ranks=group,
            tag=tag,
            comm_slot=slot,
            clbits=clbits,
            data_qubit=qubit if rank == root else None,
        )
        self._open_exposures.setdefault(tuple(group), []).append(op)
        self._held_comm_slots.add(slot)
        self._add(op)

        return qubit if rank == root else self.comm_qubit(slot)

    def unexpose(
        self,
        ranks: List[int],
        root: Optional[int] = None,
    ) -> Circuit:
        """
        Close the telegate window opened by the matching :meth:`expose`.

        Collective as well: the same ranks that opened the window must
        close it. The communication qubit and the protocol classical bits
        it held are returned to the pool, so a later window can reuse them.

        Args:
            ranks: Ranks the qubit was exposed to.
            root: Rank that exposed its qubit. Defaults to the calling rank.

        Returns:
            The current circuit instance.

        Raises:
            RuntimeError: If no matching expose window is open.
        """
        rank = self._comm.rank
        root = rank if root is None else root
        group = self._expose_group(ranks, root)
        if rank not in group:
            return self

        open_windows = self._open_exposures.get(tuple(group))
        if not open_windows:
            raise RuntimeError(
                f"rank {rank} called unexpose(ranks={ranks}, root={root}) "
                f"without a matching open expose window.")

        # Innermost window first, so nested exposures unwind like scopes.
        expose = open_windows.pop()
        self._add(Unexpose.closing(expose))
        self._held_comm_slots.discard(expose.comm_slot)
        self._comm_pool.release([expose.comm_slot])
        self._protocol_clbit_pool.release(expose.clbits)
        return self

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