"""
Circuit adapter for the NetQASM eager-execution model.

NetQASM does not provide a circuit object: each instruction is dispatched
to the simulator as soon as it is invoked on a
:class:`netqasm.sdk.qubit.Qubit`. Therefore, this adapter:

1. Allocates a :class:`Qubit` array in ``__init__`` using the active
   connection exposed by ``self._comm``.
2. Overrides the gate methods of the base :class:`Circuit` so that each
   method first delegates to ``super()`` to record the operation in the
   :class:`OperationContainer`, and then executes the corresponding
   NetQASM SDK call immediately.
3. Overrides the inter-rank communication primitives (``qsend``,
   ``qrecv``, ``qscatter``, ``qgather``, ``expose``, and ``unexpose``)
   with the concrete teleportation and collective protocols implemented
   through the NetQASM SDK.
4. Keeps ``translate()`` as a no-op, since execution has already taken
   place eagerly.
5. Flushes the connection in ``build()`` and returns the qubit array
   together with the classical measurement results.
"""
from __future__ import annotations
import numpy as np
from typing import TYPE_CHECKING, Any, List, Optional, Dict

from netqasm.sdk.classical_communication.message import StructuredMessage
from netqasm.sdk import EPRSocket, Qubit
from netqasm.sdk.external import Socket
from netqasm.sdk.toolbox import create_ghz

from netqmpi.sdk.circuit import Circuit, _ExposeContext
from netqmpi.sdk.operations.operation import Operation
from netqmpi.sdk.operations.qmpi import Expose, Unexpose

if TYPE_CHECKING:
    from netqmpi.runtime.adapters.netqasm import NetQASMCommunicator


class NetQASMCircuitAdapter(Circuit):
    """
    Eager-execution circuit adapter for NetQASM.

    This adapter executes operations immediately on live NetQASM qubits
    while still recording them through the base :class:`Circuit`
    interface.

    Attributes:
        _qubits: Live NetQASM qubits allocated at construction time.
        _results: Classical measurement results indexed by classical bit.
    """

    if TYPE_CHECKING:
        _comm: NetQASMCommunicator

    def __init__(
        self,
        num_qubits: int,
        num_clbits: int,
        comm: NetQASMCommunicator,
    ) -> None:
        """
        Initialize the NetQASM circuit adapter.

        Args:
            num_qubits: Number of qubits to allocate.
            num_clbits: Number of classical result slots.
            comm: Communicator providing the active NetQASM connection.
        """
        super().__init__(num_qubits, num_clbits, comm)
        
        self._qubits: List[Qubit] = [
            self._comm.create_qubit() for _ in range(num_qubits)
        ]
        
        self._results: List[Any] = [None] * num_clbits

        # -- Expose / GHZ bookkeeping (used by the circuit adapter) ---------
        self.qubits_exposed: List[Any] = []
        self.ghz_qubit: Optional[Any] = None

    # ------------------------------------------------------------------
    # Circuit abstract interface
    # ------------------------------------------------------------------

    def translate(self, op: Operation) -> Any:
        """
        Translate an operation.

        Args:
            op: Operation to translate.

        Returns:
            None, because operations are executed eagerly and no translation
            step is required.
        """
        return None

    def build(self) -> dict:
        """
        Finalize the circuit execution and return the collected data.

        Returns:
            A dictionary containing the allocated qubits and the classical
            measurement results.
        """
        self._comm.flush()
        return {'qubits': self._qubits, 'results': self._results}

    # ------------------------------------------------------------------
    # Single-qubit gates
    # ------------------------------------------------------------------

    def h(self, qubit: int) -> NetQASMCircuitAdapter:
        super().h(qubit)
        self._qubits[qubit].H()
        return self

    def x(self, qubit: int) -> NetQASMCircuitAdapter:
        super().x(qubit)
        self._qubits[qubit].X()
        return self

    def y(self, qubit: int) -> NetQASMCircuitAdapter:
        super().y(qubit)
        self._qubits[qubit].Y()
        return self

    def z(self, qubit: int) -> NetQASMCircuitAdapter:
        super().z(qubit)
        self._qubits[qubit].Z()
        return self

    def s(self, qubit: int) -> NetQASMCircuitAdapter:
        super().s(qubit)
        self._qubits[qubit].S()
        return self

    def t(self, qubit: int) -> NetQASMCircuitAdapter:
        super().t(qubit)
        self._qubits[qubit].T()
        return self

    def k(self, qubit: int) -> NetQASMCircuitAdapter:
        """
        Apply the NetQASM-specific K gate.

        Args:
            qubit: Index of the target qubit.

        Returns:
            The current circuit adapter instance.
        """
        self._check_qubit(qubit)
        self._qubits[qubit].K()
        return self

    # ------------------------------------------------------------------
    # Parametric single-qubit gates
    # ------------------------------------------------------------------

    def rx(self, theta: float, qubit: int) -> NetQASMCircuitAdapter:
        super().rx(theta, qubit)
        # NetQASM discrete rotation: angle = n / 2^d * pi
        self._qubits[qubit].rot_X(n=round(theta), d=16)
        return self

    def ry(self, theta: float, qubit: int) -> NetQASMCircuitAdapter:
        super().ry(theta, qubit)
        self._qubits[qubit].rot_Y(n=round(theta), d=16)
        return self

    def rz(self, theta: float, qubit: int) -> NetQASMCircuitAdapter:
        super().rz(theta, qubit)
        self._qubits[qubit].rot_Z(n=round(theta), d=16)
        return self

    # ------------------------------------------------------------------
    # Two-qubit gates
    # ------------------------------------------------------------------

    def cx(self, control: int, target: int) -> NetQASMCircuitAdapter:
        super().cx(control, target)
        self._qubits[control].cnot(self._qubits[target])
        return self

    def cz(self, control: int, target: int) -> NetQASMCircuitAdapter:
        super().cz(control, target)
        self._qubits[control].cphase(self._qubits[target])
        return self

    def swap(self, qubit1: int, qubit2: int) -> NetQASMCircuitAdapter:
        super().swap(qubit1, qubit2)
        # Decompose SWAP into 3 CNOTs
        q1, q2 = self._qubits[qubit1], self._qubits[qubit2]
        q1.cnot(q2)
        q2.cnot(q1)
        q1.cnot(q2)
        return self

    # ------------------------------------------------------------------
    # Non-unitary operations
    # ------------------------------------------------------------------

    def measure(self, qubit: int, cbit: int) -> NetQASMCircuitAdapter:
        super().measure(qubit, cbit)
        self._results[cbit] = self._qubits[qubit].measure()
        return self

    def measure_all(self) -> NetQASMCircuitAdapter:
        """
        Measure every qubit eagerly on NetQASM.

        Returns:
            The current circuit adapter instance.

        Raises:
            ValueError: If there are fewer classical bits than qubits.
        """
        if self._num_clbits < self._num_qubits:
            raise ValueError(
                "Not enough classical bits to measure all qubits "
                f"({self._num_clbits} clbits < {self._num_qubits} qubits)."
            )
        for i in range(self._num_qubits):
            self.measure(i, i)
        return self

    def reset(self, qubit: int) -> NetQASMCircuitAdapter:
        super().reset(qubit)
        # NetQASM has no native reset; the intent is recorded in the container.
        return self

    # ------------------------------------------------------------------
    # Inter-rank communication — P2P (teleportation protocol)
    # ------------------------------------------------------------------

    def qsend(self, qubits: List[int], dest_rank: int) -> NetQASMCircuitAdapter:
        """
        Send qubits to another rank using teleportation.

        Args:
            qubits: Indices of the qubits to send.
            dest_rank: Destination rank.

        Returns:
            The current circuit adapter instance.
        """
        super().qsend(qubits, dest_rank)

        epr_socket = self._comm.get_epr_socket(self._comm.rank, dest_rank)
        socket = self._comm.get_socket(self._comm.rank, dest_rank)

        for q_idx in qubits:
            qubit = self._qubits[q_idx]
            # Create EPR pair
            epr = epr_socket.create_keep()[0]
            # Teleport
            qubit.cnot(epr)
            qubit.H()
            m1 = qubit.measure()
            m2 = epr.measure()
            socket.send_structured(StructuredMessage("Corrections", (m1, m2)))  # type: ignore[arg-type]

            self._comm.flush()

        return self

    def qrecv(self, qubits: List[int], src_rank: int) -> NetQASMCircuitAdapter:
        """
        Receive qubits from another rank into local slots via teleportation.

        Args:
            qubits: Indices of the local qubit slots that will receive the qubits.
            src_rank: Source rank.

        Returns:
            The current circuit adapter instance.
        """
        super().qrecv(qubits, src_rank)
        
        epr_socket = self._comm.get_epr_socket(self._comm.rank, src_rank)
        socket = self._comm.get_socket(self._comm.rank, src_rank)

        for q_idx in qubits:
            epr = epr_socket.recv_keep()[0]
            self._comm.flush()

            # Receive corrections
            m1, m2 = socket.recv_structured().payload
            if m2 == 1:
                epr.X()
            if m1 == 1:
                epr.Z()
            self._comm.flush()
            
            # SWAP the corrected EPR qubit into the local slot
            new_q = self._comm.create_qubit()
            epr.cnot(new_q)
            new_q.cnot(epr)
            epr.cnot(new_q)

            self._qubits[q_idx] = new_q
            self._comm.flush()

        return self

    # ------------------------------------------------------------------
    # Inter-rank communication — collective operations
    # ------------------------------------------------------------------

    @staticmethod
    def _list_split(lst: list, n: int) -> List[list]:
        """
        Split a list into ``n`` chunks as evenly as possible.

        Args:
            lst: List to split.
            n: Number of chunks.

        Returns:
            A list containing the generated chunks.
        """
        avg = len(lst) // n
        rem = len(lst) % n
        chunks: List[list] = []
        start = 0
        for i in range(n):
            end = start + avg + (1 if i < rem else 0)
            chunks.append(lst[start:end])
            start = end
        return chunks

    def qscatter(self, qubits: List[int], sender_rank: int) -> NetQASMCircuitAdapter:
        """
        Scatter qubits from one rank to all ranks.

        Args:
            qubits: Qubits involved in the scatter operation.
            sender_rank: Rank acting as the sender.

        Returns:
            The current circuit adapter instance.
        """
        super().qscatter(qubits, sender_rank)

        rank = self._comm.rank
        size = self._comm.size

        if rank == sender_rank:
            chunks = self._list_split(qubits, size)
            for i in range(size):
                if i != sender_rank:
                    self.qsend(chunks[i], i)
        else:
            self.qrecv(qubits, sender_rank)

        return self

    def qgather(self, qubits: List[int], recv_rank: int) -> NetQASMCircuitAdapter:
        """
        Gather qubits from all ranks into a receiver rank.

        Args:
            qubits: Qubits involved in the gather operation.
            recv_rank: Rank that receives the gathered qubits.

        Returns:
            The current circuit adapter instance.
        """
        super().qgather(qubits, recv_rank)

        self._comm = self._comm
        rank = self._comm.rank
        size = self._comm.size

        if rank == recv_rank:
            for i in range(size):
                if i != recv_rank:
                    self.qrecv(qubits, i)
        else:
            self.qsend(qubits, recv_rank)

        return self

    # ------------------------------------------------------------------
    # Inter-rank communication — expose / unexpose (telegate protocol)
    # ------------------------------------------------------------------

    def expose(self, qubits: List[int], rank: int = 0) -> _NetQASMExposeContext:
        """
        Expose qubits through a GHZ-based telegate protocol.

        Args:
            qubits: Indices of the qubits to expose.
            rank: Rank that acts as the exposer.

        Returns:
            A context manager that automatically performs the matching
            unexpose operation when leaving the ``with`` block.
        """
        for q in qubits:
            self._check_qubit(q)
        return _NetQASMExposeContext(self, qubits, rank)

    def _do_expose(self, qubits: List[int], rank: int) -> None:
        """
        Execute the expose protocol eagerly.

        Args:
            qubits: Indices of the qubits to expose.
            rank: Rank acting as the exposer.
        """
        # Record the operation in the container
        self._ops.add(Expose(qubits, rank))
        
        self.qubits_exposed = [self._qubits[q] for q in qubits]

        # Create GHZ across all ranks
        self.ghz_qubit = self.create_ghz()

        if rank == self._comm.rank:
            # Exposer: CNOT exposed qubit with GHZ, measure GHZ, broadcast
            self.qubits_exposed[0].cnot(self.ghz_qubit)  # type: ignore[union-attr]
            measure = self.ghz_qubit.measure()  # type: ignore[union-attr]

            for rnk in range(self._comm.size):
                if rnk != self._comm.rank:
                    socket = self._comm.get_socket(self._comm.rank, rnk)
                    socket.send_structured(StructuredMessage("Expose", (measure,)))  # type: ignore[arg-type]
                    self._comm.flush()
        else:
            # Non-exposer: receive correction, conditionally apply X
            measure = self._comm.get_socket(self._comm.rank, rank).recv_structured().payload[0]
            bit = int(measure)
            if bit:
                self.ghz_qubit.X()  # type: ignore[union-attr]
            # Make the GHZ qubit accessible as the first "exposed" qubit
            self._qubits[qubits[0]] = self.ghz_qubit  # type: ignore[assignment]

    def _do_unexpose(self, rank: int) -> None:
        """
        Execute the unexpose protocol eagerly.

        Args:
            rank: Rank acting as the exposer.
        """
        self._ops.add(Unexpose(rank))

        self.qubits_exposed.clear()

        if rank == self._comm.rank:
            bits: List[int] = []
            for other_rank in range(self._comm.size):
                if other_rank != self._comm.rank:
                    socket = self._comm.get_socket(self._comm.rank, other_rank)
                    bits.append(int(socket.recv_structured().payload[0]))
                    self._comm.flush()
            # Compute AND of all bits — apply Z correction if all are 1
            if np.bitwise_and.reduce(bits):
                self.ghz_qubit.Z()  # type: ignore[union-attr]
        else:
            # Non-exposer: H on GHZ, measure, send to exposer
            self.ghz_qubit.H()  # type: ignore[union-attr]
            measure = self.ghz_qubit.measure()  # type: ignore[union-attr]
            self._comm.flush()
            socket = self._comm.get_socket(self._comm.rank, rank)
            socket.send_structured(StructuredMessage("Unexpose", (measure,)))  # type: ignore[arg-type]

        self._comm.flush()

    # ------------------------------------------------------------------
    # Collective helpers
    # ------------------------------------------------------------------

    def create_ghz(self) -> Qubit:
        """
        Create a GHZ state across all ranks.

        Returns:
            The local qubit belonging to the distributed GHZ state.
        """
        my_eprs = self._epr_sockets[self._comm.get_rank_name(self._rank)]
        next_epr: Optional[EPRSocket] = None
        prev_epr: Optional[EPRSocket] = None
        next_socket: Optional[Socket] = None
        prev_socket: Optional[Socket] = None

        if self._comm.rank != 0:
            prev_epr = my_eprs[self._comm.get_rank_name(self._comm.get_prev_rank(self._comm.rank))]
            prev_socket = self._comm.get_socket(self._comm.rank, self._comm.get_prev_rank(self._comm.rank))

        if self._comm.rank != self._size - 1:
            next_epr = my_eprs[self._comm.get_rank_name(self._comm.get_next_rank(self._comm.rank))]
            next_socket = self._comm.get_socket(self._comm.rank, self._comm.get_next_rank(self._comm.rank))

        ghz_qubit, _measurement = create_ghz(
            down_epr_socket=prev_epr,
            up_epr_socket=next_epr,
            down_socket=prev_socket,
            up_socket=next_socket,
        )

        return ghz_qubit


# ---------------------------------------------------------------------------
# Expose context manager for NetQASM eager execution
# ---------------------------------------------------------------------------

class _NetQASMExposeContext(_ExposeContext):
    """
    Context manager for the NetQASM expose/unexpose protocol.

    This class inherits from :class:`_ExposeContext` so that its return
    type remains compatible with the base :meth:`Circuit.expose`
    signature.
    """

    def __init__(self, circuit: NetQASMCircuitAdapter, qubits: List[int], rank: int) -> None:
        """
        Initialize the expose context manager.

        Args:
            circuit: Circuit adapter that executes the expose protocol.
            qubits: Indices of the qubits to expose.
            rank: Rank acting as the exposer.
        """
        # Skip _ExposeContext.__init__ — we override __enter__/__exit__ entirely
        self._circuit = circuit
        self._qubits = qubits
        self._rank = rank

    def __enter__(self) -> NetQASMCircuitAdapter:
        """
        Enter the expose context.

        Returns:
            The circuit adapter after executing the expose protocol.
        """
        self._circuit._do_expose(self._qubits, self._rank)
        return self._circuit

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """
        Exit the expose context and execute the unexpose protocol.

        Args:
            exc_type: Exception type, if one was raised.
            exc_val: Exception instance, if one was raised.
            exc_tb: Traceback, if one was raised.

        Returns:
            ``False`` so that exceptions, if any, are not suppressed.
        """
        self._circuit._do_unexpose(self._rank)
        return False