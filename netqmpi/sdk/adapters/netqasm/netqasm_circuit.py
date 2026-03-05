"""
Circuit adapter for the NetQASM backend — eager execution model.

NetQASM has no circuit object: every instruction is dispatched to the
simulator the moment it is called on a :class:`netqasm.sdk.qubit.Qubit`.
Therefore this adapter:

1. Allocates a :class:`Qubit` array in ``__init__`` using the live
   ``connection`` obtained via ``environment.comm.connection``.
2. Overrides every gate method of the base :class:`Circuit` so that each
   one *first* delegates to ``super()`` (recording the operation in the
   :class:`OperationContainer`) and *then* executes the corresponding
   NetQASM SDK call on the qubit immediately.
3. Overrides the inter-rank communication primitives (``qsend``,
   ``qrecv``, ``qscatter``, ``qgather``, ``expose``, ``unexpose``) with
   the concrete teleportation / collective protocols from the NetQASM SDK.
4. Keeps ``translate()`` as a no-op — execution has already happened.
5. In ``build()`` flushes the connection and returns the qubit array
   together with the classical results.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, List, Optional

import numpy as np
from netqasm.sdk.classical_communication.message import StructuredMessage
from netqasm.sdk.qubit import Qubit

from netqmpi.sdk.core.circuit import Circuit, _ExposeContext
from netqmpi.sdk.core.operations.operation import Operation
from netqmpi.sdk.core.operations.qmpi import Expose, Unexpose

if TYPE_CHECKING:
    from netqmpi.runtime.adapters.netqasm.netqasm_communicator import NetQASMCommunicator
    from netqmpi.sdk.core.environment import Environment


class NetQASMCircuitAdapter(Circuit):
    """
    Eager-execution circuit adapter for NetQASM.

    ``environment`` must be an
    :class:`~netqmpi.sdk.core.environment.Environment` whose ``comm``
    exposes a ``connection`` of type
    :class:`netqasm.sdk.external.NetQASMConnection`.

    Attributes:
        _qubits  (List[Qubit]): Live NetQASM qubits allocated at construction.
        _results (List[Any]):   Classical measurement results indexed by cbit.
    """

    def __init__(
        self,
        num_qubits: int,
        num_clbits: int,
        environment: Optional[Environment] = None,
    ) -> None:
        """
        Args:
            num_qubits:  Number of qubits to allocate.
            num_clbits:  Number of classical result slots.
            environment: The owning
                         :class:`~netqmpi.sdk.core.environment.Environment`.
                         Its ``comm.connection`` is used to allocate
                         :class:`Qubit` instances immediately.
        """
        super().__init__(num_qubits, num_clbits, environment)

        if environment is not None:
            connection = environment.comm.connection
            self._qubits: List[Qubit] = [
                Qubit(connection) for _ in range(num_qubits)
            ]
        else:
            self._qubits: List[Qubit] = []
        self._results: List[Any] = [None] * num_clbits

    # ------------------------------------------------------------------
    # Private helpers – safe accessors for the environment objects
    # ------------------------------------------------------------------

    @property
    def _comm(self):
        """Return ``self._environment.comm`` — assumes environment is set."""
        assert self._environment is not None, (
            "NetQASMCircuitAdapter requires an Environment for communication ops."
        )
        return self._environment.comm

    @property
    def _backend(self) -> NetQASMCommunicator:
        """Return the underlying :class:`NetQASMCommunicator` backend."""
        # noinspection PyProtectedMember
        from netqmpi.runtime.adapters.netqasm.netqasm_communicator import NetQASMCommunicator as _Comm
        backend = self._comm._backend
        assert isinstance(backend, _Comm)
        return backend

    # ------------------------------------------------------------------
    # Circuit abstract interface
    # ------------------------------------------------------------------

    def operations_supported(self) -> List[str]:
        return [
            'H', 'X', 'Y', 'Z', 'S', 'T', 'K',
            'CNOT', 'CZ', 'CPHASE',
            'RX', 'RY', 'RZ',
            'measure', 'reset',
        ]

    def translate(self, op: Operation) -> Any:
        """No-op: every operation is executed eagerly in the gate overrides."""
        return None

    def build(self) -> dict:
        """Flush the connection and return qubits + classical results."""
        if self._environment is not None:
            self._environment.comm.flush()
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
        """K gate (NetQASM-specific, not in the base Circuit API)."""
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
        """Measure every qubit, executing eagerly on NetQASM."""
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
        """Send *qubits* to *dest_rank* using teleportation."""
        super().qsend(qubits, dest_rank)

        comm = self._comm
        epr_socket = comm.get_epr_socket(comm.rank, dest_rank)
        socket = comm.get_socket(comm.rank, dest_rank)

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

            comm.flush()

        return self

    def qrecv(self, qubits: List[int], src_rank: int) -> NetQASMCircuitAdapter:
        """Receive qubits from *src_rank* into local slots via teleportation."""
        super().qrecv(qubits, src_rank)

        comm = self._comm
        epr_socket = comm.get_epr_socket(comm.rank, src_rank)
        socket = comm.get_socket(comm.rank, src_rank)

        for q_idx in qubits:
            epr = epr_socket.recv_keep()[0]
            comm.flush()

            # Receive corrections
            m1, m2 = socket.recv_structured().payload
            if m2 == 1:
                epr.X()
            if m1 == 1:
                epr.Z()
            comm.flush()

            # SWAP the corrected EPR qubit into the local slot
            new_q = Qubit(comm.connection)
            epr.cnot(new_q)
            new_q.cnot(epr)
            epr.cnot(new_q)

            self._qubits[q_idx] = new_q

            comm.flush()

        return self

    # ------------------------------------------------------------------
    # Inter-rank communication — collective operations
    # ------------------------------------------------------------------

    @staticmethod
    def _list_split(lst: list, n: int) -> List[list]:
        """Split *lst* into *n* chunks as evenly as possible."""
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
        """Scatter *qubits* from *sender_rank* to all ranks."""
        super().qscatter(qubits, sender_rank)

        comm = self._comm
        rank = comm.rank
        size = comm.size

        if rank == sender_rank:
            chunks = self._list_split(qubits, size)
            for i in range(size):
                if i != sender_rank:
                    self.qsend(chunks[i], i)
        else:
            self.qrecv(qubits, sender_rank)

        return self

    def qgather(self, qubits: List[int], recv_rank: int) -> NetQASMCircuitAdapter:
        """Gather each rank's *qubits* into *recv_rank*."""
        super().qgather(qubits, recv_rank)

        comm = self._comm
        rank = comm.rank
        size = comm.size

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
        Expose *qubits* via a GHZ-based telegate protocol.

        Returns a context manager; the matching ``unexpose`` runs
        automatically when the ``with`` block exits.
        """
        for q in qubits:
            self._check_qubit(q)
        return _NetQASMExposeContext(self, qubits, rank)

    def _do_expose(self, qubits: List[int], rank: int) -> None:
        """Execute the expose protocol eagerly (called by the context manager)."""
        # Record the operation in the container
        self._ops.add(Expose(qubits, rank))

        comm = self._comm
        backend = self._backend
        backend.qubits_exposed = [self._qubits[q] for q in qubits]

        # Create GHZ across all ranks
        backend.ghz_qubit = comm.create_ghz()

        if rank == comm.rank:
            # Exposer: CNOT exposed qubit with GHZ, measure GHZ, broadcast
            backend.qubits_exposed[0].cnot(backend.ghz_qubit)  # type: ignore[union-attr]
            measure = backend.ghz_qubit.measure()  # type: ignore[union-attr]

            for rnk in range(comm.size):
                if rnk != comm.rank:
                    socket = comm.get_socket(comm.rank, rnk)
                    socket.send_structured(StructuredMessage("Expose", (measure,)))  # type: ignore[arg-type]
                    comm.flush()
        else:
            # Non-exposer: receive correction, conditionally apply X
            measure = comm.get_socket(comm.rank, rank).recv_structured().payload[0]
            bit = int(measure)
            if bit:
                backend.ghz_qubit.X()  # type: ignore[union-attr]
            # Make the GHZ qubit accessible as the first "exposed" qubit
            self._qubits[qubits[0]] = backend.ghz_qubit  # type: ignore[assignment]

    def _do_unexpose(self, rank: int) -> None:
        """Execute the unexpose protocol eagerly (called by the context manager)."""
        self._ops.add(Unexpose(rank))

        comm = self._comm
        backend = self._backend

        backend.qubits_exposed.clear()

        if rank == comm.rank:
            bits: List[int] = []
            for other_rank in range(comm.size):
                if other_rank != comm.rank:
                    socket = comm.get_socket(comm.rank, other_rank)
                    bits.append(int(socket.recv_structured().payload[0]))
                    comm.flush()
            # Compute AND of all bits — apply Z correction if all are 1
            if np.bitwise_and.reduce(bits):
                backend.ghz_qubit.Z()  # type: ignore[union-attr]
        else:
            # Non-exposer: H on GHZ, measure, send to exposer
            backend.ghz_qubit.H()  # type: ignore[union-attr]
            measure = backend.ghz_qubit.measure()  # type: ignore[union-attr]
            comm.flush()
            socket = comm.get_socket(comm.rank, rank)
            socket.send_structured(StructuredMessage("Unexpose", (measure,)))  # type: ignore[arg-type]

        comm.flush()


# ---------------------------------------------------------------------------
# Expose context manager for NetQASM eager execution
# ---------------------------------------------------------------------------

class _NetQASMExposeContext(_ExposeContext):
    """Context manager that runs the NetQASM expose/unexpose protocol eagerly.

    Inherits from :class:`_ExposeContext` so the return type is compatible
    with the base :meth:`Circuit.expose` signature.
    """

    def __init__(self, circuit: NetQASMCircuitAdapter, qubits: List[int], rank: int) -> None:
        # Skip _ExposeContext.__init__ — we override __enter__/__exit__ entirely
        self._circuit = circuit
        self._qubits = qubits
        self._rank = rank

    def __enter__(self) -> NetQASMCircuitAdapter:
        self._circuit._do_expose(self._qubits, self._rank)
        return self._circuit

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self._circuit._do_unexpose(self._rank)
        return False
