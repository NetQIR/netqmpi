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
from typing import TYPE_CHECKING, Any, List, Optional

from netqasm.sdk import EPRSocket, Qubit
from netqasm.sdk.external import Socket
from netqasm.sdk.toolbox import create_ghz
from netqasm.sdk.classical_communication.message import StructuredMessage

from netqmpi.sdk.circuit import Circuit

from netqmpi.sdk.operations import (
    Operation,
    Gate, ControlledGate, ClassicalControlledGate,
    Measure, Reset, Barrier,
    OperationContainer,
    QSend, QRecv, QScatter, QGather, Expose, Unexpose,
)


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
        
        self._qubits: List[Qubit] = []
        
        self._translated_ops: List[Any] = []
        self._results: List[Any] = [None] * num_clbits

        # -- Expose / GHZ bookkeeping (used by the circuit adapter) ---------
        self.qubits_exposed: List[Any] = []
        self.ghz_qubit: Optional[Any] = None

    # ------------------------------------------------------------------
    # Circuit abstract interface
    # ------------------------------------------------------------------

    def _gate_not_implemented(self, name: str):
        def throw_exception():
            raise NotImplementedError(f"{name} is not yet implemented for the CUNQA backend.")
        
        return throw_exception
        

    def _translate_gate(self, op: Gate):
        """
        Translate a single-qubit unitary gate into a CUNQA instruction.

        Args:
            op: Gate operation to translate.
        """
        
        gate_map = {
            # 1 qubit
            "H":   lambda: self._qubits[op.qubits[0]].H(),
            "X":   lambda: self._qubits[op.qubits[0]].X(),
            "Z":   lambda: self._qubits[op.qubits[0]].Z(),
            "Y":   lambda: self._qubits[op.qubits[0]].Y(),
            "S":   lambda: self._qubits[op.qubits[0]].S(),
            "SDG": self._gate_not_implemented("SDG"),
            "T":   lambda: self._qubits[op.qubits[0]].T(),
            "TDG": self._gate_not_implemented("TDG"),

            # 1 qubit
            "RX": lambda: self._qubits[op.qubits[0]].rot_X(n=round(op.params[0]), d=16),
            "RY": lambda: self._qubits[op.qubits[0]].rot_Y(n=round(op.params[0]), d=16),
            "RZ": lambda: self._qubits[op.qubits[0]].rot_Z(n=round(op.params[0]), d=16),
        }

        if op.name in gate_map:
            self._translated_ops.append(gate_map[op.name])

    def _translate_controlled_gate(self, op: ControlledGate):
        """
        Translate a controlled quantum gate into a CUNQA instruction.

        Args:
            op: Controlled gate operation to translate.
        """
        
        gate_2q = {
            # 2 qubits
            "CX": lambda: self._qubits[op.qubits[0]].cnot(self._qubits[op.qubits[1]]),
            "CZ": lambda: self._qubits[op.qubits[0]].cphase(self._qubits[op.qubits[1]]),
            "SWAP": lambda: self._qubits[op.qubits[0]].cnot(self._qubits[op.qubits[1]]),
            
            # 2 qubits
            "CRZ": self._gate_not_implemented("CRZ"),
        }

        if op.name in gate_2q:
            self._translated_ops.append(gate_2q[op.name])
            

    def _translate_classical_controlled_gate(self, op: ClassicalControlledGate):
        """
        Translate a classically controlled gate into a CUNQA instruction.

        Args:
            op: Classically controlled gate operation to translate.

        Raises:
            NotImplementedError: Always, because this operation is not yet supported.
        """
        raise NotImplementedError("ClassicalControlledGate is not yet implemented for the CUNQA backend.")

    def _translate_measure(self, op: Measure):
        """
        Translate a measurement operation into a CUNQA instruction.

        Args:
            op: Measurement operation to translate.
        """
        def netqasm_measure():
            result = self._qubits[op.qubits[0]].measure()
            return result
        
        self._translated_ops.append(netqasm_measure)

    def _translate_reset(self, op: Reset):
        """
        Translate a reset operation into a CUNQA instruction.

        Args:
            op: Reset operation to translate.
        """
        raise NotImplementedError("Reset is not yet implemented for the CUNQA backend.")

    def _translate_barrier(self, op: Barrier):
        """
        Translate a barrier operation into a CUNQA instruction.

        Args:
            op: Barrier operation to translate.

        Raises:
            NotImplementedError: Always, because this operation is not yet supported.
        """
        raise NotImplementedError("Barrier is not implemented for the CUNQA backend.")

    def _translate_operation_container(self, op: OperationContainer):
        """
        Translate an operation container by recursively translating its children.

        Args:
            op: Operation container to translate.
        """
        
        for child in op.flatten():
            self.translate(child)

    def _translate_qsend(self, op: QSend):
        """
        Translate a quantum send operation into a CUNQA instruction.

        Args:
            op: Quantum send operation to translate.
        """
        
        def netqasm_qsend():
            epr_socket = self._comm.get_epr_socket(self._comm.rank, op.dest_rank)
            socket = self._comm.get_socket(self._comm.rank, op.dest_rank)

            for q_idx in op.qubits:
                qubit = self._qubits[q_idx]
                # Create EPR pair
                epr = epr_socket.create_keep()[0]
                # Teleport
                qubit.cnot(epr)
                qubit.H()
                m1 = qubit.measure()
                m2 = epr.measure()
                socket.send_structured(StructuredMessage("Corrections", (m1, m2))) 
        
        self._translated_ops.append(netqasm_qsend)

    def _translate_qrecv(self, op: QRecv):
        """
        Translate a quantum receive operation into a CUNQA instruction.

        Args:
            op: Quantum receive operation to translate.
        """
        def netqasm_qrecv():
            epr_socket = self._comm.get_epr_socket(self._comm.rank, op.src_rank)
            socket = self._comm.get_socket(self._comm.rank, op.src_rank)

            for q_idx in op.qubits:
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
        
        self._translated_ops.append(netqasm_qrecv)

    def _translate_qscatter(self, op: QScatter):
        """
        Translate a quantum scatter operation into CUNQA instructions.

        Args:
            op: Quantum scatter operation to translate.

        Raises:
            NotImplementedError: Always, because this operation is not yet supported.
        """
        def netqasm_qscatter():
            rank = self._comm.rank
            size = self._comm.size

            if rank == op.sender_rank:
                chunks = self._list_split(op.qubits, size)
                for i in range(size):
                    if i != op.sender_rank:
                        self.qsend(chunks[i], i)
            else:
                self.qrecv(op.qubits, op.sender_rank)
        
        self._translated_ops.append(netqasm_qscatter)

    def _translate_qgather(self, op: QGather):
        """
        Translate a quantum gather operation into CUNQA instructions.

        Args:
            op: Quantum gather operation to translate.

        Raises:
            NotImplementedError: Always, because this operation is not yet supported.
        """
        def netqasm_qgather():
            rank = self._comm.rank
            size = self._comm.size

            if rank == op.recv_rank:
                for i in range(size):
                    if i != op.recv_rank:
                        self.qrecv(op.qubits, i)
            else:
                self.qsend(op.qubits, op.recv_rank)
        
        self._translated_ops.append(netqasm_qgather)

    def _translate_expose(self, op: Expose):
        """
        Translate an expose operation into a CUNQA instruction.

        Args:
            op: Expose operation to translate.

        Raises:
            NotImplementedError: Always, because this operation is not yet supported.
        """
        raise NotImplementedError("Expose is not yet implemented for the CUNQA backend.")

    def _translate_unexpose(self, op: Unexpose):
        """
        Translate an unexpose operation into a CUNQA instruction.

        Args:
            op: Unexpose operation to translate.

        Raises:
            NotImplementedError: Always, because this operation is not yet supported.
        """
        raise NotImplementedError("Unexpose is not yet implemented for the CUNQA backend.")
    
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
        self._qubits = [
            self._comm.create_qubit() for _ in range(self.num_qubits)
        ]
        super().translate(op)
        return self._translated_ops
    
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

        ghz_qubit, _ = create_ghz(
            down_epr_socket=prev_epr,
            up_epr_socket=next_epr,
            down_socket=prev_socket,
            up_socket=next_socket,
        )

        return ghz_qubit