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
        
        # One slot per data qubit, filled on first use. Allocating them all
        # up front meant a rank that only ever receives still held a qubit
        # per slot that qrecv then had to free, and freeing it raced with
        # the transfer often enough to abort roughly one run in four.
        self._qubits: List[Optional[Qubit]] = [None] * num_qubits
        
        self._translated_ops: List[Any] = []
        self._results: List[Any] = [None] * num_clbits

        # -- Expose / GHZ bookkeeping (used by the circuit adapter) ---------
        self.qubits_exposed: List[Any] = []
        self.ghz_qubit: Optional[Any] = None

    # ------------------------------------------------------------------
    # Circuit abstract interface
    # ------------------------------------------------------------------

    def reset_round(self) -> None:
        """
        Clear the state a simulated round leaves behind.

        Only the qubits are per-round. The emitted operations are closures
        that look their qubits up when they run, so one translation serves
        every shot; re-translating each time simply grew the list by a full
        copy of the program.
        """
        self._qubits = [None] * self.num_qubits

    @property
    def translated_ops(self) -> List[Any]:
        """The operations emitted for this circuit, in program order."""
        return self._translated_ops

    def _qubit(self, index: int) -> Qubit:
        """
        Return the live NetQASM qubit backing a slot, allocating it if empty.

        A slot is empty before its first use, and again after a measurement
        or a ``qsend`` consumed what it held; in both cases the SDK's
        contract is that the index still names a qubit in ``|0>``, which is
        what allocating here provides.

        Args:
            index: Data qubit index.

        Returns:
            The qubit currently backing that slot.
        """
        if self._qubits[index] is None:
            self._qubits[index] = self._comm.create_qubit()
        return self._qubits[index]

    def _swap(self, first: int, second: int) -> None:
        """
        Exchange two local qubits.

        NetQASM has no SWAP instruction, so it is built from three CNOTs.
        This used to emit a single CNOT, which is a different gate
        altogether and quietly produced the wrong state.

        Args:
            first: First qubit index.
            second: Second qubit index.
        """
        a, b = self._qubit(first), self._qubit(second)
        a.cnot(b)
        b.cnot(a)
        a.cnot(b)

    def _gate_not_implemented(self, name: str):
        def throw_exception():
            raise NotImplementedError(
                f"{name} is not implemented for the NetQASM backend.")
        
        return throw_exception
        

    def _translate_gate(self, op: Gate):
        """
        Translate a single-qubit unitary gate into a NetQASM instruction.

        Args:
            op: Gate operation to translate.
        """
        
        gate_map = {
            # 1 qubit
            "H":   lambda: self._qubit(op.qubits[0]).H(),
            "X":   lambda: self._qubit(op.qubits[0]).X(),
            "Z":   lambda: self._qubit(op.qubits[0]).Z(),
            "Y":   lambda: self._qubit(op.qubits[0]).Y(),
            "S":   lambda: self._qubit(op.qubits[0]).S(),
            "SDG": self._gate_not_implemented("SDG"),
            "T":   lambda: self._qubit(op.qubits[0]).T(),
            "TDG": self._gate_not_implemented("TDG"),

            # 1 qubit
            "RX": lambda: self._qubit(op.qubits[0]).rot_X(angle=op.params[0]),
            "RY": lambda: self._qubit(op.qubits[0]).rot_Y(angle=op.params[0]),
            "RZ": lambda: self._qubit(op.qubits[0]).rot_Z(angle=op.params[0]),

            # 2 qubits. The SDK records a swap as a plain Gate over two
            # qubits, so it is dispatched here and not through the
            # controlled-gate table, where it used to sit unreachable.
            "SWAP": lambda: self._swap(op.qubits[0], op.qubits[1]),
        }

        if op.name not in gate_map:
            # Silently skipping produced a circuit missing the gate and a
            # plausible-looking histogram for a program that never ran.
            raise NotImplementedError(
                f"Gate '{op.name}' is not implemented for the NetQASM "
                f"backend.")
        self._translated_ops.append(gate_map[op.name])

    def _translate_controlled_gate(self, op: ControlledGate):
        """
        Translate a controlled quantum gate into a NetQASM instruction.

        Args:
            op: Controlled gate operation to translate.
        """
        
        # A ControlledGate carries its controls and a list of target *gates*;
        # it has no name of its own. Reading op.name here raised
        # AttributeError for every controlled gate, so CX and CZ had never
        # reached the backend at all.
        if len(op.controls) != 1 or len(op.targets) != 1:
            raise NotImplementedError(
                "Only single-control, single-target gates are implemented "
                "for the NetQASM backend.")

        control = op.controls[0]
        target_gate = op.targets[0]
        target = target_gate.qubits[0]

        gate_2q = {
            "X": lambda: self._qubit(control).cnot(self._qubit(target)),
            "Z": lambda: self._qubit(control).cphase(self._qubit(target)),
        }

        if target_gate.name not in gate_2q:
            raise NotImplementedError(
                f"Controlled-{target_gate.name} is not implemented for the "
                f"NetQASM backend.")
        self._translated_ops.append(gate_2q[target_gate.name])
            

    def _translate_classical_controlled_gate(self, op: ClassicalControlledGate):
        """
        Translate a classically controlled gate into a NetQASM instruction.

        Args:
            op: Classically controlled gate operation to translate.

        Raises:
            NotImplementedError: Always, because this operation is not yet supported.
        """
        raise NotImplementedError("ClassicalControlledGate is not implemented for the NetQASM backend.")

    def _translate_measure(self, op: Measure):
        """
        Translate a measurement operation into a NetQASM instruction.

        Args:
            op: Measurement operation to translate.
        """
        def netqasm_measure():
            result = self._qubit(op.qubits[0]).measure()
            # Measuring consumes the qubit in NetQASM. Emptying the slot
            # keeps the index usable: the SDK lets a program measure and
            # then carry on with that qubit, which it would read as |0>.
            self._qubits[op.qubits[0]] = None
            # The classical bit travels with the outcome so the caller can
            # place it in the shot's bit string; returning the outcome alone
            # left no way to tell which bit it belonged to.
            return op.cbit, result
        
        self._translated_ops.append(netqasm_measure)

    def _translate_reset(self, op: Reset):
        """
        Translate a reset operation into a NetQASM instruction.

        Args:
            op: Reset operation to translate.
        """
        raise NotImplementedError("Reset is not implemented for the NetQASM backend.")

    def _translate_barrier(self, op: Barrier):
        """
        Translate a barrier operation into a NetQASM instruction.

        Args:
            op: Barrier operation to translate.

        Raises:
            NotImplementedError: Always, because this operation is not yet supported.
        """
        raise NotImplementedError("Barrier is not implemented for the NetQASM backend.")

    def _translate_operation_container(self, op: OperationContainer):
        """
        Translate an operation container by recursively translating its children.

        Args:
            op: Operation container to translate.
        """
        
        for child in op.children:
            self.translate(child)

    def _translate_qsend(self, op: QSend):
        """
        Translate a quantum send operation into a NetQASM instruction.

        Args:
            op: Quantum send operation to translate.
        """
        
        def netqasm_qsend():
            epr_socket = self._comm.get_epr_socket(self._comm.rank, op.dest_rank)
            socket = self._comm.get_socket(self._comm.rank, op.dest_rank)

            for q_idx in op.qubits:
                qubit = self._qubit(q_idx)
                # Create EPR pair
                epr = epr_socket.create_keep()[0]
                # Bell measurement
                qubit.cnot(epr)
                qubit.H()
                m1 = qubit.measure()
                m2 = epr.measure()

                # The outcomes are futures until the subroutine is flushed;
                # sending them unresolved put placeholder objects on the
                # socket instead of the two correction bits.
                self._comm.flush()
                socket.send_structured(
                    StructuredMessage("Corrections", (int(m1), int(m2))))

                # Teleporting moves the state: measuring consumed the qubit,
                # so the slot goes back to being empty and the SDK's promise
                # that a sent qubit is left in |0> is kept by re-allocating
                # it the next time the index is used.
                self._qubits[q_idx] = None
        
        self._translated_ops.append(netqasm_qsend)

    def _translate_qrecv(self, op: QRecv):
        """
        Translate a quantum receive operation into a NetQASM instruction.

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

                # The corrected EPR half *is* the teleported state, so it
                # becomes the slot. Swapping it into a freshly created qubit
                # instead, as this did before, left both the EPR half and the
                # slot's original qubit allocated for the rest of the run:
                # three qubits held where one was needed.
                occupant = self._qubits[q_idx]
                if occupant is not None and occupant.active:
                    # Receiving into a slot destroys whatever it held; the
                    # SDK documents that, so the qubit is released rather
                    # than leaked.
                    occupant.free()
                self._qubits[q_idx] = epr

                self._comm.flush()
        
        self._translated_ops.append(netqasm_qrecv)

    def _translate_qscatter(self, op: QScatter):
        """
        Translate a quantum scatter operation into NetQASM instructions.

        The record already holds this rank's half of the exchange as
        teleportation blocks — sends on the root, receives elsewhere — so
        translating them in order is the whole scatter.

        Args:
            op: Quantum scatter operation to translate.
        """
        self._translate_operation_container(op)

    def _translate_qgather(self, op: QGather):
        """
        Translate a quantum gather operation into NetQASM instructions.

        Args:
            op: Quantum gather operation to translate.
        """
        self._translate_operation_container(op)

    def _translate_expose(self, op: Expose):
        """
        Translate an expose operation into a NetQASM instruction.

        Args:
            op: Expose operation to translate.

        Raises:
            NotImplementedError: Always, because this operation is not yet supported.
        """
        raise NotImplementedError("Expose is not implemented for the NetQASM backend.")

    def _translate_unexpose(self, op: Unexpose):
        """
        Translate an unexpose operation into a NetQASM instruction.

        Args:
            op: Unexpose operation to translate.

        Raises:
            NotImplementedError: Always, because this operation is not yet supported.
        """
        raise NotImplementedError("Unexpose is not implemented for the NetQASM backend.")
    
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