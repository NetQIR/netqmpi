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
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set, Tuple

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
from netqmpi.sdk.operations.qmpi import CollectiveOperation


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
        # Outcomes of the telegate protocol, one per protocol classical bit
        # reserved by the trace. They live apart from the user's bits, so a
        # window never overwrites a measurement, and they are per round.
        self._protocol_bits: List[int] = []

        # -- Expose / GHZ bookkeeping (used by the circuit adapter) ---------
        self.qubits_exposed: List[Any] = []
        self.ghz_qubit: Optional[Any] = None

    # ------------------------------------------------------------------
    # Circuit abstract interface
    # ------------------------------------------------------------------

    def reset_round(self) -> None:
        """
        Clear the state a simulated round leaves behind.

        Only the qubits and the protocol bits are per-round. The emitted
        operations are closures that look their qubits up when they run, so
        one translation serves every shot; re-translating each time simply
        grew the list by a full copy of the program.

        The qubit slots run past the data qubits: one per communication
        qubit the trace reserved, where a receiver keeps the control an
        ``expose`` window lends it, at the index
        :meth:`~netqmpi.sdk.circuit.Circuit.comm_qubit` gives.
        """
        self._qubits = [None] * (self.num_qubits + self.num_comm_qubits)
        self._protocol_bits = [0] * self.num_protocol_clbits

    @property
    def translated_ops(self) -> List[Any]:
        """The operations emitted for this circuit, in program order."""
        return self._translated_ops

    def peers(self) -> Set[int]:
        """
        Return the ranks this circuit shares EPR pairs or messages with.

        A transfer pairs the two ranks it joins. A telegate window is a
        star: the root shares a pair with every receiver, and a receiver
        only with the root.

        Returns:
            The peer ranks, from the traced program.
        """
        peers = set()
        for op in self.ops.flatten():
            if isinstance(op, QSend):
                peers.add(op.dest_rank)
            elif isinstance(op, QRecv):
                peers.add(op.src_rank)
            elif isinstance(op, (Expose, Unexpose)):
                if op.rank == op.root:
                    peers.update(op.receivers)
                else:
                    peers.add(op.root)
        return peers

    def peak_qubits(self) -> int:
        """
        Return the most qubits this circuit holds at once on its node.

        Follows the adapter's allocation rules over the traced program: a
        slot holds a qubit from its first use until it is measured or sent,
        a ``qsend`` holds the EPR half next to the qubit it sends, and a
        ``qrecv`` holds the incoming EPR half next to whatever its slot held
        until the correction lands. In a telegate window a receiver holds
        its communication qubit from ``expose`` to ``unexpose``, and the
        root holds its data qubit plus one EPR half at a time while it
        opens the window.

        Returns:
            The peak number of qubits live at the same time.
        """
        live: Set[int] = set()
        peak = 0
        for op in self.ops.flatten():
            if isinstance(op, QSend):
                for qubit in op.qubits:
                    live.add(qubit)
                    peak = max(peak, len(live) + 1)
                    live.discard(qubit)
            elif isinstance(op, QRecv):
                for qubit in op.qubits:
                    peak = max(peak, len(live) + 1)
                    live.add(qubit)
            elif isinstance(op, Measure):
                live.update(op.qubits)
                peak = max(peak, len(live))
                live.difference_update(op.qubits)
            elif isinstance(op, Expose):
                if op.rank == op.root:
                    # Each receiver's EPR half is measured before the next
                    # one is created, so the root never holds two.
                    live.add(op.data_qubit)
                    peak = max(peak, len(live) + 1)
                else:
                    # The incoming EPR half *is* the communication qubit.
                    live.add(self.comm_qubit(op.comm_slot))
                    peak = max(peak, len(live))
            elif isinstance(op, Unexpose):
                if op.rank == op.root:
                    live.add(op.data_qubit)
                    peak = max(peak, len(live))
                else:
                    live.discard(self.comm_qubit(op.comm_slot))
            else:
                live.update(q for q in op.qubits if q < self.num_qubits)
                peak = max(peak, len(live))
        return peak

    def _qubit(self, index: int) -> Qubit:
        """
        Return the live NetQASM qubit backing a slot, allocating it if empty.

        A slot is empty before its first use, and again after a measurement
        or a ``qsend`` consumed what it held; in both cases the SDK's
        contract is that the index still names a qubit in ``|0>``, which is
        what allocating here provides.

        A communication qubit is never allocated here: it only exists while
        an ``expose`` window has lent it a control, and a fresh ``|0>`` in
        its place would be a control that silently never fires.

        Args:
            index: Data or communication qubit index.

        Returns:
            The qubit currently backing that slot.

        Raises:
            RuntimeError: If ``index`` is a communication qubit holding no
                exposed control, because it was measured inside the window.
        """
        qubit = self._qubits[index]
        if qubit is None:
            if index >= self.num_qubits:
                raise RuntimeError(
                    f"rank {self._comm.rank}: communication qubit {index} "
                    f"(slot {index - self.num_qubits}) holds no exposed "
                    f"control: it was measured before its expose window "
                    f"closed, and the NetQASM backend cannot close the "
                    f"window without it.")
            qubit = self._qubits[index] = self._comm.create_qubit()
        return qubit

    def _swap(self, first: int, second: int) -> None:
        """
        Exchange two local qubits.

        Three CNOTs, on both NetQASM releases. 2.x does expose
        ``Qubit.swap``, but it builds an instruction the SquidASM release
        this backend runs on cannot execute — the simulation simply never
        finishes — so the hand-assembled version is the portable one. See
        :mod:`netqmpi.runtime.adapters.netqasm._compat`.

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
        Translate this rank's part of an expose window (cat-entangler).

        NetQASM runs one program per rank, so every participant emits its
        own half of the protocol, as ``qsend``/``qrecv`` do, and no joint
        expansion is needed. The window is opened as a star, one receiver
        at a time in the order of ``op.ranks``:

        - the receiver tells the root it is ready, once the qubits it
          already holds are committed (see the receiver's comment);
        - the root creates an EPR pair with the receiver, applies
          ``CNOT(data, half)``, measures its half and sends the outcome;
        - the receiver keeps the other half as its communication qubit and
          applies ``X`` to it if the outcome is 1.

        This leaves ``a|0>|0...0> + b|1>|1...1>`` over the root's data qubit
        and the receivers' communication qubits, and the root holds a single
        extra EPR half at a time. The outcomes go to the protocol bits
        ``op.clbits`` the trace reserved: one per receiver on the root, one
        on each receiver.

        Args:
            op: This rank's expose record.
        """
        header = f"expose {op.tag}"

        def netqasm_expose_root():
            data = self._qubit(op.data_qubit)
            for receiver, cbit in zip(op.receivers, op.clbits):
                epr_socket = self._comm.get_epr_socket(self._comm.rank, receiver)
                socket = self._comm.get_socket(self._comm.rank, receiver)

                # See the receiver: no pair is requested until both nodes
                # have committed the qubits they already hold.
                self._comm.flush()
                self._receive(socket, f"ready {op.tag}", receiver)
                half = epr_socket.create_keep()[0]
                data.cnot(half)
                outcome = half.measure()
                # Resolve the future before it goes on the wire; see qsend.
                self._comm.flush()
                self._protocol_bits[cbit] = int(outcome)
                socket.send_structured(
                    StructuredMessage(header, self._protocol_bits[cbit]))

        def netqasm_expose_receiver():
            epr_socket = self._comm.get_epr_socket(self._comm.rank, op.root)
            socket = self._comm.get_socket(self._comm.rank, op.root)

            # Commit the qubits this rank already holds, then tell the root
            # it may request the pair. The link layer picks the memory
            # position for this node's half when the root's request arrives;
            # a local qubit not yet allocated by then could be given that
            # same position, and the first gate between the two failed with
            # "qubit_mapping [0, 0] contains duplicate entries" — a few
            # rounds in twenty, since it races the root's request.
            self._comm.flush()
            socket.send_structured(StructuredMessage(f"ready {op.tag}", 0))
            half = epr_socket.recv_keep()[0]
            self._comm.flush()
            outcome = self._receive(socket, header, op.root)
            self._protocol_bits[op.clbits[0]] = outcome
            if outcome == 1:
                half.X()
            self._qubits[self.comm_qubit(op.comm_slot)] = half
            self._comm.flush()

        self._translated_ops.append(
            netqasm_expose_root if op.rank == op.root else netqasm_expose_receiver)

    def _translate_unexpose(self, op: Unexpose):
        """
        Translate this rank's part of an unexpose (cat-disentangler).

        Each receiver measures its communication qubit in the X basis and
        sends the outcome to the root, which applies ``Z`` to its data qubit
        if the outcomes have odd parity. The root reads the receivers in
        the order of ``op.ranks`` into its protocol bits, overwriting the
        expose outcomes it no longer needs.

        Args:
            op: This rank's unexpose record.
        """
        header = f"unexpose {op.tag}"

        def netqasm_unexpose_root():
            parity = 0
            for receiver, cbit in zip(op.receivers, op.clbits):
                socket = self._comm.get_socket(self._comm.rank, receiver)
                self._protocol_bits[cbit] = self._receive(socket, header, receiver)
                parity ^= self._protocol_bits[cbit]
            if parity == 1:
                self._qubit(op.data_qubit).Z()

        def netqasm_unexpose_receiver():
            socket = self._comm.get_socket(self._comm.rank, op.root)
            index = self.comm_qubit(op.comm_slot)

            control = self._qubit(index)
            control.H()
            outcome = control.measure()
            self._comm.flush()
            # Measuring consumed the qubit, and the window is closed.
            self._qubits[index] = None
            self._protocol_bits[op.clbits[0]] = int(outcome)
            socket.send_structured(
                StructuredMessage(header, self._protocol_bits[op.clbits[0]]))

        self._translated_ops.append(
            netqasm_unexpose_root if op.rank == op.root else netqasm_unexpose_receiver)

    def _receive(self, socket: Socket, header: str, peer: int) -> int:
        """
        Receive one telegate outcome and check it belongs to this call.

        The messages between two ranks share one socket per round and arrive
        in the order they were sent, so a header that does not match means
        the two ranks reached their calls in different orders. Reading the
        bit anyway would apply another call's correction and return a
        plausible, wrong histogram.

        Args:
            socket: Classical socket to the peer.
            header: Header the expected message carries.
            peer: Rank at the other end, for the error message.

        Returns:
            The outcome, 0 or 1.

        Raises:
            RuntimeError: If the message is not the one this call expects.
        """
        message = socket.recv_structured()
        if message.header != header:
            raise RuntimeError(
                f"rank {self._comm.rank} expected the {header!r} outcome from "
                f"rank {peer} but received {message.header!r} "
                f"({message.payload!r}): the two ranks reached their transfers "
                f"and telegate windows in different orders.")
        return int(message.payload)

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

# ----------------------------------------------------------------------
# Cross-rank check of the collective calls
# ----------------------------------------------------------------------

def check_collectives(circuits: Dict[int, NetQASMCircuitAdapter], size: int) -> None:
    """
    Refuse a program whose ranks cannot all get through their collectives.

    Each rank runs its own program, so an ``expose`` that a participant
    never reaches, or reaches after a different window, does not fail: the
    ranks wait on each other until SquidASM's wall-clock timeout, if ever.
    This replays the traced order of the collective calls instead — every
    rank advances to its next collective, and a call goes through once all
    its participants sit on it — and reports, before the simulation starts,
    the calls that can never complete.

    Transfers are left out: a ``qsend`` and its ``qrecv`` are paired at run
    time and do not take part in this check.

    Args:
        circuits: One circuit per rank, keyed by rank; the circuits that form
            one distributed program.
        size: Number of ranks in the run.

    Raises:
        RuntimeError: If a collective names a rank outside the run, or if
            the ranks block on collectives that never match — the trace
            equivalent of a deadlock.
    """
    streams = {
        rank: [op for op in circuit.ops.flatten()
               if isinstance(op, CollectiveOperation)]
        for rank, circuit in circuits.items()
    }
    cursors = {rank: 0 for rank in streams}

    while True:
        blocked = {rank: streams[rank][cursors[rank]] for rank in streams
                   if cursors[rank] < len(streams[rank])}
        if not blocked:
            return

        ready: Optional[CollectiveOperation] = None
        for rank in sorted(blocked):
            op = blocked[rank]
            outside = [r for r in op.ranks if not 0 <= r < size]
            if outside:
                raise RuntimeError(
                    f"rank {rank} called {_describe_collective(op)}, which names "
                    f"{_rank_list(outside)}, but this run has {size} ranks, "
                    f"numbered 0 to {size - 1}. A collective can only name ranks "
                    f"taking part in the run, so nothing can ever match that call.")
            if all(r in blocked and op.matches(blocked[r]) for r in op.ranks):
                ready = op
                break

        if ready is None:
            raise RuntimeError(_deadlock_report(blocked))
        for rank in ready.ranks:
            cursors[rank] += 1


def _rank_list(ranks: List[int]) -> str:
    """
    Spell a list of ranks the way a sentence would.

    Args:
        ranks: Ranks to name.

    Returns:
        ``"rank 7"``, ``"ranks 7 and 9"``, ``"ranks 7, 9 and 11"``.
    """
    names = [str(r) for r in sorted(ranks)]
    if len(names) == 1:
        return f"rank {names[0]}"
    return "ranks " + ", ".join(names[:-1]) + " and " + names[-1]


def _describe_collective(op: CollectiveOperation) -> str:
    """
    Spell a collective record the way the user wrote the call.

    Args:
        op: Record of an ``expose`` or ``unexpose``.

    Returns:
        The call as it appears in the program, receivers and root.
    """
    name = "expose" if isinstance(op, Expose) else "unexpose"
    return f"{name}(ranks={op.ranks[1:]}, root={op.ranks[0]})"


def _deadlock_report(blocked: Dict[int, CollectiveOperation]) -> str:
    """
    Explain which ranks reached a collective call and which did not.

    The wording is the CUNQA backend's, so a program that deadlocks reads
    the same on both.

    Args:
        blocked: Collective each blocked rank is waiting on, keyed by rank.

    Returns:
        The message of the deadlock error.
    """
    calls: Dict[Tuple[Any, ...], CollectiveOperation] = {}
    for rank in sorted(blocked):
        op = blocked[rank]
        calls.setdefault((type(op).__name__, op.tag, tuple(op.ranks)), op)

    lines: List[str] = []
    never_called = False
    called_something_else = False
    for op in calls.values():
        reached = [r for r in op.ranks
                   if r in blocked and op.matches(blocked[r])]
        missing = [r for r in op.ranks if r not in reached]
        lines.append(f"  {_describe_collective(op)}")
        lines.append(
            "      called by:  "
            + ", ".join(f"rank {r}" for r in sorted(reached)))
        for rank in sorted(missing):
            if rank in blocked:
                called_something_else = True
                doing = f"is waiting on {_describe_collective(blocked[rank])}"
            else:
                never_called = True
                doing = "finished its block without calling it"
            lines.append(f"      missing:    rank {rank}, which {doing}")

    if never_called:
        lines.append(
            "  Every rank a collective names has to call it, just as every rank "
            "of an MPI_Bcast calls it.")
    if called_something_else:
        lines.append(
            "  These ranks are waiting on different calls: the participants of a "
            "collective have to reach theirs in the same order, and open and "
            "close nested windows in the same order too.")

    return (
        "Deadlock while translating the group: a collective call is waiting "
        "for ranks that never reached it.\n"
        + "\n".join(lines)
    )
