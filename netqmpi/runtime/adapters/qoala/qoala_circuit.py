"""
Circuit adapter for the Qoala backend (simulation only).

Qoala programs are structured very differently from an eager gate stream:
a program is a list of *host-code basic blocks* (typed ``CL``/``CC``/``QL``/``QC``)
that invoke *local routines* (NetQASM subroutines) and *request routines*
(EPR generation). This adapter therefore acts as a small compiler that walks
the flat NetQMPI :class:`~netqmpi.sdk.operations.OperationContainer` and emits
the textual ``.iqoala`` representation of one program per rank.

The generated text is parsed into a ``QoalaProgram`` by
:mod:`~netqmpi.runtime.adapters.qoala.qoala_executor` (the only module that
imports ``qoala.*``). Keeping this file free of ``qoala`` imports means the
whole ``.iqoala`` generation is pure-Python and independently testable, and it
preserves the lazy-import contract used by the other backends.

Compilation rules (see ``docs/design/qoala-backend.md`` for the full mapping):

* Consecutive local gates / measurements are accumulated into a single local
  routine, flushed as a ``QL`` block whenever a communication boundary
  (``qsend``/``qrecv``) is reached or at the end of the circuit.
* ``qsend`` becomes the teleportation *sender* triad: a ``QC`` EPR-create
  request, a ``QL`` Bell-state-measurement routine, and a ``CL`` block sending
  the two correction bits.
* ``qrecv`` becomes the teleportation *receiver* triad: a ``QC`` EPR-receive
  request, two ``CC`` blocks receiving the correction bits, and a ``QL`` block
  applying the Pauli corrections. The received qubit lands directly in the
  target virtual-qubit slot.

Scope (v1): local gates, ``measure``, ``qsend`` and ``qrecv``. Every other
inter-rank primitive raises :class:`NotImplementedError`, mirroring the CUNQA
adapter.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, List, Tuple

from netqmpi.sdk.circuit import Circuit
from netqmpi.sdk.operations import (
    Operation,
    Gate, ControlledGate, ClassicalControlledGate,
    Measure, Reset, Barrier,
    OperationContainer,
    QSend, QRecv, QScatter, QGather, Expose, Unexpose,
)

if TYPE_CHECKING:
    from netqmpi.runtime.adapters.qoala.qoala_communicator import QoalaCommunicator


# Rotation resolution used when translating RX/RY/RZ (and S/T) to NetQASM
# ``rot_*`` instructions. A ``rot_x Q n d`` instruction rotates by ``n * pi / 2**d``.
_ROT_DENOM_EXP = 4  # d, i.e. angle unit is pi / 16


@dataclass
class QoalaProgramSpec:
    """
    Backend-neutral description of the ``.iqoala`` program built for one rank.

    Attributes:
        iqoala_text: Full serialized ``.iqoala`` program, ready to be parsed by
            ``QoalaParser``.
        program_input: Mapping of program parameter names to values (remote node
            ids), consumed as a ``ProgramInput`` by the executor.
        num_qubits: Minimum number of physical qubits the node must expose for
            this program (user qubits plus one teleportation scratch slot).
        outputs: Ordered ``(clbit_index, host_var_name)`` pairs describing which
            host variables are returned via ``return_result``, used to rebuild a
            measurement bitstring.
    """

    iqoala_text: str
    program_input: dict
    num_qubits: int
    outputs: List[Tuple[int, str]] = field(default_factory=list)


class QoalaCircuitAdapter(Circuit):
    """
    Compiles a NetQMPI circuit into a textual Qoala program (simulation only).

    The adapter does not execute anything: it records operations through the
    base :class:`~netqmpi.sdk.circuit.Circuit` fluent API and, on
    :meth:`build_program`, walks them to emit the ``.iqoala`` text for this rank.
    """

    if TYPE_CHECKING:
        _comm: QoalaCommunicator

    def __init__(self, num_qubits: int, num_clbits: int, comm: QoalaCommunicator) -> None:
        """
        Initialize the Qoala circuit adapter.

        Args:
            num_qubits: Number of user qubits in the circuit.
            num_clbits: Number of classical bits in the circuit.
            comm: Communicator providing rank/size and rank naming helpers.
        """
        super().__init__(num_qubits, num_clbits, comm)
        self._reset_builder()

    # ------------------------------------------------------------------
    # Builder state
    # ------------------------------------------------------------------

    def _reset_builder(self) -> None:
        """Reset all mutable compilation state."""
        self._host_blocks: List[Tuple[str, List[str]]] = []  # (block type, lines)
        self._subroutines: List[str] = []
        self._requests: List[str] = []
        self._pending: List[Operation] = []                  # buffered local ops
        self._peers: set = set()                             # remote ranks talked to
        self._outputs: List[Tuple[int, str]] = []            # (clbit, host var)
        self._allocated: set = set()                         # virt ids already alive
        self._counter: int = 0                               # routine/request names
        self._var_counter: int = 0                           # host variable names
        # Scratch virtual-qubit slot used to hold the EPR half during a qsend.
        self._scratch: int = self.num_qubits
        self._phys_qubits: int = self.num_qubits

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def build_program(self) -> QoalaProgramSpec:
        """
        Compile the recorded operations into a :class:`QoalaProgramSpec`.

        Returns:
            The ``.iqoala`` text plus the metadata needed to run it and to
            reconstruct measurement results.
        """
        self._reset_builder()
        # Dispatches to the _translate_* hooks below, which populate the builder.
        self.translate(self.ops)
        self._flush_local()
        text = self._assemble()
        program_input = {f"peer_{p}_id": p for p in sorted(self._peers)}
        return QoalaProgramSpec(
            iqoala_text=text,
            program_input=program_input,
            num_qubits=self._phys_qubits,
            outputs=list(self._outputs),
        )

    # ------------------------------------------------------------------
    # Circuit abstract interface — accumulate, do not execute
    # ------------------------------------------------------------------

    def _translate_operation_container(self, op: OperationContainer):
        for child in op.flatten():
            self.translate(child)

    def _translate_gate(self, op: Gate):
        self._pending.append(op)

    def _translate_controlled_gate(self, op: ControlledGate):
        self._pending.append(op)

    def _translate_measure(self, op: Measure):
        self._pending.append(op)

    def _translate_qsend(self, op: QSend):
        self._flush_local()
        self._emit_qsend(op)

    def _translate_qrecv(self, op: QRecv):
        self._flush_local()
        self._emit_qrecv(op)

    # -- Out-of-scope operations (v1) ----------------------------------

    def _translate_classical_controlled_gate(self, op: ClassicalControlledGate):
        raise NotImplementedError("ClassicalControlledGate is not implemented for the Qoala backend yet.")

    def _translate_reset(self, op: Reset):
        raise NotImplementedError("Reset is not implemented for the Qoala backend yet.")

    def _translate_barrier(self, op: Barrier):
        raise NotImplementedError("Barrier is not implemented for the Qoala backend yet.")

    def _translate_qscatter(self, op: QScatter):
        raise NotImplementedError("QScatter is not implemented for the Qoala backend yet.")

    def _translate_qgather(self, op: QGather):
        raise NotImplementedError("QGather is not implemented for the Qoala backend yet.")

    def _translate_expose(self, op: Expose):
        raise NotImplementedError("Expose is not implemented for the Qoala backend yet.")

    def _translate_unexpose(self, op: Unexpose):
        raise NotImplementedError("Unexpose is not implemented for the Qoala backend yet.")

    # ------------------------------------------------------------------
    # Local gate/measure routine emission
    # ------------------------------------------------------------------

    def _flush_local(self) -> None:
        """Flush any buffered local gates/measurements into a ``QL`` block."""
        if not self._pending:
            return

        name = f"local_{self._counter}"
        self._counter += 1

        lines, ret_vars, ret_clbits, uses, keeps = self._emit_gate_body(self._pending)
        self._subroutines.append(
            self._format_subroutine(name, params=[], ret_vars=ret_vars,
                                    uses=uses, keeps=keeps, body=lines)
        )

        if ret_vars:
            call = f"tuple<{'; '.join(ret_vars)}> = run_subroutine() : {name}"
        else:
            call = f"run_subroutine() : {name}"
        self._host_blocks.append(("QL", [call]))

        for cbit, var in zip(ret_clbits, ret_vars):
            self._outputs.append((cbit, var))

        self._pending = []

    def _emit_gate_body(
        self, ops: List[Operation]
    ) -> Tuple[List[str], List[str], List[int], List[int], List[int]]:
        """
        Turn a run of local operations into NetQASM assembly lines.

        Returns:
            Tuple ``(lines, ret_vars, ret_clbits, uses, keeps)`` where ``lines``
            is the subroutine body, ``ret_vars`` the host return variables for
            each measurement, ``ret_clbits`` their classical-bit indices, and
            ``uses``/``keeps`` the virtual-qubit lifetime metadata.
        """
        virt_order: List[int] = []
        for op in ops:
            for q in self._op_qubits(op):
                if q not in virt_order:
                    virt_order.append(q)

        lines: List[str] = [f"set Q{v} {v}" for v in virt_order]
        for v in virt_order:
            if v not in self._allocated:
                lines.append(f"init Q{v}")
                self._allocated.add(v)

        ret_vars: List[str] = []
        ret_clbits: List[int] = []
        measured: set = set()
        out_idx = 0
        meas_idx = 0

        for op in ops:
            if isinstance(op, Measure):
                lines.append(f"meas Q{op.qubit} M{meas_idx}")
                lines.append(f"store M{meas_idx} @output[{out_idx}]")
                var = f"m_{self._var_counter}"
                self._var_counter += 1
                ret_vars.append(var)
                ret_clbits.append(op.cbit)
                measured.add(op.qubit)
                out_idx += 1
                meas_idx += 1
            elif isinstance(op, ControlledGate):
                lines.extend(self._controlled_gate_lines(op))
            elif isinstance(op, Gate):
                lines.extend(self._gate_lines(op))
            else:  # pragma: no cover - guarded by dispatch
                raise NotImplementedError(f"{type(op).__name__} is not a local operation.")

        uses = list(virt_order)
        keeps = [v for v in virt_order if v not in measured]
        return lines, ret_vars, ret_clbits, uses, keeps

    def _op_qubits(self, op: Operation) -> List[int]:
        """Return the virtual-qubit ids touched by a local operation."""
        if isinstance(op, Measure):
            return [op.qubit]
        return list(op.qubits)

    def _gate_lines(self, op: Gate) -> List[str]:
        """Translate a single-qubit gate into one NetQASM instruction."""
        q = op.qubits[0]
        name = op.name
        direct = {"H": "h", "X": "x", "Y": "y", "Z": "z"}
        if name in direct:
            return [f"{direct[name]} Q{q}"]

        fixed_rot = {  # name -> (axis, n) with d = _ROT_DENOM_EXP (unit pi/16)
            "S": ("z", 8), "SDG": ("z", 24),
            "T": ("z", 4), "TDG": ("z", 28),
        }
        if name in fixed_rot:
            axis, n = fixed_rot[name]
            return [f"rot_{axis} Q{q} {n} {_ROT_DENOM_EXP}"]

        param_rot = {"RX": "x", "RY": "y", "RZ": "z"}
        if name in param_rot:
            n = self._angle_to_n(op.params[0])
            return [f"rot_{param_rot[name]} Q{q} {n} {_ROT_DENOM_EXP}"]

        raise NotImplementedError(f"Gate '{name}' is not implemented for the Qoala backend yet.")

    def _controlled_gate_lines(self, op: ControlledGate) -> List[str]:
        """Translate a two-qubit controlled gate into a NetQASM instruction."""
        if len(op.controls) != 1 or len(op.targets) != 1:
            raise NotImplementedError("Only single-control, single-target gates are supported.")
        control = op.controls[0]
        target_gate = op.targets[0]
        target = target_gate.qubits[0]
        if target_gate.name == "X":
            return [f"cnot Q{control} Q{target}"]
        if target_gate.name == "Z":
            return [f"cphase Q{control} Q{target}"]
        raise NotImplementedError(
            f"Controlled-{target_gate.name} is not implemented for the Qoala backend yet."
        )

    @staticmethod
    def _angle_to_n(theta: float) -> int:
        """Discretize a rotation angle to ``n`` for ``rot_* Q n d`` (d fixed)."""
        unit = math.pi / (2 ** _ROT_DENOM_EXP)
        return int(round(theta / unit)) % (2 ** (_ROT_DENOM_EXP + 1))

    # ------------------------------------------------------------------
    # Communication primitive emission (teleportation)
    # ------------------------------------------------------------------

    def _emit_qsend(self, op: QSend) -> None:
        """Emit the sender-side teleportation blocks for a ``qsend``."""
        dest = op.dest_rank
        self._peers.add(dest)
        self._phys_qubits = max(self._phys_qubits, self.num_qubits + 1)

        for q in op.qubits:
            tag = self._counter
            self._counter += 1
            scratch = self._scratch

            # QC: create an EPR pair with the destination (scratch slot).
            req_name = f"epr_send_{tag}"
            self._requests.append(
                self._format_request(req_name, remote=dest, socket_id=dest,
                                    virt_id=scratch, role="create")
            )
            self._host_blocks.append(("QC", [f"run_request() : {req_name}"]))

            # QL: Bell-state measurement of the user qubit against the EPR half.
            m1 = f"m_{self._var_counter}"; self._var_counter += 1
            m2 = f"m_{self._var_counter}"; self._var_counter += 1
            bsm_name = f"bsm_{tag}"
            body = [
                f"set Q{q} {q}",
                f"set Q{scratch} {scratch}",
                f"cnot Q{q} Q{scratch}",
                f"h Q{q}",
                f"meas Q{q} M0",
                f"meas Q{scratch} M1",
                "store M0 @output[0]",
                "store M1 @output[1]",
            ]
            # Both qubits are consumed by the measurement (nothing kept).
            self._subroutines.append(
                self._format_subroutine(bsm_name, params=[], ret_vars=[m1, m2],
                                        uses=[q, scratch], keeps=[], body=body)
            )
            self._host_blocks.append(("QL", [f"tuple<{m1}; {m2}> = run_subroutine() : {bsm_name}"]))

            # CL: forward the two correction bits to the destination.
            self._host_blocks.append(
                ("CL", [f"send_cmsg(csocket_{dest}, {m1})",
                        f"send_cmsg(csocket_{dest}, {m2})"])
            )
            # The user qubit slot is freed after the BSM.
            self._allocated.discard(q)

    def _emit_qrecv(self, op: QRecv) -> None:
        """Emit the receiver-side teleportation blocks for a ``qrecv``."""
        src = op.src_rank
        self._peers.add(src)

        for q in op.qubits:
            tag = self._counter
            self._counter += 1

            # QC: receive an EPR pair from the source, landing in target slot q.
            req_name = f"epr_recv_{tag}"
            self._requests.append(
                self._format_request(req_name, remote=src, socket_id=src,
                                    virt_id=q, role="receive")
            )
            self._host_blocks.append(("QC", [f"run_request() : {req_name}"]))
            # The EPR-receive request allocates the target qubit.
            self._allocated.add(q)

            # CC: receive the two correction bits (one block each, as in Qoala examples).
            m1 = f"m_{self._var_counter}"; self._var_counter += 1
            m2 = f"m_{self._var_counter}"; self._var_counter += 1
            self._host_blocks.append(("CC", [f"{m1} = recv_cmsg(csocket_{src})"]))
            self._host_blocks.append(("CC", [f"{m2} = recv_cmsg(csocket_{src})"]))

            # QL: apply Pauli corrections (Z if m1, X if m2), keeping the qubit.
            corr_name = f"corr_{tag}"
            body = [
                "set C15 0",
                "set C14 1",
                "load C0 @input[C15]",
                "load C1 @input[C14]",
                f"set Q{q} {q}",
                "set R0 0",
                "beq C0 R0 2",
                f"z Q{q}",
                "beq C1 R0 2",
                f"x Q{q}",
                "set R15 0",
            ]
            self._subroutines.append(
                self._format_subroutine(corr_name, params=[m1, m2], ret_vars=[],
                                        uses=[q], keeps=[q], body=body)
            )
            self._host_blocks.append(
                ("QL", [f"run_subroutine(tuple<{m1}; {m2}>) : {corr_name}"])
            )

    # ------------------------------------------------------------------
    # Text assembly helpers
    # ------------------------------------------------------------------

    def _assemble(self) -> str:
        """Serialize meta, host blocks, subroutines and requests into ``.iqoala``."""
        blocks: List[Tuple[str, List[str]]] = []

        # b0: bind a classical-socket handle for every peer we talk to.
        if self._peers:
            assigns = [f"csocket_{p} = assign_cval() : {p}" for p in sorted(self._peers)]
            blocks.append(("CL", assigns))

        blocks.extend(self._host_blocks)

        # Final block: return every measured value to the batch result.
        if self._outputs:
            blocks.append(("CL", [f"return_result({var})" for _, var in self._outputs]))

        host_code = "\n\n".join(
            self._format_block(f"b{i}", btype, lines)
            for i, (btype, lines) in enumerate(blocks)
        )

        params = [f"peer_{p}_id" for p in sorted(self._peers)]
        meta = self._format_meta(params)

        parts = [meta, host_code]
        if self._subroutines:
            parts.append("\n".join(self._subroutines))
        if self._requests:
            parts.append("\n".join(self._requests))
        return "\n\n".join(parts) + "\n"

    def _format_meta(self, params: List[str]) -> str:
        name = self._comm.get_rank_name(self._comm.rank)
        sockets = ", ".join(
            f"{p} -> {self._comm.get_rank_name(p)}" for p in sorted(self._peers)
        )
        return (
            "META_START\n"
            f"    name: {name}\n"
            f"    parameters: {', '.join(params)}\n"
            f"    csockets: {sockets}\n"
            f"    epr_sockets: {sockets}\n"
            "META_END"
        )

    @staticmethod
    def _format_block(name: str, btype: str, lines: List[str]) -> str:
        body = "\n".join(f"    {line}" for line in lines)
        return f"^{name} {{type = {btype}}}:\n{body}"

    @staticmethod
    def _format_subroutine(
        name: str, params: List[str], ret_vars: List[str],
        uses: List[int], keeps: List[int], body: List[str],
    ) -> str:
        body_text = "\n".join(f"    {line}" for line in body)
        return (
            f"SUBROUTINE {name}\n"
            f"    params: {', '.join(params)}\n"
            f"    returns: {', '.join(ret_vars)}\n"
            f"    uses: {', '.join(str(u) for u in uses)}\n"
            f"    keeps: {', '.join(str(k) for k in keeps)}\n"
            "    request: \n"
            "  NETQASM_START\n"
            f"{body_text}\n"
            "  NETQASM_END"
        )

    @staticmethod
    def _format_request(name: str, remote: int, socket_id: int, virt_id: int, role: str) -> str:
        return (
            f"REQUEST {name}\n"
            "  callback_type: \n"
            "  callback: \n"
            "  return_vars: \n"
            f"  remote_id: {{peer_{remote}_id}}\n"
            f"  epr_socket_id: {socket_id}\n"
            "  num_pairs: 1\n"
            f"  virt_ids: all {virt_id}\n"
            "  timeout: 1000\n"
            "  fidelity: 1.0\n"
            "  typ: create_keep\n"
            f"  role: {role}"
        )
