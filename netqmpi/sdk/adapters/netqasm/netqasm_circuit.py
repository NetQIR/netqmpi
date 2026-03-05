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
3. Keeps ``translate()`` as a no-op — execution has already happened.
4. In ``build()`` flushes the connection and returns the qubit array
   together with the classical results.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, List, Optional

from netqasm.sdk.qubit import Qubit

from netqmpi.sdk.core.circuit import Circuit
from netqmpi.sdk.core.operations.operation import Operation

if TYPE_CHECKING:
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
