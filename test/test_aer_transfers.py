"""
Regression tests for how the Aer adapter emits cross-rank calls.

Aer runs every rank inside one ``QuantumCircuit``, so the order in which
instructions are appended *is* the order in which they execute. The adapter
used to translate one rank fully before starting the next, which only
happens to be right when a program's dependencies follow rank order — and
when it was wrong it said nothing, returning a plausible-looking histogram
for a circuit that had been reordered. These tests pin the two properties
that were missing.

Every probe here is an *echo*: it does something and then undoes it, so the
correct answer is all-zeros and any reordering shows up as a wrong result
rather than as a subtly different distribution.
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

pytest.importorskip("qiskit", reason="the Aer backend needs Qiskit")
pytest.importorskip("qiskit_aer", reason="the Aer backend needs qiskit-aer")

from netqmpi.runtime.adapters.aer import (  # noqa: E402
    AerExecutorAdapter, AerSimulatorConfig,
)
from netqmpi.sdk.environment import Environment  # noqa: E402

SHOTS = 256


def run_app(source: str, ranks: int, tmp_path: Path):
    """
    Run a NetQMPI program on the Aer backend and return each rank's results.

    Args:
        source: Body of the app module, which must define ``main(env)``.
        ranks: Number of ranks to run.
        tmp_path: Directory to write the app to.

    Returns:
        The results of rank 0, which under Aer is the joint histogram over
        every rank's classical bits.
    """
    app = tmp_path / "app.py"
    app.write_text(textwrap.dedent(source))

    captured = []
    original = Environment.__init__

    def capture(self, comm, executor):
        original(self, comm, executor)
        captured.append(self)

    Environment.__init__ = capture
    try:
        config = AerSimulatorConfig()
        config.shots = SHOTS
        executor = AerExecutorAdapter(ranks, config)
        executor.run(executor.build_apps(str(app), ranks))
    finally:
        Environment.__init__ = original

    captured.sort(key=lambda env: env.comm.rank)
    return captured[0].comm.results


def all_zero_share(results) -> float:
    """
    Return the share of shots that came back all zeros.

    Args:
        results: A histogram keyed by bit string.

    Returns:
        The probability of the all-zeros outcome.
    """
    total = sum(results.values())
    hits = sum(count for key, count in results.items()
               if set(key.replace(" ", "")) == {"0"})
    return hits / total if total else 0.0


#: A star: rank 0's control visits every other rank and comes back between
#: hops, then the whole tour is repeated to undo it. Rank order cannot
#: linearise this, so translating rank by rank silently broke it.
STAR_ECHO = """
    from netqmpi.sdk.environment import Environment

    def tour(comm, circuit, rank, size):
        if rank == 0:
            circuit.swap(0, 1)
        for host in range(1, size):
            if rank == 0:
                comm.qsend(circuit, [1], host)
                comm.qrecv(circuit, [1], host)
            elif rank == host:
                comm.qrecv(circuit, [1], 0)
                circuit.cx(1, 0)
                comm.qsend(circuit, [1], 0)
        if rank == 0:
            circuit.swap(0, 1)

    def main(env: Environment = None):
        comm, rank, size = env.comm, env.comm.rank, env.comm.size
        with comm:
            circuit = env.create_circuit(num_qubits=2, num_clbits=1)
            if rank == 0:
                circuit.h(0)
            tour(comm, circuit, rank, size)
            tour(comm, circuit, rank, size)
            if rank == 0:
                circuit.h(0)
            circuit.measure(0, 0)
"""

#: Sender and receiver name *different* local indices for the same qubit.
#: The transfer has to land where the receiver asked, not where the sender
#: happened to keep it.
ASYMMETRIC_INDICES = """
    from netqmpi.sdk.environment import Environment

    def main(env: Environment = None):
        comm, rank = env.comm, env.comm.rank
        with comm:
            circuit = env.create_circuit(num_qubits=3, num_clbits=3)
            if rank == 0:
                circuit.x(0)                       # payload |1> on qubit 0
                comm.qsend(circuit, [0], 1)
            else:
                comm.qrecv(circuit, [2], 0)        # lands on qubit 2, not 0
                circuit.x(2)                       # undo it: back to |0>
            for i in range(3):
                circuit.measure(i, i)
"""

#: An exposed control driving a gate on another rank. Rank 0 lends its
#: qubit, rank 1 uses it as the control of a CNOT, and the window closes.
#: Rank 1 must end up holding whatever rank 0 was.
REMOTE_CONTROL = """
    from netqmpi.sdk.environment import Environment

    def main(env: Environment = None):
        comm, rank = env.comm, env.comm.rank
        with comm:
            circuit = env.create_circuit(num_qubits=1, num_clbits=1)
            if rank == 0 and {payload}:
                circuit.x(0)
            handle = comm.expose(circuit, 0, [1], root=0)
            if rank == 1:
                circuit.cx(handle, 0)
            comm.unexpose(circuit, [1], root=0)
            circuit.measure(0, 0)
"""

#: One control lent to every other rank at once, the MPI_Bcast shape.
FAN_OUT = """
    from netqmpi.sdk.environment import Environment

    def main(env: Environment = None):
        comm, rank, size = env.comm, env.comm.rank, env.comm.size
        receivers = list(range(1, size))
        with comm:
            circuit = env.create_circuit(num_qubits=1, num_clbits=1)
            if rank == 0:
                circuit.x(0)
            handle = comm.expose(circuit, 0, receivers, root=0)
            if rank in receivers:
                circuit.cx(handle, 0)
            comm.unexpose(circuit, receivers, root=0)
            circuit.measure(0, 0)
"""

#: The root lends a superposition and must get it back untouched: the
#: receiver applies its CNOT twice, which is the identity, so rank 0's |+>
#: has to survive the window and map back to |0> under a final H.
CONTROL_RETURNED = """
    from netqmpi.sdk.environment import Environment

    def main(env: Environment = None):
        comm, rank = env.comm, env.comm.rank
        with comm:
            circuit = env.create_circuit(num_qubits=1, num_clbits=1)
            if rank == 0:
                circuit.h(0)
            handle = comm.expose(circuit, 0, [1], root=0)
            if rank == 1:
                circuit.cx(handle, 0)
                circuit.cx(handle, 0)
            comm.unexpose(circuit, [1], root=0)
            if rank == 0:
                circuit.h(0)
            circuit.measure(0, 0)
"""

#: A qsend whose qrecv was never traced. Nothing can pair it.
DANGLING_SEND = """
    from netqmpi.sdk.environment import Environment

    def main(env: Environment = None):
        comm, rank = env.comm, env.comm.rank
        with comm:
            circuit = env.create_circuit(num_qubits=1, num_clbits=1)
            if rank == 0:
                circuit.h(0)
                comm.qsend(circuit, [0], 1)
            circuit.measure(0, 0)
"""


@pytest.mark.parametrize("ranks", [2, 3, 4])
def test_star_topology_survives_translation(ranks, tmp_path):
    """A control that revisits rank 0 between hops must still echo exactly."""
    results = run_app(STAR_ECHO, ranks, tmp_path)
    assert all_zero_share(results) == 1.0


def test_transfer_lands_on_the_receivers_index(tmp_path):
    """The receiver chooses where an incoming qubit lands, not the sender."""
    results = run_app(ASYMMETRIC_INDICES, 2, tmp_path)
    # Rank 1 cancels the payload it received on qubit 2; if the transfer had
    # landed anywhere else, that X would hit |0> and leave a 1 behind.
    assert all_zero_share(results) == 1.0


def test_unmatched_transfer_is_reported(tmp_path):
    """A qsend with no qrecv is a deadlock, and must be named as one."""
    with pytest.raises(RuntimeError, match="never match"):
        run_app(DANGLING_SEND, 2, tmp_path)


@pytest.mark.parametrize("payload, expected", [(0, 0.0), (1, 1.0)])
def test_exposed_control_drives_a_remote_gate(payload, expected, tmp_path):
    """A control lent by expose must actually control the receiver's gate."""
    results = run_app(REMOTE_CONTROL.format(payload=payload), 2, tmp_path)
    total = sum(results.values())
    ones = sum(count for key, count in results.items()
               if key.replace(" ", "")[0] == "1")     # rank 1's bit
    assert ones / total == expected


@pytest.mark.parametrize("ranks", [2, 3, 4])
def test_expose_fans_out_to_every_receiver(ranks, tmp_path):
    """One window serves the whole group, not just its first member."""
    results = run_app(FAN_OUT, ranks, tmp_path)
    observed = {key.replace(" ", ""): value for key, value in results.items()}
    assert observed == {"1" * ranks: SHOTS}, results


def test_exposed_control_comes_back_untouched(tmp_path):
    """The root lends a superposition and gets it back unchanged."""
    results = run_app(CONTROL_RETURNED, 2, tmp_path)
    total = sum(results.values())
    # Rank 0's bit is the rightmost; it must read 0 on every shot.
    zeros = sum(count for key, count in results.items()
                if key.replace(" ", "")[-1] == "0")
    assert zeros == total, results


def test_lone_transfer_translation_is_refused():
    """Translating a transfer outside the joint pass must not be guessed at."""
    from netqmpi.runtime.adapters.aer.aer_circuit import AerCircuitAdapter
    from netqmpi.sdk.operations import QSend

    refuse = AerCircuitAdapter.__dict__["_translate_qsend"]
    with pytest.raises(RuntimeError, match="translate_group"):
        refuse(None, QSend([0], 1, tag="t"))


def test_lone_expose_translation_is_refused():
    """A telegate window cannot be opened from one rank's stream either."""
    from netqmpi.runtime.adapters.aer.aer_circuit import AerCircuitAdapter
    from netqmpi.sdk.operations import Expose

    refuse = AerCircuitAdapter.__dict__["_translate_expose"]
    window = Expose(rank=0, root=0, ranks=[0, 1], tag="t",
                    comm_slot=0, clbits=[0], data_qubit=0)
    with pytest.raises(RuntimeError, match="translate_group"):
        refuse(None, window)
