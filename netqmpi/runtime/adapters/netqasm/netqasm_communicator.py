"""
Concrete :class:`~netqmpi.sdk.communicator.base.BaseCommunicator` backed
by the NetQASM SDK.

This is the only communicator module allowed to import from
``netqasm.*``. It provides the low-level resource management delegated by
the backend-agnostic :class:`QMPICommunicator` facade, including
connections, EPR sockets, and classical sockets.
"""
from __future__ import annotations

import os
import importlib
from typing import Any, List, Dict, Optional, Set, TYPE_CHECKING

from netqasm.sdk import EPRSocket, Qubit
from netqasm.sdk.external import NetQASMConnection, Socket
from netqasm.runtime.app_config import AppConfig
from netqasm.runtime import env as netqasm_env
from netqasm.runtime.application import (
    Application, ApplicationInstance, Program, network_cfg_from_path,
)
from netqasm.runtime.process_logs import create_app_instr_logs, make_last_log
from netqasm.runtime.settings import Formalism, Simulator, set_simulator
from netqasm.util.yaml import load_yaml

from netqmpi.runtime.adapters.netqasm import _simulation
from netqmpi.sdk.operations import QRecv, QSend

if TYPE_CHECKING:
    from netqmpi.runtime.adapters.netqasm.netqasm_executor import NetQASMRunConfig
    
from netqmpi.sdk import QMPICommunicator

class NetQASMCommunicator(QMPICommunicator):
    """
    NetQASM-backed communicator for a single rank.

    This class provides the backend-specific communication resources used
    by the NetQASM runtime adapter, including the NetQASM connection,
    EPR sockets, and lazily created classical sockets.

    Both kinds of socket are opened only towards the *peers* of the rank —
    the ranks its traced program sends to or receives from. Opening an EPR
    socket makes SquidASM set up a circuit with the remote node under a
    wall-clock timeout, so doing it towards every rank cost O(size^2)
    set-ups per shot where a chain or a star needs O(size).

    Args:
        rank: Numeric index of the current rank.
        size: Total number of ranks in the communicator.
        _config: NetQASM application configuration associated with this rank.
    """
    
    netqasm_circuits = []
    #: Every rank's communicator for the current run, so the state that a
    #: single simulated round leaves behind can be cleared before the next.
    communicators: List["NetQASMCommunicator"] = []

    def __init__(self, rank: int, size: int, config: NetQASMRunConfig) -> None:
        """
        Initialize the NetQASM communicator.

        Args:
            rank: Numeric index of the current rank.
            size: Total number of ranks in the communicator.
            _config: NetQASM application configuration associated with this rank.
        """
        super().__init__(rank, size)

        # -- EPR sockets ---------------------------------------------------
        self._epr_sockets: Dict[str, Dict[str, EPRSocket]] = {}

        for i in range(self.size):
            self._epr_sockets[self.get_rank_name(i)] = {}

        for i in range(self.size):
            if i != self.rank:
                self._epr_sockets[self.get_rank_name(self.rank)][
                    self.get_rank_name(i)
                ] = EPRSocket(self.get_rank_name(i))

        self._epr_sockets_list: List[EPRSocket] = list(
            self._epr_sockets[self.get_rank_name(self.rank)].values()
        )
        
        self._connection = None

        # Classical sockets of the current round, one per peer. See get_socket.
        self._sockets: Dict[int, Socket] = {}
        # Ranks the traced program talks to; filled in by __exit__.
        self._peers: Optional[Set[int]] = None

        self._config = config
        NetQASMCommunicator.communicators.append(self)

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------
    
    @property
    def epr_sockets(self) -> Dict[str, Dict[str, EPRSocket]]:
        """
        Return the EPR sockets indexed by rank name.

        Returns:
            A nested mapping of EPR sockets.
        """
        return self._epr_sockets
    
    def __enter__(self) -> NetQASMCommunicator:
        """
        Enter the NetQASM connection context.

        Returns:
            The current communicator instance.
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> Any:
        """
        Exit the NetQASM connection context.

        Args:
            exc_type: Exception type, if one was raised.
            exc_val: Exception instance, if one was raised.
            exc_tb: Traceback, if one was raised.

        Returns:
            The result of the underlying connection ``__exit__`` method.
        """
        # A rank that failed while tracing has nothing to simulate, and the
        # run cannot go on without it: forget the run and let the exception
        # propagate.
        if exc_type is not None:
            NetQASMCommunicator.reset_run()
            return None

        # SquidASM looks up program_inputs[party] for every program it runs
        # and then writes the AppConfig back into that mapping, so each party
        # needs an entry, and its own mutable one. An empty mapping made
        # run_app raise KeyError before any program started.
        argv = load_yaml(self._config.argv) if self._config.argv else {}
        argv_per_rank: dict = {
            self.get_rank_name(r): dict(argv.get(self.get_rank_name(r), {}))
            for r in range(self.size)
        }
        
        self._peers = set()
        for circuit in self.circuits:
            self._peers |= circuit.peers()

        for circuit in self.circuits:
            # Translate before the simulator is anywhere near: the emitted
            # operations are closures that only touch qubits when they run,
            # so nothing here needs a connection. Doing it inside the
            # simulation instead meant an unsupported gate raised in a
            # program thread, and the run hung waiting for a thread that had
            # already died rather than reporting the gate.
            circuit.translate(circuit.ops)

            # ``circuit`` is bound as a default argument: without it every
            # program built by this loop would close over the last circuit
            # of the rank rather than its own.
            def entry(app_config=None, circuit=circuit):
                # One call per shot. The qubits of the previous shot were
                # consumed with it, so the slots start empty and are filled
                # again on first use; keeping them would hand this shot
                # qubits belonging to a connection that no longer exists.
                circuit.reset_round()
                self.close_sockets()

                self._connection = NetQASMConnection(
                    app_name=app_config.app_name,
                    log_config=app_config.log_config, # TODO: Change none
                    epr_sockets=self._epr_sockets_list,
                )
                self._connection.__enter__()

                # One bit string per shot, unmeasured bits reading 0, most
                # significant first — the shape every other backend returns.
                # Counting each measurement on its own instead made a rank
                # with two classical bits report twice as many single-bit
                # outcomes as it ran shots, which no other backend does and
                # nothing downstream could compare against.
                shot = [0] * circuit.num_clbits
                try:
                    for op in circuit.translated_ops:
                        outcome = op()
                        if outcome is not None:
                            cbit, result = outcome
                            self.flush()
                            shot[cbit] = int(result)
                finally:
                    # The round's sockets must be gone before SquidASM starts
                    # the next round; see get_socket.
                    self.close_sockets()

                key = "".join(str(bit) for bit in reversed(shot))
                self.results[key] = self.results.get(key, 0) + 1

                self._connection.__exit__(exc_type, exc_val, exc_tb)

            NetQASMCommunicator.netqasm_circuits.append(
                Program(party=f"rank_{self.rank}", entry=entry, args=[], results=[])
            )

        if len(NetQASMCommunicator.netqasm_circuits) == self.size:
            try:
                self._simulate(argv_per_rank)
            finally:
                # Class-level state belongs to one run. Leaving it behind
                # meant a second run in the same process assembled an
                # application out of both runs' programs, so the ranks never
                # matched their own count and the simulation was never
                # reached.
                NetQASMCommunicator.reset_run()

        return None

    def _simulate(self, argv_per_rank: dict) -> None:
        """
        Run every rank's program through SquidASM, once per shot.

        Called by the last rank to leave its block, once every rank's
        program is registered.

        Args:
            argv_per_rank: Program inputs, one mutable mapping per party.

        Raises:
            ValueError: If the network cannot hold the program (see
                :func:`~netqmpi.runtime.adapters.netqasm._simulation.check_capacity`)
                or a configured network file does not exist.
            Exception: Whatever a rank's program raised during the
                simulation. The SquidASM backend is stopped first, so the
                process can still exit and run again.
        """
        programs = NetQASMCommunicator.netqasm_circuits
        roles = netqasm_env.load_roles_config(self._config.roles)
        if roles is None:
            roles = {prog.party: prog.party for prog in programs}

        app_instance = ApplicationInstance(
            app=Application(programs=programs, metadata=None),
            program_inputs=argv_per_rank,
            network=None,
            party_alloc=roles,
            logging_cfg=None,
        )

        simulator = os.environ.get("NETQASM_SIMULATOR", Simulator.NETSQUID.value)
        set_simulator(simulator)

        simulate_application = importlib.import_module("netqasm.sdk.external").simulate_application

        formalism = getattr(self._config, "formalism", Formalism.KET)
        log_cfg = _simulation.log_config(self._config)

        if self._config.network_config is not None:
            network_config = network_cfg_from_path(".", self._config.network_config)
            if network_config is None:
                raise ValueError(
                    f"NetQASM network configuration "
                    f"{self._config.network_config!r} does not exist.")
        else:
            # SquidASM's default network, but sized and tuned by the config.
            network_config = _simulation.build_network(
                sorted(set(roles.values())), self._config)

        _simulation.check_capacity(network_config, roles,
                                   NetQASMCommunicator.communicators)

        # ``num_rounds`` is how SquidASM repeats an experiment, and it
        # was pinned at 1: every run returned a single sample whatever
        # --shots asked for, so a 50/50 outcome came back as a certainty.
        # Asking for real repeats needs the classical sockets to live one
        # round only (see get_socket) and each shot to start from empty
        # qubit slots. Results accumulate across the rounds.
        for comm in NetQASMCommunicator.communicators:
            comm._reset_sockets()

        with _simulation.squidasm_patches(self._config.epr_setup_timeout,
                                          self._config.poll_interval):
            try:
                simulate_application(
                    app_instance=app_instance,
                    num_rounds=max(1, self._config.shots),
                    network_cfg=network_config,
                    formalism=formalism,
                    post_function=self._config.post_function,
                    log_cfg=log_cfg,
                    use_app_config=True,
                    enable_logging=self._config.enable_logging,
                    hardware=self._config.hardware,
                )
            except BaseException:
                _simulation.stop_backend()
                raise

        if self._config.enable_logging and log_cfg is not None:
            create_app_instr_logs(log_cfg.log_subroutines_dir)
            make_last_log(log_cfg.log_subroutines_dir)

    @classmethod
    def reset_run(cls) -> None:
        """Forget the programs and communicators of a finished run."""
        cls.netqasm_circuits = []
        cls.communicators = []


    def _reset_sockets(self) -> None:
        """
        Give this rank the EPR sockets the coming run will be connected with.

        Called once per run, before the rounds begin; the classical sockets
        are built on demand by :meth:`get_socket`. Only the ranks the traced
        program talks to get one, or every other rank if the program has not
        been traced.
        """
        peers = self._peers
        if peers is None:
            peers = set(range(self.size)) - {self.rank}

        self._epr_sockets = {self.get_rank_name(i): {} for i in range(self.size)}
        for other in sorted(peers):
            if other != self.rank:
                self._epr_sockets[self.get_rank_name(self.rank)][
                    self.get_rank_name(other)
                ] = EPRSocket(self.get_rank_name(other))
        self._epr_sockets_list = list(
            self._epr_sockets[self.get_rank_name(self.rank)].values())

        for circuit in self.circuits:
            circuit.reset_round()

    def get_socket(self, my_rank: int, other_rank: int) -> Socket:
        """
        Return the classical socket between two ranks for the current round.

        One socket per peer and per round: the first transfer with a peer
        in a round opens it, every later one in the same round reuses it,
        and :meth:`close_sockets` drops it when the round ends.

        *Not one per transfer.* Every socket between the same two ranks has
        the same key in NetQASM's socket hub, and closing one removes the
        marker its peer's socket left there. With a fresh socket per
        transfer, a sender that ran ahead — its second transfer's socket
        opened and closed while the receiver was still in the first one —
        had that marker erased when the receiver closed its first socket,
        and the receiver's second socket then waited for a peer that was
        gone. Two transfers in a row in the same direction hung that way.

        *Not one per run either.* SquidASM resets the hub between rounds, so
        a socket kept across rounds is no longer connected ("Socket is not
        connected so cannot send"), and one released late would, when
        collected, erase the hub entries of the next round's socket.

        Args:
            my_rank: Rank requesting the socket.
            other_rank: Rank at the other endpoint of the socket.

        Returns:
            A classical socket connecting the two ranks.
        """
        socket = self._sockets.get(other_rank)
        if socket is None:
            socket = Socket(self.get_rank_name(my_rank),
                            self.get_rank_name(other_rank))
            self._sockets[other_rank] = socket
        return socket

    def close_sockets(self) -> None:
        """
        Drop the classical sockets of the current round.

        A NetQASM thread socket leaves the hub when it is collected, and
        this holds the only lasting reference to it, so dropping it closes
        it.
        """
        self._sockets = {}

    def get_epr_socket(self, my_rank: int, other_rank: int) -> EPRSocket:
        """
        Return the EPR socket between two ranks.

        Args:
            my_rank: Rank requesting the socket.
            other_rank: Rank at the other endpoint of the socket.

        Returns:
            The EPR socket connecting the two ranks.

        Raises:
            RuntimeError: If the requested EPR socket does not exist.
        """
        my_eprs = self.epr_sockets[self.get_rank_name(my_rank)]
        other_name = self.get_rank_name(other_rank)

        if other_name not in my_eprs:
            raise RuntimeError(
                f"EPR socket between rank {my_rank} and rank {other_rank} "
                "does not exist."
            )

        return my_eprs[other_name]

    def flush(self) -> None:
        """
        Flush the underlying NetQASM connection.
        """
        self._connection.flush()
        
    def create_qubit(self):
        """
        Create a new qubit on the underlying NetQASM connection.

        Returns:
            A newly allocated NetQASM qubit.
        """
        return Qubit(self._connection)

    # ------------------------------------------------------------------
    # Connection helpers
    # ------------------------------------------------------------------

    @property
    def connection(self) -> NetQASMConnection:
        """
        Return the underlying NetQASM connection.

        Returns:
            The active NetQASM connection.
        """
        return self._connection
    
    