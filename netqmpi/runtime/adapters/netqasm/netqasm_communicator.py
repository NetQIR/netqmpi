"""
Concrete :class:`~netqmpi.sdk.communicator.base.BaseCommunicator` backed
by the **NetQASM** SDK.

This is the only communicator file that is allowed to import from
``netqasm.*``.  It provides all the low-level resource management
(connections, EPR sockets, classical sockets, GHZ creation) that the
backend-agnostic :class:`QMPICommunicator` facade delegates to.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import netqasm.sdk.toolbox
from netqasm.sdk import EPRSocket, Qubit
from netqasm.sdk.external import NetQASMConnection, Socket

from netqmpi.sdk.communicator.base import BaseCommunicator


class NetQASMCommunicator(BaseCommunicator):
    """
    NetQASM-backed communicator for a single rank.

    Created by the runtime
    (:func:`~netqmpi.runtime.adapters.netqasm.netqasm_executor._make_environment_injector`)
    and injected into :class:`~netqmpi.sdk.communicator.QMPICommunicator`.

    Args:
        rank:       This rank's numeric index.
        size:       Total number of ranks in the world.
        app_config: NetQASM ``AppConfig`` returned by the simulator for this
                    rank.
    """

    def __init__(self, rank: int, size: int, app_config: Any) -> None:
        super().__init__(rank, size)
        self._app_config = app_config

        # -- EPR sockets ---------------------------------------------------
        self._epr_sockets: Dict[str, Dict[str, EPRSocket]] = {}

        for i in range(size):
            self._epr_sockets[self.get_rank_name(i)] = {}

        for i in range(size):
            if i != rank:
                self._epr_sockets[self.get_rank_name(rank)][
                    self.get_rank_name(i)
                ] = EPRSocket(self.get_rank_name(i))

        self._epr_sockets_list: List[EPRSocket] = list(
            self._epr_sockets[self.get_rank_name(rank)].values()
        )

        # -- NetQASM connection ---------------------------------------------
        self._connection = NetQASMConnection(
            app_name=app_config.app_name,
            log_config=app_config.log_config,
            epr_sockets=self._epr_sockets_list,
        )

        # -- Classical sockets (created lazily) -----------------------------
        self._sockets: Dict[str, Dict[str, Socket]] = {}
        for i in range(size):
            self._sockets[self.get_rank_name(i)] = {}

        # -- Expose / GHZ bookkeeping (used by the circuit adapter) ---------
        self.qubits_exposed: List[Any] = []
        self.ghz_qubit: Optional[Any] = None

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> NetQASMCommunicator:
        self._connection.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> Any:
        return self._connection.__exit__(exc_type, exc_val, exc_tb)

    # ------------------------------------------------------------------
    # Connection helpers
    # ------------------------------------------------------------------

    @property
    def connection(self) -> NetQASMConnection:
        return self._connection

    def flush(self) -> None:
        self._connection.flush()

    def create_qubit(self) -> Qubit:
        return Qubit(self._connection)

    # ------------------------------------------------------------------
    # Socket access
    # ------------------------------------------------------------------

    def get_socket(self, my_rank: int, other_rank: int) -> Socket:
        my_sockets = self._sockets[self.get_rank_name(my_rank)]
        other_name = self.get_rank_name(other_rank)

        if other_name not in my_sockets:
            my_sockets[other_name] = Socket(
                self.get_rank_name(my_rank), other_name
            )

        return my_sockets[other_name]

    def get_epr_socket(self, my_rank: int, other_rank: int) -> EPRSocket:
        my_eprs = self._epr_sockets[self.get_rank_name(my_rank)]
        other_name = self.get_rank_name(other_rank)

        if other_name not in my_eprs:
            raise RuntimeError(
                f"EPR socket between rank {my_rank} and rank {other_rank} "
                "does not exist."
            )

        return my_eprs[other_name]

    def get_epr_sockets_list(self) -> List[EPRSocket]:
        return self._epr_sockets_list

    # ------------------------------------------------------------------
    # Collective helpers
    # ------------------------------------------------------------------

    def create_ghz(self) -> Qubit:
        """Create a GHZ state across all ranks and return the local qubit."""
        my_eprs = self._epr_sockets[self.get_rank_name(self._rank)]
        next_epr: Optional[EPRSocket] = None
        prev_epr: Optional[EPRSocket] = None
        next_socket: Optional[Socket] = None
        prev_socket: Optional[Socket] = None

        if self._rank != 0:
            prev_epr = my_eprs[self.get_rank_name(self.get_prev_rank(self._rank))]
            prev_socket = self.get_socket(self._rank, self.get_prev_rank(self._rank))

        if self._rank != self._size - 1:
            next_epr = my_eprs[self.get_rank_name(self.get_next_rank(self._rank))]
            next_socket = self.get_socket(self._rank, self.get_next_rank(self._rank))

        ghz_qubit, _measurement = netqasm.sdk.toolbox.create_ghz(
            down_epr_socket=prev_epr,
            up_epr_socket=next_epr,
            down_socket=prev_socket,
            up_socket=next_socket,
        )

        return ghz_qubit
