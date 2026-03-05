"""
Concrete :class:`~netqmpi.sdk.communicator.base.BaseCommunicator` backed
by the **NetQASM** SDK.

This is the only communicator file that is allowed to import from
``netqasm.*``.  It provides all the low-level resource management
(connections, EPR sockets, classical sockets, GHZ creation) that the
backend-agnostic :class:`QMPICommunicator` facade delegates to.
"""
from __future__ import annotations

from typing import Any, List

from netqmpi.sdk.communicator.base import BaseCommunicator


class CunqaCommunicator(BaseCommunicator):
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

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> CunqaCommunicator:
        pass

    def __exit__(self, exc_type, exc_val, exc_tb) -> Any:
        pass

    # ------------------------------------------------------------------
    # Connection helpers
    # ------------------------------------------------------------------

    @property
    def connection(self):
        pass

    def flush(self) -> None:
        pass

    def create_qubit(self):
        pass

    # ------------------------------------------------------------------
    # Socket access
    # ------------------------------------------------------------------

    def get_socket(self, my_rank: int, other_rank: int):
        pass

    def get_epr_socket(self, my_rank: int, other_rank: int):
        pass

    def get_epr_sockets_list(self):
        pass

    # ------------------------------------------------------------------
    # Collective helpers
    # ------------------------------------------------------------------

    def create_ghz(self):
        pass
