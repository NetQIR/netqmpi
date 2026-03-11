"""
Concrete :class:`~netqmpi.sdk.communicator.base.BaseCommunicator` backed
by the **NetQASM** SDK.

This is the only communicator file that is allowed to import from
``netqasm.*``.  It provides all the low-level resource management
(connections, EPR sockets, classical sockets, GHZ creation) that the
backend-agnostic :class:`QMPICommunicator` facade delegates to.
"""
from __future__ import annotations

from netqmpi.sdk import QMPICommunicator

class CunqaCommunicator(QMPICommunicator):
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

    def __init__(self, rank: int, size: int) -> None:
        super().__init__(rank, size)

    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        return None