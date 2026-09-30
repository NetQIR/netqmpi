"""
How the NetQASM adapter drives a SquidASM simulation, and where it has to
work around SquidASM to do it.

Everything here runs once per NetQMPI run, from the last rank's
``__exit__``, around the call to ``simulate_application``:

* :func:`build_network` describes the simulated network. SquidASM's default
  gives every node five perfect qubits and perfect links, and offers no way
  to change either short of writing a network YAML; this builds the same
  network with the size and noise taken from
  :class:`~netqmpi.runtime.adapters.netqasm.NetQASMRunConfig`.
* :func:`check_capacity` refuses a run whose ranks cannot fit in their
  nodes, before SquidASM gets the chance to fail on it half-way through.
* :func:`log_config` keeps concurrent runs from colliding over SquidASM's
  log directory.
* :func:`squidasm_patches` and :func:`stop_backend` are the workarounds: a
  wall-clock timeout that SquidASM hard-codes and a busy wait, and a backend
  thread that SquidASM leaves running when a program fails.

SquidASM is imported lazily and only by the workarounds, so the rest of this
module works against NetQASM alone (the ``debug`` simulator included).
"""
from __future__ import annotations

import copy
import functools
import os
import tempfile
from contextlib import contextmanager
from datetime import datetime
from typing import TYPE_CHECKING, Dict, Iterable, Iterator, List, Optional

from netqasm.runtime.interface.config import (
    Link, NetworkConfig, Node, NoiseType, QuantumHardware, Qubit,
)
from netqasm.sdk.config import LogConfig

if TYPE_CHECKING:
    from netqmpi.runtime.adapters.netqasm.netqasm_communicator import (
        NetQASMCommunicator,
    )
    from netqmpi.runtime.adapters.netqasm.netqasm_executor import NetQASMRunConfig

#: The ``hardware`` values SquidASM's own default network understands.
_HARDWARE = {"generic": QuantumHardware.Generic, "nv": QuantumHardware.NV}


def build_network(node_names: Iterable[str],
                  config: "NetQASMRunConfig") -> NetworkConfig:
    """
    Describe a fully connected network sized and tuned by the run config.

    With the config's defaults this is exactly SquidASM's default network —
    five qubits per node, no memory decoherence (``t1 = t2 = 0``), perfect
    gates and noiseless links — which is to say the simulation is **ideal**
    unless the config asks otherwise.

    Args:
        node_names: The nodes the ranks are placed on.
        config: Run configuration providing ``num_qubits``, ``t1``, ``t2``,
            ``gate_fidelity``, ``link_fidelity``, ``link_noise`` and
            ``hardware``.

    Returns:
        The network, ready for ``simulate_application(network_cfg=...)``.

    Raises:
        ValueError: If ``hardware`` or ``link_noise`` is not a value
            SquidASM knows.
    """
    try:
        hardware = _HARDWARE[config.hardware]
    except KeyError:
        raise ValueError(
            f"Unsupported NetQASM hardware {config.hardware!r}; expected one "
            f"of {sorted(_HARDWARE)}.") from None

    if config.link_fidelity >= 1.0:
        noise = NoiseType.NoNoise
    else:
        try:
            noise = NoiseType(config.link_noise)
        except ValueError:
            raise ValueError(
                f"Unknown NetQASM link noise {config.link_noise!r}; expected "
                f"one of {[n.value for n in NoiseType]}.") from None

    names = list(node_names)
    nodes = [
        Node(name=name, hardware=hardware,
             qubits=[Qubit(id=i, t1=config.t1, t2=config.t2)
                     for i in range(config.num_qubits)],
             gate_fidelity=config.gate_fidelity)
        for name in names
    ]
    # One link per ordered pair, as SquidASM's default network has.
    links = [
        Link(name=f"link_{a}_{b}", node_name1=a, node_name2=b,
             noise_type=noise, fidelity=config.link_fidelity)
        for a in names for b in names if a != b
    ]
    return NetworkConfig(nodes=nodes, links=links)


def check_capacity(network: NetworkConfig, roles: Dict[str, str],
                   communicators: List["NetQASMCommunicator"]) -> None:
    """
    Refuse a run whose ranks need more qubits than their nodes have.

    A rank needs the most qubits it ever holds at once: its live data
    qubits plus, during a transfer, the EPR half. SquidASM finds out only
    when the allocation fails inside the simulation, far from the program
    that caused it, and a run can hang rather than report it.

    Args:
        network: The network the run will be simulated on.
        roles: Party (``rank_<i>``) to node name.
        communicators: Every rank's communicator, with its circuits traced.

    Raises:
        ValueError: Naming every rank that does not fit, and how to fix it.
    """
    capacity = {node.name: len(node.qubits) for node in network.nodes}
    short = []
    for comm in communicators:
        need = max((circuit.peak_qubits() for circuit in comm.circuits),
                   default=0)
        node = roles.get(comm.get_rank_name(comm.rank),
                         comm.get_rank_name(comm.rank))
        have = capacity.get(node)
        if have is None:
            raise ValueError(
                f"rank {comm.rank} is placed on node {node!r}, which the "
                f"network configuration does not define.")
        if need > have:
            short.append(f"  rank {comm.rank} needs {need} qubits at once, "
                         f"node {node!r} has {have}")
    if short:
        raise ValueError(
            "The simulated network is too small for this program:\n"
            + "\n".join(short)
            + "\nA rank holds its live data qubits plus one EPR half while "
            "it transfers. Raise 'num_qubits' in the netqasm block of the "
            "config file (NetQASMRunConfig.num_qubits; SquidASM's default is "
            "5), or the qubits of those nodes in the network YAML.")


def log_config(config: "NetQASMRunConfig") -> Optional[LogConfig]:
    """
    Return the log configuration to hand SquidASM, if logging is on.

    SquidASM writes each run to ``<log_dir>/<YYYYmmdd-HHMMSS>`` and creates
    that directory with a check-then-``mkdir``, so two runs sharing a
    working directory that start within the same second collide with
    ``FileExistsError``. Unless the config names a ``log_dir`` of its own,
    every run gets a fresh one under ``./log`` instead.

    Args:
        config: Run configuration providing ``enable_logging`` and
            ``log_cfg``.

    Returns:
        ``None`` when logging is off, otherwise a log configuration whose
        ``log_dir`` no other run shares.
    """
    if not config.enable_logging:
        return config.log_cfg

    log_cfg = copy.copy(config.log_cfg) if config.log_cfg is not None else LogConfig()
    if log_cfg.log_dir is None:
        base = os.path.abspath("log")
        os.makedirs(base, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S-")
        log_cfg.log_dir = tempfile.mkdtemp(prefix=stamp, dir=base)
    return log_cfg


@contextmanager
def squidasm_patches(epr_setup_timeout: float,
                     poll_interval: float) -> Iterator[None]:
    """
    Work around two SquidASM behaviours for the duration of one simulation.

    Neither can be configured through SquidASM's API, so both are patched,
    and restored on the way out:

    * **EPR-socket set-up timeout.** ``NetworkStack.setup_epr_socket`` waits
      for the remote node to install its rules for at most ``timeout``
      seconds — 5 by default, measured in *wall-clock* time, not simulated
      time. With many ranks in threads, or on a loaded machine, a remote
      node does not get there in time and the run dies with ``TimeoutError:
      Remote node did not initialize the correct rules`` (1 in 16 runs at 10
      ranks, 3 in 7 at 12, every run from 16 in the benchmark). The default
      is raised to ``epr_setup_timeout``.
    * **Busy wait on the programs.** ``SquidAsmRuntimeManager.run_app``
      waits for the program threads with ``as_completed(..., sleep_time=0)``,
      a loop that never sleeps and so competes for the GIL with the very
      threads it waits for. It is given a ``poll_interval`` sleep.

    Outside SquidASM (NetQASM's ``debug`` simulator, say) this does nothing.

    Args:
        epr_setup_timeout: Wall-clock seconds a node waits for its peer to
            set up an EPR socket. Never lowers the timeout a caller passes.
        poll_interval: Seconds the main thread sleeps between checks on the
            program threads.
    """
    try:
        from squidasm.nqasm.netstack import NetworkStack
        from squidasm.run.multithread import runtime_mgr
    except ImportError:
        yield
        return

    original_setup = NetworkStack.setup_epr_socket
    original_as_completed = runtime_mgr.as_completed

    @functools.wraps(original_setup)
    def setup_epr_socket(self, epr_socket_id, remote_node_id,
                         remote_epr_socket_id, timeout=epr_setup_timeout):
        return original_setup(self, epr_socket_id, remote_node_id,
                              remote_epr_socket_id,
                              timeout=max(timeout, epr_setup_timeout))

    NetworkStack.setup_epr_socket = setup_epr_socket
    runtime_mgr.as_completed = functools.partial(original_as_completed,
                                                 sleep_time=poll_interval)
    try:
        yield
    finally:
        NetworkStack.setup_epr_socket = original_setup
        runtime_mgr.as_completed = original_as_completed


def stop_backend() -> None:
    """
    Stop the SquidASM backend a failed simulation left running.

    When a program thread raises, ``simulate_application`` re-raises in the
    caller but never reaches its ``stop_backend()``: the NetSquid thread,
    which is not a daemon, keeps running, so the process cannot exit, and
    the next simulation in the same process fails with "Already a backend
    running". Program threads blocked inside NetQASM on the dead run cannot
    be interrupted; they are daemons and do not keep the process alive.
    """
    try:
        from squidasm.sim.glob import get_running_backend
    except ImportError:
        return
    backend = get_running_backend(block=False)
    if backend is not None:
        backend.stop_backend()
