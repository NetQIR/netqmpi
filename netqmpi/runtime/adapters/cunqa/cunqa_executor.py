"""
Executor adapter for the CUNQA backend.

This module provides an implementation of the ``Executor`` interface for
running applications with the CUNQA backend.

A NetQMPI run needs one vQPU per rank. There are two ways to get them, and
the configuration picks between them:

- *Attach* (the default): the vQPUs are already up, raised by the user
  before the ``netqmpi`` command with ``qraise``, and the run takes the
  ones it needs from a single family. The allocation outlives the run, so
  several programs can be launched against the same vQPUs without paying
  for a SLURM job each time, and a family larger than the run is fine —
  the vQPUs no rank is using are kept busy with a trivial circuit, since
  the family's executor runs a round only once every one of its vQPUs has
  submitted something.
- *Raise* (``qraise: true`` in the config): the adapter raises the vQPUs
  itself, runs, and drops them again. The vQPU definition to raise them
  with is the ``backend`` setting, so the qubit budget of the run is under
  the user's control.
"""
import os, sys
sys.path.append(os.getenv("HOME"))

from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from cunqa.qpu import QPU, qraise, get_QPUs, qdrop

from netqmpi.runtime.executor import Executor
from netqmpi.sdk.environment import Environment
from netqmpi.runtime.adapters.cunqa.cunqa_circuit import CunqaCircuitAdapter
from netqmpi.runtime.adapters.cunqa.cunqa_communicator import (
    CunqaCommunicator,
    CunqaSession,
)
from netqmpi.runtime.run_config import RunConfig
from netqmpi.helpers import load_main  


@dataclass
class CunqaRunConfig(RunConfig):
    """
    Extension of :class:`~netqmpi.runtime.run_config.RunConfig` with
    CUNQA-specific execution parameters.

    Attributes:
        qraise: Whether to raise the vQPUs for this run and drop them
            afterwards. When ``False`` (the default), the run attaches to
            vQPUs that are already up.
        backend: Path to the vQPU definition file the vQPUs are raised
            with, which is what fixes their qubit budget. Only meaningful
            when ``qraise`` is enabled; ``None`` leaves the choice to
            CUNQA's own default.
        simulator: Simulator backing each vQPU. Raise-mode only.
        time: Wall-clock reservation for the SLURM job, as ``D-HH:MM:SS``
            or ``HH:MM:SS``. Raise-mode only.
        family: Family the vQPUs belong to. Selects which of the running
            vQPUs to attach to, or names the family to raise.
        co_located: Whether the vQPUs are reachable from other nodes
            (CUNQA's *co-located* mode) rather than only from the node
            they run on (*hpc* mode).
    """

    qraise: bool = False
    backend: Optional[str] = None
    simulator: str = "Munich"
    time: str = "00:10:00"
    family: Optional[str] = None
    co_located: bool = True


#: Settings that only have an effect when the adapter raises the vQPUs.
_RAISE_ONLY_FIELDS = ("backend", "simulator", "time")


class CunqaExecutorAdapter(Executor):
    """
    Executor adapter for the CUNQA backend.

    This adapter enables execution through CUNQA while conforming to the
    common interface defined by :class:`Executor`.
    """
    
    def __init__(self, size: int, config: CunqaRunConfig = None):
        """
        Initialize the CUNQA executor adapter.

        Args:
            size: Number of available CUNQA nodes.
            config: Backend-specific configuration parameters.
        """
        super().__init__(size, config or CunqaRunConfig())
        # Family raised by this run, and therefore ours to drop. Stays None
        # when attaching to vQPUs somebody else raised.
        self._family: Optional[str] = None
    
    def create_circuit(
        self, 
        num_qubits: int, 
        num_clbits: int, 
        comm: CunqaCommunicator
    ) -> CunqaCircuitAdapter:
        """
        Create a CUNQA circuit adapter.

        Args:
            num_qubits: Number of qubits in the circuit.
            num_clbits: Number of classical bits in the circuit.
            comm: Communicator associated with the circuit.

        Returns:
            A ``CunqaCircuitAdapter`` instance.
        """
        return CunqaCircuitAdapter(num_qubits, num_clbits, comm)

    # ------------------------------------------------------------------
    # vQPUs
    # ------------------------------------------------------------------

    def _check_config(self) -> None:
        """
        Reject configurations whose settings could not take effect.

        Raises:
            ValueError: If raise-only settings are given while attaching to
                already-raised vQPUs, or if the vQPU definition file does
                not exist.
        """
        config = self._config
        if not config.qraise:
            defaults = CunqaRunConfig()
            ignored = [name for name in _RAISE_ONLY_FIELDS
                       if getattr(config, name) != getattr(defaults, name)]
            if ignored:
                raise ValueError(
                    f"{', '.join(ignored)} only appl{'ies' if len(ignored) == 1 else 'y'} "
                    f"when NetQMPI raises the vQPUs itself. Either add 'qraise: true' "
                    f"to the cunqa block of the config file, or drop "
                    f"{'that setting' if len(ignored) == 1 else 'those settings'} and "
                    f"raise the vQPUs yourself with qraise before running.")
            return

        if config.backend is not None and not os.path.isfile(config.backend):
            raise ValueError(
                f"The vQPU definition file given as 'backend' does not exist: "
                f"{config.backend}.")

    def _attach_qpus(self, size: int) -> List[QPU]:
        """
        Return the vQPUs of one already-raised family.

        The whole family is returned, not just ``size`` of them: CUNQA runs
        one executor per family and it waits for a circuit from each of the
        family's vQPUs before running a round, so the ones no rank uses
        cannot simply be left out — the caller keeps them busy instead.

        Args:
            size: Number of vQPUs the run needs, one per rank. Only used to
                report a family that is too small.

        Returns:
            Every vQPU of the chosen family, or an empty list when none are
            running at all.

        Raises:
            RuntimeError: If vQPUs of several families are up and none was
                named, since a run cannot spread across families.
        """
        config = self._config
        qpus = get_QPUs(co_located=config.co_located, family=config.family) or []
        if not qpus:
            return []

        families: Dict[Any, List[QPU]] = {}
        for qpu in qpus:
            families.setdefault(qpu.family, []).append(qpu)

        if len(families) > 1:
            listing = ", ".join(
                f"'{name}' ({len(members)} vQPU{'s' if len(members) != 1 else ''})"
                for name, members in sorted(families.items(), key=lambda kv: str(kv[0])))
            raise RuntimeError(
                f"Found vQPUs of more than one family running: {listing}. A run "
                f"cannot spread across families, because each of them is executed "
                f"on its own, so name the one to use with 'family: <name>' in the "
                f"cunqa block of the config file.")

        return next(iter(families.values()))

    def _raise_qpus(self, size: int) -> List[QPU]:
        """
        Raise one vQPU per rank and return them.

        The family is recorded so that :meth:`run` drops exactly what this
        run raised, and nothing that was already up.

        Args:
            size: Number of vQPUs to raise, one per rank.

        Returns:
            The vQPUs raised for this run.
        """
        config = self._config
        self._family = qraise(
            size,
            config.time,
            simulator=config.simulator,
            co_located=config.co_located,
            quantum_comm=True,
            backend=config.backend,
            family=config.family,
        )
        return get_QPUs(co_located=config.co_located, family=self._family) or []

    def _get_qpus(self, size: int) -> List[QPU]:
        """
        Obtain one vQPU per rank, raising them first if so configured.

        Args:
            size: Number of vQPUs the run needs, one per rank.

        Returns:
            At least ``size`` vQPUs, all of the same family, in the order
            CUNQA reports them. There may be more: a family bigger than the
            run is allowed, and the caller keeps the spare ones busy.

        Raises:
            RuntimeError: If there are not enough vQPUs for the run.
        """
        self._check_config()

        qpus = self._raise_qpus(size) if self._config.qraise else self._attach_qpus(size)

        if len(qpus) < size:
            family = (f" of family '{self._config.family}'"
                      if self._config.family else "")
            if self._config.qraise:
                raise RuntimeError(
                    f"Asked CUNQA for {size} vQPUs but only {len(qpus)} came "
                    f"up{family}. The SLURM job may still be starting or may have "
                    f"failed; check it with squeue.")
            raise RuntimeError(
                f"This run needs {size} vQPUs but found {len(qpus)}{family} "
                f"already raised. Raise them before running, for instance with "
                f"'qraise -n {size} -t {self._config.time} --quantum_comm "
                f"--co-located', or add 'qraise: true' to the cunqa block of the "
                f"config file to have NetQMPI raise and drop them for you.")

        return qpus

    # ------------------------------------------------------------------
    # Executor interface
    # ------------------------------------------------------------------

    def build_apps(self, file: str, size: int) -> Any:
        """
        Build one application wrapper per rank from the provided file.

        Args:
            file: Path to the file containing the main entry point.
            size: Number of ranks to instantiate.

        Returns:
            A collection of wrapped application callables, one per rank.
        """
        main_func = load_main(file)

        try:
            qpus = self._get_qpus(size)
            # The family can be larger than the run; the vQPUs beyond the
            # ranks take no part in the program but are still submitted to.
            rank_qpus, idle_qpus = qpus[:size], qpus[size:]

            # One session shared by every rank: the ranks trace one after the
            # other, but their circuits are translated and submitted together.
            session = CunqaSession(size, rank_qpus, self._config,
                                   idle_qpus=idle_qpus)

            apps = []
            for rank, qpu in enumerate(rank_qpus):
                comm = CunqaCommunicator(rank, size, qpu, self._config, session=session)
                env = Environment(comm, self)
                wrapped_main = lambda env=env: main_func(env=env)
                apps.append(wrapped_main)
        except Exception:
            # Never leave a family of our own behind on a failed setup.
            self._drop_raised_qpus()
            raise

        return apps
        
    def run(self, apps: Any) -> None:
        """
        Execute the provided applications on the CUNQA backend.

        The applications are invoked one after the other to build their
        circuits; the last rank to leave its ``with comm:`` block triggers
        the joint translation and the actual run.

        Args:
            apps: Applications to execute.

        Raises:
            Exception: Propagates any exception raised while the ranks
                build their circuits or while the circuits are executed.
        """
        try:
            for app in apps:
                app()
        finally:
            # Only vQPUs this run raised are dropped: ones the user raised
            # beforehand stay up for their next run.
            self._drop_raised_qpus()

    def _drop_raised_qpus(self) -> None:
        """Drop the vQPUs this run raised, if any."""
        if self._family is not None:
            qdrop(self._family)
            self._family = None
