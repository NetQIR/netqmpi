"""
A NetQMPI plugin that adds readout error to the NetQASM/SquidASM backend.

SquidASM measures without error: a qubit in ``|1⟩`` always reads 1. Real
detectors misread, and usually asymmetrically. This plugin flips the outcome
of every measurement with a probability that depends on the outcome:

    netqasm:
      plugin: readout_error:readout_error
      plugin_config:
        prob_error_0: 0.02    # a 0 reads as 1
        prob_error_1: 0.10    # a 1 reads as 0

The flip happens after the qubit has been measured, so the post-measurement
state is the true one and only the reported bit is wrong, which is what a
readout error is. It applies to every measurement the simulated node makes,
including the Bell measurement that ``qsend`` performs behind the scenes, so
it also degrades teleportation.

Extension point: the *executor*, the SquidASM class that turns each NetQASM
instruction into an operation on the simulated device. SquidASM asks
``SubroutineHandler._get_executor_class`` which one to use; the plugin
answers with a subclass of SquidASM's own executor for the duration of the
run. Nodes with NV hardware keep their executor unchanged.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Dict, Iterator, Mapping

from netqasm.lang.instr.flavour import VanillaFlavour
from netsquid.util.simtools import get_random_state
from squidasm.nqasm.executor.vanilla import VanillaNetSquidExecutor
from squidasm.nqasm.qnodeos import SubroutineHandler

#: The keys this plugin understands in ``plugin_config``.
KEYS = ("prob_error_0", "prob_error_1")


def readout_error(config: Mapping[str, Any]):
    """
    The plugin entry point, named in the config file.

    NetQMPI calls it once per run, with ``plugin_config``, just before the
    simulation, and enters the context manager it returns for the length of
    that simulation.

    Args:
        config: ``plugin_config``: ``prob_error_0`` and ``prob_error_1``,
            each a probability in [0, 1] that defaults to 0.

    Returns:
        A context manager under which SquidASM measures with readout error.

    Raises:
        ValueError: On an unknown key or a value that is not a probability.
            Validating here, before anything is patched, reports a typo in
            the config file instead of silently ignoring it.
    """
    probs = _probabilities(config)
    return _readout_error(probs["prob_error_0"], probs["prob_error_1"])


def _probabilities(config: Mapping[str, Any]) -> Dict[str, float]:
    """Check ``plugin_config`` and return both probabilities."""
    unknown = set(config) - set(KEYS)
    if unknown:
        raise ValueError(f"readout_error: unknown keys {sorted(unknown)}; "
                         f"expected {list(KEYS)}.")
    probs = {}
    for key in KEYS:
        value = float(config.get(key, 0.0))
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"readout_error: {key} must be in [0, 1], got {value}.")
        probs[key] = value
    return probs


# Optional members of the plugin contract (netqmpi.plugins.Plugin), which a
# plain function can carry as attributes. NetQMPI calls ``configure`` when the
# config is built, so a typo in plugin_config is reported before any rank is
# traced instead of after; and ``backends`` refuses the plugin on any backend
# but the one whose internals it replaces.
readout_error.configure = lambda plugin_config, run_config: _probabilities(plugin_config)
readout_error.backends = ("netqasm",)


@contextmanager
def _readout_error(prob_error_0: float, prob_error_1: float) -> Iterator[None]:
    """
    Make SquidASM's vanilla executor misread measurements while active.

    Args:
        prob_error_0: Probability that an outcome 0 is reported as 1.
        prob_error_1: Probability that an outcome 1 is reported as 0.
    """

    class ReadoutErrorExecutor(VanillaNetSquidExecutor):
        """SquidASM's executor for generic hardware, with readout error."""

        def _meas_physical_qubit(self, position):
            outcome = yield from super()._meas_physical_qubit(position)
            flip = prob_error_0 if outcome == 0 else prob_error_1
            # NetSquid's random state, so that ns.set_random_state(seed)
            # makes a run with readout error reproducible too.
            if flip and get_random_state().random_sample() < flip:
                return 1 - outcome
            return outcome

    # The function stored in the class, not the bound classmethod, so that
    # it can be put back exactly as it was.
    original = SubroutineHandler.__dict__["_get_executor_class"]

    def executor_class(cls, flavour=None):
        if flavour is None or isinstance(flavour, VanillaFlavour):
            return ReadoutErrorExecutor
        return original.__func__(cls, flavour)

    SubroutineHandler._get_executor_class = classmethod(executor_class)
    try:
        yield
    finally:
        SubroutineHandler._get_executor_class = original
