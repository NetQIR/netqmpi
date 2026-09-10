"""
Phase-resolved instrumentation of a NetQMPI run.

The point of the benchmark is to say *where* the wall-clock and the memory
of a distributed quantum program actually go, and in particular how much of
it NetQMPI's own abstraction costs. This module splits one run into five
phases and measures each of them:

===============  =============================================================
``import``       Importing the backend adapter and the simulator it pulls in.
``setup``        Building the executor and ``build_apps``: resource discovery,
                 vQPU allocation (CUNQA's ``qraise``), topology construction.
``trace``        Running the user's ``main()`` so the SDK records operations
                 into the ``OperationContainer``. **NetQMPI SDK cost.**
``translate``    ``Circuit.translate``: turning those operations into native
                 backend instructions. **NetQMPI adapter cost.**
``backend``      The backend actually executing: submit, simulate, collect.
===============  =============================================================

``trace`` and ``translate`` together are the price of the abstraction;
``setup`` and ``backend`` are the price of the platform underneath it.

How the split is obtained
-------------------------
``translate`` is measured by wrapping :meth:`netqmpi.sdk.circuit.Circuit.translate`,
which every adapter funnels through, counting only the outermost call so
that the recursion into an ``OperationContainer`` is not charged twice.
``backend`` is measured by wrapping one call per backend (see
:data:`BACKEND_PROBES`).

``trace`` is measured *directly* rather than by subtraction, which matters
more than it sounds. Everything a backend does — translating, simulating,
and synchronising the ranks — happens inside the communicator's ``__exit__``,
so timing each rank's ``main()`` and removing the time it spent in that
``__exit__`` leaves exactly the recording of operations. Taking ``trace`` as
"whatever is left of ``executor.run()``" instead would fold in the barrier
waits of the thread-per-rank backends, and those waits are concurrency
latency, not a cost of the abstraction: on Aer they swamp the real trace by
an order of magnitude and vary at random from run to run.

What is left inside ``__exit__`` after removing translation and execution is
reported separately as ``sync``. For a backend that runs its ranks
concurrently the per-rank figures are summed, so they can add up to more
than the wall-clock; ``total`` is always the real elapsed time.

Memory
------
Two figures, because neither alone is honest:

* ``py_peak`` — :mod:`tracemalloc` peak, Python allocations only. This is
  what actually captures NetQMPI's own footprint: the ``Operation`` objects
  the trace builds. It does **not** see a C++ state vector.
* ``rss_peak`` — :func:`resource.getrusage` peak RSS of the whole process,
  which does include the simulator's own memory.

``tracemalloc`` perturbs timing badly, so a measured run never does both:
:func:`run_profiled` takes a ``memory`` flag and the caller does separate
timing and memory passes.

Only the standard library is used: the four backend environments do not
share a third-party dependency (the ``qoala`` env has no ``psutil``, the
CUNQA image no ``qiskit``), and the profiler has to import in all of them.
"""
from __future__ import annotations

import gc
import importlib
import resource
import time
import tracemalloc
from typing import Any, Callable, Dict, List, Optional, Tuple

#: Backend execution calls to wrap, per backend name. Each entry is
#: ``(module, class_or_None, attribute)``. The module is the one the call is
#: *looked up in at call time*, which is not always where it is defined:
#: the CUNQA communicator does ``from cunqa.qpu import run``, so the name to
#: patch lives in the communicator's namespace, not in ``cunqa.qpu``.
BACKEND_PROBES: Dict[str, List[Tuple[str, Optional[str], str]]] = {
    "aer": [
        ("netqmpi.runtime.adapters.aer.aer_executor",
         "AerExecutorAdapter", "_run_simulation"),
    ],
    "cunqa": [
        ("netqmpi.runtime.adapters.cunqa.cunqa_communicator", None, "run"),
        ("netqmpi.runtime.adapters.cunqa.cunqa_communicator", None, "gather"),
    ],
    "qoala": [
        ("netqmpi.runtime.adapters.qoala.qoala_executor",
         "QoalaExecutorAdapter", "run_simulation"),
    ],
    "netqasm": [
        ("netqasm.sdk.external", None, "simulate_application"),
    ],
}

#: Backends whose translation is interleaved with execution rather than
#: happening in a separate pass, so ``translate`` cannot be isolated from
#: ``backend``. The NetQASM adapter returns callables from ``translate``
#: and runs them inside the simulation, so its ``translate`` figure covers
#: only the emission of those callables and its ``trace`` absorbs the rest.
INTERLEAVED_BACKENDS = frozenset({"netqasm"})


class Profiler:
    """
    Wall-clock and memory instrumentation for a single NetQMPI run.

    Install the hooks with :meth:`install`, drive the run through
    :meth:`phase` context managers, and read the totals off :attr:`times`
    and :attr:`counts` afterwards.

    Args:
        backend: Backend name, used to select the execution probe.
    """

    def __init__(self, backend: str) -> None:
        self.backend = backend
        self.times: Dict[str, float] = {
            "import": 0.0, "setup": 0.0, "trace": 0.0, "translate": 0.0,
            "backend": 0.0, "sync": 0.0, "total": 0.0,
        }
        # Aggregates the phase split is derived from: time inside each
        # rank's main(), and time inside the communicator __exit__ it calls.
        self._app_seconds = 0.0
        self._exit_seconds = 0.0
        self.counts: Dict[str, int] = {"translate_calls": 0}
        self._depth = 0
        self._restore: List[Tuple[Any, str, Any]] = []

    # ------------------------------------------------------------------
    # Hook installation
    # ------------------------------------------------------------------

    def install(self) -> None:
        """
        Wrap :meth:`Circuit.translate` and the backend's execution call.

        A backend whose probe cannot be resolved (the module is not
        importable in this environment, or the adapter has been renamed)
        is left unprobed: ``backend`` stays at zero and its time is
        absorbed into ``trace``, which :meth:`report` flags.
        """
        from netqmpi.sdk.circuit import Circuit

        original = Circuit.translate

        def timed_translate(circuit_self, op):
            # Only the outermost call is charged: adapters recurse into
            # OperationContainer through this same method.
            if self._depth:
                self.counts["translate_calls"] += 1
                return original(circuit_self, op)
            self._depth = 1
            self.counts["translate_calls"] += 1
            start = time.perf_counter()
            try:
                return original(circuit_self, op)
            finally:
                self.times["translate"] += time.perf_counter() - start
                self._depth = 0

        self._patch(Circuit, "translate", timed_translate)

        for module_name, class_name, attr in BACKEND_PROBES.get(self.backend, []):
            try:
                module = importlib.import_module(module_name)
            except ImportError:
                continue
            target = getattr(module, class_name) if class_name else module
            if not hasattr(target, attr):
                continue
            self._patch(target, attr, self._timed(getattr(target, attr), "backend"))

    def hook_communicator(self, comm_class: type) -> None:
        """
        Time the communicator's ``__exit__``, where the backend does its work.

        Args:
            comm_class: Concrete communicator class used by this run.
        """
        if "__exit__" not in vars(comm_class):
            return
        original = comm_class.__exit__

        def timed_exit(comm_self, *args):
            start = time.perf_counter()
            try:
                return original(comm_self, *args)
            finally:
                self._exit_seconds += time.perf_counter() - start

        self._patch(comm_class, "__exit__", timed_exit)

    def wrap_app(self, app: Callable) -> Callable:
        """
        Return a rank's entry point wrapped so its wall-clock is recorded.

        Args:
            app: Zero-argument callable running one rank's ``main()``.

        Returns:
            The wrapped callable.
        """
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                return app(*args, **kwargs)
            finally:
                self._app_seconds += time.perf_counter() - start
        return wrapper

    def _timed(self, func: Callable, bucket: str) -> Callable:
        """
        Return *func* wrapped so its wall-clock lands in ``times[bucket]``.

        Args:
            func: Callable to wrap.
            bucket: Key of :attr:`times` to accumulate into.

        Returns:
            The wrapped callable.
        """
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                self.times[bucket] += time.perf_counter() - start
        return wrapper

    def _patch(self, target: Any, attr: str, replacement: Any) -> None:
        """
        Replace ``target.attr`` and remember how to put it back.

        Args:
            target: Module or class holding the attribute.
            attr: Attribute name.
            replacement: Value to install.
        """
        self._restore.append((target, attr, getattr(target, attr)))
        setattr(target, attr, replacement)

    def uninstall(self) -> None:
        """Undo every patch, most recent first."""
        for target, attr, original in reversed(self._restore):
            setattr(target, attr, original)
        self._restore.clear()

    # ------------------------------------------------------------------
    # Measurement
    # ------------------------------------------------------------------

    def finish_run(self, run_seconds: float) -> None:
        """
        Attribute the measured aggregates to their phases.

        ``trace`` is the time the ranks spent in ``main()`` outside the
        communicator's ``__exit__``; ``sync`` is what remains inside that
        ``__exit__`` once translation and backend execution are removed —
        barrier waits and results plumbing. If the communicator could not be
        hooked, the split falls back to subtraction and ``sync`` stays zero.

        Args:
            run_seconds: Wall-clock of the whole ``executor.run()`` call.
        """
        if self._app_seconds > 0:
            self.times["trace"] = max(0.0, self._app_seconds - self._exit_seconds)
            self.times["sync"] = max(
                0.0,
                self._exit_seconds - self.times["translate"] - self.times["backend"])
        else:
            self.times["trace"] = max(
                0.0,
                run_seconds - self.times["translate"] - self.times["backend"])

        self.times["total"] = (
            self.times["import"] + self.times["setup"] + run_seconds)

    @property
    def netqmpi_seconds(self) -> float:
        """Time spent inside NetQMPI itself: tracing plus translating."""
        return self.times["trace"] + self.times["translate"]

    @property
    def overhead_fraction(self) -> float:
        """NetQMPI's share of the total wall-clock, in ``[0, 1]``."""
        total = self.times["total"]
        return self.netqmpi_seconds / total if total > 0 else float("nan")


def peak_rss_bytes() -> int:
    """
    Return the peak resident set size of this process, in bytes.

    ``ru_maxrss`` is reported in kilobytes on Linux and is monotonic over
    the life of the process, which is why the benchmark runs one
    configuration per process.

    Returns:
        Peak RSS in bytes.
    """
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


class MemoryPass:
    """
    Context manager taking a :mod:`tracemalloc` peak around a run.

    Args:
        enabled: When ``False`` the manager does nothing and
            :attr:`peak_bytes` stays ``None``, which is what a timing pass
            wants — ``tracemalloc`` roughly doubles the cost of the trace.
    """

    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled
        self.peak_bytes: Optional[int] = None

    def __enter__(self) -> "MemoryPass":
        if self.enabled:
            gc.collect()
            tracemalloc.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self.enabled:
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            self.peak_bytes = peak
        return None
