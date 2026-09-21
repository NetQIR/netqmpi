"""
Which NetQASM is installed, and what that changes.

NetQASM 1.x and 2.x are not installable side by side, and 2.x needs Python
3.9 or newer, so a machine runs one or the other and a NetQMPI run has to
be pointed at the environment that has the one it wants. That is what the
``--netqasm`` and ``--netqasm1.0`` flags select between.

What they do *not* select between is two adapters. The API this backend
uses — ``Qubit``, ``EPRSocket``, ``NetQASMConnection``, ``Socket``,
``Application``/``ApplicationInstance``/``Program``, and SquidASM's
``simulate_application`` — is the same in both releases, and SquidASM
itself is a pure-Python wheel with no upper bound on the NetQASM version it
accepts. One adapter serves both; this module names the handful of places
where the newer release offers something better, and checks that the
environment is the one the run asked for instead of failing later with an
error that says nothing about versions.
"""
from __future__ import annotations

import netqasm

#: Version of the installed NetQASM, as a ``(major, minor)`` pair. Anything
#: after the minor is ignored: nothing here depends on a patch release.
def installed_version() -> tuple:
    """
    Return the installed NetQASM version as ``(major, minor)``.

    Returns:
        The two leading version components, as integers. Components that do
        not parse as integers (a development suffix, say) count as zero.
    """
    parts = str(netqasm.__version__).split(".")[:2]
    numbers = []
    for part in parts:
        digits = "".join(c for c in part if c.isdigit())
        numbers.append(int(digits) if digits else 0)
    while len(numbers) < 2:
        numbers.append(0)
    return tuple(numbers)


#: Major version of the installed NetQASM.
INSTALLED_MAJOR = installed_version()[0]

#: Highest NetQASM that SquidASM accepts, as of squidasm 0.13.6, whose
#: metadata reads ``netqasm<=2.0.0,>=1.0.0``. NetQASM ships 2.1, 2.2 and 2.3
#: as well, but no SquidASM release supports them — 2.3 is the one qoala-sim
#: builds on, which is a different simulator entirely.
SQUIDASM_MAX_NETQASM = (2, 0)

#: Where the compatibility actually ends.
#:
#: As far as this backend is concerned the 2.x SDK is a drop-in for 1.x:
#: ``Qubit``, ``EPRSocket``, ``NetQASMConnection``, ``Socket`` and the
#: application classes are the same calls. What is *not* portable is
#: anything the newer instruction set adds on top.
#:
#: ``Qubit.swap`` is the case in point. It does not exist in 2.0, the
#: newest release SquidASM accepts; it exists in 2.3, and calling it there
#: against SquidASM 0.13.4 builds a swap instruction the executor never
#: handles, so the simulation *hangs* rather than failing. The adapter
#: therefore keeps assembling a swap from three CNOTs on every release.
#:
#: The rule this encodes: use 2.x through the API 1.x also had, and treat
#: anything newer as unavailable until a SquidASM that understands it
#: exists.
USE_ONLY_SHARED_INSTRUCTIONS = True

#: Environment each major version is expected to live in, named in the error
#: raised when the wrong one is active. These are the conda environments the
#: project's documentation sets up.
_SUGGESTED_ENV = {1: "squidasm", 2: "netqasm2"}


def require_major(expected: int) -> None:
    """
    Check that the installed NetQASM is the one the run asked for.

    Args:
        expected: Major version the selected backend flag stands for.

    Raises:
        RuntimeError: If a different major version is installed. Failing
            here, before any circuit is built, is the difference between a
            message naming the version and one about a missing attribute
            several layers into SquidASM.
    """
    if INSTALLED_MAJOR == expected:
        return

    flag = "--netqasm" if expected >= 2 else "--netqasm1.0"
    other = "--netqasm1.0" if expected >= 2 else "--netqasm"
    env = _SUGGESTED_ENV.get(expected, "one with that version")

    raise RuntimeError(
        f"{flag} asks for NetQASM {expected}.x but {netqasm.__version__} is "
        f"installed. The two releases cannot share an environment, and 2.x "
        f"needs Python 3.9 or newer, so run this from the '{env}' "
        f"environment — or use {other} to run against what is installed here."
    )
