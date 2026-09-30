"""
Which NetQMPI this is: the release, and the git tag and commit behind it.

A release number alone does not say which code ran: every commit after a tag
reports the tag's release until the next bump. Benchmark results and bug
reports need the commit, so it is recorded here.

Where the answer comes from, in order:

1. **The build.** ``setup.py`` writes ``netqmpi/_build_info.py`` into every
   build and every source distribution, freezing the tag and commit of the
   checkout it was built from. An installed wheel reports that, whatever
   repository — if any — it later runs next to.
2. **The checkout.** An editable install (``pip install -e .``) or a plain
   checkout on ``PYTHONPATH`` has no build step, so git is asked directly.
   That is also the right answer there: the code that runs is whatever is
   checked out *now*, not what it was when it was installed.
3. **Neither.** The fields are ``None`` and :func:`describe` says so.

Usage::

    >>> import netqmpi
    >>> netqmpi.__version__
    '0.3.1'
    >>> print(netqmpi.version.describe())
    NetQMPI 0.3.1 (tag v0.3.1 + 1 commit, commit 662436b0544560689612d7546eab9089a423f783)

or, from the shell, ``netqmpi --version``.
"""
from __future__ import annotations

import os
import re
import subprocess
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Dict, Optional

#: Name of the module the build writes, next to this one.
BUILD_INFO_MODULE = "_build_info"

#: ``git describe --long`` output: ``<tag>-<distance>-g<hash>[-dirty]``.
_DESCRIBE = re.compile(r"^(?P<tag>.+)-(?P<distance>\d+)-g(?P<short>[0-9a-f]+)(?P<dirty>-dirty)?$")


@dataclass(frozen=True)
class BuildInfo:
    """
    Where this copy of NetQMPI comes from.

    Attributes:
        version: The release, as declared in ``setup.py``.
        tag: The most recent git tag reachable from the commit, if any.
        commits_since_tag: How many commits the code is past ``tag``.
        commit: Full hash of the commit the code was built from.
        dirty: Whether the working tree had uncommitted changes to tracked
            files. The commit alone does not then identify the code.
        source: ``"build"`` if frozen when the package was built, ``"git"``
            if read from the checkout at import, ``"unknown"`` otherwise.
    """

    version: str
    tag: Optional[str] = None
    commits_since_tag: Optional[int] = None
    commit: Optional[str] = None
    dirty: bool = False
    source: str = "unknown"

    def as_dict(self) -> Dict[str, object]:
        """Return the fields as a plain dict, e.g. to store with results."""
        return asdict(self)


def parse_describe(output: str) -> Dict[str, object]:
    """
    Split the output of ``git describe --tags --long --dirty --always``.

    Args:
        output: What git printed. Without any tag, ``--always`` makes it a
            bare abbreviated hash.

    Returns:
        ``tag``, ``commits_since_tag`` and ``dirty``; the first two are
        ``None`` when the repository has no tag.
    """
    output = output.strip()
    match = _DESCRIBE.match(output)
    if match is None:
        return {"tag": None, "commits_since_tag": None,
                "dirty": output.endswith("-dirty")}
    return {"tag": match.group("tag"),
            "commits_since_tag": int(match.group("distance")),
            "dirty": match.group("dirty") is not None}


def _git(directory: str, *args: str) -> Optional[str]:
    """Run git in ``directory``; ``None`` if git or the repository is missing."""
    try:
        done = subprocess.run(["git", "-C", directory, *args],
                              capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout.strip() if done.returncode == 0 else None


def read_git(directory: str) -> Optional[Dict[str, object]]:
    """
    Read tag, commit and dirtiness from the NetQMPI checkout at ``directory``.

    Only a repository whose top level holds NetQMPI's own ``setup.py`` and
    ``netqmpi`` package counts: a copy installed somewhere that happens to be
    inside another project's repository must not report *that* project's
    commit.

    Args:
        directory: Root of the checkout, or any directory inside it.

    Returns:
        ``tag``, ``commits_since_tag``, ``commit`` and ``dirty``, or ``None``
        if ``directory`` is not in a NetQMPI checkout or git is unavailable.
    """
    top = _git(directory, "rev-parse", "--show-toplevel")
    if top is None or not (
            os.path.isfile(os.path.join(top, "setup.py"))
            and os.path.isdir(os.path.join(top, "netqmpi"))):
        return None

    commit = _git(top, "rev-parse", "HEAD")
    if commit is None:            # a repository with no commit yet
        return None
    info = parse_describe(
        _git(top, "describe", "--tags", "--long", "--dirty", "--always") or "")
    info["commit"] = commit
    info["version"] = setup_py_version(top) or "unknown"
    return info


def setup_py_version(directory: str) -> Optional[str]:
    """
    Read the release declared in the ``setup.py`` of a checkout.

    Args:
        directory: Root of the checkout.

    Returns:
        The declared version, or ``None`` if there is none to read.
    """
    try:
        with open(os.path.join(directory, "setup.py"), encoding="utf-8") as handle:
            match = re.search(r"version\s*=\s*['\"]([^'\"]+)['\"]", handle.read())
    except OSError:
        return None
    return match.group(1) if match else None


def _installed_version() -> str:
    """The release recorded in the installed distribution's metadata."""
    try:
        from importlib.metadata import PackageNotFoundError, version
    except ImportError:            # Python < 3.8
        return "unknown"
    try:
        return version("netqmpi")
    except PackageNotFoundError:
        return "unknown"


@lru_cache(maxsize=1)
def get_build_info() -> BuildInfo:
    """
    Return where this copy of NetQMPI comes from.

    Computed once per process; see the module docstring for the order in
    which the sources are tried.

    Returns:
        The :class:`BuildInfo`.
    """
    try:
        from netqmpi import _build_info  # type: ignore[attr-defined]
    except ImportError:
        _build_info = None

    if _build_info is not None:
        frozen = dict(_build_info.BUILD_INFO)
        return BuildInfo(source="build", **frozen)

    # In a checkout the installed metadata can be stale — an editable install
    # keeps the version it had when it was installed — so the checkout's own
    # setup.py is what counts there.
    checkout = read_git(os.path.dirname(os.path.abspath(__file__)))
    if checkout is not None:
        return BuildInfo(source="git", **checkout)
    return BuildInfo(version=_installed_version())


def describe(info: Optional[BuildInfo] = None) -> str:
    """
    Return a one-line, human-readable description of this NetQMPI.

    Args:
        info: What to describe; this copy of NetQMPI by default.

    Returns:
        For instance ``NetQMPI 0.3.1 (tag v0.3.1 + 1 commit, commit 662436b...)``.
    """
    info = info or get_build_info()
    if info.commit is None:
        return f"NetQMPI {info.version} (no git information)"

    parts = []
    if info.tag is not None:
        distance = info.commits_since_tag or 0
        if distance:
            plural = "commit" if distance == 1 else "commits"
            parts.append(f"tag {info.tag} + {distance} {plural}")
        else:
            parts.append(f"tag {info.tag}")
    else:
        parts.append("no tag")
    parts.append(f"commit {info.commit}")
    if info.dirty:
        parts.append("uncommitted changes")
    return f"NetQMPI {info.version} ({', '.join(parts)})"


def render_build_info(info: Dict[str, object]) -> str:
    """
    Return the source of the ``_build_info`` module a build writes.

    Args:
        info: ``version``, ``tag``, ``commits_since_tag``, ``commit``, ``dirty``.

    Returns:
        Python source defining ``BUILD_INFO``.
    """
    fields = ("version", "tag", "commits_since_tag", "commit", "dirty")
    body = "".join(f"    {name!r}: {info.get(name)!r},\n" for name in fields)
    return ('"""Written by setup.py when NetQMPI was built. Do not edit."""\n'
            f"BUILD_INFO = {{\n{body}}}\n")


def __getattr__(name: str):
    # ``__version__`` is resolved on first use rather than at import, so that
    # importing NetQMPI never waits on git.
    if name == "__version__":
        return get_build_info().version
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
