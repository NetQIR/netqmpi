"""
Tests for the version identifier: release, git tag and commit.

NetQMPI reports which code is running — the release declared in
``setup.py``, the latest git tag and the commit — frozen into the package
when it is built, or read from git when it runs from a checkout.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import types
from pathlib import Path

import pytest

import netqmpi
from netqmpi import version
from netqmpi.version import BuildInfo, describe, parse_describe

REPO_ROOT = Path(__file__).resolve().parents[1]

needs_git = pytest.mark.skipif(
    shutil.which("git") is None
    or subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
                      capture_output=True).returncode != 0,
    reason="needs git and a git checkout of NetQMPI")


@pytest.fixture(autouse=True)
def fresh_cache():
    """Every test computes the build info anew."""
    version.get_build_info.cache_clear()
    yield
    version.get_build_info.cache_clear()


def git(*args: str, cwd: Path = REPO_ROOT) -> str:
    return subprocess.run(["git", "-C", str(cwd), *args], capture_output=True,
                          text=True, check=True).stdout.strip()


@pytest.mark.parametrize("output, expected", [
    ("v0.3.1-0-g662436b", {"tag": "v0.3.1", "commits_since_tag": 0, "dirty": False}),
    ("v0.3.1-12-g662436b-dirty", {"tag": "v0.3.1", "commits_since_tag": 12, "dirty": True}),
    # A tag that itself contains dashes.
    ("release-2-rc-3-gabcdef0", {"tag": "release-2-rc", "commits_since_tag": 3, "dirty": False}),
    # No tag at all: --always falls back to the bare hash.
    ("662436b", {"tag": None, "commits_since_tag": None, "dirty": False}),
    ("662436b-dirty", {"tag": None, "commits_since_tag": None, "dirty": True}),
])
def test_git_describe_is_parsed(output, expected):
    assert parse_describe(output) == expected


@needs_git
def test_a_checkout_reports_its_own_tag_and_commit():
    info = version.get_build_info()
    assert info.source == "git"
    assert info.commit == git("rev-parse", "HEAD")
    assert info.tag == git("describe", "--tags", "--abbrev=0")
    assert info.version == version.setup_py_version(str(REPO_ROOT))
    assert netqmpi.__version__ == info.version


@needs_git
def test_another_repository_is_not_mistaken_for_netqmpi(tmp_path):
    """A copy living inside some other project's repository reports nothing."""
    git("init", "-q", cwd=tmp_path)
    git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q",
        "--allow-empty", "-m", "unrelated", cwd=tmp_path)
    assert version.read_git(str(tmp_path)) is None


def test_the_frozen_build_info_wins(monkeypatch):
    """An installed build reports what it was built from, not what is around it."""
    frozen = types.ModuleType("netqmpi._build_info")
    frozen.BUILD_INFO = {"version": "9.9.9", "tag": "v9.9.9",
                         "commits_since_tag": 0, "commit": "f" * 40,
                         "dirty": False}
    monkeypatch.setitem(sys.modules, "netqmpi._build_info", frozen)
    monkeypatch.setattr(netqmpi, "_build_info", frozen, raising=False)

    info = version.get_build_info()
    assert info == BuildInfo(version="9.9.9", tag="v9.9.9", commits_since_tag=0,
                             commit="f" * 40, dirty=False, source="build")
    assert netqmpi.__version__ == "9.9.9"


def test_the_build_info_module_round_trips():
    info = {"version": "0.3.1", "tag": "v0.3.1", "commits_since_tag": 2,
            "commit": "a" * 40, "dirty": True}
    namespace = {}
    exec(version.render_build_info(info), namespace)
    assert namespace["BUILD_INFO"] == info


@pytest.mark.parametrize("info, text", [
    (BuildInfo("0.3.1", "v0.3.1", 0, "abc", False, "build"),
     "NetQMPI 0.3.1 (tag v0.3.1, commit abc)"),
    (BuildInfo("0.3.1", "v0.3.1", 1, "abc", False, "git"),
     "NetQMPI 0.3.1 (tag v0.3.1 + 1 commit, commit abc)"),
    (BuildInfo("0.3.1", "v0.3.1", 4, "abc", True, "git"),
     "NetQMPI 0.3.1 (tag v0.3.1 + 4 commits, commit abc, uncommitted changes)"),
    (BuildInfo("0.3.1", None, None, "abc", False, "git"),
     "NetQMPI 0.3.1 (no tag, commit abc)"),
    (BuildInfo("0.3.1"), "NetQMPI 0.3.1 (no git information)"),
])
def test_describe(info, text):
    assert describe(info) == text


def test_the_cli_prints_the_version_without_other_arguments():
    """``--version`` works even though ``-n`` and a script are required."""
    done = subprocess.run([sys.executable, "-m", "netqmpi.runtime.cli", "--version"],
                          cwd=REPO_ROOT, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    assert done.stdout.startswith("NetQMPI "), done.stdout


@needs_git
def test_a_build_freezes_the_tag_and_commit(tmp_path):
    """``setup.py build_py`` writes _build_info.py into the build, not the sources."""
    done = subprocess.run(
        [sys.executable, "setup.py", "-q", "build_py", "--build-lib", str(tmp_path)],
        cwd=REPO_ROOT, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr

    namespace = {}
    exec((tmp_path / "netqmpi" / "_build_info.py").read_text(), namespace)
    frozen = namespace["BUILD_INFO"]
    assert frozen["commit"] == git("rev-parse", "HEAD")
    assert frozen["tag"] == git("describe", "--tags", "--abbrev=0")
    assert frozen["version"] == version.setup_py_version(str(REPO_ROOT))
    assert not (REPO_ROOT / "netqmpi" / "_build_info.py").exists()
