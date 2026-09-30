"""
Regression tests for :func:`netqmpi.helpers.load_main` (G3).

``load_main`` ran ``runpy.run_path`` on every call, which reads and compiles
the script each time: 7-10 ms per run on the cluster's shared file system
against 0.4 ms on a local disk. It now compiles a script once per version of
the file — but must still *run* it every time, because scripts read their
parameters from the environment when they load.
"""
from __future__ import annotations

import builtins
import os
import textwrap

import pytest

from netqmpi import helpers

APP = """
    import os
    WIDTH = int(os.environ.get("WIDTH", "1"))

    def main(env=None):
        return {marker!r}, WIDTH
"""


@pytest.fixture
def compiles(monkeypatch):
    """Count the calls to ``compile`` made by the helpers module."""
    calls = []

    def counting_compile(*args, **kwargs):
        calls.append(args[1])
        return builtins.compile(*args, **kwargs)

    monkeypatch.setattr(helpers, "compile", counting_compile, raising=False)
    helpers._CODE_CACHE.clear()
    yield calls
    helpers._CODE_CACHE.clear()


def write(path, marker):
    path.write_text(textwrap.dedent(APP.format(marker=marker)))


def test_a_script_is_compiled_once(tmp_path, compiles):
    app = tmp_path / "app.py"
    write(app, "first")
    for _ in range(3):
        assert helpers.load_main(str(app))()[0] == "first"
    assert compiles == [str(app)]


def test_a_changed_script_is_compiled_again(tmp_path, compiles):
    app = tmp_path / "app.py"
    write(app, "first")
    assert helpers.load_main(str(app))()[0] == "first"

    write(app, "second, and longer")
    stat = app.stat()
    os.utime(app, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
    assert helpers.load_main(str(app))()[0] == "second, and longer"
    assert len(compiles) == 2


def test_the_script_still_runs_every_time(tmp_path, compiles, monkeypatch):
    """Module-level code sees the environment of each call, not the first."""
    app = tmp_path / "app.py"
    write(app, "x")
    monkeypatch.setenv("WIDTH", "2")
    assert helpers.load_main(str(app))() == ("x", 2)
    monkeypatch.setenv("WIDTH", "6")
    assert helpers.load_main(str(app))() == ("x", 6)


def test_the_script_runs_as_runpy_would_run_it(tmp_path, compiles):
    app = tmp_path / "app.py"
    app.write_text(textwrap.dedent("""
        import sys
        SEEN = (__name__, __file__, sys.argv[0], __name__ in sys.modules)
        def main(env=None):
            return SEEN
    """))
    assert helpers.load_main(str(app))() == (
        "<run_path>", str(app), str(app), True)
    assert "<run_path>" not in __import__("sys").modules


@pytest.mark.parametrize("path, message", [
    (None, "must be provided"), ("app.txt", "must be a .py"),
])
def test_bad_paths_are_refused(path, message):
    with pytest.raises(ValueError, match=message):
        helpers.load_main(path)
