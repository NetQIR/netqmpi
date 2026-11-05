"""
Tests for :func:`netqmpi.helpers.ensure_cunqa_importable`.

The CUNQA adapters used to run ``sys.path.append(os.getenv("HOME"))`` at
import, which put ``None`` on ``sys.path`` when ``HOME`` was unset and changed
``sys.path`` even when ``cunqa`` was already importable. None of these tests
needs CUNQA.
"""
from __future__ import annotations

import importlib.machinery
import sys
import types

import pytest

from netqmpi import helpers

BASE = ["/site-packages"]


@pytest.fixture
def cunqa_missing(monkeypatch):
    """Make ``cunqa`` look not importable, and give ``sys.path`` a known state."""
    monkeypatch.setattr(sys, "path", list(BASE))
    monkeypatch.delitem(sys.modules, "cunqa", raising=False)
    monkeypatch.setattr(helpers.importlib.util, "find_spec",
                        lambda name: None)


def test_an_importable_cunqa_leaves_sys_path_alone(monkeypatch):
    monkeypatch.setattr(sys, "path", list(BASE))
    monkeypatch.delitem(sys.modules, "cunqa", raising=False)
    installed = importlib.machinery.ModuleSpec("cunqa", loader=object())
    monkeypatch.setattr(helpers.importlib.util, "find_spec",
                        lambda name: installed)

    helpers.ensure_cunqa_importable({"NETQMPI_CUNQA_PATH": "/opt/cunqa",
                                     "HOME": "/h"})

    assert sys.path == BASE


def test_the_env_var_is_tried_before_home(cunqa_missing):
    helpers.ensure_cunqa_importable({"NETQMPI_CUNQA_PATH": "/opt/cunqa",
                                     "HOME": "/h"})
    assert sys.path == BASE + ["/opt/cunqa", "/h"]


def test_home_alone_still_works(cunqa_missing):
    """CUNQA installs that live in $HOME keep working with no setup."""
    helpers.ensure_cunqa_importable({"HOME": "/h"})
    assert sys.path == BASE + ["/h"]


def test_unset_home_adds_nothing(cunqa_missing):
    helpers.ensure_cunqa_importable({})
    assert sys.path == BASE


def test_a_bare_cunqa_directory_does_not_count_as_installed(tmp_path,
                                                           monkeypatch):
    """A ``cunqa/`` folder without ``__init__.py`` is only a namespace."""
    (tmp_path / "cunqa").mkdir()
    monkeypatch.setattr(sys, "path", [str(tmp_path)])
    monkeypatch.delitem(sys.modules, "cunqa", raising=False)

    helpers.ensure_cunqa_importable({"HOME": "/h"})

    assert sys.path == [str(tmp_path), "/h"]


def test_an_already_imported_cunqa_leaves_sys_path_alone(monkeypatch):
    """Stubs registered by hand have no ``__spec__``; they still count."""
    monkeypatch.setattr(sys, "path", list(BASE))
    monkeypatch.setitem(sys.modules, "cunqa", types.ModuleType("cunqa"))

    helpers.ensure_cunqa_importable({"HOME": "/h"})

    assert sys.path == BASE


def test_a_path_already_present_is_not_added_twice(cunqa_missing):
    sys.path.append("/h")
    helpers.ensure_cunqa_importable({"HOME": "/h"})
    assert sys.path.count("/h") == 1
