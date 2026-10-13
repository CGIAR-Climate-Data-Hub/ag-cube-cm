"""Tests for the lock-tolerant filesystem helpers.

The real failure these guard against is a Windows file lock, which cannot be
reproduced portably — so the retry *policy* is tested with injected errors
instead, and the real-world behaviour is covered by the concurrency test below.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from ag_cube_cm.models._fileops import copy_retry, retry_on_oserror, rmtree_retry


def test_retry_succeeds_after_transient_failures():
    """An operation that fails twice then succeeds must return its value."""
    attempts = {"n": 0}

    def flaky():
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise PermissionError(32, "being used by another process")
        return "ok"

    assert retry_on_oserror(flaky, "flaky op", delay=0.001) == "ok"
    assert attempts["n"] == 3


def test_retry_reraises_after_exhausting_attempts():
    """A persistent failure must propagate, not be swallowed."""
    def always_fails():
        raise PermissionError(32, "locked forever")

    with pytest.raises(PermissionError):
        retry_on_oserror(always_fails, "doomed op", retries=3, delay=0.001)


def test_retry_does_not_catch_other_exceptions():
    """Only OSError is transient; a logic error must surface immediately."""
    def bad_logic():
        raise ValueError("this is a bug, not a lock")

    with pytest.raises(ValueError):
        retry_on_oserror(bad_logic, "op", delay=0.001)


def test_copy_retry_copies_file(tmp_path):
    src = tmp_path / "src.txt"
    src.write_text("DSSAT")
    dst = tmp_path / "nested" / "dst.txt"
    dst.parent.mkdir()

    copy_retry(src, dst)
    assert dst.read_text() == "DSSAT"


def test_rmtree_retry_removes_tree(tmp_path):
    root = tmp_path / "run"
    (root / "deep").mkdir(parents=True)
    (root / "deep" / "Summary.OUT").write_text("results")

    assert rmtree_retry(root) is True
    assert not root.exists()


def test_rmtree_retry_handles_readonly_files(tmp_path):
    """DSSAT writes some outputs read-only, which blocks deletion on Windows."""
    root = tmp_path / "run"
    root.mkdir()
    locked = root / "readonly.OUT"
    locked.write_text("x")
    os.chmod(locked, stat.S_IREAD)

    try:
        assert rmtree_retry(root) is True
        assert not root.exists()
    finally:
        if locked.exists():
            os.chmod(locked, stat.S_IWRITE)


def test_rmtree_retry_is_noop_for_missing_path(tmp_path):
    assert rmtree_retry(tmp_path / "never-existed") is True


def test_bootstrap_is_idempotent(sim_config, tmp_path):
    """A second bootstrap must not rewrite files a sibling thread may be reading.

    This is the race that silently dropped pixels: the old implementation
    unlinked and recopied every file on every call.
    """
    from ag_cube_cm.models.dssat.base import DSSATModel

    model = DSSATModel(sim_config)
    base = str(tmp_path / "dssat_base")

    home, _ = model._bootstrap_dssat_home(base)
    probe = home / "DATA.CDE"
    if not probe.exists():
        pytest.skip("bundled static/ has no DATA.CDE to probe")

    before = probe.stat().st_mtime_ns
    model._bootstrap_dssat_home(base)
    assert probe.stat().st_mtime_ns == before, (
        "bootstrap rewrote an already-present file — the race window is still open"
    )
