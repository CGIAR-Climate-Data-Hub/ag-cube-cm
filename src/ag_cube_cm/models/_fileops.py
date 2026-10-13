"""
ag_cube_cm.models._fileops
===========================

Filesystem helpers that tolerate Windows file locking.

Crop models run as subprocesses over a tree of small files, many of them shared
between threads.  On Windows a file that another process still holds open cannot
be deleted or overwritten — the attempt raises ``PermissionError`` (WinError 32)
or ``WinError 145`` for a non-empty directory.  Both are transient: the handle
is released a moment later.

A failure here surfaces as a *lost pixel*, not a crash — the orchestrator records
the job as a model error and carries on — so an unhandled lock quietly removes
data from the output map and makes two identical runs disagree.  Retrying with a
short backoff removes that class of failure.

These live in one module rather than being copied per model, so the retry policy
has a single definition.
"""

from __future__ import annotations

import logging
import os
import shutil
import stat
import time
from pathlib import Path
from typing import Callable, TypeVar

logger = logging.getLogger(__name__)

__all__ = ["rmtree_retry", "copy_retry", "retry_on_oserror"]

T = TypeVar("T")

#: Attempts before giving up. Observed contention clears within one or two.
_DEFAULT_RETRIES = 5
#: Initial backoff, doubled each attempt (0.1, 0.2, 0.4, 0.8 s).
_DEFAULT_DELAY = 0.1


def retry_on_oserror(
    func: Callable[[], T],
    what: str,
    retries: int = _DEFAULT_RETRIES,
    delay: float = _DEFAULT_DELAY,
) -> T:
    """Call *func*, retrying on :class:`OSError` with exponential backoff.

    Parameters
    ----------
    func : callable
        Zero-argument operation to attempt.
    what : str
        Short description used in log messages, e.g. ``"copy MODEL.ERR"``.
    retries : int
        Maximum attempts.  Default: 5.
    delay : float
        Initial backoff in seconds, doubled per attempt.  Default: 0.1.

    Returns
    -------
    T
        Whatever *func* returns.

    Raises
    ------
    OSError
        Re-raised from the final attempt if every try fails.
    """
    wait = delay
    for attempt in range(1, retries + 1):
        try:
            return func()
        except OSError as exc:
            if attempt == retries:
                logger.warning("%s failed after %d attempts: %s", what, retries, exc)
                raise
            logger.debug(
                "%s failed (attempt %d/%d): %s — retrying in %.2fs",
                what, attempt, retries, exc, wait,
            )
            time.sleep(wait)
            wait *= 2
    raise AssertionError("unreachable")  # pragma: no cover


def _force_writable(func, path, _exc) -> None:
    """``onerror`` hook: clear the read-only bit and retry once.

    DSSAT writes some outputs read-only, which blocks deletion on Windows even
    when no handle is open.
    """
    try:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    except OSError:
        pass


def rmtree_retry(
    path: str | Path,
    retries: int = _DEFAULT_RETRIES,
    delay: float = _DEFAULT_DELAY,
) -> bool:
    """Remove a directory tree, tolerating transient locks and read-only files.

    Never raises: cleanup failure must not fail an otherwise good simulation.
    A tree left behind is a disk-space problem, not a correctness one.

    Parameters
    ----------
    path : str | Path
        Directory to remove.
    retries, delay : int, float
        Retry policy, as in :func:`retry_on_oserror`.

    Returns
    -------
    bool
        True if the tree is gone, False if it survived every attempt.
    """
    target = Path(path)
    if not target.exists():
        return True

    wait = delay
    for attempt in range(1, retries + 1):
        try:
            shutil.rmtree(target, onerror=_force_writable)
            if not target.exists():
                return True
        except OSError as exc:
            logger.debug("rmtree %s failed (attempt %d/%d): %s", target, attempt, retries, exc)
        if attempt < retries:
            time.sleep(wait)
            wait *= 2

    shutil.rmtree(target, ignore_errors=True)
    survived = target.exists()
    if survived:
        logger.warning(
            "Could not remove working directory %s — it will be left on disk.", target
        )
    return not survived


def copy_retry(
    src: str | Path,
    dst: str | Path,
    retries: int = _DEFAULT_RETRIES,
    delay: float = _DEFAULT_DELAY,
) -> None:
    """Copy a file, retrying while the source or destination is locked.

    Parameters
    ----------
    src, dst : str | Path
        Source and destination paths.
    retries, delay : int, float
        Retry policy, as in :func:`retry_on_oserror`.

    Raises
    ------
    OSError
        If every attempt fails — unlike cleanup, a missing input file *is*
        fatal to the run, so this one propagates.
    """
    retry_on_oserror(
        lambda: shutil.copy2(str(src), str(dst)),
        what=f"copy {Path(src).name}",
        retries=retries,
        delay=delay,
    )
