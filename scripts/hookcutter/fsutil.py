"""Filesystem helpers shared by every step that caches or writes artifacts.

`tmp_path` gives each writer its own temp name next to the destination, so two processes (or
threads) producing the same cached file never truncate each other's work; the final
`os.replace` is atomic. `file_lock` serializes whole steps (one render of a clip x format at a
time, one energy decode per source) with an advisory OS lock that dies with the process.

Stdlib only and Python 3.9 compatible (it is imported next to env.py).
"""

from __future__ import annotations

import contextlib
import os
import secrets
import sys
import tempfile
import time
from pathlib import Path
from typing import Iterator, Union

from . import env

PathLike = Union[str, "os.PathLike[str]"]


def tmp_path(dest: PathLike) -> Path:
    """A unique temp path in dest's directory: `<stem>.<pid><8 hex>.tmp<suffix>`.

    The suffix is kept so tools that infer a format from the extension (ffmpeg, numpy) still
    work, and the `.tmp.` infix lets cleanup globs (`*.tmp.*`) find leftovers."""
    d = Path(dest)
    return d.with_name("%s.%d%s.tmp%s" % (d.stem, os.getpid(), secrets.token_hex(4), d.suffix))


def _atomic_write(dest: PathLike, data: bytes) -> Path:
    p = Path(dest)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix="." + p.name + ".", suffix=".tmp", dir=str(p.parent))
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        os.chmod(tmp, env.file_mode())
        os.replace(tmp, str(p))
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return p


def atomic_write_bytes(dest: PathLike, data: bytes) -> Path:
    """Write bytes to dest via a same-directory temp file and an atomic rename."""
    return _atomic_write(dest, bytes(data))


def atomic_write_text(dest: PathLike, text: str, encoding: str = "utf-8") -> Path:
    """Write text to dest via a same-directory temp file and an atomic rename."""
    return _atomic_write(dest, str(text).encode(encoding))


def _try_lock(fh) -> bool:
    if sys.platform.startswith("win"):
        import msvcrt
        try:
            fh.seek(0)
            msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
            return True
        except OSError:
            return False
    import fcntl
    try:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except (BlockingIOError, PermissionError):
        return False
    except OSError as exc:  # pragma: no cover (EWOULDBLOCK spelled differently)
        import errno
        if exc.errno in (errno.EAGAIN, errno.EACCES):
            return False
        raise


def _unlock(fh) -> None:
    if sys.platform.startswith("win"):
        import msvcrt
        try:
            fh.seek(0)
            msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
        except OSError:
            pass
        return
    import fcntl
    try:
        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
    except OSError:
        pass


@contextlib.contextmanager
def file_lock(lock_path: PathLike, *, timeout: float = 900.0, poll: float = 0.25,
              what: str = "") -> Iterator[Path]:
    """Hold an exclusive advisory lock on `lock_path` for the `with` block.

    Another holder (a second process, or another thread with its own open) makes this wait,
    logging one "waiting for <what>" line; after `timeout` seconds it raises HCError (code 3).
    The lock file itself is left in place (deleting it would race a waiter)."""
    p = Path(lock_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fh = open(str(p), "a+b")
    try:
        if not _try_lock(fh):
            label = what or p.name
            env.log("lock", "waiting for %s (another hook-cutter process holds %s)" % (label, p))
            t_end = time.monotonic() + max(0.0, float(timeout))
            while True:
                if time.monotonic() >= t_end:
                    raise env.HCError("timed out after %.0f s waiting for %s (lock %s is held by "
                                      "another hook-cutter process; if none is running, delete it)"
                                      % (float(timeout), label, p), code=3)
                time.sleep(max(0.01, float(poll)))
                if _try_lock(fh):
                    break
        try:
            yield p
        finally:
            _unlock(fh)
    finally:
        fh.close()
