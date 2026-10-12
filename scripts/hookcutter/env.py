"""Paths, tools, the ffmpeg runner, errors, JSON io and hashing.

This module is imported by the bootstrap path (``hc doctor`` under the system
python), so it must stay stdlib-only and Python 3.9 compatible: no ``X | Y``
types at runtime, no ``match``, no 3.10+ stdlib APIs.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence, Tuple, Union

from . import __version__

PathLike = Union[str, "os.PathLike[str]"]

SKILL_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPTS_DIR = SKILL_ROOT / "scripts"
APP_NAME = "hook-cutter"

# ffmpeg stderr substrings that signal timestamp damage: a hit fails the step.
FF_DENY = (
    "non monotonically increasing dts",
    "non-monotonous dts",
    "queue input is backward in time",
    "invalid dts",
    "invalid pts",
    "timestamps are unset",
    "application provided invalid",
)
# ffmpeg stderr substrings worth recording, never fatal.
FF_WARN = (
    "past duration",
    "error while decoding",
    "corrupt decoded frame",
    "concealing",
)

FINGERPRINT_CHUNK = 4 * 1024 * 1024


class HCError(Exception):
    """User-facing failure. code: 1 user/spec error, 2 evidence gap, 3 tool/env error."""

    def __init__(self, msg: str, code: int = 1):
        super().__init__(msg)
        self.msg = msg
        self.code = int(code)

    def __str__(self) -> str:
        return self.msg


# ---------------------------------------------------------------- paths

def skill_root() -> Path:
    return SKILL_ROOT


def data_dir() -> Path:
    """Per-user data dir: $HOOKCUTTER_HOME, else the platform default."""
    override = os.environ.get("HOOKCUTTER_HOME")
    if override:
        return Path(override).expanduser()
    home = Path.home()
    if sys.platform == "darwin":
        return home / "Library" / "Application Support" / APP_NAME
    if sys.platform.startswith("win"):
        base = os.environ.get("LOCALAPPDATA") or str(home / "AppData" / "Local")
        return Path(base) / APP_NAME
    base = os.environ.get("XDG_DATA_HOME") or str(home / ".local" / "share")
    return Path(base) / APP_NAME


def venv_dir() -> Path:
    return data_dir() / "venv"


def venv_python(venv: Optional[Path] = None) -> Path:
    v = Path(venv) if venv else venv_dir()
    if sys.platform.startswith("win"):
        return v / "Scripts" / "python.exe"
    return v / "bin" / "python"


def in_venv() -> bool:
    """True when the running interpreter is the skill's venv."""
    try:
        return Path(sys.prefix).resolve() == venv_dir().resolve()
    except OSError:
        return False


def hf_home() -> Path:
    return data_dir() / "models" / "hf"


def fw_models_dir() -> Path:
    return data_dir() / "models" / "fw"


def user_presets_dir() -> Path:
    return data_dir() / "presets"


def user_fonts_dir() -> Path:
    return data_dir() / "fonts"


def cache_dir() -> Path:
    return data_dir() / "cache"


def bundled_fonts_dir() -> Path:
    return SKILL_ROOT / "fonts"


def bundled_models_dir() -> Path:
    return SKILL_ROOT / "models"


def bundled_presets_dir() -> Path:
    return SKILL_ROOT / "presets"


def setup_runtime_env() -> None:
    """Point HF caches into the data dir. Call before importing any HF code."""
    os.environ["HF_HOME"] = str(hf_home())
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


def code_version() -> str:
    return __version__


# ---------------------------------------------------------------- logging

def verbose() -> bool:
    return os.environ.get("HC_VERBOSE", "") not in ("", "0", "false", "no")


def log(step: str, msg: str) -> None:
    print("[%s] %s" % (step, msg), file=sys.stderr, flush=True)


# ---------------------------------------------------------------- tools

def _tool(env_var: str, name: str) -> str:
    override = os.environ.get(env_var)
    if override:
        p = Path(override).expanduser()
        if p.exists():
            return str(p)
        found = shutil.which(override)
        if found:
            return found
        raise HCError("%s=%s does not exist" % (env_var, override), code=3)
    found = shutil.which(name)
    if not found:
        raise HCError("%s not found on PATH (install ffmpeg >= 6, or set %s). Run: hc doctor"
                      % (name, env_var), code=3)
    return found


def ffmpeg() -> str:
    return _tool("HOOKCUTTER_FFMPEG", "ffmpeg")


def ffprobe() -> str:
    return _tool("HOOKCUTTER_FFPROBE", "ffprobe")


def quote_cmd(args: Sequence[Any]) -> str:
    out = []
    for a in args:
        s = str(a)
        if not s or re.search(r"[^\w@%+=:,./-]", s):
            s = "'" + s.replace("'", "'\"'\"'") + "'"
        out.append(s)
    return " ".join(out)


class FFResult(object):
    """Outcome of an ffmpeg run. stdout is bytes; stderr is text."""

    def __init__(self, cmd: List[str], returncode: int, stdout: bytes, stderr: str,
                 warnings: List[str]):
        self.cmd = cmd
        self.args = cmd
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        self.warnings = warnings

    @property
    def text(self) -> str:
        return (self.stdout or b"").decode("utf-8", "replace")

    def __repr__(self) -> str:
        return "FFResult(rc=%d, warnings=%d)" % (self.returncode, len(self.warnings))


def _ff_args(args: Sequence[Any]) -> List[str]:
    args = [str(a) for a in args]
    pre = [ffmpeg(), "-hide_banner", "-nostdin"]
    if "-loglevel" not in args and "-v" not in args:
        pre += ["-loglevel", "warning"]
    if "-n" not in args and "-y" not in args:
        pre.append("-y")
    return pre + args


def scan_stderr(text: str, allow: Iterable[str] = ()) -> Tuple[List[str], List[str]]:
    """Split ffmpeg stderr into (deny_hits, warn_hits) per FF_DENY / FF_WARN."""
    allow_l = [a.lower() for a in allow if a]
    deny, warn = [], []
    for line in (text or "").splitlines():
        low = line.lower()
        hit = next((d for d in FF_DENY if d in low), None)
        if hit is not None:
            if hit in allow_l or any(a in low for a in allow_l):
                warn.append(line.strip())
            else:
                deny.append(line.strip())
            continue
        if any(w in low for w in FF_WARN):
            warn.append(line.strip())
    return deny, warn


def _tail(text: str, n: int = 15) -> str:
    lines = [ln for ln in (text or "").splitlines() if ln.strip()]
    return "\n".join(lines[-n:])


def _check_ff(cmd: List[str], rc: int, err: str, allow: Iterable[str], check: bool) -> List[str]:
    deny, warn = scan_stderr(err, allow)
    if check and rc != 0:
        raise HCError("ffmpeg failed (exit %d): %s\n%s" % (rc, quote_cmd(cmd[1:8]) + " ...",
                                                           _tail(err)), code=3)
    if deny:
        shown = "\n  ".join(deny[:10])
        more = "" if len(deny) <= 10 else "\n  (+%d more)" % (len(deny) - 10)
        raise HCError("ffmpeg reported timestamp damage (refusing the output):\n  %s%s"
                      % (shown, more), code=3)
    return warn


def _append_log(log_to: Optional[PathLike], cmd: List[str], err: str) -> None:
    if not log_to:
        return
    p = Path(log_to)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as fh:
        fh.write("$ %s\n%s\n" % (quote_cmd(cmd), err or ""))


def ff(args: Sequence[Any], *, input: Optional[bytes] = None, capture: bool = True,
       check: bool = True, allow: Iterable[str] = (), log_to: Optional[PathLike] = None) -> FFResult:
    """Run ffmpeg with -hide_banner -nostdin -loglevel warning -y prepended.

    Raises HCError(code 3) on a non-zero exit (when check) or on any FF_DENY line
    not covered by ``allow``. FF_WARN lines are returned in ``result.warnings``.
    ``capture=False`` discards stdout. ``log_to`` appends the command and stderr.
    """
    cmd = _ff_args(args)
    if verbose():
        log("ffmpeg", quote_cmd(cmd))
    try:
        proc = subprocess.run(cmd, input=input,
                              stdin=None if input is not None else subprocess.DEVNULL,
                              stdout=subprocess.PIPE if capture else subprocess.DEVNULL,
                              stderr=subprocess.PIPE)
    except OSError as exc:
        raise HCError("cannot run ffmpeg: %s" % exc, code=3)
    err = proc.stderr.decode("utf-8", "replace")
    _append_log(log_to, cmd, err)
    warn = _check_ff(cmd, proc.returncode, err, allow, check)
    return FFResult(cmd, proc.returncode, proc.stdout if capture else b"", err, warn)


def ff_popen(args: Sequence[Any], log_path: PathLike) -> "subprocess.Popen[bytes]":
    """Start ffmpeg with stdin=PIPE (caption strip pipe); stderr goes to log_path."""
    cmd = _ff_args(args)
    if verbose():
        log("ffmpeg", quote_cmd(cmd))
    lp = Path(log_path)
    lp.parent.mkdir(parents=True, exist_ok=True)
    fh = open(lp, "wb")
    try:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=fh)
    except OSError as exc:
        fh.close()
        raise HCError("cannot run ffmpeg: %s" % exc, code=3)
    proc.hc_log_file = fh  # type: ignore[attr-defined]
    proc.hc_cmd = cmd  # type: ignore[attr-defined]
    return proc


def ff_wait(proc: "subprocess.Popen[bytes]", log_path: PathLike, allow: Iterable[str] = (),
            check: bool = True) -> FFResult:
    """Close stdin, wait for an ff_popen process, then scan its stderr log."""
    try:
        if proc.stdin and not proc.stdin.closed:
            proc.stdin.close()
    except (BrokenPipeError, OSError):
        pass
    rc = proc.wait()
    fh = getattr(proc, "hc_log_file", None)
    if fh is not None and not fh.closed:
        fh.close()
    try:
        err = Path(log_path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        err = ""
    cmd = list(getattr(proc, "hc_cmd", []) or [])
    warn = _check_ff(cmd or ["ffmpeg"], rc, err, allow, check)
    return FFResult(cmd, rc, b"", err, warn)


def ffprobe_json(path: PathLike, *args: Any) -> dict:
    """ffprobe -v error -of json <args> <path> → parsed dict."""
    cmd = [ffprobe(), "-v", "error", "-of", "json"] + [str(a) for a in args] + [str(path)]
    if verbose():
        log("ffprobe", quote_cmd(cmd))
    try:
        proc = subprocess.run(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE)
    except OSError as exc:
        raise HCError("cannot run ffprobe: %s" % exc, code=3)
    if proc.returncode != 0:
        raise HCError("ffprobe failed on %s: %s" % (path, _tail(proc.stderr.decode("utf-8", "replace"), 5)),
                      code=3)
    try:
        return json.loads(proc.stdout.decode("utf-8", "replace") or "{}")
    except ValueError as exc:
        raise HCError("ffprobe returned invalid JSON for %s: %s" % (path, exc), code=3)


def run(cmd: Sequence[Any], *, check: bool = True, capture: bool = True,
        env: Optional[dict] = None, cwd: Optional[PathLike] = None,
        timeout: Optional[float] = None) -> "subprocess.CompletedProcess[str]":
    """Generic subprocess runner (text mode). Missing tool → HCError code 3."""
    cmd = [str(c) for c in cmd]
    if verbose():
        log("run", quote_cmd(cmd))
    try:
        proc = subprocess.run(cmd, stdin=subprocess.DEVNULL,
                              stdout=subprocess.PIPE if capture else None,
                              stderr=subprocess.PIPE if capture else None,
                              universal_newlines=True, env=env,
                              cwd=str(cwd) if cwd else None, timeout=timeout)
    except FileNotFoundError:
        raise HCError("command not found: %s" % cmd[0], code=3)
    except subprocess.TimeoutExpired:
        raise HCError("command timed out: %s" % quote_cmd(cmd[:4]), code=3)
    if check and proc.returncode != 0:
        raise HCError("%s failed (exit %d): %s" % (Path(cmd[0]).name, proc.returncode,
                                                  _tail(proc.stderr or proc.stdout or "", 10)), code=3)
    return proc


_FILTER_CACHE = {}  # type: dict


def ffmpeg_capabilities(kind: str = "filters") -> set:
    """Names listed by ``ffmpeg -<kind>`` (filters, encoders, decoders), cached."""
    key = (ffmpeg(), kind)
    if key in _FILTER_CACHE:
        return _FILTER_CACHE[key]
    proc = run([ffmpeg(), "-hide_banner", "-" + kind], check=False)
    names = set()
    flag_re = re.compile(r"^[A-Z.|]{2,7}$")
    for line in (proc.stdout or "").splitlines():
        parts = line.split()
        if len(parts) >= 2 and flag_re.match(parts[0]) and not parts[1].startswith("="):
            names.add(parts[1])
    _FILTER_CACHE[key] = names
    return names


def has_filter(name: str) -> bool:
    try:
        return name in ffmpeg_capabilities("filters")
    except HCError:
        return False


# ---------------------------------------------------------------- JSON io

def file_mode() -> int:
    """Permission bits a normal open() would give a new file (0o666 minus the umask)."""
    um = os.umask(0)
    os.umask(um)
    return 0o666 & ~um


def write_json(path: PathLike, obj: Any) -> Path:
    """Atomic write (tmp + rename), indent 2, ensure_ascii=False, trailing newline."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(obj, indent=2, ensure_ascii=False, default=_json_default) + "\n"
    fd, tmp = tempfile.mkstemp(prefix="." + p.name + ".", suffix=".tmp", dir=str(p.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(data)
        os.chmod(tmp, file_mode())  # mkstemp creates 0600
        os.replace(tmp, str(p))
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return p


def read_json(path: PathLike) -> Any:
    """Parse a JSON file. A UTF-8 byte-order mark (Windows editors add one) is accepted."""
    p = Path(path)
    try:
        with open(p, "r", encoding="utf-8-sig") as fh:
            return json.load(fh)
    except FileNotFoundError:
        raise HCError("missing file: %s" % p, code=1)
    except ValueError as exc:
        raise HCError("invalid JSON in %s: %s" % (p, exc), code=1)


def unknown_clip(clip_id: Any, known: Iterable[Any]) -> HCError:
    """The error for a clip id that is not in clips.json (code 1), with a suggestion.

    Callers ``raise env.unknown_clip(cid, ids)``. The message names the closest ids
    (difflib) and lists the clips that exist, so a typo is fixed in one step."""
    import difflib
    ids = [str(k) for k in known]
    cid = str(clip_id)
    # a bare prefix ("c01") points at "c01-fired-big-client" before any fuzzy match
    near = [k for k in ids if cid and k.startswith(cid)][:3]
    if not near:
        near = difflib.get_close_matches(cid, ids, n=3, cutoff=0.6)
    msg = "unknown clip %r" % cid
    if near:
        msg += " (did you mean %s?)" % " or ".join(repr(k) for k in near)
    if ids:
        shown = ", ".join(ids[:20]) + (", ..." if len(ids) > 20 else "")
        msg += ". clips.json has: %s" % shown
    else:
        msg += ". clips.json has no clips yet"
    return HCError(msg, code=1)


def _json_default(o: Any) -> Any:
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    if hasattr(o, "numerator") and hasattr(o, "denominator"):
        return [int(o.numerator), int(o.denominator)]
    if hasattr(o, "tolist"):  # numpy arrays and scalars
        return o.tolist()
    if hasattr(o, "item"):
        return o.item()
    raise TypeError("not JSON serializable: %r" % type(o))


# ---------------------------------------------------------------- hashing

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: PathLike) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fingerprint(path: PathLike) -> str:
    """sha256 of (size, first 4 MiB, last 4 MiB): fast identity for large media."""
    p = Path(path)
    size = p.stat().st_size
    h = hashlib.sha256()
    h.update(str(size).encode("ascii") + b"\0")
    with open(p, "rb") as fh:
        h.update(fh.read(FINGERPRINT_CHUNK))
        if size > FINGERPRINT_CHUNK:
            fh.seek(max(FINGERPRINT_CHUNK, size - FINGERPRINT_CHUNK))
            h.update(fh.read(FINGERPRINT_CHUNK))
    return h.hexdigest()


_PACKAGE_DIGESTS = {}  # type: dict


def package_digest(root: Optional[PathLike] = None) -> str:
    """Identity of the code behind the QA checks: sha256 over the version plus the name and
    bytes of every ``hookcutter/*.py`` (cached per process). The whole package on purpose: a
    hand-kept list of what QA depends on would fall behind, and an extra QA run is cheap.
    ``root`` is the package folder (tests pass a copy)."""
    base = Path(root) if root is not None else Path(__file__).resolve().parent
    key = str(base)
    if key not in _PACKAGE_DIGESTS:
        h = hashlib.sha256()
        h.update(str(__version__).encode("utf-8") + b"\0")
        for f in sorted(base.glob("*.py"), key=lambda x: x.name):
            h.update(f.name.encode("utf-8") + b"\0")
            try:
                h.update(sha256_file(f).encode("ascii"))
            except OSError:
                h.update(b"unreadable")
            h.update(b"\0")
        _PACKAGE_DIGESTS[key] = h.hexdigest()
    return _PACKAGE_DIGESTS[key]


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      default=_json_default)


def stable_hash(obj: Any) -> str:
    """sha256 of canonical JSON (sorted keys, compact separators)."""
    return sha256_bytes(canonical_json(obj).encode("utf-8"))


# ---------------------------------------------------------------- time helpers

_TIME_RE = re.compile(r"^(?:(\d+):)?(?:(\d+):)?(\d+(?:\.\d*)?|\.\d+)$")


def parse_time(value: Any) -> float:
    """Seconds from 681.5, "681.5", "11:21", "11:21.40" or "1:02:03.4"."""
    if isinstance(value, bool):
        raise HCError("invalid time: %r" % (value,), code=1)
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip()
    neg = s.startswith("-")
    if neg:
        s = s[1:].strip()
    m = _TIME_RE.match(s)
    if not m:
        raise HCError("invalid time %r (use seconds, MM:SS(.s) or H:MM:SS(.s))" % (value,), code=1)
    a, b, c = m.group(1), m.group(2), m.group(3)
    if a is not None and b is not None:
        h, mi, sec = int(a), int(b), float(c)
    elif a is not None:
        h, mi, sec = 0, int(a), float(c)
    else:
        h, mi, sec = 0, 0, float(c)
    if (a is not None) and sec >= 60:
        raise HCError("invalid time %r: seconds field must be < 60" % (value,), code=1)
    if a is not None and b is not None and mi >= 60:
        raise HCError("invalid time %r: minutes field must be < 60" % (value,), code=1)
    t = h * 3600 + mi * 60 + sec
    return -t if neg else t


def fmt_time(t: Optional[float], decimals: int = 1) -> str:
    """mm:ss.s (or h:mm:ss.s) for labels and tables."""
    if t is None:
        return "-"
    neg = t < 0
    t = abs(float(t))
    scale = 10 ** decimals
    total = int(round(t * scale))
    whole, frac = divmod(total, scale)
    h, rem = divmod(whole, 3600)
    m, s = divmod(rem, 60)
    sec = ("%02d" % s) + (("." + str(frac).zfill(decimals)) if decimals > 0 else "")
    out = ("%d:%02d:%s" % (h, m, sec)) if h else ("%02d:%s" % (m, sec))
    return "-" + out if neg else out


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def human_bytes(n: float) -> str:
    n = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1000 or unit == "TB":
            return ("%d %s" % (n, unit)) if unit == "B" else ("%.1f %s" % (n, unit))
        n /= 1000.0
    return "%.1f TB" % n
