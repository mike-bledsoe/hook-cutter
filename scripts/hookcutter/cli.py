"""argparse dispatch for ``hc``. Command modules are imported lazily.

Stays Python 3.9 compatible and stdlib-only at import time: ``hc doctor`` runs
through here under the system python before the venv exists.
"""

from __future__ import annotations

import argparse
import errno
import importlib
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, NoReturn, Optional, Sequence, Tuple

from . import __version__, env

# command -> (module, function)
COMMANDS = {
    "doctor": ("doctor", "cmd_doctor"),
    "ingest": ("ingest", "cmd_ingest"),
    "transcribe": ("transcribe", "cmd_transcribe"),
    "find": ("find", "cmd_find"),
    "words": ("find", "cmd_words"),
    "peek": ("sheets", "cmd_peek"),
    "shots": ("shots", "cmd_shots"),
    "plan": ("plan", "cmd_plan"),
    "render": ("render", "cmd_render"),
    "cover": ("cover", "cmd_cover"),
    "qa": ("qa", "cmd_qa"),
    "export": ("export", "cmd_export"),
    "all": ("cli", "cmd_all"),
    "status": ("cli", "cmd_status"),
    "test": ("cli", "cmd_test"),
    "demo": ("demo", "cmd_demo"),
}
# Commands that read the source media through source.json: they stop when the file changed
# since ingest (ingest, status, find, words, export, doctor and test never decode it).
SOURCE_BOUND = ("transcribe", "shots", "peek", "plan", "render", "cover", "qa", "all")
# Frames of slack when a cover time is floored to its frame. Must stay equal to
# cover_frames.FRAME_TOL (hc cover floors with it); kept as a copy here so `hc status` does
# not import the numpy-bearing cover modules. tests/test_codex_fg.py pins the two together.
COVER_FRAME_TOL = 1e-3
STAMPS_FILE = "all-stamps.json"
STAMPS_LOCK = ".stamps.lock"
STAMP_SECTIONS = ("render", "cover", "qa")
VOLATILE_PLAN_KEYS = ("created", "timings", "elapsed")
# INCOMPLETE = every gate that ran passed, but some could not run (evidence gaps).
STATUS_RANK = {"PASS": 0, "SKIPPED": 1, "INCOMPLETE": 2, "WARN": 3, "FAIL": 4}
EXIT_RANK = {0: 0, 2: 1, 1: 2, 3: 3}   # worst-of order for exit codes (3 tool > 1 fail > 2 gap)


class Parser(argparse.ArgumentParser):
    """argparse with the documented exit code for usage mistakes: 1 (user error), not 2."""

    def error(self, message: str) -> NoReturn:  # type: ignore[override]
        self.print_usage(sys.stderr)
        self.exit(1, "%s: error: %s\n" % (self.prog, message))


def q(path: Any) -> str:
    """A path quoted for a shell command line the user may paste."""
    return shlex.quote(str(path))


def worst_exit(*codes: int) -> int:
    """The worst of several exit codes (tool error 3 > FAIL 1 > evidence gap 2 > 0)."""
    known = [int(c) for c in codes if c is not None]
    return max(known, key=lambda c: EXIT_RANK.get(c, 3)) if known else 0


def _csv(value: str) -> List[str]:
    return [x.strip() for x in str(value).split(",") if x.strip()]


def _time(value: str) -> float:
    """A time argument as clips.json takes it: seconds ("12.4") or a clock time ("0:12",
    "1:02:03.4"). A bad value is a usage error (exit 1) that names the two forms."""
    try:
        try:
            from . import spec as spec_mod
            parse = spec_mod.parse_time
        except ImportError:
            parse = env.parse_time
        return float(parse(value))
    except env.HCError as exc:
        raise argparse.ArgumentTypeError(exc.msg)


def build_parser() -> argparse.ArgumentParser:
    p = Parser(
        prog="hc", description="hook-cutter: find the hook, make the cuts, add the captions.",
        epilog="First time? hc demo shows a finished Short in about a minute. Typical run: doctor, ingest, transcribe, (write clips.json), plan, render, cover, qa, export.")
    p.add_argument("--version", action="store_true", help="print version and interpreter, then exit")
    sub = p.add_subparsers(dest="cmd", metavar="COMMAND")

    def add(name: str, help_text: str) -> argparse.ArgumentParser:
        return sub.add_parser(name, help=help_text, description=help_text)

    def project(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("project", help="project folder P (durable, never /tmp)")

    def clips(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("clips", nargs="*", metavar="CLIP", help="clip ids (default: all in clips.json)")

    def styles(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--styles", type=_csv, default=None, help="comma-separated preset names")

    def formats(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--formats", type=_csv, default=None, help="comma-separated: vertical,landscape")

    sp = add("doctor", "check the environment; --install sets it up")
    sp.add_argument("--install", action="store_true", help="create the venv and install pinned deps")
    sp.add_argument("--with", dest="extras", type=_csv, default=None, metavar="EXTRAS",
                    help="extras: mlx,url,whispercpp")
    sp.add_argument("--model", default=None, metavar="NAME",
                    help="pre-download an ASR model (large-v3-turbo or mlx-community/whisper-large-v3-turbo)")
    sp.add_argument("--backend", default="auto", choices=("auto", "mlx", "faster-whisper"),
                    help="with --model: the backend the model is for (default: what transcribe "
                         "--backend auto picks)")
    sp.add_argument("--python", default=None, metavar="PATH",
                    help="interpreter to build the venv from (3.12 or 3.13); a different "
                         "version rebuilds the venv")
    sp.add_argument("--json", action="store_true", help="machine-readable output")

    sp = add("ingest", "link/copy a local file or download a URL into P; probe; survey sheet")
    project(sp)
    sp.add_argument("source", help="video file or URL")
    sp.add_argument("--copy", action="store_true", help="copy instead of symlinking a local file")
    sp.add_argument("--section", default=None, metavar="A-B", help="URL only: download just A-B")
    sp.add_argument("--cfr", action="store_true",
                    help="transcode a variable-frame-rate source to constant frame rate first")
    sp.add_argument("--force", action="store_true", help="replace an existing source")

    sp = add("transcribe", "speech to text with word times -> transcript.json + transcript.md")
    project(sp)
    sp.add_argument("--backend", default="auto", choices=("auto", "mlx", "faster-whisper", "whispercpp"))
    sp.add_argument("--model", default=None)
    sp.add_argument("--language", default="auto", metavar="L|auto")
    sp.add_argument("--glossary", default=None, metavar="TEXT")
    sp.add_argument("--speakers-vtt", default=None, metavar="PATH")
    sp.add_argument("--file", default=None, metavar="VIDEO", help="caption-only: transcribe this file")
    sp.add_argument("--clip", default=None, metavar="ID", help="caption-only: the clip id")
    sp.add_argument("--retime", action="store_true",
                    help="re-time the existing transcript's words from the audio (no ASR)")
    sp.add_argument("--force", action="store_true")

    sp = add("find", "fuzzy-find a phrase in the transcript")
    project(sp)
    sp.add_argument("phrase")
    sp.add_argument("--near", default=None, metavar="MM:SS")
    sp.add_argument("--max", type=int, default=10)
    sp.add_argument("--clip", default=None, metavar="ID", help="search a caption-only clip's transcript")

    sp = add("words", "print transcript words A..B with times, gaps, fillers, speakers")
    project(sp)
    sp.add_argument("a", type=int)
    sp.add_argument("b", type=int)
    sp.add_argument("--clip", default=None, metavar="ID", help="read a caption-only clip's transcript")

    sp = add("peek", "contact sheet of a time range of the source (or of --file)")
    project(sp)
    sp.add_argument("start")
    sp.add_argument("end")
    sp.add_argument("--n", type=int, default=12)
    sp.add_argument("--file", default=None, metavar="PATH")

    sp = add("shots", "shot boundaries + faces + framing proposal")
    project(sp)
    clips(sp)
    sp.add_argument("--range", nargs=2, default=None, metavar=("START", "END"))

    sp = add("plan", "validate clips.json and build plans")
    project(sp)
    clips(sp)
    sp.add_argument("--quiet", action="store_true")

    sp = add("render", "render clips (stage A segments + stage B final)")
    project(sp)
    clips(sp)
    styles(sp)
    formats(sp)
    sp.add_argument("--force", action="store_true")

    sp = add("cover", "cover candidates + cover images")
    project(sp)
    clips(sp)
    styles(sp)
    sp.add_argument("--at", type=_time, default=None, metavar="SECONDS",
                    help="cut the cover at this output time (seconds or [h:]mm:ss[.ff])")
    sp.add_argument("--grid", action="store_true",
                    help="only rebuild sheets/covers-grid.jpg from the covers on disk")

    sp = add("qa", "quality gates -> qa/*.json, qa/*.md, qa/summary.md")
    project(sp)
    clips(sp)
    styles(sp)
    formats(sp)
    sp.add_argument("--no-asr", action="store_true")
    sp.add_argument("--require-coverage", action="store_true")

    sp = add("export", "export folder + manifest")
    project(sp)
    clips(sp)
    styles(sp)
    formats(sp)
    sp.add_argument("--force", action="store_true",
                    help="also export clips whose QA is FAIL, incomplete, stale or missing")

    sp = add("all", "plan, render, cover, qa, export (skips up-to-date work)")
    project(sp)
    clips(sp)
    styles(sp)
    formats(sp)
    sp.add_argument("--no-asr", action="store_true")
    sp.add_argument("--force", action="store_true", help="re-render and re-check everything")

    sp = add("status", "one line per clip: plan / render / cover / qa state")
    project(sp)

    sp = add("demo", "cut one Short from a made-up 20-second recording, to see the output")
    sp.add_argument("dir", nargs="?", default=None, metavar="DIR",
                    help="where to make it (default ~/hook-cutter-demo)")
    sp.add_argument("--force", action="store_true", help="make it again from the start")
    sp.add_argument("--open", action="store_true", help="open the finished Short when done")

    sp = add("test", "run the test suite under the venv")
    sp.add_argument("--slow", action="store_true", help="include ASR tests")
    sp.add_argument("pattern", nargs="?", default=None)
    return p


def _resolve(cmd: str) -> Any:
    mod_name, func_name = COMMANDS[cmd]
    if mod_name == "cli":
        return globals()[func_name]
    full = "hookcutter." + mod_name
    try:
        mod = importlib.import_module(full)
    except ImportError as exc:
        if getattr(exc, "name", None) == full:
            raise env.HCError("'%s' is not available in this build (hookcutter/%s.py is missing)"
                              % (cmd, mod_name), code=3)
        raise env.HCError("'%s' could not load: %s. Run: hc doctor" % (cmd, exc), code=3)
    func = getattr(mod, func_name, None)
    if func is None:
        raise env.HCError("'%s' is not available in this build (%s.%s missing)" % (cmd, mod_name, func_name),
                          code=3)
    return func


def _require_source_current(args: argparse.Namespace) -> None:
    """Stop a source-bound command when P's source file is no longer the one ingest probed.

    source.json's fingerprint, duration, stream layout and picture facts describe that file,
    and the transcript, plans, cached segments and renders were made from it, so a file
    replaced at the same path must be ingested again first (a hard stop, not a cache key).
    Skipped for ``--file`` (a caption-only transcript, a peek at a rendered file) and when
    every clip named is caption-only: those read their own file, which is checked live."""
    if args.cmd not in SOURCE_BOUND or getattr(args, "file", None):
        return
    project = getattr(args, "project", None)
    if not project:
        return
    from .project import Project
    proj = Project(project)
    if not proj.root.is_dir():
        return
    named = [c for c in (getattr(args, "clips", None) or []) if isinstance(c, str)]
    try:
        if named and all(proj.clip_kind(c) == "caption_only" for c in named):
            return
    except env.HCError:
        pass
    proj.require_source_current()


def dispatch(args: argparse.Namespace) -> int:
    _require_source_current(args)
    rc = _resolve(args.cmd)(args)
    return int(rc or 0)


def _utf8_stdio() -> None:
    """Print any transcript text: a Windows pipe defaults to the ANSI code page (cp1252)."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except (AttributeError, ValueError, OSError):
            pass


def _os_error(exc: OSError) -> str:
    what = exc.strerror or (os.strerror(exc.errno) if exc.errno else "") or str(exc) \
        or type(exc).__name__
    names = [n for n in (exc.filename, getattr(exc, "filename2", None)) if n]
    hint = ""
    if exc.errno in (errno.EACCES, errno.EPERM, errno.EROFS):
        hint = " (check the folder is writable and the file is not open in another app)"
    elif exc.errno == errno.ENOSPC:
        hint = " (free some disk space)"
    return "%s: %s%s" % (what, " -> ".join(str(n) for n in names), hint) if names else what + hint


def main(argv: Optional[Sequence[str]] = None) -> int:
    _utf8_stdio()
    env.setup_runtime_env()
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.version:
        print("hook-cutter %s (python %s at %s; venv %s)" % (
            __version__, ".".join(str(x) for x in sys.version_info[:3]), sys.executable,
            "active" if env.in_venv() else "not active"))
        return 0
    if not args.cmd:
        parser.print_help()
        return 0
    try:
        return dispatch(args)
    except env.HCError as exc:
        sys.stdout.flush()
        print("error: %s" % exc.msg, file=sys.stderr)
        return exc.code
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
    except BrokenPipeError:
        return 0
    except OSError as exc:
        # disk full, read-only folder, a file held open by a player: a tool/environment error
        # (exit 3) with the file named, not a traceback
        try:
            sys.stdout.flush()
        except OSError:
            pass
        print("error: %s" % _os_error(exc), file=sys.stderr)
        if env.verbose():
            import traceback
            traceback.print_exc()
        return 3


# ================================================================ test

def cmd_test(args: argparse.Namespace) -> int:
    runner = env.SKILL_ROOT / "tests" / "run.py"
    if not runner.exists():
        raise env.HCError("tests/run.py is missing from this install", code=3)
    py = sys.executable if env.in_venv() or not env.venv_python().exists() else str(env.venv_python())
    cmd = [py, "-s", str(runner)]
    if getattr(args, "slow", False):
        cmd.append("--slow")
    if getattr(args, "pattern", None):
        cmd.append(args.pattern)
    e = dict(os.environ)
    e["PYTHONNOUSERSITE"] = "1"
    e["PYTHONPATH"] = str(env.SCRIPTS_DIR)
    e["PYTHONUTF8"] = "1"
    e["PYTHONPYCACHEPREFIX"] = str(env.cache_dir() / "pyc")
    return subprocess.call(cmd, env=e)


# ================================================================ status helpers

def _sub_args(argv: List[str]) -> argparse.Namespace:
    """Namespace for a sub-step, parsed by the real parser so every attribute exists."""
    return build_parser().parse_args(argv)


def _flag_list(flag: str, values: Optional[List[str]]) -> List[str]:
    return [flag, ",".join(values)] if values else []


def _plan_hash(plan: Dict[str, Any]) -> str:
    return env.stable_hash({k: v for k, v in plan.items() if k not in VOLATILE_PLAN_KEYS})


def _spec_and_eff(proj: Any) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    try:
        return proj.spec(), None
    except env.HCError as exc:
        return None, exc.msg
    except Exception as exc:  # spec module missing or broken: status must still work
        return None, "%s: %s" % (type(exc).__name__, exc)


def _effective(spec: Optional[Dict[str, Any]], raw: Dict[str, Any], cid: str) -> Tuple[List[str], List[str]]:
    if spec:
        c = (spec.get("clips") or {}).get(cid) or {}
        eff = c.get("_eff") or {}
        if eff.get("styles") and eff.get("formats"):
            return list(eff["styles"]), list(eff["formats"])
    d = raw.get("defaults") or {}
    c = (raw.get("clips") or {}).get(cid) or {}
    return (list(c.get("styles") or d.get("styles") or ["clean"]),
            list(c.get("formats") or d.get("formats") or ["vertical"]))


def _worst(statuses: List[str]) -> Optional[str]:
    statuses = [s.split("(")[0].strip().upper() for s in statuses if isinstance(s, str)]
    statuses = [s for s in statuses if s in STATUS_RANK]
    return max(statuses, key=lambda s: STATUS_RANK[s]) if statuses else None


def qa_status(data: Any) -> Optional[str]:
    """Overall status of a qa/<clip>.json: its "status", else the worst nested gate status."""
    if isinstance(data, dict) and isinstance(data.get("status"), str):
        return _worst([data["status"]]) or data["status"]
    found = []

    def walk(o: Any) -> None:
        if isinstance(o, dict):
            if isinstance(o.get("status"), str) and ("id" in o or "name" in o):
                found.append(o["status"])
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
    walk(data)
    return _worst(found)


def qa_has_gaps(rep: Any) -> bool:
    """True when a qa/<clip>.json says some checks could not run (``complete`` false or
    ``gaps`` above 0). A report without those keys counts as complete. The same test holds a
    clip back in ``hc export`` and gives exit 2 under --require-coverage."""
    if not isinstance(rep, dict):
        return False
    try:
        gaps = int(rep.get("gaps") or 0)
    except (TypeError, ValueError):
        gaps = 0
    return rep.get("complete") is False or gaps > 0


def _qa_word(rep: Any) -> str:
    """The QA state shown per clip: the status, plus " (incomplete)" when checks did not run."""
    status = qa_status(rep) or "?"
    if status != "INCOMPLETE" and qa_has_gaps(rep):
        status += " (incomplete)"
    return status


def _mtime(p: Path) -> float:
    try:
        return p.stat().st_mtime
    except OSError:
        return 0.0


def _transcript_changed(plan: Dict[str, Any], transcript: Any) -> bool:
    """True when the transcript the plan was built from is not the transcript now: other
    words or times (the sha), or another speaker label on one of the plan's own words
    (``plan.transcript_changed``, the test ``plan._stale`` uses, so status and render agree).
    A plan or transcript without a recorded sha is never called changed."""
    if not isinstance(transcript, dict) or not transcript.get("sha256") or not plan.get("transcript_sha"):
        return False
    try:
        from . import plan as plan_mod
        fn = getattr(plan_mod, "transcript_changed", None)
        if callable(fn):
            return bool(fn(plan, transcript))
    except Exception:
        pass
    return plan["transcript_sha"] != transcript["sha256"]


def _plan_state(proj: Any, spec: Optional[Dict[str, Any]], cid: str, src: Optional[Dict[str, Any]],
                tr: Optional[Dict[str, Any]], source_changed: bool = False,
                ) -> Tuple[str, Optional[Dict[str, Any]]]:
    """(state text, plan) of one clip. ``tr`` is the project transcript; a caption-only plan
    is compared with its own transcripts/<clip>.json. ``source_changed``: the source file is
    not the one ingest probed, so every cut plan is stale whatever its recorded fingerprint."""
    pp = proj.root / "plans" / ("%s.plan.json" % cid)
    if not pp.exists():
        return "missing", None
    try:
        plan = env.read_json(pp)
    except env.HCError:
        return "unreadable", None
    stale = []
    media_fp = src.get("fingerprint") if src else None
    if plan.get("kind") == "caption_only":
        # a caption-only plan is made from its own file, not the project source
        mp = (plan.get("source") or {}).get("path")
        media_fp = None
        if mp:
            f = Path(str(mp)).expanduser()
            f = f if f.is_absolute() else proj.root / f
            try:
                media_fp = env.fingerprint(f) if f.exists() else "missing"
            except OSError:
                media_fp = "missing"
    caption_only = plan.get("kind") == "caption_only"
    if media_fp and plan.get("source_fingerprint") and plan["source_fingerprint"] != media_fp:
        stale.append("source")
    elif source_changed and not caption_only:
        stale.append("source")
    clip_tr = tr
    if caption_only:
        clip_tr = None
        try:
            if proj.has_transcript(cid):
                clip_tr = proj.transcript(cid)
        except env.HCError:
            clip_tr = None
    if _transcript_changed(plan, clip_tr):
        stale.append("transcript")
    if plan.get("code_version") and plan["code_version"] != __version__:
        stale.append("code")
    if spec is not None:
        try:
            from . import spec as spec_mod
            if plan.get("spec_hash") and plan["spec_hash"] != spec_mod.clip_hash(spec, cid):
                stale.append("spec")
        except Exception:
            pass
    elif _mtime(proj.clips_path()) > _mtime(pp):
        stale.append("spec?")
    if _shots_behind(proj, spec, cid, plan):
        stale.append("shots")
    lints = plan.get("lints") or []
    nf = sum(1 for x in lints if str(x.get("level")).upper() == "FAIL")
    nw = sum(1 for x in lints if str(x.get("level")).upper() == "WARN")
    word = ("STALE(%s)" % ",".join(stale)) if stale else ("FAIL" if nf else "ok")
    extra = []
    if nf:
        extra.append("%dF" % nf)
    if nw:
        extra.append("%dW" % nw)
    if plan.get("duration_out") is not None:
        extra.append("%.1fs" % float(plan["duration_out"]))
    plan["_stale"] = [x.rstrip("?") for x in stale]          # in memory only: read by cmd_status
    return word + (" " + " ".join(extra) if extra else ""), plan


def _shots_sha(proj: Any, cid: str) -> Optional[str]:
    sp = proj.root / "shots" / ("%s.json" % cid)
    try:
        return env.sha256_file(sp) if sp.exists() else None
    except OSError:
        return None


def _shots_behind(proj: Any, spec: Optional[Dict[str, Any]], cid: str, plan: Dict[str, Any]) -> bool:
    """True when the plan file was built from other face data than shots/<clip>.json holds now
    (the file appeared, changed or went away after the plan was written). Framing follows the
    face samples, so such a plan, and every render made from it, is stale."""
    if plan.get("kind", "cut") != "cut" or "shots_hash" not in plan:
        return False
    try:
        from . import plan as plan_mod
        fn = getattr(plan_mod, "shots_stale", None)
        if callable(fn):
            return bool(fn(proj, cid, plan))
    except Exception:
        pass
    return plan.get("shots_hash") != _shots_sha(proj, cid)


def _plan_hash_fn() -> Any:
    """render.plan_hash (the hash a render log records for its plan), or None if unavailable."""
    try:
        from . import render
    except Exception:
        return None
    return getattr(render, "plan_hash", None)


def render_fresh(render_path: Path, plan: Optional[Dict[str, Any]], plan_mtime: float = 0.0) -> bool:
    """True when the render was made from this plan.

    Judged by content: the render log's ``hashes.plan`` against ``render.plan_hash(plan)``, so
    copying a project without preserving times (``cp -R``, sync tools) does not mark every
    render stale. Falls back to modification times only when no log or hash is available."""
    fn = _plan_hash_fn()
    log_path = render_path.with_name(render_path.name[:-len(render_path.suffix)] + ".log.json")
    if plan is not None and fn is not None and log_path.exists():
        try:
            recorded = (env.read_json(log_path).get("hashes") or {}).get("plan")
        except env.HCError:
            recorded = None
        if recorded:
            try:
                return recorded == fn(plan)
            except Exception:
                pass
    return not (plan_mtime and _mtime(render_path) < plan_mtime)


def _iso_time(value: Any) -> Optional[float]:
    """Epoch seconds of an ISO-8601 timestamp ("2026-10-08T14:20:56+00:00" or "...-0500")."""
    if not isinstance(value, str) or not value:
        return None
    from datetime import datetime
    for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S.%f%z"):
        try:
            return datetime.strptime(value.replace("Z", "+0000"), fmt).timestamp()
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(value).timestamp()
    except ValueError:
        return None


def _qa_older_than_renders(qa_path: Path, renders: List[Path]) -> bool:
    """True when a render was made after the QA ran. Uses the "created" stamps written inside
    the render logs and the QA json (they survive a copy), else file modification times."""
    try:
        qa_t = _iso_time(env.read_json(qa_path).get("created"))
    except env.HCError:
        qa_t = None
    for r in renders:
        log_path = r.with_name(r.name[:-len(r.suffix)] + ".log.json")
        r_t = None
        if qa_t is not None and log_path.exists():
            try:
                r_t = _iso_time(env.read_json(log_path).get("created"))
            except env.HCError:
                r_t = None
        if qa_t is not None and r_t is not None:
            if r_t > qa_t:
                return True
        elif _mtime(r) > _mtime(qa_path):
            return True
    return False


def _log_of(render_path: Path) -> Dict[str, Any]:
    """The render log next to a render ({} when missing or unreadable)."""
    log_path = render_path.with_name(render_path.name[:-len(render_path.suffix)] + ".log.json")
    try:
        data = env.read_json(log_path) if log_path.exists() else {}
    except env.HCError:
        data = {}
    return data if isinstance(data, dict) else {}


def _current_plan(proj: Any, cid: str) -> Optional[Dict[str, Any]]:
    """The plan a render made now would use (rebuilt in memory when the plan file is stale,
    never written), or None when it cannot be built."""
    try:
        from . import plan as plan_mod
        return plan_mod.current_plan(proj, cid)
    except Exception:
        return None


def render_stale_reason(proj: Any, cid: str, fmt: str, style: str, render_path: Path,
                        plan: Optional[Dict[str, Any]], plan_mtime: float,
                        cur_plan: Optional[Dict[str, Any]], plan_stale: List[str]) -> Optional[str]:
    """None when the render is the one ``hc render`` would make now, else why it is not.

    "plan": made from another plan than the plan file. Otherwise the input hash its log
    recorded (``hashes.render``: clip entry, preset, segments with their framing tracks, code)
    is compared with ``render.expected_hashes`` for the current plan; on a difference the
    reason is why the plan file is stale ("shots", "spec", ...), or "inputs" (the preset or
    the render code changed) when the plan file is current. A log without that hash, or
    inputs that cannot be computed, never make a render stale."""
    if not render_fresh(render_path, plan, plan_mtime):
        return "plan"
    recorded = (_log_of(render_path).get("hashes") or {}).get("render")
    if not recorded or cur_plan is None:
        return None
    try:
        from . import render
        want = render.expected_hashes(proj, cid, fmt, style, plan=cur_plan)["render"]
    except Exception:
        return None
    if want == recorded:
        return None
    return ",".join(plan_stale) if plan_stale else "inputs"


def _cover_headline(clip: Dict[str, Any], plan: Optional[Dict[str, Any]]) -> str:
    """The headline ``hc cover`` would set: cover.headline, else the hook card, else the title."""
    try:
        from . import cover
        return str(cover.headline_for(clip, plan))
    except Exception:
        c = clip.get("cover") if isinstance(clip.get("cover"), dict) else {}
        if c.get("headline"):
            return str(c["headline"])
        hc = (plan or {}).get("hook_card") or clip.get("hook_card") or {}
        if isinstance(hc, dict) and hc.get("lines"):
            return " ".join(str(x) for x in hc["lines"])
        return str(clip.get("title") or "")


def cover_stale_reasons(proj: Any, clip: Dict[str, Any], raw_clip: Dict[str, Any], cid: str, style: str,
                        plan: Optional[Dict[str, Any]], render_path: Optional[Path],
                        render_stale: bool) -> List[str]:
    """Why covers/<clip>__<style>.jpg is not the cover ``hc cover`` would make now ([] = current).

    Read from the cover's own json against the three things a cover is made from:
    the clip's cover spec ("at_out": the frame it was cut at, or a frame forced with --at
    while clips.json says auto; "headline": the words on it; "shift": an explicit cover.shift),
    the style (each style has its own cover and its own render), and the render it was cut
    from ("render": that render is stale, or was made after the cover). A cover with no json
    cannot be judged and is never called stale."""
    jp = proj.root / "covers" / ("%s__%s.json" % (cid, style))
    try:
        meta = env.read_json(jp) if jp.exists() else None
    except env.HCError:
        meta = None
    if not isinstance(meta, dict):
        return []
    out = []
    cc = clip.get("cover") if isinstance(clip.get("cover"), dict) else {}
    want = cc.get("at_out", "auto")
    if isinstance(want, (int, float)) and not isinstance(want, bool):
        t = float(want)
        try:
            num, den = (int(x) for x in (plan or {}).get("fps") or (0, 0))
            if num > 0 and den > 0:
                # hc cover floors the time to its frame: the same expression as
                # cover_frames.frame_index, then the rounding cover.make_cover stores
                import math
                t = round(int(math.floor(t * num / den + COVER_FRAME_TOL)) * den / num, 6)
        except (TypeError, ValueError):
            pass
        try:
            same = bool(meta.get("manual")) and abs(float(meta.get("at_out")) - t) < 1e-3
        except (TypeError, ValueError):
            same = False
        if not same:
            out.append("at_out")
    elif meta.get("manual"):
        out.append("at_out")
    if meta.get("headline") is not None and str(meta["headline"]) != _cover_headline(clip, plan):
        out.append("headline")
    rc = raw_clip.get("cover") if isinstance(raw_clip.get("cover"), dict) else {}
    if "shift_manual" in meta:
        explicit = rc.get("shift") is not None
        try:
            moved = explicit and int(rc["shift"]) != int(meta.get("shift") or 0)
        except (TypeError, ValueError):
            moved = False
        if explicit != bool(meta.get("shift_manual")) or moved:
            out.append("shift")
    if render_stale:
        out.append("render")
    elif render_path is not None:
        made = _iso_time(meta.get("created"))
        r_t = _iso_time(_log_of(render_path).get("created"))
        if made is not None and r_t is not None:
            if r_t > made:
                out.append("render")
        elif render_path.exists() and _mtime(_log_path(render_path)) > _mtime(jp) > 0:
            out.append("render")
    return out


def _render_mark(render_path: Path) -> Tuple[bool, Any]:
    """(exists, the "created" stamp of its log): changes exactly when the file is rendered again
    (an up-to-date render leaves its log alone)."""
    return render_path.exists(), _log_of(render_path).get("created")


def _log_path(render_path: Path) -> Path:
    return render_path.with_name(render_path.name[:-len(render_path.suffix)] + ".log.json")


def cmd_status(args: argparse.Namespace) -> int:
    from .project import Project
    proj = Project(args.project).require()
    src = None
    drift = False
    try:
        src = proj.source()
        sp = proj.source_path()
        changed = ""
        try:
            if env.fingerprint(sp) != src.get("fingerprint"):
                drift = bool(src.get("fingerprint"))
                changed = "  CHANGED since ingest (re-run: hc ingest %s %s)" % (
                    q(proj.root), q(src.get("original") or "<the video file or URL>"))
        except OSError:
            changed = "  MISSING"
        v = src["video"]
        fps = ("%.3f" % (v["fps"][0] / float(v["fps"][1]))).rstrip("0").rstrip(".")
        print("source      %s  %s  %dx%d %s fps%s" % (src.get("path"), env.fmt_time(src.get("duration"), 0),
                                                     v["width"], v["height"], fps, changed))
    except env.HCError as exc:
        print("source      missing (%s)" % exc.msg)
    tr = None
    if proj.has_transcript():
        try:
            tr = proj.transcript()
            tr_sha = tr.get("sha256")
            print("transcript  %d words  %s %s  sha %s" % (len(tr.get("words") or []), tr.get("backend"),
                                                         tr.get("model"), (tr_sha or "?")[:10]))
        except env.HCError as exc:
            tr = None
            print("transcript  unreadable (%s)" % exc.msg)
    else:
        print("transcript  missing (run: hc transcribe %s)" % q(proj.root))
    if not proj.clips_path().exists():
        print("clips.json  missing (write it after picking moments; see references/spec.md)")
        return 0
    raw = env.read_json(proj.clips_path())
    ids = proj.clip_ids()
    spec, err = _spec_and_eff(proj)
    print("clips.json  %d clips%s" % (len(ids), ("  (spec: %s)" % err) if err else ""))
    if not ids:
        return 0
    w = max(len(c) for c in ids) + 2
    for cid in ids:
        styles, formats = _effective(spec, raw, cid)
        pstate, plan = _plan_state(proj, spec, cid, src, tr, drift)
        pp_m = _mtime(proj.root / "plans" / ("%s.plan.json" % cid))
        plan_stale = list((plan or {}).pop("_stale", []) or [])
        pairs = [(f, s) for f in formats for s in styles]
        renders = {fs: proj.root / "renders" / ("%s__%s__%s.mp4" % (cid, fs[0], fs[1])) for fs in pairs}
        have = [r for r in renders.values() if r.exists()]
        cur_plan = _current_plan(proj, cid) if have and plan is not None else None
        why_r = {}                       # (format, style) -> why that render is stale
        for (f, s), r in renders.items():
            if r.exists():
                why = render_stale_reason(proj, cid, f, s, r, plan, pp_m, cur_plan, plan_stale)
                if why:
                    why_r[(f, s)] = why
                elif drift and (plan or (raw.get("clips") or {}).get(cid) or {}).get("kind") != "caption_only":
                    why_r[(f, s)] = "source"     # cut from the file that was there at ingest
        reasons = sorted({x for why in why_r.values() for x in why.split(",")})
        rstate = "%d/%d%s" % (len(have), len(renders),
                              (" STALE(%s)" % ",".join(reasons)) if why_r else "")
        cstate = "-"
        if "vertical" in formats:
            covers = [proj.root / "covers" / ("%s__%s.jpg" % (cid, s)) for s in styles]
            clip = ((spec or {}).get("clips") or {}).get(cid) or (raw.get("clips") or {}).get(cid) or {}
            raw_clip = (raw.get("clips") or {}).get(cid) or {}
            why_c = []                   # every reason a cover of this clip is stale
            for s, c in zip(styles, covers):
                if c.exists():
                    r = renders.get(("vertical", s))
                    for x in cover_stale_reasons(proj, clip, raw_clip, cid, s, plan,
                                                 r if r is not None and r.exists() else None,
                                                 ("vertical", s) in why_r):
                        if x not in why_c:
                            why_c.append(x)
            cstate = "%d/%d%s" % (sum(1 for c in covers if c.exists()), len(covers),
                                  (" STALE(%s)" % ",".join(why_c)) if why_c else "")
        qp = proj.root / "qa" / ("%s.json" % cid)
        if qp.exists():
            try:
                qs = _qa_word(env.read_json(qp))
            except env.HCError:
                qs = "unreadable"
            if _qa_older_than_renders(qp, have):
                qs += " STALE"
        else:
            qs = "-"
        print("%-*splan %-22s render %-10s cover %-5s qa %s" % (w, cid, pstate, rstate, cstate, qs))
    return 0


# ================================================================ all

def _stamps_path(proj: Any) -> Path:
    return proj.root / "cache" / STAMPS_FILE


def _load_stamps(proj: Any) -> Dict[str, Any]:
    p = _stamps_path(proj)
    if p.exists():
        try:
            data = env.read_json(p)
            if isinstance(data, dict):
                return data
        except env.HCError:
            pass
    return {}


class Stamps(object):
    """cache/all-stamps.json, safe for parallel ``hc all`` runs on one project.

    Every change is recorded per (section, clip). ``save`` takes the project's stamps lock,
    re-reads the file, applies only this run's changes and writes it back, so a run for c02
    never drops the c01 entries another run wrote in the meantime."""

    def __init__(self, proj: Any):
        self.proj = proj
        self.data = _load_stamps(proj)
        for sec in STAMP_SECTIONS:
            if not isinstance(self.data.get(sec), dict):
                self.data[sec] = {}
        self.changes = {}  # type: Dict[Tuple[str, str], Optional[Dict[str, Any]]]

    def get(self, section: str, cid: str) -> Optional[Dict[str, Any]]:
        return self.data[section].get(cid)

    def set(self, section: str, cid: str, entry: Dict[str, Any]) -> None:
        self.data[section][cid] = entry
        self.changes[(section, cid)] = entry

    def drop(self, section: str, cid: str) -> None:
        self.data[section].pop(cid, None)
        self.changes[(section, cid)] = None

    def save(self) -> None:
        from . import fsutil
        path = _stamps_path(self.proj)
        with fsutil.file_lock(path.with_name(STAMPS_LOCK), what="the project stamps file"):
            disk = _load_stamps(self.proj)
            for sec in STAMP_SECTIONS:
                if not isinstance(disk.get(sec), dict):
                    disk[sec] = {}
            for (sec, cid), entry in self.changes.items():
                if entry is None:
                    disk[sec].pop(cid, None)
                else:
                    disk[sec][cid] = entry
            env.write_json(path, disk)
        self.data = disk
        self.changes = {}


def _covers_current(proj: Any, spec: Any, raw: Dict[str, Any], cid: str, styles: List[str],
                    paths: List[Path]) -> bool:
    """True when every cover file of the clip exists and none is stale by the same test
    `hc status` uses (cover_stale_reasons against the cover spec, the style and the render)."""
    if not paths or not all(Path(p).exists() for p in paths):
        return False
    try:
        plan = env.read_json(proj.plan_path(cid))
    except env.HCError:
        return False
    clip = ((spec or {}).get("clips") or {}).get(cid) or (raw.get("clips") or {}).get(cid) or {}
    raw_clip = (raw.get("clips") or {}).get(cid) or {}
    for st in styles:
        r = Path(proj.render_path(cid, "vertical", st))
        if not r.exists() or not (proj.root / "covers" / ("%s__%s.json" % (cid, st))).exists():
            return False
        if cover_stale_reasons(proj, clip, raw_clip, cid, st, plan, r, False):
            return False
    return True


def _outputs_ok(proj: Any, entry: Optional[Dict[str, Any]], key: str) -> bool:
    if not entry or entry.get("in") != key:
        return False
    for rel, fp in (entry.get("out") or {}).items():
        p = proj.root / rel
        if not p.exists():
            return False
        try:
            if env.fingerprint(p) != fp:
                return False
        except OSError:
            return False
    return bool(entry.get("out"))


def _stamp(proj: Any, key: str, paths: List[Path]) -> Dict[str, Any]:
    out = {}
    for p in paths:
        if p.exists():
            out[proj.rel(p)] = env.fingerprint(p)
    return {"in": key, "out": out}


def _preset_hashes(proj: Any, styles: List[str]) -> Dict[str, str]:
    from . import presets
    out = {}
    for s in styles:
        try:
            out[s] = env.stable_hash(presets.load(s, proj))
        except env.HCError:
            out[s] = "missing"
    return out


def _render_code_hash() -> str:
    """The render module's own code hash (render, framing, captions, typeset sources), so `all`
    re-renders after a code change exactly when `render` itself would."""
    try:
        from . import render
        return render._code_hash(render.CODE_FILES + render.CAPTION_FILES)
    except Exception:
        return "unknown"


def _asset_hashes(proj: Any, clip: Dict[str, Any], styles: List[str], formats: List[str]) -> Any:
    """The files outside clips.json and the preset json that a render's pixels depend on, per
    format and style: the bytes of the caption and hook-card fonts and the safe-zone margins
    in force (``captions.render_assets``, the value ``hc render`` hashes into its own key).
    "unknown" when it cannot be computed: `all` then follows its other inputs and render
    itself decides."""
    try:
        from . import captions
        from . import render_util as ru
        fn = getattr(captions, "render_assets", None)
        if not callable(fn):
            return "unknown"
        return dict((f, dict((s, fn(ru.load_preset(s, proj, f), clip, f, proj.root)) for s in styles))
                    for f in formats)
    except Exception:
        return "unknown"


def _cover_font_hashes(proj: Any, styles: List[str]) -> Any:
    """sha256 of the font files each style's cover is set in (cover.headline_font and
    cover.lockup_font, the built-in defaults included), so `all` re-makes the covers after a
    font file is replaced. "unknown" when it cannot be computed."""
    try:
        from . import presets, typeset
        try:
            from . import cover as cover_mod
            defaults = dict(getattr(cover_mod, "DEFAULT_COVER", {}) or {})
        except Exception:
            defaults = {}
        out = {}
        for s in styles:
            cfg = dict(defaults)
            cfg.update(presets.load(s, proj).get("cover") or {})
            fonts = {}
            for k in ("headline_font", "lockup_font"):
                if not cfg.get(k):
                    continue
                try:
                    fonts[str(cfg[k])] = env.sha256_file(typeset.resolve_font(str(cfg[k]), proj.root)[0])
                except (env.HCError, OSError):
                    fonts[str(cfg[k])] = "missing"
            out[s] = fonts
        return out
    except Exception:
        return "unknown"


def _source_facts(src: Any) -> str:
    """Hash of what ingest measured about the source's sound and picture (channel layout and
    levels, stream starts, pixel shape, colour, frame facts). `hc render` hashes the ones it
    uses into every segment key; `all` takes them all, so a re-ingest that changes any of them
    for the same file sends the clip to render, which then decides for itself."""
    src = src if isinstance(src, dict) else {}
    return env.stable_hash({"audio": src.get("audio"), "video": src.get("video")})


def _qa_exit_from_reports(proj: Any, ids: List[str], require_coverage: bool) -> int:
    """The exit code `qa` gave for these clips, read back from their qa/<clip>.json: 1 for a
    FAIL, else 2 under --require-coverage when a report is INCOMPLETE or has evidence gaps
    (a WARN clip whose checks did not all run), as qa.exit_code does."""
    worst = 0
    for cid in ids:
        qp = proj.qa_path(cid, ".json")
        try:
            rep = env.read_json(qp) if qp.exists() else None
        except env.HCError:
            rep = None
        status = qa_status(rep) if rep is not None else None
        if status == "FAIL":
            worst = worst_exit(worst, 1)
        elif require_coverage and (status == "INCOMPLETE" or qa_has_gaps(rep)):
            worst = worst_exit(worst, 2)
    return worst


def cmd_all(args: argparse.Namespace) -> int:
    """plan, render, cover, qa, export; skips clips whose inputs are unchanged.

    ``--styles``/``--formats`` reach every step. QA runs with --require-coverage unless
    --no-asr. Export runs after a QA FAIL too (it holds the failing clips back), and the exit
    code is the worst of the steps."""
    from .project import Project
    proj = Project(args.project).require()
    ids = list(args.clips or proj.clip_ids())
    if not ids:
        raise env.HCError("no clips: write %s first" % proj.clips_path(), code=1)
    known = proj.clip_ids()
    for cid in ids:
        if cid not in known:
            raise env.unknown_clip(cid, known)
    P = str(proj.root)
    sty, fmt = list(args.styles or []), list(args.formats or [])
    force = bool(getattr(args, "force", False))
    no_asr = bool(getattr(args, "no_asr", False))
    require_coverage = not no_asr

    env.log("all", "plan")
    rc = dispatch(_sub_args(["plan", P] + ids + ["--quiet"]))
    if rc != 0:
        env.log("all", "plan reported FAIL lints: fix clips.json, then rerun")
        return rc
    spec, _ = _spec_and_eff(proj)
    raw = env.read_json(proj.clips_path())
    src = proj.source()
    stamps = Stamps(proj)

    def render_key(cid: str, styles: List[str], formats: List[str]) -> str:
        """Everything a clip's renders are made from, the plan file as it is on disk included."""
        plan = env.read_json(proj.plan_path(cid))
        entry = (raw.get("clips") or {}).get(cid) or {}
        media = None
        if entry.get("kind") == "caption_only" and entry.get("source_file"):
            sf = Path(str(entry["source_file"])).expanduser()
            sf = sf if sf.is_absolute() else proj.root / sf
            media = env.fingerprint(sf) if sf.exists() else "missing"
        clip = ((spec or {}).get("clips") or {}).get(cid) or entry
        return env.stable_hash({"plan": _plan_hash(plan), "clip": entry, "defaults": raw.get("defaults"),
                                "speakers": raw.get("speakers"),
                                "styles": styles, "formats": formats, "media": media,
                                "presets": _preset_hashes(proj, styles), "source": src.get("fingerprint"),
                                "source_facts": _source_facts(src),
                                "assets": _asset_hashes(proj, clip, styles, formats),
                                "code": __version__, "render_code": _render_code_hash()})

    per = {}  # cid -> (styles, formats, render input key, render paths)
    for cid in ids:
        styles, formats = _effective(spec, raw, cid)
        styles, formats = (sty or styles), (fmt or formats)
        paths = [proj.render_path(cid, f, s) for f in formats for s in styles]
        per[cid] = (styles, formats, render_key(cid, styles, formats), paths)

    did = dict((c, []) for c in ids)   # cid -> what this run did for it, in order (one line each)
    style_flags = _flag_list("--styles", sty)
    user_flags = style_flags + _flag_list("--formats", fmt)
    todo = [c for c in ids if force or not _outputs_ok(proj, stamps.get("render", c), per[c][2])]
    skipped = [c for c in ids if c not in todo]
    if skipped:
        env.log("all", "render up to date: %s" % ", ".join(skipped))
    for cid in skipped:
        did[cid].append("render skipped (up to date)")
    for cid in todo:
        styles, formats, key, paths = per[cid]
        env.log("all", "render %s" % cid)
        shots_before = _shots_sha(proj, cid)
        made_before = dict((str(x), _render_mark(Path(x))) for x in paths)
        rc = dispatch(_sub_args(["render", P, cid] + user_flags + (["--force"] if force else [])))
        if rc != 0:
            return rc
        if _shots_sha(proj, cid) != shots_before:
            # render analysed the shots itself (the framing needed face data) and re-planned
            # with them: say so, and stamp the plan as it is now so the next run skips this clip
            did[cid].append("shots analysed by render (the framing needed face data: look at %s)"
                            % proj.rel(proj.sheet_path("shots_%s.jpg" % cid)))
            try:
                key = render_key(cid, styles, formats)
                per[cid] = (styles, formats, key, paths)
            except env.HCError:
                pass
        made = [x for x in paths if Path(x).exists() and _render_mark(Path(x)) != made_before[str(x)]]
        kept = [x for x in paths if Path(x).exists() and x not in made]
        did[cid].append("; ".join(([("rendered %d file(s)" % len(made))] if made else []) +
                                  ([("%d render(s) already up to date" % len(kept))] if kept else []))
                        or "nothing rendered")
        stamps.set("render", cid, _stamp(proj, key, paths))
        stamps.drop("cover", cid)
        stamps.drop("qa", cid)
        stamps.save()

    # QA runs for every clip whenever any clip is stale, so qa/summary.md always describes the
    # whole set; covers are made only for stale clips and the grid is rebuilt from all of them.
    ckeys = {}
    for cid in ids:
        styles, formats, rkey, paths = per[cid]
        if "vertical" not in formats:
            continue
        ckeys[cid] = (env.stable_hash({"render": (stamps.get("render", cid) or {}).get("out"),
                                       "cover": ((raw.get("clips") or {}).get(cid) or {}).get("cover"),
                                       "styles": styles, "fonts": _cover_font_hashes(proj, styles)}),
                      [proj.cover_path(cid, s, k) for s in styles for k in ("main", "grid", "mobile", "json")])
    for cid in ids:
        if cid not in ckeys:
            did[cid].append("no cover (no vertical format)")
    # Only stale covers are made again: a cover takes minutes on a long 4K source, and a remade
    # one is never byte-identical to the one somebody already reviewed. A cover with no stamp
    # (made by `hc cover`, or copied in from a clone) is kept when `hc status` would call it
    # current (a stamped cover whose inputs changed, such as a replaced font, is made again).
    # The grid is rebuilt whenever a cover changed or it is missing.
    stale = []
    for cid, (ckey, cpaths) in ckeys.items():
        entry = stamps.get("cover", cid)
        if force or not _outputs_ok(proj, entry, ckey):
            inputs_same = entry is None or entry.get("in") == ckey
            if not force and inputs_same and _covers_current(proj, spec, raw, cid, per[cid][0], cpaths):
                stamps.set("cover", cid, _stamp(proj, ckey, cpaths))
                did[cid].append("cover kept (current, made outside hc all)")
            else:
                stale.append(cid)
        else:
            did[cid].append("cover skipped (up to date)")
    if stale:
        env.log("all", "cover %s" % ", ".join(stale))
        rc = dispatch(_sub_args(["cover", P] + stale + style_flags))
        if rc != 0:
            return rc
        for cid in stale:
            stamps.set("cover", cid, _stamp(proj, ckeys[cid][0], ckeys[cid][1]))
            did[cid].append("cover made")
    elif ckeys:
        env.log("all", "covers up to date")
        if not Path(proj.sheet_path("covers-grid.jpg")).is_file():
            from . import cover as cover_mod
            for line in cover_mod.refresh_grids(proj):
                print(line)
    if ckeys:
        stamps.save()

    qkeys = {}
    try:
        qa_code = env.package_digest()      # QA re-runs when any of the code behind the checks changes
    except Exception:
        qa_code = "unknown"
    for cid in ids:
        qkeys[cid] = env.stable_hash({"render": (stamps.get("render", cid) or {}).get("out"),
                                      "cover": (stamps.get("cover", cid) or {}).get("out"),
                                      "plan": per[cid][2], "no_asr": no_asr,
                                      "styles": per[cid][0], "formats": per[cid][1],
                                      "require_coverage": require_coverage, "qa_code": qa_code})
    worst = 0
    if force or any(not _outputs_ok(proj, stamps.get("qa", c), qkeys[c]) for c in ids):
        env.log("all", "qa")
        rc = dispatch(_sub_args(["qa", P] + ids + user_flags
                                + (["--no-asr"] if no_asr else ["--require-coverage"])))
        if rc == 3:
            return rc
        worst = rc
        for cid in ids:
            stamps.set("qa", cid, _stamp(proj, qkeys[cid], [proj.qa_path(cid, ".json")]))
        stamps.save()
        ran_qa = True
    else:
        worst = _qa_exit_from_reports(proj, ids, require_coverage)
        env.log("all", "qa up to date%s" % (" (FAIL or gaps recorded: see qa/summary.md)"
                                            if worst else ""))
        ran_qa = False
    for cid in ids:
        try:
            qp = proj.qa_path(cid, ".json")
            status = _qa_word(env.read_json(qp)) if qp.exists() else "?"
        except env.HCError:
            status = "?"
        did[cid].append("qa %s%s" % (status, "" if ran_qa else " (skipped, up to date)"))
        env.log("all", "%s: %s" % (cid, "; ".join(did[cid])))

    env.log("all", "export%s" % (" (clips that did not pass QA are held back)" if worst else ""))
    rc = dispatch(_sub_args(["export", P] + ids + user_flags))
    return worst_exit(worst, rc)


if __name__ == "__main__":
    sys.exit(main())
