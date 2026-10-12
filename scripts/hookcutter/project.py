"""Project directory model (contract section 5) and path helpers.

Every path helper creates the parent directory on demand, so callers can write
straight to the returned path.
"""

from __future__ import annotations

import os
import shlex
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import env

MEDIA_EXTS = (".mp4", ".mov", ".m4v", ".mkv", ".webm", ".avi", ".mts", ".m2ts", ".ts",
              ".mxf", ".flv", ".wmv", ".mpg", ".mpeg", ".3gp", ".video")
COVER_KINDS = {"main": ".jpg", "grid": ".grid.jpg", "mobile": ".mobile.jpg", "json": ".json"}
TEMP_ROOTS = ("/tmp/", "/private/tmp/", "/var/tmp/")
_LIVE_FINGERPRINTS = {}  # type: Dict[tuple, str]   # (real path, size, mtime ns) -> fingerprint


def root_of(project: Any) -> Optional[Path]:
    """The folder of a Project, or of a plain path (str / PathLike). None stays None.

    Never use ``getattr(x, "root", x)``: a ``pathlib.Path`` has its own ``.root`` ("/")."""
    if project is None:
        return None
    if isinstance(project, Project):
        return project.root
    if isinstance(project, (str, os.PathLike)):
        return Path(project).expanduser().absolute()
    root = getattr(project, "root", None)
    return Path(root) if root is not None else None


def _norm_dir(path: Any) -> str:
    """Resolved, case-normalized folder path with a trailing separator (for prefix tests)."""
    try:
        p = os.path.realpath(os.path.expanduser(str(path)))
    except (OSError, ValueError):
        p = os.path.abspath(str(path))
    p = os.path.normcase(p)
    if sys.platform == "darwin":
        p = p.lower()           # APFS and HFS+ are case-insensitive by default
    return p.rstrip("/\\") + os.sep


def temp_roots() -> List[str]:
    """Normalized temp folder prefixes for this machine."""
    cands = list(TEMP_ROOTS)
    for var in ("TMPDIR", "TEMP", "TMP"):
        if os.environ.get(var):
            cands.append(os.environ[var])
    try:
        cands.append(tempfile.gettempdir())
    except (OSError, IndexError):
        pass
    out = []
    for c in cands:
        n = _norm_dir(c)
        if len(n) > 1 and n not in out:
            out.append(n)
    return out


def _ext(ext: str) -> str:
    ext = str(ext)
    return ext if ext.startswith(".") else "." + ext


class Project(object):
    """A hook-cutter project folder P (created by ``ingest``)."""

    def __init__(self, root: Any, *, create: bool = False):
        self.root = Path(root).expanduser().absolute()
        if create:
            self.root.mkdir(parents=True, exist_ok=True)
        self._source_cache = None  # type: Optional[tuple]

    def __repr__(self) -> str:
        return "Project(%r)" % str(self.root)

    def __fspath__(self) -> str:
        return str(self.root)

    # ------------------------------------------------------------ basics

    def exists(self) -> bool:
        return self.root.is_dir()

    def require(self) -> "Project":
        if not self.root.is_dir():
            raise env.HCError("project folder not found: %s (create it with: hc ingest %s SOURCE)"
                              % (self.root, shlex.quote(str(self.root))), code=1)
        return self

    def is_temp(self) -> bool:
        """True when P sits in a temporary folder (cleared by the OS): /tmp, /var/tmp, the
        system temp dir (macOS $TMPDIR under /var/folders, Windows %TEMP%) or TMPDIR/TEMP/TMP."""
        mine = _norm_dir(self.root)
        return any(mine.startswith(t) for t in temp_roots())

    def _file(self, *parts: str) -> Path:
        p = self.root.joinpath(*parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def _dir(self, *parts: str) -> Path:
        p = self.root.joinpath(*parts)
        p.mkdir(parents=True, exist_ok=True)
        return p

    # ------------------------------------------------------------ source

    def source_json_path(self) -> Path:
        return self.root / "source.json"

    def source_candidates(self) -> List[Path]:
        """Every ``source.<ext>`` media entry in P (symlinks included, json excluded)."""
        if not self.root.is_dir():
            return []
        out = []
        for p in sorted(self.root.iterdir()):
            if not p.name.startswith("source."):
                continue
            if p.name.endswith((".json", ".part", ".ytdl", ".tmp")):
                continue
            if p.is_symlink() or p.is_file():
                out.append(p)
        return out

    def source_path(self) -> Path:
        """Path of the source media (``P/source.<ext>``)."""
        sj = self.source_json_path()
        if sj.exists():
            try:
                name = env.read_json(sj).get("path")
            except env.HCError:
                name = None
            if name:
                p = self.root / name
                if p.exists():
                    return p
                if p.is_symlink():
                    raise env.HCError("%s is a broken link (the original file moved?). Re-run: "
                                      "hc ingest %s <file> --force" % (p, shlex.quote(str(self.root))), code=1)
        cands = self.source_candidates()
        if not cands:
            raise env.HCError("no source in %s. Run: hc ingest %s <video or URL>"
                              % (self.root, shlex.quote(str(self.root))), code=1)
        for p in cands:
            if not p.exists():
                raise env.HCError("%s is a broken link (the original file moved?). Re-run: "
                                  "hc ingest %s <file> --force" % (p, shlex.quote(str(self.root))), code=1)
        return cands[0]

    def source(self) -> Dict[str, Any]:
        """source.json as a dict (cached until the file changes)."""
        sj = self.source_json_path()
        try:
            st = sj.stat()
        except OSError:
            raise env.HCError("no source.json in %s. Run: hc ingest %s <video or URL>"
                              % (self.root, shlex.quote(str(self.root))), code=1)
        key = (st.st_mtime_ns, st.st_size)
        if self._source_cache and self._source_cache[0] == key:
            return self._source_cache[1]
        data = env.read_json(sj)
        self._source_cache = (key, data)
        return data

    def source_drift(self) -> Optional[str]:
        """None when the source file is still the one ``ingest`` probed, else one sentence
        saying it changed and how to re-ingest it.

        source.json holds the fingerprint, duration, stream layout and picture facts of the
        file as it was at ingest; the transcript, plans, cached segments and renders were all
        made from that file. A file replaced at the same path (a re-export over a linked
        original) keeps the name and invalidates all of them. Nothing recorded to compare, no
        source, or a broken link give None: those have their own errors."""
        try:
            src = self.source()
        except env.HCError:
            return None
        have = src.get("fingerprint") if isinstance(src, dict) else None
        if not have:
            return None
        try:
            p = self.source_path()
        except env.HCError:
            return None
        try:
            st = os.stat(str(p))
            key = (os.path.realpath(str(p)), st.st_size, st.st_mtime_ns)
            live = _LIVE_FINGERPRINTS.get(key)
            if live is None:
                live = env.fingerprint(p)
                _LIVE_FINGERPRINTS[key] = live
        except OSError:
            return None
        if live == have:
            return None
        again = src.get("original") or "<the video file or URL>"
        return ("the source file changed since ingest: %s is not the file source.json, the "
                "transcript, the plans and the renders were made from. Run: hc ingest %s %s "
                "(then hc transcribe when the sound changed)"
                % (p.name, shlex.quote(str(self.root)), shlex.quote(str(again))))

    def require_source_current(self) -> "Project":
        """Stop (exit 1) when the source file changed since ingest; see ``source_drift``."""
        why = self.source_drift()
        if why:
            raise env.HCError(why, code=1)
        return self

    def signals_path(self) -> Path:
        return self.root / "signals.json"

    def signals(self) -> Optional[Dict[str, Any]]:
        p = self.signals_path()
        return env.read_json(p) if p.exists() else None

    # ------------------------------------------------------------ transcript

    def _raw_clip(self, clip: str) -> Optional[Dict[str, Any]]:
        p = self.clips_path()
        if not p.exists():
            return None
        try:
            data = env.read_json(p)
        except env.HCError:
            return None
        clips = data.get("clips") if isinstance(data, dict) else None
        if isinstance(clips, dict):
            c = clips.get(clip)
            return c if isinstance(c, dict) else None
        return None

    def clip_kind(self, clip: str) -> Optional[str]:
        c = self._raw_clip(clip)
        if c is None:
            return None
        return c.get("kind") or "cut"

    def transcripts_dir(self) -> Path:
        return self._dir("transcripts")

    def transcript_path(self, clip: Optional[str] = None) -> Path:
        """transcript.json, or transcripts/<clip>.json for caption-only clips."""
        if clip:
            per_clip = self.root / "transcripts" / ("%s.json" % clip)
            kind = self.clip_kind(clip)
            if kind == "caption_only":
                return per_clip
            if kind is None and per_clip.exists():
                return per_clip
        return self.root / "transcript.json"

    def transcript_md_path(self, clip: Optional[str] = None) -> Path:
        return self.transcript_path(clip).with_suffix(".md")

    def transcript(self, clip: Optional[str] = None) -> Dict[str, Any]:
        p = self.transcript_path(clip)
        if not p.exists():
            if clip and p.parent.name == "transcripts":
                raise env.HCError("no transcript for caption-only clip %s. Run: hc transcribe %s "
                                  "--file <video> --clip %s" % (clip, shlex.quote(str(self.root)), clip), code=1)
            raise env.HCError("no transcript in %s. Run: hc transcribe %s"
                              % (self.root, shlex.quote(str(self.root))), code=1)
        return env.read_json(p)

    def has_transcript(self, clip: Optional[str] = None) -> bool:
        return self.transcript_path(clip).exists()

    # ------------------------------------------------------------ spec

    def clips_path(self) -> Path:
        return self.root / "clips.json"

    def moments_path(self) -> Path:
        return self.root / "moments.md"

    def glossary_path(self) -> Path:
        return self.root / "glossary.txt"

    def spec(self) -> Dict[str, Any]:
        """clips.json normalized by spec.load_spec (B3)."""
        from . import spec as spec_mod  # lazy: owned by another module
        return spec_mod.load_spec(self)

    def clip_ids(self) -> List[str]:
        """Clip ids in clips.json order (raw read; no validation). Keys that start with "_"
        are comments, as everywhere in clips.json, and are not clips."""
        p = self.clips_path()
        if not p.exists():
            return []
        data = env.read_json(p)
        clips = data.get("clips") if isinstance(data, dict) else None
        if not isinstance(clips, dict):
            return []
        return [k for k in clips.keys() if not (isinstance(k, str) and k.startswith("_"))]

    # ------------------------------------------------------------ outputs

    def plan_path(self, c: str) -> Path:
        return self._file("plans", "%s.plan.json" % c)

    def shots_path(self, c: str) -> Path:
        return self._file("shots", "%s.json" % c)

    def work_dir(self, c: str, fmt: str) -> Path:
        return self._dir("work", c, fmt)

    def render_path(self, c: str, fmt: str, style: str, ext: str = ".mp4") -> Path:
        return self._file("renders", "%s__%s__%s%s" % (c, fmt, style, _ext(ext)))

    def cover_path(self, c: str, style: str, kind: str = "main") -> Path:
        if kind not in COVER_KINDS:
            raise env.HCError("unknown cover kind %r (use %s)" % (kind, ", ".join(COVER_KINDS)),
                              code=1)
        return self._file("covers", "%s__%s%s" % (c, style, COVER_KINDS[kind]))

    def sheet_path(self, name: str) -> Path:
        if not os.path.splitext(name)[1]:
            name += ".jpg"
        return self._file("sheets", name)

    def qa_path(self, c: str, ext: str = ".json") -> Path:
        return self._file("qa", "%s%s" % (c, _ext(ext)))

    def cache_path(self, name: str) -> Path:
        return self._file("cache", name)

    def cache_dir(self) -> Path:
        return self._dir("cache")

    def export_dir(self) -> Path:
        return self._dir("export")

    def presets_dir(self) -> Path:
        return self.root / "presets"

    def fonts_dir(self) -> Path:
        return self.root / "fonts"

    def rel(self, path: Any) -> str:
        """Path relative to P when inside it, else absolute (for messages and manifests)."""
        p = Path(path)
        try:
            return str(p.absolute().relative_to(self.root))
        except ValueError:
            return str(p)
