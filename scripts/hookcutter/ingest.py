"""``ingest``: bring a local file or URL into a project, probe it, draw the survey sheet.

Local files are symlinked by default (``--copy`` copies). URLs download through
yt-dlp from the skill venv (``hc doctor --install --with url``) with an SDR-first,
H.264-leaning format sort, and YouTube's "most replayed" heatmap and chapters are
saved to ``signals.json`` on the src clock.

yt-dlp format sort (verified against yt-dlp 2026.08 ``FormatSorter``): fields are
compared left to right; ``field:value`` prefers the best value no better than
``value``. ``hdr:SDR`` therefore picks SDR whenever it exists (yt-dlp's own default
prefers HDR), ``res:2160`` then takes the tallest picture up to 2160p (a 9:16 crop
needs height), and ``vcodec:h264`` breaks ties toward H.264 (then VP9 before AV1),
which every ffmpeg decodes. ``acodec:aac`` prefers AAC audio. Override with the
environment variable ``HOOKCUTTER_YTDLP_SORT``.

The format selector ``bv*[format_id!$=-sr]+ba/b`` skips YouTube's AI-upscaled ("-sr",
super-resolution) formats: their faces are model-generated pixels and they hide the true
resolution. Override with ``HOOKCUTTER_YTDLP_FORMAT``. yt-dlp runs quiet (progress every 5 s,
no ffmpeg banner), so signed media URLs never reach the console.

Ingest also samples the survey frames for black bars: a letterboxed or gallery (tiled) source
gets ``video.active_area`` in source.json, and the upscale warning uses that picture height.

Replacing a source (``--force``, ``--cfr``) never deletes first: the new file is checked,
then put in place by rename, source.json is written, and only then is the one file the old
source.json named removed. Other ``source.*`` files in P are never touched, and a file that
was ingested in place (the user's original) is never removed. One ingest runs per project at
a time (``cache/ingest.lock``).
"""

from __future__ import annotations

import importlib.util
import os
import re
import shlex
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from . import env, probe

YTDLP_SORT = "hdr:SDR,res:2160,vcodec:h264,acodec:aac"
YTDLP_FORMAT = "bv*[format_id!$=-sr]+ba/b"
YTDLP_QUIET = ["--quiet", "--no-warnings", "--progress"]
YTDLP_SECTION_FF_ARGS = "ffmpeg:-hide_banner -loglevel error -nostats"
CFR_AUDIO_MP4 = ("aac", "mp3", "ac3", "eac3", "alac")
YTDLP_MERGE = "mp4/mkv"
JS_RUNTIMES = ("deno", "node", "bun", "qjs")
SURVEY_FRAMES = 16
SURVEY_COLS = 4
URL_RE = re.compile(r"^(https?|ftp)://", re.I)


def is_url(s: str) -> bool:
    return bool(URL_RE.match(str(s).strip()))


# ---------------------------------------------------------------- helpers

def parse_section(spec: str) -> Tuple[float, Optional[float]]:
    """``A-B`` → (a, b) seconds; ``B`` may be ``inf``. Times as in parse_time."""
    m = re.match(r"^\s*([0-9:.]+)\s*-\s*([0-9:.]+|inf)\s*$", str(spec or ""))
    if not m:
        raise env.HCError("invalid --section %r (use A-B, e.g. 10:00-25:30 or 600-inf)" % spec, code=1)
    a = env.parse_time(m.group(1))
    b = None if m.group(2) == "inf" else env.parse_time(m.group(2))
    if b is not None and b <= a:
        raise env.HCError("invalid --section %r: end must be after start" % spec, code=1)
    return a, b


def _ytdlp_available() -> bool:
    return importlib.util.find_spec("yt_dlp") is not None


def js_runtime() -> Optional[str]:
    for name in JS_RUNTIMES:
        if shutil.which(name):
            return name
    return None


def _ytdlp_help(py: str) -> str:
    try:
        return subprocess.run([py, "-m", "yt_dlp", "--help"], stdout=subprocess.PIPE,
                              stderr=subprocess.DEVNULL, universal_newlines=True, timeout=60).stdout
    except (OSError, subprocess.TimeoutExpired):
        return ""


def _signals(info: Dict[str, Any], offset: float, duration: float, approx: bool) -> Dict[str, Any]:
    def shift(items: Any, keep: Tuple[str, ...]) -> List[Dict[str, Any]]:
        out = []
        for it in items or []:
            try:
                s = float(it.get("start_time")) - offset
                e = float(it.get("end_time")) - offset
            except (TypeError, ValueError):
                continue
            if e <= 0 or s >= duration:
                continue
            row = {"start_src": round(max(0.0, s), 3), "end_src": round(min(duration, e), 3)}
            for k in keep:
                row[k] = it.get(k)
            out.append(row)
        return out
    return {
        "heatmap": shift(info.get("heatmap"), ("value",)),
        "chapters": shift(info.get("chapters"), ("title",)),
        "section_offset": round(offset, 3),
        "section_offset_exact": not approx,
        "source": "yt-dlp",
        "video_duration": info.get("duration"),
    }


# ---------------------------------------------------------------- local / url

def _in_place_original(proj: Any, old: Optional[Dict[str, Any]]) -> Optional[Path]:
    """The file in P that IS the user's original (ingested in place, not a link or a copy
    hook-cutter made), per the old source.json; None when there is none."""
    if not isinstance(old, dict) or not old.get("path"):
        return None
    p = proj.root / Path(str(old["path"])).name
    if p.is_symlink() or not p.is_file():
        return None
    if (old.get("ingest") or {}).get("method") == "in-place":
        return p
    try:
        if old.get("original") and not is_url(str(old["original"])) \
                and p.resolve() == Path(str(old["original"])).expanduser().resolve():
            return p
    except OSError:
        pass
    return None


def _foreign_file(proj: Any, old: Optional[Dict[str, Any]], dest: Path, src: Path) -> bool:
    """True when P/source.<ext> is a regular file that may be the user's own, so putting
    ``src`` there would destroy it. Not foreign: a link, the file being ingested itself
    (in place), the copy or download the old source.json recorded, and a copy with the same
    content as ``src`` (an earlier ingest of this file that did not finish)."""
    if dest.is_symlink() or not dest.is_file():
        return False
    try:
        if dest.resolve() == src.resolve():
            return False
        recorded = isinstance(old, dict) and Path(str(old.get("path") or "")).name == dest.name
        if recorded and _in_place_original(proj, old) is None:
            return False
        if dest.stat().st_size == src.stat().st_size and env.fingerprint(dest) == env.fingerprint(src):
            return False
    except OSError:
        pass
    return True


def _drop_old_source(proj: Any, old: Optional[Dict[str, Any]], keep: Any) -> None:
    """After the new source is in place and source.json names it: remove the ONE file the old
    source.json named. Never any other ``source.*`` file in P (notes, subtitles, backups),
    never the new source, and never a file that was ingested in place (the user's original)."""
    if not isinstance(old, dict) or not old.get("path"):
        return
    name = Path(str(old["path"])).name
    if name == Path(str(keep)).name or not name.startswith("source."):
        return
    p = proj.root / name
    if not (p.is_symlink() or p.is_file()):
        return
    if _in_place_original(proj, old) is not None:
        env.log("ingest", "kept %s: it is the original file itself (ingested in place), not a "
                          "link or a copy" % p.name)
        return
    p.unlink()


def _preflight(src: Path) -> None:
    """Refuse a file that is not a video with sound BEFORE anything in the project changes
    (a header read; the full probe runs once the file is in place)."""
    info = probe.probe_media(src)
    if info.get("video") is None:
        raise env.HCError("%s has no video stream" % src.name, code=1)
    if info.get("audio") is None:
        raise env.HCError("%s has no audio stream: hook-cutter cuts on speech, so the source needs "
                          "audio" % src.name, code=1)


def _tmp_beside(dest: Path) -> Path:
    """A private temp name next to ``dest`` (per process and thread). It ends in ``.tmp``, so
    Project.source_candidates() never takes it for a source."""
    return dest.with_name("%s.%d.%d.tmp" % (dest.name, os.getpid(), threading.get_ident()))


def _unlink_quiet(path: Path) -> None:
    try:
        if os.path.lexists(str(path)):
            os.unlink(str(path))
    except OSError:
        pass


def _copy_into(src: Path, dest: Path) -> None:
    """Copy ``src`` to ``dest`` through a private temp name and a rename: nothing is written
    through a link that sits at ``dest`` or at a leftover temp name, ``dest`` is whole or
    untouched, and a failed copy leaves no temp file."""
    tmp = _tmp_beside(dest)
    try:
        if os.path.lexists(str(tmp)):
            os.unlink(str(tmp))
        shutil.copy2(str(src), str(tmp))
        os.replace(str(tmp), str(dest))
    finally:
        _unlink_quiet(tmp)
    # the fixed temp name older versions used: an interrupted copy may have left one behind
    _unlink_quiet(dest.with_name(dest.name + ".tmp"))


def _link_local(proj: Any, src: Path, copy: bool) -> Tuple[Path, str]:
    """Put ``src`` at P/source.<ext> as a link (default) or a copy. The old entry of that name
    is replaced by rename, so it stays until the new one is complete; a rename replaces a link
    itself, never the file it points to."""
    ext = src.suffix.lower() if src.suffix else ".video"
    dest = proj.root / ("source" + ext)
    if dest.exists() and not dest.is_symlink() and dest.resolve() == src.resolve():
        return dest, "in-place"
    if copy:
        env.log("ingest", "copying %s (%s)" % (src.name, env.human_bytes(src.stat().st_size)))
        _copy_into(src, dest)
        return dest, "copy"
    tmp = _tmp_beside(dest)
    try:
        if os.path.lexists(str(tmp)):
            os.unlink(str(tmp))
        os.symlink(str(src), str(tmp))
        os.replace(str(tmp), str(dest))
        return dest, "symlink"
    except (OSError, NotImplementedError):
        env.log("ingest", "symlinks unavailable here: copying instead")
        _copy_into(src, dest)
        return dest, "copy"
    finally:
        _unlink_quiet(tmp)


def _download(proj: Any, url: str, section: Optional[Tuple[float, Optional[float]]]) -> Tuple[Path, Dict[str, Any]]:
    if not _ytdlp_available():
        raise env.HCError("URL ingest needs yt-dlp. Run: hc doctor --install --with url", code=3)
    dl = proj.root / "cache" / "download"
    if dl.exists():
        shutil.rmtree(str(dl))
    dl.mkdir(parents=True)
    py = sys.executable
    help_text = _ytdlp_help(py)
    cmd = [py, "-m", "yt_dlp", "--no-playlist", "--newline", "--no-mtime",
           "-f", os.environ.get("HOOKCUTTER_YTDLP_FORMAT") or YTDLP_FORMAT,
           "-S", os.environ.get("HOOKCUTTER_YTDLP_SORT") or YTDLP_SORT,
           "--merge-output-format", YTDLP_MERGE,
           "--write-info-json", "--no-write-playlist-metafiles",
           "-o", str(dl / "source.%(ext)s")] + quiet_args(help_text)
    try:
        cmd += ["--ffmpeg-location", env.ffmpeg()]
    except env.HCError:
        pass
    rt = js_runtime()
    if "--js-runtimes" in help_text and rt and rt != "deno":
        cmd += ["--js-runtimes", rt]
    if rt is None and re.search(r"youtu\.?be", url, re.I):
        env.log("ingest", "warning: no JavaScript runtime found (deno or node). YouTube downloads "
                          "may fail or miss formats; install deno (https://deno.com)")
    if section:
        a, b = section
        cmd += ["--download-sections", "*%s-%s" % (_ytdlp_time(a), "inf" if b is None else _ytdlp_time(b)),
                "--downloader-args", YTDLP_SECTION_FF_ARGS]
    cmd.append(url)
    env.log("ingest", "downloading with yt-dlp (sort %s)" % cmd[cmd.index("-S") + 1])
    if env.verbose():
        env.log("run", env.quote_cmd(cmd))
    rc = subprocess.call(cmd, stdin=subprocess.DEVNULL)
    if rc != 0:
        raise env.HCError("yt-dlp failed (exit %d). Update it with: hc doctor --install --with url"
                          % rc, code=3)
    media = [p for p in dl.iterdir() if p.name.startswith("source.")
             and not p.name.endswith((".json", ".part", ".ytdl", ".temp"))]
    if not media:
        raise env.HCError("yt-dlp finished but produced no media file in %s" % dl, code=3)
    media.sort(key=lambda p: p.stat().st_size, reverse=True)
    m = media[0]
    _preflight(m)          # a download without picture or sound never replaces a source in P
    dest = proj.root / ("source" + m.suffix.lower())
    os.replace(str(m), str(dest))
    info = {}  # type: Dict[str, Any]
    info_path = dl / "source.info.json"
    if info_path.exists():
        info = env.read_json(info_path)
        os.replace(str(info_path), str(proj.cache_path("ytdlp.info.json")))
    shutil.rmtree(str(dl), ignore_errors=True)
    return dest, info


def _ytdlp_time(t: float) -> str:
    return ("%.3f" % t).rstrip("0").rstrip(".")


def quiet_args(help_text: str = "") -> List[str]:
    """yt-dlp output flags: no warnings or info chatter, a progress line every 5 s."""
    out = list(YTDLP_QUIET)
    if not help_text or "--progress-delta" in help_text:
        out += ["--progress-delta", "5"]
    return out


def upscaled_format(info: Dict[str, Any]) -> bool:
    """True when yt-dlp picked a YouTube AI-upscaled ("-sr") format."""
    text = " ".join(str(info.get(k) or "") for k in ("format_note", "format")).lower()
    ids = str(info.get("format_id") or "").split("+")
    return "ai-upscaled" in text or any(i.endswith("-sr") for i in ids)


# ---------------------------------------------------------------- constant frame rate

def to_cfr(proj: Any, path: Path, meta: Dict[str, Any]) -> Path:
    """Transcode a variable-frame-rate source to constant frame rate at its nominal rate
    (libx264 crf 16 veryfast; audio copied bit for bit, so transcript times stay valid).
    Returns the new P/source.<ext>; the link or copy it was made from (``path``) is removed,
    and no other file in P is."""
    v, a = meta["video"], meta["audio"]
    num, den = int(v["fps"][0]), int(v["fps"][1])
    if not path.is_symlink() and path.resolve() == Path(str(meta.get("original") or "")).resolve():
        raise env.HCError("--cfr would replace %s, which is the original file itself: ingest it "
                          "from outside the project folder (or with --copy)" % path, code=1)
    codec = str(a.get("codec") or "")
    ext = ".mp4" if codec in CFR_AUDIO_MP4 else (".mov" if codec.startswith("pcm_") else ".mkv")
    final = proj.root / ("source" + ext)
    if final.name != path.name:
        try:
            kept = _in_place_original(proj, proj.source())
        except env.HCError:
            kept = None
        if kept is not None and kept.name == final.name:
            raise env.HCError("--cfr would write %s, which is an original file that was ingested "
                              "in place: move it out of the project folder first" % final, code=1)
    tmp = proj.cache_path("cfr.%d.tmp%s" % (os.getpid(), ext))
    rate = ("%.3f" % (num / float(den))).rstrip("0").rstrip(".")
    env.log("ingest", "re-encoding the variable-frame-rate video to constant %s fps (libx264 crf 16; "
                      "audio copied). This takes a while for long sources" % rate)
    args = ["-i", str(path), "-map", "0:%d" % int(v["index"]), "-map", "0:%d" % int(a["index"]),
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "16",
            "-fps_mode", "cfr", "-r", "%d/%d" % (num, den), "-c:a", "copy"]
    if ext == ".mp4":
        args += ["-movflags", "+faststart"]
    try:
        env.ff(args + [str(tmp)])
        os.replace(str(tmp), str(final))
        if path.name != final.name:
            _unlink_quiet(path)
    finally:
        if tmp.exists():
            tmp.unlink()
    return final


# ---------------------------------------------------------------- survey

def survey_frames(path: Any, src: Dict[str, Any]) -> List[Tuple[float, Any]]:
    """[(src time, RGB image)] for the survey: 16 frames across the runtime, at most 640 px
    wide (area-scaled, so the bar measurement sees no ringing at the picture edge)."""
    from . import sheets
    v = src["video"]
    dur = float(src.get("duration") or 0.0)
    start = float(v.get("start") or 0.0)
    width = min(probe.ACTIVE_SAMPLE_W, probe.display_size(v)[0]) or probe.ACTIVE_SAMPLE_W
    return [(t, sheets.frame_at(path, t, width=width, flags="area"))
            for t in sheets.even_times(start, max(start + 0.04, dur), SURVEY_FRAMES)]


def measure_layout(src: Dict[str, Any], frames: List[Tuple[float, Any]]) -> Optional[List[int]]:
    """Set src["video"]["active_area"] from the survey frames and refresh the warnings."""
    area = probe.active_area([img.convert("L") for _, img in frames], probe.display_size(src["video"]))
    src["video"]["active_area"] = area
    return area


def survey(proj: Any, src: Dict[str, Any], frames: Optional[List[Tuple[float, Any]]] = None) -> Path:
    """sheets/survey.jpg: 16 frames across the whole runtime, 4x4, with timecodes."""
    from . import sheets
    path = proj.source_path()
    v = src["video"]
    dur = float(src.get("duration") or 0.0)
    cell = sheets.default_cell_w(v["width"], v["height"])
    if frames is None:
        frames = survey_frames(path, src)
    decimals = 0 if dur / SURVEY_FRAMES >= 2.0 else 1
    items = [(img, env.fmt_time(t, decimals)) for t, img in frames]
    area = v.get("active_area")
    note = ("letterboxed: picture %dx%d at %d,%d" % (area[2], area[3], area[0], area[1])) if area \
        else "count people, spot bars, tiles and layout changes"
    title = "%s  survey of %s  (%d frames: %s)" % (src.get("path"), env.fmt_time(dur, 0), len(items), note)
    return sheets.contact_sheet(items, SURVEY_COLS, proj.sheet_path("survey.jpg"), cell, title=title)


# ---------------------------------------------------------------- command

def ingest(project: Any, source: str, *, copy: bool = False, section: Optional[str] = None,
           force: bool = False, cfr: bool = False) -> Dict[str, Any]:
    """Ingest ``source`` into ``project``; returns the source.json dict."""
    from .project import Project
    proj = project if isinstance(project, Project) else Project(project)
    url = is_url(source)
    sec = parse_section(section) if section else None
    if sec and not url:
        raise env.HCError("--section applies to URL downloads. For a local file, ingest the whole "
                          "file (clips pick their own ranges)", code=1)
    original = source.strip() if url else str(Path(source).expanduser().resolve())
    if not url and not Path(original).is_file():
        raise env.HCError("source file not found: %s" % source, code=1)
    proj.root.mkdir(parents=True, exist_ok=True)
    if proj.is_temp():
        env.log("ingest", "warning: %s is a temporary folder; projects should live somewhere durable"
                % proj.root)
    from . import fsutil
    # one ingest per project at a time: a second run waits, then sees what the first one did
    with fsutil.file_lock(proj.cache_path("ingest.lock"), what="ingest into %s" % proj.root):
        return _ingest_locked(proj, original, url, sec, copy=copy, force=force, cfr=cfr)


def _ingest_locked(proj: Any, original: str, url: bool, sec: Any, *, copy: bool, force: bool,
                   cfr: bool) -> Dict[str, Any]:
    old = None
    if proj.source_json_path().exists():
        try:
            old = proj.source()
        except env.HCError:
            old = None
    if old and not force:
        if old.get("original") == original:
            env.log("ingest", "already ingested; refreshing source.json (use --force to re-fetch)")
            if not url:
                return _finish(proj, Path(old.get("path") or proj.source_path().name), original,
                               old, "refresh", None, cfr=cfr)
            return _finish(proj, proj.source_path(), original, old, "refresh", None, cfr=cfr)
        raise env.HCError("%s already holds %s. Use --force to replace it (derived files then need "
                          "re-running)" % (proj.root, old.get("original")), code=1)
    kept = _in_place_original(proj, old)
    if url:
        if kept is not None:
            # the download's file name is not known until it is done, and it would be renamed
            # onto P/source.<ext>: never onto the user's own file
            raise env.HCError("%s is the original file itself (this project was ingested in "
                              "place) and a download could replace it: move that file out of "
                              "the project folder, or download into another project folder"
                              % kept, code=1)
        dest, info = _download(proj, original, sec)
        meta = _finish(proj, dest, original, old, "url", (info, sec), cfr=cfr)
        _drop_old_source(proj, old, meta["path"])
        return meta
    src = Path(original)
    _preflight(src)
    dest = proj.root / ("source" + (src.suffix.lower() if src.suffix else ".video"))
    if _foreign_file(proj, old, dest, src):
        raise env.HCError("%s already exists and source.json does not record it as a link, copy "
                          "or download hook-cutter made, so it may be your own file: ingesting "
                          "%s would replace it. Move it out of the project folder (or delete "
                          "it), then run this again" % (dest, src.name), code=1)
    dest, method = _link_local(proj, src, copy)
    meta = _finish(proj, dest, original, old, method, None, cfr=cfr)
    _drop_old_source(proj, old, meta["path"])
    return meta


def _frames_missing(video: Dict[str, Any]) -> bool:
    """probe.frames_missing: the shortfall test behind the "frames are missing ... re-ingest
    with --cfr" warning, so --cfr acts on every file that warning is printed for."""
    fn = getattr(probe, "frames_missing", None)
    try:
        return bool(fn(video)) if callable(fn) else False
    except Exception:
        return False


def _transcript_from_other_source(proj: Any, meta: Dict[str, Any]) -> Optional[bool]:
    """True when transcript.json names another source file than ``meta`` describes, False
    when it names this one (or the file this one was re-encoded from with --cfr), None when
    it cannot be told (no fingerprint recorded, unreadable, or an older --cfr record)."""
    try:
        tfp = (env.read_json(proj.transcript_path()) or {}).get("source_fingerprint")
    except (env.HCError, AttributeError):
        return None
    if not tfp or not meta.get("fingerprint"):
        return None
    if tfp == meta["fingerprint"]:
        return False
    rec = (meta.get("ingest") or {}).get("cfr")
    if isinstance(rec, dict):
        if "from_fingerprint" not in rec:
            return None
        if tfp == rec.get("from_fingerprint"):
            return False
    return True


def _finish(proj: Any, dest: Path, original: str, old: Optional[Dict[str, Any]], method: str,
            url_bits: Optional[Tuple[Dict[str, Any], Any]], cfr: bool = False) -> Dict[str, Any]:
    dest = proj.root / Path(dest).name
    meta = probe.probe_source(dest, original=original, name=dest.name)
    cfr_done = None
    if cfr:
        if meta["video"]["vfr"] or _frames_missing(meta["video"]):
            # the fingerprint before the re-encode: the audio is copied bit for bit, so a
            # transcript made from that file stays valid for the new one (plan accepts it)
            from_fps, from_fp = meta["video"].get("avg_fps"), meta.get("fingerprint")
            dest = to_cfr(proj, dest, meta)
            meta = probe.probe_source(dest, original=original, name=dest.name)
            cfr_done = {"from_avg_fps": from_fps, "fps": meta["video"]["fps"],
                        "from_fingerprint": from_fp}
        else:
            env.log("ingest", "the source is already constant frame rate: --cfr changes nothing")
    frames = survey_frames(dest, meta)
    measure_layout(meta, frames)
    meta["warnings"] = probe.source_warnings(meta)
    meta["ingest"] = {"method": method, "section": None}
    if method == "refresh" and old:
        # Re-probe only: keep what the original ingest learned (URL facts, section notes).
        meta["ingest"] = dict(old.get("ingest") or meta["ingest"])
        if old.get("url_info"):
            meta["url_info"] = old["url_info"]
        meta["warnings"] += [w for w in old.get("warnings") or []
                             if w.startswith(("section download", "the download is"))]
    if cfr_done:
        meta["ingest"]["cfr"] = cfr_done
        avg = cfr_done["from_avg_fps"]
        avg = ("%.3f" % avg).rstrip("0").rstrip(".") if isinstance(avg, (int, float)) else "?"
        num, den = cfr_done["fps"]
        rate = ("%.3f" % (num / float(den))).rstrip("0").rstrip(".")
        meta["warnings"].append("re-encoded to constant frame rate (%s fps average -> %s fps); the "
                                "audio is unchanged" % (avg, rate))
    if url_bits is not None:
        info, sec = url_bits
        approx = bool(sec)
        offset = sec[0] if sec else 0.0
        preroll = 0.0
        if sec:
            # A stream-copied section starts each stream on its own keyframe at or before the
            # requested start; audio frames are all keyframes, so the later-starting stream
            # marks the requested start and everything before it is keyframe pre-roll.
            preroll = max(meta["video"]["start"], meta["audio"]["start"])
            offset = sec[0] - preroll
        meta["ingest"]["section"] = [sec[0], sec[1]] if sec else None
        meta["url_info"] = {k: info.get(k) for k in ("id", "title", "uploader", "channel", "duration",
                                                     "webpage_url", "upload_date", "format_id",
                                                     "dynamic_range", "vcodec", "acodec")}
        env.write_json(proj.signals_path(), _signals(info, offset, meta["duration"], approx))
        if sec:
            meta["warnings"].append(
                "section download: the file starts %.2f s before the requested %s (keyframe "
                "pre-roll); signals.json times are shifted to this file's clock (estimated offset "
                "%s)" % (preroll, env.fmt_time(sec[0]), env.fmt_time(offset, 2)))
        if (info.get("dynamic_range") or "SDR").upper() != "SDR":
            meta["warnings"].append("the download is %s: no SDR format was offered"
                                    % info.get("dynamic_range"))
        if upscaled_format(info):
            meta["warnings"].append("the download is an AI-upscaled format (%s): faces are "
                                    "model-generated pixels; unset HOOKCUTTER_YTDLP_FORMAT or pick "
                                    "a native format" % (info.get("format_id") or "?"))
    changed = bool(old and old.get("fingerprint") and old.get("fingerprint") != meta["fingerprint"])
    if proj.has_transcript():
        if changed and cfr_done:
            meta["warnings"].append("the source was re-encoded: re-run plan and render (the "
                                    "audio is unchanged, so transcript.json stays valid)")
        # Judged from the transcript itself when it says which file it was made from, so the
        # warning is repeated on every later ingest until transcribe is re-run; a transcript
        # without that record falls back to "the file changed in this very ingest".
        stale = _transcript_from_other_source(proj, meta)
        if stale is None:
            stale = changed and not cfr_done
        if stale:
            meta["warnings"].append("the source changed: transcript.json, plans and renders "
                                    "were made from the old file; re-run transcribe")
    env.write_json(proj.source_json_path(), meta)
    sheet = survey(proj, meta, frames)
    print(probe.describe(meta))
    for w in meta["warnings"]:
        print("warning: %s" % w)
    print("survey: %s (open it: count people, check for black bars or tiles, spot layout changes)"
          % sheet)
    if not proj.has_transcript():
        print("next: hc transcribe %s" % shlex.quote(str(proj.root)))
    return meta


def cmd_ingest(args: Any) -> int:
    ingest(args.project, args.source, copy=bool(getattr(args, "copy", False)),
           section=getattr(args, "section", None), force=bool(getattr(args, "force", False)),
           cfr=bool(getattr(args, "cfr", False)))
    return 0
