"""ffprobe → source.json (contract section 6.1).

Clock rule: ``video.start`` and ``audio.start`` are stream start_time minus the
format start_time, i.e. positions on the ``src`` clock (the timeline ``ffmpeg -ss``
uses). A decoder's first output sample/frame of a stream sits at that time.
"""

from __future__ import annotations

import re
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import env

STANDARD_RATES = (Fraction(24000, 1001), Fraction(24), Fraction(25), Fraction(30000, 1001),
                  Fraction(30), Fraction(48), Fraction(50), Fraction(60000, 1001), Fraction(60),
                  Fraction(100), Fraction(120000, 1001), Fraction(120))
VFR_AVG_TOLERANCE = 0.005      # avg vs nominal rate, relative
VFR_DELTA_FRAMES = 1.5          # spread of packet pts deltas, in frame durations
VFR_SAMPLE_PACKETS = 300
CHANNEL_SAMPLE_S = 60.0
FOLD_MONO_DB = 20.0
DUP_MONO_DB = 40.0
SILENT_DB = -70.0
DB_FLOOR = -120.0
UPSCALE_WARN = 1.8
SHORT_SOURCE_S = 60.0
VERTICAL_OUT_H = 1920
# Letterbox / gallery detection (ingest samples the survey frames, grey, 640 px wide).
ACTIVE_SAMPLE_W = 640
BAR_LUMA = 20                   # a row/column whose max luma over every sample is <= this is a bar
BAR_MIN_TRIM = 0.02             # a side is trimmed only when its bars cover >= 2% of the frame
ACTIVE_MIN_SHARE = 0.10         # smaller "pictures" are noise (a black video with a logo)
AUDIO_HOLE_FRAMES = 2.0         # audio packet or gap longer than this many video frames

_CACHE = {}  # type: Dict[Tuple[str, int, int], Dict[str, Any]]


# ---------------------------------------------------------------- small parsers

def _frac(value: Any) -> Optional[Fraction]:
    if value in (None, "", "N/A"):
        return None
    try:
        s = str(value)
        if "/" in s:
            num, den = s.split("/", 1)
            if int(den) == 0:
                return None
            f = Fraction(int(num), int(den))
        else:
            f = Fraction(s)
    except (ValueError, ZeroDivisionError):
        return None
    return f if f > 0 else None


def _float(value: Any) -> Optional[float]:
    if value in (None, "", "N/A"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _clean(value: Any) -> Optional[str]:
    if value is None:
        return None
    s = str(value).strip().rstrip(",").strip()
    return s or None


def _snap_rate(rate: Fraction) -> Fraction:
    """Snap odd rationals (e.g. 2997/100) to the standard rate they mean."""
    if rate.denominator in (1, 1001):
        return rate
    for std in STANDARD_RATES:
        if abs(float(rate) / float(std) - 1.0) < 0.001:
            return std
    return rate.limit_denominator(1001)


def hdr_kind(transfer: Optional[str]) -> str:
    t = (_clean(transfer) or "").lower()
    if t == "arib-std-b67":
        return "hlg"
    if t == "smpte2084":
        return "pq"
    return "none"


def _rotation(stream: Dict[str, Any]) -> int:
    rot = None
    for sd in stream.get("side_data_list") or []:
        if "rotation" in sd:
            rot = _float(sd.get("rotation"))
            break
    if rot is None:
        tag = (stream.get("tags") or {}).get("rotate")
        if tag not in (None, ""):
            r = _float(tag)
            rot = -r if r is not None else None  # tag is clockwise; side data is not
    if rot is None:
        return 0
    r = int(round(rot)) % 360
    return r - 360 if r > 180 else r


# ---------------------------------------------------------------- media probe

def _pick(streams: List[Dict[str, Any]], kind: str) -> Optional[Dict[str, Any]]:
    cands = [s for s in streams if s.get("codec_type") == kind
             and not (s.get("disposition") or {}).get("attached_pic")]
    if not cands:
        return None
    default = [s for s in cands if (s.get("disposition") or {}).get("default")]
    return (default or cands)[0]


def _packet_deltas(path: Any, index: int, time_base: Fraction) -> List[float]:
    data = env.ffprobe_json(path, "-select_streams", str(index), "-show_entries", "packet=pts",
                            "-read_intervals", "%%+#%d" % VFR_SAMPLE_PACKETS)
    pts = sorted(int(p["pts"]) for p in data.get("packets") or []
                 if str(p.get("pts", "N/A")).lstrip("-").isdigit())
    return [float((b - a) * time_base) for a, b in zip(pts, pts[1:]) if b > a]


def _video_info(path: Any, s: Dict[str, Any], format_start: float, duration: Optional[float],
                sample_packets: bool) -> Dict[str, Any]:
    rot = _rotation(s)
    w, h = int(s.get("width") or 0), int(s.get("height") or 0)
    if rot in (90, -90):
        w, h = h, w
    r_rate = _frac(s.get("r_frame_rate"))
    avg_rate = _frac(s.get("avg_frame_rate"))
    rate = r_rate or avg_rate or Fraction(25)
    # r_frame_rate can be a timebase-like value on VFR phone files (e.g. 90000/1).
    if avg_rate and (rate > 240 or rate > 2 * avg_rate):
        rate = avg_rate
    rate = _snap_rate(rate)
    vfr = False
    if avg_rate and abs(float(avg_rate) / float(rate) - 1.0) > VFR_AVG_TOLERANCE:
        vfr = True
    tb = _frac(s.get("time_base")) or Fraction(1, 90000)
    if sample_packets and not vfr:
        deltas = _packet_deltas(path, int(s["index"]), tb)
        if len(deltas) >= 3:
            spread = max(deltas) - min(deltas)
            if spread > VFR_DELTA_FRAMES * float(1 / rate):
                vfr = True
    st = _float(s.get("start_time"))
    start = (st - format_start) if st is not None else 0.0
    sdur = _float(s.get("duration"))
    if sdur is None and duration is not None:
        sdur = max(0.0, duration - start)
    nb = s.get("nb_frames")
    nb_frames = int(nb) if str(nb or "").isdigit() and int(nb) > 0 else None
    frames = nb_frames if nb_frames else (int(round((sdur or 0.0) * rate)) if sdur else None)
    sar = _clean(s.get("sample_aspect_ratio"))
    if not sar or sar in ("0:1", "N/A"):
        sar = "1:1"
    transfer = _clean(s.get("color_transfer"))
    return {
        "index": int(s["index"]),
        "codec": s.get("codec_name"),
        "width": w, "height": h,
        "coded_width": int(s.get("coded_width") or s.get("width") or 0),
        "coded_height": int(s.get("coded_height") or s.get("height") or 0),
        "rotation": rot,
        "sar": sar,
        "fps": [rate.numerator, rate.denominator],
        "avg_fps": round(float(avg_rate), 6) if avg_rate else None,
        "vfr": vfr,
        "start": round(start, 6),
        "duration": round(sdur, 6) if sdur is not None else None,
        "pix_fmt": s.get("pix_fmt"),
        "color_space": _clean(s.get("color_space")),
        "color_transfer": transfer,
        "color_primaries": _clean(s.get("color_primaries")),
        "color_range": _clean(s.get("color_range")),
        "field_order": _clean(s.get("field_order")),
        "hdr": hdr_kind(transfer),
        "frames": frames,
        "nb_frames": nb_frames,
        "time_base": [tb.numerator, tb.denominator],
    }


def _audio_info(s: Dict[str, Any], format_start: float) -> Dict[str, Any]:
    st = _float(s.get("start_time"))
    start = (st - format_start) if st is not None else 0.0
    return {
        "index": int(s["index"]),
        "codec": s.get("codec_name"),
        "sample_rate": int(s.get("sample_rate") or 0),
        "channels": int(s.get("channels") or 0),
        "channel_layout": s.get("channel_layout"),
        "start": round(start, 6),
        "duration": _float(s.get("duration")),
    }


def probe_media(path: Any, *, sample_packets: bool = True) -> Dict[str, Any]:
    """Format + first video/audio stream facts for any media file (cached per file state)."""
    p = Path(path)
    try:
        st = p.stat()
    except OSError:
        raise env.HCError("file not found: %s" % p, code=1)
    key = (str(p.resolve()), st.st_size, st.st_mtime_ns)
    if key in _CACHE:
        return _CACHE[key]
    data = env.ffprobe_json(p, "-show_format", "-show_streams")
    fmt = data.get("format") or {}
    streams = data.get("streams") or []
    format_start = _float(fmt.get("start_time")) or 0.0
    duration = _float(fmt.get("duration"))
    vs, aus = _pick(streams, "video"), _pick(streams, "audio")
    video = _video_info(p, vs, format_start, duration, sample_packets) if vs else None
    audio = _audio_info(aus, format_start) if aus else None
    if duration is None:
        cands = [x.get("start", 0) + (x.get("duration") or 0) for x in (video, audio) if x]
        duration = max(cands) if cands else None
    info = {
        "duration": round(duration, 6) if duration is not None else None,
        "format_start": round(format_start, 6),
        "format_name": fmt.get("format_name"),
        "size": int(fmt.get("size") or st.st_size),
        "bit_rate": int(fmt["bit_rate"]) if str(fmt.get("bit_rate", "")).isdigit() else None,
        "video": video,
        "audio": audio,
    }
    _CACHE[key] = info
    return info


# ---------------------------------------------------------------- channel levels

_CH_RE = re.compile(r"Channel:\s*(\d+)")
_RMS_RE = re.compile(r"RMS level dB:\s*(\S+)")


def _db(value: str) -> float:
    try:
        v = float(value)
    except ValueError:
        return DB_FLOOR
    if v != v or v < DB_FLOOR:  # nan or -inf
        return DB_FLOOR
    return round(v, 2)


def channel_levels(path: Any, audio: Dict[str, Any], duration: float) -> Tuple[List[float], Optional[float]]:
    """Per-channel RMS dB over a 60 s sample from the middle (astats).

    Returns (channel_db, diff_db) where diff_db is the RMS of (c0 - c1) / 2,
    used to spot duplicated mono; None for mono sources.
    """
    n = int(audio.get("channels") or 0)
    if n <= 0:
        return [], None
    t = min(CHANNEL_SAMPLE_S, max(duration, 0.1))
    ss = max(0.0, duration / 2.0 - t / 2.0)
    stats = "astats=measure_perchannel=RMS_level:measure_overall=none"
    if n >= 2:
        maps = "|".join("c%d=c%d" % (i, i) for i in range(n))
        af = "pan=%dc|%s|c%d=0.5*c0-0.5*c1,%s" % (n + 1, maps, n, stats)
    else:
        af = stats
    res = env.ff(["-loglevel", "info", "-ss", "%.3f" % ss, "-t", "%.3f" % t, "-i", str(path),
                  "-map", "0:%d" % int(audio["index"]), "-af", af, "-f", "null", "-"])
    levels = {}  # type: Dict[int, float]
    cur = None
    for line in res.stderr.splitlines():
        m = _CH_RE.search(line)
        if m:
            cur = int(m.group(1))
            continue
        m = _RMS_RE.search(line)
        if m and cur is not None and cur not in levels:
            levels[cur] = _db(m.group(1))
    if not levels:
        raise env.HCError("could not measure audio levels of %s (astats printed nothing)" % path, code=3)
    chans = [levels.get(i + 1, DB_FLOOR) for i in range(n)]
    diff = levels.get(n + 1) if n >= 2 else None
    return chans, diff


# ---------------------------------------------------------------- picture area

def display_size(video: Dict[str, Any]) -> Tuple[int, int]:
    """Upright, square-pixel (width, height) of a source.json video block (as framing uses)."""
    w, h = int(video.get("width") or 0), int(video.get("height") or 0)
    raw = str(video.get("sar") or "1:1").replace("/", ":")
    try:
        a, b = raw.split(":")
        sar = Fraction(int(a), int(b))
    except (ValueError, ZeroDivisionError):
        sar = Fraction(1)
    if sar > 0 and sar != 1:
        w = int(round(w * sar / 2)) * 2
    return w, h


def _even(v: float) -> int:
    return int(round(v / 2.0)) * 2


def active_area(gray_frames: Sequence[Any], display: Tuple[int, int]) -> Optional[List[int]]:
    """[x, y, w, h] in display pixels of the picture inside black bars, or None (full frame).

    ``gray_frames``: same-size 2-D luma images (numpy arrays or PIL "L" images) sampled across
    the source, scaled from the display frame. A row or column is a bar when its brightest
    pixel over every sample is <= BAR_LUMA; the picture is the bounding box of the rest. A
    side is trimmed only when its bars cover at least BAR_MIN_TRIM of the frame."""
    import numpy as np
    frames = [np.asarray(f, dtype=np.uint8) for f in gray_frames]
    frames = [f for f in frames if f.ndim == 2 and f.size]
    if not frames:
        return None
    shape = frames[0].shape
    frames = [f for f in frames if f.shape == shape]
    peak = np.maximum.reduce(frames) if len(frames) > 1 else frames[0]
    ih, iw = peak.shape
    rows = np.flatnonzero(peak.max(axis=1) > BAR_LUMA)
    cols = np.flatnonzero(peak.max(axis=0) > BAR_LUMA)
    if rows.size == 0 or cols.size == 0:
        return None                     # nothing but black in the samples: no layout to report
    top = int(rows[0]) if rows[0] >= BAR_MIN_TRIM * ih else 0
    bottom = int(rows[-1]) + 1 if ih - rows[-1] - 1 >= BAR_MIN_TRIM * ih else ih
    left = int(cols[0]) if cols[0] >= BAR_MIN_TRIM * iw else 0
    right = int(cols[-1]) + 1 if iw - cols[-1] - 1 >= BAR_MIN_TRIM * iw else iw
    if (top, bottom, left, right) == (0, ih, 0, iw):
        return None
    W, H = int(display[0]), int(display[1])
    x0 = min(max(0, _even(left * W / float(iw))), W)
    x1 = min(max(0, _even(right * W / float(iw))), W)
    y0 = min(max(0, _even(top * H / float(ih))), H)
    y1 = min(max(0, _even(bottom * H / float(ih))), H)
    w, h = x1 - x0, y1 - y0
    if w < ACTIVE_MIN_SHARE * W or h < ACTIVE_MIN_SHARE * H:
        return None
    if (x0, y0, w, h) == (0, 0, W, H):
        return None
    return [x0, y0, w, h]


def audio_holes(path: Any, audio: Dict[str, Any], frame_s: float) -> List[Tuple[float, float]]:
    """[(src time, seconds)] of audio timestamp holes and over-long packets (a hole muxed as
    one packet) longer than AUDIO_HOLE_FRAMES video frames. Raw PCM (big packets by design)
    is not checked; a probe failure returns []."""
    codec = str(audio.get("codec") or "")
    if codec.startswith("pcm_") or frame_s <= 0:
        return []
    try:
        proc = env.run([env.ffprobe(), "-v", "error", "-select_streams", str(int(audio["index"])),
                        "-show_entries", "packet=pts_time,duration_time", "-of", "csv=p=0",
                        str(path)], check=False, timeout=300)
    except env.HCError:
        return []
    if proc.returncode != 0:
        return []
    limit = AUDIO_HOLE_FRAMES * frame_s
    out = []
    prev_end = None
    for line in (proc.stdout or "").splitlines():
        parts = line.strip().split(",")
        if len(parts) < 2:
            continue
        t, d = _float(parts[0]), _float(parts[1])
        if t is None:
            continue
        if prev_end is not None and t - prev_end > limit:
            out.append((round(prev_end, 3), round(t - prev_end, 3)))
        if d is not None and d > limit:
            out.append((round(t, 3), round(d, 3)))
        if d is not None:
            prev_end = t + d
        else:
            prev_end = None
    return out


# ---------------------------------------------------------------- source.json

def source_warnings(src: Dict[str, Any]) -> List[str]:
    """The probe warnings for a source.json dict (call again after setting active_area)."""
    return _warnings({"duration": src.get("duration")}, src["video"], src["audio"])


def frames_missing(video_meta) -> bool:
    """True when the video stream holds fewer frames than its duration needs at the nominal rate
    (by more than one frame): frames were dropped or the rate varies somewhere in the file, even
    when the first packets and the average rate look constant. The ingest warning that says to
    re-ingest with --cfr and the --cfr gate both use this one test."""
    v = video_meta or {}
    try:
        num, den = v["fps"][0], v["fps"][1]
        fps = float(num) / float(den)
        nb, vdur = v.get("nb_frames"), v.get("duration")
        if not nb or not vdur or fps <= 0:
            return False
        return float(vdur) - float(nb) / fps > 1.0 / fps + 1e-6
    except (KeyError, IndexError, TypeError, ValueError, ZeroDivisionError):
        return False


def _warnings(info: Dict[str, Any], v: Dict[str, Any], a: Dict[str, Any]) -> List[str]:
    out = []
    w, h = v["width"], v["height"]
    area = v.get("active_area")
    if area:
        out.append("letterboxed/gallery layout: the picture is %dx%d at (%d,%d) inside the frame; "
                   "framing stays inside it" % (area[2], area[3], area[0], area[1]))
    pic_h = int(area[3]) if area else h
    if pic_h > 0:
        up = VERTICAL_OUT_H / float(pic_h)
        if up > UPSCALE_WARN:
            what = ("source is %dx%d" % (w, h)) if not area else \
                ("the picture is %dx%d" % (area[2], area[3]))
            out.append("%s: a full-height 9:16 crop upscales %.1fx and will look soft"
                       % (what, up))
    if v["hdr"] != "none":
        if env.has_filter("zscale"):
            out.append("source is HDR (%s): renders tone-map it to SDR BT.709" % v["hdr"].upper())
        else:
            out.append("source is HDR (%s): renders tone-map to SDR BT.709, which needs an ffmpeg "
                       "with zscale (run: hc doctor)" % v["hdr"].upper())
    if v["vfr"]:
        nominal = "%g" % round(v["fps"][0] / float(v["fps"][1]), 3)
        avg = ("average %.3f fps vs nominal %s" % (v["avg_fps"], nominal)) if v.get("avg_fps") \
            else "nominal %s fps" % nominal
        out.append("source is variable frame rate (%s): cuts snap to the nominal grid and renders "
                   "are constant frame rate, so check seams closely" % avg)
    if v["rotation"]:
        out.append("source carries a %d degree rotation: dimensions are reported upright (%dx%d)"
                   % (v["rotation"], w, h))
    if v["sar"] != "1:1":
        out.append("source has non-square pixels (SAR %s): framing uses stored pixels" % v["sar"])
    if (v.get("field_order") or "progressive") not in ("progressive", "unknown"):
        out.append("source is interlaced (%s): expect combing on motion; deinterlace it first "
                   "for best results" % v["field_order"])
    dbs = a.get("channel_db") or []
    if dbs and max(dbs) < SILENT_DB:
        out.append("audio is near silent in the sampled minute (%.0f dB RMS): check the right "
                   "audio track was recorded" % max(dbs))
    elif a.get("fold_mono") and len(dbs) >= 2:
        lo = 0 if dbs[0] < dbs[1] else 1
        out.append("channel %d is %.0f dB quieter than channel %d in the sampled minute: both "
                   "channels are mixed and the audio is folded to mono (nothing is dropped)"
                   % (lo + 1, abs(dbs[0] - dbs[1]), 2 - lo))
    frame = v["fps"][1] / float(v["fps"][0])
    off = a["start"] - v["start"]
    if abs(off) > frame / 2.0:
        out.append("audio stream starts %.3f s %s the video: times use the container timeline, "
                   "so this is handled" % (abs(off), "after" if off > 0 else "before"))
    fps = v["fps"][0] / float(v["fps"][1])
    nb, vdur = v.get("nb_frames"), v.get("duration")
    if nb and vdur and fps > 0 and abs(nb / fps - vdur) > 1.0 / fps + 1e-6:
        rate = ("%.3f" % fps).rstrip("0").rstrip(".")
        if not frames_missing(v):
            out.append("the video stream holds %d frames (%.2f s at %s fps) but plays %.2f s: an "
                       "edit list hides %.2f s of it (normal for a section download) or the frame "
                       "rate varies" % (nb, nb / fps, rate, vdur, nb / fps - vdur))
        else:
            out.append("the video stream has %d frames (%.2f s at %s fps) but lasts %.2f s: "
                       "frames are missing (variable frame rate or dropped frames); if cuts or "
                       "the end of a clip misbehave, re-ingest with --cfr"
                       % (nb, nb / fps, rate, vdur))
    holes = a.get("holes") or []
    if holes:
        t, d = max(holes, key=lambda x: x[1])
        out.append("the audio has %d timestamp hole(s) or over-long packet(s) (largest %.2f s at "
                   "src %s): words near them may be timed late; re-mux the file or re-encode its "
                   "audio" % (len(holes), d, env.fmt_time(t, 2)))
    dur = info.get("duration") or 0.0
    if dur < SHORT_SOURCE_S:
        out.append("source is only %.0f s long: shorter than most target clip lengths" % dur)
    return out


def probe_source(path: Any, original: Optional[str] = None, *, name: Optional[str] = None) -> Dict[str, Any]:
    """Probe a source file → the source.json dict. No audio or no video → HCError."""
    p = Path(path)
    info = probe_media(p)
    v, a = info["video"], info["audio"]
    if v is None:
        raise env.HCError("%s has no video stream" % p.name, code=1)
    if a is None:
        raise env.HCError("%s has no audio stream: hook-cutter cuts on speech, so the source needs "
                          "audio" % p.name, code=1)
    duration = info["duration"] or 0.0
    a = dict(a)
    chans, diff = channel_levels(p, a, duration)
    a["channel_db"] = chans
    fold = dup = False
    if len(chans) >= 2:
        c0, c1 = chans[0], chans[1]
        loud = max(c0, c1)
        fold = loud > SILENT_DB and abs(c0 - c1) > FOLD_MONO_DB
        dup = (not fold) and loud > SILENT_DB and diff is not None and diff <= min(c0, c1) - DUP_MONO_DB
    a["fold_mono"] = bool(fold)
    a["dup_mono"] = bool(dup)
    a["diff_db"] = diff
    fs = float(info.get("format_start") or 0.0)
    a["holes"] = [[round(t - fs, 3), d]       # src clock (stream time minus format start)
                  for t, d in audio_holes(p, a, float(Fraction(v["fps"][1], v["fps"][0])))][:50]
    out = {
        "version": 1,
        "path": name or p.name,
        "original": original or str(p.resolve()),
        "fingerprint": env.fingerprint(p),
        "size": info["size"],
        "duration": duration,
        "format_start": info["format_start"],
        "format_name": info["format_name"],
        "video": v,
        "audio": a,
        "warnings": [],
    }
    out["warnings"] = _warnings(info, v, a)
    return out


def describe(src: Dict[str, Any]) -> str:
    """One-line human summary of a source.json dict."""
    v, a = src["video"], src["audio"]
    fps = v["fps"][0] / float(v["fps"][1])
    return "%s  %s  %dx%d  %s fps%s  %s  %s %d Hz %d ch" % (
        src.get("path"), env.fmt_time(src.get("duration"), 0), v["width"], v["height"],
        ("%.3f" % fps).rstrip("0").rstrip("."), " VFR" if v["vfr"] else "",
        v["codec"], a["codec"], a["sample_rate"], a["channels"])
