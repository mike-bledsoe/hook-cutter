"""Render a planned clip (contract 9.2).

Stage A: one lossless intermediate per plan segment (input-seeked, frame-exact video, sample-exact
audio, framing applied), cached by content hash. Then a loudness measurement and a 5 fps luma
track of the concatenated base. Stage B: concat filter over the segments + the caption strip on
a pipe at the exact rational fps + dip overlays + linear loudnorm + TPDF dither, encoded once.

Sync rules that keep cuts from drifting: every segment's video is exactly `frames` frames and
its audio exactly `samples` samples (verified with ffprobe); stage B regenerates timestamps from
frame and sample counts before the concat filter; AAC is encoded exactly once.
"""

from __future__ import annotations

import hashlib
import os
import time
from fractions import Fraction
from pathlib import Path

import numpy as np

from . import env, framing, fsutil
from . import render_util as ru

SR = 48000
SEEK_PREROLL = 2.0          # s decoded before a segment so the seek lands on clean frames
TAIL_READ = 1.0             # s read past a segment end
CONCAT_GROUP = 48           # more segments than this are pre-joined losslessly in groups
LUMA_FPS = 5
LUMA_SIZE = {"vertical": (270, 480), "landscape": (480, 270)}
DIP_ALPHA = 0.6
DIP_HALF = 0.4
FINAL_FADE_OUT = 0.06
LOUDNORM = (-14.0, -1.5, 11.0)   # I, TP, LRA
CONTACT_FRAMES = 12
CONTACT_COLS = 4
CELL_W = {"vertical": 270, "landscape": 480}
LOSSLESS_V = ["-c:v", "libx264", "-qp", "0", "-preset", "ultrafast", "-pix_fmt", "yuv420p"]
LOSSLESS_A = ["-c:a", "pcm_s24le", "-ar", str(SR)]
FINAL_V = ["-c:v", "libx264", "-crf", "18", "-preset", "medium", "-profile:v", "high",
           "-pix_fmt", "yuv420p", "-color_primaries", "bt709", "-color_trc", "bt709",
           "-colorspace", "bt709", "-color_range", "tv"]
FINAL_A = ["-c:a", "aac", "-b:a", "192k", "-ar", str(SR)]
SCALE_MATRIX = {"bt709": "bt709", "smpte170m": "smpte170m", "bt470bg": "bt470", "bt601": "bt601",
                "fcc": "fcc", "smpte240m": "smpte240m", "bt2020nc": "bt2020", "bt2020c": "bt2020",
                "bt2020": "bt2020"}
TONEMAP = ("zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,"
           "tonemap=tonemap=hable:desat=0,zscale=t=bt709:m=bt709:r=tv,format=yuv420p")
CODE_FILES = ("render.py", "framing.py", "render_util.py")
CAPTION_FILES = ("captions.py", "typeset.py")

_CODE_HASH = {}


def _code_hash(files) -> str:
    key = tuple(files)
    if key not in _CODE_HASH:
        h = hashlib.sha256(env.code_version().encode())
        here = Path(__file__).resolve().parent
        for name in files:
            p = here / name
            h.update(name.encode() + b"\0" + (p.read_bytes() if p.is_file() else b"-"))
        _CODE_HASH[key] = h.hexdigest()[:16]
    return _CODE_HASH[key]


# ---------------------------------------------------------------- small helpers

def _fps(plan) -> tuple:
    f = framing.fps_of(plan)
    return f.numerator, f.denominator


def _samples_for(frames: int, num: int, den: int) -> int:
    return int(round(Fraction(frames * SR * den, num)))


def _ff_escape(value: str) -> str:
    """Escape a string for a filter option value inside a filtergraph (both levels)."""
    lvl1 = "".join("\\" + c if c in "\\':" else c for c in str(value))
    return "".join("\\" + c if c in "\\'[],;" else c for c in lvl1)


def _list(v):
    if v is None:
        return None
    if isinstance(v, str):
        v = [x.strip() for x in v.split(",")]
    out = [str(x) for x in v if str(x).strip()]
    return out or None


def _rel(proj, p) -> str:
    return proj.rel(p) if hasattr(proj, "rel") else str(p)


def _color_args(video: dict, hdr: bool) -> str:
    if hdr:
        return ":in_color_matrix=bt709:in_range=tv:out_color_matrix=bt709:out_range=tv"
    cs = (video.get("color_space") or "").lower()
    m = SCALE_MATRIX.get(cs) or ("bt709" if int(video.get("height") or 0) >= 720 else "smpte170m")
    pc = (video.get("color_range") or "").lower() in ("pc", "jpeg", "full") or \
        str(video.get("pix_fmt") or "").startswith("yuvj")
    return ":in_color_matrix=%s:in_range=%s:out_color_matrix=bt709:out_range=tv" % (
        m, "pc" if pc else "tv")


def _hdr_chain(video: dict) -> str:
    hdr = (video.get("hdr") or "none").lower()
    if hdr in ("", "none", "sdr"):
        return ""
    if not env.has_filter("zscale"):
        raise env.HCError("the source is HDR (%s) and this ffmpeg has no zscale filter to tone-map "
                          "it to SDR. Run: hc doctor (it can point HOOKCUTTER_FFMPEG at an ffmpeg "
                          "build with zscale)" % hdr, code=3)
    return TONEMAP


def _deinterlace(video: dict) -> str:
    """bwdif (or yadif) for interlaced sources, applied before any trim (it needs both fields)."""
    if (video.get("field_order") or "progressive") in ("progressive", "unknown", ""):
        return ""
    name = "bwdif" if env.has_filter("bwdif") else ("yadif" if env.has_filter("yadif") else "")
    return ("%s=mode=send_frame:parity=auto:deint=all," % name) if name else ""


MONO_SUM = "pan=stereo|c0=0.5*c0+0.5*c1|c1=0.5*c0+0.5*c1"


def _pan(audio: dict) -> str:
    """The stereo layout of a stage A segment. A fold (`fold_mono`: one channel far quieter in
    the sampled minute) is an equal-weight SUM of both channels, never a copy of the louder one:
    a channel that was quiet in that minute can carry the other speaker everywhere else, and a
    truly dead channel adds only its noise floor. It is also what the transcriber heard (-ac 1).
    More than two channels always go through the standard downmix first, so no channel is ever
    picked or dropped by level."""
    ch = int(audio.get("channels") or 2)
    if ch == 1:
        return "pan=stereo|c0=c0|c1=c0"
    fold = bool(audio.get("fold_mono"))
    if ch == 2:
        return MONO_SUM if fold else "pan=stereo|c0=c0|c1=c1"
    return "aformat=channel_layouts=stereo" + ("," + MONO_SUM if fold else "")


# ---------------------------------------------------------------- inputs

def _media(proj, plan: dict, clip: dict) -> tuple:
    """(media path, source info dict in source.json shape) for a clip."""
    kind = plan.get("kind") or clip.get("kind") or "cut"
    if kind != "caption_only":
        return proj.source_path(), proj.source()
    sf = clip.get("source_file") or plan.get("source_file")
    if not sf:
        raise env.HCError("caption_only clip %s has no source_file" % plan.get("clip"), code=1)
    p = Path(sf).expanduser()
    if not p.is_absolute():
        p = proj.root / p
    if not p.exists():
        raise env.HCError("caption_only source_file not found: %s" % p, code=1)
    from . import probe
    info = dict(probe.probe_media(p))
    if not info.get("video") or not info.get("audio"):
        raise env.HCError("%s needs both video and audio" % p.name, code=1)
    info["fingerprint"] = env.fingerprint(p)
    return p, info


def _load_plan(proj, clip_id: str) -> dict:
    try:
        from . import plan as plan_mod
    except ImportError:
        plan_mod = None
    if plan_mod is not None and hasattr(plan_mod, "load_plan"):
        try:
            return plan_mod.load_plan(proj, clip_id, require_ok=True)
        except TypeError:
            return plan_mod.load_plan(proj, clip_id)
    p = proj.plan_path(clip_id)
    if not p.exists():
        raise env.HCError("no plan for %s. Run: hc plan %s %s" % (clip_id, proj.root, clip_id), code=1)
    return env.read_json(p)


def _load_shots(proj, clip_id: str):
    p = proj.root / "shots" / ("%s.json" % clip_id)
    if p.exists():
        try:
            return env.read_json(p)
        except env.HCError:
            return None
    return None


def needs_shots(proj, clip_id: str, clip: dict, plan: dict, formats, info: dict = None) -> bool:
    """True when the framing of `plan` in one of `formats` reads face data (a track or speaker
    mark, or the per-shot default) that shots/<clip>.json does not hold: the file is missing,
    was made from another source or by an older sampler, or does not cover the source the plan
    plays. Nothing is analysed or written here."""
    if (plan.get("kind") or clip.get("kind") or "cut") == "caption_only":
        return False
    from . import shots as shots_mod
    shots = _load_shots(proj, clip_id)
    wins = []
    for fmt in formats or []:
        wins += framing.face_windows(plan, clip, fmt, shots)
    if not wins:
        return False
    return not shots_mod.covers(shots, wins, info if info is not None else proj.source())


def ensure_shots(proj, clip_id: str, clip: dict, plan: dict, formats, info: dict = None) -> tuple:
    """(shots json or None, analysed). Framing that follows faces needs face samples: without
    them a track mark would hold a static crop at its hint x. When the data is missing or does
    not cover the plan (see needs_shots), the plan's source ranges are analysed first, exactly
    as `hc shots` does, and the shots json and sheet are written."""
    if not needs_shots(proj, clip_id, clip, plan, formats, info):
        return _load_shots(proj, clip_id), False
    from . import shots as shots_mod
    info = info if info is not None else proj.source()
    lock = proj.root / "work" / clip_id / ".shots.lock"
    with fsutil.file_lock(lock, what="shots analysis of %s" % clip_id):
        if not needs_shots(proj, clip_id, clip, plan, formats, info):    # another process did it
            return _load_shots(proj, clip_id), False
        try:
            shots_mod.analyze(proj, clip_id, ranges=shots_mod.plan_ranges(
                plan, float(info.get("duration") or 0.0)))
        except env.HCError as exc:
            raise env.HCError("%s is framed by faces (a track or speaker mark, or no mark yet) and "
                              "the shots analysis that finds them failed: %s. Fix that (`hc shots "
                              "%s %s` shows the same error), or frame the clip with fixed or "
                              "letterbox marks only." % (clip_id, exc.msg, proj.root, clip_id),
                              code=exc.code)
    env.log("render", "analysed shots for %s (framing needs face data)" % clip_id)
    return _load_shots(proj, clip_id), True


# ---------------------------------------------------------------- stage A

def _seek_frame(seg_src0: float, fps: Fraction, grid0: float) -> int:
    """Grid frame index to seek to for a segment: SEEK_PREROLL before it, never before 0 s."""
    k_seg = int(round((Fraction(seg_src0).limit_denominator(10 ** 7)
                       - Fraction(grid0).limit_denominator(10 ** 7)) * fps))
    k_min = int(-(-(Fraction(-grid0).limit_denominator(10 ** 7) * fps) // 1))   # ceil
    return max(k_min, k_seg - int(-(-(Fraction(SEEK_PREROLL) * fps) // 1)))


def _sub_chain(inp: str, outp: str, sub: dict, i: int, g0: float, fps: Fraction, video: dict,
               out_wh: tuple, cmd_file, vfr: bool) -> str:
    """One SubSeg's video chain. The input was seeked to the grid time g0, so the decoded
    timeline starts on the source frame grid. `fps` (round=down, after a quarter-frame lean)
    maps every slot to the source frame on screen at that grid time BEFORE any trim, so VFR
    sources (dropped frames, static gaps) keep the frame that is showing at the cut instead of
    pulling the next one back to it; then trim by frame index is exact."""
    num, den = fps.numerator, fps.denominator
    q = 0.25 / float(fps)
    n = int(sub["frames"])
    fa = int(round((Fraction(sub["src"][0]).limit_denominator(10 ** 7)
                    - Fraction(g0).limit_denominator(10 ** 7)) * fps))
    ow, oh = out_wh
    sw, sh = framing.display_size(video)
    color = _color_args(video, bool(_hdr_chain(video)))
    pre = ("%ssetpts=PTS+%.6f/TB,fps=%d/%d:round=down:start_time=0,"
           "trim=start_frame=%d:end_frame=%d,setpts=PTS-STARTPTS" % (inp, q, num, den, fa, fa + n))
    if framing.sar_of(video) != 1:
        pre += ",scale=%d:%d:flags=lanczos,setsar=1" % (sw, sh)
    hdr = _hdr_chain(video)
    if hdr:
        pre += "," + hdr
    tail = ",setsar=1,trim=end_frame=%d" % n
    if sub["mode"] == "crop":
        cw, ch = sub["size"]
        cmds = sub.get("cmds") or []
        x0, y0 = (cmds[0][1], cmds[0][2]) if cmds else ((sw - cw) // 4 * 2, (sh - ch) // 4 * 2)
        sc = "sendcmd=f=%s," % _ff_escape(str(cmd_file)) if (cmd_file and len(cmds) > 1) else ""
        return ("%s,%scrop@c%d=w=%d:h=%d:x=%d:y=%d,scale=%d:%d:flags=lanczos%s,format=yuv420p%s%s"
                % (pre, sc, i, cw, ch, x0, y0, ow, oh, color, tail, outp))
    # letterbox / full: the active picture (or a mark's rect) over a blurred, dimmed fill of it
    rect = sub.get("rect")
    if rect and tuple(int(v) for v in rect) != (0, 0, sw, sh):
        rx, ry, rw, rh = (int(v) for v in rect)
        pre += ",crop=%d:%d:%d:%d" % (rw, rh, rx, ry)
        sw, sh = rw, rh
    if sub["mode"] == "letterbox":
        fw, fh = ow, max(2, int(round(ow * sh / float(sw) / 2)) * 2)
        if fh > oh:
            fw, fh = max(2, int(round(oh * sw / float(sh) / 2)) * 2), oh
        yc = float(sub.get("y_center") or oh / 2)
    else:
        if Fraction(sw, sh) >= Fraction(ow, oh):
            fw, fh = ow, max(2, int(round(ow * sh / float(sw) / 2)) * 2)
        else:
            fw, fh = max(2, int(round(oh * sw / float(sh) / 2)) * 2), oh
        yc = oh / 2.0
    if (fw, fh) == (ow, oh):
        return "%s,scale=%d:%d:flags=lanczos%s,format=yuv420p%s%s" % (pre, ow, oh, color, tail, outp)
    fy = int(min(max(0.0, yc - fh / 2.0), oh - fh)) // 2 * 2
    fx = (ow - fw) // 4 * 2
    k = max(0.0, min(1.0, 1.0 - float(sub.get("dim", 0.35))))
    sigma = max(1.0, float(sub.get("blur", 28)) / 4.0)
    bw, bh = framing.even_floor(ow / 4), framing.even_floor(oh / 4)
    lut = "lutyuv=y='16+(val-16)*%.4f':u='128+(val-128)*%.4f':v='128+(val-128)*%.4f'" % (k, k, k)
    return ";".join([
        "%s,split=2[lbb%d][lbf%d]" % (pre, i, i),
        "[lbb%d]scale=%d:%d:force_original_aspect_ratio=increase:flags=bilinear%s,crop=%d:%d,"
        "gblur=sigma=%.2f,scale=%d:%d:flags=bicubic,format=yuv420p,%s[lbg%d]"
        % (i, bw, bh, color, bw, bh, sigma, ow, oh, lut, i),
        "[lbf%d]scale=%d:%d:flags=lanczos%s,format=yuv420p[lfg%d]" % (i, fw, fh, color, i),
        "[lbg%d][lfg%d]overlay=x=%d:y=%d:format=yuv420%s%s" % (i, i, fx, fy, tail, outp),
    ])


def _audio_chain(aidx: int, seg: dict, last: bool, s0: float, audio: dict) -> str:
    s, e = float(seg["src"][0]), float(seg["src"][1])
    ns = int(seg["samples"])
    off, end = s - s0, e - s0
    fin = min(ns // 2, int(round(float(seg.get("fade_in") or 0.0) * SR)))
    fo = float(seg.get("fade_out") or 0.0)
    if last:
        fo = max(fo, FINAL_FADE_OUT)
    fout = min(ns // 2, int(round(fo * SR)))
    g = float(seg.get("gain_db") or 0.0)
    ch = ("[0:%d]atrim=start=%.6f:end=%.6f,asetpts=PTS-%.6f/TB,aresample=%d:first_pts=0,%s"
          % (aidx, off, end, off, SR, _pan(audio)))
    if abs(g) > 1e-6:
        ch += ",volume=%.3fdB" % g
    ch += ",apad=whole_len=%d,atrim=end_sample=%d" % (ns, ns)
    if fin > 0:
        ch += ",afade=t=in:ss=0:ns=%d" % fin
    if fout > 0:
        ch += ",afade=t=out:ss=%d:ns=%d" % (ns - fout, fout)
    return ch + "[aout]"


def _probe_counts(path) -> tuple:
    """(video frames, audio samples) of a stage A file from its packets (pcm_s24le stereo)."""
    d = env.ffprobe_json(path, "-show_entries", "packet=stream_index,size:stream=index,codec_type,channels")
    kinds = {int(s["index"]): (s.get("codec_type"), int(s.get("channels") or 2))
             for s in d.get("streams") or []}
    frames, abytes, chans = 0, 0, 2
    for p in d.get("packets") or []:
        kind, ch = kinds.get(int(p["stream_index"]), (None, 2))
        if kind == "video":
            frames += 1
        elif kind == "audio":
            abytes += int(p["size"])
            chans = ch
    return frames, abytes // (3 * max(1, chans))


def _segment_key(fp: str, seg: dict, subs: list, fmt: str, out_wh: tuple, last: bool, audio: dict,
                 video: dict) -> dict:
    return {
        "fp": fp, "code": _code_hash(CODE_FILES), "fmt": fmt, "out": list(out_wh),
        "seg": {k: seg.get(k) for k in ("src", "frames", "samples", "fade_in", "fade_out", "gain_db")},
        "last": last, "subs": [framing.render_view(x) for x in subs],
        "audio": {k: audio.get(k) for k in ("index", "channels", "fold_mono", "channel_db", "start")},
        "video": {k: video.get(k) for k in ("index", "width", "height", "sar", "color_space",
                                            "color_range", "pix_fmt", "hdr", "start", "vfr",
                                            "field_order")},
    }


def _audio_key(fp: str, seg: dict, last: bool, audio: dict) -> dict:
    return {"fp": fp, "code": _code_hash(CODE_FILES), "last": last,
            "seg": {k: seg.get(k) for k in ("src", "samples", "fade_in", "fade_out", "gain_db")},
            "audio": {k: audio.get(k) for k in ("index", "channels", "fold_mono", "channel_db", "start")}}


def segment_entries(proj, clip_id: str, fmt: str, plan: dict, tracks: list, media, info: dict) -> list:
    """Cache keys and paths of every stage A segment (nothing is encoded here)."""
    fps = framing.fps_of(plan)
    num, den = fps.numerator, fps.denominator
    out_wh = framing.out_size(fmt)
    video, audio = info["video"], info["audio"]
    if not audio:
        raise env.HCError("the source has no audio stream", code=1)
    fp = info.get("fingerprint") or env.fingerprint(media)
    wd = proj.work_dir(clip_id, fmt)
    segs = plan.get("segments") or []
    if not segs:
        raise env.HCError("plan for %s has no segments" % clip_id, code=1)
    out = []
    for k, seg in enumerate(segs):
        seg = dict(seg)
        n = int(seg["frames"])
        if n <= 0:
            raise env.HCError("plan segment %d of %s has no frames" % (k, clip_id), code=1)
        seg.setdefault("samples", _samples_for(n, num, den))
        last = k == len(segs) - 1
        subs = tracks[k]
        if sum(int(x["frames"]) for x in subs) != n:
            raise env.HCError("framing for segment %d covers %d frames, plan says %d"
                              % (k, sum(int(x["frames"]) for x in subs), n), code=3)
        key = env.stable_hash(_segment_key(fp, seg, subs, fmt, out_wh, last, audio, video))
        akey = env.stable_hash(_audio_key(fp, seg, last, audio))
        out.append({"k": k, "path": str(wd / ("seg_%s.mkv" % key[:20])), "key": key, "akey": akey,
                    "frames": n, "samples": int(seg["samples"]), "src": seg["src"], "cached": True,
                    "_seg": seg, "_subs": subs, "_last": last})
    return out


def stage_a(proj, clip_id: str, fmt: str, plan: dict, tracks: list, media, info: dict,
            force: bool = False, entries: list = None) -> list:
    """Encode (or reuse) every segment → [{"k", "path", "key", "akey", "frames", "samples", "src",
    "cached"}] in plan order."""
    entries = entries or segment_entries(proj, clip_id, fmt, plan, tracks, media, info)
    fps = framing.fps_of(plan)
    out_wh = framing.out_size(fmt)
    wd = proj.work_dir(clip_id, fmt)
    for e in entries:
        path = Path(e["path"])
        if force or not path.exists():
            e["cached"] = False
            e["cmd"], warns = _encode_segment(media, info, e["_seg"], e["_subs"], e["k"], e["_last"],
                                              fps, out_wh, wd, e["key"], path)
            e["warnings"] = ["segment %d: %s" % (e["k"], w) for w in warns]
    return entries


def _encode_segment(media, info: dict, seg: dict, subs: list, k: int, last: bool, fps: Fraction,
                    out_wh: tuple, wd: Path, key: str, path: Path) -> list:
    video, audio = info["video"], info["audio"]
    s, e = float(seg["src"][0]), float(seg["src"][1])
    grid0 = float(video.get("start") or 0.0)
    s0 = float(Fraction(grid0).limit_denominator(10 ** 7)
               + _seek_frame(s, fps, grid0) / fps)
    s0 = max(0.0, s0)
    dur = e - s0 + TAIL_READ
    vfr = bool(video.get("vfr"))
    parts, labels = [], []
    m = len(subs)
    vin = "[0:%d]" % int(video["index"])
    dein = _deinterlace(video)
    if dein:
        parts.append("%s%snull[vdi]" % (vin, dein))
        vin = "[vdi]"
    srcs = [vin] if m == 1 else ["[sp%d]" % i for i in range(m)]
    if m > 1:
        parts.append("%ssplit=%d%s" % (vin, m, "".join(srcs)))
    for i, sub in enumerate(subs):
        cmd_file = None
        if sub["mode"] == "crop" and len(sub.get("cmds") or []) > 1:
            cmd_file = wd / ("cmds_%s_%d.txt" % (key[:20], i))
            cmd_file.write_text(framing.sendcmd_text(sub, "crop@c%d" % i, fps), encoding="utf-8")
        lab = "[vs%d]" % i
        parts.append(_sub_chain(srcs[i], lab, sub, i, s0, fps, video, out_wh, cmd_file, vfr))
        labels.append(lab)
    if m > 1:
        parts.append("%sconcat=n=%d:v=1:a=0[vout]" % ("".join(labels), m))
        vlab = "[vout]"
    else:
        vlab = labels[0]
    parts.append(_audio_chain(int(audio["index"]), seg, last, s0, audio))
    tmp = fsutil.tmp_path(path)
    args = ["-ss", "%.6f" % s0, "-t", "%.6f" % dur, "-i", str(media),
            "-filter_complex", ";".join(parts), "-map", vlab, "-map", "[aout]",
            "-map_metadata", "-1", "-map_chapters", "-1",
            *LOSSLESS_V, *LOSSLESS_A, "-ac", "2", "-f", "matroska", str(tmp)]
    try:
        res = env.ff(args, capture=False)
        frames, samples = _probe_counts(tmp)
        if frames != int(seg["frames"]) or samples != int(seg["samples"]):
            hint = ""
            if vfr:
                hint = (" (the source is VFR: re-ingest it as constant frame rate with "
                        "`hc ingest P SRC --cfr`, then re-plan)")
            raise env.HCError("stage A segment %d (src %.3f-%.3f) came out %d frames / %d samples, "
                              "expected %d / %d%s" % (k, s, e, frames, samples, int(seg["frames"]),
                                                      int(seg["samples"]), hint), code=3)
        os.replace(str(tmp), str(path))
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
    return args, list(res.warnings or [])


def _group(paths: list, fps: Fraction, wd: Path, force: bool) -> list:
    """Pre-join > CONCAT_GROUP segments losslessly in groups (contract 9.2)."""
    if len(paths) <= CONCAT_GROUP:
        return [Path(p["path"]) for p in paths]
    out = []
    num, den = fps.numerator, fps.denominator
    for g in range(0, len(paths), CONCAT_GROUP):
        part = paths[g:g + CONCAT_GROUP]
        key = env.stable_hash([p["key"] for p in part])
        path = wd / ("grp_%s.mkv" % key[:20])
        if force or not path.exists():
            n = len(part)
            ins = []
            for p in part:
                ins += ["-i", p["path"]]
            graph = ";".join("[%d:v]settb=expr=%d/%d,setpts=N[v%d];[%d:a]asetpts=N/SR/TB[a%d]"
                             % (i, den, num, i, i, i) for i in range(n))
            graph += ";%sconcat=n=%d:v=1:a=1[vc][ac];[vc]settb=expr=%d/%d,setpts=N[vo]" % (
                "".join("[v%d][a%d]" % (i, i) for i in range(n)), n, den, num)
            tmp = fsutil.tmp_path(path)
            try:
                env.ff(ins + ["-filter_complex", graph, "-map", "[vo]", "-map", "[ac]",
                              *LOSSLESS_V, *LOSSLESS_A, "-ac", "2", "-f", "matroska", str(tmp)],
                       capture=False)
                frames, samples = _probe_counts(tmp)
                want_f = sum(p["frames"] for p in part)
                want_s = sum(p["samples"] for p in part)
                if (frames, samples) != (want_f, want_s):
                    raise env.HCError("segment group came out %d frames / %d samples, expected "
                                      "%d / %d" % (frames, samples, want_f, want_s), code=3)
                os.replace(str(tmp), str(path))
            finally:
                if tmp.exists():
                    tmp.unlink()
        out.append(path)
    return out


# ---------------------------------------------------------------- loudness and luma

def _clamp(v, lo, hi, default):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    if f != f or f in (float("inf"), float("-inf")):
        return default
    return max(lo, min(hi, f))


def measure_loudness(paths: list, cache: Path) -> dict:
    """loudnorm first pass over the concatenated audio (cached json)."""
    if cache.exists():
        try:
            return env.read_json(cache)
        except env.HCError:
            pass
    ins = []
    for p in paths:
        ins += ["-i", str(p)]
    n = len(paths)
    prefix = "%sconcat=n=%d:v=0:a=1[hc_cat]" % ("".join("[%d:a]" % i for i in range(n)), n)
    try:
        from . import audio as audio_mod
        fn = getattr(audio_mod, "loudnorm_measure", None)
    except ImportError:
        fn = None
    if fn is not None:
        meas = fn(ins, prefix)
    else:
        i, tp, lra = LOUDNORM
        res = env.ff(["-loglevel", "info", "-nostats"] + ins +
                     ["-filter_complex", "%s;[hc_cat]loudnorm=I=%s:TP=%s:LRA=%s:print_format=json[hc_ln]"
                      % (prefix, i, tp, lra), "-map", "[hc_ln]", "-f", "null", "-"])
        meas = _parse_loudnorm(res.stderr)
    env.write_json(cache, meas)
    return meas


def _parse_loudnorm(text: str) -> dict:
    import json
    import re
    for blk in reversed(re.findall(r"\{[^{}]*\}", text or "")):
        try:
            raw = json.loads(blk)
        except ValueError:
            continue
        if "input_i" in raw:
            return raw
    raise env.HCError("loudnorm printed no measurement", code=3)


def loudnorm_filter(meas: dict) -> str:
    i, tp, lra = LOUDNORM
    return ("loudnorm=I=%s:TP=%s:LRA=%s:measured_I=%.2f:measured_TP=%.2f:measured_LRA=%.2f:"
            "measured_thresh=%.2f:offset=%.2f:linear=true"
            % (i, tp, lra, _clamp(meas.get("input_i"), -99, 0, -70.0),
               _clamp(meas.get("input_tp"), -99, 99, -70.0),
               _clamp(meas.get("input_lra"), 0, 99, 0.0),
               _clamp(meas.get("input_thresh"), -99, 0, -80.0),
               _clamp(meas.get("target_offset"), -99, 99, 0.0)))


def luma_track(paths: list, fmt: str, fps: Fraction, cache: Path) -> np.ndarray:
    """Concatenated base decoded at 5 fps as gray uint8 [n, h, w] (cached .npy)."""
    if cache.exists():
        try:
            return np.load(str(cache))
        except (OSError, ValueError):
            pass
    lw, lh = LUMA_SIZE.get(fmt, LUMA_SIZE["vertical"])
    num, den = fps.numerator, fps.denominator
    ins = []
    for p in paths:
        ins += ["-i", str(p)]
    n = len(paths)
    graph = "".join("[%d:v]settb=expr=%d/%d,setpts=N[l%d];" % (i, den, num, i) for i in range(n))
    graph += ("%sconcat=n=%d:v=1:a=0,settb=expr=%d/%d,setpts=N,fps=%d,"
              "scale=%d:%d:flags=area:out_range=pc,format=gray[luma]"
              % ("".join("[l%d]" % i for i in range(n)), n, den, num, LUMA_FPS, lw, lh))
    res = env.ff(ins + ["-filter_complex", graph, "-map", "[luma]", "-f", "rawvideo",
                        "-pix_fmt", "gray", "-"])
    fb = lw * lh
    arr = np.frombuffer(res.stdout[: len(res.stdout) // fb * fb], dtype=np.uint8).reshape(-1, lh, lw)
    cache.parent.mkdir(parents=True, exist_ok=True)
    tmp = fsutil.tmp_path(cache)
    try:
        with open(str(tmp), "wb") as fh:
            np.save(fh, arr)
        os.replace(str(tmp), str(cache))
    finally:
        if tmp.exists():
            tmp.unlink()
    return arr


# ---------------------------------------------------------------- stage B

def _dips(plan: dict, fps: Fraction) -> list:
    out = []
    for j in plan.get("joins") or []:
        if j.get("kind") != "dip":
            continue
        jf = Fraction(int(round(float(j["out"]) * fps)), 1) / fps
        out.append((float(jf) - DIP_HALF, float(jf), float(jf) + DIP_HALF))
    return out


def _caption_track(plan, clip, preset, fmt, fps, luma, proj):
    """captions.build_track (with `captions: false` it holds only the hook card), or None when
    there is nothing to draw or the module is unavailable."""
    try:
        from . import captions as capmod
    except ImportError as exc:
        return None, "captions module unavailable (%s)" % exc
    track = capmod.build_track(plan, clip, preset, fmt, framing.out_size(fmt),
                               (fps.numerator, fps.denominator), luma, project_root=proj.root)
    empty = not (getattr(track, "pages", None) or getattr(track, "hook", None))
    if int(getattr(track, "band_h", 0) or 0) <= 0 or empty:
        if clip.get("captions") is False:
            return None, "captions off for this clip"
        return None, "no caption pages or hook card in this clip"
    return track, None


def stage_b(proj, clip_id: str, fmt: str, style: str, plan: dict, clip: dict, preset: dict,
            inputs: list, meas: dict, track, out_path: Path, log_txt: Path,
            log_path: Path = None) -> dict:
    """Encode the final file. The new file is verified while it is still the temp file; then
    the previous render log (`log_path`) is removed BEFORE the new file takes the MP4's name.
    So a failed verification leaves the previous MP4 and its log as they were, and an
    interruption anywhere between the rename and the new log leaves an MP4 with no log, which
    nothing calls up to date (a log that described the previous picture would)."""
    fps = framing.fps_of(plan)
    num, den = fps.numerator, fps.denominator
    ow, oh = framing.out_size(fmt)
    frames_out = int(plan.get("frames_out") or 0)
    n = len(inputs)
    args = []
    for p in inputs:
        args += ["-i", str(p)]
    g = ["[%d:v]settb=expr=%d/%d,setpts=N[v%d];[%d:a]asetpts=N/SR/TB[a%d]" % (i, den, num, i, i, i)
         for i in range(n)]
    g.append("%sconcat=n=%d:v=1:a=1[vc][ac]" % ("".join("[v%d][a%d]" % (i, i) for i in range(n)), n))
    g.append("[vc]settb=expr=%d/%d,setpts=N[b0]" % (den, num))
    cur = "[b0]"
    if track is not None:
        args += ["-f", "rawvideo", "-pix_fmt", "rgba", "-s", "%dx%d" % (ow, int(track.band_h)),
                 "-framerate", "%d/%d" % (num, den), "-i", "pipe:0"]
        g.append("[%d:v]scale=out_color_matrix=bt709:out_range=tv,format=yuva420p[strip]" % n)
        g.append("%s[strip]overlay=x=0:y=%d:eof_action=pass:format=yuv420[b1]" % (cur, int(track.band_y)))
        cur = "[b1]"
    dur = float(Fraction(frames_out * den, num)) + 1.0
    for i, (ts, tj, te) in enumerate(_dips(plan, fps)):
        g.append("color=c=black@%.2f:s=%dx%d:r=%d/%d:d=%.6f,format=yuva420p,"
                 "fade=t=in:st=%.6f:d=%.3f:alpha=1,fade=t=out:st=%.6f:d=%.3f:alpha=1[dc%d]"
                 % (DIP_ALPHA, ow, oh, num, den, dur, max(0.0, ts), DIP_HALF, tj, DIP_HALF, i))
        g.append("%s[dc%d]overlay=x=0:y=0:eof_action=pass:format=yuv420:enable='between(t,%.6f,%.6f)'[bd%d]"
                 % (cur, i, max(0.0, ts), te, i))
        cur = "[bd%d]" % i
    g.append("%sformat=yuv420p,setparams=color_primaries=bt709:color_trc=bt709:colorspace=bt709:"
             "range=tv[vout]" % cur)
    # loudnorm (192 kHz inside) stamps the frames it flushes from its 3 s look-ahead at EOF with
    # a jump (measured: an 80 ms hole 2.9 s before the end, so the tail played late). Re-stamp
    # by sample count after it and pin the exact length before the single dithered s16 step.
    n_samples = int(round(frames_out * SR * den / float(num)))
    g.append("[ac]%s,aresample=%d,asetpts=N/SR/TB,apad=whole_len=%d,atrim=end_sample=%d,"
             "aresample=%d:osf=s16:dither_method=triangular_hp[aout]"
             % (loudnorm_filter(meas), SR, n_samples, n_samples, SR))
    tmp = fsutil.tmp_path(out_path)
    args += ["-filter_complex", ";".join(g), "-map", "[vout]", "-map", "[aout]",
             *FINAL_V, "-r", "%d/%d" % (num, den), *FINAL_A, "-ac", "2",
             "-map_metadata", "-1", "-map_chapters", "-1",
             "-metadata", "comment=hook-cutter %s" % env.code_version(),
             "-movflags", "+faststart", "-f", "mp4", str(tmp)]
    t0 = time.time()
    try:
        res = _run_stage_b(args, track, frames_out, ow, log_txt)
        ver = _verify_final(tmp, plan)
        if log_path is not None and Path(log_path).exists():
            Path(log_path).unlink()
        os.replace(str(tmp), str(out_path))
    finally:
        if tmp.exists():
            tmp.unlink()
    return {"args": args, "warnings": list(res.warnings or []), "seconds": round(time.time() - t0, 2),
            "verify": ver}


def _run_stage_b(args: list, track, frames_out: int, ow: int, log_txt: Path):
    if track is None:
        res = env.ff(args, capture=False, log_to=log_txt)
    else:
        blank = bytes(ow * int(track.band_h) * 4)
        tf = int(track.frames)
        if tf and len(track.frame(0)) != len(blank):
            raise env.HCError("caption strip frames are %d bytes, expected %d (%dx%d RGBA)"
                              % (len(track.frame(0)), len(blank), ow, int(track.band_h)), code=3)
        proc = env.ff_popen(args, log_txt)
        try:
            for k in range(frames_out):
                proc.stdin.write(track.frame(k) if k < tf else blank)
        except (BrokenPipeError, OSError):
            pass                                 # ffmpeg exited early: ff_wait reports why
        except BaseException:
            proc.kill()
            proc.wait()
            fh = getattr(proc, "hc_log_file", None)
            if fh is not None:
                fh.close()
            raise
        res = env.ff_wait(proc, log_txt)
    return res


def audio_md5(path) -> str:
    res = env.ff(["-i", str(path), "-map", "0:a:0", "-c:a", "pcm_s16le", "-f", "md5", "-"])
    txt = res.text.strip()
    return txt.split("=", 1)[1] if "=" in txt else txt


def _verify_final(path, plan: dict) -> dict:
    d = env.ffprobe_json(path, "-count_packets", "-show_entries",
                         "stream=codec_type,nb_read_packets,duration,width,height,r_frame_rate")
    info = {}
    for s in d.get("streams") or []:
        info[s.get("codec_type")] = s
    v = info.get("video") or {}
    frames = int(v.get("nb_read_packets") or 0)
    want = int(plan.get("frames_out") or 0)
    if frames != want:
        raise env.HCError("final render has %d frames, plan says %d" % (frames, want), code=3)
    a = info.get("audio") or {}
    return {"frames": frames, "v_duration": float(v.get("duration") or 0),
            "a_duration": float(a.get("duration") or 0)}


# ---------------------------------------------------------------- sheets

def contact_frames(frames: int) -> list:
    """Frame indices of the contact sheet: cell 0 is frame 0 (the hook card and the default
    thumbnail a platform shows), the other cells sit at (2i+1)/22 of the clip."""
    idx = [0]
    rest = CONTACT_FRAMES - 1
    for i in range(rest):
        n = int(Fraction(2 * i + 1, 2 * rest) * frames)
        idx.append(min(max(0, n), max(0, frames - 1)))
    return idx


def contact_sheet(proj, clip_id: str, fmt: str, style: str, path, plan: dict) -> Path:
    fps = framing.fps_of(plan)
    frames = int(plan.get("frames_out") or 0)
    ow, oh = framing.out_size(fmt)
    idx = contact_frames(frames)
    imgs = ru.grab_frames(path, idx, CELL_W[fmt], size=(ow, oh))
    items = []
    for c, n in enumerate(idx):
        if n not in imgs:
            continue
        label = "%s  f%d" % (ru.tc(float(n / fps)), n)
        if c == 0:
            label += " (frame 0: hook card / default thumbnail)"
        items.append((imgs[n], label))
    out = proj.sheet_path("%s__%s__%s.contact.jpg" % (clip_id, fmt, style))
    return ru.contact_sheet(items, CONTACT_COLS, out, CELL_W[fmt])


def seam_label(k: int, t: float, join: dict) -> str:
    """Caption under one seam pair. The pair is cut from stage A segments: the base picture
    before captions and before a dip join's darkening, so a dip says that it is not shown."""
    kind = join.get("kind", "cut")
    label = "#%d  %s  %s" % (k, ru.tc(t), kind)
    return label + " (darkening not shown)" if kind == "dip" else label


def seams_sheet(proj, clip_id: str, fmt: str, plan: dict, segs: list) -> Path:
    """Every join: the base picture just before and just after it (last frame of one stage A
    segment, first frame of the next), before captions and before any dip darkening."""
    ow, oh = framing.out_size(fmt)
    cw = CELL_W[fmt]
    joins = {int(j["k"]): j for j in plan.get("joins") or [] if "k" in j}
    want = {}
    for k in range(1, len(segs)):
        want.setdefault(k - 1, set()).add(segs[k - 1]["frames"] - 1)
        want.setdefault(k, set()).add(0)
    grabbed = {}
    for k, idxs in want.items():
        grabbed[k] = ru.grab_frames(segs[k]["path"], idxs, cw, size=(ow, oh))
    items = []
    for k in range(1, len(segs)):
        a = grabbed.get(k - 1, {}).get(segs[k - 1]["frames"] - 1)
        b = grabbed.get(k, {}).get(0)
        if a is None or b is None:
            continue
        j = joins.get(k, {})
        t = float(j.get("out", (plan["segments"][k].get("out") or [0])[0]))
        pair = ru.seam_pair(a, b, "seg %d last" % (k - 1), "seg %d first" % k)
        items.append((pair, seam_label(k, t, j)))
    out = proj.sheet_path("%s__%s.seams.jpg" % (clip_id, fmt))
    if not items:
        return ru.contact_sheet([(ru.placeholder("no joins in this clip"), "")], 1, out, 540,
                                title="%s %s: seams" % (clip_id, fmt))
    cols = 3 if fmt == "vertical" else 2
    return ru.contact_sheet(items, cols, out, items[0][0].width,
                            title="%s %s: seams (last frame before | first frame after)"
                            % (clip_id, fmt))


# ---------------------------------------------------------------- public API

def segment_paths(project, clip_id: str, fmt: str, style: str = None, plan: dict = None,
                  clip: dict = None):
    """Ordered stage A segment files of clip x fmt, or None.

    With `style` and `plan`: the files of exactly that plan and style, found by their content
    keys (source fingerprint, segment, crop tracks, letterbox fill, code), so a file that exists
    holds the right picture. None when any is missing (the clip was not rendered from this plan
    in this style) or the inputs cannot be resolved. No render log is read: a log can describe
    an older plan, or another style whose framing differs.

    Without them: the segments of the newest render log of clip x fmt whose files all exist,
    whatever plan or style made it (kept for callers that only want "a" rendered picture)."""
    proj = ru.as_project(project)
    if style is not None and plan is not None:
        try:
            clip = clip or ru.clip_entry(proj, clip_id)
            plan = _with_frames(plan)
            media, info = _media(proj, plan, clip)
            segs = _render_inputs(proj, clip_id, fmt, style, plan, clip, media, info,
                                  _load_shots(proj, clip_id))["segs"]
        except (env.HCError, KeyError, TypeError, ValueError):
            return None
        paths = [Path(sg["path"]) for sg in segs]
        return paths if paths and all(q.exists() for q in paths) else None
    logs = sorted(proj.root.glob("renders/%s__%s__*.log.json" % (clip_id, fmt)),
                  key=lambda p: p.stat().st_mtime, reverse=True)
    for lp in logs:
        try:
            data = env.read_json(lp)
        except env.HCError:
            continue
        paths = [proj.root / s["path"] for s in data.get("segments") or []]
        if paths and all(p.exists() for p in paths):
            return paths
    return None


WORK_PATTERNS = ("seg_*.mkv", "grp_*.mkv", "cmds_*.txt", "luma-*.npy", "loudnorm-*.json", "*.tmp.*")


def prune_work(proj, clip_id: str, fmt: str) -> int:
    """Delete cached work files of clip x fmt that no render log references any more."""
    keep = set()
    for lp in proj.root.glob("renders/%s__%s__*.log.json" % (clip_id, fmt)):
        try:
            data = env.read_json(lp)
        except env.HCError:
            return 0                      # unreadable log: keep everything
        for rel in data.get("work_files") or []:
            keep.add((proj.root / rel).resolve())
    wd = proj.root / "work" / clip_id / fmt
    removed = 0
    for pat in WORK_PATTERNS:
        for f in wd.glob(pat):
            if f.resolve() not in keep and f.is_file():
                try:
                    f.unlink()
                    removed += 1
                except OSError:
                    pass
    return removed


# Plan and clips.json keys that never change a render's bytes: a new cover headline, post copy,
# title or another clip's range must not make a render stale (G0) or re-render it. A clip's own
# `transcript_sha` is a pin on its caption-only transcript (plan lints it): writing or updating
# the pin changes no frame.
PLAN_NOT_RENDERED = ("lints", "created", "spec_hash", "set_hash", "transcript_sha")
CLIP_NOT_RENDERED = ("cover", "post", "title", "ledger", "transcript_sha")


def _plan_core(plan: dict) -> dict:
    return {k: v for k, v in plan.items() if k not in PLAN_NOT_RENDERED}


def _with_frames(plan: dict) -> dict:
    frames_out = int(plan.get("frames_out") or
                     sum(int(s["frames"]) for s in plan.get("segments") or []))
    return dict(plan, frames_out=frames_out)


def plan_hash(plan: dict) -> str:
    """The render log's `hashes.plan` for this plan (what a render made from it records)."""
    return env.stable_hash(_plan_core(_with_frames(plan)))


def _assets(preset: dict, clip: dict, fmt: str, proj):
    """The content of the files a render draws with that the preset only NAMES: the bytes of
    the caption, emphasis and hook-card fonts and the margins of the safe-zone profile in use
    (captions.render_assets, which never raises). Swapping a font file or editing
    safezones.json therefore moves the render key like a preset edit does."""
    try:
        from . import captions as capmod
    except ImportError:
        return None
    fn = getattr(capmod, "render_assets", None)
    return fn(preset, clip, fmt, proj.root) if fn is not None else None


def _render_inputs(proj, clip_id: str, fmt: str, style: str, plan: dict, clip: dict, media,
                   info: dict, shots) -> dict:
    """Everything that decides a render's bytes, and its hash (the log's `hashes.render`)."""
    preset = ru.load_preset(style, proj, fmt)
    read = {}
    tracks = framing.build_tracks(plan, clip, shots, info, fmt, preset, read)
    segs = segment_entries(proj, clip_id, fmt, plan, tracks, media, info)
    akey = env.stable_hash([s["akey"] for s in segs])[:20]
    vkey = env.stable_hash([s["key"] for s in segs])[:20]
    # `faces`: the face samples the framing trackers read (shots json). The crop moves they
    # produce are in `segs` already; hashing the samples themselves makes the key follow the
    # face data even where a dead zone absorbs the difference.
    rkey = env.stable_hash({"plan": _plan_core(plan), "clip": {k: v for k, v in clip.items()
                                                               if not k.startswith("_")
                                                               and k not in CLIP_NOT_RENDERED},
                            "preset": preset, "segs": vkey, "audio": akey, "fmt": fmt,
                            "style": style, "faces": read.get("faces"),
                            "assets": _assets(preset, clip, fmt, proj),
                            "code": _code_hash(CODE_FILES + CAPTION_FILES)})
    return {"preset": preset, "tracks": tracks, "segs": segs, "akey": akey, "vkey": vkey,
            "rkey": rkey, "faces": read.get("faces"), "trackers": read.get("trackers", 0)}


def expected_hashes(project, clip_id: str, fmt: str, style: str, *, plan: dict = None,
                    clip: dict = None) -> dict:
    """{"plan", "render"}: the hashes a render of clip x fmt x style made NOW would record, so a
    render log whose hashes differ is stale. `plan` defaults to the project's current plan
    (rebuilt in memory when stale, never written; FAIL lints allowed); `clip` to the clips.json
    entry. The render hash follows the face samples the framing reads from shots/<clip>.json
    (the plan's shots_hash, the crop tracks and the samples themselves), so a render made
    before that data existed or changed is stale. Nothing is analysed or written here."""
    proj = ru.as_project(project)
    if clip is None:
        clip = ru.clip_entry(proj, clip_id)
    if plan is None:
        from . import plan as plan_mod
        plan = plan_mod.current_plan(proj, clip_id)
    framing.out_size(fmt)
    plan = _with_frames(plan)
    media, info = _media(proj, plan, clip)
    k = _render_inputs(proj, clip_id, fmt, style, plan, clip, media, info,
                       _load_shots(proj, clip_id))
    return {"plan": plan_hash(plan), "render": k["rkey"]}


def render_clip(project, clip_id: str, formats=None, styles=None, force: bool = False, *,
                plan: dict = None, clip: dict = None) -> list:
    """Render one clip for each format x style → [{"clip", "format", "style", "path", "duration",
    "frames"}]. `plan` / `clip` override the project's plan and clips.json entry (tests)."""
    proj = ru.as_project(project)
    if clip is None:
        clip = ru.clip_entry(proj, clip_id)
    plan_given = plan is not None
    if plan is None:
        plan = _load_plan(proj, clip_id)
    eff = clip.get("_eff") or {}
    formats = _list(formats) or clip.get("formats") or eff.get("formats") or ["vertical"]
    styles = _list(styles) or clip.get("styles") or eff.get("styles") or ["clean"]
    for f in formats:
        framing.out_size(f)
    media, info = _media(proj, plan, clip)
    # Face data first: a track mark with no samples would silently hold a static crop.
    shots, analysed = ensure_shots(proj, clip_id, clip, _with_frames(plan), formats, info)
    if analysed and not plan_given:
        plan = _load_plan(proj, clip_id)         # the plan records the shots data it was made from
    fps = framing.fps_of(plan)
    plan = _with_frames(plan)
    frames_out = int(plan["frames_out"])
    results = []
    for fmt in formats:
        # One render of a clip x format at a time: the stage A cache, the seams sheet and the
        # prune pass are shared by every style of it. A second process waits here.
        with fsutil.file_lock(proj.work_dir(clip_id, fmt) / ".lock",
                              what="render %s %s" % (clip_id, fmt)):
            results += _render_format(proj, clip_id, fmt, styles, plan, clip, media, info, shots,
                                      force, fps, frames_out, analysed)
    return results


def _render_format(proj, clip_id, fmt, styles, plan, clip, media, info, shots, force, fps,
                   frames_out, shots_analysed: bool = False) -> list:
    results = []
    seams_done = False
    for style in styles:
        t_start = time.time()
        ins = _render_inputs(proj, clip_id, fmt, style, plan, clip, media, info, shots)
        preset, tracks, segs = ins["preset"], ins["tracks"], ins["segs"]
        akey, vkey, rkey = ins["akey"], ins["vkey"], ins["rkey"]
        wd = proj.work_dir(clip_id, fmt)
        out_path = proj.render_path(clip_id, fmt, style)
        log_path = proj.render_path(clip_id, fmt, style, ".log.json")
        cap_on = clip.get("captions") is not False
        if not force and out_path.exists() and log_path.exists():
            try:
                old = env.read_json(log_path)
            except env.HCError:
                old = {}
            if (old.get("hashes") or {}).get("render") == rkey:
                # Up to date: nothing is written or touched (freshness is judged by the hashes
                # in the render log, never by modification times).
                env.log("render", "%s %s %s: up to date" % (clip_id, fmt, style))
                results.append(_entry(clip_id, fmt, style, out_path, plan))
                if not seams_done:
                    seams_done = proj.sheet_path("%s__%s.seams.jpg" % (clip_id, fmt)).exists()
                continue
        ta = time.time()
        segs = stage_a(proj, clip_id, fmt, plan, tracks, media, info, force, entries=segs)
        inputs = _group(segs, fps, wd, force)
        t_a = round(time.time() - ta, 2)
        if not seams_done:
            seams_sheet(proj, clip_id, fmt, plan, segs)
            seams_done = True
        tl = time.time()
        meas = measure_loudness(inputs, wd / ("loudnorm-%s.json" % akey))
        luma = luma_track(inputs, fmt, fps, wd / ("luma-%s.npy" % vkey))
        t_l = round(time.time() - tl, 2)
        track, why = _caption_track(plan, clip, preset, fmt, fps, luma, proj)
        if why and cap_on:
            env.log("render", "%s: rendering without captions: %s" % (clip_id, why))
        log_txt = proj.render_path(clip_id, fmt, style, ".ffmpeg.txt")
        if log_txt.exists():
            log_txt.unlink()
        b = stage_b(proj, clip_id, fmt, style, plan, clip, preset, inputs, meas, track,
                    out_path, log_txt, log_path)
        ver = b["verify"]                        # checked on the temp file, before the rename
        md5 = audio_md5(out_path)
        cap_json = proj.render_path(clip_id, fmt, style, ".captions.json")
        srt = proj.render_path(clip_id, fmt, style, ".srt")
        if track is not None:
            env.write_json(cap_json, track.to_json())
            if cap_on:
                track.write_srt(srt)
            elif srt.exists():
                srt.unlink()                     # hook card only: no spoken-word captions
        else:
            env.write_json(cap_json, {"disabled": True, "reason": why, "band": None, "pages": [],
                                      "hook": None})
            if srt.exists():
                srt.unlink()
        sheet = contact_sheet(proj, clip_id, fmt, style, out_path, plan)
        work = [s["path"] for s in segs] + [str(p) for p in inputs] + \
            [str(wd / ("loudnorm-%s.json" % akey)), str(wd / ("luma-%s.npy" % vkey))] + \
            [str(p) for p in wd.glob("cmds_*.txt")
             if any(p.name.startswith("cmds_%s_" % s["key"][:20]) for s in segs)]
        log = {
            "version": 1, "clip": clip_id, "format": fmt, "style": style,
            "code_version": env.code_version(), "created": env.now_iso(),
            "fps": [fps.numerator, fps.denominator], "size": list(framing.out_size(fmt)),
            "frames": ver["frames"], "duration": round(float(Fraction(frames_out) / fps), 6),
            "v_duration": ver["v_duration"], "a_duration": ver["a_duration"],
            "audio_md5": md5, "captions": track is not None, "captions_note": why,
            "loudnorm": meas, "dips": [list(d) for d in _dips(plan, fps)],
            "segments": [{"k": s["k"], "path": _rel(proj, s["path"]), "frames": s["frames"],
                          "samples": s["samples"], "src": s["src"], "key": s["key"],
                          "cached": s["cached"]} for s in segs],
            "inputs": [_rel(proj, p) for p in inputs],
            "work_files": sorted({_rel(proj, p) for p in work}),
            "framing": framing.summary(tracks),
            "framing_windows": framing.windows(tracks, plan, info),
            "shots": {"used": shots is not None, "analysed_by_render": bool(shots_analysed),
                      "face_tracks": int(ins.get("trackers") or 0), "faces": ins.get("faces")},
            "cmds": {"stage_a": [s.get("cmd") for s in segs if s.get("cmd")],
                     "stage_b": b["args"]},
            "warnings": [w for sg in segs for w in sg.get("warnings") or []] + b["warnings"],
            "timings": {"stage_a": t_a, "loudness_luma": t_l, "stage_b": b["seconds"],
                        "total": round(time.time() - t_start, 2)},
            "hashes": {"render": rkey, "plan": plan_hash(plan),
                       "segments": vkey, "audio": akey},
            "sheets": {"contact": _rel(proj, sheet),
                       "seams": _rel(proj, proj.sheet_path("%s__%s.seams.jpg" % (clip_id, fmt)))},
        }
        env.write_json(log_path, log)
        env.log("render", "%s %s %s: %s (%.2f s, %d frames)"
                % (clip_id, fmt, style, _rel(proj, out_path), log["duration"], ver["frames"]))
        results.append(_entry(clip_id, fmt, style, out_path, plan))
    n = prune_work(proj, clip_id, fmt)
    if n:
        env.log("render", "%s %s: removed %d stale work file%s" % (clip_id, fmt, n, "" if n == 1 else "s"))
    return results


def _entry(clip_id, fmt, style, path, plan) -> dict:
    fps = framing.fps_of(plan)
    frames = int(plan.get("frames_out") or 0)
    return {"clip": clip_id, "format": fmt, "style": style, "path": str(path),
            "duration": round(float(Fraction(frames) / fps), 6), "frames": frames}


def cmd_render(args) -> int:
    proj = ru.as_project(args.project)
    spec = proj.spec()
    ids = list(getattr(args, "clips", None) or []) or list((spec.get("clips") or {}).keys())
    if not ids:
        raise env.HCError("clips.json has no clips to render", code=1)
    for cid in ids:
        if cid not in (spec.get("clips") or {}):
            raise ru.unknown_clip(cid, spec)
    rows = []
    for cid in ids:
        rows += render_clip(proj, cid, formats=getattr(args, "formats", None),
                            styles=getattr(args, "styles", None), force=bool(getattr(args, "force", False)))
    for r in rows:
        print("%-28s %-9s %-10s %7.2f s %6d fr  %s" % (r["clip"], r["format"], r["style"],
                                                      r["duration"], r["frames"], _rel(proj, r["path"])))
    sheets = sorted({"sheets/%s__%s.seams.jpg" % (r["clip"], r["format"]) for r in rows} |
                    {"sheets/%s__%s__%s.contact.jpg" % (r["clip"], r["format"], r["style"]) for r in rows})
    print("look at: " + ", ".join(sheets))
    return 0
