"""Framing marks + face samples -> per-segment crop tracks (contract 9.1).

`build_tracks` turns each plan segment into a list of SubSegs (video-only splits at framing
mark changes; audio is never split here). A SubSeg is
`{"src": [s, e], "frames": n, "mode": "crop"|"letterbox"|"full", "size": [w, h],
"cmds": [[t_rel, x, y], ...], "zoom": z}` where, for crop, `size` is the crop rect in upright
square-pixel source pixels and `cmds` holds the crop's top-left corner from frame 0 on, one
entry per change. Letterbox and full SubSegs carry `rect` = the source area they show (the
source's active picture, or a mark's `rect`), plus the fill settings (`y_center`, `blur`, `dim`).

Which framing applies to a frame:
- an explicit mark in effect at its source time (last `at_src` <= t);
- else the per-shot default: a multi-face shot (two or more people of similar size) is
  letterboxed, a single-face shot tracks its face, no face gives a fixed centre crop;
- speaker-follow: with a `speakers` x map covering every speaker label in the clip, and either no
  marks at all or a mark with mode "speaker", multi-face shots follow the speaker turns (one
  track window per turn, switching at turn starts; short interjections do not switch).

Every crop is clamped inside the source's active picture area (`video.active_area`), so black
letterbox bars or gallery borders are never part of the picture.

Face tracks are reproducible: a tracker reads only the face samples inside fixed windows (each
run of frames it frames, widened by TRACK_MARGIN), and shots analysis samples on an absolute
grid, so any shots json that covers those windows gives the same tracks (`face_windows`,
`shots.covers`). A framing mark placed inside a source cross dissolve switches at the
dissolve's midpoint (`snap_marks`).
"""

from __future__ import annotations

import bisect
from fractions import Fraction

from . import env

OUT_SIZES = {"vertical": (1080, 1920), "landscape": (1920, 1080)}
ASPECTS = {"vertical": Fraction(9, 16), "landscape": Fraction(16, 9)}
TRACK_WINDOW = 0.3       # s: a frame uses face samples this close, else holds
ANCHOR_TOL = 0.14        # a tracked face must stay this close to the mark's hint x
EMA_ALPHA = 0.3          # smoothing on the 4 fps samples (run forward then backward: no lag)
DEAD_ZONE = 0.005        # fraction of width/height the target may wander before the crop moves
FACE_Y = 0.38            # face centre sits this far down the crop
MIN_FACE_H = 0.012       # smaller detections are ignored by the tracker
COLUMN_TOL = 0.06        # candidates this close in x to the best one share its column (a known y decides)
TRACK_MARGIN = 1.0       # s of samples either side of a run of frames that its tracker reads
DISSOLVE_SNAP = 0.1      # a mark this close to a source dissolve switches at its midpoint
PUNCH_JOINS = ("tighten", "omit", "piece")
PUNCH_MIN_HOLD = 1.0     # s: no punch-in when a new framing mark would undo it sooner
CROP_MODES = ("track", "fixed")
DEFAULT_MOTION = {"punch_in": 1.0, "min_gap": 3.5, "max_upscale": 3.0}
DEFAULT_LETTERBOX = {"y_center": 820, "blur": 28, "dim": 0.35}
# speaker-follow (turns)
TURN_MIN_S = 1.0         # a run of one speaker's words shorter than this (output time) ...
TURN_MIN_WORDS = 3       # ... or with fewer words is absorbed into the previous turn
TURN_LEAD = 0.15         # the window switches this long before the turn's first word
TURN_SNAP = 0.3          # ... or exactly at a join when one is this close
MULTI_MINOR = 0.40       # samples fallback: faces under this share of the largest are background


# ---------------------------------------------------------------- geometry

def out_size(fmt: str) -> tuple:
    if fmt not in OUT_SIZES:
        raise env.HCError("unknown format %r (use vertical or landscape)" % (fmt,), code=1)
    return OUT_SIZES[fmt]


def fps_of(obj) -> Fraction:
    """Fraction frame rate from a plan dict, an [num, den] pair or a Fraction."""
    fps = obj.get("fps") if isinstance(obj, dict) else obj
    if isinstance(fps, Fraction):
        return fps
    if isinstance(fps, (list, tuple)) and len(fps) == 2:
        return Fraction(int(fps[0]), int(fps[1]))
    if isinstance(fps, (int, float)) and fps > 0:
        return Fraction(fps).limit_denominator(1001)
    raise env.HCError("plan has no usable fps", code=1)


def display_size(source: dict) -> tuple:
    """Upright, square-pixel (width, height) of a source.json dict (or its video block)."""
    v = source.get("video", source) if isinstance(source, dict) else {}
    w, h = int(v.get("width") or 0), int(v.get("height") or 0)
    if w <= 0 or h <= 0:
        raise env.HCError("source has no video size (re-run ingest)", code=1)
    sar = sar_of(source)
    if sar != 1:
        w = int(round(w * sar / 2)) * 2
    return w, h


def sar_of(source: dict) -> Fraction:
    v = source.get("video", source) if isinstance(source, dict) else {}
    raw = str(v.get("sar") or "1:1")
    try:
        a, b = raw.replace("/", ":").split(":")
        sar = Fraction(int(a), int(b))
        return sar if sar > 0 else Fraction(1)
    except (ValueError, ZeroDivisionError):
        return Fraction(1)


def active_rect(source: dict) -> tuple:
    """(x, y, w, h) of the source's active picture in display pixels: `video.active_area` (the
    picture inside letterbox bars or a gallery border), else the whole display frame."""
    width, height = display_size(source)
    v = source.get("video", source) if isinstance(source, dict) else {}
    aa = v.get("active_area") if isinstance(v, dict) else None
    if isinstance(aa, (list, tuple)) and len(aa) == 4:
        try:
            x, y, w, h = (float(a) for a in aa)
        except (TypeError, ValueError):
            return 0, 0, width, height
        x0, y0 = max(0, int(round(x))), max(0, int(round(y)))
        x1, y1 = min(width, int(round(x + w))), min(height, int(round(y + h)))
        if x1 - x0 >= 16 and y1 - y0 >= 16:
            return x0, y0, x1 - x0, y1 - y0
    return 0, 0, width, height


def _even_rect(rect, width: int, height: int) -> tuple:
    """A rect shrunk to even edges (yuv420 crops need even offsets and sizes)."""
    x, y, w, h = rect
    x0, y0 = min(width, (int(x) + 1) // 2 * 2), min(height, (int(y) + 1) // 2 * 2)
    x1, y1 = max(x0 + 2, int(x + w) // 2 * 2), max(y0 + 2, int(y + h) // 2 * 2)
    x1, y1 = min(x1, width // 2 * 2), min(y1, height // 2 * 2)
    return x0, y0, max(2, x1 - x0), max(2, y1 - y0)


def norm_rect_px(rect, width: int, height: int):
    """A mark's normalized `rect` [x, y, w, h] as an even pixel rect inside the frame (or None)."""
    if not isinstance(rect, (list, tuple)) or len(rect) != 4:
        return None
    try:
        x, y, w, h = (float(v) for v in rect)
    except (TypeError, ValueError):
        return None
    x0, y0 = max(0.0, x) * width, max(0.0, y) * height
    x1, y1 = min(1.0, x + w) * width, min(1.0, y + h) * height
    if x1 - x0 < 16 or y1 - y0 < 16:
        return None
    return _even_rect((x0, y0, x1 - x0, y1 - y0), width, height)


def even_floor(v: float) -> int:
    return max(2, int(v) // 2 * 2)


def max_rect(width: int, height: int, fmt: str) -> tuple:
    """Largest rect of the format's aspect that fits the source (floats)."""
    asp = ASPECTS[fmt] if fmt in ASPECTS else ASPECTS["vertical"]
    if Fraction(width, height) > asp:
        return float(height * asp), float(height)
    return float(width), float(width / asp)


def crop_size(width: int, height: int, fmt: str, crop: float = 1.0, zoom: float = 1.0,
              rect=None) -> tuple:
    """Crop rect (even ints) = max rect x crop x zoom. With `rect` (x, y, w, h: the active
    picture), crop 1.0 is the largest rect of the format's aspect inside that rect."""
    rw, rh = (int(rect[2]), int(rect[3])) if rect else (int(width), int(height))
    mw, mh = max_rect(rw, rh, fmt)
    f = max(0.05, float(crop or 1.0)) * max(0.05, float(zoom or 1.0))
    return min(even_floor(mw * f), even_floor(rw)), min(even_floor(mh * f), even_floor(rh))


def upscale(width: int, height: int, fmt: str, crop: float = 1.0, zoom: float = 1.0,
            rect=None) -> float:
    """Output height / crop rect height (how much the picture is enlarged)."""
    _, ch = crop_size(width, height, fmt, crop, zoom, rect)
    return out_size(fmt)[1] / float(ch)


def crop_xy(fx: float, fy, cw: int, ch: int, width: int, height: int, mode: str,
            rect=None) -> tuple:
    """Top-left corner (even ints) for a target centre (normalized to the display frame),
    clamped so the crop stays inside `rect` (the active picture; default the whole frame)."""
    rx, ry, rw, rh = rect if rect else (0, 0, width, height)
    x = fx * width - cw / 2.0
    if fy is None:
        y = ry + (rh - ch) / 2.0
    elif mode == "fixed":
        y = fy * height - ch / 2.0
    else:
        y = fy * height - FACE_Y * ch
    lo_x, lo_y = (int(rx) + 1) // 2 * 2, (int(ry) + 1) // 2 * 2
    hi_x = min(max(0, width - cw) // 2 * 2, max(lo_x, int(rx + rw - cw) // 2 * 2))
    hi_y = min(max(0, height - ch) // 2 * 2, max(lo_y, int(ry + rh - ch) // 2 * 2))
    lo_x, lo_y = min(lo_x, hi_x), min(lo_y, hi_y)
    xi = min(hi_x, max(lo_x, int(round(x / 2.0)) * 2))
    yi = min(hi_y, max(lo_y, int(round(y / 2.0)) * 2))
    return xi, yi


# ---------------------------------------------------------------- marks and shots

def norm_marks(marks) -> list:
    out = []
    for m in marks or []:
        if not isinstance(m, dict):
            continue
        at = m.get("at_src", m.get("at"))
        try:
            at = env.parse_time(at)
        except env.HCError:
            continue
        mode = m.get("mode") or "track"
        mk = {"at_src": float(at), "mode": mode,
              "x": None if m.get("x") is None else float(m["x"]),
              "y": None if m.get("y") is None else float(m["y"]),
              "crop": float(m.get("crop") or 1.0),
              "reaction": bool(m.get("reaction", False))}
        if isinstance(m.get("rect"), (list, tuple)) and len(m["rect"]) == 4:
            try:
                mk["rect"] = [float(v) for v in m["rect"]]
            except (TypeError, ValueError):
                pass
        out.append(mk)
    out.sort(key=lambda m: m["at_src"])
    return out


def mark_index_at(marks: list, t: float, tol: float = 0.0) -> int:
    """Index of the mark in effect at t (last at_src <= t + tol), or -1 before the first."""
    idx = -1
    for i, m in enumerate(marks):
        if m["at_src"] <= t + tol:
            idx = i
        else:
            break
    return idx


def shot_index(shots, t: float):
    for i, s in enumerate((shots or {}).get("shots") or []):
        if s["start_src"] <= t < s["end_src"]:
            return i
    return None


def same_shot(shots, a: float, b: float) -> bool:
    """True when no detected shot boundary separates a and b (no shots data: assume same)."""
    if not shots or not shots.get("shots"):
        return True
    ia, ib = shot_index(shots, a), shot_index(shots, b)
    if ia is None or ib is None:
        return True
    return ia == ib


def shot_is_multi(shot: dict) -> bool:
    """A shot with two or more people of similar size (gallery, two-shot, group; also two small
    faces of near-equal size in a wide shot, see shots.similar_faces)."""
    if not isinstance(shot, dict):
        return False
    from . import shots as shots_mod        # local: shots imports this module
    return shots_mod.is_multi(shot)


def _samples_multi(shots, t: float) -> bool:
    """No shot covers t: judge from the face samples within 2 s (two or more similar faces)."""
    near = [s for s in (shots or {}).get("samples") or [] if abs(s["t_src"] - t) <= 2.0]
    if not near:
        return False
    from . import shots as shots_mod        # local: shots imports this module
    return shots_mod.faces_multi(shots_mod.shot_faces(near))


def snap_marks(marks: list, shots) -> list:
    """Marks (norm_marks output) with every mark that sits inside a source cross dissolve, or
    within DISSOLVE_SNAP of one, moved to the dissolve's midpoint: the framing then changes on
    the frame where the new shot takes over, not a few frames into or after the blend."""
    trans = [t for t in (shots or {}).get("transitions") or []
             if isinstance(t, dict) and t.get("kind") == "dissolve"]
    if not trans or not marks:
        return marks
    out = []
    for m in marks:
        for t in trans:
            try:
                a, b, at = float(t["start_src"]), float(t["end_src"]), float(t["at_src"])
            except (KeyError, TypeError, ValueError):
                continue
            if a - DISSOLVE_SNAP <= m["at_src"] <= b + DISSOLVE_SNAP:
                if at != m["at_src"]:
                    m = dict(m, at_src=at)
                break
        out.append(m)
    out.sort(key=lambda m: m["at_src"])
    return out


def _merge_windows(spans: list) -> list:
    out = []
    for a, b in sorted(spans):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


def face_windows(plan: dict, clip: dict, fmt: str, shots=None) -> list:
    """Source-time windows [(lo, hi)] whose face samples a render of this plan reads: every run
    of frames framed by a track or speaker mark, or by the per-shot default (no mark yet),
    widened by TRACK_MARGIN. Empty for landscape, for caption-only clips and for clips framed
    only by fixed, letterbox or full marks. `shots.covers(data, windows, source)` tells whether
    a shots json holds them."""
    clip = clip or {}
    if fmt != "vertical" or (plan.get("kind") or clip.get("kind") or "cut") == "caption_only":
        return []
    fps = fps_of(plan)
    dt = Fraction(1) / fps
    half = float(dt) / 2
    marks = snap_marks(norm_marks(plan.get("framing") if plan.get("framing") is not None
                                  else clip.get("framing")), shots)
    out = []
    for seg in plan.get("segments") or []:
        n_total = int(seg["frames"])
        s0 = Fraction(seg["src"][0]).limit_denominator(10 ** 7)
        start = None
        for n in range(n_total + 1):
            need = False
            if n < n_total:
                mi = mark_index_at(marks, float(s0 + n * dt), half)
                need = mi < 0 or marks[mi]["mode"] in ("track", "speaker")
            if need and start is None:
                start = n
            elif not need and start is not None:
                out.append((float(s0 + start * dt) - TRACK_MARGIN,
                            float(s0 + (n - 1) * dt) + TRACK_MARGIN))
                start = None
    return _merge_windows(out)


def multi_face_at(shots, t: float) -> bool:
    if not shots:
        return False
    i = shot_index(shots, t)
    if i is not None:
        return shot_is_multi(shots["shots"][i])
    return _samples_multi(shots, t)


def _largest_face_x(shots, t: float):
    """x of the largest face at t. Only meaningful in single-face shots: never use it to pick a
    person in a multi-face shot (the largest tile is not the speaker)."""
    if not shots:
        return None
    i = shot_index(shots, t)
    cands = []
    if i is not None:
        cands = [f for f in shots["shots"][i].get("faces") or [] if not f.get("minor")]
    if not cands:
        near = [s for s in shots.get("samples") or [] if abs(s["t_src"] - t) <= 2.0]
        cands = [{"x": f[0], "h": f[3], "presence": 1.0} for s in near for f in s["faces"]]
    if not cands:
        return None
    best = max(cands, key=lambda f: (f.get("h", 0), f.get("presence", 0)))
    return float(best["x"])


def default_mark(fmt: str, shots, t0: float, source=None) -> dict:
    """The default framing at t0 when no mark applies (contract 7): landscape is full; a
    multi-face shot is letterboxed (never the largest face); a single-face shot tracks its
    face; otherwise a fixed crop at the centre of the active picture."""
    if fmt == "landscape":
        return {"at_src": t0, "mode": "full", "x": None, "y": None, "crop": 1.0}
    if multi_face_at(shots, t0):
        return {"at_src": t0, "mode": "letterbox", "x": None, "y": None, "crop": 1.0,
                "multi_face": True}
    x = _largest_face_x(shots, t0)
    if x is not None:
        return {"at_src": t0, "mode": "track", "x": x, "y": None, "crop": 1.0}
    cx, cy = 0.5, 0.5
    if source:
        try:
            w, h = display_size(source)
            ax, ay, aw, ah = active_rect(source)
            cx, cy = (ax + aw / 2.0) / w, (ay + ah / 2.0) / h
        except env.HCError:
            pass
    return {"at_src": t0, "mode": "fixed", "x": cx, "y": cy, "crop": 1.0}


# ---------------------------------------------------------------- speaker turns

def _visible(w: dict) -> bool:
    return not (w.get("hidden") or w.get("omitted") or w.get("removed") or w.get("filler")
                or w.get("edge"))


def speaker_map(plan: dict) -> dict:
    """{label: {"x", "y"}} from plan["speakers"] (labels with a usable x only)."""
    out = {}
    for lab, v in ((plan or {}).get("speakers") or {}).items():
        if isinstance(v, dict) and v.get("x") is not None:
            try:
                out[str(lab)] = {"x": float(v["x"]),
                                 "y": None if v.get("y") is None else float(v["y"])}
            except (TypeError, ValueError):
                continue
    return out


def clip_labels(plan: dict) -> list:
    """Speaker labels of the plan's visible words, in order of first appearance."""
    seen = []
    for w in sorted((w for w in (plan or {}).get("words") or [] if _visible(w) and w.get("sp")
                     and w.get("s_out") is not None), key=lambda w: (w["s_out"], w["i"])):
        if w["sp"] not in seen:
            seen.append(w["sp"])
    return seen


def follow_ok(plan: dict) -> bool:
    """True when plan["speakers"] has an x for every speaker label among the visible words."""
    labs, smap = clip_labels(plan), speaker_map(plan)
    return bool(labs) and all(lab in smap for lab in labs)


def speaker_turns(plan: dict) -> list:
    """Speaker turns over the plan's visible words, in output order:
    [{"sp", "t_out", "at_src", "words": [i, ...]}]. A run of one label shorter than TURN_MIN_S
    (output time) or TURN_MIN_WORDS words is absorbed into the previous turn (an interjection
    such as "Yeah." does not switch). Turn 0 starts at -inf; every later turn switches
    TURN_LEAD before its first word, or exactly at a join when one is within TURN_SNAP."""
    ws = sorted((w for w in (plan or {}).get("words") or [] if _visible(w) and w.get("sp")
                 and w.get("s_out") is not None), key=lambda w: (w["s_out"], w["i"]))
    runs = []
    for w in ws:
        if runs and runs[-1][0] == w["sp"]:
            runs[-1][1].append(w)
        else:
            runs.append([w["sp"], [w]])
    turns, pending = [], []
    for lab, rw in runs:
        dur = float(rw[-1].get("e_out") or rw[-1]["s_out"]) - float(rw[0]["s_out"])
        if dur < TURN_MIN_S or len(rw) < TURN_MIN_WORDS:
            if turns:
                turns[-1]["w"].extend(rw)
            else:
                pending.extend(rw)
            continue
        if turns and turns[-1]["sp"] == lab:
            turns[-1]["w"].extend(rw)
        else:
            turns.append({"sp": lab, "w": pending + rw, "first": rw[0]})
            pending = []
    if not turns and pending:
        tot = {}
        for w in pending:
            tot[w["sp"]] = tot.get(w["sp"], 0.0) + max(0.0, float(w.get("e_out") or 0)
                                                        - float(w["s_out"]))
        lab = max(tot, key=lambda k: (tot[k], -[x["sp"] for x in pending].index(k)))
        turns = [{"sp": lab, "w": pending, "first": pending[0]}]
    joins = [j for j in (plan or {}).get("joins") or [] if j.get("out") is not None]
    out = []
    for k, tn_ in enumerate(turns):
        first = tn_["first"]
        if k == 0:
            t_out, at_src = float("-inf"), float(first.get("s_src", 0.0)) - TURN_LEAD
        else:
            t_out = float(first["s_out"]) - TURN_LEAD
            at_src = float(first.get("s_src", 0.0)) - TURN_LEAD
            near = [j for j in joins if abs(float(j["out"]) - t_out) <= TURN_SNAP + 1e-9]
            if near:
                j = min(near, key=lambda j: abs(float(j["out"]) - t_out))
                t_out = float(j["out"])
                at_src = float(j.get("after_src", at_src))
        out.append({"sp": tn_["sp"], "t_out": t_out, "at_src": round(at_src, 6),
                    "words": [w["i"] for w in tn_["w"]]})
    return out


def _label_at_src(plan: dict, t: float):
    """Speaker label of the visible word at (or next after) source time t."""
    best = None
    for w in (plan or {}).get("words") or []:
        if not (_visible(w) and w.get("sp")):
            continue
        s, e = float(w.get("s_src", 0.0)), float(w.get("e_src", 0.0))
        if s <= t < e:
            return w["sp"]
        if s >= t and (best is None or s < best[0]):
            best = (s, w["sp"])
    return best[1] if best else None


# ---------------------------------------------------------------- tracker

def _ema(values: list, alpha: float) -> list:
    out, acc = [], None
    for v in values:
        acc = v if acc is None else acc + alpha * (v - acc)
        out.append(acc)
    return out


class Tracker(object):
    """Hint-anchored face track over a src time range.

    Association is causal: each 4 fps sample takes the face nearest the previous match that
    is also within ANCHOR_TOL of the hint x (so the track never creeps onto a neighbour); no
    candidate holds the last value. x decides first. When a hint y is known and several
    candidates sit in the best one's column (within COLUMN_TOL in x: people stacked in a
    gallery), the one nearest the previous match's y (the hint y at the start) is taken, not the
    taller face. The matched series is smoothed with a forward-backward EMA (no lag), and
    frames read it by linear interpolation.
    """

    def __init__(self, samples: list, hint_x: float, lo: float, hi: float, hint_y=None):
        self.hint_x = float(hint_x)
        self.hint_y = None if hint_y is None else float(hint_y)
        sel = sorted((s for s in samples or [] if lo <= s["t_src"] <= hi),
                     key=lambda s: s["t_src"])
        self.used = [[s["t_src"], [list(f) for f in s.get("faces") or []]] for s in sel]
        ts, xs, ys, hit = [], [], [], []
        prev = None
        for s in sel:
            cands = [f for f in s.get("faces") or []
                     if f[3] >= MIN_FACE_H and abs(f[0] - self.hint_x) <= ANCHOR_TOL]
            ref = prev[0] if prev else self.hint_x
            if cands:
                f = min(cands, key=lambda f: (abs(f[0] - ref), -f[3]))
                if self.hint_y is not None and len(cands) > 1:
                    near = abs(f[0] - ref) + COLUMN_TOL
                    column = [c for c in cands if abs(c[0] - ref) <= near]
                    if len(column) > 1:
                        ref_y = prev[1] if prev else self.hint_y
                        f = min(column, key=lambda c: (abs(c[1] - ref_y), abs(c[0] - ref), -c[3]))
                prev = (float(f[0]), float(f[1]))
                hit.append(True)
            else:
                hit.append(False)
            ts.append(float(s["t_src"]))
            xs.append(prev[0] if prev else None)
            ys.append(prev[1] if prev else None)
        first = next((i for i, h in enumerate(hit) if h), None)
        if first is None:
            self.ts, self.xs, self.ys, self.hit_ts = [], [], [], []
            return
        for i in range(first):              # back-fill before the first detection
            xs[i], ys[i] = xs[first], ys[first]
        a = EMA_ALPHA
        self.xs = _ema(_ema(xs, a)[::-1], a)[::-1]
        self.ys = _ema(_ema(ys, a)[::-1], a)[::-1]
        self.ts = ts
        self.hit_ts = [t for t, h in zip(ts, hit) if h]

    @property
    def found(self) -> bool:
        return bool(self.ts)

    def near_face(self, t: float) -> bool:
        if not self.hit_ts:
            return False
        i = bisect.bisect_left(self.hit_ts, t)
        best = min((abs(self.hit_ts[j] - t) for j in (i - 1, i) if 0 <= j < len(self.hit_ts)))
        return best <= TRACK_WINDOW

    def value(self, t: float) -> tuple:
        """(x, y) at t by interpolation of the smoothed series (hint y, else None, when no face)."""
        if not self.ts:
            return self.hint_x, self.hint_y
        i = bisect.bisect_right(self.ts, t)
        if i <= 0:
            return self.xs[0], self.ys[0]
        if i >= len(self.ts):
            return self.xs[-1], self.ys[-1]
        t0, t1 = self.ts[i - 1], self.ts[i]
        u = 0.0 if t1 <= t0 else (t - t0) / (t1 - t0)
        return (self.xs[i - 1] + u * (self.xs[i] - self.xs[i - 1]),
                self.ys[i - 1] + u * (self.ys[i] - self.ys[i - 1]))


def _dead_zone(cam, target: float, dz: float) -> float:
    if cam is None:
        return target
    d = target - cam
    if abs(d) <= dz:
        return cam
    return target - dz if d > 0 else target + dz


# ---------------------------------------------------------------- build

def _merge(base: dict, *over) -> dict:
    out = dict(base)
    for o in over:
        for k, v in (o or {}).items():
            if v is not None:
                out[k] = v
    return out


def build_tracks(plan: dict, clip: dict, shots, source: dict, fmt: str, preset: dict,
                 report: dict = None) -> list:
    """Per plan segment, the list of SubSegs render executes (see module docstring). `report`
    (a dict) receives `faces`: a hash of every face sample the trackers read (the render key
    uses it) and `trackers`: how many track windows there were."""
    clip = clip or {}
    preset = preset or {}
    fps = fps_of(plan)
    dt = Fraction(1) / fps
    width, height = display_size(source)
    act = _even_rect(active_rect(source), width, height)
    segments = plan.get("segments") or []
    lb = _merge(DEFAULT_LETTERBOX, preset.get("letterbox"))
    fill = {"y_center": int(lb["y_center"]), "blur": float(lb["blur"]), "dim": float(lb["dim"])}
    kind = plan.get("kind") or clip.get("kind") or "cut"

    def frame_src(seg, n):
        return Fraction(seg["src"][0]).limit_denominator(10 ** 7) + n * dt

    def sub(seg, n0, n1, mode, **extra):
        d = {"src": [round(float(frame_src(seg, n0)), 6), round(float(frame_src(seg, n1)), 6)],
             "frames": int(n1 - n0), "mode": mode, "size": [width, height], "cmds": [],
             "zoom": 1.0}
        d.update(extra)
        return d

    if kind == "caption_only":
        cw, ch = crop_size(width, height, fmt, 1.0, 1.0, act)
        x0, y0 = crop_xy((act[0] + act[2] / 2.0) / width, (act[1] + act[3] / 2.0) / height,
                         cw, ch, width, height, "fixed", act)
        return [[sub(seg, 0, int(seg["frames"]), "crop", size=[cw, ch], cmds=[[0.0, x0, y0]],
                     mark=-1)] for seg in segments]
    if fmt == "landscape":
        return [[sub(seg, 0, int(seg["frames"]), "full", mark=-1, rect=list(act), **fill)]
                for seg in segments]

    marks = snap_marks(norm_marks(plan.get("framing") if plan.get("framing") is not None
                                  else clip.get("framing")), shots)
    smap = speaker_map(plan)
    has_speaker_mark = any(m["mode"] == "speaker" for m in marks)
    turns = speaker_turns(plan) if (follow_ok(plan) and (not marks or has_speaker_mark)) else []
    turn_t = [t["t_out"] for t in turns]
    shot_list = (shots or {}).get("shots") or []
    motion = _merge(DEFAULT_MOTION, preset.get("motion"), clip.get("motion"))
    punch = float(motion.get("punch_in") or 1.0)
    min_gap = float(motion.get("min_gap") or 0.0)
    max_up = float(motion.get("max_upscale") or 99.0)
    samples = (shots or {}).get("samples") or []
    joins = {int(j["k"]): j for j in plan.get("joins") or [] if "k" in j}
    half = float(dt) / 2

    # Pass 1: the framing key of every frame. A key is (kind, index, explicit mark index):
    # ("m", i) explicit mark i; ("def", shot) per-shot default; ("turn", j) speaker turn j.
    per_seg, spans, multi, key_runs = [], {}, {}, {}
    out_cursor = Fraction(0)
    for seg in segments:
        n_total = int(seg["frames"])
        seg_out0 = Fraction(seg["out"][0]).limit_denominator(10 ** 7) if seg.get("out") \
            else out_cursor
        out_cursor += n_total * dt
        keys = []
        for n in range(max(0, n_total)):
            ts = float(frame_src(seg, n))
            mi = mark_index_at(marks, ts, half)
            m = marks[mi] if mi >= 0 else None
            sh = shot_index(shots, ts) if shot_list else None
            if m is not None and m["mode"] != "speaker":
                key = ("m", mi, mi)
            elif turns and (m is not None or not marks) and \
                    (sh is None or shot_is_multi(shot_list[sh])):
                to = float(seg_out0 + n * dt)
                key = ("turn", max(0, bisect.bisect_right(turn_t, to + 1e-9) - 1), mi)
            else:
                key = ("def", sh, mi)
            keys.append(key)
            lo, hi = spans.get(key[:2], (ts, ts))
            spans[key[:2]] = (min(lo, ts), max(hi, ts))
            if sh is not None:
                multi[key[:2]] = multi.get(key[:2], False) or shot_is_multi(shot_list[sh])
            elif key[:2] not in multi:
                multi[key[:2]] = multi_face_at(shots, ts) if shots else False
        per_seg.append(keys)
        start = 0
        for n in range(1, len(keys) + 1):       # runs of one key: the tracker windows
            if n == len(keys) or keys[n][:2] != keys[start][:2]:
                key_runs.setdefault(keys[start][:2], []).append(
                    (float(frame_src(seg, start)), float(frame_src(seg, n - 1))))
                start = n

    t_first = float(segments[0]["src"][0]) if segments else 0.0
    objs = {}

    def mark_obj(key):
        # A speaker turn takes its crop from the speaker mark in effect, so a turn that runs
        # across a later speaker mark is one object per mark (the face track, the camera and
        # the windows stay per turn: they do not depend on the crop, so the picture is
        # continuous across the mark).
        k2 = key[:2]
        ck = key if k2[0] == "turn" else k2
        if ck in objs:
            return objs[ck]
        kind_, idx = k2
        if kind_ == "m":
            m = dict(marks[idx])
            m["source"] = "mark"
            if m["mode"] == "track" and m.get("x") is None:
                if multi_face_at(shots, m["at_src"]):
                    lab = _label_at_src(plan, m["at_src"])
                    if lab in smap:
                        m.update(x=smap[lab]["x"], speaker=lab)
                        if m.get("y") is None:
                            m["y"] = smap[lab]["y"]
                    else:
                        m.update(mode="letterbox", x=None, y=None, multi_face=True)
                else:
                    hx = _largest_face_x(shots, m["at_src"]) if shots else None
                    m["x"] = hx if hx is not None else 0.5
            if m["mode"] == "fixed":
                m["x"] = 0.5 if m.get("x") is None else m["x"]
                m["y"] = 0.5 if m.get("y") is None else m["y"]
        elif kind_ == "turn":
            tn_ = turns[idx]
            ent = smap[tn_["sp"]]
            m = {"at_src": tn_["at_src"], "mode": "track", "x": ent["x"], "y": ent["y"],
                 "crop": 1.0, "speaker": tn_["sp"], "source": "speaker", "reaction": False}
            mi = key[2]
            if mi >= 0 and marks[mi].get("crop"):
                m["crop"] = float(marks[mi]["crop"])
        else:
            t0 = spans.get(k2, (t_first, t_first))[0] if idx is not None else t_first
            m = default_mark(fmt, shots, t0, source)
            m["source"] = "default"
            m.setdefault("reaction", False)
        objs[ck] = m
        return m

    # A tracker reads only the samples inside its window: a run of frames with one framing key,
    # widened by TRACK_MARGIN (windows of one key that overlap are one window). The windows
    # follow from the plan and the marks alone, so the tracks do not depend on how much of the
    # source was analysed around them.
    wins, trackers, cams = {}, {}, {}

    def tracker(key, t):
        k2 = key[:2]
        if k2 not in wins:
            wins[k2] = _merge_windows([(a - TRACK_MARGIN, b + TRACK_MARGIN)
                                       for a, b in key_runs.get(k2, [])])
        wi = next((i for i, (a, b) in enumerate(wins[k2]) if a <= t <= b), 0)
        if (k2, wi) not in trackers:
            m = mark_obj(key)
            lo, hi = wins[k2][wi] if wins[k2] else (float("-inf"), float("inf"))
            trackers[(k2, wi)] = Tracker(samples, m["x"], lo, hi, m.get("y"))
        return trackers[(k2, wi)]

    zoom, last_toggle, prev_mark = 1.0, 0.0, None
    out_cursor = Fraction(0)
    tracks = []
    for k, seg in enumerate(segments):
        n_total = int(seg["frames"])
        seg_out0 = float(seg["out"][0]) if seg.get("out") else float(out_cursor)
        out_cursor += n_total * dt
        if n_total <= 0:
            tracks.append([])
            continue
        fmarks = per_seg[k]
        m0 = fmarks[0]
        hold = next((n for n in range(n_total) if fmarks[n] != m0), None)
        short_hold = hold is not None and hold * dt < Fraction(PUNCH_MIN_HOLD).limit_denominator(1000)
        if m0 != prev_mark:
            zoom = 1.0
        elif k in joins and joins[k].get("kind") in PUNCH_JOINS and punch > 1.0 and not short_hold:
            j, mk = joins[k], mark_obj(m0)
            before = float(j.get("before_src", seg["src"][0]))
            after = float(j.get("after_src", seg["src"][0]))
            if (mk["mode"] in CROP_MODES and same_shot(shots, before, after)
                    and seg_out0 - last_toggle >= min_gap - 1e-9):
                target = 1.0 / punch if zoom == 1.0 else 1.0
                if target >= 1.0 or upscale(width, height, fmt, mk["crop"], target, act) <= max_up:
                    zoom, last_toggle = target, seg_out0
        runs, start = [], 0
        for n in range(1, n_total + 1):
            if n == n_total or fmarks[n] != fmarks[start]:
                runs.append((start, n, fmarks[start]))
                start = n
        subs = []
        for r, (n0, n1, key) in enumerate(runs):
            if r > 0:
                zoom = 1.0                   # a new mark starts unzoomed
            m = mark_obj(key)
            info = {"mark": key[2], "reaction": bool(m.get("reaction")),
                    "speaker": m.get("speaker"), "framing": m.get("source", "mark"),
                    "multi_face": bool(multi.get(key[:2], False))}
            if m["mode"] in ("letterbox", "full"):
                rect = norm_rect_px(m.get("rect"), width, height) if m["mode"] == "letterbox" \
                    else None
                extra = dict(info, rect=list(rect or act), **fill)
                if rect:
                    extra["rect_mark"] = True
                subs.append(sub(seg, n0, n1, m["mode"], **extra))
                continue
            z = zoom
            cw, ch = crop_size(width, height, fmt, m["crop"], z, act)
            cmds, last = [], None
            trk = tracker(key, float(frame_src(seg, n0))) if m["mode"] == "track" else None
            cam = cams.get(key[:2])
            for n in range(n0, n1):
                if trk is None:
                    fx, fy = m["x"], m["y"]
                else:
                    t = float(frame_src(seg, n))
                    if cam is not None and not trk.near_face(t):
                        fx, fy = cam                       # no face nearby: hold
                    else:
                        fx, fy = trk.value(t)
                        cx = _dead_zone(cam[0] if cam else None, fx, DEAD_ZONE)
                        cy = None if fy is None else _dead_zone(
                            cam[1] if cam and cam[1] is not None else None, fy, DEAD_ZONE)
                        fx, fy = cx, cy
                    cam = (fx, fy)
                xy = crop_xy(fx, fy, cw, ch, width, height, m["mode"], act)
                if xy != last:
                    cmds.append([round(float((n - n0) * dt), 6), xy[0], xy[1]])
                    last = xy
            if trk is not None:
                cams[key[:2]] = cam
            subs.append(sub(seg, n0, n1, "crop", size=[cw, ch], cmds=cmds, zoom=round(z, 6),
                            anchor=m.get("x"), **info))
        prev_mark = runs[-1][2]
        tracks.append(subs)
    if report is not None:
        read = sorted(([round(wins[k2][wi][0], 6), round(wins[k2][wi][1], 6), trk.hint_x,
                        trk.used] for (k2, wi), trk in trackers.items() if wins.get(k2)),
                      key=lambda r: (r[0], r[1], r[2]))
        report["faces"] = env.stable_hash(read)
        report["trackers"] = len(trackers)
    return tracks


# Keys of a SubSeg that change the encoded picture (the rest is bookkeeping for logs and QA).
RENDER_KEYS = ("src", "frames", "mode", "size", "cmds", "zoom", "rect", "y_center", "blur", "dim")


def render_view(sub: dict) -> dict:
    """The part of a SubSeg that determines the encoded frames (used for cache keys)."""
    return {k: sub[k] for k in RENDER_KEYS if k in sub}


def windows(tracks: list, plan: dict, source: dict) -> list:
    """Per SubSeg, the normalized source interval on screen (render log `framing_windows`):
    lo/hi = the x interval covered for the whole SubSeg (intersection over time), lo_any/hi_any
    = the union; y_lo/y_hi = the y interval covered for the whole SubSeg, so people stacked in
    one column can be told apart. Letterbox and full show their whole `rect` (0..1 for a
    full-frame source)."""
    width, height = display_size(source)
    fps = fps_of(plan)
    out = []
    segs = plan.get("segments") or []
    cur = Fraction(0)
    for k, subs in enumerate(tracks):
        seg = segs[k] if k < len(segs) else {}
        t = Fraction(seg["out"][0]).limit_denominator(10 ** 7) if seg.get("out") else cur
        for s in subs:
            n = int(s["frames"])
            t1 = t + Fraction(n) / fps
            if s["mode"] == "crop":
                cw, ch = float(s["size"][0]), float(s["size"][1])
                xs = [float(c[1]) for c in s.get("cmds") or []] or [0.0]
                ys = [float(c[2]) for c in s.get("cmds") or []] or [0.0]
                lo, hi = max(xs) / width, (min(xs) + cw) / width
                lo_any, hi_any = min(xs) / width, (max(xs) + cw) / width
                y_lo, y_hi = max(ys) / height, (min(ys) + ch) / height
                mode = "crop"
            else:
                rx, ry, rw, rh = s.get("rect") or (0, 0, width, height)
                lo = lo_any = float(rx) / width
                hi = hi_any = float(rx + rw) / width
                y_lo, y_hi = float(ry) / height, float(ry + rh) / height
                mode = "rect" if s.get("rect_mark") else s["mode"]
            out.append({"seg": k, "out": [round(float(t), 6), round(float(t1), 6)], "mode": mode,
                        "lo": round(lo, 4), "hi": round(hi, 4), "lo_any": round(lo_any, 4),
                        "hi_any": round(hi_any, 4), "y_lo": round(y_lo, 4), "y_hi": round(y_hi, 4),
                        "mark": int(s.get("mark", -1)),
                        "reaction": bool(s.get("reaction")),
                        "multi_face": bool(s.get("multi_face")),
                        "speaker": s.get("speaker"), "framing": s.get("framing", "mark")})
            t = t1
        cur = t
    return out


def sendcmd_text(sub: dict, target: str, fps) -> str:
    """sendcmd script that moves `target` (a crop filter instance) per sub["cmds"][1:].

    Each command fires half a frame before its frame so float rounding can never push it
    one frame late (the SubSeg timeline starts at 0 and runs at the plan fps)."""
    rate = fps_of(fps)
    lines = []
    for t_rel, x, y in (sub.get("cmds") or [])[1:]:
        n = round(Fraction(t_rel).limit_denominator(10 ** 7) * rate)
        ts = float((n - Fraction(1, 2)) / rate)
        lines.append("%.6f %s x %d, %s y %d;" % (ts, target, int(x), target, int(y)))
    return "\n".join(lines) + ("\n" if lines else "")


def summary(tracks: list) -> list:
    """Compact per-segment description for logs: [[mode, frames, size, zoom, n_cmds], ...]."""
    return [[[s["mode"], s["frames"], s["size"], s["zoom"], len(s["cmds"])] for s in subs]
            for subs in tracks]
