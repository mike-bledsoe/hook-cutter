"""Shot boundaries, faces per shot and a framing proposal (contract 8.2).

Analysis decodes the clip's source ranges at 4 fps, 640 px wide, on an ABSOLUTE grid: sample k
is the source frame on screen at t = k / 4 s of the source clock, whatever range is analysed.
Two analyses that both cover a stretch of the source therefore hold the same samples for it,
and framing reads samples by fixed windows (framing.TRACK_MARGIN), so a render can be
reproduced from a fresh `hc shots`. Shot cuts come from a lagged frame difference (frame t vs
frame t-2, so slow dissolves register as one sustained change) with an adaptive threshold;
each boundary is then looked at frame by frame: a hard cut lands on the exact source frame, a
cross dissolve gets its span and midpoint (`transitions`). YuNet faces per sample are clustered
per shot by x. The proposal is one framing mark per shot, ready to paste into clips.json, and
the sheet shows one frame per shot with face boxes, x labels and the 9:16 window the proposal
would use.
"""

from __future__ import annotations

import json
import math
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from . import env, faces, framing
from . import render_util as ru

SAMPLE_FPS = 4
GRID = "absolute"         # samples sit at k / SAMPLE_FPS on the source clock (shots json "grid")
SHOTS_VERSION = 2         # 2: absolute sample grid, transitions, multi_face by similar size
ANALYSIS_WIDTH = 640
DIFF_W = 64               # gray thumbnails for the difference signal (64 x 36 for 16:9)
LAG = 2                   # samples (0.5 s at 4 fps)
MIN_THRESHOLD = 6.0       # mean abs difference on 0..255
MAD_K = 6.0
MIN_SHOT_GAP = 1.0        # s between accepted cuts
EDGE_GUARD = 0.25         # s: cuts this close to a range edge are ignored
REFINE_WIN = 0.6          # s decoded around a cut to place it on the exact frame
REFINE_MIN = 0.5          # refined step must reach this share of the threshold
TRANS_WIN = (1.6, 3.6)    # s decoded at the native rate around a boundary: cut or dissolve?
                          # (the wider window is tried when the first cannot see both shots clean)
TRANS_LO = 0.05           # blend share up to this: the frame is still the old shot ...
TRANS_HI = 0.95           # ... from this: it is the new shot
TRANS_MIN_STEP = 2.0      # the two shots must differ by this much (rms gray) to measure a blend
TRANS_BACK = 0.12         # a blend share may fall back this much inside a dissolve (motion)
DISSOLVE_MIN = 2          # blended frames between the two shots that make it a dissolve
CELL_DIFF = 20.0          # a thumbnail cell "changed" when it moves by more than this (0..255)
MIN_CHANGED = 0.40        # a cut changes at least this share of cells; a gesture or a lean does not
SPIKE_THR = 2.0           # ... unless, at the native frame rate, it is one isolated frame step: at
SPIKE_ISOLATION = 3.0     # least SPIKE_THR x the threshold, SPIKE_ISOLATION x the next largest step
SPIKE_RATIO = 10.0        # and SPIKE_RATIO x the median step around it (+/- SPIKE_WIN). A hard switch
                          # between two similar framings (dark close-up to dark wide) rewrites few
                          # cells but in one frame; a gesture ramps over several. Measured: two
                          # missed TED camera cuts have max / next 4.8 and 5.9 and max / median 15
                          # to 62; every gesture or lean on a one-camera video call has max / next
                          # 1.6 or less, or a max below SPIKE_THR x threshold.
SPIKE_WIN = 1.0           # s decoded either side of a candidate for the native-rate check
CLUSTER_TOL = 0.06        # face centres closer than this in x are one person
WIDE_FACE_H = 0.12        # largest face shorter than this → wide shot
MIN_PRESENCE = 0.10       # face clusters seen in fewer samples are dropped as noise
COUNT_PRESENCE = 0.30     # clusters at or above this count toward single / two / group
MINOR_FACE = 0.40         # a face under this share of the largest face's height is background
                          # (a photo on a shelf, a passer-by): listed with "minor", never counted
MULTI_PRESENCE = 0.50     # two or more faces seen in at least this share of a shot's samples ...
MULTI_RATIO = 1.6         # ... with heights within this factor are a multi-face shot, however
                          # small they are (a wide two-shot)
RANGE_PAD = 2.0           # s analysed before and after each stretch the clip plays (more than
                          # framing.TRACK_MARGIN, so small edge moves need no new analysis)
MERGE_GAP = 5.0           # piece ranges closer than this are analysed as one
SHEET_W = 640
SHEET_COLS = 2
RULER = (255, 216, 77)
FACE_BOX = (120, 230, 140)
WINDOW = (80, 200, 255)


# ---------------------------------------------------------------- ranges

def plan_ranges(plan: dict, duration: float) -> list:
    """Source ranges to analyse for a plan: every segment it plays ± RANGE_PAD, merged."""
    spans = [(float(sg["src"][0]), float(sg["src"][1])) for sg in (plan or {}).get("segments") or []]
    return merge_ranges(spans, duration) if spans else []


def _plan_spans(proj, clip_id: str) -> list:
    """Source spans the clip's current plan plays (snapped edges, holds and offsets included);
    [] when no plan can be built yet."""
    try:
        from . import plan as plan_mod
        plan = plan_mod.current_plan(proj, clip_id)
    except (env.HCError, OSError, ValueError, KeyError, TypeError, IndexError):
        return []
    return [(float(sg["src"][0]), float(sg["src"][1])) for sg in plan.get("segments") or []]


def clip_ranges(proj, clip_id: str, duration: float) -> list:
    """Source ranges to analyse for a clip: its pieces and every stretch its plan plays (a
    hold_after tail or an edge offset reaches past the piece's words), each ± RANGE_PAD, merged
    when close."""
    clip = ru.clip_entry(proj, clip_id)
    if (clip.get("kind") or "cut") == "caption_only":
        raise env.HCError("%s is caption_only: shots analyses source ranges, run it on cut clips "
                          "or use --range" % clip_id, code=1)
    words = None
    spans = []
    for p in clip.get("pieces") or []:
        if p.get("time_src"):
            spans.append((float(p["time_src"][0]), float(p["time_src"][1])))
        elif p.get("words"):
            if words is None:
                words = proj.transcript().get("words") or []
            a, b = int(p["words"][0]), int(p["words"][1])
            if not (0 <= a <= b < len(words)):
                raise env.HCError("%s: piece words %s are outside the transcript (0..%d)"
                                  % (clip_id, [a, b], len(words) - 1), code=1)
            spans.append((float(words[a]["s"]), float(words[b]["e"])))
    if not spans:
        raise env.HCError("%s has no pieces to analyse" % clip_id, code=1)
    return merge_ranges(spans + _plan_spans(proj, clip_id), duration)


def merge_ranges(spans: list, duration: float) -> list:
    out = []
    for a, b in sorted(spans):
        a = max(0.0, a - RANGE_PAD)
        b = min(duration, b + RANGE_PAD) if duration > 0 else b + RANGE_PAD
        if out and a <= out[-1][1] + MERGE_GAP:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return [(round(a, 3), round(b, 3)) for a, b in out if b - a > 0.2]


# ---------------------------------------------------------------- decoding

def _stream(path, args: list, frame_bytes: int):
    """Yield raw frames from an ffmpeg pipe; the stderr scan follows env.ff's policy."""
    cmd = [env.ffmpeg(), "-hide_banner", "-nostdin", "-loglevel", "warning"] + [str(a) for a in args]
    if env.verbose():
        env.log("ffmpeg", env.quote_cmd(cmd))
    with tempfile.TemporaryFile() as err:
        proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=err)
        try:
            while True:
                buf = proc.stdout.read(frame_bytes)
                if len(buf) < frame_bytes:
                    break
                yield buf
        finally:
            proc.stdout.close()
            rc = proc.wait()
            err.seek(0)
            text = err.read().decode("utf-8", "replace")
        deny, _ = env.scan_stderr(text)
        if rc != 0:
            raise env.HCError("ffmpeg failed while decoding %s: %s" % (Path(path).name, text[-600:]), code=3)
        if deny:
            raise env.HCError("ffmpeg reported timestamp damage:\n  %s" % "\n  ".join(deny[:5]), code=3)


def _small_gray(bgr: np.ndarray) -> np.ndarray:
    import cv2
    h, w = bgr.shape[:2]
    dh = max(2, int(round(DIFF_W * h / float(w))))
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return cv2.resize(g, (DIFF_W, dh), interpolation=cv2.INTER_AREA).astype(np.float32)


def _analysis_size(src: dict) -> tuple:
    w, h = framing.display_size(src)
    return ANALYSIS_WIDTH, max(2, int(round(ANALYSIS_WIDTH * h / float(w) / 2)) * 2)


def sample_index_range(a: float, b: float) -> tuple:
    """(k0, k1): the absolute sample indices k with a <= k / SAMPLE_FPS <= b (k1 < k0: none)."""
    return (int(math.ceil(a * SAMPLE_FPS - 1e-6)), int(math.floor(b * SAMPLE_FPS + 1e-6)))


def sample_times(a: float, b: float) -> list:
    """The absolute sample times inside [a, b]."""
    k0, k1 = sample_index_range(a, b)
    return [k / float(SAMPLE_FPS) for k in range(max(0, k0), k1 + 1)]


def _decode_range(path, src: dict, a: float, b: float):
    """(times, small gray frames, faces per frame) for the absolute samples inside [a, b].

    Sample k is the source frame on screen at t = k / SAMPLE_FPS (the last frame whose
    timestamp is <= t), so the frame chosen for a sample time never depends on where the
    analysed range starts. ffmpeg seeks to a whole second at least one second earlier (whole
    seconds are exact in every stream time base) and `fps=...:round=up:start_time=0` then puts
    the last frame at or before each slot time into that slot; the slots before `a` are
    dropped (the first slot after a seek has no earlier frame to show)."""
    aw, ah = _analysis_size(src)
    k0, k1 = sample_index_range(a, b)
    k0 = max(0, k0)
    if k1 < k0:
        return [], [], []
    tone = ""
    if (src["video"].get("hdr") or "none") not in ("none", "", "sdr") and env.has_filter("zscale"):
        from .render import TONEMAP
        tone = TONEMAP + ","
    seek = max(0, k0 // SAMPLE_FPS - 1)                  # whole seconds
    skip = k0 - seek * SAMPLE_FPS
    want = k1 - k0 + 1
    args = ["-ss", "%d" % seek, "-t", "%.6f" % (k1 / float(SAMPLE_FPS) - seek + 0.75),
            "-i", str(path), "-map", "0:v:0",
            "-vf", "fps=%d:round=up:start_time=0,%sscale=%d:%d:flags=area"
            % (SAMPLE_FPS, tone, aw, ah),
            "-frames:v", str(skip + want), "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    times, grays, found = [], [], []
    for j, buf in enumerate(_stream(path, args, aw * ah * 3)):
        if j < skip or j >= skip + want:
            continue
        bgr = np.frombuffer(buf, dtype=np.uint8).reshape(ah, aw, 3)
        times.append(round((k0 + j - skip) / float(SAMPLE_FPS), 3))
        grays.append(_small_gray(bgr))
        found.append(faces.detect(bgr))
    return times, grays, found


# ---------------------------------------------------------------- cuts

def lagged_diff(grays: list, lag: int = LAG) -> np.ndarray:
    d = np.zeros(len(grays), dtype=np.float64)
    for t in range(lag, len(grays)):
        d[t] = float(np.mean(np.abs(grays[t] - grays[t - lag])))
    return d


def changed_share(a: np.ndarray, b: np.ndarray) -> float:
    """Share of thumbnail cells whose gray level moved by more than CELL_DIFF."""
    return float(np.mean(np.abs(a - b) > CELL_DIFF))


def threshold(d: np.ndarray) -> float:
    vals = d[LAG:] if d.size > LAG else d
    if vals.size == 0:
        return MIN_THRESHOLD
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med)))
    return max(MIN_THRESHOLD, med + MAD_K * mad)


def step_diff(grays: list) -> np.ndarray:
    """Single-step mean absolute difference (frame t vs t-1); step[0] = 0."""
    st = np.zeros(len(grays), dtype=np.float64)
    for t in range(1, len(grays)):
        st[t] = float(np.mean(np.abs(grays[t] - grays[t - 1])))
    return st


def spike_ok(steps, thr: float) -> bool:
    """True when the largest of the native-rate steps around a candidate cut is an isolated
    spike: >= SPIKE_THR x thr, >= SPIKE_ISOLATION x the next largest step and >= SPIKE_RATIO x
    the median of the others."""
    st = np.asarray(steps, dtype=np.float64)
    if st.size < 3:
        return False
    k = int(np.argmax(st))
    rest = np.delete(st, k)
    med, second = float(np.median(rest)), float(np.max(rest))
    return bool(st[k] >= SPIKE_THR * thr and st[k] >= SPIKE_ISOLATION * max(second, 1e-3)
                and st[k] >= SPIKE_RATIO * max(med, 1e-3))


def detect_cuts(times: list, grays: list, spike=None) -> tuple:
    """Coarse cuts from the lagged difference → ([(time, strength)], threshold); cuts are at
    least MIN_SHOT_GAP apart, each placed at the largest single-step change in its lag window.

    A cut must also rewrite MIN_CHANGED of the thumbnail cells (a gesture does not). A hot run
    below that floor whose largest 4 fps step still reaches SPIKE_THR x the threshold is a
    candidate hard switch between similar framings: `spike(t, thr)` (analyze passes a
    native-rate check, see spike_ok) decides it. Without `spike` such runs are rejected."""
    if len(grays) <= LAG:
        return [], MIN_THRESHOLD
    d = lagged_diff(grays)
    thr = threshold(d)
    st = step_diff(grays)
    hot = d > thr
    cands = []
    t = LAG
    while t < len(d):
        if not hot[t]:
            t += 1
            continue
        run_end = t
        while run_end + 1 < len(d) and hot[run_end + 1]:
            run_end += 1
        p = t + int(np.argmax(d[t:run_end + 1]))
        lo = max(1, p - LAG + 1)
        j = lo + int(np.argmax(st[lo:p + 1]))
        # Spatial extent: a camera change rewrites most of the frame, while a hand, a lean or a
        # head turn moves a large mean difference through a minority of cells.
        changed = max(changed_share(grays[q], grays[q - LAG]) for q in range(t, run_end + 1))
        if changed < MIN_CHANGED:
            tc = (times[j - 1] + times[j]) / 2.0
            if spike is None or st[j] < SPIKE_THR * thr or not spike(tc, thr):
                t = run_end + 1
                continue
        cands.append(((times[j - 1] + times[j]) / 2.0, float(d[p])))
        t = run_end + 1
    edge_lo, edge_hi = times[0] + EDGE_GUARD, times[-1] - EDGE_GUARD
    cands = [c for c in cands if edge_lo < c[0] < edge_hi]
    keep = []
    for c in sorted(cands, key=lambda c: -c[1]):
        if all(abs(c[0] - k[0]) >= MIN_SHOT_GAP for k in keep):
            keep.append(c)
    return sorted(keep), thr


def off_grid(video: dict) -> bool:
    """True when the source's frames do not all sit on its nominal frame grid: it was flagged
    variable frame rate at ingest, or its stream holds fewer frames than its duration needs
    (dropped frames anywhere in the file). Frame ordinals are then not grid slots."""
    if video.get("vfr"):
        return True
    from . import probe
    return bool(probe.frames_missing(video))


def native_frame_count(win: float, fps) -> int:
    """Frames a constant-rate source yields for a `win` s native window (see _native_frames)."""
    return int(math.ceil((win + 0.2) * float(fps) - 0.5 - 1e-9))


def _native_frames(path, src: dict, t: float, win: float) -> tuple:
    """(t0, small gray frames at the native rate) from the grid frame nearest t - win/2 for
    about win + 0.2 s; frame j is the picture on screen at t0 + j / fps. `-enc_time_base -1`
    keeps the rawvideo muxer on the decoder's timestamps: MKV/WebM sources otherwise print "non
    monotonically increasing dts" on some seeks, which the timestamp-damage deny list turns
    into an error.

    A constant-rate source is decoded frame by frame (frame j IS grid slot k0 + j). A source
    with missing frames (off_grid) is decoded onto the nominal grid the way render's stage A
    does it, so a slot whose frame is missing holds the frame that was on screen: without that,
    every missing frame before a camera change moved the reported boundary one frame early."""
    v = src["video"]
    fps = framing.fps_of(v["fps"])
    grid0 = float(v.get("start") or 0.0)
    k0 = int(round((max(0.0, t - win / 2) - grid0) * fps))
    t0 = grid0 + float(k0 / fps)
    w, h = framing.display_size(src)
    dh = max(2, int(round(DIFF_W * h / float(w))))
    scale = "scale=%d:%d:flags=area,format=gray" % (DIFF_W, dh)
    if off_grid(v):
        # Seek to a grid time up to one second early (never before 0 s) so the frame showing at
        # t0 is decoded even when the frame AT t0 is missing; `fps` (round=down after a quarter
        # frame lean) fills every grid slot with the frame on screen; trim then counts slots.
        kmin = int(math.ceil(-grid0 * float(fps) - 1e-9))
        pre = max(0, min(k0 - kmin, int(math.ceil(float(fps)))))
        g = max(0.0, grid0 + float((k0 - pre) / fps))
        n = native_frame_count(win, fps)
        vf = ("setpts=PTS+%.6f/TB,fps=%d/%d:round=down:start_time=0,trim=start_frame=%d:end_frame=%d,"
              "setpts=PTS-STARTPTS,%s" % (0.25 / float(fps), fps.numerator, fps.denominator,
                                          pre, pre + n, scale))
        args = ["-ss", "%.6f" % g, "-i", str(path), "-map", "0:v:0", "-vf", vf]
    else:
        seek = max(0.0, t0 - 0.5 / float(fps))
        args = ["-ss", "%.6f" % seek, "-t", "%.6f" % (win + 0.2), "-i", str(path), "-map", "0:v:0",
                "-vf", scale]
    args += ["-fps_mode", "passthrough", "-enc_time_base", "-1", "-f", "rawvideo", "-pix_fmt", "gray", "-"]
    frames = [np.frombuffer(b, dtype=np.uint8).astype(np.float32).reshape(dh, DIFF_W)
              for b in _stream(path, args, DIFF_W * dh)]
    return t0, frames


def spike_at(path, src: dict, t: float, thr: float) -> bool:
    """Native-rate check of a candidate hard cut near t (see spike_ok)."""
    _t0, frames = _native_frames(path, src, t, 2 * SPIKE_WIN)
    steps = [float(np.mean(np.abs(frames[j] - frames[j - 1]))) for j in range(1, len(frames))]
    return spike_ok(steps, thr)


def refine_cut(path, src: dict, t: float, thr: float) -> float:
    """Place a coarse cut on the exact source frame (first frame of the new shot)."""
    fps = framing.fps_of(src["video"]["fps"])
    t0, frames = _native_frames(path, src, t, REFINE_WIN)
    if len(frames) < 3:
        return t
    steps = [float(np.mean(np.abs(frames[j] - frames[j - 1]))) for j in range(1, len(frames))]
    j = 1 + int(np.argmax(steps))
    if steps[j - 1] < REFINE_MIN * thr:
        return t                      # soft change (dissolve): keep the coarse midpoint
    return round(t0 + float(j / fps), 4)


def blend_shares(frames: list):
    """Per frame, how far the picture is from the window's opening shot (0.0) to its closing
    shot (1.0): the least-squares mix factor against the first and last frames. None when the
    two ends are too alike to measure a blend."""
    n = len(frames)
    if n < 6:
        return None
    k = max(1, min(3, n // 6))
    a = np.mean(np.stack(frames[:k]), axis=0)
    b = np.mean(np.stack(frames[-k:]), axis=0)
    d = (b - a).ravel().astype(np.float64)
    den = float(d @ d)
    if den / d.size < TRANS_MIN_STEP ** 2:
        return None
    return [float(((f - a).ravel().astype(np.float64) @ d) / den) for f in frames]


def classify_transition(shares):
    """("cut" | "dissolve", first, mid, full) frame indices from blend shares, or None.

    `mid` = the frame where the new shot takes over (the one upward crossing of one half),
    `first` = the first frame that mixes in the new shot, `full` = the first frame that is the
    new shot alone. A hard cut has first == mid == full. A dissolve has DISSOLVE_MIN or more
    mixed frames (first .. full - 1) that rise all the way. None when the window does not show
    one clean change (two crossings, or the blend runs off either end)."""
    if not shares:
        return None
    n = len(shares)
    ups = [j for j in range(1, n) if shares[j] >= 0.5 > shares[j - 1]]
    if len(ups) != 1:
        return None
    mid = first = full = ups[0]
    while first > 0 and shares[first - 1] > TRANS_LO:
        first -= 1
    while full < n and shares[full] < TRANS_HI:
        full += 1
    if first == 0 or full >= n:
        return None
    if any(v > TRANS_LO + TRANS_BACK for v in shares[:first]) or \
            any(v < TRANS_HI - TRANS_BACK for v in shares[full:]):
        return None
    rising = all(shares[j + 1] >= shares[j] - TRANS_BACK for j in range(first, full))
    if full - first >= DISSOLVE_MIN and rising:
        return "dissolve", first, mid, full
    return "cut", full, full, full


def transition_at(path, src: dict, t: float, thr: float) -> dict:
    """What a detected boundary near t is: {"at_src", "kind": "cut" | "dissolve", "start_src",
    "end_src"}. A cut sits on the first frame of the new shot (start == end == at). A cross
    dissolve reports the first blended frame (start), the first clean frame of the new shot
    (end) and the first frame that is mostly the new shot (at: its midpoint)."""
    res, t0, fps = None, 0.0, None
    for win in TRANS_WIN:
        try:
            fps = framing.fps_of(src["video"]["fps"])
            t0, frames = _native_frames(path, src, t, win)
            res = classify_transition(blend_shares(frames))
        except (env.HCError, KeyError, TypeError, ValueError):
            res = None
        if res is not None:
            # a blend that runs to the edge of the window may be longer than the window shows
            edge = max(2, int(round(0.15 * float(fps))))
            if res[0] == "cut" or (res[1] >= edge and res[3] <= len(frames) - 1 - edge):
                break
            res = None
    if res and res[0] == "dissolve":
        _kind, first, mid, full = res
        tm = lambda j: round(t0 + float(j / fps), 4)   # noqa: E731
        return {"at_src": tm(mid), "kind": "dissolve", "start_src": tm(first), "end_src": tm(full)}
    rt = refine_cut(path, src, t, thr)
    return {"at_src": rt, "kind": "cut", "start_src": rt, "end_src": rt}


# ---------------------------------------------------------------- faces per shot

def cluster_faces(entries: list) -> list:
    """[(sample_index, face)] → clusters (single linkage on x with CLUSTER_TOL)."""
    pts = sorted(entries, key=lambda e: e[1][0])
    clusters, cur = [], []
    for e in pts:
        if cur and e[1][0] - cur[-1][1][0] > CLUSTER_TOL:
            clusters.append(cur)
            cur = []
        cur.append(e)
    if cur:
        clusters.append(cur)
    return clusters


def shot_faces(samples: list) -> list:
    """Per-shot face summary: [{"x", "y", "h", "presence"}] left to right."""
    n = len(samples)
    if n == 0:
        return []
    entries = [(i, f) for i, s in enumerate(samples) for f in s["faces"]]
    out = []
    for c in cluster_faces(entries):
        presence = len({i for i, _ in c}) / float(n)
        if presence < MIN_PRESENCE:
            continue
        arr = np.array([f[:4] for _, f in c], dtype=np.float64)
        out.append({"x": round(float(np.median(arr[:, 0])), 3), "y": round(float(np.median(arr[:, 1])), 3),
                    "h": round(float(np.median(arr[:, 3])), 3), "presence": round(presence, 2)})
    if out:
        top = max(f["h"] for f in out)
        if top >= WIDE_FACE_H:
            for f in out:
                if f["h"] < MINOR_FACE * top:
                    f["minor"] = True
    return sorted(out, key=lambda f: f["x"])


def shot_kind(fl: list) -> str:
    if not fl:
        return "none"
    if max(f["h"] for f in fl) < WIDE_FACE_H:
        return "wide"
    n = max(1, sum(1 for f in fl if f["presence"] >= COUNT_PRESENCE and not f.get("minor")))
    return {1: "single", 2: "two"}.get(n, "group")


def similar_faces(fl: list) -> bool:
    """Two or more faces present in at least MULTI_PRESENCE of the samples whose heights are
    within MULTI_RATIO of the largest: people of similar size, whatever the absolute size (two
    small faces in a wide two-shot count; a face on a poster far smaller than the speaker does
    not)."""
    seen = [f for f in fl or [] if float(f.get("presence") or 0.0) >= MULTI_PRESENCE
            and float(f.get("h") or 0.0) > 0]
    if len(seen) < 2:
        return False
    top = max(float(f["h"]) for f in seen)
    return sum(1 for f in seen if float(f["h"]) * MULTI_RATIO >= top) >= 2


def faces_multi(fl: list, kind: str = None) -> bool:
    """The shots json `multi_face` flag for a shot's face summary."""
    return (kind if kind is not None else shot_kind(fl)) in ("two", "group") or similar_faces(fl)


SPEC_KEYS = ("at_src", "mode", "x", "y", "crop", "rect", "reaction")   # framing mark keys
TURN_WORDS = 3            # speaker_turns: a run of one label needs this many words


def is_multi(shot: dict) -> bool:
    """Two or more people of similar size: the largest face is not the speaker, so the default
    is letterbox and proposals follow the speakers map instead (never one face's track). A shots
    json written before the similar-size rule is judged again from its face list."""
    if shot.get("multi_face"):
        return True
    return shot.get("kind") in ("two", "group") or similar_faces(shot.get("faces"))


def label_faces(faces: list) -> list:
    """Give the counted (non-minor) faces ids "A", "B", ... left to right (in place)."""
    k = 0
    for f in sorted(faces, key=lambda f: f["x"]):
        if f.get("minor"):
            f.pop("id", None)
            continue
        f["id"] = chr(ord("A") + k) if k < 26 else "F%d" % k
        k += 1
    return faces


def speaker_turns(words: list, ranges: list, min_words: int = TURN_WORDS) -> list:
    """[{"at_src", "sp"}]: where the speaking label changes inside the analysed ranges, from
    transcript runs of one label with at least `min_words` words (shorter runs are skipped)."""
    out = []
    for a, b in ranges:
        run = []
        for w in words:
            if not (a <= float(w["s"]) < b) or not w.get("sp"):
                continue
            if run and run[-1]["sp"] != w["sp"]:
                if len(run) >= min_words and (not out or out[-1]["sp"] != run[0]["sp"]):
                    out.append({"at_src": round(float(run[0]["s"]), 3), "sp": run[0]["sp"]})
                run = []
            run.append(w)
        if len(run) >= min_words and (not out or out[-1]["sp"] != run[0]["sp"]):
            out.append({"at_src": round(float(run[0]["s"]), 3), "sp": run[0]["sp"]})
    return out


def _shot_turns(words: list, a: float, b: float) -> list:
    """Speaker turns inside [a, b) by the framing rules: a run shorter than framing.TURN_MIN_S
    or with fewer than framing.TURN_MIN_WORDS words does not switch (an interjection)."""
    runs = []
    for w in words:
        if not (a <= float(w["s"]) < b) or not w.get("sp"):
            continue
        if runs and runs[-1][0] == w["sp"]:
            runs[-1][1].append(w)
        else:
            runs.append([w["sp"], [w]])
    turns = []
    for lab, rw in runs:
        if float(rw[-1]["e"]) - float(rw[0]["s"]) < framing.TURN_MIN_S or \
                len(rw) < framing.TURN_MIN_WORDS:
            continue
        if turns and turns[-1]["sp"] == lab:
            continue
        turns.append({"sp": lab, "s": float(rw[0]["s"])})
    return turns


def propose(shots: list, speakers=None, words=None) -> list:
    """One framing proposal per shot (per speaker turn in a multi-face shot with a speakers map).

    Single-face shots track their face. Multi-face shots never track the largest face: with a
    `speakers` x map covering every label speaking in the shot they get one track mark per turn
    (`speaker` names who), else a letterbox entry listing the faces (`multi_face`, `faces`).
    Extra keys are informational; paste only SPEC_KEYS (print_result does)."""
    smap = {str(k): v for k, v in (speakers or {}).items()
            if isinstance(v, dict) and v.get("x") is not None}
    out = []
    for s in shots:
        main = [f for f in s.get("faces") or [] if not f.get("minor")]
        if is_multi(s):
            end = s.get("end_src")
            turns = _shot_turns(words or [], s["start_src"], end if end is not None else math.inf)
            if turns and all(t["sp"] in smap for t in turns):
                for k, t in enumerate(turns):
                    at = s["start_src"] if k == 0 else round(max(s["start_src"],
                                                                 t["s"] - framing.TURN_LEAD), 3)
                    out.append({"at_src": at, "mode": "track", "x": float(smap[t["sp"]]["x"]),
                                "crop": 1.0, "speaker": t["sp"]})
                continue
            out.append({"at_src": s["start_src"], "mode": "letterbox", "multi_face": True,
                        "faces": [{"id": f.get("id"), "x": f["x"], "h": f["h"]} for f in main]})
            continue
        if s["kind"] == "single" and main:
            best = max(main, key=lambda f: (f["presence"], f["h"]))
            out.append({"at_src": s["start_src"], "mode": "track", "x": best["x"], "crop": 1.0})
        else:
            out.append({"at_src": s["start_src"], "mode": "letterbox"})
    return out


def paste_block(proposal: list) -> list:
    """The proposal with only framing-mark keys (what may be pasted into clips.json)."""
    return [{k: m[k] for k in SPEC_KEYS if k in m} for m in proposal]


# ---------------------------------------------------------------- analyze

def _range_id(a: float, b: float) -> str:
    return "range-%d-%d" % (int(a), int(round(b)))


def _transcript_words(proj) -> list:
    """The source transcript's words ([] when there is none yet)."""
    try:
        return list((proj.transcript() or {}).get("words") or [])
    except (env.HCError, OSError, ValueError):
        return []


def _speakers_map(proj) -> dict:
    """clips.json "speakers" ({} when there is no clips.json or no map)."""
    try:
        spec = proj.spec() or {}
    except (env.HCError, OSError, ValueError):
        return {}
    sp = spec.get("speakers") if isinstance(spec, dict) else None
    return sp if isinstance(sp, dict) else {}


def covers(data, windows: list, source: dict) -> bool:
    """True when a shots json holds every sample the given source-time windows need: it was
    made from this source on the absolute sample grid, and each window (clamped to the source)
    lies inside one analysed range. Framing reads samples by fixed windows, so data that covers
    them gives the same tracks whatever else was analysed."""
    if not isinstance(data, dict) or data.get("grid") != GRID:
        return False
    if int(data.get("sample_fps") or 0) != SAMPLE_FPS:
        return False
    fp, have = (source or {}).get("fingerprint"), data.get("source_fingerprint")
    if fp and have and fp != have:
        return False
    dur = float((source or {}).get("duration") or 0.0)
    ranges = [(float(r[0]), float(r[1])) for r in data.get("ranges") or [] if len(r) == 2]
    for lo, hi in windows or []:
        lo, hi = max(0.0, float(lo)), (min(dur, float(hi)) if dur > 0 else float(hi))
        ts = sample_times(lo, hi)
        if not ts:
            continue
        if not any(a - 1e-6 <= ts[0] and ts[-1] <= b + 1e-6 for a, b in ranges):
            return False
    return True


def analyze(project, clip_id: str = None, range_src=None, *, ranges=None) -> dict:
    """Analyse a clip's source ranges (or one explicit range) → shots json dict (written).
    `ranges` ([(a, b), ...] in source seconds) replaces the clip's own ranges (render passes the
    ranges of the plan it is about to render)."""
    proj = ru.as_project(project)
    src = proj.source()
    path = proj.source_path()
    duration = float(src.get("duration") or 0.0)
    if ranges is not None and clip_id:
        ranges = [(round(max(0.0, float(a)), 3),
                   round(min(duration, float(b)) if duration > 0 else float(b), 3))
                  for a, b in ranges if float(b) - float(a) > 0.2]
        if not ranges:
            raise env.HCError("%s has no source ranges to analyse" % clip_id, code=1)
        cid = clip_id
    elif range_src is not None:
        a, b = (env.parse_time(x) for x in range_src)
        if duration > 0:
            a, b = max(0.0, a), min(duration, b)
        if b - a < 0.5:
            raise env.HCError("--range needs END after START (at least 0.5 s)", code=1)
        ranges = [(round(a, 3), round(b, 3))]
        cid = clip_id or _range_id(a, b)
    elif clip_id:
        ranges = clip_ranges(proj, clip_id, duration)
        cid = clip_id
    else:
        raise env.HCError("shots needs a clip id or --range START END", code=1)
    samples, shots, cuts_log, transitions = [], [], [], []
    for a, b in ranges:
        times, grays, found = _decode_range(path, src, a, b)
        if not times:
            continue
        rs = [{"t_src": t, "faces": [f.rounded(4) for f in fl]} for t, fl in zip(times, found)]
        samples += rs
        cuts, thr = detect_cuts(times, grays,
                                spike=lambda tc, th: spike_at(path, src, tc, th))
        bounds = []
        for t, strength in cuts:
            tr = transition_at(path, src, t, thr)
            rt = tr["at_src"]
            if not a < rt < b:
                continue
            bounds.append(rt)
            transitions.append(tr)
            cuts_log.append({"coarse_src": round(t, 3), "src": rt, "strength": round(strength, 2),
                             "threshold": round(thr, 2), "kind": tr["kind"],
                             "start_src": tr["start_src"], "end_src": tr["end_src"]})
        edges = [a] + sorted(bounds) + [b]
        for s0, s1 in zip(edges, edges[1:]):
            if s1 - s0 <= 0:
                continue
            inside = [s for s in rs if s0 <= s["t_src"] < s1] or \
                [min(rs, key=lambda s: abs(s["t_src"] - (s0 + s1) / 2))]
            fl = label_faces(shot_faces(inside))
            kind = shot_kind(fl)
            shots.append({"start_src": round(s0, 4), "end_src": round(s1, 4), "kind": kind,
                          "multi_face": faces_multi(fl, kind), "faces": fl})
    words, smap = _transcript_words(proj), _speakers_map(proj)
    transitions.sort(key=lambda tr: tr["at_src"])
    result = {"version": SHOTS_VERSION, "clip": cid, "range_src": [ranges[0][0], ranges[-1][1]],
              "ranges": [list(r) for r in ranges], "sample_fps": SAMPLE_FPS, "grid": GRID,
              "analysis_width": ANALYSIS_WIDTH, "source_fingerprint": src.get("fingerprint"),
              "multi_face": any(s["multi_face"] for s in shots),
              "speaker_turns": speaker_turns(words, ranges),
              "shots": shots, "transitions": transitions, "samples": samples,
              "proposal": propose(shots, smap, words), "cuts": cuts_log}
    env.write_json(proj.shots_path(cid), result)
    result["sheet"] = str(write_sheet(proj, cid, result, path, src))
    return result


# ---------------------------------------------------------------- sheet

def _annotate(bgr: np.ndarray, shot: dict, mark: dict, src: dict):
    from PIL import Image, ImageDraw
    img = Image.fromarray(bgr[:, :, ::-1].copy())
    w, h = img.size
    dr = ImageDraw.Draw(img)
    font = ru.label_font(15)
    for i in range(1, 10):
        x = int(w * i / 10.0)
        dr.line([(x, 0), (x, 8)], fill=RULER, width=2)
        dr.line([(x, h - 9), (x, h - 1)], fill=RULER, width=2)
        dr.text((x + 2, 9), ".%d" % i, fill=RULER, font=font)
    for f in faces.detect(bgr):
        x0, y0, fw, fh = f.box_px(w, h)
        dr.rectangle([x0, y0, x0 + fw, y0 + fh], outline=FACE_BOX, width=2)
    for fc in shot["faces"]:
        x = int(fc["x"] * w)
        dr.polygon([(x, h - 26), (x - 8, h - 12), (x + 8, h - 12)], fill=WINDOW)
        tag = ("%s " % fc["id"]) if fc.get("id") else ""
        dr.text((min(w - 110, max(2, x - 40)), h - 46),
                "%sx=%.2f %d%%" % (tag, fc["x"], round(fc["presence"] * 100)), fill=WINDOW, font=font)
    if mark.get("mode") in ("track", "fixed"):
        sw, sh = framing.display_size(src)
        cw, ch = framing.crop_size(sw, sh, "vertical", mark.get("crop", 1.0))
        fy = next((fc["y"] for fc in shot["faces"] if abs(fc["x"] - mark["x"]) < 1e-6), None)
        x0, y0 = framing.crop_xy(mark["x"], fy, cw, ch, sw, sh, "track")
        k = w / float(sw)
        dr.rectangle([x0 * k, y0 * k, (x0 + cw) * k - 1, (y0 + ch) * k - 1], outline=WINDOW, width=3)
    else:
        dr.text((10, h // 2 - 8), "LETTERBOX", fill=WINDOW, font=ru.label_font(22))
    return img


def _shot_marks(result: dict) -> list:
    """The first proposal entry of each shot (a multi-face shot may have one per turn)."""
    out = []
    for shot in result["shots"]:
        a, b = shot["start_src"], shot.get("end_src", math.inf)
        mk = next((m for m in result["proposal"] if a - 1e-6 <= m["at_src"] < b), None)
        out.append(mk or {"at_src": a, "mode": "letterbox"})
    return out


def write_sheet(proj, cid: str, result: dict, path, src: dict) -> Path:
    out = proj.sheet_path("shots_%s.jpg" % cid)
    sw, sh = framing.display_size(src)
    items = []
    for i, (shot, mark) in enumerate(zip(result["shots"], _shot_marks(result))):
        mid = (shot["start_src"] + shot["end_src"]) / 2.0
        bgr = ru.grab_at(path, mid, SHEET_W, (sw, sh))
        if bgr is None:
            continue
        what = ("track x=%.2f" % mark["x"]) if mark["mode"] == "track" else mark["mode"]
        if is_multi(shot):
            what = "MULTI-FACE " + (("follows %s" % mark["speaker"]) if mark.get("speaker")
                                    else "letterbox (set \"speakers\")")
        label = "#%d  %s-%s  %s  %s" % (i + 1, env.fmt_time(shot["start_src"], 2),
                                        env.fmt_time(shot["end_src"], 2), shot["kind"], what)
        items.append((_annotate(bgr, shot, mark, src), label))
    if not items:
        items = [(ru.placeholder("no frames decoded"), "")]
    return ru.contact_sheet(items, SHEET_COLS, out, SHEET_W, title="shots: %s" % cid)


# ---------------------------------------------------------------- command

def print_result(proj, res: dict) -> None:
    n = len(res["shots"])
    print("shots %s  %s-%s  (%d shot%s, %d samples)" % (
        res["clip"], env.fmt_time(res["range_src"][0]), env.fmt_time(res["range_src"][1]), n,
        "" if n == 1 else "s", len(res["samples"])))
    print("  %-3s %-11s %-11s %-7s %s" % ("#", "start", "end", "kind", "faces (x presence h)"))
    for i, s in enumerate(res["shots"]):
        fl = "  ".join("x=%.2f %d%% h=%.2f%s" % (f["x"], round(f["presence"] * 100), f["h"],
                                                " (background)" if f.get("minor") else "")
                       for f in s["faces"]) or "-"
        print("  %-3d %-11s %-11s %-7s %s" % (i + 1, env.fmt_time(s["start_src"], 2),
                                             env.fmt_time(s["end_src"], 2), s["kind"], fl))
    for line in multi_face_lines(res):
        print(line)
    for tr in res.get("transitions") or []:
        if tr.get("kind") == "dissolve":
            print("DISSOLVE %s-%s in the source: the shot changes at its midpoint %s (framing "
                  "marks placed inside it switch there); frames inside it show both shots, so "
                  "keep piece edges and cover frames out of it."
                  % (env.fmt_time(tr["start_src"], 2), env.fmt_time(tr["end_src"], 2),
                     env.fmt_time(tr["at_src"], 2)))
    print("proposed framing marks (paste into clips.json \"framing\", then check the sheet):")
    print(json.dumps(paste_block(res["proposal"]), indent=2))
    print("sheet: %s" % proj.rel(res["sheet"]))


def multi_face_lines(res: dict) -> list:
    """One MULTI-FACE line per multi-face shot: the faces, the speaker turns in it and how to
    make framing follow them."""
    out = []
    turns = res.get("speaker_turns") or []
    mapped = {m.get("speaker") for m in res.get("proposal") or [] if m.get("speaker")}
    for s in res["shots"]:
        if not is_multi(s):
            continue
        a, b = s["start_src"], s.get("end_src", math.inf)
        faces = ", ".join("%s x=%.2f" % (f.get("id") or "?", f["x"]) for f in s["faces"]
                          if not f.get("minor")) or "none counted"
        inside = [t for t in turns if a <= t["at_src"] < b]
        before = [t for t in turns if t["at_src"] < a]
        if before and (not inside or inside[0]["at_src"] > a + 0.5):
            inside.insert(0, {"at_src": a, "sp": before[-1]["sp"]})    # still talking at the cut
        labs = []
        for t in inside:
            if t["sp"] not in labs:
                labs.append(t["sp"])
        tl = ", ".join("%s %s" % (env.fmt_time(t["at_src"], 1), t["sp"]) for t in inside) or \
            "none (no speaker labels in the transcript: transcribe with --speakers-vtt)"
        if labs and all(lb in mapped for lb in labs):
            how = "the speakers map covers it: the proposal follows the turns."
        else:
            example = ", ".join('"%s": {"x": ...}' % lb for lb in (labs or ["<label>"]))
            how = ('set clips.json "speakers": {%s} once (x of each person\'s face above), '
                   "then framing follows turns." % example)
        out.append("MULTI-FACE shot %s-%s: faces %s; speaker turns: %s; %s Never paste a "
                   "single-face mark for this shot." % (env.fmt_time(a, 2), env.fmt_time(b, 2),
                                                        faces, tl, how))
    return out


def cmd_shots(args) -> int:
    proj = ru.as_project(args.project)
    rng = getattr(args, "range", None)
    if rng:
        print_result(proj, analyze(proj, range_src=list(rng)))
        return 0
    ids = list(getattr(args, "clips", None) or [])
    if not ids:
        spec = proj.spec()
        ids = [c for c, v in (spec.get("clips") or {}).items() if (v.get("kind") or "cut") == "cut"]
    if not ids:
        raise env.HCError("no clips to analyse (give clip ids or --range START END)", code=1)
    for cid in ids:
        print_result(proj, analyze(proj, clip_id=cid))
    return 0
