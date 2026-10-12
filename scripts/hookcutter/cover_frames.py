"""Clean base frames and face boxes for cover and QA (private helper of cover.py / qa.py).

Base frames are the rendered picture without captions: read from the stage A segments when
render has produced them, else approximated from the source at the plan's src times (the
framing crop approximated from the framing marks, centre crop by default).
"""

from __future__ import annotations

import inspect
import math
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from . import env

SK_ROOT = Path(__file__).resolve().parents[2]
YUNET_MODEL = SK_ROOT / "models" / "face_detection_yunet_2023mar.onnx"
FACE_MIN_SCORE = 0.6
SELECT_CHUNK = 90          # frame indices per ffmpeg select expression
OUT_SIZES = {"vertical": (1080, 1920), "landscape": (1920, 1080)}


def out_size(fmt: str) -> tuple[int, int]:
    """Output frame size for a format name."""
    return OUT_SIZES.get(fmt, OUT_SIZES["vertical"])


# ---------------------------------------------------------------- faces

_DETECTORS: dict = {}


def _yunet(w: int, h: int):
    import cv2
    det = _DETECTORS.get("yunet")
    if det is None:
        if not YUNET_MODEL.is_file():
            raise env.HCError(f"face model missing: {YUNET_MODEL}", code=3)
        det = cv2.FaceDetectorYN.create(str(YUNET_MODEL), "", (w, h), FACE_MIN_SCORE, 0.3, 50)
        _DETECTORS["yunet"] = det
    det.setInputSize((w, h))
    return det


def detect_faces(img, min_score: float = FACE_MIN_SCORE) -> list[dict]:
    """YuNet faces on an RGB image (PIL or HxWx3 uint8).

    Returns pixel boxes sorted largest first:
    {"box": [x, y, w, h], "score": s, "landmarks": [[x, y] * 5]} (right eye, left eye,
    nose tip, right mouth corner, left mouth corner, as YuNet orders them).
    """
    arr = np.asarray(img.convert("RGB")) if isinstance(img, Image.Image) else np.asarray(img)
    if arr.ndim != 3 or arr.shape[0] < 16 or arr.shape[1] < 16:
        return []
    h, w = arr.shape[:2]
    bgr = np.ascontiguousarray(arr[:, :, ::-1])
    try:  # B4's shared YuNet wrapper when present
        from . import faces as _faces
        found = _faces.detect(bgr, score_threshold=min_score)
    except ImportError:
        found = None
    if found is not None:
        out = []
        for f in found:
            x, y, fw, fh = f.box_px(w, h)
            lms = [[float(a), float(b)] for a, b in f.landmarks_px(w, h)]
            if len(lms) != 5:
                lms = [[x + fw * 0.3, y + fh * 0.4], [x + fw * 0.7, y + fh * 0.4], [x + fw * 0.5, y + fh * 0.6],
                       [x + fw * 0.35, y + fh * 0.8], [x + fw * 0.65, y + fh * 0.8]]
            out.append({"box": [round(x, 1), round(y, 1), round(fw, 1), round(fh, 1)],
                        "score": round(float(f[4]), 3), "landmarks": lms})
        out.sort(key=lambda f: f["box"][3], reverse=True)
        return out
    _, rows = _yunet(w, h).detect(bgr)
    faces = []
    if rows is None:
        return faces
    for r in rows:
        score = float(r[14])
        if score < min_score:
            continue
        x, y, fw, fh = (float(v) for v in r[:4])
        x0, y0 = max(0.0, x), max(0.0, y)
        x1, y1 = min(float(w), x + fw), min(float(h), y + fh)
        if x1 - x0 < 4 or y1 - y0 < 4:
            continue
        lms = [[float(r[4 + 2 * i]), float(r[5 + 2 * i])] for i in range(5)]
        faces.append({"box": [round(x0, 1), round(y0, 1), round(x1 - x0, 1), round(y1 - y0, 1)],
                      "score": round(score, 3), "landmarks": lms})
    faces.sort(key=lambda f: f["box"][3], reverse=True)
    return faces


def box_overlap(a, b) -> float:
    """Intersection area of two [x, y, w, h] boxes."""
    ix = min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])
    iy = min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1])
    return max(0.0, ix) * max(0.0, iy)


# ---------------------------------------------------------------- the top of the head

CROWN_REACH = 0.9          # search this far above the face-box top (x face height)
CROWN_CENTRE = 0.18        # half-width of the column band through the middle of the head (x face width)
CROWN_SIDE = (0.85, 1.25)  # the background bands either side of the head (x face width from its centre)
CROWN_HEAD_MIN = 40.0      # the forehead must differ from the background by this (RGB distance) ...
CROWN_T = (0.35, 16.0, 48.0)   # ... and a head row by this share of that difference (clamped)
CROWN_RUN = 0.05           # background must hold for this x face height above the head
CROWN_FOLLOW = 24.0        # the running background follows rows within this of it (RGB distance)
CROWN_MARGIN = 0.02        # the band turns to the head a little below its apex (x face height)


def crown_from_image(img, box) -> float | None:
    """y of the top of the head (hair or scalp) above a face box [x, y, w, h], read from the
    picture, or None when the picture does not show it clearly.

    A face box starts near the hairline; how far the head goes on above it depends on the hair
    (about 0.3 x the face height for a full head of hair, under 0.1 for a bald or close-cropped
    one), so it is measured. A row above the face box belongs to the head while the middle of
    the head (a band CROWN_CENTRE wide) differs in colour both from the background bands beside
    the head and from the background above it (followed down from the top of the search
    window, so a lighting gradient is not mistaken for a head). Walking up from the forehead,
    the head ends where that stops for CROWN_RUN of the face height. Medians over each band
    ignore thin structures (slats, a lamp cord).

    None when the head does not stand out from what is around it, when no background shows
    above it inside the frame, or when there is no room for either side band. Hair the same
    colour as the background behind it is not seen (the viewer does not see where it ends
    either). The value errs high (a smaller y) rather than low.
    """
    arr = np.asarray(img.convert("RGB")) if isinstance(img, Image.Image) else np.asarray(img)
    if arr.ndim != 3 or arr.shape[2] < 3:
        return None
    H, W = arr.shape[:2]
    try:
        x, y, w, h = (float(v) for v in box[:4])
    except (TypeError, ValueError):
        return None
    if w < 16 or h < 16:
        return None
    cx = x + w / 2.0
    top = int(max(0, math.floor(y - CROWN_REACH * h)))
    bot = int(min(H, math.ceil(y + 0.10 * h)))
    y_box = int(round(y))
    if y_box - top < 4 or bot - y_box < 2 or y_box >= H:
        return None

    def band(a, b):
        a, b = int(round(max(0.0, a))), int(round(min(float(W), b)))
        if b - a < max(6, 0.15 * w):
            return None
        return np.median(arr[top:bot, a:b, :3].astype(np.float32), axis=1)      # rows x 3

    centre = band(cx - CROWN_CENTRE * w, cx + CROWN_CENTRE * w)
    sides = [s for s in (band(cx - CROWN_SIDE[1] * w, cx - CROWN_SIDE[0] * w),
                         band(cx + CROWN_SIDE[0] * w, cx + CROWN_SIDE[1] * w)) if s is not None]
    if centre is None or not sides:
        return None
    # How far the head's middle is from the nearer side band ...
    d_side = np.min([np.linalg.norm(centre - s, axis=1) for s in sides], axis=0)
    # ... and from the background above it: a running colour that follows the middle band down
    # from the top of the window and stops following where the picture departs from it.
    n_top = max(2, int(round(0.06 * h)))
    bg = np.median(centre[:n_top], axis=0)
    d_bg = np.empty(len(centre), np.float32)
    for r in range(len(centre)):
        dist = float(np.linalg.norm(centre[r] - bg))
        d_bg[r] = dist
        if dist <= CROWN_FOLLOW:
            bg = 0.9 * bg + 0.1 * centre[r]
    d = np.minimum(d_side, d_bg)
    k = max(1, int(round(0.02 * h)))
    if k > 1:
        pad = np.pad(d, (k // 2, k - 1 - k // 2), mode="edge")
        d = np.convolve(pad, np.ones(k, np.float32) / k, mode="valid")
    row_box = y_box - top
    head = float(np.median(d[row_box:]))            # the forehead rows, inside the face box
    if head < CROWN_HEAD_MIN:
        return None
    share, lo, hi = CROWN_T
    thr = min(hi, max(lo, share * head))
    need = max(3, int(round(CROWN_RUN * h)))
    # Up from the forehead: the head ends below the first stretch of `need` background rows.
    # Shorter background-coloured stretches are part of the head (the hairline, where skin in
    # shadow can pass through the colour of the wall; a highlight; a headband).
    last_head, run = None, 0
    for r in range(len(d) - 1, -1, -1):
        if d[r] > thr:
            last_head, run = r, 0
        else:
            run += 1
            if run >= need:
                break
    else:
        return None                                 # no clear background above the head
    if last_head is None:
        return None
    return float(max(0.0, top + last_head - CROWN_MARGIN * h))


# ---------------------------------------------------------------- frame extraction

def _probe_size(path: Path) -> tuple[int, int]:
    data = env.ffprobe_json(path, "-select_streams", "v:0", "-show_entries",
                            "stream=width,height,sample_aspect_ratio:stream_side_data=rotation:stream_tags=rotate")
    streams = data.get("streams") or []
    if not streams:
        raise env.HCError(f"no video stream in {path}", code=1)
    s = streams[0]
    w, h = int(s["width"]), int(s["height"])
    rot = 0
    for sd in s.get("side_data_list") or []:
        if "rotation" in sd:
            rot = int(float(sd["rotation"]))
    if not rot and (s.get("tags") or {}).get("rotate"):
        rot = int(float(s["tags"]["rotate"]))
    sar = str(s.get("sample_aspect_ratio") or "1:1")
    try:
        sn, sd_ = (int(v) for v in sar.split(":"))
        if sn > 0 and sd_ > 0 and sn != sd_:
            w = int(round(w * sn / sd_))
    except ValueError:
        pass
    if abs(rot) % 180 == 90:
        w, h = h, w
    return w, h


def grab_frames(path, indices, size: tuple[int, int] | None = None, ss: float | None = None,
                pre_vf: str = "", grid: tuple[int, int] | None = None) -> dict:
    """Decode frames by index (index 0 = first frame at/after `ss`) as RGB arrays.

    One decode pass per chunk of indices; frames come back in index order. `size` (w, h)
    scales every frame (square pixels assumed after the internal SAR fix); None keeps the
    upright square-pixel size. `pre_vf` is a filter chain applied to the selected frames
    before scaling (the HDR tone-map).

    `grid` (num, den) decodes onto that frame grid the way render's stage A does (`ss` must
    then be a grid time): index n is the frame ON SCREEN at ss + n * den / num, so a source
    with missing frames (variable frame rate, dropped frames) keeps the frame that is showing
    instead of counting the next one early. Without it an index is a decoded-frame ordinal,
    which is the same thing only for a constant-rate file (the stage A segments).
    """
    path = Path(path)
    want = sorted({int(i) for i in indices if i is not None and int(i) >= 0})
    if not want:
        return {}
    if size is None:
        size = _probe_size(path)
    w, h = int(size[0]), int(size[1])
    got: dict[int, np.ndarray] = {}
    for c0 in range(0, len(want), SELECT_CHUNK):
        chunk = want[c0:c0 + SELECT_CHUNK]
        expr = "+".join(f"eq(n\\,{i})" for i in chunk)
        pre = (pre_vf.rstrip(",") + ",") if pre_vf else ""
        head = ""
        if grid:
            num, den = int(grid[0]), int(grid[1])       # render._sub_chain's mapping
            head = "setpts=PTS+%.6f/TB,fps=%d/%d:round=down:start_time=0," % (0.25 * den / num, num, den)
        vf = f"{head}select='{expr}',{pre}scale={w}:{h}:flags=bicubic,setsar=1,format=rgb24"
        cmd = [env.ffmpeg(), "-hide_banner", "-nostdin", "-loglevel", "error"]
        if ss is not None and ss > 0:
            cmd += ["-ss", f"{ss:.6f}"]
        # -enc_time_base -1: the rawvideo muxer keeps the decoder's timestamps, so MKV/WebM
        # sources never trip "non monotonically increasing dts" after a seek.
        cmd += ["-i", str(path), "-map", "0:v:0", "-vf", vf, "-fps_mode", "passthrough",
                "-enc_time_base", "-1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
        proc = subprocess.run(cmd, capture_output=True)
        if proc.returncode != 0:
            msg = proc.stderr.decode("utf-8", "replace").strip().splitlines()[-3:]
            raise env.HCError(f"frame extraction failed for {path.name}: {' | '.join(msg)}", code=3)
        frame_bytes = w * h * 3
        n = len(proc.stdout) // frame_bytes
        for j in range(min(n, len(chunk))):
            arr = np.frombuffer(proc.stdout, np.uint8, frame_bytes, j * frame_bytes).reshape(h, w, 3)
            got[chunk[j]] = arr
    return got


# ---------------------------------------------------------------- stage A segment lookup

def _strict_lookup(fn) -> bool:
    """render.segment_paths takes a plan and a style (the content-keyed lookup)."""
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    return "plan" in params and "style" in params


def segment_files(project, clip_id: str, fmt: str, plan: dict, style: str | None = None) -> list[Path] | None:
    """Stage A segment files in plan segment order, or None when render has not made them.

    With a `style`: the content-keyed files a render of exactly this plan in this style uses
    (render.segment_paths with style and plan), or None when any is missing. A render made
    from another plan, and a style whose stage A differs (another letterbox position or punch
    in), are then never mistaken for this one: the caller falls back to the source and says so.
    Without a style: the files of the newest render log of clip x format, whatever plan and
    style made it."""
    segs = plan.get("segments") or []
    if not segs:
        return None
    try:
        from . import render  # B4
        fn = render.segment_paths
        if style is not None and _strict_lookup(fn):
            paths = fn(project, clip_id, fmt, style=style, plan=plan)
        else:
            paths = fn(project, clip_id, fmt)
    except Exception:
        paths = None
    if not paths or len(paths) != len(segs):
        return None
    paths = [Path(p) if Path(p).is_absolute() else Path(project.root) / p for p in paths]
    if not all(p.is_file() for p in paths):
        return None
    return paths


# ---------------------------------------------------------------- source fallback framing

FACE_Y = 0.38              # a tracked face centre sits this far down the crop (as render frames it)
TONEMAP = ("zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,"
           "tonemap=tonemap=hable:desat=0,zscale=t=bt709:m=bt709:r=tv,format=yuv420p")


def _framing():
    try:
        from . import framing
        return framing
    except Exception:  # pragma: no cover - framing is part of the package
        return None


def mark_at(plan: dict, t_src: float, default: dict) -> dict:
    """Framing mark in effect at a src time (last mark with at_src <= t, else default)."""
    best = None
    for m in plan.get("framing") or []:
        try:
            if float(m.get("at_src", 0.0)) <= t_src + 1e-6:
                best = m
        except (TypeError, ValueError):
            continue
    return best or default


def _source_info(project) -> dict | None:
    """source.json (video size, sar, active_area, hdr), or None when unreadable."""
    try:
        data = project.source()
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _shots(project, clip_id: str):
    try:
        sp = Path(project.shots_path(clip_id))
        if sp.is_file():
            return env.read_json(sp)
    except Exception:
        pass
    return None


def default_mark(project, clip_id: str, fmt: str, t_src: float | None = None,
                 shots=None, source=None) -> dict:
    """Clip default framing at a source time, as render picks it (contract 7, framing
    .default_mark): landscape is full; a multi-face shot is letterboxed; a single-face shot
    tracks that face; otherwise a fixed crop at the centre of the active picture."""
    if fmt == "landscape":
        return {"mode": "full"}
    fr = _framing()
    shots = shots if shots is not None else _shots(project, clip_id)
    source = source if source is not None else _source_info(project)
    if fr is not None:
        try:
            m = fr.default_mark(fmt, shots, float(t_src or 0.0), source=source)
            if isinstance(m, dict):
                return m
        except Exception:
            pass
    return {"mode": "fixed", "x": 0.5, "y": 0.5, "crop": 1.0}


def _visible(w: dict) -> bool:
    return not (w.get("hidden") or w.get("omitted") or w.get("removed") or w.get("filler")
                or w.get("edge"))


def _label_at(plan: dict, t_src: float):
    """Speaker label of the visible word at (or next after) a source time."""
    best = None
    for w in plan.get("words") or []:
        if not (_visible(w) and w.get("sp")):
            continue
        try:
            s, e = float(w.get("s_src", 0.0)), float(w.get("e_src", 0.0))
        except (TypeError, ValueError):
            continue
        if s <= t_src < e:
            return w["sp"]
        if s >= t_src and (best is None or s < best[0]):
            best = (s, w["sp"])
    return best[1] if best else None


def _speaker_mark(plan: dict, t_src: float, mark: dict | None) -> dict | None:
    """The speaker-follow framing render uses (plan "speakers" map covering every label, and
    either no marks or a "speaker" mark in effect): a track on the speaking person's x."""
    fr = _framing()
    if fr is None:
        return None
    try:
        if not fr.follow_ok(plan):
            return None
        if (plan.get("framing") and not (mark and str(mark.get("mode")) == "speaker")):
            return None
        smap = fr.speaker_map(plan)
        lab = _label_at(plan, t_src)
        if lab not in smap:
            return None
        crop = float((mark or {}).get("crop") or 1.0)
        return {"mode": "track", "x": smap[lab]["x"], "y": smap[lab].get("y"), "crop": crop,
                "speaker": lab}
    except Exception:
        return None


def _active_px(source, img_w: int, img_h: int) -> tuple:
    """The source's active picture (framing.active_rect) in the pixels of a decoded frame."""
    fr = _framing()
    if fr is None or not source:
        return 0, 0, img_w, img_h
    try:
        dw, dh = fr.display_size(source)
        ax, ay, aw, ah = fr.active_rect(source)
    except Exception:
        return 0, 0, img_w, img_h
    sx, sy = img_w / float(dw), img_h / float(dh)
    x0, y0 = max(0, int(round(ax * sx))), max(0, int(round(ay * sy)))
    x1, y1 = min(img_w, int(round((ax + aw) * sx))), min(img_h, int(round((ay + ah) * sy)))
    if x1 - x0 < 8 or y1 - y0 < 8:
        return 0, 0, img_w, img_h
    return x0, y0, x1 - x0, y1 - y0


def _rect_px(rect, img_w: int, img_h: int):
    """A mark's normalized `rect` [x, y, w, h] in decoded-frame pixels (None when unusable)."""
    if not isinstance(rect, (list, tuple)) or len(rect) != 4:
        return None
    try:
        x, y, w, h = (float(v) for v in rect)
    except (TypeError, ValueError):
        return None
    x0, y0 = max(0.0, x) * img_w, max(0.0, y) * img_h
    x1, y1 = min(1.0, x + w) * img_w, min(1.0, y + h) * img_h
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    return int(round(x0)), int(round(y0)), int(round(x1 - x0)), int(round(y1 - y0))


def _num(v, default):
    try:
        return default if v is None else float(v)
    except (TypeError, ValueError):
        return default


def fit_frame(img: Image.Image, mark: dict, out_w: int, out_h: int, letterbox_y: float | None = None,
              active: tuple | None = None) -> Image.Image:
    """Approximate the render's framing of one source frame (upright, square pixels).

    `active` (x, y, w, h in `img` pixels) is the active picture inside any letterbox bars or
    gallery border (default the whole frame): crops stay inside it, and a letterbox shows it
    (or the mark's normalized `rect`) as foreground and blurred fill. `letterbox_y` is the
    letterbox centre as a fraction of the output height (default 0.5).
    """
    W, H = img.size
    ax, ay, aw, ah = active if active else (0, 0, W, H)
    mark = mark or {}
    mode = mark.get("mode", "fixed")
    a = out_w / out_h
    if mode in ("letterbox", "full"):
        rect = _rect_px(mark.get("rect"), W, H) if mode == "letterbox" else None
        rx, ry, rw, rh = rect or (ax, ay, aw, ah)
        pic = img.crop((rx, ry, rx + rw, ry + rh))
        if mode == "full" and abs(rw / rh - a) < 0.01:
            return pic.resize((out_w, out_h), Image.LANCZOS)
        scale = max(out_w / rw, out_h / rh)
        bg = pic.resize((max(1, int(rw * scale)), max(1, int(rh * scale))), Image.BILINEAR)
        left, top = (bg.width - out_w) // 2, (bg.height - out_h) // 2
        bg = bg.crop((left, top, left + out_w, top + out_h)).filter(ImageFilter.GaussianBlur(28))
        bg = Image.eval(bg, lambda v: int(v * 0.65))
        fw = out_w
        fh = max(1, int(round(rh * out_w / rw)))
        if fh > out_h:
            fh = out_h
            fw = max(1, int(round(rw * out_h / rh)))
        fg = pic.resize((fw, fh), Image.LANCZOS)
        cy = (letterbox_y if letterbox_y is not None else 0.5) * out_h
        y0 = int(round(min(max(cy - fh / 2, 0), out_h - fh)))
        bg.paste(fg, ((out_w - fw) // 2, y0))
        return bg
    if aw / ah > a:
        mh, mw = ah, ah * a
    else:
        mw, mh = aw, aw / a
    c = min(1.0, max(0.3, _num(mark.get("crop"), 1.0)))
    cw, ch = mw * c, mh * c
    cx = _num(mark.get("x"), (ax + aw / 2.0) / W) * W
    fy = mark.get("y")
    if fy is None:
        y_top = ay + (ah - ch) / 2.0
    elif mode == "fixed":
        y_top = _num(fy, 0.5) * H - ch / 2.0
    else:
        y_top = _num(fy, 0.5) * H - FACE_Y * ch
    x0 = min(max(cx - cw / 2, ax), ax + aw - cw)
    y0 = min(max(y_top, ay), ay + ah - ch)
    box = (int(round(x0)), int(round(y0)), int(round(x0 + cw)), int(round(y0 + ch)))
    return img.crop(box).resize((out_w, out_h), Image.LANCZOS)


def source_hdr(project, plan: dict | None = None) -> str | None:
    """The source's HDR transfer ("hlg", "pq", ...) or None for SDR."""
    src = _source_info(project) or {}
    hdr = (src.get("video") or {}).get("hdr")
    if hdr is None:
        hdr = ((plan or {}).get("source") or {}).get("hdr")
    hdr = str(hdr or "none").lower()
    return None if hdr in ("", "none", "sdr") else hdr


def _hdr_chain(hdr: str | None) -> str:
    """Tone-map chain for an HDR source, as render uses; HCError(code 3) without zscale."""
    if not hdr:
        return ""
    if not env.has_filter("zscale"):
        raise env.HCError("the source is HDR (%s) and this ffmpeg has no zscale filter to tone-map "
                          "it to SDR. Run: hc doctor (it can point HOOKCUTTER_FFMPEG at an ffmpeg "
                          "build with zscale)" % hdr, code=3)
    try:
        from .render import TONEMAP as chain
    except Exception:
        chain = TONEMAP
    return chain


# ---------------------------------------------------------------- base frames at out times

def _fps(plan: dict) -> tuple[int, int]:
    fps = plan.get("fps") or [25, 1]
    return int(fps[0]), int(fps[1])


FRAME_TOL = 1e-3           # frames: plan and cover times are rounded to 6 decimals


def frame_index(seconds: float, fps) -> int:
    """Index of the frame on screen `seconds` after frame 0 on a (num, den) frame grid.

    Times in a plan and in a cover json are rounded to 6 decimals, so the time of frame n can
    be a hair under n * den / num; the tolerance (a thousandth of a frame) maps a recorded time
    back to its own frame instead of the one before it. `hc status` uses the same expression
    (cli.cover_stale_reasons): keep the two identical."""
    num, den = int(fps[0]), int(fps[1])
    return int(math.floor(float(seconds) * num / den + FRAME_TOL))


def seg_at(plan: dict, t_out: float) -> tuple[int, dict] | None:
    """Plan segment containing an out time (last segment for t at the very end)."""
    segs = plan.get("segments") or []
    for k, s in enumerate(segs):
        o0, o1 = float(s["out"][0]), float(s["out"][1])
        if o0 - 1e-6 <= t_out < o1 - 1e-6:
            return k, s
    if segs and t_out >= float(segs[-1]["out"][0]):
        return len(segs) - 1, segs[-1]
    return None


def source_media(project, clip: dict) -> Path:
    """The media the plan's src clock refers to (source, or the clip's source_file)."""
    if (clip or {}).get("kind") == "caption_only" and clip.get("source_file"):
        p = Path(clip["source_file"]).expanduser()
        return p if p.is_absolute() else Path(project.root) / p
    return Path(project.source_path())


def frame_mark(project, clip_id: str, clip: dict, plan: dict, fmt: str, t_src: float, *,
               shots=None, source=None) -> dict:
    """The framing render applies at a source time: an explicit mark, else speaker-follow
    (plan "speakers" map), else the clip default (framing.default_mark). A track mark without
    x takes the default's x (multi-face: letterbox)."""
    if (clip or {}).get("kind") == "caption_only":
        return {"mode": "fixed", "x": None, "y": None, "crop": 1.0}
    if fmt == "landscape":
        return {"mode": "full"}
    m = mark_at(plan, t_src, None)
    if m is not None and str(m.get("mode")) != "speaker":
        m = dict(m)
        if m.get("mode") == "track" and m.get("x") is None:
            d = default_mark(project, clip_id, fmt, t_src, shots, source)
            if d.get("mode") in ("letterbox", "full"):
                return dict(d, crop=m.get("crop", 1.0))
            m["x"] = d.get("x")
        return m
    sm = _speaker_mark(plan, t_src, m)
    if sm is not None:
        return sm
    return default_mark(project, clip_id, fmt, t_src, shots, source)


def base_frames(project, clip_id: str, clip: dict, plan: dict, fmt: str, times_out,
                size: tuple[int, int], letterbox_y: float | None = None,
                style: str | None = None) -> tuple[dict, str]:
    """Clean frames (RGB arrays at `size`) at output times.

    Returns ({t_out: array}, origin) with origin "segments" (stage A files) or "source"
    (approximated from the source with render's framing rules, inside the active picture).
    `letterbox_y` is a fraction of the output height. An HDR source is tone-mapped as render
    does it; without zscale that is HCError(code 3), the same error render gives.
    `style` names the style the frames are for: its own stage A files of this plan are used
    (see segment_files), and when render has not made them the origin is "source". The source
    is read on the plan's frame grid as render reads it, so a source with missing frames gives
    the frame render would show.
    """
    times = [float(t) for t in times_out]
    num, den = _fps(plan)
    files = segment_files(project, clip_id, fmt, plan, style=style)
    want: dict[int, list] = {}
    for t in times:
        hit = seg_at(plan, t)
        if hit is None:
            continue
        k, s = hit
        want.setdefault(k, []).append((t, s))
    out: dict[float, np.ndarray] = {}
    if files:
        for k, items in want.items():
            s = plan["segments"][k]
            frames = int(s.get("frames") or 0)
            idx = {}
            for t, _ in items:
                n = frame_index(t - float(s["out"][0]), (num, den))
                idx[t] = min(max(n, 0), max(frames - 1, 0))
            got = grab_frames(files[k], idx.values(), size=size)
            for t, n in idx.items():
                if n in got:
                    out[t] = got[n]
        if len(out) == len(times):
            return out, "segments"
        out = {}
    # source fallback
    src = source_media(project, clip)
    if not src.exists():
        raise env.HCError(f"cannot read base frames: {src} is missing", code=2)
    caption_only = (clip or {}).get("kind") == "caption_only"
    source = None if caption_only else _source_info(project)
    pre_vf = "" if caption_only else _hdr_chain(source_hdr(project, plan))
    shots = None if caption_only or fmt == "landscape" else _shots(project, clip_id)
    sw, sh = _probe_size(src)
    scale = min(1.0, max(size) * 1.6 / max(sw, sh))
    dec = (max(2, int(round(sw * scale / 2)) * 2), max(2, int(round(sh * scale / 2)) * 2))
    active = _active_px(source, dec[0], dec[1]) if source else None
    grid0 = float(plan.get("grid_offset") or 0.0)
    for k, items in want.items():
        s = plan["segments"][k]
        s0 = float(s["src"][0])
        # seek to a grid time half a second early and decode onto the grid (as stage A does):
        # slot n is then grid frame k0 + n even where the source has missing frames
        k0 = math.ceil((max(0.0, s0 - 0.5) - grid0) * num / den - 1e-6)
        ss = max(0.0, grid0 + k0 * den / num)
        idx = {}
        for t, _ in items:
            t_src = s0 + (t - float(s["out"][0]))
            kk = int(round((t_src - grid0) * num / den))
            idx[t] = (max(0, kk - k0), t_src)
        got = grab_frames(src, [v[0] for v in idx.values()], size=dec, ss=ss, pre_vf=pre_vf,
                          grid=(num, den))
        for t, (n, t_src) in idx.items():
            arr = got.get(n)
            if arr is None:
                continue
            mark = frame_mark(project, clip_id, clip, plan, fmt, t_src, shots=shots, source=source)
            img = fit_frame(Image.fromarray(arr), mark, size[0], size[1], letterbox_y, active)
            out[t] = np.asarray(img)
    return out, "source"
