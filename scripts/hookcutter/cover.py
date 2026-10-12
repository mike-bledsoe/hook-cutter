"""Cover frames: score candidates, compose the 1080x1920 cover, grid and mobile previews.

Contract section 11. The cover is what the profile grid shows, so composition is judged on the
3:4 grid crop (y 240..1680) and a phone-scale preview, not only the full frame.
"""

from __future__ import annotations

import itertools
import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from . import cover_frames as cf
from . import env

COVER_W, COVER_H = 1080, 1920
CANDIDATE_STEP = 0.5          # seconds between candidate frames
MAX_CANDIDATES = 240
ANALYSIS_SIZE = (360, 640)    # candidate scoring resolution (9:16)
TOP_N = 6                     # candidates on the sheet
MIN_SEPARATION = 1.5          # seconds between sheet candidates
SHIFT_STEP = 40
SHIFT_MAX = 480
HEADLINE_STEP_DOWN = 8         # px per step when shrinking a headline so the head stays in the tile
FADE_PX = 260                 # photo bottom-edge fade when shifted
GRID_TILE_W = 360             # covers-grid tile width (phone scale, 3 columns)
MOBILE_W = 240
JPEG_Q = 92
GRID_LOCK = ".covers-grid.lock"   # in the project's cache folder: one covers-grid rebuild at a time
CANDIDATE_VERSION = 5          # bump when candidate framing or scoring changes
BLEND_STEP = 2                 # a candidate is compared with the frames this far either side of it
BLEND_MIN = 8.0                # mean luma change (0..255) between those two frames worth a look ...
BLEND_LINEAR = 0.25            # ... a dissolve when the frame is their average to within this share
BLEND_HARD = 18.0              # ... and a picture changing this fast is no still either way
BLEND_SIZE = (45, 80)          # the frames are compared at this size
FACE_OVERLAP = 0.35            # two face boxes sharing this much of the smaller one: two pictures
CROWN_FRAC = 0.30             # the top of the head sits about this x the face-box height above the box
CLOSEUP_SHIFT_MAX = 80        # a close-up (head above the grid band even unshifted) may shift this far
TINT_LEAD = 300               # the tint starts this far above the headline top ...
TINT_FULL = 120               # ... reaches 0.8 x tint_alpha at the top and tint_alpha this far below
LINE_CONTRAST = 4.5           # local tint is raised until each headline line reaches this ...
LINE_CONTRAST_WARN = 3.0      # ... and a line still under this is reported
ZERO_WIDTH = frozenset("\u200b\u200c\u200d\u2060\ufeff\u00ad")   # dropped from headlines (as captions do)
DEFAULT_COVER = {
    "headline_font": "ArchivoBlack-Regular.ttf", "size_range": [96, 168], "max_lines": 4,
    "color": "#FFFFFF", "accent": "#FFD84D", "band": [240, 1680], "margin": 70,
    "tint": "#0B0B0D", "tint_from": 0.42, "tint_alpha": 0.86, "rule": False,
    "lockup": [], "lockup_font": "Montserrat[wght].ttf:SemiBold", "lockup_size": 34,
}
DEFAULT_SHADOW = [{"blur": 16, "alpha": 0.45, "dy": 4}, {"blur": 3, "alpha": 0.55, "dy": 2}]
SK_FONTS = cf.SK_ROOT / "fonts"


# ---------------------------------------------------------------- small utils

def parse_color(value, default=(255, 255, 255, 255)) -> tuple[int, int, int, int]:
    """'#RRGGBB' or '#RRGGBBAA' -> RGBA tuple."""
    if not isinstance(value, str):
        return default
    h = value.strip().lstrip("#")
    try:
        if len(h) == 6:
            return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16), 255
        if len(h) == 8:
            return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16), int(h[6:8], 16)
    except ValueError:
        pass
    return default


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def save_jpeg(img: Image.Image, path: Path, quality: int = JPEG_Q) -> None:
    """Atomic JPEG write (tmp + rename). The temp name is this writer's own: two runs on
    different clips both rewrite the shared sheets/covers-grid.jpg."""
    from . import fsutil
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = fsutil.tmp_path(path)
    try:
        img.convert("RGB").save(tmp, "JPEG", quality=quality, subsampling=0, optimize=True)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def strip_invisible(text: str) -> tuple[str, list[str]]:
    """`text` without zero-width characters and soft hyphens (fonts draw them as boxes), plus
    the characters removed."""
    text = str(text or "")
    gone = [ch for ch in text if ch in ZERO_WIDTH]
    if not gone:
        return text, []
    return "".join(ch for ch in text if ch not in ZERO_WIDTH), list(dict.fromkeys(gone))


def parse_runs(text: str) -> list[tuple[str, bool]]:
    """Headline words with accent flags; *...* marks accent runs. Stray asterisks are dropped,
    and so are zero-width characters (see strip_invisible)."""
    text, _ = strip_invisible(text)
    words: list[tuple[str, bool]] = []
    pos = 0
    glue = False                 # the previous chunk ended inside a word ("*VALUE*," -> "VALUE,")

    def add(chunk: str, acc: bool) -> None:
        nonlocal glue
        parts = chunk.replace("*", " ").split()
        if parts and glue and words and not chunk[:1].isspace():
            prev, pacc = words[-1]
            head = parts.pop(0)
            # the merged word takes the accent of its letters ('"' + WORD keeps WORD's accent)
            words[-1] = (prev + head, acc if not any(ch.isalnum() for ch in prev) else pacc)
        words.extend((w, acc) for w in parts)
        glue = bool(chunk) and not chunk[-1:].isspace()

    for m in re.finditer(r"\*([^*]+)\*", text or ""):
        add(text[pos:m.start()], False)
        add(m.group(1), True)
        pos = m.end()
    add((text or "")[pos:], False)
    return words


def runs_to_markup(words: list[tuple[str, bool]]) -> str:
    """Inverse of parse_runs for one line: accent runs wrapped in asterisks."""
    out, prev = [], False
    for w, acc in words:
        if acc and not prev:
            w = "*" + w
        if prev and not acc:
            out[-1] = out[-1] + "*"
        out.append(w)
        prev = acc
    if prev and out:
        out[-1] = out[-1] + "*"
    return " ".join(out)


# ---------------------------------------------------------------- fonts (typeset when available)

try:  # B5's typeset is the shared text engine; the local fallback keeps covers working alone
    from . import typeset as _ts
except Exception:  # pragma: no cover - only when typeset is absent or broken
    _ts = None

_FONT_CACHE: dict = {}


def _root(project):
    from .project import root_of
    return root_of(project)


def font_path(spec: str, project=None) -> tuple[Path, str | None]:
    """Resolve 'File.ttf' or 'File.ttf:Instance' to (path, instance) via the font search path."""
    if _ts is not None:
        path, inst = _ts.resolve_font(spec, _root(project))
        return Path(path), inst
    m = re.match(r"^(.*\.(?:ttf|otf|ttc))(?::([^/\\:]+))?$", spec or "", re.I)
    name, inst = (m.group(1), m.group(2)) if m else (spec or "", None)
    cands = []
    p = Path(name).expanduser()
    if p.is_absolute():
        cands.append(p)
    else:
        if project is not None:
            cands.append(Path(project.root) / "fonts" / name)
        try:
            cands.append(Path(env.data_dir()) / "fonts" / name)
        except Exception:
            pass
        cands.append(SK_FONTS / name)
    for c in cands:
        if c.is_file():
            return c, (inst or None)
    raise env.HCError(f"font not found: {name} (searched project fonts/, user fonts/, skill fonts/)", code=1)


def load_font(spec: str, size: int, project=None) -> ImageFont.FreeTypeFont:
    """Pillow font for a preset font spec at a pixel size (BASIC layout, variable instance set)."""
    if _ts is not None:
        return _ts.load_font(spec, size, _root(project))
    path, inst = font_path(spec, project)
    key = (str(path), inst, int(size))
    f = _FONT_CACHE.get(key)
    if f is None:
        f = ImageFont.truetype(str(path), int(size), layout_engine=ImageFont.Layout.BASIC)
        if inst:
            try:
                f.set_variation_by_name(inst)
            except Exception as exc:
                raise env.HCError(f"font {path.name} has no instance '{inst}': {exc}", code=1)
        _FONT_CACHE[key] = f
    return f


def missing_glyphs(spec: str, text: str, project=None) -> list[str]:
    """Characters of `text` the font cannot draw (fontTools cmap check; whitespace ignored)."""
    path, _ = font_path(spec, project)
    if _ts is not None:
        return list(_ts.missing_glyphs(str(path), text))
    key = ("cmap", str(path))
    cmap = _FONT_CACHE.get(key)
    if cmap is None:
        from fontTools.ttLib import TTFont
        with TTFont(str(path), lazy=True) as tt:
            cmap = set(tt.getBestCmap() or {})
        _FONT_CACHE[key] = cmap
    return sorted({ch for ch in text if not ch.isspace() and ord(ch) not in cmap})


# ---------------------------------------------------------------- wrap and fit

def balanced_wrap(widths: list[float], space: float, max_w: float, max_lines: int,
                  accents: list[bool] | None = None) -> list[list[int]] | None:
    """Fewest lines (<= max_lines) that fit max_w, balanced; None when nothing fits.

    Local fallback for typeset.balanced_wrap. Among partitions with that line count:
    minimise line-width variance, avoid a one-word last line (widow) and avoid splitting an
    accent run across lines.
    """
    n = len(widths)
    if n == 0:
        return []
    if max(widths) > max_w:
        return None
    acc = accents or [False] * n

    def line_w(a, b):
        return sum(widths[a:b]) + space * (b - a - 1)

    for k in range(1, min(max_lines, n) + 1):
        best, best_cost = None, None
        for cuts in itertools.combinations(range(1, n), k - 1):
            bounds = (0,) + cuts + (n,)
            ws = [line_w(bounds[i], bounds[i + 1]) for i in range(k)]
            if max(ws) > max_w:
                continue
            mean = sum(ws) / k
            cost = sum((w - mean) ** 2 for w in ws) / (max_w ** 2)
            if k >= 2 and n >= 3 and bounds[-1] - bounds[-2] == 1:
                cost += 0.08
            for c in cuts:
                if acc[c - 1] and acc[c]:
                    cost += 0.05
            key = (round(cost, 9), -sum(w * (k - i) for i, w in enumerate(ws)))  # ties: top-heavy
            if best_cost is None or key < best_cost:
                best, best_cost = bounds, key
        if best is not None:
            return [list(range(best[i], best[i + 1])) for i in range(k)]
    return None


def wrap_words(texts: list[str], accents: list[bool], font, max_w: float, max_lines: int):
    """Balanced wrap of headline words (typeset's engine when present). Lists of word indices."""
    if _ts is not None:
        def measure(idxs):
            return font.getlength(" ".join(texts[i] for i in idxs))
        return _ts.balanced_wrap(list(range(len(texts))), measure, max_w, max_lines)
    widths = [font.getlength(t) for t in texts]
    return balanced_wrap(widths, font.getlength(" "), max_w, max_lines, accents)


def fit_headline(words: list[tuple[str, bool]], spec: str, size_range, max_lines: int,
                 max_w: float, max_h: float, line_gap: float = 1.0, project=None) -> dict:
    """Largest size in size_range (step 2 px) whose balanced wrap fits width, line count and height.

    Falls back below the range (never below 24 px) rather than letting text leave the frame;
    `fallback` is set when that happened.
    """
    lo, hi = int(size_range[0]), int(size_range[1])
    texts = [w for w, _ in words]
    accents = [a for _, a in words]

    def attempt(size, lines_cap):
        f = load_font(spec, size, project)
        wrap = wrap_words(texts, accents, f, max_w, lines_cap)
        if wrap is None:
            return None
        if len(wrap) * size * line_gap > max_h and size > 24:
            return None
        return {"size": size, "wrap": wrap, "fallback": None}

    for size in range(hi, lo - 1, -2):
        got = attempt(size, max_lines)
        if got:
            return got
    for size in range(lo - 2, 23, -2):
        got = attempt(size, max_lines)
        if got:
            got["fallback"] = f"headline needed {size} px, below the preset minimum {lo} px"
            return got
    for size in range(lo, 23, -2):
        got = attempt(size, len(words))
        if got:
            got["fallback"] = f"headline needed {len(got['wrap'])} lines at {size} px (max_lines {max_lines})"
            return got
    got = {"size": 24, "wrap": [list(range(len(words)))],
           "fallback": "headline does not fit the frame even at 24 px; shorten it"}
    return got


# ---------------------------------------------------------------- layout

def layout_text(words, fit: dict, spec: str, cfg: dict, lockup: list[str], project=None,
                width: int = COVER_W) -> dict:
    """Place headline lines (centred) so the block bottom sits at band bottom - margin - lockup.

    Returns {"items": [(x, y_baseline, text, accent)], "headline_box", "lockup_items",
    "rule_box", "text_box", "size", "lines"} in cover coordinates.
    """
    size = fit["size"]
    font = load_font(spec, size, project)
    gap = float(cfg.get("line_gap", 1.0))
    band_top, band_bot = (int(v) for v in cfg.get("band", [240, 1680]))
    margin = int(cfg.get("margin", 70))
    lk_size = int(cfg.get("lockup_size", 34))
    rule = bool(cfg.get("rule"))
    lockup_h = 0
    if rule:
        lockup_h += 28 + 6
    if lockup:
        lockup_h += (22 if rule else 30) + int(round(len(lockup) * lk_size * 1.3))
    target_bottom = band_bot - margin - lockup_h

    draw = ImageDraw.Draw(Image.new("L", (8, 8)))
    items, ink = [], None
    lines_markup = []
    for li, idxs in enumerate(fit["wrap"]):
        line_words = [words[i] for i in idxs]
        text = " ".join(w for w, _ in line_words)
        lw = font.getlength(text)
        x0 = (width - lw) / 2
        yb = li * size * gap
        pos = 0
        for w, acc in line_words:
            x = x0 + font.getlength(text[:pos])
            items.append([x, yb, w, acc])
            bb = draw.textbbox((x, yb), w, font=font, anchor="ls")
            ink = bb if ink is None else (min(ink[0], bb[0]), min(ink[1], bb[1]),
                                          max(ink[2], bb[2]), max(ink[3], bb[3]))
            pos += len(w) + 1
        lines_markup.append(runs_to_markup(line_words))
    if ink is None:
        ink = (width / 2, target_bottom, width / 2, target_bottom)
    dy = target_bottom - ink[3]
    for it in items:
        it[1] += dy
    hb = [ink[0], ink[1] + dy, ink[2] - ink[0], ink[3] - ink[1]]
    text_box = list(hb)
    rule_box, lockup_items = None, []
    y = hb[1] + hb[3]
    if rule:
        rw = min(220.0, max(hb[2] * 0.4, 120.0))
        rule_box = [(width - rw) / 2, y + 28, rw, 6]
        y = rule_box[1] + 6
    if lockup:
        lk_font = load_font(cfg.get("lockup_font", DEFAULT_COVER["lockup_font"]), lk_size, project)
        y += 22 if rule else 30
        for line in lockup:
            lw = lk_font.getlength(line)
            lockup_items.append([(width - lw) / 2, y + lk_size, line])
            bb = draw.textbbox(((width - lw) / 2, y + lk_size), line, font=lk_font, anchor="ls")
            text_box = _union(text_box, [bb[0], bb[1], bb[2] - bb[0], bb[3] - bb[1]])
            y += lk_size * 1.3
    if rule_box:
        text_box = _union(text_box, rule_box)
    return {"items": items, "headline_box": [round(v, 1) for v in hb], "rule_box": rule_box,
            "lockup_items": lockup_items, "text_box": [round(v, 1) for v in text_box],
            "size": size, "lines": lines_markup}


def _union(a, b):
    x0, y0 = min(a[0], b[0]), min(a[1], b[1])
    x1, y1 = max(a[0] + a[2], b[0] + b[2]), max(a[1] + a[3], b[1] + b[3])
    return [x0, y0, x1 - x0, y1 - y0]


def choose_shift(text_box, faces: list[dict], base_shift: int = 0,
                 manual: bool = False) -> tuple[int, bool]:
    """Smallest shift (base, +40 ... <= 480) at which no important face overlaps the text box.

    Faces are in photo coordinates at shift 0. Returns (shift, still_overlapping); when every
    shift overlaps, the shift with the least overlap is returned. A `manual` shift (an explicit
    cover.shift, 0 included) is kept as given; only the overlap is reported.
    """
    important = _important_faces(faces)
    if not important or text_box is None:
        return int(base_shift), False
    pad = [text_box[0] - 10, text_box[1] - 10, text_box[2] + 20, text_box[3] + 20]
    if manual:
        s = int(base_shift)
        area = sum(cf.box_overlap(pad, [f["box"][0], f["box"][1] - s, f["box"][2], f["box"][3]])
                   for f in important)
        return s, area > 0
    shifts = list(range(int(base_shift), max(int(base_shift), SHIFT_MAX) + 1, SHIFT_STEP))
    if shifts[-1] < SHIFT_MAX and base_shift < SHIFT_MAX:
        shifts.append(SHIFT_MAX)
    best = (None, int(base_shift))
    for s in shifts:
        area = sum(cf.box_overlap(pad, [f["box"][0], f["box"][1] - s, f["box"][2], f["box"][3]])
                   for f in important)
        if area <= 0:
            return s, False
        if best[0] is None or area < best[0]:
            best = (area, s)
    return best[1], True


def _important_faces(faces: list[dict]) -> list[dict]:
    if not faces:
        return []
    big = max(f["box"][3] for f in faces)
    return [f for f in faces if f["box"][3] >= 0.6 * big]


def crown_y(box) -> float:
    """Estimated top of the head for a face box [x, y, w, h]: the box starts at the forehead.
    A guess for a full head of hair; a bald or close-cropped head ends far lower (see
    head_top, which measures it when the picture allows)."""
    return float(box[1]) - CROWN_FRAC * float(box[3])


def head_top(photo, box) -> tuple[float, str]:
    """(y of the top of the head, method) for a face box in `photo`: "image" when the picture
    shows where the head ends (cover_frames.crown_from_image), else "estimate" (crown_y: the
    face-box top minus CROWN_FRAC x its height). An estimate is a layout guess, not a finding:
    nothing warns that a head is trimmed on the strength of it."""
    try:
        y = cf.crown_from_image(photo, box)
    except Exception:           # a measurement aid: never the reason a cover fails
        y = None
    if y is not None:
        return float(min(y, float(box[1]))), "image"
    return crown_y(box), "estimate"


# ---------------------------------------------------------------- drawing

def _tint_alpha(text_top: float | None, cfg: dict) -> np.ndarray:
    """Tint alpha per row (COVER_H x 1). Anchored to the headline when there is one: 0 above
    text_top - TINT_LEAD, a smoothstep to 0.8 x tint_alpha at text_top, tint_alpha from
    text_top + TINT_FULL down. Without a headline: the preset ramp from tint_from."""
    t_alpha = float(cfg.get("tint_alpha", 0.86))
    alpha = np.zeros((COVER_H, 1), np.float32)
    if t_alpha <= 0:
        return alpha
    y = np.arange(COVER_H, dtype=np.float32)
    if text_top is None:
        y0 = int(round(min(max(float(cfg.get("tint_from", 0.42)), 0.0), 1.0) * COVER_H))
        if y0 < COVER_H:
            r = np.linspace(0.0, 1.0, COVER_H - y0)
            alpha[y0:, 0] = r * r * (3 - 2 * r) * t_alpha
        return alpha
    top = float(text_top)
    a0, a1 = top - TINT_LEAD, top
    r = np.clip((y - a0) / max(1.0, a1 - a0), 0.0, 1.0)
    ramp = r * r * (3 - 2 * r) * 0.8 * t_alpha
    full = np.clip((y - top) / float(TINT_FULL), 0.0, 1.0)
    ramp = np.where(y > top, (0.8 + 0.2 * full) * t_alpha, ramp)
    alpha[:, 0] = ramp
    return alpha


def _apply_tint(canvas: Image.Image, alpha: np.ndarray, tint) -> Image.Image:
    base = np.asarray(canvas.convert("RGB"), np.float32)
    tint_rgb = np.array(tint[:3], np.float32)[None, None, :]
    out = base * (1 - alpha[:, :, None]) + tint_rgb * alpha[:, :, None]
    return Image.fromarray(np.clip(out + 0.5, 0, 255).astype(np.uint8))


def _photo_layer(photo: Image.Image, shift: int, cfg: dict,
                 text_top: float | None = None) -> Image.Image:
    """Photo slid up by `shift` (bottom edge faded into the tint) plus the tint ramp, anchored
    to the headline top when given (see _tint_alpha)."""
    tint = parse_color(cfg.get("tint", "#0B0B0D"), (11, 11, 13, 255))
    canvas = Image.new("RGB", (COVER_W, COVER_H), tint[:3])
    ph = photo.convert("RGB").resize((COVER_W, COVER_H), Image.LANCZOS) \
        if photo.size != (COVER_W, COVER_H) else photo.convert("RGB")
    if shift > 0:
        # mask rows are in photo coordinates: the photo's own bottom rows fade into the tint
        mask = np.full((COVER_H, COVER_W), 255, np.uint8)
        fade = min(FADE_PX, COVER_H - shift)
        if fade > 0:
            ramp = np.linspace(1.0, 0.0, fade)
            ramp = ramp * ramp * (3 - 2 * ramp)
            mask[COVER_H - fade:, :] = (ramp * 255).astype(np.uint8)[:, None]
        canvas.paste(ph, (0, -shift), Image.fromarray(mask))
    else:
        canvas = ph.copy()
    alpha = _tint_alpha(text_top, cfg)
    if float(alpha.max()) > 0:
        canvas = _apply_tint(canvas, alpha, tint)
    return canvas


def _gray(img: Image.Image) -> np.ndarray:
    """Gamma-space luma 0..1 (Rec. 709 weights) of an RGB image."""
    a = np.asarray(img.convert("RGB"), np.float32) / 255.0
    return a[..., 0] * 0.2126 + a[..., 1] * 0.7152 + a[..., 2] * 0.0722


def _line_boxes(layout: dict, font) -> list[tuple]:
    """Ink box (x0, y0, x1, y1) of each headline line, in cover coordinates."""
    lines: dict = {}
    for x, yb, w, acc in layout["items"]:
        l, t, r, b = font.getbbox(w, anchor="ls")
        key = round(yb, 3)
        box = (x + l, yb + t, x + r, yb + b)
        cur = lines.get(key)
        lines[key] = box if cur is None else (min(cur[0], box[0]), min(cur[1], box[1]),
                                              max(cur[2], box[2]), max(cur[3], box[3]))
    return [lines[k] for k in sorted(lines)]


def _line_colors(layout: dict, cfg: dict) -> list:
    """Per headline line, the darkest text colour drawn on it (accent or plain)."""
    color = parse_color(cfg.get("color", "#FFFFFF"))[:3]
    accent = parse_color(cfg.get("accent", "#FFD84D"))[:3]
    by: dict = {}
    for _, yb, _, acc in layout["items"]:
        by.setdefault(round(yb, 3), set()).add(accent if acc else color)
    from .captions import _rel_lum
    return [min(by[k], key=_rel_lum) for k in sorted(by)]


def _line_contrast(canvas: Image.Image, box, rgb, shadow_s: float) -> float:
    """Shadow-aware contrast of `rgb` text on the p90 luma behind one line box."""
    from .captions import contrast_ratio
    g = _gray(canvas)
    x0, y0 = max(0, int(box[0]) - 4), max(0, int(box[1]) - 4)
    x1, y1 = min(COVER_W, int(math.ceil(box[2])) + 4), min(COVER_H, int(math.ceil(box[3])) + 4)
    if x1 <= x0 or y1 <= y0:
        return 21.0
    p90 = float(np.percentile(g[y0:y1, x0:x1], 90))
    return contrast_ratio(rgb, p90 * (1.0 - shadow_s))


def _raise_contrast(canvas: Image.Image, layout: dict, spec: str, cfg: dict,
                    project=None) -> tuple[Image.Image, list[float]]:
    """Darken the tint behind any headline line whose shadow-aware contrast is under
    LINE_CONTRAST (lines top to bottom; each extra tint feathers in over 160 px above the line and
    holds to the bottom of the frame). Returns (canvas, final contrast per line)."""
    from .captions import contrast_ratio
    font = load_font(spec, layout["size"], project)
    shadow = cfg.get("shadow") or DEFAULT_SHADOW
    s = _ts.shadow_strength(font, shadow) if _ts is not None and hasattr(_ts, "shadow_strength") \
        else 0.28
    tint = parse_color(cfg.get("tint", "#0B0B0D"), (11, 11, 13, 255))
    t_luma = float(_gray(Image.new("RGB", (1, 1), tint[:3]))[0, 0])
    boxes, colors = _line_boxes(layout, font), _line_colors(layout, cfg)
    y = np.arange(COVER_H, dtype=np.float32)
    for box, rgb in zip(boxes, colors):
        if _line_contrast(canvas, box, rgb, s) >= LINE_CONTRAST:
            continue
        g = _gray(canvas)
        x0, y0 = max(0, int(box[0]) - 4), max(0, int(box[1]) - 4)
        x1, y1 = min(COVER_W, int(math.ceil(box[2])) + 4), min(COVER_H, int(math.ceil(box[3])) + 4)
        p90 = float(np.percentile(g[y0:y1, x0:x1], 90))
        lo, hi = 0.0, 0.97
        for _ in range(24):          # smallest extra tint that reaches LINE_CONTRAST
            mid = (lo + hi) / 2.0
            bg = (1 - mid) * p90 + mid * t_luma
            if contrast_ratio(rgb, bg * (1.0 - s)) >= LINE_CONTRAST + 0.05:
                hi = mid
            else:
                lo = mid
        top = float(box[1])
        r = np.clip((y - (top - 180.0)) / 160.0, 0.0, 1.0)
        extra = (r * r * (3 - 2 * r) * hi)[:, None]
        canvas = _apply_tint(canvas, extra, tint)
    return canvas, [round(_line_contrast(canvas, b, c, s), 2) for b, c in zip(boxes, colors)]


def _draw_text(canvas: Image.Image, layout: dict, spec: str, cfg: dict, project=None) -> Image.Image:
    size = layout["size"]
    font = load_font(spec, size, project)
    color = parse_color(cfg.get("color", "#FFFFFF"))
    accent = parse_color(cfg.get("accent", "#FFD84D"))
    lk_font = None
    if layout["lockup_items"]:
        lk_font = load_font(cfg.get("lockup_font", DEFAULT_COVER["lockup_font"]),
                            int(cfg.get("lockup_size", 34)), project)
    text = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    mask = Image.new("L", canvas.size, 0)
    dt, dm = ImageDraw.Draw(text), ImageDraw.Draw(mask)
    for x, yb, w, acc in layout["items"]:
        fill = accent if acc else color
        dt.text((x, yb), w, font=font, fill=fill, anchor="ls")
        dm.text((x, yb), w, font=font, fill=255, anchor="ls")
    if layout["rule_box"]:
        x, y, w, h = layout["rule_box"]
        dt.rectangle([x, y, x + w - 1, y + h - 1], fill=accent)
        dm.rectangle([x, y, x + w - 1, y + h - 1], fill=255)
    lk_color = color[:3] + (int(color[3] * 0.9),)
    for x, yb, line in layout["lockup_items"]:
        dt.text((x, yb), line, font=lk_font, fill=lk_color, anchor="ls")
        dm.text((x, yb), line, font=lk_font, fill=255, anchor="ls")
    out = canvas.convert("RGBA")
    for sh in cfg.get("shadow") or DEFAULT_SHADOW:
        blur, alpha, dy = float(sh.get("blur", 12)), float(sh.get("alpha", 0.4)), int(sh.get("dy", 3))
        m = mask.filter(ImageFilter.GaussianBlur(blur)) if blur > 0 else mask
        m = m.point(lambda v, a=alpha: int(v * a))
        layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        layer.putalpha(m)
        shifted = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        shifted.paste(layer, (0, dy))
        out = Image.alpha_composite(out, shifted)
    out = Image.alpha_composite(out, text)
    return out.convert("RGB")


def compose(photo: Image.Image, faces: list[dict], headline: str, cfg: dict, *,
            base_shift: int = 0, lockup: list[str] | None = None, project=None,
            shift_manual: bool = False) -> tuple[Image.Image, dict]:
    """Compose a 1080x1920 cover from a clean frame and its faces (photo coordinates).

    Returns (image, info) with info {headline_box, text_box, size, lines, shift, face_box,
    faces, crown_y, crown_method, font_fallback, line_contrast, warnings}. `shift_manual` keeps
    base_shift exactly (an explicit cover.shift). The shift only slides the photo up: a negative
    base_shift counts as 0, so the recorded face position is always the one in the picture. The
    tint is anchored to the headline and
    darkened behind any line under LINE_CONTRAST. `crown_y` is the top of the main head in the
    finished cover and `crown_method` how it was found ("image": measured on the photo;
    "estimate": guessed from the face box, see head_top); only a measured crown, or a face box
    that itself starts above the grid band, is reported as a head the profile tile trims.
    Pure apart from font loading, so tests can drive it directly.
    """
    try:
        base_shift = max(0, int(base_shift or 0))
    except (TypeError, ValueError):
        base_shift = 0
    cfg = {**DEFAULT_COVER, **(cfg or {})}
    warnings: list[str] = []
    case = str(cfg.get("case", "upper"))
    _, gone = strip_invisible(headline or "")
    if gone:
        warnings.append("removed invisible %s from the headline"
                        % ", ".join("U+%04X" % ord(ch) for ch in gone))
    words = parse_runs(headline or "")
    if case == "upper":
        words = [(w.upper(), a) for w, a in words]
    spec = cfg["headline_font"]
    lockup = [strip_invisible(x)[0] for x in (lockup if lockup is not None else cfg.get("lockup") or [])]
    if words:
        miss = missing_glyphs(spec, "".join(w for w, _ in words), project)
        if miss:
            raise env.HCError(f"cover font {spec} cannot draw: {' '.join(miss)} (fix the headline or the preset font)", code=1)
    if lockup:
        miss = missing_glyphs(cfg["lockup_font"], "".join(lockup), project)
        if miss:
            raise env.HCError(f"lockup font {cfg['lockup_font']} cannot draw: {' '.join(miss)}", code=1)
    band_top, band_bot = (int(v) for v in cfg["band"])
    margin = int(cfg["margin"])
    layout, fit = None, {}
    main_face = max(_important_faces(faces), key=lambda f: f["box"][3]) if faces else None
    crown0, crown_method = None, None
    if main_face is not None:
        ph = photo if photo.size == (COVER_W, COVER_H) else photo.convert("RGB").resize(
            (COVER_W, COVER_H), Image.BILINEAR)
        crown0, crown_method = head_top(ph, main_face["box"])
    if words:
        lk_h = (34 if cfg.get("rule") else 0) + (
            ((22 if cfg.get("rule") else 30) + int(round(len(lockup) * int(cfg["lockup_size"]) * 1.3))) if lockup else 0)
        max_h = (band_bot - margin - lk_h) - (band_top + margin)
        lo_size, hi_size = (int(v) for v in cfg["size_range"])

        def fit_at(top_size):
            f_ = fit_headline(words, spec, (lo_size, top_size), int(cfg["max_lines"]),
                              COVER_W - 2 * margin, max_h, float(cfg.get("line_gap", 1.0)), project)
            return f_, layout_text(words, f_, spec, cfg, lockup, project)

        fit, layout = fit_at(hi_size)
        # A huge headline that pushes the photo up until the profile tile cuts the head is worse
        # than a slightly smaller one: step the size down until the face clears the text without
        # the shift pushing the top of the head (the estimated crown, not the face-box top) out
        # of the grid band (keep the largest such size; none found -> the original). A close-up
        # whose crown is outside the band even unshifted protects the forehead (face-box top).
        # A close-up (crown above the band even unshifted) cannot keep the whole head in the
        # tile: protect the forehead (face-box top) and shift at most CLOSEUP_SHIFT_MAX, so the
        # tile trims no more of the head than the framing already does plus that allowance. When
        # no size meets the rule, keep the size that trims the head least (then the largest).
        if main_face is not None:
            closeup = crown0 < band_top
            forehead = float(main_face["box"][1])

            def trim(lay):
                sh, ov = choose_shift(lay["text_box"], faces, base_shift, shift_manual)
                head = max(0.0, band_top - (crown0 - sh)) - (max(0.0, band_top - crown0) if closeup else 0.0)
                cut = ov or forehead - sh < band_top - 1e-6 or (
                    sh > CLOSEUP_SHIFT_MAX if closeup else crown0 - sh < band_top - 1e-6)
                ok = not ov and forehead - sh >= band_top - 1e-6     # face clear, forehead in
                return cut, (round(head, 1) if ok else None)
            cut0, head0 = trim(layout)
            if cut0:
                best = (head0, -int(layout["size"]), fit, layout) if head0 is not None else None
                size = hi_size - HEADLINE_STEP_DOWN
                while size >= lo_size:
                    f2, l2 = fit_at(size)
                    cut2, head2 = trim(l2)
                    if not cut2:
                        fit, layout = f2, l2
                        break
                    if head2 is not None and (best is None or (head2, -int(l2["size"])) < best[:2]):
                        best = (head2, -int(l2["size"]), f2, l2)
                    size = min(size, int(l2["size"])) - HEADLINE_STEP_DOWN
                else:
                    if best is not None:        # else keep the original (face overlap reported)
                        fit, layout = best[2], best[3]
        if fit.get("fallback"):
            warnings.append(fit["fallback"])
        hb = layout["headline_box"]
        if hb[1] < band_top or hb[1] + hb[3] > band_bot or hb[0] < 0 or hb[0] + hb[2] > COVER_W:
            warnings.append("headline leaves the 240-1680 grid band")
    else:
        warnings.append("no headline: cover is the photo only")
    text_box = layout["text_box"] if layout else None
    shift, overlapping = choose_shift(text_box, faces, base_shift, shift_manual)
    if overlapping:
        warnings.append(f"headline still covers a face at shift {shift} px (try another frame with cover.at_out, or a shorter headline)")
    img = _photo_layer(photo, shift, cfg, layout["headline_box"][1] if layout else None)
    contrast = []
    if layout:
        img, contrast = _raise_contrast(img, layout, spec, cfg, project)
        for n, c in enumerate(contrast, 1):
            if c < LINE_CONTRAST_WARN:
                warnings.append(f"cover headline contrast {c:.1f}:1 on line {n}")
        img = _draw_text(img, layout, spec, cfg, project)
    shifted = [{**f, "box": [f["box"][0], round(f["box"][1] - shift, 1), f["box"][2], f["box"][3]]} for f in faces]
    face_box = shifted[0]["box"] if shifted else None
    crown = None if crown0 is None else round(crown0 - shift, 1)
    if crown is not None and crown < band_top:
        forehead = float(main_face["box"][1]) - shift
        if forehead < band_top:
            warnings.append(f"face top at y {forehead:.0f} is above the grid crop ({band_top}); the profile "
                            "tile cuts the head")
        elif crown_method != "image":
            pass        # a guess from the face box: a bald head ends far below it. No finding.
        elif crown0 < band_top and shift <= CLOSEUP_SHIFT_MAX:
            warnings.append(f"close-up: the top of the head (y {crown:.0f}) is above the grid crop ({band_top}) even "
                            "unshifted, so the profile tile trims it; the forehead stays inside")
        else:
            warnings.append(f"top of the head at y {crown:.0f} is above the grid crop ({band_top}); the profile tile cuts the head")
    info = {"headline_box": layout["headline_box"] if layout else None, "text_box": text_box,
            "size": layout["size"] if layout else None, "lines": layout["lines"] if layout else [],
            "shift": int(shift), "shift_manual": bool(shift_manual), "face_box": face_box,
            "faces": [f["box"] for f in shifted], "crown_y": crown, "crown_method": crown_method,
            "font_fallback": bool(fit.get("fallback")), "line_contrast": contrast,
            "warnings": warnings}
    return img, info


def grid_crop(img: Image.Image, band=(240, 1680)) -> Image.Image:
    """The 3:4 profile-grid crop of a 9:16 cover."""
    return img.crop((0, int(band[0]), img.width, int(band[1])))


def mobile_preview(grid: Image.Image, width: int = MOBILE_W) -> Image.Image:
    """Grid crop at phone-tile scale."""
    h = int(round(grid.height * width / grid.width))
    return grid.resize((width, h), Image.LANCZOS)


# ---------------------------------------------------------------- candidate scoring

def score_frame(img: np.ndarray, faces: list[dict]) -> tuple[float, dict]:
    """Cover score = face size x centring x frontalness x sharpness (0..1-ish).

    Faceless frames score a small sharpness-only value so they rank below any usable face.
    """
    import cv2
    h, w = img.shape[:2]
    gray = cv2.cvtColor(np.ascontiguousarray(img), cv2.COLOR_RGB2GRAY)
    if not faces:
        g = cv2.resize(gray, (180, int(180 * h / w)))
        v = float(cv2.Laplacian(g, cv2.CV_64F).var())
        return round(0.05 * v / (v + 200.0), 4), {"size": 0, "centre": 0, "frontal": 0, "sharp": round(v / (v + 200.0), 3)}
    f = faces[0]
    x, y, fw, fh = f["box"]
    size = min(fh / h, 0.5) / 0.5
    cx, cy = (x + fw / 2) / w, (y + fh / 2) / h
    centre = max(0.15, 1.0 - 1.4 * abs(cx - 0.5))
    centre *= max(0.35, 1.0 - 2.0 * max(0.0, abs(cy - 0.38) - 0.12))
    (rex, rey), (lex, ley), (nx, ny), (rmx, rmy), (lmx, lmy) = f["landmarks"]
    a, b = abs(nx - rex), abs(lex - nx)
    sym_e = min(a, b) / max(a, b) if max(a, b) > 1e-6 else 0.0
    c, d = abs(nx - rmx), abs(lmx - nx)
    sym_m = min(c, d) / max(c, d) if max(c, d) > 1e-6 else 0.0
    eye_dist = math.hypot(lex - rex, ley - rey)
    roll = abs(math.atan2(ley - rey, (lex - rex) or 1e-6)) if eye_dist > 1e-6 else 0.0
    frontal = (0.25 + 0.75 * (0.6 * sym_e + 0.4 * sym_m)) * max(0.5, 1.0 - roll)
    x0, y0 = max(0, int(x)), max(0, int(y))
    x1, y1 = min(w, int(x + fw)), min(h, int(y + fh))
    crop = gray[y0:y1, x0:x1]
    if crop.size < 64:
        sharp = 0.0
    else:
        crop = cv2.resize(crop, (max(8, int(128 * crop.shape[1] / crop.shape[0])), 128))
        v = float(cv2.Laplacian(crop, cv2.CV_64F).var())
        sharp = v / (v + 60.0)
    score = size * centre * frontal * (0.3 + 0.7 * sharp)
    return round(score, 4), {"size": round(size, 3), "centre": round(centre, 3),
                             "frontal": round(frontal, 3), "sharp": round(sharp, 3)}


def candidate_times(duration: float, fps: tuple[int, int]) -> list[float]:
    """Candidate out times every 0.5 s (wider for long clips), on the frame grid."""
    if duration <= 0:
        return [0.0]
    step = max(CANDIDATE_STEP, duration / MAX_CANDIDATES)
    num, den = fps
    out, t = [], 0.0
    while t < duration - 0.1 or not out:
        n = cf.frame_index(t, (num, den))
        out.append(round(n * den / num, 6))
        t += step
    return sorted(set(out))


def _diverse_top(cands: list[dict], n: int = TOP_N) -> list[dict]:
    chosen: list[dict] = []
    for c in sorted(cands, key=lambda c: (-c["score"], c["at_out"])):
        if all(abs(c["at_out"] - o["at_out"]) >= MIN_SEPARATION for o in chosen):
            chosen.append(c)
        if len(chosen) >= n:
            break
    return chosen


def dissolve_spans(project, clip_id: str) -> list[tuple]:
    """(start_src, end_src) of every source cross dissolve the shot analysis found for the clip
    (shots json "transitions" of kind "dissolve"); [] when there is no shots data."""
    shots = cf._shots(project, clip_id)
    out = []
    for tr in (shots or {}).get("transitions") or []:
        if not isinstance(tr, dict) or str(tr.get("kind")) != "dissolve":
            continue
        try:
            a = float(tr.get("start_src", tr.get("at_src")))
            b = float(tr.get("end_src", tr.get("at_src")))
        except (TypeError, ValueError):
            continue
        out.append((min(a, b), max(a, b)))
    return sorted(out)


def in_dissolve(spans: list, t_src, margin: float = 0.0) -> bool:
    """A source time inside (or within `margin` seconds of) one of the dissolve spans."""
    if t_src is None:
        return False
    return any(a - margin <= float(t_src) <= b + margin for a, b in spans)


def _neighbours(plan: dict, t: float) -> tuple | None:
    """Out times of the frames BLEND_STEP either side of the frame at out time t, kept inside
    its plan segment (a join is a cut the editor made, not a dissolve in the picture). None
    when the segment is too short to tell."""
    hit = cf.seg_at(plan, t)
    if hit is None:
        return None
    _, sg = hit
    num, den = cf._fps(plan)
    o0 = float(sg["out"][0])
    frames = int(sg.get("frames") or round((float(sg["out"][1]) - o0) * num / den))
    n = cf.frame_index(t - o0, (num, den))
    a, b = max(0, n - BLEND_STEP), min(frames - 1, n + BLEND_STEP)
    if n - a < 1 or b - n < 1:
        return None
    return round(o0 + a * den / num, 6), round(o0 + b * den / num, 6)


def _small_gray(arr) -> np.ndarray:
    return np.asarray(Image.fromarray(np.asarray(arr)).convert("L").resize(BLEND_SIZE, Image.BOX),
                      np.float32)


def blend_measure(before, frame, after) -> dict:
    """How a frame sits between the frames either side of it: {"change": mean luma difference
    between the two neighbours (0..255), "linear": how far the frame is from their average, as
    a share of that change}. In a cross dissolve every pixel moves in a straight line from one
    picture to the other, so the middle frame IS the average (linear near 0) while the picture
    changes a lot; a moving head or hand changes the picture too, but the middle frame is not
    the average of a hand in two places (linear about 0.3 to 0.7)."""
    a, f, b = _small_gray(before), _small_gray(frame), _small_gray(after)
    change = float(np.abs(a - b).mean())
    resid = float(np.abs(f - (a + b) / 2.0).mean())
    return {"change": round(change, 2), "linear": round(resid / change, 3) if change > 1e-6 else 0.0}


def is_blend(m: dict | None) -> bool:
    """The measure of a frame inside a dissolve, or of a picture changing too fast to be a still."""
    if not m:
        return False
    return (m["change"] >= BLEND_HARD
            or (m["change"] >= BLEND_MIN and m["linear"] <= BLEND_LINEAR))


def faces_superimposed(boxes: list) -> bool:
    """Two face boxes that share more than FACE_OVERLAP of the smaller one: two pictures laid
    over each other (people in one picture do not overlap like that)."""
    for i, a in enumerate(boxes):
        for b in boxes[i + 1:]:
            small = min(float(a[2]) * float(a[3]), float(b[2]) * float(b[3]))
            if small > 0 and cf.box_overlap(a, b) > FACE_OVERLAP * small:
                return True
    return False


def frame_skip(plan: dict, t: float, faces: list, spans: list, frames: dict) -> tuple:
    """(reason to keep a frame off the cover or None, its blend measure or None): "dissolve"
    (inside a source dissolve the shot analysis found, or measured as one against its
    neighbours in `frames`, {out time: array}) or "superimposed" (two overlapping faces)."""
    num, den = cf._fps(plan)
    measure = None
    nb = _neighbours(plan, t)
    if nb and all(frames.get(x) is not None for x in (nb[0], t, nb[1])):
        measure = blend_measure(frames[nb[0]], frames[t], frames[nb[1]])
    if in_dissolve(spans, _src_exact(plan, t), margin=1.5 * den / num) or is_blend(measure):
        return "dissolve", measure
    if faces_superimposed(faces):
        return "superimposed", measure
    return None, measure


def usable(cands: list[dict]) -> list[dict]:
    """Candidates that may be picked or shown: not inside a dissolve, no superimposed faces.
    (All of them when every frame was set aside, so a cover can still be made.)"""
    good = [c for c in cands if not c.get("skip")]
    return good or list(cands)


def find_candidates(project, clip_id: str, clip: dict, plan: dict, preset: dict,
                    style: str | None = None) -> dict:
    """Scored candidates (cached by plan + segment files). {"origin", "candidates", "key"}.

    `style` is the style the cover is for (the preset's file name): the frames are read from
    that style's own stage A files of this plan. The cache key also carries the letterbox
    position and the shots data the source fallback frames with, so candidates scored before
    either changed are not reused.

    A candidate inside a source cross dissolve (the shots json "transitions", or measured
    against the frames either side of it: see blend_measure) or showing two superimposed faces
    carries "skip": the reason. It keeps its score for the record, but `usable` drops it from
    the automatic pick and the sheet: a dissolve ghost can score as a sharp frontal face."""
    fps = cf._fps(plan)
    files = cf.segment_files(project, clip_id, "vertical", plan, style=style)
    sig = [[p.name, p.stat().st_size, int(p.stat().st_mtime)] for p in files] if files else None
    lb_y = _letterbox_y(preset)
    video = (cf._source_info(project) or {}).get("video") or {}
    spans = dissolve_spans(project, clip_id)
    key = env.stable_hash({"v": CANDIDATE_VERSION, "spec": plan.get("spec_hash"),
                           "frames": plan.get("frames_out"),
                           "segs": [s.get("src") for s in plan.get("segments") or []],
                           "files": sig, "fp": plan.get("source_fingerprint"),
                           "framing": plan.get("framing"), "speakers": plan.get("speakers"),
                           "active": video.get("active_area"), "hdr": video.get("hdr"),
                           "dissolves": [list(x) for x in spans],
                           "lb_y": round(lb_y, 4), "shots": plan.get("shots_hash")})[:16]
    cache = Path(project.cache_path(f"cover-cands-{clip_id}-{key}.json"))
    if cache.is_file():
        try:
            data = env.read_json(cache)
            if data.get("candidates"):
                return data
        except Exception:
            pass
    times = candidate_times(float(plan.get("duration_out") or 0.0), fps)
    near = {t: _neighbours(plan, t) for t in times}
    want = sorted(set(times) | {x for nb in near.values() if nb for x in nb})
    frames, origin = cf.base_frames(project, clip_id, clip, plan, "vertical", want, ANALYSIS_SIZE, lb_y,
                                    **_for_style(style))
    cands = []
    for t in times:
        arr = frames.get(t)
        if arr is None:
            continue
        faces = cf.detect_faces(arr)
        score, parts = score_frame(arr, faces)
        cand = {"at_out": t, "score": score, "parts": parts, "faces": [f["box"] for f in faces]}
        skip, measure = frame_skip(plan, t, cand["faces"], spans, frames)
        if measure:
            cand["blend"] = measure
        if skip:
            cand["skip"] = skip
        cands.append(cand)
    if not cands:
        raise env.HCError(f"{clip_id}: no candidate frames could be read for the cover", code=2)
    data = {"origin": origin, "key": key, "analysis_size": list(ANALYSIS_SIZE), "candidates": cands}
    env.write_json(cache, data)
    return data


def _for_style(style) -> dict:
    """The keyword that points cover_frames at one style's own stage A files (none: any)."""
    return {} if style is None else {"style": style}


def _letterbox_y(preset: dict) -> float:
    try:
        return float(preset.get("letterbox", {}).get("y_center", 820)) / COVER_H
    except (TypeError, ValueError):
        return 820 / COVER_H


def _label_font(size: int = 22):
    try:
        return load_font("Montserrat[wght].ttf:SemiBold", size)
    except Exception:
        return ImageFont.load_default()


def candidates_sheet(project, clip_id: str, clip: dict, plan: dict, preset: dict, data: dict,
                     chosen_at: float, style: str | None = None) -> Path:
    """sheets/<clip>.cover-candidates.jpg: the top candidates with out time + score."""
    top = _diverse_top(data["candidates"])
    times = [c["at_out"] for c in top]
    if all(abs(chosen_at - t) > 1e-6 for t in times):
        times.append(chosen_at)
    frames, _ = cf.base_frames(project, clip_id, clip, plan, "vertical", times, ANALYSIS_SIZE,
                               _letterbox_y(preset), **_for_style(style))
    by_t = {c["at_out"]: c for c in data["candidates"]}
    cw, ch = ANALYSIS_SIZE
    cols = min(len(times), 4) or 1
    rows = math.ceil(len(times) / cols)
    sheet = Image.new("RGB", (cols * (cw + 8) + 8, rows * (ch + 44) + 8), (24, 24, 27))
    font = _label_font(22)
    d = ImageDraw.Draw(sheet)
    for i, t in enumerate(times):
        arr = frames.get(t)
        x, y = 8 + (i % cols) * (cw + 8), 8 + (i // cols) * (ch + 44)
        if arr is not None:
            tile = Image.fromarray(arr)
            td = ImageDraw.Draw(tile)
            for fb in (by_t.get(t) or {}).get("faces", []):
                td.rectangle([fb[0], fb[1], fb[0] + fb[2], fb[1] + fb[3]], outline=(80, 220, 120), width=2)
            sheet.paste(tile, (x, y))
        c = by_t.get(t)
        label = f"{t:.2f}s  score {c['score']:.3f}" if c else f"{t:.2f}s  (manual)"
        if abs(t - chosen_at) < 1e-6:
            label = "CHOSEN  " + label
            d.rectangle([x - 4, y - 4, x + cw + 3, y + ch + 3], outline=(255, 216, 77), width=4)
        d.text((x + 2, y + ch + 10), label, font=font, fill=(235, 235, 235))
    out = Path(project.sheet_path(f"{clip_id}.cover-candidates.jpg"))
    save_jpeg(sheet, out, 88)
    return out


# ---------------------------------------------------------------- make_cover

def _clip_cover(clip: dict) -> dict:
    c = clip.get("cover") or {}
    return c if isinstance(c, dict) else {}


def headline_for(clip: dict, plan: dict | None = None) -> str:
    """Cover headline: clip cover.headline, else the hook card lines, else the title.

    The text comes from claims.effective_headline, the one definition the claim checks (plan
    lint L13, QA G11) read as well, so what is drawn on a cover is what was checked."""
    from . import claims
    fn = getattr(claims, "effective_headline", None)
    if callable(fn):
        return str(fn(clip, plan)[0] or "")
    c = _clip_cover(clip)
    if c.get("headline"):
        return str(c["headline"])
    hc = (plan or {}).get("hook_card") or clip.get("hook_card") or {}
    if isinstance(hc, dict) and hc.get("lines"):
        return " ".join(str(x) for x in hc["lines"])
    return str(clip.get("title") or "")


def _name_words(name) -> list[str]:
    import unicodedata
    t = unicodedata.normalize("NFKD", str(name or ""))
    t = "".join(ch for ch in t if not unicodedata.combining(ch)).lower()
    return re.findall(r"[a-z0-9]+", t)


def same_speaker(a, b) -> bool:
    """Speaker names match when one's words are a prefix of the other's, case and accents
    ignored: "Sam" matches "Sam Rivera" and "sam rivera"; "Samantha" does not match "Sam"."""
    x, y = _name_words(a), _name_words(b)
    if not x or not y:
        return False
    n = min(len(x), len(y))
    return x[:n] == y[:n]


def speaker_runs(plan: dict) -> list[tuple]:
    """(label, s_out, e_out) runs of consecutive visible words with the same speaker label."""
    ws = []
    for w in plan.get("words") or []:
        if (w.get("hidden") or w.get("omitted") or w.get("removed") or w.get("filler")
                or w.get("s_out") is None or w.get("e_out") is None):
            continue
        ws.append((float(w["s_out"]), float(w["e_out"]), w.get("sp")))
    ws.sort()
    runs: list[list] = []
    for s, e, sp in ws:
        if runs and runs[-1][0] == sp:
            runs[-1][2] = max(runs[-1][2], e)
        else:
            runs.append([sp, s, e])
    return [tuple(r) for r in runs]


def speaker_at(plan: dict, t: float):
    """Speaker label of the words around out time t (None between speakers or unlabelled)."""
    for sp, s, e in speaker_runs(plan):
        if s - 1e-6 <= t <= e + 1e-6:
            return sp
    return None


def _explicit_shift(project, clip_id: str) -> bool:
    """True when clips.json itself sets cover.shift for the clip (0 included). The normalized
    spec fills shift 0 by default, so this reads the file as written."""
    try:
        raw = env.read_json(project.clips_path())
        c = ((raw.get("clips") or {}).get(clip_id) or {}).get("cover")
        return isinstance(c, dict) and c.get("shift") is not None
    except Exception:
        return False


def candidate_pool(cands: list[dict], plan: dict, owner) -> tuple[list[dict], str | None]:
    """The candidates the automatic pick may use: with a clip `speaker` and labelled words,
    only frames while that speaker talks (see same_speaker). Returns (pool, warning or None);
    when no candidate falls in the speaker's words the whole clip is used, with a warning."""
    if not owner:
        return cands, None
    runs = speaker_runs(plan)
    if not any(sp for sp, _, _ in runs):
        return cands, None
    spans = [(s, e) for sp, s, e in runs if sp and same_speaker(sp, owner)]
    pool = [c for c in cands if any(s - 1e-6 <= float(c["at_out"]) <= e + 1e-6 for s, e in spans)]
    if not pool:
        return cands, f"no candidate frame while {owner} speaks; picked from the whole clip"
    return pool, None


def make_cover(project, clip_id: str, style: str, fmt: str = "vertical", at_out=None) -> dict:
    """Build covers/<clip>__<style>.jpg (+ .grid.jpg, .mobile.jpg, .json) and the candidates sheet.

    The automatic frame comes from the clip's own speaker when `clip.speaker` is set and the
    words carry speaker labels (the claim owner on the cover, not the interviewer)."""
    from . import plan as plan_mod
    from . import presets

    spec = project.spec()
    clips = spec.get("clips") or {}
    clip = clips.get(clip_id)
    if clip is None:
        raise env.unknown_clip(clip_id, clips)
    plan = plan_mod.load_plan(project, clip_id)
    preset = presets.load(style, project)
    cfg = {**DEFAULT_COVER, **(preset.get("cover") or {})}
    if "case" not in (preset.get("cover") or {}):  # follow the hook card's case (minimal is as_is)
        cfg["case"] = str((preset.get("hook_card") or {}).get("case") or "upper")
    cc = _clip_cover(clip)
    data = find_candidates(project, clip_id, clip, plan, preset, style)
    cands = data["candidates"]
    warnings: list[str] = []
    good = usable(cands)
    if cands and all(c.get("skip") for c in cands):
        warnings.append("every candidate frame is inside a dissolve or shows superimposed faces; "
                        "picked from them anyway (set cover.at_out)")
    pool, note = candidate_pool(good, plan, clip.get("speaker"))
    if note:
        warnings.append(note)
    best = max(pool, key=lambda c: (c["score"], -c["at_out"]))
    duration = float(plan.get("duration_out") or 0.0)
    pick = at_out if at_out is not None else cc.get("at_out", "auto")
    if isinstance(pick, (int, float)) and not isinstance(pick, bool):
        t = float(pick)
        if not 0 <= t < max(duration, 1e-6):
            raise env.HCError(f"{clip_id}: cover time {t:.2f}s is outside the clip (0 to {duration:.2f}s)", code=1)
        num, den = cf._fps(plan)
        # the frame the time names (a time copied from a cover json maps back to its own
        # frame); `hc status` floors the same way (cli.cover_stale_reasons)
        t = round(cf.frame_index(t, (num, den)) * den / num, 6)
        manual = True
    elif pick in (None, "auto"):
        t, manual = best["at_out"], False
    else:
        raise env.HCError(f"{clip_id}: cover.at_out must be 'auto' or seconds, got {pick!r}", code=1)
    frames, origin = cf.base_frames(project, clip_id, clip, plan, "vertical", [t], (COVER_W, COVER_H),
                                    _letterbox_y(preset), **_for_style(style))
    arr = frames.get(t)
    if arr is None:
        raise env.HCError(f"{clip_id}: could not read the cover frame at {t:.2f}s", code=2)
    if manual:
        # A frame the editor chose is kept, but a dissolve ghost on a cover is worth a warning.
        why = _manual_skip(project, clip_id, clip, plan, preset, t, cands, style)
        if why == "dissolve":
            warnings.append(f"the cover frame at {t:.2f}s is inside a dissolve (two pictures laid over each "
                            "other): pick a frame before or after it with cover.at_out")
        elif why == "superimposed":
            warnings.append(f"the cover frame at {t:.2f}s shows two faces laid over each other: pick another "
                            "frame with cover.at_out")
    # Rendered segments are tone-mapped by render; source frames by base_frames (it raises
    # without zscale), so an HDR cover is always tone-mapped.
    hdr = cf.source_hdr(project, plan) if clip.get("kind") != "caption_only" else None
    if origin == "source":
        warnings.append("frames approximated from the source with a framing estimate (render the clip for exact frames)")
    faces = cf.detect_faces(arr)
    # Report the same score the candidates sheet shows for this frame (candidates are scored
    # at analysis size; re-scoring the full-size frame gave a different number for one frame).
    same = [c for c in cands if abs(float(c["at_out"]) - t) < 1e-3]
    if same:
        score, parts = same[0]["score"], same[0].get("parts") or {}
    else:
        score, parts = score_frame(arr, faces)
    if not faces:
        warnings.append("no face found in the chosen frame")
    headline = headline_for(clip, plan)
    lockup = cc.get("lockup")
    if isinstance(lockup, str):
        lockup = [lockup]
    shift_manual = _explicit_shift(project, clip_id)    # an explicit shift, 0 included
    try:
        base_shift = int(cc.get("shift") or 0)
    except (TypeError, ValueError):
        base_shift, shift_manual = 0, False
    img, info = compose(Image.fromarray(arr), faces, headline, cfg, base_shift=base_shift,
                        lockup=lockup, project=project, shift_manual=shift_manual)
    warnings.extend(info.pop("warnings"))
    main = Path(project.cover_path(clip_id, style, "main"))
    grid_p = Path(project.cover_path(clip_id, style, "grid"))
    mob_p = Path(project.cover_path(clip_id, style, "mobile"))
    json_p = Path(project.cover_path(clip_id, style, "json"))
    save_jpeg(img, main, JPEG_Q)
    g = grid_crop(img, cfg["band"])
    save_jpeg(g, grid_p, 90)
    save_jpeg(mobile_preview(g), mob_p, 88)
    sheet = candidates_sheet(project, clip_id, clip, plan, preset, {**data, "candidates": pool}, t, style)
    top = [{"at_out": c["at_out"], "score": c["score"]} for c in _diverse_top(pool)]
    out = {
        "version": 1, "clip": clip_id, "style": style, "format": fmt,
        "at_out": t, "at_src": _src_of(plan, t), "manual": manual, "score": score, "score_parts": parts,
        "face_box": info["face_box"], "faces": info["faces"], "faces_found": len(faces),
        "crown_y": info["crown_y"], "crown_method": info["crown_method"],
        "frame_speaker": speaker_at(plan, t),
        "headline": headline, "headline_box": info["headline_box"], "text_box": info["text_box"],
        "ink_box": info["text_box"], "size": info["size"], "lines": info["lines"],
        "line_contrast": info["line_contrast"], "font_fallback": info["font_fallback"],
        "shift": info["shift"], "shift_manual": info["shift_manual"],
        "band": list(cfg["band"]), "frame_source": origin,
        "hdr_tonemapped": True if hdr else None, "candidates": top,
        "candidates_sheet": _rel(project, sheet),
        "files": {"main": _rel(project, main), "grid": _rel(project, grid_p), "mobile": _rel(project, mob_p)},
        "spec_hash": plan.get("spec_hash"), "created": _now(), "warnings": warnings,
    }
    env.write_json(json_p, out)
    return out


def _src_exact(plan: dict, t: float):
    hit = cf.seg_at(plan, t)
    if not hit:
        return None
    _, s = hit
    return float(s["src"][0]) + (t - float(s["out"][0]))


def _src_of(plan: dict, t: float):
    v = _src_exact(plan, t)
    return None if v is None else round(v, 3)


def _manual_skip(project, clip_id: str, clip: dict, plan: dict, preset: dict, t: float,
                 cands: list[dict], style: str | None = None) -> str | None:
    """Why a hand-picked cover frame would have been set aside as a candidate (or None)."""
    same = [c for c in cands if abs(float(c["at_out"]) - t) < 1e-3]
    if same:
        return same[0].get("skip")
    try:
        nb = _neighbours(plan, t)
        want = [t] + (list(nb) if nb else [])
        frames, _ = cf.base_frames(project, clip_id, clip, plan, "vertical", want, ANALYSIS_SIZE,
                                   _letterbox_y(preset), **_for_style(style))
        if frames.get(t) is None:
            return None
        faces = [f["box"] for f in cf.detect_faces(frames[t])]
        return frame_skip(plan, t, faces, dissolve_spans(project, clip_id), frames)[0]
    except env.HCError:
        return None


def _rel(project, p: Path) -> str:
    try:
        return str(Path(p).resolve().relative_to(Path(project.root).resolve()))
    except ValueError:
        return str(p)


# ---------------------------------------------------------------- covers-grid mosaic

def covers_grid(project, style: str | dict, clip_ids: list[str], out_path: Path | None = None) -> Path | None:
    """3-column mosaic of the clips' grid crops at phone scale (the profile grid view).
    `style` is one style for every clip, or a {clip_id: style} map (each clip's own style)."""
    tiles = []
    for cid in clip_ids:
        st = style.get(cid) if isinstance(style, dict) else style
        if not st:
            continue
        p = Path(project.cover_path(cid, st, "grid"))
        if p.is_file():
            try:
                with Image.open(p) as im:
                    im.load()
                    tiles.append(im.convert("RGB").resize((GRID_TILE_W, int(GRID_TILE_W * 4 / 3)), Image.LANCZOS))
            except OSError:
                continue
    if not tiles:
        return None
    cols, gap = 3, 3
    th = tiles[0].height
    rows = math.ceil(len(tiles) / cols)
    mosaic = Image.new("RGB", (cols * GRID_TILE_W + (cols - 1) * gap, rows * th + (rows - 1) * gap), (0, 0, 0))
    for i, tile in enumerate(tiles):
        mosaic.paste(tile, ((i % cols) * (GRID_TILE_W + gap), (i // cols) * (th + gap)))
    out = Path(out_path or project.sheet_path("covers-grid.jpg"))
    save_jpeg(mosaic, out, 88)
    return out


def refresh_grids(project, spec: dict | None = None) -> list[str]:
    """Rebuild the set-level grids from every cover on disk, whatever was just made:
    sheets/covers-grid.jpg shows each clip in its own first style (the profile as it will look),
    and sheets/covers-grid__<style>.jpg one per style when the set uses more than one.

    Two `hc cover` runs on different clips both rebuild these shared sheets, so the tiles are
    read and the mosaics written under one lock (cache/.covers-grid.lock): the last run to
    finish is the one that saw every cover."""
    from . import fsutil
    with fsutil.file_lock(Path(project.cache_dir()) / GRID_LOCK, what="the covers grid"):
        return _refresh_grids(project, spec)


def _refresh_grids(project, spec: dict | None = None) -> list[str]:
    spec = spec or project.spec()
    clips = spec.get("clips") or {}
    ids = list(clips)
    first = {}
    styles: list[str] = []
    for cid, c in clips.items():
        eff = list((c.get("_eff") or {}).get("styles") or ["clean"])
        have = [st for st in eff if Path(project.cover_path(cid, st, "grid")).is_file()]
        if have:
            first[cid] = have[0]
        for st in eff:
            if st not in styles:
                styles.append(st)
    out = []
    grid = covers_grid(project, first, ids)
    if grid:
        out.append(f"covers grid (each clip's first style): {_rel(project, grid)}")
    if len(styles) > 1:
        for st in styles:
            p = covers_grid(project, st, ids, project.sheet_path(f"covers-grid__{st}.jpg"))
            if p:
                out.append(f"covers grid ({st}): {_rel(project, p)}")
    return out


# ---------------------------------------------------------------- command

def cmd_cover(args) -> int:
    """hc cover P [CLIP...] [--styles a,b] [--at SECONDS] | hc cover P --grid"""
    from .project import Project

    project = Project(args.project).require()
    spec = project.spec()
    if getattr(args, "grid", False):
        # only the set-level sheets, from the covers already on disk (seconds, not minutes)
        if getattr(args, "clips", None) or getattr(args, "at", None) is not None:
            raise env.HCError("--grid rebuilds the covers grid from every cover on disk: "
                              "name no clips and no --at", code=1)
        lines = refresh_grids(project, spec)
        if not lines:
            raise env.HCError("no covers on disk yet: run hc cover P first", code=1)
        for line in lines:
            print(line)
        return 0
    all_ids = list((spec.get("clips") or {}).keys())
    named = list(getattr(args, "clips", None) or [])
    unknown = [c for c in named if c not in all_ids]
    if unknown:
        raise env.unknown_clip(unknown[0], all_ids)
    ids = named or [c for c in all_ids
                    if "vertical" in ((spec["clips"][c].get("_eff") or {}).get("formats") or ["vertical"])]
    skipped = [c for c in all_ids if not named and c not in ids]
    if skipped:
        env.log("cover", f"skipping clips with no vertical format: {', '.join(skipped)}")
    at = getattr(args, "at", None)
    if at is not None:
        if len(ids) != 1:
            raise env.HCError("--at sets one clip's cover frame: name exactly one clip", code=1)
        try:
            at = float(at)
        except (TypeError, ValueError):
            from . import spec as spec_mod
            at = float(spec_mod.parse_time(at))
    req_styles = getattr(args, "styles", None)
    made = 0
    styles_seen: list[str] = []
    for cid in ids:
        clip = spec["clips"][cid]
        styles = list(req_styles or (clip.get("_eff") or {}).get("styles") or ["clean"])
        for st in styles:
            res = make_cover(project, cid, st, "vertical", at)
            made += 1
            if st not in styles_seen:
                styles_seen.append(st)
            note = f"  WARN: {'; '.join(res['warnings'])}" if res["warnings"] else ""
            print(f"{cid} [{st}] at {res['at_out']:.2f}s score {res['score']:.3f} shift {res['shift']} "
                  f"size {res['size']} -> {res['files']['main']}{note}")
    if styles_seen:
        for line in refresh_grids(project, spec):
            print(line)
    env.log("cover", f"{made} cover(s) written; open the candidates sheets and the covers grid")
    return 0
