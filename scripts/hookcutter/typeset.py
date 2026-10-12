"""Fonts, glyph coverage, measuring, size fitting and balanced line wrapping.

Every piece of burned-in text (captions, hook card, cover) is drawn with Pillow from a font
FILE, never through fontconfig, so a missing face fails loudly instead of silently becoming
some fallback. Layout is pinned to Pillow's BASIC engine so measurements (and therefore line
breaks and auto-fit sizes) are identical on machines with and without libraqm.

Tracking is expressed in 1/1000 em (the design-tool convention), so one preset value works at
any font size.
"""
from __future__ import annotations

import itertools
import math
import os
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Callable, Iterable, Sequence

from PIL import Image, ImageDraw, ImageFont

from . import env

LAYOUT = ImageFont.Layout.BASIC
FONT_EXTS = (".ttf", ".otf", ".ttc")
SKILL_ROOT = Path(__file__).resolve().parents[2]
MAX_EXHAUSTIVE_TOKENS = 24      # balanced_wrap enumerates partitions up to this many tokens
WIDOW_FRACTION = 0.5            # a lone last-line token narrower than this x widest line is a widow
# Zero-width space/joiners, word joiner, BOM and soft hyphen: stripped from burned-in text.
ZERO_WIDTH = frozenset("\u200b\u200c\u200d\u2060\ufeff\u00ad")
SHADOW_SAMPLE = "HAMBURGEFONTS" # type-test word whose shadow halo is measured (shadow_strength)


def font_dirs(project_root: str | os.PathLike | None = None) -> list[Path]:
    """Font search path in priority order: P/fonts, <data>/fonts, SK/fonts."""
    dirs = []
    if project_root:
        dirs.append(Path(project_root) / "fonts")
    dirs.append(Path(env.data_dir()) / "fonts")
    dirs.append(SKILL_ROOT / "fonts")
    return dirs


def parse_font_spec(spec: str) -> tuple[str, str | None]:
    """Split "File.ttf:Instance" into (file, instance). Windows drive colons are left alone."""
    spec = str(spec).strip()
    if ":" in spec:
        head, tail = spec.rsplit(":", 1)
        if tail and not any(sep in tail for sep in ("/", "\\")) and head:
            return head, tail.strip()
    return spec, None


def resolve_font(spec: str, project_root: str | os.PathLike | None = None) -> tuple[Path, str | None]:
    """Find the font file for `spec`. Raises HCError (code 1) listing where it looked."""
    name, instance = parse_font_spec(spec)
    candidates = [name]
    if not name.lower().endswith(FONT_EXTS):
        candidates += [name + ext for ext in FONT_EXTS]
    searched = []
    for cand in candidates:
        p = Path(cand).expanduser()
        if p.is_absolute():
            if p.is_file():
                return p, instance
            searched.append(str(p))
            continue
        for d in font_dirs(project_root):
            q = d / cand
            if q.is_file():
                return q, instance
            searched.append(str(q))
    raise env.HCError(f"font not found: {spec!r}. Looked in: " + ", ".join(dict.fromkeys(searched)))


@lru_cache(maxsize=128)
def _load(path: str, instance: str | None, size: int) -> ImageFont.FreeTypeFont:
    font = ImageFont.truetype(path, size, layout_engine=LAYOUT)
    if instance:
        _apply_instance(font, path, instance)
    return font


def _apply_instance(font: ImageFont.FreeTypeFont, path: str, instance: str) -> None:
    if "=" in instance:                        # explicit axes, e.g. "wght=650"
        try:
            axes = {k.strip().lower(): float(v) for k, v in
                    (part.split("=", 1) for part in instance.split(","))}
            order = [a["name"].decode("latin-1").lower() if isinstance(a["name"], bytes) else
                     str(a["name"]).lower() for a in font.get_variation_axes()]
            tags = {"weight": "wght", "width": "wdth", "slant": "slnt", "italic": "ital"}
            values = []
            for nm in order:
                key = tags.get(nm, nm)
                values.append(axes.get(key, axes.get(nm)))
            if any(v is None for v in values):
                raise ValueError("missing axis")
            font.set_variation_by_axes(values)
            return
        except (OSError, ValueError, KeyError) as exc:
            raise env.HCError(f"font {Path(path).name}: cannot apply axes {instance!r} ({exc})")
    try:
        names = [n.decode("latin-1") if isinstance(n, bytes) else str(n)
                 for n in font.get_variation_names()]
    except OSError:
        raise env.HCError(f"font {Path(path).name} is not a variable font, so instance "
                     f"{instance!r} cannot be selected")
    match = next((n for n in names if n.lower() == instance.lower()), None)
    if match is None:
        raise env.HCError(f"font {Path(path).name} has no instance {instance!r}. "
                     f"Available: {', '.join(names)}")
    font.set_variation_by_name(match)


def load_font(spec: str, size: float, project_root: str | os.PathLike | None = None
              ) -> ImageFont.FreeTypeFont:
    """Load `spec` ("File.ttf" or "File.ttf:Instance") at `size` px with BASIC layout (cached)."""
    path, instance = resolve_font(spec, project_root)
    return _load(str(path), instance, max(1, int(round(size))))


@lru_cache(maxsize=32)
def _cmap(path: str) -> frozenset:
    from fontTools.ttLib import TTFont
    kw = {"fontNumber": 0} if path.lower().endswith(".ttc") else {}
    with TTFont(path, lazy=True, **kw) as tt:
        cmap = tt.getBestCmap() or {}
        return frozenset(cmap.keys())


def _font_path(font) -> str:
    if isinstance(font, ImageFont.FreeTypeFont):
        return str(font.path)
    return str(font)


def strip_invisible(text: str) -> tuple[str, list[str]]:
    """`text` without zero-width and soft-hyphen characters, plus the characters removed.

    These are often pasted in from documents; fonts rarely carry them, and Pillow draws a
    .notdef box for any character missing from the font, so they are removed before measuring
    and drawing.
    """
    text = str(text)
    gone = [ch for ch in text if ch in ZERO_WIDTH]
    if not gone:
        return text, []
    return "".join(ch for ch in text if ch not in ZERO_WIDTH), list(dict.fromkeys(gone))


def missing_glyphs(font_path, text: str) -> list[str]:
    """Characters of `text` the font has no glyph for (unique, first-seen order).

    Whitespace and control characters are ignored. Format characters (category Cf) are
    reported like any other character: callers strip the zero-width ones first
    (strip_invisible), and anything else would be drawn as a box. `font_path` may be a path or
    a loaded FreeTypeFont.
    """
    cmap = _cmap(_font_path(font_path))
    out = []
    for ch in dict.fromkeys(text):
        if ch.isspace() or unicodedata.category(ch) == "Cc":
            continue
        if ord(ch) not in cmap:
            out.append(ch)
    return out


def describe_chars(chars: Iterable[str]) -> str:
    return ", ".join(f"{c!r} (U+{ord(c):04X})" for c in chars)


def tracking_px(font: ImageFont.FreeTypeFont, tracking: float) -> float:
    return float(tracking or 0) * font.size / 1000.0


def text_width(font: ImageFont.FreeTypeFont, text: str, tracking: float = 0) -> float:
    """Advance width in px of `text`, with tracking (1/1000 em) between characters."""
    if not text:
        return 0.0
    return font.getlength(text) + tracking_px(font, tracking) * (len(text) - 1)


def cap_height(font: ImageFont.FreeTypeFont) -> int:
    """Height of a capital H above the baseline, in px."""
    return -font.getbbox("H", anchor="ls")[1]


def char_offsets(font: ImageFont.FreeTypeFont, text: str, tracking: float = 0) -> list[float]:
    """x offset of each character's origin when drawn with tracking (kerning kept via prefixes)."""
    t = tracking_px(font, tracking)
    return [font.getlength(text[:i]) + t * i for i in range(len(text))]


def text_mask(font: ImageFont.FreeTypeFont, text: str, tracking: float = 0,
              stroke: int = 0) -> tuple[Image.Image, tuple[int, int]]:
    """Render `text` as an 8-bit coverage mask.

    Returns (mask, (ox, oy)) where (ox, oy) is the pixel position of the text origin (left
    edge of the first advance, on the baseline) inside the mask. With `stroke` > 0 the glyphs
    are dilated by that many px (used for outlines).
    """
    ascent, descent = font.getmetrics()
    pad = int(stroke) + max(4, font.size // 4)
    width = int(math.ceil(text_width(font, text, tracking))) + 2 * pad
    height = ascent + descent + 2 * pad
    img = Image.new("L", (max(1, width), max(1, height)), 0)
    draw = ImageDraw.Draw(img)
    ox, oy = pad, pad + ascent
    kw = {"stroke_width": int(stroke), "stroke_fill": 255} if stroke else {}
    if not tracking:
        draw.text((ox, oy), text, font=font, fill=255, anchor="ls", **kw)
    else:
        for ch, dx in zip(text, char_offsets(font, text, tracking)):
            draw.text((ox + dx, oy), ch, font=font, fill=255, anchor="ls", **kw)
    return img, (ox, oy)


def fit_size(lines: Sequence[str], spec: str, max_width: float, lo: int, hi: int,
             tracking: float = 0, project_root: str | os.PathLike | None = None) -> int:
    """Largest integer size in [lo, hi] at which every line is <= max_width px.

    Returns `lo` when even the smallest size overflows (callers decide what to do about it).
    """
    lo, hi = int(math.floor(lo)), int(math.ceil(hi))
    if hi < lo:
        lo, hi = hi, lo
    lines = [ln for ln in lines if ln]
    if not lines:
        return hi

    def fits(size: int) -> bool:
        font = load_font(spec, size, project_root)
        return all(text_width(font, ln, tracking) <= max_width for ln in lines)

    if fits(hi):
        return hi
    if not fits(lo):
        return lo
    a, b = lo, hi          # invariant: fits(a), not fits(b)
    while b - a > 1:
        mid = (a + b) // 2
        if fits(mid):
            a = mid
        else:
            b = mid
    # Hinting can make width non-monotonic by a pixel; walk down until it truly fits.
    while a > lo and not fits(a):
        a -= 1
    return a


def _partitions(n: int, k: int):
    for cuts in itertools.combinations(range(1, n), k - 1):
        bounds = (0,) + cuts + (n,)
        yield [(bounds[i], bounds[i + 1]) for i in range(k)]


def balanced_wrap(tokens: Sequence, measure: Callable[[Sequence], float], max_width: float,
                  max_lines: int, bad_break: Callable | None = None) -> list[list] | None:
    """Split `tokens` into the fewest lines (<= max_lines) that fit, then balance them.

    `measure(list_of_tokens)` returns the rendered width of those tokens set as one line
    (return math.inf to forbid a combination). Among partitions with the fewest lines it
    picks, in order: the fewest bad line breaks (`bad_break(last_token_of_line,
    first_token_of_next_line)` is True, for example a line ending on "the"), no widow (a lone
    short token on the last line under a fuller line above), the narrowest widest line, the
    least raggedness (sum of squared shortfalls), then the top-heavier shape. Returns None
    when no partition fits.
    """
    toks = list(tokens)
    n = len(toks)
    if n == 0:
        return []
    max_lines = max(1, int(max_lines))
    if n > MAX_EXHAUSTIVE_TOKENS:
        return _greedy_wrap(toks, measure, max_width, max_lines)
    cache: dict = {}

    def width(a: int, b: int) -> float:
        if (a, b) not in cache:
            cache[(a, b)] = float(measure(toks[a:b]))
        return cache[(a, b)]

    for k in range(1, min(max_lines, n) + 1):
        best, best_key = None, None
        for parts in _partitions(n, k):
            widths = [width(a, b) for a, b in parts]
            if any(w > max_width for w in widths):
                continue
            widest = max(widths)
            widow = 0
            if k > 1:
                a, b = parts[-1]
                pa, pb = parts[-2]
                if b - a == 1 and pb - pa > 1 and widths[-1] < WIDOW_FRACTION * widest:
                    widow = 1
            bad = 0
            if bad_break is not None:
                bad = sum(1 for _, b in parts[:-1] if bad_break(toks[b - 1], toks[b]))
            ragged = sum((widest - w) ** 2 for w in widths)
            top_heavy = -sum(w * (k - i) for i, w in enumerate(widths))
            key = (bad, widow, round(widest, 3), round(ragged, 3), round(top_heavy, 3))
            if best_key is None or key < best_key:
                best, best_key = parts, key
        if best is not None:
            return [toks[a:b] for a, b in best]
    return None


def _greedy_wrap(toks, measure, max_width, max_lines):
    lines, cur = [], []
    for t in toks:
        if cur and measure(cur + [t]) > max_width:
            lines.append(cur)
            cur = []
        if not cur and measure([t]) > max_width:
            return None
        cur.append(t)
    if cur:
        lines.append(cur)
    return lines if len(lines) <= max_lines else None


_SHADOW_CACHE: dict = {}


def _shadow_layers_key(layers) -> tuple:
    out = []
    for layer in layers or []:
        out.append((float(layer.get("blur", 0) or 0), float(layer.get("alpha", 0.5) or 0),
                    int(layer.get("dy", 0) or 0)))
    return tuple(out)


def shadow_strength(font: ImageFont.FreeTypeFont, layers, outline_px: int = 0,
                    outline_luma: float = 0.0) -> float:
    """How much the text's shadow (and outline) darkens the background right around the
    glyphs: 0 = not at all, 1 = fully.

    Measured on a sample word as the mean darkening of the ring 2-5 px outside the glyphs,
    drawn with the same blur/alpha/offset layers the captions and the cover use (an outline of
    luma `outline_luma`, 0..1, is laid over the shadow). A background of grey g behind the
    shadowed text reads as about g * (1 - s), which is what shadow-aware contrast uses.
    """
    import numpy as np                      # local: typeset itself does not need numpy
    from PIL import ImageFilter

    key = _shadow_layers_key(layers)
    outline_px = max(0, int(outline_px or 0))
    if not key and outline_px == 0:
        return 0.0
    try:
        face = font.getname()
    except (OSError, AttributeError):
        face = None
    ck = (_font_path(font), face, font.size, key, outline_px, round(float(outline_luma), 3))
    hit = _SHADOW_CACHE.get(ck)
    if hit is not None:
        return hit
    glyph, (ox, oy) = text_mask(font, SHADOW_SAMPLE, 0)
    reach = max([int(math.ceil(3 * b)) + abs(dy) for b, _, dy in key] or [0]) + outline_px + 8
    w, h = glyph.size
    canvas = Image.new("L", (w + 2 * reach, h + 2 * reach), 0)
    canvas.paste(glyph, (reach, reach))
    shape = canvas
    if outline_px > 0:
        stroke, (sx, sy) = text_mask(font, SHADOW_SAMPLE, 0, stroke=outline_px)
        shape = Image.new("L", canvas.size, 0)
        shape.paste(stroke, (reach + ox - sx, reach + oy - sy))
    clear = np.ones((canvas.size[1], canvas.size[0]), np.float32)   # 1 = background untouched
    for blur, alpha, dy in key:
        img = shape.filter(ImageFilter.GaussianBlur(blur)) if blur > 0 else shape
        m = np.asarray(img, np.float32) / 255.0 * alpha
        if dy:
            m = np.roll(m, dy, axis=0)
            if dy > 0:
                m[:dy] = 0
            else:
                m[dy:] = 0
        clear *= 1.0 - np.clip(m, 0.0, 1.0)
    if outline_px > 0:
        o = np.asarray(shape, np.float32) / 255.0
        clear = clear * (1.0 - o) + o * float(outline_luma)
    solid = canvas.point(lambda v: 255 if v >= 242 else 0)       # glyph interior
    near = np.asarray(solid.filter(ImageFilter.MaxFilter(3)), np.uint8) > 0
    ring = (np.asarray(solid.filter(ImageFilter.MaxFilter(11)), np.uint8) > 0) & ~near
    value = 0.0 if not ring.any() else float(min(1.0, max(0.0, 1.0 - clear[ring].mean())))
    _SHADOW_CACHE[ck] = value
    return value
