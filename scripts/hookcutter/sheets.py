"""Exact-frame extraction, contact sheets, seam pairs and the ``peek`` command.

All text is drawn with Pillow from a bundled font (no drawtext/libass needed).
"""

from __future__ import annotations

import math
import os
import tempfile
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont

from . import env, probe

LABEL_FONT = "Montserrat[wght].ttf"
LABEL_INSTANCE = "SemiBold"
FALLBACK_FONT = "ArchivoBlack-Regular.ttf"
BG = (16, 16, 18)
FG = (238, 238, 240)
MUTED = (150, 150, 158)
SEAM = (255, 196, 0)
TIME_EPS = 1e-4
TONEMAP_CHAIN = ("zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,"
                 "tonemap=tonemap=hable:desat=0,zscale=t=bt709:m=bt709:r=tv,format=yuv420p")
PEEK_COLS = 4


# ---------------------------------------------------------------- fonts

@lru_cache(maxsize=32)
def label_font(size: int) -> "ImageFont.FreeTypeFont":
    """Bundled UI font for labels (Montserrat SemiBold), Pillow BASIC layout."""
    size = max(8, int(size))
    for name, inst in ((LABEL_FONT, LABEL_INSTANCE), (FALLBACK_FONT, None)):
        path = env.bundled_fonts_dir() / name
        if not path.exists():
            continue
        try:
            font = ImageFont.truetype(str(path), size, layout_engine=ImageFont.Layout.BASIC)
            if inst:
                try:
                    font.set_variation_by_name(inst)
                except Exception:  # variation support missing in this FreeType build
                    pass
            return font
        except OSError:
            continue
    return ImageFont.load_default(size)


def _text_w(font: Any, text: str) -> int:
    box = font.getbbox(text)
    return int(box[2] - box[0])


def _line_h(font: Any) -> int:
    asc, desc = font.getmetrics()
    return int(asc + desc)


def _fit(text: str, font: Any, width: int) -> str:
    if _text_w(font, text) <= width:
        return text
    while text and _text_w(font, text + "...") > width:
        text = text[:-1]
    return text + "..."


# ---------------------------------------------------------------- frames

def _tonemap_ok() -> bool:
    return env.has_filter("zscale")


def _filters(v: dict, width: Optional[int], tonemap: Any,
             flags: str = "lanczos") -> Tuple[List[str], bool]:
    chain = []
    toned = False
    hdr = v.get("hdr", "none") != "none"
    if hdr and tonemap and _tonemap_ok():
        chain.append(TONEMAP_CHAIN)
        toned = True
    sar = v.get("sar") or "1:1"
    if sar not in ("1:1", "0:1"):
        chain.append("scale='trunc(iw*sar/2)*2':ih,setsar=1")
    if width:
        chain.append("scale=%d:-2:flags=%s" % (int(width), flags))
    chain.append("format=rgb24")
    return chain, toned


def _grab(path: Path, ss: float, dur: Optional[float], chain: List[str], last: bool) -> Optional[Image.Image]:
    with tempfile.TemporaryDirectory(prefix="hc-frame-") as td:
        out = Path(td) / "f.png"
        args = ["-ss", "%.6f" % max(0.0, ss), "-i", str(path)]
        if dur is not None:
            # trim (not -t): output -t lets one frame past the limit through.
            chain = ["trim=end=%.6f" % max(dur, TIME_EPS)] + list(chain)
        args += ["-map", "0:v:0", "-an", "-sn", "-vf", ",".join(chain)]
        if last:
            args += ["-update", "1"]
        else:
            args += ["-frames:v", "1"]
        args += ["-f", "image2", "-c:v", "png", str(out)]
        env.ff(args, capture=False)
        if not out.exists():
            return None
        with Image.open(out) as im:
            im.load()
            return im.convert("RGB")


def frame_at(path: Any, t: float, width: Optional[int] = None, *, tonemap: Any = True,
             flags: str = "lanczos") -> Image.Image:
    """The frame displayed at time ``t`` (seconds on the file's own clock, from its format
    start: ``src`` for a source, ``out`` for a render), as an RGB PIL image.

    Exact: decodes a short window ending at ``t`` and keeps the last frame whose pts <= t,
    so it works for CFR and VFR files alike. ``width`` scales (aspect kept, even height) with
    the swscale ``flags`` ("area" for measurements: no ringing next to hard edges). HDR sources
    are tone-mapped to SDR when ffmpeg has zscale.
    """
    p = Path(path)
    info = probe.probe_media(p, sample_packets=False)
    v = info.get("video")
    if not v:
        raise env.HCError("%s has no video stream" % p.name, code=1)
    t = float(t)
    dur = info.get("duration") or 0.0
    if t < 0 or (dur and t > dur + 0.5):
        raise env.HCError("time %.3f s is outside %s (0 to %.3f s)" % (t, p.name, dur), code=1)
    chain, _ = _filters(v, width, tonemap, flags)
    frame_s = float(Fraction(v["fps"][1], v["fps"][0])) if v.get("fps") else 0.04
    for back in (max(0.5, 3 * frame_s), 5.0, 30.0):
        ss = max(0.0, t - back)
        img = _grab(p, ss, t - ss + TIME_EPS, chain, last=True)
        if img is not None:
            return img
        if ss <= 0.0:
            break
    # Nothing displayed yet at t (t before the first frame): use the first frame.
    img = _grab(p, t, None, chain, last=False) or _grab(p, 0.0, None, chain, last=False)
    if img is None:
        raise env.HCError("could not decode a frame at %.3f s from %s" % (t, p.name), code=3)
    return img


def frames_at(path: Any, times: Sequence[float], width: Optional[int] = None, *,
              tonemap: Any = True) -> List[Image.Image]:
    return [frame_at(path, t, width, tonemap=tonemap) for t in times]


def even_times(start: float, end: float, n: int) -> List[float]:
    """Centres of n equal bins across [start, end]."""
    n = max(1, int(n))
    span = float(end) - float(start)
    return [float(start) + (i + 0.5) * span / n for i in range(n)]


# ---------------------------------------------------------------- sheets

def _label_block(img_w: int, labels: str, font: Any, pad: int) -> int:
    lines = [ln for ln in (labels or "").split("\n")]
    if not labels:
        return 0
    return pad * 2 + len(lines) * _line_h(font)


def contact_sheet(images_with_labels: Sequence[Tuple[Image.Image, Optional[str]]], cols: int,
                  out_path: Any, cell_w: int, *, title: Optional[str] = None, gap: int = 6,
                  quality: int = 88) -> Path:
    """Grid of images (scaled to ``cell_w`` wide) with a label strip under each.

    ``images_with_labels``: list of (PIL image, label or None); labels may hold newlines.
    Writes JPEG (or PNG by suffix) atomically and returns the path.
    """
    items = list(images_with_labels)
    if not items:
        raise env.HCError("contact sheet needs at least one image", code=1)
    cols = max(1, min(int(cols), len(items)))
    cell_w = int(cell_w)
    font = label_font(max(12, int(round(cell_w * 0.045))))
    pad = max(3, cell_w // 80)
    scaled = []
    for img, label in items:
        im = img.convert("RGB")
        if im.width != cell_w:
            h = max(1, int(round(im.height * cell_w / float(im.width))))
            im = im.resize((cell_w, h), Image.LANCZOS)
        scaled.append((im, label or ""))
    img_h = max(im.height for im, _ in scaled)
    lab_h = max(_label_block(cell_w, lb, font, pad) for _, lb in scaled)
    cell_h = img_h + lab_h
    rows = int(math.ceil(len(scaled) / float(cols)))
    title_font = label_font(max(14, int(round(cell_w * 0.055))))
    title_h = (_line_h(title_font) + 2 * gap + pad) if title else 0
    W = cols * cell_w + (cols + 1) * gap
    H = title_h + rows * cell_h + (rows + 1) * gap
    sheet = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(sheet)
    if title:
        draw.text((gap + pad, gap + pad // 2), _fit(title, title_font, W - 2 * (gap + pad)),
                  font=title_font, fill=FG)
    for i, (im, label) in enumerate(scaled):
        r, c = divmod(i, cols)
        x = gap + c * (cell_w + gap)
        y = title_h + gap + r * (cell_h + gap)
        sheet.paste(im, (x, y + (img_h - im.height) // 2))
        if label:
            ty = y + img_h + pad
            for ln in label.split("\n"):
                draw.text((x + pad, ty), _fit(ln, font, cell_w - 2 * pad), font=font, fill=FG)
                ty += _line_h(font)
    return save_image(sheet, out_path, quality=quality)


def save_image(img: Image.Image, out_path: Any, *, quality: int = 88) -> Path:
    """Atomic save; JPEG unless the suffix says otherwise."""
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    suffix = out.suffix.lower()
    fd, tmp = tempfile.mkstemp(prefix="." + out.stem + ".", suffix=suffix or ".jpg", dir=str(out.parent))
    os.close(fd)
    try:
        if suffix in ("", ".jpg", ".jpeg"):
            img.convert("RGB").save(tmp, "JPEG", quality=quality, optimize=True)
        else:
            img.save(tmp)
        os.chmod(tmp, env.file_mode())  # mkstemp creates 0600
        os.replace(tmp, str(out))
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    return out


def seam_pair(before: Image.Image, after: Image.Image, label_before: str = "", label_after: str = "",
              *, title: Optional[str] = None, height: Optional[int] = None) -> Image.Image:
    """Two frames side by side (last before a join | first after it) with a marker bar
    between them, a label under each and an optional title above. Returns the image."""
    h = int(height or min(before.height, after.height))
    ims = []
    for im in (before, after):
        im = im.convert("RGB")
        if im.height != h:
            im = im.resize((max(1, int(round(im.width * h / float(im.height)))), h), Image.LANCZOS)
        ims.append(im)
    font = label_font(max(12, int(round(h * 0.035))))
    pad = max(3, h // 120)
    bar = max(4, h // 160)
    lab_h = max(_label_block(0, label_before, font, pad), _label_block(0, label_after, font, pad))
    title_h = (_line_h(font) + 2 * pad) if title else 0
    W = ims[0].width + bar + ims[1].width
    H = title_h + h + lab_h
    out = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(out)
    if title:
        draw.text((pad, pad), _fit(title, font, W - 2 * pad), font=font, fill=FG)
    out.paste(ims[0], (0, title_h))
    draw.rectangle([ims[0].width, title_h, ims[0].width + bar - 1, title_h + h - 1], fill=SEAM)
    out.paste(ims[1], (ims[0].width + bar, title_h))
    for x0, im, label in ((0, ims[0], label_before), (ims[0].width + bar, ims[1], label_after)):
        ty = title_h + h + pad
        for ln in (label or "").split("\n"):
            if ln:
                draw.text((x0 + pad, ty), _fit(ln, font, im.width - 2 * pad), font=font, fill=FG)
            ty += _line_h(font)
    return out


def default_cell_w(width: int, height: int) -> int:
    return 300 if height > width else 480


# ---------------------------------------------------------------- peek

def _fmt_name(t: float) -> str:
    """A time for a file name: as many decimals as it has (up to 3, at least 1), so two
    different sub-second requests never share a sheet ("748.95", not "749.0")."""
    s = ("%.3f" % t).rstrip("0")
    s = s + "0" if s.endswith(".") else s
    return s.replace("-", "m")


def peek(project: Any, start: Any, end: Any, n: int = 12, file: Optional[Any] = None) -> Path:
    """Contact sheet of n evenly spaced frames in [start, end] of the source (src clock) or of
    ``file`` (its own clock, e.g. a render) → sheets/peek_*.jpg. Returns the path."""
    from .project import Project
    proj = project if isinstance(project, Project) else Project(project)
    proj.require()
    a, b = env.parse_time(start), env.parse_time(end)
    if b <= a:
        raise env.HCError("peek: END must be after START (%s to %s)" % (env.fmt_time(a), env.fmt_time(b)),
                          code=1)
    if file:
        fp = Path(file).expanduser()
        if not fp.is_absolute() and not fp.exists():
            fp = proj.root / fp
        if not fp.exists():
            raise env.HCError("peek: file not found: %s" % file, code=1)
        clock, stem = "out", fp.stem
    else:
        fp = proj.source_path()
        clock, stem = "src", None
    info = probe.probe_media(fp, sample_packets=False)
    v = info.get("video")
    if not v:
        raise env.HCError("%s has no video stream" % fp.name, code=1)
    dur = info.get("duration") or b
    b = min(b, dur)
    if b <= a:
        raise env.HCError("peek: %s is only %s long" % (fp.name, env.fmt_time(dur)), code=1)
    n = max(1, min(int(n), 64))
    fps = Fraction(v["fps"][0], v["fps"][1])
    cell = default_cell_w(v["width"], v["height"])
    items = []
    for t in even_times(a, b, n):
        img = frame_at(fp, t, width=cell)
        k = int(math.floor((t - v.get("start", 0.0)) * fps + 1e-6))
        items.append((img, "%s %s  f%d" % (clock, env.fmt_time(t, 2), k)))
    hdr = v.get("hdr", "none") != "none"
    note = ""
    if hdr:
        note = "  (HDR tone-mapped)" if _tonemap_ok() else "  (HDR, NOT tone-mapped: colours look flat)"
    title = "%s  %s to %s  %d frames%s" % (fp.name, env.fmt_time(a, 1), env.fmt_time(b, 1), n, note)
    name = "peek_%s%s-%s%s.jpg" % ((stem + "_") if stem else "", _fmt_name(a), _fmt_name(b),
                                   "" if n == 12 else "_n%d" % n)
    return contact_sheet(items, PEEK_COLS, proj.sheet_path(name), cell, title=title)


def cmd_peek(args: Any) -> int:
    out = peek(args.project, args.start, args.end, n=getattr(args, "n", 12) or 12,
               file=getattr(args, "file", None))
    print(str(out))
    return 0
