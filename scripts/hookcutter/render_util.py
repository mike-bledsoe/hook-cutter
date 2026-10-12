"""Private helpers for shots / framing / render: project coercion, presets, frame grabs,
labelled sheets. Each prefers the shared module (project, presets, sheets) and keeps a small
local fallback so the picture path works on its own."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import env

LABEL_BG = (12, 12, 14)
LABEL_FG = (236, 236, 236)
SHEET_BG = (24, 24, 27)


# ---------------------------------------------------------------- project

def as_project(p):
    """A project.Project for a path (or the object itself when it already is one)."""
    if hasattr(p, "root") and hasattr(p, "work_dir"):
        return p
    from .project import Project
    return Project(p).require()


def unknown_clip(clip_id, spec_or_known) -> "env.HCError":
    """The HCError for a clip id clips.json does not have (env.unknown_clip: code 1 with a
    "did you mean" suggestion). `spec_or_known` is a spec dict or a list of known ids."""
    if isinstance(spec_or_known, dict):
        known = list((spec_or_known.get("clips") or {}).keys())
    else:
        known = list(spec_or_known or [])
    fn = getattr(env, "unknown_clip", None)
    if fn is not None:
        return fn(clip_id, known)
    import difflib
    near = difflib.get_close_matches(str(clip_id), [str(k) for k in known], n=1)
    hint = (" (did you mean %r?)" % near[0]) if near else ""
    return env.HCError("unknown clip %r%s; clips.json has: %s"
                       % (clip_id, hint, ", ".join(known[:12]) or "none"), code=1)


def clip_entry(proj, clip_id: str) -> dict:
    spec = proj.spec()
    clips = spec.get("clips") or {}
    if clip_id not in clips:
        raise unknown_clip(clip_id, spec)
    return clips[clip_id]


# ---------------------------------------------------------------- presets

def deep_merge(base: dict, over: dict) -> dict:
    out = dict(base or {})
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _preset_dirs(project) -> list:
    dirs = []
    if project is not None:
        from .project import root_of
        dirs.append(root_of(project) / "presets")
    dirs += [env.user_presets_dir(), env.bundled_presets_dir()]
    return dirs


def _load_local(name: str, project, seen=()) -> dict:
    if name in seen:
        raise env.HCError("preset %r extends itself (cycle: %s)" % (name, " > ".join(seen)), code=1)
    for d in _preset_dirs(project):
        p = d / ("%s.json" % name)
        if p.is_file():
            data = env.read_json(p)
            parent = data.get("extends")
            if parent:
                data = deep_merge(_load_local(parent, project, tuple(seen) + (name,)), data)
            data.setdefault("name", name)
            return data
    raise env.HCError("unknown preset %r (looked in %s)"
                      % (name, ", ".join(str(d) for d in _preset_dirs(project))), code=1)


def load_preset(name: str, project=None, fmt: str = "vertical") -> dict:
    """Preset `name` resolved for `fmt` (presets.load + for_format when available)."""
    try:
        from . import presets as pm
    except ImportError:
        pm = None
    if pm is not None and hasattr(pm, "load"):
        preset = pm.load(name, project=project)
        if hasattr(pm, "for_format"):
            return pm.for_format(preset, fmt)
    else:
        preset = _load_local(name, project)
    if fmt == "landscape" and isinstance(preset.get("landscape"), dict):
        preset = deep_merge(preset, preset["landscape"])
    return preset


# ---------------------------------------------------------------- frames

def scaled_size(w: int, h: int, width: int) -> tuple:
    width = int(width)
    return width, max(2, int(round(h * width / float(w) / 2.0)) * 2)


def grab_frames(path, indices, width: int, size=None) -> dict:
    """Exact frames by decode index from a media file → {index: PIL RGB image}.

    `size` = (w, h) of the file's frames (probed when None). One decode pass."""
    from PIL import Image
    idx = sorted({int(i) for i in indices if int(i) >= 0})
    if not idx:
        return {}
    if size is None:
        st = env.ffprobe_json(path, "-select_streams", "v:0", "-show_entries", "stream=width,height")
        s = (st.get("streams") or [{}])[0]
        size = (int(s["width"]), int(s["height"]))
    ow, oh = scaled_size(size[0], size[1], width)
    out = {}
    chunk = 60
    for c in range(0, len(idx), chunk):
        part = idx[c:c + chunk]
        sel = "+".join("eq(n\\,%d)" % i for i in part)
        res = env.ff(["-i", str(path), "-map", "0:v:0",
                      "-vf", "select='%s',scale=%d:%d:flags=area" % (sel, ow, oh),
                      "-fps_mode", "passthrough", "-enc_time_base", "-1",
                      "-f", "rawvideo", "-pix_fmt", "rgb24", "-"])
        fb = ow * oh * 3
        data = res.stdout
        got = len(data) // fb
        for j in range(min(got, len(part))):
            out[part[j]] = Image.frombytes("RGB", (ow, oh), data[j * fb:(j + 1) * fb])
    return out


def grab_at(path, t: float, width: int, src_size: tuple) -> "np.ndarray":
    """One BGR frame at time t of a source (input seek), scaled to `width`."""
    ow, oh = scaled_size(src_size[0], src_size[1], width)
    res = env.ff(["-ss", "%.6f" % max(0.0, t), "-i", str(path), "-map", "0:v:0", "-frames:v", "1",
                  "-vf", "scale=%d:%d:flags=area" % (ow, oh), "-f", "rawvideo", "-pix_fmt", "bgr24", "-"])
    fb = ow * oh * 3
    if len(res.stdout) < fb:
        return None
    return np.frombuffer(res.stdout[:fb], dtype=np.uint8).reshape(oh, ow, 3).copy()


# ---------------------------------------------------------------- sheets

def label_font(size: int = 18):
    from PIL import ImageFont
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        f = env.bundled_fonts_dir() / "ArchivoBlack-Regular.ttf"
        try:
            return ImageFont.truetype(str(f), size)
        except OSError:
            return ImageFont.load_default()


def labelled(img, label: str, font=None):
    """Image with a label strip underneath."""
    from PIL import Image, ImageDraw
    font = font or label_font(18)
    lh = 28
    out = Image.new("RGB", (img.width, img.height + lh), LABEL_BG)
    out.paste(img.convert("RGB"), (0, 0))
    ImageDraw.Draw(out).text((8, img.height + 5), label, fill=LABEL_FG, font=font)
    return out


def _local_sheet(items, cols: int, out_path, cell_w: int, title=None) -> Path:
    from PIL import Image, ImageDraw
    cells = []
    for img, lab in items:
        if img.width != cell_w:
            img = img.resize(scaled_size(img.width, img.height, cell_w), Image.LANCZOS)
        cells.append(labelled(img, lab) if lab else img.convert("RGB"))
    cols = max(1, min(cols, len(cells) or 1))
    rows = (len(cells) + cols - 1) // cols
    ch = max((c.height for c in cells), default=40)
    gap, th = 8, (36 if title else 0)
    sheet = Image.new("RGB", (cols * cell_w + (cols + 1) * gap, th + rows * ch + (rows + 1) * gap),
                      SHEET_BG)
    if title:
        ImageDraw.Draw(sheet).text((gap, 8), title, fill=LABEL_FG, font=label_font(20))
    for i, c in enumerate(cells):
        r, k = divmod(i, cols)
        sheet.paste(c, (gap + k * (cell_w + gap), th + gap + r * (ch + gap)))
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(str(p), quality=88)
    return p


def contact_sheet(items, cols: int, out_path, cell_w: int, title=None) -> Path:
    """[(PIL image, label)] → jpg grid (sheets.contact_sheet when available)."""
    try:
        from . import sheets
        fn = getattr(sheets, "contact_sheet", None)
    except ImportError:
        fn = None
    if fn is not None:
        try:
            res = fn(items, cols, out_path, cell_w, title=title) if title else \
                fn(items, cols, out_path, cell_w)
            return Path(res) if res else Path(out_path)
        except TypeError:
            pass
    return _local_sheet(items, cols, out_path, cell_w, title)


def seam_pair(before, after, label_before: str, label_after: str, title=None):
    """Two frames side by side (sheets.seam_pair when available)."""
    try:
        from . import sheets
        fn = getattr(sheets, "seam_pair", None)
    except ImportError:
        fn = None
    if fn is not None:
        try:
            return fn(before, after, label_before, label_after, title=title)
        except TypeError:
            pass
    from PIL import Image
    pair = Image.new("RGB", (before.width + after.width + 6, before.height), (255, 210, 0))
    pair.paste(before.convert("RGB"), (0, 0))
    pair.paste(after.convert("RGB"), (before.width + 6, 0))
    return labelled(pair, title or ("%s | %s" % (label_before, label_after)))


def placeholder(text: str, size=(540, 120)):
    from PIL import Image, ImageDraw
    img = Image.new("RGB", size, SHEET_BG)
    ImageDraw.Draw(img).text((12, size[1] // 2 - 10), text, fill=LABEL_FG, font=label_font(20))
    return img


def tc(t: float) -> str:
    """m:ss.ss timecode for sheet labels."""
    t = max(0.0, float(t))
    m, s = divmod(t, 60.0)
    return "%d:%05.2f" % (int(m), s)

