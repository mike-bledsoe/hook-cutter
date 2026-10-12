"""clips.json: load, normalize, validate (contract section 7).

`load_spec` never fails on content problems; it normalizes what it can and
records issues (plan lint L01) on the spec (`_issues`) or the clip
(`clip["_issues"]`). `validate` adds the checks that need the transcript or
the source (index bounds, expect anchors, transcript_sha) and returns the
whole lint list. Keys starting with "_" are treated as comments.
"""
from __future__ import annotations

import copy
import difflib
import json
import math
import re
from pathlib import Path

from . import env

CLIP_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,47}$")
ROLES = ("cold_open", "setup", "body", "payoff", "close", "reaction", "broll")
JOINS = ("cut", "dip")
KINDS = ("cut", "caption_only")
FORMATS = ("vertical", "landscape")
FRAMING_MODES = ("track", "fixed", "letterbox", "full", "speaker")

TOP_KEYS = {"version", "transcript_sha", "defaults", "clips", "speakers"}
DEFAULT_KEYS = {"styles", "formats", "tighten", "safezone", "length"}
TIGHTEN_KEYS = {"enabled", "gap", "keep", "min_cut", "fillers"}
CLIP_KEYS = {"ledger", "title", "speaker", "kind", "source_file", "pieces", "omit", "verbatim",
             "fix", "emphasis", "hook_card", "captions", "caption_y", "caption_backing", "framing",
             "motion", "cover", "styles", "formats", "tighten", "post", "insert",
             "transcript_sha"}
LEGACY_CLIP_KEYS = {"spans"}
PIECE_KEYS = {"words", "time_src", "role", "join", "expect", "hold_after", "gain_db",
              "tighten", "chapter", "in_offset", "out_offset"}
HOOK_KEYS = {"lines", "dur", "text"}
COVER_KEYS = {"headline", "at_out", "shift", "lockup"}
FRAMING_KEYS = {"at_src", "mode", "x", "y", "crop", "reaction", "rect"}
SPEAKER_KEYS = {"x", "y"}
INSERT_KEYS = {"after", "text", "src"}
MOTION_KEYS = {"punch_in", "min_gap", "max_upscale"}
POST_KEYS = {"title", "caption"}      # the post texts lint L13 and QA G11 check
BACKING_MODES = ("auto", "always", "never")   # clip "caption_backing": overrides the preset's mode
EDGE_OFFSET_MAX = 60          # piece in_offset / out_offset bound, in output frames
EDGE_OFFSET_NOTE = 12         # a larger offset is named in seconds by a plan lint (INFO)
OUT_H = 1920                  # vertical output height: caption_y must land on the frame
HOOK_DUR_WARN = 3.0           # a hook card held longer than this costs the opening

DEFAULTS = {
    "styles": ["clean"],
    "formats": ["vertical"],
    "tighten": {"enabled": True, "gap": 0.6, "keep": 0.3, "min_cut": 0.12, "fillers": True},
    "safezone": "organic",
    "length": {"vertical": [15.0, 30.0, 60.0, 90.0], "landscape": [60.0, 300.0, 900.0, 1800.0]},
}
EXPECT_MIN_RATIO = 0.8
_TIME_RE = re.compile(r"^\d+(\.\d*)?$|^\.\d+$")


# ---- small helpers ----------------------------------------------------------

def lint(lid: str, level: str, msg: str, clip=None, at_out=None) -> dict:
    """One lint record (shared shape for spec and plan lints)."""
    return {"id": lid, "level": level, "msg": msg, "clip": clip, "at_out": at_out}


def parse_time(v) -> float:
    """Seconds from a number or "681.5", "11:21", "11:21.40", "1:02:03.4"."""
    if isinstance(v, bool) or v is None:
        raise env.HCError(f"bad time {v!r}: use seconds (681.5) or [h:]mm:ss[.ff]", 1)
    if isinstance(v, (int, float)):
        f = float(v)
        if not math.isfinite(f) or f < 0:
            raise env.HCError(f"bad time {v!r}: must be a finite number >= 0", 1)
        return f
    if isinstance(v, str):
        s = v.strip()
        parts = s.split(":")
        if 1 <= len(parts) <= 3 and all(parts):
            *hi, last = parts
            if _TIME_RE.match(last) and all(p.isdigit() for p in hi):
                sec = float(last)
                nums = [int(p) for p in hi]
                ok = True
                if nums and sec >= 60:
                    ok = False
                if len(nums) == 2 and nums[1] >= 60:
                    ok = False
                if ok:
                    total = sec
                    for k, n in enumerate(reversed(nums)):
                        total += n * (60 ** (k + 1))
                    return total
    raise env.HCError(f"bad time {v!r}: use seconds (681.5) or [h:]mm:ss[.ff]", 1)


def fmt_time(t) -> str:
    """mm:ss.ss (or h:mm:ss.ss) for messages."""
    if t is None:
        return "?"
    t = max(0.0, float(t))
    h, rem = divmod(t, 3600)
    m, s = divmod(rem, 60)
    return f"{int(h)}:{int(m):02d}:{s:05.2f}" if h else f"{int(m)}:{s:05.2f}"


def _as_int(v):
    if isinstance(v, bool):
        raise ValueError
    if isinstance(v, int):
        return v
    if isinstance(v, float) and v.is_integer():
        return int(v)
    if isinstance(v, str) and v.strip().lstrip("-").isdigit():
        return int(v.strip())
    raise ValueError


def _as_float(v):
    if isinstance(v, bool) or v is None:
        raise ValueError
    f = float(v)
    if not math.isfinite(f):
        raise ValueError
    return f


def _str_list(v):
    if v is None:
        return None
    if isinstance(v, str):
        return [p.strip() for p in v.split(",") if p.strip()]
    if isinstance(v, (list, tuple)) and all(isinstance(x, str) for x in v):
        return [x.strip() for x in v if x.strip()]
    raise ValueError


class _Issues:
    def __init__(self, sink: list, clip=None):
        self.sink, self.clip = sink, clip

    def fail(self, msg):
        self.sink.append(lint("L01", "FAIL", msg, self.clip))

    def warn(self, msg):
        self.sink.append(lint("L01", "WARN", msg, self.clip))

    def unknown(self, obj: dict, known: set, where: str):
        for k in obj:
            if isinstance(k, str) and k.startswith("_"):
                continue
            if k not in known:
                self.warn(f"unknown key {k!r} in {where} (ignored)")


# ---- loading / normalizing -------------------------------------------------

def spec_path(project) -> Path:
    from .project import root_of
    p = root_of(project)
    return p if p.suffix == ".json" else p / "clips.json"


def load_spec(project) -> dict:
    """Read and normalize P/clips.json. Raises HCError only when unreadable."""
    path = spec_path(project)
    if not path.exists():
        raise env.HCError(f"no clips.json at {path}: write the edit spec first "
                          "(see references/spec.md)", 1)
    try:
        # utf-8-sig: a byte-order mark (Notepad, PowerShell) is not a JSON error
        raw = json.loads(path.read_text(encoding="utf-8-sig"))
    except ValueError as exc:
        raise env.HCError(f"clips.json is not valid JSON: {exc}", 1)
    spec = normalize_spec(raw)
    _check_styles(spec, path.parent)
    return spec


def _check_styles(spec: dict, root) -> None:
    """L01 FAIL on every clip whose effective styles do not resolve to a preset in this project
    (name rules, search path, extends chain, value checks), so a typo fails at plan time."""
    from . import presets
    seen: dict = {}
    loaded: dict = {}
    for cid, clip in spec["clips"].items():
        for st in (clip.get("_eff") or {}).get("styles") or []:
            if st not in seen:
                try:
                    loaded[st] = presets.load(st, root)
                    seen[st] = None
                except env.HCError as exc:
                    seen[st] = exc.msg
            if seen[st]:
                clip.setdefault("_issues", []).append(
                    lint("L01", "FAIL", f"style {st!r} cannot be used: {seen[st]}", cid))
    _check_safezones(spec, root, loaded)


def _check_safezones(spec: dict, root, loaded: dict) -> None:
    """L01 FAIL for a safezone profile that does not exist, so a typo fails at plan time and
    not at render (where it also silently dropped lint L20). `defaults.safezone` written in
    clips.json is checked once, for the whole file. A preset's own profile is checked per clip
    and format, and only where the render reads it: when clips.json does not name one and the
    resolver presets.zone_name is there. `loaded`: {style: resolved preset} of the styles that
    load."""
    from . import presets
    try:
        zones = presets.safezones(root)
    except env.HCError:
        return                           # an unreadable safezones.json is reported at render
    have = ", ".join(sorted(zones)) or "none"
    d = spec.get("defaults") or {}
    if d.get("safezone_set"):
        if d.get("safezone") not in zones:
            spec.setdefault("_issues", []).append(
                lint("L01", "FAIL", f"defaults.safezone {d.get('safezone')!r} is not a safezone "
                     f"profile (available: {have}; add one in P/presets/safezones.json)"))
        return
    resolve = getattr(presets, "zone_name", None)
    if resolve is None:
        return
    for cid, clip in spec["clips"].items():
        eff = clip.get("_eff") or {}
        for st in eff.get("styles") or []:
            if st not in loaded:
                continue                 # already an L01 FAIL in _check_styles
            for fmt in eff.get("formats") or []:
                try:
                    preset = presets.for_format(loaded[st], fmt)
                    if not isinstance(preset.get("safezone"), str):
                        continue         # absent, or margins given inline
                    name = resolve(eff, preset, fmt)
                except env.HCError:
                    continue
                if name not in zones:
                    clip.setdefault("_issues", []).append(
                        lint("L01", "FAIL", f"style {st!r} cannot be used: its {fmt} safezone "
                             f"{name!r} is not a safezone profile (available: {have})", cid))


def normalize_spec(raw) -> dict:
    """Normalize a parsed clips.json object (pure; see module docstring)."""
    issues: list = []
    top = _Issues(issues)
    if not isinstance(raw, dict):
        raise env.HCError("clips.json must be a JSON object with a \"clips\" map", 1)
    if "clips" not in raw and "defaults" not in raw and raw and \
            all(isinstance(v, dict) for v in raw.values()):
        top.warn("clips.json is a bare map of clips; wrap it as {\"version\": 1, \"clips\": {...}}")
        raw = {"version": 1, "clips": raw}
    top.unknown(raw, TOP_KEYS, "clips.json")
    if raw.get("version", 1) != 1:
        top.warn(f"clips.json version {raw.get('version')!r} is not 1; reading it as version 1")
    tsha = raw.get("transcript_sha")
    if tsha is not None and not isinstance(tsha, str):
        top.fail("transcript_sha must be a string")
        tsha = None
    defaults = _norm_defaults(raw.get("defaults"), top)
    speakers = _norm_speakers(raw.get("speakers"), top)
    clips_raw = raw.get("clips")
    clips: dict = {}
    if clips_raw is None:
        top.fail("clips.json has no \"clips\" map")
    elif not isinstance(clips_raw, dict):
        top.fail("\"clips\" must be a map of clip id -> clip")
    else:
        for cid, entry in clips_raw.items():
            if isinstance(cid, str) and cid.startswith("_"):
                continue                 # a comment key ("_note", a parked "_c02-off" clip)
            clips[cid] = normalize_clip(cid, entry, defaults)
    return {"version": 1, "transcript_sha": tsha, "defaults": defaults, "speakers": speakers,
            "clips": clips, "_issues": issues}


def _norm_speakers(v, iss: _Issues) -> dict:
    """Top-level "speakers": {"<label>": {"x": 0..1, "y": 0..1}}: where each person sits in
    the source frame (normalized), so multi-face framing can follow who is talking."""
    out: dict = {}
    if v is None:
        return out
    if not isinstance(v, dict):
        iss.fail("\"speakers\" must map a speaker label to {\"x\": 0..1}, e.g. "
                 "{\"Host\": {\"x\": 0.27}}")
        return out
    for label, val in v.items():
        if isinstance(label, str) and label.startswith("_"):
            continue
        where = f"speakers[{label!r}]"
        if not isinstance(label, str) or not label.strip():
            iss.fail("speakers labels must be non-empty strings (the transcript's speaker names)")
            continue
        if not isinstance(val, dict):
            iss.fail(f"{where} must be an object like {{\"x\": 0.27}} (x from the shots sheet)")
            continue
        iss.unknown(val, SPEAKER_KEYS, where)
        ent = {}
        ok = True
        for key in ("x", "y"):
            if val.get(key) is None:
                continue
            try:
                f = _as_float(val[key])
                if not 0.0 <= f <= 1.0:
                    raise ValueError
                ent[key] = f
            except (ValueError, TypeError):
                iss.fail(f"{where}.{key} must be a number in 0..1 (share of the source width"
                         f"{'' if key == 'x' else ' / height'})")
                ok = False
        if "x" not in ent:
            if ok:
                iss.fail(f"{where} needs \"x\": the face centre as a share of the source width")
            continue
        out[label] = ent
    return out


def _norm_tighten(v, base: dict, iss: _Issues, where: str) -> dict:
    out = dict(base)
    if v is None:
        return out
    if isinstance(v, bool):
        out["enabled"] = v
        return out
    if not isinstance(v, dict):
        iss.fail(f"{where} must be true, false or an object")
        return out
    iss.unknown(v, TIGHTEN_KEYS, where)
    for k in ("gap", "keep", "min_cut"):
        if k in v:
            try:
                f = _as_float(v[k])
                if f < 0 or (k == "gap" and f <= 0):
                    raise ValueError
                out[k] = f
            except (ValueError, TypeError):
                iss.fail(f"{where}.{k} must be a number > 0")
    for k in ("enabled", "fillers"):
        if k in v:
            if isinstance(v[k], bool):
                out[k] = v[k]
            else:
                iss.fail(f"{where}.{k} must be true or false")
    return out


def _norm_defaults(d, iss: _Issues) -> dict:
    out = copy.deepcopy(DEFAULTS)
    if d is None:
        return out
    if not isinstance(d, dict):
        iss.fail("\"defaults\" must be an object")
        return out
    iss.unknown(d, DEFAULT_KEYS, "defaults")
    for k in ("styles", "formats"):
        if k in d:
            try:
                lst = _str_list(d[k])
                if not lst:
                    raise ValueError
                if k == "formats" and any(f not in FORMATS for f in lst):
                    iss.fail(f"defaults.formats: unknown format in {lst} (use vertical, landscape)")
                    continue
                out[k] = lst
            except ValueError:
                iss.fail(f"defaults.{k} must be a list of names")
    out["tighten"] = _norm_tighten(d.get("tighten"), out["tighten"], iss, "defaults.tighten")
    if "safezone" in d:
        if isinstance(d["safezone"], str) and d["safezone"]:
            out["safezone"] = d["safezone"]
            # Written in clips.json, so it overrides a preset's own "safezone"
            # (presets.zone_name). The marker is absent otherwise: specs that never set the
            # key keep their hashes.
            out["safezone_set"] = True
        else:
            iss.fail("defaults.safezone must be a profile name")
    if "length" in d:
        ln = d["length"]
        if not isinstance(ln, dict):
            iss.fail("defaults.length must map format -> [min, target_lo, target_hi, max]")
        else:
            for fmt, band in ln.items():
                if isinstance(fmt, str) and fmt.startswith("_"):
                    continue             # a comment key
                if fmt not in FORMATS:
                    iss.warn(f"defaults.length: unknown format {fmt!r}")
                    continue
                try:
                    vals = [_as_float(x) for x in band]
                    if len(vals) != 4 or vals != sorted(vals) or vals[0] < 0:
                        raise ValueError
                    out["length"][fmt] = vals
                except (ValueError, TypeError):
                    iss.fail(f"defaults.length.{fmt} must be 4 ascending seconds [min, lo, hi, max]")
    return out


def _norm_expect(v, iss, where):
    if v is None:
        return ["", ""]
    if isinstance(v, str):
        return [v, ""]
    if isinstance(v, (list, tuple)) and len(v) <= 2 and all(x is None or isinstance(x, str) for x in v):
        vals = [x or "" for x in v] + ["", ""]
        return vals[:2]
    iss.fail(f"{where}.expect must be [first_text, last_text]")
    return ["", ""]


def _norm_piece(raw, n: int, iss: _Issues) -> dict:
    where = f"pieces[{n}]"
    p = {"words": None, "time_src": None, "role": "body", "join": "cut", "expect": ["", ""],
         "hold_after": 0.0, "gain_db": 0.0, "tighten": True, "chapter": None,
         "in_offset": 0, "out_offset": 0}
    if isinstance(raw, (list, tuple)):
        raw = {"words": list(raw)}
    if not isinstance(raw, dict):
        iss.fail(f"{where} must be an object like {{\"words\": [a, b]}}")
        return p
    iss.unknown(raw, PIECE_KEYS, where)
    has_w, has_t = raw.get("words") is not None, raw.get("time_src") is not None
    if has_w == has_t:
        iss.fail(f"{where} needs exactly one of \"words\": [a, b] or \"time_src\": [t0, t1]")
    elif has_w:
        try:
            w = raw["words"]
            if not isinstance(w, (list, tuple)) or len(w) != 2:
                raise ValueError
            a, b = _as_int(w[0]), _as_int(w[1])
            if a < 0 or b < a:
                iss.fail(f"{where}.words {[a, b]}: need 0 <= a <= b")
            else:
                p["words"] = [a, b]
        except (ValueError, TypeError):
            iss.fail(f"{where}.words must be two word indices [a, b]")
    else:
        try:
            t = raw["time_src"]
            if not isinstance(t, (list, tuple)) or len(t) != 2:
                raise ValueError
            t0, t1 = parse_time(t[0]), parse_time(t[1])
            if t1 <= t0:
                iss.fail(f"{where}.time_src end must be after start")
            else:
                p["time_src"] = [t0, t1]
        except (ValueError, TypeError, env.HCError):
            iss.fail(f"{where}.time_src must be [t0, t1] (seconds or mm:ss)")
    role = raw.get("role", "body")
    if role not in ROLES:
        iss.warn(f"{where}.role {role!r} is not one of {', '.join(ROLES)}; using body")
        role = "body"
    p["role"] = role
    join = raw.get("join", "cut")
    if join not in JOINS:
        iss.fail(f"{where}.join must be \"cut\" or \"dip\"")
        join = "cut"
    p["join"] = join
    p["expect"] = _norm_expect(raw.get("expect"), iss, where)
    if "hold_after" in raw:
        try:
            h = _as_float(raw["hold_after"])
            if h < 0:
                raise ValueError
            p["hold_after"] = h
        except (ValueError, TypeError):
            iss.fail(f"{where}.hold_after must be seconds >= 0")
    if "gain_db" in raw:
        try:
            g = _as_float(raw["gain_db"])
            if not -12.0 <= g <= 12.0:
                raise ValueError
            p["gain_db"] = g
        except (ValueError, TypeError):
            iss.fail(f"{where}.gain_db must be a number in -12..12")
    if "tighten" in raw:
        if isinstance(raw["tighten"], bool):
            p["tighten"] = raw["tighten"]
        else:
            iss.fail(f"{where}.tighten must be true or false")
    if raw.get("chapter") is not None:
        if isinstance(raw["chapter"], str):
            p["chapter"] = raw["chapter"]
        else:
            iss.fail(f"{where}.chapter must be a string")
    for key in ("in_offset", "out_offset"):
        if raw.get(key) is None:
            continue
        try:
            k = _as_int(raw[key])
            if not -EDGE_OFFSET_MAX <= k <= EDGE_OFFSET_MAX:
                raise ValueError
            p[key] = k
        except (ValueError, TypeError):
            iss.fail(f"{where}.{key} must be a whole number of frames in "
                     f"-{EDGE_OFFSET_MAX}..{EDGE_OFFSET_MAX} (+ moves the edge later; to move "
                     "further, change the piece's words or use time_src)")
    return p


def split_hook_text(text: str) -> list:
    """Legacy hook card text: lines separated by a literal \\N, \\n or a newline."""
    parts = re.split(r"\\N|\\n|\n", text)
    return [p.strip() for p in parts if p.strip()]


def _norm_hook(v, iss: _Issues):
    if v is None or v is False:
        return None
    if isinstance(v, str):
        v = {"text": v}
    elif isinstance(v, (list, tuple)):
        v = {"lines": list(v)}
    if not isinstance(v, dict):
        iss.fail("hook_card must be {\"lines\": [...], \"dur\": \"auto\"} or null")
        return None
    iss.unknown(v, HOOK_KEYS, "hook_card")
    lines = v.get("lines")
    if lines is None and isinstance(v.get("text"), str):
        lines = split_hook_text(v["text"])
    if lines is None:
        iss.fail("hook_card needs \"lines\" (list of strings)")
        return None
    if not isinstance(lines, (list, tuple)) or not all(isinstance(x, str) for x in lines):
        iss.fail("hook_card.lines must be a list of strings")
        return None
    lines = [x.strip() for x in lines if x.strip()]
    if not lines:
        iss.warn("hook_card has no text; treating it as null")
        return None
    for ln in lines:
        if ln.count("*") % 2:
            iss.warn(f"hook_card line {ln!r} has an unbalanced * accent marker")
    dur = v.get("dur", "auto")
    if dur != "auto":
        try:
            dur = _as_float(dur)
            if dur <= 0:
                raise ValueError
            if dur > HOOK_DUR_WARN:
                iss.warn(f"hook_card.dur {dur:g} s is longer than {HOOK_DUR_WARN:g} s: the card "
                         "hides the first captions that long (\"auto\" sizes it to the text)")
        except (ValueError, TypeError):
            iss.fail("hook_card.dur must be \"auto\" or seconds > 0")
            dur = "auto"
    return {"lines": lines, "dur": dur}


def _norm_framing(v, iss: _Issues) -> list:
    if v is None:
        return []
    if not isinstance(v, (list, tuple)):
        iss.fail("framing must be a list of marks")
        return []
    out = []
    for k, m in enumerate(v):
        where = f"framing[{k}]"
        if not isinstance(m, dict):
            iss.fail(f"{where} must be an object")
            continue
        iss.unknown(m, FRAMING_KEYS, where)
        try:
            at = parse_time(m.get("at_src"))
        except env.HCError:
            iss.fail(f"{where}.at_src must be a time")
            continue
        mode = m.get("mode")
        if mode not in FRAMING_MODES:
            iss.fail(f"{where}.mode must be one of {', '.join(FRAMING_MODES)}")
            continue
        mk = {"at_src": at, "mode": mode}
        ok = True
        for key, lo, hi in (("x", 0.0, 1.0), ("y", 0.0, 1.0), ("crop", 0.3, 1.0)):
            if m.get(key) is None:
                continue
            try:
                f = _as_float(m[key])
                if not lo <= f <= hi:
                    raise ValueError
                mk[key] = f
            except (ValueError, TypeError):
                iss.fail(f"{where}.{key} must be a number in {lo}..{hi}")
                ok = False
        if mode == "speaker":
            # follow the speakers map from here: the label speaking sets x (and y)
            for key in ("x", "y"):
                if key in mk:
                    iss.warn(f"{where}.{key} does not apply to mode speaker (x comes from "
                             "\"speakers\"; ignored)")
                    mk.pop(key)
        elif mode == "fixed":
            mk.setdefault("x", 0.5)
            mk.setdefault("y", 0.5)
        elif "y" in mk:
            iss.warn(f"{where}.y only applies to mode fixed (ignored)")
            mk.pop("y")
        if mode in ("track", "fixed", "speaker"):
            mk.setdefault("crop", 1.0)
        rv = m.get("reaction", False)
        if isinstance(rv, bool):
            if rv:                       # absent means false (keeps older specs' hashes)
                mk["reaction"] = True
        else:
            iss.fail(f"{where}.reaction must be true or false")
            ok = False
        if m.get("rect") is not None:
            rect = _norm_rect(m["rect"])
            if rect is None:
                iss.fail(f"{where}.rect must be [x, y, w, h] as shares of the source frame "
                         "(0..1, w and h > 0, inside the frame)")
                ok = False
            elif mode != "letterbox":
                iss.warn(f"{where}.rect only applies to mode letterbox (ignored)")
            else:
                mk["rect"] = rect
        if ok:
            out.append(mk)
    out.sort(key=lambda m: m["at_src"])
    return out


def _norm_rect(v):
    """[x, y, w, h] normalized to the display frame, or None when invalid."""
    if not isinstance(v, (list, tuple)) or len(v) != 4:
        return None
    try:
        x, y, w, h = (_as_float(t) for t in v)
    except (ValueError, TypeError):
        return None
    eps = 1e-6
    if min(x, y) < 0 or w <= 0 or h <= 0 or x + w > 1 + eps or y + h > 1 + eps:
        return None
    return [x, y, w, h]


def _norm_cover(v, iss: _Issues) -> dict:
    out = {"headline": None, "at_out": "auto", "shift": 0, "lockup": None}
    if v is None:
        return out
    if not isinstance(v, dict):
        iss.fail("cover must be an object")
        return out
    iss.unknown(v, COVER_KEYS, "cover")
    if v.get("headline") is not None:
        if isinstance(v["headline"], str):
            out["headline"] = v["headline"]
        else:
            iss.fail("cover.headline must be a string")
    at = v.get("at_out", "auto")
    if at not in (None, "auto"):
        try:
            out["at_out"] = parse_time(at)
        except env.HCError:
            iss.fail("cover.at_out must be \"auto\" or output seconds")
    if v.get("shift") is not None:
        try:
            out["shift"] = int(round(_as_float(v["shift"])))
        except (ValueError, TypeError):
            iss.fail("cover.shift must be a number of pixels")
        if out["shift"] < 0:
            # Sliding the photo down would leave an empty band at the top, so the cover is made
            # as with shift 0 while every recorded face position would be off by the shift.
            iss.fail("cover.shift must be 0 or more: it only slides the photo up (a face that "
                     "sits too high needs another frame: cover.at_out)")
            out["shift"] = 0
    lk = v.get("lockup")
    if lk is not None:
        if isinstance(lk, str):
            lk = [lk]
        if isinstance(lk, list) and all(isinstance(x, str) for x in lk):
            out["lockup"] = lk
        else:
            iss.fail("cover.lockup must be a list of strings or null")
    return out


def _norm_motion(v, iss: _Issues):
    """Clip "motion": overrides of the preset's motion (punch_in, min_gap, max_upscale). A valid
    object is stored exactly as written, comment keys included (the clip hash and the render key
    hash it). A value that is not a finite number >= 0 is L01 FAIL and that key is dropped, so
    framing never reads it; null keeps meaning "not set"."""
    if v is None:
        return None
    if not isinstance(v, dict):
        iss.fail("motion must be an object or null")
        return None
    iss.unknown(v, MOTION_KEYS, "motion")
    out = {}
    for k, val in v.items():
        if k in MOTION_KEYS and val is not None:
            try:
                if _as_float(val) < 0:
                    raise ValueError
            except (ValueError, TypeError):
                iss.fail(f"motion.{k} must be a number >= 0 (punch_in 1.0 turns punch-ins off), "
                         f"got {val!r}")
                continue
        out[k] = val
    return out


def _norm_post(v, iss: _Issues):
    """Clip "post": stored exactly as written (the export manifest copies it whole). Only
    post.title and post.caption are checked against the clip's words (lint L13, QA G11), so any
    other key is named; a title or caption that is not a string is L01 FAIL."""
    if v is None:
        return None
    if not isinstance(v, dict):
        iss.fail("post must be an object")
        return None
    for k, val in v.items():
        if isinstance(k, str) and k.startswith("_"):
            continue                     # a comment key
        if k not in POST_KEYS:
            iss.warn(f"unknown key {k!r} in post: it is copied into the export manifest as "
                     "written, but only title and caption are checked against the clip's words "
                     "(L13, QA G11)")
        elif val is not None and not isinstance(val, str):
            iss.fail(f"post.{k} must be a string")
    return v


def _norm_insert(v, iss: _Issues) -> list:
    """Clip "insert": speech the transcript missed, typed by the editor so it is captioned:
    [{"after": word index, "text": "what is said", "src": [s, e]}]. `after` is the word the
    speech follows, `src` the source span it is heard in (any time format), `text` verbatim."""
    if v is None:
        return []
    if isinstance(v, dict):
        v = [v]
    if not isinstance(v, (list, tuple)):
        iss.fail("insert must be a list like [{\"after\": 120, \"text\": \"...\", "
                 "\"src\": [s, e]}]")
        return []
    out = []
    for n, m in enumerate(v):
        where = f"insert[{n}]"
        if not isinstance(m, dict):
            iss.fail(f"{where} must be an object with \"after\", \"text\" and \"src\"")
            continue
        iss.unknown(m, INSERT_KEYS, where)
        ok = True
        after = None
        try:
            after = _as_int(m.get("after"))
            if after < 0:
                raise ValueError
        except (ValueError, TypeError):
            iss.fail(f"{where}.after must be a word index (the word the inserted speech follows)")
            ok = False
        text = m.get("text")
        if not isinstance(text, str) or not text.split():
            iss.fail(f"{where}.text must be the words that are said (a non-empty string)")
            ok = False
        src = None
        try:
            t = m.get("src")
            if not isinstance(t, (list, tuple)) or len(t) != 2:
                raise ValueError
            s0, s1 = parse_time(t[0]), parse_time(t[1])
            if s1 <= s0:
                iss.fail(f"{where}.src end must be after its start")
                ok = False
            else:
                src = [s0, s1]
        except (ValueError, TypeError, env.HCError):
            iss.fail(f"{where}.src must be [start, end] in source time (seconds or mm:ss.ff)")
            ok = False
        if ok:
            out.append({"after": after, "text": " ".join(text.split()), "src": src})
    out.sort(key=lambda m: (m["src"][0], m["after"]))
    return out


def _norm_index_list(v, iss, where) -> list:
    """List of ints; nested [a, b] pairs are inclusive ranges."""
    if v is None:
        return []
    if not isinstance(v, (list, tuple)):
        iss.fail(f"{where} must be a list of word indices")
        return []
    out = set()
    for x in v:
        try:
            if isinstance(x, (list, tuple)):
                if len(x) != 2:
                    raise ValueError
                a, b = _as_int(x[0]), _as_int(x[1])
                if b < a or a < 0:
                    raise ValueError
                out.update(range(a, b + 1))
            else:
                i = _as_int(x)
                if i < 0:
                    raise ValueError
                out.add(i)
        except (ValueError, TypeError):
            iss.fail(f"{where}: {x!r} is not a word index")
    return sorted(out)


def _norm_ranges(v, iss, where) -> list:
    if v is None:
        return []
    if isinstance(v, (list, tuple)) and len(v) == 2 and all(isinstance(x, int) for x in v):
        v = [v]
    if not isinstance(v, (list, tuple)):
        iss.fail(f"{where} must be a list of [a, b] ranges")
        return []
    out = []
    for r in v:
        try:
            if not isinstance(r, (list, tuple)) or len(r) != 2:
                raise ValueError
            a, b = _as_int(r[0]), _as_int(r[1])
            if a < 0 or b < a:
                raise ValueError
            out.append([a, b])
        except (ValueError, TypeError):
            iss.fail(f"{where}: {r!r} is not an [a, b] range")
    return sorted(out)


def normalize_clip(cid, raw, defaults: dict) -> dict:
    """Normalize one clip entry; issues land in clip["_issues"]."""
    issues: list = []
    iss = _Issues(issues, cid)
    if not isinstance(cid, str) or not CLIP_ID_RE.match(cid):
        iss.fail(f"clip id {cid!r} is invalid: use lowercase letters, digits and hyphens "
                 "(max 48 chars, starting with a letter or digit)")
    c = {"ledger": None, "title": None, "speaker": None, "kind": "cut", "source_file": None,
         "pieces": [], "omit": [], "verbatim": [], "fix": {}, "emphasis": [], "hook_card": None,
         "captions": True, "caption_y": None, "caption_backing": None, "framing": [],
         "motion": None, "cover": _norm_cover(None, iss), "styles": None, "formats": None,
         "tighten": None, "post": None, "insert": []}
    if not isinstance(raw, dict):
        iss.fail("clip entry must be an object")
        c["_issues"] = issues
        c["_eff"] = _eff(c, defaults)
        return c
    iss.unknown(raw, CLIP_KEYS | LEGACY_CLIP_KEYS, "clip")
    for k in ("ledger", "title", "speaker", "source_file"):
        if raw.get(k) is not None:
            if isinstance(raw[k], str):
                c[k] = raw[k]
            else:
                iss.fail(f"{k} must be a string")
    kind = raw.get("kind", "cut")
    if kind not in KINDS:
        iss.fail(f"kind must be \"cut\" or \"caption_only\" (got {kind!r})")
        kind = "cut"
    c["kind"] = kind
    pieces_raw = raw.get("pieces")
    if pieces_raw is None and raw.get("spans") is not None:
        iss.warn("legacy \"spans\" converted to \"pieces\"")
        pieces_raw = raw["spans"]
    if kind == "caption_only":
        if not c["source_file"]:
            iss.fail("caption_only clips need \"source_file\" (the finished video)")
        if pieces_raw:
            iss.fail("caption_only clips must not have \"pieces\"")
        for k in ("omit", "verbatim"):
            if raw.get(k):
                iss.warn(f"{k} is ignored for caption_only clips")
        c["insert"] = _norm_insert(raw.get("insert"), iss)
    else:
        if c["source_file"]:
            iss.warn("source_file is only used by caption_only clips (ignored)")
        if not isinstance(pieces_raw, (list, tuple)) or not pieces_raw:
            iss.fail("clip has no pieces: add \"pieces\": [{\"words\": [a, b]}]")
        else:
            c["pieces"] = [_norm_piece(p, n, iss) for n, p in enumerate(pieces_raw)]
            c["pieces"][0]["join"] = "cut"
    if kind == "cut":
        c["omit"] = _norm_index_list(raw.get("omit"), iss, "omit")
        c["verbatim"] = _norm_ranges(raw.get("verbatim"), iss, "verbatim")
        c["insert"] = _norm_insert(raw.get("insert"), iss)
    fix = raw.get("fix")
    if fix is not None:
        if not isinstance(fix, dict):
            iss.fail("fix must map word index -> display text")
        else:
            for k, v in fix.items():
                if isinstance(k, str) and k.startswith("_"):
                    continue             # a comment key
                try:
                    i = _as_int(k)
                    if i < 0 or not isinstance(v, str):
                        raise ValueError
                    c["fix"][str(i)] = v
                except (ValueError, TypeError):
                    iss.fail(f"fix entry {k!r}: {v!r} must be \"index\": \"text\"")
    c["emphasis"] = _norm_index_list(raw.get("emphasis"), iss, "emphasis")
    c["hook_card"] = _norm_hook(raw.get("hook_card"), iss)
    if "captions" in raw:
        if isinstance(raw["captions"], bool):
            c["captions"] = raw["captions"]
        else:
            iss.fail("captions must be true or false")
    if raw.get("caption_y") is not None:
        try:
            cy = _as_float(raw["caption_y"])
            if not 0.0 <= cy <= OUT_H:
                iss.fail(f"caption_y {cy:g} is off the frame: use a pixel y in 0..{OUT_H} of the "
                         "vertical output (the caption block centre), or null")
            else:
                c["caption_y"] = cy
        except (ValueError, TypeError):
            iss.fail("caption_y must be a pixel y or null")
    if raw.get("caption_backing") is not None:
        cb = raw["caption_backing"]
        if isinstance(cb, str) and cb.strip().lower() in BACKING_MODES:
            c["caption_backing"] = cb.strip().lower()
        else:
            iss.fail("caption_backing must be \"auto\", \"always\" or \"never\" (it overrides "
                     "the preset's caption backing mode for this clip; the hook card follows it)")
    c["framing"] = _norm_framing(raw.get("framing"), iss)
    c["motion"] = _norm_motion(raw.get("motion"), iss)
    c["cover"] = _norm_cover(raw.get("cover"), iss)
    for k in ("styles", "formats"):
        if raw.get(k) is not None:
            try:
                lst = _str_list(raw[k])
                if not lst:
                    raise ValueError
                if k == "formats" and any(f not in FORMATS for f in lst):
                    iss.fail(f"formats: unknown format in {lst} (use vertical, landscape)")
                else:
                    c[k] = lst
            except ValueError:
                iss.fail(f"{k} must be a list of names")
    if raw.get("tighten") is not None:
        t = raw["tighten"]
        if isinstance(t, (bool, dict)):
            _norm_tighten(t, defaults["tighten"], iss, "tighten")   # validation only
            c["tighten"] = t if isinstance(t, bool) else {k: t[k] for k in t if k in TIGHTEN_KEYS}
        else:
            iss.fail("tighten must be true, false or an object")
    c["post"] = _norm_post(raw.get("post"), iss)
    # S1 / I10: the pin on a caption-only clip's own transcript. Stored ONLY when given: a
    # None default here would change the hash of every clip in every project.
    pin = raw.get("transcript_sha")
    if pin is not None:
        if not isinstance(pin, str) or not pin.strip():
            iss.fail("transcript_sha must be a string: the sha256 of this clip's transcript "
                     "(transcripts/<clip>.json)")
        elif kind != "caption_only":
            iss.warn("transcript_sha on a clip is only used by caption_only clips (ignored); a "
                     "cut clip is pinned by the top-level transcript_sha")
        else:
            c["transcript_sha"] = pin.strip()
    c["_issues"] = issues
    c["_eff"] = _eff(c, defaults)
    return c


def _eff(c: dict, defaults: dict) -> dict:
    sink: list = []
    tight = _norm_tighten(c.get("tighten"), defaults["tighten"], _Issues(sink), "tighten")
    if c.get("kind") == "caption_only":
        tight["enabled"] = False
    eff = {"styles": list(c.get("styles") or defaults["styles"]),
           "formats": list(c.get("formats") or defaults["formats"]),
           "tighten": tight,
           "safezone": defaults["safezone"],
           "length": copy.deepcopy(defaults["length"])}
    if defaults.get("safezone_set"):
        eff["safezone_set"] = True       # only when clips.json set defaults.safezone (see there)
    return eff


def public_clip(clip: dict) -> dict:
    """Clip entry without private (underscore) keys."""
    return {k: v for k, v in clip.items() if not k.startswith("_")}


def clip_hash(spec: dict, clip_id: str) -> str:
    """Stable hash of the clip entry, its effective defaults and the speakers map (framing
    follows the map, so moving a person's x re-plans every clip).

    A value the normaliser rejects is reset to its default, so on its own an invalid edit
    would hash like no edit and the old plan (and render) would pass as current. The FAIL
    issues of the spec and the clip are therefore hashed too, but only when there are any: a
    clip that normalises without a FAIL keeps the hash it always had, and taking the bad value
    out again restores it."""
    clip = spec["clips"][clip_id]
    data = {"clip": public_clip(clip), "eff": clip.get("_eff"),
            "speakers": spec.get("speakers") or {}}
    fails = [[x.get("id"), x.get("level"), x.get("msg")]
             for x in list(spec.get("_issues") or []) + list(clip.get("_issues") or [])
             if x.get("level") == "FAIL"]
    if fails:
        data["issues"] = fails
    return env.stable_hash(data)


# ---- validation --------------------------------------------------------------

def _tokens(text: str) -> list:
    from . import textnorm
    return list(textnorm.norm_tokens(text or ""))


def _range_tokens(words: list, idxs) -> list:
    from . import textnorm
    glued = textnorm.glue_tokens([{"w": words[i]["w"], "i": i} for i in idxs])
    return list(textnorm.norm_tokens([g["text"] for g in glued]))


def anchor_ratio(expect_text: str, words: list, idxs, at_end: bool = False) -> float | None:
    """Share of expect tokens found in order at the start (or end) of the range.
    None when the expect text has no tokens."""
    exp = _tokens(expect_text)
    if not exp:
        return None
    idxs = list(idxs)
    take = idxs[-(len(exp) + 4):] if at_end else idxs[:len(exp) + 4]
    rng = _range_tokens(words, take)
    head = rng[-len(exp):] if at_end else rng[:len(exp)]
    sm = difflib.SequenceMatcher(None, exp, head, autojunk=False)
    hit = sum(b.size for b in sm.get_matching_blocks())
    return hit / len(exp)


def piece_word_indices(piece: dict, words: list) -> list:
    """Word indices a piece covers (time pieces: words whose midpoint lies inside)."""
    if piece.get("words"):
        a, b = piece["words"]
        return list(range(a, min(b, len(words) - 1) + 1)) if a < len(words) else []
    if piece.get("time_src"):
        t0, t1 = piece["time_src"]
        return [k for k, w in enumerate(words) if t0 <= (w["s"] + w["e"]) / 2 < t1]
    return []


def validate(spec: dict, transcript: dict | None = None, source: dict | None = None,
             clip_ids=None, clip_transcripts: dict | None = None) -> list:
    """All spec lints (L01 schema/bounds, L02 expect anchors, L03 transcript_sha).

    `transcript`: the project transcript (cut clips). `clip_transcripts`:
    {clip_id: transcript} for caption_only clips. Lints carry "clip"."""
    out = list(spec.get("_issues", []))
    ids = list(spec["clips"]) if clip_ids is None else list(clip_ids)
    tsha = spec.get("transcript_sha")
    if transcript is not None:
        out.extend(_check_speakers(spec, transcript))
    for cid in ids:
        clip = spec["clips"].get(cid)
        if clip is None:
            out.append(lint("L01", "FAIL", env.unknown_clip(cid, list(spec["clips"])).msg, cid))
            continue
        out.extend(clip.get("_issues", []))
        if clip["kind"] == "caption_only":
            tr = (clip_transcripts or {}).get(cid)
            if tr is not None:
                _check_indices(clip, cid, tr.get("words") or [], out, pieces=False)
                _check_inserts(clip, cid, tr.get("words") or [], out, pieces=False)
                _check_clip_pin(clip, cid, tr, out)
            continue
        if transcript is None:
            continue
        words = transcript.get("words") or []
        if tsha and transcript.get("sha256") and not sha_matches(tsha, transcript):
            out.append(lint("L03", "FAIL", "transcript_sha in clips.json does not match "
                            "transcript.json: the transcript changed, re-anchor word indices "
                            "(then update transcript_sha)", cid))
        _check_indices(clip, cid, words, out, pieces=True)
        _check_inserts(clip, cid, words, out, pieces=True)
        _check_hook_dur(clip, cid, words, out)
        dur = (source or {}).get("duration")
        for n, p in enumerate(clip["pieces"]):
            if p.get("time_src") and dur is not None and p["time_src"][1] > float(dur) + 1e-6:
                out.append(lint("L01", "FAIL", f"pieces[{n}].time_src ends at "
                                f"{fmt_time(p['time_src'][1])}, after the source end "
                                f"{fmt_time(dur)}", cid))
            _check_expect(p, n, clip, cid, words, out)
    return out


def sha_matches(tsha: str, transcript: dict) -> bool:
    """True when clips.json's transcript_sha still anchors this transcript: its sha256, its
    word-text sha, or a sha of an earlier version with the same word texts (a re-time keeps
    every index)."""
    if tsha in (transcript.get("sha256"), transcript.get("text_sha")):
        return True
    return tsha in (transcript.get("text_compatible_shas") or [])


def _check_clip_pin(clip, cid, tr, out):
    """L03 for a caption_only clip: its fix / emphasis / insert keys are word indices into
    transcripts/<clip>.json, and nothing else anchors them (no pieces, so no expect; the
    top-level transcript_sha is about transcript.json). The clip's own "transcript_sha" pins
    them. Without a pin the clip FAILs once its transcript has been re-made with other word
    texts (`hc transcribe` records that as "text_changed_from"); a clip that uses no index key
    has nothing to drift."""
    if not (clip.get("fix") or clip.get("emphasis") or clip.get("insert")):
        return
    pin = clip.get("transcript_sha")
    now = tr.get("sha256")
    tail = (f"check them with `hc words P A B --clip {cid}`, then set this clip's "
            f"\"transcript_sha\" to {now!r}" if now else
            f"check them with `hc words P A B --clip {cid}`")
    if pin:
        if now and not sha_matches(pin, tr):
            out.append(lint("L03", "FAIL", f"this clip's transcript_sha does not match "
                            f"transcripts/{cid}.json: the clip's transcript changed, so its fix / "
                            f"emphasis / insert indices may point at other words; {tail}", cid))
    elif tr.get("text_changed_from"):
        out.append(lint("L03", "FAIL", f"transcripts/{cid}.json was re-made with different words "
                        "and this clip has no transcript_sha: its fix / emphasis / insert indices "
                        f"may point at other words; {tail}", cid))


def _check_speakers(spec: dict, transcript: dict) -> list:
    """L01 WARN for speakers-map labels that no transcript word or clip speaker uses."""
    smap = spec.get("speakers") or {}
    if not smap:
        return []
    known = {w.get("sp") for w in transcript.get("words") or [] if w.get("sp")}
    known |= {c.get("speaker") for c in spec["clips"].values() if c.get("speaker")}
    out = []
    for label in smap:
        if label not in known:
            have = ", ".join(sorted(str(k) for k in known)) or "none (no --speakers-vtt labels)"
            out.append(lint("L01", "WARN", f"speakers label {label!r} matches no transcript "
                            f"speaker or clip speaker (labels: {have}): framing will not use it"))
    return out


def estimate_duration(clip: dict, words: list) -> float | None:
    """Rough output length of a cut clip from its pieces (before tighten), or None."""
    total, any_piece = 0.0, False
    for p in clip.get("pieces") or []:
        if p.get("words"):
            a, b = p["words"]
            if b >= len(words) or a > b:
                return None
            total += float(words[b]["e"]) - float(words[a]["s"])
        elif p.get("time_src"):
            total += float(p["time_src"][1]) - float(p["time_src"][0])
        else:
            continue
        total += float(p.get("hold_after") or 0.0)
        any_piece = True
    return total if any_piece else None


def _check_hook_dur(clip, cid, words, out):
    hook = clip.get("hook_card") or {}
    dur = hook.get("dur")
    if not isinstance(dur, (int, float)) or isinstance(dur, bool):
        return
    est = estimate_duration(clip, words)
    if est is not None and dur >= est - 1.0:
        out.append(lint("L01", "FAIL", f"hook_card.dur {dur:g} s covers the whole clip (about "
                        f"{est:.1f} s): no caption would ever show; use \"auto\" or a few "
                        "seconds", cid))


def _check_indices(clip, cid, words, out, pieces: bool):
    nw = len(words)
    covered = set()
    if pieces:
        for n, p in enumerate(clip["pieces"]):
            if p.get("words"):
                a, b = p["words"]
                if b >= nw:
                    out.append(lint("L01", "FAIL", f"pieces[{n}].words [{a}, {b}] is out of range "
                                    f"(transcript has {nw} words, last index {nw - 1})", cid))
                    continue
                covered.update(range(a, b + 1))
        word_pieces = covered.copy()
        for p in clip["pieces"]:
            if p.get("time_src"):
                covered.update(piece_word_indices(p, words))
        for i in clip["omit"]:
            if i >= nw:
                out.append(lint("L01", "FAIL", f"omit index {i} is out of range", cid))
            elif i not in word_pieces:
                out.append(lint("L01", "FAIL", f"omit index {i} is not inside any word piece", cid))
        for a, b in clip["verbatim"]:
            if b >= nw:
                out.append(lint("L01", "FAIL", f"verbatim [{a}, {b}] is out of range", cid))
            elif not set(range(a, b + 1)) & word_pieces:
                out.append(lint("L01", "WARN", f"verbatim [{a}, {b}] is outside every piece", cid))
    else:
        covered = set(range(nw))
    for k in clip["fix"]:
        i = int(k)
        if i >= nw:
            out.append(lint("L01", "FAIL", f"fix index {i} is out of range", cid))
        elif i not in covered:
            out.append(lint("L01", "WARN", f"fix index {i} is not inside any piece (no effect)", cid))
    for i in clip["emphasis"]:
        if i >= nw:
            out.append(lint("L01", "FAIL", f"emphasis index {i} is out of range", cid))
        elif i not in covered:
            out.append(lint("L01", "WARN", f"emphasis index {i} is not inside any piece", cid))


def _check_inserts(clip, cid, words, out, pieces: bool):
    """L01 FAIL for an insert whose `after` is not a word of the clip, or whose `src` starts
    before that word does. (The planner checks that `src` lies inside the kept audio.)"""
    nw = len(words)
    covered = set(range(nw))
    if pieces:
        covered = set()
        for p in clip["pieces"]:
            covered.update(piece_word_indices(p, words))
    for n, ins in enumerate(clip.get("insert") or []):
        a = ins["after"]
        if a >= nw:
            out.append(lint("L01", "FAIL", f"insert[{n}].after {a} is out of range (transcript "
                            f"has {nw} words, last index {nw - 1})", cid))
        elif a not in covered:
            out.append(lint("L01", "FAIL", f"insert[{n}].after {a} is not inside any piece: use "
                            "the index of the word the inserted speech follows", cid))
        elif ins["src"][0] < float(words[a]["s"]) - 1e-6:
            out.append(lint("L01", "FAIL", f"insert[{n}].src starts at {fmt_time(ins['src'][0])}, "
                            f"before word {a} ({words[a]['w']!r}) starts at "
                            f"{fmt_time(words[a]['s'])}: src must lie after the word it follows",
                            cid))


def _check_expect(p, n, clip, cid, words, out):
    first, last = p.get("expect") or ["", ""]
    if not (first or last):
        return
    idxs = piece_word_indices(p, words)
    if p.get("words") and p["words"][1] >= len(words):
        return
    if not idxs:
        return
    fix = clip.get("fix") or {}
    for text, at_end, label in ((first, False, "first"), (last, True, "last")):
        if not text:
            continue
        r = anchor_ratio(text, words, idxs, at_end)
        if r is not None and r < EXPECT_MIN_RATIO and fix:
            # the agent may have written the anchor from fixed (display) text
            fixed = {i: dict(words[i], w=fix.get(str(i)) or words[i]["w"]) for i in idxs}
            r = max(r, anchor_ratio(text, fixed, idxs, at_end) or 0.0)
        if r is not None and r < EXPECT_MIN_RATIO:
            sl = idxs[-6:] if at_end else idxs[:6]
            got = " ".join(words[i]["w"] for i in sl)
            out.append(lint("L02", "FAIL", f"pieces[{n}] expect {label} {text!r} does not match "
                            f"the {label} words ({got!r}, {r:.0%}): indices drifted, re-anchor "
                            "with `find`", cid))
