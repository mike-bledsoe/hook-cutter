"""Burned-in captions: display tokens, emphasis, paging, hook card, legibility, strip frames, SRT.

`build_track(plan, clip, preset, fmt, out_size, fps, luma)` turns a plan (contract 8.1) into a
CaptionTrack: a full-width transparent RGBA strip (`band_y`, `band_h`) with one frame per output
frame, piped into the final encode. Everything is measured with real font metrics and every
frame is a pure function of (page, per-token visual state), so renders are deterministic and
identical frames are drawn once.
"""
from __future__ import annotations

import math
import re
import unicodedata
from bisect import bisect_right
from collections import OrderedDict
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Iterator

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from . import env, typeset

SKILL_ROOT = Path(__file__).resolve().parents[2]
PRESET_DIR = SKILL_ROOT / "presets"
LUMA_FPS = 5                 # luma array sample rate (contract 9.2)
PAGE_TAIL = 0.4              # a page holds this long after its last word ends
HOOK_MIN_REMAIN = 0.4        # a page cut by the hook card is dropped below this (safety net)
SHRINK_FLOOR = 0.5           # a page with an over-wide single word may shrink to this x size
WORD_SHRUNK = 0.9            # a page shrunk below this x size is reported (issue word_shrunk)
PILL_CONTRAST = 3.0          # pill when shadow-aware contrast of the text on the page p90 is below
TEXTURE_EDGE = 0.08          # luma step (0..1) between neighbouring luma pixels that counts as an edge
TEXTURE_SHARE = 0.15         # pill when more than this share of the box is edges (busy background)
SHOT_NEED_SHARE = 0.25       # a shot is pilled when at least this share of its pages need it ...
SEVERE_CONTRAST = 2.5        # ... or any page (or the hook card) falls under this contrast ...
SEVERE_TEXTURE = 0.30        # ... or over this edge density: one marginal page never pills a shot
POP_KEYS = ((0.0, 0.82), (0.11, 1.08), (0.22, 1.0))   # (seconds, scale) for the emphasis pop
FRAME_CACHE = 16
INVISIBLE = (0, 0, 0, 0, 0, 1000)   # token state: not drawn (unspoken in build reveal)
PAGE_COST, ORPHAN_COST, DANGLER_COST, BALANCE_COST = 100.0, 30.0, 8.0, 0.5   # paging weights
HARD_DANGLER_COST = 120.0    # a page or line ending on an article/possessive/"of"/"to" (soft break)
EXTRA_WORD_COST = 60.0       # a page of max_words + 1 (only when it still fits)
SHORT_PAGE_COST = 150.0      # x (min_page - on-screen time) / min_page for a page shorter than min_page
NUM_SPLIT_COST = 1000.0      # a page break inside a number list ("80, 90%")
BACKFILL_MAX = 0.4           # min_page never shows a page more than this before its first word
HOLD_GAP = 0.7               # a blank gap shorter than this between pages is filled by the earlier page
HESITATION_GAP = 1.0         # a one-word hesitation page followed within this joins the next page
AUTO_GAP = 4.0               # seconds between auto accents
HOOK_MAX_LINES = 3           # default hook_card.max_lines: re-break up to this many lines
HOOK_SPLIT_GAIN = 1.15       # ...but only when that makes it at least this much bigger
HOOK_KEEP_AT = 0.80          # ...and only when the authored lines set below this share of max_size
HOOK_MIN_RATIO = 1.3         # default hook_card.min_ratio: the card should set >= this x caption size
SRT_LINE, SRT_LINES, SRT_MAX_DUR, SRT_TAIL = 42, 2, 7.0, 0.4   # sidecar subtitle limits
SRT_GAP = 1.5                # a silence this long always ends a subtitle cue
SRT_ORPHAN = 70.0            # cost of a one-word cue in a run of several words
SRT_SPLIT_NUMBER = 80.0      # cost of a cue break inside a number phrase ("12 to | 18 months")
_RANGE_WORDS = {"to", "through", "-", "\u2013", "or", "and"}
_SOFT_DANGLERS = {"at", "in", "on", "for", "with", "from", "by", "about", "into", "and", "but",
                  "or", "so", "that", "because", "if", "when"}   # a cue reads badly ending on one
HARD_JOINS = ("piece", "omit", "dip")   # joins no page or subtitle cue may cross (tighten: may)
BACKING_MODES = ("auto", "always", "never")   # caption.backing.mode and clip caption_backing
FRAME_FUZZ = Fraction(1, 1000)          # frames: a time this close after a frame is on that frame
_HARD_DANGLERS = {"a", "an", "the", "my", "your", "his", "her", "our", "their", "its", "of", "to"}
_HESITATIONS = {"i", "and", "but", "so", "it", "the", "a"}
GLUE_SUFFIXES = ("'s", "'re", "'ll", "'t", "'ve", "'d", "'m", "n't")
ABBREV_RE = re.compile(r"^(?:[A-Za-z]\.){2,}$")
NOT_SENTENCE_END = {"mr.", "mrs.", "ms.", "dr.", "st.", "vs.", "e.g.", "i.e.", "jr.", "sr.",
                    "prof.", "approx."}
CLAUSE_END = (",", ";", ":")
SENTENCE_END = (".", "?", "!")
CLOSERS = "\"'”’)]}"
EDGE_PUNCT = "\"'“”‘’()[]{}!?;:,.…"
CURRENCY = "$£€"
NUM_RE = re.compile(r"^([$£€])?(\d+(?:,\d{3})*(?:\.\d+)?|\.\d+)(k|mm|m|bn|b|t)?(%|x)?$")
NUM_MULT = {"k": 1e3, "m": 1e6, "mm": 1e6, "b": 1e9, "bn": 1e9, "t": 1e12}
NUM_UNITS = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
             "seven": 7, "eight": 8, "nine": 9}
NUM_TEENS = {"ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
             "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19}
NUM_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
            "eighty": 80, "ninety": 90}
NUM_SCALES = {"thousand": 1e3, "grand": 1e3, "million": 1e6, "billion": 1e9, "trillion": 1e12}

# Private word lists (used alone, or unioned with hookcutter.textnorm's lists when present).
_STOPWORDS = set("""
a an the and or but nor if then else so as at by for from in into of off on onto out over
under up down to with within without about above after again against before behind below
between during through until upon via per than too very just also only even still yet
i me my mine myself we us our ours ourselves you your yours yourself yourselves he him his
himself she her hers herself it its itself they them their theirs themselves this that these
those there here what which who whom whose when where why how all any both each either
neither few many more most much other others some such own same is am are was were be been
being do does did doing have has had having will would shall should can could may might must
ought get got gets getting go goes going gone went let lets thats theres heres whats whos
its im ive ill id youre youve youll youd hes shes weve were well wed theyre theyve theyll
theyd dont s t ll re ve d m o y like okay ok yeah yes oh um uh hmm mm gonna wanna gotta kind
sort lot lots thing things one ones way stuff
""".split())
_NEGATIONS = set("""
no not never nothing nobody none nowhere nor neither cannot cant dont doesnt didnt wont
wouldnt shouldnt couldnt isnt arent wasnt werent havent hasnt hadnt aint mustnt neednt
""".split())
_HEDGES = set("""
actually basically literally really honestly probably maybe perhaps kinda sorta pretty quite
somewhat definitely obviously certainly totally absolutely simply essentially generally
typically usually sometimes anyway anyways seriously clearly frankly truly genuinely
""".split())
_GENERIC = set("""
people person something someone somebody everything everyone everybody anything anyone
anybody nothing because through really little thinking saying talking things whatever
believe believed thought thinks knowing wanted wanting needed needing feeling saying
getting making having taking coming looking trying should would could always another
""".split())
# Words a page should not end on when the break is only for length (subtitle convention).
_DANGLERS = set("""
a an the and but or nor so because if that than when while as of to in on at for with from
by into onto about my your his her our their its i we they he she is are was were am be been
have has had will would can could should shall may might must do does did very really just
im ive id ill youre youve youd youll were weve wed theyre theyve theyd theyll hes shes thats
theres whats its
""".split())


# ----------------------------------------------------------------------------- utilities

def _deep_merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _clean_defaults() -> dict:
    import json
    try:
        return json.loads((PRESET_DIR / "clean.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def resolve_style(preset: dict, fmt: str) -> dict:
    """Preset with clean.json filling any missing keys and landscape overrides applied once.

    A preset from presets.for_format(..., "landscape") (marked `_format`) already carries its
    own landscape block; clean's landscape values only fill keys it lacks. A raw preset used for
    landscape gets clean's landscape block, then its own base, then its own landscape block.
    """
    preset = preset or {}
    clean = _clean_defaults()
    clean_wo = {k: v for k, v in clean.items() if k != "landscape"}
    preset_wo = {k: v for k, v in preset.items() if k != "landscape"}
    if preset.get("_format") == "landscape":
        return _deep_merge(_deep_merge(clean_wo, clean.get("landscape") or {}), preset_wo)
    if fmt == "landscape":
        base = _deep_merge(clean_wo, clean.get("landscape") or {})
        return _deep_merge(base, _deep_merge(preset_wo, preset.get("landscape") or {}))
    return _deep_merge(clean_wo, preset_wo)


def _safezone(name, project_root=None) -> dict:
    from . import presets
    return presets.safezone(name, project_root)


def _zone_name(style: dict, clip: dict | None, fmt: str) -> str:
    """The safe-zone profile in force (presets.zone_name, the one resolver QA shares):
    clips.json defaults.safezone when the file set it, else the preset's own, else organic."""
    from . import presets
    return presets.zone_name((clip or {}).get("_eff"), style, fmt)


def render_assets(preset: dict, clip: dict | None, fmt: str, project_root=None) -> dict:
    """What a render draws with that the preset only NAMES, for the render key: the bytes of
    the caption, emphasis and hook-card font files (sha256, "missing" for one that does not
    resolve: the render itself reports it) and the margins of the safe-zone profile in force
    ("unknown" when it cannot be read). Never raises."""
    fonts: dict = {}
    zone: object = "unknown"
    try:
        style = resolve_style(preset, fmt)
        cap, hk = style.get("caption") or {}, style.get("hook_card") or {}
        base = cap.get("font")
        specs = {base, (cap.get("emphasis") or {}).get("font") or base, hk.get("font") or base}
        for spec in sorted(str(s) for s in specs if s):
            try:
                path, _instance = typeset.resolve_font(spec, project_root)
                fonts[spec] = env.sha256_file(path)
            except Exception:
                fonts[spec] = "missing"
        got = _safezone(_zone_name(style, clip, fmt), project_root)
        zone = {k: got[k] for k in ("top", "bottom", "left", "right")}
    except Exception:
        pass
    return {"fonts": fonts, "safezone": zone}


def parse_color(value, default=(255, 255, 255, 255)) -> tuple[int, int, int, int]:
    """'#RRGGBB' or '#RRGGBBAA' (or an RGB/RGBA list) -> (r, g, b, a) ints."""
    if value is None:
        return default
    if isinstance(value, (list, tuple)):
        vals = [int(v) for v in value] + [255]
        return tuple(vals[:4]) if len(value) >= 4 else (vals[0], vals[1], vals[2], 255)
    s = str(value).strip().lstrip("#")
    if len(s) in (3, 4):
        s = "".join(c * 2 for c in s)
    if len(s) not in (6, 8) or not re.fullmatch(r"[0-9a-fA-F]+", s):
        raise env.HCError(f"bad colour {value!r}; use #RRGGBB or #RRGGBBAA")
    r, g, b = int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16)
    a = int(s[6:8], 16) if len(s) == 8 else 255
    return (r, g, b, a)


def _fps(fps) -> tuple[int, int]:
    if isinstance(fps, Fraction):
        return fps.numerator, fps.denominator
    if isinstance(fps, (list, tuple)):
        f = Fraction(int(fps[0]), int(fps[1]) if len(fps) > 1 else 1)
    elif isinstance(fps, str):
        f = Fraction(fps) if "/" in fps else Fraction(fps).limit_denominator(1001)
    else:
        f = Fraction(fps).limit_denominator(1001)
    return f.numerator, f.denominator


def _frac(t: float) -> Fraction:
    """Seconds as an exact Fraction, quantized to the microsecond (kills float fuzz)."""
    return Fraction(int(round(float(t) * 1_000_000)), 1_000_000)


def _frame_ceil(t: float, num: int, den: int) -> int:
    """First frame whose time n*den/num is >= t."""
    return math.ceil(_frac(t) * num / den)


def _frame_snap(t: float, num: int, den: int) -> int:
    """_frame_ceil for a time that is meant to sit ON the frame grid (a join's `out`): rounding
    to the microsecond can leave it a hair past its frame (308/30 s reads 10.266667), which
    plain ceil would push to the next frame."""
    return math.ceil(_frac(t) * num / den - FRAME_FUZZ)


class _Seams:
    """The plan's hard joins (kind piece, omit or dip) on the output frame grid.

    A caption page and a subtitle cue live on one side of every hard join: `span(seg0, seg1)`
    gives the first frame and the end frame (exclusive) of the stretch between the hard joins
    around segments seg0..seg1, `bounds` the same in seconds. Tighten joins are not seams (the
    picture and the sentence run on across them). Join frames come from the segments' frame
    counts (exact); a plan without them falls back to each join's `out` time."""

    def __init__(self, plan: dict, num: int, den: int, frames: int, duration: float):
        self.num, self.den = int(num), int(den)
        self.frames = max(0, int(frames))
        self.duration = float(duration)
        first: dict = {}
        cum = 0
        for n, sg in enumerate(plan.get("segments") or []):
            f = sg.get("frames") if isinstance(sg, dict) else None
            if isinstance(f, bool) or not isinstance(f, int) or f < 0:
                first = {}
                break
            first[int(sg.get("k", n))] = cum
            cum += f
        hard = []
        for j in plan.get("joins") or []:
            if j.get("kind") not in HARD_JOINS or j.get("k") is None:
                continue
            k = int(j["k"])
            f = first.get(k)
            if f is None and j.get("out") is not None:
                f = _frame_snap(float(j["out"]), self.num, self.den)
            if f is not None:
                hard.append((k, max(0, f)))
        hard.sort()
        self.ks = [k for k, _ in hard]
        self._fs = [f for _, f in hard]

    def _time(self, f: int) -> float:
        return float(Fraction(f * self.den, self.num))

    def span(self, seg0: int, seg1: int) -> tuple:
        """(first frame, end frame) of the stretch holding segments seg0..seg1."""
        a = bisect_right(self.ks, int(seg0)) - 1
        b = bisect_right(self.ks, int(seg1))
        lo = self._fs[a] if a >= 0 else 0
        hi = self._fs[b] if b < len(self._fs) else self.frames
        if self.frames:
            lo, hi = min(lo, self.frames), min(hi, self.frames)
        return lo, max(lo, hi)

    def bounds(self, seg0: int, seg1: int) -> tuple:
        """(start, end) seconds of the same stretch; the last one ends with the clip."""
        a = bisect_right(self.ks, int(seg0)) - 1
        b = bisect_right(self.ks, int(seg1))
        lo = self._time(self._fs[a]) if a >= 0 else 0.0
        hi = self._time(self._fs[b]) if b < len(self._fs) else self.duration
        hi = min(hi, self.duration) if self.duration > 0 else hi
        return lo, max(lo, hi)

    def between(self, seg_a: int, seg_b: int) -> bool:
        """A hard join lies after segment seg_a and at or before segment seg_b."""
        return bisect_right(self.ks, int(seg_b)) > bisect_right(self.ks, int(seg_a))

    def of(self, toks: list) -> tuple:
        """bounds() of a run of tokens (first token's first segment, last token's last)."""
        return self.bounds(toks[0].seg0, toks[-1].seg1)


def _even_floor(v: float) -> int:
    return int(math.floor(v / 2.0)) * 2


def _even_ceil(v: float) -> int:
    return int(math.ceil(v / 2.0)) * 2


def _srgb_to_lin(c: float) -> float:
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def _rel_lum(rgb) -> float:
    r, g, b = (_srgb_to_lin(v / 255.0) for v in rgb[:3])
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast_ratio(text_rgb, bg_gray: float, pill_rgba=None) -> float:
    """WCAG contrast of `text_rgb` over a grey background (0..1), optionally under a pill."""
    bg = (bg_gray * 255.0,) * 3
    if pill_rgba is not None:
        a = pill_rgba[3] / 255.0
        bg = tuple(a * pill_rgba[i] + (1 - a) * bg[i] for i in range(3))
    l1, l2 = _rel_lum(text_rgb), _rel_lum(bg)
    hi, lo = max(l1, l2), min(l1, l2)
    return (hi + 0.05) / (lo + 0.05)


def _ease_out(p: float) -> float:
    p = min(1.0, max(0.0, p))
    return 1.0 - (1.0 - p) ** 3


def _pop_scale(dt: float) -> float:
    if dt >= POP_KEYS[-1][0]:
        return 1.0
    for (t0, s0), (t1, s1) in zip(POP_KEYS, POP_KEYS[1:]):
        if dt <= t1:
            return s0 + (s1 - s0) * (dt - t0) / (t1 - t0)
    return 1.0


# ----------------------------------------------------------------------------- word lists

def _key(word: str) -> str:
    return re.sub(r"[^\w]", "", word.lower().replace("’", "'").replace("'", ""))


def _word_lists() -> tuple[set, set, set]:
    """(accent stopwords, negations, auto-pick exclusions), merged with textnorm when present."""
    stop, neg, hedges = set(_STOPWORDS), set(_NEGATIONS), set(_HEDGES) | _GENERIC
    try:
        from . import textnorm
        for name, bucket in (("STOPWORDS", stop), ("NEGATIONS", neg), ("HEDGES", hedges)):
            extra = getattr(textnorm, name, None)
            if isinstance(extra, (set, frozenset, list, tuple)):
                bucket.update(_key(w) for w in extra if isinstance(w, str))
    except ImportError:
        pass
    stop = {w for w in stop if w} - neg
    return stop, neg, hedges


_LISTS = None


def _lists():
    global _LISTS
    if _LISTS is None:
        _LISTS = _word_lists()
    return _LISTS


# ----------------------------------------------------------------------------- phrase matching

def _canon_num(v: float) -> str:
    if abs(v - round(v)) < 1e-9:
        return "#" + str(int(round(v)))
    return "#" + ("%.6g" % v)


def _fold(text: str) -> str:
    """Diacritics folded ("José" -> "Jose"); other characters kept."""
    t = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in t if not unicodedata.combining(ch))


def word_units(text: str) -> list[str]:
    """Normalized match units of one word: lowercase, diacritics folded, apostrophes removed,
    hyphens split, numerals canonical ("$70K" -> "#70000", "80%" -> "#80", "percent")."""
    t = _fold(text).lower().replace("’", "'").replace("‘", "'")
    out = []
    for part in re.split(r"[\s/\-\u2013\u2014]+", t):
        core = part.strip(EDGE_PUNCT)
        if not core:
            continue
        m = NUM_RE.match(core)
        if m:
            v = float(m.group(2).replace(",", "")) * NUM_MULT.get(m.group(3) or "", 1.0)
            out.append(_canon_num(v))
            if m.group(4) == "%":
                out.append("percent")
            elif m.group(4) == "x":
                out.append("times")
            continue
        if core in ("%",):
            out.append("percent")
            continue
        core = re.sub(r"[^\w]", "", core.replace("'", ""))
        if core:
            out.append(core)
    return out


def _num_cat(units: list, j: int):
    k = units[j][0]
    if k.startswith("#"):
        return "digit", float(k[1:])
    if k in NUM_UNITS:
        return "unit", NUM_UNITS[k]
    if k in NUM_TEENS:
        return "teen", NUM_TEENS[k]
    if k in NUM_TENS:
        return "ten", NUM_TENS[k]
    if k == "hundred":
        return "hundred", 100
    if k in NUM_SCALES:
        return "scale", NUM_SCALES[k]
    if k == "a" and j + 1 < len(units) and (units[j + 1][0] == "hundred"
                                            or units[j + 1][0] in NUM_SCALES):
        return "unit", 1
    return None, None


def _merge_numbers(units: list) -> list:
    """Collapse spoken number runs ("seventy thousand", "two and a half") into one unit
    ("#70000", "#2.5")."""
    out, i = [], 0
    while i < len(units):
        total, cur, last, j = 0.0, None, None, i
        while j < len(units):
            cat, v = _num_cat(units, j)
            if cat is None:
                if (units[j][0] == "and" and last == "hundred" and j + 1 < len(units)
                        and _num_cat(units, j + 1)[0] in ("unit", "teen", "ten")):
                    j += 1
                    continue
                if (units[j][0] == "and" and last in ("unit", "teen", "ten", "digit")
                        and j + 2 < len(units) and units[j + 1][0] == "a"
                        and units[j + 2][0] == "half"):
                    cur = (cur or 0) + 0.5           # "one and a half (million)"
                    last = "half"
                    j += 3
                    continue
                break
            if cat == "digit":
                if last is not None:
                    break
                cur = v
            elif cat in ("unit", "teen", "ten"):
                if last in (None, "scale", "hundred"):
                    cur = (cur or 0) + v
                elif last == "ten" and cat == "unit":
                    cur += v
                else:
                    break
            elif cat == "hundred":
                if last in ("unit", "teen", "digit"):
                    cur = (cur or 0) * 100
                else:
                    break
            elif cat == "scale":
                if last in ("unit", "teen", "ten", "hundred", "digit", "half"):
                    total += (cur or 0) * v
                    cur = 0
                else:
                    break
            last = cat
            j += 1
        if j > i and not (j == i + 1 and units[i][0].startswith("#")):
            toks = frozenset().union(*(u[1] for u in units[i:j]))
            out.append((_canon_num(total + (cur or 0)), toks))
            i = j
        else:
            out.append(units[i])
            i += 1
    return out


def phrase_units(phrase: str) -> list[str]:
    units = [(u, frozenset()) for w in phrase.split() for u in word_units(w)]
    return [u for u, _ in _merge_numbers(units)]


def match_phrases(phrases: list[str], words: list[str]) -> list[list[list[int]]]:
    """For each phrase, the list of matches; each match = sorted word positions it covers.

    Matching is on normalized units with stopwords removed from both sides (negations are
    content), so "*an afternoon*" marks only "afternoon" and "$70K" matches "seventy
    thousand". A phrase made only of stopwords is matched literally.
    """
    stop = _lists()[0]
    stream = [(u, frozenset([pos])) for pos, w in enumerate(words) for u in word_units(w)]
    stream = _merge_numbers(stream)
    content = [(u, toks, k) for k, (u, toks) in enumerate(stream) if u not in stop]
    full = [(u, toks, k) for k, (u, toks) in enumerate(stream)]
    result = []
    for phrase in phrases:
        units = phrase_units(phrase)
        target = [u for u in units if u not in stop]
        seq = content
        if not target:
            target, seq = units, full
        found = []
        if target:
            n = len(target)
            for start in range(len(seq) - n + 1):
                if not all(seq[start + j][0] == target[j] for j in range(n)):
                    continue
                # "SELL FIRST" must not match "sell it to them first": at most 2 skipped
                # stopwords between matched units
                if any(seq[start + j + 1][2] - seq[start + j][2] - 1 > 2 for j in range(n - 1)):
                    continue
                found.append(sorted(set().union(*(seq[start + j][1] for j in range(n)))))
        result.append(found)
    return result


def parse_accent_runs(line: str) -> list[tuple[str, bool]]:
    """'AFRAID OF *LOOKING STUPID*' -> [('AFRAID OF ', False), ('LOOKING STUPID', True)]."""
    runs, buf, accent = [], [], False
    for ch in line:
        if ch == "*":
            if buf:
                runs.append(("".join(buf), accent))
                buf = []
            accent = not accent
        else:
            buf.append(ch)
    if buf:
        runs.append(("".join(buf), accent))
    return runs


def accent_phrases(lines: list[str]) -> list[str]:
    out = []
    for line in lines or []:
        for text, acc in parse_accent_runs(line):
            if acc and text.strip():
                out.append(text.strip())
    return out


def hook_lines(hook: dict | None) -> list[str]:
    if not hook:
        return []
    lines = hook.get("lines")
    if lines is None and hook.get("text"):
        lines = re.split(r"\\N|\n", str(hook["text"]))
    return [str(x) for x in (lines or []) if str(x).strip()]


def hook_duration(hook: dict, lines: list[str]) -> float:
    dur = hook.get("dur", "auto")
    if isinstance(dur, (int, float)):
        return float(dur)
    chars = sum(len(ln.replace("*", "")) for ln in lines)
    return min(3.0, max(1.5, 0.5 + chars / 20.0))


# ----------------------------------------------------------------------------- tokens

@dataclass
class Token:
    natural: str               # glued text as spoken (natural case, punctuation kept)
    text: str                  # display text (case applied, strip_punct removed)
    idx: list
    s: float
    e: float
    sp: object = None
    piece: object = None
    seg0: int = 0
    seg1: int = 0
    sentence_start: bool = False
    emph: str | None = None    # "hook" | "manual" | "auto"
    run: tuple | None = None   # identity of the emphasis run this token belongs to
    phrase: int | None = None  # hook accent phrase index


def _glues_to_prev(text: str, prev: str | None = None) -> bool:
    t = text.replace("’", "'")
    if not t:
        return False
    if len(t) >= 2 and t[0] == "." and t[1].isalnum():
        # ".5" continues a number split by ASR ("3" + ".5" -> "3.5"); after anything else a
        # word that starts with a dot is a word of its own (".5%", ".22", ".com", ".NET").
        return t[1].isdigit() and bool(prev) and prev.rstrip(CLOSERS)[-1:].isdigit()
    if t[0] in ",.%":
        return True
    if t.lower() in GLUE_SUFFIXES:
        return True
    if len(t) >= 2 and t[0] in "-–" and t[1].isdigit():
        # a number range split by ASR: "80" + "-90" -> "80-90"
        return bool(prev) and prev.rstrip(CLOSERS)[-1:].isdigit()
    return len(t) >= 2 and t[0] == "-" and t[1].islower()


def _is_number(raw: str) -> bool:
    units = word_units(raw)
    return bool(units) and units[0].startswith("#")


def _number_list(a: str, b: str) -> bool:
    """`a` is a number ending in a comma and `b` a number: "80," + "90%" (a range or list)."""
    return a.rstrip(CLOSERS).endswith(",") and _is_number(a) and _is_number(b)


def _ends_sentence(raw: str) -> bool:
    t = raw.rstrip(CLOSERS)
    if not t.endswith(SENTENCE_END):
        return False
    low = t.lower()
    if low in NOT_SENTENCE_END or ABBREV_RE.match(t) or re.fullmatch(r"[A-Z]\.", t):
        return False
    return True


def _ends_clause(raw: str) -> bool:
    return raw.rstrip(CLOSERS).endswith(CLAUSE_END)


def _strip_display(text: str, strip: str) -> str:
    """Strip `strip` chars from both ends, keeping the dots of an abbreviation ("U.S.,") and
    a leading dot that belongs to the word (".5%" is not "5%"; ".22", ".com", ".NET")."""
    text = text.strip()
    if not strip:
        return text
    if "." in strip:
        core = text.strip(strip.replace(".", ""))
        if ABBREV_RE.match(core):
            return core
        out = text.strip(strip)
        if out and len(core) >= 2 and core[0] == "." and core[1].isalnum():
            return "." + out
        return out
    return text.strip(strip)


def _closing_mark(raw: str) -> str:
    """The sentence or clause mark a word ends on ("deck." -> ".", "AI,\"" -> ","), else ""."""
    if not (_ends_sentence(raw) or _ends_clause(raw)):
        return ""
    m = re.search(r"[.?!,;:]+$", raw.rstrip(CLOSERS))
    return m.group(0) if m else ""


def _take_mark(stem: str, mark: str) -> str:
    """`stem` closed with `mark`, unless it is empty or already ends in a sentence or clause
    mark of its own (closing quotes and brackets stay outside the mark)."""
    core = stem.rstrip(CLOSERS)
    if not mark or not core or core.endswith(SENTENCE_END + CLAUSE_END):
        return stem
    return core + mark + stem[len(core):]


def _lone(cur: dict) -> dict:
    return {"text": cur["text"], "idx": cur["idx"], "s": cur["s"], "e": cur["e"], "sp": cur["sp"],
            "piece": cur["piece"], "seg0": cur["seg"], "seg1": cur["seg"]}


def display_tokens(plan: dict, clip: dict | None, caption: dict,
                   issues: list | None = None) -> list[Token]:
    """Visible plan words -> glued, cased, stripped display tokens in output order.

    Zero-width characters are removed (each removal is appended to `issues` as
    ("glyph_stripped", message, out_time) when a list is given).

    An inserted word (plan word with `i` null and `ins` set: speech the transcript missed,
    typed by the editor with the clip key `insert`) is a token like any other, with an empty
    `idx`: it pages, reveals and times normally, and `fix` / `emphasis` (which address
    transcript indices) never touch it.

    A display merge (`fix` {"4": "back deck", "5": ""}): the hidden word is still heard, so
    the token before it, when the editor retyped that token's own word, lasts until the hidden
    word ends and takes its closing mark ("deck." gives "back deck."). Forward only: a word
    hidden BEFORE its carrier ({"4": "", "5": "both"}) never pulls the carrier's start back."""
    clip = clip or {}
    fix = {str(k): v for k, v in (clip.get("fix") or {}).items()}
    hard_ks = sorted(int(j["k"]) for j in plan.get("joins") or []
                     if j.get("kind") in HARD_JOINS and j.get("k") is not None)
    rows = []
    last_i = -1
    for n, w in enumerate(plan.get("words") or []):
        i = w.get("i", 0)
        inserted = i is None
        # Words that start together keep transcript order; an inserted word sorts right after
        # the transcript word it follows in the plan's word list.
        order = last_i + 0.5 if inserted else int(i)
        if not inserted:
            last_i = int(i)
        if w.get("removed") or w.get("omitted") or w.get("filler"):
            continue
        if w.get("s_out") is None or w.get("e_out") is None:
            continue
        if w.get("hidden"):
            # Hidden with fix "" inside the piece: heard, not shown. A row with no text, which
            # only the retyped token before it can absorb (see the merge loop).
            if not w.get("edge") and not inserted:
                rows.append((float(w["s_out"]), order, n, w, None))
            continue
        text = w.get("text")
        if text is None:
            raw = w.get("raw", w.get("w", ""))
            text = raw if inserted else fix.get(str(i), raw)
        text, gone = typeset.strip_invisible(str(text))
        text = text.strip()
        if gone and issues is not None:
            what = "an inserted word" if inserted else "word #%s" % i
            issues.append(("glyph_stripped", "removed invisible %s from %s %r"
                           % (typeset.describe_chars(gone), what, text), float(w["s_out"])))
        if not text:
            continue
        rows.append((float(w["s_out"]), order, n, w, text))
    rows.sort(key=lambda r: r[:3])

    merged: list[dict] = []
    pending_currency = None
    for s, _, _, w, text in rows:
        own = [] if w.get("i", 0) is None else [int(w.get("i", 0))]
        seg = int(w.get("seg") or 0)
        e = max(float(w["e_out"]), s)
        if text is None:
            prev = merged[-1] if merged else None
            if (pending_currency is None and prev is not None and prev.get("retyped")
                    and prev["piece"] == w.get("piece")
                    and not any(prev["seg1"] < k <= seg for k in hard_ks)):
                prev["e"] = max(prev["e"], e)
                prev["seg1"] = max(prev["seg1"], seg)
                stem = prev.setdefault("stem", prev["text"])
                prev["text"] = _take_mark(stem, _closing_mark(str(w.get("raw") or "")))
            continue
        retyped = bool(own) and bool(str(fix.get(str(own[0])) or "").strip())
        if pending_currency is not None and pending_currency["piece"] != w.get("piece"):
            merged.append(_lone(pending_currency))          # never glue "$" across a cut
            pending_currency = None
        glued = pending_currency is not None
        if glued:
            text = pending_currency["text"] + text
            s = pending_currency["s"]
            idx = pending_currency["idx"] + own
            pending_currency = None
        else:
            idx = own
        prev = merged[-1] if merged else None
        if (prev is not None and _glues_to_prev(text, prev["text"]) and prev["piece"] == w.get("piece")
                and not glued):
            prev["text"] += text
            prev["idx"] += idx
            prev["e"] = max(prev["e"], e)
            prev["retyped"] = retyped       # the token's latest word decides (display merges)
            prev.pop("stem", None)
            if not any(prev["seg1"] < k <= seg for k in hard_ks):
                prev["seg1"] = seg      # glued across an omit join: the token stays before it
            continue
        if text in CURRENCY:
            pending_currency = {"text": text, "s": s, "e": e, "idx": idx, "sp": w.get("sp"),
                                "piece": w.get("piece"), "seg": seg}
            continue
        merged.append({"text": text, "idx": idx, "s": s, "e": e, "sp": w.get("sp"),
                       "piece": w.get("piece"), "seg0": seg, "seg1": seg, "retyped": retyped})
    if pending_currency is not None:          # a dangling "$" at the very end: show it alone
        merged.append(_lone(pending_currency))

    case = (caption.get("case") or "upper").lower()
    strip = caption.get("strip_punct") or ""
    tokens: list[Token] = []
    sentence_start = True
    for k, m in enumerate(merged):
        disp = m["text"]
        if case == "upper":
            disp = disp.upper()
        elif case == "lower":
            disp = disp.lower()
        cut = strip
        nxt = merged[k + 1] if k + 1 < len(merged) else None
        if "," in cut and nxt is not None and m["text"].rstrip(CLOSERS).endswith(","):
            # Keep the comma where dropping it misleads: a repeated word reads as a stammer
            # ("WITH, WITH THE") and a number list as one number ("80, 90%").
            if (_key(m["text"]) and _key(m["text"]) == _key(nxt["text"])) or \
                    _number_list(m["text"], nxt["text"]):
                cut = cut.replace(",", "")
        disp = _strip_display(disp, cut)
        if disp:
            tokens.append(Token(natural=m["text"], text=disp, idx=m["idx"], s=m["s"], e=m["e"],
                                sp=m["sp"], piece=m["piece"], seg0=m["seg0"], seg1=m["seg1"],
                                sentence_start=sentence_start))
        elif tokens:
            # Pure punctuation (for example a lone "..."): keep its paging meaning.
            tokens[-1].natural += m["text"]
        sentence_start = _ends_sentence(m["text"])
    return tokens


# ----------------------------------------------------------------------------- typesetting

class _Part:
    """One space-separated piece of a display token whose text holds spaces (a display merge
    made with `fix`, such as "Started dating"). The token stays one unit for paging, timing
    and emphasis; its parts may sit on different lines of the page."""
    __slots__ = ("tok", "k", "text", "natural")

    def __init__(self, tok: Token, k: int, text: str, natural: str):
        self.tok, self.k, self.text, self.natural = tok, k, text, natural

    @property
    def emph(self):
        return self.tok.emph


def _line_runs(units: list) -> list[tuple]:
    """A line of wrap units (tokens and parts) as [(token, text)] runs: neighbouring parts of
    one token are one run again ("STARTED" + "DATING" -> "STARTED DATING"), so a merged token
    that stays whole on a line is measured and drawn exactly as one word."""
    out = []
    last_k = None
    for u in units:
        if isinstance(u, _Part):
            if out and out[-1][0] is u.tok and last_k == u.k - 1:
                out[-1] = (u.tok, out[-1][1] + " " + u.text)
            else:
                out.append((u.tok, u.text))
            last_k = u.k
        else:
            out.append((u, u.text))
            last_k = None
    return out


class _Setter:
    """Measures tokens with the caption font (and the emphasis font for emphasis tokens)."""

    def __init__(self, cap: dict, project_root):
        self.cap = cap
        self.project_root = project_root
        self.spec = cap["font"]
        emph = cap.get("emphasis") or {}
        self.emph_spec = emph.get("font") or self.spec
        self.mult = float(emph.get("size_mult") or 1.0)
        self.tracking = float(cap.get("tracking") or 0)
        self.size = int(cap["size"])
        lay = cap["layout"]
        self.max_chars = int(lay.get("max_chars") or 10 ** 6)
        self.max_lines = int(lay.get("max_lines") or 2)
        self.line_gap = float(lay.get("line_gap") or 1.0)
        self.case = (cap.get("case") or "upper").lower()
        self._wraps: dict = {}
        self._parts: dict = {}

    def font(self, size: int, emph: bool = False):
        if emph:
            return typeset.load_font(self.emph_spec, max(1, round(size * self.mult)), self.project_root)
        return typeset.load_font(self.spec, size, self.project_root)

    def width(self, tok: Token, size: int) -> float:
        return typeset.text_width(self.font(size, bool(tok.emph)), tok.text, self.tracking)

    def run_width(self, tok: Token, text: str, size: int) -> float:
        return typeset.text_width(self.font(size, bool(tok.emph)), text, self.tracking)

    def space(self, size: int) -> float:
        f = self.font(size)
        return typeset.text_width(f, " ") + typeset.tracking_px(f, self.tracking)

    def measure(self, toks: list, size: int) -> float:
        """Line width in px; inf when a multi-word line breaks max_chars. `toks` may mix
        tokens and parts of tokens (see _Part)."""
        runs = _line_runs(toks)
        if len(toks) > 1 and sum(len(text) for _, text in runs) + len(runs) - 1 > self.max_chars:
            return math.inf
        return (sum(self.run_width(tok, text, size) for tok, text in runs)
                + self.space(size) * (len(runs) - 1))

    def parts(self, tok: Token) -> list:
        """The token itself, or its space-separated parts when its text holds spaces."""
        key = (id(tok), tok.text)
        hit = self._parts.get(key)
        if hit is None:
            texts = tok.text.split()
            if len(texts) < 2:
                hit = [tok]
            else:
                nat = tok.natural.split()
                if len(nat) != len(texts):
                    nat = texts
                hit = [_Part(tok, k, t, n) for k, (t, n) in enumerate(zip(texts, nat))]
            self._parts[key] = hit
        return hit

    def wrap(self, toks: list[Token], size: int, max_width: float):
        """Balanced lines of `toks` (None when they do not fit), avoiding line ends on a hard
        dangler ("THE / PRODUCT") or inside a number list. Cached per token run and size.

        Tokens are set whole when that fits. When it does not and a token's text holds spaces
        (a display merge made with `fix`), the token may wrap at them: the lines then hold
        _Part units for that token ("STARTED" / "DATING MY WIFE"), so a long merge takes two
        lines at full size instead of being shrunk as one unbreakable word."""
        key = (tuple((id(t), t.emph, t.text) for t in toks), size, round(max_width, 3))
        if key not in self._wraps:
            def measure(ts):
                return self.measure(ts, size)
            lines = typeset.balanced_wrap(toks, measure, max_width, self.max_lines,
                                          bad_break=_bad_break)
            if lines is None:
                units = [u for t in toks for u in self.parts(t)]
                if len(units) > len(toks):
                    lines = typeset.balanced_wrap(units, measure, max_width, self.max_lines,
                                                  bad_break=_bad_break)
            self._wraps[key] = lines
        return self._wraps[key]


def _bad_break(a: Token, b: Token) -> bool:
    """A line or page should not end between `a` and `b` (soft breaks only)."""
    if _ends_clause(a.natural) and not _number_list(a.natural, b.natural):
        return False
    return _key(a.natural) in _HARD_DANGLERS or _number_list(a.natural, b.natural)


# ----------------------------------------------------------------------------- pages

@dataclass
class Page:
    tokens: list
    brk: str = "start"                    # why the page started: start | hard:<why> | soft:<why>
    start: float = 0.0
    end: float = 0.0
    size: int = 0
    lines: list = field(default_factory=list)       # lists of token positions
    line_texts: list = field(default_factory=list)  # the text runs of each line, as drawn
    pos: list = field(default_factory=list)         # (x, baseline) per token, frame coords
    runs: list = field(default_factory=list)        # per token: [(text, x, baseline)] as drawn
    box: tuple = (0, 0, 0, 0)                       # text box x, y, w, h (frame coords)
    typo: tuple = (0, 0, 0, 0)                      # typographic box (cap top to baseline)
    line_typos: list = field(default_factory=list)  # typographic box of lines 0..k
    backing: bool = False
    pill: tuple | None = None                       # build reveal: the pill of the whole page
    token_typos: list = field(default_factory=list)  # typographic box of tokens 0..k (reading order)
    token_pills: list = field(default_factory=list)  # pill that has grown to token k (build reveal)
    p90: float | None = None
    edges: float | None = None                      # share of edge pixels behind the box
    contrast: float | None = None                   # shadow-aware when bare, with the pill when backed
    need: bool = False                              # this page alone calls for a pill
    reason: str | None = None                       # contrast | texture | shot | always
    forced: bool = False
    start_f: int = 0
    end_f: int = 0
    lo: float = 0.0                                 # the stretch between hard joins the page
    hi: float = math.inf                            # lives in (seconds; see _Seams)


def _hard_break(prev: Token, tok: Token, lay: dict, join_ks: list) -> str | None:
    """Breaks that always end a page: speaker change, plan join, pause, sentence end."""
    if prev.sp is not None and tok.sp is not None and prev.sp != tok.sp:
        return "speaker"
    if any(prev.seg1 < k <= tok.seg0 for k in join_ks):
        return "join"
    if tok.s - prev.e > float(lay.get("pause_break", 0.45)):
        return "pause"
    if _ends_sentence(prev.natural):
        return "sentence"
    return None


def _clause_at(run: list[Token], q: int) -> bool:
    """Token q ends a clause (,;:), except a comma inside a number list ("80, 90%")."""
    if not _ends_clause(run[q].natural):
        return False
    return not (q + 1 < len(run) and _number_list(run[q].natural, run[q + 1].natural))


def _page_run(run: list[Token], setter: _Setter, max_width: float, lay: dict, *,
              lead: float = 0.0, next_s: float | None = None,
              bounds: tuple | None = None) -> list[tuple]:
    """Optimal paging of one run (no hard break inside), by dynamic programming.

    Costs: one PAGE_COST per page (fewest pages first), a page of max_words + 1 words costs
    EXTRA_WORD_COST, one-word orphans, a soft page end on a dangling function word (heavier for
    articles, possessives, "of" and "to", also at a line end inside the page), uneven page
    lengths, and on-screen time below min_page (the page shows from first.s - lead until
    min(last.e + PAGE_TAIL, next.s - lead); `next_s` is the start of the token after this run;
    `bounds` are the seams around the run, which no page outlasts: see _Seams).
    Clauses: a page may hold a clause end (,;:) only when fewer than comma_min_words of its
    words come before it AND that clause is whole on the page (the page starts the run or
    follows another clause end); "WELL, THEY GET" is fine, "MET, NOT YOUR" is not. A page
    break inside a number list ("80, 90%") is avoided unless nothing else fits.
    """
    max_words = max(1, int(lay.get("max_words") or 4))
    cmin = int(lay.get("comma_min_words", 2))
    min_page = float(lay.get("min_page") or 0.0)
    m = len(run)
    best = [math.inf] * (m + 1)
    back = [0] * (m + 1)
    best[0] = 0.0
    clause = [_clause_at(run, q) for q in range(m)]

    def clause_ok(j: int, i: int) -> bool:
        whole = j == 0 or clause[j - 1]
        for q in range(j, i - 1):
            if clause[q] and (q - j + 1 >= cmin or not whole):
                return False
        return True

    for i in range(1, m + 1):
        for j in range(max(0, i - max_words - 1), i):
            if best[j] == math.inf or not clause_ok(j, i):
                continue
            page = run[j:i]
            n = i - j
            lines = setter.wrap(page, setter.size, max_width) if n > 1 else [page]
            if lines is None:
                continue
            cost = best[j] + PAGE_COST + BALANCE_COST * max(0, max_words - n) ** 2
            if n > max_words:
                cost += EXTRA_WORD_COST
            if n == 1 and m > 1:
                cost += ORPHAN_COST
            last = run[i - 1].natural
            if i < m and not clause[i - 1]:
                if _number_list(last, run[i].natural):
                    cost += NUM_SPLIT_COST
                elif _key(last) in _HARD_DANGLERS:
                    cost += HARD_DANGLER_COST
                elif _key(last) in _DANGLERS:
                    cost += DANGLER_COST
            for line, nxt in zip(lines, lines[1:]):
                if _bad_break(line[-1], nxt[0]):
                    cost += HARD_DANGLER_COST
            if min_page > 0:
                end = run[i - 1].e + PAGE_TAIL
                nxt_s = run[i].s if i < m else next_s
                if nxt_s is not None:
                    end = min(end, nxt_s - lead)
                begin = run[j].s - lead
                if bounds is not None:
                    begin, end = max(begin, bounds[0]), min(end, bounds[1])
                dur = end - begin
                if dur < min_page:
                    cost += SHORT_PAGE_COST * (min_page - dur) / min_page
            if cost < best[i] - 1e-9:
                best[i], back[i] = cost, j
    cuts, i = [], m
    while i > 0:
        cuts.append((back[i], i))
        i = back[i]
    cuts.reverse()
    out = []
    for k, (a, b) in enumerate(cuts):
        why = "start"
        if k:
            why = "hard:comma" if clause[a - 1] else "soft:fit"
        out.append((run[a:b], why))
    return out


def paginate(tokens: list[Token], setter: _Setter, max_width: float, lay: dict,
             join_ks: list, lead: float = 0.0, seams: _Seams | None = None) -> list[Page]:
    """Split tokens into runs at hard breaks (speaker, join, pause, sentence), then page each
    run optimally (see _page_run). Page.brk records why each page starts. With `seams` the
    paging counts a page's on-screen time only up to the seams around its run."""
    runs: list[tuple] = []
    cur: list[Token] = []
    brk = "start"
    for tok in tokens:
        if cur:
            hard = _hard_break(cur[-1], tok, lay, join_ks)
            if hard:
                runs.append((cur, brk))
                cur, brk = [], f"hard:{hard}"
        cur.append(tok)
    if cur:
        runs.append((cur, brk))
    pages = []
    for r, (run, run_brk) in enumerate(runs):
        next_s = runs[r + 1][0][0].s if r + 1 < len(runs) else None
        bounds = seams.of(run) if seams is not None else None
        for k, (toks, why) in enumerate(_page_run(run, setter, max_width, lay, lead=lead,
                                                  next_s=next_s, bounds=bounds)):
            pages.append(Page(tokens=toks, brk=run_brk if k == 0 else why))
    return pages


def _seam_between(a: Page, b: Page, join_ks: list) -> bool:
    """A cut (piece, omit or dip join) or a speaker change lies between pages a and b."""
    x, y = a.tokens[-1], b.tokens[0]
    if x.sp is not None and y.sp is not None and x.sp != y.sp:
        return True
    return any(x.seg1 < k <= y.seg0 for k in join_ks)


def _clauses_ok(toks: list[Token], whole: bool, cmin: int) -> bool:
    """The clause rule of _page_run for one candidate page: a clause end inside the page needs
    fewer than `cmin` words before it, and the clause must be whole (`whole`: the page starts a
    run or follows a clause end)."""
    for q in range(len(toks) - 1):
        if _clause_at(toks, q) and (q + 1 >= cmin or not whole):
            return False
    return True


def _starts_whole(p: Page) -> bool:
    """Page p starts a run or follows a clause end (see _clauses_ok)."""
    return not p.brk.startswith("soft:")


def _merge_ok(a: Page, b: Page, setter: _Setter, max_width: float, limit: int,
              cmin: int = 2) -> bool:
    """May short page `a` and its successor `b` share one page (at most `limit` words)?

    Never across a seam, a speaker change or a pause. Never across a sentence end either,
    except when one side is a single word and the merged page is itself whole sentences
    ("Run." + "No." -> "RUN NO"): "IT WORKS PEOPLE SAY" and "DONE THE / PRODUCT CAME" read
    as run-on captions. Across a clause end the merged page must pass the clause rule (a clause
    tail never leads the next clause: "MET, NOT YOUR") or be whole clauses from end to end
    ("PEOPLE SAY, I CAN'T DO THAT.")."""
    if b.brk in ("hard:join", "hard:speaker", "hard:pause"):
        return False
    last_a, last_b = a.tokens[-1].natural, b.tokens[-1].natural
    if _ends_sentence(last_a):
        single = len(a.tokens) == 1 or len(b.tokens) == 1
        if not (single and a.tokens[0].sentence_start and _ends_sentence(last_b)):
            return False
    toks = a.tokens + b.tokens
    if not _ends_sentence(last_a) and not _clauses_ok(toks, _starts_whole(a), cmin):
        whole = _starts_whole(a) and (_ends_clause(last_b) or _ends_sentence(last_b))
        if not whole:
            return False
    return len(toks) <= limit and setter.wrap(toks, setter.size, max_width) is not None


def schedule(pages: list[Page], setter: _Setter, max_width: float, lay: dict, lead: float,
             hook_dur: float, duration: float, join_ks: list | None = None,
             seams: _Seams | None = None) -> list[Page]:
    """Display spans and min_page.

    1. Each page shows [first.s - lead, min(last.e + PAGE_TAIL, next page start)]; pages
       starting under the hook card start when it ends (and are dropped below HOOK_MIN_REMAIN).
    2. A blank gap under HOLD_GAP between two pages is filled by the earlier page (no blink),
       except across a seam.
    3. A one-word hesitation page ("I", "AND", "SO", ...) followed within HESITATION_GAP joins
       the next page when it fits.
    4. A page under min_page grows into free time on either side (forward first; backward
       never to more than BACKFILL_MAX before its first word, which stays well inside what QA
       G10 calls an early page), else merges with a neighbour (up to max_words + 2 words,
       still fitting max_lines and the width; see _merge_ok). min_page is a target: pages
       still under it are reported by build_track.

    With `seams` (see _Seams) every step is clamped to the page's own stretch between hard
    joins: the lead before the first word, the tail after the last, the hold and the min_page
    growth in both directions all stop at a piece, omit or dip join, so a page never shows over
    the picture on the other side of a cut. (Without it only steps 2 to 4 respect the joins.)
    """
    join_ks = join_ks or []
    max_words = int(lay.get("max_words") or 4)
    limit = max_words + 2
    cmin = int(lay.get("comma_min_words", 2))
    min_page = float(lay.get("min_page") or 0.0)

    def bounds(p):
        if seams is None:
            return 0.0, duration
        lo, hi = seams.of(p.tokens)
        return lo, min(hi, duration)

    def spans(ps):
        for p in ps:
            p.start = max(0.0, bounds(p)[0], p.tokens[0].s - lead)
        for i, p in enumerate(ps):
            nxt = ps[i + 1].start if i + 1 < len(ps) else duration
            p.end = min(p.tokens[-1].e + PAGE_TAIL, nxt, duration, bounds(p)[1])

    spans(pages)
    if hook_dur > 0:
        kept = []
        for p in pages:
            if p.start < hook_dur:
                p.start = hook_dur
                if p.end - p.start < HOOK_MIN_REMAIN:
                    continue                    # spoken under the hook card
            kept.append(p)
        pages = kept
    floor = hook_dur if hook_dur > 0 else 0.0

    def hold(ps):
        for p, q in zip(ps, ps[1:]):
            if 0 < q.start - p.end < HOLD_GAP and not _seam_between(p, q, join_ks):
                p.end = max(p.end, min(q.start, bounds(p)[1]))

    hold(pages)
    i = 0
    while i < len(pages) - 1:
        p, q = pages[i], pages[i + 1]
        tok = p.tokens[0]
        if (len(p.tokens) == 1 and _key(tok.natural) in _HESITATIONS
                and not _ends_sentence(tok.natural) and q.tokens[0].s - tok.e <= HESITATION_GAP
                and not _seam_between(p, q, join_ks) and len(q.tokens) + 1 <= limit
                and _clauses_ok(p.tokens + q.tokens, _starts_whole(p), cmin)
                and setter.wrap(p.tokens + q.tokens, setter.size, max_width) is not None):
            q.tokens = p.tokens + q.tokens
            q.start, q.brk = p.start, p.brk
            del pages[i]
            continue
        i += 1

    changed = True
    while changed:
        changed = False
        for i, p in enumerate(pages):
            need = min_page - (p.end - p.start)
            if need <= 1e-9:
                continue
            lo, hi = bounds(p)
            nxt = min(pages[i + 1].start if i + 1 < len(pages) else duration, hi)
            grow = max(0.0, min(need, nxt - p.end))
            p.end += grow
            need -= grow
            if need > 1e-9:
                prev_end = pages[i - 1].end if i > 0 else floor
                early = p.tokens[0].s - BACKFILL_MAX
                back = max(0.0, min(need, p.start - max(prev_end, floor, lo, early)))
                p.start -= back
                need -= back
            if need > 1e-9:
                if (i > 0 and not _seam_between(pages[i - 1], p, join_ks)
                        and _merge_ok(pages[i - 1], p, setter, max_width, limit, cmin)):
                    a = pages[i - 1]
                    a.tokens = a.tokens + p.tokens
                    a.end = max(a.end, p.end)
                    del pages[i]
                    changed = True
                    break
                if (i + 1 < len(pages) and not _seam_between(p, pages[i + 1], join_ks)
                        and _merge_ok(p, pages[i + 1], setter, max_width, limit, cmin)):
                    b = pages[i + 1]
                    p.tokens = p.tokens + b.tokens
                    p.end = max(p.end, b.end)
                    del pages[i + 1]
                    changed = True
                    break
    hold(pages)
    return pages


# ----------------------------------------------------------------------------- emphasis

def mark_emphasis(tokens: list[Token], phrases: list[str], manual: set, sources: list) -> list:
    """Token-level hook + manual emphasis. Returns the matches per phrase (token positions)."""
    matches = []
    if "hook" in sources and phrases:
        matches = match_phrases(phrases, [t.natural for t in tokens])
        for pi, found in enumerate(matches):
            for mi, positions in enumerate(found):
                for pos in positions:
                    t = tokens[pos]
                    if t.emph is None:
                        t.emph, t.run, t.phrase = "hook", ("hook", pi, mi), pi
    if "manual" in sources and manual:
        run_id = None
        for pos, t in enumerate(tokens):
            if t.emph is None and any(i in manual for i in t.idx):
                prev = tokens[pos - 1] if pos else None
                if not (prev is not None and prev.emph == "manual" and prev.run == run_id):
                    run_id = ("manual", pos)
                t.emph, t.run = "manual", run_id
    return matches


def cap_emphasis(page: Page, sources: list, per_page) -> None:
    if per_page is None:
        return
    order = {s: i for i, s in enumerate(sources)}
    runs: list[tuple] = []
    for pos, t in enumerate(page.tokens):
        if t.emph and (not runs or runs[-1][1] != t.run):
            runs.append((order.get(t.emph, 99), t.run, pos))
    keep = {r[1] for r in sorted(runs, key=lambda r: (r[0], r[2]))[: int(per_page)]}
    for t in page.tokens:
        if t.emph and t.run not in keep:
            t.emph, t.run, t.phrase = None, None, None


def _auto_candidates(page: Page) -> list[tuple]:
    stop, neg, hedges = _lists()
    cands = []
    for pos, t in enumerate(page.tokens):
        if t.emph:
            continue
        units = word_units(t.natural)
        key = _key(t.natural)
        if any(u.startswith("#") for u in units) or any(c in t.natural for c in "$%£€"):
            tier = 3
        elif (t.natural[:1].isupper() and not t.sentence_start and key not in stop
              and key not in ("i", "im", "ive", "ill", "id")):
            tier = 2
        elif (len(key) >= 6 and key not in stop and key not in hedges and key not in neg
              and not key.endswith("ly")):
            tier = 1
        else:
            continue
        cands.append((tier, len(key), -pos, pos))
    return sorted(cands, reverse=True)


# ----------------------------------------------------------------------------- the track

class CaptionTrack:
    """Caption strip for one clip x format x style. See module docstring."""

    def __init__(self, *, width, band_y, band_h, frames, fps, pages, hook, style, fmt,
                 setter, tiles, hook_tiles, accents, meta, srt_tokens=None, issues=None,
                 seams=None):
        self.width = width
        self._srt_tokens = list(srt_tokens or [])
        self._seams = seams
        self._issues = list(issues or [])
        self.band_y = band_y
        self.band_h = band_h
        self.frames = frames
        self.fps = fps
        self.fmt = fmt
        self._pages = pages
        self._hook = hook
        self._style = style
        self._setter = setter
        self._tiles = tiles
        self._hook_tiles = hook_tiles
        self._accents = accents
        self._meta = meta
        self._starts = [p.start_f for p in pages]
        self._cache: OrderedDict = OrderedDict()
        self._blank = bytes(width * band_h * 4)
        self._pill_cache: dict = {}
        self._scaled: dict = {}
        cap = style["caption"]
        self._color = parse_color(cap.get("color"))[:3]
        accent = cap.get("accent")
        self._accent = parse_color(accent)[:3] if accent else self._color
        self._ua = float(cap.get("unspoken_alpha", 0.55))
        self._reveal = (cap.get("reveal") or "fill").lower()
        reveal_mode = (cap.get("accent_reveal") or "on_spoken").lower()
        self._accent_early = reveal_mode in ("page_start", "page", "always")
        anim = cap.get("anim") or {}
        self._in_s = max(0.0, float(anim.get("in_ms", 100)) / 1000.0)
        self._rise = float(anim.get("rise_px", 0))
        self._pop = bool(anim.get("pop"))
        self._shadow = list(cap.get("shadow") or [])
        outline = cap.get("outline") or {}
        self._outline_px = int(outline.get("px") or 0)
        self._outline_rgb = parse_color(outline.get("color", "#000000"))[:3]
        self._pill_rgba = parse_color((cap.get("backing") or {}).get("color", "#0B0B0DD9"))
        self._pill_radius = int((cap.get("backing") or {}).get("radius", 22))
        num, den = fps
        self._onsets = [[_frac(t.s - setter_lead(style)) for t in p.tokens] for p in pages]
        if hook:
            self._hook_end = Fraction(hook["_dur_us"], 1_000_000)
            self._hook_fade = Fraction(hook["_fade_us"], 1_000_000)
            self._hook_end_f = math.ceil(self._hook_end * num / den)
        else:
            self._hook_end_f = 0

    # --- public API

    @property
    def pages(self) -> list[dict]:
        return [self._page_json(i, p) for i, p in enumerate(self._pages)]

    @property
    def hook(self) -> dict | None:
        return self._hook_json()

    def time(self, n: int) -> Fraction:
        num, den = self.fps
        return Fraction(n * den, num)

    def frame_key(self, n: int) -> tuple:
        """Visual state of frame n; frames with equal keys are byte-identical."""
        if n < 0 or n >= self.frames:
            raise IndexError(f"frame {n} outside 0..{self.frames - 1}")
        t = self.time(n)
        if self._hook and n < self._hook_end_f:
            return ("hook", self._hook_alpha(t))
        i = bisect_right(self._starts, n) - 1
        if i >= 0 and n < self._pages[i].end_f:
            page = self._pages[i]
            states = tuple(self._token_state(i, j, t) for j in range(len(page.tokens)))
            pill_alone = page.backing and self._reveal != "build"
            if pill_alone or any(st != INVISIBLE for st in states):
                return ("page", i, states)
        return ("blank",)

    def frame(self, n: int) -> bytes:
        key = self.frame_key(n)
        if key[0] == "blank":
            return self._blank
        hit = self._cache.get(key)
        if hit is not None:
            self._cache.move_to_end(key)
            return hit
        data = self._render(key)
        self._cache[key] = data
        if len(self._cache) > FRAME_CACHE:
            self._cache.popitem(last=False)
        return data

    def iter_frames(self) -> Iterator[bytes]:
        for n in range(self.frames):
            yield self.frame(n)

    def frame_image(self, n: int) -> Image.Image:
        """Frame n as an RGBA image of the band (for previews and sheets)."""
        return Image.frombytes("RGBA", (self.width, self.band_h), self.frame(n))

    def to_json(self) -> dict:
        cap = self._style["caption"]
        return {
            "version": 1,
            "style": self._style.get("name"),
            "format": self.fmt,
            "band": [self.band_y, self.band_h],
            "size": [self.width, self._meta["height"]],
            "fps": list(self.fps),
            "frames": self.frames,
            "font": {"spec": cap["font"], "size": int(cap["size"]),
                     "emphasis_spec": self._setter.emph_spec,
                     "emphasis_size_mult": self._setter.mult},
            "reveal": self._reveal,
            "lead_ms": float(cap.get("lead_ms", 0)),
            "safezone": self._meta["safezone"],
            "max_width": self._meta["max_width"],
            "backing_mode": self._meta.get("backing_mode"),
            "backing_override": self._meta.get("backing_override"),
            "pages": self.pages,
            "hook": self._hook_json(),
            "accents": self._accents,
            "warnings": list(self._meta["warnings"]),
            "issues": [dict(i) for i in self._issues],
        }

    @property
    def issues(self) -> list[dict]:
        return [dict(i) for i in self._issues]

    def srt_cues(self) -> list[tuple]:
        """(start, end, text) subtitle cues from every visible word (see _srt_cues): broken at
        every piece, omit and dip join and held inside the clip's duration."""
        return _srt_cues(self._srt_tokens, self._seams, self._meta.get("duration"))

    def write_srt(self, path) -> Path:
        """Sidecar subtitles from every visible word, natural case, UTF-8.

        Independent of the burned-in paging and the hook card: words spoken under the card are
        included and the card's headline is not. Cues follow subtitle conventions (at most
        SRT_LINES lines of SRT_LINE characters, SRT_MAX_DUR seconds, broken at sentence and
        clause ends). Like the pages, a cue never crosses a piece, omit or dip join, and no cue
        ends after the clip does."""
        out = []
        for n, (a, b, text) in enumerate(self.srt_cues(), 1):
            out.append(f"{n}\n{_srt_time(a)} --> {_srt_time(b)}\n{text}\n")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text("\n".join(out), encoding="utf-8")
        tmp.replace(path)
        return path

    # --- state

    def _hook_alpha(self, t: Fraction) -> int:
        fade_start = self._hook_end - self._hook_fade
        if t < fade_start or self._hook_fade <= 0:
            return 255
        a = float((self._hook_end - t) / self._hook_fade)
        return max(1, int(round(255 * min(1.0, max(0.0, a)))))      # visible until dur

    def _token_state(self, pi: int, ti: int, t: Fraction) -> tuple:
        """(alpha, r, g, b, dy, scale) quantized. Anything fully transparent collapses to one
        sentinel, so equal keys mean equal pixels and different keys mean different pixels."""
        st = self._raw_state(pi, ti, t)
        return INVISIBLE if st[0] <= 0 else st

    def _raw_state(self, pi: int, ti: int, t: Fraction) -> tuple:
        tok = self._pages[pi].tokens[ti]
        emph = bool(tok.emph)
        if self._reveal == "static":
            col = self._accent if emph else self._color
            return (255,) + tuple(col) + (0, 1000)
        onset = self._onsets[pi][ti]
        if t < onset:                                   # not yet spoken: a fixed sentinel
            if self._reveal == "build":
                return INVISIBLE
            col = self._accent if (emph and self._accent_early) else self._color
            return (int(round(255 * self._ua)),) + tuple(col) + (0, 1000)
        dt = float(t - onset)
        p = 1.0 if self._in_s <= 0 else _ease_out(dt / self._in_s)
        if self._reveal == "build":
            col = self._accent if emph else self._color
            scale = int(round(1000 * _pop_scale(dt))) if (emph and self._pop) else 1000
            return (int(round(255 * p)),) + tuple(col) + (int(round((1 - p) * self._rise)), scale)
        alpha = self._ua + (1 - self._ua) * p
        if emph and not self._accent_early:
            col = tuple(int(round(c0 + (c1 - c0) * p)) for c0, c1 in zip(self._color, self._accent))
        else:
            col = self._accent if emph else self._color
        return (int(round(255 * alpha)),) + tuple(col) + (0, 1000)

    # --- drawing

    def _render(self, key: tuple) -> bytes:
        h, w = self.band_h, self.width
        prem = np.zeros((h, w, 3), np.float32)
        alpha = np.zeros((h, w), np.float32)
        pieces = []          # (tile, x, y, rgb, a, stroke_tile, sx, sy)
        pill = None
        if key[0] == "hook":
            a = key[1] / 255.0
            hk = self._hook
            for tile in self._hook_tiles:
                pieces.append((tile["mask"], tile["x"], tile["y"], tile["rgb"], a,
                               tile.get("stroke"), tile.get("sx"), tile.get("sy")))
            if hk["backing"]:
                pill = (hk["_pill"], a)
        else:
            pi, states = key[1], key[2]
            page = self._pages[pi]
            for ti, st in enumerate(states):
                a = st[0] / 255.0
                if a <= 0:
                    continue
                tile = self._tiles[pi][ti]
                mask, x, y = tile["mask"], tile["x"], tile["y"] + st[4]
                stroke, sx, sy = tile.get("stroke"), tile.get("sx"), tile.get("sy")
                if st[5] != 1000:
                    mask, x, y, stroke, sx, sy = self._scaled_tile(pi, ti, st[5], st[4])
                pieces.append((mask, x, y, st[1:4], a, stroke, sx, sy))
            if page.backing:
                box, pa = page.pill, 1.0
                if self._reveal == "build":
                    box = self._build_pill(page, states)
                if box is not None:
                    pill = (box, pa)
        if pill is not None:
            box, a = pill
            mask, x, y = self._pill_mask(box)
            _over(prem, alpha, mask * (a * self._pill_rgba[3] / 255.0), x, y - self.band_y,
                  self._pill_rgba[:3])
        elif self._shadow and pieces:
            self._draw_shadows(prem, alpha, pieces)
        if self._outline_px > 0:
            for mask, x, y, rgb, a, stroke, sx, sy in pieces:
                if stroke is not None:
                    _over(prem, alpha, stroke * a, sx, sy - self.band_y, self._outline_rgb)
        for mask, x, y, rgb, a, stroke, sx, sy in pieces:
            _over(prem, alpha, mask * a, x, y - self.band_y, rgb)
        out = np.zeros((h, w, 4), np.uint8)
        nz = alpha > 0
        straight = np.zeros_like(prem)
        straight[nz] = prem[nz] / alpha[nz][:, None]
        out[..., :3] = np.clip(np.rint(straight * 255.0), 0, 255).astype(np.uint8)
        out[..., 3] = np.clip(np.rint(alpha * 255.0), 0, 255).astype(np.uint8)
        return out.tobytes()

    @staticmethod
    def _build_pill(page: Page, states: tuple):
        """The pill of a build-reveal page for one frame: it reaches the newest word on screen
        from the first frame that word is drawn (however faint), at full strength, so a word is
        never on screen without its backing and there is never a dark bar ahead of the speech.
        (It used to ease to the new word over the word's own fade, and fade in with the first
        word: for a frame or two the incoming word was drawn faint, outside the pill.)"""
        shown = [j for j, st in enumerate(states) if st[0] > 0]
        if not shown:
            return None
        pills = page.token_pills or []
        k = shown[-1]
        if k < len(pills) and pills[k] is not None:
            return pills[k]
        return page.pill

    def _draw_shadows(self, prem, alpha, pieces) -> None:
        # The shadow follows the outline when there is one, else the glyphs.
        shapes = [((p[5], p[6], p[7]) if p[5] is not None else (p[0], p[1], p[2]), p[4])
                  for p in pieces]
        reach = max(int(math.ceil(3 * float(l.get("blur", 0)))) + abs(int(l.get("dy", 0)))
                    for l in self._shadow) + 2
        x0 = max(0, min(x for (m, x, y), _ in shapes) - reach)
        y0 = max(0, min(y - self.band_y for (m, x, y), _ in shapes) - reach)
        x1 = min(self.width, max(x + m.shape[1] for (m, x, y), _ in shapes) + reach)
        y1 = min(self.band_h, max(y - self.band_y + m.shape[0] for (m, x, y), _ in shapes) + reach)
        if x1 <= x0 or y1 <= y0:
            return
        cov = np.zeros((y1 - y0, x1 - x0), np.float32)
        for (m, x, y), a in shapes:
            _union(cov, m * a, x - x0, y - self.band_y - y0)
        base = Image.fromarray(np.clip(np.rint(cov * 255.0), 0, 255).astype(np.uint8), "L")
        for layer in self._shadow:
            blur = float(layer.get("blur", 0))
            img = base.filter(ImageFilter.GaussianBlur(blur)) if blur > 0 else base
            m = np.asarray(img, np.float32) / 255.0 * float(layer.get("alpha", 0.5))
            _over(prem, alpha, np.clip(m, 0, 1), x0, y0 + int(layer.get("dy", 0)), (0, 0, 0))

    def _pill_mask(self, box) -> tuple:
        hit = self._pill_cache.get(box)
        if hit is None:
            x, y, w, h = box
            r = min(self._pill_radius, h // 2, w // 2)
            ss = 4
            img = Image.new("L", (w * ss, h * ss), 0)
            ImageDraw.Draw(img).rounded_rectangle((0, 0, w * ss - 1, h * ss - 1), radius=r * ss,
                                                  fill=255)
            img = img.resize((w, h), Image.Resampling.BOX)
            hit = (np.asarray(img, np.float32) / 255.0, x, y)
            self._pill_cache[box] = hit
        return hit

    def _scaled_tile(self, pi, ti, scale_milli, dy) -> tuple:
        k = (pi, ti, scale_milli)
        if k not in self._scaled:
            tile = self._tiles[pi][ti]
            s = scale_milli / 1000.0
            cx, cy = tile["cx"], tile["cy"]          # frame coords of the token's visual centre

            def scale(arr, x, y):
                hgt, wid = arr.shape
                nw, nh = max(1, int(round(wid * s))), max(1, int(round(hgt * s)))
                img = Image.fromarray(np.clip(np.rint(arr * 255), 0, 255).astype(np.uint8), "L")
                out = np.asarray(img.resize((nw, nh), Image.Resampling.BICUBIC), np.float32) / 255.0
                nx = int(round(cx - (cx - x) * s))
                ny = int(round(cy - (cy - y) * s))
                return out, nx, ny

            mask, x, y = scale(tile["mask"], tile["x"], tile["y"])
            stroke = sx = sy = None
            if tile.get("stroke") is not None:
                stroke, sx, sy = scale(tile["stroke"], tile["sx"], tile["sy"])
            self._scaled[k] = (mask, x, y, stroke, sx, sy)
        mask, x, y, stroke, sx, sy = self._scaled[k]
        return mask, x, y + dy, stroke, sx, (sy + dy if sy is not None else None)

    # --- json

    def _page_json(self, i: int, p: Page) -> dict:
        return {
            "i": i,
            "start_out": _round_in(p.start, p.lo, p.hi),
            "end_out": _round_in(p.end, p.lo, p.hi),
            "frames": [p.start_f, p.end_f],
            "text": " ".join(t.text for t in p.tokens),
            "lines": [list(line) for line in p.line_texts] if p.line_texts
            else [[p.tokens[j].text for j in line] for line in p.lines],
            "tokens": [{"text": t.text, "s_out": round(t.s, 3), "e_out": round(t.e, 3),
                        "emph": t.emph is not None, "source": t.emph, "idx": list(t.idx)}
                       for t in p.tokens],
            "size": p.size,
            "box": list(p.box),
            "backing": bool(p.backing),
            "backing_box": list(p.pill) if p.backing and p.pill else None,
            "backing_forced": bool(p.forced),
            "backing_reason": p.reason,
            "p90": None if p.p90 is None else round(p.p90, 3),
            "edge_density": None if p.edges is None else round(p.edges, 3),
            "contrast": None if p.contrast is None else round(p.contrast, 2),
            "break": p.brk,
        }

    def _hook_json(self) -> dict | None:
        hk = self._hook
        if not hk:
            return None
        return {"lines": list(hk["src_lines"]), "rendered_lines": list(hk["rendered"]),
                "size": hk["size"], "caption_size": hk.get("caption_size"), "box": list(hk["box"]),
                "start_out": 0.0, "end_out": round(hk["dur"], 3), "backing": bool(hk["backing"]),
                "p90": None if hk["p90"] is None else round(hk["p90"], 3),
                "contrast": None if hk["contrast"] is None else round(hk["contrast"], 2),
                "frames": [0, self._hook_end_f]}


def _round_in(t: float, lo: float, hi: float) -> float:
    """t to the millisecond, never rounded out of [lo, hi]: a page held at a join at 9.666667 s
    reads 9.666, not 9.667 (which would be past the join)."""
    r = round(t, 3)
    if r < lo - 1e-9:
        r = math.ceil(lo * 1000.0 - 1e-6) / 1000.0
    if math.isfinite(hi) and r > hi + 1e-9:
        r = math.floor(hi * 1000.0 + 1e-6) / 1000.0
    return round(r, 3)


def setter_lead(style: dict) -> float:
    return float(style["caption"].get("lead_ms", 0)) / 1000.0


def _splits_number(a: str, b: str) -> bool:
    """A break between `a` and `b` would split a number phrase: a number and the word that
    belongs to it ("30 | seconds", "12 | to 18") or a range word and its number ("to | 18")."""
    if a.rstrip(CLOSERS).endswith(CLAUSE_END + SENTENCE_END):
        return False
    if _numberish(a):
        return True
    return _key(a) in _RANGE_WORDS and _numberish(b)


def _numberish(raw: str) -> bool:
    """A number written in digits ("18", "$2,400") or in words ("eighteen", "three thousand")."""
    if _is_number(raw):
        return True
    from . import textnorm
    return textnorm.num_value(raw.strip(CLOSERS + ".,;:!?")) is not None


def _srt_split(words: list[str]) -> list[str] | None:
    """One or two subtitle lines of at most SRT_LINE characters (None when they do not fit).
    Two lines are balanced, preferring a break after a clause mark and never after a hard
    dangler or inside a number phrase when another split fits."""
    text = " ".join(words)
    if len(text) <= SRT_LINE:
        return [text]
    best, best_key = None, None
    for k in range(1, len(words)):
        a, b = " ".join(words[:k]), " ".join(words[k:])
        if len(a) > SRT_LINE or len(b) > SRT_LINE:
            continue
        bad = 1 if (_key(words[k - 1]) in _HARD_DANGLERS
                    or _splits_number(words[k - 1], words[k])) else 0
        clause = 0 if words[k - 1].rstrip(CLOSERS).endswith(CLAUSE_END + SENTENCE_END) else 1
        soft = 1 if _key(words[k - 1]) in _SOFT_DANGLERS else 0
        key = (bad, clause, soft, max(len(a), len(b)), -len(a))
        if best_key is None or key < best_key:
            best, best_key = [a, b], key
    return best


def _srt_cues(tokens: list[Token], seams: _Seams | None = None,
              duration: float | None = None) -> list[tuple]:
    """Subtitle cues (start, end, text) over display tokens, natural case.

    Hard breaks: a speaker change, a silence of SRT_GAP or more and, with `seams`, every piece,
    omit or dip join (a cue never reads across a cut). Inside a run, a dynamic programme picks
    the fewest cues that fit SRT_LINES x SRT_LINE characters and SRT_MAX_DUR seconds,
    preferring to end cues at sentence ends, then clause ends, then pauses. A cue starts at its
    first word and ends SRT_TAIL after its last word, or at the next cue, or at the join that
    closes its run, whichever comes first; with `duration` every cue lies inside the file
    (0 to duration), on whole milliseconds."""
    if not tokens:
        return []
    runs, cur = [], [tokens[0]]
    for a, b in zip(tokens, tokens[1:]):
        if ((a.sp is not None and b.sp is not None and a.sp != b.sp) or b.s - a.e >= SRT_GAP
                or (seams is not None and seams.between(a.seg1, b.seg0))):
            runs.append(cur)
            cur = []
        cur.append(b)
    runs.append(cur)
    cues = []
    for run in runs:
        m = len(run)
        words = [t.natural for t in run]
        best = [math.inf] * (m + 1)
        back = [0] * (m + 1)
        lines_of: dict = {}
        best[0] = 0.0
        for i in range(1, m + 1):
            for j in range(i - 1, -1, -1):
                if i - j > 1 and (sum(len(w) + 1 for w in words[j:i]) - 1 > SRT_LINE * SRT_LINES
                                  or run[i - 1].e - run[j].s > SRT_MAX_DUR):
                    break
                lines = _srt_split(words[j:i]) if i - j > 1 else [words[j]]
                if lines is None or best[j] == math.inf:
                    continue
                last = words[i - 1]
                if i == m or _ends_sentence(last):
                    end_cost = 0.0
                elif _ends_clause(last):
                    end_cost = 15.0
                elif run[i].s - run[i - 1].e >= 0.45:
                    end_cost = 20.0
                else:
                    end_cost = 60.0 + (30.0 if _key(last) in _HARD_DANGLERS else
                                       20.0 if _key(last) in _SOFT_DANGLERS else 0.0)
                if i < m and _splits_number(last, words[i]):
                    end_cost += SRT_SPLIT_NUMBER
                if i - j == 1 and m > 1:
                    end_cost += SRT_ORPHAN      # a word alone on screen reads as a flash
                cost = best[j] + 100.0 + end_cost
                if cost < best[i] - 1e-9:
                    best[i], back[i] = cost, j
                    lines_of[i] = lines
        cuts, i = [], m
        while i > 0:
            cuts.append((back[i], i, lines_of[i]))
            i = back[i]
        lo, hi = seams.of(run) if seams is not None else (0.0, math.inf)
        if duration is not None:
            hi = min(hi, float(duration))
        for a, b, lines in reversed(cuts):
            cues.append([run[a].s, run[b - 1].e + SRT_TAIL, "\n".join(lines), lo, hi])
    if seams is None and duration is None:
        for k in range(len(cues) - 1):
            cues[k][1] = max(cues[k][0] + 0.001, min(cues[k][1], cues[k + 1][0]))
        return [tuple(c[:3]) for c in cues]
    # Whole milliseconds, rounded inward at the bounds (a join at 9.666667 s ends its cue at
    # 9.666), so the written timestamps never cross a join or run past the end of the file.
    ms: list[list] = []
    for k, (start, end, text, lo, hi) in enumerate(cues):
        if k + 1 < len(cues):
            end = min(end, cues[k + 1][0])
        lo_ms = int(math.ceil(max(0.0, lo) * 1000.0 - 1e-6))
        hi_ms = int(math.floor(hi * 1000.0 + 1e-6)) if math.isfinite(hi) else None
        a_ms, b_ms = max(int(round(start * 1000.0)), lo_ms), int(round(end * 1000.0))
        if hi_ms is not None:
            b_ms = min(b_ms, hi_ms)
            a_ms = min(a_ms, max(lo_ms, hi_ms - 1))     # a word stamped on the join itself
        b_ms = max(b_ms, a_ms + 1)
        if ms and ms[-1][1] > a_ms:                     # ... never under the cue before it
            ms[-1][1] = max(ms[-1][0] + 1, a_ms)
        ms.append([a_ms, b_ms, text])
    return [(a / 1000.0, b / 1000.0, text) for a, b, text in ms]


def _srt_time(t: float) -> str:
    ms = max(0, int(round(t * 1000)))
    return f"{ms // 3600000:02d}:{ms % 3600000 // 60000:02d}:{ms % 60000 // 1000:02d},{ms % 1000:03d}"


def _region(arr_shape, src_shape, x, y):
    """Clip a src tile placed at (x, y) to a destination of arr_shape; return slices or None."""
    H, W = arr_shape[:2]
    h, w = src_shape[:2]
    dx0, dy0 = max(0, x), max(0, y)
    dx1, dy1 = min(W, x + w), min(H, y + h)
    if dx1 <= dx0 or dy1 <= dy0:
        return None
    return (slice(dy0, dy1), slice(dx0, dx1)), (slice(dy0 - y, dy1 - y), slice(dx0 - x, dx1 - x))


def _over(prem, alpha, mask, x, y, rgb) -> None:
    reg = _region(alpha.shape, mask.shape, int(x), int(y))
    if reg is None:
        return
    d, s = reg
    a = mask[s]
    col = np.asarray(rgb, np.float32) / 255.0
    inv = 1.0 - a
    prem[d] = col[None, None, :] * a[..., None] + prem[d] * inv[..., None]
    alpha[d] = a + alpha[d] * inv


def _union(cov, mask, x, y) -> None:
    reg = _region(cov.shape, mask.shape, int(x), int(y))
    if reg is None:
        return
    d, s = reg
    cov[d] = 1.0 - (1.0 - cov[d]) * (1.0 - mask[s])


# ----------------------------------------------------------------------------- build

def _tile(font, text, tracking, x, baseline, outline_px, cap_h) -> dict:
    img, (ox, oy) = typeset.text_mask(font, text, tracking)
    tx, ty = int(round(x)) - ox, int(round(baseline)) - oy
    tile = {"mask": np.asarray(img, np.float32) / 255.0, "x": tx, "y": ty,
            "cx": x + typeset.text_width(font, text, tracking) / 2.0, "cy": baseline - cap_h / 2.0}
    if outline_px > 0:
        simg, (sox, soy) = typeset.text_mask(font, text, tracking, stroke=outline_px)
        tile["stroke"] = np.asarray(simg, np.float32) / 255.0
        tile["sx"], tile["sy"] = int(round(x)) - sox, int(round(baseline)) - soy
    return tile


def _tile_runs(font, runs: list, tracking, outline_px, cap_h) -> dict:
    """One tile for a token drawn as several runs (a merged token wrapped over lines): the
    runs' masks on one canvas, so the token still fades, rises and pops as a single unit."""
    tiles = [_tile(font, text, tracking, x, baseline, outline_px, cap_h)
             for text, x, baseline in runs]

    def merge(mask_key, x_key, y_key):
        x0 = min(t[x_key] for t in tiles)
        y0 = min(t[y_key] for t in tiles)
        x1 = max(t[x_key] + t[mask_key].shape[1] for t in tiles)
        y1 = max(t[y_key] + t[mask_key].shape[0] for t in tiles)
        canvas = np.zeros((y1 - y0, x1 - x0), np.float32)
        for t in tiles:
            m = t[mask_key]
            ys, xs = t[y_key] - y0, t[x_key] - x0
            region = canvas[ys:ys + m.shape[0], xs:xs + m.shape[1]]
            np.maximum(region, m, out=region)
        return canvas, x0, y0

    mask, x, y = merge("mask", "x", "y")
    left = min(rx for _, rx, _ in runs)
    right = max(rx + typeset.text_width(font, text, tracking) for text, rx, _ in runs)
    top = min(b for _, _, b in runs) - cap_h
    bottom = max(b for _, _, b in runs)
    out = {"mask": mask, "x": x, "y": y, "cx": (left + right) / 2.0, "cy": (top + bottom) / 2.0}
    if outline_px > 0:
        out["stroke"], out["sx"], out["sy"] = merge("stroke", "sx", "sy")
    return out


def _ink_box(font, text, tracking, x, baseline) -> tuple:
    l, t, r, b = font.getbbox(text, anchor="ls")
    r += typeset.tracking_px(font, tracking) * max(0, len(text) - 1)
    return (x + l, baseline + t, x + r, baseline + b)


def _union_box(boxes) -> tuple:
    boxes = [b for b in boxes if b]
    return (min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes))


def _xywh(box) -> tuple:
    x0, y0 = int(math.floor(box[0])), int(math.floor(box[1]))
    return (x0, y0, int(math.ceil(box[2])) - x0, int(math.ceil(box[3])) - y0)


def _layout_page(page: Page, setter: _Setter, max_width: float, cx: float, cy: float,
                 anchor: str = "center") -> list[tuple]:
    """Lay the page out around (cx, cy). `anchor` puts the first line ("top"), the block
    ("center") or the last line ("bottom") on cy. Returns notes [(issue code or None, msg)]."""
    notes = []
    size = setter.size
    lines = setter.wrap(page.tokens, size, max_width)
    text = " ".join(t.text for t in page.tokens)
    if lines is None:
        floor = max(8, int(size * SHRINK_FLOOR))
        while lines is None and size > floor:
            size -= 1
            lines = setter.wrap(page.tokens, size, max_width)
        if lines is None:
            lines = typeset.balanced_wrap(page.tokens, lambda ts: setter.measure(ts, size),
                                          math.inf, setter.max_lines) or [page.tokens]
            notes.append(("overflow", f"page '{text}' does not fit {int(max_width)} px even at "
                                      f"{size} px"))
        else:
            code = "word_shrunk" if size < WORD_SHRUNK * setter.size else None
            notes.append((code, f"page '{text}' shrunk to {size} px to fit"))
    page.size = size
    base = setter.font(size)
    cap_h = typeset.cap_height(base)
    pitch = setter.line_gap * size
    n = len(lines)
    if anchor == "top":
        b0 = cy + cap_h / 2.0
    elif anchor == "bottom":
        b0 = cy + cap_h / 2.0 - (n - 1) * pitch
    else:
        b0 = cy + cap_h / 2.0 - (n - 1) * pitch / 2.0
    space = setter.space(size)
    position = {id(t): i for i, t in enumerate(page.tokens)}
    page.lines, page.pos = [], [None] * len(page.tokens)
    page.line_texts, page.runs = [], [[] for _ in page.tokens]
    ink, xs = [], []
    desc = 0 if setter.case == "upper" else round(0.22 * size)
    grown = [None] * len(page.tokens)    # x range revealed through each token, per line
    for li, line in enumerate(lines):
        # A line holds whole tokens, or parts of a merged token that wraps (see _Setter.wrap).
        runs = _line_runs(line)
        widths = [setter.run_width(t, text, size) for t, text in runs]
        lw = sum(widths) + space * (len(runs) - 1)
        x = cx - lw / 2.0
        xs.append((x, x + lw))
        baseline = b0 + li * pitch
        page.lines.append([position[id(t)] for t, _ in runs])
        page.line_texts.append([text for _, text in runs])
        x_line0 = x
        for (t, text), w in zip(runs, widths):
            ti = position[id(t)]
            if page.pos[ti] is None:
                page.pos[ti] = (x, baseline)
            page.runs[ti].append((text, x, baseline))
            ink.append(_ink_box(setter.font(size, bool(t.emph)), text, setter.tracking, x, baseline))
            prev = [xs[k] for k in range(li)]
            lo = min([a for a, _ in prev] + [x_line0])
            hi = max([b for _, b in prev] + [x + w])
            grown[ti] = (lo, b0 - cap_h, hi, baseline + desc)     # through the token's last run
            x += w + space
    page.token_typos = grown
    page.line_typos = [(min(a for a, _ in xs[:k + 1]), b0 - cap_h, max(b for _, b in xs[:k + 1]),
                        b0 + k * pitch + desc) for k in range(n)]
    typo = page.line_typos[-1]
    page.typo = typo
    page.box = _xywh(_union_box(ink + [typo]))
    return notes


def _reveal_inks(page: Page, setter: _Setter, peak: float = 1.0) -> list:
    """Per token, the ink box of everything on screen once that token is revealed (build
    reveal, reading order), so the pill that has grown to a token covers accents, descenders
    and a larger emphasis face as the page pill does. With `peak` above 1 an emphasised
    token's own ink is taken at the top of its pop: scaled about the tile centre, as
    _scaled_tile draws it."""
    cap_h = typeset.cap_height(setter.font(page.size))
    out, seen = [], []
    for tok, runs in zip(page.tokens, page.runs):
        if runs:
            font = setter.font(page.size, bool(tok.emph))
            own = _union_box([_ink_box(font, text, setter.tracking, x, b) for text, x, b in runs])
            if peak > 1.0 and tok.emph:
                left = min(x for _, x, _ in runs)
                right = max(x + typeset.text_width(font, text, setter.tracking) for text, x, _ in runs)
                cx = (left + right) / 2.0
                cy = (min(b for _, _, b in runs) - cap_h + max(b for _, _, b in runs)) / 2.0
                own = (cx - (cx - own[0]) * peak, cy - (cy - own[1]) * peak,
                       cx + (own[2] - cx) * peak, cy + (own[3] - cy) * peak)
            seen.append(own)
        out.append(_union_box(seen) if seen else None)
    return out


def _pill_box(typo, ink_xywh, pad, W, H) -> tuple:
    px, py = int(pad[0]), int(pad[1])
    x0 = min(typo[0] - px, ink_xywh[0] - 4)
    y0 = min(typo[1] - py, ink_xywh[1] - 4)
    x1 = max(typo[2] + px, ink_xywh[0] + ink_xywh[2] + 4)
    y1 = max(typo[3] + py, ink_xywh[1] + ink_xywh[3] + 4)
    x0, y0 = max(0, int(math.floor(x0))), max(0, int(math.floor(y0)))
    x1, y1 = min(W, int(math.ceil(x1))), min(H, int(math.ceil(y1)))
    return (x0, y0, x1 - x0, y1 - y0)


def _hook_line(markup: str, case: str) -> tuple[str, list]:
    """One authored hook line -> (display text, per-character accent flags). Zero-width
    characters are dropped (see typeset.strip_invisible)."""
    chars, flags = [], []
    markup, _ = typeset.strip_invisible(markup)
    for t, a in parse_accent_runs(markup):
        t = t.upper() if case == "upper" else (t.lower() if case == "lower" else t)
        chars.extend(t)
        flags.extend([a] * len(t))
    text = "".join(chars)
    lead = len(text) - len(text.lstrip())
    text = text.strip()
    return text, flags[lead:lead + len(text)]


def _split_line(line: tuple, font) -> list | None:
    """Split one hook line into two width-balanced lines at a space (None if one word)."""
    text, flags = line
    words, pos = [], 0
    for wd in text.split(" "):
        if wd:
            words.append((wd, flags[pos:pos + len(wd)]))
        pos += len(wd) + 1
    if len(words) < 2:
        return None
    parts = typeset.balanced_wrap(words, lambda ws: typeset.text_width(
        font, " ".join(x for x, _ in ws)), math.inf, 2)
    if not parts or len(parts) < 2:
        best = min(range(1, len(words)), key=lambda k: max(
            typeset.text_width(font, " ".join(x for x, _ in words[:k])),
            typeset.text_width(font, " ".join(x for x, _ in words[k:]))))
        parts = [words[:best], words[best:]]
    out = []
    for part in parts:
        f = []
        for k, (x, fl) in enumerate(part):
            f.extend(fl)
            if k + 1 < len(part):
                f.append(bool(fl and fl[-1] and part[k + 1][1] and part[k + 1][1][0]))
        out.append((" ".join(x for x, _ in part), f))
    return out


def _hook_layout(hook: dict, style: dict, max_w: float, cx: float, cy: float, y_lim: tuple,
                 project_root, warnings: list, caption_size: float | None = None) -> dict:
    """Fit the hook card: authored lines first; split the widest line (up to max_lines,
    default HOOK_MAX_LINES) when the authored lines overflow, or when they set below
    HOOK_KEEP_AT of max_size and the split buys at least HOOK_SPLIT_GAIN more size. A card
    still under hook_card.min_ratio x the caption size is re-broken for size regardless of
    HOOK_KEEP_AT; if it stays under, the result carries `small` and the line length
    (`max_chars`) that would reach it."""
    hc = style["hook_card"]
    case = (hc.get("case") or "upper").lower()
    spec = hc["font"]
    lo, hi = int(hc.get("min_size", 56)), int(hc.get("max_size", 104))
    authored = [_hook_line(ln, case) for ln in hook["lines"]]
    authored = [ln for ln in authored if ln[0]] or authored
    ref = typeset.load_font(spec, hi, project_root)

    def fit(lines):
        return typeset.fit_size([t for t, _ in lines], spec, max_w, lo, hi, project_root=project_root)

    def overflows(lines, size):
        f = typeset.load_font(spec, size, project_root)
        return any(typeset.text_width(f, t) > max_w for t, _ in lines)

    max_lines = int(hc.get("max_lines") or HOOK_MAX_LINES)
    cands = [authored]
    while len(cands[-1]) < max(max_lines, len(authored)) or (
            len(cands[-1]) < len(authored) + 2 and overflows(cands[-1], fit(cands[-1]))):
        cur = cands[-1]
        order = sorted(range(len(cur)), key=lambda i: -typeset.text_width(ref, cur[i][0]))
        split = None
        for i in order:
            parts = _split_line(cur[i], ref)
            if parts:
                split = cur[:i] + parts + cur[i + 1:]
                break
        if split is None:
            break
        cands.append(split)
    sizes = [fit(c) for c in cands]
    choice = 0
    if overflows(authored, sizes[0]):
        choice = next((k for k, c in enumerate(cands) if not overflows(c, sizes[k])), len(cands) - 1)
    elif sizes[0] < HOOK_KEEP_AT * hi:
        # The author's breaks are an editorial choice: keep them once the card already sets
        # near full size; re-break only a card that would otherwise set small.
        choice = next((k for k in range(1, len(cands)) if len(cands[k]) <= max_lines
                       and sizes[k] >= HOOK_SPLIT_GAIN * sizes[0]), 0)
    ratio = float(hc.get("min_ratio", HOOK_MIN_RATIO) or 0.0)
    target = ratio * float(caption_size) if caption_size else 0.0
    if target and sizes[choice] < target - 1e-9:
        # Smaller than the minimum title-to-caption ratio: size wins over the authored breaks.
        ok = [k for k in range(len(cands)) if len(cands[k]) <= max(max_lines, len(authored))
              and not overflows(cands[k], sizes[k])]
        reach = [k for k in ok if sizes[k] >= target - 1e-9]
        pick = reach[0] if reach else (max(ok, key=lambda k: (sizes[k], -k)) if ok else choice)
        if sizes[pick] > sizes[choice]:
            choice = pick
    lines, size = cands[choice], sizes[choice]
    if choice:
        warnings.append(f"hook card re-broken into {len(lines)} lines to set it at {size} px")
    font = typeset.load_font(spec, size, project_root)
    if overflows(lines, size):
        warnings.append(f"hook card overflows {int(max_w)} px at the minimum size {size}")
    cap_h = typeset.cap_height(font)
    pitch = float(hc.get("line_gap", 1.0)) * size
    n = len(lines)
    desc = 0 if case == "upper" else round(0.22 * size)
    b0 = cy + cap_h / 2.0 - (n - 1) * pitch / 2.0
    top, bottom = b0 - cap_h, b0 + (n - 1) * pitch + desc
    if bottom > y_lim[1]:                       # keep the card above the bottom UI zone
        b0 -= bottom - y_lim[1]
    elif top < y_lim[0]:
        b0 += y_lim[0] - top
    color = parse_color(hc.get("color", style["caption"].get("color")))[:3]
    accent_v = hc.get("accent", style["caption"].get("accent"))
    accent = parse_color(accent_v)[:3] if accent_v else color
    outline_px = int((style["caption"].get("outline") or {}).get("px") or 0)

    def draw(b0):
        tiles, ink, xs = [], [], []
        for li, (text, flags) in enumerate(lines):
            lw = typeset.text_width(font, text)
            x0 = cx - lw / 2.0
            baseline = b0 + li * pitch
            xs.append((x0, x0 + lw))
            k = 0
            while k < len(text):
                j = k
                while j < len(text) and flags[j] == flags[k]:
                    j += 1
                run = text[k:j]
                if run.strip():
                    rx = x0 + typeset.text_width(font, text[:k])
                    tile = _tile(font, run, 0, rx, baseline, outline_px, cap_h)
                    tile["rgb"] = accent if flags[k] else color
                    tiles.append(tile)
                    ink.append(_ink_box(font, run, 0, rx, baseline))
                k = j
        typo = (min(a for a, _ in xs), b0 - cap_h, max(b for _, b in xs), b0 + (n - 1) * pitch + desc)
        return tiles, typo, _xywh(_union_box(ink + [typo]))

    tiles, typo, box = draw(b0)
    over = box[1] + box[3] - y_lim[1]
    if over > 0 and box[1] - over >= y_lim[0]:  # ink below the baseline (round letters, commas)
        b0 -= math.ceil(over)
        tiles, typo, box = draw(b0)
    small = bool(target and size < target - 1e-9)
    max_chars = None
    if target:
        # Line length (characters) that sets at the target size in this width, from the card's
        # own average advance.
        tf = typeset.load_font(spec, max(1, min(hi, int(math.ceil(target)))), project_root)
        joined = "".join(t for t, _ in lines)
        adv = typeset.text_width(tf, joined) / max(1, len(joined))
        max_chars = int(max_w // adv) if adv > 0 else None
    return {"tiles": tiles, "size": size, "box": box, "typo": typo, "font": font,
            "text": "".join(t for t, _ in lines), "lines_rendered": [t for t, _ in lines],
            "small": small, "target": target, "max_chars": max_chars}


def backing_override(clip: dict | None) -> str | None:
    """The clip's `caption_backing` ("auto", "always" or "never"), or None when it sets none.
    It replaces the preset's caption.backing.mode for this clip, and the hook card follows it."""
    clip = clip or {}
    v = clip.get("caption_backing")
    if v is None:
        v = (clip.get("_eff") or {}).get("caption_backing")
    v = str(v).strip().lower() if isinstance(v, str) else None
    return v if v in BACKING_MODES else None


def _mode(block: dict | None) -> str:
    m = str((block or {}).get("mode") or "auto").lower()
    return m if m in BACKING_MODES else "auto"


def _geometry(style: dict, clip: dict, fmt: str, W: int, H: int, project_root=None,
              safezone: dict | None = None) -> dict:
    """Caption and hook-card geometry: safe zone, centre, line widths and caption y. `notes`
    lists (issue code, message) for a clamped layout.max_width and an ignored caption_y.
    `mode` / `hook_mode` are the backing modes in force (the clip's caption_backing over the
    preset's)."""
    cap, lay = style["caption"], style["caption"]["layout"]
    notes = []
    zone_name = _zone_name(style, clip, fmt)
    zone = safezone if safezone is not None else _safezone(zone_name, project_root)
    cx = W / 2.0
    # Centred at the frame centre, never wider than the symmetric safe span.
    safe_span = 2.0 * min(cx - float(zone.get("left", 0)), W - float(zone.get("right", 0)) - cx)
    bk = cap.get("backing") or {}
    pad = bk.get("pad") or [28, 14]
    over = backing_override(clip)
    mode = over or _mode(bk)
    # Reserve the pill's side padding whenever a pill can appear, so the pill (not just the
    # text) stays inside the safe zone and page breaks never depend on the background. A clip
    # that only switches the pill off (caption_backing "never" on a preset that can pill)
    # keeps the reservation, so the override changes the pill and nothing else: same lines,
    # same pages, same position.
    room = not (mode == "never" and _mode(bk) == "never")
    pill_room = 2.0 * float(pad[0]) if room else 0.0
    want = float(lay.get("max_width") or W)
    max_width = max(64.0, min(want, safe_span - pill_room))
    if lay.get("max_width") and want > 1.1 * max_width:
        notes.append(("width_clamped",
                      f"layout.max_width {want:.0f} px is clamped to {max_width:.0f} px by the "
                      f"{zone_name} safe zone: lines break and long words shrink sooner than "
                      f"the preset intends"))
    cy = float(lay.get("y") or H * 0.65)
    if clip.get("caption_y") is not None:
        if fmt == "landscape":
            notes.append(("caption_y_clamped", "caption_y applies to vertical renders only; "
                                               "landscape uses the preset's layout.y"))
        else:
            cy = float(clip["caption_y"])
    hb = _deep_merge(bk, style["hook_card"].get("backing") or {})
    hook_mode = over or _mode(hb)
    hk_room = 0.0 if (hook_mode == "never" and _mode(hb) == "never") else 2.0 * float(
        (hb.get("pad") or pad)[0])
    hk_max = max(64.0, min(float(style["hook_card"].get("max_width") or W), safe_span - hk_room))
    y_lim = (float(zone.get("top", 0)), H - float(zone.get("bottom", 0)))
    return {"zone_name": zone_name, "zone": zone, "cx": cx, "cy": cy, "max_width": max_width,
            "bk": bk, "pad": pad, "hb": hb, "hk_max": hk_max, "y_lim": y_lim, "notes": notes,
            "mode": mode, "hook_mode": hook_mode, "override": over, "pill_room": room}


def _format_size(fmt: str) -> tuple:
    return (1920, 1080) if fmt == "landscape" else (1080, 1920)


def hook_fit(clip: dict | None, preset: dict, fmt: str, project_root=None) -> dict | None:
    """How the clip's hook card sets in this preset and format, without rendering anything.

    Returns {"hook_px", "caption_px", "ratio", "lines", "max_chars"}, or None when the clip has
    no hook card. `ratio` is hook_px / caption_px; `max_chars` is the line length (characters)
    that would let the card reach hook_card.min_ratio x the caption size in this width."""
    clip = clip or {}
    lines = hook_lines(clip.get("hook_card"))
    if not lines:
        return None
    style = resolve_style(preset, fmt)
    W, H = _format_size(fmt)
    geo = _geometry(style, clip, fmt, W, H, project_root)
    cap_px = int(style["caption"]["size"])
    hk = _hook_layout({"lines": lines}, style, geo["hk_max"], geo["cx"], geo["cy"], geo["y_lim"],
                      project_root, [], cap_px)
    return {"hook_px": int(hk["size"]), "caption_px": cap_px,
            "ratio": round(hk["size"] / float(cap_px), 3), "lines": list(hk["lines_rendered"]),
            "max_chars": hk["max_chars"]}


def _auto_emphasis(pages: list[Page], setter: _Setter, max_width: float, per_page) -> None:
    """Auto accents: numbers, currency and mid-sentence proper names only (tier >= 2), at
    least AUTO_GAP seconds apart, never a word already accented, and never on or next to a page
    that carries a hook accent (the hook's own words keep the colour)."""
    hooked = {k for k, p in enumerate(pages) if any(t.emph == "hook" for t in p.tokens)}
    accented = {_key(t.natural) for p in pages for t in p.tokens if t.emph}
    last = -math.inf
    for k, p in enumerate(pages):
        runs = {t.run for t in p.tokens if t.emph}
        room = per_page is None or len(runs) < int(per_page)
        if room and not ({k - 1, k, k + 1} & hooked):
            for tier, _, _, pos in _auto_candidates(p):
                if tier < 2:
                    break
                tok = p.tokens[pos]
                if _key(tok.natural) in accented or tok.s - last < AUTO_GAP:
                    continue
                tok.emph, tok.run = "auto", ("auto", k, pos)
                if setter.wrap(p.tokens, setter.size, max_width) is not None:
                    last = tok.s
                    break
                tok.emph, tok.run = None, None
        accented.update(_key(t.natural) for t in p.tokens if t.emph)


def _seg_out(plan: dict, t_src: float, end: bool = False):
    """Output time of source time t inside a plan segment (None when no segment holds it)."""
    for sg in plan.get("segments") or []:
        try:
            s0, s1, o0 = float(sg["src"][0]), float(sg["src"][1]), float(sg["out"][0])
        except (KeyError, TypeError, ValueError, IndexError):
            continue
        if s0 <= t_src < s1 or (end and abs(t_src - s1) < 1e-6):
            return o0 + t_src - s0
    return None


def _shot_bounds(plan: dict) -> list[float]:
    """Output times where the framed picture changes: framing-mark changes (inside a segment, or
    between two segments that sit under different marks) and speaker turns when the framing
    follows the speaker (plan "speakers" map, see framing). A piece join inside one framing is a
    jump cut on the same background, so the caption backing holds across it."""
    cuts = set()
    marks = []
    for m in plan.get("framing") or []:
        try:
            marks.append((float(m.get("at_src")), m))
        except (TypeError, ValueError):
            continue
    marks.sort(key=lambda x: x[0])
    at = [a for a, _ in marks]

    def mark_index(t):
        return bisect_right(at, t + 1e-6) - 1

    segs = plan.get("segments") or []
    for a, _ in marks:
        o = _seg_out(plan, a)
        if o is not None and o > 1e-6:
            cuts.add(round(o, 6))
    for prev, sg in zip(segs, segs[1:]):
        try:
            p_end, s0, o0 = float(prev["src"][1]), float(sg["src"][0]), float(sg["out"][0])
        except (KeyError, TypeError, ValueError, IndexError):
            continue
        if mark_index(p_end - 1e-3) != mark_index(s0):
            cuts.add(round(o0, 6))
    try:
        from . import framing
        follow = framing.follow_ok(plan) and (
            not marks or any(str(m.get("mode")) == "speaker" for _, m in marks))
        if follow:
            for k, turn in enumerate(framing.speaker_turns(plan)):
                if k == 0 or not math.isfinite(float(turn.get("t_out", math.nan))):
                    continue
                if marks:
                    i = mark_index(float(turn.get("at_src", 0.0)))
                    if i < 0 or str(marks[i][1].get("mode")) != "speaker":
                        continue
                cuts.add(round(max(0.0, float(turn["t_out"])), 6))
    except Exception:          # framing data unusable: joins and marks still bound the shots
        pass
    return sorted(c for c in cuts if c > 1e-6)


def _luma_region(luma, box, W, H):
    """(array, y0, y1, x0, x1, scale) of the luma samples behind a frame box, or None."""
    if luma is None:
        return None
    arr = np.asarray(luma)
    if arr.ndim != 3 or arr.shape[0] == 0:
        return None
    n, lh, lw = arr.shape
    x, y, w, h = box
    x0 = min(lw - 1, max(0, int(math.floor(x * lw / W))))
    y0 = min(lh - 1, max(0, int(math.floor(y * lh / H))))
    x1 = max(x0 + 1, min(lw, int(math.ceil((x + w) * lw / W))))
    y1 = max(y0 + 1, min(lh, int(math.ceil((y + h) * lh / H))))
    scale = 65535.0 if arr.dtype == np.uint16 else (1.0 if arr.dtype.kind == "f" else 255.0)
    return arr, y0, y1, x0, x1, scale


def _luma_index(arr, t: float) -> int:
    return min(arr.shape[0] - 1, max(0, int(round(t * LUMA_FPS))))


def _p90(luma, box, times, W, H) -> float | None:
    """90th percentile luma (0..1) behind `box` over the sample times."""
    reg = _luma_region(luma, box, W, H)
    if reg is None:
        return None
    arr, y0, y1, x0, x1, scale = reg
    v = np.concatenate([arr[_luma_index(arr, t), y0:y1, x0:x1].ravel() for t in times])
    return float(np.percentile(v.astype(np.float32), 90)) / scale


def _edge_density(luma, box, times, W, H) -> float | None:
    """Share of the luma samples behind `box` where brightness steps by more than TEXTURE_EDGE
    (0..1, gradient magnitude between neighbouring samples): high on busy backgrounds such as
    foliage, a bookshelf or printed text, low on walls, skin and clothing."""
    reg = _luma_region(luma, box, W, H)
    if reg is None:
        return None
    arr, y0, y1, x0, x1, scale = reg
    if y1 - y0 < 2 or x1 - x0 < 2:
        return 0.0
    shares = []
    for t in times:
        a = arr[_luma_index(arr, t), y0:y1, x0:x1].astype(np.float32) / scale
        gx = np.abs(np.diff(a, axis=1))[:-1, :]
        gy = np.abs(np.diff(a, axis=0))[:, :-1]
        shares.append(float((np.hypot(gx, gy) > TEXTURE_EDGE).mean()))
    return float(np.mean(shares))


def _issue(code: str, msg: str, at=None, page=None) -> tuple:
    return (code, msg, at, page)


def build_track(plan: dict, clip: dict | None, preset: dict, fmt: str, out_size, fps, luma,
                *, project_root=None, safezone: dict | None = None) -> CaptionTrack:
    """Build the caption strip for one clip x format x style (contract 9.3).

    `luma`: uint8 array [n, h, w] sampled at 5 fps from the base, or None. `project_root` adds
    P/fonts to the font search path. `safezone` overrides the profile named by the clip/preset.
    Problems that do not stop the render (a page shorter than min_page, a shrunk word, a small
    hook card, ...) are listed in `issues` (captions.json) with a code, message and out time.
    """
    clip = clip or {}
    style = resolve_style(preset, fmt)
    cap, lay = style["caption"], style["caption"]["layout"]
    W, H = int(out_size[0]), int(out_size[1])
    num, den = _fps(fps)
    frames = int(plan.get("frames_out") or 0)
    duration = float(plan.get("duration_out") or Fraction(frames * den, num))
    warnings: list[str] = []
    raw_issues: list[tuple] = []          # (code, msg, at_out, page object or None)

    def issue(code, msg, at=None, page=None, warn=True):
        raw_issues.append(_issue(code, msg, at, page))
        if warn:
            warnings.append(msg)

    geo = _geometry(style, clip, fmt, W, H, project_root, safezone)
    for code, msg in geo["notes"]:
        issue(code, msg)
    zone_name, cx, cy = geo["zone_name"], geo["cx"], geo["cy"]
    max_width, bk, pad, hb = geo["max_width"], geo["bk"], geo["pad"], geo["hb"]

    setter = _Setter(cap, project_root)
    lead = setter_lead(style)

    # Hook card.
    hook_src = plan.get("hook_card") or clip.get("hook_card")
    src_lines = hook_lines(hook_src)
    hook = None
    if src_lines:
        dur = min(hook_duration(hook_src, src_lines), duration)
        hook = {"lines": src_lines, "src_lines": src_lines, "dur": dur}
        gone = [ch for ln in src_lines for ch in typeset.strip_invisible(ln)[1]]
        if gone:
            issue("glyph_stripped", "removed invisible %s from the hook card"
                  % typeset.describe_chars(list(dict.fromkeys(gone))), 0.0)
    hook_dur = hook["dur"] if hook else 0.0

    # Tokens. Every visible word feeds the SRT sidecar; the burned captions skip the words
    # spoken under the hook card, so the first page after the card starts on a fresh phrase.
    tok_notes: list = []
    all_tokens = display_tokens(plan, clip, cap, tok_notes)
    for code, msg, at in tok_notes:
        issue(code, msg, at)
    tokens = list(all_tokens) if clip.get("captions", True) is not False else []
    if hook_dur > 0 and tokens:
        tokens = [t for t in tokens if (t.s + t.e) / 2.0 >= hook_dur - lead]
        if tokens:
            tokens[0].sentence_start = True

    # Emphasis, pages.
    emph_cfg = cap.get("emphasis") or {}
    sources = list(emph_cfg.get("sources") or [])
    per_page = emph_cfg.get("per_page", 1)
    phrases = accent_phrases(src_lines)
    manual = {int(i) for i in (clip.get("emphasis") or [])}
    mark_emphasis(tokens, phrases, manual, sources)
    join_ks = sorted(int(j["k"]) for j in plan.get("joins") or []
                     if j.get("kind") in HARD_JOINS and j.get("k") is not None)
    seams = _Seams(plan, num, den, frames, duration)
    pages = paginate(tokens, setter, max_width, lay, join_ks, lead, seams)
    pages = schedule(pages, setter, max_width, lay, lead, hook_dur, duration, join_ks, seams)
    for p in pages:
        cap_emphasis(p, sources, per_page)
    if "auto" in sources:
        _auto_emphasis(pages, setter, max_width, per_page)

    # Frame spans. A page lives between the hard joins around its words (schedule clamps the
    # seconds; here the frames): it starts on the join's own frame at the earliest and its last
    # frame is the one before the next join, so no caption shows over the far side of a cut.
    # A page due less than one frame after its cut (the first word is spoken a lead and a few
    # milliseconds in) starts on the cut frame too: rounding it up to the next frame would leave
    # the cut frame with no caption at all, a one-frame blink between two pages.
    kept = []
    frame_s = float(Fraction(den, num))
    for p in pages:
        lo_f, hi_f = seams.span(p.tokens[0].seg0, p.tokens[-1].seg1)
        p.lo, p.hi = seams.of(p.tokens)
        p.start_f = min(frames, max(lo_f, _frame_ceil(p.start, num, den)))
        at_cut = p.lo > 0 and p.lo >= hook_dur - 1e-9 and p.start - p.lo < frame_s - 1e-6
        if p.start <= p.lo + 1e-6 or at_cut:
            p.start_f = min(frames, lo_f)       # held at the join: exactly its frame
        p.end_f = min(frames, hi_f, _frame_ceil(p.end, num, den))
        if p.end_f > p.start_f:
            kept.append(p)
    pages = kept

    # Layout. A build reveal anchors the first line (a page growing to two lines does not jump);
    # the block is then kept inside the safe zone vertically (caption_y can push it out).
    reveal = (cap.get("reveal") or "fill").lower()
    anchor = str(lay.get("anchor") or ("top" if reveal == "build" else "center")).lower()
    if anchor not in ("top", "center", "bottom"):
        anchor = "center"
    mode = geo["mode"]               # the clip's caption_backing, else the preset's mode

    def layout_all(y):
        return [_layout_page(p, setter, max_width, cx, y, anchor) for p in pages]

    notes = layout_all(cy)
    if pages:
        reach_y = max(int(pad[1]), 4) if geo["pill_room"] else 0
        top = min(p.box[1] for p in pages) - reach_y
        bottom = max(p.box[1] + p.box[3] for p in pages) + reach_y
        lo_y, hi_y = geo["y_lim"]
        dy = 0.0
        if bottom > hi_y:
            dy = hi_y - bottom
        if top + dy < lo_y:
            dy = lo_y - top
        if abs(dy) >= 1.0:
            notes = layout_all(cy + dy)
            issue("caption_y_clamped", f"caption pages moved from y {cy:.0f} to {cy + dy:.0f} to "
                                       f"stay inside the {zone_name} safe zone "
                                       f"({lo_y:.0f}-{hi_y:.0f})")
            cy += dy
    for p, ns in zip(pages, notes):
        for code, msg in ns:
            if code:
                issue(code, msg, p.start, p)
            else:
                warnings.append(msg)

    # Glyph coverage: every character that will be drawn, per font file.
    need: dict[str, set] = {}
    for p in pages:
        for t in p.tokens:
            f = setter.font(p.size, bool(t.emph))
            need.setdefault(str(f.path), set()).update(t.text)
    hk_layout = None
    cap_px = int(cap["size"])
    if hook:
        hk_layout = _hook_layout(hook, style, geo["hk_max"], cx, cy, geo["y_lim"], project_root,
                                 warnings, cap_px)
        need.setdefault(str(hk_layout["font"].path), set()).update(hk_layout["text"])
        if hk_layout["small"]:
            k = hk_layout["max_chars"]
            ratio = float(style["hook_card"].get("min_ratio", HOOK_MIN_RATIO))
            issue("hook_small", f"hook card sets at {hk_layout['size']} px, under {ratio:g} x the "
                                f"{cap_px} px captions" + (f": shorten its lines to <= {k} characters"
                                                           if k else ""), 0.0)
    problems = []
    for path, chars in need.items():
        miss = typeset.missing_glyphs(path, "".join(sorted(chars)))
        if miss:
            problems.append(f"{Path(path).name} has no glyph for {typeset.describe_chars(miss)}")
    if problems:
        raise env.HCError("caption text uses characters the font cannot draw: " + "; ".join(problems)
                        + ". Respell them with the clip's \"fix\" map or pick a font that covers them.")

    # Legibility. A page needs a pill when its text, with the shadow it is drawn with, would fall
    # under min_contrast on the page's own p90 background (shadow-aware: the halo darkens what is
    # right behind the glyphs), or when that background is busy (edge density). The decision is
    # held per shot (one framing, see _shot_bounds), so the pill never blinks from page to page or
    # across a jump cut: a shot is pilled when SHOT_NEED_SHARE of its pages (the hook card counts
    # as a page of the first shot) need it, or any of them is severe (SEVERE_CONTRAST,
    # SEVERE_TEXTURE). One marginal page does not pill twenty.
    pill_rgba = parse_color(bk.get("color", "#0B0B0DD9"))
    text_rgb = parse_color(cap.get("color"))[:3]
    min_contrast = float(bk.get("min_contrast", PILL_CONTRAST))
    max_edges = float(bk.get("max_edges", TEXTURE_SHARE))
    outline = cap.get("outline") or {}
    o_rgb = parse_color(outline.get("color", "#000000"))[:3]
    o_luma = (0.2126 * o_rgb[0] + 0.7152 * o_rgb[1] + 0.0722 * o_rgb[2]) / 255.0

    def judge(box, times, rgb, font):
        p90 = _p90(luma, box, times, W, H)
        edges = _edge_density(luma, box, times, W, H)
        eff = None
        if p90 is not None:
            s = typeset.shadow_strength(font, cap.get("shadow") or [], int(outline.get("px") or 0),
                                        o_luma)
            eff = contrast_ratio(rgb, p90 * (1.0 - s))
        reason = None
        if eff is not None and eff < min_contrast:
            reason = "contrast"
        elif edges is not None and edges > max_edges:
            reason = "texture"
        return p90, edges, eff, reason

    bounds = _shot_bounds(plan)

    def shot_of(t):
        return bisect_right(bounds, t + 1e-9)

    tally: dict = {}                         # shot -> [pages, needing, severe]

    def count(shot, reason, eff, edges):
        t = tally.setdefault(shot, [0, 0, False])
        t[0] += 1
        if reason:
            t[1] += 1
            t[2] = t[2] or (eff is not None and eff < SEVERE_CONTRAST) or (
                edges is not None and edges > SEVERE_TEXTURE)

    effs = []
    for p in pages:
        d = p.end - p.start
        times = [p.start + d * f for f in (1 / 6, 0.5, 5 / 6)]
        p.p90, p.edges, eff, p.reason = judge(p.box, times, text_rgb, setter.font(p.size))
        effs.append(eff)
        p.need = p.reason is not None
        count(shot_of((p.start + p.end) / 2.0), p.reason, eff, p.edges)
    if hook and hk_layout:
        hmode = geo["hook_mode"]     # the hook card follows the clip's caption_backing too
        times = [hook_dur * f for f in (1 / 6, 0.5, 5 / 6)]
        hook_rgb = parse_color(style["hook_card"].get("color", cap.get("color")))[:3]
        hp90, hedges, heff, hreason = judge(hk_layout["box"], times, hook_rgb, hk_layout["font"])
        if hmode == "auto":
            count(shot_of(0.0), hreason, heff, hedges)
    shot_need = {k: bool(t[2] or (t[0] and t[1] / t[0] >= SHOT_NEED_SHARE - 1e-9))
                 for k, t in tally.items()}
    if hook and hk_layout:
        hbacking = hmode == "always" or (hmode == "auto" and bool(shot_need.get(shot_of(0.0))))
        hook.update({
            "size": hk_layout["size"], "caption_size": cap_px, "box": hk_layout["box"],
            "backing": hbacking, "rendered": hk_layout["lines_rendered"],
            "p90": hp90,
            "contrast": None if hp90 is None else (
                contrast_ratio(hook_rgb, hp90, pill_rgba) if hbacking else heff),
            "_pill": _pill_box(hk_layout["typo"], hk_layout["box"], hb.get("pad") or pad, W, H),
            "_dur_us": int(round(hook_dur * 1_000_000)),
            "_fade_us": int(round(min(float(style["hook_card"].get("fade_out_ms", 260)) / 1000.0,
                                      max(0.0, hook_dur - float(Fraction(den, num))))
                                  * 1_000_000)),
        })
    # The emphasis pop is drawn in a build reveal only (see CaptionTrack._raw_state).
    pop_peak = max(s for _, s in POP_KEYS) if (reveal == "build"
                                               and (cap.get("anim") or {}).get("pop")) else 1.0
    for p, eff in zip(pages, effs):
        if mode == "always":
            p.backing, p.reason = True, "always"
        elif mode == "never":
            p.backing = False
        else:
            p.backing = bool(shot_need.get(shot_of((p.start + p.end) / 2.0)))
            if p.backing and not p.need:
                p.reason = "shot"
        p.forced = bool(p.need and mode == "auto")
        p.pill = _pill_box(p.typo, p.box, pad, W, H)
        # Build reveal: the pill grows token by token and follows the ink (it used to be the
        # cap-height box alone, so an accent, a descender or a larger emphasis word stood
        # outside it). The pill of the last token is the page's pill.
        inks = _reveal_inks(p, setter, pop_peak)
        p.token_pills = [_pill_box(t, _xywh(_union_box([t, ink])), pad, W, H) if t else None
                         for t, ink in zip(p.token_typos, inks)]
        if reveal == "build" and p.token_pills and p.token_pills[-1] is not None:
            p.pill = p.token_pills[-1]
        if p.p90 is not None:
            p.contrast = contrast_ratio(text_rgb, p.p90, pill_rgba) if p.backing else eff

    # Band: union of every page box, pill and the hook card, plus shadow/animation reach.
    reach = 2 + int(cap.get("anim", {}).get("rise_px", 0) or 0)
    reach += max([int(math.ceil(3 * float(l.get("blur", 0)))) + abs(int(l.get("dy", 0)))
                  for l in cap.get("shadow") or []] or [0])
    reach += int((cap.get("outline") or {}).get("px") or 0)
    boxes = []
    for p in pages:
        grow = reach + (int(0.1 * p.size) if (cap.get("anim") or {}).get("pop") else 0)
        x, y, w, h = p.box
        boxes.append((x - grow, y - grow, x + w + grow, y + h + grow))
        if p.backing:
            for x, y, w, h in [p.pill] + ([b for b in p.token_pills if b] if reveal == "build" else []):
                boxes.append((x, y, x + w, y + h))
    if hook:
        x, y, w, h = hook["box"]
        boxes.append((x - reach, y - reach, x + w + reach, y + h + reach))
        if hook["backing"]:
            x, y, w, h = hook["_pill"]
            boxes.append((x, y, x + w, y + h))
    if boxes:
        top = max(0, _even_floor(min(b[1] for b in boxes)))
        bottom = min(_even_floor(H), _even_ceil(max(b[3] for b in boxes)))
        if bottom <= top:
            bottom = min(_even_floor(H), top + 2)
    else:
        top = min(max(0, _even_floor(cy)), max(0, _even_floor(H) - 2))
        bottom = top + 2
    band_y, band_h = top, bottom - top

    # Pre-rendered glyph tiles (positions in frame coords).
    outline_px = int((cap.get("outline") or {}).get("px") or 0)
    tiles = []
    for p in pages:
        cap_h = typeset.cap_height(setter.font(p.size))
        row = []
        for t, (x, baseline), runs in zip(p.tokens, p.pos, p.runs):
            font = setter.font(p.size, bool(t.emph))
            if len(runs) > 1:               # a merged token wrapped over two lines: one tile
                row.append(_tile_runs(font, runs, setter.tracking, outline_px, cap_h))
            else:                           # the text as laid out (t.text for a whole token)
                row.append(_tile(font, runs[0][0] if runs else t.text, setter.tracking, x,
                                 baseline, outline_px, cap_h))
        tiles.append(row)

    # Accent report (G14): a phrase counts only when an emphasized token shows on a page.
    shown = {t.phrase for p in pages for t in p.tokens if t.emph == "hook" and t.phrase is not None}
    accents = {"phrases": phrases, "matched": [ph for i, ph in enumerate(phrases) if i in shown]}
    missing = [ph for i, ph in enumerate(phrases) if i not in shown]
    if missing:
        issue("accent_unshown", "hook accent never shown in captions: "
              + ", ".join(repr(m) for m in missing))

    # Pages still under min_page: a target, not a guarantee (fast speech boxed in by seams).
    min_page = float(lay.get("min_page") or 0.0)
    for p in pages:
        shown_s = (p.end_f - p.start_f) * den / num
        if min_page > 0 and shown_s < min_page - 1e-6:
            issue("min_page", f"page '{' '.join(t.text for t in p.tokens)}' shows for "
                              f"{shown_s:.2f} s (min_page {min_page:g} s)", p.start, p, warn=False)

    index = {id(p): i for i, p in enumerate(pages)}
    issues = [{"code": code, "msg": msg, "at_out": None if at is None else round(float(at), 3),
               "page": None if pg is None else index.get(id(pg))}
              for code, msg, at, pg in raw_issues]

    n_pill = sum(1 for p in pages if p.backing)
    env.log("captions", f"{len(pages)} pages, {n_pill} with backing, band y={band_y} h={band_h}"
         + (f", hook {hook['size']} px" if hook else ""))
    for wmsg in warnings:
        env.log("captions", "warning: " + wmsg)
    short = sum(1 for i in issues if i["code"] == "min_page")
    if short:
        env.log("captions", f"note: {short} page(s) under min_page (listed in captions.json issues)")

    return CaptionTrack(width=W, band_y=band_y, band_h=band_h, frames=frames, fps=(num, den),
                        pages=pages, hook=hook, style=style, fmt=fmt, setter=setter, tiles=tiles,
                        hook_tiles=hk_layout["tiles"] if hk_layout else [], accents=accents,
                        meta={"height": H, "safezone": zone_name, "max_width": round(max_width, 1),
                              "warnings": warnings, "duration": duration, "backing_mode": mode,
                              "backing_override": geo["override"]},
                        srt_tokens=all_tokens, issues=issues, seams=seams)
