"""QA gates G0-G16 (contract section 10).

Every gate reports PASS / WARN / FAIL / SKIPPED with evidence. "Could not check" is never PASS:
a gate that cannot run is SKIPPED with its reason, and when the missing piece is something the
check needed (an artefact, a tool, the ASR pass) it is an evidence gap. A clip whose gates all
PASS but that has gaps is INCOMPLETE, and gaps fail the run under --require-coverage (exit 2).

Exit codes: 0 no FAIL, 1 any FAIL, 2 evidence gap with --require-coverage, 3 tool error (an ASR
backend crash counts as one).
"""

from __future__ import annotations

import bisect
import difflib
import inspect
import math
import re
import shlex
import subprocess
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

import numpy as np

from . import env
from . import textnorm as tn

STATUSES = ("PASS", "WARN", "FAIL", "SKIPPED")
SEVERITY = {"PASS": 0, "SKIPPED": 1, "WARN": 2, "FAIL": 3}
CLIP_STATUSES = ("PASS", "WARN", "FAIL", "INCOMPLETE")
GATES = {
    "G0": "render fresh", "G1": "plan lints", "G2": "duration", "G3": "streams", "G5": "word diff",
    "G6": "open/close", "G7": "loudness", "G8": "black/freeze", "G9": "caption layout",
    "G10": "caption sync", "G11": "faithfulness", "G12": "cover", "G13": "set",
    "G14": "accent fired", "G15": "render log", "G16": "framing attribution",
}
TARGET_LUFS = -14.0
LUFS_TOL = 1.0
MAX_TRUE_PEAK = -1.0
MAX_ODD_AAC_PACKETS = 5
PTS_JITTER_SAMPLES = 2           # G3: an audio packet this many samples off the grid is not a hole
BLACK_FILTER = "blackdetect=d=0:pix_th=0.06"     # d=0: a single black frame anywhere is seen
FREEZE_FILTER = "freezedetect=n=0.003:d=1.0"
JOIN_WINDOW_FRAMES = 3
MID_SOUND_DB = 20.0
OPEN_WIN = 0.03
CLOSE_FADE = 0.06
CLOSE_WIN = 0.06
FLOOR_PCT = 10
FLOOR_MIN_DB = -70.0             # digital silence never makes the floor lower than this
SYNC_MEDIAN_WARN = 0.15
SYNC_DRIFT_FAIL = 0.10
SYNC_MIN_PAIRS = 6
SET_OVERLAP_S = 15.0
DUP_MIN_SHARED = 2               # G13 near-duplicate hooks: shared stemmed content tokens ...
DUP_JACCARD = 0.5                # ... and Jaccard, or
DUP_OVERLAP = 0.67               # ... overlap coefficient with both sets at least DUP_MIN_SET
DUP_MIN_SET = 3
ECHO_OVERLAP = 0.8
CLOSEUP_SHIFT_MAX = 80    # cover.CLOSEUP_SHIFT_MAX: a close-up cover may shift this far (G12)
ECHO_MIN_SHARED = 2       # one shared word ("SELL" on both) is not an echo
FACE_LOWER = 0.6
MOUTH_CLEAR = 0.35        # below the mouth line, in eye-to-mouth distances (lower lip + margin)
CONTENT_MIN_LETTERS = 4
SILENT_VOICED_FRAC = 0.15        # a word span with < 15 % voiced hops is silence
UNCAPTIONED_WORDS = 6            # G5: this many consecutive extra content words = uncaptioned speech
# G10 caption timing, measured on the render's own audio (no ASR model involved)
G10_PAUSE = 0.25                 # pauses at least this long are timing anchors
G10_ONSET_SLACK = 0.15           # a token shown inside a pause with this much pause left is early
G10_SPAN_PAUSE = 0.20            # a token whose span holds this much pause is mis-timed
G10_PAGE_EARLY = 0.6             # a page this long before the next voiced onset FAILs
G10_FAR = 0.5                    # tokens at least this far off ...
G10_FAR_COUNT = 3                # ... this many of them FAIL
G10_UNCOVERED_WARN = 1.5         # voiced render audio with no caption token near it
G10_UNCOVERED_FAIL = 3.0
G10_UNCOVERED_ASR_WORDS = 4      # ... FAIL when ASR also heard this many extra words inside it
G10_LAG_WARN = 0.12              # display lag (cross-correlation) thresholds
G10_LAG_FAIL = 0.25
G10_LAG_MAX = 1.0
G10_LAG_MIN_SPAN = 2.0           # seconds of audio after the hook card needed for the lag
G10_LEAD_RANGE = (0.0, 0.20)     # a caption lead outside this window is a preset defect
G10_SPEECH_EARLY = 0.20          # a token shown this long before the first speech-like sound
G10_SPEECH_LOOK = 1.5            # ... looked for within this many seconds of the token
CLOSE_ATTACH = 0.08              # G6: a final consonant follows a voiced sound within this
DEAD_AIR = 0.35                  # G6: first voiced audio this long after a segment start
DEAD_AIR_OPEN = 0.25             # ... at out 0
FLASH_JUMP = 40.0                # G8 single-frame flash: band-masked mean luma (0-255)
FLASH_SAME = 15.0
DARK_LUMA = 20                   # G8 / G12 dark rows: max luma at most this over the row
DARK_MIN_FRAC = 0.03             # a dark band taller than this fraction of the height is reported
DARK_FAIL_FRAC = 0.10
DARK_SAMPLES = 8
LUMA_SCALE = 8                   # luma scans decode at 1/8 size
GRID_BAND = (240, 1680)          # the platform's 3:4 profile-grid crop of a 9:16 cover
COVER_TINT_LEAD = 300            # G12 dark band: the cover tint ramp starts this far above the headline
COVER_TINT_FROM = 0.42           # ... or at this fraction of the height without a headline box
WINDOW_SLACK = 0.02              # G16: a face x this close outside the window still counts
G16_WINDOW_MODES = ("crop", "rect")   # G16: windows that can show one face of a multi-face shot
G16_MERGE_GAP = 0.5
G16_FAIL_S = 1.5
QA_VERSION = 2
QA_CODE_CHANGED = "the QA checks changed since this report was made"     # qa_staleness reason
ASR_CACHE_VERSION = 2
CAPTION_ISSUES_WARN = ("min_page", "word_shrunk", "overflow", "hook_small", "caption_y_clamped")


# ---------------------------------------------------------------- result helpers

def gate(gid: str, status: str, evidence: str, details=None, *, target: str = "clip",
         reason: str | None = None, gap: bool = False) -> dict:
    """One gate result. `gap` marks a SKIPPED (or partial) check that lacked needed evidence."""
    assert status in STATUSES, status
    out = {"id": gid, "name": GATES.get(gid, gid), "target": target, "status": status,
           "evidence": evidence, "details": list(details or [])}
    if status == "SKIPPED":
        out["reason"] = reason or evidence
    if gap:
        out["gap"] = True
    return out


def worst(statuses) -> str:
    """Most severe status (FAIL > WARN > SKIPPED > PASS); PASS for an empty list."""
    best = "PASS"
    for s in statuses:
        if SEVERITY[s] > SEVERITY[best]:
            best = s
    return best


def clip_status(results: list[dict], gaps: int) -> str:
    """FAIL / WARN from the gates; PASS with evidence gaps is INCOMPLETE, never PASS."""
    st = worst(r["status"] for r in results if r["status"] != "SKIPPED")
    if st == "PASS" and gaps > 0:
        return "INCOMPLETE"
    return st


def _fmt_t(t) -> str:
    return "-" if t is None else f"{float(t):.2f}s"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _q(path) -> str:
    return shlex.quote(str(path))


# ---------------------------------------------------------------- subprocess helpers

def _run(cmd: list, timeout: float | None = None) -> tuple[int, bytes, str]:
    """Run an analysis command; returns (rc, stdout, stderr text). Missing binary -> HCError 3."""
    try:
        p = subprocess.run([str(c) for c in cmd], stdin=subprocess.DEVNULL, capture_output=True,
                           timeout=timeout)
    except FileNotFoundError as exc:
        raise env.HCError(f"cannot run {cmd[0]}: {exc}", code=3)
    except subprocess.TimeoutExpired:
        return 124, b"", "timed out"
    return p.returncode, p.stdout, p.stderr.decode("utf-8", "replace")


def probe(path: Path) -> dict:
    """ffprobe streams + format of a render (raises HCError when unreadable)."""
    return env.ffprobe_json(path, "-show_streams", "-show_format")


def _stream(info: dict, kind: str) -> dict | None:
    for s in info.get("streams") or []:
        if s.get("codec_type") == kind:
            return s
    return None


def _frac(v) -> Fraction | None:
    try:
        if isinstance(v, (list, tuple)):
            return Fraction(int(v[0]), int(v[1]))
        n, _, d = str(v).partition("/")
        return Fraction(int(n), int(d or 1))
    except (ValueError, ZeroDivisionError, TypeError):
        return None


def decode_mono(path: Path, sr: int = 16000) -> np.ndarray:
    """First audio stream as mono float32 at `sr` (empty array when there is no audio)."""
    rc, out, err = _run([env.ffmpeg(), "-hide_banner", "-nostdin", "-loglevel", "error", "-i", str(path),
                         "-map", "0:a:0", "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"])
    if rc != 0:
        raise env.HCError(f"cannot decode audio of {Path(path).name}: {err.strip()[-200:]}", code=2)
    return np.frombuffer(out, np.float32).copy()


def rms_db(x: np.ndarray) -> float:
    if x.size == 0:
        return -120.0
    p = float(np.mean(np.square(x, dtype=np.float64)))
    return max(-120.0, 10.0 * math.log10(p + 1e-12))


def hop_db(x: np.ndarray, sr: int = 16000, hop: float = 0.01) -> np.ndarray:
    n = int(sr * hop)
    k = x.size // n
    if k == 0:
        return np.array([rms_db(x)])
    seg = x[: k * n].reshape(k, n).astype(np.float64)
    return np.maximum(-120.0, 10.0 * np.log10(np.mean(seg * seg, axis=1) + 1e-12))


def speech_track(path, x: np.ndarray | None = None):
    """speechband.Track of a file's first audio stream (or of mono 16 kHz samples `x`); None
    when there is no readable audio."""
    from . import speechband
    try:
        if x is None:
            x = decode_mono(Path(path), speechband.SR)
    except env.HCError:
        return None
    if x is None or x.size == 0:
        return None
    return speechband.analyse(x, speechband.SR)


# ---------------------------------------------------------------- ASR (G5 / G10)

def _asr_words(project, render: Path, transcript: dict) -> list[dict]:
    """Re-transcribe a render with the transcript's backend/model/glossary (cached).

    Returns [{"w", "s", "e"}] on the render's out clock.
    """
    from . import transcribe

    fn = getattr(transcribe, "transcribe_media", None)
    if not callable(fn):
        raise env.HCError("transcribe.transcribe_media is not available", code=3)
    opts = {"backend": transcript.get("backend"), "model": transcript.get("model"),
            "language": transcript.get("language"), "glossary": transcript.get("glossary")}
    km = getattr(transcribe, "asr_key_material", None)
    try:
        material = km(opts["backend"], opts["model"], opts["language"], opts["glossary"]) if callable(km) else opts
    except Exception:
        material = opts
    key = env.stable_hash({"v": ASR_CACHE_VERSION, "fp": env.fingerprint(render), "asr": material,
                           "post": getattr(transcribe, "POST_VERSION", 0)})[:20]
    cache = Path(project.cache_path(f"qa-asr-{key}.json"))
    if cache.is_file():
        try:
            data = env.read_json(cache)
            if isinstance(data.get("words"), list):
                return data["words"]
        except env.HCError:
            pass
    try:
        params = inspect.signature(fn).parameters
        takes_kw = any(p.kind == p.VAR_KEYWORD for p in params.values())
        kwargs = {k: v for k, v in opts.items() if v is not None and (takes_kw or k in params)}
    except (TypeError, ValueError):
        kwargs = {k: v for k, v in opts.items() if v is not None}
    res = fn(render, **kwargs)
    raw = res.get("words") if isinstance(res, dict) else res
    words = []
    for w in raw or []:
        if not isinstance(w, dict):
            continue
        text = w.get("w", w.get("word", w.get("text")))
        s = w.get("s", w.get("start"))
        e = w.get("e", w.get("end", s))
        if text is None or s is None:
            continue
        words.append({"w": str(text).strip(), "s": float(s), "e": float(e if e is not None else s)})
    env.write_json(cache, {"render": str(render), "opts": opts, "words": words})
    return words


def _toks(texts: list[str]) -> list[tuple[str, int]]:
    """Normalized tokens (contractions expanded, numbers canonical, fillers dropped) with the
    index of the word each came from."""
    out = []
    for tok, a, _b in tn.norm_stream(texts, expand=True):
        if not tok or tn.is_filler(tok):
            continue
        out.append((tok, a))
    return out


def _is_number(tok: str) -> bool:
    return bool(re.fullmatch(r"\d+(?:\.\d+)?", tok))


def _is_negation(tok: str) -> bool:
    return tok in tn.NEGATIONS


def _is_content(tok: str) -> bool:
    if _is_number(tok):
        return True
    return tok.isalpha() and len(tok) >= CONTENT_MIN_LETTERS and tok not in tn.STOPWORDS


def diff_words(ref_texts: list[str], hyp_texts: list[str], silent=None, *, dup=None,
               hyp_times=None) -> dict:
    """Align intended words with re-transcribed words (difflib on normalized tokens).

    Rules (ported from the word-diff harness): a missing negation or content word (>= 4
    letters, not a stopword, or a number) is FAIL unless it is a stutter repeat whose twin
    survived; other missing words are soft (WARN); replacements and extra words are WARN,
    except a replacement that loses a negation, which is FAIL. `silent(hyp_word_idx)` (optional)
    says whether the render is silent where ASR placed a word: a replacement heard only in
    silence is an ASR hallucination over a gap, so its intended words count as missing.

    `dup` maps intended word positions (index into `ref_texts`) that two overlapping pieces
    both play to a label ("pieces 1/2"): a missing copy there is FAIL "duplicated at join",
    never a stutter or a plain miss. UNCAPTIONED_WORDS or more extra content words heard in a
    row are FAIL "uncaptioned speech" (speech in the render that the captions do not carry);
    `hyp_times[hyp_word_idx]` gives their out time for the message.
    Returns {"fail": [...], "warn": [...], "pairs": [(ref_word_idx, hyp_word_idx)], "ref_n",
    "hyp_n", "matched", "extra": [hyp_word_idx heard but not intended], "dup_missing": n}.
    """
    ref = _toks(ref_texts)
    hyp = _toks(hyp_texts)
    rt = [t for t, _ in ref]
    ht = [t for t, _ in hyp]
    hyp_set = frozenset(ht)
    dup = dup or {}
    sm = difflib.SequenceMatcher(None, rt, ht, autojunk=False)
    fails, warns, pairs, extra = [], [], [], set()
    matched = 0
    dup_missing = 0

    def ctx(i: int) -> str:
        lo, hi = max(0, i - 4), min(len(rt), i + 5)
        return " ".join(f"[{rt[k]}]" if k == i else rt[k] for k in range(lo, hi))

    def at(j: int) -> str:
        if hyp_times is None or not (0 <= j < len(hyp)):
            return ""
        try:
            return f" at out {float(hyp_times[hyp[j][1]]):.2f}s"
        except (IndexError, KeyError, TypeError, ValueError):
            return ""

    def judge_missing(i: int):
        nonlocal dup_missing
        tok = rt[i]
        label = dup.get(ref[i][1])
        if label:
            dup_missing += 1
            fails.append(f'duplicated at join ({label} share word "{tok}"): one copy is not heard: {ctx(i)}')
            return
        stutter = tok in hyp_set and ((i > 0 and rt[i - 1] == tok) or (i + 1 < len(rt) and rt[i + 1] == tok))
        if _is_negation(tok):
            fails.append(f'missing negation "{tok}": {ctx(i)}')
        elif _is_content(tok) and not stutter:
            fails.append(f'missing "{tok}": {ctx(i)}')
        else:
            why = "stutter repeat" if stutter else "function word"
            warns.append(f'soft missing "{tok}" ({why}): {ctx(i)}')

    def judge_extra(j1: int, j2: int, after: int, n_content: int):
        words = " ".join(ht[j1:j2])
        if n_content >= UNCAPTIONED_WORDS:
            fails.append(f"uncaptioned speech: {n_content} words heard{at(j1)} are not in the captions: "
                         f'"{words[:120]}"')
        else:
            warns.append(f'extra "{words}" heard after: {ctx(max(after, 0)) if rt else ""}')

    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal":
            matched += i2 - i1
            seen = set()
            for k in range(i2 - i1):
                ri, hj = ref[i1 + k][1], hyp[j1 + k][1]
                if ri not in seen:
                    pairs.append((ri, hj))
                    seen.add(ri)
        elif op == "delete":
            for i in range(i1, i2):
                judge_missing(i)
        elif op == "insert":
            extra.update(hyp[j][1] for j in range(j1, j2))
            judge_extra(j1, j2, i1 - 1, sum(1 for t in ht[j1:j2] if _is_content(t)))
        elif op == "replace":
            rs, hs = rt[i1:i2], ht[j1:j2]
            if any(_is_negation(t) for t in rs) and not any(_is_negation(t) for t in hs):
                neg = next(t for t in rs if _is_negation(t))
                fails.append(f'negation "{neg}" lost ("{" ".join(rs)}" heard as "{" ".join(hs)}")')
                continue
            if silent is not None and all(silent(hyp[j][1]) for j in range(j1, j2)):
                warns.append(f'"{" ".join(hs)}" was heard where the render is silent (ASR filling a gap)')
                for i in range(i1, i2):
                    judge_missing(i)
                continue
            surplus = len(rs) - len(hs)
            if surplus > 0:
                sims = []
                for k, t in enumerate(rs):
                    best = max((difflib.SequenceMatcher(None, t, h).ratio() for h in hs), default=0.0)
                    sims.append((best, k))
                for _, k in sorted(sims)[:surplus]:
                    judge_missing(i1 + k)
            elif surplus < 0:
                # more heard than intended: the unmatched heard words are extra speech
                odd = [j for j in range(j1, j2)
                       if max((difflib.SequenceMatcher(None, ht[j], t).ratio() for t in rs), default=0.0) < 0.6]
                extra.update(hyp[j][1] for j in odd)
                n_content = sum(1 for j in odd if _is_content(ht[j])) - sum(1 for t in rs if _is_content(t))
                if n_content >= UNCAPTIONED_WORDS:
                    judge_extra(j1, j2, i1, n_content)
                    continue
            warns.append(f'heard "{" ".join(rs)}" as "{" ".join(hs)}"')
    return {"fail": fails, "warn": warns, "pairs": pairs, "ref_n": len(rt), "hyp_n": len(ht),
            "matched": matched, "extra": sorted(extra), "dup_missing": dup_missing}


def theil_sen(x, y) -> float | None:
    """Median of pairwise slopes (pairs at least 0.5 s apart); None with too few points."""
    x = np.asarray(x, np.float64)
    y = np.asarray(y, np.float64)
    if x.size > 300:
        idx = np.linspace(0, x.size - 1, 300).round().astype(int)
        x, y = x[idx], y[idx]
    dx = x[None, :] - x[:, None]
    dy = y[None, :] - y[:, None]
    mask = np.triu(dx > 0.5, 1)
    if not mask.any():
        return None
    return float(np.median(dy[mask] / dx[mask]))


# ---------------------------------------------------------------- faithfulness helpers

def surface_numbers(text: str) -> list[tuple[float, str]]:
    """Numbers in a burned surface as (value, raw words). A bare "one"/"a" is not a claim."""
    raws = (text or "").replace("*", " ").split()
    out = []
    for tok, a, b in tn.norm_stream(raws):
        if not _is_number(tok):
            continue
        raw = " ".join(raws[a:b + 1])
        if b == a and tn.norm(raws[a]) in ("1",) and re.sub(r"[^a-z]", "", raws[a].lower()) in ("one", "a"):
            continue
        out.append((float(tok), raw))
    return out


def spoken_numbers(texts: list[str]) -> set[float]:
    return {float(t) for t, _, _ in tn.norm_stream(texts) if _is_number(t)}


def _proper_noun_evidence(words: list[dict]) -> dict:
    """norm token -> [capitalized_count, lowercase_count] over non-sentence-initial transcript words."""
    ev: dict = {}
    prev = None
    for w in words:
        raw = str(w.get("w", ""))
        core = raw.strip("\"'()[]{}.,;:!?")
        initial = prev is None or tn.is_sentence_end(prev)
        prev = raw
        if not core or not core[0].isalpha():
            continue
        key = tn.norm(core)
        if not key or initial:
            continue
        rec = ev.setdefault(key, [0, 0])
        if core[0].isupper():
            rec[0] += 1
        else:
            rec[1] += 1
    return ev


def surface_names(text: str, evidence: dict, glossary: list[str]) -> list[str]:
    """Name-like tokens of a surface: glossary terms, capitalized words mid-text in a
    mixed-case surface, or (in all-caps / title-case text) words the transcript only ever
    capitalizes mid-sentence."""
    clean = (text or "").replace("*", " ")
    names: list[str] = []
    gl = {tn.norm(g): g for g in glossary or [] if tn.norm(g)}
    words = clean.split()
    letters = [w for w in words if any(c.isalpha() for c in w)]
    content = [w for w in letters[1:] if tn.norm(w) not in tn.STOPWORDS]
    caps_like = bool(content) and sum(1 for w in content if w[:1].isupper()) >= 0.6 * len(content)
    all_caps = all(w.upper() == w for w in letters) if letters else False
    for i, w in enumerate(words):
        core = w.strip("\"'()[]{}.,;:!?")
        if not core or not core[0].isalpha():
            continue
        if core.lower().endswith("'s") or core.lower().endswith("’s"):
            core = core[:-2]
        key = tn.norm(core)
        if not key or key in tn.STOPWORDS:
            continue
        if key in gl:
            names.append(core)
            continue
        if all_caps or caps_like:
            rec = evidence.get(key)
            if rec and rec[0] > 0 and rec[1] == 0:
                names.append(core)
        elif i > 0 and core[0].isupper():
            names.append(core)
    out, seen = [], set()
    for n in names:
        k = tn.norm(n)
        if k not in seen:
            seen.add(k)
            out.append(n)
    return out


def _hook_text(plan: dict | None, clip: dict) -> str:
    hc = (plan or {}).get("hook_card") or clip.get("hook_card") or {}
    if isinstance(hc, dict) and hc.get("lines"):
        return " ".join(str(x) for x in hc["lines"])
    return ""


def accent_phrases(text_lines) -> list[str]:
    out = []
    for line in text_lines or []:
        out.extend(m.strip() for m in re.findall(r"\*([^*]+)\*", str(line)) if m.strip())
    return out


# ---------------------------------------------------------------- context

class _Ctx:
    def __init__(self, project, spec: dict, styles, asr: bool, require_coverage: bool,
                 formats=None):
        self.project = project
        self.spec = spec
        self.styles = list(styles) if styles else None
        self.formats = list(formats) if formats else None
        self.asr = asr
        self.require_coverage = require_coverage
        self.tool_errors: list[str] = []
        self._plans: dict = {}
        self._probe: dict = {}
        self._faces: dict = {}
        self._energy: dict = {}
        self._speech: dict = {}
        self._md5: dict = {}
        self._presets: dict = {}
        self._evidence = None
        self.set_issues: dict = {}
        self.flag_sheets: dict = {}      # (clip, fmt, style) -> G8 peek sheet path

    def energy(self, path: Path | None):
        """EnergyTrack of a render's audio (10 ms, out clock), cached per file; None if unreadable."""
        if path is None:
            return None
        key = str(path)
        if key not in self._energy:
            from .audio import EnergyTrack
            try:
                E = EnergyTrack.from_media(path)
                self._energy[key] = E if E.n else None
            except env.HCError:
                self._energy[key] = None
        return self._energy[key]

    def speech(self, path: Path | None):
        """speechband.Track of a render's audio (speech-like runs, out clock), cached per file;
        None when the audio cannot be read."""
        if path is None:
            return None
        key = str(path)
        if key not in self._speech:
            self._speech[key] = speech_track(path)
        return self._speech[key]

    def preset(self, style: str, fmt: str) -> dict:
        """The style's preset resolved for `fmt`, as render loads it; {} when it cannot be
        loaded (the gate that needed it then falls back to the shipped defaults). Cached."""
        key = (style, fmt)
        if key not in self._presets:
            try:
                from . import render_util as ru
                got = ru.load_preset(style, self.project, fmt)
                self._presets[key] = got if isinstance(got, dict) else {}
            except Exception:
                self._presets[key] = {}
        return self._presets[key]

    def cmd(self, *args) -> str:
        """A copy-pasteable hc command for this project."""
        return "hc " + " ".join(_q(a) for a in (args[0], self.project.root) + tuple(args[1:]))

    def file_plan(self, cid: str) -> dict | None:
        if cid not in self._plans:
            p = Path(self.project.plan_path(cid))
            try:
                self._plans[cid] = env.read_json(p) if p.is_file() else None
            except env.HCError:
                self._plans[cid] = None
        return self._plans[cid]

    def probe(self, path: Path):
        key = str(path)
        if key not in self._probe:
            try:
                self._probe[key] = (probe(path), None)
            except env.HCError as exc:
                self._probe[key] = (None, str(exc))
        return self._probe[key]

    def name_evidence(self) -> dict:
        if self._evidence is None:
            try:
                self._evidence = _proper_noun_evidence(self.project.transcript().get("words") or [])
            except Exception:
                self._evidence = {}
        return self._evidence


# ---------------------------------------------------------------- gates

def g1_lints(fresh: dict | None, err: str | None) -> dict:
    if fresh is None:
        return gate("G1", "FAIL", f"plan could not be built: {err}")
    lints = fresh.get("lints") or []
    fails = [l for l in lints if l.get("level") == "FAIL"]
    warns = [l for l in lints if l.get("level") == "WARN"]
    det = [f"{l.get('level')} {l.get('id')}: {l.get('msg')}" + (f" (at {_fmt_t(l['at_out'])})" if l.get("at_out") is not None else "")
           for l in lints if l.get("level") in ("FAIL", "WARN")]
    if fails:
        return gate("G1", "FAIL", f"{len(fails)} FAIL lint(s): " + "; ".join(f"{l.get('id')} {l.get('msg')}" for l in fails[:3]), det)
    if warns:
        return gate("G1", "WARN", f"{len(warns)} WARN lint(s): " + ", ".join(sorted({str(l.get('id')) for l in warns})), det)
    return gate("G1", "PASS", f"{len(lints)} lint(s), none WARN or FAIL")


def g0_fresh(ctx: _Ctx, cid: str, fmt: str, style: str, plan: dict | None, target: str) -> dict:
    """The render was made from the current plan and inputs: its log's hashes.plan equals the
    hash of the fresh plan, and hashes.render equals the key a render made now would record
    (clip entry, preset, the font files and safe-zone margins the captions are drawn with,
    segments and render code)."""
    lp = Path(ctx.project.render_path(cid, fmt, style, ext=".log.json"))
    if plan is None:
        return gate("G0", "SKIPPED", "no plan: render freshness not checkable", target=target, gap=True)
    if not lp.is_file():
        return gate("G0", "SKIPPED", f"render log missing ({ctx.project.rel(lp)}): freshness not checkable",
                    target=target, gap=True)
    try:
        hashes = (env.read_json(lp) or {}).get("hashes") or {}
    except env.HCError as exc:
        return gate("G0", "SKIPPED", f"render log unreadable: {exc}", target=target, gap=True)
    if not hashes.get("plan") or not hashes.get("render"):
        return gate("G0", "SKIPPED", "render log has no hashes (an older render): re-render to make it checkable",
                    target=target, gap=True)
    from . import render as render_mod
    try:
        ph = render_mod.plan_hash(plan)
    except Exception as exc:
        return gate("G0", "SKIPPED", f"plan hash not computable: {type(exc).__name__}: {exc}",
                    target=target, gap=True)
    if hashes["plan"] != ph:
        return gate("G0", "FAIL", "render predates the plan: " + ctx.cmd("render", cid, "--styles", style),
                    [f"render log plan hash {hashes['plan'][:12]}, current plan {ph[:12]}"], target=target)
    try:
        want = render_mod.expected_hashes(ctx.project, cid, fmt, style, plan=plan)["render"]
    except Exception as exc:
        return gate("G0", "PASS", f"plan hash matches ({ph[:12]}); render inputs not checkable: "
                    f"{type(exc).__name__}: {exc}", target=target, gap=True)
    if hashes["render"] != want:
        return gate("G0", "FAIL", "render predates its inputs (clip entry, preset, fonts, safe zone or code): "
                    + ctx.cmd("render", cid),
                    [f"render log key {hashes['render'][:12]}, a render made now {want[:12]}"], target=target)
    return gate("G0", "PASS", f"render made from the current plan and inputs (plan {ph[:12]})", target=target)


def g2_duration(ctx: _Ctx, plan: dict, clip: dict, fmt: str, render: Path, target: str) -> dict:
    info, err = ctx.probe(render)
    if info is None:
        return gate("G2", "FAIL", f"render unreadable: {err}", target=target)
    v = _stream(info, "video")
    if v is None:
        return gate("G2", "FAIL", "render has no video stream", target=target)
    fps = _frac(plan.get("fps")) or Fraction(25)
    fdur = float(1 / fps)
    want_frames = int(plan.get("frames_out") or 0)
    want_dur = float(plan.get("duration_out") or want_frames * fdur)
    try:
        frames = int(v.get("nb_frames"))
    except (TypeError, ValueError):
        frames = None
    if frames is None:
        try:
            p = env.ffprobe_json(render, "-select_streams", "v:0", "-count_packets", "-show_entries", "stream=nb_read_packets")
            frames = int(p["streams"][0]["nb_read_packets"])
        except Exception:
            frames = None
    try:
        vdur = float(v.get("duration"))
    except (TypeError, ValueError):
        vdur = float((info.get("format") or {}).get("duration") or 0.0)
    problems = []
    if frames is not None and abs(frames - want_frames) > 1:
        problems.append(f"{frames} frames vs plan {want_frames}")
    if abs(vdur - want_dur) > fdur + 1e-3:
        problems.append(f"video {vdur:.3f}s vs plan {want_dur:.3f}s")
    ev = f"{vdur:.3f}s, {frames if frames is not None else '?'} frames (plan {want_dur:.3f}s, {want_frames})"
    if problems:
        return gate("G2", "FAIL", "duration mismatch: " + "; ".join(problems) + " (stale or broken render; re-render)", target=target)
    band = ((clip.get("_eff") or {}).get("length") or {}).get(fmt)
    if band and len(band) == 4:
        lo, tlo, thi, hi = (float(x) for x in band)
        if want_dur < lo or want_dur > hi:
            return gate("G2", "WARN", f"{ev}; outside the {fmt} length band {lo:g}-{hi:g}s", target=target)
        if not (tlo <= want_dur <= thi):
            ev += f"; inside the band but outside the {tlo:g}-{thi:g}s target"
    return gate("G2", "PASS", ev, target=target)


def _audio_pts_gaps(render: Path, a: dict) -> tuple[int | None, float]:
    """Holes or overlaps in the audio packet timeline: (count, largest in seconds). A hole means
    every later sample plays late in a player that honours timestamps, even though a straight
    decode to PCM (what a sample-count check sees) looks perfect.

    A packet that starts PTS_JITTER_SAMPLES or fewer samples off the frame grid is not a hole
    as long as the timeline comes back to the grid (the encoder does that now and then: one
    packet a sample short, the next a sample long; a sample is 0.02 ms and nothing after it
    moves). Such steps are summed, so a timeline that keeps slipping the same way is still
    reported once the slip passes the allowance; a real break is a partial frame, hundreds of
    samples."""
    try:
        data = env.ffprobe_json(render, "-select_streams", "a:0", "-show_entries", "packet=pts,duration")
    except env.HCError:
        return None, 0.0
    tb = _frac(a.get("time_base")) or Fraction(1, int(a.get("sample_rate") or 48000))
    sr = int(a.get("sample_rate") or 48000)
    fs = int(round(Fraction(int(a.get("frame_size") or 1024), sr) / tb))   # one AAC frame, in tb
    pk = []
    for p in data.get("packets") or []:
        try:
            pk.append((int(p["pts"]), int(p["duration"])))
        except (KeyError, TypeError, ValueError):
            continue
    tol = Fraction(PTS_JITTER_SAMPLES, sr) / tb     # the allowance, in time-base ticks
    n, worst, slip = 0, 0.0, 0
    for k, (pts, dur) in enumerate(pk[:-1]):
        nxt = pk[k + 1][0]
        hole = nxt - (pts + min(dur, fs))       # mp4 stores a hole as an over-long packet duration
        if hole == 0:
            short = fs - (nxt - pts)            # ... and an early packet as a short one before it
            if 0 < short <= tol:
                slip = max(slip - short, -tol)
            continue
        if abs(hole) <= tol:
            slip += hole
            if abs(slip) <= tol:
                continue                        # a wobble around the grid: nothing has moved
            hole, slip = slip, 0                # small steps that added up: one hole of their sum
        n += 1
        worst = max(worst, abs(float(hole * tb)))
    return n, worst


def _odd_aac_packets(render: Path, a: dict) -> tuple[int | None, int]:
    try:
        data = env.ffprobe_json(render, "-select_streams", "a:0", "-show_entries", "packet=duration")
    except env.HCError:
        return None, 0
    durs = [p.get("duration") for p in data.get("packets") or []]
    tb = _frac(a.get("time_base")) or Fraction(1, int(a.get("sample_rate") or 48000))
    sr = int(a.get("sample_rate") or 48000)
    fs = int(a.get("frame_size") or 1024)
    expect = float(Fraction(fs, sr) / tb)
    odd = 0
    for d in durs[:-1]:
        try:
            if abs(float(d) - expect) > 1.0:
                odd += 1
        except (TypeError, ValueError):
            odd += 1
    return odd, len(durs)


def g3_streams(ctx: _Ctx, plan: dict, fmt: str, render: Path, target: str) -> dict:
    from .cover_frames import out_size

    info, err = ctx.probe(render)
    if info is None:
        return gate("G3", "FAIL", f"render unreadable: {err}", target=target)
    v, a = _stream(info, "video"), _stream(info, "audio")
    bad = []
    W, H = out_size(fmt)
    fps = _frac(plan.get("fps"))
    fdur = float(1 / fps) if fps else 0.04
    if v is None:
        bad.append("no video stream")
    else:
        if (int(v.get("width") or 0), int(v.get("height") or 0)) != (W, H):
            bad.append(f"size {v.get('width')}x{v.get('height')} (want {W}x{H})")
        if v.get("codec_name") != "h264":
            bad.append(f"video codec {v.get('codec_name')} (want h264)")
        if v.get("pix_fmt") != "yuv420p":
            bad.append(f"pix_fmt {v.get('pix_fmt')} (want yuv420p)")
        rate = _frac(v.get("r_frame_rate"))
        if fps and rate != fps:
            bad.append(f"fps {v.get('r_frame_rate')} (plan {fps.numerator}/{fps.denominator})")
        tags = {k: v.get(k) for k in ("color_space", "color_transfer", "color_primaries")}
        off = [f"{k}={val}" for k, val in tags.items() if val != "bt709"]
        if off:
            bad.append("colour tags not bt709: " + ", ".join(off))
    if a is None:
        bad.append("no audio stream")
    else:
        if a.get("codec_name") != "aac":
            bad.append(f"audio codec {a.get('codec_name')} (want aac)")
        if int(a.get("sample_rate") or 0) != 48000:
            bad.append(f"audio {a.get('sample_rate')} Hz (want 48000)")
        if int(a.get("channels") or 0) != 2:
            bad.append(f"audio {a.get('channels')} channel(s) (want 2)")
    av = None
    if v is not None and a is not None:
        try:
            av = abs(float(v.get("duration")) - float(a.get("duration")))
            if av > fdur + 1e-3:
                bad.append(f"audio/video durations differ by {av:.3f}s (> 1 frame)")
        except (TypeError, ValueError):
            bad.append("stream durations unavailable")
    odd = None
    if a is not None:
        odd, total = _odd_aac_packets(render, a)
        if odd is None:
            bad.append("could not read audio packets")
        elif odd > MAX_ODD_AAC_PACKETS:
            bad.append(f"{odd} of {total} AAC packets are not full frames (timeline break signature)")
        gaps, worst = _audio_pts_gaps(render, a)
        if gaps:
            bad.append(f"audio timeline has {gaps} timestamp hole(s), largest {worst * 1000:.0f} ms: "
                       "audio after it plays out of sync")
    rc, _, errtxt = _run([env.ffmpeg(), "-hide_banner", "-nostdin", "-v", "error", "-i", str(render),
                          "-map", "0", "-f", "null", "-"])
    dec = [ln for ln in errtxt.splitlines() if ln.strip()]
    if rc != 0 or dec:
        bad.append(f"full decode not clean ({len(dec)} error line(s)): " + " | ".join(dec[:3]))
    if bad:
        return gate("G3", "FAIL", "; ".join(bad), dec[:10], target=target)
    ev = (f"{W}x{H} h264 yuv420p bt709 {v.get('r_frame_rate')}, aac 48000 Hz 2 ch, "
          f"A/V diff {av:.3f}s, {odd} odd AAC packet(s), decode clean")
    return gate("G3", "PASS", ev, target=target)


def file_audio_md5(ctx: _Ctx, path: Path) -> str | None:
    """md5 of the decoded PCM of a render's first audio stream (what the render log records as
    audio_md5), recomputed from the file; None when it cannot be decoded."""
    key = str(path)
    if key not in ctx._md5:
        rc, out, _err = _run([env.ffmpeg(), "-hide_banner", "-nostdin", "-loglevel", "error", "-i", str(path),
                              "-map", "0:a:0", "-c:a", "pcm_s16le", "-f", "md5", "-"])
        txt = out.decode("utf-8", "replace").strip()
        ctx._md5[key] = (txt.split("=", 1)[1] if "=" in txt else txt) if rc == 0 and txt else None
    return ctx._md5[key]


def g_styles_audio(ctx: _Ctx, cid: str, fmt: str, renders: dict) -> dict | None:
    """Each render's audio is what its render log recorded, and all styles of a clip x format
    carry identical audio. The md5 is recomputed from each render file (the log's audio_md5 is
    only a fallback when the file is gone), so a replaced or re-encoded audio track (a render
    edited after the fact) is caught even when its log was left alone. With one style only the
    file-against-log test applies. None when there is no render."""
    if not renders:
        return None
    one = len(renders) == 1
    md5s, missing, notes, unlogged = {}, [], [], []
    for st, rp in renders.items():
        lp = Path(ctx.project.render_path(cid, fmt, st, ext=".log.json"))
        try:
            logged = env.read_json(lp).get("audio_md5") if lp.is_file() else None
        except env.HCError:
            logged = None
        got = file_audio_md5(ctx, Path(rp)) if Path(rp).is_file() else None
        if got and logged and got != logged:
            notes.append(f"{st}: the file's audio ({got[:8]}) is not what its render log recorded ({logged[:8]})")
        m = got or logged
        if m:
            md5s[st] = m
        else:
            missing.append(st)
        if not (got and logged):
            unlogged.append(st)
    target = f"{fmt} (all styles)"
    if missing:
        return gate("G3", "SKIPPED", f"audio identity across styles not checkable: no decodable audio or audio_md5 for {', '.join(missing)}",
                    target=target, gap=True)
    if one and unlogged and not notes:
        return gate("G3", "SKIPPED", f"the audio of {unlogged[0]} could not be compared with its render log "
                    "(no audio_md5 in the log, or the file's audio does not decode)", target=target, gap=True)
    if len(set(md5s.values())) > 1:
        return gate("G3", "FAIL", "styles diverged: audio differs between styles (" +
                    ", ".join(f"{s} {m[:8]}" for s, m in md5s.items()) + ")", notes, target=target)
    if notes:
        return gate("G3", "FAIL", "; ".join(notes), notes, target=target)
    if one:
        return gate("G3", "PASS", f"audio md5 {next(iter(md5s.values()))[:12]} decoded from the file is what its "
                    "render log recorded", target=target)
    return gate("G3", "PASS", f"{len(md5s)} styles share audio md5 {next(iter(md5s.values()))[:12]} (decoded from the files)",
                target=target)


def intent_sequence(plan: dict | None, prefer: str = "raw") -> list[tuple]:
    """What the clip is meant to say, in play order: [(source word index or None, text)].

    plan["intent"] (source word indices) plus the tokens of plan["inserts"] (speech the
    transcript missed, declared with the clip key `insert`), each run placed after its `after`
    word. Inserted tokens carry index None: no transcript word stands behind them. `prefer`
    picks the word text ("raw" as transcribed, "text" as displayed)."""
    plan = plan or {}
    other = "text" if prefer == "raw" else "raw"
    occ: dict = {}                                   # word index -> its plan words, in play order
    for w in plan.get("words") or []:
        if isinstance(w, dict) and w.get("i") is not None:
            occ.setdefault(w["i"], []).append(w)
    seq, pieces, seen = [], [], {}
    for i in plan.get("intent") or []:
        ws = occ.get(i)
        if not ws:
            continue
        k = seen.get(i, 0)
        seen[i] = k + 1
        w = ws[min(k, len(ws) - 1)]
        seq.append((i, str(w.get(prefer) or w.get(other) or "")))
        pieces.append(w.get("piece"))
    ins_words: dict = {}
    for w in plan.get("words") or []:
        if isinstance(w, dict) and w.get("i") is None and w.get("ins") is not None:
            ins_words.setdefault(w["ins"], []).append(w)
    runs = []
    for k, item in enumerate(plan.get("inserts") or []):
        if not isinstance(item, dict):
            continue
        n = item.get("n", k)
        mine = ins_words.get(n) or []
        toks = [str(t) for t in item.get("tokens") or [] if str(t).strip()] or \
            [str(w.get(prefer) or w.get(other) or "") for w in mine]
        if not toks:
            continue
        try:
            after = int(item.get("after"))
        except (TypeError, ValueError):
            after = None
        piece = mine[0].get("piece") if mine else None
        runs.append((after, piece, toks))
    for after, piece, toks in runs:
        at = None
        if after is not None:
            hits = [p for p, (i, _t) in enumerate(seq) if i == after]
            same = [p for p in hits if piece is not None and pieces[p] == piece]
            if same or hits:
                at = (same or hits)[-1] + 1
            else:   # `after` itself is not intended (a filler, an omitted word): the word before it
                prev = [p for p, (i, _t) in enumerate(seq)
                        if i is not None and i < after and (piece is None or pieces[p] == piece)]
                nxt = [p for p, (i, _t) in enumerate(seq)
                       if i is not None and i > after and (piece is None or pieces[p] == piece)]
                at = max(prev, key=lambda p: (seq[p][0], p)) + 1 if prev else (min(nxt) if nxt else None)
        if at is None:
            at = len(seq)
        while at < len(seq) and seq[at][0] is None:
            at += 1                                  # a second run after the same word follows the first
        seq[at:at] = [(None, t) for t in toks]
        pieces[at:at] = [piece] * len(toks)
    return seq


def _dup_positions(plan: dict, intent: list) -> dict:
    """Intended word positions that two overlapping pieces both play (plan `overlaps`, I14) ->
    "pieces i/j" label. Inserted tokens (index None) are never part of an overlap."""
    out = {}
    for ov in plan.get("overlaps") or []:
        ws = ov.get("words") if isinstance(ov, dict) else None
        if not ws or len(ws) != 2:
            continue
        try:
            a, b = int(ws[0]), int(ws[1])
        except (TypeError, ValueError):
            continue
        pcs = ov.get("pieces") or []
        label = "pieces " + "/".join(str(x) for x in pcs) if pcs else "overlapping pieces"
        for pos, wi in enumerate(intent):
            if wi is not None and a <= wi <= b:
                out[pos] = label
    return out


def g5_words(ctx: _Ctx, cid: str, plan: dict, render: Path | None, target: str):
    """Returns (gate, diff_or_None, asr_words_or_None)."""
    if not ctx.asr:
        return gate("G5", "SKIPPED", "--no-asr: render audio was not re-transcribed", target=target,
                    reason="--no-asr", gap=True), None, None
    if render is None:
        return gate("G5", "SKIPPED", "no render to transcribe", target=target, gap=True), None, None
    try:
        tr = ctx.project.transcript(cid)
    except env.HCError as exc:
        return gate("G5", "SKIPPED", f"no transcript: {exc}", target=target, gap=True), None, None
    try:
        hyp = _asr_words(ctx.project, render, tr)
    except env.HCError as exc:
        if exc.code == 3:
            ctx.tool_errors.append(f"{cid} G5: {exc}")
        return gate("G5", "SKIPPED", f"ASR unavailable: {exc}", target=target, gap=True), None, None
    except Exception as exc:  # backend crash (missing model, runtime error): a tool error, exit 3
        msg = f"ASR failed: {type(exc).__name__}: {str(exc)[:300]}"
        ctx.tool_errors.append(f"{cid} G5: {msg}")
        return gate("G5", "SKIPPED", msg, target=target, gap=True), None, None
    seq = intent_sequence(plan)
    intent = [i for i, _t in seq]                    # None marks an inserted token
    if not intent:
        return gate("G5", "SKIPPED", "plan has no intent words", target=target, gap=True), None, hyp
    ref_texts = [t for _i, t in seq]
    n_ins = sum(1 for i in intent if i is None)
    dup = _dup_positions(plan, intent)
    d = diff_words(ref_texts, [w["w"] for w in hyp], _silence_oracle(render, hyp),
                   dup=dup, hyp_times=[w["s"] for w in hyp])
    d["intent"] = intent
    heard = {ri for ri, _hj in d["pairs"]}
    by_label: dict = {}
    for pos, label in dup.items():
        by_label.setdefault(label, []).append(pos)
    for label, poss in by_label.items():
        if all(pos in heard for pos in poss):
            d["warn"].append(f"duplicated at join ({label}): their shared words are heard twice in the render "
                             "(plan lint L17)")
    rate = d["matched"] / d["ref_n"] if d["ref_n"] else 0.0
    ev = (f"{d['ref_n']} intended tokens" + (f" ({n_ins} inserted)" if n_ins else "") +
          f", {d['matched']} matched ({rate:.0%}); "
          f"{len(d['fail'])} hard miss(es), {len(d['warn'])} soft/replace/extra note(s)")
    if d["fail"]:
        return gate("G5", "FAIL", ev + ": " + "; ".join(d["fail"][:3]), d["fail"] + d["warn"], target=target), d, hyp
    if d["warn"]:
        return gate("G5", "WARN", ev, d["warn"], target=target), d, hyp
    return gate("G5", "PASS", ev, target=target), d, hyp


def _silence_oracle(render: Path, hyp: list[dict]):
    """hyp word index -> True when the render has no voiced audio (floor + 10 dB) under it."""
    try:
        x = decode_mono(render)
    except env.HCError:
        return None
    if x.size == 0:
        return None
    db = hop_db(x)
    floor = max(FLOOR_MIN_DB, float(np.percentile(db, FLOOR_PCT)))

    def silent(j: int) -> bool:
        w = hyp[j]
        a = max(0, int(float(w["s"]) / 0.01))
        b = max(a + 1, int(math.ceil(float(w["e"]) / 0.01)))
        seg = db[a:b]
        return seg.size == 0 or float(np.mean(seg > floor + 10.0)) < SILENT_VOICED_FRAC
    return silent


def dead_air(plan: dict | None, E) -> list[tuple[int, float, float | None]]:
    """[(segment k, out start, seconds to the first voiced audio or None)] for every plan segment
    whose first voiced audio comes more than DEAD_AIR (DEAD_AIR_OPEN at out 0) after its start."""
    from . import align
    out = []
    if E is None or not plan:
        return out
    for k, seg in enumerate(plan.get("segments") or []):
        try:
            t = float((seg.get("out") or [None])[0])
        except (TypeError, ValueError):
            continue
        if t >= E.end - 0.05:
            continue
        lim = DEAD_AIR_OPEN if t < 1e-6 else DEAD_AIR
        on = align.next_onset(E, t, min(t + 1.5, E.end))
        gap = None if on is None else on - t
        if gap is None or gap > lim + 1e-9:
            out.append((k, t, gap))
    return out


def close_sound(x: np.ndarray, sr: int, t0: float, t1: float, S=None) -> dict:
    """What sounds in the close window [t0, t1] (the CLOSE_WIN before the final fade) of mono
    audio `x`: {"kind", "centroid", "sounded", "note"}.

    kind "consonant": unpitched frication (speechband "fricative": "s", "sh", the release of
    "t") that follows a voiced sound with no pause, so it is the last consonant of a word; it
    has sounded for most of the window and the fade is its decay. "stray": the same spectrum
    with no voiced sound right before it, the first consonant of the next word. "voiced": a
    pitched or formant sound (a vowel or a nasal) at full level, the clipped word G6 warns
    about. "other": anything else that is loud."""
    if S is None:
        from . import speechband
        S = speechband.analyse(x, sr)
    d = S.describe(t0, t1)
    cen = float(d.get("centroid") or 0.0)
    sounded = 0.0
    if d["kind"] == "fricative":
        start = S.frication_start(t1)
        sounded = max(0.0, t1 - start)
        kind = "consonant" if S.speech_before(start, CLOSE_ATTACH) else "stray"
    elif d["kind"] == "voiced":
        kind = "voiced"
    else:
        kind = "other"
    note = {"consonant": f"an unvoiced consonant (frication centred at {cen / 1000:.1f} kHz for {sounded * 1000:.0f} ms, "
                         "straight after a voiced sound): the last consonant of a word, not a clipped vowel",
            "stray": f"a consonant (frication centred at {cen / 1000:.1f} kHz) with no voiced sound right before it: "
                     "the next word is starting",
            "voiced": "a voiced sound (a vowel or a nasal) is cut at full level",
            "other": "the sound is neither a clear vowel nor a consonant"}[kind]
    return {"kind": kind, "centroid": cen, "sounded": sounded, "note": note}


def g6_open_close(fresh: dict | None, render: Path | None, target: str, plan: dict | None = None,
                  E=None, S=None) -> dict:
    """Opener and closer: lints L04 / L05, the first 30 ms and the 60 ms before the final fade
    measured on the render, and dead air. A loud close is "ends mid-sound" unless what sounds
    there is the last consonant of a word (close_sound): frication straight after a voiced
    sound, which the final fade lets run out."""
    det, statuses = [], []
    for l in (fresh or {}).get("lints") or []:
        if l.get("id") in ("L04", "L05") and l.get("level") in ("FAIL", "WARN"):
            statuses.append(l["level"])
            det.append(f"{l['level']} {l['id']}: {l.get('msg')}")
    gap = False
    if render is None:
        det.append("render missing: open/close audio not measured")
        gap = True
    else:
        try:
            x = decode_mono(render)
        except env.HCError as exc:
            x = None
            det.append(f"audio not measured: {exc}")
            gap = True
        if x is not None and x.size:
            sr = 16000
            floor = max(FLOOR_MIN_DB, float(np.percentile(hop_db(x, sr), FLOOR_PCT)))
            dur = x.size / sr
            head = rms_db(x[: int(OPEN_WIN * sr)])
            t1 = max(0.0, dur - CLOSE_FADE)
            t0 = max(0.0, t1 - CLOSE_WIN)
            tail = rms_db(x[int(t0 * sr): int(t1 * sr)])
            det.append(f"floor {floor:.1f} dB, first 30 ms {head:.1f} dB, last 60 ms before the fade {tail:.1f} dB")
            if head > floor + MID_SOUND_DB:
                statuses.append("WARN")
                det.append(f"WARN starts mid-sound: first 30 ms at {head:.1f} dB (floor {floor:.1f} dB)")
            if tail > floor + MID_SOUND_DB:
                cs = close_sound(x, sr, t0, t1, S)
                if cs["kind"] == "consonant":
                    det.append(f"ends on a finished consonant: 60 ms before the final fade at {tail:.1f} dB "
                               f"(floor {floor:.1f} dB) is {cs['note']}")
                else:
                    statuses.append("WARN")
                    det.append(f"WARN ends mid-sound: 60 ms before the final fade at {tail:.1f} dB "
                               f"(floor {floor:.1f} dB): {cs['note']}")
    for k, t, wait in dead_air(plan or fresh, E):
        statuses.append("WARN")
        what = f"{wait:.2f}s" if wait is not None else "more than 1.5 s"
        det.append(f"WARN dead air: segment {k} starts at out {t:.2f}s and the first voice comes {what} later")
    st = worst(statuses) if statuses else "PASS"
    if st == "PASS" and gap:
        return gate("G6", "SKIPPED", "; ".join(det), det, target=target, gap=True)
    issues = [_bare(d) for d in det if d.startswith(("FAIL", "WARN"))]
    ev = "; ".join(issues) if issues else (det[-1] if det else "opener and closer lints clean")
    return gate("G6", st, ev, det, target=target, gap=gap)


_EBU_I = re.compile(r"I:\s*(-?[\d.]+|-inf)\s*LUFS")
_EBU_TP = re.compile(r"Peak:\s*(-?[\d.]+|-inf)\s*dBFS")


def parse_ebur128(stderr: str) -> tuple[float | None, float | None]:
    """Integrated loudness and true peak from the ebur128 Summary block."""
    k = stderr.rfind("Summary:")
    block = stderr[k:] if k >= 0 else stderr
    mi, mp = _EBU_I.findall(block), _EBU_TP.findall(block)

    def val(xs):
        if not xs:
            return None
        return -math.inf if xs[-1] == "-inf" else float(xs[-1])
    return val(mi), val(mp)


def g7_loudness(render: Path | None, target: str) -> dict:
    if render is None:
        return gate("G7", "SKIPPED", "no render", target=target, gap=True)
    rc, _, err = _run([env.ffmpeg(), "-hide_banner", "-nostdin", "-nostats", "-loglevel", "info",
                       "-i", str(render), "-map", "0:a:0", "-af", "ebur128=peak=true", "-f", "null", "-"])
    i, tp = parse_ebur128(err)
    if rc != 0 or i is None or tp is None:
        return gate("G7", "SKIPPED", "ebur128 measurement failed: " + err.strip()[-200:], target=target, gap=True)
    ev = f"I {i:.1f} LUFS (target {TARGET_LUFS:g} +/- {LUFS_TOL:g}), true peak {tp:.1f} dBTP (max {MAX_TRUE_PEAK:g})"
    bad = []
    if not (abs(i - TARGET_LUFS) <= LUFS_TOL):
        bad.append(f"integrated {i:.1f} LUFS")
    if tp > MAX_TRUE_PEAK:
        bad.append(f"true peak {tp:.1f} dBTP")
    if bad:
        return gate("G7", "FAIL", ev + ": " + ", ".join(bad) + " out of range", target=target)
    return gate("G7", "PASS", ev, target=target)


_BLACK = re.compile(r"black_start:\s*([\d.]+)\s+black_end:\s*([\d.]+)\s+black_duration:\s*([\d.]+)")
_FREEZE_S = re.compile(r"freeze_start:\s*([\d.]+)")
_FREEZE_D = re.compile(r"freeze_duration:\s*([\d.]+)")
_FREEZE_E = re.compile(r"freeze_end:\s*([\d.]+)")


def parse_black_freeze(stderr: str, duration: float) -> tuple[list, list]:
    """(black intervals [(s, e)], freezes [(s, e)]) from blackdetect/freezedetect logs."""
    blacks = [(float(a), float(b)) for a, b, _ in _BLACK.findall(stderr)]
    freezes, open_s = [], None
    for line in stderr.splitlines():
        m = _FREEZE_S.search(line)
        if m:
            open_s = float(m.group(1))
            continue
        m = _FREEZE_E.search(line)
        if m and open_s is not None:
            freezes.append((open_s, float(m.group(1))))
            open_s = None
    if open_s is not None:
        freezes.append((open_s, duration))
    return blacks, freezes


def judge_black_freeze(blacks, freezes, joins: list[dict], fps: Fraction) -> tuple[str, list[str]]:
    """Status + notes. Black at frame 0 or within +/-3 frames of a cut join FAILs; near a dip
    join, or anywhere else (even a single frame), WARNs; a freeze >= 1 s WARNs."""
    fd = float(1 / fps)
    win = JOIN_WINDOW_FRAMES * fd
    statuses, notes = [], []
    for s, e in blacks:
        if s < 0.5 * fd:
            statuses.append("FAIL")
            notes.append(f"FAIL black at frame 0 ({s:.3f}-{e:.3f}s)")
            continue
        near = [j for j in joins if s - win <= float(j.get("out", -99)) <= e + win]
        hard = [j for j in near if j.get("kind") != "dip"]
        if hard:
            j = hard[0]
            statuses.append("FAIL")
            notes.append(f"FAIL black {s:.3f}-{e:.3f}s at the {j.get('kind')} join {float(j['out']):.3f}s")
        elif near:
            statuses.append("WARN")
            notes.append(f"WARN black {s:.3f}-{e:.3f}s at the dip join {float(near[0]['out']):.3f}s (dip too dark?)")
        else:
            statuses.append("WARN")
            n = max(1, int(round((e - s) / fd)))
            notes.append(f"WARN black {s:.3f}-{e:.3f}s ({n} frame{'s' if n != 1 else ''}) away from any join")
    for s, e in freezes:
        if e - s >= 1.0 - 1e-6:
            statuses.append("WARN")
            notes.append(f"WARN frozen picture {s:.2f}-{e:.2f}s ({e - s:.2f}s)")
    return (worst(statuses) if statuses else "PASS"), notes


def find_flashes(means) -> list[int]:
    """Frames k whose band-masked mean luma jumps against both neighbours (> FLASH_JUMP) while
    the neighbours agree (< FLASH_SAME): a single-frame flash."""
    y = np.asarray(means, dtype=np.float64)
    out = []
    for k in range(1, y.size - 1):
        if (abs(y[k] - y[k - 1]) > FLASH_JUMP and abs(y[k] - y[k + 1]) > FLASH_JUMP
                and abs(y[k - 1] - y[k + 1]) < FLASH_SAME):
            out.append(k)
    return out


def luma_frames(render: Path, W: int, H: int) -> np.ndarray | None:
    """Every frame's luma at 1/LUMA_SCALE size as uint8 (n, h, w), or None."""
    w, h = max(1, W // LUMA_SCALE), max(1, H // LUMA_SCALE)
    rc, out, _err = _run([env.ffmpeg(), "-hide_banner", "-nostdin", "-loglevel", "error", "-i", str(render),
                          "-map", "0:v:0", "-vf", f"scale={w}:{h}:flags=area,format=gray", "-f", "rawvideo", "-"])
    if rc != 0 or not out:
        return None
    n = len(out) // (w * h)
    if n == 0:
        return None
    return np.frombuffer(out[: n * w * h], np.uint8).reshape(n, h, w)


def _band_rows(band, H: int, h: int):
    if not band or len(band) < 2 or not band[1]:
        return None
    return (float(band[0]) * h / H, (float(band[0]) + float(band[1])) * h / H)


def _source_has_active_area(ctx: _Ctx) -> bool:
    try:
        v = (ctx.project.source() or {}).get("video") or {}
    except Exception:
        return False
    return bool(v.get("active_area"))


def _flags_sheet(ctx: _Ctx, cid: str, fmt: str, style: str, render: Path, times: list[float],
                 labels: list[str]) -> Path | None:
    """sheets/<clip>__<fmt>__<style>.flags.jpg: the exact frames G8 flagged."""
    from . import sheets
    if not times:
        return None
    try:
        imgs = sheets.frames_at(render, times[:12], 270)
        out = Path(ctx.project.sheet_path(f"{cid}__{fmt}__{style}.flags.jpg"))
        sheets.contact_sheet(list(zip(imgs, labels[:12])), min(4, len(imgs)), out, 270,
                             title=f"{cid} {fmt} {style}: frames QA G8 flagged")
        return out
    except Exception:
        return None


def g8_black_freeze(plan: dict, render: Path, band: list | None, target: str, ctx: _Ctx | None = None,
                    cid: str | None = None, fmt: str = "vertical", style: str = "") -> dict:
    vf = []
    if band and band[1] > 0:
        vf.append(f"drawbox=x=0:y={int(band[0])}:w=iw:h={int(band[1])}:color=black:t=fill")
    vf += [BLACK_FILTER, FREEZE_FILTER]
    rc, _, err = _run([env.ffmpeg(), "-hide_banner", "-nostdin", "-nostats", "-loglevel", "info",
                       "-i", str(render), "-map", "0:v:0", "-vf", ",".join(vf), "-an", "-f", "null", "-"])
    if rc != 0:
        return gate("G8", "SKIPPED", "blackdetect/freezedetect failed: " + err.strip()[-200:], target=target, gap=True)
    fps = _frac(plan.get("fps")) or Fraction(25)
    fd = float(1 / fps)
    blacks, freezes = parse_black_freeze(err, float(plan.get("duration_out") or 0.0))
    st, notes = judge_black_freeze(blacks, freezes, plan.get("joins") or [], fps)
    statuses = [st]
    flag_t, flag_l = [], []
    for s, e in blacks:
        if s >= 0.5 * fd:
            flag_t.append(s + 0.5 * fd)
            flag_l.append(f"{s:.2f}s black")
    extra = ""
    from .cover_frames import out_size
    try:
        W, H = out_size(fmt)
    except Exception:
        W, H = 1080, 1920
    lum = luma_frames(render, W, H)
    if lum is None:
        notes.append("luma scan failed: flashes and dark bands not checked")
        extra = "; flash/dark-band scan failed"
        gap = True
    else:
        gap = False
        n, h, w = lum.shape
        rows = _band_rows(band, H, h)
        keep = np.ones(h, bool)
        if rows:
            keep[max(0, int(rows[0])):min(h, int(math.ceil(rows[1])))] = False
        means = lum[:, keep, :].reshape(n, -1).mean(axis=1) if keep.any() else lum.reshape(n, -1).mean(axis=1)
        for k in find_flashes(means):
            t = k * fd
            if any(s - fd <= t <= e for s, e in blacks):
                continue                       # already reported as black
            statuses.append("WARN")
            notes.append(f"WARN single-frame flash at {t:.3f}s (frame {k}: mean luma {means[k]:.0f} "
                         f"between {means[k - 1]:.0f} and {means[k + 1]:.0f})")
            flag_t.append(t + 0.5 * fd)
            flag_l.append(f"{t:.2f}s flash f{k}")
        idx = np.unique(np.linspace(0, n - 1, min(DARK_SAMPLES, n)).round().astype(int))
        row_max = lum[idx].max(axis=(0, 2))
        bands = dark_bands(row_max, skip=rows)
        if bands:
            big = any((b - a) / h >= DARK_FAIL_FRAC for a, b in bands)
            lvl = "FAIL" if (big or (ctx is not None and _source_has_active_area(ctx))) else "WARN"
            statuses.append(lvl)
            notes.append(f"{lvl} dark band {_band_text(bands, h, H)} across all {len(idx)} sampled frames "
                         "(letterbox bars or gallery border in the picture)")
    if ctx is not None and cid and flag_t:
        sheet = _flags_sheet(ctx, cid, fmt, style, render, flag_t, flag_l)
        if sheet is not None:
            ctx.flag_sheets[(cid, fmt, style)] = sheet
            notes.append(f"flagged frames: {ctx.project.rel(sheet)}")
    final = worst(statuses)
    masked = " (caption band masked)" if band else ""
    if final == "PASS":
        return gate("G8", "PASS", f"no black at frame 0 or at {len(plan.get('joins') or [])} join(s), no black or "
                    f"single-frame flash elsewhere, no freeze >= 1 s, no dark band{masked}{extra}",
                    notes, target=target, gap=gap)
    issues = [n_ for n_ in notes if n_.startswith(("FAIL", "WARN"))]
    return gate("G8", final, "; ".join(_bare(n_) for n_ in issues[:4]) + masked, notes, target=target, gap=gap)


def _safezone(name: str, project=None) -> dict:
    try:
        from . import presets
        z = presets.safezone(name, project)
        if z:
            return z
    except Exception:
        pass
    data = env.read_json(env.bundled_presets_dir() / "safezones.json")
    if name not in data:
        raise env.HCError(f"unknown safezone profile {name!r}", code=1)
    return data[name]


def layout_issues(boxes: list[tuple[str, list]], W: int, H: int, zone: dict) -> tuple[str, list[str]]:
    """FAIL for a box leaving the frame, WARN for a box outside the safezone."""
    sx0, sy0 = zone.get("left", 0), zone.get("top", 0)
    sx1, sy1 = W - zone.get("right", 0), H - zone.get("bottom", 0)
    statuses, notes = [], []
    for label, b in boxes:
        x, y, w, h = (float(v) for v in b)
        if x < 0 or y < 0 or x + w > W or y + h > H:
            statuses.append("FAIL")
            notes.append(f"FAIL {label} box [{x:.0f},{y:.0f},{w:.0f},{h:.0f}] leaves the {W}x{H} frame")
        elif x < sx0 or y < sy0 or x + w > sx1 or y + h > sy1:
            statuses.append("WARN")
            notes.append(f"WARN {label} box [{x:.0f},{y:.0f},{w:.0f},{h:.0f}] outside the safe area x {sx0}-{sx1}, y {sy0}-{sy1}")
    return (worst(statuses) if statuses else "PASS"), notes


def mouth_zone(f: list) -> list:
    """The part of a face captions must not cover: [x, y, w, h]. With YuNet landmarks
    (f[4] = [[x, y] * 5]: eyes, nose tip, mouth corners) it runs from the nose tip to just below
    the lower lip (mouth line + MOUTH_CLEAR x eye-to-mouth distance), which is where a caption
    hides speech; a caption over the chin or beard is fine. Without landmarks: the box's lower
    60 % (contract G9)."""
    x, y, w, h = (float(v) for v in f[:4])
    lm = f[4] if len(f) > 4 else None
    if lm and len(lm) >= 5:
        eye_y = (lm[0][1] + lm[1][1]) / 2.0
        nose_y = lm[2][1]
        mouth_y = (lm[3][1] + lm[4][1]) / 2.0
        d = max(1.0, mouth_y - eye_y)
        top, bot = min(nose_y, mouth_y), mouth_y + MOUTH_CLEAR * d
        return [x, top, w, max(1.0, bot - top)]
    return [x, y + (1 - FACE_LOWER) * h, w, FACE_LOWER * h]


def face_hits(box, faces: list[list]) -> list[list]:
    """Faces whose mouth zone (see mouth_zone) the box overlaps."""
    from .cover_frames import box_overlap
    return [f for f in faces if box_overlap(box, mouth_zone(f)) > 0]


def _captions(ctx: _Ctx, cid: str, fmt: str, style: str) -> tuple[dict | None, Path, str | None]:
    """(captions json or None, its path, reason when the render was made without captions)."""
    p = Path(ctx.project.render_path(cid, fmt, style, ext=".captions.json"))
    if not p.is_file():
        return None, p, None
    try:
        data = env.read_json(p)
    except env.HCError:
        return None, p, None
    if isinstance(data, dict) and data.get("disabled"):
        return None, p, str(data.get("reason") or "captions disabled")
    return data, p, None


def _no_captions(clip: dict, disabled: str | None, gid: str, target: str) -> dict | None:
    """Result for a render deliberately or accidentally made without captions, else None."""
    if disabled is None:
        return None
    if clip.get("captions") is False:
        return gate(gid, "SKIPPED", f"captions off for this clip ({disabled})", target=target, reason="not applicable")
    if gid == "G9":
        return gate(gid, "FAIL", f"captions were expected but the render has none: {disabled}", target=target)
    return gate(gid, "SKIPPED", f"render has no captions: {disabled}", target=target, gap=True)


def _takes(fn, name: str) -> bool:
    """`fn` accepts the keyword `name` (a sibling module may predate the argument)."""
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    return name in params or any(p.kind == p.VAR_KEYWORD for p in params.values())


def _letterbox_centre(ctx: _Ctx, style: str, fmt: str) -> float | None:
    """Where the style centres a letterboxed picture, as a fraction of the output height
    (vertical only): base_frames needs it when it rebuilds a frame from the source, or the
    picture lands mid-frame where no render of this style puts it."""
    if fmt != "vertical":
        return None
    try:
        from . import cover_frames as cf
        from .framing import DEFAULT_LETTERBOX
        lb = ctx.preset(style, fmt).get("letterbox")
        yc = (lb if isinstance(lb, dict) else {}).get("y_center", DEFAULT_LETTERBOX["y_center"])
        return float(yc) / cf.out_size(fmt)[1]
    except Exception:
        return None


def _base_faces(ctx: _Ctx, cid: str, clip: dict, plan: dict, fmt: str, style: str, times: list[float]) -> dict:
    """Faces (output pixel boxes) on clean base frames at out times; cached per clip x format
    x style. Each style is judged on its OWN stage A picture (two styles can frame one clip
    differently: another punch-in, another letterbox centre), never on whichever style was
    rendered last."""
    from . import cover_frames as cf
    W, H = cf.out_size(fmt)
    size = (W // 2, H // 2)
    cache = ctx._faces.setdefault((cid, fmt, style), {})
    need = [t for t in times if round(t, 3) not in cache]
    if need:
        kw = {"letterbox_y": _letterbox_centre(ctx, style, fmt)}
        if _takes(cf.base_frames, "style"):
            kw["style"] = style
        frames, _ = cf.base_frames(ctx.project, cid, clip, plan, fmt, need, size, **kw)
        for t in need:
            arr = frames.get(t)
            if arr is None:
                continue
            faces = cf.detect_faces(arr)
            cache[round(t, 3)] = [[v * 2 for v in f["box"]]
                                  + ([[[c * 2 for c in pt] for pt in f["landmarks"]]]
                                     if f.get("landmarks") else [])
                                  for f in faces]
    return {t: cache.get(round(t, 3)) for t in times}


def _zone_name(ctx: _Ctx, clip: dict, style: str, fmt: str) -> str:
    """The safe-zone profile the render of this style used. One resolver, presets.zone_name,
    shared with the caption layout: clips.json defaults.safezone when the file set it, else the
    preset's own "safezone", else organic (a landscape render: the landscape profile unless its
    preset names another)."""
    eff = clip.get("_eff") or {}
    try:
        from . import presets
        resolver = getattr(presets, "zone_name", None)
        if resolver is not None:
            return str(resolver(eff, ctx.preset(style, fmt), fmt))
    except Exception:
        pass
    return "landscape" if fmt == "landscape" else (eff.get("safezone") or "organic")


def g9_layout(ctx: _Ctx, cid: str, clip: dict, plan: dict, fmt: str, style: str, cap: dict | None,
              cap_path: Path, target: str) -> dict:
    """Caption and hook-card boxes inside the frame and the safe zone of the profile the render
    used, the caption layout's own issues, and no caption over a mouth on this style's own
    base picture. A face-tool or frame-grab failure (HCError code 3) is a tool error as well
    as an evidence gap."""
    from .cover_frames import out_size
    W, H = out_size(fmt)
    has_hook = bool(_hook_text(plan, clip))
    if cap is None:
        if not clip.get("captions", True) and not has_hook:
            return gate("G9", "SKIPPED", "captions off and no hook card: nothing burned in", target=target,
                        reason="not applicable")
        return gate("G9", "SKIPPED", f"captions json missing: {ctx.project.rel(cap_path)}", target=target, gap=True)
    zone_name = _zone_name(ctx, clip, style, fmt)
    zone = _safezone(zone_name, ctx.project)
    boxes = []
    pages = cap.get("pages") or []
    for p in pages:
        b = p.get("backing_box") if p.get("backing") and p.get("backing_box") else p.get("box")
        if b:
            boxes.append((f"page {p.get('i')} ({_fmt_t(p.get('start_out'))})", b))
    hook = cap.get("hook") or None
    if hook and hook.get("box"):
        boxes.append(("hook card", hook["box"]))
    if not boxes:
        return gate("G9", "SKIPPED", "captions json has no page or hook boxes", target=target, gap=True)
    st, notes = layout_issues(boxes, W, H, zone)
    statuses = [st]
    for iss in cap.get("issues") or []:
        if not isinstance(iss, dict):
            continue
        code, msg = str(iss.get("code") or "?"), str(iss.get("msg") or "")
        where = f" (at {_fmt_t(iss['at_out'])})" if iss.get("at_out") is not None else ""
        if code in CAPTION_ISSUES_WARN:
            statuses.append("WARN")
            notes.append(f"WARN caption {code}: {msg}{where}")
        else:
            notes.append(f"caption note {code}: {msg}{where}")
    face_note, gap = "", False
    try:
        mids = []
        for p in pages:
            if p.get("box") and p.get("start_out") is not None and p.get("end_out") is not None:
                mids.append((f"page {p.get('i')}", (float(p["start_out"]) + float(p["end_out"])) / 2, p["box"]))
        if hook and hook.get("box"):
            hs = float(hook.get("start_out") or 0.0)
            he = float(hook.get("end_out") or hs + 1.0)
            mids.append(("hook card", (hs + he) / 2, hook["box"]))
        dur = float(plan.get("duration_out") or 0.0)
        mids = [(lab, min(max(t, 0.0), max(dur - 0.05, 0.0)), b) for lab, t, b in mids]
        faces = _base_faces(ctx, cid, clip, plan, fmt, style, [t for _, t, _ in mids])
        covered = 0
        for lab, t, b in mids:
            fs = faces.get(t) or []
            hits = face_hits(b, fs)
            if hits:
                covered += 1
                statuses.append("WARN")
                f = hits[0]
                bh = float(b[3])
                mz = mouth_zone(f)
                y_sugg = mz[1] + mz[3] + bh / 2 + 30
                limit = H - zone.get("bottom", 0) - bh / 2
                hint = f"try caption_y {int(round(y_sugg))}" if y_sugg <= limit else "no room below the face: consider letterbox or another framing"
                notes.append(f"WARN {lab} at {t:.2f}s covers the mouth of the face at [{f[0]:.0f},{f[1]:.0f},{f[2]:.0f},{f[3]:.0f}]: {hint}")
        face_note = f"; faces checked at {len(mids)} page midpoint(s), {covered} covered"
    except Exception as exc:
        face_note = f"; face check not run: {exc}"
        gap = True
        if isinstance(exc, env.HCError) and exc.code == 3:      # the face tool or the frame grab broke
            ctx.tool_errors.append(f"{cid} G9: {exc}")
    final = worst(statuses)
    issues = [n for n in notes if n.startswith(("FAIL", "WARN"))]
    ev = (f"{len(boxes)} box(es) vs '{zone_name}' safezone" + face_note) if final == "PASS" else \
        ("; ".join(_bare(n) for n in issues[:3]) + (f" (+{len(issues) - 3} more)" if len(issues) > 3 else "") + face_note)
    return gate("G9", final, ev, notes, target=target, gap=gap)


def _caption_starts(cap: dict) -> dict:
    """Source word index -> caption token s_out (first token carrying it)."""
    out = {}
    for p in cap.get("pages") or []:
        for t in p.get("tokens") or []:
            if t.get("s_out") is None:
                continue
            for i in t.get("idx") or []:
                out.setdefault(i, float(t["s_out"]))
    return out


def caption_tokens(cap: dict, lead: float) -> list[dict]:
    """Every caption token as {"text", "s", "e", "d" (display time = s - lead), "page", "idx"},
    in display order."""
    out = []
    for p in cap.get("pages") or []:
        for t in p.get("tokens") or []:
            if t.get("s_out") is None or t.get("e_out") is None:
                continue
            s, e = float(t["s_out"]), float(t["e_out"])
            out.append({"text": str(t.get("text") or ""), "s": s, "e": max(e, s), "d": s - lead,
                        "page": p.get("i"), "idx": list(t.get("idx") or [])})
    out.sort(key=lambda t: t["d"])
    return out


def energy_timing(E, toks: list[dict], pages: list[dict], a: float, b: float) -> dict:
    """G10 (a): caption timing against the render's own audio, no ASR model involved.

    - onset in silence: a token displayed inside a pause (>= G10_PAUSE) with at least
      G10_ONSET_SLACK of the pause still to come; `off` = seconds until the next voiced onset.
    - spans pause: a token whose [s, e] holds >= G10_SPAN_PAUSE of a pause.
    - early page: a page shown G10_PAGE_EARLY or more before the next voiced onset.
    Only tokens and pages displayed inside [a, b] count (a = hook card end)."""
    from . import align
    P = align.pauses(E, a, b, min_pause=G10_PAUSE)
    starts = [p[0] for p in P]
    in_sil, spans, early = [], [], []
    for t in toks:
        d = t["d"]
        if d < a - 1e-6 or d > b:
            continue
        k = bisect.bisect_right(starts, d - 0.02 - 1e-9) - 1
        if k >= 0:
            pa, pb = P[k]
            if pa + 0.02 < d < pb - G10_ONSET_SLACK:
                on = align.next_onset(E, d, min(d + 1.5, E.end))
                in_sil.append((t, (on - d) if on is not None else 1.5))
        best = 0.0
        lo = max(0, bisect.bisect_right(starts, t["s"]) - 1)
        for pa, pb in P[lo:]:
            if pa >= t["e"]:
                break
            best = max(best, min(t["e"], pb) - max(t["s"], pa))
        if best >= G10_SPAN_PAUSE - 1e-9:
            spans.append((t, best))
    for pg in pages:
        try:
            s0 = float(pg["start_out"])
        except (KeyError, TypeError, ValueError):
            continue
        if s0 < a - 1e-6 or s0 > b - 0.1 or not pg.get("tokens"):
            continue
        on = align.next_onset(E, s0, min(s0 + 2.0, E.end))
        wait = (on - s0) if on is not None else 2.0
        if wait >= G10_PAGE_EARLY - 1e-9:
            early.append((pg, wait))
    return {"in_silence": in_sil, "spans": spans, "early": early, "pauses": len(P)}


def speech_timing(S, toks: list[dict], a: float, b: float) -> list[tuple[dict, float, float]]:
    """G10 (a): tokens displayed more than G10_SPEECH_EARLY before the first speech-like sound.

    Broadband energy calls a laugh, a breath or a lip click "voiced", so a caption that lights
    under one of them 0.3 s before its word is invisible to energy_timing. `S` is the render's
    speechband.Track: its onsets need sustained, pitched or formant-shaped sound in the speech
    band, with a word-initial consonant counted in. Returns [(token, seconds early, seconds of
    other sound in the gap)] for tokens displayed inside [a, b] (a = hook card end). A strongly
    voiced laugh is speech-like and is not seen here (see speechband)."""
    out = []
    end = min(float(b), S.duration)
    for t in toks:
        d = t["d"]
        if d < a - 1e-6 or d > end:
            continue
        on = S.onset_after(d, min(end, d + G10_SPEECH_LOOK))
        off = (on - d) if on is not None else min(G10_SPEECH_LOOK, end - d)
        if off > G10_SPEECH_EARLY + 1e-9:
            out.append((t, off, S.loud_seconds(d, d + off)))
    return out


def uncertain_words(project, cid: str | None = None) -> set:
    """Source word indices the transcript flags `timing_uncertain`: the start of the word could
    not be placed because another sound (a laugh, a second voice) overlaps it."""
    out: set = set()
    try:
        flags = project.transcript(cid).get("flags") or []
    except Exception:
        return out
    for f in flags:
        if isinstance(f, dict) and f.get("kind") == "timing_uncertain":
            try:
                out.update(range(int(f["a"]), int(f.get("b", f["a"])) + 1))
            except (KeyError, TypeError, ValueError):
                continue
    return out


def uncaptioned_runs(E, toks: list[dict], a: float, b: float) -> list[tuple[float, float]]:
    """G10 (b): voiced render audio (>= G10_UNCOVERED_WARN) that no caption token is near."""
    from . import align
    spans = [{"s": t["s"] - 0.15, "e": t["e"] + 0.15} for t in toks]
    return align.uncovered_voiced(E, spans, a, b, min_run=G10_UNCOVERED_WARN, bridge=0.3)


def display_lag(E, toks: list[dict], lead: float, a: float, b: float):
    """G10 (c): (lag, r) of the displayed captions against the render's voicing, or None.

    Cross-correlates "a token is displayed" ([s - lead, e - lead] on a 10 ms grid) with soft
    voicing clip((dB - floor - 6) / 12, 0, 1) over +/- G10_LAG_MAX. A positive lag means the
    captions show later than the speech; lag + lead is the deviation from the intended lead."""
    db = np.asarray(E.db, dtype=np.float64)
    n = db.size
    hop = float(E.hop)
    if n == 0:
        return None
    floor = max(FLOOR_MIN_DB, float(np.percentile(db, FLOOR_PCT)))
    voiced = np.clip((db - floor - 6.0) / 12.0, 0.0, 1.0)
    cov = np.zeros(n)
    for t in toks:
        j0 = int(round((t["d"] - E.t0) / hop))
        j1 = int(round((t["e"] - lead - E.t0) / hop))
        if j1 > j0:
            cov[max(0, j0):max(0, min(n, j1))] = 1.0
    m = int(round(G10_LAG_MAX / hop))
    lo = max(0, int(round((a - E.t0) / hop))) + m
    hi = min(n, int(round((b - E.t0) / hop))) - m
    if hi - lo < int(round(G10_LAG_MIN_SPAN / hop)):
        return None
    vv = voiced[lo:hi] - voiced[lo:hi].mean()
    vn = float(np.sqrt((vv * vv).sum()))
    if vn <= 1e-9:
        return None
    best, best_l = -2.0, 0
    for L in range(-m, m + 1):
        cc = cov[lo + L:hi + L]
        cc = cc - cc.mean()
        den = vn * float(np.sqrt((cc * cc).sum()))
        if den <= 1e-9:
            continue
        r = float((vv * cc).sum() / den)
        if r > best:
            best, best_l = r, L
    if best <= -1.5:
        return None
    return best_l * hop, best


def g10_sync(ctx: _Ctx, plan: dict, cap: dict | None, diff: dict | None, hyp: list | None,
             lead: float, target: str, E=None, render: Path | None = None, S=None,
             uncertain: set | None = None) -> dict:
    """Caption timing, measured on what is displayed (token s_out - lead) against the render's
    own audio: (a) energy timing (onsets in silence, tokens spanning pauses, early pages) and,
    with the render's speech track `S`, tokens shown before the first speech-like sound (a
    caption that lights under a laugh or a click), (b) voiced audio no caption covers,
    (c) display lag by cross-correlation; these run without ASR. (d) With the G5
    re-transcription: median offset and Theil-Sen drift of the words. `uncertain` (source word
    indices the transcript flags timing_uncertain) names the caption tokens whose onset the
    audio cannot confirm: a voiced laugh over a word start looks like speech to (a)."""
    if cap is None:
        return gate("G10", "SKIPPED", "captions json missing", target=target, gap=True)
    if cap.get("lead_ms") is not None:
        try:
            lead = float(cap["lead_ms"]) / 1000.0
        except (TypeError, ValueError):
            pass
    if E is None and render is not None:
        E = ctx.energy(render)
    if E is None:
        return gate("G10", "SKIPPED", "render audio not readable: caption timing not measured",
                    target=target, gap=True)
    if S is None and render is not None:
        S = ctx.speech(render)
    toks = caption_tokens(cap, lead)
    if not toks:
        return gate("G10", "SKIPPED", "captions json has no timed tokens", target=target, gap=True)
    hook = cap.get("hook") or {}
    try:
        a = float(hook.get("end_out") or 0.0) if hook else 0.0
    except (TypeError, ValueError):
        a = 0.0
    b = float(plan.get("duration_out") or E.end) if plan else E.end
    b = min(b, E.end)
    statuses, det, ev = [], [], []

    # (a) energy timing
    et = energy_timing(E, toks, cap.get("pages") or [], a, b)
    far = [(t, off) for t, off in et["in_silence"] if off >= G10_FAR - 1e-9]
    if et["early"] or len(far) >= G10_FAR_COUNT:
        statuses.append("FAIL")
    elif et["in_silence"] or et["spans"]:
        statuses.append("WARN")
    ev.append(f"energy: {len(et['in_silence'])} token(s) shown in silence ({len(far)} >= {G10_FAR:g}s early), "
              f"{len(et['spans'])} spanning a pause, {len(et['early'])} early page(s)")
    for pg, wait in et["early"][:10]:
        det.append(f"{'FAIL'} page {pg.get('i')} shows at {float(pg['start_out']):.2f}s, the voice starts {wait:.2f}s later")
    for t, off in sorted(et["in_silence"], key=lambda z: -z[1])[:10]:
        lvl = "FAIL" if len(far) >= G10_FAR_COUNT and off >= G10_FAR else "WARN"
        det.append(f'{lvl} "{t["text"]}" shown at {t["d"]:.2f}s inside a pause, the voice starts {off:.2f}s later')
    for t, ov in sorted(et["spans"], key=lambda z: -z[1])[:10]:
        det.append(f'WARN "{t["text"]}" {t["s"]:.2f}-{t["e"]:.2f}s holds {ov:.2f}s of silence')
    if S is not None:
        seen = {id(t) for t, _off in et["in_silence"]}
        early = [z for z in speech_timing(S, toks, a, b) if id(z[0]) not in seen]
        if early:
            statuses.append("WARN")
        ev.append(f"speech: {len(early)} more token(s) shown over {G10_SPEECH_EARLY:g}s before the first "
                  "speech-like sound")
        for t, off, other in sorted(early, key=lambda z: -z[1])[:10]:
            what = (f"{other:.2f}s of other sound in between (a laugh, a breath or a click is not the word)"
                    if other >= 0.05 else "the gap is quiet")
            det.append(f'WARN "{t["text"]}" shown at {t["d"]:.2f}s, {off:.2f}s before the first speech-like '
                       f"sound; {what}: check the caption onset in the render")
    elif render is not None:
        ev.append("speech-like onsets not measured (render audio not decodable)")
    if uncertain:
        told = " ".join(det)
        unsure = [t for t in toks if a - 1e-6 <= t["d"] <= b and set(t["idx"]) & set(uncertain)]
        fresh_ones = [t for t in unsure if f'"{t["text"]}" shown at {t["d"]:.2f}s' not in told]
        if unsure:
            statuses.append("WARN")
            ev.append(f"{len(unsure)} token(s) on words the transcript flags timing_uncertain")
        for t in fresh_ones[:10]:
            det.append(f'WARN "{t["text"]}" shown at {t["d"]:.2f}s sits on a word whose start the transcript '
                       "flags timing_uncertain (another sound overlaps it); the audio here cannot confirm "
                       "the onset: look at the frames around it (hc peek with --file)")

    # (b) voiced audio no caption covers
    runs = uncaptioned_runs(E, toks, a, b)
    extra = set((diff or {}).get("extra") or [])
    n_fail = 0
    for rs, re_ in runs:
        heard = [hyp[j]["w"] for j in sorted(extra) if hyp and j < len(hyp) and rs - 0.1 <= float(hyp[j]["s"]) <= re_ + 0.1]
        long = re_ - rs >= G10_UNCOVERED_FAIL - 1e-9
        if long and len(heard) >= G10_UNCOVERED_ASR_WORDS:
            n_fail += 1
            statuses.append("FAIL")
            det.append(f'FAIL uncaptioned speech {rs:.2f}-{re_:.2f}s ({re_ - rs:.1f}s): heard "{" ".join(heard)[:100]}"')
        else:
            statuses.append("WARN")
            what = f', ASR heard "{" ".join(heard)[:80]}"' if heard else ""
            det.append(f"WARN voiced audio {rs:.2f}-{re_:.2f}s ({re_ - rs:.1f}s) has no caption{what}")
    if runs:
        ev.append(f"{len(runs)} uncaptioned voiced run(s) ({n_fail} FAIL)")

    # (c) display lag. The reference is the lead clamped to G10_LEAD_RANGE: a preset whose own
    # lead_ms puts captions late or far early is judged by what the viewer sees, not excused.
    ref = min(max(lead, G10_LEAD_RANGE[0]), G10_LEAD_RANGE[1])
    if abs(lead - ref) > 1e-9:
        lvl = "FAIL" if abs(lead - ref) > G10_LAG_FAIL else "WARN"
        statuses.append(lvl)
        det.insert(0, f"{lvl} caption lead_ms {lead * 1000:.0f} is outside {G10_LEAD_RANGE[0] * 1000:.0f}-"
                      f"{G10_LEAD_RANGE[1] * 1000:.0f} ms: captions are designed to show "
                      f"{abs(lead) * 1000:.0f} ms {'after' if lead < 0 else 'before'} the words (fix the preset)")
    lag = display_lag(E, toks, lead, a, b)
    if lag is None:
        ev.append("display lag not measured (too little audio after the hook card)")
    else:
        L, r = lag
        dev = L + ref
        txt = f"display lag {L:+.2f}s, deviation from a {ref * 1000:.0f} ms lead {dev:+.2f}s (r={r:.2f})"
        if r < 0.2:
            ev.append(txt + ", too weak to judge")
        elif abs(dev) > G10_LAG_FAIL:
            statuses.append("FAIL")
            ev.append(txt)
            det.insert(0, f"FAIL captions show {abs(dev):.2f}s {'late' if dev > 0 else 'early'} against the speech "
                          f"(> {G10_LAG_FAIL:g}s)")
        elif abs(dev) > G10_LAG_WARN:
            statuses.append("WARN")
            ev.append(txt)
            det.insert(0, f"WARN captions show {abs(dev):.2f}s {'late' if dev > 0 else 'early'} against the speech "
                          f"(> {G10_LAG_WARN:g}s)")
        else:
            ev.append(txt)

    # (d) ASR words (repaired QA transcription): median offset, drift, worst page
    if diff is not None and hyp is not None:
        starts = _caption_starts(cap)
        page_of = {}
        for p in cap.get("pages") or []:
            for t in p.get("tokens") or []:
                for i in t.get("idx") or []:
                    page_of.setdefault(i, p.get("i"))
        intent = diff.get("intent") or []
        xs, ds, per_page = [], [], {}
        for ri, hj in diff["pairs"]:
            if ri >= len(intent) or hj >= len(hyp):
                continue
            widx = intent[ri]
            if widx not in starts:
                continue
            s_asr = float(hyp[hj]["s"])
            if s_asr < a:
                continue
            xs.append(s_asr)
            ds.append(starts[widx] - s_asr)
            pg = page_of.get(widx)
            per_page[pg] = max(per_page.get(pg, 0.0), abs(starts[widx] - s_asr))
        if len(xs) >= SYNC_MIN_PAIRS:
            med = float(np.median(np.abs(ds)))
            bias = float(np.median(ds))
            slope = theil_sen(xs, ds)
            dur = float(plan.get("duration_out") or (max(xs) - min(xs)))
            drift = abs(slope) * dur if slope is not None else 0.0
            wp = max(per_page.items(), key=lambda kv: kv[1]) if per_page else (None, 0.0)
            ev.append(f"ASR: {len(xs)} words, median |offset| {med:.3f}s (bias {bias:+.3f}s), drift {drift:.3f}s, "
                      f"worst page {wp[0]} {wp[1]:.2f}s")
            if drift > SYNC_DRIFT_FAIL:
                statuses.append("FAIL")
                det.insert(0, f"FAIL captions drift {drift:.3f}s over the clip (> {SYNC_DRIFT_FAIL:g}s)")
            elif med > SYNC_MEDIAN_WARN:
                statuses.append("WARN")
                det.insert(0, f"WARN median caption offset {med:.3f}s against the re-transcription (> {SYNC_MEDIAN_WARN:g}s)")
        else:
            ev.append(f"ASR: only {len(xs)} caption words matched the re-transcription (drift not measured)")
    elif not ctx.asr:
        ev.append("ASR drift check skipped (--no-asr)")
    st = worst(statuses) if statuses else "PASS"
    det = [d for d in det if d.startswith("FAIL")] + [d for d in det if not d.startswith("FAIL")]
    head = "; ".join(_bare(d) for d in det if d.startswith(("FAIL", "WARN")))[:400] if st != "PASS" else ""
    evidence = (head + " | " if head else "") + "; ".join(ev)
    return gate("G10", st, evidence, det, target=target)


def claim_surfaces(clip: dict, plan: dict | None) -> list[tuple[str, str]]:
    """The written surfaces hard rule 4 covers, as [(label, text)]. There is one definition,
    claims.written_surfaces, shared with plan lint L13 and (through claims.effective_headline)
    with the cover drawer, so the words drawn and the words checked cannot differ: hook card,
    cover headline (also when a vertical clip with neither a cover.headline nor a hook card
    gets its title printed on the cover), and the other surfaces that function lists."""
    from . import claims
    return [(str(label), str(text)) for label, text in claims.written_surfaces(clip, plan) or []]


def g11_faith(ctx: _Ctx, cid: str, clip: dict, plan: dict | None) -> dict:
    """Every number and name on a written surface is said in the clip (claims.py, shared with
    plan lint L13): unspoken numbers FAIL; unspoken names, and content words of all-caps or
    title-case surfaces that are neither spoken nor common, WARN."""
    from . import claims
    surfaces = claim_surfaces(clip, plan)
    if not surfaces:
        return gate("G11", "SKIPPED", "no hook card, cover headline, title or post text to check", reason="not applicable")
    if plan is None:
        return gate("G11", "SKIPPED", "no plan: intent words unknown", gap=True)
    by_i = {w.get("i"): w for w in plan.get("words") or [] if w.get("i") is not None}
    intent = [by_i[i] for i in plan.get("intent") or [] if i in by_i]
    spoken = [str(w.get("raw") or "") for w in intent] + [str(w.get("text") or w.get("raw") or "") for w in intent]
    spoken += [t for i, t in intent_sequence(plan) if i is None]     # inserted speech is spoken too
    glossary = []
    try:
        glossary = list(ctx.project.transcript(cid).get("glossary") or [])
    except Exception:
        pass
    exempt = [clip.get("speaker")] + sorted({str(w.get("sp")) for w in intent if w.get("sp")})
    exempt = [e for e in exempt if e]
    sp = claims.Spoken(spoken)
    res = claims.unspoken_claims(surfaces, sp, glossary=glossary, exempt_names=exempt,
                                 name_evidence=ctx.name_evidence())
    ok = claims.spoken_claims_found(surfaces, sp, glossary=glossary, name_evidence=ctx.name_evidence())
    fails = [r["msg"] for r in res if r["level"] == "FAIL"]
    warns = [r["msg"] for r in res if r["level"] != "FAIL"]
    det = [f"FAIL {m}" for m in fails] + [f"WARN {m}" for m in warns] + [f"found {o}" for o in ok]
    checked = ", ".join(lab for lab, _ in surfaces)
    if fails:
        return gate("G11", "FAIL", "; ".join(fails[:3]) + (f" (+{len(fails) + len(warns) - 3} more)" if len(fails) + len(warns) > 3 else ""), det)
    if warns:
        return gate("G11", "WARN", "; ".join(warns[:4]) + (f" (+{len(warns) - 4} more)" if len(warns) > 4 else ""), det)
    return gate("G11", "PASS", (f"{len(ok)} number/name claim(s) found in the clip's words" if ok else
                                "no unspoken numbers, names or words") + f" ({checked})", det)


def check_cover_file(path: Path) -> list[str]:
    """Problems with a cover JPEG: decode fully, EOI marker, 1080x1920. Empty list = OK."""
    from PIL import Image, ImageFile
    probs = []
    if not path.is_file():
        return ["missing"]
    data = path.read_bytes()
    if len(data) < 4 or data[:3] != b"\xff\xd8\xff":
        return ["not a JPEG (bad magic bytes)"]
    if data.rstrip(b"\x00")[-2:] != b"\xff\xd9":
        probs.append("JPEG has no end-of-image marker (truncated)")
    old = ImageFile.LOAD_TRUNCATED_IMAGES
    ImageFile.LOAD_TRUNCATED_IMAGES = False
    try:
        with Image.open(path) as im:
            im.load()
            size = im.size
        if size != (1080, 1920):
            probs.append(f"size {size[0]}x{size[1]} (want 1080x1920)")
    except Exception as exc:
        probs.append(f"does not decode: {exc}")
    finally:
        ImageFile.LOAD_TRUNCATED_IMAGES = old
    return probs


def dark_bands(row_max, skip=None) -> list[tuple[int, int]]:
    """Runs [r0, r1) of full-width dark rows (max luma <= DARK_LUMA across the row) taller than
    DARK_MIN_FRAC of the height. `skip` = (r0, r1) rows that are never dark (the caption band)."""
    rm = np.asarray(row_max, dtype=np.float64).copy()
    h = rm.size
    if h == 0:
        return []
    if skip:
        a, b = max(0, int(skip[0])), min(h, int(math.ceil(skip[1])))
        rm[a:b] = 255.0
    dark = rm <= DARK_LUMA
    out, k = [], 0
    while k < h:
        if dark[k]:
            j = k
            while j < h and dark[j]:
                j += 1
            if (j - k) / h > DARK_MIN_FRAC:
                out.append((k, j))
            k = j
        else:
            k += 1
    return out


def _band_text(bands, h: int, H: int) -> str:
    return ", ".join(f"y {int(round(a * H / h))}-{int(round(b * H / h))} ({(b - a) / h:.0%})" for a, b in bands)


def _same_speaker(a, b) -> bool:
    """cover.same_speaker (the match `hc cover` picks its frame with: prefix of name words,
    case and accents ignored); exact text when the cover module cannot be loaded."""
    try:
        from . import cover as cover_mod
        return bool(cover_mod.same_speaker(a, b))
    except Exception:
        return str(a) == str(b)


def _cover_rows(path: Path) -> np.ndarray | None:
    """Per-row max luma of a cover image at 1/LUMA_SCALE size."""
    from PIL import Image
    try:
        with Image.open(path) as im:
            g = im.convert("L")
            g = g.resize((max(1, g.width // LUMA_SCALE), max(1, g.height // LUMA_SCALE)), Image.BOX)
            return np.asarray(g, dtype=np.uint8).max(axis=1)
    except Exception:
        return None


def _stamp_time(value) -> float | None:
    """Epoch seconds of a "created" stamp in either form the tool writes ("...+00:00" in cover
    and QA json, "...-0500" in render logs); None when it does not parse."""
    if not isinstance(value, str) or not value:
        return None
    text = value.replace("Z", "+0000")
    for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S.%f%z"):
        try:
            return datetime.strptime(text, fmt).timestamp()
        except ValueError:
            continue
    return None


def _cover_outdated(ctx: _Ctx, cid: str, style: str, plan: dict | None, meta: dict) -> list[str]:
    """Why covers/<clip>__<style>.jpg is not the cover `hc cover` would make now: the words on
    it are not the clip's headline (G11 reads the clips.json text, never the JPEG), or its
    render was made again after it (the frame it was cut from is gone). [] = current, or not
    judgeable: a cover json with no "headline", a stamp that is missing or does not parse."""
    out = []
    clips = ctx.spec.get("clips") if isinstance(ctx.spec, dict) else None
    clip = (clips or {}).get(cid) or {}
    if meta.get("headline") is not None:
        try:
            from . import cover as cover_mod
            want = str(cover_mod.headline_for(clip, plan))
        except Exception:
            want = None
        if want is not None and str(meta["headline"]) != want:
            out.append(f'it reads "{meta["headline"]}", the clip\'s headline is now "{want}"')
    made = _stamp_time(meta.get("created"))
    if made is not None:
        lp = Path(ctx.project.render_path(cid, "vertical", style, ext=".log.json"))
        try:
            log = env.read_json(lp) if lp.is_file() else None
        except env.HCError:
            log = None
        r_t = _stamp_time(log.get("created")) if isinstance(log, dict) else None
        if r_t is not None and r_t > made:
            out.append(f"the render {ctx.project.rel(lp.with_name(lp.name[:-len('.log.json')] + '.mp4'))} "
                       "was made again after it, so its frame is from a render that is gone")
    return out


def g12_cover(ctx: _Ctx, cid: str, style: str, plan: dict | None, render_stale: bool = False) -> dict:
    """The cover file, judged against the platform's fixed 240-1680 grid crop (never the
    preset's own band): headline (ink box when the generator records it) inside the band, the
    head not cut by the tile, a face found and clear of the headline, no dark band, and the
    generator's own warnings surfaced. A cover that is not the one `hc cover` would make now
    (other words on it than the clip's headline, or a render made again after it) is a FAIL,
    as is one recorded with a negative shift."""
    from .cover_frames import box_overlap
    proj = ctx.project
    target = f"cover/{style}"
    main = Path(proj.cover_path(cid, style, "main"))
    probs = check_cover_file(main)
    if probs == ["missing"]:
        return gate("G12", "FAIL", f"no cover {proj.rel(main)}: run hc cover (a platform picks frame 0 without one)", target=target)
    if probs:
        return gate("G12", "FAIL", f"{proj.rel(main)}: " + "; ".join(probs), target=target)
    fails, warns, gap = [], [], False
    for kind in ("grid", "mobile"):
        p = Path(proj.cover_path(cid, style, kind))
        if not p.is_file():
            warns.append(f"{kind} preview missing")
    jp = Path(proj.cover_path(cid, style, "json"))
    meta = None
    if jp.is_file():
        try:
            meta = env.read_json(jp)
        except env.HCError:
            meta = None
    rows = _cover_rows(main)
    if rows is not None:
        # Only the photo above the tint ramp: the ramp darkens the lower cover on purpose.
        hb0 = (meta or {}).get("ink_box") or (meta or {}).get("headline_box")
        top = float(hb0[1]) - COVER_TINT_LEAD if hb0 else COVER_TINT_FROM * 1920
        h = rows.size
        cut = max(0, min(h, int(top * h / 1920)))
        bands = dark_bands(rows, skip=(cut, h)) if cut > 0 else []
        if bands:
            fails.append(f"dark band on the cover: {_band_text(bands, h, 1920)} above the headline "
                         "(letterbox bars or a black frame in the picture)")
    faces_n = None
    if meta is None:
        warns.append("cover json missing: headline band and face overlap not checked")
        gap = True
    else:
        g0, g1 = GRID_BAND
        hb = meta.get("ink_box") or meta.get("headline_box")
        what = "headline ink" if meta.get("ink_box") else "headline box"
        if hb:
            if hb[1] < g0 - 0.5 or hb[1] + hb[3] > g1 + 0.5 or hb[0] < 0 or hb[0] + hb[2] > 1080:
                fails.append(f"{what} [{hb[0]:.0f},{hb[1]:.0f},{hb[2]:.0f},{hb[3]:.0f}] leaves the {g0}-{g1} grid band "
                             "(the profile tile crops it)")
        crown = meta.get("crown_y")
        # crown_method "image": the top of the head was found in the picture. Anything else
        # ("estimate", or a cover made before the field existed) is the face box top minus a
        # share of its height, which sits far too high on a bald or close-cropped head: an
        # estimate never decides a head-cut finding, only the face box itself does.
        measured = str(meta.get("crown_method") or "") == "image"
        if crown is None:            # covers made before crown_y was recorded: read the warning
            m = next((re.search(r"face top at y (-?\d+(?:\.\d+)?)", str(w)) for w in meta.get("warnings") or []
                      if re.search(r"face top at y (-?\d+(?:\.\d+)?)", str(w))), None)
            crown = float(m.group(1)) if m else None
            measured = False
        # A close-up whose estimated crown is above the tile even unshifted cannot fit the tile at
        # any shift: the generator then keeps the forehead inside and shifts at most
        # CLOSEUP_SHIFT_MAX (WARN). FAIL when the face box itself is cut, when the headline shift
        # pushed a head that fitted out of the tile, or when a close-up was shifted further.
        fb0 = meta.get("face_box")
        try:
            sh = float(meta.get("shift") or 0)
        except (TypeError, ValueError):
            sh = 0.0
        if sh < 0:
            # cover.shift only slides the photo up: a negative one never moved a pixel, but a
            # cover made before that was refused recorded its face and head that far too low.
            fails.append(f"cover was made with a negative shift ({sh:.0f}), so its recorded face position is "
                         "wrong: set cover.shift to 0 or more and re-run " + ctx.cmd("cover", cid))
            sh = 0.0
        forehead = float(fb0[1]) if fb0 else None
        if forehead is not None and forehead < g0 - 0.5:
            fails.append(f"face cut by the grid tile: the face box starts at y {forehead:.0f}, above {g0} "
                         + ("(pick another frame with cover.at_out, or a smaller cover.shift)" if sh > 0 else
                            "(pick another frame with cover.at_out: cover.shift only slides the photo up)"))
        elif crown is not None and float(crown) < g0 and measured:
            if float(crown) + sh >= g0 or forehead is None:
                fails.append(f"head cut by the grid tile: the top of the head is at y {float(crown):.0f}, above {g0}"
                             + (f" (the {sh:.0f} px headline shift pushed it out: shorten the headline or set "
                                "cover.shift)" if sh and float(crown) + sh >= g0 else ""))
            elif sh > CLOSEUP_SHIFT_MAX:
                fails.append(f"close-up shifted {sh:.0f} px (more than {CLOSEUP_SHIFT_MAX}): the grid tile cuts "
                             f"the head down to the forehead (top of the head at y {float(crown):.0f}); shorten "
                             "the headline or pick another frame")
            else:
                warns.append(f"close-up: the grid tile trims the top of the head (y {float(crown):.0f}, above "
                             f"{g0} even unshifted); the face stays inside")
        faces_n = meta.get("faces_found")
        if faces_n is None:
            faces_n = len(meta.get("faces") or []) or (1 if meta.get("face_box") else 0)
        tb = meta.get("text_box") or hb
        fb = meta.get("face_box")
        if not faces_n:
            warns.append("no face found: face overlap unchecked")
        elif tb and fb and box_overlap(tb, fb) > 0:
            warns.append("headline covers the face (pick another frame with cover.at_out or shorten the headline)")
        if meta.get("font_fallback"):
            warns.append("the headline font fell back to another face (check the cover lettering)")
        fsp, csp = meta.get("frame_speaker"), ctx.spec["clips"].get(cid, {}).get("speaker") if ctx.spec.get("clips") else None
        if fsp and csp and str(fsp) != str(csp) and not _same_speaker(fsp, csp):
            warns.append(f"cover frame shows {fsp}, not the clip speaker {csp}")
        for w in meta.get("warnings") or []:
            w = str(w)
            if "no face" in w.lower() and not faces_n:
                continue
            if ("face top at y" in w or "top of the head" in w) and \
                    (not measured or (crown is not None and float(crown) < g0)):
                continue        # judged above when measured; an estimate is not a finding
            if w not in warns:
                warns.append(f"cover: {w}")
        why = _cover_outdated(ctx, cid, style, plan, meta)
        if why:
            fails.append("cover is not the one hc cover would make now (" + "; ".join(why) + "): "
                         + ("re-render, then re-cover (" + ctx.cmd("render", cid) + ", then " + ctx.cmd("cover", cid) + ")"
                            if render_stale else "re-run " + ctx.cmd("cover", cid)))
        elif plan and meta.get("spec_hash") and plan.get("spec_hash") and meta["spec_hash"] != plan["spec_hash"]:
            if render_stale:
                warns.append("cover predates the current plan and the render is stale too: re-render, then re-cover ("
                             + ctx.cmd("render", cid) + ", then " + ctx.cmd("cover", cid) + ")")
            else:
                warns.append("cover predates the current plan: re-run " + ctx.cmd("cover", cid))
    det = [f"FAIL {f}" for f in fails] + [f"WARN {w}" for w in warns]
    if fails:
        return gate("G12", "FAIL", "; ".join(fails + warns), det, target=target, gap=gap)
    if warns:
        return gate("G12", "WARN", "; ".join(warns), det, target=target, gap=gap)
    return gate("G12", "PASS", f"decodes, 1080x1920, headline inside the {GRID_BAND[0]}-{GRID_BAND[1]} grid band, "
                f"{faces_n} face(s) found and clear of the headline, no dark band, grid + mobile present", target=target)


def surface_echo(a: str, b: str) -> float | None:
    """Two surfaces of one clip say the same thing: overlap coefficient >= ECHO_OVERLAP over
    stemmed content tokens with at least ECHO_MIN_SHARED of them shared. Returns the overlap, or
    None."""
    sa = {tn.stem(t) for t in tn.content_tokens(a)}
    sb = {tn.stem(t) for t in tn.content_tokens(b)}
    if not sa or not sb or len(sa & sb) < ECHO_MIN_SHARED:
        return None
    oc = len(sa & sb) / min(len(sa), len(sb))
    return oc if oc >= ECHO_OVERLAP else None


def near_duplicate(a: str, b: str) -> str | None:
    """Two hook texts say the same thing: at least DUP_MIN_SHARED shared stemmed content tokens
    with Jaccard >= DUP_JACCARD, or an overlap coefficient >= DUP_OVERLAP with both sets at least
    DUP_MIN_SET tokens. Returns a short description, or None."""
    sa = {tn.stem(t) for t in tn.content_tokens(a)}
    sb = {tn.stem(t) for t in tn.content_tokens(b)}
    if not sa or not sb:
        return None
    shared = len(sa & sb)
    jac = shared / len(sa | sb)
    ov = shared / min(len(sa), len(sb))
    if (shared >= DUP_MIN_SHARED and jac >= DUP_JACCARD) or (ov >= DUP_OVERLAP and min(len(sa), len(sb)) >= DUP_MIN_SET):
        return f"{shared} shared word(s), Jaccard {jac:.2f}, overlap {ov:.2f}"
    return None


def compute_set_issues(ctx: _Ctx) -> dict:
    """G13 issues per clip id over every clip in clips.json."""
    clips = ctx.spec.get("clips") or {}
    info = {}
    for cid, clip in clips.items():
        plan = ctx.file_plan(cid)
        ranges = []
        if plan:
            for p in plan.get("pieces") or []:
                if p.get("src"):
                    ranges.append((float(p["src"][0]), float(p["src"][1])))
        info[cid] = {"ranges": ranges, "hook": _hook_text(plan, clip),
                     "head": (clip.get("cover") or {}).get("headline") or "",
                     "title": str(((clip.get("post") or {}) if isinstance(clip.get("post"), dict) else {}).get("title") or ""),
                     "kind": clip.get("kind")}
    issues = {cid: [] for cid in clips}
    ids = list(clips)
    for a_i, a in enumerate(ids):
        for b in ids[a_i + 1:]:
            if info[a]["kind"] != "caption_only" and info[b]["kind"] != "caption_only":
                ov = 0.0
                for s0, e0 in info[a]["ranges"]:
                    for s1, e1 in info[b]["ranges"]:
                        ov += max(0.0, min(e0, e1) - max(s0, s1))
                if ov > SET_OVERLAP_S:
                    msg = f"shares {ov:.1f}s of source with {{other}}"
                    issues[a].append(msg.format(other=b))
                    issues[b].append(msg.format(other=a))
            ha, hb_ = info[a]["hook"], info[b]["hook"]
            if ha and hb_:
                nd = near_duplicate(ha, hb_)
                if nd:
                    issues[a].append(f"hook card near-duplicates {b} ({nd})")
                    issues[b].append(f"hook card near-duplicates {a} ({nd})")
    for cid in ids:
        surf = [(n, info[cid][k]) for n, k in (("hook card", "hook"), ("cover headline", "head"), ("post title", "title")) if info[cid][k]]
        for i in range(len(surf)):
            for j in range(i + 1, len(surf)):
                oc = surface_echo(surf[i][1], surf[j][1])
                if oc is not None:
                    issues[cid].append(f"{surf[i][0]} and {surf[j][0]} echo each other (overlap {oc:.2f}); give each surface its own job")
    missing = [cid for cid in ids if not info[cid]["ranges"] and info[cid]["kind"] != "caption_only"]
    ctx.set_missing = missing
    return issues


def g13_set(ctx: _Ctx, cid: str) -> dict:
    iss = ctx.set_issues.get(cid) or []
    missing = [m for m in getattr(ctx, "set_missing", []) if m != cid]
    note = f" ({len(missing)} other clip(s) have no plan yet, overlap not checked for them)" if missing else ""
    if iss:
        return gate("G13", "WARN", "; ".join(iss[:3]) + note, iss, target="set")
    n = len(ctx.spec.get("clips") or {})
    return gate("G13", "PASS", f"no overlap, duplicate hook or surface echo across {n} clip(s){note}", target="set")


def g14_accent(ctx: _Ctx, plan: dict, clip: dict, style: str, fmt: str, cap: dict | None, target: str) -> dict:
    hc = (plan or {}).get("hook_card") or clip.get("hook_card") or {}
    phrases = accent_phrases((hc or {}).get("lines") if isinstance(hc, dict) else None)
    if not phrases:
        return gate("G14", "SKIPPED", "no hook-card accent phrases", target=target, reason="not applicable")
    try:
        from . import presets
        pre = presets.for_format(presets.load(style, ctx.project), fmt)
        sources = ((pre.get("caption") or {}).get("emphasis") or {}).get("sources") or []
    except Exception:
        sources = ["hook"]
    if "hook" not in sources:
        return gate("G14", "SKIPPED", f"style {style} does not colour hook accents", target=target, reason="not applicable")
    if cap is None:
        return gate("G14", "SKIPPED", "captions json missing", target=target, gap=True)
    acc = cap.get("accents") or {}
    matched = acc.get("matched") or []
    norm_matched = set()
    for m in matched:
        if isinstance(m, str):
            norm_matched.add(" ".join(tn.norm_tokens(m)))
        elif isinstance(m, dict) and m.get("phrase"):
            norm_matched.add(" ".join(tn.norm_tokens(str(m["phrase"]))))
    missing = [p for p in phrases if " ".join(tn.norm_tokens(p)) not in norm_matched]
    if missing:
        spoken = " " + " ".join(t for t, _, _ in tn.norm_stream(
            [t for _i, t in intent_sequence(plan, prefer="text")])) + " "
        notes = []
        for m in missing:
            key = " ".join(tn.norm_tokens(m))
            why = ("spoken, but never shown highlighted (its page may sit under the hook card, or emphasis was capped)"
                   if key and f" {key} " in spoken else "not spoken in the clip (or spelled differently)")
            notes.append(f'"{m}": {why}')
        return gate("G14", "WARN", "accent never fired in the captions: " + "; ".join(notes), notes, target=target)
    return gate("G14", "PASS", f"{len(phrases)} accent phrase(s) matched in the captions", target=target)


def _flatten_strings(obj) -> list[str]:
    out = []
    if isinstance(obj, str):
        out.append(obj)
    elif isinstance(obj, dict):
        for v in obj.values():
            out.extend(_flatten_strings(v))
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            out.extend(_flatten_strings(v))
    return out


def g15_log(ctx: _Ctx, cid: str, fmt: str, style: str, target: str) -> dict:
    lp = Path(ctx.project.render_path(cid, fmt, style, ext=".log.json"))
    if not lp.is_file():
        return gate("G15", "SKIPPED", f"render log missing: {ctx.project.rel(lp)}", target=target, gap=True)
    try:
        log = env.read_json(lp)
    except env.HCError as exc:
        return gate("G15", "SKIPPED", f"render log unreadable: {exc}", target=target, gap=True)
    lines = _flatten_strings(log.get("warnings") if isinstance(log, dict) else None)
    deny, warn = env.scan_stderr("\n".join(lines))
    if deny:
        return gate("G15", "FAIL", f"{len(deny)} timestamp-damage line(s) recorded: " + " | ".join(deny[:2]), deny, target=target)
    if warn:
        return gate("G15", "WARN", f"{len(warn)} decode warning(s) recorded: " + " | ".join(warn[:2]), warn, target=target)
    return gate("G15", "PASS", f"no timestamp-damage or decode warnings in {len(lines)} logged line(s)", target=target)


def _shot_multi(shot: dict) -> bool:
    if "multi_face" in shot:
        return bool(shot.get("multi_face"))
    main = [f for f in shot.get("faces") or [] if not f.get("minor")]
    return shot.get("kind") in ("two", "group") or len(main) >= 2


def multi_face_spans(plan: dict, shots: dict) -> list[tuple[float, float, dict]]:
    """Out-time spans [(o0, o1, shot)] whose source lies in a multi-face shot."""
    out = []
    multi = [s for s in (shots or {}).get("shots") or [] if _shot_multi(s)]
    if not multi:
        return out
    for seg in plan.get("segments") or []:
        try:
            s0, s1 = (float(x) for x in seg["src"])
            o0 = float(seg["out"][0])
        except (KeyError, TypeError, ValueError, IndexError):
            continue
        for sh in multi:
            a = max(s0, float(sh.get("start_src", 0.0)))
            b = min(s1, float(sh.get("end_src", 0.0)))
            if b > a:
                out.append((o0 + (a - s0), o0 + (b - s0), sh))
    return out


def framing_attribution(plan: dict, clip: dict, shots: dict, windows: list, speakers: dict,
                        aspect: float = 16 / 9) -> tuple[str, list[str], str]:
    """G16 core (pure): (status, notes, summary). See g16_framing. A window is checked when it
    is a crop, or the `rect` tile of a letterbox mark, and does not hold every main face of
    its multi-face shot: what decides is where the window sits (each face x inside lo..hi),
    never how wide it is. A window that also records its rows (y_lo / y_hi) must hold the
    speaker's y too, when clips.json gives one. `aspect` (source width / height) is kept for
    callers; the test no longer needs a face width."""
    spans = multi_face_spans(plan, shots)
    if not spans and not any(w.get("multi_face") for w in windows or []):
        return "SKIPPED", [], "no multi-face shot in this clip"

    def shot_at(t0: float, t1: float):
        best, ov = None, 0.0
        for a, b, sh in spans:
            o = min(b, t1) - max(a, t0)
            if o > ov:
                best, ov = sh, o
        return best

    single = []           # (window, shot) for single-face crops of multi-face shots
    for w in windows or []:
        try:
            t0, t1 = (float(x) for x in w["out"])
        except (KeyError, TypeError, ValueError):
            continue
        sh = shot_at(t0, t1)
        if sh is None and not w.get("multi_face"):
            continue
        if w.get("mode") not in G16_WINDOW_MODES:
            continue
        try:
            lo, hi = float(w["lo"]), float(w["hi"])
        except (KeyError, TypeError, ValueError):
            continue
        faces = [f for f in (sh or {}).get("faces") or [] if not f.get("minor")] if sh else []
        if len(faces) >= 2:
            try:
                xs = [float(f["x"]) for f in faces]
            except (KeyError, TypeError, ValueError):
                xs = [-1.0]              # a face with no x cannot be shown to be inside
            if not all(lo - WINDOW_SLACK <= x <= hi + WINDOW_SLACK for x in xs):
                single.append(w)
        elif hi - lo < (0.0 if not w.get("multi_face") else 1.0):
            single.append(w)
    if not single:
        return "PASS", [], f"{len(spans)} multi-face span(s), every crop and rect tile shows all faces, or it is a letterbox"
    roles = {int(pc.get("n", k)): str(pc.get("role") or "") for k, pc in enumerate(plan.get("pieces") or [])}
    words = [w for w in plan.get("words") or []
             if w.get("s_out") is not None and w.get("e_out") is not None
             and not w.get("omitted") and not w.get("removed")]
    default = clip.get("speaker")
    unlabeled, no_x, bad = [], {}, []
    checked = 0
    for wd in words:
        mid = (float(wd["s_out"]) + float(wd["e_out"])) / 2
        win = next((w for w in single if float(w["out"][0]) <= mid < float(w["out"][1])), None)
        if win is None:
            continue
        checked += 1
        label = wd.get("sp") or default
        if not label:
            unlabeled.append(wd)
            continue
        x = (speakers or {}).get(label, {}).get("x") if isinstance((speakers or {}).get(label), dict) else None
        if x is None:
            no_x.setdefault(label, []).append(wd)
            continue
        inside = float(win["lo"]) - WINDOW_SLACK <= float(x) <= float(win["hi"]) + WINDOW_SLACK
        if inside:                      # rows, when both the log and clips.json give them
            try:
                y, y_lo, y_hi = float(speakers[label]["y"]), float(win["y_lo"]), float(win["y_hi"])
                if y_hi > y_lo:         # a crop that travelled up or down has no common rows: x decides
                    inside = y_lo - WINDOW_SLACK <= y <= y_hi + WINDOW_SLACK
            except (KeyError, TypeError, ValueError):
                pass
        if not inside:
            reaction = bool(win.get("reaction")) or roles.get(wd.get("piece"), "") == "reaction"
            bad.append((wd, label, reaction))
    statuses, notes = [], []
    if unlabeled:
        statuses.append("FAIL")
        notes.append(f"FAIL multi-face crop with no speaker labels or clip speaker ({len(unlabeled)} word(s), "
                     f"first at out {float(unlabeled[0]['s_out']):.2f}s): set the clip's \"speaker\"")
    for label, ws in no_x.items():
        statuses.append("FAIL")
        notes.append(f'FAIL multi-face shot framed on one face: set clips.json "speakers": {{"{label}": {{"x": ...}}}} '
                     f"(x from the shots sheet) so QA can check who is on screen ({len(ws)} word(s) of {label})")
    runs = []
    for wd, label, reaction in bad:
        s, e = float(wd["s_out"]), float(wd["e_out"])
        if runs and runs[-1]["label"] == label and s - runs[-1]["e"] < G16_MERGE_GAP:
            r = runs[-1]
            r["e"] = max(r["e"], e)
            r["reaction"] = r["reaction"] and reaction
        else:
            runs.append({"label": label, "s": s, "e": e, "reaction": reaction})
    for r in runs:
        dur = r["e"] - r["s"]
        lvl = "FAIL" if dur >= G16_FAIL_S - 1e-9 and not r["reaction"] else "WARN"
        why = " (reaction shot)" if r["reaction"] and dur >= G16_FAIL_S else ""
        statuses.append(lvl)
        notes.append(f"{lvl} framed face is not {r['label']} for {dur:.1f}s at out {r['s']:.2f}s{why}")
    st = worst(statuses) if statuses else "PASS"
    summary = (f"{len(single)} single-face crop or rect window(s) of multi-face shots, {checked} word(s) checked, "
               f"{len(runs)} mismatch span(s)")
    return st, notes, summary


def shots_uncovered(plan: dict, shots: dict, windows: list | None = None) -> tuple[float, float] | None:
    """(seconds of the clip's cropped source the shots analysis never looked at, seconds of
    cropped source in the clip), or None when the shots json does not say what it analysed.
    `windows` (a render log's framing_windows) limits the count to what is shown as a crop or
    as the `rect` tile of a letterbox mark: a plain letterboxed or full-frame stretch needs no
    face data. Without it every segment counts."""
    ranges = shots.get("ranges") or ([shots["range_src"]] if shots.get("range_src") else None)
    if not ranges:
        ss = [s for s in shots.get("shots") or [] if s.get("start_src") is not None and s.get("end_src") is not None]
        ranges = [[s["start_src"], s["end_src"]] for s in ss] or None
    if not ranges:
        return None
    try:
        rs = sorted((float(a), float(b)) for a, b in ranges)
    except (TypeError, ValueError):
        return None
    crops = None
    if windows:
        crops = []
        for w in windows:
            try:
                if w.get("mode") in G16_WINDOW_MODES:
                    crops.append((float(w["out"][0]), float(w["out"][1])))
            except (KeyError, TypeError, ValueError, IndexError, AttributeError):
                continue
    total = miss = 0.0
    for seg in plan.get("segments") or []:
        try:
            s0, s1 = (float(v) for v in seg["src"])
            o0 = float(seg["out"][0]) if crops is not None else 0.0
        except (KeyError, TypeError, ValueError, IndexError):
            continue
        spans = [(s0, s1)]
        if crops is not None:        # the parts of this segment a crop window shows, on the src clock
            spans = [(max(s0, s0 + (a - o0)), min(s1, s0 + (b - o0))) for a, b in crops]
            spans = [(a, b) for a, b in spans if b > a]
        for a0, b0 in spans:
            cur, cov = a0, 0.0
            for a, b in rs:
                lo, hi = max(a, cur), min(b, b0)
                if hi > lo:
                    cov += hi - lo
                    cur = hi
            total += b0 - a0
            miss += max(0.0, (b0 - a0) - cov)
    return miss, total


def g16_shots(ctx: _Ctx, cid: str, clip: dict, plan: dict | None) -> tuple[dict | None, dict | None]:
    """The face data G16 needs, read once per clip: (shots json, None), or (None, gate) when
    there is none. `hc render` analyses shots itself when the framing needs face data, so a
    clip rendered with face tracking has the file; when it is really absent (or was analysed
    for other pieces) that is ONE evidence gap for the clip, not one per style."""
    target = "vertical (all styles)"
    sp = Path(ctx.project.shots_path(cid))
    if not sp.is_file():
        return None, gate("G16", "SKIPPED", "no shots json: who is on screen was not checked for any style ("
                          + ctx.cmd("shots", cid) + ", then look at the sheet)", target=target, gap=True)
    try:
        shots = env.read_json(sp)
    except env.HCError as exc:
        return None, gate("G16", "SKIPPED", f"shots json unreadable: {exc}", target=target, gap=True)
    windows = None
    for st in (clip.get("_eff") or {}).get("styles") or ctx.styles or ["clean"]:
        lp = Path(ctx.project.render_path(cid, "vertical", st, ext=".log.json"))
        try:
            windows = (env.read_json(lp) or {}).get("framing_windows") if lp.is_file() else None
        except env.HCError:
            windows = None
        if windows:
            break
    unc = shots_uncovered(plan or {}, shots, windows) if plan else None
    if unc and unc[0] >= G16_FAIL_S - 1e-9:
        return None, gate("G16", "SKIPPED", f"shots json covers {unc[1] - unc[0]:.1f}s of the {unc[1]:.1f}s of source "
                          "this clip shows cropped (it was analysed for other pieces): " + ctx.cmd("shots", cid),
                          target=target, gap=True)
    return shots, None


def g16_framing(ctx: _Ctx, cid: str, clip: dict, plan: dict | None, fmt: str, style: str, target: str,
                shots: dict | None = None) -> dict:
    """In a multi-face shot cropped to one face (a crop, or the `rect` tile of a letterbox
    mark), the face on screen is the one speaking: each visible word's speaker (word label, else
    the clip speaker) must have its declared x (clips.json "speakers") inside the window the
    render log records. `shots` is the clip's shots json when the caller already read it
    (g16_shots)."""
    if plan is None:
        return gate("G16", "SKIPPED", "no plan", target=target, gap=True)
    if plan.get("kind") == "caption_only" or clip.get("kind") == "caption_only":
        return gate("G16", "SKIPPED", "caption-only clip: the video is already framed", target=target,
                    reason="not applicable")
    if shots is None:
        shots, missing = g16_shots(ctx, cid, clip, plan)
        if missing is not None:
            return dict(missing, target=target)
    lp = Path(ctx.project.render_path(cid, fmt, style, ext=".log.json"))
    try:
        log = env.read_json(lp) if lp.is_file() else {}
    except env.HCError:
        log = {}
    windows = log.get("framing_windows")
    speakers = plan.get("speakers") if isinstance(plan.get("speakers"), dict) else (ctx.spec.get("speakers") or {})
    src = plan.get("source") or {}
    try:
        aspect = float(src.get("width")) / float(src.get("height"))
    except (TypeError, ValueError, ZeroDivisionError):
        aspect = 16 / 9
    if windows is None:
        if not multi_face_spans(plan, shots):
            return gate("G16", "SKIPPED", "no multi-face shot in this clip", target=target, reason="not applicable")
        return gate("G16", "SKIPPED", "render log has no framing_windows (an older render): re-render to check who is on screen",
                    target=target, gap=True)
    st, notes, summary = framing_attribution(plan, clip, shots, windows, speakers, aspect)
    if st == "SKIPPED":
        return gate("G16", "SKIPPED", summary, target=target, reason="not applicable")
    if st == "PASS":
        return gate("G16", "PASS", summary, target=target)
    return gate("G16", st, "; ".join(_bare(n) for n in notes[:3]) + f" ({summary})", notes, target=target)


# ---------------------------------------------------------------- per clip

def _set_styles(spec: dict) -> list[str]:
    out = []
    for c in (spec.get("clips") or {}).values():
        for st in (c.get("_eff") or {}).get("styles") or ["clean"]:
            if st not in out:
                out.append(st)
    return out


def _newer_or_same(sheet: Path, parts: list[Path]) -> bool:
    """A mosaic contains a clip when it was rebuilt after the clip's grid cover."""
    have = [q for q in parts if q.is_file()]
    if not sheet.is_file() or not have:
        return False
    return sheet.stat().st_mtime + 1e-6 >= max(q.stat().st_mtime for q in have)


def _eyes(ctx: _Ctx, cid: str, plan: dict | None, formats: list[str], styles: list[str]) -> list[dict]:
    proj = ctx.project
    out = []

    def add(p: Path, why: str, contains: bool = True, fix: str | None = None):
        ok = p.is_file() and contains
        e = {"path": proj.rel(p), "exists": ok, "why": why}
        if p.is_file() and not contains:
            e["note"] = "does not show this clip (stale)"
        if not ok and fix:
            e["fix"] = fix
        out.append(e)
    for fmt in formats:
        for st in styles:
            add(Path(proj.sheet_path(f"{cid}__{fmt}__{st}.contact.jpg")),
                f"contact sheet ({fmt}, {st}): cell 1 is frame 0 (hook card and default thumbnail); framing, captions, hook card")
            flags = ctx.flag_sheets.get((cid, fmt, st))
            if flags:
                add(Path(flags), f"flagged frames ({fmt}, {st}): the black or flash frames G8 found")
        if plan and (plan.get("joins") or []):
            add(Path(proj.sheet_path(f"{cid}__{fmt}.seams.jpg")),
                f"seams ({fmt}): the base picture just before and just after every join, before captions and "
                "before any dip darkening; the join kind is printed under each pair")
    if "vertical" in formats:
        multi = len(_set_styles(ctx.spec)) > 1
        for st in styles:
            mine = Path(proj.cover_path(cid, st, "grid"))
            add(mine, f"cover grid crop ({st}): this cover as the profile tile shows it")
            sheet = Path(proj.sheet_path(f"covers-grid__{st}.jpg" if multi else "covers-grid.jpg"))
            add(sheet, f"covers grid ({st}): how the profile grid will look", _newer_or_same(sheet, [mine]),
                fix="rebuild it from the covers on disk: hc cover P --grid")
        add(Path(proj.sheet_path(f"{cid}.cover-candidates.jpg")), "cover candidates: is the chosen frame the best one")
    return out


def check_clip(ctx: _Ctx, cid: str) -> dict:
    from . import plan as plan_mod
    clip = ctx.spec["clips"][cid]
    eff = clip.get("_eff") or {}
    formats = list(ctx.formats or eff.get("formats") or ["vertical"])
    styles = list(ctx.styles or eff.get("styles") or ["clean"])
    results = []
    fresh, err = None, None
    try:
        try:  # QA never rewrites plans/<clip>.plan.json
            fresh = plan_mod.build_plan(ctx.project, cid, spec=ctx.spec, write=False)
        except TypeError:
            fresh = plan_mod.build_plan(ctx.project, cid)
    except env.HCError as exc:
        err = str(exc)
    except Exception as exc:
        err = f"{type(exc).__name__}: {exc}"
    results.append(g1_lints(fresh, err))
    plan = fresh or ctx.file_plan(cid)
    lead_by_style = {}
    for st in styles:
        try:
            from . import presets
            lead_by_style[st] = float((presets.load(st, ctx.project).get("caption") or {}).get("lead_ms", 0)) / 1000.0
        except Exception:
            lead_by_style[st] = 0.0
    renders_seen, stale = [], {}
    unsure_words = uncertain_words(ctx.project, cid)
    cut = plan is not None and plan.get("kind") != "caption_only" and clip.get("kind") != "caption_only"
    g16_data, g16_gap = g16_shots(ctx, cid, clip, plan) if cut and "vertical" in formats else (None, None)
    for fmt in formats:
        renders = {}
        for st in styles:
            rp = Path(ctx.project.render_path(cid, fmt, st))
            if rp.is_file():
                renders[st] = rp
                renders_seen.append(ctx.project.rel(rp))
            else:
                tgt = f"{fmt}/{st}"
                gids = ("G0", "G2", "G3", "G8", "G9", "G10", "G14", "G15") + (("G16",) if fmt == "vertical" else ())
                for gid in gids:
                    results.append(gate(gid, "SKIPPED", f"render missing: {ctx.project.rel(rp)}", target=tgt, gap=True))
        atarget = f"{fmt} (audio)"
        for st in renders:
            g0 = g0_fresh(ctx, cid, fmt, st, plan, f"{fmt}/{st}")
            results.append(g0)
            if g0["status"] == "FAIL":
                stale[(fmt, st)] = ctx.project.rel(renders[st])
        if plan is None:
            for gid in ("G5", "G6", "G7"):
                results.append(gate(gid, "SKIPPED", "no plan", target=atarget, gap=True))
            continue
        first = next(iter(renders.values()), None)
        sa = g_styles_audio(ctx, cid, fmt, renders)
        if sa:
            results.append(sa)
        g5, diff, hyp = g5_words(ctx, cid, plan, first, atarget)
        results.append(g5)
        results.append(g6_open_close(fresh, first, atarget, plan, ctx.energy(first), ctx.speech(first)))
        results.append(g7_loudness(first, atarget))
        for st, rp in renders.items():
            tgt = f"{fmt}/{st}"
            results.append(g2_duration(ctx, plan, clip, fmt, rp, tgt))
            results.append(g3_streams(ctx, plan, fmt, rp, tgt))
            cap, cap_path, disabled = _captions(ctx, cid, fmt, st)
            band = (cap or {}).get("band")
            results.append(g8_black_freeze(plan, rp, band, tgt, ctx, cid, fmt, st))
            if (fmt, st) in stale:
                for gid in ("G9", "G10", "G14"):
                    results.append(gate(gid, "SKIPPED", "render is stale (G0): its captions describe an older plan",
                                        target=tgt, gap=True))
            else:
                results.append(_no_captions(clip, disabled, "G9", tgt) or
                               g9_layout(ctx, cid, clip, plan, fmt, st, cap, cap_path, tgt))
                results.append(_no_captions(clip, disabled, "G10", tgt) or
                               g10_sync(ctx, plan, cap, diff, hyp, lead_by_style.get(st, 0.0), tgt,
                                        E=ctx.energy(rp), render=rp, uncertain=unsure_words))
                results.append(_no_captions(clip, disabled, "G14", tgt) or
                               g14_accent(ctx, plan, clip, st, fmt, cap, tgt))
            results.append(g15_log(ctx, cid, fmt, st, tgt))
            if fmt == "vertical" and g16_gap is None:
                results.append(g16_framing(ctx, cid, clip, plan, fmt, st, tgt, shots=g16_data))
        if fmt == "vertical" and g16_gap is not None and renders:
            results.append(g16_gap)              # face data absent: said once for the clip
    results.append(g11_faith(ctx, cid, clip, plan))
    if "vertical" in formats:
        for st in styles:
            results.append(g12_cover(ctx, cid, st, plan, render_stale=("vertical", st) in stale))
    else:
        results.append(gate("G12", "SKIPPED", "no vertical format: covers not required", target="cover", reason="not applicable"))
    results.append(g13_set(ctx, cid))
    order = {g: k for k, g in enumerate(GATES)}
    results.sort(key=lambda r: (order.get(r["id"], 99), r["target"]))
    eyes = _eyes(ctx, cid, plan, formats, styles)
    counts = {s: sum(1 for r in results if r["status"] == s) for s in STATUSES}
    gaps = sum(1 for r in results if r.get("gap")) + sum(1 for e in eyes if not e["exists"])
    status = clip_status(results, gaps)
    return {"version": QA_VERSION, "clip": cid, "title": clip.get("title"), "created": _now(),
            "code_version": env.code_version(), "qa_code": qa_code(), "status": status, "complete": gaps == 0,
            "counts": counts, "gaps": gaps,
            "styles": styles, "formats": formats, "checked": {"styles": styles, "formats": formats},
            "renders": renders_seen, "stale_renders": sorted(stale.values()),
            "plan_spec_hash": (plan or {}).get("spec_hash"), "gates": results, "eyes": eyes}


# ---------------------------------------------------------------- staleness (shared with export)

def qa_code() -> str | None:
    """Identity of the code that makes QA verdicts (env.package_digest: the version and every
    module of the package, since the gates lean on claims, textnorm, align, speechband and
    more). Written into each report as "qa_code"; None when the digest is not available."""
    digest = getattr(env, "package_digest", None)
    if digest is None:
        return None
    try:
        return str(digest())
    except Exception:
        return None


def current_spec_hash(spec: dict, cid: str) -> str | None:
    try:
        from . import spec as specmod
        return specmod.clip_hash(spec, cid)
    except Exception:
        return None


def qa_staleness(project, spec: dict, cid: str, rep: dict | None, qa_file: Path | None = None) -> list[str]:
    """Why a recorded qa/<clip>.json no longer describes the clip: a checked render, its
    captions or the cover changed after the QA ran, the clips.json entry changed (plan
    spec_hash), QA itself found the render stale (G0), or the checks themselves changed since
    (the report's "qa_code" is not this code's; a report from before the field counts as
    changed). Empty list = current."""
    if not rep:
        return ["QA has not run"]
    qa_file = Path(qa_file or project.qa_path(cid, ".json"))
    reasons = []
    try:
        qm = qa_file.stat().st_mtime
    except OSError:
        return ["QA json missing"]
    checked = rep.get("checked") or {"styles": rep.get("styles") or [], "formats": rep.get("formats") or []}
    newer = []
    for fmt in checked.get("formats") or []:
        for st in checked.get("styles") or []:
            for ext in (".mp4", ".captions.json"):
                q = Path(project.render_path(cid, fmt, st, ext=ext))
                try:
                    if q.stat().st_mtime > qm + 1e-6:
                        newer.append(project.rel(q))
                except OSError:
                    pass
            if fmt == "vertical":
                for kind in ("json", "main"):
                    q = Path(project.cover_path(cid, st, kind))
                    try:
                        if q.stat().st_mtime > qm + 1e-6:
                            newer.append(project.rel(q))
                    except OSError:
                        pass
    if newer:
        reasons.append("changed since QA: " + ", ".join(sorted(set(newer))[:4]))
    cur = current_spec_hash(spec, cid)
    if cur and rep.get("plan_spec_hash") and cur != rep.get("plan_spec_hash"):
        reasons.append("the clips.json entry changed since QA")
    if rep.get("stale_renders"):
        reasons.append("QA found stale render(s) (G0): " + ", ".join(rep["stale_renders"][:3]))
    code = qa_code()
    if code is not None and rep.get("qa_code") != code:
        reasons.append(QA_CODE_CHANGED)
    return reasons


# ---------------------------------------------------------------- reports

def _bare(note: str) -> str:
    """A detail line without its leading status word (for evidence strings)."""
    return re.sub(r"^(FAIL|WARN) ", "", note)


def _md_cell(s: str) -> str:
    return str(s).replace("|", "/").replace("\n", " ")


STATUS_MEANING = {"PASS": "every gate that ran passed and nothing was left unchecked",
                  "WARN": "no FAIL; read every WARN",
                  "FAIL": "at least one gate failed",
                  "INCOMPLETE": "no WARN or FAIL, but some checks could not run (evidence gaps): not a PASS"}


def clip_markdown(rep: dict) -> str:
    chk = rep.get("checked") or {"styles": rep.get("styles") or [], "formats": rep.get("formats") or []}
    lines = [f"# QA: {rep['clip']}" + (f" ({rep['title']})" if rep.get("title") else ""), "",
             f"Status **{rep['status']}** ({STATUS_MEANING.get(rep['status'], '')}): "
             + ", ".join(f"{rep['counts'][s]} {s}" for s in STATUSES) +
             f"; {rep['gaps']} evidence gap(s). Generated {rep['created']}.", "",
             f"Checked styles {', '.join(chk.get('styles') or []) or '-'}; formats {', '.join(chk.get('formats') or []) or '-'}."]
    if rep.get("stale_renders"):
        lines.append("Stale renders (G0): " + ", ".join(rep["stale_renders"]) + ".")
    lines += ["", "| Gate | Target | Status | Evidence |", "|---|---|---|---|"]
    for r in rep["gates"]:
        st = r["status"] + (" (gap)" if r.get("gap") else "")
        lines.append(f"| {r['id']} {r['name']} | {_md_cell(r['target'])} | {st} | {_md_cell(r['evidence'])} |")
    det = [r for r in rep["gates"] if r["details"] and r["status"] in ("FAIL", "WARN")]
    if det:
        lines += ["", "## Details", ""]
        for r in det:
            lines.append(f"### {r['id']} {r['name']} ({r['target']}): {r['status']}")
            lines += [f"- {_md_cell(d)}" for d in r["details"][:40]]
            lines.append("")
    lines += ["", "## EYES (open every image below and look before calling this clip done)", ""]
    for e in rep["eyes"]:
        miss = "" if e["exists"] else f" **({e.get('note') or 'missing'}: evidence gap)**"
        lines.append(f"- `{e['path']}`: {e['why']}" + miss)
    if not rep["eyes"]:
        lines.append("- none")
    return "\n".join(lines) + "\n"


def summary_markdown(reports: list[dict], exit_code: int, ctx: _Ctx) -> str:
    """qa/summary.md over EVERY clip in clips.json: the clips just checked, plus the recorded
    qa/<clip>.json of the others, marked "not re-run" and "stale" when they no longer describe
    the clip."""
    meaning = {0: "no FAIL", 1: "at least one FAIL", 2: "evidence gaps with --require-coverage",
               3: "a tool error stopped some checks"}
    now = {r["clip"]: r for r in reports}
    proj = ctx.project
    lines = ["# QA summary", "",
             f"Exit {exit_code} for the clip(s) checked in this run ({', '.join(now) or '-'}): "
             f"{meaning.get(exit_code, '')}. Generated {_now()}."
             + ("" if ctx.asr else " ASR gates were skipped (--no-asr): run a full QA with ASR before the final export."), "",
             "INCOMPLETE = no WARN or FAIL, but some checks could not run: not a PASS.", "",
             "| Clip | Status | PASS | WARN | FAIL | SKIPPED | Gaps | Top issue |", "|---|---|---|---|---|---|---|---|"]
    shown = []
    for cid in list((ctx.spec.get("clips") or {}).keys()):
        rep = now.get(cid)
        marks = []
        if rep is None:
            qp = Path(proj.qa_path(cid, ".json"))
            try:
                rep = env.read_json(qp) if qp.is_file() else None
            except env.HCError:
                rep = None
            if rep is None or not isinstance(rep.get("counts"), dict):
                lines.append(f"| {cid} | not checked | - | - | - | - | - | run {ctx.cmd('qa', cid)} |")
                continue
            marks.append(f"not re-run: checked {rep.get('created') or '?'}")
            if qa_staleness(proj, ctx.spec, cid, rep, qp):
                marks.append("stale")
        elif rep.get("stale_renders"):
            marks.append("stale")
        shown.append(rep)
        top = next((r for r in rep.get("gates") or [] if r["status"] == "FAIL"), None) or \
            next((r for r in rep.get("gates") or [] if r["status"] == "WARN"), None) or \
            next((r for r in rep.get("gates") or [] if r.get("gap")), None)
        top_s = f"{top['id']} {top['target']}: {top['evidence']}" if top else "-"
        c = rep["counts"]
        st = rep.get("status", "?") + (f" ({'; '.join(marks)})" if marks else "")
        lines.append(f"| {cid} | {st} | {c.get('PASS', 0)} | {c.get('WARN', 0)} | {c.get('FAIL', 0)} | "
                     f"{c.get('SKIPPED', 0)} | {rep.get('gaps', 0)} | {_md_cell(top_s)[:160]} |")
    if ctx.tool_errors:
        lines += ["", "## Tool errors", ""] + [f"- {_md_cell(e)}" for e in ctx.tool_errors]
    lines += ["", "## EYES", ""]
    seen = set()
    for rep in shown:
        for e in rep.get("eyes") or []:
            if e["path"] in seen:
                continue
            seen.add(e["path"])
            lines.append(f"- `{e['path']}`" + ("" if e["exists"] else f" ({e.get('note') or 'missing'})"))
    lines += ["", "Per-clip detail: `qa/<clip>.md`."]
    return "\n".join(lines) + "\n"


def exit_code(reports: list[dict], tool_errors: list, require_coverage: bool) -> int:
    """3 tool error (an ASR backend crash counts) > 1 any FAIL > 2 gaps with --require-coverage > 0."""
    if tool_errors:
        return 3
    if any(r["status"] == "FAIL" for r in reports):
        return 1
    if require_coverage and any(r["gaps"] for r in reports):
        return 2
    return 0


# ---------------------------------------------------------------- entry points

def run(project, clip_ids=None, styles=None, asr: bool = True, require_coverage: bool = False,
        formats=None) -> int:
    """Run every gate for the clips; write qa/<clip>.json + .md and qa/summary.md (covering every
    clip in clips.json); return the exit code. `styles` / `formats` override each clip's own lists."""
    env.ffmpeg()
    env.ffprobe()
    spec = project.spec()
    all_ids = list((spec.get("clips") or {}).keys())
    ids = list(clip_ids or []) or all_ids
    for c in ids:
        if c not in all_ids:
            raise env.unknown_clip(c, all_ids)
    ctx = _Ctx(project, spec, styles, asr, require_coverage, formats)
    ctx.set_issues = compute_set_issues(ctx)
    reports = []
    for cid in ids:
        try:
            rep = check_clip(ctx, cid)
        except env.HCError as exc:
            if exc.code == 3:
                ctx.tool_errors.append(f"{cid}: {exc}")
            st = "FAIL" if exc.code != 3 else "SKIPPED"
            rep = {"version": QA_VERSION, "clip": cid, "title": spec["clips"][cid].get("title"), "created": _now(),
                   "code_version": env.code_version(), "qa_code": qa_code(),
                   "status": "FAIL" if exc.code != 3 else "INCOMPLETE",
                   "complete": False, "counts": {s: 0 for s in STATUSES}, "gaps": 1, "styles": [], "formats": [],
                   "checked": {"styles": [], "formats": []}, "renders": [], "stale_renders": [],
                   "plan_spec_hash": None, "eyes": [],
                   "gates": [gate("G1", st, f"QA could not run: {exc}", gap=exc.code == 3)]}
            rep["counts"][st] = 1
        env.write_json(project.qa_path(cid, ".json"), rep)
        Path(project.qa_path(cid, ".md")).write_text(clip_markdown(rep), encoding="utf-8")
        reports.append(rep)
        c = rep["counts"]
        env.log("qa", f"{cid}: {rep['status']} ({c['FAIL']} FAIL, {c['WARN']} WARN, {c['SKIPPED']} SKIPPED, {rep['gaps']} gap(s))")
        for r in rep["gates"]:
            if r["status"] in ("FAIL", "WARN"):
                print(f"  {r['status']:4} {r['id']:3} {r['target']}: {r['evidence']}")
            elif r.get("gap"):
                print(f"  GAP  {r['id']:3} {r['target']}: {r['evidence']}")
        for e in rep.get("eyes") or []:
            if not e["exists"]:
                fix = f" ({e['fix']})" if e.get("fix") else ""
                print(f"  GAP  EYES {e['path']}: {e.get('note') or 'missing'}{fix}")
    code = exit_code(reports, ctx.tool_errors, require_coverage)
    summary = summary_markdown(reports, code, ctx)
    Path(project.qa_path("summary", ".md")).write_text(summary, encoding="utf-8")
    for e in ctx.tool_errors:
        print(f"  TOOL ERROR {e}")
    print(f"qa: exit {code}; summary {project.rel(project.qa_path('summary', '.md'))}")
    return code


def cmd_qa(args) -> int:
    """hc qa P [CLIP...] [--styles ...] [--formats ...] [--no-asr] [--require-coverage]"""
    from .project import Project
    project = Project(args.project).require()
    return run(project, list(getattr(args, "clips", None) or []), getattr(args, "styles", None),
               asr=not getattr(args, "no_asr", False),
               require_coverage=bool(getattr(args, "require_coverage", False)),
               formats=getattr(args, "formats", None))
