"""Energy-guided word timing: pauses, voiced runs, a hallucination gate and the word-time repair.

Whisper-family word times make words contiguous: the word after a pause (or after speech the
model did not write down, such as a dropped "um" or a repeated "I") gets its START pulled back
to the previous word's end, so its span covers silence. Its END stays close to the truth. The
repair below moves such starts to the voiced onset that really begins the word, using the
audio energy track (audio.EnergyTrack, 10 ms frames on the src clock).

Everything here is pure: functions take an EnergyTrack and plain word dicts ({"w", "s", "e"},
other keys are carried along) and return new values. Levels (noise floor and local speech level)
are computed once per track on a 0.5 s grid and interpolated per frame.

    pauses(E, a, b)            quiet runs >= 0.20 s (adaptive threshold)
    voiced_runs(E, a, b)       runs above floor + 10 dB lasting >= 30 ms
    sustained_runs(runs)       the runs a word can start on (a lip click is not one)
    next_onset(E, t, limit)    first voiced run start in [t, limit]
    uncovered_voiced(E, spans, a, b)   voiced audio no span covers
    sound_into_starts(E, words)        words that unworded sound runs straight into
    burst_dips(E, s, e)        dips that split a span into separate bursts of sound
    audio_gate(words, E)       drop words with no audible frame (ASR hallucinations)
    repair_times(words, E)     move mis-stamped starts and ends onto the audio

The energy track cannot tell a word from a laugh it starts under (both are loud). BandShapes
(the spectrum's shape per frame, from the transcript wav) can:

    spectral_onset(B, E, pre, s, e)    where [s, e] stops being the sound heard in `pre`
    repair_overlapped(words, E, B)     move starts stamped inside someone else's sound
    uncertain_starts(words, E, B)      starts that stay a guess (flag timing_uncertain)
"""
from __future__ import annotations

import math
import re
import weakref
from bisect import bisect_left, bisect_right
from typing import Any, Iterable, Sequence

import numpy as np

from . import audio

# --- thresholds (dB relative to the local noise floor, seconds) -----------------------------
PAUSE_DB = 6.0           # quiet = below floor + this ...
PAUSE_DB_MAX = 15.0      # ... raised with the local speech level, but never above floor + this
SPEECH_PCT = 90          # local speech level = this percentile of dB within +/- SPEECH_RADIUS
SPEECH_RADIUS = 5.0
SPEECH_DROP = 25.0       # a frame this far below the local speech level counts as quiet
VOICED_DB = 10.0         # voiced = above floor + this
MIN_PAUSE = 0.20         # a pause lasts at least this long
MIN_OVERLAP = 0.15       # pause seconds inside a word span = evidence of a mis-stamp
MIN_VOICED = 0.05        # voiced seconds a moved word must keep
OWN_CONTINUED = 0.06     # seconds at speech level before a pause, in a run begun under the
OWN_LOUD_DB = 12.0       # previous word (within this of the local speech level), that make the
                         # audio the word's own: its end ran on into the pause. A fading tail of
                         # the previous word is quieter than that and does not count.
ONSET_RUN = 0.03         # a voiced run lasts at least this long
ONSET_SUSTAIN = 0.06     # a word onset is a voiced run of this long (a lip click, a tick is shorter) ...
ONSET_ATTACH = 0.04      # ... or a shorter one that such a run follows this closely (a stop burst)
EDGE_SLACK = 0.02
SHORT_GAP = 0.08         # short pass: a start inside a quiet run this long moves to the next onset
OVERLONG_BASE = 0.30     # overlong pass: plausible max duration = base + per_letter * letters
OVERLONG_PER_LETTER = 0.08
OVERLONG_GAP = 0.10      # ... and an overlong word must hold a quiet run this long
OWN_AFTER_PREV = True    # voiced runs starting after the previous word's end are this word's own
ONSET_RULE = False       # (measured worse; kept off) keep the start when it sits on an onset
GATE_PAD = 0.10          # audio_gate looks this far around a word for an audible frame
GRID = 0.5               # seconds between precomputed level points
SNAP_MIN = 0.05          # snap_back: a start this far ...
SNAP_MAX = 0.40          # ... up to this far after a post-pause onset moves back to it
SNAP_LATE_MAX = 0.30     # the same for every backend (a small model starts the word after a pause
                         # late), kept under INTO_MIN so longer unworded sound is still flagged
ABUT_SLACK = 0.02        # uncovered sound "runs out of a word" when voiced this close to its edge
ABUT_DB = 12.0           # ... and counts as speech when its median is this close to speech level
DIP_DB = 18.0            # burst_dips: a dip this far under the span's own level (90th percentile) ...
DIP_MIN = 0.08           # ... lasting this long ...
BURST_DB = 10.0          # ... with sound within this of that level ...
BURST_MIN = 0.05         # ... for this long on each side
INTO_MIN = 0.30          # sound_into_starts: unworded sound of this long running into a word start
SHAPE_EDGES = (100.0, 300.0, 600.0, 1200.0, 2400.0, 4800.0, 8000.0)   # band-shape bands, Hz
SHAPE_WIN = 0.025        # band-shape analysis window, seconds
SHAPE_PRE = 0.6          # the sound before a word is described by at most its last ... seconds
SHAPE_MIN_DB = 8.0       # "no longer that sound": a band shape this far (RMS dB) from it ...
SHAPE_MARGIN = 2.5       # ... and this far beyond that sound's own spread (90th percentile) ...
SHAPE_RUN = 0.06         # ... for this long without a break
SHAPE_AT_START = 0.04    # a change this close to the stamped start confirms it
SHAPE_KEEP = 0.08        # a word moved to a change keeps at least this much of its span

_LEVELS: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()
_LETTER_RE = re.compile(r"[A-Za-z0-9]")


# ---------------------------------------------------------------------------
# Levels (floor, speech level) on a 0.5 s grid
# ---------------------------------------------------------------------------


class _Levels:
    """Lazily filled grid of the noise floor (10th percentile, +/- 20 s, clamped like
    EnergyTrack.floor) and the local speech level (90th percentile, +/- 5 s) at absolute times
    k * GRID."""

    def __init__(self, E: Any):
        self.E = E
        self.k0 = int(math.floor(E.t0 / GRID)) - 1
        k1 = int(math.ceil(E.end / GRID)) + 1
        size = max(1, k1 - self.k0 + 1)
        self.floor = np.full(size, np.nan)
        self.speech = np.full(size, np.nan)
        self.db = np.asarray(E.db, dtype=np.float64)

    def _fill(self, ka: int, kb: int) -> None:
        E = self.E
        fl_pct = getattr(audio, "FLOOR_PCT", 10)
        fl_rad = getattr(audio, "FLOOR_RADIUS", 20.0)
        fl_min = getattr(audio, "FLOOR_MIN_DB", -80.0)
        n = len(self.db)
        for k in range(max(ka, 0), min(kb, len(self.floor))):
            if not math.isnan(self.floor[k]):
                continue
            tc = (self.k0 + k) * GRID
            a = max(0, E.index(tc - fl_rad))
            b = min(n, E.index(tc + fl_rad) + 1)
            self.floor[k] = (max(float(np.percentile(self.db[a:b], fl_pct)), fl_min)
                             if b > a else fl_min)
            a = max(0, E.index(tc - SPEECH_RADIUS))
            b = min(n, E.index(tc + SPEECH_RADIUS) + 1)
            self.speech[k] = float(np.percentile(self.db[a:b], SPEECH_PCT)) if b > a else -120.0

    def frames(self, j0: int, j1: int) -> tuple[np.ndarray, np.ndarray]:
        """(floor, speech level) per frame j0..j1-1 (linear interpolation between grid points)."""
        E = self.E
        t = E.t0 + np.arange(j0, j1, dtype=np.float64) * E.hop
        if not len(t):
            return np.empty(0), np.empty(0)
        ka = int(math.floor(t[0] / GRID)) - self.k0
        kb = int(math.ceil(t[-1] / GRID)) - self.k0 + 1
        ka, kb = max(ka, 0), min(kb, len(self.floor))
        self._fill(ka, kb)
        gx = (self.k0 + np.arange(ka, kb, dtype=np.float64)) * GRID
        return (np.interp(t, gx, self.floor[ka:kb]), np.interp(t, gx, self.speech[ka:kb]))


def _levels(E: Any) -> _Levels:
    lv = _LEVELS.get(E)
    if lv is None or lv.db.shape[0] != len(E.db):
        lv = _Levels(E)
        _LEVELS[E] = lv
    return lv


def _span(E: Any, a: float, b: float) -> tuple[int, int]:
    """Frame range [j0, j1) of [a, b] clipped to the track."""
    return max(0, E.index(a)), min(E.n, E.index(b) + 1)


def frame_masks(E: Any, a: float, b: float) -> tuple[int, np.ndarray, np.ndarray]:
    """(j0, quiet mask, voiced mask) for the frames of [a, b] inside the track."""
    j0, j1 = _span(E, a, b)
    if j1 <= j0:
        return j0, np.zeros(0, bool), np.zeros(0, bool)
    fl, sp = _levels(E).frames(j0, j1)
    db = np.asarray(E.db[j0:j1], dtype=np.float64)
    thr = np.maximum(fl + PAUSE_DB, np.minimum(fl + PAUSE_DB_MAX, sp - SPEECH_DROP))
    return j0, db < thr, db > fl + VOICED_DB


def _runs(mask: np.ndarray, minlen: int = 1) -> list[tuple[int, int]]:
    """[(start, end)) of True runs with length >= minlen (vectorized)."""
    if not len(mask):
        return []
    m = np.concatenate([[0], mask.astype(np.int8), [0]])
    d = np.diff(m)
    starts = np.flatnonzero(d == 1)
    ends = np.flatnonzero(d == -1)
    keep = (ends - starts) >= minlen
    return list(zip(starts[keep].tolist(), ends[keep].tolist()))


def _minframes(E: Any, seconds: float) -> int:
    return max(1, int(math.ceil(seconds / E.hop - 1e-9)))


# ---------------------------------------------------------------------------
# Public queries
# ---------------------------------------------------------------------------


def pauses(E: Any, a: float, b: float, min_pause: float = MIN_PAUSE) -> list[tuple[float, float]]:
    """Quiet runs inside [a, b] lasting at least `min_pause`, using the adaptive threshold
    max(floor + 6, min(floor + 15, speech_p90 - 25)) dB."""
    j0, quiet, _ = frame_masks(E, a, b)
    return [(round(E.time(j0 + s), 6), round(E.time(j0 + e), 6))
            for s, e in _runs(quiet, _minframes(E, min_pause))]


def voiced_runs(E: Any, a: float, b: float, min_run: float = ONSET_RUN) -> list[tuple[float, float]]:
    """Runs above floor + 10 dB inside [a, b] lasting at least `min_run` (clipped to [a, b])."""
    j0, _, voiced = frame_masks(E, a, b)
    return [(max(a, round(E.time(j0 + s), 6)), min(b, round(E.time(j0 + e), 6)))
            for s, e in _runs(voiced, _minframes(E, min_run))]


def next_onset(E: Any, t: float, limit: float) -> float | None:
    """Start of the first voiced run in [t, limit] (t itself when audio is already voiced at t),
    or None."""
    if limit < t:
        return None
    for s, _e in voiced_runs(E, t, limit + ONSET_RUN + E.hop):
        if s <= limit + 1e-9:
            return max(t, s)
        break
    return None


def _merge(intervals: Iterable[tuple[float, float]], gap: float = 0.0) -> list[list[float]]:
    out: list[list[float]] = []
    for s, e in sorted(intervals):
        if out and s - out[-1][1] <= gap:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def _uncovered(E: Any, spans: Iterable[Any], a: float, b: float, bridge: float,
               pad: float) -> list[tuple[float, float, bool, bool]]:
    """Voiced audio in [a, b] outside every span [s - pad, e + pad], as pieces trimmed to their
    voiced frames: (first, last, runs out of a span, runs into a span). "Runs out of / into":
    the piece is cut by a span on that side and is voiced within ABUT_SLACK of the cut."""
    V = voiced_runs(E, a, b)
    if not V:
        return []
    Vs = [r[0] for r in V]
    cov = []
    for sp in spans:
        s, e = (sp["s"], sp["e"]) if isinstance(sp, dict) else (sp[0], sp[1])
        cov.append((float(s) - pad, float(e) + pad))
    cov = _merge(cov)
    cs = [c[0] for c in cov]
    out: list[tuple[float, float, bool, bool]] = []
    for ra, rb in _merge(V, bridge):
        pieces = []          # (start, end, starts at a span's end, ends at a span's start)
        cur = ra
        after_span = False
        k = max(0, bisect_right(cs, ra) - 1)
        while k < len(cov) and cov[k][0] < rb:
            ca, cb = cov[k]
            if cb > cur:
                if ca > cur:
                    pieces.append((cur, min(ca, rb), after_span, ca <= rb))
                cur = max(cur, cb)
                after_span = True
            k += 1
        if cur < rb:
            pieces.append((cur, rb, after_span, False))
        for pa, pb, from_span, to_span in pieces:
            i = bisect_right(Vs, pa) - 1
            if i < 0 or V[i][1] <= pa:
                i += 1
            first = last = None
            while i < len(V) and V[i][0] < pb:
                s, e = max(V[i][0], pa), min(V[i][1], pb)
                if e > s:
                    first = s if first is None else first
                    last = e
                i += 1
            if first is not None:
                out.append((first, last, from_span and first - pa <= ABUT_SLACK,
                            to_span and pb - last <= ABUT_SLACK))
    return out


def uncovered_voiced(E: Any, spans: Iterable[Any], a: float, b: float, *, min_run: float = 0.8,
                     bridge: float = 0.3, pad: float = 0.05, abut_run: float | None = None,
                     abut_db: float = ABUT_DB) -> list[tuple[float, float]]:
    """Voiced audio in [a, b] that no span covers.

    Voiced runs (floor + 10 dB, >= 30 ms) merged across dips of at most `bridge` seconds, minus
    every span [s - pad, e + pad] (`spans` are dicts with "s"/"e" or (s, e) pairs). Each
    remaining piece is trimmed to its voiced frames and kept when it lasts >= `min_run`.

    With `abut_run`, a shorter piece (>= `abut_run`) is kept too when it runs straight out of
    (or into) a span with no dip between them and is as loud as speech (its median level within
    `abut_db` of the local speech level): a short phrase the recogniser skipped right after a
    word, not a word's fading tail or a breath."""
    out: list[tuple[float, float]] = []
    for first, last, out_of, into in _uncovered(E, spans, a, b, bridge, pad):
        if last - first >= min_run - 1e-9:
            out.append((round(first, 3), round(last, 3)))
        elif (abut_run is not None and last - first >= abut_run - 1e-9 and (out_of or into)
              and _speech_loud(E, first, last, abut_db)):
            out.append((round(first, 3), round(last, 3)))
    return out


def sound_into_starts(E: Any, words: Sequence[dict], *, min_run: float = INTO_MIN,
                      within: float = ABUT_DB, bridge: float = 0.3,
                      pad: float = 0.05) -> list[tuple[int, tuple[float, float]]]:
    """Words that someone else's sound runs straight into -> [(word index, (a, b)), ...].

    [a, b] is voiced audio no word covers (a laugh, crosstalk), at least `min_run` s long and
    as loud as speech (median within `within` dB of the local speech level), that ends at the
    word's start with no dip between them. Such a word has no onset of its own in the energy
    track: where it starts inside that sound is the recogniser's guess. `words` are sorted by
    start."""
    if not words or E is None or E.n == 0:
        return []
    starts = [float(w["s"]) for w in words]
    out = []
    for first, last, _out_of, into in _uncovered(E, words, E.t0, E.end, bridge, pad):
        if not into or last - first < min_run - 1e-9 or not _speech_loud(E, first, last, within):
            continue
        k = bisect_left(starts, last - 1e-6)
        if k < len(words) and starts[k] - last <= pad + ABUT_SLACK + 1e-6:
            out.append((k, (round(first, 3), round(last, 3))))
    return out


def _speech_loud(E: Any, a: float, b: float, within: float) -> bool:
    """True when the median level of [a, b] lies within `within` dB of the local speech level
    (90th percentile, +/- 5 s)."""
    j0, j1 = _span(E, a, b - 1e-9)
    if j1 <= j0:
        return False
    _fl, sp = _levels(E).frames(j0, j1)
    db = np.asarray(E.db[j0:j1], dtype=np.float64)
    return float(np.median(db - sp)) >= -within


def _loud_seconds(E: Any, a: float, b: float, within: float) -> float:
    """Seconds of [a, b] whose level lies within `within` dB of the local speech level."""
    j0, j1 = _span(E, a, b - 1e-9)
    if j1 <= j0:
        return 0.0
    _fl, sp = _levels(E).frames(j0, j1)
    db = np.asarray(E.db[j0:j1], dtype=np.float64)
    return float(np.count_nonzero(db - sp >= -within)) * E.hop


def unvoiced_words(words: Sequence[dict], E: Any, pad: float = GATE_PAD) -> list[int]:
    """Indices of words with no frame above floor + 10 dB inside [s - pad, e + pad]. A word
    that lies outside the track (beyond its end, before its start) is never listed: the track
    cannot judge it."""
    if not words or E is None or E.n == 0:
        return []
    _, _, voiced = frame_masks(E, E.t0, E.end)
    cum = np.concatenate([[0], np.cumsum(voiced.astype(np.int64))])
    out = []
    for k, w in enumerate(words):
        s, e = float(w["s"]), float(w["e"])
        if s < E.t0 or e > E.end:
            continue
        j0, j1 = max(0, E.index(s - pad)), min(E.n, E.index(e + pad) + 1)
        if j1 > j0 and cum[j1] - cum[j0] == 0:
            out.append(k)
    return out


def audio_gate(words: Sequence[dict], E: Any) -> tuple[list[dict], list[dict]]:
    """Split words into (kept, dropped): dropped words have no audible frame (floor + 10 dB)
    within 0.1 s of their span. Whisper's "Thank you." in quiet room tone peaks near floor + 1-2
    dB; real words reach floor + 26 dB or more. Run it BEFORE repair_times."""
    bad = set(unvoiced_words(words, E))
    return ([w for k, w in enumerate(words) if k not in bad],
            [w for k, w in enumerate(words) if k in bad])


def _letters(text: str) -> int:
    return max(1, len(_LETTER_RE.findall(text or "")))


class _Runs:
    """Sorted, disjoint runs with bisect lookups."""

    def __init__(self, runs: Sequence[tuple[float, float]]):
        self.r = list(runs)
        self.s = [x[0] for x in self.r]
        self.e = [x[1] for x in self.r]

    def overlapping(self, a: float, b: float) -> list[tuple[float, float]]:
        """Runs with end > a and start < b."""
        out = []
        i = bisect_right(self.e, a)
        while i < len(self.r) and self.s[i] < b:
            out.append(self.r[i])
            i += 1
        return out

    def first_start_from(self, t: float, before: float) -> tuple[float, float] | None:
        """First run with start >= t and start < before."""
        i = bisect_left(self.s, t)
        if i < len(self.r) and self.s[i] < before:
            return self.r[i]
        return None

    def first_end_after(self, t: float, before: float) -> tuple[float, float] | None:
        """First run with end > t and start < before."""
        i = bisect_right(self.e, t)
        if i < len(self.r) and self.s[i] < before:
            return self.r[i]
        return None

    def containing(self, t: float, slack: float) -> tuple[float, float] | None:
        """The run with start + slack < t < end - slack, if any."""
        i = bisect_right(self.s, t) - 1
        if i >= 0 and self.s[i] + slack < t < self.e[i] - slack:
            return self.r[i]
        return None

    def seconds(self, a: float, b: float, start_from: float | None = None) -> float:
        """Seconds of [a, b] the runs cover (only runs starting >= start_from when given)."""
        tot = 0.0
        for x, y in self.overlapping(a, b):
            if start_from is not None and x < start_from:
                continue
            tot += max(0.0, min(y, b) - max(x, a))
        return tot


def sustained_runs(runs: Sequence[tuple[float, float]], sustain: float = ONSET_SUSTAIN,
                   attach: float = ONSET_ATTACH) -> list[tuple[float, float]]:
    """The voiced runs a word can start on: runs of at least `sustain` s, plus a shorter run
    that a kept run follows within `attach` s (the burst of a stop consonant before its vowel).
    An isolated short run (a lip click, a tick, a knock: tens of milliseconds, then quiet) is
    left out."""
    runs = list(runs)
    keep = [False] * len(runs)
    for i in range(len(runs) - 1, -1, -1):
        a, b = runs[i]
        keep[i] = (b - a >= sustain - 1e-9
                   or (i + 1 < len(runs) and keep[i + 1] and runs[i + 1][0] - b <= attach + 1e-9))
    return [r for r, k in zip(runs, keep) if k]


class _Onsets:
    """Voiced-run lookups for the repair: the sustained runs first (sustained_runs), any voiced
    run when the audio in reach holds no sustained one."""

    def __init__(self, V: _Runs):
        self.V = V
        self.S = _Runs(sustained_runs(V.r))

    def first_start_from(self, t: float, before: float) -> tuple[float, float] | None:
        return self.S.first_start_from(t, before) or self.V.first_start_from(t, before)

    def first_end_after(self, t: float, before: float) -> tuple[float, float] | None:
        return self.S.first_end_after(t, before) or self.V.first_end_after(t, before)

    def seconds(self, a: float, b: float, start_from: float | None = None) -> float:
        """Sustained voiced seconds in [a, b]; all voiced seconds when it holds none."""
        return self.S.seconds(a, b, start_from) or self.V.seconds(a, b, start_from)


def repair_times(words: Sequence[dict], E: Any) -> tuple[list[dict], list[dict]]:
    """Move mis-stamped word times onto the audio (see the module docstring).

    For each word in order:
      1. pause pass: a word covering >= 0.15 s of pause keeps its END and its start moves to the
         voiced onset after the last pause inside it (a); else, when it owns voiced audio before
         the pause, its end moves to the pause start (b); else it was stamped into silence and
         moves to the next onset (c). A small overlap moves only a start sitting inside a pause
         ("s~") or an end sitting inside one ("e~").
      2. short pass: a start inside a quiet run >= 0.08 s moves to the onset after it ("s.").
      3. overlong pass: a word longer than 0.30 + 0.08 * letters s that holds a quiet run of
         0.10 s starts at the onset after the last such run ("s+").
      4. click pass: a start sitting on an isolated short voiced run (a lip click before the
         voice) moves to the sustained onset after it inside the word ("s!").
    Then times are made monotonic. Words without evidence are not touched.

    An onset is a sustained voiced run (see sustained_runs): a lip click or a tick before the
    voice (tens of milliseconds, then quiet again) is not where a word starts, and is not voiced
    audio a word owns. Only when the audio in reach holds no sustained run is a short one used.

    Returns (new words, log). Each log entry is {"i", "kind": "energy", "tag", "s0", "e0", "s",
    "e"}: one per word whose times changed (i = its index in `words`)."""
    W = [dict(w) for w in words]
    n = len(W)
    if not n or E is None or E.n == 0:
        return W, []
    s_orig = [(float(w["s"]), float(w["e"])) for w in W]
    lo = min(w["s"] for w in W) - 1.0
    hi = max(w["e"] for w in W) + 1.0
    P = _Runs(pauses(E, lo, hi))
    V = _Onsets(_Runs(voiced_runs(E, lo, hi)))
    G = _Runs(pauses(E, lo, hi, min_pause=SHORT_GAP))
    tags: dict[int, list[str]] = {}
    INF = float("inf")

    def nxt_s(k: int) -> float:
        return W[k + 1]["s"] if k + 1 < n else INF

    def move_start(k: int, ns: float, tag: str) -> bool:
        w = W[k]
        if ns <= w["s"] + 1e-3 or ns >= w["e"] - 0.03 or ns >= nxt_s(k) - 0.03:
            return False
        w["s"] = round(ns, 3)
        tags.setdefault(k, []).append(tag)
        return True

    for k in range(n):
        w = W[k]
        s, e = w["s"], w["e"]
        # 1. pause pass
        inside = [(max(pa, s), min(pb, e)) for pa, pb in P.overlapping(s, e)
                  if min(pb, e) - max(pa, s) > 0]
        cover = sum(y - x for x, y in inside)
        if cover >= MIN_OVERLAP:
            # a pause starting in the word's last 50 ms is its trailing silence, not inside it
            core = [(x, y) for x, y in inside if x < e - MIN_VOICED] or inside
            last_pb = max(y for _x, y in core)
            first_pa = min(x for x, _y in core)
            prev_e = W[k - 1]["e"] if k else -1e9
            own_from = (min(s, prev_e + 0.05) if OWN_AFTER_PREV else s) - 0.011
            right = V.first_end_after(last_pb + EDGE_SLACK, e)
            vright = V.seconds(last_pb, e)
            vleft = V.seconds(s, first_pa, start_from=own_from)
            if vleft < MIN_VOICED:
                # one voiced run carries the previous word straight into this one (connected
                # speech has no gap between words): a good stretch of it inside the span,
                # before the pause, is this word, and the model ran its end on into the pause
                cont = _loud_seconds(E, s, first_pa, OWN_LOUD_DB)
                if cont >= OWN_CONTINUED:
                    vleft = cont
            if right is not None and vright >= MIN_VOICED:
                if move_start(k, max(right[0], last_pb), "s"):
                    continue
            if vleft >= MIN_VOICED:
                w["e"] = round(first_pa, 3)
                tags.setdefault(k, []).append("e")
                continue
            # no voiced audio of its own: it was stamped into the silence and belongs to the
            # next voiced onset after the pause (before the next word ends)
            nxt_e = W[k + 1]["e"] if k + 1 < n else INF
            on = V.first_start_from(last_pb - 0.011, nxt_e - MIN_VOICED)
            if on is not None:
                ns = on[0]
                ne = max(ns + 0.08, min(nxt_s(k), nxt_e - MIN_VOICED)) if k + 1 < n else on[1]
                w["s"], w["e"] = round(ns, 3), round(ne, 3)
                tags.setdefault(k, []).append("S")
                if k + 1 < n and W[k + 1]["s"] < w["e"]:
                    W[k + 1]["s"] = w["e"]
                    tags.setdefault(k + 1, []).append("push")
            continue
        p = P.containing(s, EDGE_SLACK)
        if p is not None:
            on = V.first_start_from(s - 1e-6, e)
            if on is not None:
                move_start(k, on[0], "s~")
        p = P.containing(w["e"], EDGE_SLACK)
        if p is not None and p[0] > w["s"]:
            w["e"] = round(p[0], 3)
            tags.setdefault(k, []).append("e~")
        # 2. short pass
        s, e = w["s"], w["e"]
        g = G.containing(s, 0.01)
        if g is not None:
            on = V.first_start_from(g[1] - 0.011, e)
            if on is not None:
                move_start(k, on[0], "s.")
        # 3. overlong pass
        s, e = w["s"], w["e"]
        if e - s > OVERLONG_BASE + OVERLONG_PER_LETTER * _letters(w.get("w", "")):
            gaps = [(ga, gb) for ga, gb in G.overlapping(s, e)
                    if ga > s + 0.02 and gb < e - MIN_VOICED and gb - ga >= OVERLONG_GAP]
            if gaps:
                on = V.first_start_from(gaps[-1][1] - 0.011, e)
                if on is not None and V.seconds(on[0], e) >= MIN_VOICED:
                    move_start(k, on[0], "s+")
        # 4. click pass
        s, e = w["s"], w["e"]
        blip = V.V.first_end_after(s, s + 0.011)
        if blip is not None and blip[0] >= s - 0.011 and not V.S.overlapping(s - 0.011, blip[1]):
            on = V.S.first_start_from(blip[1], e - MIN_VOICED)
            if on is not None:
                move_start(k, on[0], "s!")
    for k in range(1, n):
        if W[k]["s"] < W[k - 1]["s"]:
            W[k]["s"] = W[k - 1]["s"]
        if W[k - 1]["e"] > W[k]["s"]:
            W[k - 1]["e"] = W[k]["s"]
        if W[k - 1]["e"] < W[k - 1]["s"]:
            W[k - 1]["e"] = W[k - 1]["s"]
    log = []
    for k, w in enumerate(W):
        w["s"], w["e"] = round(w["s"], 3), round(w["e"], 3)
        s0, e0 = s_orig[k]
        if abs(w["s"] - s0) > 5e-4 or abs(w["e"] - e0) > 5e-4:
            log.append({"i": k, "kind": "energy", "tag": ",".join(tags.get(k) or ["mono"]),
                        "s0": round(s0, 3), "e0": round(e0, 3), "s": w["s"], "e": w["e"]})
    return W, log


def snap_back(words: Sequence[dict], E: Any, lo: float = SNAP_MIN,
              hi: float = SNAP_MAX, sustained: bool = False) -> tuple[list[dict], list[dict]]:
    """Backward onset snap (for backends whose starts run late, such as whisper.cpp DTW): a start
    sitting lo..hi s after a voiced onset that follows a pause of at least 0.2 s, with no earlier
    word reaching past that onset, moves back to the onset. With `sustained` only a sustained
    onset counts (a lip click after the pause is not where the word starts). Returns (words, log)."""
    W = [dict(w) for w in words]
    if not W or E is None or E.n == 0:
        return W, []
    a = min(w["s"] for w in W) - 1.0
    b = max(w["e"] for w in W) + 1.0
    V = _Onsets(_Runs(voiced_runs(E, a, b)))
    onsets = []
    for _pa, pb in pauses(E, a, b):
        on = (V.S.first_start_from(pb - 0.011, pb + 0.05) if sustained
              else V.first_start_from(pb - 0.011, pb + 0.05))
        if on is not None:
            onsets.append(on[0])
    log = []
    for k, w in enumerate(W):
        s = w["s"]
        i = bisect_right(onsets, s - lo + 1e-9) - 1
        if i < 0:
            continue
        on = onsets[i]
        if not (lo - 1e-9 <= s - on <= hi + 1e-9):
            continue
        if k and W[k - 1]["e"] > on + 1e-6:
            continue
        log.append({"i": k, "kind": "energy", "tag": "snap", "s0": s, "e0": w["e"],
                    "s": round(on, 3), "e": w["e"]})
        w["s"] = round(on, 3)
    return W, log


def burst_dips(E: Any, s: float, e: float, *, dip_db: float = DIP_DB, min_dip: float = DIP_MIN,
               burst_db: float = BURST_DB, min_burst: float = BURST_MIN) -> list[tuple[float, float]]:
    """Dips that split the sound in [s, e] into separate bursts -> [(dip start, dip end), ...].

    The level of the span is its 90th percentile. A dip is a run of at least `min_dip` s more
    than `dip_db` dB under it, with at least `min_burst` s of sound within `burst_db` dB of it on
    each side. One spoken word holds short stop closures (a few 10 ms); two attempts at a word,
    or a laugh burst followed by a word, are separated by a dip like this."""
    j0, j1 = _span(E, s, e - 1e-9)
    if j1 - j0 < _minframes(E, min_dip + 2 * min_burst):
        return []
    db = np.asarray(E.db[j0:j1], dtype=np.float64)
    ref = float(np.percentile(db, 90))
    loud = np.concatenate([[0], np.cumsum(db > ref - burst_db)])
    need = _minframes(E, min_burst)
    out = []
    for a, b in _runs(db < ref - dip_db, _minframes(E, min_dip)):
        if loud[a] >= need and loud[-1] - loud[b] >= need:
            out.append((round(E.time(j0 + a), 3), round(E.time(j0 + b), 3)))
    return out


def voiced_end(E: Any, t: float, limit: float, bridge: float = 0.2) -> float:
    """End of the voiced stretch that starts at or after `t`: the first voiced run in
    [t, limit] extended across quiet dips shorter than `bridge` s, capped at `limit`. Returns `t`
    when no voiced audio lies in [t, limit]."""
    if limit <= t:
        return t
    runs = voiced_runs(E, t, limit)
    if not runs:
        return t
    end = runs[0][1]
    for a, b in runs[1:]:
        if a - end >= bridge - 1e-9:
            break
        end = b
    return min(limit, end)


# ---------------------------------------------------------------------------
# Band shapes: is this still the same sound?
# ---------------------------------------------------------------------------


class BandShapes:
    """The spectrum's shape per 10 ms frame: dB in six bands (100 Hz to 8 kHz) with the frame's
    own mean removed, so loudness drops out and only the colour of the sound is left. Frame j is
    a 25 ms window centred on energy frame j (same clock, same hop as the EnergyTrack).

    The energy track cannot tell a word from the laugh it starts under: both are loud. Their
    colours differ, and that is what spectral_onset looks for."""

    def __init__(self, shapes: Any, hop: float = 0.01, t0: float = 0.0):
        S = np.asarray(shapes, dtype=np.float32)
        self.S = S.reshape(-1, S.shape[-1]) if S.size else np.zeros((0, len(SHAPE_EDGES) - 1),
                                                                    np.float32)
        self.hop = float(hop)
        self.t0 = float(t0)

    @property
    def n(self) -> int:
        return int(self.S.shape[0])

    def index(self, t: float) -> int:
        return int(math.floor((t - self.t0) / self.hop + 1e-9))

    @staticmethod
    def _plan(sr: int, hop: float) -> tuple:
        hop_n = max(1, int(round(sr * hop)))
        win_n = max(hop_n, int(round(sr * SHAPE_WIN)))
        nfft = 1 << (win_n - 1).bit_length()
        freqs = np.fft.rfftfreq(nfft, 1.0 / sr)
        top = min(SHAPE_EDGES[-1], sr / 2.0)
        masks = [(freqs >= lo) & (freqs <= hi if hi >= top else freqs < hi)
                 for lo, hi in zip(SHAPE_EDGES[:-1], SHAPE_EDGES[1:])]
        return hop_n, win_n, nfft, masks, np.hanning(win_n).astype(np.float32)

    @staticmethod
    def _block(x: np.ndarray, count: int, plan: tuple) -> np.ndarray:
        """Shapes of `count` frames from samples `x` (frame i = x[i * hop : i * hop + win])."""
        hop_n, win_n, nfft, masks, window = plan
        frames = x[np.arange(count)[:, None] * hop_n + np.arange(win_n)[None, :]] * window
        power = np.abs(np.fft.rfft(frames, nfft, axis=1)) ** 2
        out = np.empty((count, len(masks)), dtype=np.float32)
        for b, m in enumerate(masks):
            band = power[:, m].sum(axis=1) if m.any() else np.zeros(count)
            out[:, b] = 10.0 * np.log10(band / win_n + 1e-12)
        return out - out.mean(axis=1, keepdims=True)

    @classmethod
    def _build(cls, total: int, read: Any, sr: int, hop: float, t0: float) -> "BandShapes":
        """`read(i0, i1)` returns samples [i0, i1) as float32 (zeros outside the audio)."""
        plan = cls._plan(sr, hop)
        hop_n, win_n = plan[0], plan[1]
        lead = (win_n - hop_n) // 2
        n = total // hop_n
        out = np.empty((max(n, 0), len(plan[3])), dtype=np.float32)
        step = 20000                                   # frames per block: about 40 MB of work
        for c0 in range(0, n, step):
            c1 = min(n, c0 + step)
            out[c0:c1] = cls._block(read(c0 * hop_n - lead, (c1 - 1) * hop_n - lead + win_n),
                                    c1 - c0, plan)
        return cls(out, hop, t0)

    @classmethod
    def from_samples(cls, x: Any, sr: int = 16000, hop: float = 0.01,
                     t0: float = 0.0) -> "BandShapes":
        """From mono float samples (sample 0 = src time `t0`)."""
        x = np.asarray(x, dtype=np.float32).reshape(-1)

        def read(i0: int, i1: int) -> np.ndarray:
            out = np.zeros(i1 - i0, dtype=np.float32)
            a, b = max(i0, 0), min(i1, len(x))
            if b > a:
                out[a - i0:b - i0] = x[a:b]
            return out
        return cls._build(len(x), read, sr, hop, t0)

    @classmethod
    def from_wav(cls, path: Any, t0: float = 0.0, hop: float = 0.01) -> "BandShapes":
        """From a 16-bit PCM wav (the transcript's 16 kHz mono wav), read block by block.
        Sample 0 = src time `t0`."""
        import wave
        with wave.open(str(path), "rb") as w:
            sr, ch, total = w.getframerate(), w.getnchannels(), w.getnframes()
            if w.getsampwidth() != 2:
                raise ValueError("band shapes need a 16-bit PCM wav")

            def read(i0: int, i1: int) -> np.ndarray:
                out = np.zeros(i1 - i0, dtype=np.float32)
                a, b = max(i0, 0), min(i1, total)
                if b > a:
                    w.setpos(a)
                    data = np.frombuffer(w.readframes(b - a), dtype="<i2").astype(np.float32)
                    if ch > 1:
                        data = data[:len(data) // ch * ch].reshape(-1, ch).mean(axis=1)
                    out[a - i0:a - i0 + len(data)] = data / 32768.0
                return out
            return cls._build(total, read, sr, hop, t0)


def spectral_onset(B: Any, E: Any, pre: tuple[float, float], s: float, e: float) -> float | None:
    """Where the sound in [s, e] stops being the sound heard in `pre` = (a, b) just before it.

    The sound before is described by the mean band shape of the louder half of its last
    SHAPE_PRE seconds, and by how far its own frames stray from that mean (90th percentile).
    Returns the start of the first voiced stretch of at least SHAPE_RUN s in [s, e] whose
    frames are all farther from it than both SHAPE_MIN_DB and that spread + SHAPE_MARGIN, or
    None when [s, e] never leaves the sound (or the sound before is too short to describe)."""
    if B is None or E is None or B.n == 0 or e <= s:
        return None
    a = max(pre[0], pre[1] - SHAPE_PRE)
    i0, i1 = max(0, B.index(a)), min(B.n, B.index(pre[1] - 1e-9) + 1)
    j0, j1 = max(0, B.index(s)), min(B.n, B.index(e - 1e-9) + 1)
    if i1 - i0 < _minframes(E, 0.1) or j1 <= j0:
        return None
    at = np.array([E.index(B.t0 + (i + 0.5) * B.hop) for i in range(i0, i1)])
    lvl = np.where((at >= 0) & (at < E.n), np.asarray(E.db, dtype=np.float64)[np.clip(at, 0, E.n - 1)],
                   -120.0)
    loud = B.S[i0:i1][lvl >= np.median(lvl)].astype(np.float64)
    tpl = loud.mean(axis=0)
    spread = float(np.percentile(np.sqrt(((loud - tpl) ** 2).mean(axis=1)), 90))
    thr = max(SHAPE_MIN_DB, spread + SHAPE_MARGIN)
    dist = np.sqrt(((B.S[j0:j1].astype(np.float64) - tpl) ** 2).mean(axis=1))
    _f0, _quiet, voiced = frame_masks(E, B.t0 + j0 * B.hop, B.t0 + (j1 - 1) * B.hop)
    ok = dist >= thr
    m = min(len(ok), len(voiced))
    ok[:m] &= voiced[:m]
    ok[m:] = False
    runs = _runs(ok, _minframes(E, SHAPE_RUN))
    return round(B.t0 + (j0 + runs[0][0]) * B.hop, 3) if runs else None


def repair_overlapped(words: Sequence[dict], E: Any, B: Any) -> tuple[list[dict], list[dict]]:
    """Starts stamped inside someone else's sound: for each word that unworded sound runs
    straight into (sound_into_starts), find where its span stops being that sound
    (spectral_onset) and move the start there (tag "sp") when that is later than the stamped
    start by more than SHAPE_AT_START and leaves the word at least SHAPE_KEEP s. The end is
    kept. Returns (new words, log) like repair_times."""
    W = [dict(w) for w in words]
    log: list[dict] = []
    if B is None or E is None or not W:
        return W, log
    for k, pre in sound_into_starts(E, W):
        w = W[k]
        s, e = float(w["s"]), float(w["e"])
        on = spectral_onset(B, E, pre, s, e)
        if on is None or on - s <= SHAPE_AT_START + 1e-9 or e - on < SHAPE_KEEP - 1e-9:
            continue
        w["s"] = round(on, 3)
        log.append({"i": k, "kind": "energy", "tag": "sp", "s0": round(s, 3), "e0": round(e, 3),
                    "s": w["s"], "e": round(e, 3)})
    return W, log


def uncertain_starts(words: Sequence[dict], E: Any, B: Any = None) -> list[int]:
    """Indices of words whose start is a guess: unworded sound as loud as speech runs straight
    into them (sound_into_starts) and, when band shapes are given, the sound does not change
    colour within SHAPE_AT_START of the stamped start (spectral_onset)."""
    out = []
    for k, pre in sound_into_starts(E, words):
        s, e = float(words[k]["s"]), float(words[k]["e"])
        if B is not None:
            on = spectral_onset(B, E, pre, s, e)
            if on is not None and on - s <= SHAPE_AT_START + 1e-9:
                continue
        out.append(k)
    return out
