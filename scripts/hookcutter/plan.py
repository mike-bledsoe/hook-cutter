"""Edit planning (contract 8.1): where every cut lands, in source time, on the frame grid.

For each piece the planner snaps the in and out edges into the gap between
words using the energy track (audio.EnergyTrack), splits pieces at `omit`
words, tightens long quiet gaps and fillers inside each piece, and rounds
every cut to the source frame grid with exact integer frame counts. Picture
and audio are cut at the same grid times, so they cannot drift apart. The
result (`plans/<clip>.plan.json`) is what render, captions, cover and qa read.
"""
from __future__ import annotations

import bisect
import math
import re
from fractions import Fraction
from pathlib import Path

from . import env
from . import spec as specmod
from . import textnorm as tn

SAMPLE_RATE = 48000
# in-edge (8.1.1)
IN_LEAD = 0.25           # window starts at most this long before the first word
IN_LATE = 0.03           # ... and ends this long after its whisper start
IN_TARGET = 0.06         # preferred cut: this long before the word
ONSET_PAD = 0.02         # cut at least this long before a voiced onset
GUARD_SLACK = 0.05       # the onset guard may reach this far below the window
COST_DB_PER_S = 40.0     # distance penalty (dB per second from the target)
WIN_DB = 0.02            # energy window around a candidate cut
TIGHT_DB = 12.0          # tight = db at the cut > floor + TIGHT_DB
LATE_REACH = 0.60        # a tight in-edge may look back this far for a late ASR word start
LATE_AFTER_PREV = 0.15   # ... but never earlier than this after the previous word's start
LATE_QUIET_DB = 6.0      # the audio just before the rescued onset must sit within this of floor
KEPT_WIN = 0.01          # tighten cuts: loudness of the 10 ms kept next to the cut
# in-edge after a dropped, touching previous word: cut in the dip between the two words
DIP_TOUCH = 0.10         # the previous word ends at most this long before the word starts
DIP_FRAC = 0.6           # search no earlier than this share into the previous word ...
DIP_REACH = 0.12         # ... and no earlier than this long before the word's start
DIP_DB = 6.0             # the dip must sit this far below the energy at the default target
VOICED_RUN = 0.03        # ... and the cut it replaces must keep at least this much voiced audio
END_GUARD = 0.01         # never pick a cut this close to (or past) the end of the audio
# out-edge
OUT_PRE = 0.02           # window starts this long before the true word end
TAIL = 0.10              # base tail after the last word
TAIL_LAST = 0.40         # base tail on the clip's last piece
OUT_REACH = 0.25         # window reaches this far past the target
NEXT_CLEAR = 0.03        # stay this far clear of the next (or omitted) word
OFFSET_PAD = 0.02        # cut at least this long after the last voiced frame
# clip-final out-edge when the last word's final consonant runs into the next word (no pause)
CONS_BACK = 0.02         # look for the dip between them from this long before the snapped cut ...
CONS_REACH = 0.06        # ... to this long after the next word's stamped start
CONS_DIP = 6.0           # the dip sits this far below the loudest audio on both sides of it
# tighten (8.1.2)
PROTECT_PRE = 0.10
PROTECT_POST = 0.05
PROTECT_EDGE = 0.25      # first/last seconds of a piece are never tightened
FILLER_PAD = 0.02
MIN_KEEP_FRAMES = 2
LAUGH_DB = 10.0          # a voiced run inside a pause (laugh, chuckle, breath laugh) ...
LAUGH_MIN = 0.12         # ... lasting at least this long ...
LAUGH_BRIDGE = 0.02      # ... (dips up to this long do not end it) ...
LAUGH_PAD = 0.10         # ... is kept whole, with this much room either side
# segments
FADE = 0.008
FADE_LAST = 0.06
# lints (8.1.3)
SENTENCE_PAUSE = 0.3
PAYOFF_BY = {"vertical": 8.0, "landscape": 20.0}
HOOK_ACCENT_LEAD = 0.06  # caption.lead_ms of the shipped presets: a word whose midpoint is earlier
                         # than the card end less this is not in the burned captions
CUT_RATE_MAX = 0.5
UPSCALE_MAX = 3.0
OVERLAP_MAX = 15.0
OUT_SIZE = {"vertical": (1080, 1920), "landscape": (1920, 1080)}
QUALIFIERS = frozenset({"just", "except", "unless", "until", "even", "if", "only"})
L10_DB = 18.0            # a piece / clip edge WARNs only above floor + this (else INFO)
L10_JOIN_DB = 24.0       # omit and tighten joins WARN only above floor + this (else INFO)
DEMONSTRATIVES = frozenset({"this", "that", "these", "those"})
PRONOUN_VERBS = frozenset({"is", "was", "are", "were", "'s", "s", "means", "does", "did", "will",
                           "would", "can", "could"})
RESTART_REACH = 6        # L06: an omitted qualifier said again this many words later = restart
CLOSE_GAP = 0.15         # L05: a sentence-final last word needs this much air after it ...
CLOSE_QUIET = 0.08       # ... or a quiet run (floor + 6 dB) this long within ...
CLOSE_WIN = 0.20         # ... this window after its end
OVERLAP_SRC = 0.1        # L17: pieces sharing more source than this overlap
DEAD_AIR = 0.35          # L18: silence before the first word at a join (s)
DEAD_AIR_START = 0.25    # L18: ... at the clip start
DEAD_AIR_LOOK = 1.0      # L18: onset search window
HOLE_WARN = 1.5          # L19: uncovered voiced audio this long WARNs ...
HOLE_FAIL = 3.0          # ... and this long FAILs (unless a transcript flag covers it)
HOOK_RATIO_MIN = 1.3     # L20: the hook card should set at least this much larger than captions
HOOK_AUTO = (0.5, 20.0, 1.5, 3.0)    # dur = clamp(a + chars / b, lo, hi)
EPS = 1e-6


def _code_version() -> str:
    try:
        from . import __version__
        return __version__
    except ImportError:  # pragma: no cover
        return "0.1.0"


# ---- frame grid ----------------------------------------------------------------

class Grid:
    """Source frame grid: frame k sits at offset + k * den / num (exact integers in, floats out)."""

    def __init__(self, num: int, den: int, offset: float = 0.0):
        if num <= 0 or den <= 0:
            raise env.HCError(f"invalid frame rate {num}/{den}", 3)
        self.num, self.den, self.offset = int(num), int(den), float(offset)
        self.fd = Fraction(self.den, self.num)

    def t(self, k: int) -> float:
        return self.offset + k * self.den / self.num

    def _pos(self, t: float) -> float:
        return (t - self.offset) * self.num / self.den

    def ceil(self, t: float) -> int:
        return int(math.ceil(self._pos(t) - EPS))

    def floor(self, t: float) -> int:
        return int(math.floor(self._pos(t) + EPS))

    def nearest(self, t: float) -> int:
        return int(math.floor(self._pos(t) + 0.5))

    def dur(self, frames: int) -> float:
        return float(frames * self.fd)

    def samples(self, frames: int) -> int:
        return int(round(Fraction(frames * SAMPLE_RATE * self.den, self.num)))


# ---- context -----------------------------------------------------------------

class _Ctx:
    def __init__(self, clip: dict, words: list, grid: Grid, energy, kmax: int):
        self.clip, self.W, self.G, self.E, self.kmax = clip, words, grid, energy, kmax
        self.N = len(words)
        # Last grid frame a snapped cut may use: the video end (kmax) and, for word cuts, the
        # end of the decoded audio. Past it db_at reads SILENCE_DB, which looks like perfect
        # silence and would pull an out-edge beyond the end of the file.
        self.kend = kmax
        if energy is not None and getattr(energy, "n", 0):
            self.kend = max(0, min(kmax, grid.floor(energy.end - END_GUARD)))
        self.omit = set(clip.get("omit") or [])
        self.fix = clip.get("fix") or {}
        self.verb = set()
        for a, b in clip.get("verbatim") or []:
            self.verb.update(range(a, b + 1))
        self.opts = (clip.get("_eff") or {}).get("tighten") or dict(specmod.DEFAULTS["tighten"])
        self.inserts = [m for m in clip.get("insert") or [] if isinstance(m, dict)]
        self._te: dict = {}
        self._starts = [w["s"] for w in words]
        self._filler: dict = {}

    def te(self, i: int) -> float:
        if i not in self._te:
            self._te[i] = self.E.true_end(self.W[i])
        return self._te[i]

    def filler(self, i: int) -> bool:
        if i not in self._filler:
            self._filler[i] = _is_filler_word(self.W[i]["w"], self.fix.get(str(i)))
        return self._filler[i]

    def clampk(self, k: int) -> int:
        return min(max(k, 0), self.kmax)


def _is_filler_word(raw, fixv=None) -> bool:
    """A filler ("um", "uh") is cut by tighten, never captioned and not in intent. A non-empty
    `fix` on the word is the editor saying what was really said (an "a" the transcript wrote as
    "uh"): then the fix text decides, so a corrected word is kept, captioned and in intent. An
    empty fix only hides the caption and changes nothing here."""
    return bool(tn.is_filler(fixv if fixv else raw))


def _edge(ctx: _Ctx, k: int, forced: bool, guard, word: int, fl: float) -> dict:
    t = ctx.G.t(k)
    db = ctx.E.db_at(t, WIN_DB)
    return {"src": round(t, 6), "k": int(k), "tight": bool(forced or db > fl + TIGHT_DB),
            "db": round(db, 1), "floor_db": round(fl, 1), "guard": guard, "word": word}


def _pick(ctx: _Ctx, lo: float, hi: float, target: float):
    """Grid time in [lo, hi] minimizing db + 40 * |t - target|; (k, forced)."""
    G = ctx.G
    k0, k1 = max(G.ceil(lo), 0), min(G.floor(hi), ctx.kend)
    if hi >= lo and k1 >= k0:
        best = None
        for k in range(k0, k1 + 1):
            t = G.t(k)
            cost = ctx.E.db_at(t, WIN_DB) + COST_DB_PER_S * abs(t - target)
            key = (round(cost, 9), round(abs(t - target), 9), k)
            if best is None or key < best:
                best = key
        return best[2], False
    tt = min(max(target, lo), hi) if hi >= lo else (lo + hi) / 2
    return min(ctx.clampk(G.nearest(tt)), ctx.kend), True


def _voiced_s(E, a: float, b: float, fl: float) -> float:
    """Seconds of voiced audio (above floor + 10 dB) in [a, b)."""
    if b <= a:
        return 0.0
    n, t, step = 0, a, E.hop
    while t < b:
        if E.db_at(t, step / 2) > fl + 10.0:
            n += 1
        t += step
    return n * step


def _dip_frame(ctx: _Ctx, i: int, k: int, lo_min: float = -math.inf):
    """Grid frame in the energy dip between word i and a touching previous word, or None.

    When the previous word is dropped (not kept right before this cut) and its stamped end
    touches word i, the default target (IN_TARGET before the start) sits in the previous word's
    tail, and the distance-weighted pick often prefers the deep silence BEFORE the previous
    word, so the cut keeps all of it ("and", a "but" burst). The quietest grid frame between
    DIP_FRAC into the previous word (at most DIP_REACH before the start) and IN_LATE after the
    start is the boundary; it is taken when it is DIP_DB below the energy at the default
    target (the previous word's tail) AND moving the cut there removes sound: the current pick
    sits before the dip with voiced audio (the dropped word) between them, or after it inside
    louder audio. A pick already in the silence after the previous word stays (a late ASR
    start makes the window look like a dip when it is the next word's onset)."""
    W, G, E = ctx.W, ctx.G, ctx.E
    s, p = W[i]["s"], W[i - 1]
    lo = max(p["s"] + DIP_FRAC * (s - p["s"]), s - DIP_REACH, lo_min)
    k0, k1 = max(G.ceil(lo), 0), min(G.floor(s + IN_LATE), ctx.kend)
    if k1 < k0:
        return None
    target = s - IN_TARGET
    best = min(range(k0, k1 + 1), key=lambda kk: (round(E.db_at(G.t(kk), WIN_DB), 6),
                                                   abs(G.t(kk) - target), kk))
    if best == k:
        return None
    db = E.db_at(G.t(best), WIN_DB)
    if db > E.db_at(target, WIN_DB) - DIP_DB:
        return None
    if best > k:
        fl = E.floor(s)
        return best if _voiced_s(E, G.t(k), G.t(best), fl) >= VOICED_RUN else None
    return best if db <= E.db_at(G.t(k), WIN_DB) - DIP_DB else None


def snap_in(ctx: _Ctx, i: int, prev_omitted: bool = False, prev_kept: bool = False) -> dict:
    """In-edge before word i (8.1.1). `prev_kept`: word i-1 is heard right before this cut (it
    ends the previous run or a source-adjacent preceding piece), so no dip search is needed."""
    W, G, E = ctx.W, ctx.G, ctx.E
    s = W[i]["s"]
    p = W[i - 1] if i > 0 else None
    lo = s - IN_LEAD
    if p is not None and p["e"] < s:
        lo = max(lo, (p["e"] + s) / 2)
    guard_lo = lo - GUARD_SLACK
    if prev_omitted and p is not None:
        # the omitted word must not leak back in
        lo = max(lo, min(ctx.te(i - 1), s) - NEXT_CLEAR)
        guard_lo = lo
    hi = s + IN_LATE
    fl = E.floor(s)
    k, forced = _pick(ctx, lo, hi, s - IN_TARGET)
    guard = None
    on = E.onset(s, guard_lo, fl=fl)
    if on is not None and G.t(k) > on - ONSET_PAD + 1e-9:
        kg = max(G.floor(on - ONSET_PAD), G.ceil(guard_lo), 0)
        if kg < k:
            k, guard = kg, "onset"
    if p is not None and not prev_kept and p["e"] >= s - DIP_TOUCH:
        kd = _dip_frame(ctx, i, k, lo if prev_omitted else -math.inf)
        if kd is not None:
            k, forced, guard = kd, False, "dip"
    if (not prev_omitted and p is not None
            and E.db_at(G.t(k), WIN_DB) > fl + TIGHT_DB):
        # Late ASR start: Whisper often stamps a segment-initial word (and the previous
        # segment's last word end) a few hundred ms late, so the whole window sits inside the
        # word. If the voiced run holding the word began after a real silence that lies inside
        # the previous word's stamped span, that silence is the true boundary: cut there.
        rlo = max(p["s"] + LATE_AFTER_PREV, s - LATE_REACH)
        if rlo < guard_lo:
            on2 = E.onset(s, rlo, fl=fl)
            if (on2 is not None and on2 < guard_lo
                    and E.db_at(on2 - 0.03, WIN_DB) <= fl + LATE_QUIET_DB
                    and _voiced_s(E, p["s"], on2, fl) >= _voiced_s(E, on2, min(p["e"], s), fl)):
                # ...and only when most of the previous word's stamped span sounds BEFORE that
                # silence. Otherwise the silence is the pause before the previous word, stamped
                # early ("common [pause] issue I" read as "issue [pause] I"), and cutting there
                # would put the previous word back into the clip.
                kg = max(G.floor(on2 - ONSET_PAD), G.ceil(on2 - 0.15), 0)
                if kg < k and E.db_at(G.t(kg), WIN_DB) <= fl + TIGHT_DB:
                    k, guard = kg, "late_asr"
                    edge = _edge(ctx, k, forced, guard, i, fl)
                    edge["onset"] = round(on2, 4)   # the word's audible start (captions use it)
                    return edge
    return _edge(ctx, k, forced, guard, i, fl)


def _consonant_frame(ctx: _Ctx, i: int, k: int, fl: float):
    """Clip-final out-edge when word i's last consonant runs straight into word i+1, or None.

    The closing fade (FADE_LAST) ends at the cut. With no pause before the next word the snapped
    cut sits NEXT_CLEAR before its stamped start, inside the consonant, so the fade covers the
    whole consonant ("sense" heard as "sen"). When the snapped cut is inside sound and the audio
    dips (CONS_DIP below both sides) between there and the next word, the cut moves to the last
    grid frame that centres the fade on that dip: the consonant plays at full level up to half a
    fade before the dip and the next word's onset is under the second half of the fade."""
    W, G, E = ctx.W, ctx.G, ctx.E
    q = W[i + 1]
    t_k = G.t(k)
    if E.db_at(t_k, WIN_DB) <= fl + TIGHT_DB:
        return None                                   # the cut is already in quiet audio
    j0, j1 = E.index(t_k - CONS_BACK), E.index(q["s"] + CONS_REACH)
    if j1 - j0 < 2:
        return None
    vals = [E._db_j(j) for j in range(j0, j1 + 1)]
    jm = min(range(1, len(vals) - 1), key=lambda j: (vals[j], j))
    if vals[jm] > max(vals[:jm]) - CONS_DIP or vals[jm] > max(vals[jm + 1:]) - CONS_DIP:
        return None                                   # no dip: one continuous sound
    t_dip = E.time(j0 + jm) + E.hop / 2
    kc = min(G.floor(t_dip + FADE_LAST / 2), ctx.kend)
    return kc if kc > k else None


def snap_out(ctx: _Ctx, i: int, tail: float, hold: float = 0.0, next_omitted: bool = False,
             final: bool = False):
    """Out-edge after word i (8.1.1). Returns (edge, extra word indices pulled in by hold_after).
    `final`: this is the clip's last out-edge and the piece sets no out_offset, so a final
    consonant that runs into the next word gets the consonant guard (see _consonant_frame)."""
    W, G, E = ctx.W, ctx.G, ctx.E
    q = W[i + 1] if i + 1 < ctx.N else None
    te = ctx.te(i)
    total = tail + hold
    lo = te - OUT_PRE
    hi = te + total + OUT_REACH
    if q is not None and (hold <= 0 or next_omitted):
        hi = min(hi, q["s"] - NEXT_CLEAR)
    fl = E.floor(te)
    k, forced = _pick(ctx, lo, hi, te + total)

    def extras_for(kk):
        out, j = [], i + 1
        if hold > 0 and not next_omitted:
            while j < ctx.N and W[j]["s"] < G.t(kk) - OUT_PRE:
                out.append(j)
                j += 1
        return out

    extras = extras_for(k)
    last = extras[-1] if extras else i
    lim = hi if hi >= lo else lo
    t_from = ctx.te(last) - 0.03
    vend = E.voiced_end(t_from, max(lim, ctx.te(last)) + 0.05, fl)
    guard = None
    if vend > t_from and G.t(k) < vend + OFFSET_PAD - 1e-9:
        kg = min(G.ceil(vend + OFFSET_PAD), G.floor(lim), ctx.kend)
        if kg > k:
            k, guard = kg, "offset"
            extras = extras_for(k)
    auto = 0
    if final and q is not None and hold <= 0 and not next_omitted:
        kc = _consonant_frame(ctx, i, k, fl)
        if kc is not None:
            auto, k, guard = kc - k, kc, "consonant"
    edge = _edge(ctx, k, forced, guard, extras[-1] if extras else i, fl)
    if auto:
        edge["auto"] = auto              # frames the consonant guard moved the cut (later)
    if hold > 0:
        # why this grid frame: hold_after asks for a time, the edge lands on a frame
        why = "forced" if forced and guard is None else ("after_sound" if guard == "offset"
                                                         else "quietest")
        edge["hold"] = {"asked": round(float(hold), 3), "target_src": round(te + total, 4),
                        "frame": int(k), "frame_src": round(G.t(k), 6),
                        "got": round(G.t(k) - te, 3), "why": why}
    return edge, extras


# ---- tighten -----------------------------------------------------------------

def _merge(spans):
    """Union of (a, b[, kind]) spans; kind 'filler' wins over 'gap'."""
    out = []
    for sp in sorted(spans, key=lambda x: (x[0], x[1])):
        a, b = sp[0], sp[1]
        kind = sp[2] if len(sp) > 2 else None
        if out and a <= out[-1][1] + 1e-9:
            pa, pb, pk = out[-1]
            out[-1] = (pa, max(pb, b), "filler" if "filler" in (pk, kind) else (pk or kind))
        else:
            out.append((a, b, kind))
    return out


class _Prot:
    """Merged, sorted protected intervals with fast overlap queries."""

    def __init__(self, spans):
        self.m = [(a, b) for a, b, _ in _merge([(a, b) for a, b in spans])]
        self.starts = [a for a, _ in self.m]
        self.ends = [b for _, b in self.m]

    def clear(self, x: float, y: float) -> bool:
        """(x, y) touches no protected interval."""
        k = bisect.bisect_left(self.starts, y) - 1
        return not (k >= 0 and self.ends[k] > x)

    def subtract(self, spans) -> list:
        """spans minus the protected union."""
        out = []
        for a, b, kind in spans:
            cur = a
            k = bisect.bisect_right(self.ends, cur)
            while k < len(self.m) and self.m[k][0] < b:
                pa, pb = self.m[k]
                if pa > cur:
                    out.append((cur, pa, kind))
                cur = max(cur, pb)
                if cur >= b:
                    break
                k += 1
            if cur < b:
                out.append((cur, b, kind))
        return out


def kept_side_loud(E, t: float, side: str) -> bool:
    """True when the audio kept next to a cut at t (the 10 ms `before` or `after` it) is voiced
    above floor + 12 dB, i.e. the cut slices through sound instead of silence."""
    c = t - KEPT_WIN / 2 if side == "before" else t + KEPT_WIN / 2
    return E.db_at(c, KEPT_WIN) > E.floor(t) + TIGHT_DB


def _sound_runs(E, a: float, b: float, exclude=()) -> list:
    """Voiced runs (floor + LAUGH_DB, dips up to LAUGH_BRIDGE bridged) of at least LAUGH_MIN
    inside [a, b): laughs, chuckles and other sound in a pause. Frames inside `exclude` spans
    (fillers being removed) do not count."""
    if b <= a:
        return []
    j0, j1 = E.index(a), max(E.index(a) + 1, E.index(b))
    thr = E.floor((a + b) / 2) + LAUGH_DB
    out = []
    bridge = max(0, int(round(LAUGH_BRIDGE / E.hop)))
    minf = max(1, int(round(LAUGH_MIN / E.hop)))
    start = last = None
    gap = 0
    for j in range(j0, j1):
        t = E.time(j) + E.hop / 2
        on = E._db_j(j) > thr and not any(x <= t < y for x, y in exclude)
        if on:
            if start is None:
                start = j
            last, gap = j, 0
        elif start is not None:
            gap += 1
            if gap > bridge:
                if last - start + 1 >= minf:
                    out.append((E.time(start), E.time(last + 1)))
                start, gap = None, 0
    if start is not None and last - start + 1 >= minf:
        out.append((E.time(start), E.time(last + 1)))
    return out


def _kept_excess(E, t: float, side: str) -> float:
    """dB above the floor of the 10 ms kept next to a cut at t (see kept_side_loud)."""
    c = t - KEPT_WIN / 2 if side == "before" else t + KEPT_WIN / 2
    return E.db_at(c, KEPT_WIN) - E.floor(t)


def _edge_excess(edge: dict) -> float:
    return float(edge.get("db", -120.0)) - float(edge.get("floor_db", -80.0))


def _snap_removals(G: Grid, spans, prot: _Prot, k_in: int, k_out: int,
                   min_cut: float = 0.0) -> list:
    """Grid-snap removal spans: inward (a gap keeps at least its `keep` beat); a removal that
    contains a filler snaps outward when that stays clear of protected material, so no sliver
    of the filler's onset or tail survives. No protected span is ever cut."""
    out = []
    for a, b, kind in spans:
        k0, k1 = G.ceil(a), G.floor(b)
        if kind == "filler":
            if prot.clear(G.t(G.floor(a)), a):
                k0 = G.floor(a)
            if prot.clear(b, G.t(G.ceil(b))):
                k1 = G.ceil(b)
        k0, k1 = max(k0, k_in), min(k1, k_out)
        if k1 > k0 and G.dur(k1 - k0) >= min_cut - 1e-9:
            out.append((k0, k1, kind))
    return out


def tighten_run(ctx: _Ctx, run: dict, piece: dict) -> list:
    """Removals [(k0, k1, kind)] for one run of kept words (8.1.2).

    A filler is only removed when both cut points land in quiet audio; one glued to a
    neighbouring word (so the cut would clip it) is kept whole and treated as a kept word,
    never left as a fragment."""
    W, G, E, o = ctx.W, ctx.G, ctx.E, ctx.opts
    k_in, k_out = run["k_in"], run["k_out"]
    t_in, t_out = G.t(k_in), G.t(k_out)
    idxs = run["idxs"]
    use_fillers = bool(o.get("fillers", True))
    cand = [i for i in idxs if use_fillers and ctx.filler(i) and i not in ctx.verb]
    cset = set(cand)

    def guard(i):
        return (W[i]["s"] - PROTECT_PRE, ctx.te(i) + PROTECT_POST)

    prot = [guard(i) for i in idxs if i not in cset]
    for a, b in ctx.clip.get("verbatim") or []:
        if b >= idxs[0] and a <= idxs[-1]:
            prot.append((W[max(a, idxs[0])]["s"] - PROTECT_PRE,
                         ctx.te(min(b, idxs[-1])) + PROTECT_POST))
    prot.append((-math.inf, t_in + (PROTECT_EDGE if run["first"] else 0.0)))
    prot.append((t_out - (PROTECT_EDGE if run["last"] else 0.0), math.inf))
    if run["last"] and piece.get("hold_after", 0) > 0:
        prot.append((ctx.te(idxs[-1]), math.inf))
    for ins in ctx.inserts:                 # speech the editor typed in: never tightened
        a, b = float(ins["src"][0]), float(ins["src"][1])
        if b > t_in and a < t_out:
            prot.append((a - PROTECT_PRE, b + PROTECT_POST))

    fill = list(cand)
    changed = True
    while changed:                          # a kept filler protects its neighbours: iterate
        changed = False
        pr = _Prot(prot)
        for f in list(fill):
            parts = _snap_removals(G, pr.subtract([(W[f]["s"] - FILLER_PAD, W[f]["e"] + FILLER_PAD,
                                                    "filler")]), pr, k_in, k_out)
            if not parts or any(kept_side_loud(E, G.t(k0), "before") or
                                kept_side_loud(E, G.t(k1), "after") for k0, k1, _ in parts):
                fill.remove(f)
                prot.append(guard(f))       # cannot be cut cleanly: keep it whole
                changed = True
    fset = set(fill)
    kept = [i for i in idxs if i not in fset]
    raw = []
    gap, keep = float(o.get("gap", 0.6)), float(o.get("keep", 0.3))
    for u, v in zip(kept, kept[1:]):
        teu, vs = ctx.te(u), W[v]["s"]
        if vs - teu > gap:
            excl = [(W[f]["s"], W[f]["e"]) for f in fill if teu - 0.05 <= W[f]["s"] < vs]
            if E.quiet_gap(teu, vs, exclude=excl):
                raw.append((teu + 0.4 * keep, vs - 0.6 * keep, "gap"))
                # A chuckle or short laugh passes the p75 quiet test; keep it whole (with room
                # either side) so removals only take the quiet parts of the pause.
                for la, lb in _sound_runs(E, teu, vs, excl):
                    prot.append((la - LAUGH_PAD, lb + LAUGH_PAD))
    for f in fill:
        raw.append((W[f]["s"] - FILLER_PAD, W[f]["e"] + FILLER_PAD, "filler"))
    if not raw:
        return []
    pr = _Prot(prot)
    snapped = _snap_removals(G, pr.subtract(_merge(raw)), pr, k_in, k_out,
                             float(o.get("min_cut", 0.12)))
    # a removal edge that would slice through sound (the tail of a laugh, a breath) is dropped
    snapped = [(k0, k1, kind) for k0, k1, kind in snapped
               if not (kept_side_loud(E, G.t(k0), "before") or kept_side_loud(E, G.t(k1), "after"))]
    out, prev = [], k_in
    for k0, k1, kind in snapped:
        if k0 - prev < MIN_KEEP_FRAMES:
            continue
        out.append((k0, k1, kind))
        prev = k1
    while out and k_out - out[-1][1] < MIN_KEEP_FRAMES:
        out.pop()
    return out


# ---- inputs ------------------------------------------------------------------

def _as_project(project):
    if hasattr(project, "root") and hasattr(project, "source"):
        return project
    from .project import Project
    return Project(project)


def _words(transcript: dict) -> list:
    ws = (transcript or {}).get("words") or []
    for k, w in enumerate(ws):
        if w.get("i", k) != k:
            raise env.HCError("transcript word indices are not 0..n-1 in order; re-run transcribe", 2)
    return ws


def _grid_for(video: dict) -> Grid:
    fps = video.get("fps") or [25, 1]
    return Grid(int(fps[0]), int(fps[1]), float(video.get("start") or 0.0))


def _kmax(grid: Grid, video: dict, duration) -> int:
    """Number of grid frames the source really has, i.e. the last usable grid index.

    The grid is timed: frame k shows at offset + k / fps, so the count comes from the stream's
    end time. A VFR file's nb_frames is a packet count, far short of duration x fps when frames
    are dropped or held (a 40 s file with 1089 packets at 30 fps still runs to 40.34 s), so it
    never caps the grid. A CFR file can have more packets than its timed end allows (an mp4 edit
    list trims the tail), so it is capped at the smaller of the two."""
    vdur = video.get("duration")
    try:
        vdur = float(vdur) if vdur is not None else None
    except (TypeError, ValueError):
        vdur = None
    if vdur and vdur > 0:
        end_t = grid.offset + vdur
    else:
        end_t = float(duration or 0.0)
    timed = max(0, grid.floor(end_t)) if end_t > 0 else 0
    nb = video.get("nb_frames") if "nb_frames" in video else video.get("frames")
    nb = nb if isinstance(nb, int) and nb > 0 else None
    if video.get("vfr"):
        return timed if timed > 0 else (nb or 0)
    if nb and timed > 0:
        return min(nb, timed)
    return nb or timed


def _set_hash(spec: dict) -> str:
    return env.stable_hash([[cid, c.get("kind"), [[p.get("words"), p.get("time_src")]
                                                  for p in c.get("pieces") or []]]
                            for cid, c in spec["clips"].items()])


# ---- the planner ---------------------------------------------------------------

def plan_clip(spec: dict, clip_id: str, transcript: dict, source: dict, energy,
              shots: dict | None = None) -> dict:
    """Build the plan for a `cut` clip from already-loaded inputs (pure, no file writes)."""
    clip = spec["clips"][clip_id]
    lints = specmod.validate(spec, transcript, source, [clip_id])
    W = _words(transcript)
    video = source.get("video") or {}
    if not video:
        raise env.HCError("source has no video stream", 1)
    G = _grid_for(video)
    ctx = _Ctx(clip, W, G, energy, _kmax(G, video, source.get("duration")))
    recs = _build_pieces(ctx, clip, lints)
    segs, joins, removals = _segments(ctx, recs)
    words = _plan_words(ctx, recs, segs)
    words, inserts = _insert_words(ctx, recs, segs, words, lints)
    intent = [i for r in recs for i in r["idxs"]
              if not (r["type"] == "words" and i in ctx.omit) and not ctx.filler(i)]
    frames_out = sum(s["frames"] for s in segs)
    dur = G.dur(frames_out)
    plan = {
        "version": 1, "clip": clip_id, "kind": "cut",
        "spec_hash": specmod.clip_hash(spec, clip_id),
        "transcript_sha": transcript.get("sha256"),
        "source_fingerprint": source.get("fingerprint"),
        "code_version": _code_version(),
        "set_hash": _set_hash(spec),
        "shots_hash": (shots or {}).get("_hash"),
        "fps": [G.num, G.den], "grid_offset": G.offset, "sample_rate": SAMPLE_RATE,
        "source": {"width": video.get("width"), "height": video.get("height"),
                   "duration": source.get("duration"), "hdr": video.get("hdr", "none"),
                   "sar": video.get("sar"), "active_area": video.get("active_area")},
        "eff": clip.get("_eff"),
        "speakers": _speakers(spec),
        "pieces": _piece_records(recs, segs, G, W, ctx.omit),
        "segments": segs, "removals": removals, "joins": joins, "words": words,
        "intent": intent, "inserts": inserts,
        "duration_out": round(dur, 6), "frames_out": frames_out,
        "hook_card": _hook(clip, dur),
        "framing": [dict(m) for m in clip.get("framing") or []],
        "cut_rate": {"joins": len(joins), "per_sec": round(len(joins) / dur, 3) if dur > 0 else 0.0},
        "tighten": {"removed_s": round(sum(r["src"][1] - r["src"][0] for r in removals
                                           if r["kind"] != "omit"), 3),
                    "count": sum(1 for r in removals if r["kind"] != "omit")},
    }
    plan["overlaps"] = _overlaps(ctx, recs, segs)
    _lint_hook_covers(clip, dur, lints, clip_id)
    if not segs:
        lints.append(specmod.lint("L01", "FAIL", "nothing to render: no valid pieces", clip_id))
    else:
        lints.extend(_lints(ctx, spec, clip_id, transcript, source, plan, recs, shots))
    plan["lints"] = _sorted_lints(lints, clip_id)
    return plan


def _kept_words(ctx: _Ctx, p: dict) -> list:
    a, b = p["words"]
    return [i for i in range(a, b + 1) if i not in ctx.omit]


def _re_edge(ctx: _Ctx, old: dict, k: int, guard, word: int, **extra) -> dict:
    """A moved edge (same floor), keeping the caption onset of a late_asr in-edge."""
    e = _edge(ctx, k, False, guard, word, old["floor_db"])
    if old.get("onset") is not None:
        e["onset"] = old["onset"]
    if old.get("hold") is not None:
        e["hold"] = dict(old["hold"])
    e.update(extra)
    return e


def _offset_note(ctx: _Ctx, n: int, key: str, asked: int, k_from: int, k_to: int, lints: list,
                 cid=None, own=None) -> None:
    """L01 INFO for an in_offset / out_offset beyond EDGE_OFFSET_NOTE frames: the shift in
    seconds and where the edge landed (so a long skip, e.g. over a false start, is visible).
    `own` (the piece's word indices): when the offset widens a words piece, the note also names
    the transcript words it brings in, which are heard but not captioned."""
    if abs(asked) <= specmod.EDGE_OFFSET_NOTE:
        return
    G = ctx.G
    moved = k_to - k_from
    sec = G.dur(abs(moved))
    edge = "in-edge" if key == "in_offset" else "out-edge"
    way = "later" if moved > 0 else "earlier"
    if moved:
        msg = (f"pieces[{n}] {key} {asked:+d} moves the {edge} {sec:.2f} s {way} "
               f"(src {specmod.fmt_time(G.t(k_from))} -> {specmod.fmt_time(G.t(k_to))})")
    else:
        msg = f"pieces[{n}] {key} {asked:+d} does not move the {edge}"
    if moved != asked:
        msg += (f"; it was held to {moved:+d} frames so the edge stays inside the piece's own "
                "audio and never replays another kept word")
    widened = (key == "in_offset" and moved < 0) or (key == "out_offset" and moved > 0)
    if own is not None and widened:
        lo, hi = sorted((G.t(k_from), G.t(k_to)))
        a = bisect.bisect_left(ctx._starts, lo - 30.0)
        b = bisect.bisect_left(ctx._starts, hi)
        extra = [j for j in range(a, b) if j not in own
                 and lo <= (ctx.W[j]["s"] + ctx.W[j]["e"]) / 2 < hi]
        if extra:
            text = " ".join(ctx.W[j]["w"] for j in extra[:6]) + (" ..." if len(extra) > 6 else "")
            msg += (f"; that brings in words {extra[0]}-{extra[-1]} ('{text}') the piece does "
                    "not list: they are heard but not captioned (widen the piece's words instead)")
    msg += f": more than {specmod.EDGE_OFFSET_NOTE} frames, so check on the seam sheet " + \
        ("what it adds" if widened else "that it skips only what you mean to skip")
    lints.append(specmod.lint("L01", "INFO", msg, cid))


def _apply_offsets(ctx: _Ctx, p: dict, runs: list, owned: dict, n: int = 0,
                   lints: list | None = None) -> None:
    """Piece `in_offset` / `out_offset` (whole frames, + = later), applied after snapping (and
    after the hold_after snap) and clamped so the edge never crosses into another kept word (no
    replay) nor past the piece's own first / last word."""
    G, W = ctx.G, ctx.W
    d_in, d_out = int(p.get("in_offset") or 0), int(p.get("out_offset") or 0)
    sink = lints if lints is not None else []

    def owned_by(piece):                 # the piece's own words plus what hold_after pulled in
        ws = set(range(piece["words"][0], piece["words"][1] + 1)) if piece.get("words") else set()
        return ws | {i for run in runs for i in run["idxs"]}
    if d_in and runs:
        r = runs[0]
        i = r["idxs"][0]
        lo = G.ceil(ctx.te(i - 1)) if i > 0 and (i - 1) in owned else 0
        hi = max(lo, min(G.floor(W[i]["e"]), r["k_out"] - MIN_KEEP_FRAMES))
        k = min(max(r["k_in"] + d_in, lo), hi)
        _offset_note(ctx, n, "in_offset", d_in, r["k_in"], k, sink, own=owned_by(p))
        r["in_moved"] = k - r["k_in"]        # frames the in-edge really moved (lint L04 reads it)
        r["k_in"] = k
        r["edge_in"] = _re_edge(ctx, r["edge_in"], k, r["edge_in"].get("guard"), i, offset=d_in)
    if d_out and runs:
        r = runs[-1]
        j = r["idxs"][-1]
        hi = G.floor(W[j + 1]["s"]) if j + 1 < ctx.N and (j + 1) in owned else ctx.kend
        lo = min(hi, max(G.ceil(W[j]["s"]), r["k_in"] + MIN_KEEP_FRAMES))
        k = max(min(r["k_out"] + d_out, hi), lo)
        _offset_note(ctx, n, "out_offset", d_out, r["k_out"], k, sink, own=owned_by(p))
        r["out_moved"] = k - r["k_out"]      # frames the out-edge really moved
        r["k_out"] = k
        r["edge_out"] = _re_edge(ctx, r["edge_out"], k, r["edge_out"].get("guard"), j,
                                 offset=d_out)


def _clamp_holds(ctx: _Ctx, recs: list, lints: list, cid) -> None:
    """hold_after never replays another piece: when the held out-edge runs into a word another
    piece of this clip keeps, or past the in-edge of the next piece in source order, the edge
    stops there. It also stops before a word the clip omits (the leading word of the next piece
    in source order): an omitted word is never heard. L16 says what a hold pulls in (heard and
    captioned) and where it was clamped."""
    G, W = ctx.G, ctx.W
    owner = {}
    for r in recs:
        if r["type"] == "words":
            for i in _kept_words(ctx, r["p"]):
                owner.setdefault(i, r)
    for r in recs:
        if r["type"] != "words" or not r["runs"] or float(r["p"].get("hold_after") or 0) <= 0:
            continue
        last = r["runs"][-1]
        lims = []
        for j in r["extras"]:
            if j in ctx.omit:
                # a word the clip omits (whichever piece lists it) is never played by a hold
                lims.append((G.floor(W[j]["s"] - NEXT_CLEAR), j, None))
                break
            o = owner.get(j)
            if o is not None and o is not r:
                k_o = next((run["k_in"] for run in o["runs"] if run["idxs"][0] == j), None)
                if k_o is None:
                    k_o = G.floor(W[j]["s"] - OUT_PRE)
                lims.append((k_o, j, o["n"]))
                break
        k_in0 = r["runs"][0]["k_in"]
        later = [o for o in recs if o is not r and o["runs"] and o["runs"][0]["k_in"] > k_in0]
        if later:
            o = min(later, key=lambda o: (o["runs"][0]["k_in"], o["n"]))
            if o["runs"][0]["k_in"] < last["k_out"]:
                lims.append((o["runs"][0]["k_in"], o["runs"][0]["idxs"][0]
                             if o["runs"][0]["idxs"] else None, o["n"]))
        extras = list(r["extras"])
        clamp = None
        if lims:
            k_lim, w_lim, n_lim = min(lims, key=lambda x: x[0])
            if k_lim < last["k_out"]:
                k_new = max(k_lim, last["k_in"] + MIN_KEEP_FRAMES)
                if w_lim is not None:
                    extras = [j for j in extras if j < w_lim]
                extras = [j for j in extras if W[j]["s"] < G.t(k_new) - OUT_PRE]
                base = [i for i in last["idxs"] if i not in set(r["extras"])]
                last["idxs"] = base + extras
                r["extras"] = extras
                last["k_out"] = k_new
                last["edge_out"] = _re_edge(ctx, last["edge_out"], k_new, "hold_clamp",
                                            extras[-1] if extras else base[-1])
                if last["edge_out"].get("hold"):
                    h = last["edge_out"]["hold"]
                    h.update(frame=int(k_new), frame_src=round(G.t(k_new), 6),
                             why="clamped" if n_lim is not None else "omit",
                             got=round(h["got"] + G.t(k_new) - G.t(h["frame"]), 3))
                clamp = (w_lim, n_lim)
        msgs = []
        if extras:
            text = " ".join(W[j]["w"] for j in extras[:6]) + (" ..." if len(extras) > 6 else "")
            msgs.append(f"pulls in words {extras[0]}-{extras[-1]} ('{text}'): they will be heard "
                        "and captioned")
        if clamp:
            where = f"word {clamp[0]}" if clamp[0] is not None else "the in-edge"
            if clamp[1] is None:
                msgs.append(f"stopped before word {clamp[0]}, which the clip omits")
            else:
                msgs.append(f"clamped at {where} where pieces[{clamp[1]}] starts")
        if msgs:
            lints.append(specmod.lint("L16", "WARN", f"hold_after on pieces[{r['n']}] "
                                      + "; ".join(msgs), cid))


def _build_pieces(ctx: _Ctx, clip: dict, lints: list) -> list:
    """Snap every piece (and every omit run) to grid edges; tighten; return piece records."""
    W, G, N = ctx.W, ctx.G, ctx.N
    cid = None
    valid = []
    for n, p in enumerate(clip.get("pieces") or []):
        if p.get("words") and p["words"][1] < N:
            valid.append((n, p))
        elif p.get("time_src"):
            valid.append((n, p))
    owned: dict = {}
    for n, p in valid:
        if p.get("words"):
            for i in _kept_words(ctx, p):
                owned.setdefault(i, n)
    recs = []
    for vi, (n, p) in enumerate(valid):
        last_piece = vi == len(valid) - 1
        if p.get("words"):
            a, b = p["words"]
            kept = _kept_words(ctx, p)
            prev_p = valid[vi - 1][1] if vi > 0 else None
            prev_kept = _kept_words(ctx, prev_p) if prev_p and prev_p.get("words") else []
            if not kept:
                lints.append(specmod.lint("L01", "FAIL", f"pieces[{n}] has no words left after omit",
                                          cid))
                continue
            groups, cur = [], [kept[0]]
            for i in kept[1:]:
                if i == cur[-1] + 1:
                    cur.append(i)
                else:
                    groups.append(cur)
                    cur = [i]
            groups.append(cur)
            runs, extras = [], []
            for gi, g in enumerate(groups):
                ra, rb = g[0], g[-1]
                first, last = gi == 0, gi == len(groups) - 1
                prev_om = ra - 1 >= a and (ra - 1) in ctx.omit
                next_om = rb + 1 <= b and (rb + 1) in ctx.omit
                # the word before this run is heard right before the cut only when it ends the
                # previous piece in clip order (source-adjacent); otherwise it is dropped
                pk = first and bool(prev_kept) and prev_kept[-1] == ra - 1
                e_in = snap_in(ctx, ra, prev_omitted=prev_om, prev_kept=pk)
                tail = (TAIL_LAST if last_piece else TAIL) if last else TAIL
                hold = p.get("hold_after", 0.0) if last else 0.0
                final = last_piece and last and not int(p.get("out_offset") or 0)
                e_out, ex = snap_out(ctx, rb, tail, hold, next_om, final=final)
                if last:
                    extras = ex
                runs.append({"idxs": list(g) + (ex if last else []), "k_in": e_in["k"],
                             "k_out": e_out["k"], "edge_in": e_in, "edge_out": e_out,
                             "first": first, "last": last})
            _apply_offsets(ctx, p, runs, owned, n, lints)
            for r0, r1 in zip(runs, runs[1:]):
                if r1["k_in"] < r0["k_out"] + 1:
                    # overlapping speech around a very short omitted word: never replay audio
                    r1["k_in"] = r0["k_out"]
                    r1["edge_in"] = _edge(ctx, r1["k_in"], True, "no_replay", r1["idxs"][0],
                                          r1["edge_in"]["floor_db"])
                    r0["edge_out"]["tight"] = True
                    lints.append(specmod.lint("L10", "WARN", f"omit before word {r1['idxs'][0]} "
                                              "could not be cut cleanly (continuous speech): the "
                                              "omitted word may still be audible", cid))
            recs.append({"n": n, "p": p, "type": "words", "a": a, "b": b, "runs": runs,
                         "extras": extras})
        else:
            t0, t1 = p["time_src"]
            k_in, k_out = ctx.clampk(G.nearest(t0)), ctx.clampk(G.nearest(t1))
            d_in, d_out = int(p.get("in_offset") or 0), int(p.get("out_offset") or 0)
            if d_in or d_out:
                k_in0, k_out0 = k_in, k_out
                k_in = min(max(0, k_in + d_in), max(0, k_out - MIN_KEEP_FRAMES))
                k_out = max(min(ctx.kmax, k_out + d_out), min(ctx.kmax, k_in + MIN_KEEP_FRAMES))
                if d_in:
                    _offset_note(ctx, n, "in_offset", d_in, k_in0, k_in, lints)
                if d_out:
                    _offset_note(ctx, n, "out_offset", d_out, k_out0, k_out, lints)
            fl0, fl1 = ctx.E.floor(t0), ctx.E.floor(t1)
            ts0, ts1 = G.t(k_in), G.t(k_out)
            idxs = [k for k in range(N) if ts0 <= (W[k]["s"] + W[k]["e"]) / 2 < ts1] \
                if N else []
            run = {"idxs": idxs, "k_in": k_in, "k_out": k_out,
                   "edge_in": _edge(ctx, k_in, False, None, idxs[0] if idxs else None, fl0),
                   "edge_out": _edge(ctx, k_out, False, None, idxs[-1] if idxs else None, fl1),
                   "first": True, "last": True}
            recs.append({"n": n, "p": p, "type": "time", "a": None, "b": None, "runs": [run],
                         "extras": []})
    _clamp_holds(ctx, recs, lints, cid)
    # Source-adjacent consecutive word pieces joined by a plain cut: when their snapped edges
    # overlap (continuous speech would replay) or leave less than min_cut between them, they
    # share one cut point, so the join is seamless (no replay, no micro-gap, no fade dip).
    min_cut = float(ctx.opts.get("min_cut", 0.12))
    for r0, r1 in zip(recs, recs[1:]):
        if r0["type"] == r1["type"] == "words" and r1["p"].get("join") == "cut" and \
                not r0["p"].get("hold_after") and \
                r1["runs"][0]["idxs"][0] == r0["runs"][-1]["idxs"][-1] + 1:
            head, last = r1["runs"][0], r0["runs"][-1]
            shared = head["edge_in"]
            # An explicit offset that trims this join (out_offset < 0 on the first piece,
            # in_offset > 0 on the second) is the editor asking for a cut here: once it leaves
            # a gap it is never merged away. Both the spec value and the real movement count
            # (a negative in_offset that the clamp pushed a frame later is not a trim).
            d_out, d_in = int(r0["p"].get("out_offset") or 0), int(r1["p"].get("in_offset") or 0)
            trimmed = shared["k"] > last["k_out"] and (
                (d_out < 0 and int(last.get("out_moved") or 0) < 0) or
                (d_in > 0 and int(head.get("in_moved") or 0) > 0))
            if not trimmed and G.dur(shared["k"] - last["k_out"]) < min_cut and \
                    shared["k"] - last["k_in"] >= MIN_KEEP_FRAMES:
                last["k_out"] = shared["k"]
                last["edge_out"] = dict(shared, guard="shared", word=last["idxs"][-1])
                if d_out < 0 or d_in > 0:
                    # the two snapped edges overlap, so the offset has not cleared the other
                    # piece's edge yet: say so, a control that does nothing reads as a bug
                    what = ([f"pieces[{r0['n']}] out_offset {d_out:+d}"] if d_out < 0 else []) + \
                        ([f"pieces[{r1['n']}] in_offset {d_in:+d}"] if d_in > 0 else [])
                    m_in = int(head.get("in_moved") or 0) if d_in > 0 else 0
                    lints.append(specmod.lint(
                        "L01", "INFO", " and ".join(what) + f" removed 0 frames: pieces[{r0['n']}] "
                        f"and pieces[{r1['n']}] run into each other and share one cut point"
                        + (f" (it moved {m_in} frame{'' if m_in == 1 else 's'} later)"
                           if m_in > 0 else "")
                        + "; an offset shortens this join once it clears the other piece's edge "
                        "(use a larger one)", cid))
    for r in recs:
        keep_runs = []
        for run in r["runs"]:
            if run["k_out"] - run["k_in"] < MIN_KEEP_FRAMES:
                lints.append(specmod.lint("L01", "FAIL", f"pieces[{r['n']}] is shorter than "
                                          f"{MIN_KEEP_FRAMES} frames after snapping", cid))
                continue
            enabled = ctx.opts.get("enabled", True) and r["p"].get("tighten", True) \
                and r["type"] == "words"
            run["removals"] = tighten_run(ctx, run, r["p"]) if enabled else []
            keeps, cur = [], run["k_in"]
            for k0, k1, _kind in run["removals"]:
                keeps.append((cur, k0))
                cur = k1
            keeps.append((cur, run["k_out"]))
            run["keeps"] = keeps
            keep_runs.append(run)
        r["runs"] = keep_runs
        r["idxs"] = [i for run in r["runs"] for i in run["idxs"]] if r["type"] == "time" else \
            ([i for i in range(r["a"], r["b"] + 1)] + list(r["extras"]) if r["runs"] else [])
        r["idxset"] = set(r["idxs"])
    return [r for r in recs if r["runs"]]


def _segments(ctx: _Ctx, recs: list):
    G, E = ctx.G, ctx.E
    segs, joins, removals = [], [], []
    cum = 0
    prev_run = None
    for r in recs:
        p = r["p"]
        for ri, run in enumerate(r["runs"]):
            if ri > 0 and run["k_in"] > prev_run["k_out"]:
                removals.append({"src": [round(G.t(prev_run["k_out"]), 6), round(G.t(run["k_in"]), 6)],
                                 "kind": "omit", "piece": r["n"]})
            for k0, k1, kind in run["removals"]:
                removals.append({"src": [round(G.t(k0), 6), round(G.t(k1), 6)], "kind": kind,
                                 "piece": r["n"]})
            for ki, (k0, k1) in enumerate(run["keeps"]):
                frames = k1 - k0
                if not segs:
                    join = "start"
                elif ki > 0:
                    join = "tighten"
                elif ri > 0:
                    join = "omit"
                else:
                    join = "dip" if p.get("join") == "dip" else "piece"
                seg = {"k": len(segs), "piece": r["n"],
                       "src": [round(G.t(k0), 6), round(G.t(k1), 6)],
                       "out": [round(G.dur(cum), 6), round(G.dur(cum + frames), 6)],
                       "frames": frames, "samples": G.samples(frames), "join_before": join,
                       "fade_in": FADE, "fade_out": FADE, "gain_db": float(p.get("gain_db", 0.0)),
                       "grid": [k0, k1]}
                if segs:
                    prev = segs[-1]
                    seamless = prev["grid"][1] == k0
                    excess = None
                    if seamless:
                        tight = False
                        prev["fade_out"] = 0.0
                        seg["fade_in"] = 0.0
                    elif join == "tighten":
                        tight = (kept_side_loud(E, G.t(prev["grid"][1]), "before") or
                                 kept_side_loud(E, G.t(k0), "after"))
                        excess = max(_kept_excess(E, G.t(prev["grid"][1]), "before"),
                                     _kept_excess(E, G.t(k0), "after"))
                    else:
                        tight = bool(prev_run["edge_out"]["tight"] or run["edge_in"]["tight"])
                        excess = max(_edge_excess(prev_run["edge_out"]), _edge_excess(run["edge_in"]))
                    joins.append({"k": seg["k"], "out": seg["out"][0], "kind": join,
                                  "before_src": prev["src"][1], "after_src": seg["src"][0],
                                  "tight": bool(tight), "seamless": seamless,
                                  "excess_db": None if excess is None else round(excess, 1)})
                segs.append(seg)
                cum += frames
            prev_run = run
    if segs:
        segs[-1]["fade_out"] = FADE_LAST
    return segs, joins, removals


def _seg_lookup(segs_p: list, t: float, starts: list | None = None):
    """Segment (of one piece, in source order) whose src range holds t, else None."""
    if starts is None:
        starts = [s["src"][0] - EPS for s in segs_p]
    k = bisect.bisect_right(starts, t) - 1
    if k >= 0 and t < segs_p[k]["src"][1]:
        return segs_p[k]
    return None


def _plan_words(ctx: _Ctx, recs: list, segs: list) -> list:
    W, N = ctx.W, ctx.N
    by_piece: dict = {}
    for s in segs:
        by_piece.setdefault(s["piece"], []).append(s)
    # Words whose in-edge found the ASR start late: captions start at the audible onset.
    late = {}
    for r in recs:
        for run in r.get("runs") or []:
            e = run.get("edge_in") or {}
            if e.get("guard") == "late_asr" and e.get("onset") is not None:
                late[int(e["word"])] = float(e["onset"])
    out = []
    for r in recs:
        segs_p = by_piece.get(r["n"]) or []
        if not segs_p:
            continue
        ps0, ps1 = segs_p[0]["src"][0], segs_p[-1]["src"][1]
        starts = [s["src"][0] - EPS for s in segs_p]
        lo = bisect.bisect_left(ctx._starts, ps0 - 30.0)
        hi = bisect.bisect_left(ctx._starts, ps1)
        cands = set(range(lo, hi)) | r["idxset"]
        for i in sorted(cands, key=lambda j: (W[j]["s"], j)):
            w = W[i]
            in_piece = i in r["idxset"]
            if not in_piece and not (w["s"] < ps1 and max(w["e"], w["s"] + 1e-3) > ps0):
                continue
            omitted = in_piece and r["type"] == "words" and i in ctx.omit
            fixv = ctx.fix.get(str(i))
            rec = {"i": i, "raw": w["w"], "text": w["w"] if fixv is None else fixv,
                   "s_src": w["s"], "e_src": w["e"], "s_out": None, "e_out": None,
                   "piece": r["n"], "seg": None, "sp": w.get("sp"),
                   "filler": ctx.filler(i), "hidden": (not in_piece) or fixv == "",
                   "omitted": omitted, "removed": False}
            if not in_piece:
                rec["edge"] = True
            mid = (w["s"] + max(w["e"], w["s"])) / 2
            seg_s, seg_m = _seg_lookup(segs_p, w["s"], starts), _seg_lookup(segs_p, mid, starts)
            nk = bisect.bisect_left(starts, w["s"] - EPS)
            following = segs_p[nk] if nk < len(segs_p) else segs_p[-1]
            if omitted:
                seg = following
            elif in_piece and seg_m is None and ps0 <= mid < ps1:
                seg = following                  # its middle lies in a tighten removal
                rec["removed"] = True
            elif seg_s is not None or seg_m is not None:
                seg = seg_s if seg_s is not None else seg_m
                ws = min(w["s"], late[i]) if in_piece and i in late else w["s"]
                s0 = max(ws, seg["src"][0])
                rec["s_out"] = round(seg["out"][0] + (s0 - seg["src"][0]), 4)
                e = min(max(w["e"], s0), seg["src"][1])
                rec["e_out"] = round(seg["out"][0] + (e - seg["src"][0]), 4)
            elif w["s"] < ps0:
                seg = segs_p[0]
                rec["s_out"] = seg["out"][0]
                e = min(max(w["e"], ps0), seg["src"][1])
                rec["e_out"] = round(seg["out"][0] + (e - ps0), 4)
            else:
                seg = segs_p[-1]
                rec["s_out"] = rec["e_out"] = seg["out"][1]
            rec["seg"] = seg["k"]
            out.append(rec)
    return out


def insert_tokens(text: str, s: float, e: float) -> list:
    """[(token, s, e)]: the words of an insert spread over [s, e] in proportion to their
    character length (the editor gives one span for the whole phrase)."""
    toks = str(text).split()
    total = float(sum(len(t) for t in toks)) or 1.0
    out, acc = [], 0
    for t in toks:
        a = s + (e - s) * acc / total
        acc += len(t)
        b = s + (e - s) * acc / total
        out.append((t, a, b))
    return out


def _insert_words(ctx: _Ctx, recs: list, segs: list, words: list, lints: list, cid=None):
    """Plan words for the clip's `insert` entries (speech the transcript missed) -> (words with
    the inserted tokens placed in time order inside their piece, plan["inserts"]).

    An inserted token is a plan word with `i` None and `ins` = the insert's number: it is
    heard, captioned and timed like any word, has no transcript index and no speaker label, and
    is never addressed by fix / emphasis / omit / verbatim. The whole `src` span must lie in one
    kept segment of the piece that holds `after` (tighten protects it; an omit or a piece edge
    that cuts through it is an L01 FAIL)."""
    if not ctx.inserts:
        return words, []
    by_piece: dict = {}
    for sg in segs:
        by_piece.setdefault(sg["piece"], []).append(sg)
    meta, extra = [], {}
    for n, ins in enumerate(ctx.inserts):
        after, (s, e) = int(ins["after"]), (float(ins["src"][0]), float(ins["src"][1]))
        owners = [r for r in recs if after in r["idxset"]]
        if not owners:
            if 0 <= after < ctx.N:           # in range but dropped with its piece: spec said so
                lints.append(specmod.lint("L01", "FAIL", f"insert[{n}].after {after} is not in "
                                          "any kept piece", cid))
            continue
        hit = None
        for r in owners:
            for sg in by_piece.get(r["n"], []):
                if sg["src"][0] - 1e-3 <= s and e <= sg["src"][1] + 1e-3:
                    hit = (r, sg)
                    break
            if hit:
                break
        if hit is None:
            r = owners[0]
            kept = ", ".join(f"{specmod.fmt_time(sg['src'][0])}-{specmod.fmt_time(sg['src'][1])}"
                             for sg in by_piece.get(r["n"], [])) or "nothing"
            lints.append(specmod.lint(
                "L01", "FAIL", f"insert[{n}].src {specmod.fmt_time(s)}-{specmod.fmt_time(e)} is not "
                f"inside the kept audio of pieces[{r['n']}] (kept: {kept}): it must lie in one "
                "stretch the clip plays (extend the piece with hold_after or out_offset, or "
                "remove the omit that cuts it)", cid))
            continue
        r, sg = hit
        toks = insert_tokens(ins["text"], s, e)
        for tok, a, b in toks:
            a_c, b_c = max(a, sg["src"][0]), min(b, sg["src"][1])
            extra.setdefault(r["n"], []).append({
                "i": None, "ins": n, "raw": tok, "text": tok,
                "s_src": round(a, 4), "e_src": round(b, 4),
                "s_out": round(sg["out"][0] + (a_c - sg["src"][0]), 4),
                "e_out": round(sg["out"][0] + (max(b_c, a_c) - sg["src"][0]), 4),
                "piece": r["n"], "seg": sg["k"], "sp": None, "filler": False, "hidden": False,
                "omitted": False, "removed": False, "_after": after})
        meta.append({"n": n, "after": after, "tokens": [t for t, _a, _b in toks],
                     "src": [round(s, 4), round(e, 4)], "piece": r["n"]})
    if not extra:
        return words, meta
    out, k = [], 0
    while k < len(words):
        pn = words[k]["piece"]
        block = []
        while k < len(words) and words[k]["piece"] == pn:
            block.append(words[k])
            k += 1
        for tok in extra.pop(pn, []):
            # time order, and never before the word it follows
            pos = 0
            for j, w in enumerate(block):
                if w.get("i") == tok["_after"] or w["s_src"] <= tok["s_src"]:
                    pos = j + 1
            tok = {key: v for key, v in tok.items() if key != "_after"}
            block.insert(pos, tok)
        out.extend(block)
    return out, meta


def _piece_records(recs: list, segs: list, G: Grid, W: list = (), omit=frozenset()) -> list:
    out = []
    for r in recs:
        mine = [s for s in segs if s["piece"] == r["n"]]
        p = r["p"]
        first, last = r["runs"][0], r["runs"][-1]
        labels = []
        for i in r["idxs"]:
            sp = W[i].get("sp") if 0 <= i < len(W) else None
            if sp and sp not in labels and not (r["type"] == "words" and i in omit):
                labels.append(sp)
        out.append({
            "n": r["n"], "role": p.get("role", "body"), "join": p.get("join", "cut"),
            "gain_db": float(p.get("gain_db", 0.0)), "words": p.get("words"),
            "time_src": p.get("time_src"), "hold_after": p.get("hold_after", 0.0),
            "chapter": p.get("chapter"), "extra_words": list(r["extras"]),
            "speakers": labels, "in_offset": int(p.get("in_offset") or 0),
            "out_offset": int(p.get("out_offset") or 0),
            "src": [round(G.t(first["k_in"]), 6), round(G.t(last["k_out"]), 6)],
            "out": [mine[0]["out"][0], mine[-1]["out"][1]] if mine else None,
            "frames": sum(s["frames"] for s in mine),
            "edge_in": first["edge_in"], "edge_out": last["edge_out"],
        })
        hold = (last["edge_out"] or {}).get("hold")
        if hold:
            # the frame the hold landed on, before and after out_offset (the fine control)
            out[-1]["hold"] = dict(hold, frame_out=int(last["k_out"]),
                                   src=round(G.t(last["k_out"]), 6))
    return out


def _speakers(spec: dict) -> dict:
    """plan["speakers"]: a copy of the clips.json speakers map ({label: {"x", "y"?}})."""
    out = {}
    for lab, v in ((spec or {}).get("speakers") or {}).items():
        if isinstance(v, dict):
            out[str(lab)] = {k: v[k] for k in ("x", "y") if v.get(k) is not None}
    return out


def _overlaps(ctx: _Ctx, recs: list, segs: list) -> list:
    """Pairs of pieces the audience hears twice (I14 plan["overlaps"]): shared kept words, or
    more than OVERLAP_SRC seconds of shared snapped source."""
    by_piece: dict = {}
    for sg in segs:
        by_piece.setdefault(sg["piece"], []).append((float(sg["src"][0]), float(sg["src"][1])))
    words = {}
    for r in recs:
        ws = set(r["idxs"])
        if r["type"] == "words":
            ws -= (ctx.omit - set(r["extras"]))
        words[r["n"]] = ws
    out = []
    ns = [r["n"] for r in recs]
    for x in range(len(ns)):
        for y in range(x + 1, len(ns)):
            i, j = ns[x], ns[y]
            shared = sorted(words[i] & words[j])
            A, B = _union(by_piece.get(i, [])), _union(by_piece.get(j, []))
            sec = _overlap(A, B)
            if not shared and sec <= OVERLAP_SRC:
                continue
            spans = [(max(a, c), min(b, d)) for a, b in A for c, d in B if min(b, d) > max(a, c)]
            out.append({"pieces": [i, j],
                        "words": [shared[0], shared[-1]] if shared else None,
                        "src": ([round(min(p for p, _ in spans), 6), round(max(q for _, q in spans), 6)]
                                if spans else None),
                        "seconds": round(sec, 3)})
    return out


def accent_phrases(lines) -> list:
    """`*...*` accent runs of a hook card (may span lines)."""
    return [m.strip() for m in re.findall(r"\*([^*]+)\*", " ".join(lines or [])) if m.strip()]


def hook_chars(lines) -> int:
    return sum(len(ln.replace("*", "")) for ln in lines or [])


def _hook(clip: dict, dur_out: float):
    hc = clip.get("hook_card")
    if not hc:
        return None
    a, b, lo, hi = HOOK_AUTO
    if hc.get("dur", "auto") == "auto":
        d = min(max(a + hook_chars(hc["lines"]) / b, lo), hi)
    else:
        d = float(hc["dur"])
    if dur_out > 0:
        d = min(d, dur_out)
    return {"lines": list(hc["lines"]), "dur": round(d, 3)}


def _lint_hook_covers(clip: dict, dur_out: float, lints: list, cid) -> None:
    """L01 FAIL: an explicit hook_card.dur that reaches the end of the PLANNED clip (after
    tighten, omit and snapping; a caption-only clip's whole video), less a second. The spec
    check (spec._check_hook_dur) only has the raw word spans; when it already fired, this adds
    nothing."""
    d = (clip.get("hook_card") or {}).get("dur")
    if isinstance(d, bool) or not isinstance(d, (int, float)) or dur_out <= 0:
        return
    if d >= dur_out - 1.0 and not any(x.get("id") == "L01" and "covers the whole clip" in
                                      str(x.get("msg")) for x in lints):
        lints.append(specmod.lint("L01", "FAIL", f"hook_card.dur {d:g} s covers the whole clip "
                                  f"({dur_out:.1f} s as planned): no caption would ever show; "
                                  "use \"auto\" or a few seconds", cid))


def _sorted_lints(lints: list, clip_id: str) -> list:
    out, seen = [], set()
    for x in lints:
        rec = dict(x)
        rec["clip"] = clip_id
        rec.setdefault("at_out", None)
        key = (rec["id"], rec["level"], rec["msg"], rec["at_out"])
        if key in seen:
            continue
        seen.add(key)
        out.append(rec)
    order = {"FAIL": 0, "WARN": 1, "INFO": 2}
    return sorted(out, key=lambda r: (order.get(r["level"], 3), r["id"],
                                      -1 if r["at_out"] is None else r["at_out"]))


# ---- lints -----------------------------------------------------------------------

def _visible(w: dict) -> bool:
    return not (w["hidden"] or w["removed"] or w["omitted"] or w["filler"])


def _lints(ctx, spec, cid, transcript, source, plan, recs, shots) -> list:
    out = []
    L = lambda lid, lvl, msg, at=None: out.append(specmod.lint(lid, lvl, msg, cid,  # noqa: E731
                                                               None if at is None else round(at, 3)))
    W, G, E = ctx.W, ctx.G, ctx.E
    clip = spec["clips"][cid]
    words = plan["words"]
    eff = clip.get("_eff") or {}
    formats = eff.get("formats") or ["vertical"]
    dur = plan["duration_out"]
    # Heard words: in a piece, not cut by tighten or omit. `fix: ""` hides a word's CAPTION
    # only (the audio still plays it), so a hidden word is heard: hiding an opening filler does
    # not clear the opener FAIL, and a display merge ({"7": "back deck", "8": ""}) at the first
    # or last words of a clip is judged on the words that are said.
    audible = [w for w in words if not (w["removed"] or w["omitted"]) and not w.get("edge")]
    heard = [w for w in audible if not w["filler"]]
    cap = [w for w in words if _visible(w)]

    # L04 opener
    if audible and audible[0]["filler"]:
        L("L04", "FAIL", f"opens on a filler {audible[0]['raw']!r}: start the piece one word later",
          0.0)
    if heard:
        w0 = heard[0]
        i = w0["i"]                      # None for an inserted word (no transcript index)
        why = []
        raw = w0["raw"]
        head = " ".join(w["raw"] for w in heard[:6])
        pre = tn.match_preamble(head)
        if pre:
            why.append(f"preamble phrase {pre!r}")
        elif tn.is_continuation(raw):
            why.append(f"continuation word {raw!r}")
        if tn.is_pronoun_opener(raw) and not _is_determiner(raw, heard[1]["raw"]
                                                            if len(heard) > 1 else None):
            why.append(f"bare pronoun {raw!r}")
        if tn.is_negation(raw):
            why.append(f"negation {raw!r}")
        if i is not None and i > 0:
            pw = W[i - 1]
            pause = W[i]["s"] - ctx.te(i - 1)
            # A time_src piece opens where the editor put it: a transcript word that ends before
            # that in-edge is not in the clip, and its stamped distance to the next word says
            # nothing about the audio at the cut (a false start folded into one word run).
            first = recs[0] if recs else None
            outside = (first is not None and first["type"] == "time"
                       and w0.get("piece") == first["n"]
                       and float(pw["e"]) <= G.t(first["runs"][0]["k_in"]) + EPS)
            # The same holds for a words piece whose in-edge an explicit in_offset moved more
            # than EDGE_OFFSET_NOTE frames later, past the end of the word before it (the remedy
            # for a false start folded into the first word; the L01 INFO names the shift): the
            # editor placed that opening by hand, and what was stepped over is not in the clip.
            stepped = (first is not None and first["type"] == "words"
                       and w0.get("piece") == first["n"]
                       and int(first["runs"][0].get("in_moved") or 0) > specmod.EDGE_OFFSET_NOTE
                       and float(pw["e"]) <= G.t(first["runs"][0]["k_in"]) + EPS)
            if not (outside or stepped) and not tn.is_sentence_end(pw["w"], W[i]["w"]) \
                    and pause < SENTENCE_PAUSE:
                why.append(f"not at a sentence start (previous word {pw['w']!r}, "
                           f"pause {max(pause, 0):.2f} s)")
        if why:
            L("L04", "WARN", "opens on " + "; ".join(why), 0.0)

    # L05 closer
    if heard:
        wl = heard[-1]
        i = wl["i"]
        if i is None:
            # the clip closes on an inserted word: judge it from its own text and end time
            k = bisect.bisect_left(ctx._starts, float(wl["e_src"]) - EPS)
            nxt = W[k] if k < ctx.N else None
            if not tn.is_sentence_end(wl["raw"], nxt["w"] if nxt else None):
                pause = (nxt["s"] - float(wl["e_src"])) if nxt else math.inf
                if pause < SENTENCE_PAUSE:
                    L("L05", "WARN", f"ends on {wl['raw']!r}, not a sentence end, and the speaker "
                      f"continues {max(pause, 0):.2f} s later", dur)
        else:
            nxt = W[i + 1] if i + 1 < ctx.N else None
            if not tn.is_sentence_end(W[i]["w"], nxt["w"] if nxt else None):
                pause = (nxt["s"] - ctx.te(i)) if nxt else math.inf
                if pause < SENTENCE_PAUSE:
                    L("L05", "WARN", f"ends on {wl['raw']!r}, not a sentence end, and the speaker "
                      f"continues {max(pause, 0):.2f} s later", dur)
            elif nxt is not None and not _close_has_air(ctx, i):
                more = " ".join(W[j]["w"] for j in range(i + 1, min(ctx.N, i + 5)))
                L("L05", "WARN", f"ends on {wl['raw']!r} but speech continues immediately "
                  f"({more!r}): the cut will sound clipped (end a sentence earlier or later)", dur)

    # L06 meaning-changing words dropped between kept words
    risky = _risky_word

    def restart(i, kept):
        # "I just, I just think": the omitted word is said again a few words later
        key = tn.norm(W[i]["w"])
        return any(tn.norm(W[j]["w"]) == key for j in range(i + 1, min(ctx.N, i + 1 + RESTART_REACH))
                   if j in kept)
    told = set()
    for r in recs:
        if r["type"] != "words":
            continue
        kept_r = r["idxset"] - ctx.omit
        for i in sorted(ctx.omit & r["idxset"]):
            if r["a"] < i < r["b"] and risky(W[i]["w"]) and not restart(i, kept_r):
                told.add(i)
                L("L06", "WARN", f"omit removes {W[i]['w']!r} (word {i}) between kept words: "
                  "check the meaning survives", _out_near(plan, W[i]["s"]))
    for r0, r1 in zip(recs, recs[1:]):
        if r0["type"] == r1["type"] == "words":
            b0 = max(r0["idxset"])
            a1 = r1["a"]
            kept_0, kept_1 = r0["idxset"] - ctx.omit, r1["idxset"] - ctx.omit
            # The 5-word window is on the pieces as written; the scan runs between the words
            # really heard either side of the cut, so an omit that is the last word of one
            # piece or the first word of the next is seen too.
            if kept_0 and kept_1 and 0 <= a1 - b0 - 1 <= 5:
                for i in range(max(kept_0) + 1, min(kept_1)):
                    if i not in told and risky(W[i]["w"]) and not restart(i, kept_1):
                        L("L06", "WARN", f"the cut between pieces {r0['n']} and {r1['n']} drops "
                          f"{W[i]['w']!r} (word {i}): check the meaning survives",
                          _seg_out_start(plan, r1["n"]))
    _lint_fix_text(ctx, words, L)

    # L07 length band
    for fmt in formats:
        band = (eff.get("length") or {}).get(fmt)
        if band and not band[0] <= dur <= band[3]:
            L("L07", "WARN", f"{fmt}: duration {dur:.1f} s is outside {band[0]:g}-{band[3]:g} s")

    # L08 payoff timing
    hc = plan.get("hook_card")
    if hc:
        first = _first_spoken(cap, accent_phrases(hc["lines"]))
        for phrase, t in first:
            for fmt in formats:
                lim = PAYOFF_BY.get(fmt, 8.0)
                if t is None:
                    L("L08", "WARN", f"hook accent {phrase!r} is never spoken in the clip")
                    break
                if t > lim:
                    L("L08", "WARN", f"{fmt}: hook accent {phrase!r} is first spoken at {t:.1f} s "
                      f"(payoff should land by {lim:g} s)", t)
        # The burned captions skip every word whose midpoint falls before the card ends (less the
        # caption lead), so an accent spoken only under the card is never highlighted in the
        # captions (the captions.json issue accent_unshown and QA G14 report it after the render).
        hdur = float(hc.get("dur") or 0.0)
        for phrase, hits in _spoken_times(cap, accent_phrases(hc["lines"]), mid=True):
            hits = [h for h in hits if h is not None]
            if hits and hdur > 0 and max(hits) < hdur - HOOK_ACCENT_LEAD:
                L("L08", "WARN", f"hook accent {phrase!r} is spoken only under the hook card "
                  f"(at {max(hits):.1f} s; the card holds until {hdur:.1f} s), so the captions "
                  "never highlight it: accent a word spoken later, or shorten hook_card.dur", hits[0])

    # L09 cut rate
    if dur > 0 and plan["cut_rate"]["per_sec"] > CUT_RATE_MAX:
        L("L09", "WARN", f"cut rate {plan['cut_rate']['per_sec']:.2f} joins/s is above "
          f"{CUT_RATE_MAX} (choppy); loosen tighten or merge pieces")

    # L10 tight edges: WARN only when the cut is clearly inside sound (floor + L10_DB at piece
    # edges, floor + L10_JOIN_DB at omit and tighten joins, which are short and masked by the
    # surrounding speech); a merely tight cut is INFO.
    segs = plan["segments"]
    if recs and recs[0]["runs"][0]["edge_in"]["tight"]:
        e = recs[0]["runs"][0]["edge_in"]
        L("L10", "WARN" if _edge_excess(e) > L10_DB else "INFO",
          f"clip starts inside voiced audio ({e['db']} dB, floor {e['floor_db']} dB)", 0.0)
    for j in plan["joins"]:
        if j["tight"]:
            lim = L10_JOIN_DB if j["kind"] in ("omit", "tighten") else L10_DB
            ex = j.get("excess_db")
            lvl = "WARN" if ex is None or ex > lim else "INFO"
            L("L10", lvl, f"{j['kind']} join cuts inside voiced audio "
              f"(src {specmod.fmt_time(j['before_src'])} -> {specmod.fmt_time(j['after_src'])}"
              + (f", {ex:+.0f} dB over the floor)" if ex is not None else ")"), j["out"])
    if recs and recs[-1]["runs"][-1]["edge_out"].get("guard") == "consonant":
        e = recs[-1]["runs"][-1]["edge_out"]
        n_auto = int(e.get("auto") or 0)
        L("L10", "INFO", f"the last word's final consonant runs into the next word: the out-edge "
          f"moved {n_auto} frame{'' if n_auto == 1 else 's'} later ({G.dur(n_auto):.2f} s) so "
          "the closing fade is centred on the dip between them and the consonant is heard (an "
          f"out_offset on the last piece replaces this; out_offset {n_auto} keeps this edge)", dur)
    elif recs and recs[-1]["runs"][-1]["edge_out"]["tight"]:
        e = recs[-1]["runs"][-1]["edge_out"]
        L("L10", "WARN" if _edge_excess(e) > L10_DB else "INFO",
          f"clip ends inside voiced audio ({e['db']} dB, floor {e['floor_db']} dB)", dur)

    # L11 shots without framing marks
    if shots and shots.get("shots"):
        _lint_shots(plan, clip, shots, formats, L)

    # L12 upscale
    if "vertical" in formats:
        _lint_upscale(plan, clip, source, L)

    # L13 hook card / cover / post facts
    _lint_facts(ctx, clip, plan, L, transcript)

    # L14 overlap with other clips
    mine = _union([tuple(s["src"]) for s in segs])
    for oid, oc in spec["clips"].items():
        if oid == cid or oc.get("kind") != "cut":
            continue
        rng = []
        for p in oc.get("pieces") or []:
            if p.get("words") and p["words"][1] < ctx.N:
                rng.append((W[p["words"][0]]["s"], W[p["words"][1]]["e"]))
            elif p.get("time_src"):
                rng.append(tuple(p["time_src"]))
        ov = _overlap(mine, _union(rng))
        if ov > OVERLAP_MAX:
            L("L14", "WARN", f"shares {ov:.1f} s of source with clip {oid!r}")

    # L15 transcript flags
    _lint_flags(ctx, plan, recs, transcript, L)

    # L17 pieces that overlap each other (heard twice)
    roles = {pc["n"]: pc.get("role") for pc in plan["pieces"]}
    for ov in plan.get("overlaps") or []:
        i, j = ov["pieces"]
        what = []
        if ov.get("words"):
            a, b = ov["words"]
            text = " ".join(W[k]["w"] for k in range(a, min(b, a + 5) + 1)) + \
                (" ..." if b - a > 5 else "")
            what.append(f"words {a}-{b} ('{text}')")
        if ov.get("seconds", 0.0) > OVERLAP_SRC or not what:
            what.append(f"{ov.get('seconds', 0.0):.2f} s of source")
        lvl = "FAIL" if "cold_open" in (roles.get(i), roles.get(j)) else "WARN"
        L("L17", lvl, f"pieces[{i}] and pieces[{j}] share {' / '.join(what)}: the audience "
          "hears it twice", _seg_out_start(plan, j))

    # L18 dead air at a segment start, L19 speech with no transcript words
    if E is not None:
        _lint_dead_air(ctx, plan, recs, L)
        _lint_holes(ctx, plan, recs, transcript, L)
    return out


def _risky_word(raw) -> bool:
    """A word whose loss changes the meaning: a negation, a hedge or a qualifier (lint L06)."""
    return bool(tn.is_negation(raw) or tn.is_hedge(raw) or tn.norm(raw) in QUALIFIERS)


def _lint_fix_text(ctx, words: list, L) -> None:
    """L06 for `fix`: it changes what the captions (and the .srt) show, never the audio, so a
    fix that hides a negation, hedge or qualifier, or that changes a number, makes the captions
    say something the speaker does not. Consecutive fixed words are judged as one run, so the
    honest uses stay quiet: a display merge ("back deck" + ""), "do not" -> "don't" + "",
    "twenty five grand" -> "$25K", a masked word, a corrected name. QA G5 compares the audio
    with the raw words and G11 counts fix text as spoken, both on purpose, so this lint is the
    only place a caption that contradicts the audio is named."""
    from . import claims
    runs, cur = [], []
    for w in words:
        i = w.get("i")
        fixed = (i is not None and not w.get("edge") and not w["omitted"] and not w["removed"]
                 and str(i) in ctx.fix)
        if fixed and cur and cur[-1]["i"] == i - 1 and cur[-1]["piece"] == w["piece"]:
            cur.append(w)
            continue
        if cur:
            runs.append(cur)
        cur = [w] if fixed else []
    if cur:
        runs.append(cur)

    def values(tokens) -> set:
        try:
            return {round(float(v), 6) for v in claims.Spoken(tokens).values}
        except Exception:                # a lint never stops a plan
            return set()

    for run in runs:
        raw = [str(w["raw"]) for w in run]
        disp = [t for w in run for t in str(w["text"]).split()]
        if raw == disp:
            continue
        at = next((w["s_out"] for w in run if w.get("s_out") is not None), None)
        if not any(_risky_word(t) for t in disp):
            for w in run:
                if _risky_word(w["raw"]):
                    L("L06", "WARN", f"fix hides {w['raw']!r} (word {w['i']}) from the captions "
                      "while the audio keeps it: the captions will not say what is said", at)
        rv, dv = values(raw), values(disp)
        if (dv - rv - values(["".join(raw)])) or (rv and not dv):
            where = f"word {run[0]['i']}" if len(run) == 1 else \
                f"words {run[0]['i']}-{run[-1]['i']}"
            L("L06", "WARN", f"fix changes a number {' '.join(raw)!r} -> {' '.join(disp)!r} "
              f"({where}): the captions will not match the audio; if the transcript misheard "
              "it, confirm with the user", at)


def _is_determiner(raw: str, next_raw) -> bool:
    """'That conversation ...': this/that/these/those before a noun is a determiner, not a bare
    pronoun opener ('That is why ...' / 'That.' stay bare)."""
    core = str(raw).strip("\"'()[]{}")
    if tn.norm(core) not in DEMONSTRATIVES or not core or not core[-1].isalnum():
        return False
    if next_raw is None:
        return False
    nxt = str(next_raw).strip("\"'()[]{}.,;:!?").lower()
    return bool(nxt) and nxt not in PRONOUN_VERBS and tn.norm(nxt) not in PRONOUN_VERBS


def _close_has_air(ctx, i: int) -> bool:
    """A sentence-final last word is a clean close when the next word starts CLOSE_GAP or more
    after its true end, or the audio dips (below floor + 6 dB) for CLOSE_QUIET within
    CLOSE_WIN after it."""
    W, E = ctx.W, ctx.E
    te = ctx.te(i)
    if i + 1 >= ctx.N or W[i + 1]["s"] - te >= CLOSE_GAP:
        return True
    if E is None:
        return False
    thr = E.floor(te) + 6.0
    need = max(1, int(round(CLOSE_QUIET / E.hop)))
    run = 0
    for j in range(E.index(te), E.index(te + CLOSE_WIN) + 1):
        if E._db_j(j) < thr:
            run += 1
            if run >= need:
                return True
        else:
            run = 0
    return False


TIME_FLAGS = ("unworded_audio", "dropout")
INSERT_COVERS = 0.5      # an unworded_audio flag this much covered by an insert is dealt with


def _insert_spans(plan) -> list:
    """Source spans of the clip's inserts (speech the editor typed in), merged."""
    return _union([tuple(m["src"]) for m in plan.get("inserts") or []])


def _lint_flags(ctx, plan, recs, transcript, L) -> None:
    """L15: transcript flags inside the clip (word flags by index, time flags by source).
    `false_start` and `timing_uncertain` are word flags: they WARN only when a flagged word is
    inside the clip, and a `false_start` only while the clip still plays the abandoned attempt
    (its s..e span): once an edge sits on the restart there is nothing left to check. An
    `unworded_audio` flag the clip captions with an `insert` is settled."""
    W = ctx.W
    idxs = set(plan["intent"]) | {i for r in recs for i in r["idxs"]}
    spans = _union([tuple(s["src"]) for s in plan["segments"]])
    typed = _insert_spans(plan)
    for f in transcript.get("flags") or []:
        kind = f.get("kind")
        if kind == "respliced":
            continue
        if kind in TIME_FLAGS and isinstance(f.get("s"), (int, float)) \
                and isinstance(f.get("e"), (int, float)):
            fs, fe = float(f["s"]), float(f["e"])
            hit = [(max(a, fs), min(b, fe)) for a, b in spans if min(b, fe) > max(a, fs)]
            if hit and kind == "unworded_audio" and typed:
                inside = sum(b - a for a, b in hit)
                if inside > 0 and _overlap(hit, typed) >= INSERT_COVERS * inside:
                    continue
            if hit:
                L("L15", "WARN", _flag_msg(f, f.get("a"), f.get("b")),
                  _out_near(plan, hit[0][0]))
            continue
        try:
            a, b = int(f["a"]), int(f["b"])
        except (KeyError, TypeError, ValueError):
            continue
        hit = sorted(i for i in idxs if a <= i <= b)
        if hit and kind == "false_start" and isinstance(f.get("s"), (int, float)) \
                and isinstance(f.get("e"), (int, float)):
            # s..e is the abandoned attempt. A clip that plays none of it (under one output
            # frame) has already cut the false start: the in-edge sits on the restart.
            fs, fe = float(f["s"]), float(f["e"])
            played = sum(min(y, fe) - max(x, fs) for x, y in spans if min(y, fe) > max(x, fs))
            if played <= ctx.G.dur(1) + EPS:
                continue
        if hit:
            t_src = f.get("s") if isinstance(f.get("s"), (int, float)) else \
                (W[a]["s"] if 0 <= a < ctx.N else None)
            L("L15", "WARN", _flag_msg(f, a, b, t_src),
              _out_near(plan, W[hit[0]]["s"]) if hit[0] < ctx.N else None)


def _lint_dead_air(ctx, plan, recs, L) -> None:
    """L18: a segment that opens on silence before its first sound (dead air under a caption)."""
    from . import align
    E = ctx.E
    kinds = {r["n"]: r["type"] for r in recs}
    joins = {j["k"]: j for j in plan["joins"]}
    for seg in plan["segments"]:
        kind = seg.get("join_before", "start")
        if kind not in ("start", "piece", "omit", "tighten") or kinds.get(seg["piece"]) == "time":
            continue
        if seg["k"] in joins and joins[seg["k"]].get("seamless"):
            continue
        s0 = float(seg["src"][0])
        lim = DEAD_AIR_START if seg["out"][0] <= 1e-6 else DEAD_AIR
        look = min(DEAD_AIR_LOOK, float(seg["src"][1]) - s0)
        if look <= lim:
            continue
        on = align.next_onset(E, s0, s0 + look)
        gap = (on - s0) if on is not None else look
        if gap > lim:
            more = "" if on is not None else " or more"
            if kind == "start":
                msg = f"the clip opens on {gap:.2f} s{more} of silence before the first word"
            else:
                msg = (f"{kind} join at out {seg['out'][0]:.2f} opens on {gap:.2f} s{more} of "
                       "silence before the first word")
            L("L18", "WARN", msg + " (start the piece later, or move its in-edge with in_offset)",
              seg["out"][0])


def _lint_holes(ctx, plan, recs, transcript, L) -> None:
    """L19: voiced audio inside the clip that no transcript word covers (ASR dropped it, so it
    plays uncaptioned and unchecked)."""
    from . import align
    E, W = ctx.E, ctx.W
    flags = [(float(f["s"]), float(f["e"])) for f in transcript.get("flags") or []
             if f.get("kind") == "unworded_audio" and isinstance(f.get("s"), (int, float))
             and isinstance(f.get("e"), (int, float))]
    kinds = {r["n"]: r["type"] for r in recs}
    for seg in plan["segments"]:
        a, b = float(seg["src"][0]), float(seg["src"][1])
        lo = bisect.bisect_left(ctx._starts, a - 30.0)
        hi = bisect.bisect_left(ctx._starts, b)
        spans = [W[i] for i in range(lo, hi) if W[i]["e"] > a and W[i]["s"] < b]
        spans += [{"s": x, "e": y} for x, y in _insert_spans(plan) if y > a and x < b]
        spans.sort(key=lambda w: (w["s"], w["e"]))
        for hs, he in align.uncovered_voiced(E, spans, a, b, min_run=HOLE_WARN):
            d = he - hs
            covered = any(min(he, fe) > max(hs, fs) for fs, fe in flags)
            at = seg["out"][0] + (hs - a)
            if covered:
                L("L19", "WARN", f"{d:.1f} s of sound at out {at:.2f} has no transcript words "
                  "(transcribe flagged it: laughter, music or speech ASR missed): listen to it, "
                  "captions and QA cannot see it", at)
                continue
            lvl = "FAIL" if d >= HOLE_FAIL and kinds.get(seg["piece"]) != "time" else "WARN"
            L("L19", lvl, f"{d:.1f} s of speech at out {at:.2f} has no transcript words: "
              "re-run hc transcribe (it re-listens to gaps) or keep it out of the clip", at)


def _flag_msg(f: dict, a, b, t_src=None) -> str:
    kind = f.get("kind", "flag")
    if kind == "false_start":
        t = f.get("s") if isinstance(f.get("s"), (int, float)) else t_src
        return (f"possible false start or repeated attempt at src {specmod.fmt_time(t)} "
                f"(words {a}-{b}): check with hc words and a slice transcription before opening "
                "on it")
    if kind == "timing_uncertain":
        return (f"word timing near src {specmod.fmt_time(t_src)} is uncertain (overlapping "
                "sound): check the caption onset in the render")
    if kind == "repetition_loop":
        return (f"transcript repetition loop on words {a}-{b} is inside the clip: the text there "
                "is unreliable, read it with hc words before keeping it")
    if kind == "low_confidence":
        w = f" {f['w']!r}" if f.get("w") else ""
        p = f" (p {f['p']:.2f})" if isinstance(f.get("p"), (int, float)) else ""
        return (f"low-confidence transcript word{w}{p} at words {a}-{b}: check it and add a "
                "\"fix\" if it is wrong")
    if kind == "unworded_audio":
        d = float(f.get("e", 0.0)) - float(f.get("s", 0.0))
        return (f"{d:.1f} s of sound with no words at src {specmod.fmt_time(f.get('s', 0.0))} is "
                "inside the clip (laughter, music or speech ASR missed): transcribe that stretch on "
                "its own before keeping it (hc transcribe P --file SNIPPET --clip SCRATCH-ID) and "
                "caption speech the transcript missed with \"insert\"")
    if kind == "dropout":
        n = f.get("vtt_words")
        n = len(n) if isinstance(n, (list, tuple)) else n
        return (f"the speakers VTT has {n if n is not None else 'some'} words at src "
                f"{specmod.fmt_time(f.get('s', 0.0))} that ASR did not transcribe")
    if kind == "speaker_uncertain":
        return f"speaker label of words {a}-{b} is uncertain: check attribution"
    if kind == "no_audio":
        return (f"words {a}-{b} have no audible sound (likely an ASR hallucination): omit them")
    return f"transcript flag {kind!r} on words {a}-{b} is inside the clip: check it"


def _out_near(plan, t_src):
    v = out_time(plan, t_src)
    if v is not None:
        return v
    for s in plan["segments"]:
        if s["src"][0] >= t_src:
            return s["out"][0]
    return plan["duration_out"]


def _seg_out_start(plan, piece_n):
    for s in plan["segments"]:
        if s["piece"] == piece_n:
            return s["out"][0]
    return None


def _spoken_times(cap: list, phrases: list, mid: bool = False) -> list:
    """[(phrase, [s_out of every occurrence])] matching content tokens consecutively (`mid`: the
    first matched word's midpoint instead of its s_out)."""
    raws = [w["text"] or w["raw"] for w in cap]
    stream = [(t, k) for t, k, _ in tn.norm_stream(raws)
              if t not in tn.STOPWORDS and not tn.is_filler(t)]
    toks = [t for t, _ in stream]
    out = []
    for ph in phrases:
        P = tn.content_tokens(ph)
        hits = []
        if P:
            for j in range(0, len(toks) - len(P) + 1):
                if toks[j:j + len(P)] == P:
                    w = cap[stream[j][1]]
                    hits.append((w["s_out"] + w["e_out"]) / 2.0
                                if mid and w.get("e_out") is not None else w["s_out"])
        out.append((ph, hits))
    return out


def _first_spoken(cap: list, phrases: list) -> list:
    """[(phrase, first s_out or None)] matching content tokens consecutively."""
    return [(ph, hits[0] if hits else None) for ph, hits in _spoken_times(cap, phrases)]


def _union(spans):
    out = []
    for a, b in sorted(spans):
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def _overlap(xs, ys) -> float:
    tot = 0.0
    for a, b in xs:
        for c, d in ys:
            tot += max(0.0, min(b, d) - max(a, c))
    return tot


def _lint_shots(plan, clip, shots, formats, L):
    marks = clip.get("framing") or []
    if formats == ["landscape"] and not any(m["mode"] in ("track", "fixed") for m in marks):
        return
    sh = sorted((float(s["start_src"]), float(s.get("end_src") or math.inf))
                for s in shots["shots"] if s.get("start_src") is not None)
    # A shot that "starts" where an analysed range starts is not a cut: the analysis simply began
    # there, and whatever framing is in effect before it still applies.
    ranges = shots.get("ranges") or ([shots["range_src"]] if shots.get("range_src") else [])
    range_starts = [float(r[0]) for r in ranges if r]
    cut_starts = [float(c["src"]) for c in shots.get("cuts") or [] if c.get("src") is not None]
    virtual = {k for k, (a, _b) in enumerate(sh)
               if any(abs(a - r) < 1e-3 for r in range_starts)
               and not any(abs(a - c) < 1e-3 for c in cut_starts)}
    seen, first = set(), True
    for seg in plan["segments"]:
        s0, s1 = seg["src"]
        for k, (a, b) in enumerate(sh):
            if b <= s0 or a >= s1 or k in seen:
                continue
            seen.add(k)
            has = any(a - 0.25 <= m["at_src"] < b for m in marks)
            prior = any(m["at_src"] < a - 0.25 for m in marks)
            if not has and (prior or not first) and k not in virtual:
                at = seg["out"][0] + max(0.0, a - s0)
                L("L11", "WARN", f"shot starting at src {specmod.fmt_time(a)} has no framing mark: "
                  "it inherits the previous framing (add a mark at or after its start)", at)
            first = False


def _lint_upscale(plan, clip, source, L):
    """L12: how much the vertical framing enlarges the picture, measured against the active
    picture (a letterboxed or gallery source's real image, not its black bars)."""
    from . import framing
    src = source if (source or {}).get("video") else {"video": dict(plan.get("source") or {})}
    try:
        dw, dh = framing.display_size(src)
    except (env.HCError, KeyError, TypeError, ValueError, ZeroDivisionError):
        return
    if not dw or not dh:
        return
    act = framing.active_rect(src)
    out_w, out_h = OUT_SIZE["vertical"]
    segs = plan["segments"]
    if not segs:
        return
    lo = min(s["src"][0] for s in segs)
    hi = max(s["src"][1] for s in segs)
    marks = clip.get("framing") or []
    active = [m for m in marks if lo <= m["at_src"] < hi]
    before = [m for m in marks if m["at_src"] < lo]
    if before:
        active.insert(0, before[-1])
    if not active or active[0]["at_src"] > lo:
        active.insert(0, {"at_src": None, "mode": "default", "crop": 1.0})
    worst = None
    for m in active:
        if m["mode"] in ("letterbox", "full"):
            r = framing.norm_rect_px(m.get("rect"), dw, dh) if m.get("rect") else None
            rw, rh = (r or act)[2], (r or act)[3]
            up = min(out_w / float(rw), out_h / float(rh))
        else:
            ch = framing.crop_size(dw, dh, "vertical", float(m.get("crop") or 1.0), 1.0, act)[1]
            up = out_h / float(ch)
        if up > UPSCALE_MAX and (worst is None or up > worst[0]):
            worst = (up, m)
    if worst:
        up, m = worst
        where = "default framing" if m["at_src"] is None else \
            f"mark at {specmod.fmt_time(m['at_src'])} ({m['mode']}, crop {m.get('crop', 1.0)})"
        area = "" if (act[2], act[3]) == (dw, dh) else \
            f" whose active picture is {act[2]}x{act[3]} (bars or a gallery border around it)"
        L("L12", "WARN", f"vertical {where} upscales {up:.1f}x from a {dw}x{dh} source{area}: it "
          "will look soft (use a wider crop, letterbox, or letterbox with a \"rect\" around one "
          "tile)")


_EVIDENCE: dict = {}


def _name_evidence(transcript) -> dict:
    """norm token -> [capitalized, lowercase] counts over non-sentence-initial transcript words
    (claims: a word the transcript only ever capitalizes mid-sentence is a name)."""
    words = (transcript or {}).get("words") or []
    key = ((transcript or {}).get("sha256"), len(words))
    hit = _EVIDENCE.get(key)
    if hit is not None:
        return hit
    ev: dict = {}
    prev = None
    for w in words:
        raw = str(w.get("w", ""))
        core = raw.strip("\"'()[]{}.,;:!?")
        initial = prev is None or tn.is_sentence_end(prev)
        prev = raw
        if not core or not core[0].isalpha() or initial:
            continue
        k = tn.norm(core)
        if not k:
            continue
        rec = ev.setdefault(k, [0, 0])
        rec[0 if core[0].isupper() else 1] += 1
    _EVIDENCE.clear()
    _EVIDENCE[key] = ev
    return ev


def _written_surfaces(claims, clip, plan) -> list:
    """[(label, text)] of everything the clip puts in writing. claims.written_surfaces is the
    one definition (the cover drawer and QA G11 use it too): it knows that a clip with no
    cover.headline and no hook card prints its TITLE on the cover. The inline list below is
    the older, narrower one, kept only for a claims module that does not have it."""
    fn = getattr(claims, "written_surfaces", None)
    if fn is not None:
        return list(fn(clip, plan))
    surfaces = []
    hc = clip.get("hook_card")
    if isinstance(hc, dict) and hc.get("lines"):
        surfaces.append(("hook card", " ".join(str(x) for x in hc["lines"])))
    cover = clip.get("cover") if isinstance(clip.get("cover"), dict) else {}
    if cover.get("headline"):
        surfaces.append(("cover headline", str(cover["headline"])))
    post = clip.get("post") if isinstance(clip.get("post"), dict) else {}
    for key, label in (("title", "post title"), ("caption", "post caption")):
        if post.get(key):
            surfaces.append((label, str(post[key])))
    return surfaces


def _lint_facts(ctx, clip, plan, L, transcript=None):
    """L13: numbers, names and words on the hook card, cover headline and post that the clip's
    words never say (claims.unspoken_claims, the extractor QA G11 uses; always WARN here)."""
    from . import claims
    surfaces = _written_surfaces(claims, clip, plan)
    if not surfaces:
        return
    spoken = []
    for i in plan["intent"]:
        if 0 <= i < ctx.N:
            spoken.append(ctx.W[i]["w"])
            fx = ctx.fix.get(str(i))
            if fx:
                spoken.append(fx)
    for m in plan.get("inserts") or []:
        spoken.extend(m.get("tokens") or [])
    labels = [clip.get("speaker")] + sorted({w.get("sp") for w in plan["words"]
                                             if w.get("sp") and not w.get("hidden")})
    items = claims.unspoken_claims(surfaces, spoken,
                                   glossary=list((transcript or {}).get("glossary") or []),
                                   exempt_names=[x for x in labels if x],
                                   name_evidence=_name_evidence(transcript))
    for it in items:
        L("L13", "WARN", f"{it['surface']} {it['kind']} {it['raw']!r} is not in the clip's words")


# ---- caption-only clips ------------------------------------------------------

def plan_caption_only(spec: dict, clip_id: str, transcript: dict | None, media: dict,
                      media_fingerprint: str | None = None) -> dict:
    """One segment = the whole finished video; out = src (minus its video start)."""
    clip = spec["clips"][clip_id]
    lints = specmod.validate(spec, None, None, [clip_id],
                             clip_transcripts={clip_id: transcript} if transcript else None)
    video = media.get("video") or {}
    if not video:
        raise env.HCError(f"{clip['source_file']} has no video stream", 1)
    G = _grid_for(video)
    frames = _kmax(G, video, media.get("duration"))
    W = _words(transcript) if transcript else []
    if transcript is None:
        lints.append(specmod.lint("L01", "FAIL", f"no transcript for caption-only clip: run "
                                  f"`hc transcribe P --file {clip['source_file']} --clip {clip_id}`",
                                  clip_id))
    seg = {"k": 0, "piece": 0, "src": [round(G.t(0), 6), round(G.t(frames), 6)],
           "out": [0.0, round(G.dur(frames), 6)], "frames": frames, "samples": G.samples(frames),
           "join_before": "start", "fade_in": FADE, "fade_out": FADE_LAST, "gain_db": 0.0,
           "grid": [0, frames]}
    words = []
    for k, w in enumerate(W):
        fixv = clip["fix"].get(str(k))
        s_out = min(max(w["s"] - G.offset, 0.0), seg["out"][1])
        e_out = min(max(w["e"] - G.offset, s_out), seg["out"][1])
        words.append({"i": k, "raw": w["w"], "text": w["w"] if fixv is None else fixv,
                      "s_src": w["s"], "e_src": w["e"], "s_out": round(s_out, 4),
                      "e_out": round(e_out, 4), "piece": 0, "seg": 0, "sp": w.get("sp"),
                      "filler": _is_filler_word(w["w"], fixv), "hidden": fixv == "",
                      "omitted": False, "removed": False})
    dur = G.dur(frames)
    ctx = _Ctx(clip, W, G, None, frames)
    words, inserts = _insert_words(ctx, [{"n": 0, "idxset": set(range(len(W)))}], [seg], words,
                                   lints, clip_id)
    plan = {
        "version": 1, "clip": clip_id, "kind": "caption_only",
        "spec_hash": specmod.clip_hash(spec, clip_id),
        "transcript_sha": (transcript or {}).get("sha256"),
        "source_fingerprint": media_fingerprint, "code_version": _code_version(),
        "set_hash": _set_hash(spec), "shots_hash": None,
        "fps": [G.num, G.den], "grid_offset": G.offset, "sample_rate": SAMPLE_RATE,
        "source": {"width": video.get("width"), "height": video.get("height"),
                   "duration": media.get("duration"), "hdr": video.get("hdr", "none"),
                   "path": clip.get("source_file")},
        "eff": clip.get("_eff"),
        "speakers": _speakers(spec),
        "pieces": [{"n": 0, "role": "body", "join": "cut", "gain_db": 0.0, "words": None,
                    "time_src": None, "hold_after": 0.0, "chapter": None, "extra_words": [],
                    "speakers": sorted({w.get("sp") for w in W if w.get("sp")}),
                    "in_offset": 0, "out_offset": 0,
                    "src": seg["src"], "out": seg["out"], "frames": frames,
                    "edge_in": None, "edge_out": None}],
        "overlaps": [],
        "segments": [seg], "removals": [], "joins": [], "words": words,
        "intent": [w["i"] for w in words if not w["filler"] and w["i"] is not None],
        "inserts": inserts,
        "duration_out": round(dur, 6), "frames_out": frames,
        "hook_card": _hook(clip, dur),
        "framing": [dict(m) for m in clip.get("framing") or []],
        "cut_rate": {"joins": 0, "per_sec": 0.0},
        "tighten": {"removed_s": 0.0, "count": 0},
    }
    eff = clip.get("_eff") or {}
    for fmt in eff.get("formats") or ["vertical"]:
        band = (eff.get("length") or {}).get(fmt)
        if band and frames and not band[0] <= dur <= band[3]:
            lints.append(specmod.lint("L07", "WARN", f"{fmt}: duration {dur:.1f} s is outside "
                                      f"{band[0]:g}-{band[3]:g} s", clip_id))
    if frames == 0:
        lints.append(specmod.lint("L01", "FAIL", "the caption-only video has no frames", clip_id))
    _lint_hook_covers(clip, dur, lints, clip_id)
    sink: list = []
    if plan["hook_card"]:
        cap = [w for w in words if _visible(w)]
        for phrase, t in _first_spoken(cap, accent_phrases(plan["hook_card"]["lines"])):
            if t is None:
                sink.append(specmod.lint("L08", "WARN", f"hook accent {phrase!r} is never spoken "
                                         "in the clip", clip_id))
    _lint_fix_text(ctx, words, lambda lid, lvl, msg, at=None: sink.append(
        specmod.lint(lid, lvl, msg, clip_id, at)))
    _lint_facts(ctx, clip, plan, lambda lid, lvl, msg, at=None: sink.append(
        specmod.lint(lid, lvl, msg, clip_id, at)), transcript)
    for f in (transcript or {}).get("flags") or []:
        if f.get("kind") == "respliced":
            continue
        fa = f.get("a")
        t_src = W[fa]["s"] if isinstance(fa, int) and 0 <= fa < len(W) else None
        sink.append(specmod.lint("L15", "WARN", _flag_msg(f, fa, f.get("b"), t_src), clip_id))
    lints.extend(sink)
    plan["lints"] = _sorted_lints(lints, clip_id)
    return plan


# ---- project-level API -----------------------------------------------------------

_ENERGY: dict = {}


def _energy(project):
    from .audio import EnergyTrack
    src = project.source()
    key = (str(project.root), src.get("fingerprint"), (src.get("audio") or {}).get("start"))
    if key not in _ENERGY:
        _ENERGY.clear()
        _ENERGY[key] = EnergyTrack.for_project(project)
    return _ENERGY[key]


def _media_path(project, clip) -> Path:
    p = Path(clip["source_file"]).expanduser()
    return p if p.is_absolute() else Path(project.root) / p


def _shots(project, clip_id):
    p = Path(project.root) / "shots" / f"{clip_id}.json"
    if not p.exists():
        return None
    try:
        data = env.read_json(p)
    except env.HCError:
        return None
    if isinstance(data, dict):
        data["_hash"] = env.sha256_file(p)
    return data


def _unknown_clip(clip_id, spec) -> env.HCError:
    fn = getattr(env, "unknown_clip", None)
    if fn is not None:
        return fn(clip_id, list(spec["clips"]))
    return env.HCError(f"unknown clip {clip_id!r} (clips.json has: "
                       f"{', '.join(spec['clips']) or 'none'})", 1)


def _check_id(spec, clip_id):
    if clip_id not in spec["clips"]:
        raise _unknown_clip(clip_id, spec)
    if not isinstance(clip_id, str) or not specmod.CLIP_ID_RE.match(clip_id):
        raise env.HCError(f"clip id {clip_id!r} is invalid: use lowercase letters, digits and "
                          "hyphens", 1)


def _require_transcript_source(project, transcript: dict | None = None) -> dict:
    """The project transcript, after checking it was made from the source now in the project.

    `hc ingest P other.mp4 --force` replaces the source but leaves transcript.json: its words,
    times and indices belong to the other file, and nothing else stops a plan that burns them
    over the new one. Both files record the source's fingerprint; when they differ this raises
    (code 1). One case is fine: `hc ingest --cfr` re-encodes the picture and copies the audio
    bit for bit, so a transcript made before it stays valid (source.json records the earlier
    fingerprint as ingest.cfr.from_fingerprint; an older record without that key is trusted).
    It is raised live, not baked into the plan as a lint: a transcript re-made from the same
    audio has the same sha256, so a cached FAIL would never clear."""
    project = _as_project(project)
    tr = transcript if transcript is not None else project.transcript()
    src = project.source()
    t_fp, s_fp = tr.get("source_fingerprint"), src.get("fingerprint")
    if not t_fp or not s_fp or t_fp == s_fp:
        return tr
    rec = (src.get("ingest") or {}).get("cfr")
    if isinstance(rec, dict) and ("from_fingerprint" not in rec
                                  or rec.get("from_fingerprint") == t_fp):
        return tr
    raise env.HCError("transcript.json was made from another source file than the one in the "
                      "project now (its source_fingerprint is not source.json's fingerprint): "
                      "its words and times belong to that other file. Run `hc transcribe P`, "
                      "check the word indices in clips.json, then `hc plan P`", 1)


def _require_clip_transcript_source(clip_id: str, clip: dict, transcript: dict | None,
                                    fingerprint: str | None) -> None:
    """The same check for a caption_only clip: transcripts/<clip>.json against the finished
    video the clip names (it has no ingest step, so nothing else would notice a re-export)."""
    t_fp = (transcript or {}).get("source_fingerprint")
    if t_fp and fingerprint and t_fp != fingerprint:
        f = clip.get("source_file")
        raise env.HCError(f"transcripts/{clip_id}.json was made from another version of {f}: "
                          f"its words and times belong to that version. Run `hc transcribe P "
                          f"--file {f} --clip {clip_id}`, then `hc plan P`", 1)


def transcript_changed(plan: dict, transcript: dict | None) -> bool:
    """True when `plan` no longer matches `transcript`: another sha256 (word texts or times),
    or another speaker label on a word the plan uses. The sha does not cover labels (it pins
    word indices and must not move when a label is corrected), so the labels are compared with
    the ones the plan itself recorded: every label a plan uses (words[].sp, pieces[].speakers,
    the speaker turns framing follows) comes from those words. Inserted words (i None) have no
    transcript word. For a caption_only plan pass the clip's own transcript."""
    tr = transcript or {}
    if (plan or {}).get("transcript_sha") != tr.get("sha256"):
        return True
    words = tr.get("words") or []
    for w in (plan or {}).get("words") or []:
        i = w.get("i")
        if isinstance(i, int) and not isinstance(i, bool):
            if not 0 <= i < len(words) or w.get("sp") != words[i].get("sp"):
                return True
    return False


def build_plan(project, clip_id: str, *, spec: dict | None = None, write: bool = True) -> dict:
    """Build (and by default write) plans/<clip>.plan.json."""
    project = _as_project(project)
    spec = spec if spec is not None else specmod.load_spec(project)
    _check_id(spec, clip_id)
    clip = spec["clips"][clip_id]
    if clip["kind"] == "caption_only":
        if not clip.get("source_file"):
            raise env.HCError(f"caption_only clip {clip_id} needs \"source_file\"", 1)
        from . import probe
        path = _media_path(project, clip)
        if not path.exists():
            raise env.HCError(f"caption_only clip {clip_id}: {path} not found", 1)
        tr = project.transcript(clip_id) if project.has_transcript(clip_id) else None
        fp = env.fingerprint(path)
        _require_clip_transcript_source(clip_id, clip, tr, fp)
        plan = plan_caption_only(spec, clip_id, tr, probe.probe_media(path), fp)
    else:
        tr = _require_transcript_source(project)
        plan = plan_clip(spec, clip_id, tr, project.source(), _energy(project),
                         _shots(project, clip_id))
    extra = _lint_hook_fit(project, clip, clip_id)
    if extra:
        plan["lints"] = _sorted_lints(list(plan["lints"]) + extra, clip_id)
    if write:
        path = project.plan_path(clip_id)
        # An identical rebuild leaves the file (and its mtime) alone, so `hc status` does not
        # call renders made from this very plan stale.
        same = False
        if path.exists():
            try:
                same = env.stable_hash(env.read_json(path)) == env.stable_hash(plan)
            except env.HCError:
                same = False
        if not same:
            env.write_json(path, plan)
    return plan


def _lint_hook_fit(project, clip: dict, clip_id: str) -> list:
    """L20: the hook card sets less than HOOK_RATIO_MIN x the caption size in a vertical style
    (a card smaller than the captions reads as a caption, not a headline)."""
    if not clip.get("hook_card"):
        return []
    eff = clip.get("_eff") or {}
    if "vertical" not in (eff.get("formats") or ["vertical"]):
        return []
    try:
        from . import captions
        from . import render_util as ru
    except ImportError:
        return []
    fit_fn = getattr(captions, "hook_fit", None)
    if fit_fn is None:
        return []
    out = []
    for style in eff.get("styles") or ["clean"]:
        try:
            preset = ru.load_preset(style, project, "vertical")
            fit = fit_fn(clip, preset, "vertical", project.root)
        except env.HCError:
            continue
        if not fit or not fit.get("ratio") or fit["ratio"] >= HOOK_RATIO_MIN:
            continue
        out.append(specmod.lint(
            "L20", "WARN", f"hook card sets at {fit.get('hook_px', 0):.0f} px against "
            f"{fit.get('caption_px', 0):.0f} px captions in {style}: shorten lines to <= "
            f"{fit.get('max_chars', '?')} characters", clip_id))
    return out


def shots_stale(project, clip_id: str, plan: dict) -> bool:
    """True when `plan` was made from other shots data than shots/<clip>.json holds now (the file
    appeared, changed or went away since). Framing tracks and lint L11 follow that file, so the
    plan and every render made from it are stale (`hc status`: STALE(shots))."""
    project = _as_project(project)
    sp = Path(project.root) / "shots" / f"{clip_id}.json"
    return (plan or {}).get("shots_hash") != (env.sha256_file(sp) if sp.exists() else None)


def _stale(project, spec, clip_id, plan, transcript: dict | None = None) -> bool:
    """True when the plan file was made from other inputs than the project holds now.
    `transcript`: the project transcript when the caller has already read it (cut clips)."""
    clip = spec["clips"][clip_id]
    if plan.get("version") != 1 or plan.get("code_version") != _code_version():
        return True
    if plan.get("spec_hash") != specmod.clip_hash(spec, clip_id):
        return True
    if plan.get("set_hash") != _set_hash(spec):
        return True
    if clip["kind"] == "caption_only":
        path = _media_path(project, clip)
        if not path.exists():
            return True
        fp = env.fingerprint(path)
        if plan.get("source_fingerprint") != fp:
            return True
        tp = Path(project.transcript_path(clip_id))
        tr = env.read_json(tp) if tp.exists() else None
        if (tr or {}).get("source_fingerprint") not in (None, "", fp):
            return True                  # made from another version of the file: the rebuild says so
        return transcript_changed(plan, tr)
    src = project.source()
    if plan.get("source_fingerprint") != src.get("fingerprint"):
        return True
    if (plan.get("source") or {}).get("active_area") != (src.get("video") or {}).get("active_area"):
        return True                      # re-ingest found (or dropped) letterbox bars
    tr = transcript if transcript is not None else project.transcript()
    if transcript_changed(plan, tr):
        return True
    # The top-level transcript_sha pin is not in the clip hash (pinning must not re-plan every
    # clip), so the plan follows its verdict instead: stale when lint L03 would now appear or
    # go away.
    tsha = spec.get("transcript_sha")
    l03 = bool(tsha and tr.get("sha256") and not specmod.sha_matches(tsha, tr))
    if l03 != any(x.get("id") == "L03" for x in plan.get("lints") or []):
        return True
    return shots_stale(project, clip_id, plan)


def load_plan(project, clip_id: str, require_ok: bool = False) -> dict:
    """The clip's plan, rebuilt when stale (spec, transcript, source, shots or code changed).
    With require_ok, FAIL lints raise HCError (code 1)."""
    project = _as_project(project)
    spec = specmod.load_spec(project)
    _check_id(spec, clip_id)
    path = Path(project.root) / "plans" / f"{clip_id}.plan.json"
    tr = _require_transcript_source(project) if spec["clips"][clip_id]["kind"] == "cut" else None
    plan = None
    if path.exists():
        try:
            plan = env.read_json(path)
        except env.HCError:
            plan = None
    if plan is None or _stale(project, spec, clip_id, plan, tr):
        plan = build_plan(project, clip_id, spec=spec)
    if require_ok:
        fails = fail_lints(plan)
        if fails:
            raise env.HCError(f"plan for {clip_id} has FAIL lints (run `plan` to see them): " +
                              "; ".join(f"{x['id']} {x['msg']}" for x in fails[:5]), 1)
    return plan


def current_plan(project, clip_id: str) -> dict:
    """The plan a render made now would use: the plan file when fresh, else a rebuild that is
    NOT written (QA and freshness checks never rewrite plans)."""
    project = _as_project(project)
    spec = specmod.load_spec(project)
    _check_id(spec, clip_id)
    path = Path(project.root) / "plans" / f"{clip_id}.plan.json"
    tr = _require_transcript_source(project) if spec["clips"][clip_id]["kind"] == "cut" else None
    plan = None
    if path.exists():
        try:
            plan = env.read_json(path)
        except env.HCError:
            plan = None
    if plan is None or _stale(project, spec, clip_id, plan, tr):
        plan = build_plan(project, clip_id, spec=spec, write=False)
    return plan


def fail_lints(plan: dict) -> list:
    return [x for x in plan.get("lints") or [] if x.get("level") == "FAIL"]


# ---- time maps -----------------------------------------------------------------

def out_time(plan: dict, t_src: float):
    """Output time of source time t_src (first segment containing it), or None if cut."""
    for s in plan["segments"]:
        a, b = s["src"]
        if a - EPS <= t_src < b:
            return s["out"][0] + (t_src - a)
    for s in plan["segments"]:
        if abs(t_src - s["src"][1]) <= EPS:
            return s["out"][1]
    return None


def src_time(plan: dict, t_out: float):
    """(segment index, source time) for output time t_out (clamped to the clip)."""
    segs = plan["segments"]
    if not segs:
        raise env.HCError(f"plan for {plan.get('clip')} has no segments", 1)
    t = min(max(float(t_out), 0.0), segs[-1]["out"][1])
    for s in segs:
        if s["out"][0] - EPS <= t < s["out"][1]:
            return s["k"], s["src"][0] + (t - s["out"][0])
    last = segs[-1]
    return last["k"], last["src"][1]


# ---- printing / command ------------------------------------------------------------

def _ft(t) -> str:
    return specmod.fmt_time(t)


def print_plan(plan: dict, file=None) -> None:
    """Human-readable plan table + lints (stdout)."""
    import sys
    f = file or sys.stdout
    fps = plan["fps"]
    rate = f"{fps[0]}/{fps[1]}" if fps[1] != 1 else str(fps[0])
    cr = plan["cut_rate"]
    tg = plan["tighten"]
    print(f"{plan['clip']}  {plan['duration_out']:.2f} s  {plan['frames_out']} frames @ {rate} fps  "
          f"{cr['joins']} joins ({cr['per_sec']:.2f}/s)  tighten -{tg['removed_s']:.2f} s "
          f"({tg['count']})", file=f)
    print(f"  {'#':>2}  {'role':<9} {'words':<13} {'src':<21} {'out':<17} {'frames':>6}  edges",
          file=f)
    for p in plan["pieces"]:
        wr = f"{p['words'][0]}-{p['words'][1]}" if p.get("words") else "time"
        if p.get("extra_words"):
            wr += f"+{len(p['extra_words'])}"
        src = f"{_ft(p['src'][0])}-{_ft(p['src'][1])}"
        out = f"{_ft(p['out'][0])}-{_ft(p['out'][1])}" if p.get("out") else "-"
        edges = ""
        if p.get("edge_in"):
            ei, eo = p["edge_in"], p["edge_out"]
            edges = (f"in {ei['db']:.0f} dB{' TIGHT' if ei['tight'] else ''}"
                     f"{' (' + ei['guard'] + ')' if ei.get('guard') else ''} / "
                     f"out {eo['db']:.0f} dB{' TIGHT' if eo['tight'] else ''}"
                     f"{' (' + eo['guard'] + ')' if eo.get('guard') else ''}")
        print(f"  {p['n']:>2}  {p['role']:<9} {wr:<13} {src:<21} {out:<17} {p['frames']:>6}  {edges}",
              file=f)
        if p.get("hold"):
            print("      " + hold_line(p), file=f)
    if plan["joins"]:
        js = ", ".join(f"{j['out']:.2f} {j['kind']}{'!' if j['tight'] else ''}"
                       for j in plan["joins"][:24])
        more = f" (+{len(plan['joins']) - 24} more)" if len(plan["joins"]) > 24 else ""
        print(f"  joins: {js}{more}", file=f)
    if plan.get("hook_card"):
        print(f"  hook card: {' / '.join(plan['hook_card']['lines'])} ({plan['hook_card']['dur']} s)",
              file=f)
    for m in plan.get("inserts") or []:
        ws = [w for w in plan["words"] if w.get("i") is None and w.get("ins") == m["n"]]
        out = f" -> out {_ft(ws[0]['s_out'])}-{_ft(ws[-1]['e_out'])}" if ws else ""
        print(f"  insert {m['n']}: after word {m['after']} \"{' '.join(m['tokens'])}\" "
              f"src {_ft(m['src'][0])}-{_ft(m['src'][1])}{out} (captioned, not in the transcript)",
              file=f)
    lints = plan.get("lints") or []
    if lints:
        print("  lints:", file=f)
        for x in lints:
            at = f" @{_ft(x['at_out'])}" if x.get("at_out") is not None else ""
            print(f"    {x['level']:<4} {x['id']}{at} {x['msg']}", file=f)
    else:
        print("  lints: none", file=f)


HOLD_WHY = {
    "quietest": "the quietest frame near the target",
    "after_sound": "the first frame after the sound that runs past the target",
    "clamped": "stopped where the next piece of the clip starts",
    "omit": "stopped before a word the clip omits",
    "forced": "no frame in reach of the target, the nearest one",
}


def hold_line(piece: dict) -> str:
    """Plan-table line for a piece with hold_after: the time asked for, the grid frame the
    out-edge landed on and why. The edge snaps to the frame grid at the quietest moment near
    the target, so hold values a few hundredths apart land on the same frame; out_offset is
    applied after this snap and moves the edge one frame at a time."""
    h = piece["hold"]
    line = (f"hold_after {h['asked']:g} s: target {_ft(h['target_src'])}, out-edge on frame "
            f"{h['frame']} ({_ft(h['frame_src'])}, {h['got']:.2f} s after the last word): "
            f"{HOLD_WHY.get(h.get('why'), h.get('why'))}")
    if h.get("frame_out") is not None and h["frame_out"] != h["frame"]:
        line += (f"; out_offset {piece.get('out_offset', 0):+d} then put it on frame "
                 f"{h['frame_out']} ({_ft(h.get('src'))})")
    else:
        line += "; nearby hold values land on this same frame, out_offset moves it frame by frame"
    return line


def cmd_plan(args) -> int:
    """`plan P [CLIP...] [--quiet]`: build plans, print tables, exit 1 on FAIL lints."""
    project = _as_project(args.project)
    if hasattr(project, "require"):
        project.require()
    spec = specmod.load_spec(project)
    ids = list(getattr(args, "clips", None) or []) or list(spec["clips"])
    if not ids:
        raise env.HCError("clips.json has no clips", 1)
    for cid in ids:
        if cid not in spec["clips"]:
            raise _unknown_clip(cid, spec)
    worst = 0
    for cid in ids:
        plan = build_plan(project, cid, spec=spec)
        fails = fail_lints(plan)
        if getattr(args, "quiet", False):
            nw = sum(1 for x in plan["lints"] if x["level"] == "WARN")
            print(f"{cid}  {plan['duration_out']:.2f} s  {plan['cut_rate']['joins']} joins  "
                  f"{len(fails)} FAIL  {nw} WARN")
            for x in fails:
                print(f"    FAIL {x['id']} {x['msg']}")
        else:
            print_plan(plan)
            print()
        if fails:
            worst = 1
    return worst
