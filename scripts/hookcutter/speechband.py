"""Speech-likeness of a render's audio, for the QA gates that judge sound (G6, G10).

Broadband energy cannot tell a word from a laugh, a lip click or a mic thump: all of them
are "above the floor". This module looks at the shape of the sound in 10 ms steps:

    band_db    energy in the speech band (300 to 3400 Hz)
    full_db    energy of the whole frame
    nccf       periodicity: the peak of the normalized autocorrelation over pitch lags
               (60 to 400 Hz); a vowel is near 1, noise and breath are near 0.2
    flat       spectral flatness of 300 to 4000 Hz; a formant spectrum is near 0, noise near 0.3
    centroid   spectral centre of gravity in Hz; a sibilant sits above 2.5 kHz

A frame is speech-like when the speech band is at least LOUD_DB over the floor AND the sound
is either periodic (nccf >= PERIODIC) or has a peaked, formant-like spectrum (flat <= PEAKY).
A speech-like RUN lasts at least SUSTAIN_S (dips of BRIDGE_S or less are bridged), so one
click is never an onset. The frication of a word-initial consonant ("s", "sh", "ch", the
burst of "t" or "k") directly before a run belongs to that word (`Track.onset_after`).

Limits, stated plainly: a breathy laugh, applause, a click and a thump are not speech-like.
A strongly voiced laugh ("ha ha" sung on a vowel) and music are: nothing in the signal
separates them from a spoken vowel, so a caption that lights early under such a sound is
not seen here.

Pure numpy; everything takes a mono float array at SR and returns plain values.
"""
from __future__ import annotations

import math

import numpy as np

SR = 16000
HOP_S = 0.01
WIN_S = 0.04
BAND = (300.0, 3400.0)           # speech band
FLAT_BAND = (300.0, 4000.0)      # flatness is measured here
PITCH = (60.0, 400.0)            # pitch lags searched for periodicity
LOUD_DB = 10.0                   # above the floor by this much = sound, not room tone
FLOOR_PCT = 10
FLOOR_MIN_DB = -70.0
PERIODIC = 0.6                   # nccf at least this = periodic (voiced)
PEAKY = 0.05                     # flatness at most this = a formant-like spectrum
SUSTAIN_S = 0.06                 # a speech-like run lasts at least this long
BRIDGE_S = 0.02                  # dips this short do not end a run
FRIC_HZ = 2000.0                 # loud, not speech-like and centred above this = frication
SIBILANT_HZ = 2500.0             # a sibilant ("s", "sh") sits above this
UNVOICED = 0.45                  # nccf below this = no pitch
ATTACK_S = 0.03                  # loud frames this close before a run are its attack
CONS_BACK_S = 0.25               # a word-initial consonant reaches at most this far back
CONS_BRIDGE_S = 0.08             # ... across a stop closure this long
CHUNK = 2000                     # frames analysed per block (bounds memory)


def _runs(mask: np.ndarray, minlen: int, bridge: int = 0) -> list[tuple[int, int]]:
    """[(start, end)) of True runs at least `minlen` long; gaps of at most `bridge` are bridged."""
    m = np.asarray(mask, dtype=bool)
    n = m.size
    if n == 0:
        return []
    d = np.diff(np.concatenate([[0], m.astype(np.int8), [0]]))
    starts = np.flatnonzero(d == 1).tolist()
    ends = np.flatnonzero(d == -1).tolist()
    merged: list[list[int]] = []
    for s, e in zip(starts, ends):
        if merged and s - merged[-1][1] <= bridge:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged if e - s >= minlen]


class Track:
    """Per-hop features of one audio file. Frame j is centred on time j * hop."""

    def __init__(self, band_db, full_db, nccf, flat, centroid, hop: float = HOP_S,
                 duration: float | None = None):
        self.band_db = np.asarray(band_db, dtype=np.float64)
        self.full_db = np.asarray(full_db, dtype=np.float64)
        self.nccf = np.asarray(nccf, dtype=np.float64)
        self.flat = np.asarray(flat, dtype=np.float64)
        self.centroid = np.asarray(centroid, dtype=np.float64)
        self.hop = float(hop)
        self.n = int(self.band_db.size)
        self.duration = float(duration) if duration is not None else self.n * self.hop
        if self.n:
            self.band_floor = max(FLOOR_MIN_DB, float(np.percentile(self.band_db, FLOOR_PCT)))
            self.full_floor = max(FLOOR_MIN_DB, float(np.percentile(self.full_db, FLOOR_PCT)))
        else:
            self.band_floor = self.full_floor = FLOOR_MIN_DB
        self.loud = self.full_db > self.full_floor + LOUD_DB
        self.speech = (self.band_db > self.band_floor + LOUD_DB) & (
            (self.nccf >= PERIODIC) | (self.flat <= PEAKY))
        self.fric = self.loud & ~self.speech & (self.centroid >= FRIC_HZ)
        self._runs: list[tuple[int, int, int]] | None = None

    # ---- frames and times ---------------------------------------------
    def index(self, t: float) -> int:
        return int(round(float(t) / self.hop))

    def time(self, j: int) -> float:
        return j * self.hop

    def _span(self, a: float, b: float) -> tuple[int, int]:
        return max(0, self.index(a)), min(self.n, max(self.index(a) + 1, self.index(b)))

    # ---- speech-like runs ---------------------------------------------
    def runs(self) -> list[tuple[int, int, int]]:
        """[(onset, voiced start, end)] frame indices of the speech-like runs. `onset` reaches
        back over the frication of a word-initial consonant, never into the previous run."""
        if self._runs is None:
            minlen = max(1, int(round(SUSTAIN_S / self.hop)))
            bridge = max(0, int(round(BRIDGE_S / self.hop)))
            back = int(round(CONS_BACK_S / self.hop))
            gap_max = int(round(CONS_BRIDGE_S / self.hop))
            attack = int(round(ATTACK_S / self.hop))
            out, prev_end = [], 0
            for s, e in _runs(self.speech, minlen, bridge):
                on = s
                # the attack: the frame centred on an onset is half silence, so it is loud
                # before it is periodic
                while on > prev_end and s - on < attack and self.loud[on - 1]:
                    on -= 1
                k, gap = on, 0
                while k > prev_end and s - k < back:
                    k -= 1
                    if self.fric[k]:
                        on, gap = k, 0
                    else:
                        gap += 1
                        if gap > gap_max:
                            break
                out.append((on, s, e))
                prev_end = e
            self._runs = out
        return self._runs

    def speech_runs(self) -> list[tuple[float, float]]:
        """[(start, end)] seconds of the speech-like runs (consonant onsets included)."""
        return [(self.time(on), self.time(e)) for on, _s, e in self.runs()]

    def onset_after(self, t: float, limit: float | None = None) -> float | None:
        """The first moment at or after `t` that is speech-like: `t` itself inside a run, else
        the next run's onset when it comes by `limit` (default: the end of the file)."""
        j = self.index(t)
        lim = self.n if limit is None else self.index(limit)
        for on, _s, e in self.runs():
            if e <= j:
                continue
            if on <= j:
                return float(t)
            return self.time(on) if on <= lim else None
        return None

    def frication_start(self, t: float, bridge: float = BRIDGE_S, back: float = 0.4) -> float:
        """Start of the unpitched frication that runs up to `t` (dips of `bridge` or less are
        bridged; at most `back` seconds); `t` when nothing of the kind sounds right before it."""
        j = min(self.n, self.index(t))
        sib = self.loud & (self.centroid >= SIBILANT_HZ) & (self.nccf < UNVOICED)
        gap_max = max(0, int(round(bridge / self.hop)))
        lim = max(0, j - int(round(back / self.hop)))
        first, k, gap = j, j, 0
        while k > lim:
            k -= 1
            if sib[k]:
                first, gap = k, 0
            else:
                gap += 1
                if gap > gap_max:
                    break
        return self.time(first)

    def speech_before(self, t: float, within: float) -> bool:
        """True when a speech-like frame lies in the `within` seconds before `t`."""
        j1 = min(self.n, self.index(t) + 1)
        j0 = max(0, self.index(t - within))
        return bool(j1 > j0 and np.any(self.speech[j0:j1]))

    def loud_seconds(self, a: float, b: float) -> float:
        """Seconds of [a, b] that hold sound (LOUD_DB over the floor), speech-like or not."""
        j0, j1 = self._span(a, b)
        return float(np.count_nonzero(self.loud[j0:j1])) * self.hop if j1 > j0 else 0.0

    # ---- what a short stretch is ---------------------------------------
    def describe(self, a: float, b: float) -> dict:
        """What sounds in [a, b]: {"kind", "frames", "loud", "voiced", "fricative", "centroid",
        "nccf", "level_db"}. kind is "quiet" (under half the frames hold sound), "voiced" (a
        pitched or formant sound: a vowel, a nasal), "fricative" (unpitched noise centred above
        SIBILANT_HZ: "s", "sh", the release of "t") or "other"."""
        j0, j1 = self._span(a, b)
        n = j1 - j0
        if n <= 0:
            return {"kind": "quiet", "frames": 0, "loud": 0.0, "voiced": 0.0, "fricative": 0.0,
                    "centroid": 0.0, "nccf": 0.0, "level_db": FLOOR_MIN_DB}
        loud = self.loud[j0:j1]
        k = int(np.count_nonzero(loud))
        if k == 0:
            cen, per = 0.0, 0.0
        else:
            cen = float(np.median(self.centroid[j0:j1][loud]))
            per = float(np.median(self.nccf[j0:j1][loud]))
        voiced = float(np.count_nonzero(self.speech[j0:j1] & loud)) / n
        sib = loud & (self.centroid[j0:j1] >= SIBILANT_HZ) & (self.nccf[j0:j1] < UNVOICED)
        fric = float(np.count_nonzero(sib)) / n
        if k < 0.5 * n:
            kind = "quiet"
        elif fric >= 0.6:
            kind = "fricative"
        elif voiced >= 0.4:
            kind = "voiced"
        else:
            kind = "other"
        return {"kind": kind, "frames": n, "loud": k / n, "voiced": voiced, "fricative": fric,
                "centroid": cen, "nccf": per,
                "level_db": float(np.max(self.full_db[j0:j1]))}


def analyse(x, sr: int = SR, hop: float = HOP_S, win: float = WIN_S) -> Track:
    """Feature track of mono audio `x` at `sr` (an empty track for no audio)."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    h = max(1, int(round(sr * hop)))
    n = max(8, int(round(sr * win)))
    if x.size == 0:
        z = np.zeros(0)
        return Track(z, z, z, z, z, hop=h / sr, duration=0.0)
    k = x.size // h + 1
    pad = n // 2
    xp = np.concatenate([np.zeros(pad), x, np.zeros(n + pad)])
    win_fn = np.hanning(n)
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    m_band = (freqs >= BAND[0]) & (freqs < BAND[1])
    m_flat = (freqs >= FLAT_BAND[0]) & (freqs < FLAT_BAND[1])
    lo = max(1, int(sr / PITCH[1]))
    hi = min(n - 8, int(math.ceil(sr / PITCH[0])))
    lags = np.arange(lo, hi + 1)
    band_db = np.empty(k)
    full_db = np.empty(k)
    nccf = np.zeros(k)
    flat = np.ones(k)
    cen = np.zeros(k)
    base = np.arange(n)[None, :]
    for c0 in range(0, k, CHUNK):
        c1 = min(k, c0 + CHUNK)
        F = xp[base + (np.arange(c0, c1) * h)[:, None]]
        power = np.mean(F * F, axis=1)
        full = 10.0 * np.log10(np.maximum(power, 1e-12))
        S = np.abs(np.fft.rfft(F * win_fn, axis=1)) ** 2 + 1e-18
        tot = S.sum(axis=1)
        full_db[c0:c1] = full
        band_db[c0:c1] = np.maximum(-120.0, full + 10.0 * np.log10(S[:, m_band].sum(axis=1) / tot))
        cen[c0:c1] = (S * freqs[None, :]).sum(axis=1) / tot
        fb = S[:, m_flat]
        flat[c0:c1] = np.exp(np.mean(np.log(fb), axis=1)) / np.mean(fb, axis=1)
        G = F - F.mean(axis=1, keepdims=True)
        r = np.fft.irfft(np.abs(np.fft.rfft(G, 2 * n, axis=1)) ** 2, axis=1)[:, lags]
        cs = np.concatenate([np.zeros((G.shape[0], 1)), np.cumsum(G * G, axis=1)], axis=1)
        head = cs[:, n - lags]                       # energy of g[0 : n - lag]
        tail = cs[:, [n]] - cs[:, lags]              # energy of g[lag : n]
        den = np.sqrt(head * tail)
        ok = den > 1e-10
        val = np.where(ok, r / np.where(ok, den, 1.0), 0.0)
        peak = val.max(axis=1)
        nccf[c0:c1] = np.where(power > 1e-10, np.clip(peak, 0.0, 1.0), 0.0)
    return Track(band_db, full_db, nccf, flat, cen, hop=h / sr, duration=x.size / sr)
