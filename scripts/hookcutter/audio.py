"""Audio energy track and loudness measurement.

The planner never trusts word timestamps alone: Whisper-family word edges are
off by roughly 50 to 150 ms. This module gives it the actual signal: a 10 ms
RMS energy track of the source (16 kHz, power-averaged over two channels, on
the `src` clock), a local noise floor, voiced onset/offset search, a
silence-aware true word end and the quiet-gap test used by tighten.
"""
from __future__ import annotations

import json
import math
import os
import re
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from . import env, fsutil

SR = 16000                 # analysis sample rate
HOP_S = 0.01               # 10 ms hops
HOP_N = int(SR * HOP_S)    # samples per hop
CHANNELS = 2               # decode to stereo, power-average (no phase cancellation)
SILENCE_DB = -120.0        # value used for "no audio here" (outside the stream)
FLOOR_MIN_DB = -80.0       # floors below this (digital silence) are clamped
FLOOR_RADIUS = 20.0        # seconds either side of t for the floor percentile
FLOOR_PCT = 10             # percentile used as the noise floor
VOICED_DB = 10.0           # voiced = db > floor + VOICED_DB
VOICED_MIN_S = 0.03        # a voiced run must last this long
SILENT_DB = 6.0            # true_end silence = db < floor + SILENT_DB
SILENT_MIN_S = 0.15        # ... for at least this long
QUIET_DB = 8.0             # quiet gap = p75(db in gap) <= floor + QUIET_DB
QUIET_PCT = 75
BRIDGE_S = 0.02            # unvoiced dips up to this long do not end a voiced run
ENERGY_VERSION = 2         # 2: decoded on the gap-filled clock (aresample async=1)
LOUDNORM_TARGET = (-14.0, -1.5, 11.0)   # I, TP, LRA


def _win(t, hop):
    return int(math.floor(t / hop + 1e-9))


class EnergyTrack:
    """RMS energy in dB per 10 ms hop. Frame j covers src [t0 + j*hop, t0 + (j+1)*hop)."""

    def __init__(self, db, hop: float = HOP_S, t0: float = 0.0):
        self.db = np.asarray(db, dtype=np.float64).reshape(-1)
        self.hop = float(hop)
        self.t0 = float(t0)
        self._pw = np.power(10.0, self.db / 10.0)
        self._cum = np.concatenate([[0.0], np.cumsum(self._pw)])
        self._floor_cache: dict = {}

    # ---- construction -------------------------------------------------
    @classmethod
    def from_media(cls, path, *, t0: float = 0.0, stream=None, cache_dir=None,
                   key: str | None = None) -> "EnergyTrack":
        """Decode `path` (audio stream `stream`, absolute index; default first audio)
        and build the track. `t0` = src time of the stream's first sample.
        With `cache_dir` + `key` the decoded dB array is cached as
        `energy-<key>.npy` + `.json` meta."""
        path = Path(path)
        meta = {"version": ENERGY_VERSION, "key": key, "hop": HOP_S, "sr": SR,
                "channels": CHANNELS, "stream": stream}
        if cache_dir is None or not key:
            return cls(decode_energy(path, stream=stream), hop=HOP_S, t0=t0)
        cache_dir = Path(cache_dir)
        npy = cache_dir / f"energy-{key}.npy"
        meta_p = cache_dir / f"energy-{key}.json"

        def cached():
            if npy.exists() and meta_p.exists():
                try:
                    old = json.loads(meta_p.read_text())
                    if all(old.get(k) == v for k, v in meta.items()):
                        return np.load(npy)
                except (OSError, ValueError):
                    pass
            return None

        db = cached()
        if db is not None:
            return cls(db, hop=HOP_S, t0=t0)
        # One decode per source at a time: a second process waits, then reads the cache.
        with fsutil.file_lock(cache_dir / f"energy-{key}.lock", what="the audio energy decode"):
            db = cached()
            if db is None:
                db = decode_energy(path, stream=stream)
                tmp = fsutil.tmp_path(npy)
                try:
                    with open(tmp, "wb") as fh:
                        np.save(fh, db.astype(np.float32))
                    os.replace(tmp, npy)
                finally:
                    if tmp.exists():
                        tmp.unlink()
                meta["frames"] = int(len(db))
                env.write_json(meta_p, meta)
        return cls(db, hop=HOP_S, t0=t0)

    @classmethod
    def for_project(cls, project) -> "EnergyTrack":
        """Cached track for the project's source (src clock: audio.start applied)."""
        src = project.source()
        a = src.get("audio") or {}
        if not a:
            raise env.HCError("source has no audio stream: cannot snap cuts", 1)
        fp = src.get("fingerprint") or env.fingerprint(project.source_path())
        cache_dir = Path(project.cache_path(f"energy-{fp}.npy")).parent
        return cls.from_media(project.source_path(), t0=float(a.get("start") or 0.0),
                              stream=a.get("index"), cache_dir=cache_dir, key=fp)

    # ---- basic access -------------------------------------------------
    @property
    def n(self) -> int:
        return len(self.db)

    @property
    def end(self) -> float:
        return self.t0 + self.n * self.hop

    def index(self, t: float) -> int:
        """Frame index containing src time t (may be out of range)."""
        return _win(t - self.t0, self.hop)

    def time(self, j: int) -> float:
        """Start time of frame j."""
        return self.t0 + j * self.hop

    def _db_j(self, j: int) -> float:
        return float(self.db[j]) if 0 <= j < self.n else SILENCE_DB

    def db_at(self, t: float, win: float = 0.02) -> float:
        """Power-mean dB over [t - win/2, t + win/2] (at least the frame containing t)."""
        lo = t - win / 2 - self.t0
        hi = t + win / 2 - self.t0
        j0 = _win(lo, self.hop)
        j1 = max(j0 + 1, int(math.ceil(hi / self.hop - 1e-9)))
        n = j1 - j0
        a, b = max(j0, 0), min(j1, self.n)
        s = float(self._cum[b] - self._cum[a]) if b > a else 0.0
        s += (n - max(0, b - a)) * 10.0 ** (SILENCE_DB / 10.0)
        return 10.0 * math.log10(max(s / n, 1e-30))

    def floor(self, t: float, radius: float = FLOOR_RADIUS) -> float:
        """Noise floor near t: 10th percentile dB within +/- radius (clamped >= FLOOR_MIN_DB)."""
        key = (round(t * 2.0) / 2.0, float(radius))
        hit = self._floor_cache.get(key)
        if hit is not None:
            return hit
        tc = key[0]
        a = max(0, self.index(tc - radius))
        b = min(self.n, self.index(tc + radius) + 1)
        if b <= a:
            val = FLOOR_MIN_DB
        else:
            val = max(float(np.percentile(self.db[a:b], FLOOR_PCT)), FLOOR_MIN_DB)
        self._floor_cache[key] = val
        return val

    def _mask(self, j0: int, j1: int, thr: float, above: bool = True) -> np.ndarray:
        """Boolean mask for frames j0..j1-1 (out-of-range frames = SILENCE_DB)."""
        out = np.full(max(0, j1 - j0), SILENCE_DB)
        a, b = max(j0, 0), min(j1, self.n)
        if b > a:
            out[a - j0:b - j0] = self.db[a:b]
        return out > thr if above else out < thr

    # ---- voicing ------------------------------------------------------
    def voiced(self, t: float, fl: float | None = None) -> bool:
        if fl is None:
            fl = self.floor(t)
        return self._db_j(self.index(t)) > fl + VOICED_DB

    def onset(self, s: float, lo: float, after: float = 0.10, fl: float | None = None):
        """Voiced onset for a word whisper says starts at `s`.

        Returns the start of the voiced run (>= 30 ms above floor + 10 dB) that
        contains `s`, or else the first such run starting in (s, s + after].
        A run that already started at or before `lo` is continuous speech, not an
        onset: returns None (no guard possible). Also None when nothing voiced."""
        if fl is None:
            fl = self.floor(s)
        thr = fl + VOICED_DB
        minf = max(1, int(round(VOICED_MIN_S / self.hop)))
        j0 = self.index(lo)
        j1 = self.index(s + after) + minf + 1
        m = self._mask(j0, j1, thr)
        js = self.index(s) - j0
        runs = _runs(m, minf)
        for rs, re_ in runs:
            if rs <= js < re_:
                if rs == 0:
                    return None
                return self.time(j0 + rs)
        for rs, _re in runs:
            if rs > js and self.time(j0 + rs) <= s + after + 1e-9:
                return self.time(j0 + rs)
        return None

    def voiced_end(self, t_from: float, t_limit: float, fl: float | None = None) -> float:
        """End of the voiced run active at `t_from` (bridging dips <= 20 ms), capped at t_limit.
        Returns t_from when t_from is not voiced."""
        if fl is None:
            fl = self.floor(t_from)
        thr = fl + VOICED_DB
        j = self.index(t_from)
        jl = self.index(t_limit)
        if jl < j or not self._db_j(j) > thr:
            return t_from
        bridge = max(1, int(round(BRIDGE_S / self.hop)))
        m = self._mask(j, jl + 1, thr)
        last = 0
        gap = 0
        for k in range(1, len(m)):
            if m[k]:
                last, gap = k, 0
            else:
                gap += 1
                if gap > bridge:
                    break
        return min(self.time(j + last + 1), t_limit)

    def true_end(self, w, e: float | None = None) -> float:
        """Silence-aware word end. `w` is a word dict (s, e) or a start time with `e`.

        Whisper stretches word ends through pauses. If a silence (>= 0.15 s below
        floor + 6 dB) starts strictly inside (s + 0.05, e), the word really ended
        there; otherwise the end is e."""
        if isinstance(w, dict):
            s, e = float(w["s"]), float(w["e"])
        else:
            s, e = float(w), float(e)
        if e - s <= 0.05:
            return e
        fl = self.floor((s + e) / 2)
        thr = fl + SILENT_DB
        minf = max(1, int(round(SILENT_MIN_S / self.hop)))
        j0 = self.index(s + 0.05) + 1          # first frame starting after s + 0.05
        j1 = self.index(e - 1e-9)              # last frame starting before e
        if j1 < j0:
            return e
        m = self._mask(j0, j1 + minf, thr, above=False).astype(np.int64)
        # windows of minf frames that are all silent, starting at frames j0..j1
        full = np.convolve(m, np.ones(minf, dtype=np.int64), mode="valid")[:j1 - j0 + 1]
        for k in np.flatnonzero(full >= minf):
            t = self.time(j0 + int(k))
            if s + 0.05 < t < e:
                return t
        return e

    def quiet_gap(self, a: float, b: float, exclude=()) -> bool:
        """True when the span [a, b) is quiet: p75 dB <= floor + 8 dB.

        Laughter, applause and music fail this test, so tighten never squashes
        them. Frames inside `exclude` spans (e.g. filler words) are ignored."""
        if b <= a:
            return True
        j0, j1 = self.index(a), max(self.index(a) + 1, self.index(b))
        vals = np.full(j1 - j0, SILENCE_DB)
        lo, hi = max(j0, 0), min(j1, self.n)
        if hi > lo:
            vals[lo - j0:hi - j0] = self.db[lo:hi]
        if exclude:
            mid = self.t0 + (np.arange(j0, j1) + 0.5) * self.hop
            keep = np.ones(len(vals), dtype=bool)
            for x, y in exclude:
                keep &= ~((mid >= x) & (mid < y))
            vals = vals[keep]
        if not len(vals):
            return True
        return float(np.percentile(vals, QUIET_PCT)) <= self.floor((a + b) / 2) + QUIET_DB


def _runs(mask: np.ndarray, minlen: int):
    """[(start, end)] of True runs with length >= minlen (end exclusive)."""
    out, start = [], None
    for k, v in enumerate(mask):
        if v and start is None:
            start = k
        elif not v and start is not None:
            if k - start >= minlen:
                out.append((start, k))
            start = None
    if start is not None and len(mask) - start >= minlen:
        out.append((start, len(mask)))
    return out


def decode_energy(path, stream=None) -> np.ndarray:
    """Stream-decode `path` to 16 kHz stereo float and return per-hop RMS dB.

    Decoded on the same clock as the transcript wav and the render: `aresample async=1` fills
    audio timestamp holes (an encoder hiccup stored as one over-long packet) with silence, so a
    hole does not shift every later frame of the track earlier than the words and the cuts."""
    cmd = [env.ffmpeg(), "-hide_banner", "-nostdin", "-loglevel", "error", "-i", str(path),
           "-map", f"0:{stream}" if stream is not None else "0:a:0",
           "-vn", "-sn", "-dn", "-af", f"aresample={SR}:async=1", "-ac", str(CHANNELS),
           "-f", "f32le", "-"]
    if os.environ.get("HC_VERBOSE"):
        env.log("audio", " ".join(cmd))
    frame_bytes = HOP_N * CHANNELS * 4
    chunk = frame_bytes * 2000                         # 20 s per read
    out, rest = [], b""
    # stderr goes to a file, never a pipe: nobody reads it while stdout is being read, so a
    # damaged stretch (one error line per undecodable frame) would fill a 64 KiB pipe, ffmpeg
    # would block writing to it and this loop would wait on stdout forever.
    with tempfile.TemporaryFile() as errf:
        proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                stderr=errf)
        try:
            while True:
                buf = proc.stdout.read(chunk)
                if not buf:
                    break
                buf = rest + buf
                n = len(buf) // frame_bytes
                rest = buf[n * frame_bytes:]
                if n:
                    x = np.frombuffer(buf[:n * frame_bytes], dtype="<f4").astype(np.float64)
                    out.append(_hop_db(x.reshape(n, HOP_N * CHANNELS)))
        finally:
            proc.stdout.close()
            rc = proc.wait()
            errf.seek(0)
            err = errf.read().decode("utf-8", "replace")
    if rc == 0 and err.strip():
        lines = [ln for ln in err.splitlines() if ln.strip()]
        env.log("audio", f"warning: ffmpeg reported {len(lines)} error line(s) while decoding "
                f"the audio of {Path(path).name}; the audio may be damaged there (last: "
                f"{lines[-1].strip()[:160]})")
    if rc != 0:
        raise env.HCError(f"ffmpeg could not decode audio from {path}: {err.strip()[-400:]}", 3)
    whole = len(rest) // (CHANNELS * 4)
    if whole:
        x = np.frombuffer(rest[:whole * CHANNELS * 4], dtype="<f4").astype(np.float64)
        out.append(_hop_db(x.reshape(1, -1)))
    if not out:
        raise env.HCError(f"no audio decoded from {path}", 3)
    # float32 so a fresh decode and a cached .npy give bit-identical tracks (deterministic plans)
    return np.concatenate(out).astype(np.float32)


def _hop_db(x: np.ndarray) -> np.ndarray:
    ms = np.mean(x * x, axis=1)
    return 10.0 * np.log10(np.maximum(ms, 10.0 ** (SILENCE_DB / 10.0)))


# ---- loudness ---------------------------------------------------------------

def loudnorm_measure(input_args, filter_prefix: str | None = None, *,
                     target=LOUDNORM_TARGET, allow=()) -> dict:
    """Measure loudness with ffmpeg loudnorm (first pass).

    `input_args`: ffmpeg input arguments, e.g. ["-i", "a.mkv", "-i", "b.mkv"].
    `filter_prefix`: optional filtergraph producing ONE audio stream, e.g.
    "[0:a][1:a]concat=n=2:v=0:a=1" (an unlabeled or labeled final pad). Without
    it, the first audio stream of input 0 is measured.
    Returns ffmpeg's measured dict with numeric values as floats (non-finite
    values such as -inf for digital silence become None)."""
    i, tp, lra = target
    ln = f"loudnorm=I={i}:TP={tp}:LRA={lra}:print_format=json"
    args = ["-loglevel", "info", "-nostats"] + [str(a) for a in input_args]
    if filter_prefix:
        fp = filter_prefix.strip().rstrip(";")
        m = re.search(r"(\[[^\[\]]+\])\s*$", fp)
        graph = f"{fp};{m.group(1)}{ln}[hc_ln]" if m else f"{fp},{ln}[hc_ln]"
        args += ["-filter_complex", graph, "-map", "[hc_ln]"]
    else:
        args += ["-map", "0:a:0", "-af", ln]
    args += ["-vn", "-sn", "-dn", "-f", "null", "-"]
    res = env.ff(args, capture=True, check=True, allow=allow)
    text = _stderr_text(res)
    return parse_loudnorm(text)


def parse_loudnorm(text: str) -> dict:
    """Parse the JSON block loudnorm prints at the end of stderr."""
    blocks = re.findall(r"\{[^{}]*\}", text or "")
    for blk in reversed(blocks):
        try:
            raw = json.loads(blk)
        except ValueError:
            continue
        if "input_i" not in raw:
            continue
        out = {}
        for k, v in raw.items():
            try:
                f = float(v)
                out[k] = f if math.isfinite(f) else None
            except (TypeError, ValueError):
                out[k] = v
        return out
    raise env.HCError("loudnorm printed no measurement (is the input silent or missing audio?)", 3)


def _stderr_text(res) -> str:
    """Pull stderr text out of whatever env.ff returned."""
    if res is None:
        return ""
    if isinstance(res, (str, bytes)):
        return res.decode("utf-8", "replace") if isinstance(res, bytes) else res
    if isinstance(res, tuple):
        for part in reversed(res):
            if isinstance(part, (str, bytes)) and part:
                return _stderr_text(part)
        return ""
    if isinstance(res, dict):
        return _stderr_text(res.get("stderr"))
    err = getattr(res, "stderr", None)
    return _stderr_text(err)
