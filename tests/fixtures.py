"""Synthetic media fixtures shared by the tests.

Video comes from lavfi (a dimmed testsrc, so it reads as dark grey with a moving
pattern and a frame counter). Audio is synthesized with numpy and written as a
WAV, which keeps clicks and tone bursts sample-exact. Every fixture writes a
sidecar ``<file>.json`` with the exact flash frames / times it contains
(``fixture_meta(path)``). Files are cached by parameter hash inside ``dir``.
"""

from __future__ import annotations

import json
import math
import os
import sys
import unittest
import wave
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

HERE = Path(__file__).resolve().parent
SK = HERE.parent
if str(SK / "scripts") not in sys.path:
    sys.path.insert(0, str(SK / "scripts"))

from hookcutter import env  # noqa: E402

SR = 48000
NOISE_DB = -60.0       # pink-noise floor RMS, dBFS
CLICK_HZ = 1000.0
CLICK_S = 0.010
CLICK_AMP = 0.5
TONE_HZ = 200.0
TONE_DB = -14.0
TONE_RAMP_S = 0.005
FIXTURE_VERSION = 2

slow = unittest.skipUnless(os.environ.get("HC_SLOW") == "1", "slow test (run: hc test --slow)")


def _fps(fps: Any) -> Fraction:
    s = str(fps)
    if "/" in s:
        a, b = s.split("/", 1)
        return Fraction(int(a), int(b))
    return Fraction(s)


def _key(params: Dict[str, Any]) -> str:
    return env.stable_hash(dict(params, _v=FIXTURE_VERSION))[:12]


def fixture_meta(path: Any) -> Dict[str, Any]:
    """The sidecar written next to a fixture: fps, flash frames/times, audio_delay, words."""
    return json.loads(Path(str(path) + ".json").read_text(encoding="utf-8"))


def _cached(path: Path, params: Dict[str, Any]) -> bool:
    side = Path(str(path) + ".json")
    if not (path.exists() and side.exists()):
        return False
    try:
        return json.loads(side.read_text(encoding="utf-8")).get("params") == params
    except ValueError:
        return False


def _pink_noise(n: int, rms_db: float, seed: int):
    import numpy as np
    rng = np.random.default_rng(seed)
    white = rng.standard_normal(n)
    spec = np.fft.rfft(white)
    f = np.arange(spec.size, dtype=np.float64)
    f[0] = 1.0
    pink = np.fft.irfft(spec / np.sqrt(f), n)
    pink -= pink.mean()
    rms = float(np.sqrt(np.mean(pink ** 2))) or 1.0
    return pink * (10 ** (rms_db / 20.0) / rms)


def _write_wav(path: Path, samples) -> None:
    import numpy as np
    pcm = np.clip(np.round(samples * 32767.0), -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes(pcm.tobytes())


def _video_filter(size: str, rate: Fraction, dur: float, flash_frames: Sequence[int], hdr: bool) -> str:
    vf = ("testsrc=size=%s:rate=%d/%d:duration=%s,format=yuv420p,"
          "lutyuv=y='16+(val-16)*0.18':u='128+(val-128)*0.15':v='128+(val-128)*0.15'"
          % (size, rate.numerator, rate.denominator, repr(float(dur))))
    if flash_frames:
        cond = "+".join("eq(n\\,%d)" % k for k in flash_frames)
        vf += ",drawbox=x=0:y=0:w=iw:h=ih:color=white:t=fill:enable='%s'" % cond
    if hdr:
        vf += ",setparams=color_primaries=bt2020:color_trc=arib-std-b67:colorspace=bt2020nc:range=tv"
    else:
        vf += ",setparams=color_primaries=bt709:color_trc=bt709:colorspace=bt709:range=tv"
    return vf


def _encode_video(out: Path, vf: str) -> None:
    args = ["-f", "lavfi", "-i", vf, "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "16",
             "-pix_fmt", "yuv420p", "-g", "50", "-bf", "2", "-f", "mp4", str(out)]
    env.ff(args)


def _mux(out: Path, video: Path, wav: Path, audio_offset: float, video_offset: float = 0.0) -> None:
    args = []
    if video_offset > 0:
        args += ["-itsoffset", "%.6f" % video_offset]
    args += ["-i", str(video)]
    if audio_offset > 0:
        args += ["-itsoffset", "%.6f" % audio_offset]
    tmp = out.with_name(out.stem + ".tmp.mp4")
    args += ["-i", str(wav), "-map", "0:v", "-map", "1:a", "-c:v", "copy",
             "-c:a", "aac", "-b:a", "160k", "-ar", str(SR), "-ac", "2", "-f", "mp4", str(tmp)]
    env.ff(args)
    os.replace(str(tmp), str(out))


def make_av_fixture(dir: Any, *, fps: str = "25", dur: float = 40, flashes: Optional[Iterable[float]] = None,
                    audio_delay: float = 0.0, size: str = "640x360", hdr: bool = False) -> Path:
    """Dark testsrc video with one white frame per flash and a 1 kHz 10 ms click at the same
    container time, over a -60 dB pink-noise floor; mono synthesized, stereo 48 kHz AAC in mp4.

    Flash k sits on frame round(t * fps); its exact time is that frame's time. ``audio_delay``
    (seconds, may be negative) delays the audio stream start with -itsoffset while keeping the
    clicks aligned with the flashes on the container timeline.
    """
    import numpy as np
    d = Path(dir)
    d.mkdir(parents=True, exist_ok=True)
    rate = _fps(fps)
    flashes = list(flashes or [])
    params = {"kind": "av", "fps": str(fps), "dur": float(dur), "flashes": [float(x) for x in flashes],
              "audio_delay": float(audio_delay), "size": size, "hdr": bool(hdr)}
    out = d / ("av_%s.mp4" % _key(params))
    if _cached(out, params):
        return out
    n_frames = int(math.floor(float(dur) * rate + 1e-9))
    frames = sorted({int(round(float(t) * rate)) for t in flashes})
    frames = [k for k in frames if 0 <= k < n_frames]
    v_off = max(0.0, -float(audio_delay))
    a_off = max(0.0, float(audio_delay))
    a_len = max(0.1, float(dur) + v_off - a_off)
    n = int(round(a_len * SR))
    floor = _pink_noise(n, NOISE_DB, seed=1234)
    tt = np.arange(int(round(CLICK_S * SR))) / SR
    click = CLICK_AMP * np.sin(2 * np.pi * CLICK_HZ * tt)

    def synth(click_times_rel):
        audio = floor.copy()
        for t in click_times_rel:
            i0 = int(round(t * SR))
            if 0 <= i0 < n:
                seg = click[: n - i0]
                audio[i0:i0 + seg.size] += seg
        return audio

    vtmp = out.with_name(out.stem + ".video.mp4")
    wav = out.with_suffix(".wav")
    try:
        _encode_video(vtmp, _video_filter(size, rate, float(dur), frames, hdr))
        # Pass 1: nominal placement. Then measure where the muxer really put both streams
        # (B-frame delay, AAC priming and mp4 edit-list rounding all move them) and correct.
        nominal = [v_off + float(Fraction(k) / rate) for k in frames]
        rel = [t - a_off for t in nominal]
        info = {}  # type: Dict[str, Any]
        for _pass in range(3):
            _write_wav(wav, synth(rel))
            _mux(out, vtmp, wav, a_off, v_off)
            info = _measure(out, frames, rate)
            err = info["sync_error"]
            if err is None or abs(err) <= 0.0002:
                break
            rel = [t - err for t in rel]
    finally:
        for f in (vtmp, wav):
            if f.exists():
                f.unlink()
    meta = {"params": params, "fps": [rate.numerator, rate.denominator], "flash_frames": frames,
            "flash_times": info.get("flash_times", []), "click_times": info.get("click_times", []),
            "video_start": info.get("video_start"), "audio_start": info.get("audio_start"),
            "sync_error": info.get("sync_error")}
    Path(str(out) + ".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return out


def _measure(path: Path, frames: Sequence[int], rate: Fraction) -> Dict[str, Any]:
    """Container-timeline flash times (video start + k/fps) and decoded click onsets."""
    data = env.ffprobe_json(path, "-show_format", "-show_streams")
    fstart = float((data.get("format") or {}).get("start_time") or 0.0)
    starts = {}
    for s in data.get("streams") or []:
        starts[s.get("codec_type")] = float(s.get("start_time") or 0.0) - fstart
    v0, a0 = starts.get("video", 0.0), starts.get("audio", 0.0)
    flash = [v0 + float(Fraction(k) / rate) for k in frames]
    clicks = [a0 + t for t in click_onsets(decode_mono(path))]
    err = None
    if flash and clicks:
        diffs = []
        for t in flash:
            near = min(clicks, key=lambda c: abs(c - t))
            if abs(near - t) < 0.2:
                diffs.append(near - t)
        if diffs:
            diffs.sort()
            err = diffs[len(diffs) // 2]
    return {"video_start": round(v0, 6), "audio_start": round(a0, 6), "flash_times": flash,
            "click_times": clicks, "sync_error": err}


def _norm_words(words: Iterable[Sequence[Any]]) -> List[Tuple[str, float, float, Optional[str]]]:
    out = []
    for w in words:
        text, s, e = str(w[0]), float(w[1]), float(w[2])
        sp = str(w[3]) if len(w) > 3 and w[3] is not None else None
        out.append((text, s, e, sp))
    out.sort(key=lambda x: (x[1], x[2]))
    return out


def fake_transcript(words: Iterable[Sequence[Any]], *, media: str = "source.mp4",
                    fingerprint: Optional[str] = None) -> Dict[str, Any]:
    """A transcript.json dict, shaped as `hc transcribe` writes it, for (text, s, e[, sp])."""
    ws = _norm_words(words)
    items = []
    for i, (text, s, e, sp) in enumerate(ws):
        items.append({"i": i, "w": text, "s": round(s, 3), "e": round(max(e, s), 3), "p": 0.99, "sp": sp})
    lines = []
    a = 0
    for i, it in enumerate(items):
        last = i == len(items) - 1
        brk = last or it["w"].endswith((".", "?", "!")) or (items[i + 1]["s"] - it["e"] >= 0.6) \
            or (items[i + 1]["sp"] != it["sp"]) or (i - a + 1 >= 30)
        if brk:
            lines.append({"a": a, "b": i, "s": items[a]["s"], "e": it["e"], "sp": items[a]["sp"]})
            a = i + 1
    return {
        "version": 1, "backend": "fixture", "model": "fixture", "backend_version": "0",
        "language": "en", "glossary": [], "audio_offset": 0.0,
        "source_fingerprint": fingerprint, "media": media,
        "sha256": env.stable_hash([[w["w"], w["s"], w["e"]] for w in items]),
        "created": "2026-01-01T00:00:00+0000",
        "words": items, "lines": lines, "flags": [],
    }


def make_speech_fixture(dir: Any, words: Iterable[Sequence[Any]], *, fps: str = "25",
                        size: str = "640x360", dur: Optional[float] = None,
                        tone_db: float = TONE_DB) -> Path:
    """mp4 with a 200 Hz tone burst over each word span (energy-based snapping sees "speech")
    on a -60 dB pink floor, plus ``dir/transcript.json`` matching the words."""
    import numpy as np
    d = Path(dir)
    d.mkdir(parents=True, exist_ok=True)
    ws = _norm_words(words)
    total = float(dur) if dur else (max([w[2] for w in ws] or [0.0]) + 2.0)
    rate = _fps(fps)
    params = {"kind": "speech", "fps": str(fps), "size": size, "dur": total, "tone_db": tone_db,
              "words": [[w[0], w[1], w[2], w[3]] for w in ws]}
    out = d / ("speech_%s.mp4" % _key(params))
    if not _cached(out, params):
        n = int(round(total * SR))
        audio = _pink_noise(n, NOISE_DB, seed=99)
        amp = 10 ** (tone_db / 20.0) * math.sqrt(2.0)
        ramp = int(round(TONE_RAMP_S * SR))
        for _, s, e, _sp in ws:
            i0, i1 = max(0, int(round(s * SR))), min(n, int(round(e * SR)))
            if i1 <= i0:
                continue
            t = np.arange(i1 - i0) / SR
            burst = amp * np.sin(2 * np.pi * TONE_HZ * t)
            r = min(ramp, burst.size // 2)
            if r > 0:
                env_ = np.ones(burst.size)
                env_[:r] = np.linspace(0.0, 1.0, r)
                env_[-r:] = np.linspace(1.0, 0.0, r)
                burst *= env_
            audio[i0:i1] += burst
        wav = out.with_suffix(".wav")
        _write_wav(wav, audio)
        vtmp = out.with_name(out.stem + ".video.mp4")
        try:
            _encode_video(vtmp, _video_filter(size, rate, total, [], False))
            _mux(out, vtmp, wav, 0.0)
        finally:
            for f in (vtmp, wav):
                if f.exists():
                    f.unlink()
        meta = {"params": params, "fps": [rate.numerator, rate.denominator], "flash_frames": [],
                "flash_times": [], "video_start": 0.0, "audio_start": 0.0}
        Path(str(out) + ".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    env.write_json(d / "transcript.json", fake_transcript(ws, media=out.name, fingerprint=env.fingerprint(out)))
    return out


def make_project(dir: Any, source_path: Any, words: Optional[Iterable[Sequence[Any]]] = None) -> Path:
    """Create ``dir/project`` with source.<ext> (symlink), source.json (via probe) and,
    when ``words`` is given, transcript.json. Returns the project path P."""
    from hookcutter import probe
    src = Path(source_path).resolve()
    P = Path(dir) / "project"
    P.mkdir(parents=True, exist_ok=True)
    link = P / ("source" + (src.suffix.lower() or ".mp4"))
    for old in P.glob("source.*"):
        if old.name != "source.json" and (old.is_symlink() or old.is_file()):
            old.unlink()
    try:
        os.symlink(str(src), str(link))
    except OSError:
        import shutil
        shutil.copy2(str(src), str(link))
    meta = probe.probe_source(link, original=str(src), name=link.name)
    env.write_json(P / "source.json", meta)
    if words is not None:
        env.write_json(P / "transcript.json",
                       fake_transcript(words, media=link.name, fingerprint=meta["fingerprint"]))
    return P


def decode_mono(path: Any, *, sr: int = SR, stream: str = "0:a:0"):
    """Decoded audio as float32 mono numpy array (sample 0 = the stream's start time)."""
    import numpy as np
    res = env.ff(["-i", str(path), "-map", stream, "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"])
    return np.frombuffer(res.stdout, dtype="<f4")


def click_onsets(samples, *, sr: int = SR, threshold: float = 0.05, min_gap: float = 0.1) -> List[float]:
    """Sample-accurate onset times (s, from sample 0) of clicks/bursts above ``threshold``."""
    import numpy as np
    idx = np.flatnonzero(np.abs(samples) > threshold)
    out, last = [], -1e9
    for i in idx:
        t = i / float(sr)
        if t - last > min_gap:
            out.append(t)
        last = t
    return out
