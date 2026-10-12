"""Media and project builders for the cover / QA / export tests (B6).

Everything is synthesized with ffmpeg lavfi + numpy so the tests own their fixtures: a short
1080x1920 render with speech-like tone bursts at a chosen loudness, optional seeded defects
(sample rate, black frames), a small landscape source, and hand-written plan / captions /
render-log json per contract sections 8.1 and 9.3.
"""

from __future__ import annotations

import subprocess
import sys
import wave
from pathlib import Path

HERE = Path(__file__).resolve().parent
SK = HERE.parent
if str(SK / "scripts") not in sys.path:
    sys.path.insert(0, str(SK / "scripts"))

import numpy as np  # noqa: E402

from hookcutter import env  # noqa: E402

SR = 48000
FPS = 25
OUT_W, OUT_H = 1080, 1920


def _wav(path: Path, x: np.ndarray, sr: int = SR) -> None:
    pcm = np.clip(np.round(x * 32767.0), -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm.tobytes())


def speechy(dur: float, spans, seed: int = 7) -> np.ndarray:
    """-60 dB noise floor plus 220 Hz bursts (with harmonics) inside `spans`."""
    rng = np.random.default_rng(seed)
    n = int(round(dur * SR))
    x = rng.standard_normal(n) * (10 ** (-60 / 20))
    t = np.arange(n) / SR
    tone = 0.25 * (np.sin(2 * np.pi * 220 * t) + 0.5 * np.sin(2 * np.pi * 440 * t) + 0.25 * np.sin(2 * np.pi * 660 * t))
    ramp = int(0.01 * SR)
    for s, e in spans:
        a, b = int(s * SR), min(n, int(e * SR))
        env_ = np.ones(b - a)
        k = min(ramp, (b - a) // 2)
        if k:
            env_[:k] = np.linspace(0, 1, k)
            env_[-k:] = np.linspace(1, 0, k)
        x[a:b] += tone[a:b] * env_
    return x


def measure_lufs(path: Path) -> float:
    p = subprocess.run([env.ffmpeg(), "-hide_banner", "-nostdin", "-nostats", "-i", str(path),
                        "-af", "ebur128=peak=true", "-f", "null", "-"], capture_output=True, text=True)
    k = p.stderr.rfind("Summary:")
    import re
    m = re.findall(r"I:\s*(-?[\d.]+)\s*LUFS", p.stderr[k:])
    return float(m[-1])


def make_render(path: Path, *, dur: float = 3.0, lufs: float = -14.0, ar: int = 48000,
                black_frames=(), spans=((0.25, 1.3), (1.7, 2.7))) -> Path:
    """A render-shaped mp4: 1080x1920 h264 yuv420p bt709 25 fps + stereo AAC at `ar` Hz."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    wav = path.with_suffix(".src.wav")
    _wav(wav, speechy(dur, spans))
    gain = lufs - measure_lufs(wav)
    tags = "setparams=color_primaries=bt709:color_trc=bt709:colorspace=bt709:range=tv"
    vf = f"format=yuv420p,{tags}"
    if black_frames:
        cond = "+".join(f"eq(n\\,{k})" for k in black_frames)
        vf = f"drawbox=x=0:y=0:w=iw:h=ih:color=black:t=fill:enable='{cond}',format=yuv420p,{tags}"
    frames = int(round(dur * FPS))
    env.ff(["-f", "lavfi", "-i", f"testsrc2=size={OUT_W}x{OUT_H}:rate={FPS}:duration={dur}",
            "-i", str(wav), "-map", "0:v", "-map", "1:a", "-vf", vf, "-frames:v", str(frames),
            "-af", f"volume={gain:.3f}dB", "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
            "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709", "-color_range", "tv",
            "-c:a", "aac", "-b:a", "192k", "-ar", str(ar), "-ac", "2", "-movflags", "+faststart", str(path)])
    wav.unlink()
    return path


def make_source(path: Path, dur: float = 30.0, size: str = "1280x720") -> Path:
    """Landscape testsrc2 source with tone audio (no faces)."""
    path = Path(path)
    wav = path.with_suffix(".wav")
    _wav(wav, speechy(dur, [(s + 0.25, s + 3.0) for s in range(0, int(dur), 4)]))
    env.ff(["-f", "lavfi", "-i", f"testsrc2=size={size}:rate={FPS}:duration={dur}", "-i", str(wav),
            "-map", "0:v", "-map", "1:a", "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-ar", "48000", "-ac", "2", "-shortest", str(path)])
    wav.unlink()
    return path


def transcript_for(clips_words: dict) -> dict:
    """transcript.json words: each clip k gets its sentence at src 4k + 0.3 ..."""
    words, i = [], 0
    for k, (_cid, sentence) in enumerate(clips_words.items()):
        t = 4.0 * k + 0.3
        for w in sentence.split():
            words.append({"i": i, "w": w, "s": round(t, 3), "e": round(t + 0.3, 3), "p": 0.99, "sp": None})
            i += 1
            t += 0.42
    return {"version": 1, "backend": "test", "model": "none", "language": "en", "glossary": [],
            "audio_offset": 0.0, "media": "source.mp4", "sha256": "t", "words": words,
            "lines": [], "flags": []}


def plan_for(cid: str, k: int, transcript: dict, hook_lines, spec_hash: str = "h") -> dict:
    """Two-piece plan (75 frames, join at 1.52 s) over clip k's src window [4k, 4k + 4)."""
    s0 = 4.0 * k
    segs = [{"k": 0, "piece": 0, "src": [s0, s0 + 1.52], "out": [0.0, 1.52], "frames": 38,
             "samples": 72960, "join_before": "start", "fade_in": 0.008, "fade_out": 0.008, "gain_db": 0.0},
            {"k": 1, "piece": 1, "src": [s0 + 2.0, s0 + 3.48], "out": [1.52, 3.0], "frames": 37,
             "samples": 71040, "join_before": "cut", "fade_in": 0.008, "fade_out": 0.06, "gain_db": 0.0}]
    words = []
    for w in transcript["words"]:
        if not (s0 <= w["s"] < s0 + 4.0):
            continue
        seg = 0 if w["s"] < s0 + 1.52 else 1
        o = segs[seg]["out"][0] + (w["s"] - segs[seg]["src"][0])
        words.append({"i": w["i"], "raw": w["w"], "text": w["w"], "s_src": w["s"], "e_src": w["e"],
                      "s_out": round(max(0.0, o), 3), "e_out": round(max(0.0, o) + 0.3, 3), "piece": seg,
                      "seg": seg, "sp": None, "filler": False, "hidden": False, "omitted": False, "removed": False})
    idx = [w["i"] for w in words]
    return {
        "version": 1, "clip": cid, "kind": "cut", "spec_hash": spec_hash, "transcript_sha": "t",
        "source_fingerprint": "f", "code_version": env.code_version(), "fps": [FPS, 1], "grid_offset": 0.0,
        "sample_rate": SR, "source": {"width": 1280, "height": 720, "duration": 30.0, "hdr": "none"},
        "pieces": [{"n": 0, "role": "body", "join": "cut", "gain_db": 0.0, "words": [idx[0], idx[0]],
                    "src": segs[0]["src"], "out": segs[0]["out"], "frames": 38},
                   {"n": 1, "role": "close", "join": "cut", "gain_db": 0.0, "words": [idx[-1], idx[-1]],
                    "src": segs[1]["src"], "out": segs[1]["out"], "frames": 37}],
        "segments": segs, "removals": [],
        "joins": [{"k": 1, "out": 1.52, "kind": "piece", "before_src": s0 + 1.52, "after_src": s0 + 2.0, "tight": False}],
        "words": words, "intent": idx, "duration_out": 3.0, "frames_out": 75,
        "hook_card": {"lines": list(hook_lines), "dur": 1.5} if hook_lines else None,
        "framing": [], "cut_rate": {"joins": 1, "per_sec": 0.33}, "tighten": {"removed_s": 0, "count": 0},
        "lints": [],
    }


def captions_for(hook_lines, page_box=(200, 1150, 600, 110), matched=None) -> dict:
    """captions.json shaped like captions.CaptionTrack.to_json (the fields QA reads)."""
    import re
    phrases = [m for ln in hook_lines or [] for m in re.findall(r"\*([^*]+)\*", ln)]
    return {
        "version": 1, "band": [1000, 400], "lead_ms": 60,
        "pages": [{"i": 0, "start_out": 1.5, "end_out": 2.4, "box": list(page_box), "backing": False,
                   "tokens": [{"text": "WORD", "s_out": 1.6, "e_out": 1.9, "emph": False, "idx": [0]}]}],
        "hook": {"lines": list(hook_lines or []), "size": 96, "box": [150, 1050, 700, 200],
                 "start_out": 0.0, "end_out": 1.5} if hook_lines else None,
        "accents": {"phrases": phrases, "matched": phrases if matched is None else matched},
    }


def write_project(root: Path, clips: dict, sentences: dict, source: Path) -> Path:
    """clips.json + source + transcript for a project folder."""
    root.mkdir(parents=True, exist_ok=True)
    link = root / "source.mp4"
    if not link.exists():
        link.symlink_to(source)
    env.write_json(root / "source.json", {"version": 1, "path": "source.mp4", "fingerprint": "f",
                                          "duration": 30.0, "format_start": 0.0,
                                          "video": {"width": 1280, "height": 720, "fps": [FPS, 1], "start": 0.0},
                                          "audio": {"start": 0.0}})
    env.write_json(root / "transcript.json", transcript_for(sentences))
    env.write_json(root / "clips.json", {"version": 1, "defaults": {
        "styles": ["clean"], "formats": ["vertical"],
        "length": {"vertical": [2, 2, 10, 60], "landscape": [2, 2, 10, 60]}}, "clips": clips})
    return root
