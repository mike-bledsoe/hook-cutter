"""ASR backends -> transcript.json + transcript.md (contract section 6.2).

Backends: mlx (mlx-whisper, Apple Silicon), faster-whisper (CPU/CUDA, everywhere) and whispercpp
(`whisper-cli`). `auto` = mlx on arm64 macOS when importable, else faster-whisper.

Pipeline: source audio -> 16 kHz mono wav (cached by source fingerprint) -> backend words (cached
by wav sha + asr_key_material: backend, model, language, glossary, runner options, backend
version) -> normalization (audio offset, non-speech removal, sort, dedupe) -> audio gate (drop
words with no audible sound) -> energy re-timing (align.repair_times) -> re-listen to voiced
audio no word covers (each gap transcribed on its own, words spliced in) -> unsure words that
hold two bursts of sound (settle_bursts: a false start is flagged, a start stamped on a laugh
burst moves past it) -> starts stamped inside someone else's laugh (align.repair_overlapped, on
the band shapes of the same wav) -> speakers (optional VTT) -> flags, lines -> json + md.

Flags never change the words: "false_start" (an abandoned attempt folded into one word, or a
visible repeat holding an overlong word) and "timing_uncertain" (a start inside sound that is
not transcribed, with nothing to mark where the word begins) sit next to "repetition_loop",
"low_confidence", "respliced", "unworded_audio", "dropout", "speaker_uncertain", "no_audio".

Library entry for QA: `transcribe_media(path, backend, model, language, glossary) -> words`
(gated, re-timed and gap-filled the same way, on the media's own clock).
"""

from __future__ import annotations

import contextlib
import importlib
import importlib.util
import json
import math
import os
import platform
import re
import shutil
import sys
import tempfile
import time
import wave
from bisect import bisect_left, bisect_right
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from . import env, textnorm

STEP = "transcribe"
BACKENDS = ("auto", "mlx", "faster-whisper", "whispercpp")
BACKEND_ALIASES = {
    "mlx-whisper": "mlx", "mlx_whisper": "mlx", "faster_whisper": "faster-whisper",
    "fw": "faster-whisper", "fasterwhisper": "faster-whisper", "whisper.cpp": "whispercpp",
    "whisper-cpp": "whispercpp", "whisper_cpp": "whispercpp", "cpp": "whispercpp",
}
DEFAULT_MODELS = {"mlx": "mlx-community/whisper-large-v3-turbo", "faster-whisper": "large-v3-turbo"}
MLX_MODEL_ALIASES = {
    "large-v3-turbo": "mlx-community/whisper-large-v3-turbo",
    "turbo": "mlx-community/whisper-large-v3-turbo",
    "large-v3": "mlx-community/whisper-large-v3-mlx",
    "large": "mlx-community/whisper-large-v3-mlx",
    "medium": "mlx-community/whisper-medium-mlx",
    "small": "mlx-community/whisper-small-mlx",
    "small.en": "mlx-community/whisper-small.en-mlx",
    "base": "mlx-community/whisper-base-mlx",
    "tiny": "mlx-community/whisper-tiny-mlx",
}
FW_MODEL_ALIASES = {"turbo": "large-v3-turbo"}
WHISPERCPP_MODEL_ENV = "HOOKCUTTER_WHISPERCPP_MODEL"
DEVICE_ENV = "HOOKCUTTER_DEVICE"   # faster-whisper device: cpu | cuda | auto (default)
# whisper.cpp 1.9 with --prompt and -ml 1 drops opening words and shifts word times by up to
# ~1 s (measured against the same audio without a prompt), so the glossary is not sent to it.
WHISPERCPP_USE_PROMPT = False

SAMPLE_RATE = 16000
ASR_CACHE_VERSION = 2          # raw-ASR cache key layout (v1 keys are still read, see transcribe_wav)
POST_VERSION = 6               # bump when the audio gate, the re-timing or the gap fill change
HALLUCINATION_SILENCE = 2.0    # seconds (mlx + faster-whisper)
FW_BEAM_SIZE = 5
FW_VAD = {"min_silence_duration_ms": 1000, "speech_pad_ms": 400, "threshold": 0.35}
PROMPT_MAX_CHARS = 600         # whisper prompts are capped near 224 tokens

MIN_WORD_DUR = 0.08            # zero-length words get e = min(s + this, next.s)
CPP_MAX_WORD = 1.5             # whisper.cpp DTW words: end = next start, at most this long
LINE_PAUSE = 0.6               # a pause this long starts a new transcript line
LINE_MAX_WORDS = 30            # soft cap: from here a line ends at the next clause mark
LINE_HARD_WORDS = 45           # hard cap: the line ends at the largest gap in its last 10 words
LINE_LOOKBACK = 10
LOOP_MAX_N = 15                # repetition loops: n-grams up to this long
LOOP_EXEMPT = frozenset({"very", "really", "no"})   # repeated for emphasis, not a model loop
# A doubled function word is a stumble whatever its length ("the the", "I I"); "that that" and
# "had had" are grammar and stay out.
STUMBLE_WORDS = frozenset({"the", "a", "an", "i", "to", "and", "of", "in", "we", "it", "my", "is",
                           "was", "you", "he", "she", "they", "so", "but", "on", "for", "with"})
RESTART_MARKS = (",", "-", "\u2013", "\u2014", "...", "\u2026")   # the model's mark for a restart
LOW_CONF_P = 0.5               # names / numbers below this probability get flagged
LOW_CONF_REPORT_MAX = 25
SILENT_FRACTION = 0.2          # a word with less voiced audio than this share is "in silence"
VTT_SLACK = 0.3                # a word midpoint this close to a cue still takes its speaker
VTT_GAP_FILL = 4.0              # words in a gap of at most this between two same-speaker cues take it
VTT_BLIP_WORDS = 2              # a mid-sentence speaker flip this short is cue jitter (see _smooth_speaker_blips)
TURN_PAUSE = 0.2               # a speaker turn sits on a sentence end or a pause this long
VTT_DROPOUT_WORDS = 4          # a cue with at least this many words ...
VTT_DROPOUT_COVER = 0.4        # ... that ASR covers below this share is a dropout
HOLE_MIN_RUN = 0.8             # re-listen to voiced audio of at least this long with no word
HOLE_ABUT_RUN = 0.5            # ... or this long when it runs straight out of (or into) a word at speech level
HOLE_BRIDGE = 0.3
HOLE_MERGE = 2.0               # gaps closer than this are re-listened together
HOLE_CAP = 300
HOLE_SPLIT = 120.0             # a gap longer than this is re-listened in HOLE_SLICE pieces
HOLE_SLICE = 60.0
HOLE_PAD = 1.0                 # context either side of a gap sent to the model
HOLE_KEEP = 0.15               # a spliced word's midpoint must lie this close to the gap
HOLE_OVERLAP = 0.05            # ... and overlap no existing word by more than this
HOLE_RETIME = 2.0              # re-time words this close to a splice
HOLE_MOVED = 0.3               # a matched context word the re-listen hears this far away moves
HOLE_EDGE = 0.2                # ... unless the slice heard it this close to its own edge
HOLE_RETRY_MIN = 2.0           # still unworded voiced audio this long is re-listened without the prompt
HOLE_SUB_DT = 1.0              # a slice word this close to an existing word it resembles replaces it
HOLE_SUB_SIM = 0.6             # ... resembles = difflib character ratio at least this
ECHO_MAX_P = 0.05              # a glossary term a re-listen hears with a word this unsure is a prompt echo
SOUND_REPORT_MAX = 8           # distinct sound-event texts the transcribe summary names
FS_LOW_P = 0.8                 # false starts: a word this unsure whose span holds two bursts of sound
FS_LEAD = 1.2                  # its lead-in is re-listened from a word boundary at least this far back
FS_LEAD_MAX = 3.0              # ... from the last pause within this, when there is one
FS_LEAD_PAUSE = 0.2
FS_LEAD_PAD = 0.1              # sound kept before that boundary
FS_WORD_MIN = 0.1              # the first burst holds at most one heard word per this many seconds
FS_TAIL = 1.2                  # a start moved past a dip is checked by re-listening this far on
FS_TAIL_PAD = 0.03             # ... from this far before the new start
FS_CAP = 60                    # at most this many lead-ins are re-listened (the least sure words)
FS_REPORT_MAX = 25
NONSPEECH_MAX_PAREN = 4        # "(audience laughs)" style events are short
NONSPEECH_MAX_BRACKET = 8

CLIP_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,47}$")
_MUSIC_RE = re.compile(r"^[♪♫♬♩\s*~-]+$")
_SPECIAL_RE = re.compile(r"^(?:\[_[A-Z]+_?\d*\]|<\|[^|]*\|>)$")
_TRAIL_PUNCT = ".,!?;:\"'”’"
_CLOSERS = "\"')]}”’"
_IGNORED_EVENTS = frozenset({"blankaudio", "silence", "nospeech", "noaudio", "blank"})

# ---------------------------------------------------------------------------
# Text files (glossary, VTT): BOM / UTF-16 aware
# ---------------------------------------------------------------------------


def decode_text(data: bytes) -> tuple[str, list[str]]:
    """Bytes of a user text file -> (text, warnings). Handles a UTF-8 BOM, UTF-16 with a BOM or
    without one (NUL-heavy), strips NULs and reports undecodable bytes."""
    warns: list[str] = []
    if data.startswith(b"\xef\xbb\xbf"):
        text = data[3:].decode("utf-8", errors="replace")
    elif data.startswith((b"\xff\xfe", b"\xfe\xff")):
        text = data.decode("utf-16", errors="replace")
    elif len(data) >= 4 and data.count(b"\x00") >= len(data) // 4:
        even, odd = data[0::2].count(b"\x00"), data[1::2].count(b"\x00")
        text = data.decode("utf-16-le" if odd >= even else "utf-16-be", errors="replace")
    else:
        text = data.decode("utf-8", errors="replace")
    if "\x00" in text:
        text = text.replace("\x00", "")
        warns.append("NUL characters removed")
    text = text.lstrip("\ufeff")
    if "\ufffd" in text:
        warns.append("%d characters could not be decoded (save the file as UTF-8)"
                     % text.count("\ufffd"))
    return text, warns


def read_text_file(path: str | os.PathLike) -> str:
    """Read a user text file (glossary, VTT) in any common encoding; warnings go to the log."""
    text, warns = decode_text(Path(path).read_bytes())
    for w in warns:
        env.log(STEP, "warning: %s: %s" % (Path(path).name, w))
    return text


# ---------------------------------------------------------------------------
# Backend and model resolution
# ---------------------------------------------------------------------------


def _hf_cached(repo: str) -> bool:
    """True when a Hugging Face repo's config is already in the local cache (no download)."""
    if os.path.isdir(str(repo)):
        return True
    try:
        from huggingface_hub import try_to_load_from_cache
        hit = try_to_load_from_cache(str(repo), "config.json")
        return isinstance(hit, str)
    except Exception:
        return False


def _fw_cached(model: str, root: str | os.PathLike) -> bool:
    """True when a faster-whisper model (alias, repo id or directory) is already under `root`."""
    if os.path.isdir(str(model)):
        return True
    try:
        from faster_whisper.utils import _MODELS  # type: ignore
    except Exception:
        _MODELS = {}
    repo = str(model) if "/" in str(model) else _MODELS.get(str(model))
    if not repo:
        return False
    try:
        from huggingface_hub import try_to_load_from_cache
        hit = try_to_load_from_cache(repo, "model.bin", cache_dir=str(root))
        return isinstance(hit, str)
    except Exception:
        return False


def _mlx_supported() -> bool:
    return (sys.platform == "darwin" and platform.machine() == "arm64"
            and importlib.util.find_spec("mlx_whisper") is not None)


def resolve_backend(name: str | None) -> str:
    """Map a user backend name (or "auto") to mlx | faster-whisper | whispercpp."""
    b = (name or "auto").strip().lower()
    b = BACKEND_ALIASES.get(b, b)
    if b not in BACKENDS:
        raise env.HCError("unknown ASR backend %r (use %s)" % (name, ", ".join(BACKENDS)), code=1)
    if b == "auto":
        return "mlx" if _mlx_supported() else "faster-whisper"
    return b


def resolve_model(backend: str, model: str | None) -> str:
    """Backend-specific model id: HF repo for mlx, alias/path for faster-whisper, file for cpp."""
    if backend == "mlx":
        if not model:
            return DEFAULT_MODELS["mlx"]
        return MLX_MODEL_ALIASES.get(model, model)
    if backend == "faster-whisper":
        if not model:
            return DEFAULT_MODELS["faster-whisper"]
        return FW_MODEL_ALIASES.get(model, model)
    if backend == "whispercpp":
        path = model or os.environ.get(WHISPERCPP_MODEL_ENV)
        if not path:
            raise env.HCError("whispercpp needs a ggml model file: pass --model PATH or set %s"
                              % WHISPERCPP_MODEL_ENV, code=3)
        p = Path(path).expanduser()
        if not p.is_file():
            raise env.HCError("whisper.cpp model not found: %s" % p, code=3)
        return str(p.resolve())
    raise env.HCError("unknown ASR backend %r" % backend, code=1)


def model_short_name(model: str) -> str:
    """"mlx-community/whisper-large-v3-turbo" -> "large-v3-turbo"; ggml file -> its stem."""
    name = Path(str(model)).name if ("/" in str(model) or os.sep in str(model)) else str(model)
    if name.endswith(".bin"):
        name = name[:-4]
    for prefix in ("ggml-", "whisper-", "faster-whisper-"):
        if name.startswith(prefix):
            name = name[len(prefix):]
    for suffix in ("-mlx",):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name


_VERSIONS: dict = {}


def backend_version(backend: str) -> str | None:
    if backend in _VERSIONS:
        return _VERSIONS[backend]
    out = None
    if backend in ("mlx", "faster-whisper"):
        from importlib import metadata
        dist = "mlx-whisper" if backend == "mlx" else "faster-whisper"
        try:
            out = metadata.version(dist)
        except metadata.PackageNotFoundError:
            out = None
    elif backend == "whispercpp":
        exe = shutil.which("whisper-cli")
        if exe:
            proc = env.run([exe, "--version"], check=False)
            m = re.search(r"version:?\s*v?([0-9][\w.\-]*)", (proc.stdout or "") + (proc.stderr or ""))
            out = m.group(1) if m else None
    _VERSIONS[backend] = out
    return out


def runner_options(backend: str, model: str, prompt: str | None) -> dict:
    """The "options" dict a runner records for these inputs (mlx and faster-whisper; whisper.cpp
    records its argv)."""
    if backend == "mlx":
        return {"word_timestamps": True, "condition_on_previous_text": False,
                "hallucination_silence_threshold": HALLUCINATION_SILENCE,
                "initial_prompt": bool(prompt), "prompt_every_window": bool(prompt)}
    if backend == "faster-whisper":
        return {"word_timestamps": True, "condition_on_previous_text": False,
                "hallucination_silence_threshold": HALLUCINATION_SILENCE,
                "vad_filter": True, "vad_parameters": dict(FW_VAD),
                "beam_size": FW_BEAM_SIZE, "hotwords": bool(prompt)}
    if backend == "whispercpp":
        turbo = "turbo" in Path(str(model)).name.lower()
        return {"ml": 1, "sow": True, "json_full": True,
                "dtw": "large.v3.turbo" if turbo else None, "nfa": turbo,
                "prompt": bool(prompt) and WHISPERCPP_USE_PROMPT, "words": "t_dtw"}
    return {}


def asr_key_material(backend: str | None, model: str | None = None, language: str | None = None,
                     glossary: Sequence[str] | str | None = None) -> dict:
    """Everything that decides a raw ASR result except the audio: concrete backend, resolved
    model, language, glossary, the runner options and the backend version. Never raises (an
    unresolvable backend or model is kept as given)."""
    try:
        b = resolve_backend(backend)
    except env.HCError:
        b = str(backend)
    try:
        mdl = resolve_model(b, model)
    except env.HCError:
        mdl = model
    terms = glossary_terms(glossary) if isinstance(glossary, str) else list(glossary or [])
    lang = "auto" if language in (None, "", "auto") else language
    return {"backend": b, "model": mdl, "language": lang, "glossary": terms,
            "options": runner_options(b, mdl, build_prompt(terms)),
            "backend_version": backend_version(b)}


# ---------------------------------------------------------------------------
# Glossary
# ---------------------------------------------------------------------------


def glossary_terms(text: str | Sequence[str] | None = None,
                   path: str | os.PathLike | None = None) -> list[str]:
    """Terms from a glossary file (one per line, "#" lines are comments) plus --glossary text
    (comma, semicolon or newline separated). Deduped case-insensitively, order kept. The file
    may be UTF-8 (with or without a BOM) or UTF-16."""
    raw: list[str] = []
    if path is not None and Path(path).is_file():
        for line in read_text_file(path).splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                raw.append(line)
    if text:
        parts = [text] if isinstance(text, str) else list(text)
        for part in parts:
            raw.extend(re.split(r"[,;\n]", str(part)))
    out: list[str] = []
    seen: set[str] = set()
    for t in raw:
        t = " ".join(t.replace("\x00", "").split())
        if t and t.lower() not in seen:
            seen.add(t.lower())
            out.append(t)
    return out


def build_prompt(terms: Sequence[str]) -> str | None:
    """Comma-joined glossary prompt, cut at a term boundary to fit the decoder prompt window."""
    out = ""
    for t in terms:
        nxt = t if not out else out + ", " + t
        if len(nxt) > PROMPT_MAX_CHARS:
            break
        out = nxt
    return out + "." if out else None


# ---------------------------------------------------------------------------
# Audio extraction
# ---------------------------------------------------------------------------


def wav_ok(path: str | os.PathLike) -> bool:
    """True when path is a readable 16 kHz mono PCM wav with at least one sample."""
    try:
        with wave.open(str(path), "rb") as w:
            return (w.getnchannels() == 1 and w.getframerate() == SAMPLE_RATE
                    and w.getsampwidth() == 2 and w.getnframes() > 0)
    except (OSError, EOFError, wave.Error):
        return False


def extract_wav(media: str | os.PathLike, out: str | os.PathLike, stream: int | None = None) -> Path:
    """Decode one audio stream to 16 kHz mono s16 wav (atomic). Sample 0 = the stream's first
    decoded sample, so wav time + audio.start = src time. Timestamp gaps are filled with silence."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(".%s.%d.part.wav" % (out.stem, os.getpid()))
    smap = "0:%d" % int(stream) if stream is not None else "0:a:0"
    try:
        env.ff(["-i", str(media), "-map", smap, "-vn", "-sn", "-dn",
                "-af", "aresample=%d:async=1" % SAMPLE_RATE, "-ac", "1",
                "-c:a", "pcm_s16le", "-f", "wav", str(tmp)])
        if not wav_ok(tmp):
            raise env.HCError("no audio samples decoded from %s" % media, code=1)
        os.replace(tmp, out)
    finally:
        if tmp.exists():
            tmp.unlink()
    return out


def ensure_wav(project: Any, force: bool = False) -> tuple[Path, float]:
    """Cached 16 kHz mono wav of the project source -> (path, audio offset in src seconds)."""
    src = project.source()
    audio = src.get("audio") or {}
    if not audio:
        raise env.HCError("the source has no audio stream; nothing to transcribe", code=1)
    wav = project.cache_path("wav-%s.wav" % src["fingerprint"])
    if force or not wav_ok(wav):
        env.log(STEP, "extracting 16 kHz mono audio")
        extract_wav(project.source_path(), wav, audio.get("index"))
    return wav, float(audio.get("start") or 0.0)


def probe_audio(media: str | os.PathLike) -> dict:
    """First audio stream of a file -> {"index", "start" (rel. to format start), "duration"}."""
    info = env.ffprobe_json(media, "-show_streams", "-show_format")
    fmt = info.get("format") or {}
    fstart = _float(fmt.get("start_time"), 0.0)
    for st in info.get("streams") or []:
        if st.get("codec_type") == "audio":
            return {"index": int(st.get("index", 0)),
                    "start": round(_float(st.get("start_time"), fstart) - fstart, 6),
                    "duration": _float(fmt.get("duration"), None)}
    raise env.HCError("no audio stream in %s" % media, code=1)


def load_wav(path: str | os.PathLike):
    """16 kHz mono s16 wav -> float32 numpy array in [-1, 1]."""
    import numpy as np
    with wave.open(str(path), "rb") as w:
        data = w.readframes(w.getnframes())
    return np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0


def wav_duration(path: str | os.PathLike) -> float:
    with wave.open(str(path), "rb") as w:
        return w.getnframes() / float(w.getframerate())


def cut_wav(src: str | os.PathLike, out: str | os.PathLike, a: float, b: float) -> Path:
    """Copy samples [a, b) seconds of a 16 kHz mono wav into a new wav."""
    with wave.open(str(src), "rb") as w:
        n = w.getnframes()
        i0 = max(0, min(n, int(round(a * SAMPLE_RATE))))
        i1 = max(i0, min(n, int(round(b * SAMPLE_RATE))))
        w.setpos(i0)
        data = w.readframes(i1 - i0)
    with wave.open(str(out), "wb") as o:
        o.setnchannels(1)
        o.setsampwidth(2)
        o.setframerate(SAMPLE_RATE)
        o.writeframes(data)
    return Path(out)


def _float(v: Any, default: Any) -> Any:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return default
    return f if math.isfinite(f) else default


# ---------------------------------------------------------------------------
# Backend runners -> raw words [[text, s, e, p], ...] on the wav clock
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def _mlx_prompt_carry(mod: Any, enabled: bool):
    """Re-send the glossary prompt to every 30 s window.

    mlx-whisper drops the initial prompt after the first window when
    condition_on_previous_text=False. The decoder receives the prompt through
    DecodingOptions(prompt=...), so wrap that constructor: the first non-empty prompt (the
    glossary tokens) is remembered and supplied whenever a later window has none.
    """
    orig = getattr(mod, "DecodingOptions", None)
    if not enabled or orig is None:
        yield False
        return
    state: dict = {"carry": None}

    def factory(*a: Any, **kw: Any) -> Any:
        p = kw.get("prompt")
        if p:
            if state["carry"] is None:
                state["carry"] = list(p) if not isinstance(p, str) else p
        elif state["carry"]:
            kw["prompt"] = list(state["carry"]) if not isinstance(state["carry"], str) else state["carry"]
        return orig(*a, **kw)

    mod.DecodingOptions = factory
    try:
        yield True
    finally:
        mod.DecodingOptions = orig


def run_mlx(wav: Path, model: str, language: str | None, prompt: str | None,
            progress: bool = True) -> dict:
    env.setup_runtime_env()
    try:
        mod = importlib.import_module("mlx_whisper.transcribe")
    except ImportError as exc:
        raise env.HCError("mlx-whisper is not installed (%s). Run: hc doctor --install --with mlx"
                          % exc, code=3)
    audio = load_wav(wav)
    if progress:
        env.log(STEP, "mlx: %s%s" % (model, "" if _hf_cached(model) else " (first use downloads the model)"))
    with _mlx_prompt_carry(mod, bool(prompt)) as carried:
        result = mod.transcribe(
            audio, path_or_hf_repo=model, word_timestamps=True, initial_prompt=prompt,
            condition_on_previous_text=False, hallucination_silence_threshold=HALLUCINATION_SILENCE,
            language=language, verbose=False if progress else None)
    opts = runner_options("mlx", model, prompt)
    opts["prompt_every_window"] = bool(carried)
    return {"words": words_from_mlx(result), "language": result.get("language") or language,
            "options": opts}


_FW_MODELS: dict = {}


def _fw_device() -> tuple[str, str]:
    """(device, compute type) for faster-whisper. HOOKCUTTER_DEVICE = cpu | cuda | auto."""
    want = (os.environ.get(DEVICE_ENV) or "auto").strip().lower()
    if want == "cpu":
        return "cpu", "int8"
    if want == "cuda":
        return "cuda", "float16"
    if want not in ("", "auto"):
        env.log(STEP, "warning: %s=%r is not cpu, cuda or auto; using auto" % (DEVICE_ENV, want))
    try:
        import ctranslate2
        if ctranslate2.get_cuda_device_count() > 0:
            return "cuda", "float16"
    except Exception:  # noqa: BLE001 - any failure means no usable CUDA
        pass
    return "cpu", "int8"


def _fw_model(model: str, device: str, compute: str, progress: bool) -> Any:
    from faster_whisper import WhisperModel
    key = (model, device, compute)
    if key not in _FW_MODELS:
        root = env.fw_models_dir()
        root.mkdir(parents=True, exist_ok=True)
        if progress:
            env.log(STEP, "faster-whisper: %s on %s/%s%s" % (
                model, device, compute, "" if _fw_cached(model, root) else
                " (first use downloads the model)"))
        _FW_MODELS[key] = WhisperModel(model, device=device, compute_type=compute,
                                       download_root=str(root))
    return _FW_MODELS[key]


def run_faster_whisper(wav: Path, model: str, language: str | None, prompt: str | None,
                       progress: bool = True) -> dict:
    env.setup_runtime_env()
    try:
        import faster_whisper  # noqa: F401
    except ImportError as exc:
        raise env.HCError("faster-whisper is not installed (%s). Run: hc doctor --install" % exc,
                          code=3)
    device, compute = _fw_device()
    audio = load_wav(wav)
    total = len(audio) / float(SAMPLE_RATE)

    def attempt(dev: str, comp: str) -> tuple[list, Any]:
        mdl = _fw_model(model, dev, comp, progress)
        segments, info = mdl.transcribe(
            audio, language=language, beam_size=FW_BEAM_SIZE, word_timestamps=True,
            vad_filter=True, vad_parameters=dict(FW_VAD), condition_on_previous_text=False,
            hotwords=prompt or None, hallucination_silence_threshold=HALLUCINATION_SILENCE)
        collected = []
        next_report = 120.0
        for seg in segments:  # decoding is lazy: CUDA library errors surface here
            collected.append(seg)
            end = _float(getattr(seg, "end", None), 0.0)
            if progress and end >= next_report:
                env.log(STEP, "%s / %s" % (env.fmt_time(end, 0), env.fmt_time(total, 0)))
                next_report = end + 120.0
        return collected, info

    try:
        collected, info = attempt(device, compute)
    except Exception as exc:  # noqa: BLE001 - missing cuBLAS/cuDNN raise many different types
        if device != "cuda":
            raise
        _FW_MODELS.pop((model, device, compute), None)
        env.log(STEP, "CUDA libraries unavailable (%s); using CPU int8" % (str(exc)[:200] or
                                                                           type(exc).__name__))
        device, compute = "cpu", "int8"
        collected, info = attempt(device, compute)
    return {"words": words_from_faster(collected),
            "language": getattr(info, "language", None) or language,
            "device": device, "compute_type": compute,
            "options": runner_options("faster-whisper", model, prompt)}


_CPP_HELP: dict = {}


def _whispercpp_help(exe: str) -> str:
    if exe not in _CPP_HELP:
        proc = env.run([exe, "--help"], check=False)
        _CPP_HELP[exe] = (proc.stdout or "") + (proc.stderr or "")
    return _CPP_HELP[exe]


def _has_flag(help_text: str, flag: str) -> bool:
    return re.search(r"(?<![\w-])%s(?![\w-])" % re.escape(flag), help_text) is not None


def whispercpp_command(exe: str, model: str, wav: Path, out_prefix: Path, language: str | None,
                       prompt: str | None, help_text: str) -> list[str]:
    """whisper-cli argv: one word per segment (-ml 1 -sow), JSON (+ token probabilities)."""
    cmd = [exe, "-m", model, "-f", str(wav), "-ml", "1", "-sow", "-oj"]
    if _has_flag(help_text, "-ojf"):
        cmd.append("-ojf")
    cmd += ["-of", str(out_prefix), "-np", "-l", language or "auto"]
    if prompt and WHISPERCPP_USE_PROMPT:
        cmd += ["--prompt", prompt]
        if _has_flag(help_text, "--carry-initial-prompt"):
            cmd.append("--carry-initial-prompt")
    if "turbo" in Path(model).name.lower() and _has_flag(help_text, "-dtw"):
        cmd += ["-dtw", "large.v3.turbo"]
        if _has_flag(help_text, "-nfa"):
            cmd.append("-nfa")  # whisper.cpp ignores DTW while flash attention is on
    return cmd


def run_whispercpp(wav: Path, model: str, language: str | None, prompt: str | None,
                   progress: bool = True) -> dict:
    exe = shutil.which("whisper-cli")
    if not exe:
        raise env.HCError("whisper-cli not found on PATH (install whisper.cpp or use another backend)",
                          code=3)
    help_text = _whispercpp_help(exe)
    with tempfile.TemporaryDirectory(prefix="hc-wcpp-") as td:
        prefix = Path(td) / "asr"
        cmd = whispercpp_command(exe, model, wav, prefix, language, prompt, help_text)
        if progress:
            env.log(STEP, "whisper.cpp: %s" % Path(model).name)
            if prompt and not WHISPERCPP_USE_PROMPT:
                env.log(STEP, "note: the glossary is not sent to whisper.cpp (its prompt breaks "
                        "word timing); use mlx or faster-whisper for glossary biasing")
        env.run(cmd, check=True)
        out = prefix.with_suffix(".json")
        if not out.is_file():
            raise env.HCError("whisper-cli wrote no JSON output", code=3)
        data = json.loads(out.read_text(encoding="utf-8", errors="replace"))
    lang = ((data.get("result") or {}).get("language")) or language
    return {"words": words_from_whispercpp(data), "language": lang,
            "options": {"args": [a for a in cmd[5:] if not a.startswith(td)],
                        "prompt": bool(prompt) and WHISPERCPP_USE_PROMPT,
                        "dtw_words": whispercpp_has_dtw(data)}}


RUNNERS = {"mlx": run_mlx, "faster-whisper": run_faster_whisper, "whispercpp": run_whispercpp}


# ---------------------------------------------------------------------------
# Backend output parsing (pure)
# ---------------------------------------------------------------------------


def words_from_mlx(result: dict) -> list[list]:
    """mlx-whisper (or openai-whisper) result dict -> [[text, s, e, p], ...]."""
    out = []
    for seg in (result or {}).get("segments") or []:
        for w in seg.get("words") or []:
            out.append([w.get("word", ""), w.get("start"), w.get("end"), w.get("probability")])
    return out


def words_from_faster(segments: Iterable[Any]) -> list[list]:
    """faster-whisper Segment objects (or dicts) -> [[text, s, e, p], ...]."""
    out = []
    for seg in segments:
        words = seg.get("words") if isinstance(seg, dict) else getattr(seg, "words", None)
        for w in words or []:
            if isinstance(w, dict):
                out.append([w.get("word", ""), w.get("start"), w.get("end"), w.get("probability")])
            else:
                out.append([getattr(w, "word", ""), getattr(w, "start", None),
                            getattr(w, "end", None), getattr(w, "probability", None)])
    return out


def _cpp_special(text: str) -> bool:
    t = (text or "").strip()
    return bool(_SPECIAL_RE.match(t))


def _cpp_text_tokens(seg: dict) -> list[dict]:
    return [t for t in seg.get("tokens") or [] if isinstance(t, dict)
            and not _cpp_special(t.get("text", "")) and (t.get("text") or "").strip()]


def _dtw(tok: dict) -> float | None:
    v = tok.get("t_dtw")
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v / 100.0 if v >= 0 else None


def whispercpp_has_dtw(data: dict) -> bool:
    """True when the -ojf JSON carries DTW token times (tokens[].t_dtw >= 0)."""
    for seg in (data or {}).get("transcription") or []:
        if any(_dtw(t) is not None for t in _cpp_text_tokens(seg)):
            return True
    return False


def words_from_whispercpp(data: dict) -> list[list]:
    """whisper-cli -ml 1 -sow -oj[f] JSON -> [[text, s, e, p], ...] (offsets are milliseconds).

    With DTW token times (-dtw, t_dtw in centiseconds, -1 = absent) a word starts at the t_dtw
    of its first text token and ends at the next word's start (at most CPP_MAX_WORD later; the
    energy track trims it later). Without them the segment offsets are used. p = mean
    probability of the segment's text tokens. A segment holding several words (output made
    without -ml 1) is split, by token boundaries with DTW times, else by text length."""
    dtw_mode = whispercpp_has_dtw(data)
    out = []
    for seg in (data or {}).get("transcription") or []:
        text = seg.get("text") or ""
        off = seg.get("offsets") or {}
        s, e = off.get("from"), off.get("to")
        if s is None:
            continue
        s = float(s) / 1000.0
        e = float(e) / 1000.0 if e is not None else s
        toks = _cpp_text_tokens(seg)
        probs = [float(t["p"]) for t in toks if "p" in t]
        p = sum(probs) / len(probs) if probs else None
        parts = text.split()
        if dtw_mode and parts:
            starts = []
            for t in toks:
                if not starts or (t.get("text") or "").startswith(" "):
                    starts.append(_dtw(t))
            if len(parts) == 1 or len(starts) == len(parts):
                for k, x in enumerate(parts if len(parts) > 1 else [text]):
                    st = starts[k] if k < len(starts) and starts[k] is not None else s
                    out.append([x if len(parts) == 1 else " " + x, st, None, p])
                continue
        if len(parts) <= 1:
            out.append([text, s, e, p])
            continue
        total = sum(len(x) for x in parts) or 1
        t = s
        for x in parts:
            d = (e - s) * len(x) / total
            out.append([" " + x, t, t + d, p])
            t += d
    if dtw_mode:
        # ends: the next word's start (ordered by start), capped
        order = sorted(range(len(out)), key=lambda k: out[k][1])
        for pos, k in enumerate(order):
            if out[k][2] is not None:
                continue
            st = out[k][1]
            nxt = out[order[pos + 1]][1] if pos + 1 < len(order) else None
            end = st + CPP_MAX_WORD if nxt is None else min(max(nxt, st), st + CPP_MAX_WORD)
            out[k][2] = end
    return out


# ---------------------------------------------------------------------------
# Normalization (pure)
# ---------------------------------------------------------------------------


def _bare(text: str) -> str:
    return text.strip().rstrip(_TRAIL_PUNCT)


def split_nonspeech(items: list[dict]) -> tuple[list[dict], list[dict]]:
    """Separate sound-event tokens ("[Music]", "(laughs)", "[BLANK_AUDIO]", music notes) from
    speech. Returns (speech words, events [{"s", "e", "text"}])."""
    keep: list[dict] = []
    events: list[dict] = []
    i, n = 0, len(items)
    while i < n:
        t = items[i]["w"].strip()
        if _SPECIAL_RE.match(t) or _MUSIC_RE.match(t):
            i += 1
            continue
        opener = t[:1]
        if opener in "[(":
            closer = "]" if opener == "[" else ")"
            limit = NONSPEECH_MAX_BRACKET if opener == "[" else NONSPEECH_MAX_PAREN
            j = i
            while j < n and j - i < limit and not _bare(items[j]["w"]).endswith(closer):
                j += 1
            if j < n and j - i < limit:
                text = " ".join(x["w"].strip() for x in items[i:j + 1])
                if textnorm.norm(text) not in _IGNORED_EVENTS:
                    events.append({"s": items[i]["s"], "e": items[j]["e"], "text": text})
                i = j + 1
                continue
        keep.append(items[i])
        i += 1
    return keep, events


def normalize_words(raw: Iterable[Sequence], offset: float = 0.0,
                    events: list | None = None) -> tuple[list[dict], list[tuple[int, int]]]:
    """Raw backend words -> clean words [{"w", "s", "e", "p"}] on the src clock.

    Strips whitespace, drops empty and non-speech tokens (appended to `events` when given), adds
    `offset`, sorts by start (stable), dedupes on (round(s, 2), w), spreads runs of identical
    starts by letter count, enforces s <= e <= next.s, gives zero-length words
    e = min(s + MIN_WORD_DUR, next.s). Returns (words, spread_runs) where spread_runs are inclusive
    (a, b) index ranges whose timing was interpolated.
    """
    items = []
    for r in raw:
        text, s, e, p = (list(r) + [None, None, None, None])[:4]
        text = (text or "").strip()
        s = _float(s, None)
        if not text or s is None:
            continue
        e = _float(e, s)
        p = _float(p, None)
        items.append({"w": text, "s": s + offset, "e": e + offset,
                      "p": None if p is None else round(min(max(p, 0.0), 1.0), 3)})
    items, ev = split_nonspeech(items)
    if events is not None:
        events.extend({"s": round(x["s"], 3), "e": round(x["e"], 3), "text": x["text"]} for x in ev)
    items.sort(key=lambda d: d["s"])
    words: list[dict] = []
    seen: set = set()
    for d in items:
        d["s"] = round(d["s"], 3)
        d["e"] = round(d["e"], 3)
        key = (round(d["s"], 2), d["w"])
        if key in seen:
            continue
        seen.add(key)
        words.append(d)
    spread = _spread_identical_starts(words)
    _clamp_ends(words)
    return words, spread


def _clamp_ends(words: list[dict]) -> None:
    n = len(words)
    for k, d in enumerate(words):
        nxt = words[k + 1]["s"] if k + 1 < n else None
        if d["e"] < d["s"]:
            d["e"] = d["s"]
        if d["e"] == d["s"]:
            d["e"] = d["s"] + MIN_WORD_DUR if nxt is None else min(d["s"] + MIN_WORD_DUR, nxt)
        elif nxt is not None and d["e"] > nxt:
            d["e"] = max(d["s"], nxt)
        d["e"] = round(d["e"], 3)


def _spread_identical_starts(words: list[dict]) -> list[tuple[int, int]]:
    """Runs of words sharing one start time (typically a zero-length first word of a segment
    plus the word after it) share their span in proportion to their letter counts."""
    runs = []
    n = len(words)
    k = 0
    while k < n:
        j = k
        while j + 1 < n and words[j + 1]["s"] == words[k]["s"]:
            j += 1
        if j > k:
            cnt = j - k + 1
            s0 = words[k]["s"]
            end = max(d["e"] for d in words[k:j + 1])
            if end <= s0:
                end = s0 + MIN_WORD_DUR * cnt
            if j + 1 < n:
                end = min(end, words[j + 1]["s"])
            weights = [max(1, len(re.sub(r"\W", "", d["w"]))) for d in words[k:j + 1]]
            total = float(sum(weights))
            t = s0
            for m in range(cnt):
                nxt_t = s0 + (end - s0) * sum(weights[:m + 1]) / total
                words[k + m]["s"] = round(t, 3)
                words[k + m]["e"] = round(nxt_t, 3)
                t = nxt_t
            runs.append((k, j))
        k = j + 1
    return runs


def prompt_sent(asr: dict) -> str | None:
    """The glossary prompt the backend really received for the result `asr`, else None.

    transcribe_wav records the prompt it built for every backend, but whisper.cpp is never given
    it (WHISPERCPP_USE_PROMPT) and a runner records whether it used one in its options
    ("initial_prompt", "hotwords" or "prompt"). A backend that was sent no prompt cannot echo
    one, so the echo filters must not run on its words. A result with none of those option keys
    (an older cache file) keeps the recorded prompt."""
    prompt = asr.get("prompt")
    if not prompt:
        return None
    if asr.get("backend") == "whispercpp" and not WHISPERCPP_USE_PROMPT:
        return None
    opts = asr.get("options")
    if isinstance(opts, dict):
        flags = [opts[k] for k in ("initial_prompt", "hotwords", "prompt") if k in opts]
        if flags and not any(flags):
            return None
    return prompt


def drop_prompt_echo(words: list[dict], prompt: str | None) -> tuple[list[dict], int]:
    """Remove runs that repeat the whole glossary prompt verbatim (a known hallucination in
    silence) when some word of the run has p < ECHO_MAX_P: an echo is written with near-zero
    confidence on its first word, while a speaker who really says the whole glossary in order
    (a one-term glossary of three words or more, every time it is mentioned) is heard with
    confidence and kept. A word without p counts as confident. Returns (words, number of words
    removed)."""
    if not prompt:
        return words, 0
    ptoks = textnorm.norm_tokens(prompt)
    m = len(ptoks)
    if m < 3:
        return words, 0
    stream = textnorm.norm_stream([w["w"] for w in words])
    toks = [t for t, _, _ in stream]
    drop: set[int] = set()
    i = 0
    while i + m <= len(toks):
        if toks[i:i + m] == ptoks:
            run = range(stream[i][1], stream[i + m - 1][2] + 1)
            if any((words[k].get("p") if words[k].get("p") is not None else 1.0) < ECHO_MAX_P
                   for k in run):
                drop.update(run)
            i += m
        else:
            i += 1
    if not drop:
        return words, 0
    return [w for k, w in enumerate(words) if k not in drop], len(drop)


def drop_echo_fragments(words: list[dict], prompt: str | None) -> tuple[list[dict], int]:
    """Re-listen slices only: remove a run of words that spells a glossary term (or a contiguous
    part of one) when any of its words has p < ECHO_MAX_P. Given a short slice of breath or filler, Whisper can write part
    of the prompt (one term, at p 0.00) instead of the full echo drop_prompt_echo catches; a
    term really spoken with the prompt in context is heard with confidence. Returns (words,
    number of words removed)."""
    if not prompt or not words:
        return words, 0
    terms = [textnorm.norm_tokens(t.strip()) for t in prompt.rstrip(".").split(",")]
    vocab = {tok for t in terms for tok in t}
    pairs = {(t[i], t[i + 1]) for t in terms for i in range(len(t) - 1)}
    if not vocab:
        return words, 0
    stream = textnorm.norm_stream([w["w"] for w in words])
    toks: dict[int, list[str]] = {}
    for tok, a, b in stream:
        for k in range(a, b + 1):
            toks.setdefault(k, []).append(tok)
    in_term = [bool(toks.get(k)) and all(t in vocab for t in toks[k]) for k in range(len(words))]
    drop: set[int] = set()
    k = 0
    while k < len(words):
        if not in_term[k]:
            k += 1
            continue
        j = k               # one run = consecutive words that continue the same term
        while (j + 1 < len(words) and in_term[j + 1]
               and (toks[j][-1], toks[j + 1][0]) in pairs):
            j += 1
        if any((words[m].get("p") if words[m].get("p") is not None else 1.0) < ECHO_MAX_P
               for m in range(k, j + 1)):
            drop.update(range(k, j + 1))
        k = j + 1
    if not drop:
        return words, 0
    return [w for m, w in enumerate(words) if m not in drop], len(drop)


def index_words(words: list[dict]) -> list[dict]:
    """Final word dicts in schema order {"i", "w", "s", "e", "p", "sp"}."""
    return [{"i": k, "w": w["w"], "s": w["s"], "e": w["e"], "p": w.get("p"), "sp": w.get("sp")}
            for k, w in enumerate(words)]


def transcript_sha(words: Sequence[dict]) -> str:
    """Stable hash of [(w, s, e), ...]: the identity clips.json "transcript_sha" pins."""
    return env.stable_hash([[w["w"], w["s"], w["e"]] for w in words])


def text_sha(words: Sequence[dict]) -> str:
    """Stable hash of the word texts only: unchanged by re-timing (indices do not move)."""
    return env.stable_hash([w["w"] for w in words])


def carry_compatible(doc: dict, previous: dict | None) -> list[str]:
    """text_compatible_shas for a rewritten transcript: the previous sha256 (and its own list)
    when the word-text sequence is unchanged, else []."""
    if not previous or not isinstance(previous.get("words"), list):
        return []
    old = [w.get("w") for w in previous["words"]]
    if old != [w["w"] for w in doc["words"]]:
        return []
    out: list[str] = []
    for sha in list(previous.get("text_compatible_shas") or []) + [previous.get("sha256")]:
        if sha and sha != doc.get("sha256") and sha not in out:
            out.append(sha)
    return out


def carry_text_change(doc: dict, previous: dict | None) -> str | None:
    """text_changed_from for a rewritten transcript: the previous sha256 when the word texts
    differ from the previous transcript's (word indices may have moved), else whatever the
    previous transcript carried (a re-timing or an identical re-run keeps the mark). None for a
    first transcript. plan reads it for caption_only clips: one that uses word-index keys fails
    L03 until the clip pins the new transcript_sha."""
    if not previous or not isinstance(previous.get("words"), list):
        return None
    mark = previous.get("text_changed_from")
    mark = mark if isinstance(mark, str) and mark else None
    old = [w.get("w") if isinstance(w, dict) else None for w in previous["words"]]
    if old != [w["w"] for w in doc["words"]]:
        sha = previous.get("sha256")
        return sha if isinstance(sha, str) and sha else mark
    return mark


# ---------------------------------------------------------------------------
# Audio post-processing: gate, re-timing, gap fill
# ---------------------------------------------------------------------------


def _retime(words: list[dict], E: Any, fn: Callable) -> tuple[list[dict], int]:
    """Run an align re-timing function and remember each word's original times in "_rep"."""
    new, log = fn(words, E)
    for ent in log:
        w = new[ent["i"]]
        old = w.get("_rep")
        if old:
            w["_rep"] = {"tag": old["tag"] + "," + ent["tag"], "s0": old["s0"], "e0": old["e0"]}
        else:
            w["_rep"] = {"tag": ent["tag"], "s0": ent["s0"], "e0": ent["e0"]}
    return new, len(log)


def _monotonic(words: list[dict]) -> None:
    for k in range(1, len(words)):
        if words[k]["s"] < words[k - 1]["s"]:
            words[k]["s"] = words[k - 1]["s"]
        if words[k - 1]["e"] > words[k]["s"]:
            words[k - 1]["e"] = words[k]["s"]
        if words[k - 1]["e"] < words[k - 1]["s"]:
            words[k - 1]["e"] = words[k - 1]["s"]
    if words and words[-1]["e"] < words[-1]["s"]:
        words[-1]["e"] = words[-1]["s"]


def _cpp_trim_ends(words: list[dict], E: Any) -> None:
    """whisper.cpp DTW words end at the next word's start: cap each end at the end of the voiced
    stretch the word starts (quiet dips under TURN_PAUSE bridged), at least MIN_WORD_DUR. A word
    with no voiced audio before the next word keeps MIN_WORD_DUR."""
    from . import align
    for k, w in enumerate(words):
        lim = words[k + 1]["s"] if k + 1 < len(words) else w["e"]
        end = align.voiced_end(E, w["s"], lim, bridge=TURN_PAUSE)
        w["e"] = round(max(min(w["e"], end), min(w["s"] + MIN_WORD_DUR, lim)), 3)


def vtt_dropouts(words: Sequence[dict], cues: Sequence[dict] | None,
                 min_words: int = VTT_DROPOUT_WORDS, cover: float = VTT_DROPOUT_COVER) -> list[dict]:
    """Cues (with >= min_words words) whose span holds ASR words (by midpoint) for less than
    `cover` of the cue's word count -> [{"s", "e", "vtt_words", "asr_words"}]."""
    if not cues:
        return []
    mids = sorted((w["s"] + w["e"]) / 2.0 for w in words)
    out = []
    for c in cues:
        n = len((c.get("text") or "").split())
        if n < min_words:
            continue
        got = bisect_right(mids, c["e"]) - bisect_left(mids, c["s"])
        if got < cover * n:
            out.append({"s": round(c["s"], 3), "e": round(c["e"], 3), "vtt_words": n,
                        "asr_words": got})
    return out


def find_holes(words: Sequence[dict], E: Any, cues: Sequence[dict] | None = None) -> list[tuple]:
    """Voiced audio no word covers (>= HOLE_MIN_RUN, or >= HOLE_ABUT_RUN when it runs straight
    out of a word at speech level: a short phrase skipped right after a word) plus VTT dropout
    cues, merged when closer than HOLE_MERGE, long ones split into HOLE_SLICE pieces, at most
    HOLE_CAP (the longest)."""
    from . import align
    holes = list(align.uncovered_voiced(E, words, E.t0, E.end, min_run=HOLE_MIN_RUN,
                                        bridge=HOLE_BRIDGE, abut_run=HOLE_ABUT_RUN))
    holes += [(d["s"], d["e"]) for d in vtt_dropouts(words, cues)]
    merged: list[list[float]] = []
    for s, e in sorted(holes):
        if merged and s - merged[-1][1] < HOLE_MERGE:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    out = []
    for s, e in merged:
        if e - s > HOLE_SPLIT:
            t = s
            while t < e - 1e-6:
                out.append((round(t, 3), round(min(e, t + HOLE_SLICE), 3)))
                t += HOLE_SLICE
        else:
            out.append((round(s, 3), round(e, 3)))
    if len(out) > HOLE_CAP:
        out = sorted(sorted(out, key=lambda h: h[0] - h[1])[:HOLE_CAP])
    return out


def _tok(text: str) -> str:
    return textnorm.norm(text) or text.strip().lower()


def _pair_similar(xs: Sequence[str], ys: Sequence[str], xt: Sequence[float],
                  yt: Sequence[float]) -> list[tuple[int, int]]:
    """Monotonic greedy pairs (i, j) of tokens xs[i] ~ ys[j] heard within HOLE_SUB_DT s of each
    other (midpoints xt, yt) with character similarity >= HOLE_SUB_SIM: the same word heard
    slightly differently ("you've" ~ "you" 0.67; "studio" ~ "so" 0.5 is not)."""
    import difflib
    pairs: list[tuple[int, int]] = []
    j0 = 0
    for i, x in enumerate(xs):
        best, bj = 0.0, None
        for j in range(j0, len(ys)):
            if abs(xt[i] - yt[j]) > HOLE_SUB_DT:
                continue
            r = difflib.SequenceMatcher(None, x, ys[j]).ratio()
            if r > best:
                best, bj = r, j
        if bj is not None and best >= HOLE_SUB_SIM:
            pairs.append((i, bj))
            j0 = bj + 1
    return pairs


def _splice(words: list[dict], hs: float, he: float, a: float, b: float,
            new: list[dict]) -> int:
    """Merge the words heard on a re-listen of src [a, b] (gap [hs, he] plus context) into
    `words` (in place). Returns the number of words inserted.

    The slice words are aligned to the existing words over [a, b] by text, so a word the slice
    heard again (or heard slightly differently, "you" for "you've") is never added twice. An
    existing word the slice also heard takes the slice's times when it lies inside the gap, or
    when it sits in a run of at least two matched words and the slice hears it inside or across
    the gap, more than HOLE_MOVED s away (the long-form pass skipped the speech and stamped it
    onto that speech); the span it leaves joins the gap. A re-time that would cross a word that
    keeps its times is dropped. Slice words the transcript lacks are inserted between their text
    neighbours when their midpoint lies within HOLE_KEEP of the gap and they overlap no word of
    their own stretch by more than HOLE_OVERLAP."""
    import difflib
    INF = float("inf")
    starts = [w["s"] for w in words]
    i0 = bisect_left(starts, a)
    while i0 > 0 and words[i0 - 1]["e"] > a:
        i0 -= 1
    i1 = max(i0, bisect_right(starts, b))
    ctx = list(range(i0, i1))
    ct = [_tok(words[k]["w"]) for k in ctx]
    nt = [_tok(w["w"]) for w in new]
    ops = difflib.SequenceMatcher(None, ct, nt, autojunk=False).get_opcodes()

    def mid(w: dict) -> float:
        return (w["s"] + w["e"]) / 2.0

    def side(t: float) -> int:
        return -1 if t < hs else (1 if t > he else 0)

    def tag(w: dict, name: str) -> None:
        rep = w.get("_rep") or {"tag": "", "s0": w["s"], "e0": w["e"]}
        w["_rep"] = {"tag": (rep["tag"] + "," if rep["tag"] else "") + name,
                     "s0": rep["s0"], "e0": rep["e0"]}

    # 1. (existing word, slice word, run length) pairs, and slice words the transcript lacks
    pairs: list[tuple[int, dict, int]] = []
    cands: list[tuple[int, int, int, dict]] = []     # (insert before ctx[x], op x1, op x2, word)
    for op, x1, x2, y1, y2 in ops:
        if op == "equal":
            pairs += [(ctx[x1 + d], new[y1 + d], x2 - x1) for d in range(x2 - x1)]
        elif op == "insert":
            cands += [(x1, x1, x2, new[y]) for y in range(y1, y2)]
        elif op == "replace":
            nb, nc = x2 - x1, y2 - y1
            if nc < nb:
                continue                     # the slice heard fewer words here: keep them
            xt = [mid(words[ctx[x]]) for x in range(x1, x2)]
            yt = [mid(new[y]) for y in range(y1, y2)]
            if nc == nb:     # word for word: a substitution where heard at about the same time
                sub = [(d, d) for d in range(nb) if abs(xt[d] - yt[d]) <= HOLE_SUB_DT]
            else:
                sub = _pair_similar(ct[x1:x2], nt[y1:y2], xt, yt)
            pairs += [(ctx[x1 + i], new[y1 + j], 1) for i, j in sub]
            paired_x = {i for i, _j in sub}
            for j in range(nc):
                if any(jj == j for _i, jj in sub):
                    continue
                # between the pairs around it, then by time among the unpaired words there
                lo_i = max([i + 1 for i, jj in sub if jj < j], default=0)
                hi_i = min([i for i, jj in sub if jj > j], default=nb)
                before = sum(1 for i in range(lo_i, hi_i)
                             if i not in paired_x and xt[i] < yt[j])
                cands.append((x1 + lo_i + before, x1, x2, new[y1 + j]))
    # 2. re-times
    prop: dict = {}
    for k, nw, run in pairs:
        w = words[k]
        if abs(w["s"] - nw["s"]) <= 5e-4 and abs(w["e"] - nw["e"]) <= 5e-4:
            continue
        was, now = side(mid(w)), side(mid(nw))
        moved = (was != 0 and now != was and run >= 2 and abs(w["s"] - nw["s"]) > HOLE_MOVED
                 and a + HOLE_EDGE <= mid(nw) <= b - HOLE_EDGE)
        if was == 0 or moved:
            prop[k] = (nw["s"], nw["e"])
    # 3. keep the order: clip the previous word's end or push the next word's start when it has
    #    room, else drop that re-time
    while True:
        clip: dict = {}
        push: dict = {}
        bad = None
        for k in sorted(prop):
            ns, ne = prop[k]
            p, q = k - 1, k + 1
            if p >= 0 and p not in prop and words[p]["e"] > ns + 1e-6:
                if words[p]["s"] < ns - 0.03:
                    clip[p] = min(clip.get(p, INF), ns)
                else:
                    bad = k
                    break
            if q < len(words) and q not in prop and words[q]["s"] < ne - 1e-6:
                if words[q]["e"] - 0.03 > ne:
                    push[q] = max(push.get(q, -INF), ne)
                else:
                    bad = k
                    break
        if bad is None:
            break
        del prop[bad]
    lo_t, hi_t = hs, he
    for k, (ns, ne) in prop.items():
        w = words[k]
        lo_t, hi_t = min(lo_t, w["s"]), max(hi_t, w["e"])
        tag(w, "relisten")
        w["s"], w["e"] = ns, ne
    for k, t in clip.items():
        tag(words[k], "clip")
        words[k]["e"] = t
    for k, t in push.items():
        tag(words[k], "push")
        words[k]["s"] = t
    # 4. inserts
    inserts: dict = {}
    for x, x1, x2, w in cands:
        if not lo_t - HOLE_KEEP <= mid(w) <= hi_t + HOLE_KEEP:
            continue
        if any(min(w["e"], words[ctx[i]]["e"]) - max(w["s"], words[ctx[i]]["s"]) > HOLE_OVERLAP
               and ctx[i] not in prop for i in range(x1, x2)):
            continue
        pos = ctx[x] if x < len(ctx) else i1
        lo = words[pos - 1]["e"] if pos > 0 else -INF
        hi = words[pos]["s"] if pos < len(words) else INF
        s, e = w["s"], w["e"]
        if s < lo or e > hi:
            s, e = max(s, lo), min(e, hi)
            if e - s < 0.02:
                continue     # no room left between its text neighbours
        inserts.setdefault(pos, []).append(dict(w, s=round(s, 3), e=round(e, 3), _new=True))
    for pos in sorted(inserts, reverse=True):     # slice order within one position
        words[pos:pos] = inserts[pos]
    return sum(len(v) for v in inserts.values())


def _settle(words: list[dict], regions: Sequence[tuple[float, float]], E: Any) -> None:
    """After splicing: clamp ends, re-time the words within HOLE_RETIME of each region, make
    the times monotonic (in place)."""
    from . import align
    _clamp_ends(words)
    for rs, re_ in regions:
        idx = [k for k, w in enumerate(words)
               if w["e"] >= rs - HOLE_RETIME and w["s"] <= re_ + HOLE_RETIME]
        if idx:
            a, b = idx[0], idx[-1] + 1
            words[a:b], _ = _retime(words[a:b], E, align.repair_times)
    _monotonic(words)


def fill_holes(words: list[dict], E: Any, listen: Callable[..., Any], *,
               cues: Sequence[dict] | None = None, prompt: str | None = None,
               duration: float | None = None) -> tuple[list[dict], list[dict]]:
    """Re-listen to every gap find_holes reports and splice in the words heard there.

    `listen(a, b)` transcribes src [a, b] (the gap plus HOLE_PAD of context) on its own and
    returns raw words [[text, s, e, p]] on the src clock (or None when it cannot). The words
    heard are gated and re-timed, then merged by text with the words already there (see
    _splice): nothing is added twice, and new words keep their place in the sentence. The words
    around each splice are re-timed again.

    A glossary prompt can itself make Whisper skip a stretch (it writes a short phrase and jumps
    ahead). When `listen.can_retry` is true, voiced audio of at least HOLE_RETRY_MIN s that a gap
    still lacks words for is re-listened once more with `listen(a, b, plain=True)` (no prompt).
    Returns (words, gaps [{"s", "e", "n"}])."""
    from . import align
    holes = find_holes(words, E, cues)
    if not holes:
        return words, []
    end_t = duration if duration else E.end
    words = [dict(w) for w in words]

    def heard(hs: float, he: float, **kw: Any) -> int:
        a, b = max(0.0, hs - HOLE_PAD), min(end_t, he + HOLE_PAD)
        raw = listen(a, b, **kw)
        new, _ = normalize_words(raw or [], 0.0)
        if not kw.get("plain"):             # the plain retry ran without a prompt: nothing to echo
            new, _ = drop_prompt_echo(new, prompt)
            new, _ = drop_echo_fragments(new, prompt)
        new, _ = align.audio_gate(new, E)
        new, _ = align.repair_times(new, E)
        return _splice(words, hs, he, a, b, new) if new else 0

    results = [{"s": hs, "e": he, "n": heard(hs, he)} for hs, he in holes]
    if any(r["n"] for r in results) or any(w.get("_rep") for w in words):
        _settle(words, [(r["s"], r["e"]) for r in results if r["n"]], E)
    if getattr(listen, "can_retry", False):
        again = []
        for r in results:
            for rs, re_ in align.uncovered_voiced(E, words, r["s"], r["e"],
                                                  min_run=HOLE_RETRY_MIN, bridge=HOLE_BRIDGE):
                m = heard(rs, re_, plain=True)
                if m:
                    r["n"] += m
                    again.append((rs, re_))
        if again:
            _settle(words, again, E)
    return words, results


def late_starts(words: Sequence[dict], E: Any) -> tuple[list[dict], list[dict]]:
    """align.snap_back for any backend, up to align.SNAP_LATE_MAX: a word that starts a little
    after the sustained onset that ends a pause (a smaller model often stamps it late, so a cut
    there clips its first sound) moves back to the onset; a lip click is never that onset."""
    from . import align
    return align.snap_back(words, E, hi=align.SNAP_LATE_MAX, sustained=True)


def postprocess(words: list[dict], E: Any, *, backend: str | None = None, dtw: bool = False,
                listen: Callable | None = None, prompt: str | None = None,
                cues: Sequence[dict] | None = None, duration: float | None = None,
                gate: bool = True, repair: bool = True,
                shapes: Any = None) -> tuple[list[dict], dict]:
    """Gate, re-time and gap-fill normalized words with the energy track `E`.

    After the gap fill two more re-timing passes run (with `repair`): settle_bursts (unsure
    words that hold two bursts of sound: the start moves past a leading laugh or tail, or the
    word is marked for a false_start flag) and, with `shapes` (align.BandShapes of the same
    audio), align.repair_overlapped (a start stamped inside someone else's laugh moves to where
    the sound changes).

    Returns (words, info) with info = {"dropped": [words], "retimed": int, "gaps": [...],
    "spliced": int}. Words keep private "_rep" (original times), "_new" (spliced) and "_fs"
    (false start) markers until index_words."""
    from . import align
    info: dict = {"dropped": [], "retimed": 0, "gaps": [], "spliced": 0}
    if E is None or getattr(E, "n", 0) == 0:
        return words, info
    if backend == "whispercpp" and dtw:
        _cpp_trim_ends(words, E)
    if gate:
        words, info["dropped"] = align.audio_gate(words, E)
    if repair:
        words, info["retimed"] = _retime(words, E, align.repair_times)
        if backend == "whispercpp" and dtw:
            words, n = _retime(words, E, align.snap_back)
        else:
            words, n = _retime(words, E, late_starts)
        info["retimed"] += n
    if listen is not None:
        words, info["gaps"] = fill_holes(words, E, listen, cues=cues, prompt=prompt,
                                         duration=duration)
        info["spliced"] = sum(g["n"] for g in info["gaps"])
    if repair:
        _monotonic(words)
        info["retimed"] += settle_bursts(words, E, listen, prompt)
        if shapes is not None:
            words, n = _retime(words, E, lambda ws, en: align.repair_overlapped(ws, en, shapes))
            info["retimed"] += n
    _monotonic(words)
    for w in words:
        w["s"], w["e"] = round(w["s"], 3), round(w["e"], 3)
    return words, info


def _quiet_hub() -> None:
    """No Hugging Face download/progress bars for the many short re-listen calls."""
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    try:
        from huggingface_hub.utils import disable_progress_bars
        disable_progress_bars()
    except Exception:  # noqa: BLE001 - cosmetic only
        pass


class Relisten:
    """listen(a, b) for fill_holes: transcribes src [a, b] of a cached 16 kHz wav on its own with
    the full run's backend, model, language and glossary. Results are cached per slice under
    `cache_dir` (one json map per wav and ASR settings)."""

    def __init__(self, wav: str | os.PathLike, offset: float, asr: dict,
                 cache_dir: str | os.PathLike | None = None, progress: bool = False,
                 force: bool = False):
        self.wav = Path(wav)
        self.offset = float(offset or 0.0)
        self.backend = asr.get("backend")
        self.model = asr.get("model")
        lang = asr.get("language") or asr.get("language_requested")
        self.language = None if lang in (None, "", "auto") else lang
        self.glossary = list(asr.get("glossary") or [])
        self.prompt = build_prompt(self.glossary)
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self.progress = progress
        self.force = force              # ignore cached slice results (they are rewritten)
        self.wav_sha = asr.get("wav_sha")
        self._map: dict | None = None
        self._dirty = False
        self.failed: str | None = None
        self.calls = 0
        self.seconds = 0.0

    @property
    def can_retry(self) -> bool:
        """A second, prompt-free listen can differ from the first only when a prompt is sent."""
        return bool(self.prompt) and (self.backend != "whispercpp" or WHISPERCPP_USE_PROMPT)

    def _material(self, plain: bool = False) -> dict:
        return asr_key_material(self.backend, self.model, self.language,
                                [] if plain else self.glossary)

    def _cache_file(self) -> Path | None:
        if self.cache_dir is None:
            return None
        tag = env.stable_hash({"wav": self.wav_sha, **self._material()})[:24]
        return self.cache_dir / ("asr-slices-%s.json" % tag)

    def _load(self) -> dict:
        if self._map is None:
            if self.wav_sha is None:
                self.wav_sha = env.sha256_file(self.wav)
            self._map = {}
            p = self._cache_file()
            if p is not None and p.is_file() and not self.force:
                try:
                    data = env.read_json(p)
                    if isinstance(data.get("slices"), dict):
                        self._map = data["slices"]
                except env.HCError:
                    pass
        return self._map

    def __call__(self, a: float, b: float, plain: bool = False) -> list | None:
        """Raw words heard in src [a, b] (src clock). `plain`: without the glossary prompt."""
        if self.failed or self.backend not in RUNNERS:
            return None
        cache = self._load()
        wa = max(0.0, a - self.offset)
        wb = max(wa, b - self.offset)
        key = env.stable_hash({"slice": [round(wa, 2), round(wb, 2)], "wav": self.wav_sha,
                               **self._material(plain)})
        hit = cache.get(key)
        if hit is None:
            t0 = time.time()
            if not self.calls:
                _quiet_hub()
            try:
                with tempfile.TemporaryDirectory(prefix="hc-slice-") as td:
                    piece = cut_wav(self.wav, Path(td) / "slice.wav", wa, wb)
                    res = RUNNERS[self.backend](piece, self.model, self.language,
                                                None if plain else self.prompt, progress=False)
            except Exception as exc:  # noqa: BLE001 - a failed re-listen only leaves the gap flagged
                self.failed = str(exc)[:200] or type(exc).__name__
                env.log(STEP, "re-listening failed (%s); gaps stay flagged" % self.failed)
                return None
            self.calls += 1
            self.seconds += time.time() - t0
            hit = {"slice": [round(wa, 3), round(wb, 3)],
                   "words": [[w[0], w[1], w[2], w[3]] for w in res.get("words") or []]}
            cache[key] = hit
            self._dirty = True
        start = round(wa, 3) + self.offset
        return [[w[0], None if w[1] is None else w[1] + start,
                 None if w[2] is None else w[2] + start, w[3]] for w in hit["words"]]

    def save(self) -> None:
        p = self._cache_file()
        if p is not None and self._dirty and self._map is not None:
            env.write_json(p, {"wav": self.wav_sha, "material": self._material(),
                               "slices": self._map})
            self._dirty = False


# ---------------------------------------------------------------------------
# Flags (pure)
# ---------------------------------------------------------------------------


def _loop_min_repeats(size: int) -> int:
    return 2 if size >= 6 else (3 if size >= 3 else 4)


def _loop_evidence(words: Sequence[dict], i: int, size: int, reps: int,
                   silent: set | None) -> bool:
    """A loop needs one repeat with mean p < LOW_CONF_P or stamped in silence. Words without any
    probability (and no silence data) cannot be judged and count as evidence."""
    known = False
    for r in range(reps):
        a = i + r * size
        ps = [w.get("p") for w in words[a:a + size] if w.get("p") is not None]
        if ps:
            known = True
            if sum(ps) / len(ps) < LOW_CONF_P:
                return True
        if silent is not None and all(k in silent for k in range(a, a + size)):
            return True
    return not known and silent is None


def find_loops(words: Sequence[dict], min_repeats: int | None = None, max_n: int = LOOP_MAX_N,
               silent: set | None = None) -> list[dict]:
    """Repetition loops: the same 1..max_n-gram (normalized) repeated in a row: at least 4 times
    for n <= 2, 3 times for n 3-5, twice for n >= 6 (`min_repeats` overrides), with one repeat at
    mean p < 0.5 or in silence (`silent` = word indices with almost no voiced audio). Single
    "very", "really", "no" are emphasis, never loops.
    -> [{"a", "b", "kind": "repetition_loop", "n", "repeats", "text"}]"""
    toks = [textnorm.norm(w["w"]) for w in words]
    n = len(toks)
    flags = []
    i = 0
    while i < n:
        best = None
        for size in range(1, max_n + 1):
            need = min_repeats or _loop_min_repeats(size)
            if i + size * need > n:
                continue
            unit = toks[i:i + size]
            if not all(unit) or (size == 1 and unit[0] in LOOP_EXEMPT):
                continue
            reps, j = 1, i + size
            while j + size <= n and toks[j:j + size] == unit:
                reps += 1
                j += size
            if reps >= need and _loop_evidence(words, i, size, reps, silent):
                end = i + reps * size - 1
                if best is None or end > best[1]:
                    best = (i, end, size, reps)
        if best:
            a, b, size, reps = best
            flags.append({"a": a, "b": b, "kind": "repetition_loop", "n": size, "repeats": reps,
                          "text": " ".join(w["w"] for w in words[a:a + size])})
            i = b + 1
        else:
            i += 1
    return flags


def _sentence_initial(words: Sequence[dict], k: int) -> bool:
    if k == 0:
        return True
    prev = words[k - 1]["w"].strip()
    return textnorm.is_sentence_end(prev, words[k]["w"]) or prev.endswith((":", '"', "“"))


def low_confidence(words: Sequence[dict], threshold: float = LOW_CONF_P) -> list[dict]:
    """Likely names (capitalized, not sentence-initial, not "I") and numbers with p < threshold.
    Adjacent words merge into one flag. -> [{"a", "b", "kind": "low_confidence", "w", "p"}]"""
    def bare(w):
        return w["w"].strip().strip("\"'([{").rstrip(_TRAIL_PUNCT + ")]}")

    # A word the transcript also writes in lowercase is an ordinary word, not a name. ASR
    # capitalizes segment-initial words without a full stop before them ("cohort, Everyone").
    lower_seen = {b.lower() for b in map(bare, words) if b and b[0].islower()}
    hits = []
    for k, w in enumerate(words):
        p = w.get("p")
        if p is None or p >= threshold:
            continue
        raw = bare(w)
        if not raw:
            continue
        is_num = any(ch.isdigit() for ch in raw) or (
            raw.lower() not in ("one", "a") and textnorm.num_value(raw) is not None)
        first_word = raw.split("'")[0]
        is_name = (raw[0].isupper() and first_word.lower() != "i"
                   and raw.lower() not in lower_seen
                   and not textnorm.is_stopword(raw.lower())
                   and not _sentence_initial(words, k))
        if is_num or is_name:
            hits.append(k)
    flags: list[dict] = []
    for k in hits:
        if flags and flags[-1]["b"] == k - 1:
            f = flags[-1]
            f["b"] = k
            f["w"] += " " + words[k]["w"]
            f["p"] = round(min(f["p"], words[k]["p"]), 3)
        else:
            flags.append({"a": k, "b": k, "kind": "low_confidence", "w": words[k]["w"],
                          "p": round(words[k]["p"], 3)})
    return flags


def missing_glossary_terms(words: Sequence[dict], terms: Sequence[str]) -> list[str]:
    """Glossary terms that never occur (normalized, contiguous) in the transcript."""
    hay = " " + " ".join(t for t, _, _ in textnorm.norm_stream([w["w"] for w in words])) + " "
    out = []
    for term in terms:
        needle = " ".join(textnorm.norm_tokens(term))
        if needle and (" " + needle + " ") not in hay:
            out.append(term)
    return out


def time_span_words(words: Sequence[dict], s: float, e: float) -> tuple[int, int]:
    """(a, b) for a time-based flag: a = the last word before s, b = the first word after e
    (clamped to the transcript)."""
    n = len(words)
    if not n:
        return 0, 0
    starts = [w["s"] for w in words]
    a = bisect_left(starts, s) - 1
    while a >= 0 and words[a]["e"] > s + 1e-6 and a > 0 and words[a]["s"] >= s:
        a -= 1
    b = bisect_right(starts, e - 1e-9)
    return max(0, min(a, n - 1)), max(0, min(b, n - 1))


def gap_flags(words: Sequence[dict], E: Any, gaps: Sequence[dict],
              cues: Sequence[dict] | None = None) -> list[dict]:
    """Coverage flags after the gap fill: "respliced" for gaps that got words, "unworded_audio"
    for voiced audio still without words, "dropout" for VTT cues ASR still barely covers."""
    from . import align
    out = []
    for g in gaps:
        if g["n"]:
            a, b = time_span_words(words, g["s"], g["e"])
            out.append({"a": a, "b": b, "kind": "respliced", "s": g["s"], "e": g["e"], "n": g["n"]})
    if E is not None and getattr(E, "n", 0):
        for s, e in align.uncovered_voiced(E, words, E.t0, E.end, min_run=HOLE_MIN_RUN,
                                           bridge=HOLE_BRIDGE, abut_run=HOLE_ABUT_RUN):
            a, b = time_span_words(words, s, e)
            out.append({"a": a, "b": b, "kind": "unworded_audio", "s": s, "e": e})
    for d in vtt_dropouts(words, cues):
        a, b = time_span_words(words, d["s"], d["e"])
        out.append(dict({"a": a, "b": b, "kind": "dropout"}, **d))
    return out


def silent_words(words: Sequence[dict], E: Any) -> set:
    """Indices of words whose span is almost unvoiced (< SILENT_FRACTION voiced frames)."""
    from . import align
    if E is None or not words or getattr(E, "n", 0) == 0:
        return set()
    out = set()
    a, b = min(w["s"] for w in words), max(w["e"] for w in words)
    V = align.voiced_runs(E, a - 0.1, b + 0.1)
    vs = [r[0] for r in V]
    for k, w in enumerate(words):
        s, e = w["s"], w["e"]
        if e - s <= 0 or s < E.t0 or e > E.end:
            continue
        i = max(0, bisect_right(vs, s) - 1)
        tot = 0.0
        while i < len(V) and V[i][0] < e:
            tot += max(0.0, min(V[i][1], e) - max(V[i][0], s))
            i += 1
        if tot / (e - s) < SILENT_FRACTION:
            out.add(k)
    return out


def _letters(text: str) -> int:
    return max(1, sum(1 for ch in text or "" if ch.isalnum()))


def false_start_candidates(words: Sequence[dict], E: Any) -> list[dict]:
    """Words that may hold two attempts at a phrase folded into one: p < FS_LOW_P and the span
    holds two bursts of sound separated by a dip (align.burst_dips).
    -> [{"k": word index, "dip": (start, end)}] with the last such dip of each word."""
    from . import align
    out: list[dict] = []
    if E is None or getattr(E, "n", 0) == 0:
        return out
    for k, w in enumerate(words):
        p = w.get("p")
        s, e = w["s"], w["e"]
        if p is None or p >= FS_LOW_P or s < E.t0 or e > E.end:
            continue
        dips = align.burst_dips(E, s, e)
        if dips:
            out.append({"k": k, "dip": dips[-1]})
    return out


def _lead_in_start(words: Sequence[dict], k: int) -> int:
    """Index of the word a false-start re-listen starts on: the first word at least FS_LEAD s
    before word k, or the word after the last pause within FS_LEAD_MAX s when there is one."""
    s = words[k]["s"]
    first = None
    lo = k
    while lo > 0 and s - words[lo - 1]["s"] <= FS_LEAD_MAX:
        lo -= 1
        if s - words[lo]["s"] < FS_LEAD:
            continue
        if first is None:
            first = lo
        if lo == 0 or words[lo]["s"] - words[lo - 1]["e"] >= FS_LEAD_PAUSE:
            return lo
    return lo if first is None else first


def lead_in(words: Sequence[dict], k: int, dip: tuple[float, float], listen: Callable,
            E: Any = None, prompt: str | None = None) -> tuple[str, list[str]]:
    """Re-listen to the sound that leads into the dip inside word k (from a word boundary a
    second or more back, up to the middle of the dip) and compare it with the transcript.
    -> (verdict, words heard):

      "false_start"  more is heard there than the transcript has before word k: the attempt the
                     long-form pass folded into word k ("and after a" where the transcript has
                     "And after" and then a long "about"). The words heard after the last
                     transcript word are returned, at most one per FS_WORD_MIN s of the sound
                     before the dip (a short slice can end on a made-up tail).
      "late"         nothing more is heard (or only a filler): the sound before the dip is the
                     previous word's tail, a laugh, a breath, an "um"; word k starts after it.
      "own"          word k itself is heard before the dip.
      "unknown"      the re-listen is unavailable or heard nothing the transcript has."""
    import difflib
    from . import align
    j = _lead_in_start(words, k)
    a = max(0.0, words[j]["s"] - FS_LEAD_PAD)
    if j > 0:
        a = max(a, min(words[j - 1]["e"], words[j]["s"]))
    b = (dip[0] + dip[1]) / 2.0
    if b - a < 0.1:
        return "unknown", []
    raw = listen(a, b)
    if raw is None:
        return "unknown", []
    new, _ = normalize_words(raw, 0.0)
    new, _ = drop_prompt_echo(new, prompt)
    new, _ = drop_echo_fragments(new, prompt)
    if E is not None:
        new, _ = align.audio_gate(new, E)
    heard = textnorm.norm_stream([x["w"] for x in new])
    htoks = [t for t, _, _ in heard]
    expected = [t for t, _, _ in textnorm.norm_stream([x["w"] for x in words[j:k]])]
    cut = 0
    if expected:
        blocks = [m for m in difflib.SequenceMatcher(None, expected, htoks,
                                                     autojunk=False).get_matching_blocks() if m.size]
        if not blocks:
            return "unknown", []
        last = blocks[-1]
        # transcript words after the last match were heard differently: they are not extras
        cut = last.b + last.size + (len(expected) - last.a - last.size)
    extra = heard[cut:]
    if not extra or all(textnorm.is_filler(t) for t, _, _ in extra):
        return "late", []
    if [t for t, _, _ in extra] == textnorm.norm_tokens(words[k]["w"]):
        return "own", []
    idx = sorted({i for _t, x, y in extra for i in range(x, y + 1)})
    room = max(1, int(math.ceil((dip[0] - words[k]["s"]) / FS_WORD_MIN - 1e-9)))
    return "false_start", [new[i]["w"] for i in idx[:room]]


def burst_verdicts(words: Sequence[dict], E: Any, listen: Callable | None = None,
                   prompt: str | None = None) -> list[dict]:
    """false_start_candidates with what a re-listen of each lead-in says (see lead_in):
    [{"k", "dip", "verdict", "heard"}]. Without `listen` every verdict is "unknown". At most
    FS_CAP candidates (the least sure words) are judged."""
    cands = false_start_candidates(words, E)
    if len(cands) > FS_CAP:
        keep = sorted(cands, key=lambda c: (words[c["k"]].get("p") or 0.0, c["k"]))[:FS_CAP]
        cands = sorted(keep, key=lambda c: c["k"])
    for c in cands:
        c["verdict"], c["heard"] = (lead_in(words, c["k"], c["dip"], listen, E, prompt)
                                    if listen is not None else ("unknown", []))
    return cands


def _fs_flag(k: int, w: dict, dip_end: float, heard: Sequence[str] = ()) -> dict:
    f = {"a": k, "b": k, "kind": "false_start", "s": round(w["s"], 3), "e": round(dip_end, 3)}
    if heard:
        f["heard"] = " ".join(heard)
    return f


def _tag(w: dict, name: str) -> None:
    """Remember a word's times before a change (private "_rep", see _retime)."""
    rep = w.get("_rep") or {"tag": "", "s0": w["s"], "e0": w["e"]}
    w["_rep"] = {"tag": (rep["tag"] + "," if rep["tag"] else "") + name,
                 "s0": rep["s0"], "e0": rep["e0"]}


_VOWEL_RE = re.compile("[aeiouy\u00e0-\u00e5\u00e8-\u00ef\u00f2-\u00f6\u00f9-\u00fc\u00fd\u00ff]+")


def _syllables(text: str) -> int:
    """Rough vowel-group count of a Latin-script word ("that's" 1, "once" 1, "people" 2,
    "experienced" 4): a contraction is counted on its first part, a final silent "e" is not
    counted. 0 for a word with no Latin vowel (another script, digits)."""
    t = "".join(ch for ch in (text or "").lower().replace("\u2019", "'") if ch.isalpha() or ch == "'")
    t = t.strip("'")
    if t.endswith("n't"):
        t = t[:-3]
    t = t.split("'")[0]
    n = len(_VOWEL_RE.findall(t))
    if n > 1 and t.endswith("e") and not t.endswith(("le", "ee", "ie", "ye", "oe", "ue")):
        n -= 1
    return n


def _heard_first(words: Sequence[dict], k: int, t: float, listen: Callable, E: Any = None,
                 prompt: str | None = None) -> bool:
    """True when a re-listen that starts just before `t` (and runs FS_TAIL s on, to a word end)
    hears word k first: the whole word lies after `t`."""
    from . import align
    j = k
    while j + 1 < len(words) and words[j]["e"] - t < FS_TAIL:
        j += 1
    raw = listen(max(0.0, t - FS_TAIL_PAD), words[j]["e"] + FS_LEAD_PAD)
    if raw is None:
        return False
    new, _ = normalize_words(raw, 0.0)
    new, _ = drop_prompt_echo(new, prompt)
    new, _ = drop_echo_fragments(new, prompt)
    if E is not None:
        new, _ = align.audio_gate(new, E)
    own = textnorm.norm_tokens(words[k]["w"])
    return bool(own) and textnorm.norm_tokens([x["w"] for x in new])[:len(own)] == own


def settle_bursts(words: list[dict], E: Any, listen: Callable | None = None,
                  prompt: str | None = None) -> int:
    """Act on burst_verdicts, in place. Returns the number of starts moved.

      * "false_start" and "unknown": the word is marked (private "_fs") for a false_start flag.
      * "late": the start moves to the end of the dip (timing tag "dip"; the end is kept) when
        all of this holds: the word has one syllable (the dip inside a longer word can be its
        own stop consonant: "ex-perienced"); sound runs into the stamped start (a burst that
        starts out of silence may be the word's own first consonant); and a second re-listen
        from the end of the dip hears the word first."""
    from . import align
    moved = 0
    for c in burst_verdicts(words, E, listen, prompt):
        k, end = c["k"], c["dip"][1]
        w = words[k]
        if c["verdict"] in ("false_start", "unknown"):
            w["_fs"] = {"e": end, "heard": list(c["heard"])}
            continue
        if c["verdict"] != "late" or w["e"] - end < MIN_WORD_DUR - 1e-9 or _syllables(w["w"]) != 1:
            continue
        _j0, _quiet, voiced = align.frame_masks(E, w["s"] - 0.03, w["s"] - 0.001)
        if not len(voiced) or not bool(voiced.all()):
            continue
        if not _heard_first(words, k, end, listen, E, prompt):
            continue
        _tag(w, "dip")
        w["s"] = round(end, 3)
        moved += 1
    return moved


def repeated_attempts(words: Sequence[dict], skip: Iterable[tuple[int, int]] = ()) -> list[dict]:
    """Visible repeats ("I was, I was", "the the": the same 1 to 3 words twice in a row) that
    read as a restart: the first attempt ends in a comma or a dash (the model's mark for a
    restart), the repeat is one function word ("the the"), or one word lasts implausibly long
    for its letters (more than 0.30 + 0.08 * letters s: a further attempt or a hesitation is
    folded into it). A single "very", "really" or "no" repeated for emphasis is never listed,
    nor a repeat inside a `skip` (a, b) index range. Flagged so `hc plan` (L15) and the by-ear
    checklist name it: the editor opens after the first attempt or cuts it out.
    -> false_start flags: a..b = both attempts, s = the first attempt's start, e = the second's."""
    from . import align
    toks = [textnorm.norm(w["w"]) for w in words]
    taken = sorted(skip)
    n = len(toks)
    out: list[dict] = []
    k = 0
    while k < n:
        hit = 0
        for size in (3, 2, 1):
            if k + 2 * size > n:
                continue
            unit = toks[k:k + size]
            if not all(unit) or unit != toks[k + size:k + 2 * size]:
                continue
            if size == 1 and unit[0] in LOOP_EXEMPT:
                continue
            if any(a <= k + 2 * size - 1 and b >= k for a, b in taken):
                continue
            restart = str(words[k + size - 1]["w"]).rstrip().endswith(RESTART_MARKS)
            stumble = size == 1 and unit[0] in STUMBLE_WORDS
            if restart or stumble or any(
                    x["e"] - x["s"] > align.OVERLONG_BASE + align.OVERLONG_PER_LETTER * _letters(x["w"])
                    for x in words[k:k + 2 * size]):
                hit = size
                break
        if hit:
            out.append({"a": k, "b": k + 2 * hit - 1, "kind": "false_start",
                        "s": round(words[k]["s"], 3), "e": round(words[k + hit]["s"], 3)})
            k += 2 * hit
        else:
            k += 1
    return out


def false_starts(words: Sequence[dict], E: Any, listen: Callable | None = None,
                 prompt: str | None = None, skip: Iterable[tuple[int, int]] = ()) -> list[dict]:
    """Likely false starts and repeated attempts -> flags kind "false_start" {a, b, s, e}.

    The long-form pass writes "And after about a decade" for "and after a w... after about a
    decade": the abandoned attempt has no words of its own, it sits inside one long, unsure
    word. Two kinds are flagged; the words are never rewritten:

      * folded: a candidate word (see false_start_candidates). With `listen` (transcribe.Relisten)
        the lead-in up to the dip is transcribed on its own and the word is flagged only when
        more is heard there than the transcript has; those words are kept in "heard". Without
        it (or when it cannot tell) the candidate is flagged as it is. a = b = the word,
        s = its start, e = the end of the dip (where the attempt that was kept begins).
      * visible: see repeated_attempts.

    build_transcript gets the folded kind from postprocess (settle_bursts), which also moves
    the starts of candidates that turn out to begin after their dip."""
    flags = [_fs_flag(c["k"], words[c["k"]], c["dip"][1], c["heard"])
             for c in burst_verdicts(words, E, listen, prompt)
             if c["verdict"] in ("false_start", "unknown")]
    taken = [(f["a"], f["b"]) for f in flags] + list(skip)
    return sorted(flags + repeated_attempts(words, taken), key=lambda f: (f["a"], f["b"]))


def timing_flags(words: Sequence[dict], E: Any, shapes: Any = None) -> list[dict]:
    """Words whose start is a guess -> flags kind "timing_uncertain" {a, b}: sound that is not
    transcribed (a laugh, crosstalk) runs straight into them at speech level and nothing marks
    where the word begins inside it (align.uncertain_starts). Adjacent words share one flag."""
    from . import align
    if E is None or getattr(E, "n", 0) == 0 or not words:
        return []
    out: list[dict] = []
    for k in align.uncertain_starts(words, E, shapes):
        if out and out[-1]["b"] == k - 1:
            out[-1]["b"] = k
        else:
            out.append({"a": k, "b": k, "kind": "timing_uncertain"})
    return out


# ---------------------------------------------------------------------------
# Speakers from WebVTT (pure)
# ---------------------------------------------------------------------------

_VTT_TIME_RE = re.compile(r"(?:(\d+):)?(\d{1,2}):(\d{2})[.,](\d{1,3})")
_VTT_CUE_RE = re.compile(r"^\s*(\S+)\s+-->\s+(\S+)")
_VTT_VOICE_RE = re.compile(r"<v(?:\.[^\s>]*)?\s+([^>]+)>")
_VTT_TAG_RE = re.compile(r"<[^>]*>")
_VTT_NAME_RE = re.compile(r"^\s*([^:<>]{1,60}?)\s*:\s+(\S.*)$")
_VTT_DASH_RE = re.compile("^[-\u2013\u2014]+\\s*")


def _vtt_time(text: str) -> float | None:
    m = _VTT_TIME_RE.fullmatch(text.strip())
    if not m:
        return None
    h = int(m.group(1) or 0)
    frac = m.group(4)
    return h * 3600 + int(m.group(2)) * 60 + int(m.group(3)) + int(frac) / (10 ** len(frac))


def _line_speaker(line: str) -> tuple[str | None, str, str, bool]:
    """-> (candidate speaker, text after the name, whole clean text, from a <v> voice tag)."""
    vm = _VTT_VOICE_RE.search(line)
    if vm:
        text = _VTT_DASH_RE.sub("", _VTT_TAG_RE.sub("", line).strip())
        return vm.group(1).strip(), text, text, True
    clean = _VTT_DASH_RE.sub("", _VTT_TAG_RE.sub("", line).strip())
    nm = _VTT_NAME_RE.match(clean)
    if nm:
        name = nm.group(1).strip()
        if (name and not re.fullmatch(r"[\d\s.:,]+", name) and len(name.split()) <= 6
                and name.lower() not in ("note", "notes") and not name.lower().startswith("http")):
            return name, nm.group(2).strip(), clean, False
    return None, clean, clean, False


def parse_vtt(text: str) -> list[dict]:
    """WebVTT (or SRT) cues -> [{"s", "e", "sp", "text"}] sorted by s.

    Speakers come from <v Name> voice tags, or from a "Name: text" prefix (a leading "-" or dash
    is ignored) when that name starts at least 2 cues or most cues carry a name, so a sentence
    such as "Step 1: sell." in an unlabelled file stays text. A cue holding lines from several
    speakers is split in time by text length."""
    raw_cues: list[tuple[float, float, list]] = []
    blocks = re.split(r"\r?\n\s*\r?\n", text.replace("\ufeff", "").replace("\x00", ""))
    for block in blocks:
        lines = [ln for ln in block.splitlines() if ln.strip()]
        for k, line in enumerate(lines):
            m = _VTT_CUE_RE.match(line)
            if not m:
                continue
            s, e = _vtt_time(m.group(1)), _vtt_time(m.group(2))
            if s is None or e is None or e < s:
                break
            body = [_line_speaker(b) for b in lines[k + 1:]]
            body = [x for x in body if x[2]]
            if body:
                raw_cues.append((s, e, body))
            break
    counts: dict = {}
    named = 0
    for _s, _e, body in raw_cues:
        names = {x[0] for x in body if x[0]}
        named += bool(names)
        for nm in names:
            counts[nm] = counts.get(nm, 0) + 1
    most = raw_cues and named > len(raw_cues) / 2.0
    cues: list[dict] = []
    for s, e, body in raw_cues:
        parts: list[list] = []  # [speaker, text]
        for name, after, whole, voice in body:
            ok = name is not None and (voice or counts.get(name, 0) >= 2 or most)
            sp, txt = (name, after) if ok else (None, whole)
            if not txt:
                continue
            if sp is None and parts:
                parts[-1][1] += " " + txt
            elif parts and sp == parts[-1][0]:
                parts[-1][1] += " " + txt
            else:
                parts.append([sp, txt])
        if not parts:
            continue
        total = sum(max(1, len(t)) for _, t in parts)
        t0 = s
        for sp, txt in parts:
            d = (e - s) * max(1, len(txt)) / total
            cues.append({"s": round(t0, 3), "e": round(t0 + d, 3), "sp": sp, "text": txt})
            t0 += d
    cues.sort(key=lambda c: (c["s"], c["e"]))
    return cues


def _clause_end(text: str) -> bool:
    t = text.strip().rstrip(_CLOSERS)
    return bool(t) and (t[-1] in ",;:" or t.endswith(("--", "\u2013", "\u2014")))


def _turn_ok(words: Sequence[dict], k: int) -> bool:
    """True when a speaker change between words k-1 and k sits on a sentence end or a pause."""
    if k <= 0 or k >= len(words):
        return True
    prev, nxt = words[k - 1], words[k]
    return (textnorm.is_sentence_end(prev["w"], nxt["w"])
            or nxt["s"] - prev["e"] >= TURN_PAUSE - 1e-9)


def assign_speakers(words: list[dict], cues: Sequence[dict], slack: float = VTT_SLACK) -> int:
    """Set w["sp"] from the cues: the speaker whose cues overlap the word the most (else the cue
    nearest its midpoint, within `slack` s). Words with no cue keep sp None. Then: gaps between
    two cues of one speaker take it, short crosstalk blips are relabelled, and each turn change
    that is not on a sentence end or a pause moves to one within one word when there is one.
    Returns the number of words given a speaker."""
    if not cues:
        return 0
    starts = [c["s"] for c in cues]
    max_dur = max(c["e"] - c["s"] for c in cues)
    cue_text: dict = {}            # word id -> normalized tokens of the cue that labelled it
    for w in words:
        s, e = w["s"], w["e"]
        m = (s + e) / 2.0
        hi = bisect_right(starts, max(e, m + slack))
        k = hi - 1
        votes: dict = {}
        top: dict = {}
        best, best_key = None, None
        while k >= 0 and cues[k]["s"] >= min(s, m - slack) - max_dur:
            c = cues[k]
            ov = min(e, c["e"]) - max(s, c["s"])
            if ov > 0:
                votes[c["sp"]] = votes.get(c["sp"], 0.0) + ov
                if ov > top.get(c["sp"], (0.0, None))[0]:
                    top[c["sp"]] = (ov, c)
            if c["e"] + slack >= m:
                inside = c["s"] <= m <= c["e"]
                dist = 0.0 if inside else min(abs(m - c["s"]), abs(m - c["e"]))
                if dist <= slack:
                    key = (0 if inside else 1, dist, abs(m - (c["s"] + c["e"]) / 2.0))
                    if best_key is None or key < best_key:
                        best, best_key = c, key
            k -= 1
        if votes:
            sp = max(votes, key=lambda x: (votes[x], best is not None and best["sp"] == x))
            best = top[sp][1]
        w["sp"] = best["sp"] if best else None
        if best is not None:
            cue_text[id(w)] = set(textnorm.norm_tokens(best.get("text") or ""))
    # A run of words in a gap between two cues of the SAME speaker belongs to that speaker
    # (caption files drop short words between cues). Between different speakers it stays None.
    k = 0
    n = len(words)
    while k < n:
        if words[k].get("sp"):
            k += 1
            continue
        j = k
        while j < n and not words[j].get("sp"):
            j += 1
        before = words[k - 1]["sp"] if k > 0 else None
        after = words[j]["sp"] if j < n else None
        if before and before == after and words[j]["s"] - words[k - 1]["e"] <= VTT_GAP_FILL:
            for t in range(k, j):
                words[t]["sp"] = before
        k = j
    _smooth_speaker_blips(words, cue_text)
    _snap_turns(words)
    return sum(1 for w in words if w.get("sp"))


def _snap_turns(words: list[dict]) -> int:
    """Move each mid-clause speaker change to a sentence end or pause one word away."""
    moved = 0
    n = len(words)
    for k in range(1, n):
        a, b = words[k - 1].get("sp"), words[k].get("sp")
        if not a or not b or a == b or _turn_ok(words, k) or _clause_end(words[k - 1]["w"]):
            continue
        left = _turn_ok(words, k - 1) and (k - 1 == 0 or words[k - 2].get("sp") == a)
        right = _turn_ok(words, k + 1) and (k + 1 >= n or words[k + 1].get("sp") == b)
        if left and right:
            gl = words[k - 1]["s"] - words[k - 2]["e"] if k >= 2 else 1e9
            gr = words[k + 1]["s"] - words[k]["e"] if k + 1 < n else 1e9
            if textnorm.is_sentence_end(words[k]["w"], words[k + 1]["w"] if k + 1 < n else None):
                gr = 1e9
            left = gl > gr
            right = not left
        if left:
            words[k - 1]["sp"] = b
            moved += 1
        elif right:
            words[k]["sp"] = a
            moved += 1
    return moved


def speaker_flags(words: Sequence[dict]) -> list[dict]:
    """speaker_uncertain flags for speaker changes still in mid-clause (not on a sentence end,
    a clause mark or a pause of TURN_PAUSE)."""
    out = []
    for k in range(1, len(words)):
        a, b = words[k - 1].get("sp"), words[k].get("sp")
        if a and b and a != b and not _turn_ok(words, k) and not _clause_end(words[k - 1]["w"]):
            out.append({"a": k - 1, "b": k, "kind": "speaker_uncertain", "sp": b, "alt": a})
    return out


def _smooth_speaker_blips(words: list[dict], cue_text: dict | None = None,
                          max_run: int = VTT_BLIP_WORDS) -> int:
    """A run of at most `max_run` words labelled B inside one sentence of speaker A ("he had
    [done for] a whole year") whose words B's cue does not even contain (Zoom's cue said "job.")
    is crosstalk or cue jitter over A, not B speaking those words: relabel it A. A real
    interjection ends its own sentence ("Yeah.") or matches its cue's text, and is kept."""
    cue_text = cue_text or {}
    n, fixed, k = len(words), 0, 1
    while k < n - 1:
        sp = words[k].get("sp")
        prev = words[k - 1].get("sp")
        if not sp or not prev or sp == prev:
            k += 1
            continue
        j = k
        while j < n and words[j].get("sp") == sp:
            j += 1
        in_cue = any(set(textnorm.norm_tokens(words[t]["w"])) & cue_text.get(id(words[t]), set())
                     for t in range(k, min(j, n)))
        if (j < n and j - k <= max_run and words[j].get("sp") == prev and not in_cue
                and not textnorm.is_sentence_end(words[k - 1]["w"], words[k]["w"])
                and not any(textnorm.is_sentence_end(words[t]["w"], words[t + 1]["w"])
                            for t in range(k, j))):
            for t in range(k, j):
                words[t]["sp"] = prev
                fixed += 1
        k = j
    return fixed


# ---------------------------------------------------------------------------
# Lines and markdown (pure)
# ---------------------------------------------------------------------------


def _glued(words: Sequence[dict], k: int) -> bool:
    """True when a line must not end after word k (the next token glues to it: "-old", ",000",
    "%", "'s", or word k is "$"/"(")."""
    if k + 1 >= len(words):
        return False
    return (textnorm._glues_prev(words[k + 1]["w"], words[k]["w"])
            or textnorm._joins_next(words[k]["w"]))


def build_lines(words: Sequence[dict], pause: float = LINE_PAUSE,
                max_words: int = LINE_MAX_WORDS, hard_words: int = LINE_HARD_WORDS) -> list[dict]:
    """Lines break at a sentence end, a pause >= `pause` or a speaker change. A line that
    reaches `max_words` ends at the last clause mark (",", ";", ":") of its last 10 words, or at
    the next one; at `hard_words` it ends at the largest gap of its last 10 words. A line never
    ends right before a glued token ("-old", ",000", "%"), except at a speaker change. Lines
    started by a cap carry "cont": True."""
    lines = []
    n = len(words)
    a = 0
    cont = False
    k = 0
    while k < n:
        natural = k == n - 1
        if not natural:
            w, nxt = words[k], words[k + 1]
            if nxt.get("sp") != w.get("sp"):
                natural = True
            else:
                natural = ((textnorm.is_sentence_end(w["w"], nxt["w"])
                            or nxt["s"] - w["e"] >= pause) and not _glued(words, k))
        end = k if natural else None
        cnt = k - a + 1
        if end is None and cnt >= max_words:
            if cnt == max_words:
                for j in range(k, max(a, k - LINE_LOOKBACK + 1) - 1, -1):
                    if _clause_end(words[j]["w"]) and not _glued(words, j):
                        end = j
                        break
            elif _clause_end(words[k]["w"]) and not _glued(words, k):
                end = k
            if end is None and cnt >= hard_words:
                cands = [j for j in range(max(a, k - LINE_LOOKBACK + 1), k + 1)
                         if not _glued(words, j)]
                if cands:
                    end = max(cands, key=lambda j: (words[j + 1]["s"] - words[j]["e"], j))
                else:
                    end = k
        if end is None:
            k += 1
            continue
        line = {"a": a, "b": end, "s": words[a]["s"], "e": words[end]["e"],
                "sp": words[a].get("sp")}
        if cont:
            line["cont"] = True
        lines.append(line)
        cont = not (natural and end == k)
        a = end + 1
        k = a
    return lines


def fmt_clock(t: float, hours: bool = False, decimals: int = 1) -> str:
    """mm:ss.s, or h:mm:ss.s when `hours` (used for sources of an hour or more)."""
    scale = 10 ** decimals
    total = int(round(max(0.0, float(t)) * scale))
    whole, frac = divmod(total, scale)
    h, rem = divmod(whole, 3600)
    m, s = divmod(rem, 60)
    sec = "%02d" % s + ("." + str(frac).zfill(decimals) if decimals else "")
    if hours or h:
        return "%d:%02d:%s" % (h, m, sec)
    return "%02d:%s" % (m, sec)


_MD_MARKS = {
    "repetition_loop": "repetition loop: transcript unreliable here",
    "no_audio": "no audible sound: likely an ASR hallucination",
    "speaker_uncertain": "speaker label uncertain",
}


def render_md(doc: dict, duration: float | None = None) -> str:
    """transcript.md text: header, then one line per transcript line. A line continued from
    the previous one (split by the length cap) starts with an ellipsis; gaps with sound but no
    words are listed in place, and possible false starts and uncertain word starts are noted
    right above the line that holds the word."""
    words = doc["words"]
    dur = duration if duration is not None else (words[-1]["e"] if words else 0.0)
    hours = (dur or 0) >= 3600
    lang = doc.get("language") or "auto"
    head = "# Transcript: %s (%s, %s words, %s %s, %s)" % (
        doc.get("media") or "media", fmt_clock(dur or 0, hours=False, decimals=0), format(len(words), ","),
        doc.get("backend"), model_short_name(doc.get("model") or ""), lang)
    out = [head,
           '# Lines: [mm:ss.s] #first-last SPEAKER: text. Indices are inclusive word indices for '
           'clips.json "words": [first, last].',
           "# Each space-separated token is one index (split tokens like \"$2 ,400\" count as two). "
           "Exact times: hc words P FIRST LAST."]
    flags = doc.get("flags") or []
    marked = [f for f in flags if f.get("kind") in _MD_MARKS]
    gaps = sorted((f for f in flags if f.get("kind") in ("unworded_audio", "dropout")),
                  key=lambda f: f.get("s", 0.0))
    # word notes (false starts, uncertain starts) go right above the line that holds the word
    firsts = [ln["a"] for ln in doc["lines"]]
    above: dict = {}
    for f in sorted((f for f in flags if f.get("kind") in _WORD_NOTES),
                    key=lambda f: (f.get("a", 0), f.get("b", 0))):
        note = _word_note(f, words, hours)
        if note:
            above.setdefault(max(0, bisect_right(firsts, f["a"]) - 1), []).append(note)
    gi = 0
    for li, ln in enumerate(doc["lines"]):
        while gi < len(gaps) and gaps[gi].get("s", 0.0) < ln["s"]:
            out.append(_gap_line(gaps[gi], hours))
            gi += 1
        out.extend(above.get(li, ()))
        text = " ".join(w["w"] for w in words[ln["a"]:ln["b"] + 1])
        if ln.get("cont"):
            text = "…" + text
        sp = ("%s: " % ln["sp"]) if ln.get("sp") else ""
        notes = []
        for f in marked:
            if f["a"] <= ln["b"] and f["b"] >= ln["a"] and _MD_MARKS[f["kind"]] not in notes:
                notes.append(_MD_MARKS[f["kind"]])
        mark = "".join("  <<%s>>" % x for x in notes)
        out.append("[%s] #%d-%d %s%s%s" % (fmt_clock(ln["s"], hours), ln["a"], ln["b"], sp, text, mark))
    while gi < len(gaps):
        out.append(_gap_line(gaps[gi], hours))
        gi += 1
    return "\n".join(out) + "\n"


_WORD_NOTES = ("false_start", "timing_uncertain")


def _word_note(f: dict, words: Sequence[dict], hours: bool) -> str | None:
    """transcript.md line for a false_start or timing_uncertain flag, or None when the flag
    does not point at words of this transcript."""
    a, b = f.get("a"), f.get("b")
    if not isinstance(a, int) or not isinstance(b, int) or not 0 <= a <= b < len(words):
        return None
    ref = "#%d" % a if a == b else "#%d-%d" % (a, b)
    if f.get("kind") == "timing_uncertain":
        what = ("the start of %s \"%s\" is uncertain: it begins under other sound (a laugh, "
                "crosstalk), check the caption onset" % (ref, words[a]["w"]))
        return "[%s] <<%s>>" % (fmt_clock(words[a]["s"], hours), what)
    s = float(f.get("s", words[a]["s"]))
    restart = fmt_clock(float(f.get("e", s)), hours, decimals=2)
    if a != b:
        what = ("repeated attempt %s (\"%s\"): one of these words holds more sound than its "
                "letters; the repeat starts at %s" % (
                    ref, " ".join(w["w"] for w in words[a:b + 1]), restart))
    elif f.get("heard"):
        what = ("possible false start inside %s \"%s\": a re-listen also hears \"%s\" there; "
                "the phrase restarts at %s" % (ref, words[a]["w"], f["heard"], restart))
    else:
        what = ("possible false start inside %s \"%s\": it holds two bursts of sound, the "
                "second starts at %s" % (ref, words[a]["w"], restart))
    return "[%s] <<%s>>" % (fmt_clock(s, hours), what)


def _gap_line(f: dict, hours: bool) -> str:
    s, e = float(f.get("s", 0.0)), float(f.get("e", 0.0))
    if f.get("kind") == "dropout":
        what = "the speakers VTT has %s words here that ASR did not transcribe" % f.get("vtt_words")
    else:
        what = "%.1f s of sound with no words (laughter, music or missed speech)" % (e - s)
    return "[%s] <<%s>>" % (fmt_clock(s, hours), what)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name("." + path.name + ".%d.tmp" % os.getpid())
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def transcribe_wav(wav: Path, backend: str = "auto", model: str | None = None,
                   language: str | None = None, glossary: Sequence[str] = (), *,
                   cache_dir: str | os.PathLike | None = None, force: bool = False,
                   progress: bool = True) -> dict:
    """Run (or load from cache) one backend on a 16 kHz mono wav.

    Returns {"backend", "model", "backend_version", "language", "language_requested",
    "glossary", "prompt", "options", "words": [[text, s, e, p], ...] (wav clock), "key",
    "wav_sha"}. The cache key is the wav sha plus asr_key_material; a result cached under the
    previous key layout is reused (and re-filed) when its recorded options equal today's.
    """
    requested = (backend or "auto").strip().lower()
    concrete = resolve_backend(requested)
    language = None if language in (None, "", "auto") else language
    glossary = list(glossary or [])
    prompt = build_prompt(glossary)
    attempts = [concrete]
    if BACKEND_ALIASES.get(requested, requested) == "auto" and concrete == "mlx":
        attempts.append("faster-whisper")
    wav_sha = None
    last_err: Exception | None = None
    for b in attempts:
        mdl = resolve_model(b, model if b == concrete else None)
        key = None
        cpath = None
        if cache_dir is not None:
            if wav_sha is None:
                wav_sha = env.sha256_file(wav)
            key = env.stable_hash({"v": ASR_CACHE_VERSION, "wav": wav_sha,
                                   **asr_key_material(b, mdl, language, glossary)})
            cpath = Path(cache_dir) / ("asr-%s.json" % key[:24])
            if not force:
                cached = _read_cached(cpath, key)
                if cached is None:
                    cached = _legacy_cached(Path(cache_dir), wav_sha, b, mdl, language, glossary,
                                            prompt, key, cpath, progress)
                if cached is not None:
                    if progress:
                        env.log(STEP, "using cached %s words (%s)" % (b, cpath.name))
                    cached.setdefault("wav_sha", wav_sha)
                    return cached
        t0 = time.time()
        try:
            res = RUNNERS[b](wav, mdl, language, prompt, progress=progress)
        except env.HCError as exc:
            if b != attempts[-1] and exc.code == 3:
                env.log(STEP, "%s unavailable (%s); falling back to faster-whisper" % (b, exc))
                last_err = exc
                continue
            raise
        data = {"key": key, "backend": b, "model": mdl, "backend_version": backend_version(b),
                "language": res.get("language"), "language_requested": language or "auto",
                "glossary": glossary, "prompt": prompt, "options": res.get("options") or {},
                "seconds": round(time.time() - t0, 1), "created": env.now_iso(),
                "words": [[w[0], w[1], w[2], w[3]] for w in res["words"]]}
        for extra in ("device", "compute_type"):
            if extra in res:
                data[extra] = res[extra]
        if cpath is not None:
            env.write_json(cpath, data)
        data["wav_sha"] = wav_sha
        return data
    raise last_err or env.HCError("no ASR backend available", code=3)


def _read_cached(path: Path, key: str) -> dict | None:
    if not path.is_file():
        return None
    try:
        cached = env.read_json(path)
    except env.HCError:
        return None
    return cached if isinstance(cached, dict) and cached.get("key") == key else None


def _legacy_cached(cache_dir: Path, wav_sha: str, backend: str, model: str,
                   language: str | None, glossary: list, prompt: str | None, key: str,
                   cpath: Path, progress: bool) -> dict | None:
    """A raw result filed under the v1 key (wav, backend, model, language, glossary) is still
    valid when its recorded runner options equal today's (mlx and faster-whisper; whisper.cpp
    words are parsed differently now). It is copied under the current key."""
    if backend not in ("mlx", "faster-whisper"):
        return None
    old_key = env.stable_hash({"v": 1, "wav": wav_sha, "backend": backend, "model": model,
                               "language": language or "auto", "glossary": glossary})
    cached = _read_cached(cache_dir / ("asr-%s.json" % old_key[:24]), old_key)
    if cached is None or cached.get("options") != runner_options(backend, model, prompt):
        return None
    cached = dict(cached, key=key)
    env.write_json(cpath, cached)
    if progress:
        env.log(STEP, "reusing words cached by an earlier version (same ASR options)")
    return cached


def _media_energy(path: Path, info: dict, cache_dir: str | os.PathLike | None) -> Any:
    from .audio import EnergyTrack
    key = env.fingerprint(path) if cache_dir is not None else None
    return EnergyTrack.from_media(path, t0=float(info.get("start") or 0.0), stream=info.get("index"),
                                  cache_dir=cache_dir, key=key)


def band_shapes(wav: str | os.PathLike, offset: float = 0.0) -> Any:
    """align.BandShapes of a transcript wav on the src clock (wav time + offset), or None when
    the wav cannot be read: the spectral passes are then skipped, nothing else changes."""
    from . import align
    try:
        return align.BandShapes.from_wav(wav, t0=float(offset or 0.0))
    except (OSError, EOFError, wave.Error, ValueError) as exc:
        env.log(STEP, "warning: no band shapes (%s): starts under overlapping sound are flagged, "
                "not moved" % (str(exc)[:120] or type(exc).__name__))
        return None


def transcribe_media(path: str | os.PathLike, backend: str = "auto", model: str | None = None,
                     language: str | None = None, glossary: Sequence[str] | str | None = None, *,
                     cache_dir: str | os.PathLike | None = None, force: bool = False,
                     progress: bool = False, repair: bool = True,
                     fill_holes: bool = True) -> list[dict]:
    """Transcribe any media file (e.g. a rendered clip for QA) -> normalized words
    [{"i", "w", "s", "e", "p", "sp": None}] on that file's own clock (format start = 0).

    Use the source transcript's backend, model, language and glossary to keep results comparable.
    With `repair` / `fill_holes` (default) the words pass the same audio gate, energy re-timing
    and gap re-listening as a project transcript, using the media's own energy track. With
    `cache_dir`, the wav, the energy track and the ASR results are cached there.
    """
    path = Path(path)
    if not path.is_file():
        raise env.HCError("media not found: %s" % path, code=1)
    name = (backend or "auto").strip().lower()
    if BACKEND_ALIASES.get(name, name) not in BACKENDS:
        if progress:
            env.log(STEP, "unknown backend %r (not produced by an ASR run); using auto" % backend)
        backend, model = "auto", None
    terms = glossary_terms(glossary) if isinstance(glossary, str) else list(glossary or [])
    info = probe_audio(path)
    with tempfile.TemporaryDirectory(prefix="hc-asr-") as td:
        if cache_dir is not None:
            wav = Path(cache_dir) / ("wav-%s.wav" % env.fingerprint(path))
        else:
            wav = Path(td) / "audio.wav"
        if not wav_ok(wav):
            extract_wav(path, wav, info["index"])
        res = transcribe_wav(wav, backend, model, language, terms, cache_dir=cache_dir,
                             force=force, progress=progress)
        words, _ = normalize_words(res["words"], info["start"])
        sent = prompt_sent(res)
        words, _ = drop_prompt_echo(words, sent)
        if repair or fill_holes:
            E = _media_energy(path, info, cache_dir)
            listen = Relisten(wav, info["start"], res, cache_dir, force=force) if fill_holes else None
            words, _info = postprocess(
                words, E, backend=res.get("backend"),
                dtw=bool((res.get("options") or {}).get("dtw_words")), listen=listen,
                prompt=sent, duration=info.get("duration"), gate=True,
                repair=repair, shapes=band_shapes(wav, info["start"]) if repair else None)
            if listen is not None:
                listen.save()
    return index_words(words)


def build_transcript(asr: dict, *, offset: float, media: str, fingerprint: str | None,
                     cues: Sequence[dict] | None = None, speakers_vtt: str | None = None,
                     duration: float | None = None, energy: Any = None,
                     listen: Callable | None = None, previous: dict | None = None,
                     progress: bool = False, shapes: Any = None) -> dict:
    """Assemble the transcript.json document from a transcribe_wav result.

    With `energy` (an audio.EnergyTrack on the src clock): words with no audible sound are
    dropped (events kind "no_audio"), times are re-timed from the audio (timing_repairs kind
    "energy"), and with `listen` (see Relisten) voiced audio no word covers is re-transcribed on
    its own and spliced in (flags "respliced", "unworded_audio", "dropout"). Likely false starts
    are flagged "false_start" (never rewritten). With `shapes` (align.BandShapes of the same
    audio) a start stamped inside someone else's laugh moves to where the sound changes; starts
    that stay a guess are flagged "timing_uncertain"."""
    events: list[dict] = []
    words, spread = normalize_words(asr["words"], offset, events=events)
    for ev in events:                   # bracketed or parenthesised runs set aside as sound events
        ev.setdefault("kind", "sound")
    for run_id, (a, b) in enumerate(spread):
        for k in range(a, b + 1):
            words[k]["_spread"] = run_id
    sent = prompt_sent(asr)
    words, echoed = drop_prompt_echo(words, sent)
    if echoed:
        env.log(STEP, "removed %d words that echoed the glossary prompt" % echoed)
    dtw = bool((asr.get("options") or {}).get("dtw_words"))
    words, info = postprocess(words, energy, backend=asr.get("backend"), dtw=dtw, listen=listen,
                              prompt=sent, cues=cues, duration=duration,
                              shapes=shapes)
    for w in info["dropped"]:
        events.append({"kind": "no_audio", "s": round(w["s"], 3), "e": round(w["e"], 3),
                       "text": w["w"], "p": w.get("p")})
    events.sort(key=lambda x: (x["s"], x["e"]))
    repairs: list[dict] = []
    runs: dict = {}
    folded: list[dict] = []
    for k, w in enumerate(words):
        if "_spread" in w:
            r = runs.setdefault(w["_spread"], [k, k])
            r[1] = k
        if "_fs" in w:
            folded.append(_fs_flag(k, w, w["_fs"]["e"], w["_fs"]["heard"]))
        rep = w.get("_rep")
        if rep and (abs(w["s"] - rep["s0"]) > 5e-4 or abs(w["e"] - rep["e0"]) > 5e-4):
            repairs.append({"i": k, "kind": "energy", "tag": rep["tag"], "s0": rep["s0"],
                            "e0": rep["e0"], "s": w["s"], "e": w["e"]})
    spread_entries = [{"a": a, "b": b, "kind": "spread"} for a, b in sorted(runs.values())
                      if b > a]
    words = index_words(words)
    if cues:
        assign_speakers(words, cues)
    silent = silent_words(words, energy) if energy is not None else None
    loops = find_loops(words, silent=silent)
    flags = loops + low_confidence(words)
    if energy is not None:
        flags += gap_flags(words, energy, info["gaps"], cues)
        flags += folded + repeated_attempts(words, [(f["a"], f["b"]) for f in loops + folded])
        flags += timing_flags(words, energy, shapes)
    if cues:
        flags += speaker_flags(words)
    flags.sort(key=lambda f: (f["a"], f["b"]))
    doc = {
        "version": 1,
        "backend": asr["backend"],
        "model": asr["model"],
        "backend_version": asr.get("backend_version"),
        "language": asr.get("language") or asr.get("language_requested"),
        "glossary": list(asr.get("glossary") or []),
        "audio_offset": round(float(offset), 6),
        "source_fingerprint": fingerprint,
        "media": media,
        "duration": duration,
        "sha256": transcript_sha(words),
        "text_sha": text_sha(words),
        "text_compatible_shas": [],
        "post_version": POST_VERSION if energy is not None else 0,
        "created": env.now_iso(),
        "speakers_vtt": speakers_vtt,
        "asr_options": asr.get("options") or {},
        "words": words,
        "lines": build_lines(words),
        "flags": flags,
        "timing_repairs": spread_entries + repairs,
        "events": events,
    }
    doc["text_compatible_shas"] = carry_compatible(doc, previous)
    changed_from = carry_text_change(doc, previous)
    if changed_from:
        doc["text_changed_from"] = changed_from
    doc["_post"] = info
    return doc


def sound_event_summary(events: Sequence[dict], limit: int = SOUND_REPORT_MAX) -> str | None:
    """One log line naming the text set aside as sound events ("(laughs)", "[Music]"), or None.

    Any parenthesised run of up to NONSPEECH_MAX_PAREN words and any bracketed run of up to
    NONSPEECH_MAX_BRACKET words is taken for a sound event whatever its confidence, so a spoken
    aside the backend wrote in parentheses would vanish from the words: printing the texts lets
    the user see one that was really said."""
    counts: dict[str, int] = {}
    for ev in events or []:
        if isinstance(ev, dict) and ev.get("kind") == "sound" and ev.get("text"):
            counts[str(ev["text"])] = counts.get(str(ev["text"]), 0) + 1
    if not counts:
        return None
    total = sum(counts.values())
    shown = ["%s x%d" % (t, n) if n > 1 else t for t, n in list(counts.items())[:limit]]
    more = len(counts) - len(shown)
    return ("set aside %d sound event%s written in brackets or parentheses (not words, never "
            "captioned): %s%s (transcript.json events; if one of them was really said, it is "
            "missing from the words)" % (total, "" if total == 1 else "s", ", ".join(shown),
                                         " (+%d more)" % more if more > 0 else ""))


def _report(doc: dict, glossary: Sequence[str], out_json: Path) -> None:
    words = doc["words"]
    flags = doc["flags"]
    info = doc.pop("_post", None) or {}
    loops = [f for f in flags if f["kind"] == "repetition_loop"]
    low = [f for f in flags if f["kind"] == "low_confidence"]
    hours = (doc.get("duration") or 0) >= 3600
    env.log(STEP, "%s words, %d lines, language %s, %s %s" % (
        format(len(words), ","), len(doc["lines"]), doc.get("language"), doc["backend"],
        model_short_name(doc["model"])))
    sounds = sound_event_summary(doc.get("events") or [])
    if sounds:
        env.log(STEP, sounds)
    if info.get("dropped"):
        env.log(STEP, "dropped %d words with no audible sound (likely ASR hallucinations; listed "
                "in transcript.json events)" % len(info["dropped"]))
    if info.get("retimed"):
        env.log(STEP, "re-timed %d words from the audio (timing_repairs)" % info["retimed"])
    gaps = info.get("gaps") or []
    if gaps:
        empty = sum(1 for g in gaps if not g["n"])
        env.log(STEP, "re-listened to %d gaps (%.1f s): spliced %d words; %d gaps still have no "
                "words (flagged)" % (len(gaps), sum(g["e"] - g["s"] for g in gaps),
                                     info.get("spliced", 0), empty))
    for f in loops:
        env.log(STEP, "repetition loop #%d-%d at %s (%r x%d): transcript unreliable there" % (
            f["a"], f["b"], fmt_clock(words[f["a"]]["s"], hours), f["text"], f["repeats"]))
    unworded = [f for f in flags if f["kind"] == "unworded_audio"]
    if unworded:
        env.log(STEP, "%d stretches of sound have no words (%.1f s; laughter, music or speech "
                "ASR missed): see transcript.md" % (len(unworded),
                                                    sum(f["e"] - f["s"] for f in unworded)))
    drops = [f for f in flags if f["kind"] == "dropout"]
    if drops:
        env.log(STEP, "%d speakers-VTT cues have words ASR did not transcribe (flag dropout)"
                % len(drops))
    starts = [f for f in flags if f["kind"] == "false_start"]
    if starts:
        env.log(STEP, "%d possible false starts or repeated attempts (flag false_start; the "
                "words are not changed, check before opening a clip on one):" % len(starts))
        for f in starts[:FS_REPORT_MAX]:
            env.log(STEP, "  #%s at %s%s" % (
                "%d" % f["a"] if f["a"] == f["b"] else "%d-%d" % (f["a"], f["b"]),
                fmt_clock(f["s"], hours),
                ": %r, a re-listen also hears %r there" % (
                    words[f["a"]]["w"], f["heard"]) if f.get("heard") else
                ": " + " ".join(w["w"] for w in words[f["a"]:f["b"] + 1])))
        if len(starts) > FS_REPORT_MAX:
            env.log(STEP, "  (+%d more in transcript.json flags)" % (len(starts) - FS_REPORT_MAX))
    unsure = [f for f in flags if f["kind"] == "timing_uncertain"]
    if unsure:
        env.log(STEP, "%d word starts are uncertain (they begin under a laugh or crosstalk; flag "
                "timing_uncertain): %s%s" % (
                    len(unsure), ", ".join("#%d at %s" % (f["a"], fmt_clock(words[f["a"]]["s"], hours))
                                           for f in unsure[:8]),
                    " ..." if len(unsure) > 8 else ""))
    if low:
        env.log(STEP, "low-confidence names/numbers (check spelling; add real names to glossary.txt):")
        for f in sorted(low, key=lambda f: f["p"])[:LOW_CONF_REPORT_MAX]:
            env.log(STEP, "  #%d %s (p %.2f) at %s" % (f["a"], f["w"], f["p"],
                                                    fmt_clock(words[f["a"]]["s"], hours)))
        if len(low) > LOW_CONF_REPORT_MAX:
            env.log(STEP, "  (+%d more in transcript.json flags)" % (len(low) - LOW_CONF_REPORT_MAX))
    # A short caption-only clip need not contain every project name: only the full transcript
    # reports glossary terms that never occur.
    if out_json.parent.name != "transcripts":
        for term in missing_glossary_terms(words, glossary):
            env.log(STEP, "glossary term never transcribed: %r" % term)
    if doc.get("speakers_vtt") and words:
        labelled = sum(1 for w in words if w.get("sp"))
        unsure = sum(1 for f in flags if f["kind"] == "speaker_uncertain")
        if not labelled:
            env.log(STEP, "warning: the speakers VTT matched no words (check that its clock matches)")
        else:
            env.log(STEP, "speaker labels: %.1f%% of words, %d uncertain turn boundaries "
                    "(flag speaker_uncertain)" % (100.0 * labelled / len(words), unsure))
    env.log(STEP, "wrote %s and %s" % (out_json.name, out_json.with_suffix(".md").name))


def _resolve_file(project: Any, value: str) -> Path:
    p = Path(value).expanduser()
    if p.is_absolute():
        return p
    if p.exists():
        return p.absolute()
    return project.root / p


def _project_energy(proj: Any, media: Path | None = None, info: dict | None = None,
                    fp: str | None = None) -> Any:
    """Energy track on the transcript's clock (the source, or a caption-only file), or None."""
    from .audio import EnergyTrack
    try:
        if media is None:
            return EnergyTrack.for_project(proj)
        return EnergyTrack.from_media(media, t0=float((info or {}).get("start") or 0.0),
                                      stream=(info or {}).get("index"),
                                      cache_dir=proj.cache_dir(), key=fp)
    except env.HCError as exc:
        env.log(STEP, "warning: no audio energy track (%s): word times are not re-timed" % exc)
        return None


def _flag_runs(idxs: Sequence[int], words: Sequence[dict], kind: str) -> list[dict]:
    out: list[dict] = []
    for k in sorted(idxs):
        if out and out[-1]["b"] == k - 1:
            out[-1]["b"] = k
            out[-1]["w"] += " " + words[k]["w"]
        else:
            out.append({"a": k, "b": k, "kind": kind, "w": words[k]["w"]})
    return out


def retime_doc(doc: dict, E: Any, shapes: Any = None) -> tuple[dict, int]:
    """Re-time an existing transcript's words from the audio (no ASR, no gate: indices and texts
    stay). With `shapes` (align.BandShapes) starts stamped inside someone else's laugh move to
    where the sound changes. Words with no audible sound get "no_audio" flags and starts that
    stay a guess "timing_uncertain" flags (both recomputed; other flags are kept).
    Returns (new doc, words moved)."""
    from . import align
    old_words = doc.get("words") or []
    words, log = align.repair_times([dict(w) for w in old_words], E)
    passes = [late_starts]
    if shapes is not None:
        passes.append(lambda ws, en: align.repair_overlapped(ws, en, shapes))
    for fn in passes:
        words, more = fn(words, E)
        first = {ent["i"]: ent for ent in log}
        for ent in more:
            old = first.get(ent["i"])
            if old:
                old.update(tag=old["tag"] + "," + ent["tag"], s=ent["s"], e=ent["e"])
            else:
                log.append(ent)
        log.sort(key=lambda ent: ent["i"])
    words = [{"i": k, "w": w["w"], "s": w["s"], "e": w["e"], "p": w.get("p"), "sp": w.get("sp")}
             for k, w in enumerate(words)]
    prev = {r["i"]: r for r in doc.get("timing_repairs") or [] if r.get("kind") == "energy"
            and isinstance(r.get("i"), int)}
    for ent in log:
        old = prev.get(ent["i"])
        if old:
            prev[ent["i"]] = dict(old, tag=old.get("tag", "") + "," + ent["tag"],
                                  s=ent["s"], e=ent["e"])
        else:
            prev[ent["i"]] = ent
    repairs = [r for r in doc.get("timing_repairs") or [] if r.get("kind") != "energy"]
    repairs += [prev[i] for i in sorted(prev)
                if abs(words[i]["s"] - prev[i]["s0"]) > 5e-4 or abs(words[i]["e"] - prev[i]["e0"]) > 5e-4]
    flags = [f for f in doc.get("flags") or []
             if f.get("kind") not in ("no_audio", "timing_uncertain")]
    flags += _flag_runs(align.unvoiced_words(words, E), words, "no_audio")
    flags += timing_flags(words, E, shapes)
    flags.sort(key=lambda f: (f.get("a", 0), f.get("b", 0)))
    new = dict(doc, words=words, flags=flags, timing_repairs=repairs, lines=build_lines(words),
               sha256=transcript_sha(words), text_sha=text_sha(words),
               post_version=POST_VERSION, retimed=env.now_iso())
    new["text_compatible_shas"] = carry_compatible(new, doc)
    return new, len(log)


def _cmd_retime(proj: Any, args: Any) -> int:
    clip = getattr(args, "clip", None)
    file_arg = getattr(args, "file", None)
    if clip:
        out_json = proj.root / "transcripts" / ("%s.json" % clip)
    else:
        out_json = proj.root / "transcript.json"
    if not out_json.is_file():
        raise env.HCError("no transcript to re-time at %s. Run: hc transcribe %s%s first"
                          % (proj.rel(out_json), proj.root,
                             " --file VIDEO --clip %s" % clip if clip else ""), code=1)
    doc = env.read_json(out_json)
    if clip:
        media = _resolve_file(proj, file_arg or doc.get("media") or "")
        if not media.is_file():
            raise env.HCError("media of caption-only clip %s not found (%s): pass --file VIDEO"
                              % (clip, media), code=1)
        info = probe_audio(media)
        fp = env.fingerprint(media)
        E = _project_energy(proj, media, info, fp)
        wav, offset = proj.cache_path("wav-%s.wav" % fp), info["start"]
        if E is not None and not wav_ok(wav):
            extract_wav(media, wav, info["index"])
    else:
        E = _project_energy(proj)
        wav, offset = ensure_wav(proj) if E is not None else (None, 0.0)
    if E is None:
        raise env.HCError("cannot re-time without the audio energy track", code=3)
    new, moved = retime_doc(doc, E, band_shapes(wav, offset))
    env.write_json(out_json, new)
    _write_text(out_json.with_suffix(".md"), render_md(new, new.get("duration")))
    nfl = sum(1 for f in new["flags"] if f.get("kind") == "no_audio")
    nun = sum(1 for f in new["flags"] if f.get("kind") == "timing_uncertain")
    env.log(STEP, "re-timed %d of %d words from the audio (indices and text unchanged)%s%s"
            % (moved, len(new["words"]), "; %d no_audio flags" % nfl if nfl else "",
               "; %d timing_uncertain flags" % nun if nun else ""))
    if new["sha256"] != doc.get("sha256"):
        env.log(STEP, "sha %s -> %s (the old sha stays valid for clips.json: text unchanged)"
                % (str(doc.get("sha256"))[:10], new["sha256"][:10]))
    env.log(STEP, "wrote %s and %s" % (out_json.name, out_json.with_suffix(".md").name))
    print(str(out_json))
    return 0


def _same_media(previous: dict, fp: str | None, src: dict | None = None) -> bool:
    """Was the `previous` transcript made from the media whose fingerprint is `fp`? Unknown
    (either fingerprint missing) counts as the same. A source that ingest only re-encoded to a
    constant frame rate (source.json ingest.cfr.from_fingerprint) keeps its timeline."""
    old = previous.get("source_fingerprint")
    if not old or not fp or old == fp:
        return True
    cfr = (src or {}).get("ingest")
    cfr = cfr.get("cfr") if isinstance(cfr, dict) else None
    return isinstance(cfr, dict) and cfr.get("from_fingerprint") == old


def cmd_transcribe(args: Any) -> int:
    """hc transcribe P [--backend] [--model] [--language] [--glossary] [--speakers-vtt]
    [--file VIDEO --clip ID] [--force] [--retime]"""
    from .project import Project

    proj = Project(args.project).require()
    force = bool(getattr(args, "force", False))
    language = getattr(args, "language", None)
    file_arg = getattr(args, "file", None)
    clip = getattr(args, "clip", None)
    if clip and not CLIP_ID_RE.match(clip):
        raise env.HCError("invalid clip id %r (lowercase letters, digits, hyphens; max 48)"
                          % clip, code=1)
    if getattr(args, "retime", False):
        if file_arg and not clip:
            raise env.HCError("--file goes with --clip (caption-only clip transcript)", code=1)
        return _cmd_retime(proj, args)
    terms = glossary_terms(getattr(args, "glossary", None), proj.glossary_path())

    if file_arg or clip:
        if not (file_arg and clip):
            raise env.HCError("--file and --clip go together (caption-only clip transcript)", code=1)
        media = _resolve_file(proj, file_arg)
        if not media.is_file():
            raise env.HCError("file not found: %s" % file_arg, code=1)
        info = probe_audio(media)
        fp = env.fingerprint(media)
        wav = proj.cache_path("wav-%s.wav" % fp)
        if force or not wav_ok(wav):
            env.log(STEP, "extracting 16 kHz mono audio from %s" % media.name)
            extract_wav(media, wav, info["index"])
        offset, duration = info["start"], info["duration"]
        out_json = proj.root / "transcripts" / ("%s.json" % clip)
        media_label = proj.rel(media)
        energy_args: tuple = (media, info, fp)
        src = None
    else:
        src = proj.source()
        fp = src.get("fingerprint")
        wav, offset = ensure_wav(proj, force=force)
        duration = _float(src.get("duration"), None)
        out_json = proj.root / "transcript.json"
        media_label = src.get("path") or proj.source_path().name
        energy_args = ()

    previous = None
    if out_json.is_file():
        try:
            previous = env.read_json(out_json)
        except env.HCError:
            previous = None

    vtt_arg = getattr(args, "speakers_vtt", None)
    vtt_path = None
    if vtt_arg and str(vtt_arg).strip().lower() in ("none", "off"):
        vtt_arg = None
        if previous:
            previous = dict(previous, speakers_vtt=None)
    if vtt_arg:
        vtt_path = _resolve_file(proj, vtt_arg)
        if not vtt_path.is_file():
            raise env.HCError("speakers VTT not found: %s" % vtt_arg, code=1)
    elif previous and previous.get("speakers_vtt") and Path(previous["speakers_vtt"]).is_file():
        if _same_media(previous, fp, src):
            vtt_path = Path(previous["speakers_vtt"])
            env.log(STEP, "reusing speakers from %s (pass --speakers-vtt to change)" % vtt_path)
        else:
            # a re-export can move the timeline: the old cues would label the wrong words
            env.log(STEP, "warning: the %s changed since the speakers VTT %s was used, so it is "
                    "not reused and the words get no speaker labels. Pass --speakers-vtt PATH "
                    "again (a VTT of this file), or --speakers-vtt none"
                    % ("file" if clip else "source", Path(previous["speakers_vtt"]).name))
            previous = dict(previous, speakers_vtt=None)
    cues = None
    if vtt_path is not None:
        cues = parse_vtt(read_text_file(vtt_path))
        if not cues:
            raise env.HCError("no timed cues found in %s" % vtt_path, code=1)
        if not any(c["sp"] for c in cues):
            env.log(STEP, "warning: no 'Name: text' speakers found in %s" % vtt_path.name)

    asr = transcribe_wav(wav, getattr(args, "backend", None) or "auto", getattr(args, "model", None),
                         language, terms, cache_dir=proj.cache_dir(), force=force, progress=True)
    energy = _project_energy(proj, *energy_args)
    listen = (Relisten(wav, offset, asr, proj.cache_dir(), progress=True, force=force)
              if energy is not None else None)
    t0 = time.time()
    doc = build_transcript(asr, offset=offset, media=media_label, fingerprint=fp, cues=cues,
                           speakers_vtt=str(vtt_path) if vtt_path else None, duration=duration,
                           energy=energy, listen=listen, previous=previous, progress=True,
                           shapes=band_shapes(wav, offset) if energy is not None else None)
    if listen is not None:
        listen.save()
        if listen.calls:
            env.log(STEP, "re-listening took %.1f s (%d model calls)" % (time.time() - t0,
                                                                          listen.calls))
    if not doc["words"]:
        env.log(STEP, "warning: no speech recognized")
    if previous and previous.get("sha256") and previous["sha256"] != doc["sha256"]:
        if doc["text_compatible_shas"]:
            env.log(STEP, "word times changed, text unchanged (sha %s -> %s): clips.json word "
                    "indices still hold" % (previous["sha256"][:10], doc["sha256"][:10]))
        elif clip:
            env.log(STEP, "transcript of clip %s changed (sha %s -> %s): its word indices may "
                    "have moved. Check the fix, emphasis and insert indices of this clip in "
                    "clips.json (hc words %s A B --clip %s). plan fails L03 for a caption_only "
                    "clip that uses them until its transcript_sha is set to the new sha256: %s"
                    % (clip, previous["sha256"][:10], doc["sha256"][:10], proj.root, clip,
                       doc["sha256"]))
        else:
            env.log(STEP, "transcript changed (sha %s -> %s): word indices in clips.json may have "
                    "drifted. plan flags drift only where clips.json pins transcript_sha (L03) or "
                    "pieces carry expect anchors (L02); check the other clips with hc words. The "
                    "new sha256 to pin: %s"
                    % (previous["sha256"][:10], doc["sha256"][:10], doc["sha256"]))
    out = {k: v for k, v in doc.items() if k != "_post"}
    env.write_json(out_json, out)
    _write_text(out_json.with_suffix(".md"), render_md(out, duration))
    _report(doc, terms, out_json)
    print(str(out_json))
    return 0
