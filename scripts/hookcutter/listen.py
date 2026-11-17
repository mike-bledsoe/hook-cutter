"""hc listen: transcribe one short stretch on its own, writing nothing into P.

The agent cannot hear, so it checks a seam, a flagged word or a stretch with no words by asking
the recogniser about that stretch alone. This command cuts START..END from the source (source
seconds) or from a clip's vertical render (`--render CLIP`, output seconds) into a temporary
folder outside P, transcribes it with the transcript's backend, model, language and glossary
(the settings QA's G5 re-transcribes a render with; `--backend` and `--model` override them),
prints the words on that clock and deletes the folder. Nothing is cached or written into P: no
`cache/`, no `transcripts/`, so a read-only reviewer can run it and nothing piles up.

A slice whose edge falls inside voiced audio makes the recogniser drop or invent the words at
that edge, so both edges are measured on the audio around the slice (the noise floor over the
planner's 20 s radius): an edge inside a voiced run (align.voiced_runs) is reported with the
nearest quiet edge outside the slice, LISTEN_MARGIN into the nearest pause (align.pauses, at
least 0.2 s), or the middle of a shorter pause.
"""

from __future__ import annotations

import json
import re
import shlex
import sys
import tempfile
import wave
from pathlib import Path
from typing import Any

from . import env

STEP = "listen"
SAMPLE_RATE = 16000
LISTEN_MAX = 120.0       # a slice is a short stretch: longer ones are hc transcribe's job
LIMIT_EPS = 1e-6         # times are typed to two decimals: 9.80..129.80 is 120 s, not over it
LISTEN_SHORT = 1.0       # slices shorter than this transcribe poorly (qa.md)
CONTEXT = 20.0           # audio read either side of the slice for the floor (audio.FLOOR_RADIUS)
EDGE_WIN = 0.02          # energy window around an edge, for the dB figure (plan.WIN_DB)
EDGE_LOOK = 0.1          # an edge is voiced when a voiced run within this of it covers it
LISTEN_MARGIN = 0.2      # a quiet edge sits this far into the pause next to the speech ...
LISTEN_REACH = 5.0       # ... found within this far outside the edge


def cut_audio(media: Any, a: float, b: float, out: Any, stream: Any = None,
              channels: int = 1) -> Path:
    """Audio [a, b) of `media` (seconds on its own clock, the clock the transcript and hc peek
    use) as a 16 kHz s16 wav whose sample 0 is time `a`: an audio stream that starts after `a`
    is padded with silence (aresample first_pts=0), and timestamp holes are filled (async=1) as
    in the transcript wav."""
    out = Path(out)
    smap = "0:%d" % int(stream) if stream is not None else "0:a:0"
    env.ff(["-ss", "%.6f" % float(a), "-i", str(media), "-map", smap, "-vn", "-sn", "-dn",
            "-af", "aresample=%d:async=1:first_pts=0" % SAMPLE_RATE, "-ac", str(int(channels)),
            "-t", "%.6f" % max(0.0, float(b) - float(a)), "-c:a", "pcm_s16le", "-f", "wav",
            str(out)])
    try:
        with wave.open(str(out), "rb") as w:
            frames = w.getnframes()
    except (OSError, EOFError, wave.Error):
        frames = 0
    if frames <= 0:
        raise env.HCError("no audio decoded from %s between %.2f and %.2f s" % (media, a, b), code=1)
    return out


def edge_report(E: Any, t: float, side: str, lo: float, hi: float) -> dict:
    """Is the slice edge at `t` inside voiced audio, and where is the nearest quiet edge?

    `side` is "start" or "end"; `lo`..`hi` is the file's own span. Returns {"t", "voiced",
    "over_floor_db", "quiet"}: `quiet` (only for a voiced edge) is the suggested edge outside the
    slice, LISTEN_MARGIN into the nearest pause before a start (after an end), the middle of a
    shorter pause, or None when no pause lies within LISTEN_REACH."""
    from . import align
    t = float(t)
    over = E.db_at(t, EDGE_WIN) - E.floor(t)
    runs = align.voiced_runs(E, t - EDGE_LOOK, t + EDGE_LOOK)
    voiced = any(s <= t < e for s, e in runs)
    quiet = None
    if voiced:
        if side == "start":
            found = align.pauses(E, max(lo, t - LISTEN_REACH), t)
            if found:
                qa, qb = found[-1]
                quiet = max((qa + qb) / 2.0, qb - LISTEN_MARGIN)
        else:
            found = align.pauses(E, t, min(hi, t + LISTEN_REACH))
            if found:
                qa, qb = found[0]
                quiet = min((qa + qb) / 2.0, qa + LISTEN_MARGIN)
    return {"t": round(t, 3), "voiced": bool(voiced), "over_floor_db": round(over, 1),
            "quiet": None if quiet is None else round(min(max(quiet, lo), hi), 2)}


def _render_file(proj: Any, clip: str) -> Path:
    """The clip's vertical render (renders/<clip>__vertical__<style>.mp4). Every style of a clip
    carries the same audio (QA fails styles that diverge), so the newest one is used."""
    from .transcribe import CLIP_ID_RE
    if not CLIP_ID_RE.match(clip):
        raise env.HCError("invalid clip id %r (lowercase letters, digits, hyphens; max 48)" % clip,
                          code=1)
    folder = proj.root / "renders"
    pat = re.compile(r"^%s__vertical__[^.]+\.mp4$" % re.escape(clip))
    found = [p for p in folder.iterdir() if pat.match(p.name) and p.is_file()] \
        if folder.is_dir() else []
    if not found:
        other = re.compile(r"^%s__landscape__[^.]+\.mp4$" % re.escape(clip))
        land = folder.is_dir() and any(other.match(p.name) for p in folder.iterdir())
        raise env.HCError("no vertical render of clip %s in %s%s: hc listen --render reads only "
                          "the vertical render. Run: hc render %s %s --formats vertical, or slice "
                          "the source (source seconds, no --render)"
                          % (clip, proj.rel(folder), " (only a landscape one)" if land else "",
                             shlex.quote(str(proj.root)), clip), code=1)
    found.sort(key=lambda p: (-p.stat().st_mtime, p.name))
    if len(found) > 1:
        env.log(STEP, "%s has %d vertical renders; listening to %s (every style has the same "
                "audio)" % (clip, len(found), found[0].name))
    return found[0]


def _asr_settings(proj: Any, clip: str | None, backend: str | None,
                  model: str | None) -> tuple[str, str | None, str | None, list]:
    """(backend, model, language, glossary): the transcript's own, as QA uses them, so the
    slice is heard the way the transcript was made. A backend given that differs from the
    transcript's takes its own default model unless --model names one. With no transcript (or
    one from no real backend) the defaults of hc transcribe apply."""
    from . import transcribe as T
    doc: dict = {}
    path = proj.transcript_path(clip)
    if path.is_file():
        try:
            doc = env.read_json(path)
        except env.HCError:
            doc = {}
    name = str(doc.get("backend") or "").strip().lower()
    known = T.BACKEND_ALIASES.get(name, name) in T.BACKENDS
    if known:
        same = backend is None or T.resolve_backend(backend) == T.resolve_backend(name)
        b = backend or name
        m = model or (doc.get("model") if same else None)
        terms = list(doc.get("glossary") or [])
    else:
        b, m = backend or "auto", model
        terms = T.glossary_terms(None, proj.glossary_path())
    lang = doc.get("language") if known else None
    lang = None if lang in (None, "", "auto") else str(lang)
    return b, m, lang, terms


def _rerun(proj: Any, a: float, b: float, args: Any) -> str:
    cmd = "hc listen %s %.2f %.2f" % (shlex.quote(str(proj.root)), a, b)
    for flag in ("render", "backend", "model"):
        v = getattr(args, flag, None)
        if v:
            cmd += " --%s %s" % (flag, shlex.quote(str(v)))
    return cmd


def listen(proj: Any, start: float, end: float, args: Any) -> dict:
    """Cut, check and transcribe one slice; returns the report (see cmd_listen)."""
    from . import transcribe as T
    from .audio import EnergyTrack, decode_energy
    clip = getattr(args, "render", None)
    start, end = float(start), float(end)
    if end <= start:
        raise env.HCError("END must be after START (%.2f to %.2f)" % (start, end), code=1)
    if end - start > LISTEN_MAX + LIMIT_EPS:
        raise env.HCError("a slice is at most %.0f s (this one is %.1f s): for a longer stretch "
                          "run hc transcribe" % (LISTEN_MAX, end - start), code=1)
    if clip:
        media, clock = _render_file(proj, clip), "out"
        info = T.probe_audio(media)
        stream, dur = info["index"], info.get("duration")
    else:
        src = proj.source()
        audio = src.get("audio") or {}
        if not audio:
            raise env.HCError("the source has no audio stream; nothing to listen to", code=1)
        media, clock = proj.source_path(), "src"
        stream = audio.get("index")
        dur = src.get("duration") or T.probe_audio(media).get("duration")
    dur = float(dur) if isinstance(dur, (int, float)) and dur > 0 else None
    label = proj.rel(media)
    notes = []
    if dur is not None:
        if start >= dur:
            raise env.HCError("START %.2f is past the end of %s (%.2f s long)"
                              % (start, label, dur), code=1)
        if end > dur:
            notes.append("END %.2f is past the end of %s: the slice ends at %.2f"
                         % (end, label, dur))
            end = dur
    backend, model, language, glossary = _asr_settings(proj, clip, getattr(args, "backend", None),
                                                       getattr(args, "model", None))
    concrete = T.resolve_backend(backend)
    shown_model = T.model_short_name(T.resolve_model(concrete, model))
    env.log(STEP, "%s %.2f-%.2f of %s (%.2f s) with %s %s, in a temporary folder (nothing is "
            "written into P)" % (clock, start, end, label, end - start, concrete, shown_model))
    with tempfile.TemporaryDirectory(prefix="hc-listen-") as td:
        lo = max(0.0, start - CONTEXT)
        hi = end + CONTEXT if dur is None else min(dur, end + CONTEXT)
        ctx = cut_audio(media, lo, hi, Path(td) / "context.wav", stream, channels=2)
        E = EnergyTrack(decode_energy(ctx), t0=lo)
        top = dur if dur is not None else E.end
        edges = {"start": edge_report(E, start, "start", 0.0, top),
                 "end": edge_report(E, end, "end", 0.0, top)}
        piece = cut_audio(media, start, end, Path(td) / "slice.wav", stream)
        T._quiet_hub()      # one short call: no model download bars
        words = T.transcribe_media(piece, backend, model, language, glossary, cache_dir=None,
                                   progress=False)
    out_words = [{"w": w["w"], "s": round(w["s"] + start, 3), "e": round(w["e"] + start, 3),
                  "p": w.get("p")} for w in words]
    warnings = list(notes)
    if end - start < LISTEN_SHORT:
        warnings.append("the slice is %.2f s: slices shorter than about a second transcribe "
                        "poorly (take 0.2 s of quiet margin each side)" % (end - start))
    for side, name, where in (("start", "START", "before"), ("end", "END", "after")):
        e = edges[side]
        if not e["voiced"]:
            continue
        msg = ("%s %.2f is inside voiced audio (%+.0f dB over the floor): the recogniser drops "
               "or invents words at an edge cut inside speech. " % (name, e["t"], e["over_floor_db"]))
        if e["quiet"] is not None:
            msg += "Nearest quiet %s: %.2f" % (side, e["quiet"])
        else:
            msg += ("No pause within %.0f s %s it: the speech runs on, so read the words at "
                    "this edge as unsure" % (LISTEN_REACH, where))
        warnings.append(msg)
    qs, qe = edges["start"]["quiet"], edges["end"]["quiet"]
    rerun = None
    if qs is not None or qe is not None:
        ra, rb = qs if qs is not None else start, qe if qe is not None else end
        if rb - ra > LISTEN_MAX + LIMIT_EPS:     # the quiet edges only widen it: hc listen
            # would refuse that re-run, so none is printed
            warnings.append("from the quiet edges the slice would be %.1f s, over the %.0f s a "
                            "slice may be: listen to it as two slices that meet in a pause"
                            % (rb - ra, LISTEN_MAX))
        else:
            rerun = _rerun(proj, ra, rb, args)
    return {"media": label, "clock": clock, "start": round(start, 3), "end": round(end, 3),
            "backend": concrete, "model": shown_model, "language": language or "auto",
            "words": out_words, "text": " ".join(w["w"] for w in out_words),
            "edges": edges, "warnings": warnings, "rerun": rerun}


def cmd_listen(args: Any) -> int:
    """hc listen P START END [--render CLIP] [--backend B] [--model M] [--json]"""
    from .project import Project
    from .transcribe import fmt_clock
    proj = Project(args.project).require()
    rep = listen(proj, args.start, args.end, args)
    if getattr(args, "json", False):
        print(json.dumps(rep, indent=2, ensure_ascii=False))
        return 0
    c = rep["clock"]
    hours = rep["end"] >= 3600
    print("%9s  %9s  %10s  %5s  %s" % ("s_" + c, "e_" + c, "at", "p", "word"))
    for w in rep["words"]:
        p = w.get("p")
        print("%9.3f  %9.3f  %10s  %5s  %s" % (w["s"], w["e"], fmt_clock(w["s"], hours, 2),
                                              "-" if p is None else "%.2f" % p, w["w"]))
    print("heard: %s" % (rep["text"] or "(no words: a laugh, a breath or room sound, or too "
                         "little to recognise)"))
    sys.stdout.flush()
    for msg in rep["warnings"]:
        env.log(STEP, "warning: " + msg)
    if rep["rerun"]:
        env.log(STEP, "re-run from the quiet edges: " + rep["rerun"])
    return 0
