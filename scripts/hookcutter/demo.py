"""``demo``: one finished Short from a made-up 20-second recording, in about a minute.

It shows a new user what the skill makes before they bring their own footage. It builds a
short source (a synthetic speaker in a room; the voice is the computer's text-to-speech), then
runs the normal commands on it: ingest, transcribe, plan, render, cover, QA and export. The
edit spec it writes is the same kind an agent writes: a cold open, the setup, the close, a hook
card and a cover headline. Nothing about it is special-cased in the pipeline.

The voice comes from ``say`` (macOS) or ``espeak-ng`` / ``espeak`` (Linux). The transcript uses
the small ``tiny.en`` speech model on faster-whisper (about 75 MB, downloaded on first use), so
the demo does not wait for the 1.6 GB default model. QA re-transcribes with the same model.
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple

from . import env

MARKER = ".hook-cutter-demo"
DEFAULT_DIR = "~/hook-cutter-demo"
ASR_BACKEND = "faster-whisper"
ASR_MODEL = "tiny.en"
CLIP_ID = "c01-nobody-watches-an-hour"
SIZE = (1920, 1080)
FPS = 30
LEAD_IN = 1.2       # seconds of room tone before the first word
TAIL = 1.5          # and after the last

# "|" marks a pause of about 0.6 s: the cuts land in those pauses, as they would between a
# speaker's sentences.
SCRIPT = ("Most people think the hard part of a podcast is the recording. It is not. "
          "The hard part is the clip. | Nobody watches an hour. They watch thirty seconds. | "
          "So you find the one line that stops the scroll, you open on it, and you cut "
          "everything that came before it. Do that every week, and the long show starts "
          "paying for itself.")
PAUSE_MS = 600

# The three pieces, found by phrase in whatever the speech model heard, so the demo does not
# depend on exact word indices. Played in this order: the claim first, then the setup, then
# the close (references/editing.md, "Cold opens").
PIECES = [
    ("cold_open", "Nobody watches an hour. They watch thirty seconds."),
    ("setup", "Most people think the hard part of a podcast is the recording. It is not. "
              "The hard part is the clip."),
    ("close", "So you find the one line that stops the scroll, you open on it, and you cut "
              "everything that came before it. Do that every week, and the long show starts "
              "paying for itself."),
]
HOOK_CARD = ["THE HARD PART", "IS *THE CLIP*"]
COVER_HEADLINE = "NOBODY WATCHES AN *HOUR*"
TITLE = "Nobody watches an hour"
POST = {"title": "The hard part of a podcast is not the recording",
        "caption": "Nobody watches an hour. Find the line that stops the scroll and open on it."}


# ---------------------------------------------------------------- the source

def _tts_command(text: str, out_base: Path) -> Tuple[List[str], Path]:
    """The text-to-speech command for this machine and the file it writes."""
    if platform.system() == "Darwin" and shutil.which("say"):
        out = out_base.with_suffix(".aiff")
        spoken = text.replace("|", "[[slnc %d]]" % PAUSE_MS)
        return ["say", "-v", "Samantha", "-r", "175", "-o", str(out), spoken], out
    for name in ("espeak-ng", "espeak"):
        if shutil.which(name):
            out = out_base.with_suffix(".wav")
            spoken = "<speak>%s</speak>" % text.replace("|", '<break time="%dms"/>' % PAUSE_MS)
            return [name, "-m", "-v", "en-us", "-s", "165", "-w", str(out), spoken], out
    raise env.HCError("the demo needs a text-to-speech voice: `say` (macOS) or espeak-ng "
                      "(Linux: apt install espeak-ng). Your own video does not need one.", code=3)


def _speak(text: str, out_base: Path) -> Path:
    cmd, out = _tts_command(text, out_base)
    proc = subprocess.run(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, universal_newlines=True)
    if proc.returncode != 0 and cmd[0] == "say":
        # the Samantha voice can be missing on a trimmed install: fall back to the system voice
        cmd = ["say", "-r", "175", "-o", str(out), cmd[-1]]
        proc = subprocess.run(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, universal_newlines=True)
    if proc.returncode != 0 or not out.is_file() or out.stat().st_size < 10000:
        raise env.HCError("text-to-speech failed (%s): %s" % (cmd[0], (proc.stderr or "").strip()[-300:]),
                          code=3)
    return out


def _draw_room(out: Path) -> Path:
    """A still of a synthetic speaker at a microphone (Pillow; no faces of real people)."""
    from PIL import Image, ImageDraw, ImageFilter

    w, h = SIZE
    im = Image.new("RGB", (w, h))
    d = ImageDraw.Draw(im)
    for y in range(h):  # backdrop: a dark wall that warms towards the floor
        t = y / h
        d.line([(0, y), (w, y)], fill=(int(24 + 26 * t), int(28 + 20 * t), int(40 + 14 * t)))
    glow = Image.new("L", (w, h), 0)  # a lamp behind the speaker, blurred into a soft pool
    ImageDraw.Draw(glow).ellipse([1240, 60, 1760, 560], fill=120)
    glow = glow.filter(ImageFilter.GaussianBlur(120))
    im = Image.composite(Image.new("RGB", (w, h), (120, 96, 70)), im, glow)
    d = ImageDraw.Draw(im)
    d.rectangle([150, 330, 640, 346], fill=(78, 68, 60))  # a shelf of books
    x = 176
    for bw, bh, c in [(60, 120, (150, 70, 60)), (48, 104, (70, 104, 140)), (70, 130, (190, 160, 90)),
                      (52, 96, (90, 132, 100)), (64, 116, (120, 90, 130))]:
        d.rectangle([x, 330 - bh, x + bw, 330], fill=c)
        x += bw + 14
    cx = w // 2
    d.rounded_rectangle([cx - 340, 640, cx + 340, h + 200], radius=180, fill=(38, 58, 88))  # shoulders
    d.rectangle([cx - 46, 560, cx + 46, 670], fill=(178, 140, 112))                        # neck
    d.ellipse([cx - 128, 300, cx + 128, 600], fill=(196, 158, 128))                        # head
    d.chord([cx - 136, 286, cx + 136, 520], 180, 360, fill=(52, 40, 34))                   # hair
    d.rounded_rectangle([cx + 250, 560, cx + 330, 720], radius=34, fill=(28, 28, 30))      # mic
    d.rectangle([cx + 284, 720, cx + 296, h], fill=(22, 22, 24))
    im.save(str(out))
    return out


def build_source(media: Path) -> Path:
    """``media/demo-source.mp4``: the still with film grain (so the picture is never frozen)
    and the spoken script over a quiet room tone, at a constant 30 fps."""
    media.mkdir(parents=True, exist_ok=True)
    voice = _speak(SCRIPT, media / "voice")
    still = _draw_room(media / "room.png")
    probe = env.ffprobe_json(voice, "-show_entries", "format=duration")
    dur = float(probe.get("format", {}).get("duration") or 20.0) + LEAD_IN + TAIL
    out = media / "demo-source.mp4"
    delay = int(LEAD_IN * 1000)
    env.ff(["-loop", "1", "-framerate", str(FPS), "-i", still, "-i", voice,
            "-f", "lavfi", "-i", "anoisesrc=color=pink:amplitude=0.002:sample_rate=48000",
            "-filter_complex",
            "[0:v]noise=alls=6:allf=t,format=yuv420p[v];"
            "[1:a]aresample=48000,adelay=%d|%d[sp];"
            "[sp][2:a]amix=inputs=2:duration=longest:normalize=0[a]" % (delay, delay),
            "-map", "[v]", "-map", "[a]", "-t", "%.2f" % dur, "-r", str(FPS),
            "-c:v", "libx264", "-preset", "fast", "-crf", "20",
            "-c:a", "aac", "-b:a", "160k", "-ac", "2", out], capture=False)
    for f in (voice, still):
        f.unlink()
    return out


# ---------------------------------------------------------------- the edit spec

def _anchor(words: list, a: int, b: int) -> List[str]:
    """Two `expect` anchors: the first and last three words of the piece as transcribed."""
    def text(i: int, j: int) -> str:
        return " ".join(str(words[k]["w"]) for k in range(i, j + 1)).strip(" ,.?!")
    return [text(a, min(b, a + 2)), text(max(a, b - 2), b)]


def write_spec(P: Path) -> dict:
    """clips.json for the demo clip, its pieces located by phrase in the transcript."""
    from . import find

    words = env.read_json(P / "transcript.json")["words"]
    pieces = []
    for role, phrase in PIECES:
        hits = find.search(words, phrase, max_hits=1)
        if not hits or hits[0]["score"] < 0.6:
            raise env.HCError("the speech model did not hear the %s line well enough to cut "
                              "it (open %s): try `hc demo --force` once more" % (role, P / "transcript.md"),
                              code=3)
        a, b = hits[0]["a"], hits[0]["b"]
        piece = {"words": [a, b], "role": role, "expect": _anchor(words, a, b)}
        if role == "close":
            piece["hold_after"] = 0.5
        pieces.append(piece)
    spec = {
        "version": 1,
        "_note": "Written by `hc demo`. The same kind of edit spec an agent writes: "
                 "references/spec.md explains every key.",
        "defaults": {"styles": ["clean"], "formats": ["vertical"]},
        "clips": {CLIP_ID: {
            "title": TITLE,
            "pieces": pieces,
            "hook_card": {"lines": HOOK_CARD, "dur": "auto"},
            "cover": {"headline": COVER_HEADLINE},
            "post": POST,
        }},
    }
    env.write_json(P / "clips.json", spec)
    return spec


# ---------------------------------------------------------------- the command

def _prepare(dest: Path, force: bool) -> None:
    if dest.exists() and any(dest.iterdir()):
        if not (dest / MARKER).is_file():
            raise env.HCError("%s already exists and is not a demo folder: pass another folder, "
                              "e.g. `hc demo ~/hook-cutter-demo-2`" % dest, code=1)
        if not force and (dest / "project" / "export").is_dir():
            return
        for name in ("project", "media"):
            shutil.rmtree(str(dest / name), ignore_errors=True)
    dest.mkdir(parents=True, exist_ok=True)
    (dest / MARKER).write_text("made by `hc demo`; safe to delete\n", encoding="utf-8")


def _step(argv: List[str]) -> int:
    from . import cli
    print("\n$ hc %s" % " ".join(cli.q(a) for a in argv), flush=True)
    return cli.main(argv)


def cmd_demo(args: argparse.Namespace) -> int:
    dest = Path(os.path.expanduser(args.dir or DEFAULT_DIR)).resolve()
    _prepare(dest, bool(args.force))
    P = dest / "project"
    if (P / "export").is_dir() and not args.force:
        print("the demo is already made: %s (pass --force to make it again)" % (P / "export"))
        return _summary(P, open_it=bool(args.open))
    env.log("demo", "making a 20-second recording to cut (text-to-speech voice, drawn picture)")
    src = build_source(dest / "media")
    for argv in (["ingest", str(P), str(src)],
                 ["transcribe", str(P), "--backend", ASR_BACKEND, "--model", ASR_MODEL]):
        rc = _step(argv)
        if rc != 0:
            return rc
    write_spec(P)
    print("\nwrote %s (the edit spec: a cold open, the setup, the close)" % (P / "clips.json"))
    rc = _step(["all", str(P)])
    if rc not in (0, 2):
        return rc
    return _summary(P, open_it=bool(args.open))


def _summary(P: Path, open_it: bool) -> int:
    exported = sorted((P / "export").rglob("*.mp4")) if (P / "export").is_dir() else []
    covers = sorted((P / "export").rglob("* - Cover.jpg")) if (P / "export").is_dir() else []
    print("""
demo finished. Open these:
  the Short      %s
  the cover      %s
  what QA says   %s
  listen for     %s
  how it looks   %s
  the edit spec  %s

The recording is made up (a drawn speaker, a text-to-speech voice), so expect a few QA WARN
lines that a real recording would not raise, such as no face found. Then try your own video:
open Claude Code and say "Cut vertical Shorts from <your video>" (README, Quick start).""" % (
        exported[0] if exported else P / "export",
        covers[0] if covers else P / "sheets" / "covers-grid.jpg",
        P / "qa" / "summary.md",
        P / "export" / "by-ear.md",
        P / "sheets" / ("%s__vertical__clean.contact.jpg" % CLIP_ID),
        P / "clips.json"))
    if open_it and exported:
        opener = {"Darwin": ["open"], "Linux": ["xdg-open"]}.get(platform.system())
        if opener and shutil.which(opener[0]):
            subprocess.run(opener + [str(exported[0])], check=False)
    return 0

