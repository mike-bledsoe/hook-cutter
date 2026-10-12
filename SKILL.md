---
name: hook-cutter
description: Turns long-form video (podcast, interview, talk, panel, webinar, livestream; a local file or a URL) into captioned vertical Shorts, Reels and TikToks, plus optional 16:9 clips. Finds the hooks, recomposes each cut hook-first, burns word-timed captions with a hook card, makes covers and runs a QA pass. Use when the user says "cut reels/shorts/clips from this video/podcast/panel/talk", "find the hooks", "find the best moments", "clip this episode", "make shorts from this recording", "turn this interview into reels", "caption this video", "add captions to my clip", "make reels covers", "give this clip a cold open", "recut this clip" or "make this short tighter", or hands over a long recording and wants short-form out of it. Transcription, framing and rendering run locally; the agent writes the moments ledger and the edit spec, and the user approves the shortlist and the final files. Never uploads or publishes.
---

# hook-cutter

Turns long-form video into captioned vertical Shorts/Reels (and optional 16:9 clips): find the
hooks, make the cuts, add the captions, make the covers, prove it with QA. You supply the
judgment: which moments, in what order, with what words on screen. The scripts do the mechanics
deterministically and write inspectable JSON at every step. The skill ends at local files.

Per approved clip it makes: `renders/<clip>__vertical__<style>.mp4` (1080x1920 H.264/AAC at
-14 LUFS, word-timed captions with one accent word, a hook card readable on frame 0), an `.srt`,
a 1080x1920 cover with grid and mobile previews, a QA report, and a named export with a manifest.

## Hard rules

1. **Never upload, post, schedule or publish.** Gate E is the user approving local files.
2. **Never attribute words to the wrong person.** If attribution is not certain, drop the clip.
3. **Never put words in anyone's mouth.** No guessed words, no meaning-changing omits, no generated
   or cloned faces or voices. A garbled phrase gets cut around, never "fixed".
4. **Every number and name on a written surface** (hook card, cover, title, post, chapter title)
   is said in the clip. One exception: the clip's speakers may be named in the title and the post.
5. **`P/moments.md` is the only status store.** A row is CUT only when its render exists.
6. **No virality scores.** Ledger scores sort the shortlist; never present them as predictions.
7. **A clean exit code proves nothing visual.** Open every image QA lists under EYES.
8. **Fix the spec, never the output.** Never hand-edit renders or QA files; never loosen a gate.
9. **Ask about rights first:** does the user have rights to this footage, and whose brand does it
   carry? Never apply another organisation's branding.
10. **The project folder `P` is durable** (never /tmp). Everything for a source lives inside it.
11. **Images cost context; old ones drop out.** Open at most two at a time, write each up in a
    notes file at once, never re-open one written up (`qa.md`, "Image budget").

## Setup

`hc` below means `python3 <skill-dir>/scripts/hc`, where `<skill-dir>` is the folder holding this
file (usually `~/.claude/skills/hook-cutter`).

```
hc doctor                          # READY table: ffmpeg, Python, venv, models, fonts
hc doctor --install                # once: venv + deps (uv builds 3.12; --python PATH for 3.13)
hc doctor --install --with url     # + yt-dlp for URL sources (YouTube also needs deno or node)
hc doctor --install --with mlx     # + faster ASR on Apple Silicon
hc doctor --model large-v3-turbo   # pre-download the ASR model (about 1.6 GB; --backend picks whose)
```

If any command prints `hook-cutter is not set up`, run the install line. Never pip install into
the venv yourself.

**A new user:** offer `hc demo` first: one finished Short from a made-up recording in about a
minute, through the same commands below (needs `say` or espeak-ng; WARNs such as "no face" are
expected on its drawn speaker).

## Workflow

Start with one message to the user covering: the source, the project folder, who the clips
serve, rights and branding, anything already posted from this source, and (only if they care)
count, formats and style. Then:

### 1. Ingest

```
hc ingest P SOURCE                 # local file (linked; --copy to copy) or URL (--section 12:00-48:30)
```

Read the warnings in `P/source.json` (low resolution, HDR, VFR, missing frames, audio holes, a
channel far quieter than the other: then mixed to mono) and open `P/sheets/survey.jpg`: how many
people, black bars or gallery tiles, layout changes. A letterboxed or gallery source gets
`video.active_area`. Re-ingest a VFR source or one with missing frames with `--cfr` (the
transcript stays valid). A source replaced after ingest stops every step until it is ingested
again (`spec.md`, "Source identity"). Tell the user what caps quality before promising sharp
verticals.

### 2. Transcribe

Write `P/glossary.txt` first: names, companies, jargon, one per line (from the user, the video
title and description, the survey). Then:

```
hc transcribe P                    # --speakers-vtt PATH if a speaker-labelled VTT exists
```

Skim `P/transcript.md` for mangled proper nouns. The same mistake everywhere: add it to the
glossary and run `hc transcribe P --force`. A single occurrence: `fix` it in clips.json later.

Transcribe drops words with no audible sound, re-times word edges from the audio and re-listens
to voiced stretches that got no words. Flags in `transcript.md`: `unworded_audio` and `dropout`
(`<<... s of sound with no words>>` lines), `false_start` (two attempts at a phrase, a visible
restart such as "I was, I was" included), `timing_uncertain` (a start under a laugh or another
voice), `speaker_uncertain`, `no_audio`. Check them with `hc words` before keeping that audio
(`hooks.md` section 1). You cannot listen: `qa.md` ("Checking sound that has no words") shows
how to transcribe one stretch alone; clip key `insert` captions speech the transcript missed.
`hc transcribe P --retime` re-times without ASR (indices stay valid). `HOOKCUTTER_DEVICE=cpu`
forces faster-whisper onto the CPU.

### 3. Mine the moments

**Read `references/hooks.md` first, every time.** Recall pass in 10-minute windows with
1-minute overlap into `P/moments.md`, then hard gates, scores, and the set pass. Tools:
`hc find P "phrase"`, `hc words P A B`, `hc peek P START END`, and `P/signals.json` for URL
sources (heatmap and chapters: they surface candidates, they never rank them).

**Gate A (user): approve the shortlist.** Show the shortlist table (hooks.md section 9) and stop.
Approved rows become PLANNED once they have a clips.json entry.

### 4. Look before you build

For every approved row:

```
hc peek P 24:00 24:40              # contact sheet of the span
hc shots P --range 24:00 24:40     # shot boundaries, faces, proposed framing
```

**Gate B (agent, hard): attribution and on-camera.** You are certain who says the hook line,
and that person is on camera when it lands (or a wide shot honestly shows the group). Not
certain: mark the row PASS `attribution` and tell the user. No exceptions.

### 5. Write the edit spec

Write `P/clips.json`: `references/editing.md` for the craft, `references/spec.md` for every key,
`examples/clips.example.json` for a full example. Every piece carries `expect` anchors. Write
`hook_card` and `cover.headline` per `references/packaging.md`.

**Multi-face shots** (a gallery call, a two-shot): `hc shots` prints `MULTI-FACE` and proposes
letterbox; never paste a single-face mark there. Set `"speakers": {"<label>": {"x": 0.27}}` once
in clips.json (x from the shots sheet, labels as in `transcript.md`): framing follows the
speaker turns, and QA G16 fails a render that frames the wrong face. Then:

```
hc shots P c01-some-id             # shots for the clip's real range: framing marks come from here
hc plan P                          # piece table + lints; exits 1 on any FAIL lint
```

**Framing needs face data.** `hc shots P <clip>` writes `P/shots/<clip>.json` and draws
`P/sheets/shots_<clip>.jpg`: LOOK at the sheet before you write a framing mark (a mark is your
judgment of who is on screen). `hc render` and `hc all` analyse the shots themselves when they
are missing, so a mark never falls back to a blind crop (the mechanics, not the looking).

Fix every FAIL. Read every WARN and fix it or note in one line why it stands. `plan` is cheap:
run it after every edit.

### 6. Render

```
hc render P                        # all clips; --styles clean,pop  --formats vertical,landscape
```

When render prints `analysed shots for <clip> (framing needs face data)`, nobody has looked at
its shots yet: open `P/sheets/shots_<clip>.jpg` and check the marks before you trust the render.

**Gate C (agent): read the sheets.** For every render open
`P/sheets/<clip>__<fmt>__<style>.contact.jpg` and `P/sheets/<clip>__<fmt>.seams.jpg`: the right
person framed in every cell, captions clear of the mouth, the hook card readable in cell 1
(frame 0, the default thumbnail), no black band, black or flash frame or framing jump at a seam.
Fix in clips.json (framing marks, `speakers`, `caption_y`, piece edges) and render again.
Parallel `hc all P <clip>` runs on different clips are safe (shared files are written under locks).

### 7. Covers

```
hc cover P c01-some-id             # candidates sheet + cover + .grid.jpg + .mobile.jpg
hc cover P c01-some-id --at 12.4   # force a frame (output time: 12.4 or 0:12.4)
hc cover P --grid                  # only rebuild sheets/covers-grid.jpg from the covers on disk
```

Open `P/sheets/<clip>.cover-candidates.jpg` and the cover's `.grid.jpg`. Pick an expressive
frame of the person who owns the claim; record it as `cover.at_out` in clips.json.

### 8. QA

```
hc qa P --require-coverage
```

**Gate D (agent): zero FAIL, every EYES image opened.** Read `P/qa/summary.md` and each
`P/qa/<clip>.md`. Zero FAIL. Every WARN fixed or explained in one line. Open every image under
EYES and list the ones you opened. SKIPPED is not PASS: a clip whose gates passed but some could
not run is INCOMPLETE; a WARN clip with such gaps is held too (exit 2 under `--require-coverage`).
Run this full QA (with ASR) before the final export. `references/qa.md` explains every gate.

### 9. Export and present

```
hc export P
```

Export holds back clips whose QA is FAIL, INCOMPLETE, stale, missing or has evidence gaps, and
renders that no longer match their inputs (exit 1, listed); `--force` exports them anyway, only
when the user says so. It also writes `P/export/by-ear.md`: every join, the opening, the ending,
flagged words and sound WARNs, each with its time. Present: the `P/export/` paths, `by-ear.md`
(you cannot listen: ask the user to play every clip with it open), `P/sheets/covers-grid.jpg`,
the QA summary (FAIL and WARN counts plus anything you explained), the three surfaces per clip
(hook card, cover headline, title), draft post copy if asked, and any cherry-picks or risks.

**Gate E (user): approval.** Approval means the user has the files. You never upload or publish.
Update `moments.md`: CUT rows get what was exported; anything deferred becomes BANKED with a
reason for next time.

## The iteration loop

User notes are one-liners ("open stronger", "the laugh got cut", "wrong face at 0:12"). Turn
each into one spec edit (`references/editing.md` ends with a note-to-edit table), then:

```
hc all P c01-some-id               # plan, render, cover, qa, export; skips work whose inputs did not change
hc status P                        # one line per clip: plan / render / cover / qa state
```

- `hc all` passes `--styles`/`--formats` on, runs QA with `--require-coverage` (`--no-asr`
  leaves QA INCOMPLETE, so export holds everything: a quick look, never the final files),
  exports after a QA FAIL (holding failing clips), exits with the worst code and ends with one
  line per clip saying what ran. It re-makes only stale covers and keeps a current one made by
  `hc cover` or copied in. `hc status` judges freshness by content and names the reason:
  `plan STALE(spec|shots|transcript|source|code)`, `render STALE(plan|shots|spec|source|inputs)`
  (inputs: preset, font, safe zone or render code), `cover STALE(at_out|headline|shift|render)`.
- One note, one edit, one re-render. Do not touch clips the user did not mention.
- Style-only changes reuse the cut; edits to pieces re-cut only the segments that changed.
- Re-run gates C and D for every clip you re-rendered.
- Log notable outcomes (a reversed PASS, a reverted reorder) in the ledger for the next session.

## Command reference

| Command | Use |
|---|---|
| `hc doctor [--install] [--with mlx,url,whispercpp] [--model NAME] [--backend auto\|mlx\|faster-whisper] [--python PATH] [--json]` | check or set up the environment |
| `hc ingest P SOURCE [--copy] [--section A-B] [--cfr] [--force]` | bring a file or URL into a project |
| `hc transcribe P [--backend auto\|mlx\|faster-whisper\|whispercpp] [--model M] [--language L\|auto] [--glossary TEXT] [--speakers-vtt PATH] [--force]` | ASR to `transcript.json` and `transcript.md` |
| `hc transcribe P --retime [--clip ID]` | re-time existing words from the audio (no ASR) |
| `hc transcribe P --file VIDEO --clip ID` | transcript for a caption-only clip |
| `hc find P "phrase" [--near MM:SS] [--max N] [--clip ID]` | phrase to word indices and times |
| `hc words P A B [--clip ID]` | words A..B with times, gaps, filler flags, speakers |
| `hc peek P START END [--n 12] [--file PATH]` | contact sheet of a source range or a rendered file |
| `hc shots P CLIP...` / `hc shots P --range START END` | shots, faces, framing proposal and sheet |
| `hc plan P [CLIP...] [--quiet]` | validate and plan; prints table and lints |
| `hc render P [CLIP...] [--styles a,b] [--formats vertical,landscape] [--force]` | render clips and sheets |
| `hc cover P CLIP... [--styles a,b] [--at SECONDS]` | cover, candidates sheet, previews (`--at` takes any time form below) |
| `hc cover P --grid` | rebuild `sheets/covers-grid.jpg` from the covers on disk (seconds; after an interrupted cover step) |
| `hc qa P [CLIP...] [--styles ...] [--formats ...] [--no-asr] [--require-coverage]` | QA gates and reports |
| `hc export P [CLIP...] [--styles ...] [--formats ...] [--force]` | named files, `export/manifest.json` and `export/by-ear.md`; holds unchecked clips (after adding, dropping or reordering clips, export all clips) |
| `hc all P [CLIP...] [--styles ...] [--formats ...] [--no-asr] [--force]` | the whole chain, skipping up-to-date work (`--force` redoes every step; held exports stay held) |
| `hc status P` | where every clip stands |
| `hc test [--slow] [PATTERN]` | the skill's own test suite |
| `hc demo [DIR] [--force] [--open]` | one finished Short from a made-up recording, to see the output (default `~/hook-cutter-demo`) |

Times accept `681.5`, `11:21`, `11:21.40` or `1:02:03.4`. Default clips = all clips in
clips.json; default styles and formats = the clips.json defaults.

Exit codes: 0 ok, 1 spec or user error, a QA FAIL, a held export or a usage mistake (the
message says what to fix), 2 evidence gap, 3 tool or environment error such as a read-only
folder or a full disk (run `hc doctor`). `HC_VERBOSE=1` prints ffmpeg command lines.

## References

| File | Read it when |
|---|---|
| `references/hooks.md` | before mining: recall pass, ledger format, taxonomy, gates, scores, set pass, shortlist, worked examples |
| `references/editing.md` | writing pieces: cold opens, preamble, repetition, endings, length bands, notes-to-edits |
| `references/captions.md` | choosing a style, accents, legibility, safe zones, hook card mechanics, glyphs |
| `references/packaging.md` | hook card vs title vs cover, parse-cold covers, frame curation, post copy |
| `references/fidelity.md` | anything touching attribution, numbers, risky claims, rights |
| `references/qa.md` | reading plan lints and QA gates, EYES discipline |
| `references/spec.md` | every clips.json and preset key with defaults and examples |
