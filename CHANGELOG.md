# Changelog

## Unreleased

- G16 (framing attribution) no longer judges a crop of a single-face shot as a crop of the
  two-shot it meets at a cut. A window now counts as part of a multi-face span only when it
  overlaps it by more than 0.005 s (`G16_MIN_OVERLAP`, under a fifth of a frame). The plan's
  out times carry float noise (a segment out of 9.700034 for 9.7), so the speaker's own single
  "overlapped" the next or previous two-shot by 1e-6 to 3.3e-5 s and was checked against their
  two-shot x: on a 14-clip run from one 4K interview, 7 clips failed G16 this way and 4 were held
  from export with no wrong face anywhere. A real crop of a two-shot, even one frame long, is
  still judged and still fails on the wrong face (regression test in `test_fix_qa.py`). QA
  reports go stale once (their code digest covers every module); renders stay current.
  Related, not fixed here: the renderer's per-frame shot lookup has the same 4-decimal boundary
  issue, which can leave one frame cropped at the previous shot's face after a cut.

- Frame grabs (covers and QA) stop decoding once the last selected frame is out (`-frames:v`).
  Before, every grab decoded its input to the end, so on a 79-minute 4K source QA of one clip
  near the start ran over an hour; a test grab ran about 30x faster. The frames, their order and
  their bytes are unchanged. QA reports go stale once (their code digest covers every module);
  renders stay current.
- Paper edit before the first render (SKILL.md step 5, `editing.md` "Paper edit"): the user
  reads each clip's words in play order with every source jump marked and sized, plus the
  improvement process below, and approves before anything renders; then a pilot of two or three clips
  for their ear before the rest. On the production run behind it, 4 of the first 14 renders
  were dropped and 2 re-cut on first listen, and the large source jumps were the ones flagged.
- The paper edit's single improvement pass becomes a process that runs every time before the
  first render (`editing.md` "Paper edit", steps 3 to 6): per clip, two or three rounds on the
  words, each a level deeper, the third only when the second found something worth its cost; a
  skeptic per clip that tries to refute each change and defaults to leave; one read across the
  set; then one plan, `P/notes/paper-edit/PLAN.md`, that the user answers yes or no per change,
  shown with each clip's words, every jump over about 30 s and the ideas the skeptic killed. A
  quick pass (on request, or for three clips or fewer) is one round and the skeptic. Evidence,
  from that same run (a 79-minute guest interview whose first 14 renders, cut straight off a
  hook-line shortlist, lost 4 clips on first listen and needed 2 re-cuts): on the 10 kept clips,
  three rounds and a skeptic per clip plus one set read (31 rounds) produced 474 ideas and kept
  11 changes on 6 clips; 4 clips were left as they were. The skeptic killed the lead idea on 6
  of the 10, and 3 of those went from change to leave. Two moves survived across the set:
  running the tape where the edit had cut a live turn, and cutting where the speaker describes
  instead of says; reorders built to chase an early-payoff number lost. The set read took a
  phrase repeated across clips from 3 plays to 1 and set a posting order. The user then answered
  each change yes or no, one by one. Cost: about 4.8M subagent tokens, 42 minutes, 21 agents.
- `hc ingest` warns when the project folder is one a sync service uploads (iCloud Desktop &
  Documents, iCloud Drive, Dropbox, OneDrive, Google Drive, other cloud storage apps): a
  project holds gigabytes of source, segments and renders that change on every pass.
- `hc doctor` shows the default home for project folders, `~/hook-cutter-projects` or
  `$HOOKCUTTER_PROJECTS`, and warns when that folder is synced.
- SKILL.md hard rule 10: `P` is local as well as durable, defaults to a folder under that
  home, and a parallel agent's working copy of `P` never goes in a synced or scratch folder.
  The version is unchanged, so 0.2.0 projects stay current; QA reruns once, because its code
  digest covers every module.

## 0.2.0

Aimed at a new user's first hour, and at the defects the first production run reported most.

### New

- `hc demo`: one finished Short from a made-up 20-second recording (a drawn speaker, a
  text-to-speech voice) in about a minute, through the real pipeline. `doctor` suggests it after
  `READY`.
- `export/by-ear.md`, written by every export: for each exported clip, the opening, every join
  with the words on both sides, the ending, the words the transcript flags and the
  sound-related QA warnings, each with its time in the file.
- `hc cover P --grid` rebuilds `sheets/covers-grid.jpg` from the covers on disk in seconds; QA
  names it when the grid is missing.

### Changed

- `hc all` makes only the stale covers again (it used to remake every cover when any one was
  stale, at minutes per cover on a long 4K source), and keeps a cover made by `hc cover` or
  copied in from another project when `hc status` calls it current.
- Word timing: a word whose end the recogniser ran on into a pause stays before the pause when
  its audio is there (it used to be moved past the pause, onto the next word's sound), and the
  start of a word after a pause moves back to the voice onset when the recogniser stamped it up
  to 0.3 s late, on every backend (never onto a lip click). A new ground-truth passage
  (`tests/data/timing/p5_tts_pause_end`) measures both; the four older passages score the same.
  Transcripts made before 0.2 are re-timed by `hc transcribe P --retime`.
- `false_start` flags a visible restart: a repeat whose first attempt ends in a comma or a dash
  ("I was, I was going") or a doubled function word ("the the"), whatever its length.
- `.srt` sidecar: no cue holds one word alone when the run has more, no cue or line breaks
  inside a number phrase ("twelve to | eighteen months"), and cues and lines avoid ending on
  "at", "for", "and" and similar words.
- `hc peek` sheet names keep sub-second times and a non-default `--n`, so two requests never
  overwrite one sheet.
- `hc words --clip` and `hc find --clip` refuse an unknown clip id instead of reading the
  project transcript.
- SKILL.md: hard rule 11 (images cost context) and `references/qa.md` "Image budget" and "By
  ear".

## 0.1.1

First shared version.
