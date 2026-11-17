# Changelog

## Unreleased

- Captions and the `.srt`: five fixes from a production run on a 75-minute 720p remote
  interview (host and guest in two fixed tiles, each clip framed with one tile over a blurred
  fill of itself, a transcript with no speaker labels). Regression tests in
  `test_fix3_captions.py`, each failing before its fix.
  - No caption-free frame right before a seam. A page holds at most 0.4 s after its last word and
    the gap fill stops at a cut, so a page ending a frame or two before a cut left those frames
    with no caption, then the next page started on the cut (one clip had to move its cut a frame
    earlier to hide it). A page that would end `BLINK_FRAMES` (2) or fewer frames before the next
    page, the cut that closes its stretch, or the clip's end now holds to it; it never holds
    past the cut, and a gap of three frames or more stays a gap. A speaker change a frame or two
    later is held the same way. On the run's 11 final plans this took caption blinks from 5 to
    1; the one left is two blank frames after a cut, before the next page's first word (closing
    it would light that word earlier than `lead_ms`).
  - A hook-card accent broken over two lines (`"THE *BANK*"`, `"*ACCOUNT*"`) is one accent where
    the speaker says the words together (at most two stopwords between them, as inside one
    phrase), so `emphasis.per_page: 1` lights both and the render no longer warns "hook accent
    never shown in captions: 'ACCOUNT'" (the QA G14 WARN is gone). The phrases themselves are
    unchanged, so plan L08 and QA G14 count the same two; said apart, or written as two runs on
    one line, they are still two accents and still capped.
  - A word whose onset is under the hook card is no longer captioned. The burned captions kept a
    word when its midpoint fell after the card's end less the caption lead, so on one clip "AND",
    which started about 0.1 s before a 1.4 s card ended, opened the first page, against the rule
    in `captions.md` and `editing.md` rule 13; the workaround (`fix` `""` on that word) also
    took it out of the `.srt`. The `.srt` still carries every word spoken under the card, with no
    `fix` needed. Plan lint L08 judges an accent by the same onset (it warns when every time the
    accent is said it starts under the card), so an accent starting 0.1 s before the card ends
    now gets the L08 WARN it used to miss.
  - An `insert` that starts at the same moment as the next transcript word (the recogniser
    folded the missed speech into that word's start) now plays before it, all of its words
    together: the plan placed it after, so the captions read "HELPING LIKE PEOPLE" for "like
    helping people" (the run worked around it by starting the insert 0.01 s earlier). That word
    is heard once the insert ends, so its caption now starts there; a multi-word insert ("kind
    of like") no longer has the word slip in after its first token. A word that starts first
    still plays first, and the word an insert follows keeps its own start.
  - The `.srt` gives the clip's first word, and the first word of a piece that follows a
    sentence end, a capital letter: a piece cut in mid-sentence opened its cue in lower case (two
    cues of one clip), and the only fix was a case-only `fix`. A piece that goes on with the
    previous piece's sentence, a word that already holds a capital ("eBook") and a word that
    starts with a number ("3pm", "90s", "$5k") are left as they are. The burned captions are
    unchanged.

  A clip whose first caption word starts under the hook card loses that word from the burned
  captions on its next render, so re-check the first page of clips already approved. Plans are
  not re-made by a code change: an existing plan keeps an insert tie and the old L08 result
  until `hc plan` runs again.
- A piece can name who speaks it: `"speaker": "<label>"` on a piece (`spec.md`, piece table).
  Before, a transcript without speaker labels gave every word to the clip's `speaker`, so on the
  same run a guest clip whose closing piece is the host, shown correctly on the host's own tile,
  failed QA G16 ("framed face is not <guest> for 2.3s") and was held from export; the only
  escape, `"reaction": true`, mislabels a speaking turn. The same gap let `hc cover`'s automatic
  pick choose a frame of the host for the guest's clip (the top candidate was in the host's
  piece). Now the piece's words take its label wherever the transcript gives none (transcript
  labels still win, with an L01 WARN when a piece's words carry another one), and once any piece
  of a clip names its speaker, the clip's other unlabelled words take the clip's `speaker`. An
  inserted word takes the transcript label of the word it follows when that word has one, else
  the piece's. G16 checks each piece against its own speaker, speaker-follow framing and a
  `track` mark with no `x` follow the labels, and the automatic cover frame skips a piece someone
  else speaks (the whole clip is still used, with the existing warning, when no other frame is
  left). The wrong face still fails: the host's piece on the guest's tile, or the guest's pieces
  on the host's tile, fail G16 as before (regression tests in `test_piece_speaker.py`). A label
  the `speakers` map lacks is L01 WARN, and so is a piece key on a clip with no `speaker` where
  another piece names none (those words stay unlabelled and G16 still FAILs them); a value that
  is not a non-empty string is L01 FAIL; a map label used only by a piece no longer warns as
  unused; a G16 mismatch on words with no label now says to name the piece's speaker. A filled
  label is recorded as `sp_from` on the plan word and does not make the plan stale against the
  transcript. Clips without the key plan and render exactly as before (the key is added to a
  piece only when set, so clip hashes do not move).
- QA no longer reads the renderer's own letterbox fill as a defect. G8 and G12 take the picture
  rows of every `letterbox`, `rect` and `full` window from the render log (`framing_windows`,
  placed the way render places them: fitted, even sizes, on the style's `y_center`) and leave the
  blurred fill above and below the tile out of the dark-band scan; G12 does the same on a cover
  cut from this plan's render (`frame_source: segments`), moved up by its `shift`. When the whole
  clip is letterboxed, G8's freeze check reads only the picture rows, its threshold scaled to the
  share of them the masked caption band leaves. On the same run a clip at the shipped `y_center`
  820 got `FAIL G12 dark band on the cover: y 104-200 (5%)` and `WARN G8 dark band y 112-192
  (4%)` from the dimmed fill of a dark wall above the tile, plus a 1.04 s "frozen picture" of a
  talking head; two clips at `y_center` 680 got 5.75 s and 3.45 s of "frozen picture" while the
  speaker talked. Replayed on that data, all three pass G8 and the cover passes the dark-band
  check. A dark band inside the tile is now measured against the tile's own height, not the
  frame's, and its evidence says `of the picture`: the fill stretched the picture to the frame
  height, so that is the share it used to show a source bar at, and the levels hold. Through
  render's own letterbox chain the bars of 2:1 content in a 16:9 picture still WARN in G8 and
  FAIL in G12, and those of 2.39:1 FAIL in both. A frozen tile (with or without the caption band
  over it), a cover cut from the source and a stale render (G0) are judged as before
  (regression tests in `test_qa_false_alarms.py`).
- The cover json no longer calls a size fit a typeface fallback. `hc cover` sets the headline
  only in the preset's own font file (a missing face or glyph stops it), so `font_fallback` is
  now always false, and a headline set below the preset's size range is recorded as
  `size_fallback` (the fit's message, also one of the cover's warnings). G12 reports it once, as
  a size; before, it also said "the headline font fell back to another face" for an Archivo
  Black cover that only needed 82 to 92 px. Older cover json, whose `font_fallback` came from the
  size fit, are read the same way when their warnings carry the size message.
- `cover.at_out` and `hc cover --at` name the frame that starts nearest to the time (a half
  frame goes to the later one, never past the last frame), not the frame on screen at it: 35.458
  at 24 fps (850.99 frames) is frame 851 and 24.333 is frame 584, where both used to land one
  frame early. `hc status` rounds the same way. A manual cover whose recorded time is not a
  frame start can show `cover STALE(at_out)` once after the update: `hc cover` (or `hc all`)
  makes it again. When `hc cover` re-picks at the output time of the cover on disk and that time
  now shows another source frame (an edge before it moved), it prints a WARN with both source
  times.
- Renders from a hot, clipped source now reach -14 LUFS (QA G7). The renderer normalises in one
  linear loudnorm step, and a source that already peaks near 0 dBFS cannot take the gain to -14
  LUFS without its peaks passing loudnorm's -1.5 dBTP, so ffmpeg switched itself to dynamic mode,
  without a warning, and landed short. On the same run every clip fell to dynamic mode: one of
  twelve landed at -15.3 LUFS and was held at export, and the other eleven passed G7 by only 0.1
  to 0.8 LU. No clip or preset key could fix it (`gain_db` moves loudness and peaks together).
  Now, when the measured peaks leave no room for the linear step, the render fits a pre-gain and
  a look-ahead limiter at 192 kHz (`alimiter`, ceiling -2 dBFS, its delay compensated so no
  sample moves) from fresh loudnorm measurements, at most four passes, then runs the same linear
  step on the limited audio. The fit never asks the limiter for more than 15 dB on the loudest
  peak: a step past that is held at the 15 dB gain, and that gain is measured before the clip is
  given up on. The held clip now measures -14.1 LUFS with a -1.4 dBTP true peak (it was -15.3
  LUFS and -1.3 dBTP), with up to 11.7 dB taken off its loudest peak. The fit is recorded in the
  render log under `loudnorm.limiter`. A clip whose peaks need more than 15 dB of limiting keeps
  the plain step, the log says why under `loudnorm.limiter_skipped` (with the loudness it
  reached at 15 dB and the loudness the linear step needed), and G7 still fails it; G7's limits
  are unchanged. A source that normalises in linear mode keeps exactly the audio filter graph it
  had, so its sound does not change. A clip that went to dynamic mode and happened to pass G7 now
  takes the limiter too and lands nearer -14 LUFS, so its sound changes on re-render: on that
  interview, the other eleven clips. `hc doctor` now requires ffmpeg's `alimiter`. Regression
  tests in `test_loudness.py`: a synthetic clipped speech track at about -19 LUFS that failed G7
  (-14.9 LUFS, true peak -0.8 dBTP) and now passes, one whose fit needs the full 15 dB that
  failed G7 at -15.8 LUFS and now passes, an ordinary source's graph pinned, two sources past the
  15 dB cap that still fail G7, the fit's steps checked on a replayed gain curve, and the limited
  chain checked for zero lag.
- New plan lint L21 (WARN): speech the transcript may have folded into a neighbouring word or
  dropped. It reads the clip and up to 0.5 s outside each kept stretch from the audio level,
  judged against the speech around it (a new helper, `align.speech_share`), so a breath on a
  quiet recording does not count: (a) a word that sounds longer than its text takes to say
  (0.30 s plus 0.06 s per letter, a digit counting three) and runs straight into the next word,
  such as a restart stamped as one word; (b) a gap of 0.2 s or more between two words at speech
  level across 80% of it; (c) 0.3 s up to L19's 1.5 s of wordless sound at speech level. Each
  WARN names the source time and the words around it and prints the `hc listen` command for that
  stretch (below), from the word before to the word after, its edges moved out to a pause when
  one lies within 5 s; an `insert` over it, or a `fix` on the word (for a gap or wordless
  sound, on the word after it), settles it. An `in_offset` past a restart does not: what is
  left after the edge can still hold more than the word's text, so the `fix` says what is
  heard from the edge on. (a) reads a word whether or not a `false_start` or
  `timing_uncertain` flag marks it (the flag can name one restart while the word holds more);
  sound L15 or L19 already reports is not repeated. On the
  same run, folded restarts, fillers and short dropouts reached the paper edit in 7 of 12 clips
  and only slice transcriptions found them: a 0.40 s "a" holding a whole second attempt at "I'm
  in a", a filler with no word in a voiced 0.26 s gap, a 0.35 s word with no token. Replayed on
  11 of those clips with their `fix` and `insert` remedies removed, L21 raised 27 WARNs (0 to 5
  per clip). Slice transcriptions of each found 11 hidden words (9 of the 16 folds the run had
  found by hand that transcribe had not flagged) and 4 words stamped 0.25 s or more off; 2 more
  were words stamped under 0.2 s off and 10 were drawn or emphatic speech. The misses were words
  folded into a word of ordinary length. With the run's own remedies in place 19 remain, 9 of
  them on speech or timing the edit had not answered. The gap threshold is 0.2 s, not 0.12 s:
  the five speech-level gaps of 0.12 to 0.2 s in those clips held no word. Lints are not in a
  render's plan hash, so L21 does not make a render stale.
- A plan lint about a stretch no segment plays now points at the nearest segment edge in the
  source (the next segment's start or the previous one's end). It took the first segment in play
  order that starts later in the source, so with a cold open a stretch just before the second
  piece, or just past the end of the first, was reported at 0:00 or at the clip's end. This
  moves L21 and, in the same cases, L06 and L15.
- `hc listen P START END [--render CLIP] [--backend B] [--model M] [--json]`: transcribe one
  short stretch on its own, the check an agent runs on a seam, a flagged word or sound with no
  words. It cuts the slice from the source (source seconds) or from the clip's vertical render
  (`--render`, output seconds) into a temporary folder outside P, transcribes it with the
  transcript's backend, model, language and glossary (the settings QA's G5 uses; `--backend` and
  `--model` override them), prints each word with its times and probability on the clock given,
  and writes nothing into P. The old recipe in `qa.md` ("Checking sound that has no words":
  ffmpeg into `P/cache`, then `hc transcribe P --file ... --clip scratch-x`) wrote into `P/cache`
  and `P/transcripts`, so a read-only reviewer could not follow it and scratch transcripts piled
  up. The recipe now uses `hc listen`, and so do the planner's L15 `unworded_audio` warning (the
  flagged stretch with 0.2 s of margin each side; a stretch over 120 s is named as slices of at
  most 120 s) and every L21 warning. Both edges are measured on the audio around the slice (the
  planner's 20 s floor radius): an edge inside a voiced run is a warning that names the nearest
  quiet edges outside the slice (0.2 s into the pause next to the speech, the middle of a shorter
  one) and prints the command that re-runs from them, unless the quiet edges would widen the
  slice past the 120 s limit, where it says to listen in two slices that meet in a pause instead
  of printing a command it would refuse; a slice under a second is a warning too. L21 moves an
  edge its margin puts inside voiced audio out to that same quiet edge before it prints the
  command: on the run, a slice cut inside the neighbouring words heard "I'm in a terminal" over a
  word that holds "I'm in a, I'm in a terminal", and from the quiet edges it heard the restart.
  A 120 s slice typed to two decimals is inside the limit. A clip with no vertical render (or
  only a landscape one) is told to run `hc render P CLIP --formats vertical` or to slice the
  source. The source-drift stop skips only `hc listen --render`, which reads a rendered file; a
  source slice of a replaced source still stops with exit 1. Evidence, from the same run: a
  render slice started mid-word dropped the whole sentence before the one it heard, and one
  started inside a repeated question invented that question's first four words; on the same
  slices `hc listen` warns on the start (+23 and +45 dB over the floor), and from the first one's
  suggested start the dropped sentence reads back. A source slice started inside a word's tail
  put a filler's start 0.24 s early, on the edge, and misheard the last word, with both edges
  warned; re-run from the suggested quiet edges it read the whole sentence, with the folded
  filler at the time a manual slice had found. Twelve runs on a copy of the project made
  read-only left its file list, sizes and times unchanged and no temporary files behind. Tests
  in `test_listen.py` (the slice clock with a delayed audio stream, the source and output
  clocks, no writes into P and none left in the temp folder, a read-only project, usage errors,
  the edge report on synthetic energy, the L15 message, the slice limit, the missing-render
  advice and the drift stop for other commands) and `test_hidden_words.py` (the L21 command and
  its quiet edges).
- After this update every render goes stale once, stage A segments included (the caption and
  render code are in its key), and every QA report goes stale once (its code digest covers every
  module). A plan on disk is not re-made by the update: the plan changes above (the insert tie,
  L08, L21) apply from the next `hc plan`, or `hc all`, which plans first.

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
