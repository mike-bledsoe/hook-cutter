# QA

`hc plan` lints the edit before anything renders. `hc qa P` runs the gates on every render and
writes `qa/<clip>.json`, `qa/<clip>.md` and `qa/summary.md`.

A clean exit means nothing was proven wrong. It does not mean the clip is good. Captions across a
face, a hook card off the frame edge and the wrong person framed have all passed clean exit codes
in real use. That is why gates C and D include looking.

## Statuses and exit codes

Every gate reports PASS, WARN, FAIL or SKIPPED (with a reason), plus its evidence. "Could not
check" is SKIPPED, never PASS. A clip's status is its worst gate: FAIL, then WARN, then PASS.
A clip that would be PASS but has an evidence gap (a gate SKIPPED because something it needed
was missing) is **INCOMPLETE**, never PASS. The console prints each gap. A WARN clip with such
a gap stays WARN and counts as incomplete: `hc status` and the `hc all` clip line show `qa WARN
(incomplete)`, `--require-coverage` exits 2 and `hc export` holds it.

| Exit | Means |
|---|---|
| 0 | no FAIL (and, under `--require-coverage`, no gap) |
| 1 | at least one FAIL |
| 2 | evidence gap under `--require-coverage`: a clip is INCOMPLETE, or WARN with checks that could not run |
| 3 | tool or environment error, including a speech backend that crashes or a face tool that fails inside G9 (run `hc doctor`) |

`--no-asr` skips the re-transcription: G5 and the ASR part of G10 are SKIPPED, while G10's
energy checks still run. It is faster, but the word diff is the only independent witness that
tightening did not eat a word. **Run a full QA (with ASR, with `--require-coverage`) before the
final export.** `--styles` and `--formats` restrict the run; the report records what was
checked (`checked: {styles, formats}`).

Audio gates run once per clip and format, and every style must share identical audio; if two
styles diverge, that is a FAIL ("styles diverged").

QA results go stale with the render and with the checks themselves: a report made by other
code is stale ("the QA checks changed since this report was made") and `hc export` holds it.
Re-run QA after every re-render and after updating the skill. G0 proves the render matches the
current plan and inputs.
`qa/summary.md` lists every clip in clips.json from its `qa/<clip>.json`, marking clips this run
did not check `(not re-run: checked <time>)` and reports older than their render or plan
`(stale)`.

`hc export` holds back clips whose QA is FAIL, INCOMPLETE, stale or missing or has evidence
gaps (a WARN made with `--no-asr` counts), and renders that no longer match their inputs
(preset, transcript, shots or code changed since the render): it lists them, exports the rest
and exits 1. A held clip keeps no exported file, also in a run restricted with `--styles` or
`--formats`. `--force` exports them anyway; use it only when the user says so. The manifest
records each clip's QA status and what was checked. The clip number in a file name is the
clip's position in clips.json: after adding, dropping or reordering clips, run `hc export P`
for all clips, because a run that names some clips renumbers only those.

## Gate D

- Zero FAIL.
- Every WARN read, then fixed or explained to the user in one line.
- Every image under EYES opened, and the ones you opened listed in your report.
- Every SKIPPED and every INCOMPLETE reported as an evidence gap, not glossed over.

## EYES

Each `qa/<clip>.md` ends with an EYES list of images, for every style checked. Open every one
with the Read tool. An image listed as missing, or one that does not show this clip, is itself a
finding.

| Image | Look for |
|---|---|
| `sheets/<clip>__<fmt>__<style>.contact.jpg` (12 frames with timecodes; cell 1 is frame 0) | the right person framed in every cell; no face cut by the crop; no black band; captions clear of the mouth; the hook card fully readable in cell 1 (frame 0 is the default thumbnail); the accent on the word you meant; letterbox only where you put it |
| `sheets/<clip>__<fmt>.seams.jpg` (the base picture just before and just after every join, before captions and before any dip darkening; the join kind is printed under each pair, and a dip says `(darkening not shown)`) | no black or flash frame; no framing jump that lands on the wrong face; punch-ins that read as intentional; dips only where you put them |
| `sheets/<clip>__<fmt>__<style>.flags.jpg` (only when G8 flagged frames) | the flagged flash or black frames: real defects or harmless (a camera flash in the source) |
| `sheets/<clip>.cover-candidates.jpg` | the chosen frame is expressive, sharp and shows the claim owner |
| `covers/<clip>__<style>.grid.jpg` | the headline and the face inside the grid crop |
| `sheets/covers-grid__<style>.jpg` and `sheets/covers-grid.jpg` | every headline reads at phone scale; faces clear the text; the set is not one tile repeated |

If an image looks wrong, extract more frames with `hc peek P START END --file <render>` before
deciding what to change. A covers grid listed as missing (an interrupted cover step) is rebuilt in
seconds from the covers on disk with `hc cover P --grid`; nothing else needs to run again.

### Image budget

An agent's context keeps only the most recent images. In a long run (mining a whole episode,
reviewing many clips) the older ones are dropped, and an agent that keeps opening sheets ends up
re-opening the same ones without making progress. What works:

- Open at most two images at a time.
- Straight after looking, write what the image shows into a notes file (`P/notes/`, the ledger,
  or the review file of the clip), in words a later step can act on without the picture:
  "contact c03 cell 7: the guest framed, captions clear; cell 9: hook card covers the chin".
- Never re-open an image you have written up. The written line is the evidence; a later agent
  reads the line, not the picture.
- When you hand image-heavy work to other agents, split it by rows of the ledger or by clip (four
  or five rows, or one clip, per agent), never the whole source to one agent.
- Name checkpoint files for what they are (`PROGRESS.md`, `notes/c03-review.md`); some agent
  harnesses refuse generic names such as `REPORT.md` from a subagent.

## By ear

`hc export` writes `P/export/by-ear.md`: for every exported clip, its opening, each join with the
words on both sides, its ending, the words the transcript flags (`timing_uncertain`,
`false_start`, `unworded_audio` and the like) and the sound-related QA warnings (G5, G6, G7,
G10), each with its time in the exported file. The agent cannot hear; the person approving the
clips plays each one with this list open and sends back a plain note with a time for anything
wrong ("clipped word at 0:03.0"). Turn each note into one spec edit (`editing.md`, notes to edits).

## Gates

| Gate | Checks | Catches | What to do |
|---|---|---|---|
| G0 render fresh | the render log's plan hash against the current plan, and its input hash (clip entry, preset, font files, safe-zone margins, code) against what a render made now would record. A stale render skips G9, G10 and G14 as gaps. A render with no log (an interrupted render) is never up to date: `hc render` makes it again | a render made before the last plan or preset change, so QA would check the old video | `hc render P CLIP --styles S`, then QA again |
| G1 plan lints | re-runs the planner: FAIL lints are FAIL, WARN lints are WARN | everything in the lint table below | fix clips.json |
| G2 duration | render duration within one frame of the plan; length band | a stale or broken render (FAIL); a clip outside its band (WARN) | FAIL: `hc render P CLIP --force`; band: trim, or explain |
| G3 streams | size, plan fps, yuv420p, BT.709 tags, H.264; AAC 48 kHz stereo; audio present; audio and video durations within one frame; regular AAC packets with no timestamp holes (a packet one or two samples off the frame grid that comes back to it is not a hole); a clean full decode; the audio checksum recomputed from each file matches its log (this row also runs for a single style; a log with no `audio_md5` is an evidence gap) | timelines that play locally and glitch after a platform transcodes them; a render edited after the fact | re-render with `--force`; if it persists, run `hc doctor` and report it |
| G5 word diff | re-transcribes the render and aligns it with the intended words (piece words minus `omit` minus fillers, never the tightened result, plus the words of every `insert`, placed after their `after` word): a missing content word or negation is FAIL; other missing words and extra words are WARN. Words two pieces share (plan `overlaps`) heard twice are FAIL "duplicated at join"; 6 or more consecutive extra content words are FAIL "uncaptioned speech" | tightening eating a word, an edge clipping a syllable, an edge swallowing a neighbouring word, a hold that replays the next piece, speech the transcript never had | missing: `verbatim` around the word, or move the edge; extra at a seam: see "An extra word at a seam" below; duplicated: end the first piece before the shared words; uncaptioned: re-transcribe (`hc transcribe P --force` re-listens to gaps) or cut it out |
| G6 open and close | lints L04 and L05, plus measured audio: loud in the first 30 ms means the clip starts mid-sound; the same for the 60 ms before the final fade, with one exception: when what sounds there is the last consonant of a word (unvoiced frication such as "s" or "sh", straight after a voiced sound) the clip ends on a finished consonant and the fade is its decay, which is not a WARN. A vowel or nasal at full level is still "ends mid-sound", and so is a consonant with no voiced sound right before it (the next word starting). Dead air: the first voiced sound more than 0.35 s after a segment start (0.25 s at 0:00) is WARN | clipped first syllables, mid-word endings, a page of captions over silence | start on a sentence in a pause; end on a sentence end; `hold_after`, then `out_offset` for the last frames; dead air: start the piece later or `in_offset` |
| G7 loudness | integrated -14 LUFS (plus or minus 1), true peak at most -1.0 dBTP | normalisation failures | re-render; if it persists, report it with the json |
| G8 picture | black within three frames of a join or at frame 0 (FAIL); a black frame anywhere else (WARN); a one-frame flash, where the mean luma jumps by more than 40 and straight back (WARN, drawn on the flags sheet); a freeze of a second or more (WARN); dark bands: rows at the frame edge (or full width), outside the caption band, whose brightest pixel over 8 frames is at most 20 and that cover more than 3% of the height (FAIL when the source has an `active_area` or the band covers 10% or more, else WARN) | a source fade at an edge, a glitch frame, a frozen screen share, letterbox bars of the source left in the vertical frame | move the edge off the fade; cut or letterbox the frozen stretch; bands: re-ingest so `active_area` is measured, or use a `letterbox` mark with `rect` |
| G9 caption layout | every page box and the hook card inside the safe zone the render used (clips.json `defaults.safezone` when set, else the preset's `safezone`, else organic) (WARN) and the frame (FAIL); captions over a detected mouth, nose tip to just below the lower lip, checked on each style's own base picture (WARN; chin and beard are fine); a face-tool failure is a tool error (exit 3); the render's caption issues (`min_page`, `word_shrunk`, `overflow`, `hook_small`, `caption_y_clamped` are WARN, others are listed) | captions under platform interface, captions across a mouth, pages too short to read, a hook card no bigger than the captions | `caption_y`, a preset `layout.y`, shorter hook lines, fewer words per line (`captions.md`) |
| G10 caption sync | four checks after the hook card ends, the first three without ASR. (a) Energy timing: a token whose display time (`s_out - lead_ms`) falls inside a pause of 0.25 s or more, a token spanning a pause of 0.20 s or more, a page whose first voiced sound comes 0.6 s or more after it appears. FAIL on any early page or on 3 or more tokens 0.5 s or more early; WARN on any other. Loudness alone calls a laugh, a breath or a lip click "voiced", so the same check also runs against speech-like sound: sustained (60 ms or more), pitched or formant-shaped sound in the speech band, with a word's first consonant counted in. A token shown more than 0.2 s before the first speech-like sound is WARN, with the seconds of other sound in between. Limit: a strongly voiced laugh is a vowel as far as the signal goes and is not seen by this check, so G10 also names every caption token that sits on a word the transcript flags `timing_uncertain` (WARN: the audio cannot confirm that onset; look at the frames). (b) Uncaptioned speech: voiced audio of 1.5 s or more with no caption token (WARN; FAIL at 3.0 s or more when G5 heard 4 or more extra words inside it). (c) Display lag: the caption display cross-correlated with the voicing over plus or minus 1 s; deviation from `lead_ms` over 0.12 s is WARN, over 0.25 s FAIL. (d) With ASR: drift across the clip over 0.10 s FAIL, a median offset over 0.15 s WARN, the worst page reported | captions that light before or after the words, captions over silence, a caption that lights under a laugh or on a click before its word, speech with no captions, a constant offset, drift | early or late tokens: `hc transcribe P --retime`, re-plan, re-render; a token early under another sound: check the frames with `hc peek`, then hide the mis-timed word and carry its text on the next one with `fix` (`"<i>": ""`, `"<i+1>": "both words"`), or move the piece edge past the sound; uncaptioned speech: re-transcribe or cut it; lag or drift: report it with the qa json |
| G11 faithfulness | claims on the hook card, the cover headline (`cover.headline`, else the hook card, else the clip `title` when a vertical clip has neither: the text `hc cover` prints), the clip `title`, the post title and caption, and the chapter titles of a landscape clip, against the clip's spoken words. Numbers (digits, multipliers like 10X or x3, ordinals, ranges like 80-90%, number words before a noun or unit: "ONE PERSON" counts, "A NORMAL ONE" does not) missing from the speech are FAIL. A spelled-out amount with a scale word counts wherever it stands (`ONE MILLION`, `FIVE HUNDRED THOUSAND`). A leading decimal point counts (`.5%` is 0.5%); a leading minus sign is not checked (direction is carried by words). `0 X` / `ZERO X` matches a spoken zero, or a spoken `no X` / `nothing on X` / `nobody X` with the negation within three words before the noun; a bare `No,` elsewhere states nothing. Names (capitalised words, also in first position) missing are WARN; a name matches only itself and its possessive (`James`, `James's`), never a word that shares its stem. The clip's speaker and the speaker labels in the clip are exempt on the clip title, post title and caption (not on a title printed on the cover, which is also checked as `cover headline (the title)`). In all-caps and title-case surfaces, an uncommon word the speaker never said is WARN. "1 YEAR" matches "a year", "2026" matches "twenty twenty six" | invented or misremembered numbers, misspelled names, words put in the speaker's mouth | use a number the speaker said; check spellings against the glossary; reword with the speaker's words |
| G12 cover | decodes fully, 1080x1920; the headline ink inside the fixed grid band y 240-1680 (FAIL); the face box inside the band (FAIL when it leaves it); the top of the head inside the band, judged only when the cover json measured it in the picture (`crown_method: image`): FAIL when the headline shift pushed a head that fitted out or a close-up was shifted more than 80 px, WARN for a close-up whose head starts above the band even unshifted. A top of the head estimated from the face box (`crown_method: estimate`, and covers made before the field existed) is not a finding: on a bald or close-cropped head the estimate sits far above the real head, so look at the `.grid.jpg`; no face found (WARN: overlap unchecked, never "face clear"); a fallback font (WARN); a frame of a speaker other than the clip's `speaker` (WARN; names are matched the way `hc cover` matches them: a prefix of the name's words, case and accents ignored); a dark band on the cover (FAIL); the words on the cover (cover json `headline`) are not the clip's headline now, the vertical render was made again after the cover, or the cover json records a negative shift (FAIL); the cover json's own warnings; grid and mobile previews exist | truncated files, headlines or heads lost to the grid crop, text over faces, a host's face over a guest's claim | `hc cover P CLIP` (re-render first when G0 failed); adjust `cover.shift` (it only slides the photo up) or `cover.at_out` |
| G13 set | more than 15 s of shared source between clips; near-duplicate hook cards (2 or more shared content words and a Jaccard overlap of 0.5 or more, or an overlap coefficient of 0.67 or more when both have at least 3 words); hook card, cover and title of one clip echoing each other (2 or more shared content words and an overlap coefficient of 0.8 or more) | a set that repeats itself, surfaces that do one job three times | carve the spans apart or bank one; rewrite one surface (`packaging.md`) |
| G14 accent fired | every hook-card accent phrase is matched in the captions | an accent on a word the speaker never says | accent a word they actually say |
| G15 render log | no timestamp-damage warnings from ffmpeg (non-monotonic or invalid timestamps) | files that play locally and break after upload | re-render with `--force`; if it persists, the source timestamps are damaged: report it, do not ship |
| G16 framing attribution | vertical renders, for the parts of the clip in multi-face shots: every word inside a single-face crop or a `rect` tile (a window that does not hold every main face of the shot, judged by position, not width) belongs to the person framed (the word's speaker label, else the clip's `speaker`, looked up in the `speakers` map). No label: FAIL. A label with no x in `speakers`: FAIL. The framed window misses that person for 1.5 s or more: FAIL (WARN inside a `reaction` piece or mark); shorter misses WARN. When the render log records the window's rows and `speakers` gives the person a `y`, that y must be inside the window too. People stacked in one column of the picture (same x) are counted as one face by the shot analysis, so a shot of only such a pair is not multi-face and is not checked here: give them `speakers` entries with `y` and look at the contact sheet. SKIPPED when the clip has no multi-face shot. `hc render` analyses the shots itself when the framing needs face data, so a rendered clip normally has `shots/<clip>.json`; when it is missing, unreadable, or was analysed for other pieces (1.5 s or more of the clip's source not covered) that is one evidence gap for the clip (target `vertical (all styles)`), not one per style | the wrong person framed while someone else talks, which a contact sheet only shows if you look at the right cell | set `"speakers"` in clips.json (x from the shots sheet) and remove pasted single-face marks; mark a deliberate cutaway `"reaction": true`; for the gap: `hc shots P CLIP`, look at the sheet, render again |

The audio/video sync of the assembly itself is proven by the test suite (`hc test`), not per
render.

## Plan lints

`hc plan` prints these; QA G1 re-reports them. INFO lines are notes, not problems.

| Lint | Level | Checks | What to do |
|---|---|---|---|
| L01 | FAIL (unknown keys WARN) | schema and bounds: indices in range, a <= b, time pieces inside the source, valid ids; `caption_y` on the 1920 px frame; every style resolves to a preset; `speakers`, `rect`, `reaction`, `in_offset`/`out_offset` (whole frames, -60 to 60), `insert` and `caption_backing` values; `motion` values (numbers of 0 or more); `cover.shift` of 0 or more; `post.title` and `post.caption` strings (another `post` key is WARN: exported, not checked); a `defaults.safezone` or preset `safezone` that names no profile; `hook_card.dur` over 3 s is WARN and as long as the planned clip (after tighten and omit; less 1 s) is FAIL; a `speakers` label no word or clip uses is WARN; INFO when a trimming offset at a shared cut is absorbed by the overlap and removes nothing | fix the key or value (`spec.md`) |
| L02 | FAIL | `expect` anchors do not match the words at those indices | the indices drifted: `hc find` the phrase and correct them |
| L03 | FAIL | `transcript_sha` matches neither the transcript's `sha256`, its `text_sha`, nor any of its `text_compatible_shas` (a re-time keeps the old sha valid). For a caption_only clip: its own clip-level `transcript_sha` does not match `transcripts/<id>.json`, or the clip uses `fix`, `emphasis` or `insert`, has no pin, and its transcript was re-made with other word texts (`text_changed_from` in that file) | re-anchor every piece (the `expect` anchors show which), then update or remove `transcript_sha`; caption_only: check the indices with `hc words P A B --clip ID`, then set the clip's `transcript_sha` to the sha256 `hc transcribe` prints |
| L04 | FAIL or WARN | opener: a filler first word (FAIL, also when `fix` hides its caption); a continuation word, preamble, bare pronoun, negation, or a start that is not a sentence start (WARN). "This/that/these/those" followed by a noun is a determiner, not a bare pronoun. Opener and closer are judged on the first and last word heard, so a display merge there is fine | `editing.md` rules 2-5; a "not a sentence start" WARN after you deliberately dropped a leading "So" or "And" is expected: confirm the sentence stands. An opening you placed by hand is not judged against the word you stepped over: a `time_src` opener, or a words piece whose `in_offset` moved the in-edge more than 12 frames later, past the end of the word before it (the false-start remedy; the L01 INFO names the shift). That word is not in the clip, so it says nothing about where the sentence starts. A nudge of 12 frames or fewer is still judged |
| L05 | WARN | closer: the last word is not sentence-final, or it is but speech continues immediately (less than 0.15 s and no dip in level) | end on a sentence end that is followed by a breath |
| L06 | WARN | an `omit` or a piece edge removes a negation, hedge or qualifier between kept words, also when it is the last word of one piece or the first of the next (not when a kept word repeats it within 6 words: a restart); or a `fix` hides one from the captions or changes a number | restore it unless the meaning is unchanged (`fidelity.md`) |
| L07 | WARN | duration outside the length band | trim, or explain why it stands |
| L08 | WARN | the hook-card accent is first spoken after 8 s (vertical) or 20 s (landscape), or never, or only while the card is up | cold open on the payoff, or accent a word said after the card |
| L09 | WARN | more than 0.5 joins per second | fewer removals: raise `tighten.gap`, use `verbatim` |
| L10 | WARN or INFO | a piece edge falls inside voiced audio louder than the floor + 18 dB (WARN); omit and tighten joins are INFO unless louder than the floor + 24 dB. INFO "the last word's final consonant runs into the next word": the clip's last out-edge was moved a frame or two later by itself, so the closing fade sits on the dip between the two words and the consonant is heard ("sense", not "sen"). This guard applies to the clip's last out-edge only and only when the last piece sets no `out_offset`; an `out_offset` there replaces it (the message names the value that keeps the same edge) | move the edge to a pause (`in_offset`/`out_offset`); check the seam sheet and G5; the consonant INFO needs no action |
| L11 | WARN | a shot starts inside the clip with no framing mark at or after it | add a framing mark (`hc shots P CLIP`), or rely on `speakers` |
| L12 | WARN | a vertical framing upscales more than 3x (measured on the picture inside any black bars) | crop closer to 1.0, a `letterbox` mark with `rect`, or letterbox; tell the user the source is the ceiling |
| L13 | WARN | the G11 claims check on the hook card, cover headline, clip title, post copy and landscape chapter titles, before rendering | as G11 |
| L14 | WARN | more than 15 s of shared source with another clip | carve apart or bank one (variants are the exception) |
| L15 | WARN | transcript flags inside the clip: a repetition loop, low confidence, `unworded_audio` (sound with no words: laughter, music or speech ASR missed), `dropout` (the speakers VTT has words ASR did not transcribe), `speaker_uncertain`, `no_audio` (words with no audible sound: likely invented), `false_start` ("possible false start or repeated attempt": one word run that holds two attempts at a phrase), `timing_uncertain` ("word timing near src T is uncertain (overlapping sound)": a word start under a laugh or another voice) | read those words with `hc words`; check unworded sound as described below before keeping it; check attribution; omit `no_audio` words; `false_start`: open after the first attempt (`in_offset`, up to 60 frames) or keep both attempts on purpose; the WARN goes away once the clip no longer plays the first attempt; `timing_uncertain`: check that caption's onset in the render (G10, `hc peek`) |
| L16 | WARN | `hold_after` pulls in words (they will be heard and captioned), or was clamped where another piece starts, or stopped before a word the clip omits | end the piece where the laugh or beat ends instead of holding over speech |
| L17 | FAIL with a cold open, else WARN | two pieces share words or more than 0.1 s of source: the audience hears it twice | make the ranges meet without overlapping |
| L18 | WARN | a segment opens on more than 0.35 s of silence (0.25 s at 0:00) before its first word | start the piece later, or `in_offset` |
| L19 | FAIL or WARN | 3.0 s or more of voiced audio inside the clip with no transcript words and no `unworded_audio` flag (FAIL); 1.5 s or more (WARN) | re-run `hc transcribe P` (it re-listens to gaps), or keep that stretch out of the clip |
| L20 | WARN | the hook card sets smaller than 1.3 times the caption size | shorten the hook lines to the length the message suggests |

### An extra word at a seam

G5 or your ears report a stray syllable at a join ("and" before a piece start, the tail of a
dropped word). The planner already cuts at the quietest frame near the edge; when two words touch
with no pause, move the edge yourself: `"in_offset": 2` on the piece moves its in-edge two frames
later (`-2` earlier), `"out_offset": -2` moves its out-edge two frames earlier. Whole frames,
-60 to 60 (two seconds at 30 fps): enough to step over a false start. Beyond 12 frames the plan
adds an INFO line with the shift in seconds, so a typo does not move an edge half a second
unnoticed. Re-plan and check L10 and L18, then the seam sheet.

### Checking sound that has no words

L15 `unworded_audio` and L19 mean the clip holds sound the transcript has no words for: a laugh,
music, or speech the recogniser missed. You cannot listen, so ask the recogniser about that
stretch alone. Cut it out of the source (`P/source.<ext>`, source seconds) with a little margin
and transcribe the snippet as a scratch clip (an id that is not in clips.json):

```
ffmpeg -ss 132.4 -to 133.8 -i P/source.mp4 -vn P/cache/scratch-a.wav
hc transcribe P --file cache/scratch-a.wav --clip scratch-a
hc words P 0 20 --clip scratch-a
```

- No words, or only a filler: it is a laugh, a breath or room sound. Keep it or cut it as the
  edit needs; nothing to caption.
- Words: it is speech the transcript missed, and without help it plays with no caption (G5
  reports it as extra words, G10 as uncaptioned voiced audio). Caption it with the clip key
  `insert` (`spec.md`): the words exactly as heard, after the transcript word they follow, with
  the source seconds they are spoken in. If the words are not clear enough to write down
  verbatim, move the piece edge so the stretch is not in the clip.

Slices shorter than about a second transcribe poorly: take 0.2 s of margin on each side, and
when the slice comes back empty but the level is clearly speech (`hc peek` shows a moving
mouth), try a slightly longer one before deciding. Delete `P/transcripts/scratch-a.*` when done.

## Rules

- Fix the spec, never the output. Do not hand-edit renders, `.captions.json`, covers or QA files.
- Never loosen a gate to get a pass. For contributors: loosening a check requires proving again
  that its seeded defect still fails.
- QA results go stale with the render and with the checks. Re-run QA after every re-render and
  after updating the skill.
- QA does not move a ledger row. CUT means a render exists; approval is gate E.
