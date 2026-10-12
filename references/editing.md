# Editing: composing the cut

A clip here is a composition, not an extraction. `pieces` render in the order you list them, so
you can open on the strongest line wherever it lives, rewind to the setup, drop a point that
repeats, and end on the best close. Every boundary is a word index; the planner snaps it to quiet
audio on the frame grid. You decide what is said and in what order; the planner decides exactly
where each cut falls.

## Order of work

1. Start from the ledger row: its span and its hook line.
2. Find exact first and last words with `hc find P "phrase"` and `hc words P A B`.
3. Pick a shape (below) and write the pieces, each with `expect` anchors.
4. `hc plan P CLIP`: read the piece table (source and output times, durations) and every lint.
5. Render, read the sheets, tighten. Two review passes is normal.

## Shapes

| Shape | When | Pieces |
|---|---|---|
| chronological | the first line is already the strongest line | one piece, or a few with tangents cut out |
| cold open | the hook lives mid-passage | claim, then setup, then payoff, then close |
| tease and passage | long (landscape) clips | 10-20 s tease of the moment the title promises, a dip, then the passage from a clean start |
| exchange | the reaction or the back-and-forth is the payoff | question, answer, reaction (or a callback from later) |

## Rules

Each rule says what to do, why, and how it looks in clips.json.

### 1. Open on the claim, not the approach to it

The first piece starts on the first word of the claim sentence. Viewers decide in the first
second or two; a wind-up spends that second on nothing. Out of source order, give it
`"role": "cold_open"`.

### 2. Kill preamble

"Yeah, so what I'm getting at is", "and it's this", "I mean, honestly", "okay so". Move
`words[0]` to the first content word of the sentence (`hc words P a-5 a+20` shows the
neighbourhood). A sentence that merely starts with "And" can stay when it stands on its own
("And then the bank called."). Plan lint L04 warns on continuation words, preamble
phrases, bare pronouns, negation openers and non-sentence starts; QA G6 measures the render.

### 3. The first five seconds must be a sentence

Line breaks in a transcript are not sentence boundaries. Piece 0 should start where the previous
source word ends in `.?!` or after a pause of at least 0.3 s. Dead agreement ("Totally.
Yeah, okay.") is not a sentence worth opening on. If the only clean start is a second
earlier, take the second.

Dropping a leading "So" or "And" from an otherwise complete sentence is fine ("So I fired my
biggest client." opens on "I"). `hc plan` still reports L04 "not at a sentence start", because
the previous source word is the marker you dropped: read the opening cold, and if it stands,
note that in one line and move on.

### 4. Never open on a negation of an unheard setup

"It's not that I didn't have a plan" answers something the viewer never heard. Start on the
affirmative sentence that follows, or put the setup in the clip (a short question piece after the
cold open is fine).

### 5. The subject must be in the cut

Never start after the subject of the first sentence, even when it costs a second or two. A
subject-less fragment can flip who did what, and speech recognition tends to invent a subject
for it ("I removed..." when the speaker said "she removed..."). Check the captions of the first
page.

### 6. Cut stammers, doubled phrases and filler hedges

Inside kept passages, remove halting entries ("we were, we were kind of, we were trying to"),
doubled phrases and throat-clearing hedges ("I mean, kind of, you know"). When a sentence starts
with a stumble ("the thing, the thing is"), put the in-point past both halves. Express with
`omit: [indices]` (the planner splits the piece and snaps both new edges) or by starting the
piece later. Fillers ("um", "uh") are removed by tightening automatically.

A false start can hide inside one transcript word: "and after a... after about a decade" may be
stored as three words, the second "after" folded into "about" (lint L15 `false_start` names such
words). No index separates the two attempts, so `omit` and `words[0]` cannot cut between them.
Start the piece on the folded word and push its in-edge past the first attempt with
`in_offset`: whole output frames, up to 60 (two seconds at 30 fps), + is later. Read the source
times in `hc words`, convert the gap to frames, re-plan, and check the first caption page and
G5: the opening must be the second attempt, whole. The plan prints an INFO line with the shift
in seconds; L04 no longer measures the opening against the word you stepped over, and the
`false_start` WARN clears once the first attempt is out of the clip. The words piece is still
tightened and snapped at its out-edge like any other: set `"tighten": false` on it (or a
`verbatim` range) when a pause inside it has to stay.

### 7. Never cut a qualifier that bounds the claim

A concession that limits nothing the clip says can go ("not that it's perfect, but..."). A qualifier that bounds the remaining claim stays: "not", "probably", "for us", "in
most cases", the "if" of a conditional. Removing one turns a careful claim into an overclaim in
the speaker's own voice. Plan lint L06 warns when an `omit` or an edge removes one between kept
words; QA cannot see planned meaning changes, so this one is on you (`fidelity.md`).

### 8. Re-pair thoughts after reordering

When the hook moves to the front, a later clause that refers back to it ("and that was the
cheap part") must travel with it, not stay behind in the setup where it now refers to
nothing. Split the piece so the clause follows the hook.

### 9. Repetition that builds versus repetition that stalls

Builds: escalation in the same breath ("so, so much better"), a rhythm (a list of three), or
new specificity each time. Keep it; it reads as conviction. Stalls: the same claim said again
after a pause with nothing new, often introduced by "like I said" or "so again". Keep one (the
most concrete or the best delivered, often the last) and cut the rest. A blanket dedupe flattens
conviction along with the padding, so decide each instance.

### 10. End clean on the strongest line

The last piece ends on the strongest closing line, at a sentence end, with `"role": "close"`.
Never append an echo of the hook: the platform's auto-replay is the loop, and an echo makes a
looping viewer hear the hook twice back to back. Do not end on "right?", "you know?" or "so,
yeah". The planner adds a 0.4 s tail to the final piece; add `hold_after` when the beat after the
line matters. Plan lint L05 warns when the last word is not sentence-final and no pause follows.

### 11. Protect reaction beats and pre-punchline pauses

- A laugh or reaction after a line: `"hold_after": 0.6` (seconds) on that piece keeps it. A hold
  keeps whatever is in the source after the last word: if someone speaks inside it, those words
  are heard and captioned (lint L16). Then end the piece where the laugh ends instead (a shorter
  hold, or a `time_src` piece over just the laugh). A hold never runs into another piece of the
  same clip: it stops where that piece starts.
- `hold_after` is coarse: the out-edge is snapped to quiet audio on the frame grid, so nearby
  values (1.30 and 1.42, say) can land on the same frame; the `hc plan` table shows the frame it
  chose. `out_offset` (whole frames, + is later) is the fine control, applied after the snap:
  set the hold roughly, then nudge the last frames with `out_offset`.
- Tightening only squashes gaps that are quiet (near the noise floor), and it protects any
  voiced sound inside a gap (a chuckle, a short laugh) with a margin around it, removing only
  the quiet parts. So laughter, applause and music survive. A silent pause before a punchline is
  quiet, so it would be squashed: protect it with `verbatim: [[a, b]]` over the words around it,
  or `"tighten": false` on that piece.
- The reverse case: a pause you want shorter that tightening leaves alone, because it is not
  quiet (a breath, a held beat with room sound, a word end the recogniser timed early). Split
  the piece in two around it: end the first piece on the last word before the pause and start
  the second on the next word. Each new edge is snapped like any piece edge, the join is a
  normal cut (the punch-in masks it), and `in_offset` / `out_offset` set how much of the beat
  stays. Do not lower `tighten.gap` for the whole clip to reach one pause.
- A reaction with no words (a nod, a face): a `time_src` piece with `"role": "reaction"`.

### 12. Cold-open tease for long clips: payoff inside 20 seconds

A mid-length landscape clip never opens chronologically. Prepend a 10-20 s tease of the exact
moment the title promises, then run the passage from a clean sentence start. The title's payoff
must exist inside the clip and arrive in the first 20 seconds; the teased line replaying later in
context is normal and good. Put `"join": "dip"` on the first passage piece: a hard cut there reads
as abrupt and a full blackout reads as a glitch, so the dip darkens the picture by 60% and back. To lengthen a
long clip, extend toward the strongest closer rather than padding the front. Plan lint L08 checks
when the hook card's accent phrase is first spoken (8 s vertical, 20 s landscape).

### 13. Motion from frame 0

The first two seconds decide whether a clip gets watched, and a still frame spends them. The
picture must already be moving on frame 0: start piece 0 on speech, not on a breath or a pause.
The hook card overlays live picture; never hold a freeze frame under it. Same-shot jump cuts are
masked by the preset's punch-in (`motion.punch_in`); turn it off for a clip with
`"motion": {"punch_in": 1.0}`.

**Words under the hook card are never captioned.** The card sits in the caption band, so every
word spoken while it is up is heard and not shown; captions start when the card ends. That is
a trade you make on purpose. Either keep the card short (leave `dur` at `"auto"`: 1.5 to 3.0 s)
and let the opening words go uncaptioned because the card says them, or, when the opening line
must be read (a name, a number, a punchline a muted viewer needs), do not put it under the
card: shorten the card until it ends before that line, or open with a beat of something else.
Never stretch the card over a line and then hold a pause so the captions can catch up: that is
dead air in the first two seconds.

### 14. Tighter beats complete

When in doubt, shorter. A typical review runs 52 s, then 44 s (hook first, repetition gone),
then 28 s (halting entry, a doubled phrase and a hedge cut, one thought re-paired). A clip that is
short of its target band but complete is fine; never pad back material you cut for a reason.

### 15. Length bands

`defaults.length` holds `[min, target_lo, target_hi, max]` in seconds per format:

| Format | Band | Meaning |
|---|---|---|
| vertical | `[15, 30, 60, 90]` | under 15 s rarely carries an idea; 30-60 s is the default target; up to 90 s when the arc holds |
| landscape | `[60, 300, 900, 1800]` | a mid-length clip is 5-15 minutes; a 50-second "clip" is a short |

Platforms accept verticals up to three minutes, but retention, not duration, is the goal. Public
data points: a peer-reviewed study of TikTok viewing (Zannettou et al., CHI 2024) found about a
quarter of views skipped before 20% of the video, so the opening decides most outcomes; a large
vendor dataset of brand Reels found engagement peaking at 45-60 s (a heuristic, not a law).
Plan lint L07 warns outside `[min, max]`.

### 16. Cut promo middles, risky claims and bad endings

Skip programme pitches, sponsor reads and "link in the description" asides with a piece split or
an `omit`. End before profanity when the user's platforms penalise it. Cut around claims you
cannot stand behind (`fidelity.md`).

### 17. Joins

`"join": "cut"` (default) for normal joins; `"join": "dip"` for a tease into a passage or a large
jump in time. Never crossfade speech: it ghosts the picture and doubles syllables. Joins of kind
piece, omit and dip also start a new caption page, so a page never straddles a seam: no caption
of the next piece shows before the cut and none of the last piece lingers after it.

### 18. Cut rate

Every removal is a picture jump. Talking-head shorts that perform well run at a median of about
one cut every three seconds (0.33 per second, Dost and Huang), and faster pacing lowered
sustained engagement in their experiment. Plan lint L09 warns above 0.5 joins per second.
Remedies: raise `tighten.gap` or `tighten.min_cut`, set `"tighten": false` on a fluent piece, or
mark a fluent stretch `verbatim`.

### 19. Framing follows the speaker

When a new shot starts inside the clip, add a framing mark at or after its start (lint L11):
`track` on the speaker who owns the line, `letterbox` for a group or wide shot, `fixed` to pin a
position. `hc shots` proposes marks and draws the 9:16 window on `sheets/shots_<clip>.jpg`.
`crop` below 1.0 zooms in at the cost of sharpness (lint L12 warns past a 3x upscale; the source
resolution is the ceiling, so say so).

**Gallery and split-screen recordings** (a video call in tiles, a two-shot, a side-by-side
interview). The largest face is not the speaker, and a crop on the wrong tile puts one person's
face over another person's words for the whole clip.

1. `hc shots` prints `MULTI-FACE` for such a shot, with each face (A, B, ...) and its x, and
   proposes letterbox. Never paste a single-face `track` mark for it.
2. Set `"speakers"` once at the top of clips.json: each speaker label from `transcript.md`
   with that person's face x (`spec.md`). Remove single-face marks you pasted for those shots.
3. With no marks in the clip (or from a `{"mode": "speaker"}` mark on), framing follows the
   turns: the window moves to whoever talks for at least a second and three words; an
   interjection ("Yeah.") does not move it.
4. Without speaker labels, keep the default letterbox, or show one tile with
   `{"mode": "letterbox", "rect": [x, y, w, h]}` while only that person talks (about 1.7x
   upscale for one tile of a 2-up call, where a crop needs about 5x). G16 judges a tile like a
   crop: the clip needs its `speaker` and that person's x in `speakers`, or the tile of a
   multi-face shot fails as unlabeled.
5. A deliberate cutaway to the listener gets `"reaction": true` on its mark.

QA G16 fails a render that frames someone other than the speaker for 1.5 s or more. Gallery
recordings often carry black bars too: ingest measures them (`video.active_area`) and framing
stays inside the picture.

### 20. Garbled audio

If a phrase is garbled or two people talk over each other, cut around it. Never guess the words
and never caption a guess (`fidelity.md`).

The opposite case is speech that is clear and simply missing from the transcript (a short
repeat right after a sentence, a quick aside): it plays with no caption, G5 hears extra words
and G10 reports uncaptioned voiced audio. Confirm what is said by transcribing that stretch on
its own (`qa.md`, "Checking sound that has no words"), then caption it with `insert`: the
words exactly as spoken, the index of the transcript word they follow, and the source seconds
they are heard in. An insert is for words you can state verbatim; anything less certain is
garbled audio, and the rule above applies.

## Each move in clips.json

| Move | clips.json |
|---|---|
| open on a mid-passage line | first piece is that sentence, `"role": "cold_open"`; later pieces carry the setup |
| kill preamble | move the first piece's `words[0]` to the first content word |
| drop a stammer or doubled phrase inside a piece | `"omit": [i, j]` |
| drop a whole tangent | split the piece into two around it |
| keep a fluent or timed stretch untouched | `"verbatim": [[a, b]]` |
| keep the laugh or beat after a line | `"hold_after": 0.6` on that piece |
| a reaction without words | `{"time_src": ["31:02.4", "31:04.0"], "role": "reaction"}` |
| soften a hard seam | `"join": "dip"` on the piece after the seam |
| a quiet speaker | `"gain_db": 3` on that piece (-12 to 12) |
| stop tightening one piece | `"tighten": false` on the piece |
| stop tightening a whole clip | `"tighten": {"enabled": false}` on the clip |
| a misheard word in the captions | `"fix": {"2215": "Kubernetes"}` |
| hide a caption token, keep the audio | `"fix": {"2216": ""}` |
| guard indices against drift | `"expect": ["first words", "last words"]` on every piece |
| who is framed, from when | `{"at_src": "24:03.5", "mode": "track", "x": 0.31}` |
| a group or wide shot | `{"at_src": "24:10.0", "mode": "letterbox"}` |
| framing follows who talks (gallery, two-shot) | top-level `"speakers": {"Ann": {"x": 0.27}, "Ben": {"x": 0.75}}` |
| one tile of a gallery | `{"at_src": "24:20.0", "mode": "letterbox", "rect": [0.5, 0.25, 0.5, 0.5]}` |
| a deliberate listener cutaway | `"reaction": true` on that mark |
| a stray syllable at a seam | `"in_offset": 2` (frames) on the piece, or `"out_offset": -2` |
| skip a false start folded into the first word | `"in_offset": 17` on that piece (frames, up to 60) |
| fine-tune an out-edge after `hold_after` | `"out_offset": 2` on that piece (frames; the hold snaps to the grid) |
| shorten a pause tightening protects (a breath, a held beat) | split the piece in two around it |
| caption speech the transcript missed | `"insert": [{"after": 2231, "text": "Say it again.", "src": [744.9, 745.6]}]` |
| pills on every page of one clip (or on none) | `"caption_backing": "always"` (or `"never"`) on the clip |
| captions over a face | `"caption_y": 1120` (output pixels, vertical) |
| no punch-ins | `"motion": {"punch_in": 1.0}` |
| landscape chapter | `"chapter": "A claim, not a topic"` on the piece that starts it |

## A worked edit

A consultant explains how one client grew to half her revenue, then says the line worth
clipping. Chronological first draft, 33 s, the claim at 18 s:

```json
"c01-fired-biggest-client": {
  "ledger": "M03",
  "title": "Why I fired my biggest client",
  "speaker": "Speaker B",
  "pieces": [
    {"words": [5120, 5244], "expect": ["Three years ago one", "best month ever"]}
  ]
}
```

After: open on the claim, rewind to the why, drop a doubled phrase, keep the laugh, end on her
strongest line. 25 s:

```json
"c01-fired-biggest-client": {
  "ledger": "M03",
  "title": "Why I fired my biggest client",
  "speaker": "Speaker B",
  "pieces": [
    {"words": [5202, 5206], "role": "cold_open", "expect": ["I fired my", "biggest client"]},
    {"words": [5131, 5180], "role": "setup", "expect": ["Half of everything we", "came from them"]},
    {"words": [5213, 5244], "role": "close", "hold_after": 0.6, "expect": ["The month after", "best month ever"]}
  ],
  "omit": [5150, 5151, 5152],
  "hook_card": {"lines": ["I FIRED MY", "*BIGGEST* CLIENT"], "dur": "auto"}
}
```

Then `hc plan P c01-fired-biggest-client` and read the table: piece durations, the joins, the tightening
total, and the lints.

## Turning review notes into edits

| The user says | Edit |
|---|---|
| "open stronger" | find the strongest sentence in the span; make it piece 0 |
| "too long" | drop the weakest piece, then omit stammers; aim for the next band down |
| "it starts mid-thought" | move `words[0]` to a sentence start (rules 2-5) |
| "ends weak" | end on the strongest line and cut what follows it |
| "the laugh got cut" | `hold_after` on that piece |
| "too choppy" | fewer removals: raise `tighten.gap`, add `verbatim` |
| "caption says the wrong word" | `fix` by index (or the glossary, if it is wrong everywhere) |
| "wrong person on screen at 0:12" | read the `hc plan` table to map output 0:12 to source time; add a framing mark there. In a gallery or two-shot: set `speakers` and remove pasted marks (rule 19) |
| "I hear a bit of a word at the cut" | `in_offset` / `out_offset` on that piece (`qa.md`) |
| "he starts the sentence twice" | start the piece on the second attempt; if both sit in one transcript word, `in_offset` (rule 6) |
| "that pause is too long" | if tightening left it (a breath, a beat): split the piece around it (rule 11) |
| "she says something there and no caption shows" | `insert` the words (rule 20) |
| "the caption boxes flicker on and off" | `"caption_backing": "always"` or `"never"` on the clip |
| "the caption comes up before he says it" | G10 names the token; hide the mis-timed word with `fix` and carry its text on the next (`qa.md` G10) |
| "someone talks over the laugh" | end the piece where the laugh ends instead of `hold_after` |
| "captions cover the mouth" | `caption_y` |
| "the hook card is unclear" | rewrite `hook_card.lines` (`packaging.md`) |
| "the cover doesn't read" | rewrite `cover.headline` (parse-cold test) or pick another `at_out` |
| "make another version with a different hook" | a new clip id with the same pieces and a different `hook_card` |

Then re-run only that clip: `hc all P <clip>`.
