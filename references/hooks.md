# Finding hooks

The scripts do the mechanics. Picking the moments is your job, and it is most of the quality.
This file is the procedure: read everything, log every candidate, gate hard, score lightly,
compose a set, and let the user choose.

Three beliefs sit under it:

1. **The hook is mostly a selection problem.** No caption style rescues a moment that needs five
   minutes of context. Every serious source on short-form converges on the same core: a
   self-contained first sentence, an arresting claim, no throat-clearing, one idea per clip.
2. **Recall errors are the expensive ones.** A moment you never logged cannot be chosen. Misses
   come from skimming, from a taxonomy with no word for what you are looking at, and from
   proximity judgments made by eye.
3. **No score predicts virality.** No commercial or open "virality score" has independent
   evidence that it predicts performance. Gates and reasons are checkable; a number posing as a
   forecast is not. Scores here only sort the shortlist.

Contents: 1 inputs, 2 before you read, 3 recall pass, 4 the ledger, 5 taxonomy, 6 hard gates,
7 scores, 8 set pass, 9 the shortlist (gate A), 10 signals, 11 PASS reasons, 12 worked
examples, 13 how hunters fail.

## 1. Inputs

- `P/transcript.md`: one line per sentence-like unit, `[mm:ss.s] #first-last Speaker: text`.
  The `#first-last` numbers are inclusive word indices; they go straight into clips.json as
  `"words": [first, last]`. Read this file, not the JSON. A line starting with "…" continues the
  previous one.
- Coverage holes in `transcript.md`: transcribe re-listens to every stretch of voiced audio that
  got no words and splices in what it finds (flag `respliced` in transcript.json; a prompt echo
  is the whole glossary, or one of its names, written with a near-zero-confidence word, and is
  dropped; a glossary name spoken with confidence is always kept, so check spliced names
  against the audio anyway). What it still
  could not fill is listed in place as `[mm:ss.s] <<4.2 s of sound with no words ...>>` (flag
  `unworded_audio`: laughter, music, or speech ASR missed) or `<<the speakers VTT has N words
  here that ASR did not transcribe>>` (flag `dropout`). Inline `<<...>>` notes mark a repetition
  loop, words with no audible sound (`no_audio`: likely invented, omit them), an uncertain
  speaker label (`speaker_uncertain`), a possible false start folded into one word
  (`false_start`: two attempts at a phrase stored as one run) and a word whose start sits under
  a laugh or another voice (`timing_uncertain`). Inspect a hole with `hc words` on both sides and
  `hc peek` before you log or keep a moment that spans it: a laugh is fine, missing speech is
  not (plan lints L15 and L19 flag holes inside a clip).
- Text the backend writes in parentheses (up to 4 words) or brackets (up to 8) is set aside as a
  sound event and never captioned. Transcribe prints `set aside N sound events ...: (laughs) x2,
  [Music]` and lists them in transcript.json `events` with kind `sound`: read that line, a
  spoken aside written in parentheses would be on it.
- Speaker labels come from `--speakers-vtt`: Zoom-style "Name: text" cues, WebVTT `<v Name>`
  voice tags and dash-prefixed lines are understood. Transcribe prints the share of words that
  got a label and the number of uncertain turn boundaries. A re-run reuses the recorded VTT
  only while the source is the same file (or a `--cfr` re-encode of it); after a new ingest
  pass `--speakers-vtt PATH` again, or `--speakers-vtt none`.
- `hc find P "phrase"`: a remembered phrase to `#i-j [t0-t1] context`. Use `--near MM:SS` when
  the phrase recurs.
- `hc words P A B`: every word in A..B with its index, source time, the gap before it, a filler
  flag and the speaker. Use it to find the exact first word of a sentence, to measure pauses,
  and to measure the distance from a candidate's edge to anything risky.
- `hc peek P START END`: a contact sheet of the span. Delivery, who is on camera, reactions.
- `P/signals.json` (URL sources only): the "most replayed" heatmap and chapters (section 10).
- `P/sheets/survey.jpg`: who is on camera across the whole runtime and whether the layout changes.
- From the user: who the clips serve, what is already posted, anything to avoid, and a target
  count (a ceiling, never a quota).

An hour of speech is roughly 9,000 words. Read `transcript.md` in slices of about ten minutes
(the Read tool's offset and limit), never all at once and never by sampling.

## 2. Before you read

Write the header of `moments.md` before the first window:

- **Served:** whose clips these are. If the user hosts the show, their own first-party argument
  matters as much as the guest's best line. If the user is producing for a guest, weigh the
  guest.
- **Already posted:** titles or hook lines already published from this source, or recently on
  this account. Anything that repeats one is a dupe.
- **Talk time:** per speaker, when the transcript has speaker labels (sum word durations per
  speaker). A set that ignores a speaker with 40% of the airtime needs a reason.
- **Avoid:** topics, names or claims the user flagged.

## 3. Recall pass (coverage, not judgment)

The goal is that every plausible moment becomes a row. Judgment comes after.

1. Split the transcript into 10-minute windows that overlap by one minute: 0:00-10:00,
   9:00-19:00, 18:00-28:00, and so on. The overlap catches moments that straddle a boundary.
2. Read every line of the window. Do not skim the slow middle of a long conversation: the best
   line in an hour often arrives in the second half, after people relax.
3. For each line that could open a clip on its own, add a CANDIDATE row: indices, time, speaker,
   the hook line verbatim, type tags. Be generous. A row costs one line; a missed moment costs
   the clip.
4. After each window, append one line to the Coverage section:
   `18:00-28:00 read, 5 candidates (M11-M15)`. Zero candidates is possible but rare; re-read the
   window once before logging zero.
5. When every window is logged, run the recall audit: every chapter, heatmap peak, in-source
   recap item and laughter beat (section 10) must map to a row (any status) or get a new one.
   Log the result.

Expect roughly 15 to 60 rows per hour, depending on density. Only then gate, score and compose.

## 4. The ledger: moments.md

`P/moments.md` is the only place a candidate's status lives. clips.json points back to a row
with `"ledger": "M07"`; render state is derived from files (`hc status P`). Two copies of a
status means one of them is wrong.

### Format

```markdown
# Moments: interview-14.mp4 (58:07, 2 speakers)
Served: Speaker A (the host) · Already posted: none · Avoid: the pricing tangent
Talk time: Speaker A 41% · Speaker B 59%

## Set finding
What is genuinely new in this source, what repeats something already posted, who carries the
strongest material, and what the set needs (an exchange piece, the host's own voice).

## Ledger
| id | #a-b | time | speaker | hook line (verbatim) | types | gates | scores | status | reason |
|---|---|---|---|---|---|---|---|---|---|
| M03 | #812-845 | 06:41-06:58 | Speaker B | "I stopped billing by the hour, and my income doubled." | contrarian, number | ok | 5 5 4 5 4 4 = 27 | SHORTLIST | names the change and the result in one breath |
| M04 | #1022-1040 | 08:12-08:19 | Speaker A | "So the question everyone asks me is..." | question | cold-read | | PASS | cold-read: host question; the answer is M05 |
| M05 | #1041-1160 | 08:19-09:22 | Speaker B | "Nobody buys the software. They buy a better Tuesday." | contrarian, tactic | ok | 4 3 4 4 5 3 = 23 | BANKED | same argument as M03; M03 is more concrete |

## Notes
M03: hook 5 the change lands in clause one; concrete 5 a real before and after; payoff 4 the
why arrives at 06:55; standalone 5; stake 4 any freelancer; delivery 4 laughs into the number.

## Risks
| id | #a-b | kind | what | decision |
|---|---|---|---|---|

## Coverage
00:00-10:00 read, 5 candidates (M01-M05)
09:00-19:00 read, 4 candidates (M06-M09)

## Recall audit
chapters 6/6 mapped · heatmap peaks 3/3 mapped (M12, M17, new M31) · in-source recap: none
```

### Columns

| Column | Content |
|---|---|
| `id` | `M` plus two or three digits. Stable forever: never renumber, never reuse. |
| `#a-b` | Inclusive word indices of the whole candidate span (from transcript.md, refined with `hc words`). |
| `time` | `mm:ss-mm:ss` (or `h:mm:ss`) on the source clock. |
| `speaker` | Whose words the hook line is. `unknown` if you cannot tell (then the attribution gate is open). |
| `hook line` | Verbatim from the transcript, in quotes. Correct an obvious ASR mishearing in [brackets]; never smooth the speech. |
| `types` | Tags from section 5, comma separated. Add `exchange` for multi-voice pieces. |
| `gates` | `ok`, or the names of the gates that failed (section 6). |
| `scores` | Only for rows that pass every gate: six 1-5 numbers in the fixed order hook, concrete, payoff, standalone, stake, delivery, then `= sum`. The whys go in Notes. |
| `status` | One of the six statuses below. |
| `reason` | One line. Required for SHORTLIST, BANKED and PASS. PASS reasons start with a code from section 11. |

### Status semantics

| Status | Means | Next |
|---|---|---|
| CANDIDATE | logged in the recall pass, not yet gated | SHORTLIST, BANKED or PASS |
| SHORTLIST | passed every gate, scored, survived the set pass, shown at gate A | PLANNED, BANKED, or PASS `user-cut` |
| PLANNED | the user approved it and a clips.json entry with `"ledger": "<id>"` exists | CUT, or back to BANKED or PASS if it fails gate B or does not survive editing |
| CUT | a render exists in `P/renders/` for that clip (confirm with `hc status P`) | stays CUT; note exports and approvals in the reason |
| BANKED | good, kept for later: a set conflict, the user deferred it, or a better source of the same argument exists | SHORTLIST in a later session |
| PASS | rejected with a named reason, so nobody re-mines it | SHORTLIST only if the reason was wrong (log why) |

Rules:

- A row moves to CUT only when its render exists. "Rendered" is a fact about disk, not a plan.
- PASS without a reason is not allowed. The reason is the most valuable column: it stops the
  next session from re-mining the same moment and it records a judgment the user can overrule.
- BANKED rows carry over. The next session on the same source starts from BANKED, not from a
  fresh read.
- Never delete rows. Change the status and append to the reason.

## 5. Taxonomy

| Type | What it is | Tell |
|---|---|---|
| contrarian | contradicts what the viewer or the category assumes | "Everyone says X. X is backwards, because..." |
| number | a real, specific figure carries the line: money, time, a count | the line goes flat if you delete the number |
| story | a scene with a turn: a person, a place, something that went differently | "We had nine days of cash left when the call came." |
| confession | a first-person admission that names a fear, failure or mistake | "I was terrified someone would notice I was guessing." |
| warning | a genuine stake for the viewer if they keep doing something | "If every lead comes from one channel, you do not own a business." |
| question | a question the viewer wants answered, answered inside the clip | never the interviewer's question as the hook |
| tactic | a concrete do-this-not-that, best with a reversal | "Don't send the proposal. Send the calendar invite first." |
| moving | lands emotionally rather than as an argument; often quiet | easy to miss, because it does not look like a hook |
| funny | a laugh that survives without the room | self-deprecation travels; inside jokes do not |

**Exchange is a piece type, not a hook type:** a question and its answer, a pushback, or a
reaction that is itself the payoff. Tag `exchange` next to the hook type. Compound tags are fine
(`warning, number`). Tags exist for set spread (section 8) and for any performance tracking the
user does later.

## 6. Hard gates

A failed gate makes the row PASS with the gate named in the reason. Many cold-read failures are
fixed by moving the in-point; a row fails only when no in-point inside the passage passes.

1. **cold-read.** Read the planned first sentence as a stranger who just swiped in with no audio
   before it. It fails when:
   - it leans on a referent the clip never gives ("this only works if...", "and that's where
     the second product came in", "they never called back");
   - it negates a setup the viewer never heard ("It's not that I didn't have a plan", answering
     a question that was cut);
   - it is the interviewer's question (the answer is the hook; the question may follow as setup);
   - it opens on a discourse marker or throat-clearing with no clean sentence after it ("and",
     "so", "but", "because", "yeah so what I'm getting at", "like I said");
   - it starts after the subject ("...rebuilt the whole onboarding in a weekend", when the
     subject was "she").
2. **one-idea.** One idea, and its payoff sits inside the span. A great setup whose answer comes
   twenty minutes later fails unless both halves are short and joining them is honest
   (example 10).
3. **viewer.** It is about the viewer's world, not the speaker's offer, pricing, product or
   credentials. A founder's launch story passes when the lesson is the viewer's; "here is what
   my programme includes" never does.
4. **cliche.** It is not a line the audience has heard a hundred times in this genre (the "ship
   before you feel ready" sermon, "work smarter, not harder"). A cliche with a specific twist is
   a different line: judge the twist.
5. **risk.** Nothing risky sits inside the span or within a few seconds of its edges,
   **measured, not eyeballed.** Run `hc words` around both edges and write the distance in words
   and seconds in the reason ("error at #2290, 31 words / 9.4 s after the out-edge"). Risks:
   wrong facts, inflated or unverifiable numbers, claims about named third parties, a live deal,
   private information, a swipe at a group, profanity on platforms that penalise it. Log each in
   the Risks table. See `fidelity.md`.
6. **material.** Enough to make a clip: at least the vertical length-band minimum (15 s) after
   tightening, and carried by audio and captions (a screen demo whose screen is not in the frame
   fails as `needs-visuals`).
7. **attribution.** You are certain who said it. Provisional here, because transcript labels can
   be wrong; confirmed at gate B by looking at the picture.

## 7. Scores

Score only rows that pass every gate. Each score is 1 to 5 with a few words of why in Notes.

| Score | 1 | 3 | 5 |
|---|---|---|---|
| **hook**: does the first line arrest? | a topic | a claim | a claim you have to hear the rest of |
| **concrete**: person, loss, scene, number | abstraction | one concrete element | a named stake: who, what, how much |
| **payoff**: does the span deliver what the first line opens? | no landing | lands softly | lands hard, inside the span |
| **standalone**: needs nothing outside the span | needs a primer | a small gap the hook card can bridge | complete cold |
| **stake**: does the viewer feel it is about them? | the speaker's business | general interest | money, identity, status or human stakes for the viewer |
| **delivery**: conviction, timing, energy (transcript, signals, peek) | flat or halting | fine | conviction, a laugh, a pause that lands |

- The sum is a sort key and nothing else. When presenting, never call it a virality, reach or
  performance score. Say "sorted by how well each stands alone and lands".
- Ties: prefer concrete, then standalone.
- Human stakes beat mechanism. The same argument told as what happened to a person usually
  travels further than the version told as how the system works. Prefer the person.

## 8. Set pass

Run it over the gated, scored rows before gate A. A set is not the top N by sum.

- **Overlap.** Two picks share at most 15 s of source (plan lint L14 checks this later). If they
  share a passage, keep the stronger, bank the other, or carve them apart.
- **No repeated argument.** Two clips making the same point in different words is one clip. Keep
  the more concrete one and bank the other.
- **Register spread.** Vary the types and registers: a warning, a confession, a story, a number.
  Five contrarian claims in a row read as one note.
- **Exchange pieces.** When the source has more than one voice, consider an exchange piece. It is
  not a quota: include one when the back-and-forth is the payoff.
- **The person being served.** Weigh toward whoever the clips are for. A host publishing their
  own show needs some of their own argument; a guest's best line is still the guest's.
- **Already posted.** Skip anything that repeats a published clip (PASS `dupe-posted`).
- **Self-promotion.** A set dominated by one speaker's success stories and offers reads as an ad.
  Ranking by hook strength alone produces exactly this; the set pass exists to stop it.
- **A moving pick.** If a quiet, human moment passed the gates, give it a slot even when its
  hook score is middling. Quotable and moving are different things.
- **Count.** Cadence is not a quota. Four strong clips beat twelve padded with repeats. When the
  source is thin, say so.

## 9. The shortlist (gate A)

Present a table, then stop and wait for the user.

| # | id | speaker | hook line | at | est. length | types | why |
|---|---|---|---|---|---|---|---|
| 1 | M03 | Speaker B | "I stopped billing by the hour, and my income doubled." | 06:41 | 35-45 s | contrarian, number | the change and the result in one breath |
| 2 | M17 | Speaker A | "I hid the numbers from my own team for a year." | 21:10 | 40-50 s | confession, story | names the fear, then what it cost |

Then, in a few lines:

- the set: "6 picks: 2 confession, 2 tactic, 1 story, 1 exchange. Your own voice in 2."
- banked: ids with one-line reasons;
- passed: the count and the top reasons ("31 passed: 9 cold-read, 6 guest-offer, 5 dupe-set");
- the ask: "Approve, cut or add? Point me at anything you remember that I missed."

Rules: hook lines are verbatim; estimated length comes from the planned span after tightening
(roughly 85 to 90 percent of the raw span); about eight picks per hour of source at most; never
show the sums as predictions. When the user answers: approved rows become PLANNED once their
clips.json entry exists; cut rows become PASS `user-cut`; deferred rows become BANKED; a moment
the user adds gets a new row and still faces every gate (say so if one fails).

## 10. Signals that surface, never rank

Signals tell you where to look twice. They never decide.

- **"Most replayed" heatmap** (`signals.json` `heatmap`, value 0-1 on the source clock, already
  shifted for `--section`). Peaks are where viewers rewound. Read every peak window again in the
  recall audit. A peak can also mean confusion, an on-screen visual, or people seeking past an
  ad read: it is a pointer, not evidence of a hook.
- **Chapters** (`signals.json` `chapters`). The creator's own map. Each chapter should hold at
  least one logged row, or a coverage note saying why not. Chapter titles hint at what the
  creator thinks matters; they are not hooks.
- **Laughter and reactions.** A pause followed by loud non-speech, a cluster of "ha", someone
  saying "wait, say that again". Check with `hc peek`. The line before a laugh is a candidate;
  so is the laugh itself, as an exchange.
- **In-source recap.** Hosts often summarise the best points near the end. Use the recap as a
  checklist: every recapped point maps to a row.
- **Energy and pace.** Shifts in loudness or speed the transcript cannot show. They inform the
  delivery score and nothing else.

## 11. PASS reasons

Start every PASS reason with one of these codes, then a few words.

| Code | Meaning |
|---|---|
| `cold-read` | no clean first sentence: orphan referent, negation of an unheard setup, host question, preamble |
| `needs-context` | needs minutes of setup the clip cannot carry |
| `payoff-elsewhere` | the payoff lives outside the span and cannot be joined honestly |
| `no-payoff` | opens a loop and never closes it |
| `two-ideas` | two half-ideas competing for one clip |
| `guest-offer` | about the speaker's offer, pricing, product or credentials, not the viewer |
| `cliche` | a genre cliche with no twist |
| `flat` | true but unquotable: no person, loss, scene or number |
| `thin` | a good line with nothing under it: a quote card, not a clip |
| `dates-fast` | tied to a version number, a news cycle or a vendor feature |
| `needs-visuals` | depends on something the camera does not show |
| `off-audience` | good, but for a different audience than the user's |
| `dupe-set` | repeats an argument already in this set |
| `dupe-posted` | repeats something already published |
| `overlap` | shares more than 15 s of source with a stronger pick |
| `risk:<kind>` | `fact`, `number`, `third-party`, `deal`, `private`, `reputation` or `profanity`, with the measured distance |
| `attribution` | the speaker is uncertain or not on camera (gate B) |
| `too-short` | under the length band after tightening, with nothing honest to add |
| `audio` | crosstalk or garbled words that cannot be cut around |
| `rights` | footage the user cannot clip, or another organisation's branded material |
| `user-cut` | the user cut it at gate A or gate E |

## 12. Worked examples

Composites: the speakers are generic and the lines are invented to show the shape of real
decisions. Each reads before, change, after, lesson.

1. **The hook was buried 18 seconds in.** A consultant spends 18 s on how one client grew to
   half her revenue, then says "So I fired my biggest client." Before: chronological, 33 s; the
   claim arrives after most viewers have swiped. After: piece 1 is the claim, starting on "I"
   rather than "So" (cold open), piece 2 the why, piece 3 her closing line about the month that
   followed; 25 s. Lesson: when the
   strongest line lives mid-passage, open on it and rewind.

2. **Two seconds of throat-clearing.** The in-point sat on "So, yeah, here's the thing." with
   the real sentence after it. After: `hc words` found the first word of "Your first ten customers
   will tell you the price", and the piece starts there, six words later. Lesson: "so, yeah, what I'm
   getting at is" never survives. Check every clip's first word (plan lint L04 warns).

3. **A negation of a setup nobody heard.** The interviewer asked whether the founder had felt
   confident; the cut opened on "It's not that I thought I knew everything." The viewer hears a
   denial of nothing. After: the in-point moves to the next sentence, the affirmative version:
   "I figured I'd have it worked out before anyone asked." Lesson: never open on "it's
   not that..." unless the thing being denied is inside the clip.

4. **An orphan referent.** "This only works if you write it down first." What is "this"?
   After: the piece starts one sentence earlier, where the tool is named, which costs 4 s and
   passes the cold-read. Lesson: read the first sentence as a stranger, every time.

5. **The subject was cut, and the credit flipped.** The cut began on "...rebuilt the whole
   onboarding in a weekend." With the subject gone, the captions read as the host's own claim
   about something a guest's colleague did. After: the in-point moves back 2 s so "She" is in the audio and the captions.
   Lesson: the subject must be in the cut. Misattribution is the worst thing this skill can ship.

6. **A reorder that made it worse.** A clip opened chronologically on a clean claim. Someone
   "improved" it by opening on a mid-sentence line ("...which, for us, ended up being the part
   that"). It read worse. After: reverted to chronological. Lesson: reorder only when the new opener
   passes the cold-read better than the old one. A rule applied without checking becomes a tic.

7. **A moving answer marked PASS.** The host asks a reflective question; the guest's answer
   wanders, then lands on a quiet line about why they still do the work. First pass: PASS,
   "answer wanders". Second read: the wander is the answer, and it was the most human moment in
   the hour. After: an exchange piece: the question, the answer, and the host's short callback
   from three minutes later as the close (a `time_src` reaction piece). Lesson: a payoff can be
   tonal rather than propositional, and a taxonomy with no "moving" type cannot find it.

8. **A proximity call made by eye.** A strong line was passed as "too close to a factual error".
   Measured with `hc words`, the planned out-edge sat 31 words (about 9 s) before the error.
   After: restored, with the distance written in the reason and the error logged in Risks.
   Lesson: measure risk adjacency in words and seconds.

9. **The hookiest passage was the wrong pick.** A rich, quotable passage about a guest's
   philosophy needed several minutes of setup and was really about the guest's product. Any
   "hookiness" ranker would have promoted it. After: PASS `needs-context`, `guest-offer`.
   Lesson: gates before scores. Standalone and viewer come first.

10. **The payoff lived twenty minutes later.** A founder describes a bet at 12:10 and reveals how
    it turned out at 31:40. The 12:10 span alone had no payoff. After: a two-piece clip: the
    reveal as an 8 s cold open, then the 20 s setup, joined with a dip. Same speaker, same topic,
    so the join is honest. Lesson: joining across minutes is fine when meaning survives; when the
    later line answers a different question, PASS `payoff-elsewhere`.

11. **A withheld payoff in a cold open.** A teaser beat ended on "...and then I opened the
    letter," with the result held back for the full video. That works in an email, where the click
    is the goal; in a short, the beat has to land on its own, and this one left a person opening
    an envelope. After: replaced with a beat that pays off inside itself. Lesson: never end a beat
    on a cliffhanger the clip does not resolve.

12. **The interviewer's question as the hook.** "What would you tell your younger self?" opened
    the cut. After: the clip opens on the answer's strongest sentence; the question follows as a
    short setup piece. Lesson: the question is context; the answer is the hook.

13. **A set that was all monologue.** Eighteen picks, every one a single speaker; the co-host
    appeared in none, and the highlights opener was built from beats already used elsewhere.
    After: three exchange pieces where the reaction was the payoff, and an opener rebuilt from
    unused beats. Lesson: the set pass is its own step. Per-clip ranking cannot see the set.

14. **Ranking alone produced an ad.** Sorting by hook strength surfaced twenty success stories
    and offers from one guest. After: the dupes passed, the host's own argument got a slot, a
    quieter moving pick beat a louder quotable one, and six pieces shipped instead of twenty.
    Lesson: cadence is not a quota. Padding means repeating an argument or promoting someone
    twenty times.

## 13. How hunters fail

| Failure | Guard |
|---|---|
| skimming the second half of a long source | 10-minute windows and the Coverage log |
| no word for what you are looking at | the taxonomy includes moving, funny and exchange |
| a proximity judgment made by eye | measure with `hc words`; distances in the reason |
| promoting the "hookiest" passage that needs scaffolding | gates before scores |
| opening mid-clause because a playbook said "start mid-sentence" | open on the strongest *sentence*; a clean sentence start beats a fragment (L04, G6) |
| treating CUT as a plan | CUT means a render exists |
| re-mining last session's rejects | PASS reasons and BANKED rows carry over |
| presenting sums as predictions | never; they are sort keys |
| padding to a count | cadence is not a quota |
