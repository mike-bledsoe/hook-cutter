# Packaging: hook card, title, cover, post copy

A clip meets people through three written surfaces before anyone hears a word. Each has a
different reader and a different job.

## The three-surface rule

| Surface | Who reads it | Job | Where it lives |
|---|---|---|---|
| hook card | the muted autoplay viewer, with the voice playing underneath | the provocation that stops the scroll | `hook_card.lines`, burned in from frame 0 |
| cover | someone browsing a profile grid: silent, cold, not yet pressing play | the claim they will believe | `cover.headline`, `covers/<clip>__<style>.jpg` |
| platform title | search results and the sidebar | the promise of what they get | `post.title`, copied into `export/manifest.json`; the clip `title` (file name, manifest title) is claim-checked as well |

**Agree in meaning, never identical in wording.** If two surfaces are interchangeable, one of
them is not doing its job. QA G13 warns when the hook card, cover headline and title of one clip
overlap too much (an overlap coefficient of 0.8 or more). The post caption must not restate the
burned card either.

## Hook card copy

Mechanics are in `captions.md`. The copy:

- restates the claim, ideally in the speaker's own words;
- up to about nine words, 1-3 lines of about 10-12 characters (longer lines set small or get
  re-broken), one `*accent*` run on the word that carries the claim (the captions accent the same
  word automatically when it is spoken after the card);
- is written to be heard: the voice underneath supplies context, so it can be terser than a cover;
- uses only numbers and names said in the clip (G11, L13);
- sets bigger than the captions: at least 1.3 times their size (L20 says how long a line can be).

`["I FIRED MY", "*BIGGEST* CLIENT"]` works. `["THIS CHANGED", "*EVERYTHING*"]` is a topic
pretending to be a claim.

## Cover copy must parse cold

A hook card plays for two or three seconds with the speaker's voice supplying the missing
context. A cover is read in silence, in a grid, by someone who has not pressed play.

The test: read the headline with no audio, no title and no context, at grid size. Does it say a
complete, true thing?

- Default to the hook card text, then run the test. Rewrite when it fails.
- Prefer the concrete instruction or metric the speaker actually said over a clever compression.
- Watch the opposite failure: a punchier rewrite that claims more than the speaker did.
- Shape: about 14 rendered characters per line, 2-4 lines, one accent, real numbers only.

Two composites:

- Card `["3 WEEKS VS", "*1 DAY*"]` works with the voice. Cold, it never says three weeks of
  what. Cover: `WHAT TOOK OUR TEAM 3 WEEKS NOW TAKES *1 DAY*`.
- Card `["BUSY ISN'T", "*BOOKED*"]` reads as wordplay at a glance. The speaker's actual
  instruction was to fill next month's calendar before this month ends. Cover:
  `BOOK NEXT MONTH *BEFORE* THIS ONE ENDS`.

## The grid crop

Profile grids show a 3:4 crop of a 9:16 cover: 240 px lost at the top and the bottom, keeping
y 240-1680. Keep the headline and the face inside that band (the presets anchor the headline
there). Check:

- `covers/<clip>__<style>.grid.jpg`: exactly what the grid shows;
- `covers/<clip>__<style>.mobile.jpg`: the grid crop at about 240 px wide. If it does not read
  there, it does not ship;
- `sheets/covers-grid.jpg`: the whole set as a phone-scale grid. Vary framing scale and
  registers so the grid does not look like one tile repeated.

## Frame curation

`hc cover` scores candidate frames every half second (face size, centring, frontal pose,
sharpness) and writes `sheets/<clip>.cover-candidates.jpg`. The score finds usable frames; you
choose the expressive one. Frames inside a source cross dissolve (two pictures laid over each
other) and frames with two superimposed faces are set aside: they are left out of the automatic
pick and the sheet, and when you force one with `cover.at_out` the cover json warns.

- Expressive and mid-emphasis: mouth closed or shaped on a vowel, never a mid-word smear. Eyes
  open: the detector cannot see eyelids, so look.
- The right speaker: the person who owns the claim on the cover. A host's face over a guest's
  claim misattributes it. When the transcript has speaker labels, `hc cover` only scores frames
  where the clip's `speaker` is talking, and QA G12 warns when the chosen frame belongs to
  someone else.
- Head-and-shoulders scale beats a wide shot.
- Record the choice as `cover.at_out` (output seconds) or run `hc cover P CLIP --at SECONDS`
  (`--at` also takes `0:12.4`).
- A cover made before the clip is rendered, or after a clips.json edit that is not rendered
  yet, is cut from the source with a framing estimate and says so (`frame_source: source` in the
  cover json, and a warning G12 repeats): render first for the exact frame.
- `cover.shift` slides the photo up so the face clears the headline (0 or more; a negative
  value is L01 FAIL: it cannot slide the photo down, so a face that sits too high needs another
  frame, `cover.at_out`). The cover json warns when they still overlap. A `shift` you write
  (even `0`) is kept as you set it.
- The photo darkens toward the headline: the tint ramps up from about 300 px above the headline
  and is raised behind each line until the text reaches 4.5:1 contrast. Under 3:1 on a line,
  the cover json warns: pick a darker frame.
- G12 checks the grid band itself (y 240-1680): the headline ink and the top of the head must
  sit inside it, and a cover with no detected face is a WARN, because face overlap could not be
  checked. A close-up whose head starts above the band even with no shift (a webcam close-up
  cropped to 9:16) cannot fit at any shift: the cover keeps the forehead inside, shifts at most
  80 px (the headline shrinks first) and G12 WARNs that the tile trims the top of the head. It
  FAILs when the face itself leaves the band, when the headline shift pushed a head that fitted
  out of it, or when a close-up still needed more than 80 px: shorten the headline or pick
  another frame. The top of the head is measured in the picture where it stands out from the
  background (the cover json says `crown_method: "image"`); only a measured head is ever
  reported as trimmed. Where it cannot be measured (`"estimate"`: hair the colour of the wall, a
  head that fills the frame) nothing is reported, so look at the `.grid.jpg` yourself.
- Never generate, relight or retouch a face with AI for a cover (`fidelity.md`).

When no cover is uploaded, platforms fall back to frame 0, which is why the hook card is fully
visible on frame 0. That fallback is a safety net, not a cover.

## Platform title

- At most 70 characters, with the hook in the first 30.
- A claim or a promise, not a topic. No episode numbers.
- No clickbait the clip cannot cash. A curiosity gap is fine only when the clip closes it early.
- For long clips the title's payoff arrives in the first 20 seconds (`editing.md` rule 12). A
  cover may pay off later in the clip, because a cover makes a claim rather than promising what
  comes next.
- Numbers and names must be said in the clip (G11). The clip's own speakers may be named in the
  title and the post caption without saying their names in the clip.

## Registers across a set

One register per piece, different registers across a set: a warning with a number, a diagnostic
challenge, a story scene, a confession that names a fear. The covers grid makes repetition
obvious.

## Post copy

`post.title` and `post.caption` in clips.json are optional and go into `export/manifest.json`
for the user. The skill never posts them.

- The first line does the work before the platform's "more" cut (about 125 characters on
  Instagram): a claim, not narration, with no wind-up.
- Add a beat the video does not have (the context, the stake, the number). Do not transcribe the
  clip or restate the burned card.
- Quotes are verbatim: copy them from the transcript and mark any cut with an ellipsis. Never
  tidy a quote.
- Handles you have not verified are written `[handle]`, never guessed.
- Do not spend the caption on a generic call to action ("follow for more", "link in bio"). Generic
  CTAs often cost reach. If the user has a real one, use theirs.
- Hashtags: a few relevant ones or none.
- Name people as they spell their own names. G11 checks the caption too: every number in it
  must be said in the clip.

## Landscape chapters

Chapters come from pieces with a `chapter` key; their times come from the plan after tightening
(export writes them into the manifest). Chapter titles are claims, not topics: L13 and G11 check
each one against the clip's words. The first chapter starts at 0:00.

## Variants

A hook variant is a new clip id with the same pieces and a different `hook_card` (and usually a
different cover). It overlaps its original, so L14 and G13 will warn; that is expected for a
variant. Note it in the ledger and space the two apart when posting.
