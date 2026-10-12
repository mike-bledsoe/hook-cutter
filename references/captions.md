# Captions

Captions do two jobs: carry the words for the many people who watch muted, and point at the one
word that carries the claim. Everything else competes with the face for attention.

All text (captions, hook card, cover headline) is drawn with Pillow from bundled font files and
piped into the final encode. No libass, drawtext or system fonts are involved, so what you see on
one machine is what you get on another.

## Reveal modes

| Mode | What the viewer sees | Use for |
|---|---|---|
| `fill` (`clean`, the default) | the whole page is visible; unspoken words are dimmed (70% opacity, `unspoken_alpha`); each word turns full white as it is spoken | talking heads. The viewer can read ahead, and the motion comes from the fill rather than from a colour jumping word to word |
| `build` (`pop`) | words appear as they are spoken with a short fade and rise; emphasis can pop. A backing pill grows word by word: it covers each word from the first frame that word is drawn and never reaches ahead of the speech | high-energy delivery and short pages. More motion competes with the face: use it on purpose |
| `static` (`minimal`) | the page appears whole, with no per-word motion | calm or reflective clips, landscape, and any time motion would distract |

Evidence check: no peer-reviewed study shows word-by-word captions beating phrase captions; the
lift figures in circulation are vendor claims. Eye-tracking work on kinetic type does show moving
text pulls the eye more (more saccades, shorter fixations). So keep each page on screen long
enough to read and animate only what helps reading.

## One accent that carries the claim

- The accent colour marks meaning, not the playhead. It goes on the word or short phrase that
  carries the claim, never on whichever word is being spoken.
- One per page (`emphasis.per_page: 1`). Two accents on a short page colour half the line and the
  emphasis stops meaning anything.
- It is sourced from the hook card. Each `*accent*` run in `hook_card.lines` is matched in the
  captions as a phrase (case, apostrophes, hyphens and number forms normalised; stopwords
  ignored), so the hook card, the captions and the cover accent the same idea by construction.
- `emphasis` (word indices) adds a deliberate accent when the card does not cover it.
- `auto` emphasis is opt-in per preset (`pop` opts in) and deliberately rare: only numbers,
  money and names that are capitalised mid-sentence, at least 4 s apart, never a word already
  accented and never on or next to a page that carries a hook accent. An accent on every page
  is the opposite of emphasis.
- `accent_reveal: "on_spoken"` (default) colours the accent as it is spoken; `"page_start"`
  colours it as soon as its page appears.
- QA G14 warns when an accent phrase never matched: the card accents a word the speaker never
  says. Rewrite the accent to the speaker's own word.

## Legibility

- **No outline on dark footage.** A heavy black stroke on a dark background is invisible, blurs
  the letterforms and reads as a phone-app default. The presets use a two-layer soft shadow.
- **Backing only when needed.** `backing.mode: "auto"` measures each page's own box on the
  actual frames: the brightest tenth of the pixels (p90 luma) and how busy the background is.
  A page needs a pill when the contrast of the text against that p90, counting the darkening
  of the preset's shadow, is below `min_contrast` (3.0:1; white text on a plain background at
  p90 of about 0.80), or when more than `max_edges` (15%) of its box is edges (foliage, text,
  a patterned shirt). The decision is held per shot (one framing: a stretch between framing
  changes and speaker turns; a jump cut inside one framing keeps the same background): the
  shot gets the pill when a quarter of its pages need it or any page is far off (contrast
  under 2.5:1 or edges over 30%), so it never blinks on and off and one borderline page does
  not box a whole shot. The hook card counts as a page of the first shot. Use `"always"` for busy footage or
  a boxed look, `"never"` only when you accept the risk. A punch-in is a new framing, so on
  borderline footage `auto` can give one shot pills and the next none; when that reads as a
  flicker, set the clip key `"caption_backing": "always"` (or `"never"`): it overrides the
  preset's mode for every page of that clip, and the hook card follows it.
- **Outline** (`outline.px` above 0) is available for bright, busy footage where a pill feels
  heavy.
- Each page's measured p90, edge density and contrast are written to `.captions.json`.
- `.captions.json` also lists `issues` the render could not avoid, which QA G9 reports:
  `min_page` (a page on screen less than `min_page`), `word_shrunk` (a single word too wide for
  the line, set below 90% size), `overflow`, `hook_small` (the card under `min_ratio` times the
  caption size), `caption_y_clamped`, `width_clamped` (`layout.max_width` wider than the safe
  zone allows), `glyph_stripped` (invisible characters removed) and `accent_unshown`.

## Safe zones

Platform interface covers parts of the 1080x1920 frame: a top bar, the caption and handle block
at the bottom, and the action rail on the right. No platform publishes an official safe zone for
organic posts; every figure here is an estimate or an ad specification.

| Profile | Top | Bottom | Left | Right | Basis |
|---|---|---|---|---|---|
| `organic` (default) | 240 | 420 | 90 | 150 | conservative composite of published organic estimates for Reels, Shorts and TikTok |
| `ads` | 270 | 672 | 65 | 242 | Meta's published ad margins (14% top, 35% bottom, 6% sides) widened to TikTok's in-feed ad overlay on the right |
| `landscape` | 54 | 54 | 96 | 96 | title-safe style margins for 1920x1080 |

Published estimates vary a lot: Reels organic top 210-270 and bottom 310-420 px; Shorts bottom
300-381 and right 96-201 px; TikTok organic bottom roughly 320-480 and right 120-140 px. Set
`"safezone": "ads"` in the clip defaults when the clips will run as ads, or in a preset of your
own; clips.json wins when both name one. Values live in `presets/safezones.json`: each profile
is an object with top, bottom, left and right.

QA G9 checks every caption page and the hook card against the profile the render used (outside: WARN, off the
frame: FAIL) and warns when captions sit over a detected mouth (from the nose tip to just
below the lower lip; a caption over the chin or beard is fine). Fix with
`caption_y` on the clip or `layout.y` in a preset.

`caption_y` is vertical only. Pages are kept between the profile's top and bottom margins: a
`caption_y` that would push them out is clamped, with a `caption_y_clamped` issue. Landscape
ignores `caption_y` (with an issue) and uses the preset's landscape `layout.y`.

## The hook card

- A 1-3 line headline restating the claim, drawn in the caption band at the start of the clip.
- **Full opacity on frame 0.** Platforms use frame 0 as the default thumbnail when no cover is
  set, and this skill never sets one on upload. A card that fades in leaves frame 0 empty, so the
  card is fully visible from the first frame and only fades out (`fade_out_ms`). The export
  manifest records `thumb_offset_ms: 0` for this reason.
- Duration `"auto"` = 0.5 s plus characters divided by 20 (about adult reading speed), clamped to
  1.5-3.0 s. Words spoken while the card is up are heard but never captioned; the first caption
  page after the card starts on a fresh phrase, never on the tail of a sentence the card covered.
- That makes card length a trade. A long card (a number over the auto duration) hides more of
  the opening line from muted viewers. Either keep the card short (its auto duration) and accept
  that the card stands in for the words under it, or do not put a line that needs captions
  under the card at all (`editing.md` rule 13). The `.srt` sidecar still carries those words.
- `captions: false` turns the captions off and keeps the hook card.
- It sits in the caption band, away from faces, and auto-fits between `min_size` and `max_size`
  so every line fits `max_width` (inside the safe zone: about 720 px on `organic`).
- The card should read bigger than the captions: at least `min_ratio` (1.3) times the caption
  size, up to `max_size` (120 px in `clean` and `pop`). Long lines set small: in Archivo Black a
  17-character line fits only at about 63 px, smaller than the captions. When your lines set
  under `min_ratio` times the caption size (or below 80% of `max_size`, if splitting makes the
  card at least 15% larger), the renderer re-breaks them (up to `hook_card.max_lines`, default
  3) and `hc render` prints "hook card re-broken". If the card is still small, the render
  reports `hook_small` and `hc plan` warns (L20) with the line length that would fit. To keep
  your own breaks, write lines of about 10-12 characters.
- An accent spoken only while the card is up never fires in the captions (captions start when
  the card ends); `hc plan` warns (L08). Accent a word spoken after the card.
- Writing it: up to about nine words, lines of about 10-12 characters, one `*accent*` run, the
  speaker's own words where possible. Numbers and names must be said in the clip (QA G11). It is
  written to be heard (the voice plays underneath), so it can be terser than a cover
  (`packaging.md`).

## Paging

A page is one or two lines of at most `max_words` words (4 in `clean`), with each line at most
`max_chars` characters (16) and `max_width` pixels, measured with the real font. A new page
starts at:

- a pause longer than `pause_break` (0.45 s);
- the end of a sentence;
- a comma, once the page has `comma_min_words` (2) words;
- every join of kind piece, omit or dip. A page never straddles such a seam: it lies wholly on
  one side, on frame boundaries, including its `lead_ms` head, its tail and any time added to
  reach `min_page`. A tighten join (a shortened pause inside a piece) may sit inside a page;
- a change of speaker.

Lines inside a page are balanced. Inside those rules the paging prefers pages that stay up at
least `min_page` (0.8 s) and never ends a page or a line on a word that needs the next one
("a", "the", "my", "of", "to"); a comma clause that would lead the next page ("met, / not your
friends") is kept whole. After paging:

- a blank gap under 0.7 s between two pages is filled by holding the earlier page (no blink),
  except across a seam;
- a one-word hesitation page ("I", "And", "So") followed within a second joins the next page;
- a page under `min_page` merges with a neighbour when it fits and the break between them is
  soft (never across a sentence end, a seam, a speaker change or a real pause);
- a page that is still short is reported as a `min_page` issue. `min_page` is a target.

Each word lights `lead_ms` (60 ms, 0-250 allowed) before it is spoken: viewers forgive text
that is slightly early far more than text that is late. The lead never reaches back across a
seam: the first word after a cut lights on the cut frame at the earliest. A page that is due
less than one frame after its cut is shown from the cut frame, so the frame of the cut is
never left without a caption between two pages.

Numbers stay whole: "80", "-90", "%" render as "80-90%", and a list such as "80, 90%" is never
split across pages.

`layout.anchor` decides which edge of a page stays put when pages have different line counts:
`top` for `build` reveal (a page growing from one to two lines does not jump), `center`
otherwise.

Broadcast subtitle guidance sets the outer limits (around 42 characters per line, two lines, 20
characters per second for adults, a minimum event of about 5/6 s; speech runs 160-180 words per
minute). Short-form pages are much shorter by design: two to four words read in a glance.

The `.srt` sidecar follows the broadcast limits instead of the burned paging: every visible word
(including the ones spoken under the hook card), natural case, at most two lines of 42
characters and 7 s of speech per cue (a cue stays up to 0.4 s after its last word, so it can
last 7.4 s), broken at sentence and clause ends. Cues also break at every join
of kind piece, omit or dip (a cue never runs across a cut), and every cue lies inside the file:
none starts before 0 or ends after the clip's duration. The hook card is not a cue.

## Caption text

- Fillers ("um", "uh") never appear: tightening removes them from the audio and the captions.
- `fix` sets display text by word index, for names and jargon the recogniser got wrong. Under
  `case: "upper"` a fix that only changes case is invisible; fix spelling, not case.
- A word that starts with a dot (".5%", ".22", ".com", ".NET") is a word of its own and keeps
  its dot under `strip_punct`; only ".5" straight after a digit joins it ("3" + ".5").
- `fix` with `""` hides a token in the captions while the audio keeps it. Use it for display
  merges: "Open" + "AI" becomes `{"812": "OpenAI", "813": ""}`. A merge whose text holds a
  space ("Started dating") still lights as one unit, and it is kept on one line when it fits;
  when it does not, it wraps at the space at full size instead of being shrunk to fit. The
  merged token stays lit and on screen until its last hidden word ends and takes that word's
  closing punctuation (a hidden "deck." ends the sentence), so do not retype the full stop; a
  mark you do type is kept. Forward only: the hidden words follow the word that carries the
  text.
- `omit` removes a word from the audio and the captions together.
- `insert` adds words the transcript missed (speech that is heard and has no word index): they
  page, light and time like any other word, spread over the `src` span you give in proportion
  to their length, and they are in the `.srt`. They have no index, so `fix`, `emphasis`, `omit`
  and `verbatim` cannot address them: write the text as it should read (`spec.md`).
- Same mistake everywhere: add the term to `P/glossary.txt` and re-transcribe. One occurrence:
  `fix` it.
- Never use `fix` to change what was said (`fidelity.md`); plan lint L06 warns when a fix hides a
  negation or changes a number.
- Invisible characters (zero-width spaces and joiners, soft hyphens, byte-order marks) are
  removed from captions, the hook card and the cover headline before measuring, with a
  `glyph_stripped` issue, so they never draw as boxes.

## Glyph coverage

Before rendering, every character of every page, the hook card and the cover headline is checked
against the font's character map. A missing glyph stops the render with the list of characters,
because Pillow would otherwise draw an empty box without complaint. Fixes: `fix` the token to
characters the font has (straight quotes for curly ones, a spelled-out currency), choose a preset
whose font covers the text, or put a font in `P/fonts/` and point a project preset at it.
Uppercasing can change spelling (German "ß" becomes "SS"); check names. Layout uses Pillow's basic
engine: Latin, Greek and Cyrillic work; scripts that need complex shaping (Arabic, Devanagari,
Thai) are not supported.

## Presets

| Preset | Look |
|---|---|
| `clean` | fill reveal, Archivo Black, white with a warm yellow accent, soft shadow, backing when needed |
| `pop` | build reveal with an emphasis pop, Montserrat Black, pages of up to three words, hot coral accent, auto emphasis on (numbers and names only), Anton cover headline |
| `minimal` | static phrases of up to six words, Montserrat SemiBold at 60 px, case as spoken, backing always, no accent colour change, no punch-in |

Make your own look as `P/presets/<name>.json` (this project) or `<data dir>/presets/<name>.json`
(every project) with `"extends": "clean"` and only the keys you change (`spec.md` lists every
key). Render several at once with `hc render P --styles clean,pop` and compare the contact
sheets side by side.

Language: the filler, stopword and negation lists are English. Other languages caption fine,
but tightening and the opener lints are weaker.
