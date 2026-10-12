# Spec reference: clips.json and presets

Every key the scripts read, with its default and an example. `hc plan P` validates clips.json and
reports problems as lints (`qa.md` lists them). The `json clips.json` blocks in this file are
complete, valid files; the test suite validates them.

## Clocks and times

- **Two clocks.** `src` is seconds from the start of the source file; `out` is seconds from the
  start of a rendered clip. Every time key ends in `_src` or `_out`.
- **Time values** are numbers or strings: `681.5`, `"681.5"`, `"11:21"`, `"11:21.40"`,
  `"1:02:03.4"`.
- **Word indices** are the `#first-last` numbers in `transcript.md` (inclusive).

## Project folder

| Path | Written by | What |
|---|---|---|
| `P/source.<ext>`, `P/source.json` | `ingest` | the source (linked or copied) and its probe |
| `P/signals.json` | `ingest` (URL sources) | heatmap and chapters on the source clock |
| `P/glossary.txt` | you | names and jargon, one per line; `transcribe` reads it |
| `P/transcript.json`, `P/transcript.md` | `transcribe` | words with indices and times; the readable version |
| `P/transcripts/<id>.json` / `.md` | `transcribe --file --clip` | caption-only clip transcripts |
| `P/moments.md` | you | the ledger: the only status store (`hooks.md`) |
| `P/clips.json` | you | the edit spec (this file) |
| `P/presets/*.json`, `P/fonts/` | you (optional) | project presets and fonts, searched first |
| `P/shots/`, `P/plans/`, `P/work/`, `P/cache/` | `shots` (and `render`, when the framing needs face data and there is none), `plan`, `render` | shot and face analysis (below), plans, lossless segments, caches and the lock files (`ingest.lock`, `.export.lock`, `.covers-grid.lock`) |
| `P/renders/<clip>__<fmt>__<style>.mp4` | `render` | finals, plus `.srt`, `.captions.json`, `.log.json` |
| `P/covers/<clip>__<style>.jpg` | `cover` | plus `.grid.jpg`, `.mobile.jpg`, `.json` |
| `P/sheets/` | several | survey, peek, shots, contact, seams, cover candidates, covers grid |
| `P/qa/` | `qa` | `<clip>.json`, `<clip>.md`, `summary.md` |
| `P/export/` | `export` | `<Style label>/NN - <Title>.mp4`, `NN - <Title> - Cover.jpg`, `manifest.json` |

### Source identity

`source.json` records the file as ingest probed it: its streams, its picture facts and a
fingerprint (the file's size and its first and last 4 MiB). The transcript, plans, cached
segments and renders are all made from that file.

- **Replaced source.** If the file is replaced after ingest (a re-export over a linked
  original), every step stops with `the source file changed since ingest` until `hc ingest P
  <original>` is run again. The fingerprint does not see an edit in the middle of a file that
  keeps its length: re-ingest by hand after any re-export.
- **Transcript of another file.** After a re-ingest, re-run `hc transcribe P` when the sound
  changed: plan, render and all stop (exit 1) while the transcript was made from another version
  of the file. The same holds for a caption-only clip and its `transcripts/<id>.json`.
- **`--cfr`** re-encodes the video of a VFR source, or of one whose ingest warning says frames
  are missing, at its nominal rate and copies the audio, so the transcript stays valid. Shot
  boundaries of a VFR or missing-frame source are placed on the nominal frame grid: re-run `hc
  shots` on a project analysed by an older version.
- **`--force`** puts the new source in place before it removes the old one, removes only the
  file `source.json` named, and never removes a file that was ingested in place (your original
  sitting in `P`). A URL `--force` over such a project is refused, and so is a local ingest over
  a `P/source.<ext>` that hook-cutter did not make and that differs from the new file: move it
  out of the project first. One ingest runs per project at a time (`P/cache/ingest.lock`).
- **Channels.** A channel far quieter than the other in the sampled minute sets
  `audio.fold_mono`: both channels are mixed to mono in equal parts, nothing is dropped. A
  source with more than two channels goes through the standard stereo downmix first.
- **Black bars.** Ingest measures them itself: a letterboxed or gallery source gets
  `video.active_area` and a "letterboxed/gallery layout" warning, and the upscale warning uses
  the real picture (a 1280x360 band in a 720p frame upscales 5.3x, not 2.7x).
- **URL sources.** Ingest skips YouTube's AI-upscaled "-sr" formats.

## clips.json

### Top level

| Key | Type | Default | Meaning |
|---|---|---|---|
| `version` | int | `1` | schema version |
| `transcript_sha` | string | none | the `sha256` of transcript.json the indices were written against; a mismatch is lint L03 (FAIL). A re-time (`hc transcribe P --retime`) keeps the old value valid (`text_compatible_shas`). A caption_only clip is pinned by its own clip-level `transcript_sha` |
| `defaults` | object | see below | defaults merged into every clip |
| `speakers` | object | `{}` | where each person sits in the source frame: `{"<label>": {"x": 0..1, "y": 0..1}}` (below) |
| `clips` | object | required | clip id to clip; the order is the export order |

Keys starting with `_` are comments and are ignored everywhere, also as a clip id in `clips`
and as a key of `fix` or `defaults.length`. Any other unknown key is a WARN (L01).

### defaults

| Key | Type | Default | Meaning |
|---|---|---|---|
| `styles` | list of preset names | `["clean"]` | styles rendered per clip |
| `formats` | list | `["vertical"]` | `vertical` (1080x1920) and/or `landscape` (1920x1080) |
| `tighten` | object | `{"enabled": true, "gap": 0.6, "keep": 0.3, "min_cut": 0.12, "fillers": true}` | pause and filler removal inside pieces |
| `safezone` | string | `"organic"` | safe-zone profile from `presets/safezones.json`. Written here it overrides a preset's own `safezone`; left out, the preset's applies (organic for the shipped styles). An unknown name is L01 FAIL |
| `length` | object | `{"vertical": [15, 30, 60, 90], "landscape": [60, 300, 900, 1800]}` | `[min, target_lo, target_hi, max]` seconds per format (L07) |

`tighten` keys: `enabled` (bool), `gap` (a quiet pause longer than this many seconds gets
shortened), `keep` (seconds of pause kept around the removal), `min_cut` (removals shorter than
this are dropped), `fillers` (remove "um", "uh" and similar). Only quiet gaps are shortened, so
laughter and applause survive.

### Clip

The id matches `^[a-z0-9][a-z0-9-]{0,47}$` (lowercase letters, digits, hyphens). A numeric
prefix keeps the order readable: `c01-no-retainers`.

| Key | Type | Default | Meaning |
|---|---|---|---|
| `ledger` | string | none | the moments.md row id, e.g. `"M07"` (recommended) |
| `title` | string | none | export file name and manifest title; claim-checked (L13 / G11); also printed on the cover when a vertical clip has neither `cover.headline` nor a hook card |
| `speaker` | string | none | whose words and claim this is (attribution) |
| `kind` | `"cut"` or `"caption_only"` | `"cut"` | a cut from the source, or captions on a finished video |
| `source_file` | path | none | caption_only only: the finished video (relative to P, or absolute) |
| `transcript_sha` | string | none | caption_only only: the `sha256` of `P/transcripts/<id>.json` that the `fix` / `emphasis` / `insert` indices were written against. A mismatch is L03 FAIL; without it the clip FAILs L03 once its transcript has been re-made with different words. `--retime` keeps it valid. Never re-renders. On a cut clip the key is an L01 WARN and is ignored (the top-level key pins those) |
| `pieces` | list | required for `cut` | what plays, in this order (below) |
| `omit` | list of word indices | `[]` | removed from audio and captions; must sit inside a word piece. `[a, b]` pairs count as inclusive ranges |
| `verbatim` | list of `[a, b]` | `[]` | inclusive ranges never tightened |
| `fix` | object `"index": "text"` | `{}` | caption display text by word index; `""` hides the caption only (audio stays). A non-empty fix on a word the transcript wrote as a filler keeps the word (not tightened away, captioned, in the intended words). Plan lint L06 warns when a fix hides a negation, hedge or qualifier or changes a number |
| `insert` | list of objects | `[]` | speech the transcript missed, so that it is captioned: `{"after": index, "text": "...", "src": [t0, t1]}` (below) |
| `emphasis` | list of word indices | `[]` | extra accent words (the preset must list `"manual"` in `emphasis.sources`, as all shipped presets do) |
| `hook_card` | object or null | `null` | `{"lines": [...], "dur": "auto"}` (below) |
| `captions` | bool | `true` | burn captions; `false` still shows the hook card |
| `caption_y` | number or null | `null` | vertical only: caption block centre y in output pixels (0-1920, else L01 FAIL), overriding the preset. Pages are kept inside the safe zone (a `caption_y_clamped` issue says when); ignored for landscape |
| `caption_backing` | `"auto"`, `"always"`, `"never"` or null | `null` | per-clip override of the preset's `caption.backing.mode` for every caption page of this clip; the hook card follows it. `null` keeps the preset's mode. Any other value is L01 FAIL. Use it when `auto` gives one shot pills and the next none |
| `framing` | list of marks | `[]` | who or what is framed, from when (below) |
| `motion` | object or null | `null` | overrides of the preset's `motion`, e.g. `{"punch_in": 1.0}` turns punch-ins off. Values are numbers of 0 or more; anything else is L01 FAIL and that key is ignored; unknown keys WARN |
| `cover` | object | `{"headline": null, "at_out": "auto", "shift": 0, "lockup": null}` | cover settings (below) |
| `styles` | list or null | `null` | per-clip override of `defaults.styles` |
| `formats` | list or null | `null` | per-clip override of `defaults.formats` |
| `tighten` | bool, object or null | `null` | per-clip override: `false` turns tightening off; an object overrides the listed keys |
| `post` | object or null | `null` | `{"title": "...", "caption": "..."}`, copied into the export manifest. Other keys are exported too, with an L01 WARN (only title and caption are claim-checked); a title or caption that is not a string is L01 FAIL |

### Piece

Exactly one of `words` or `time_src`.

| Key | Type | Default | Meaning |
|---|---|---|---|
| `words` | `[a, b]` | | inclusive word indices |
| `time_src` | `[t0, t1]` | | a source time range; never tightened; words whose midpoint falls inside are captioned. For reactions, b-roll, non-speech |
| `role` | string | `"body"` | `cold_open`, `setup`, `body`, `payoff`, `close`, `reaction` or `broll`; a label for you and for reports |
| `join` | `"cut"` or `"dip"` | `"cut"` | how this piece joins the previous one; ignored on the first piece. A dip darkens the picture by 60% (to 40% of its brightness) over 0.4 s each side |
| `expect` | `[first_text, last_text]` | `["", ""]` | anchors: the first and last words of the range. A mismatch (under 80% of tokens, in order) is lint L02, FAIL: the indices drifted. Either side may be `""` |
| `hold_after` | seconds | `0` | extra time kept after the last word (a laugh, a beat). Speech inside the hold is heard and captioned (L16); the hold stops where another piece of this clip starts, or before a word the clip omits, so it never replays a word or plays an omitted one. The resulting out-edge is snapped to quiet audio on the frame grid, so nearby values can give the same frame (the `hc plan` table shows it, and `plan.json` records it as the piece's `hold`: the time asked for, the frame chosen and why); `out_offset` then moves it frame by frame |
| `gain_db` | number, -12 to 12 | `0` | level change for this piece |
| `tighten` | bool | `true` | `false` keeps every pause in this piece |
| `chapter` | string | none | landscape exports: a chapter starts at this piece |
| `in_offset`, `out_offset` | whole frames, -60 to 60 | `0` | move the snapped in-edge or out-edge by this many output frames (+ is later), applied after every snap (and after `hold_after`). A frame or two for a stray syllable at a seam (`qa.md`); up to 60 (two seconds at 30 fps) to step over a false start that sits inside the piece's first word (`editing.md` rule 6). Beyond 12 the plan prints an INFO line with the shift in seconds; outside -60 to 60, or not a whole number, is L01 FAIL. Where two pieces that are neighbours in the source share a cut, a trimming offset (`out_offset` below 0 on the first, `in_offset` above 0 on the second) makes a real cut once it leaves a gap; while the two edges still overlap it removes nothing and an L01 INFO says so |

### Framing mark

The mark in effect at time t is the last mark with `at_src` at or before t. Before the first
mark, the clip default applies per shot (from `hc shots`): a shot with one face tracks it; a
**multi-face shot** (two or more people of similar size, such as a gallery call or a two-shot)
is letterboxed, never cropped to the largest face; no shots data or no face gives a crop at the
centre of the picture. Landscape shows the full frame. With a `speakers` map that covers every
speaker label in the clip, and either no marks or a mark with mode `speaker`, multi-face shots
follow the speaker turns instead (one window per turn; turns under 1 s or 3 words do not switch).

Every crop stays inside the source's picture area: when ingest measured black bars or a gallery
border (`video.active_area` in `source.json`), crop 1.0 is the largest window inside the picture,
and letterbox shows and blurs the picture, not the bars.

| Key | Type | Default | Meaning |
|---|---|---|---|
| `at_src` | time | required | when the mark takes effect |
| `mode` | string | required | `track` (follow a face), `fixed` (a fixed window), `letterbox` (the whole picture over a blurred fill), `full` (no crop), `speaker` (follow the `speakers` map from here) |
| `x` | 0 to 1 | | `track`: the face to follow (horizontal position); `fixed`: window centre x (default 0.5) |
| `y` | 0 to 1 | `0.5` | `fixed`: window centre y. `track`: when people are stacked in one column (a gallery), the y of the face to follow |
| `crop` | 0.3 to 1.0 | `1.0` | `track`/`fixed`/`speaker`: window size relative to the largest 9:16 (or 16:9) window that fits; below 1.0 zooms in. On a `speaker` mark it applies to the multi-face shots the mark follows; a single-face shot under the same mark keeps the default track at 1.0 (give it a `track` mark to zoom it) |
| `rect` | `[x, y, w, h]`, shares of the source frame | none | `letterbox` only: show this rectangle (one gallery tile) instead of the whole picture; it is fitted to the width and blurred for the fill. One tile of a 2-up call upscales about 1.7x where a crop would need 5x |
| `reaction` | bool | `false` | a deliberate cutaway to a listener: QA G16 warns instead of failing while it is on screen |

### speakers

`"speakers": {"Ann Lee": {"x": 0.27}, "Ben Ortiz": {"x": 0.75, "y": 0.42}}` at the top level of
clips.json. Labels are the speaker names in `transcript.md` (from `--speakers-vtt`); `x` and
`y` are the person's face centre as shares of the source frame (read them off
`sheets/shots_<clip>.jpg`). Set it once per source: framing then follows who is talking in
multi-face shots, the cover prefers frames of the clip's `speaker`, and QA G16 checks that the
framed face is the person speaking. `y` also tells apart people stacked in one column: the crop
follows the face nearest that y, and G16 checks the window's rows. A label no word or clip uses
is L01 WARN; changing an x re-plans every clip.

### insert

Speech that is clearly heard in the clip and has no transcript words (the recogniser dropped a
short repeat or an aside). Without an insert it plays uncaptioned, QA G5 reports extra words and
G10 uncaptioned voiced audio. First confirm what is said (`qa.md`, "Checking sound that has no
words"), then:

| Key | Type | Meaning |
|---|---|---|
| `after` | word index | the transcript word the speech follows; it must lie inside one of the clip's pieces (for a caption-only clip: any index of its transcript) |
| `text` | string | what is audibly said, verbatim; you are responsible for it (`fidelity.md`). Split on spaces into caption words |
| `src` | `[t0, t1]`, source time | where it is heard. It starts at or after the `after` word's start and lies inside that piece's kept audio |

A missing or wrong key, an `after` outside the clip's pieces, or a `src` that starts before
its word is L01 FAIL. The inserted words get caption times spread over `src` in proportion to
their length, appear in the captions and the `.srt`, count as spoken for QA (G5 expects to hear
them; G11 and G14 accept them), and their span is protected from tightening. They have no word
index: `fix`, `emphasis`, `omit` and `verbatim` cannot address them, so write the text as it
should read. The plan lists them in `plan.json` under `inserts` (`n`, `after`, `tokens`, `src`
and the `piece` that plays them), and each inserted word is a `words[]` entry with `i: null`
and `ins` set to that `n`.

### hook_card

| Key | Type | Default | Meaning |
|---|---|---|---|
| `lines` | list of strings | required | one string per line; `*...*` marks the accent run |
| `dur` | `"auto"` or seconds | `"auto"` | auto = 0.5 + characters / 20, clamped to 1.5-3.0 s. A number over 3 s is L01 WARN (the card hides the first captions that long); one that covers the planned clip (after tighten and omit; a caption-only clip's whole video), less 1 s, is L01 FAIL |

A legacy `"text"` with `\N` or `\n` line breaks is accepted and converted.

### cover

| Key | Type | Default | Meaning |
|---|---|---|---|
| `headline` | string | the hook card lines, else the clip `title` | cover text with `*accent*` runs; write it to parse cold (`packaging.md`) |
| `at_out` | `"auto"` or output seconds | `"auto"` | the frame; auto picks the best-scoring candidate among frames of the clip's `speaker` (when words carry speaker labels) |
| `shift` | pixels, 0 or more | `0` | slides the photo up so the face clears the headline (it cannot slide it down). Written explicitly (even `0`), it is kept as you set it; a negative value is L01 FAIL, and `hc cover` treats it as 0 |
| `lockup` | list of strings or null | `null` | small lines under the headline (a name, a show title the user owns) |

### Caption-only clips

`"kind": "caption_only"` adds captions, a hook card and a cover to a video that is already cut.
`source_file` is required, `pieces` must be absent, tightening is off, and the transcript lives
at `P/transcripts/<id>.json`: create it with `hc transcribe P --file <video> --clip <id>`. Word
indices in `fix`, `emphasis` and `insert` refer to that transcript and are pinned by the clip's
`transcript_sha`: after the transcript is re-made with different words, plan FAILs L03 until
you check them (`hc words P A B --clip <id>`) and set it. A transcript made from another
version of the video stops plan (exit 1) until `hc transcribe P --file <video> --clip <id>` is
run again.

## Examples

### Minimal

```json clips.json
{
  "version": 1,
  "clips": {
    "c01-one-channel": {
      "ledger": "M09",
      "title": "One channel is not a business",
      "speaker": "Speaker A",
      "pieces": [
        {"words": [1204, 1288], "expect": ["If every lead comes", "do not own a business"]}
      ],
      "hook_card": {"lines": ["ONE CHANNEL IS NOT", "A *BUSINESS*"], "dur": "auto"}
    }
  }
}
```

### Cold open with the editing controls

```json clips.json
{
  "version": 1,
  "defaults": {
    "styles": ["clean", "pop"],
    "formats": ["vertical"],
    "tighten": {"gap": 0.7},
    "safezone": "organic"
  },
  "clips": {
    "c02-hid-the-numbers": {
      "ledger": "M17",
      "title": "I hid the numbers from my own team",
      "speaker": "Speaker B",
      "pieces": [
        {"words": [3410, 3424], "role": "cold_open", "expect": ["I hid the numbers", "for a whole year"]},
        {"words": [3301, 3360], "role": "setup", "join": "dip", "expect": ["We had just hired", "nobody asked"]},
        {"words": [3425, 3470], "role": "payoff", "expect": ["When I finally showed them", "fixed it in a week"]},
        {"time_src": ["23:10.40", "23:12.00"], "role": "reaction"}
      ],
      "omit": [3330, 3331],
      "verbatim": [[3460, 3470]],
      "fix": {"3315": "Kubernetes"},
      "insert": [{"after": 3440, "text": "Every single month.", "src": ["22:56.40", "22:57.20"]}],
      "emphasis": [3452],
      "hook_card": {"lines": ["I HID THE NUMBERS", "FROM MY OWN *TEAM*"], "dur": 2.4},
      "caption_y": 1180,
      "caption_backing": "always",
      "framing": [
        {"at_src": "22:00.0", "mode": "track", "x": 0.64, "crop": 0.9},
        {"at_src": "22:15.5", "mode": "letterbox"},
        {"at_src": "22:21.0", "mode": "fixed", "x": 0.35, "y": 0.45, "crop": 1.0}
      ],
      "motion": {"punch_in": 1.08},
      "cover": {"headline": "THE YEAR I KEPT MY TEAM *IN THE DARK*", "at_out": 6.5, "shift": 120, "lockup": null},
      "post": {
        "title": "Hiding the numbers from your team costs more than the bad news",
        "caption": "For a year he kept the real numbers to himself. The fix took a week."
      }
    }
  }
}
```

### Landscape tease with chapters

```json clips.json
{
  "version": 1,
  "defaults": {"styles": ["minimal"], "formats": ["landscape"]},
  "clips": {
    "c03-pricing-bet": {
      "ledger": "M22",
      "title": "The pricing bet that paid off twenty minutes later",
      "speaker": "Speaker B",
      "pieces": [
        {"words": [8840, 8871], "role": "cold_open", "expect": ["It doubled", "in one quarter"]},
        {"words": [4100, 4950], "role": "body", "join": "dip", "chapter": "We raised prices before we were ready", "expect": ["So the bet was", "and we held"]},
        {"words": [8700, 8900], "role": "close", "chapter": "Doubling in one quarter", "hold_after": 0.8, "expect": ["Here is what happened", "never looked back"]}
      ],
      "hook_card": null,
      "captions": true
    }
  }
}
```

### Caption-only

```json clips.json
{
  "version": 1,
  "clips": {
    "c04-selfie-take": {
      "kind": "caption_only",
      "title": "The call I almost did not make",
      "speaker": "Speaker A",
      "source_file": "inputs/selfie-take-3.mp4",
      "fix": {"14": "Postgres"},
      "hook_card": {"lines": ["THE CALL I ALMOST", "*DIDN'T* MAKE"], "dur": "auto"},
      "cover": {"headline": "THE CALL I ALMOST *DIDN'T* MAKE", "at_out": "auto"}
    }
  }
}
```

### Annotated reference

Every key at once, with comments (JSON with comments, for reading; not a valid file):

```jsonc
{
  "version": 1,
  "transcript_sha": "…",                     // optional; transcript.json "sha256" (or a compatible sha)
  "defaults": {
    "styles": ["clean"], "formats": ["vertical"],
    "tighten": {"enabled": true, "gap": 0.6, "keep": 0.3, "min_cut": 0.12, "fillers": true},
    "safezone": "organic",
    "length": {"vertical": [15, 30, 60, 90], "landscape": [60, 300, 900, 1800]}
  },
  "speakers": {"Speaker A": {"x": 0.27}, "Speaker B": {"x": 0.75, "y": 0.4}},   // face x (and y) per label
  "clips": {
    "c01-one-plan": {                         // id: ^[a-z0-9][a-z0-9-]{0,47}$ ; order = export order
      "ledger": "M07",                         // moments.md row id
      "title": "I only had one plan",          // export file name + manifest; claim-checked
      "speaker": "Speaker B",                  // whose words and claim
      "kind": "cut",                           // "cut" | "caption_only"
      "source_file": null,                     // caption_only: path (relative to P or absolute)
      // "transcript_sha": "…",                // caption_only: pins fix / emphasis / insert indices
      "pieces": [
        {"words": [4812, 4866], "role": "cold_open", "expect": ["I had one plan", "anyone noticed"]},
        {"words": [4701, 4811], "role": "setup", "join": "dip", "expect": ["So we had", "the whole year"],
         "in_offset": 0, "out_offset": 0},   // whole frames, -60..60: move a snapped edge
        {"words": [4867, 4890], "role": "close", "hold_after": 0.6, "expect": ["That is why", "shipped it"]},
        {"time_src": ["1:02:03.40", "1:02:05.00"], "role": "reaction"}
      ],
      "omit": [4730, 4731],                    // removed from audio + captions
      "verbatim": [[4840, 4866]],              // inclusive; never tightened
      "fix": {"4755": "Kubernetes", "4756": ""},   // caption text by index; "" hides the caption only
      "insert": [{"after": 4760, "text": "Say it again.", "src": [3741.2, 3741.9]}],   // heard, not in the transcript
      "emphasis": [4850],                      // extra accent words
      "hook_card": {"lines": ["I HAD ONE PLAN:", "*SHIP* FIRST"], "dur": "auto"},   // or null
      "captions": true,                        // false keeps the hook card, drops the captions
      "caption_y": null,                       // vertical: caption block centre y (output px, 0-1920)
      "caption_backing": null,                 // "auto" | "always" | "never": overrides the preset's backing mode
      "framing": [
        {"at_src": 3725.0, "mode": "track", "x": 0.42, "crop": 0.62},
        {"at_src": 3740.1, "mode": "letterbox"},
        {"at_src": 3744.0, "mode": "letterbox", "rect": [0.0, 0.25, 0.5, 0.5]},   // one gallery tile
        {"at_src": 3747.0, "mode": "track", "x": 0.75, "reaction": true},         // a listener cutaway
        {"at_src": 3750.0, "mode": "fixed", "x": 0.5, "y": 0.45, "crop": 1.0},
        {"at_src": 3760.0, "mode": "speaker"}  // follow the speakers map from here
      ],
      "motion": null,                          // e.g. {"punch_in": 1.0}; numbers of 0 or more
      "cover": {"headline": "SHIP BEFORE ANYONE *NOTICES*", "at_out": "auto", "shift": 0, "lockup": null},   // shift: 0 or more
      "styles": null, "formats": null, "tighten": null,   // per-clip overrides of defaults
      "post": {"title": "…", "caption": "…"}   // copied into the export manifest; claim-checked
    }
  }
}
```

## Transcript flags

`transcript.json` carries `flags`: stretches the recogniser or the audio check is unsure about.
`transcript.md` shows them inline, and plan lint L15 warns when one lies inside a clip.

| `kind` | Fields | Means | L15 says |
|---|---|---|---|
| `repetition_loop`, `low_confidence` | `a`, `b` (word indices) | a loop of repeated words; a word the recogniser doubted | read the words with `hc words`; `fix` a wrong one |
| `unworded_audio` | `s`, `e` (source seconds) | sound with no words: laughter, music, or speech the recogniser missed | "N s of sound with no words at src T is inside the clip": check it (`qa.md`), caption real speech with `insert` |
| `dropout` | `s`, `e` | the speakers VTT has words here that the transcript lacks | as above |
| `speaker_uncertain` | `a`, `b` | the speaker label is a guess | check attribution (`fidelity.md`) |
| `no_audio` | `a`, `b` | words with no audible sound: likely invented | omit them |
| `false_start` | `a`, `b`, `s`, `e`, sometimes `heard` | words `a`-`b` hold two attempts at a phrase folded into one run ("after a... after about"); `s`-`e` is the first attempt in source seconds (`e` is where the kept attempt starts), and `heard` is what a re-listen of that stretch heard, when it heard words. The words are never rewritten | "possible false start or repeated attempt at src T (words a-b): check with hc words and a slice transcription before opening on it" |
| `timing_uncertain` | `a`, `b` | the start of these words could not be placed: another sound (a laugh, a second voice) overlaps it | "word timing near src T is uncertain (overlapping sound): check the caption onset in the render" |

`false_start` and `timing_uncertain` warn only when the flagged words are inside the clip, and
`false_start` only while the clip still plays the first attempt (`s`-`e`): once an edge sits on
the restart, the flag is settled.
Remedies: a false start at a piece's first word is skipped with `in_offset` (up to 60 frames);
an uncertain word start is checked on the render (QA G10 names tokens that light early) and, if
wrong, hidden with `fix` while the next word carries its text.

## shots json

`hc shots` writes `P/shots/<clip>.json` (and `hc render` writes it itself when a clip's framing
needs face data that is missing or does not cover the clip). The plan and every render record
the file's hash: when it appears or changes, `hc status` shows `STALE(shots)`. The parts you may
want to read:

| Key | What |
|---|---|
| `version`, `grid` | `2` and `"absolute"`: face samples sit on a fixed source-time grid (below). A file without them was written by an older analysis; `hc render` analyses the clip again (once) before it trusts the tracks |
| `ranges` | the source ranges that were analysed |
| `shots[]` | `start_src`, `end_src`, `kind` (`none`, `single`, `wide`, `two`, `group`), `multi_face`, and `faces[]` with `x`, `y`, `h` (shares of the frame) and `presence`. Faces are told apart by x only: people stacked in one column count as one face |
| `multi_face` (per shot) | two or more people of similar size share the shot: the default framing letterboxes it and QA G16 checks who is framed. Two faces count as similar when their heights are within a factor of 1.6, however small they are (a wide two-shot) |
| `transitions[]` | one entry per shot boundary: `{"at_src": t, "kind": "cut" or "dissolve", "start_src": a, "end_src": b}`. For a hard cut `a == b == t`. For a cross dissolve `a`-`b` is the blended span and `t` its midpoint; the shots on either side meet at `t`, framing switches marks there, and cover candidates skip frames inside the span (two faces superimposed) |
| `samples[]` | the face samples, taken on a fixed source-time grid, so analysing a longer or shorter range that covers a piece gives the same tracks for it |

## Presets

A preset is a JSON file describing a look. `--styles clean,pop` renders one file per style from
the same cut.

- **Names** are lowercase letters, digits and hyphens (`studio`, `pop-2`), matching the file
  name exactly on every disk; `safezones` is reserved. A style in clips.json that does not
  resolve is lint L01 (FAIL).
- **Search order:** `P/presets/<name>.json`, then `<data dir>/presets/<name>.json`, then the
  skill's `presets/`. The data dir is `$HOOKCUTTER_HOME`, or `~/Library/Application
  Support/hook-cutter` (macOS), `~/.local/share/hook-cutter` (Linux), `%LOCALAPPDATA%\hook-cutter`
  (Windows).
- **`"extends": "<name>"`** deep-merges your keys over another preset. Write only what changes.
- **`"landscape": {...}`** holds overrides deep-merged when rendering the landscape format.
- **`null` does not mean inherit.** `caption`, `hook_card`, `caption.layout` or `caption.anim`
  set to null, and a null or non-numeric `size`, `lead_ms`, `hook_card.min_size` / `max_size` /
  `fade_out_ms`, stop the preset loading (L01 FAIL `style ... cannot be used`): delete the key
  to inherit it. `motion`, `letterbox`, `landscape`, `backing`, `shadow`, `emphasis` and
  `outline` may be null.
- **Fonts:** `"File.ttf"`, or `"File.ttf:Instance"` for a named instance of a variable font
  (`"Montserrat[wght].ttf:Black"`). Searched in `P/fonts/`, `<data dir>/fonts/`, the skill's
  `fonts/`, or an absolute path. Bundled: `ArchivoBlack-Regular.ttf`, `Anton-Regular.ttf`,
  `Montserrat[wght].ttf` (instances such as Black, Bold, SemiBold).
- **Colours** are `#RRGGBB` or `#RRGGBBAA`.

A custom preset (`P/presets/studio.json`):

```json
{
  "name": "studio",
  "label": "Studio",
  "extends": "clean",
  "caption": {"accent": "#7FD1FF", "layout": {"max_words": 3, "y": 1300}},
  "cover": {"accent": "#7FD1FF", "lockup": ["THE WEEKLY SHOW"]}
}
```

### Top level

| Key | Meaning |
|---|---|
| `name` | preset name (matches the file name) |
| `label` | human label; used for the export folder name. Styles that share a label each get ` (<style>)` appended |
| `extends` | parent preset name or null |
| `description` | one or two sentences on the look |
| `caption` | caption settings (below) |
| `hook_card` | hook card settings (below) |
| `landscape` | overrides applied for the landscape format |
| `letterbox` | `y_center` (output y of the letterboxed frame's centre), `blur`, `dim` (0-1) of the fill. Values are numbers; anything else stops the preset loading |
| `motion` | `punch_in` (zoom factor at same-shot joins; 1.0 = off), `min_gap` (seconds between toggles), `max_upscale`. Values are numbers; anything else stops the preset loading |
| `cover` | cover settings (below) |
| `safezone` | safe-zone profile for this look, used when clips.json does not set `defaults.safezone`; `landscape.safezone` likewise for landscape. An unknown name is L01 FAIL |

### caption

| Key | `clean` value | Meaning |
|---|---|---|
| `font`, `size`, `tracking` | `ArchivoBlack-Regular.ttf`, `78`, `0` | typeface, pixel size, extra letter spacing |
| `case` | `"upper"` | `upper` or `as_is` |
| `strip_punct` | `".,"` | characters stripped from token ends (`$ % ' ?` are kept unless listed) |
| `color` | `#FFFFFF` | spoken text colour |
| `unspoken_alpha` | `0.70` | opacity of not-yet-spoken words (fill reveal) |
| `accent` | `#FFD84D` | accent colour for emphasis words |
| `accent_reveal` | `"on_spoken"` | `on_spoken` or `page_start` |
| `emphasis.sources` | `["hook", "manual"]` | accent sources in order: `hook` (card accent runs), `manual` (clip `emphasis`), `auto` (opt-in: numbers, money and mid-sentence names only, at least 4 s apart) |
| `emphasis.per_page` | `1` | maximum accent words per page |
| `emphasis.font`, `emphasis.size_mult` | `null`, `1.0` | optional different face and size for accent words |
| `reveal` | `"fill"` | `fill`, `build` or `static` |
| `anim.in_ms`, `anim.rise_px`, `anim.pop` | `100`, `8`, `false` | build reveal: fade-in time, rise distance, accent pop |
| `shadow` | two layers | list of `{"blur", "alpha", "dy"}` |
| `outline.px`, `outline.color` | `0`, `#000000` | outline width (0 = none) and colour |
| `backing.mode` | `"auto"` | `auto`, `always` or `never`; a clip's `caption_backing` overrides it |
| `backing.color`, `radius`, `pad` | `#0B0B0DD9`, `22`, `[28, 14]` | pill colour, corner radius, padding [x, y] |
| `backing.min_contrast`, `max_edges` | `3.0`, `0.15` | auto: a pill when the shadow-aware contrast of the text on its page background is below `min_contrast`, or the background is busier than `max_edges` (share of edge pixels); decided per shot (`captions.md`) |
| `layout.max_words` | `4` | words per page |
| `layout.max_chars` | `16` | characters per line |
| `layout.max_lines` | `2` | lines per page |
| `layout.max_width` | `780` | pixels per line (clamped to the safe zone's width; a `width_clamped` issue says when) |
| `layout.anchor` | by reveal | `top` (first line fixed, the default for `build`), `center` (the default otherwise) or `bottom` |
| `layout.line_gap` | `0.98` | line spacing as a multiple of the size |
| `layout.pause_break` | `0.45` | a pause longer than this starts a new page |
| `layout.comma_min_words` | `2` | a comma breaks the page once it has this many words |
| `layout.min_page` | `0.8` | target minimum seconds on screen: short pages are merged or held (never shown more than 0.4 s before their first word); one still shorter gets a `min_page` issue |
| `layout.y` | `1250` | caption block centre y (output pixels, vertical) |
| `lead_ms` | `60` | captions lead the audio by this much (0-250; anything else, null included, stops the preset loading) |

### hook_card

`font`, `max_size` (120), `min_size` (56), `max_width` (920), `min_ratio` (1.3), `color`,
`accent`, `case`, `line_gap`, `fade_out_ms` (260), `backing.mode` (`auto`, `always` or `never`).
The card is fully visible from frame 0 and fades out over `fade_out_ms`, ending at its duration.
When the lines would set below `min_ratio` times the caption size, the card is re-broken into up
to 3 lines; if it is still small, the render reports a `hook_small` issue (QA G9, lint L20).

### cover

| Key | `clean` value | Meaning |
|---|---|---|
| `headline_font` | `ArchivoBlack-Regular.ttf` | headline typeface |
| `size_range` | `[96, 168]` | auto-fit range in pixels |
| `max_lines` | `4` | headline lines |
| `color`, `accent` | `#FFFFFF`, `#FFD84D` | text and accent colours |
| `band` | `[240, 1680]` | the 3:4 grid-safe band (y range) the headline stays inside |
| `margin` | `70` | side and bottom margin inside the band |
| `tint`, `tint_from`, `tint_alpha` | `#0B0B0D`, `0.42`, `0.86` | bottom gradient: colour, start (fraction of height), maximum opacity |
| `rule` | `false` | a short accent-coloured rule that goes with the lockup |
| `lockup`, `lockup_font`, `lockup_size` | `[]`, `Montserrat[wght].ttf:SemiBold`, `34` | default small lines under the headline |

### Safe zones (`presets/safezones.json`)

```json
{
  "organic": {"top": 240, "bottom": 420, "left": 90, "right": 150},
  "ads": {"top": 270, "bottom": 672, "left": 65, "right": 242},
  "landscape": {"top": 54, "bottom": 54, "left": 96, "right": 96}
}
```

Margins in output pixels; each profile is an object with `top`, `bottom`, `left` and `right`.
Sources and caveats are in `captions.md`.
