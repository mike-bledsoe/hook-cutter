# Fidelity

These clips put words in front of strangers under someone's name. The rules here protect the
person speaking, the person publishing and the viewer. When a rule here conflicts with a craft
rule anywhere else, this file wins.

## 1. Attribution

- **Never attribute words to the wrong person. When in doubt, drop the clip** (PASS
  `attribution`).
- How it goes wrong:
  - transcript speaker labels drift or merge two voices;
  - a cut starts after the subject, so the line reads as the speaker's own act ("rebuilt the
    onboarding" when the speaker said "she rebuilt the onboarding");
  - the camera shows the listener while the speaker talks (call software's active-speaker lag,
    multicam cutaways);
  - a cover shows one person's face over another person's claim;
  - a first-person hook card ("I...") on a line that was about someone else.
- Gate B is looking: `hc peek` the span and `hc shots` it. The person who owns the claim is on
  camera when they say it, or the framing honestly shows the group (letterbox).
- In a gallery or two-shot recording the largest face is often not the speaker. Set the
  `speakers` map (`editing.md` rule 19); QA G16 fails a render that frames someone else while a
  person talks.
- `speaker` in clips.json names whose claim it is; the export manifest carries it. The cover
  shows that person: cover auto-pick prefers their frames and QA G12 warns when the chosen frame
  belongs to someone else.

## 2. Faces and voices

- **Never generate or clone a face or a voice.** No synthetic speech to patch a missing word, no
  lip-sync, no AI-extended frames, no relighting or retouching that changes expression, gaze,
  mouth or gesture.
- Covers come from real frames of the clip.

## 3. Words

- **If a phrase is garbled, cut around it. Never guess words into someone's mouth.**
- `fix` corrects the spelling of what was actually said (a name the recogniser misheard),
  confirmed against the glossary or the user. It never improves what was said (plan lint L06
  warns when a fix changes a number).
- `fix` with `""` is for display merges only, never to hide a word that changes the meaning
  (plan lint L06 warns when it hides a negation, hedge or qualifier).
- Never omit a negation, a hedge that bounds the claim, or the "if" of a conditional (plan lint
  L06 warns). Never join two clauses so that they read as a new claim the speaker did not make.
- Reordering is fine when every piece keeps its meaning. Answering question B with the answer to
  question A is not.
- The word-diff gate (G5) proves the render contains the words you planned. It cannot see a
  planned change of meaning. That check is yours.

## 4. Written claims are narrower than speech

The hook card, the cover, the title and the post are written claims. A loose phrase in
conversation becomes an assertion someone can check. Write the narrow true version, and cut an
overclaim rather than hedge it. A speaker who said "we probably tripled it" does not get a card
reading `WE *TRIPLED* IT`; use a different line, or write what they actually claimed.

## 5. Numbers

- Numbers on a written surface (hook card, cover, title, post caption) must be real and said in
  the clip (G11 fails otherwise). Never invent one. If a number would help and was not said, ask
  the user; do not supply it.
- **Record cherry-picks.** If the speaker said "three to six months" and the cover says 6, write
  that in the ledger reason and tell the user at gate E.
- Normalise the form, not the meaning: `$2.4M` for "two point four million" is fine; `$2M+` for
  "almost two million" is not.

## 6. Guest claims become the publisher's claims

Whoever posts the clip amplifies every claim in it. Keep a Risks table at the bottom of
`moments.md`:

```markdown
## Risks
| id | #a-b | kind | what | decision |
|---|---|---|---|---|
| R01 | #3302-3340 | third-party | states a named competitor's revenue | never on a written surface; M14 ends 12 s before it |
| R02 | #4410-4422 | private | names a family member's school | cut around; nothing within 20 s ships |
```

Kinds:

- **fact**: wrong dates, wrong history, a misremembered statistic;
- **number**: inflated or unverifiable counts; a price ceiling quoted as the price;
- **third-party**: claims about named companies or people;
- **deal**: a live negotiation, an unannounced partnership;
- **private**: addresses, phone numbers, health, children, other people's finances;
- **reputation**: a swipe at a group that will land as the publisher's view;
- **profanity**: when the user's platforms penalise it.

Decisions: cut around it, keep the speech but never put it on a written surface, or PASS the
clip. **Measure proximity** with `hc words` (words and seconds from each edge); never judge it by
eye. Use credentials as the person states them on their own site or bio, not as paraphrased in
conversation.

Proper nouns and links: spell names from an authoritative source (the person's own site, the
user), not from the transcript. A web address that loads is not proof it is the right one;
lookalike domains load too.

## 7. Rights and originality

- **Ask before cutting:** does the user have rights to this footage, and whose brand does it
  carry? Their own show or talk is fine. A conference panel, someone else's podcast or a webinar
  they attended needs the owner's permission.
- Platforms demote or demonetise unoriginal reposts of other people's footage (YouTube's
  inauthentic-content rules, Instagram's originality ranking). A credit line does not make a
  repost original.
- **Never apply another organisation's branding** (logos, lower thirds, show names, house colours)
  to clips made from its footage unless that organisation asked for it, and never imply an
  endorsement.
- Credit speakers by name, spelled as they spell it. Naming the clip's own speakers in the post
  title or caption is fine even when nobody says the name in the clip (QA exempts them there);
  every other name on a written surface must be said in the clip.
- PASS `rights` when the answer is no or unknown.

## 8. Privacy of the material

Speech recognition, face detection, framing and rendering run on the user's machine. The
transcript, the ledger and any sheets you open are read by you, the agent, which means they reach
your model provider. Say so before working on confidential recordings, and never copy recordings
or transcripts anywhere outside the project folder.
