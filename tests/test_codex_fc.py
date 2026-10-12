"""Regression tests for the outside-review fixes in transcription, claims and text.

transcribe.py: the glossary prompt-echo filter keeps confidently spoken terms and does not run
where no prompt was sent; sound events are tagged and reported; a re-made transcript with other
word texts carries text_changed_from; the speakers VTT of another source is not reused.
claims.py: spelled-out scaled amounts, zero claims tied to their noun, exact name matching, and
the one list of written surfaces (cover headline, clip title, chapter titles).
textnorm.py: a leading decimal point. find.py: one hit per word range."""

from __future__ import annotations

import argparse
import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts"))
sys.path.insert(0, str(HERE))

from hookcutter import claims, env, find, textnorm as tn, transcribe as T  # noqa: E402
from hookcutter.audio import EnergyTrack  # noqa: E402

try:
    import fixtures  # noqa: E402
except Exception:  # noqa: BLE001 - media tests skip when the shared fixtures are unavailable
    fixtures = None

needs_fixtures = unittest.skipIf(fixtures is None, "tests/fixtures.py unavailable")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def track(segments=(), dur=30.0, floor=-60.0, hop=0.01):
    db = np.full(int(round(dur / hop)), floor, dtype=np.float64)
    for s, e, level in segments:
        db[int(round(s / hop)):int(round(e / hop))] = level
    return EnergyTrack(db, hop=hop, t0=0.0)


def W(text, s, e, p=0.9):
    return {"w": text, "s": s, "e": e, "p": p}


def seq(text, start=0.0, step=0.3, p=0.9):
    return [W(t, round(start + k * step, 3), round(start + k * step + step * 0.8, 3), p)
            for k, t in enumerate(text.split())]


def raw(text, t0=1.0, step=0.4, dur=0.35, p=0.99):
    return [[" " + w, round(t0 + k * step, 3), round(t0 + k * step + dur, 3), p]
            for k, w in enumerate(text.split())]


def quiet(fn, *a, **kw):
    """(result, stderr text) of fn(*a, **kw)."""
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        res = fn(*a, **kw)
    return res, err.getvalue()


def asr_of(words, **kw):
    base = {"backend": "mlx", "model": "m", "glossary": [], "prompt": None, "options": {},
            "words": [list(w) for w in words]}
    base.update(kw)
    return base


def build(asr, **kw):
    doc, err = quiet(T.build_transcript, asr, offset=0.0, media="s.mp4", fingerprint=None, **kw)
    return doc, err


def unspoken(surface, speech, label="cover headline", **kw):
    return claims.unspoken_claims([(label, surface)], speech.split(), **kw)


def numbers(surface, speech, label="cover headline"):
    return [c for c in unspoken(surface, speech, label) if c["kind"] == "number"]


def names(surface, speech, label="cover headline", **kw):
    return [c["raw"] for c in unspoken(surface, speech, label, **kw) if c["kind"] == "name"]


def mk(text, step=0.4):
    return [{"i": k, "w": t, "s": round(k * step, 3), "e": round(k * step + 0.3, 3), "p": 0.9,
             "sp": None} for k, t in enumerate(text.split())]


# ---------------------------------------------------------------------------
# S2 / V13-N1: the prompt-echo filter
# ---------------------------------------------------------------------------


class PromptEcho(unittest.TestCase):
    PROMPT = "New York City."

    def test_confident_whole_glossary_is_speech(self):
        words = seq("I visited New York City yesterday", p=0.98)
        kept, n = T.drop_prompt_echo(words, self.PROMPT)
        self.assertEqual(n, 0)
        self.assertEqual(len(kept), 6)

    def test_a_near_zero_word_marks_the_echo(self):
        words = seq("I visited New York City yesterday", p=0.98)
        for w, p in zip(words[2:5], (0.0, 0.1, 0.46)):
            w["p"] = p
        kept, n = T.drop_prompt_echo(words, self.PROMPT)
        self.assertEqual(n, 3)
        self.assertEqual([w["w"] for w in kept], ["I", "visited", "yesterday"])

    def test_words_without_confidence_are_kept(self):
        words = seq("I visited New York City yesterday", p=None)
        kept, n = T.drop_prompt_echo(words, self.PROMPT)
        self.assertEqual((n, len(kept)), (0, 6))

    def test_every_mention_survives_and_only_the_echo_goes(self):
        words = seq("the Wim Hof Method works and the Wim Hof Method is free", p=0.97)
        self.assertEqual(T.drop_prompt_echo(words, "Wim Hof Method.")[1], 0)
        words[7]["p"] = 0.01                       # the second run is an echo over silence
        kept, n = T.drop_prompt_echo(words, "Wim Hof Method.")
        self.assertEqual(n, 3)
        self.assertEqual(" ".join(w["w"] for w in kept), "the Wim Hof Method works and the is free")

    def test_build_transcript_keeps_a_spoken_one_term_glossary(self):
        gl = ["New York City"]
        asr = asr_of(raw("I visited New York City yesterday"), glossary=gl,
                     prompt=T.build_prompt(gl), options={"initial_prompt": True})
        doc, err = build(asr)
        self.assertEqual([w["w"] for w in doc["words"]],
                         ["I", "visited", "New", "York", "City", "yesterday"])
        self.assertNotIn("echoed the glossary prompt", err)
        self.assertEqual(T.missing_glossary_terms(doc["words"], gl), [])

    def test_plain_retry_is_not_echo_filtered(self):
        """The prompt-free second listen cannot echo a prompt: what it hears is spliced, even a
        whole-glossary run whose first word is unsure."""
        ws = [W("I", 1.0, 1.35), W("visited", 1.4, 1.75), W("yesterday", 4.6, 4.95)]
        E = track([(1.0, 1.8, -30.0), (1.8, 4.5, -30.0), (4.6, 5.0, -30.0)], dur=8.0)
        seen = []

        class Plain:
            can_retry = True

            def __call__(self, a, b, plain=False):
                seen.append("plain" if plain else "prompted")
                if not plain:
                    return []
                return [[" New", 2.0, 2.6, 0.01], [" York", 2.7, 3.3, 0.98],
                        [" City", 3.4, 4.2, 0.99]]

        out, gaps = T.fill_holes([dict(w) for w in ws], E, Plain(), prompt=self.PROMPT,
                                 duration=8.0)
        self.assertEqual(seen, ["prompted", "plain"])
        self.assertIn("New", [w["w"] for w in out])
        self.assertGreater(gaps[0]["n"], 0)

    def test_prompted_relisten_still_drops_an_echo(self):
        ws = [W("I", 1.0, 1.35), W("visited", 1.4, 1.75), W("yesterday", 4.6, 4.95)]
        E = track([(1.0, 1.8, -30.0), (1.8, 4.5, -30.0), (4.6, 5.0, -30.0)], dur=8.0)

        class Echo:
            can_retry = False

            def __call__(self, a, b, plain=False):
                return [[" New", 2.0, 2.6, 0.0], [" York", 2.7, 3.3, 0.1],
                        [" City.", 3.4, 4.2, 0.46]]

        out, gaps = T.fill_holes([dict(w) for w in ws], E, Echo(), prompt=self.PROMPT,
                                 duration=8.0)
        self.assertEqual([w["w"] for w in out], ["I", "visited", "yesterday"])
        self.assertEqual(gaps[0]["n"], 0)

    def test_prompt_sent_reads_the_runner_options(self):
        p = self.PROMPT
        self.assertEqual(T.prompt_sent({"backend": "mlx", "prompt": p,
                                        "options": {"initial_prompt": True}}), p)
        self.assertEqual(T.prompt_sent({"backend": "faster-whisper", "prompt": p,
                                        "options": {"hotwords": True}}), p)
        self.assertIsNone(T.prompt_sent({"backend": "faster-whisper", "prompt": p,
                                         "options": {"hotwords": False}}))
        self.assertIsNone(T.prompt_sent({"backend": "whispercpp", "prompt": p,
                                         "options": {"prompt": False, "ml": 1}}))
        self.assertIsNone(T.prompt_sent({"backend": "mlx", "prompt": None, "options": {}}))
        # a result with no prompt flag at all (an older cache file) keeps the recorded prompt
        self.assertEqual(T.prompt_sent({"backend": "mlx", "prompt": p, "options": {}}), p)
        self.assertEqual(T.prompt_sent({"backend": "mlx", "prompt": p}), p)

    def test_whisper_cpp_shaped_result_keeps_the_term(self):
        """whisper.cpp is never sent the glossary, so nothing it writes is an echo: not even a
        run with an unsure word."""
        gl = ["New York City"]
        words = raw("I visited New York City yesterday", p=0.98)
        words[2][3] = 0.01
        asr = asr_of(words, backend="whispercpp", glossary=gl, prompt=T.build_prompt(gl),
                     options=T.runner_options("whispercpp", "ggml-base.bin", T.build_prompt(gl)))
        self.assertFalse(asr["options"]["prompt"])
        doc, err = build(asr)
        self.assertEqual([w["w"] for w in doc["words"]],
                         ["I", "visited", "New", "York", "City", "yesterday"])
        self.assertNotIn("echoed", err)

    def test_qa_hearing_follows_the_post_version(self):
        self.assertGreaterEqual(T.POST_VERSION, 5)


# ---------------------------------------------------------------------------
# S3: sound events are visible
# ---------------------------------------------------------------------------


class SoundEvents(unittest.TestCase):
    def test_removed_runs_are_tagged_and_reported(self):
        words = [[" It", 0.2, 0.4, 0.97], [" was", 0.5, 0.7, 0.97], [" (not", 0.8, 1.0, 0.97],
                 [" really)", 1.1, 1.5, 0.97], [" easy.", 1.6, 2.0, 0.97],
                 [" (laughs)", 2.2, 2.6, 0.5], [" So", 3.0, 3.2, 0.9],
                 [" (laughs)", 3.4, 3.8, 0.5], [" [Music]", 4.0, 5.0, 0.4]]
        doc, _ = build(asr_of(words))
        self.assertEqual([w["w"] for w in doc["words"]], ["It", "was", "easy.", "So"])
        sound = [e for e in doc["events"] if e.get("kind") == "sound"]
        self.assertEqual([e["text"] for e in sound],
                         ["(not really)", "(laughs)", "(laughs)", "[Music]"])
        self.assertEqual((sound[0]["s"], sound[0]["e"]), (0.8, 1.5))
        _, err = quiet(T._report, doc, [], Path("transcript.json"))
        line = [ln for ln in err.splitlines() if "sound event" in ln]
        self.assertEqual(len(line), 1, err)
        self.assertIn("set aside 4 sound events", line[0])
        self.assertIn("(not really)", line[0])
        self.assertIn("(laughs) x2", line[0])
        self.assertIn("[Music]", line[0])
        self.assertIn("transcript.json events", line[0])

    def test_no_line_without_sound_events(self):
        doc, _ = build(asr_of(raw("plain words only here")))
        self.assertEqual(doc["events"], [])
        _, err = quiet(T._report, doc, [], Path("transcript.json"))
        self.assertNotIn("sound event", err)

    def test_summary_is_capped(self):
        events = [{"kind": "sound", "text": "(noise %d)" % k} for k in range(12)]
        events.append({"kind": "no_audio", "text": "Thanks"})
        line = T.sound_event_summary(events, limit=3)
        self.assertIn("set aside 12 sound events", line)
        self.assertIn("(noise 2)", line)
        self.assertNotIn("(noise 3)", line)
        self.assertIn("(+9 more)", line)
        self.assertNotIn("Thanks", line)
        self.assertIsNone(T.sound_event_summary([{"kind": "no_audio", "text": "x"}]))

    def test_normalize_words_contract_is_unchanged(self):
        events = []
        words, _ = T.normalize_words([[" So", 0.0, 0.2, 0.9], [" (laughs)", 0.3, 0.6, 0.5],
                                      [" yeah.", 0.7, 0.9, 0.9]], events=events)
        self.assertEqual([w["w"] for w in words], ["So", "yeah."])
        self.assertEqual(events, [{"s": 0.3, "e": 0.6, "text": "(laughs)"}])


# ---------------------------------------------------------------------------
# S1 / V07-N3: a re-made transcript says when its word texts changed
# ---------------------------------------------------------------------------

WORDS_A = [[" Some", 0.5, 0.8, 0.9], [" times", 0.8, 1.2, 0.9], [" we", 1.3, 1.5, 0.9],
           [" ship.", 1.6, 2.0, 0.9]]
WORDS_B = [[" Sometimes", 0.5, 1.2, 0.9], [" we", 1.3, 1.5, 0.9], [" ship.", 1.6, 2.0, 0.9]]
WORDS_B_LATER = [[" Sometimes", 0.55, 1.2, 0.9], [" we", 1.3, 1.5, 0.9], [" ship.", 1.6, 2.0, 0.9]]


class TextChangedFrom(unittest.TestCase):
    def test_marker_lifecycle(self):
        first, _ = build(asr_of(WORDS_A))
        self.assertNotIn("text_changed_from", first, "a first transcript has no mark")
        second, _ = build(asr_of(WORDS_B), previous=first)
        self.assertNotEqual(second["sha256"], first["sha256"])
        self.assertEqual(second["text_compatible_shas"], [])
        self.assertEqual(second["text_changed_from"], first["sha256"])
        # same text, other times: the old sha stays compatible and the mark is carried
        third, _ = build(asr_of(WORDS_B_LATER), previous=second)
        self.assertEqual(third["text_compatible_shas"], [second["sha256"]])
        self.assertEqual(third["text_changed_from"], first["sha256"])
        # an identical re-run keeps it too
        again, _ = build(asr_of(WORDS_B_LATER), previous=third)
        self.assertEqual(again["sha256"], third["sha256"])
        self.assertEqual(again["text_changed_from"], first["sha256"])
        # the text changes once more: the mark names the transcript it was made from
        fourth, _ = build(asr_of(WORDS_A), previous=again)
        self.assertEqual(fourth["text_changed_from"], again["sha256"])

    def test_same_text_never_marks(self):
        first, _ = build(asr_of(WORDS_B))
        second, _ = build(asr_of(WORDS_B_LATER), previous=first)
        self.assertNotIn("text_changed_from", second)
        self.assertEqual(second["text_compatible_shas"], [first["sha256"]])

    def test_retime_keeps_the_mark(self):
        first, _ = build(asr_of(WORDS_A))
        second, _ = build(asr_of(WORDS_B), previous=first)
        E = track([(0.5, 2.0, -30.0)], dur=6.0)
        new, _moved = T.retime_doc({k: v for k, v in second.items() if k != "_post"}, E)
        self.assertEqual([w["w"] for w in new["words"]], [w["w"] for w in second["words"]])
        self.assertEqual(new["text_changed_from"], first["sha256"])

    def test_carry_text_change_edges(self):
        doc = {"words": [{"w": "a"}], "sha256": "new"}
        self.assertIsNone(T.carry_text_change(doc, None))
        self.assertIsNone(T.carry_text_change(doc, {"words": "broken", "sha256": "x"}))
        self.assertEqual(T.carry_text_change(doc, {"words": [{"w": "b"}], "sha256": "old"}), "old")
        self.assertEqual(T.carry_text_change(
            doc, {"words": [{"w": "b"}], "text_changed_from": "older"}), "older")
        self.assertEqual(T.carry_text_change(
            doc, {"words": [{"w": "a"}], "sha256": "old", "text_changed_from": "older"}), "older")
        self.assertIsNone(T.carry_text_change(doc, {"words": [{"w": "a"}], "sha256": "old"}))

    def test_the_identity_hash_is_untouched(self):
        """Constraint: the mark is not part of what clips.json pins."""
        first, _ = build(asr_of(WORDS_A))
        second, _ = build(asr_of(WORDS_B), previous=first)
        self.assertEqual(second["sha256"], T.transcript_sha(second["words"]))
        self.assertEqual(second["sha256"],
                         env.stable_hash([[w["w"], w["s"], w["e"]] for w in second["words"]]))


@needs_fixtures
class TranscribeCommand(unittest.TestCase):
    """cmd_transcribe with the backend result canned (no energy track: the words are taken as
    the backend wrote them)."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        cls.media = fixtures.make_av_fixture(root / "media", fps="25", dur=6,
                                             flashes=[1.0, 2.5, 4.0], audio_delay=0.2)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.P = fixtures.make_project(Path(self.dir.name), self.media)

    def _run(self, words, **kw):
        base = dict(project=str(self.P), backend="faster-whisper", model=None, language="en",
                    glossary=None, speakers_vtt=None, file=None, clip=None, force=False)
        base.update(kw)
        canned = asr_of(words, backend="faster-whisper", model="tiny", language="en")
        out, err = io.StringIO(), io.StringIO()
        with mock.patch.object(T, "transcribe_wav", return_value=canned), \
                mock.patch.object(T, "_project_energy", return_value=None), \
                contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = T.cmd_transcribe(argparse.Namespace(**base))
        self.assertEqual(rc, 0, err.getvalue())
        return err.getvalue()

    def test_caption_only_clip_is_marked_and_told(self):
        path = self.P / "transcripts" / "c09.json"
        self._run(WORDS_A, file=str(self.media), clip="c09")
        first = env.read_json(path)
        self.assertNotIn("text_changed_from", first)
        err = self._run(WORDS_B, file=str(self.media), clip="c09")
        second = env.read_json(path)
        self.assertEqual([w["w"] for w in second["words"]], ["Sometimes", "we", "ship."])
        self.assertEqual(second["text_changed_from"], first["sha256"])
        self.assertNotIn("plan will flag them", err)
        self.assertIn("transcript of clip c09 changed", err)
        self.assertIn("fix, emphasis and insert", err)
        self.assertIn("--clip c09", err)
        self.assertIn("transcript_sha", err)
        self.assertIn(second["sha256"], err, "the full sha to pin is printed")
        # the same words again: nothing is announced, the mark stays
        err = self._run(WORDS_B, file=str(self.media), clip="c09")
        third = env.read_json(path)
        self.assertEqual(third["sha256"], second["sha256"])
        self.assertEqual(third["text_changed_from"], first["sha256"])
        self.assertNotIn("changed (sha", err)

    def test_project_transcript_log_says_what_plan_checks(self):
        self._run(WORDS_A)
        first = env.read_json(self.P / "transcript.json")
        err = self._run(WORDS_B)
        second = env.read_json(self.P / "transcript.json")
        self.assertEqual(second["text_changed_from"], first["sha256"])
        self.assertNotIn("plan will flag them", err)
        self.assertIn("pins transcript_sha", err)
        self.assertIn("expect anchors", err)
        self.assertIn(second["sha256"], err)
        # times only: the old wording for that case is kept
        err = self._run(WORDS_B_LATER)
        self.assertIn("text unchanged", err)
        self.assertIn("still hold", err)

    # ---- V06-N3: the speakers VTT of another source is not reused

    def _vtt(self):
        vtt = self.P / "speakers.vtt"
        vtt.write_text("WEBVTT\n\n00:00:00.000 --> 00:00:01.250\nAlex: Some times\n\n"
                       "00:00:01.250 --> 00:00:04.000\nSam: we ship.\n", encoding="utf-8")
        return vtt

    def test_speakers_vtt_is_reused_for_the_same_source(self):
        vtt = self._vtt()
        self._run(WORDS_A, speakers_vtt=str(vtt))
        doc = env.read_json(self.P / "transcript.json")
        self.assertEqual([w["sp"] for w in doc["words"]], ["Alex", "Alex", "Sam", "Sam"])
        err = self._run(WORDS_A)
        doc = env.read_json(self.P / "transcript.json")
        self.assertIn("reusing speakers", err)
        self.assertEqual(doc["speakers_vtt"], str(vtt))
        self.assertEqual([w["sp"] for w in doc["words"]], ["Alex", "Alex", "Sam", "Sam"])

    def test_speakers_vtt_of_another_source_is_dropped(self):
        vtt = self._vtt()
        self._run(WORDS_A, speakers_vtt=str(vtt))
        path = self.P / "transcript.json"
        doc = env.read_json(path)
        self.assertEqual(doc["speakers_vtt"], str(vtt))
        env.write_json(path, dict(doc, source_fingerprint="fp-of-the-earlier-export"))
        err = self._run(WORDS_A)
        doc = env.read_json(path)
        self.assertNotIn("reusing speakers", err)
        self.assertIn("not reused", err)
        self.assertIn("--speakers-vtt", err)
        self.assertIn("speakers.vtt", err)
        self.assertIsNone(doc["speakers_vtt"])
        self.assertEqual([w["sp"] for w in doc["words"]], [None] * 4)
        # naming the file again labels the words again
        self._run(WORDS_A, speakers_vtt=str(vtt))
        doc = env.read_json(path)
        self.assertEqual([w["sp"] for w in doc["words"]], ["Alex", "Alex", "Sam", "Sam"])

    def test_speakers_vtt_survives_a_constant_frame_rate_reencode(self):
        vtt = self._vtt()
        self._run(WORDS_A, speakers_vtt=str(vtt))
        path = self.P / "transcript.json"
        env.write_json(path, dict(env.read_json(path), source_fingerprint="fp-before-cfr"))
        sj = self.P / "source.json"
        src = env.read_json(sj)
        ingest = dict(src.get("ingest") or {}, cfr={"from_fingerprint": "fp-before-cfr"})
        env.write_json(sj, dict(src, ingest=ingest))
        err = self._run(WORDS_A)
        self.assertIn("reusing speakers", err)
        self.assertEqual(env.read_json(path)["speakers_vtt"], str(vtt))


class SameMedia(unittest.TestCase):
    def test_rules(self):
        self.assertTrue(T._same_media({"source_fingerprint": "a"}, "a"))
        self.assertFalse(T._same_media({"source_fingerprint": "a"}, "b"))
        self.assertTrue(T._same_media({"source_fingerprint": None}, "b"), "unknown: as before")
        self.assertTrue(T._same_media({"source_fingerprint": "a"}, None))
        self.assertTrue(T._same_media({"source_fingerprint": "a"}, "b",
                                      {"ingest": {"cfr": {"from_fingerprint": "a"}}}))
        self.assertFalse(T._same_media({"source_fingerprint": "a"}, "b",
                                       {"ingest": {"cfr": {"from_fingerprint": "c"}}}))
        self.assertFalse(T._same_media({"source_fingerprint": "a"}, "b", {"ingest": "odd"}))


# ---------------------------------------------------------------------------
# S4: spelled-out scaled amounts
# ---------------------------------------------------------------------------


class ScaledAmounts(unittest.TestCase):
    SPEECH = "I made some money there today"

    def test_a_scaled_amount_is_a_claim_wherever_it_stands(self):
        for surface in ("I MADE ONE MILLION", "I MADE A MILLION", "WE HIT FIVE HUNDRED THOUSAND",
                        "ONE MILLION, GONE", "How I made two million in sales",
                        "HE LOST THREE HUNDRED", "I MADE ONE MILLION."):
            got = numbers(surface, self.SPEECH)
            self.assertEqual(len(got), 1, surface)
            self.assertEqual(got[0]["level"], "FAIL", surface)

    def test_values(self):
        self.assertEqual(numbers("I MADE ONE MILLION", self.SPEECH)[0]["value"], 1e6)
        self.assertEqual(numbers("WE HIT FIVE HUNDRED THOUSAND", self.SPEECH)[0]["value"], 5e5)
        self.assertEqual(numbers("How I made two million in sales", self.SPEECH)[0]["value"], 2e6)

    def test_documented_non_claims_stay_non_claims(self):
        for surface in ("THE ONE", "A NORMAL ONE", "SELL IT FIRST", "I MADE MILLIONS"):
            self.assertEqual(numbers(surface, self.SPEECH), [], surface)

    def test_a_spoken_amount_passes(self):
        self.assertEqual(numbers("I MADE ONE MILLION", "I made one million last year"), [])
        self.assertEqual(numbers("I MADE A MILLION", "we made a million dollars"), [])
        self.assertEqual(numbers("WE HIT FIVE HUNDRED THOUSAND", "we hit 500K in a year"), [])

    def test_earlier_scale_words_are_unchanged(self):
        self.assertEqual(numbers("TWENTY-FIVE GRAND", self.SPEECH)[0]["value"], 25000.0)
        self.assertEqual(numbers("TWENTY-FIVE GRAND", "it cost twenty five thousand"), [])


# ---------------------------------------------------------------------------
# S5: a negation states a zero only for the noun it negates
# ---------------------------------------------------------------------------


class ZeroClaims(unittest.TestCase):
    def test_a_stray_negation_does_not_state_zero(self):
        for surface, speech in (
                ("0 EMPLOYEES", "No, we have ten employees"),
                ("ZERO EMPLOYEES", "No, we have ten employees"),
                ("0 EMPLOYEES", "nobody told me we had employees"),
                ("ZERO EMPLOYEES", "nobody told me we had employees"),
                ("$0 IN ADS", "no I mean we spent a lot on ads"),
                ("0 CLIENTS", "nothing about this was easy we had clients"),
                ("0 CLIENTS", "none of that mattered to the clients"),
                ("0 EMPLOYEES", "oh no no no we have ten employees"),
                ("0% CHURN", "there is no way churn stays low")):
            got = numbers(surface, speech)
            self.assertEqual([c["level"] for c in got], ["FAIL"], (surface, speech))

    def test_a_zero_tied_to_its_noun_passes(self):
        for surface, speech in (
                ("0 EMPLOYEES", "we have no employees"),
                ("ZERO EMPLOYEES", "we have no employees"),
                ("0 EMPLOYEES", "zero employees"),
                ("ZERO EMPLOYEES", "we had no paying employees at all"),
                ("ZERO EMPLOYEES", "we had no full-time employees at all"),
                ("$0 IN ADS", "we spent nothing on ads"),
                ("$0 AD SPEND", "there was no ad spend"),
                ("0 CLIENTS", "none of our clients stayed"),
                ("FOLLOWERS: 0", "I had no followers"),
                ("0 PEOPLE SHOWED UP", "nobody showed up"),
                ("0 WAYS TO WIN", "there is no way to win"),
                ("FROM 0 TO 100K", "from zero to a hundred thousand")):
            self.assertEqual(numbers(surface, speech), [], (surface, speech))

    def test_other_numbers_are_untouched(self):
        self.assertEqual(numbers("10 EMPLOYEES", "No, we have ten employees"), [])
        self.assertEqual(numbers("1 YEAR", "it took a whole year"), [])
        sp = claims.Spoken("No, we have ten employees".split())
        self.assertEqual(sp.values, [10.0])
        self.assertTrue(sp.has_number(10.0))
        self.assertFalse(sp.has_number(0.0, None, ["employees"]))

    def test_found_lines_use_the_same_rule(self):
        surfaces = [("cover headline", "0 EMPLOYEES")]
        self.assertEqual(claims.spoken_claims_found(surfaces, "No, we have ten employees".split()),
                         [])
        self.assertEqual(claims.spoken_claims_found(surfaces, "we have no employees".split()),
                         ["cover headline: 0"])


# ---------------------------------------------------------------------------
# S6: a name matches itself, not its stem
# ---------------------------------------------------------------------------


class NameMatching(unittest.TestCase):
    def test_a_stem_is_not_the_name(self):
        self.assertEqual(names("I met James", "I make jam"), ["James"])
        self.assertEqual(names("What Parker told me", "I parked the car"), ["Parker"])
        self.assertEqual(names("Williams said it", "William said it"), ["Williams"])
        got = unspoken("I met James", "I make jam")
        self.assertEqual([(c["kind"], c["level"]) for c in got], [("name", "WARN")])

    def test_the_name_and_its_possessive_match(self):
        self.assertEqual(names("I met James", "I met James"), [])
        self.assertEqual(names("I met James", "I met him at James's house"), [])
        self.assertEqual(names("We moved to Kubernetes", "we moved to kubernetes last year",
                               glossary=["Kubernetes"]), [])

    def test_found_lines_do_not_list_a_stem_match(self):
        surfaces = [("cover headline", "I met James")]
        self.assertEqual(claims.spoken_claims_found(surfaces, "I make jam".split()), [])
        self.assertEqual(claims.spoken_claims_found(surfaces, "I met James".split()),
                         ["cover headline: James"])

    def test_the_word_path_keeps_its_stems(self):
        """All-caps surfaces cannot show names: ordinary words still match by stem."""
        self.assertEqual(unspoken("WE SELL IT", "we were selling it"), [])
        self.assertEqual(unspoken("I MET JAMES", "I make jam"), [])


# ---------------------------------------------------------------------------
# T2: a leading decimal point
# ---------------------------------------------------------------------------


class LeadingDecimal(unittest.TestCase):
    def test_tokens(self):
        self.assertEqual(tn.norm_tokens(".5%"), ["0.5", "percent"])
        self.assertEqual(tn.norm_tokens("ONLY .5%"), ["only", "0.5", "percent"])
        self.assertEqual(tn.norm_tokens("(.5%)"), ["0.5", "percent"])
        self.assertEqual(tn.norm_tokens("*.5%*"), ["0.5", "percent"])
        self.assertEqual(tn.norm(".5"), "0.5")
        self.assertEqual(tn.norm_tokens("is .5."), ["is", "0.5"])

    def test_what_must_not_change(self):
        self.assertEqual(tn.norm_tokens("...5"), ["5"])
        self.assertEqual(tn.norm_tokens("3 .5"), ["3.5"])
        self.assertEqual(tn.norm_tokens("$2 ,400"), ["2400", "dollars"])
        self.assertEqual(tn.norm_tokens("$.50"), ["0.5", "dollars"])
        self.assertEqual(tn.norm_tokens(".com"), ["com"])
        self.assertEqual(tn.norm_tokens("80 %"), ["80", "percent"])
        self.assertEqual(tn.norm_tokens("5."), ["5"])
        self.assertEqual(tn.norm_tokens("0.5%"), ["0.5", "percent"])
        self.assertEqual(tn.norm_tokens("-5%"), ["5", "percent"])

    def test_glue(self):
        def glue(ws):
            return [g["text"] for g in tn.glue_tokens([{"w": w} for w in ws])]
        self.assertEqual(glue(["only", ".5%"]), ["only", ".5%"])
        self.assertEqual(glue(["3", ".5"]), ["3.5"])
        self.assertEqual(glue(["$3", ".5"]), ["$3.5"])
        self.assertEqual(glue(["$2", ",400"]), ["$2,400"])
        self.assertEqual(glue(["wait", "..."]), ["wait..."])
        self.assertEqual(glue(["80", "%", "."]), ["80%."])
        self.assertTrue(tn._glues_prev(".5"), "without the previous token: the old answer")
        self.assertFalse(tn._glues_prev(".5", "only"))
        self.assertTrue(tn._glues_prev(".5", "3"))
        self.assertTrue(tn._glues_prev(",000", "only"))

    def test_transcript_lines_do_not_weld_a_decimal_to_a_word(self):
        self.assertFalse(T._glued([{"w": "Only"}, {"w": ".5%"}], 0))
        self.assertTrue(T._glued([{"w": "3"}, {"w": ".5"}], 0))
        self.assertTrue(T._glued([{"w": "80"}, {"w": "%"}], 0))

    def test_claims_see_the_decimal(self):
        got = numbers(".5% OF FOUNDERS MAKE IT", "5% of founders make it", "hook card")
        self.assertEqual([(c["level"], c["value"], c["raw"]) for c in got], [("FAIL", 0.5, ".5%")])
        got = numbers("ONLY .5% MAKE IT", "most founders never make it", "hook card")
        self.assertEqual([c["level"] for c in got], ["FAIL"])
        self.assertEqual(numbers(".5% OF FOUNDERS MAKE IT", "0.5% of founders make it"), [])
        self.assertEqual(numbers("0.5% OF FOUNDERS MAKE IT", "only .5% of founders make it"), [])
        self.assertEqual(len(numbers("5% OF FOUNDERS", "only .5% of founders make it")), 1)

    def test_sign_and_range_are_pinned(self):
        self.assertEqual(unspoken("-5% GROWTH", "we were down five percent growth"), [])
        self.assertEqual(numbers("-5% GROWTH", "we were down five percent"), [])
        self.assertEqual(numbers("80-90% OF PEOPLE", "eighty to ninety percent of people"), [])


# ---------------------------------------------------------------------------
# S8: find returns each word range once
# ---------------------------------------------------------------------------


class FindWordRanges(unittest.TestCase):
    def ranges(self, text, phrase, **kw):
        return [(h["a"], h["b"]) for h in find.search(mk(text), phrase, **kw)]

    def test_a_hyphenated_word_is_one_hit(self):
        self.assertEqual(self.ranges("well bye-bye then", "bye"), [(1, 1)])
        self.assertEqual(self.ranges("a known-unknown b", "known"), [(1, 1)])
        self.assertEqual(self.ranges("the win-win is a no-no", "win"), [(1, 1)])

    def test_repeats_do_not_use_up_the_slots(self):
        got = self.ranges("bye-bye-bye-bye-bye-bye-bye-bye and bye", "bye", max_hits=8)
        self.assertEqual(len(got), len(set(got)))
        self.assertIn((2, 2), got)
        self.assertIn((0, 0), got)

    def test_separate_words_are_still_separate_hits(self):
        self.assertEqual(self.ranges("ok bye bye now", "bye"), [(1, 1), (2, 2)])
        self.assertEqual(self.ranges("no-no no no", "no no"), [(0, 0), (1, 2)])

    def test_the_best_hit_of_a_word_is_kept(self):
        hits = find.search(mk("a known-unknown b"), "known")
        self.assertEqual(hits[0]["score"], 1.0)


# ---------------------------------------------------------------------------
# C6 / V04-N1 / V04-N2: one list of written surfaces
# ---------------------------------------------------------------------------

HOOK = {"lines": ["A SMALL", "*GARAGE*"]}


def vclip(**kw):
    clip = {"pieces": [{"words": [0, 10]}], "_eff": {"formats": ["vertical"]}}
    clip.update(kw)
    return clip


class WrittenSurfaces(unittest.TestCase):
    def test_effective_headline_precedence(self):
        eh = claims.effective_headline
        self.assertEqual(eh(vclip(title="T", hook_card=HOOK, cover={"headline": "HEAD"})),
                         ("HEAD", "cover"))
        self.assertEqual(eh(vclip(title="T", hook_card=HOOK)), ("A SMALL *GARAGE*", "hook"))
        self.assertEqual(eh(vclip(title="T", hook_card=None)), ("T", "title"))
        self.assertEqual(eh(vclip(title="T", cover={"headline": ""})), ("T", "title"))
        self.assertEqual(eh(vclip()), ("", ""))
        # the plan's hook card (the one that is drawn) wins over the clip's
        plan = {"hook_card": {"lines": ["FROM", "THE PLAN"]}}
        self.assertEqual(eh(vclip(title="T", hook_card=HOOK), plan), ("FROM THE PLAN", "hook"))
        self.assertEqual(eh(vclip(title="T"), plan), ("FROM THE PLAN", "hook"))
        self.assertEqual(eh(vclip(title="T", cover="odd")), ("T", "title"))

    def test_title_printed_on_the_cover_is_a_cover_headline(self):
        ws = claims.written_surfaces
        self.assertEqual(ws(vclip(title="We made $99M")),
                         [("cover headline (the title)", "We made $99M"), ("title", "We made $99M")])
        self.assertEqual(ws(vclip(title="We made $99M", hook_card=None)),
                         [("cover headline (the title)", "We made $99M"), ("title", "We made $99M")])
        self.assertEqual(ws({"title": "No eff", "pieces": []}),
                         [("cover headline (the title)", "No eff"), ("title", "No eff")],
                         "vertical is the default format")

    def test_other_headline_sources(self):
        ws = claims.written_surfaces
        self.assertEqual(ws(vclip(title="T", hook_card=HOOK)),
                         [("hook card", "A SMALL *GARAGE*"), ("title", "T")])
        self.assertEqual(ws(vclip(title="T", cover={"headline": "HEAD"})),
                         [("cover headline", "HEAD"), ("title", "T")])
        self.assertEqual(ws(vclip(title="T", hook_card=HOOK, cover={"headline": "HEAD"},
                                  post={"title": "PT", "caption": "PC"})),
                         [("hook card", "A SMALL *GARAGE*"), ("cover headline", "HEAD"),
                          ("title", "T"), ("post title", "PT"), ("post caption", "PC")])
        self.assertEqual(ws(vclip()), [])
        self.assertEqual(ws(vclip(post="odd", cover=None, hook_card="odd")), [])

    def test_landscape_only_clip_has_no_cover_surface(self):
        clip = {"title": "Twenty minutes later", "hook_card": None,
                "_eff": {"formats": ["landscape"]}, "pieces": [{"words": [0, 5]}]}
        self.assertEqual(claims.written_surfaces(clip), [("title", "Twenty minutes later")])
        both = dict(clip, _eff={"formats": ["vertical", "landscape"]})
        self.assertEqual([lb for lb, _t in claims.written_surfaces(both)],
                         ["cover headline (the title)", "title"])

    def test_chapter_titles_of_a_landscape_clip(self):
        pieces = [{"words": [0, 5], "chapter": "The $5M mistake"}, {"words": [6, 9]},
                  {"words": [10, 12], "chapter": "What changed"}]
        clip = {"title": "T", "hook_card": HOOK, "_eff": {"formats": ["landscape"]},
                "pieces": pieces}
        self.assertEqual(claims.written_surfaces(clip),
                         [("hook card", "A SMALL *GARAGE*"), ("title", "T"),
                          ("chapter title", "The $5M mistake"), ("chapter title", "What changed")])
        vertical = dict(clip, _eff={"formats": ["vertical"]})
        self.assertNotIn("chapter title", [lb for lb, _t in claims.written_surfaces(vertical)])
        got = claims.unspoken_claims(claims.written_surfaces(clip),
                                     "a small garage and what changed".split())
        self.assertEqual([(c["surface"], c["kind"], c["level"]) for c in got
                          if c["kind"] == "number"], [("chapter title", "number", "FAIL")])

    def test_plan_hook_card_is_the_one_listed(self):
        plan = {"hook_card": {"lines": ["FROM", "THE PLAN"]}}
        self.assertEqual(claims.written_surfaces(vclip(title="T", hook_card=HOOK), plan)[0],
                         ("hook card", "FROM THE PLAN"))

    def test_labels_are_listed(self):
        for label in ("hook card", "cover headline", "cover headline (the title)", "title",
                      "post title", "post caption", "chapter title"):
            self.assertIn(label, claims.LABELS)
        self.assertEqual(claims.TITLE_ON_COVER, "cover headline (the title)")

    def test_what_is_drawn_is_what_is_checked(self):
        """Every headline the cover prints for a vertical clip is among the checked texts."""
        plan = {"hook_card": {"lines": ["FROM", "THE PLAN"]}}
        matrix = [vclip(title="T", cover={"headline": "HEAD"}), vclip(title="T", hook_card=HOOK),
                  vclip(title="T"), vclip(title="T", hook_card=None), vclip(),
                  vclip(title="T", hook_card=HOOK, cover={"headline": "HEAD"})]
        for clip in matrix:
            for pl in (None, plan):
                head, origin = claims.effective_headline(clip, pl)
                texts = [t for _lb, t in claims.written_surfaces(clip, pl)]
                if head:
                    self.assertIn(head, texts, (clip, pl))
                else:
                    self.assertEqual(origin, "")

    def test_cover_drawer_agrees(self):
        try:
            from hookcutter import cover
        except Exception as exc:  # noqa: BLE001 - optional imaging dependency
            self.skipTest("cover module unavailable: %s" % exc)
        plan = {"hook_card": {"lines": ["FROM", "THE PLAN"]}}
        for clip in (vclip(title="T", cover={"headline": "HEAD"}), vclip(title="T", hook_card=HOOK),
                     vclip(title="T"), vclip(title="T", hook_card=None), vclip()):
            for pl in (None, plan):
                drawn = cover.headline_for(clip, pl)
                self.assertEqual(drawn, claims.effective_headline(clip, pl)[0], (clip, pl))
                if drawn:
                    self.assertIn(drawn, [t for _lb, t in claims.written_surfaces(clip, pl)])


class TitleClaims(unittest.TestCase):
    SPEECH = "we built it in a small garage"

    def check(self, clip, speech=None, **kw):
        return claims.unspoken_claims(claims.written_surfaces(clip), (speech or self.SPEECH).split(),
                                      **kw)

    def test_a_number_in_the_title_fails_once(self):
        got = self.check(vclip(title="We made $99M"))
        fails = [c for c in got if c["level"] == "FAIL"]
        self.assertEqual(len(fails), 1, got)
        self.assertEqual(fails[0]["surface"], "cover headline (the title)")
        self.assertIn("$99M", fails[0]["msg"])

    def test_the_title_is_checked_beside_an_honest_hook_card(self):
        got = self.check(vclip(title="We made $99M", hook_card=HOOK))
        self.assertEqual([(c["surface"], c["level"]) for c in got if c["kind"] == "number"],
                         [("title", "FAIL")])
        self.assertEqual(self.check(vclip(title="Built in a garage", hook_card=HOOK)), [])

    def test_an_honest_title_on_the_cover_passes(self):
        self.assertEqual(self.check(vclip(title="Built in a garage")), [])

    def test_speaker_name_is_fine_in_a_title_but_not_on_a_cover(self):
        title = "Why Dana built it in a garage"
        beside_hook = self.check(vclip(title=title, hook_card=HOOK), exempt_names=["Dana"])
        self.assertEqual(beside_hook, [], "a title may credit the speaker, like a post title")
        on_cover = self.check(vclip(title=title), exempt_names=["Dana"])
        self.assertEqual([(c["surface"], c["kind"], c["raw"]) for c in on_cover],
                         [("cover headline (the title)", "name", "Dana")])
        no_exemption = self.check(vclip(title=title, hook_card=HOOK))
        self.assertEqual([(c["surface"], c["raw"]) for c in no_exemption], [("title", "Dana")])

    def test_post_surfaces_keep_their_exemption(self):
        clip = vclip(hook_card=HOOK, post={"title": "Dana on garages", "caption": "Dana built it"})
        self.assertEqual(self.check(clip, exempt_names=["Dana"]), [])
        self.assertEqual(sorted({c["surface"] for c in self.check(clip)}),
                         ["post caption", "post title"])

    def test_found_lines_list_a_shared_text_once(self):
        surfaces = claims.written_surfaces(vclip(title="We made 5 in a garage"))
        found = claims.spoken_claims_found(surfaces, "we made five in a garage".split())
        self.assertEqual(found, ["cover headline (the title): 5"])


if __name__ == "__main__":
    unittest.main()
