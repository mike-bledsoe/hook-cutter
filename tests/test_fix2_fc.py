"""Transcription fixes from the first long production run: short skipped phrases that run
straight out of a word, false starts folded into one word, word starts stamped onto a click or
under someone else's laugh, the two new transcript flags and how transcript.md shows them."""

from __future__ import annotations

import argparse
import contextlib
import io
import sys
import tempfile
import types
import unittest
import wave
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts"))
sys.path.insert(0, str(HERE))

from hookcutter import align, env, find, transcribe as T  # noqa: E402
from hookcutter.audio import EnergyTrack  # noqa: E402

try:
    import fixtures  # noqa: E402
except Exception:  # noqa: BLE001 - media tests skip when the shared fixtures are unavailable
    fixtures = None

needs_fixtures = unittest.skipIf(fixtures is None, "tests/fixtures.py unavailable")

SR = 16000
FLOOR = -70.0
SPEECH = -28.0


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def track(segments=(), dur=30.0, floor=FLOOR, hop=0.01):
    """EnergyTrack at `floor` dB with [(s, e, dB)] segments (src clock, t0 = 0)."""
    db = np.full(int(round(dur / hop)), floor, dtype=np.float64)
    for s, e, level in segments:
        db[int(round(s / hop)):int(round(e / hop))] = level
    return EnergyTrack(db, hop=hop, t0=0.0)


def W(text, s, e, p=0.95, sp=None):
    return {"w": text, "s": s, "e": e, "p": p, "sp": sp}


def quiet(fn, *a, **kw):
    with contextlib.redirect_stderr(io.StringIO()):
        return fn(*a, **kw)


def asr_result(words):
    return {"backend": "faster-whisper", "model": "tiny", "backend_version": "0",
            "language": "en", "language_requested": "en", "glossary": [], "prompt": None,
            "options": {}, "words": [list(w) for w in words]}


class Listen(object):
    """Stands in for transcribe.Relisten: canned raw words per call, on the src clock."""

    can_retry = False

    def __init__(self, heard=()):
        self.heard = [list(w) for w in heard]
        self.calls = []

    def __call__(self, a, b, plain=False):
        self.calls.append((round(a, 3), round(b, 3)))
        return [w for w in self.heard if a - 1e-6 <= w[1] and w[2] <= b + 1e-6]


def kinds(doc, kind):
    return [f for f in doc["flags"] if f["kind"] == kind]


def build(words, E, listen=None, **kw):
    return quiet(T.build_transcript, asr_result(words), offset=0.0, media="clip.mp4",
                 fingerprint="f" * 8, energy=E, listen=listen, duration=E.end, **kw)


# ---------------------------------------------------------------------------
# item 2: a short phrase skipped right after a word
# ---------------------------------------------------------------------------

# "in this moment." ends at 6.0; 0.74 s of speech follows with no dip and no word; then a pause.
SPOKEN = [(4.0, 6.0, SPEECH), (6.0, 6.74, SPEECH - 4.0), (8.0, 10.0, SPEECH)]
RAW = [(" in", 4.0, 4.6, 0.99), (" this", 4.6, 5.2, 0.99), (" moment.", 5.2, 6.0, 0.99),
       (" And", 8.0, 8.6, 0.99), (" after", 8.6, 10.0, 0.99)]


class TestShortSkippedPhrase(unittest.TestCase):
    def test_speech_running_out_of_a_word_is_a_hole(self):
        E = track(SPOKEN)
        words, _ = T.normalize_words(RAW)
        self.assertEqual(align.uncovered_voiced(E, words, 0.0, E.end), [])     # under 0.8 s
        holes = T.find_holes(words, E)
        self.assertEqual(len(holes), 1)
        self.assertAlmostEqual(holes[0][0], 6.05, places=2)
        self.assertAlmostEqual(holes[0][1], 6.74, places=2)

    def test_relisten_recovers_the_words(self):
        E = track(SPOKEN)
        listen = Listen([(" I", 6.05, 6.2, 0.9), (" use", 6.2, 6.4, 0.9), (" it.", 6.4, 6.72, 0.9)])
        doc = build(RAW, E, listen)
        self.assertEqual(len(listen.calls), 1)
        self.assertEqual([w["w"] for w in doc["words"]],
                         ["in", "this", "moment.", "I", "use", "it.", "And", "after"])
        self.assertEqual(kinds(doc, "unworded_audio"), [])
        self.assertEqual(len(kinds(doc, "respliced")), 1)

    def test_flagged_when_the_relisten_hears_nothing(self):
        E = track(SPOKEN)
        doc = build(RAW, E, Listen())
        flags = kinds(doc, "unworded_audio")
        self.assertEqual(len(flags), 1)
        self.assertAlmostEqual(flags[0]["s"], 6.05, places=2)
        self.assertAlmostEqual(flags[0]["e"], 6.74, places=2)
        self.assertEqual((flags[0]["a"], flags[0]["b"]), (2, 3))
        self.assertIn("0.7 s of sound with no words", T.render_md(doc))

    def test_a_fading_tail_or_a_breath_is_not_a_hole(self):
        # the same 0.74 s, but 20 dB under the speech level: the word's own tail, a breath
        E = track([(4.0, 6.0, SPEECH), (6.0, 6.74, SPEECH - 20.0), (8.0, 10.0, SPEECH)])
        words, _ = T.normalize_words(RAW)
        self.assertEqual(T.find_holes(words, E), [])
        self.assertEqual(kinds(build(RAW, E, Listen()), "unworded_audio"), [])

    def test_a_short_sound_on_its_own_is_not_a_hole(self):
        # 0.6 s at speech level between two pauses (a cough, a laugh): still needs 0.8 s
        E = track([(4.0, 6.0, SPEECH), (6.6, 7.2, SPEECH), (8.0, 10.0, SPEECH)])
        words, _ = T.normalize_words(RAW)
        self.assertEqual(T.find_holes(words, E), [])

    def test_speech_running_into_a_word_is_a_hole(self):
        E = track([(4.0, 6.0, SPEECH), (7.3, 8.0, SPEECH - 3.0), (8.0, 10.0, SPEECH)])
        words, _ = T.normalize_words(RAW)
        holes = T.find_holes(words, E)
        self.assertEqual(len(holes), 1)
        self.assertAlmostEqual(holes[0][0], 7.3, places=2)
        self.assertAlmostEqual(holes[0][1], 7.95, places=2)

    def test_under_half_a_second_is_left_alone(self):
        E = track([(4.0, 6.0, SPEECH), (6.0, 6.5, SPEECH), (8.0, 10.0, SPEECH)])
        words, _ = T.normalize_words(RAW)
        self.assertEqual(T.find_holes(words, E), [])        # 0.45 s past the word's pad

    def test_default_query_is_unchanged(self):
        # callers that do not ask for abutting stretches (the plan's hole lint) see what they did
        E = track(SPOKEN)
        words, _ = T.normalize_words(RAW)
        self.assertEqual(align.uncovered_voiced(E, words, 0.0, E.end, min_run=0.8), [])
        self.assertEqual(len(align.uncovered_voiced(E, words, 0.0, E.end, min_run=0.5)), 1)


# ---------------------------------------------------------------------------
# item 1: a false start folded into one long, unsure word
# ---------------------------------------------------------------------------

# "And after a w... after about a decade," transcribed as "And after about a decade,": the
# abandoned "a w" and the real "after" both sit inside the span of "about" (5.0-5.66), with a
# 0.12 s dip between them.
FOLDED = [(4.0, 5.2, SPEECH), (5.2, 5.32, SPEECH - 22.0), (5.32, 6.5, SPEECH)]
FOLDED_RAW = [(" And", 4.0, 4.4, 0.7), (" after", 4.4, 5.0, 0.99), (" about", 5.0, 5.66, 0.61),
              (" a", 5.66, 5.9, 1.0), (" decade,", 5.9, 6.5, 1.0)]
LEAD = [(" And", 4.0, 4.4, 0.9), (" after", 4.4, 5.0, 0.9)]


def folded_words(p=0.61):
    raw = [list(w) for w in FOLDED_RAW]
    raw[2][3] = p
    return T.index_words(T.normalize_words(raw)[0])


class TestBurstDips(unittest.TestCase):
    def test_two_bursts_and_a_dip(self):
        self.assertEqual(align.burst_dips(track(FOLDED), 5.0, 5.66), [(5.2, 5.32)])

    def test_a_stop_closure_is_not_a_dip(self):
        E = track([(4.0, 5.2, SPEECH), (5.2, 5.26, SPEECH - 30.0), (5.26, 6.5, SPEECH)])
        self.assertEqual(align.burst_dips(E, 5.0, 5.66), [])      # 60 ms: one word

    def test_a_shallow_dip_is_not_a_dip(self):
        E = track([(4.0, 5.2, SPEECH), (5.2, 5.32, SPEECH - 12.0), (5.32, 6.5, SPEECH)])
        self.assertEqual(align.burst_dips(E, 5.0, 5.66), [])

    def test_needs_sound_on_both_sides(self):
        E = track([(5.0, 5.3, SPEECH), (5.3, 5.66, SPEECH - 30.0)])      # a word and its decay
        self.assertEqual(align.burst_dips(E, 5.0, 5.66), [])
        E = track([(5.0, 5.03, SPEECH), (5.03, 5.2, SPEECH - 30.0), (5.2, 5.66, SPEECH)])
        self.assertEqual(align.burst_dips(E, 5.0, 5.66), [])             # a 30 ms click first


class TestFalseStart(unittest.TestCase):
    def test_candidate_is_the_unsure_word_with_two_bursts(self):
        E = track(FOLDED)
        self.assertEqual(T.false_start_candidates(folded_words(), E),
                         [{"k": 2, "dip": (5.2, 5.32)}])
        self.assertEqual(T.false_start_candidates(folded_words(p=0.95), E), [])   # a sure word
        self.assertEqual(T.false_start_candidates(folded_words(), track([(4.0, 6.5, SPEECH)])), [])

    def test_flagged_when_the_lead_in_holds_more_than_the_transcript(self):
        E = track(FOLDED)
        words = folded_words()
        listen = Listen(LEAD + [(" a", 5.0, 5.2, 0.5)])
        flags = T.false_starts(words, E, listen)
        self.assertEqual(flags, [{"a": 2, "b": 2, "kind": "false_start", "s": 5.0, "e": 5.32,
                                  "heard": "a"}])
        # one re-listen: from the first word (after the pause) to the middle of the dip
        self.assertEqual(listen.calls, [(3.9, 5.26)])

    def test_not_flagged_when_the_lead_in_is_what_the_transcript_has(self):
        # the first burst is the previous word's tail, a breath or a laugh: nothing more is heard
        E = track(FOLDED)
        self.assertEqual(T.false_starts(folded_words(), E, Listen(LEAD)), [])
        # ... also when the last word before it is heard a little differently
        heard = [(" And", 4.0, 4.4, 0.9), (" laughter", 4.4, 5.0, 0.9)]
        self.assertEqual(T.false_starts(folded_words(), E, Listen(heard)), [])

    def test_not_flagged_when_the_first_burst_is_the_word_itself(self):
        E = track(FOLDED)
        listen = Listen(LEAD + [(" about", 5.0, 5.2, 0.9)])
        self.assertEqual(T.false_starts(folded_words(), E, listen), [])

    def test_not_flagged_for_a_filler(self):
        E = track(FOLDED)
        listen = Listen(LEAD + [(" um,", 5.0, 5.2, 0.9)])
        self.assertEqual(T.false_starts(folded_words(), E, listen), [])

    def test_flagged_without_a_relisten(self):
        E = track(FOLDED)
        want = [{"a": 2, "b": 2, "kind": "false_start", "s": 5.0, "e": 5.32}]
        self.assertEqual(T.false_starts(folded_words(), E), want)

        def broken(a, b, plain=False):       # a failed re-listen answers None
            return None
        self.assertEqual(T.false_starts(folded_words(), E, broken), want)
        # ... as does one that hears nothing the transcript has
        other = Listen([(" Thanks", 4.0, 4.4, 0.9), (" everyone", 4.4, 5.0, 0.9)])
        self.assertEqual(T.false_starts(folded_words(), E, other), want)

    def test_heard_is_capped_by_the_room_before_the_dip(self):
        # a short slice can end on a made-up tail: 0.2 s of sound holds two words at most
        E = track(FOLDED)
        tail = [(" a", 5.0, 5.05, 0.5), (" while,", 5.05, 5.1, 0.3), (" I", 5.1, 5.12, 0.2),
                (" use", 5.12, 5.16, 0.2), (" it.", 5.16, 5.2, 0.2)]
        flags = T.false_starts(folded_words(), E, Listen(LEAD + tail))
        self.assertEqual(flags[0]["heard"], "a while,")

    def test_lead_in_starts_after_the_last_pause(self):
        words = [W("x", 0.0, 0.4), W("y", 0.5, 0.9), W("So", 2.0, 2.3), W("we", 2.3, 2.6),
                 W("go", 2.6, 2.9), W("there", 2.9, 3.4), W("and", 3.4, 3.6), W("then", 3.6, 4.4)]
        self.assertEqual(T._lead_in_start(words, 7), 2)       # "So" follows a 1.1 s pause
        self.assertEqual(T._lead_in_start(words, 2), 0)       # everything within 3 s
        run = [W("w%d" % i, i * 0.3, i * 0.3 + 0.3) for i in range(30)]
        self.assertEqual(T._lead_in_start(run, 20), 16)       # no pause: 1.2 s back
        self.assertEqual(T._lead_in_start(run, 0), 0)

    def test_transcript_flags_it_and_never_rewrites_the_words(self):
        E = track(FOLDED)
        doc = build(FOLDED_RAW, E, Listen(LEAD + [(" a", 5.0, 5.2, 0.5)]))
        self.assertEqual([w["w"] for w in doc["words"]], ["And", "after", "about", "a", "decade,"])
        self.assertEqual([(w["s"], w["e"]) for w in doc["words"]],
                         [(4.0, 4.4), (4.4, 5.0), (5.0, 5.66), (5.66, 5.9), (5.9, 6.5)])
        self.assertEqual(kinds(doc, "false_start"),
                         [{"a": 2, "b": 2, "kind": "false_start", "s": 5.0, "e": 5.32,
                           "heard": "a"}])
        self.assertEqual(kinds(doc, "respliced") + kinds(doc, "unworded_audio"), [])

    def test_no_energy_no_flags(self):
        doc = quiet(T.build_transcript, asr_result(FOLDED_RAW), offset=0.0, media="clip.mp4",
                    fingerprint="f" * 8)
        self.assertEqual(kinds(doc, "false_start"), [])


class TestRepeatedAttempts(unittest.TestCase):
    def test_repeat_with_an_overlong_word(self):
        # "I was, I was worn out": the second "was" lasts 0.7 s (3 letters: 0.54 s at most)
        words = [W("I", 1.0, 1.1), W("was,", 1.1, 1.3), W("I", 1.5, 1.6), W("was", 1.6, 2.3),
                 W("worn", 2.3, 2.6), W("out.", 2.6, 2.9)]
        self.assertEqual(T.repeated_attempts(words),
                         [{"a": 0, "b": 3, "kind": "false_start", "s": 1.0, "e": 1.5}])
        self.assertEqual(T.false_starts(words, track([(1.0, 2.9, SPEECH)])),
                         [{"a": 0, "b": 3, "kind": "false_start", "s": 1.0, "e": 1.5}])

    def test_a_marked_restart_is_flagged_even_at_normal_length(self):
        # Changed in 0.2: production runs showed readers miss these, so the comma that ends the
        # first attempt is enough (L15 and the by-ear checklist then name it)
        words = [W("I", 1.0, 1.1), W("was,", 1.1, 1.3), W("I", 1.5, 1.6), W("was", 1.6, 1.8),
                 W("worn", 1.8, 2.1), W("out.", 2.1, 2.4)]
        self.assertEqual([(f["a"], f["b"]) for f in T.repeated_attempts(words)], [(0, 3)])

    def test_an_unmarked_repeat_of_content_words_is_left_alone(self):
        words = [W("we", 1.0, 1.1), W("grew", 1.1, 1.3), W("we", 1.5, 1.6), W("grew", 1.6, 1.8),
                 W("fast.", 1.8, 2.1), W("that", 2.2, 2.4), W("that", 2.4, 2.6), W("works.", 2.6, 2.9)]
        self.assertEqual(T.repeated_attempts(words), [])

    def test_a_doubled_function_word_is_a_stumble(self):
        words = [W("into", 1.0, 1.2), W("the", 1.2, 1.3), W("the", 1.3, 1.4), W("room.", 1.4, 1.8)]
        self.assertEqual([(f["a"], f["b"]) for f in T.repeated_attempts(words)], [(1, 2)])

    def test_emphasis_and_skipped_ranges(self):
        words = [W("no,", 1.0, 1.9), W("no,", 2.0, 2.2), W("the", 3.0, 3.9), W("the", 4.0, 4.2)]
        self.assertEqual([(f["a"], f["b"]) for f in T.repeated_attempts(words)], [(2, 3)])
        self.assertEqual(T.repeated_attempts(words, skip=[(3, 3)]), [])

    def test_long_words_without_a_repeat_are_not_flagged(self):
        words = [W("So", 1.0, 1.9), W("anyway", 2.0, 3.5), W("yes", 3.6, 4.6)]
        self.assertEqual(T.repeated_attempts(words), [])


# ---------------------------------------------------------------------------
# item 3: starts stamped on a click, or under someone else's laugh
# ---------------------------------------------------------------------------


def _band_noise(rng, n, lo, hi):
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1.0 / SR)
    spec[(f < lo) | (f > hi)] = 0.0
    return np.fft.irfft(spec, n)


def synth(parts, dur, seed=7):
    """16 kHz mono samples: a noise floor plus [(s, e, kind, dB)] parts. Kinds: "voice" (a
    120 Hz harmonic complex), "laugh" (a low noise band), "hiss" (a high noise band, an "sh")."""
    rng = np.random.default_rng(seed)
    n = int(round(dur * SR))
    x = rng.standard_normal(n) * 10 ** (FLOOR / 20.0)
    t = np.arange(n) / float(SR)
    for s, e, kind, level in parts:
        i0, i1 = int(round(s * SR)), int(round(e * SR))
        if kind == "voice":
            seg = sum(np.sin(2 * np.pi * 120.0 * h * t[i0:i1]) / h for h in range(1, 20))
        elif kind == "laugh":
            seg = _band_noise(rng, i1 - i0, 150.0, 500.0)
        else:
            seg = _band_noise(rng, i1 - i0, 2500.0, 5000.0)
        x[i0:i1] += seg / np.sqrt(np.mean(seg ** 2)) * 10 ** (level / 20.0)
    return x.astype(np.float32)


def energy_of(x, hop=160):
    frames = x[:len(x) // hop * hop].reshape(-1, hop).astype(np.float64)
    return EnergyTrack(10.0 * np.log10(np.mean(frames ** 2, axis=1) + 1e-12), hop=0.01, t0=0.0)


def write_wav(path, x):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((np.clip(x, -1.0, 1.0) * 32767.0).astype("<i2").tobytes())


# A click 0.2 s before the voice: "you" is spoken 2.20-2.60, the recogniser stamped it from
# the previous word's end (1.40), and a 30 ms lip click sits at 2.00 (the 0.17 s of quiet after
# it is too short to count as a pause).
CLICK = [(1.0, 1.4, SPEECH), (2.0, 2.03, SPEECH - 22.0), (2.2, 2.6, SPEECH)]
CLICKED = [(1.0, 1.4, SPEECH), (2.0, 2.03, SPEECH - 22.0), (2.2, 2.4, SPEECH)]


class TestClickIsNotAnOnset(unittest.TestCase):
    def test_start_moves_to_the_voice_not_the_click(self):
        E = track(CLICK, dur=10.0)
        fixed, log = align.repair_times([W("pick", 1.0, 1.4), W("you", 1.4, 2.6)], E)
        self.assertAlmostEqual(fixed[1]["s"], 2.2, places=2)
        self.assertEqual(fixed[1]["e"], 2.6)
        self.assertEqual([(e["i"], e["tag"]) for e in log], [(1, "s")])

    def test_a_start_already_on_a_click_moves_too(self):
        # what `hc transcribe --retime` sees in a transcript made before this fix: the start
        # sits on the click
        E = track(CLICKED, dur=10.0)
        fixed, log = align.repair_times([W("pick", 1.0, 1.4), W("you", 2.0, 2.4)], E)
        self.assertAlmostEqual(fixed[1]["s"], 2.2, places=2)
        self.assertEqual(fixed[1]["e"], 2.4)
        self.assertEqual([(e["i"], e["tag"]) for e in log], [(1, "s!")])
        again, log = align.repair_times(fixed, E)
        self.assertEqual(log, [])

    def test_a_stop_burst_before_its_vowel_is_an_onset(self):
        # 30 ms burst, 20 ms closure, vowel: the word starts on the burst
        E = track([(1.0, 1.4, SPEECH), (2.0, 2.03, SPEECH - 10.0), (2.05, 2.6, SPEECH)], dur=10.0)
        fixed, _ = align.repair_times([W("pick", 1.0, 1.4), W("two", 1.4, 2.6)], E)
        self.assertAlmostEqual(fixed[1]["s"], 2.0, places=2)
        fixed, log = align.repair_times([W("pick", 1.0, 1.4), W("two", 2.0, 2.6)], E)
        self.assertEqual(log, [])

    def test_a_short_sound_is_used_when_nothing_longer_is_in_reach(self):
        # the whole word is 40 ms of sound: it is still where the word is
        E = track([(1.0, 1.4, SPEECH), (2.0, 2.04, SPEECH)], dur=10.0)
        fixed, _ = align.repair_times([W("pick", 1.0, 1.4), W("a", 1.4, 2.04)], E)
        self.assertAlmostEqual(fixed[1]["s"], 2.0, places=2)

    def test_sustained_runs(self):
        runs = [(1.0, 1.03), (1.2, 1.24), (1.25, 1.4), (2.0, 2.05), (2.3, 2.36), (3.0, 3.03)]
        self.assertEqual(align.sustained_runs(runs), [(1.2, 1.24), (1.25, 1.4), (2.3, 2.36)])
        self.assertEqual(align.sustained_runs([]), [])

    def test_retime_doc_moves_a_start_off_a_click(self):
        E = track(CLICKED, dur=10.0)
        words = T.index_words([W("pick", 1.0, 1.4), W("you", 2.0, 2.4)])
        doc = {"words": words, "flags": [], "timing_repairs": [], "sha256": "x", "text_sha": "y"}
        new, moved = T.retime_doc(doc, E)
        self.assertEqual(moved, 1)
        self.assertAlmostEqual(new["words"][1]["s"], 2.2, places=2)
        self.assertEqual([w["w"] for w in new["words"]], ["pick", "you"])
        self.assertEqual([(r["i"], r["tag"]) for r in new["timing_repairs"]], [(1, "s!")])


# Someone laughs from 1.9 to 2.6; "She was" is spoken 2.6-3.3 ("sh" 2.6-2.7, then voice), and
# the recogniser stamped "She" at 2.3, inside the laugh.
LAUGH = [(1.0, 1.5, "voice", SPEECH), (1.9, 2.6, "laugh", SPEECH - 2.0),
         (2.6, 2.7, "hiss", SPEECH - 4.0), (2.7, 3.3, "voice", SPEECH)]
LAUGH_RAW = [(" Nah.", 1.0, 1.5, 0.99), (" She", 2.3, 2.9, 0.97), (" was", 2.9, 3.3, 0.99)]


class TestStartUnderALaugh(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.x = synth(LAUGH, 5.0)
        cls.E = energy_of(cls.x)
        cls.B = align.BandShapes.from_samples(cls.x)
        cls.words = T.index_words(T.normalize_words(LAUGH_RAW)[0])

    def test_band_shapes_follow_the_energy_frames(self):
        self.assertEqual(self.B.n, self.E.n)
        self.assertEqual(self.B.S.shape[1], 6)
        # loudness is removed: every frame's bands average to zero
        self.assertLess(float(np.abs(self.B.S.mean(axis=1)).max()), 1e-3)
        with tempfile.TemporaryDirectory() as td:
            write_wav(Path(td) / "a.wav", self.x)
            from_file = align.BandShapes.from_wav(Path(td) / "a.wav")
            ref = align.BandShapes.from_samples(T.load_wav(Path(td) / "a.wav"))
        self.assertEqual(from_file.S.shape, self.B.S.shape)
        self.assertLess(float(np.abs(from_file.S - ref.S).max()), 1e-4)

    def test_the_laugh_runs_into_the_word(self):
        self.assertEqual(align.sound_into_starts(self.E, self.words), [(1, (1.9, 2.25))])

    def test_start_moves_to_where_the_sound_changes(self):
        on = align.spectral_onset(self.B, self.E, (1.9, 2.25), 2.3, 2.9)
        self.assertAlmostEqual(on, 2.6, delta=0.021)
        fixed, log = align.repair_overlapped(self.words, self.E, self.B)
        self.assertAlmostEqual(fixed[1]["s"], 2.6, delta=0.021)
        self.assertEqual(fixed[1]["e"], 2.9)
        self.assertEqual([(e["i"], e["tag"], e["s0"]) for e in log], [(1, "sp", 2.3)])
        self.assertEqual([w["s"] for w in fixed[:1] + fixed[2:]], [1.0, 2.9])
        # a second pass finds it anchored: nothing moves, nothing is uncertain
        again, log = align.repair_overlapped(fixed, self.E, self.B)
        self.assertEqual(log, [])
        self.assertEqual(align.uncertain_starts(fixed, self.E, self.B), [])

    def test_uncertain_until_anchored(self):
        self.assertEqual(align.uncertain_starts(self.words, self.E, self.B), [1])
        self.assertEqual(align.uncertain_starts(self.words, self.E), [1])     # no band shapes
        self.assertEqual(T.timing_flags(self.words, self.E),
                         [{"a": 1, "b": 1, "kind": "timing_uncertain"}])

    def test_transcript_moves_the_start_and_keeps_the_words(self):
        doc = build(LAUGH_RAW, self.E, Listen(), shapes=self.B)
        self.assertEqual([w["w"] for w in doc["words"]], ["Nah.", "She", "was"])
        self.assertAlmostEqual(doc["words"][1]["s"], 2.6, delta=0.021)
        self.assertEqual([(r["i"], r["tag"], r["s0"]) for r in doc["timing_repairs"]],
                         [(1, "sp", 2.3)])
        self.assertEqual(kinds(doc, "timing_uncertain"), [])
        self.assertEqual(len(kinds(doc, "unworded_audio")), 1)       # the laugh itself

    def test_without_band_shapes_it_is_flagged_not_moved(self):
        doc = build(LAUGH_RAW, self.E, Listen())
        self.assertEqual(doc["words"][1]["s"], 2.3)
        self.assertEqual(kinds(doc, "timing_uncertain"),
                         [{"a": 1, "b": 1, "kind": "timing_uncertain"}])

    def test_a_word_the_same_colour_as_the_laugh_is_flagged_not_moved(self):
        # the laugh runs to 2.9 with no change of sound inside the stamped word
        x = synth([(1.0, 1.5, "voice", SPEECH), (1.9, 2.9, "laugh", SPEECH - 2.0),
                   (2.9, 3.3, "voice", SPEECH)], 5.0)
        E, B = energy_of(x), align.BandShapes.from_samples(x)
        self.assertIsNone(align.spectral_onset(B, E, (1.9, 2.25), 2.3, 2.9))
        doc = build(LAUGH_RAW, E, Listen(), shapes=B)
        self.assertEqual(doc["words"][1]["s"], 2.3)
        self.assertEqual(kinds(doc, "timing_uncertain"),
                         [{"a": 1, "b": 1, "kind": "timing_uncertain"}])

    def test_a_change_at_the_stamped_start_confirms_it(self):
        x = synth([(1.0, 1.5, "voice", SPEECH), (1.9, 2.3, "laugh", SPEECH - 2.0),
                   (2.3, 2.4, "hiss", SPEECH - 4.0), (2.4, 3.3, "voice", SPEECH)], 5.0)
        E, B = energy_of(x), align.BandShapes.from_samples(x)
        doc = build(LAUGH_RAW, E, Listen(), shapes=B)
        self.assertEqual(doc["words"][1]["s"], 2.3)
        self.assertEqual(doc["timing_repairs"], [])
        self.assertEqual(kinds(doc, "timing_uncertain"), [])

    def test_a_breath_before_a_word_is_not_a_laugh(self):
        # the same sound 22 dB under the speech level: the word's own onset is clear
        x = synth([(1.0, 1.5, "voice", SPEECH), (1.9, 2.6, "laugh", SPEECH - 22.0),
                   (2.6, 2.7, "hiss", SPEECH - 4.0), (2.7, 3.3, "voice", SPEECH)], 5.0)
        E = energy_of(x)
        self.assertEqual(align.sound_into_starts(E, self.words), [])
        self.assertEqual(T.timing_flags(self.words, E), [])

    def test_the_word_keeps_some_of_its_span(self):
        # the sound only changes 40 ms before the word's end: too little left to be the word
        x = synth([(1.0, 1.5, "voice", SPEECH), (1.9, 2.86, "laugh", SPEECH - 2.0),
                   (2.86, 3.3, "voice", SPEECH)], 5.0)
        E, B = energy_of(x), align.BandShapes.from_samples(x)
        fixed, log = align.repair_overlapped(self.words, E, B)
        self.assertEqual((fixed[1]["s"], log), (2.3, []))

    def test_retime_recomputes_the_flag_and_keeps_the_others(self):
        doc = build(LAUGH_RAW, self.E, Listen())
        doc["flags"].append({"a": 2, "b": 2, "kind": "false_start", "s": 2.9, "e": 3.0})
        new, moved = T.retime_doc(doc, self.E, self.B)
        self.assertEqual(moved, 1)
        self.assertAlmostEqual(new["words"][1]["s"], 2.6, delta=0.021)
        self.assertEqual(kinds(new, "timing_uncertain"), [])
        self.assertEqual(len(kinds(new, "false_start")), 1)
        self.assertEqual(new["text_compatible_shas"], [doc["sha256"]])
        again, moved = T.retime_doc(new, self.E, self.B)
        self.assertEqual(moved, 0)
        # without band shapes the flag comes back for the transcript that was never moved
        plain, _ = T.retime_doc(doc, self.E)
        self.assertEqual(kinds(plain, "timing_uncertain"),
                         [{"a": 1, "b": 1, "kind": "timing_uncertain"}])


# Two laugh bursts sit inside the span of an unsure "So" (5.0-5.5): the laugh runs 4.5-5.15,
# dips, and the word itself is the last burst (5.3-5.5).
BURSTS = [(3.0, 4.0, SPEECH), (4.5, 5.15, SPEECH - 3.0), (5.15, 5.3, SPEECH - 25.0),
          (5.3, 5.8, SPEECH)]
TAIL = [(" method.", 3.5, 4.0, 0.9), (" So", 5.31, 5.5, 0.9), (" where", 5.5, 5.8, 0.9)]


def burst_words(word="So", p=0.72):
    return T.normalize_words([(" method.", 3.0, 4.0, 0.99), (" " + word, 5.0, 5.5, p),
                              (" where", 5.5, 5.8, 0.99)])[0]


class TestLaughBurstsInsideAWord(unittest.TestCase):
    def test_start_moves_past_the_laugh(self):
        E = track(BURSTS)
        words = burst_words()
        listen = Listen(TAIL)
        self.assertEqual(T.settle_bursts(words, E, listen), 1)
        self.assertEqual((words[1]["s"], words[1]["e"]), (5.3, 5.5))
        self.assertEqual(words[1]["_rep"], {"tag": "dip", "s0": 5.0, "e0": 5.5})
        self.assertNotIn("_fs", words[1])
        # two re-listens: the lead-in up to the dip, then from the end of the dip on
        self.assertEqual(listen.calls, [(2.9, 5.225), (5.27, 5.9)])

    def test_transcript_records_the_move(self):
        doc = build([(" method.", 3.0, 4.0, 0.99), (" So", 5.0, 5.5, 0.72),
                     (" where", 5.5, 5.8, 0.99)], track(BURSTS), Listen(TAIL))
        self.assertEqual([w["w"] for w in doc["words"]], ["method.", "So", "where"])
        self.assertEqual(doc["words"][1]["s"], 5.3)
        self.assertEqual([(r["i"], r["tag"], r["s0"], r["s"]) for r in doc["timing_repairs"]],
                         [(1, "dip", 5.0, 5.3)])
        self.assertEqual(kinds(doc, "false_start"), [])

    def test_not_moved_when_the_word_is_not_heard_after_the_dip(self):
        E = track(BURSTS)
        words = burst_words()
        heard = [(" method.", 3.5, 4.0, 0.9), (" Where", 5.31, 5.8, 0.9)]
        self.assertEqual(T.settle_bursts(words, E, Listen(heard)), 0)
        self.assertEqual(words[1]["s"], 5.0)

    def test_not_moved_inside_a_longer_word(self):
        # the dip in a word of several syllables can be its own stop consonant
        E = track(BURSTS)
        words = burst_words("experienced")
        heard = [(" method.", 3.5, 4.0, 0.9), (" experienced", 5.31, 5.5, 0.9)]
        self.assertEqual(T.settle_bursts(words, E, Listen(heard)), 0)
        self.assertEqual(words[1]["s"], 5.0)

    def test_not_moved_when_the_first_burst_starts_out_of_silence(self):
        # no sound runs into the stamped start: the first burst may be the word's own consonant
        E = track([(3.0, 4.0, SPEECH), (5.0, 5.15, SPEECH - 3.0), (5.15, 5.3, SPEECH - 25.0),
                   (5.3, 5.8, SPEECH)])
        words = burst_words()
        self.assertEqual(T.settle_bursts(words, E, Listen(TAIL)), 0)
        self.assertEqual(words[1]["s"], 5.0)

    def test_not_moved_without_a_relisten(self):
        E = track(BURSTS)
        words = burst_words()
        self.assertEqual(T.settle_bursts(words, E), 0)
        self.assertEqual(words[1]["s"], 5.0)
        self.assertEqual(words[1]["_fs"], {"e": 5.3, "heard": []})      # flagged instead

    def test_syllables(self):
        for word, n in (("So", 1), ("once", 1), ("that's", 1), ("There's", 1), ("don't", 1),
                        ("like", 1), ("the", 1), ("people", 2), ("Rachel", 2),
                        ("experienced", 4), ("2013.", 0), ("", 0)):
            self.assertEqual(T._syllables(word), n, word)


# ---------------------------------------------------------------------------
# item 5: transcript.md and `hc words` show the two new flags
# ---------------------------------------------------------------------------


def note_doc(flags):
    words = T.index_words([W("In", 10.0, 10.2), W("this", 10.2, 10.5), W("moment.", 10.5, 11.0),
                           W("And", 12.0, 12.3), W("after", 12.3, 12.6), W("about", 12.6, 13.3),
                           W("a", 13.3, 13.4), W("decade.", 13.4, 14.0),
                           W("She", 20.0, 20.4), W("was", 20.4, 20.7), W("like.", 20.7, 21.0)])
    return {"media": "clip.mp4", "backend": "faster-whisper", "model": "tiny", "language": "en",
            "words": words, "lines": T.build_lines(words), "flags": flags, "timing_repairs": []}


class TestMarkdownNotes(unittest.TestCase):
    def test_false_start_and_uncertain_start_sit_above_their_line(self):
        doc = note_doc([
            {"a": 5, "b": 5, "kind": "false_start", "s": 12.6, "e": 12.91, "heard": "a w"},
            {"a": 8, "b": 8, "kind": "timing_uncertain"}])
        md = T.render_md(doc, 30.0).splitlines()
        body = md[3:]
        self.assertEqual(body, [
            "[00:10.0] #0-2 In this moment.",
            '[00:12.6] <<possible false start inside #5 "about": a re-listen also hears "a w" '
            "there; the phrase restarts at 00:12.91>>",
            "[00:12.0] #3-7 And after about a decade.",
            '[00:20.0] <<the start of #8 "She" is uncertain: it begins under other sound (a '
            "laugh, crosstalk), check the caption onset>>",
            "[00:20.0] #8-10 She was like."])

    def test_unconfirmed_and_repeated(self):
        doc = note_doc([{"a": 5, "b": 5, "kind": "false_start", "s": 12.6, "e": 12.91},
                        {"a": 3, "b": 4, "kind": "false_start", "s": 12.0, "e": 12.3}])
        md = T.render_md(doc, 30.0)
        self.assertIn('[00:12.6] <<possible false start inside #5 "about": it holds two bursts '
                      "of sound, the second starts at 00:12.91>>", md)
        self.assertIn('[00:12.0] <<repeated attempt #3-4 ("And after"): one of these words holds '
                      "more sound than its letters; the repeat starts at 00:12.30>>", md)

    def test_notes_keep_time_order_with_unworded_sound(self):
        doc = note_doc([{"a": 2, "b": 3, "kind": "unworded_audio", "s": 11.1, "e": 11.9},
                        {"a": 5, "b": 5, "kind": "false_start", "s": 12.6, "e": 12.91}])
        body = T.render_md(doc, 30.0).splitlines()[3:]
        self.assertTrue(body[1].startswith("[00:11.1] <<0.8 s of sound with no words"), body[1])
        self.assertTrue(body[2].startswith("[00:12.6] <<possible false start"), body[2])
        self.assertTrue(body[3].startswith("[00:12.0] #3-7"), body[3])

    def test_a_stale_flag_is_skipped(self):
        doc = note_doc([{"a": 50, "b": 50, "kind": "false_start", "s": 1.0, "e": 2.0},
                        {"a": 3, "kind": "timing_uncertain"}])
        self.assertEqual(len(T.render_md(doc, 30.0).splitlines()), 6)

    def test_hours(self):
        doc = note_doc([{"a": 8, "b": 8, "kind": "timing_uncertain"}])
        for w in doc["words"]:
            w["s"], w["e"] = w["s"] + 3600.0, w["e"] + 3600.0
        doc["lines"] = T.build_lines(doc["words"])
        self.assertIn('[1:00:20.0] <<the start of #8 "She" is uncertain', T.render_md(doc, 4000.0))

    def test_words_rows_mark_both_flags(self):
        doc = note_doc([{"a": 5, "b": 5, "kind": "false_start", "s": 12.6, "e": 12.91},
                        {"a": 8, "b": 8, "kind": "timing_uncertain"}])
        rows = find.word_rows(doc, 4, 9)
        self.assertNotRegex(rows[1], r"\s!\s")
        self.assertRegex(rows[2], r"\s!\s")           # #5
        self.assertNotRegex(rows[3], r"\s!\s")
        self.assertRegex(rows[5], r"\s!\s")           # #8

    def test_report_names_them(self):
        doc = note_doc([{"a": 5, "b": 5, "kind": "false_start", "s": 12.6, "e": 12.91,
                         "heard": "a w"}, {"a": 8, "b": 8, "kind": "timing_uncertain"}])
        doc.update(duration=30.0, speakers_vtt=None)
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            T._report(doc, [], Path("transcript.json"))
        log = err.getvalue()
        self.assertIn("1 possible false starts or repeated attempts", log)
        self.assertIn("#5 at 00:12.6: 'about', a re-listen also hears 'a w' there", log)
        self.assertIn("1 word starts are uncertain", log)
        self.assertIn("#8 at 00:20.0", log)

    def test_post_version_moved_with_the_retiming(self):
        # QA keys its cached re-transcriptions of renders on this number
        self.assertGreaterEqual(T.POST_VERSION, 4)


@needs_fixtures
class TestCommandFlagsSkippedSpeech(unittest.TestCase):
    """`hc transcribe` on real decoded audio: 0.7 s of sound runs straight out of "moment."
    and the recogniser writes no word for it."""

    def test_flagged_and_listed_in_markdown(self):
        spans = [("In", 1.0, 1.3), ("this", 1.3, 1.6), ("moment.", 1.6, 2.0), ("skipped", 2.0, 2.7),
                 ("And", 4.0, 4.4), ("after", 4.4, 5.0)]
        heard = [[" In", 1.0, 1.3, 0.99], [" this", 1.3, 1.6, 0.99], [" moment.", 1.6, 2.0, 0.99],
                 [" And", 4.0, 4.4, 0.99], [" after", 4.4, 5.0, 0.99]]
        calls = []

        def runner(wav, model, language, prompt, progress=True):
            dur = T.wav_duration(wav)
            calls.append(round(dur, 2))
            # the full pass hears five words; a re-listen of a slice hears nothing more
            return {"words": [list(w) for w in heard] if dur > 6.0 else [], "language": "en",
                    "options": T.runner_options("faster-whisper", model, prompt)}

        with tempfile.TemporaryDirectory() as td:
            media = fixtures.make_speech_fixture(Path(td) / "m", spans, dur=7.0)
            P = fixtures.make_project(Path(td), media)
            args = argparse.Namespace(project=str(P), backend="faster-whisper", model=None,
                                      language="en", glossary=None, speakers_vtt=None, file=None,
                                      clip=None, force=False)
            old = T.RUNNERS["faster-whisper"]
            T.RUNNERS["faster-whisper"] = runner
            try:
                with contextlib.redirect_stdout(io.StringIO()), \
                        contextlib.redirect_stderr(io.StringIO()):
                    self.assertEqual(T.cmd_transcribe(args), 0)
            finally:
                T.RUNNERS["faster-whisper"] = old
            doc = env.read_json(P / "transcript.json")
            md = (P / "transcript.md").read_text(encoding="utf-8")
        self.assertEqual(len(calls), 2, calls)                 # the full pass and one re-listen
        self.assertAlmostEqual(calls[1], 2.65, delta=0.1)      # the stretch plus 1 s either side
        self.assertEqual([w["w"] for w in doc["words"]], ["In", "this", "moment.", "And", "after"])
        flags = kinds(doc, "unworded_audio")
        self.assertEqual(len(flags), 1, doc["flags"])
        self.assertAlmostEqual(flags[0]["s"], 2.05, delta=0.03)
        self.assertAlmostEqual(flags[0]["e"], 2.7, delta=0.03)
        self.assertRegex(md, r"\[00:02\.[01]\] <<0\.[67] s of sound with no words")
        self.assertEqual(doc["post_version"], T.POST_VERSION)


# ---------------------------------------------------------------------------
# item 4: the faster-whisper runner never hands the model a file path
# ---------------------------------------------------------------------------


class TestFasterWhisperGetsSamples(unittest.TestCase):
    """faster-whisper decodes a file path through PyAV, and the two libraries can disagree
    (a release of one drops an argument the other still passes: a TypeError on any path). The
    runner decodes with ffmpeg itself and passes samples, so it does not depend on that."""

    def test_path_and_str_both_reach_the_model_as_samples(self):
        seen = []

        class WhisperModel(object):
            def __init__(self, model, device="auto", compute_type="default", download_root=None):
                pass

            def transcribe(self, audio, **kw):
                if not isinstance(audio, np.ndarray):
                    raise TypeError("open() got an unexpected keyword argument")
                seen.append((audio.dtype.name, audio.ndim, len(audio)))
                seg = types.SimpleNamespace(end=0.3, words=[types.SimpleNamespace(
                    word=" hi", start=0.1, end=0.3, probability=0.9)])
                return iter([seg]), types.SimpleNamespace(language="en")

        mod = types.ModuleType("faster_whisper")
        mod.WhisperModel = WhisperModel
        saved = sys.modules.get("faster_whisper")
        old_dir, old_dev = env.fw_models_dir, T._fw_device
        with tempfile.TemporaryDirectory() as td:
            wav = Path(td) / "a.wav"
            write_wav(wav, np.zeros(SR, dtype=np.float32))
            sys.modules["faster_whisper"] = mod
            env.fw_models_dir = lambda: Path(td) / "fw"
            T._fw_device = lambda: ("cpu", "int8")
            T._FW_MODELS.clear()
            try:
                for arg in (wav, str(wav)):
                    res = quiet(T.run_faster_whisper, arg, "large-v3-turbo", "en", None,
                                progress=False)
                    self.assertEqual(res["words"], [[" hi", 0.1, 0.3, 0.9]])
            finally:
                T._FW_MODELS.clear()
                env.fw_models_dir, T._fw_device = old_dir, old_dev
                if saved is None:
                    sys.modules.pop("faster_whisper", None)
                else:
                    sys.modules["faster_whisper"] = saved
        self.assertEqual(seen, [("float32", 1, SR), ("float32", 1, SR)])


if __name__ == "__main__":
    unittest.main()
