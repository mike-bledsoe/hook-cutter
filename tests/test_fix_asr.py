"""ASR and timing fixes: energy word-time repair (measured against ground truth), the audio
gate, re-listening to gaps, --retime, speakers VTT robustness, whisper.cpp DTW words, the
faster-whisper CPU fallback, cache keys and small transcript fixes."""

from __future__ import annotations

import argparse
import contextlib
import difflib
import io
import json
import os
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

from hookcutter import align, env, find, textnorm, transcribe as T  # noqa: E402
from hookcutter.audio import EnergyTrack  # noqa: E402

try:
    import fixtures  # noqa: E402
except Exception:  # noqa: BLE001 - media tests skip when the shared fixtures are unavailable
    fixtures = None

needs_fixtures = unittest.skipIf(fixtures is None, "tests/fixtures.py unavailable")

DATA = HERE / "data" / "timing"
PASSAGES = ("p1_client_phrase", "p1_client_word_noise", "p3_hesitant_phrase", "p4_fast_word")
LONG = 0.4     # a "long" pause before a word (ground truth), seconds


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def track(segments=(), dur=30.0, floor=-60.0, hop=0.01):
    """EnergyTrack at `floor` dB with [(s, e, dB)] segments (src clock, t0 = 0)."""
    db = np.full(int(round(dur / hop)), floor, dtype=np.float64)
    for s, e, level in segments:
        db[int(round(s / hop)):int(round(e / hop))] = level
    return EnergyTrack(db, hop=hop, t0=0.0)


def W(text, s, e, p=0.9, sp=None):
    return {"w": text, "s": s, "e": e, "p": p, "sp": sp}


def run_quiet(fn, *a, **kw):
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        out = fn(*a, **kw)
    return out, err.getvalue()


def _write_wav(path, seconds=1.0):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x00" * int(16000 * seconds))


class FakeRunner(object):
    """Stands in for an ASR backend: canned words on the wav clock it is given."""

    def __init__(self, words):
        self.words = words
        self.calls = []

    def __call__(self, wav, model, language, prompt, progress=True):
        with wave.open(str(wav), "rb") as w:
            dur = w.getnframes() / float(w.getframerate())
        self.calls.append({"wav": str(wav), "model": model, "language": language,
                           "prompt": prompt, "dur": round(dur, 3)})
        return {"words": [list(x) for x in self.words], "language": language or "en",
                "options": T.runner_options("faster-whisper", model, prompt)}


@contextlib.contextmanager
def fake_backend(words, name="faster-whisper"):
    runner = FakeRunner(words)
    old = T.RUNNERS[name]
    T.RUNNERS[name] = runner
    try:
        yield runner
    finally:
        T.RUNNERS[name] = old


# Port of the reviewer's ground-truth scorer: ASR words are matched 1:1 to truth words on
# normalized tokens; starts of truth words that follow a long pause are scored separately.

def _match(truth, hyp):
    ts = textnorm.norm_stream([w["w"] for w in truth])
    hs = textnorm.norm_stream([w["w"] for w in hyp])
    sm = difflib.SequenceMatcher(None, [t for t, _, _ in ts], [t for t, _, _ in hs],
                                 autojunk=False)
    pairs, seen = [], set()
    for a, b, n in sm.get_matching_blocks():
        for k in range(n):
            ta, tb = ts[a + k][1], ts[a + k][2]
            ha, hb = hs[b + k][1], hs[b + k][2]
            if ta != tb or ha != hb:
                continue
            if sum(1 for x in ts if x[1] == ta) != 1 or sum(1 for x in hs if x[1] == ha) != 1:
                continue
            if ta not in seen:
                seen.add(ta)
                pairs.append((ta, ha))
    return pairs


def _stats(xs):
    a = np.abs(np.asarray(xs, dtype=np.float64))
    return {"n": len(xs), "med": float(np.median(a)), "p90": float(np.percentile(a, 90)),
            "max": float(a.max())}


def score(truth, hyp):
    tw = truth["words"]
    longp = [p for p in truth["pauses"] if p[1] - p[0] >= LONG]
    start, after = [], []
    in_sil = spans = 0
    for ti, hi in _match(tw, hyp):
        t, h = tw[ti], hyp[hi]
        if t["exact_s"]:
            d = h["s"] - t["s"]
            start.append(d)
            if (t.get("pause_before") or 0) >= LONG:
                after.append(d)
        if any(a + 0.10 < h["s"] < b - 0.10 for a, b in longp):
            in_sil += 1
        if any(min(h["e"], b) - max(h["s"], a) >= 0.2 for a, b in longp):
            spans += 1
    return {"start": _stats(start), "after": _stats(after), "onset_in_silence": in_sil,
            "word_spans_pause": spans}


def load_passage(name):
    truth = json.loads((DATA / (name + ".truth.json")).read_text(encoding="utf-8"))
    E = EnergyTrack(np.load(DATA / (name + ".energy.npy")).astype(np.float64), hop=0.01, t0=0.0)
    words = {be: json.loads((DATA / ("%s.%s.json" % (name, be))).read_text(encoding="utf-8"))
             for be in ("mlx", "fw")}
    return truth, E, words


# ---------------------------------------------------------------------------
# (a) energy repair against ground truth
# ---------------------------------------------------------------------------


class TestRepairAgainstTruth(unittest.TestCase):
    """Whisper pulls the start of the word after a pause back to the previous word's end.
    repair_times must put those starts on the voiced onset (synthetic speech, exact truth)."""

    def test_passages(self):
        for name in PASSAGES:
            truth, E, by_backend = load_passage(name)
            for be, words in by_backend.items():
                with self.subTest(passage=name, backend=be):
                    base = score(truth, words)
                    # the defect is present in the stored ASR words
                    self.assertGreater(base["after"]["med"], 0.1)
                    self.assertGreater(base["word_spans_pause"], 1)
                    fixed, log = align.repair_times(words, E)
                    self.assertEqual([w["w"] for w in fixed], [w["w"] for w in words])
                    self.assertTrue(log)
                    r = score(truth, fixed)
                    self.assertLessEqual(r["after"]["med"], 0.010, r)
                    self.assertLessEqual(r["after"]["p90"], 0.20, r)
                    self.assertLessEqual(r["after"]["max"], 0.40, r)
                    self.assertEqual(r["onset_in_silence"], 0, r)
                    self.assertLessEqual(r["word_spans_pause"], 1, r)
                    self.assertLessEqual(r["start"]["med"], base["start"]["med"] + 0.01, r)

    def test_small_model_around_a_pause(self):
        """p5: the hc demo's text-to-speech read by faster-whisper tiny.en. Truth is the pause
        edges measured from the audio. The model runs a word's END on into the pause after it
        ("the clip." 6.60-7.10, voiced to 6.77) and starts the word after a pause late
        ("Nobody" at 7.80, voiced from 7.63). The repair keeps the first word before the pause
        (its loud audio there is its own, not the previous word's) and late_starts moves the
        second back to its onset."""
        name = "p5_tts_pause_end"
        truth = json.loads((DATA / (name + ".truth.json")).read_text(encoding="utf-8"))
        E = EnergyTrack(np.load(DATA / (name + ".energy.npy")).astype(np.float64), hop=0.01, t0=0.0)
        words = json.loads((DATA / (name + ".fw.json")).read_text(encoding="utf-8"))
        base = score(truth, words)
        self.assertGreater(base["after"]["med"], 0.1)
        fixed, _ = align.repair_times(words, E)
        fixed, _ = T.late_starts(fixed, E)
        r = score(truth, fixed)
        self.assertLessEqual(r["after"]["med"], 0.03, r)
        self.assertLessEqual(r["after"]["max"], 0.10, r)
        ends = [fixed[hi]["e"] - truth["words"][ti]["e"] for ti, hi in _match(truth["words"], fixed)
                if truth["words"][ti]["exact_e"]]
        self.assertLessEqual(float(np.median(np.abs(ends))), 0.05, ends)
        self.assertLessEqual(max(abs(x) for x in ends), 0.35, ends)
        clip = next(w for w in fixed if w["w"] == "clip.")
        self.assertLess(clip["e"], 6.8, "the word before the pause stays before it")

    def test_repair_is_monotonic_and_logged(self):
        truth, E, by_backend = load_passage("p3_hesitant_phrase")
        words = by_backend["mlx"]
        fixed, log = align.repair_times(words, E)
        for k, w in enumerate(fixed):
            self.assertLessEqual(w["s"], w["e"])
            if k:
                self.assertLessEqual(fixed[k - 1]["e"], w["s"] + 1e-9)
                self.assertGreaterEqual(w["s"], fixed[k - 1]["s"])
        for ent in log:
            self.assertEqual(set(ent), {"i", "kind", "tag", "s0", "e0", "s", "e"})
            self.assertEqual(ent["kind"], "energy")
            self.assertEqual((fixed[ent["i"]]["s"], fixed[ent["i"]]["e"]), (ent["s"], ent["e"]))
            self.assertEqual((words[ent["i"]]["s"], words[ent["i"]]["e"]), (ent["s0"], ent["e0"]))

    def test_word_after_pause_moves_to_onset(self):
        # "one" 1.0-1.4 voiced, pause, "two" voiced 2.3-2.7 but stamped 1.4-2.7
        E = track([(1.0, 1.4, -30.0), (2.3, 2.7, -30.0)], dur=10.0)
        fixed, log = align.repair_times([W("one", 1.0, 1.4), W("two", 1.4, 2.7)], E)
        self.assertAlmostEqual(fixed[1]["s"], 2.3, places=2)
        self.assertEqual(fixed[1]["e"], 2.7)
        self.assertEqual(fixed[0]["s"], 1.0)
        self.assertEqual([e["i"] for e in log], [1])

    def test_level_helpers(self):
        E = track([(1.0, 1.4, -30.0), (2.3, 2.7, -30.0), (5.0, 5.02, -30.0)], dur=10.0)
        self.assertEqual(align.voiced_runs(E, 0.0, 4.0), [(1.0, 1.4), (2.3, 2.7)])
        ps = align.pauses(E, 0.0, 4.0)
        self.assertIn((1.4, 2.3), ps)
        self.assertAlmostEqual(align.next_onset(E, 1.5, 3.0), 2.3)
        self.assertIsNone(align.next_onset(E, 1.5, 2.0))
        self.assertIsNone(align.next_onset(E, 4.0, 6.0), "a 20 ms blip is not an onset")


# ---------------------------------------------------------------------------
# (b) audio gate
# ---------------------------------------------------------------------------


class TestAudioGate(unittest.TestCase):
    def test_gate_levels(self):
        # floor -60: "Thank" peaks at floor + 2 (room tone), "real" at floor + 26
        E = track([(5.0, 5.4, -58.0), (8.0, 8.4, -34.0)], dur=20.0)
        words = [W("Thank", 5.0, 5.4), W("real", 8.0, 8.4), W("late", 25.0, 25.3)]
        kept, dropped = align.audio_gate(words, E)
        self.assertEqual([w["w"] for w in kept], ["real", "late"], "beyond the track is kept")
        self.assertEqual([w["w"] for w in dropped], ["Thank"])

    def test_gate_runs_before_repair(self):
        # a hallucination in silence just before a real word: without the gate first, the
        # repair's "stamped into silence" rule would move it onto the real onset
        E = track([(1.0, 1.5, -30.0), (6.0, 6.5, -30.0)], dur=20.0)
        asr = {"backend": "mlx", "model": "m", "words": [
            [" Yes.", 1.0, 1.5, 0.9], [" Thank", 3.5, 4.0, 0.2], [" you.", 4.0, 4.5, 0.2],
            [" Next", 6.0, 6.5, 0.9]]}
        doc, _ = run_quiet(T.build_transcript, asr, offset=0.0, media="m.mp4", fingerprint=None,
                           energy=E)
        self.assertEqual([w["w"] for w in doc["words"]], ["Yes.", "Next"])
        self.assertEqual([(e["kind"], e["text"]) for e in doc["events"]],
                         [("no_audio", "Thank"), ("no_audio", "you.")])
        self.assertEqual(doc["words"][1]["s"], 6.0)
        self.assertEqual(doc["post_version"], T.POST_VERSION)


# ---------------------------------------------------------------------------
# (c) dropout holes: re-listen and splice
# ---------------------------------------------------------------------------


class TestHoles(unittest.TestCase):
    def setUp(self):
        # speech 2-4 s (worded), 10-15 s (no words: ASR dropped it), 20-21 s (laughter: no
        # words even on a re-listen), 25-26 s (worded)
        self.E = track([(2.0, 4.0, -30.0), (10.0, 15.0, -30.0), (20.0, 21.0, -30.0),
                        (25.0, 26.0, -30.0)], dur=30.0)
        self.words = [W("we", 2.0, 2.5), W("said", 2.5, 3.0), W("it", 3.0, 3.5),
                      W("twice.", 3.5, 4.0), W("Then", 25.0, 25.5), W("more.", 25.5, 26.0)]

    def test_uncovered_voiced(self):
        holes = align.uncovered_voiced(self.E, self.words, 0.0, 30.0)
        self.assertEqual(holes, [(10.0, 15.0), (20.0, 21.0)])
        self.assertEqual(align.uncovered_voiced(self.E, self.words, 0.0, 30.0, min_run=1.5),
                         [(10.0, 15.0)])

    def test_splice_and_flags(self):
        calls = []

        def listen(a, b):
            calls.append((a, b))
            if a < 12.0 < b:   # the model hears the dropped stretch on its own
                return [[" so", 10.0, 10.6, 0.9], [" here", 10.8, 12.0, 0.9],
                        [" we", 12.2, 13.0, 0.9], [" go.", 13.2, 15.0, 0.9],
                        [" twice.", 3.5, 4.0, 0.9]]       # outside the gap: ignored
            return []

        words, gaps = T.fill_holes([dict(w) for w in self.words], self.E, listen)
        self.assertEqual(calls, [(9.0, 16.0), (19.0, 22.0)], "each gap +/- 1 s of context")
        self.assertEqual([w["w"] for w in words],
                         ["we", "said", "it", "twice.", "so", "here", "we", "go.", "Then", "more."])
        self.assertEqual(words[4]["s"], 10.0)
        self.assertEqual(gaps, [{"s": 10.0, "e": 15.0, "n": 4}, {"s": 20.0, "e": 21.0, "n": 0}])
        flags = T.gap_flags(T.index_words(words), self.E, gaps)
        kinds = [(f["kind"], f["s"], f["e"]) for f in flags]
        self.assertIn(("respliced", 10.0, 15.0), kinds)
        self.assertIn(("unworded_audio", 20.0, 21.0), kinds)
        un = [f for f in flags if f["kind"] == "unworded_audio"][0]
        self.assertEqual((un["a"], un["b"]), (7, 8), "a = last word before, b = first after")

    def test_prompt_fragment_echo_is_not_spliced(self):
        """A prompted re-listen of breath or filler can write one glossary term at p 0.00 (a
        fragment of the prompt, which drop_prompt_echo's whole-prompt match misses). Such a
        run is dropped; the same term heard with confidence, and plain words, are kept."""
        prompt = "Ada Quill, Bo Varn, Delta Works."

        def listen(a, b, plain=False):
            if a < 12.0 < b:
                return [[" Ada", 10.2, 10.6, 0.0], [" Quill.", 10.6, 11.0, 0.32],
                        [" so", 12.0, 12.4, 0.9], [" Bo", 13.0, 13.4, 0.92],
                        [" Varn", 13.4, 13.9, 0.95], [" said", 14.0, 14.6, 0.9]]
            return []

        words, gaps = T.fill_holes([dict(w) for w in self.words], self.E, listen, prompt=prompt)
        got = [w["w"] for w in words]
        self.assertNotIn("Ada", got)
        self.assertNotIn("Quill.", got)
        self.assertEqual(got[4:8], ["so", "Bo", "Varn", "said"])
        kept, n = T.drop_echo_fragments([W("Delta", 1.0, 1.4, 0.01), W("Works,", 1.4, 1.9, 0.4),
                                         W("delta", 2.0, 2.4, 0.9)], prompt)
        self.assertEqual((n, [w["w"] for w in kept]), (2, ["delta"]))
        self.assertEqual(T.drop_echo_fragments([W("Delta", 1.0, 1.4, 0.01)], None)[1], 0)

    def test_relisten_offsets_and_slice_cache(self):
        with tempfile.TemporaryDirectory() as td:
            wav = Path(td) / "a.wav"
            _write_wav(wav, 30.0)
            asr = {"backend": "faster-whisper", "model": "large-v3-turbo", "language": "en",
                   "glossary": ["Acme"]}
            with fake_backend([[" hello", 0.5, 0.9, 0.9]]) as runner:
                ln = T.Relisten(wav, 0.25, asr, cache_dir=Path(td) / "cache")
                got = ln(10.0, 12.0)            # src clock; the wav starts at src 0.25
                self.assertEqual(len(runner.calls), 1)
                self.assertAlmostEqual(runner.calls[0]["dur"], 2.0, places=2)
                self.assertEqual(runner.calls[0]["prompt"], "Acme.")
                self.assertAlmostEqual(got[0][1], 10.5)
                self.assertAlmostEqual(got[0][2], 10.9)
                ln(10.0, 12.0)
                self.assertEqual(len(runner.calls), 1, "slice result cached in memory")
                ln.save()
                ln2 = T.Relisten(wav, 0.25, asr, cache_dir=Path(td) / "cache")
                self.assertEqual(ln2(10.0, 12.0), got)
                self.assertEqual(len(runner.calls), 1, "slice result cached on disk")
                ln3 = T.Relisten(wav, 0.25, asr, cache_dir=Path(td) / "cache", force=True)
                ln3(10.0, 12.0)
                self.assertEqual(len(runner.calls), 2, "--force re-listens")
                self.assertTrue(ln.can_retry)
                ln(10.0, 12.0, plain=True)
                self.assertEqual(len(runner.calls), 3)
                self.assertIsNone(runner.calls[2]["prompt"], "the retry sends no prompt")

    def test_build_transcript_with_relisten(self):
        def listen(a, b):
            return [[" so", 10.1, 11.0, 0.9], [" go.", 11.2, 14.8, 0.9]] if a < 12.0 < b else []

        asr = {"backend": "mlx", "model": "m",
               "words": [[" " + w["w"], w["s"], w["e"], w["p"]] for w in self.words]}
        doc, err = run_quiet(T.build_transcript, asr, offset=0.0, media="m.mp4", fingerprint=None,
                             energy=self.E, listen=listen)
        self.assertEqual([w["w"] for w in doc["words"]][4:6], ["so", "go."])
        self.assertEqual([w["i"] for w in doc["words"]], list(range(8)))
        kinds = sorted({f["kind"] for f in doc["flags"]})
        self.assertEqual(kinds, ["respliced", "unworded_audio"])
        _, err = run_quiet(T._report, doc, [], Path(tempfile.gettempdir()) / "t" / "x.json")
        self.assertIn("re-listened to 2 gaps (6.0 s): spliced 2 words; 1 gaps still have no "
                      "words (flagged)", err)

    def test_vtt_dropout_holes(self):
        # the speakers VTT has a cue at 5.5-7.5 with 6 words where ASR has none (and the
        # energy sees nothing there: a quiet speaker); it is re-listened and, if still
        # empty, flagged as a dropout
        cues = [{"s": 5.5, "e": 7.5, "sp": "Sam", "text": "and all those crazy fees too"}]
        holes = T.find_holes(self.words, self.E, cues)
        self.assertIn((5.5, 7.5), holes)
        _, gaps = T.fill_holes([dict(w) for w in self.words], self.E, lambda a, b: [], cues=cues)
        flags = T.gap_flags(T.index_words(self.words), self.E, gaps, cues)
        drop = [f for f in flags if f["kind"] == "dropout"]
        self.assertEqual(len(drop), 1)
        self.assertEqual((drop[0]["vtt_words"], drop[0]["asr_words"]), (6, 0))

    def test_prompt_free_retry_for_a_skipped_stretch(self):
        # with the glossary prompt the slice writes a short phrase and skips the rest; the
        # still-unworded stretch is re-listened once without the prompt
        class Listen(object):
            can_retry = True

            def __init__(self):
                self.calls = []

            def __call__(self, a, b, plain=False):
                self.calls.append((a, b, plain))
                if not plain:
                    return [[" Studio.", 10.0, 10.6, 0.3]]
                return [[" so", 11.0, 11.5, 0.9], [" we", 11.6, 12.4, 0.9],
                        [" built", 12.6, 14.9, 0.9]]

        ln = Listen()
        words, gaps = T.fill_holes([dict(w) for w in self.words], self.E, ln)
        self.assertEqual([c[2] for c in ln.calls], [False, False, True])
        # the retry covers what is still unworded: from just after "Studio." to the gap's end
        self.assertAlmostEqual(ln.calls[2][0], 10.6 + 0.05 - T.HOLE_PAD)
        self.assertAlmostEqual(ln.calls[2][1], 15.0 + T.HOLE_PAD)
        self.assertEqual([w["w"] for w in words][4:8], ["Studio.", "so", "we", "built"])
        self.assertEqual(gaps[0]["n"], 4)
        ln.can_retry = False
        ln.calls = []
        T.fill_holes([dict(w) for w in self.words], self.E, ln)
        self.assertEqual([c[2] for c in ln.calls], [False, False])

    def test_context_words_are_not_added_twice(self):
        # the slice hears the context again ("twice.") with slightly different times; only the
        # words the transcript lacks are added, in sentence order
        E = track([(2.0, 4.0, -30.0), (4.3, 7.0, -30.0)], dur=12.0)
        words = [W("we", 2.0, 2.5), W("said", 2.5, 3.0), W("it", 3.0, 3.5), W("twice.", 3.5, 4.0)]

        def listen(a, b):
            return [[" said", 2.45, 2.9, 0.9], [" it", 2.95, 3.4, 0.9], [" twice.", 3.45, 3.95, 0.9],
                    [" And", 4.3, 4.6, 0.9], [" it", 4.6, 4.9, 0.9], [" stuck.", 4.9, 7.0, 0.9]]

        out, gaps = T.fill_holes(words, E, listen)
        self.assertEqual([w["w"] for w in out], ["we", "said", "it", "twice.", "And", "it", "stuck."])
        self.assertEqual(gaps, [{"s": 4.3, "e": 7.0, "n": 3}])
        self.assertEqual([w["s"] for w in out[:4]], [2.0, 2.5, 3.0, 3.5], "context keeps its times")

    def test_misplaced_words_take_the_slice_times(self):
        # ASR skipped "more affordably because" and stamped "We're not replacing" on that audio;
        # the re-listen hears the whole stretch: the existing words move to where the slice
        # heard them and the missing words go in before them, keeping the sentence order
        E = track([(1.0, 2.0, -30.0), (2.4, 5.0, -30.0), (5.1, 5.9, -30.0), (6.2, 7.6, -30.0)],
                  dur=12.0)
        words = [W("cases?", 1.0, 2.0), W("We're", 2.4, 2.6), W("not", 2.6, 2.8),
                 W("replacing", 2.8, 3.1), W("attorneys.", 3.1, 7.6)]
        calls = []

        def listen(a, b):
            calls.append((a, b))
            return [[" more", 2.4, 3.0, 0.9], [" affordably", 3.0, 3.8, 0.9],
                    [" because.", 3.8, 5.0, 0.9], [" We're", 5.1, 5.3, 0.9],
                    [" not", 5.3, 5.5, 0.9], [" replacing", 5.5, 5.9, 0.9],
                    [" them.", 6.2, 6.9, 0.9]]

        words, _ = align.repair_times(words, E)      # as in postprocess: repair, then fill
        self.assertEqual(words[4]["s"], 6.2)
        out, gaps = T.fill_holes(words, E, listen)
        self.assertEqual(calls, [(2.15, 6.9)])
        self.assertEqual([w["w"] for w in out],
                         ["cases?", "more", "affordably", "because.", "We're", "not", "replacing",
                          "attorneys."])
        self.assertEqual(gaps[0]["n"], 3)
        self.assertEqual([w["s"] for w in out[4:7]], [5.1, 5.3, 5.5])
        self.assertEqual(out[1]["s"], 2.4)
        for k in range(1, len(out)):
            self.assertLessEqual(out[k - 1]["e"], out[k]["s"])
        self.assertTrue(any("relisten" in (w.get("_rep") or {}).get("tag", "") for w in out))

    def test_context_heard_on_the_same_side_keeps_its_times(self):
        # the slice starts mid-word and garbles its first second; context words it places on
        # the same side of the gap keep their (energy-repaired) times
        E = track([(1.0, 2.0, -30.0), (2.2, 3.4, -30.0)], dur=10.0)
        words = [W("a", 1.0, 1.2), W("famous", 1.2, 1.6), W("lawyer.", 1.6, 2.0)]

        def listen(a, b):
            return [[" a", 1.5, 1.55, 0.5], [" famous", 1.55, 1.7, 0.5],
                    [" lawyer.", 1.7, 1.9, 0.5], [" Is", 2.2, 2.6, 0.9], [" it", 2.6, 3.4, 0.9]]

        out, gaps = T.fill_holes([dict(w) for w in words], E, listen)
        self.assertEqual([(w["w"], w["s"]) for w in out[:3]],
                         [("a", 1.0), ("famous", 1.2), ("lawyer.", 1.6)])
        self.assertEqual([w["w"] for w in out[3:]], ["Is", "it"])

    def test_word_heard_differently_is_not_added(self):
        # the slice hears "You made" where the transcript has "You've made" (stamped late):
        # "You" substitutes "You've" (no insert), and "made" cannot move in front of "You've"
        words = [W("Thank", 1.0, 1.3), W("you.", 1.3, 1.6), W("You've", 2.6, 3.0),
                 W("made", 3.0, 3.6), W("us", 3.6, 3.8), W("money.", 3.8, 4.2)]
        new = [W("Thank", 1.0, 1.3), W("you.", 1.3, 1.6), W("You", 2.0, 2.3),
               W("made", 2.3, 2.5), W("us", 3.6, 3.8), W("money.", 3.8, 4.2)]
        n = T._splice(words, 1.9, 2.55, 0.9, 3.55, new)
        self.assertEqual(n, 0)
        self.assertEqual([w["w"] for w in words], ["Thank", "you.", "You've", "made", "us", "money."])
        self.assertEqual(words[3]["s"], 3.0, "a re-time that would cross a kept word is dropped")
        # heard far apart in time, a similar word is a different word: it is inserted
        words = [W("Thank", 1.0, 1.3), W("you.", 1.3, 1.6), W("You've", 3.6, 4.0)]
        new = [W("Thank", 1.0, 1.3), W("you.", 1.3, 1.6), W("You", 2.0, 2.3)]
        self.assertEqual(T._splice(words, 1.9, 2.5, 0.9, 3.5, new), 1)
        self.assertEqual([w["w"] for w in words], ["Thank", "you.", "You", "You've"])
        for k in range(1, len(words)):
            self.assertLessEqual(words[k - 1]["e"], words[k]["s"])

    def test_same_position_inserts_keep_slice_order(self):
        words = [W("exam.", 1.0, 1.4), W("Now", 5.0, 5.3)]
        new = [W("And", 2.0, 2.2), W("you", 2.2, 2.2), W("have", 2.2, 2.5)]
        n = T._splice(words, 1.9, 2.6, 0.9, 3.6, new)
        self.assertEqual(n, 3)
        self.assertEqual([w["w"] for w in words], ["exam.", "And", "you", "have", "Now"])

    def test_split_slices_do_not_duplicate_a_boundary_word(self):
        speech = [(10.0 + 0.4 * k, 10.3 + 0.4 * k, -30.0) for k in range(475)]
        E = track(speech, dur=210.0)

        def listen(a, b):     # every slice hears the word that straddles 70.0 s
            return [[" across", 69.8, 70.25, 0.9], [" start", a + 1.0, a + 1.25, 0.9]]

        words, gaps = T.fill_holes([], E, listen)
        self.assertEqual([g["s"] for g in gaps], [10.0, 70.0, 130.0, 190.0])
        self.assertEqual([w["w"] for w in words].count("across"), 1)

    def test_hole_merge_split_cap(self):
        # 190 s of continuous speech: 0.3 s syllables with 0.1 s dips (bridged), so the local
        # floor stays at room tone
        speech = [(10.0 + 0.4 * k, 10.3 + 0.4 * k, -30.0) for k in range(475)]
        E = track([(0.5, 1.5, -30.0), (2.5, 3.5, -30.0)] + speech, dur=210.0)
        holes = T.find_holes([], E)
        self.assertEqual(holes[0], (0.5, 3.5), "gaps closer than 2 s are merged")
        self.assertEqual(holes[1:], [(10.0, 70.0), (70.0, 130.0), (130.0, 190.0), (190.0, 199.9)])
        many = [(5.0 * k, 5.0 * k + 0.9 + 0.001 * k, -30.0) for k in range(400)]
        E = track(many, dur=2010.0)
        holes = T.find_holes([], E)
        self.assertEqual(len(holes), T.HOLE_CAP, "at most HOLE_CAP gaps (the longest)")
        self.assertEqual(holes, sorted(holes))
        self.assertGreater(min(e - s for s, e in holes), 0.9 + 0.001 * 99)


# ---------------------------------------------------------------------------
# (d) --retime
# ---------------------------------------------------------------------------


class TestRetime(unittest.TestCase):
    def _doc(self, words):
        ws = T.index_words(words)
        return {"version": 1, "words": ws, "lines": T.build_lines(ws), "flags": [],
                "sha256": T.transcript_sha(ws), "text_sha": T.text_sha(ws),
                "timing_repairs": [{"a": 0, "b": 1, "kind": "spread"}]}

    def test_retime_doc(self):
        E = track([(1.0, 1.4, -30.0), (2.3, 2.7, -30.0)], dur=10.0)
        doc = self._doc([W("one", 1.0, 1.4), W("two", 1.4, 2.7), W("ghost", 5.0, 5.5)])
        new, moved = T.retime_doc(doc, E)
        self.assertEqual(moved, 1)
        self.assertEqual([(w["i"], w["w"]) for w in new["words"]],
                         [(w["i"], w["w"]) for w in doc["words"]], "indices and texts unchanged")
        self.assertAlmostEqual(new["words"][1]["s"], 2.3, places=2)
        self.assertNotEqual(new["sha256"], doc["sha256"])
        self.assertEqual(new["text_sha"], doc["text_sha"])
        self.assertEqual(new["text_compatible_shas"], [doc["sha256"]])
        self.assertEqual([(f["kind"], f["a"], f["b"]) for f in new["flags"]],
                         [("no_audio", 2, 2)], "kept, flagged instead of dropped")
        self.assertIn({"a": 0, "b": 1, "kind": "spread"}, new["timing_repairs"])
        self.assertEqual([r["i"] for r in new["timing_repairs"] if r["kind"] == "energy"], [1])
        # a second re-time keeps the chain of compatible shas
        again, _ = T.retime_doc(new, E)
        self.assertEqual(again["sha256"], new["sha256"])
        self.assertIn(doc["sha256"], again["text_compatible_shas"])

    def test_carry_compatible(self):
        doc = {"sha256": "new", "words": [{"w": "a"}, {"w": "b"}]}
        self.assertEqual(T.carry_compatible(doc, {"sha256": "old", "text_compatible_shas": ["x"],
                                                  "words": [{"w": "a"}, {"w": "b"}]}),
                         ["x", "old"])
        self.assertEqual(T.carry_compatible(doc, {"sha256": "old", "words": [{"w": "a"}]}), [])
        self.assertEqual(T.carry_compatible(doc, None), [])

    @needs_fixtures
    def test_cmd_retime(self):
        with tempfile.TemporaryDirectory() as td:
            spans = [("Hello", 1.0, 1.4), ("there.", 1.4, 1.8), ("Again", 3.0, 3.5),
                     ("now.", 3.5, 4.0)]
            media = fixtures.make_speech_fixture(Path(td) / "m", spans, dur=6.0)
            P = fixtures.make_project(Path(td), media)
            # ASR-style stamps: "Again" pulled back to the end of "there."
            stamped = [("Hello", 1.0, 1.4), ("there.", 1.4, 1.8), ("Again", 1.8, 3.5),
                       ("now.", 3.5, 4.0)]
            old = fixtures.fake_transcript(stamped, media="source.mp4")
            env.write_json(P / "transcript.json", old)
            args = argparse.Namespace(project=str(P), backend=None, model=None, language=None,
                                      glossary=None, speakers_vtt=None, file=None, clip=None,
                                      force=False, retime=True)
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                rc, err = run_quiet(T.cmd_transcribe, args)
            self.assertEqual(rc, 0, err)
            new = env.read_json(P / "transcript.json")
            self.assertEqual([(w["i"], w["w"]) for w in new["words"]],
                             [(w["i"], w["w"]) for w in old["words"]])
            self.assertAlmostEqual(new["words"][2]["s"], 3.0, delta=0.03)
            self.assertIn(old["sha256"], new["text_compatible_shas"])
            self.assertIn("re-timed 1 of 4 words", err)
            self.assertTrue((P / "transcript.md").is_file())
            # a caption-only clip transcript is re-timed on its own media's clock
            clip = fixtures.fake_transcript(stamped, media=str(media))
            env.write_json(P / "transcripts" / "c09-short.json", clip)
            args.clip = "c09-short"
            with contextlib.redirect_stdout(io.StringIO()):
                rc, err = run_quiet(T.cmd_transcribe, args)
            self.assertEqual(rc, 0, err)
            new = env.read_json(P / "transcripts" / "c09-short.json")
            self.assertAlmostEqual(new["words"][2]["s"], 3.0, delta=0.03)
            self.assertEqual(new["text_compatible_shas"], [clip["sha256"]])


# ---------------------------------------------------------------------------
# (d) speakers VTT
# ---------------------------------------------------------------------------


class TestSpeakersVtt(unittest.TestCase):
    def test_dash_prefixed_names(self):
        vtt = ("WEBVTT\n\n00:00:01.000 --> 00:00:02.000\n- Alex: Yes.\n\n"
               "00:00:02.000 --> 00:00:03.000\n\u2013 Sam: No.\n\n"
               "00:00:03.000 --> 00:00:04.000\n\u2014 Alex: Maybe.\n")
        cues = T.parse_vtt(vtt)
        self.assertEqual([c["sp"] for c in cues], ["Alex", "Sam", "Alex"])
        self.assertEqual([c["text"] for c in cues], ["Yes.", "No.", "Maybe."])

    def test_colon_sentences_are_not_speakers(self):
        vtt = ("WEBVTT\n\n00:00:01.000 --> 00:00:02.000\nThe rule is simple: charge more.\n\n"
               "00:00:02.000 --> 00:00:03.000\nStep 1: sell.\n\n"
               "00:00:03.000 --> 00:00:04.000\nthen we grew\n\n"
               "00:00:04.000 --> 00:00:05.000\nand grew\n\n"
               "00:00:05.000 --> 00:00:06.000\nand kept going\n")
        cues = T.parse_vtt(vtt)
        self.assertEqual([c["sp"] for c in cues], [None] * 5)
        self.assertEqual(cues[0]["text"], "The rule is simple: charge more.")
        self.assertEqual(cues[1]["text"], "Step 1: sell.")

    def test_repeated_name_is_a_speaker(self):
        vtt = ("WEBVTT\n\n00:00:01.000 --> 00:00:02.000\nJo: we start\n\n"
               "00:00:02.000 --> 00:00:03.000\nand then\n\n"
               "00:00:03.000 --> 00:00:04.000\nJo: we stop\n\n"
               "00:00:04.000 --> 00:00:05.000\nNote to self: buy milk\n\n"
               "00:00:05.000 --> 00:00:06.000\nmore words\n\n"
               "00:00:06.000 --> 00:00:07.000\nlast words\n")
        cues = T.parse_vtt(vtt)
        self.assertEqual([c["sp"] for c in cues], ["Jo", None, "Jo", None, None, None])
        self.assertEqual(cues[3]["text"], "Note to self: buy milk")

    def test_majority_overlap(self):
        cues = [{"s": 0.0, "e": 1.3, "sp": "A", "text": "one"},
                {"s": 1.3, "e": 3.0, "sp": "B", "text": "two"}]
        words = [W("two", 1.0, 2.0)]
        T.assign_speakers(words, cues)
        self.assertEqual(words[0]["sp"], "B")

    def test_turn_snaps_to_sentence_end(self):
        # the cue boundary falls one word late: "Pretty | sure I gave her that."
        cues = [{"s": 0.0, "e": 1.55, "sp": "A", "text": "That works. Pretty"},
                {"s": 1.55, "e": 3.0, "sp": "B", "text": "sure I gave her that."}]
        words = [W("That", 0.0, 0.3), W("works.", 0.3, 0.7), W("Pretty", 1.2, 1.5),
                 W("sure", 1.5, 1.7), W("I", 1.7, 1.8), W("gave", 1.8, 2.0), W("her", 2.0, 2.1),
                 W("that.", 2.1, 2.4)]
        T.assign_speakers(words, cues)
        self.assertEqual([w["sp"] for w in words], ["A", "A"] + ["B"] * 6)
        self.assertEqual(T.speaker_flags(words), [])

    def test_mid_clause_flip_is_flagged(self):
        cues = [{"s": 0.0, "e": 0.9, "sp": "A", "text": "we were going"},
                {"s": 0.9, "e": 3.0, "sp": "B", "text": "to the store today."}]
        words = [W("we", 0.0, 0.3), W("were", 0.3, 0.6), W("going", 0.6, 0.9), W("to", 0.9, 1.0),
                 W("the", 1.0, 1.2), W("store", 1.2, 1.5), W("today.", 1.5, 1.9)]
        T.assign_speakers(words, cues)
        flags = T.speaker_flags(words)
        self.assertEqual(flags, [{"a": 2, "b": 3, "kind": "speaker_uncertain", "sp": "B",
                                  "alt": "A"}])

    def test_label_coverage_report(self):
        words = [W("Hi.", 0.0, 0.3, sp="A"), W("Yo", 0.5, 0.8, sp="B"), W("x", 0.9, 1.0)]
        doc = {"words": T.index_words(words), "flags": [], "lines": [], "backend": "mlx",
               "model": "m", "language": "en", "speakers_vtt": "s.vtt"}
        for w, sp in zip(doc["words"], ("A", "B", None)):
            w["sp"] = sp
        _, err = run_quiet(T._report, doc, [], Path(tempfile.gettempdir()) / "t" / "x.json")
        self.assertIn("speaker labels: 66.7% of words, 0 uncertain turn boundaries", err)


# ---------------------------------------------------------------------------
# (d) loops, whisper.cpp DTW, text files, lines
# ---------------------------------------------------------------------------


class TestLoops(unittest.TestCase):
    def _seq(self, text, p=0.9):
        return [W(t, k * 0.3, k * 0.3 + 0.25, p) for k, t in enumerate(text.split())]

    def test_long_ngram_loop(self):
        unit = "and then we went back to the"
        words = self._seq("so " + unit + " " + unit + " end")
        self.assertEqual(T.find_loops(words), [], "no evidence: a confident repeat is speech")
        words[9]["p"] = words[10]["p"] = words[11]["p"] = 0.1
        words[12]["p"] = words[13]["p"] = words[14]["p"] = words[8]["p"] = 0.1
        flags = T.find_loops(words)
        self.assertEqual([(f["a"], f["b"], f["n"], f["repeats"]) for f in flags],
                         [(1, 14, 7, 2)])

    def test_repeat_thresholds(self):
        self.assertEqual(T.find_loops(self._seq("go go go stop", p=0.1)), [], "1-gram needs 4")
        self.assertEqual(len(T.find_loops(self._seq("go go go go stop", p=0.1))), 1)
        self.assertEqual(T.find_loops(self._seq("a b c a b c end", p=0.1)), [], "3-gram needs 3")
        self.assertEqual(len(T.find_loops(self._seq("a b c a b c a b c end", p=0.1))), 1)
        self.assertEqual(T.find_loops(self._seq("very very very very good", p=0.1)), [])
        self.assertEqual(T.find_loops(self._seq("really really really really", p=0.1)), [])

    def test_silence_evidence(self):
        words = self._seq("ok thank you thank you thank you thank you end")
        self.assertEqual(T.find_loops(words), [])
        self.assertEqual(len(T.find_loops(words, silent={7, 8})), 1)


CPP_DTW = {
    "result": {"language": "en"},
    "transcription": [
        {"offsets": {"from": 0, "to": 0}, "text": "",
         "tokens": [{"text": "[_BEG_]", "p": 0.818514, "t_dtw": -1}]},
        {"offsets": {"from": 0, "to": 850}, "text": " here",
         "tokens": [{"text": " here", "p": 0.100373, "t_dtw": 110}]},
        {"offsets": {"from": 850, "to": 870}, "text": " is",
         "tokens": [{"text": " is", "p": 0.988384, "t_dtw": 156}]},
        {"offsets": {"from": 870, "to": 1420}, "text": " what",
         "tokens": [{"text": " what", "p": 0.984481, "t_dtw": 202}]},
        {"offsets": {"from": 10550, "to": 11080}, "text": " ones",
         "tokens": [{"text": " ones", "p": 0.950064, "t_dtw": 1090},
                    {"text": "[_TT_554]", "p": 0.637755, "t_dtw": -1}]},
        {"offsets": {"from": 11080, "to": 11600}, "text": " week.",
         "tokens": [{"text": " week", "p": 0.9, "t_dtw": 1131},
                    {"text": ".", "p": 0.8, "t_dtw": 1150}]},
    ],
}


class TestWhisperCppDtw(unittest.TestCase):
    def test_t_dtw_words(self):
        self.assertTrue(T.whispercpp_has_dtw(CPP_DTW))
        raw = T.words_from_whispercpp(CPP_DTW)
        words, _ = T.normalize_words(raw)
        self.assertEqual([w["w"] for w in words], ["here", "is", "what", "ones", "week."])
        self.assertEqual([w["s"] for w in words], [1.10, 1.56, 2.02, 10.90, 11.31])
        # end = next word's start, at most CPP_MAX_WORD later
        self.assertEqual(words[0]["e"], 1.56)
        self.assertEqual(words[2]["e"], round(2.02 + T.CPP_MAX_WORD, 3))
        self.assertEqual(words[4]["e"], round(11.31 + T.CPP_MAX_WORD, 3))
        self.assertEqual(words[3]["p"], 0.95, "mean p of the text tokens, specials excluded")

    def test_offsets_without_dtw(self):
        data = json.loads(json.dumps(CPP_DTW))
        for seg in data["transcription"]:
            for t in seg["tokens"]:
                t["t_dtw"] = -1
        self.assertFalse(T.whispercpp_has_dtw(data))
        words, _ = T.normalize_words(T.words_from_whispercpp(data))
        self.assertEqual([(w["s"], w["e"]) for w in words][:3],
                         [(0.0, 0.85), (0.85, 0.87), (0.87, 1.42)])

    def test_postprocess_trims_ends_and_snaps_back(self):
        # voiced 1.0-1.5 and 3.0-3.4 after a pause; DTW stamps run late (+0.2 s) and the first
        # word's end reaches the next word's start
        E = track([(1.0, 1.5, -30.0), (3.0, 3.4, -30.0)], dur=10.0)
        words = [W("here", 1.2, 3.2), W("now.", 3.2, 4.7)]
        out, info = T.postprocess([dict(w) for w in words], E, backend="whispercpp", dtw=True)
        self.assertEqual(out[0]["e"], 1.5, "end capped at the voiced end")
        self.assertEqual(out[0]["s"], 1.0, "late start snapped back to the onset")
        self.assertEqual(out[1]["s"], 3.0)
        self.assertEqual(out[1]["e"], 3.4)
        # other backends snap back too, but only up to SNAP_LATE_MAX (0.30 s): a small model
        # starts the word after a pause late; longer unworded sound stays for the flags
        out, _ = T.postprocess([W("here", 1.2, 1.5)], E, backend="mlx")
        self.assertEqual(out[0]["s"], 1.0)
        out, _ = T.postprocess([W("here", 1.35, 1.5)], E, backend="mlx")
        self.assertEqual(out[0]["s"], 1.35)


class TestTextFiles(unittest.TestCase):
    def test_decode(self):
        txt = "Acme Corp\nJos\u00e9 Doe\n"
        self.assertEqual(T.decode_text(b"\xef\xbb\xbf" + txt.encode("utf-8")), (txt, []))
        self.assertEqual(T.decode_text(txt.encode("utf-16")), (txt, []))
        self.assertEqual(T.decode_text(txt.encode("utf-16-le")), (txt, []))
        self.assertEqual(T.decode_text(txt.encode("utf-16-be")), (txt, []))
        text, warns = T.decode_text(b"Acme\x00 Corp\n")
        self.assertEqual(text, "Acme Corp\n")
        self.assertEqual(warns, ["NUL characters removed"])
        text, warns = T.decode_text(b"Jos\xe9 Doe\n")       # latin-1 bytes
        self.assertIn("\ufffd", text)
        self.assertTrue(any("could not be decoded" in w for w in warns))

    def test_glossary_and_vtt_files(self):
        with tempfile.TemporaryDirectory() as td:
            g = Path(td) / "glossary.txt"
            g.write_bytes("# names\nAcme Corp\nJos\u00e9 Doe\n".encode("utf-16"))
            self.assertEqual(T.glossary_terms(None, g), ["Acme Corp", "Jos\u00e9 Doe"])
            v = Path(td) / "s.vtt"
            v.write_bytes(b"\xef\xbb\xbf" + "WEBVTT\r\n\r\n00:00:01.000 --> 00:00:02.000\r\n"
                          "Alex: Hi.\r\n\r\n00:00:02.000 --> 00:00:03.000\r\nSam: Yo.\r\n"
                          .encode("utf-8"))
            cues = T.parse_vtt(T.read_text_file(v))
            self.assertEqual([c["sp"] for c in cues], ["Alex", "Sam"])


class TestLines(unittest.TestCase):
    def _seq(self, toks, step=0.3):
        return [W(t, round(k * step, 3), round(k * step + step * 0.8, 3)) for k, t in
                enumerate(toks)]

    def test_soft_cap_ends_at_clause_mark(self):
        toks = ["word"] * 40
        toks[26] = "word,"
        lines = T.build_lines(self._seq(toks))
        self.assertEqual([(ln["a"], ln["b"]) for ln in lines], [(0, 26), (27, 39)])
        self.assertTrue(lines[1].get("cont"))
        toks = ["word"] * 40
        toks[33] = "word;"      # none in the last 10 at 30: the next clause mark ends it
        self.assertEqual([(ln["a"], ln["b"]) for ln in T.build_lines(self._seq(toks))],
                         [(0, 33), (34, 39)])

    def test_hard_cap_never_splits_glue(self):
        toks = ["word"] * 50
        toks[44] = "-old"
        toks[43] = "25"
        words = self._seq(toks)
        words[44]["s"] += 0.05          # the largest gap of the last 10 words sits before "-old"
        lines = T.build_lines(words)
        self.assertNotEqual(lines[0]["b"], 43)
        self.assertEqual(lines[0]["b"] + 1, lines[1]["a"])
        self.assertLessEqual(lines[0]["b"] - lines[0]["a"] + 1, T.LINE_HARD_WORDS)

    def test_pause_before_glue_is_not_a_line_break(self):
        words = [W("the", 0.0, 0.2), W("25", 0.2, 0.5), W("-year", 1.3, 1.6), W("-old.", 1.6, 2.0)]
        self.assertEqual([(ln["a"], ln["b"]) for ln in T.build_lines(words)], [(0, 3)])

    def test_continuation_marked_in_markdown(self):
        words = self._seq(["word"] * 50)
        doc = {"words": T.index_words(words), "flags": [], "backend": "mlx", "model": "m",
               "language": "en", "media": "s.mp4"}
        doc["lines"] = T.build_lines(doc["words"])
        md = T.render_md(doc, 20.0).splitlines()
        self.assertTrue(md[3].startswith("[00:00.0] #0-44 word"))
        self.assertTrue(md[4].startswith("[00:13.5] #45-49 \u2026word"), md[4])


# ---------------------------------------------------------------------------
# (d) faster-whisper CUDA fallback, first-use message, cache keys
# ---------------------------------------------------------------------------


class _Word(object):
    def __init__(self, word, start, end, probability):
        self.word, self.start, self.end, self.probability = word, start, end, probability


class _Seg(object):
    def __init__(self, words, end):
        self.words, self.end = words, end


class TestFasterWhisperDevice(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.wav = Path(self.tmp.name) / "a.wav"
        _write_wav(self.wav, 1.0)
        self.made = []
        made = self.made
        lazy = {"fail": False}
        self.lazy = lazy

        class WhisperModel(object):
            def __init__(self, model, device="auto", compute_type="default", download_root=None):
                made.append((device, compute_type))
                if device == "cuda" and not lazy["fail"]:
                    raise RuntimeError("Library libcublas.so.12 is not found or cannot be loaded")
                self.device = device

            def transcribe(self, audio, **kw):
                dev = self.device

                def gen():
                    if dev == "cuda":
                        raise RuntimeError("Could not load library libcudnn_ops.so.9")
                    yield _Seg([_Word(" hi", 0.1, 0.3, 0.9)], 0.3)
                return gen(), types.SimpleNamespace(language="en")

        self.mod = types.ModuleType("faster_whisper")
        self.mod.WhisperModel = WhisperModel
        self.saved = {k: sys.modules.get(k) for k in ("faster_whisper", "ctranslate2")}
        sys.modules["faster_whisper"] = self.mod
        self.old_dir = env.fw_models_dir
        env.fw_models_dir = lambda: Path(self.tmp.name) / "fw"
        self.old_env = os.environ.get(T.DEVICE_ENV)
        T._FW_MODELS.clear()

    def tearDown(self):
        for k, v in self.saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
        env.fw_models_dir = self.old_dir
        if self.old_env is None:
            os.environ.pop(T.DEVICE_ENV, None)
        else:
            os.environ[T.DEVICE_ENV] = self.old_env
        T._FW_MODELS.clear()
        self.tmp.cleanup()

    def test_cuda_construction_failure_falls_back(self):
        os.environ[T.DEVICE_ENV] = "cuda"
        res, err = run_quiet(T.run_faster_whisper, self.wav, "large-v3-turbo", "en", None)
        self.assertEqual(self.made, [("cuda", "float16"), ("cpu", "int8")])
        self.assertEqual((res["device"], res["compute_type"]), ("cpu", "int8"))
        self.assertEqual(res["words"], [[" hi", 0.1, 0.3, 0.9]])
        self.assertIn("CUDA libraries unavailable (Library libcublas.so.12", err)
        self.assertIn("using CPU int8", err)

    def test_cuda_failure_while_decoding_falls_back(self):
        os.environ[T.DEVICE_ENV] = "cuda"
        self.lazy["fail"] = True        # the model loads; the first decode fails
        res, err = run_quiet(T.run_faster_whisper, self.wav, "large-v3-turbo", "en", None)
        self.assertEqual(res["device"], "cpu")
        self.assertIn("libcudnn_ops", err)

    def test_device_env(self):
        os.environ[T.DEVICE_ENV] = "cpu"
        self.assertEqual(T._fw_device(), ("cpu", "int8"))
        os.environ[T.DEVICE_ENV] = "cuda"
        self.assertEqual(T._fw_device(), ("cuda", "float16"))
        os.environ[T.DEVICE_ENV] = "auto"
        sys.modules["ctranslate2"] = types.SimpleNamespace(get_cuda_device_count=lambda: 0)
        self.assertEqual(T._fw_device(), ("cpu", "int8"))
        sys.modules["ctranslate2"] = types.SimpleNamespace(get_cuda_device_count=lambda: 1)
        self.assertEqual(T._fw_device(), ("cuda", "float16"))

    def test_cpu_failure_is_not_retried(self):
        os.environ[T.DEVICE_ENV] = "cpu"
        self.mod.WhisperModel.__init__ = lambda *a, **k: (_ for _ in ()).throw(OSError("bad"))
        with self.assertRaises(OSError):
            run_quiet(T.run_faster_whisper, self.wav, "large-v3-turbo", "en", None)

    def test_first_use_message_only_when_not_cached(self):
        os.environ[T.DEVICE_ENV] = "cpu"
        old = T._fw_cached
        try:
            T._fw_cached = lambda model, root: True
            _, err = run_quiet(T.run_faster_whisper, self.wav, "large-v3-turbo", "en", None)
            self.assertNotIn("first use downloads", err)
            T._FW_MODELS.clear()
            T._fw_cached = lambda model, root: False
            _, err = run_quiet(T.run_faster_whisper, self.wav, "large-v3-turbo", "en", None)
            self.assertIn("first use downloads", err)
        finally:
            T._fw_cached = old

    def test_fw_cached_reads_the_download_root(self):
        root = Path(self.tmp.name) / "hf"
        model_dir = Path(self.tmp.name) / "local-model"
        model_dir.mkdir()
        self.assertTrue(T._fw_cached(str(model_dir), root), "a model directory is local")
        self.assertFalse(T._fw_cached("large-v3-turbo", root))


class TestCacheKeys(unittest.TestCase):
    def test_key_material(self):
        m = T.asr_key_material("faster-whisper", None, "en", ["Acme"])
        self.assertEqual(m["backend"], "faster-whisper")
        self.assertEqual(m["model"], "large-v3-turbo")
        self.assertEqual(m["options"]["hallucination_silence_threshold"], T.HALLUCINATION_SILENCE)
        self.assertEqual(m["options"]["vad_parameters"], T.FW_VAD)
        self.assertIn("backend_version", m)
        old = T.HALLUCINATION_SILENCE
        try:
            T.HALLUCINATION_SILENCE = 1.0
            self.assertNotEqual(env.stable_hash(T.asr_key_material("faster-whisper", None, "en",
                                                                   ["Acme"])),
                                env.stable_hash(m))
        finally:
            T.HALLUCINATION_SILENCE = old
        cpp = T.runner_options("whispercpp", "/m/ggml-large-v3-turbo.bin", None)
        self.assertEqual((cpp["dtw"], cpp["nfa"], cpp["words"]), ("large.v3.turbo", True, "t_dtw"))
        self.assertIsInstance(T.POST_VERSION, int)
        self.assertGreaterEqual(T.POST_VERSION, 2)

    def _legacy(self, cache, wav, options):
        sha = env.sha256_file(wav)
        key = env.stable_hash({"v": 1, "wav": sha, "backend": "faster-whisper",
                               "model": "large-v3-turbo", "language": "en", "glossary": ["Acme"]})
        env.write_json(cache / ("asr-%s.json" % key[:24]),
                       {"key": key, "backend": "faster-whisper", "model": "large-v3-turbo",
                        "options": options, "words": [[" cached", 0.1, 0.3, 0.9]]})

    def test_legacy_cache_reused_when_options_match(self):
        with tempfile.TemporaryDirectory() as td:
            wav = Path(td) / "a.wav"
            _write_wav(wav)
            cache = Path(td) / "cache"
            self._legacy(cache, wav, T.runner_options("faster-whisper", "large-v3-turbo", "Acme."))
            with fake_backend([[" fresh", 0.1, 0.3, 0.9]]) as runner:
                res, err = run_quiet(T.transcribe_wav, wav, "faster-whisper", None, "en", ["Acme"],
                                     cache_dir=cache)
                self.assertEqual(runner.calls, [])
                self.assertEqual(res["words"], [[" cached", 0.1, 0.3, 0.9]])
                self.assertIn("reusing words cached by an earlier version", err)
                # it is filed under the new key: the next run finds it directly
                res, err = run_quiet(T.transcribe_wav, wav, "faster-whisper", None, "en",
                                     ["Acme"], cache_dir=cache)
                self.assertNotIn("earlier version", err)
                self.assertEqual(runner.calls, [])

    def test_legacy_cache_ignored_when_options_differ(self):
        with tempfile.TemporaryDirectory() as td:
            wav = Path(td) / "a.wav"
            _write_wav(wav)
            cache = Path(td) / "cache"
            opts = T.runner_options("faster-whisper", "large-v3-turbo", "Acme.")
            opts["hallucination_silence_threshold"] = None
            self._legacy(cache, wav, opts)
            with fake_backend([[" fresh", 0.1, 0.3, 0.9]]) as runner:
                res, _ = run_quiet(T.transcribe_wav, wav, "faster-whisper", None, "en", ["Acme"],
                                   cache_dir=cache)
            self.assertEqual(len(runner.calls), 1)
            self.assertEqual(res["words"], [[" fresh", 0.1, 0.3, 0.9]])


# ---------------------------------------------------------------------------
# (d) QA entry point and the words command
# ---------------------------------------------------------------------------


@needs_fixtures
class TestTranscribeMedia(unittest.TestCase):
    def test_gate_and_repair_on_media_clock(self):
        with tempfile.TemporaryDirectory() as td:
            spans = [("Hello", 1.0, 1.4), ("there.", 1.4, 1.8), ("Again", 3.0, 3.5)]
            media = fixtures.make_speech_fixture(Path(td) / "m", spans, dur=6.0)
            canned = [[" Thank", 0.2, 0.5, 0.2], [" Hello", 1.0, 1.4, 0.9],
                      [" there.", 1.4, 1.8, 0.9], [" Again", 1.8, 3.5, 0.9]]
            with fake_backend(canned) as runner:
                words = T.transcribe_media(media, "faster-whisper", None, "en")
                raw = T.transcribe_media(media, "faster-whisper", None, "en", repair=False,
                                         fill_holes=False)
            self.assertEqual([w["w"] for w in raw], ["Thank", "Hello", "there.", "Again"])
            self.assertEqual([w["w"] for w in words], ["Hello", "there.", "Again"])
            self.assertAlmostEqual(words[2]["s"], 3.0, delta=0.03)
            self.assertEqual([w["i"] for w in words], [0, 1, 2])
            self.assertEqual(len(runner.calls), 2, "no gaps to re-listen to")


class TestWordsCommand(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.P = Path(self.tmp.name) / "project"
        self.P.mkdir()
        words = T.index_words([W("w%d" % k, k * 0.5, k * 0.5 + 0.4) for k in range(10)])
        env.write_json(self.P / "transcript.json", {
            "version": 1, "words": words, "lines": T.build_lines(words), "flags": [
                {"a": 4, "b": 4, "kind": "no_audio", "w": "w4"}],
            "timing_repairs": [{"a": 0, "b": 1, "kind": "spread"},
                               {"i": 3, "kind": "energy", "tag": "s", "s0": 1.0, "e0": 1.9,
                                "s": 1.5, "e": 1.9}]})

    def tearDown(self):
        self.tmp.cleanup()

    def _run(self, a, b):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = find.cmd_words(argparse.Namespace(project=str(self.P), a=a, b=b, clip=None))
        return rc, out.getvalue()

    def test_reversed_and_negative_ranges_are_errors(self):
        for a, b in (("5", "2"), ("-1", "3"), ("2", "-3")):
            with self.assertRaises(env.HCError) as cm:
                self._run(a, b)
            self.assertEqual(cm.exception.code, 1)
        rc, out = self._run("2", "99")
        self.assertEqual(rc, 0)
        self.assertEqual(len(out.strip().splitlines()), 9)

    def test_rows_mark_repairs_and_flags(self):
        rc, out = self._run("0", "4")
        rows = out.splitlines()
        self.assertRegex(rows[1], r"\s~\s")
        self.assertRegex(rows[4], r"\sr\s")
        self.assertRegex(rows[5], r"\s!\s")


if __name__ == "__main__":
    unittest.main()
