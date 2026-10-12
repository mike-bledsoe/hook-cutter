"""transcribe: backend output parsing, normalization, flags, VTT speakers, lines, markdown,
caching and the command path (canned ASR; real ASR only with --slow)."""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import re
import sys
import tempfile
import types
import unittest
import wave
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts"))
sys.path.insert(0, str(HERE))

from hookcutter import env, transcribe as T  # noqa: E402

try:
    import fixtures  # noqa: E402
except Exception:  # noqa: BLE001 - media tests skip when the shared fixtures are unavailable
    fixtures = None

slow = unittest.skipUnless(os.environ.get("HC_SLOW") == "1", "slow test (run: hc test --slow)")
needs_fixtures = unittest.skipIf(fixtures is None, "tests/fixtures.py unavailable")

MLX_RESULT = {
    "language": "en",
    "segments": [
        {"start": 0.0, "end": 2.0, "words": [
            {"word": " Alright,", "start": 0.10, "end": 0.42, "probability": 0.93},
            {"word": " today", "start": 0.42, "end": 0.70, "probability": 0.99},
            {"word": " [Music]", "start": 0.70, "end": 1.20, "probability": 0.40},
            {"word": " we", "start": 1.20, "end": 1.30, "probability": 0.97},
        ]},
        {"start": 1.2, "end": 3.0, "words": [
            {"word": " we", "start": 1.2, "end": 1.3, "probability": 0.97},   # window overlap dupe
            {"word": " ship.", "start": 1.30, "end": 1.30, "probability": 0.88},  # zero length
            {"word": " Next", "start": 1.35, "end": 1.60, "probability": 0.90},
        ]},
    ],
}

WCPP_JSON = {
    "result": {"language": "en"},
    "transcription": [
        {"offsets": {"from": 0, "to": 30}, "text": "",
         "tokens": [{"text": "[_BEG_]", "p": 0.6}]},
        {"offsets": {"from": 30, "to": 210}, "text": " the",
         "tokens": [{"text": " the", "p": 0.6}]},
        {"offsets": {"from": 350, "to": 640}, "text": " show.",
         "tokens": [{"text": " show", "p": 0.9}, {"text": ".", "p": 0.7},
                    {"text": "[_TT_32]", "p": 0.1}]},
        {"offsets": {"from": 1000, "to": 1600}, "text": " [BLANK_AUDIO]", "tokens": []},
        {"offsets": {"from": 2000, "to": 2600}, "text": " two pair",
         "tokens": [{"text": " two", "p": 0.5}, {"text": " pair", "p": 0.7}]},
    ],
}


def W(text, s, e, p=0.9, sp=None):
    return {"w": text, "s": s, "e": e, "p": p, "sp": sp}


def seq(text, start=0.0, step=0.3, p=0.9):
    return [W(t, round(start + k * step, 3), round(start + k * step + step * 0.8, 3), p)
            for k, t in enumerate(text.split())]


class TestBackendParsing(unittest.TestCase):
    def test_mlx(self):
        raw = T.words_from_mlx(MLX_RESULT)
        self.assertEqual(len(raw), 7)
        self.assertEqual(raw[0], [" Alright,", 0.10, 0.42, 0.93])
        events = []
        words, spread = T.normalize_words(raw, offset=0.178, events=events)
        self.assertEqual([w["w"] for w in words], ["Alright,", "today", "we", "ship.", "Next"])
        self.assertEqual(words[0]["s"], 0.278)
        self.assertEqual(words[0]["e"], 0.598)
        self.assertEqual(events, [{"s": 0.878, "e": 1.378, "text": "[Music]"}])
        ship = words[3]
        self.assertEqual((ship["s"], ship["e"]), (1.478, 1.528))  # min(s + 0.08, next.s)
        self.assertEqual(spread, [])

    def test_faster_whisper(self):
        seg = types.SimpleNamespace(words=[
            types.SimpleNamespace(word=" Hello", start=1.0, end=1.4, probability=0.8),
            types.SimpleNamespace(word=" world.", start=1.4, end=1.9, probability=0.7)])
        seg2 = {"words": [{"word": " Again", "start": 2.5, "end": 2.9, "probability": 0.95}]}
        raw = T.words_from_faster([seg, seg2])
        self.assertEqual(raw, [[" Hello", 1.0, 1.4, 0.8], [" world.", 1.4, 1.9, 0.7],
                               [" Again", 2.5, 2.9, 0.95]])
        words, _ = T.normalize_words(raw, offset=1.0)
        self.assertEqual([(w["w"], w["s"], w["e"]) for w in words],
                         [("Hello", 2.0, 2.4), ("world.", 2.4, 2.9), ("Again", 3.5, 3.9)])

    def test_whispercpp(self):
        raw = T.words_from_whispercpp(WCPP_JSON)
        texts = [r[0] for r in raw]
        self.assertEqual(texts, ["", " the", " show.", " [BLANK_AUDIO]", " two", " pair"])
        show = raw[2]
        self.assertAlmostEqual(show[1], 0.35)
        self.assertAlmostEqual(show[2], 0.64)
        self.assertAlmostEqual(show[3], 0.8)  # mean of text tokens, specials excluded
        # multi-word segment: times shared by length (3 + 4 letters over 0.6 s)
        self.assertAlmostEqual(raw[4][1], 2.0)
        self.assertAlmostEqual(raw[4][2], 2.0 + 0.6 * 3 / 7)
        self.assertAlmostEqual(raw[5][1], 2.0 + 0.6 * 3 / 7)
        self.assertAlmostEqual(raw[5][2], 2.6)
        events = []
        words, _ = T.normalize_words(raw, offset=0.0, events=events)
        self.assertEqual([w["w"] for w in words], ["the", "show.", "two", "pair"])
        self.assertEqual(events, [], "blank audio is not an event")

    def test_whispercpp_command(self):
        help_text = ("  -oj, --output-json\n  -ojf, --output-json-full\n  -dtw MODEL --dtw MODEL\n"
                     "  -nfa, --no-flash-attn\n  --prompt PROMPT\n  --carry-initial-prompt\n")
        cmd = T.whispercpp_command("whisper-cli", "/m/ggml-large-v3-turbo.bin", Path("a.wav"),
                                   Path("/tmp/x"), None, "Acme, Widgetron.", help_text)
        self.assertIn("-ojf", cmd)
        self.assertEqual(cmd[cmd.index("-ml") + 1], "1")
        self.assertIn("-sow", cmd)
        self.assertEqual(cmd[cmd.index("-l") + 1], "auto")
        self.assertEqual(cmd[cmd.index("-dtw") + 1], "large.v3.turbo")
        self.assertIn("-nfa", cmd)
        self.assertNotIn("--prompt", cmd)  # whisper.cpp prompts corrupt -ml 1 word timing
        cmd = T.whispercpp_command("whisper-cli", "/m/ggml-small.bin", Path("a.wav"),
                                   Path("/tmp/x"), "en", None, "-oj\n")
        self.assertNotIn("-dtw", cmd)
        self.assertNotIn("-ojf", cmd)
        self.assertEqual(cmd[cmd.index("-l") + 1], "en")


class TestNormalize(unittest.TestCase):
    def test_sort_dedupe_monotonic(self):
        raw = [[" b", 2.0, 2.5, 0.9], [" a", 1.0, 2.2, 0.9], [" b", 2.004, 2.5, 0.8],
               ["  ", 3.0, 3.1, 0.9], [" c", 3.0, 2.9, None], [" d", None, 4.0, 0.9]]
        words, _ = T.normalize_words(raw)
        self.assertEqual([w["w"] for w in words], ["a", "b", "c"])
        self.assertEqual(words[0]["e"], 2.0, "overlap clamped to next start")
        self.assertEqual((words[2]["s"], words[2]["e"]), (3.0, 3.08))
        self.assertIsNone(words[2]["p"])
        for k, w in enumerate(words):
            self.assertLessEqual(w["s"], w["e"])
            if k + 1 < len(words):
                self.assertLessEqual(w["e"], words[k + 1]["s"])

    def test_identical_starts_are_spread(self):
        raw = [[" one", 1.0, 1.0, 0.9], [" two", 1.0, 1.0, 0.9], [" three", 1.0, 1.0, 0.9],
               [" four", 1.5, 1.8, 0.9]]
        words, spread = T.normalize_words(raw)
        self.assertEqual(spread, [(0, 2)])
        # span = s + 0.08 per word, shared by letter count (3, 3, 5)
        self.assertEqual([(w["s"], w["e"]) for w in words[:3]],
                         [(1.0, 1.065), (1.065, 1.131), (1.131, 1.24)])
        # the common mlx shape: a zero-length segment-initial word plus the next word
        words, spread = T.normalize_words([[" He", 8.18, 8.18, 0.2], [" has", 8.18, 8.76, 0.9],
                                           [" a", 8.76, 8.96, 0.9]])
        self.assertEqual([(w["s"], w["e"]) for w in words],
                         [(8.18, 8.412), (8.412, 8.76), (8.76, 8.96)])
        doc = T.build_transcript({"backend": "mlx", "model": "m",
                                  "words": [[" He", 8.18, 8.18, 0.2], [" has", 8.18, 8.76, 0.9]]},
                                 offset=0.0, media="s.mp4", fingerprint=None)
        self.assertEqual(doc["flags"], [], "timing repairs are not transcript flags")
        self.assertEqual(doc["timing_repairs"], [{"a": 0, "b": 1, "kind": "spread"}])

    def test_nonspeech(self):
        raw = [[" (audience", 0.0, 0.2, 0.5], [" laughs)", 0.2, 0.5, 0.5], [" so", 0.6, 0.8, 0.9],
               [" \u266a", 0.9, 1.0, 0.2], [" (and", 1.0, 1.1, 0.9], [" this", 1.1, 1.2, 0.9],
               [" is", 1.2, 1.3, 0.9], [" really", 1.3, 1.4, 0.9], [" important)", 1.4, 1.6, 0.9],
               [" [_TT_12]", 1.7, 1.7, 0.1]]
        events = []
        words, _ = T.normalize_words(raw, events=events)
        self.assertEqual([w["w"] for w in words],
                         ["so", "(and", "this", "is", "really", "important)"])
        self.assertEqual(events, [{"s": 0.0, "e": 0.5, "text": "(audience laughs)"}])

    def test_prompt_echo(self):
        words = seq("hello there Acme Corp, Jane Doe, Widgetron. bye")
        # spoken with confidence, the whole glossary in order is speech, not an echo
        kept, n = T.drop_prompt_echo(words, "Acme Corp, Jane Doe, Widgetron.")
        self.assertEqual((n, len(kept)), (0, 8))
        words[2]["p"] = 0.0            # an echo is written with near-zero confidence on its first word
        kept, n = T.drop_prompt_echo(words, "Acme Corp, Jane Doe, Widgetron.")
        self.assertEqual(n, 5)
        self.assertEqual([w["w"] for w in kept], ["hello", "there", "bye"])
        kept, n = T.drop_prompt_echo(words, None)
        self.assertEqual(n, 0)


class TestFlags(unittest.TestCase):
    def test_loops(self):
        words = seq("we said the the the the end and you know you know you know you know ok")
        # Real speech repeats words too: a loop needs one repeat with p < 0.5 (or in silence).
        self.assertEqual(T.find_loops(words), [])
        words[4]["p"] = 0.3
        words[12]["p"] = words[13]["p"] = 0.2     # one whole "you know" repeat is unsure
        flags = T.find_loops(words)
        self.assertEqual([(f["a"], f["b"], f["n"], f["repeats"]) for f in flags],
                         [(2, 5, 1, 4), (8, 15, 2, 4)])
        self.assertEqual(flags[0]["kind"], "repetition_loop")
        self.assertEqual(T.find_loops(seq("no no no but yes", p=0.2)), [])
        # punctuation variants still count as the same token
        self.assertEqual(len(T.find_loops(seq("Thank you. Thank you. Thank you. Thank you.",
                                              p=0.3))), 1)

    def test_low_confidence(self):
        words = seq("We met Greptile Labs. Zed is fine and I paid 70 one times")
        words[2]["p"] = 0.3      # Greptile, mid-sentence -> flagged
        words[3]["p"] = 0.4      # Labs. -> merged into the same flag
        words[4]["p"] = 0.2      # Zed starts a sentence -> not a name signal
        words[8]["p"] = 0.1      # I -> never
        words[10]["p"] = 0.2     # 70 -> number
        words[11]["p"] = 0.2     # "one" alone is not a claim
        flags = T.low_confidence(words)
        self.assertEqual([(f["a"], f["b"], f["w"], f["p"]) for f in flags],
                         [(2, 3, "Greptile Labs.", 0.3), (10, 10, "70", 0.2)])

    def test_missing_glossary_terms(self):
        words = seq("we use Acme Corp every day")
        self.assertEqual(T.missing_glossary_terms(words, ["Acme Corp", "Widgetron"]), ["Widgetron"])


class TestSpeakers(unittest.TestCase):
    VTT = ("WEBVTT\n\nNOTE exported\n\n1\n00:00:01.000 --> 00:00:03.000\nAlex Doe: Hello there.\n\n"
           "2\n00:00:03.500 --> 00:00:05.500 align:start\n<v Sam>Hi Alex.</v>\n\n"
           "3\n00:00:06.000 --> 00:00:08.000\nAlex Doe: First part.\nSam: Other part.\n\n"
           "4\n01:00:00.000 --> 01:00:01.000\nno speaker here\n")

    def test_parse_vtt(self):
        cues = T.parse_vtt(self.VTT)
        self.assertEqual([(c["s"], c["e"], c["sp"]) for c in cues],
                         [(1.0, 3.0, "Alex Doe"), (3.5, 5.5, "Sam"), (6.0, 7.0, "Alex Doe"),
                          (7.0, 8.0, "Sam"), (3600.0, 3601.0, None)])
        self.assertEqual(cues[1]["text"], "Hi Alex.")

    def test_parse_srt_comma_times(self):
        cues = T.parse_vtt("1\n00:00:02,500 --> 00:00:04,000\nJo: yes\n")
        self.assertEqual([(c["s"], c["e"], c["sp"]) for c in cues], [(2.5, 4.0, "Jo")])

    def test_assign(self):
        cues = T.parse_vtt(self.VTT)
        words = [W("Hello", 1.1, 1.4), W("Hi", 3.2, 3.35), W("gap", 5.9, 5.95), W("far", 20.0, 20.5),
                 W("Other", 7.2, 7.6)]
        n = T.assign_speakers(words, cues)
        self.assertEqual([w["sp"] for w in words], ["Alex Doe", "Sam", "Alex Doe", None, "Sam"])
        self.assertEqual(n, 4)


class TestLinesAndMarkdown(unittest.TestCase):
    def test_lines(self):
        words = seq("Mr. Smith said e.g. this works. Then we left")
        words += [W("later", 4.0, 4.3), W("Hi.", 4.4, 4.6, sp="B"), W("Yo", 4.7, 4.9, sp="B")]
        lines = T.build_lines(words)
        self.assertEqual([(ln["a"], ln["b"]) for ln in lines], [(0, 5), (6, 8), (9, 9), (10, 10),
                                                                 (11, 11)])
        self.assertEqual(lines[3]["sp"], "B")
        # 30 words is a soft cap (a line ends at a clause mark from there); with no clause mark
        # the hard cap of 45 ends it, and the line after it is a continuation.
        long = seq(" ".join(["word"] * 65))
        lines = T.build_lines(long)
        self.assertEqual([(ln["a"], ln["b"]) for ln in lines], [(0, 44), (45, 64)])
        self.assertEqual([bool(ln.get("cont")) for ln in lines], [False, True])

    def _doc(self, words, duration=None, **kw):
        asr = {"backend": "mlx", "model": "mlx-community/whisper-large-v3-turbo",
               "backend_version": "0.4.3", "language": "en", "glossary": ["Acme"],
               "words": [[" " + w["w"], w["s"], w["e"], w["p"]] for w in words]}
        asr.update(kw)
        return T.build_transcript(asr, offset=0.0, media="source.mp4", fingerprint="fp",
                                  duration=duration)

    def test_markdown(self):
        words = seq("Alright, today we ship. the the the the yes", start=2.2)
        words[6]["p"] = 0.3    # a loop needs one low-probability repeat (or one in silence)
        doc = self._doc(words, duration=3487.0)
        md = T.render_md(doc, 3487.0)
        lines = md.splitlines()
        self.assertEqual(lines[0], "# Transcript: source.mp4 (58:07, 9 words, mlx large-v3-turbo, en)")
        self.assertEqual(lines[1], '# Lines: [mm:ss.s] #first-last SPEAKER: text. Indices are '
                                   'inclusive word indices for clips.json "words": [first, last].')
        self.assertEqual(lines[3], "[00:02.2] #0-3 Alright, today we ship.")
        self.assertTrue(lines[4].startswith("[00:03.4] #4-8 the the the the yes"), lines[4])
        self.assertIn("repetition loop", lines[4])
        self.assertNotIn("\u2013", md)
        self.assertNotIn("\u2014", md)

    def test_markdown_hours_and_speakers(self):
        words = [W("Hello", 3700.0, 3700.4, sp="Alex"), W("there.", 3700.4, 3700.8, sp="Alex"),
                 W("Hi.", 3701.5, 3701.8)]
        doc = self._doc(words, duration=3900.0)
        doc["words"][0]["sp"] = doc["words"][1]["sp"] = "Alex"
        doc["lines"] = T.build_lines(doc["words"])
        md = T.render_md(doc, 3900.0).splitlines()
        self.assertIn("(1:05:00, 3 words,", md[0])
        self.assertEqual(md[3], "[1:01:40.0] #0-1 Alex: Hello there.")
        self.assertEqual(md[4], "[1:01:41.5] #2-2 Hi.")

    def test_build_transcript_schema(self):
        words = seq("We met Acme today. Acme Acme Acme Acme")
        words[6]["p"] = 0.2    # loop evidence (see test_loops)
        doc = self._doc(words)
        for k in ("version", "backend", "model", "backend_version", "language", "glossary",
                  "audio_offset", "source_fingerprint", "media", "sha256", "created", "words",
                  "lines", "flags", "text_sha", "text_compatible_shas", "post_version",
                  "timing_repairs", "events"):
            self.assertIn(k, doc)
        self.assertEqual(list(doc["words"][0].keys()), ["i", "w", "s", "e", "p", "sp"])
        self.assertEqual([w["i"] for w in doc["words"]], list(range(len(words))))
        self.assertEqual(doc["sha256"], T.transcript_sha(doc["words"]))
        self.assertEqual(doc["sha256"], self._doc(words)["sha256"], "sha is deterministic")
        # the unsure "Acme" repeat is also a low-confidence name
        self.assertEqual([f["kind"] for f in doc["flags"]], ["repetition_loop", "low_confidence"])
        json.dumps(doc)  # serializable


class TestConfig(unittest.TestCase):
    def test_glossary_terms(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "glossary.txt"
            p.write_text("# names\nAcme Corp\n\n  Jane   Doe \nacme corp\nC#\n", encoding="utf-8")
            self.assertEqual(T.glossary_terms("Widgetron, Jane Doe; Zed", p),
                             ["Acme Corp", "Jane Doe", "C#", "Widgetron", "Zed"])
            self.assertEqual(T.glossary_terms(None, Path(td) / "missing.txt"), [])
            self.assertEqual(T.glossary_terms(["A", "B, C"]), ["A", "B", "C"])

    def test_build_prompt(self):
        self.assertIsNone(T.build_prompt([]))
        self.assertEqual(T.build_prompt(["Acme", "Jane Doe"]), "Acme, Jane Doe.")
        long = T.build_prompt(["term%03d" % k for k in range(500)])
        self.assertLessEqual(len(long), T.PROMPT_MAX_CHARS + 1)
        self.assertTrue(long.endswith("."))

    def test_resolve(self):
        self.assertEqual(T.resolve_backend("faster_whisper"), "faster-whisper")
        self.assertEqual(T.resolve_backend("whisper.cpp"), "whispercpp")
        self.assertIn(T.resolve_backend("auto"), ("mlx", "faster-whisper"))
        with self.assertRaises(env.HCError):
            T.resolve_backend("nope")
        self.assertEqual(T.resolve_model("mlx", None), "mlx-community/whisper-large-v3-turbo")
        self.assertEqual(T.resolve_model("mlx", "small"), "mlx-community/whisper-small-mlx")
        self.assertEqual(T.resolve_model("faster-whisper", "turbo"), "large-v3-turbo")
        self.assertEqual(T.resolve_model("faster-whisper", "/x/model"), "/x/model")
        old = os.environ.pop(T.WHISPERCPP_MODEL_ENV, None)
        try:
            with self.assertRaises(env.HCError) as cm:
                T.resolve_model("whispercpp", None)
            self.assertEqual(cm.exception.code, 3)
        finally:
            if old is not None:
                os.environ[T.WHISPERCPP_MODEL_ENV] = old

    def test_model_short_name(self):
        self.assertEqual(T.model_short_name("mlx-community/whisper-large-v3-turbo"), "large-v3-turbo")
        self.assertEqual(T.model_short_name("mlx-community/whisper-small-mlx"), "small")
        self.assertEqual(T.model_short_name("large-v3-turbo"), "large-v3-turbo")
        self.assertEqual(T.model_short_name("/m/ggml-large-v3-turbo.bin"), "large-v3-turbo")


class FakeRunner(object):
    """Stands in for an ASR backend: returns canned words on the wav clock."""

    def __init__(self, words):
        self.words = words
        self.calls = []

    def __call__(self, wav, model, language, prompt, progress=True):
        self.calls.append({"wav": str(wav), "model": model, "language": language, "prompt": prompt})
        return {"words": [list(w) for w in self.words], "language": language or "en",
                "options": {"fake": True}}


@contextlib.contextmanager
def fake_backend(words, name="faster-whisper"):
    runner = FakeRunner(words)
    old = T.RUNNERS[name]
    T.RUNNERS[name] = runner
    try:
        yield runner
    finally:
        T.RUNNERS[name] = old


def _write_wav(path, seconds=1.0):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x00" * int(16000 * seconds))


class TestCache(unittest.TestCase):
    def test_transcribe_wav_cache(self):
        with tempfile.TemporaryDirectory() as td:
            wav = Path(td) / "a.wav"
            _write_wav(wav)
            cache = Path(td) / "cache"
            with fake_backend([[" hi", 0.1, 0.3, 0.9]]) as runner:
                a = T.transcribe_wav(wav, "faster-whisper", None, "en", ["Acme"], cache_dir=cache,
                                     progress=False)
                b = T.transcribe_wav(wav, "faster-whisper", None, "en", ["Acme"], cache_dir=cache,
                                     progress=False)
                self.assertEqual(len(runner.calls), 1)
                self.assertEqual(a["words"], b["words"])
                self.assertEqual(runner.calls[0]["prompt"], "Acme.")
                self.assertEqual(runner.calls[0]["model"], "large-v3-turbo")
                T.transcribe_wav(wav, "faster-whisper", None, "en", ["Other"], cache_dir=cache,
                                 progress=False)
                T.transcribe_wav(wav, "faster-whisper", None, "auto", ["Acme"], cache_dir=cache,
                                 progress=False)
                T.transcribe_wav(wav, "faster-whisper", None, "en", ["Acme"], cache_dir=cache,
                                 force=True, progress=False)
                self.assertEqual(len(runner.calls), 4)
                self.assertIsNone(runner.calls[2]["language"], "auto means detect")
            self.assertEqual(a["backend"], "faster-whisper")
            self.assertTrue(a["key"])


@needs_fixtures
class TestCommand(unittest.TestCase):
    """cmd_transcribe on a real lavfi fixture with a canned backend."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        cls.flashes = [1.0, 2.5, 4.0]
        cls.media = fixtures.make_av_fixture(root / "media", fps="25", dur=6, flashes=cls.flashes,
                                             audio_delay=0.2)
        cls.P = fixtures.make_project(root, cls.media)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def _args(self, **kw):
        base = dict(project=str(self.P), backend="faster-whisper", model=None, language="en",
                    glossary=None, speakers_vtt=None, file=None, clip=None, force=False)
        base.update(kw)
        return argparse.Namespace(**base)

    def _run(self, args):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = T.cmd_transcribe(args)
        return rc, out.getvalue(), err.getvalue()

    def test_wav_clock_plus_audio_start_is_src(self):
        """The cached wav plus source.audio.start lands every click on its container time."""
        import numpy as np
        from hookcutter.project import Project
        proj = Project(self.P)
        wav, offset = T.ensure_wav(proj)
        self.assertTrue(T.wav_ok(wav))
        self.assertAlmostEqual(offset, proj.source()["audio"]["start"])
        self.assertGreater(offset, 0.1)
        samples = T.load_wav(wav)
        onsets = fixtures.click_onsets(np.asarray(samples), sr=16000)
        times = fixtures.fixture_meta(self.media)["flash_times"]
        self.assertEqual(len(onsets), len(times))
        for onset, t in zip(onsets, times):
            self.assertLess(abs(onset + offset - t), 0.003, (onset, offset, t))

    def test_command_end_to_end(self):
        from hookcutter.project import Project
        proj = Project(self.P)
        offset = proj.source()["audio"]["start"]
        canned = [[" Hello", 0.5, 0.9, 0.9], [" there.", 0.9, 1.3, 0.9], [" Acme", 2.0, 2.4, 0.3],
                  [" rocks.", 2.4, 2.8, 0.9]]
        vtt = self.P / "speakers.vtt"
        vtt.write_text("WEBVTT\n\n00:00:00.000 --> 00:00:01.800\nAlex: Hello there.\n\n"
                       "00:00:01.900 --> 00:00:04.000\nSam: Acme rocks.\n", encoding="utf-8")
        (self.P / "glossary.txt").write_text("Acme\n", encoding="utf-8")
        with fake_backend(canned) as runner:
            rc, out, err = self._run(self._args(speakers_vtt=str(vtt)))
        self.assertEqual(rc, 0, err)
        self.assertEqual(runner.calls[0]["prompt"], "Acme.")
        doc = env.read_json(self.P / "transcript.json")
        self.assertEqual(doc["audio_offset"], round(offset, 6))
        self.assertEqual(doc["words"][0]["s"], round(0.5 + offset, 3))
        self.assertEqual([w["sp"] for w in doc["words"]], ["Alex", "Alex", "Sam", "Sam"])
        self.assertEqual(doc["glossary"], ["Acme"])
        self.assertEqual(doc["source_fingerprint"], proj.source()["fingerprint"])
        self.assertEqual(doc["media"], proj.source()["path"])
        md = (self.P / "transcript.md").read_text(encoding="utf-8")
        self.assertIn("#0-1 Alex: Hello there.", md)
        self.assertIn("#2-3 Sam: Acme rocks.", md)
        self.assertEqual(out.strip(), str(self.P / "transcript.json"))
        # rerun without --speakers-vtt reuses the recorded VTT; cached ASR is not re-run
        with fake_backend(canned) as runner:
            rc, _, err = self._run(self._args())
        self.assertEqual(rc, 0)
        self.assertEqual(runner.calls, [])
        self.assertIn("reusing speakers", err)
        self.assertEqual(env.read_json(self.P / "transcript.json")["words"][2]["sp"], "Sam")
        # "--speakers-vtt none" drops the speakers again
        with fake_backend(canned):
            rc, _, err = self._run(self._args(speakers_vtt="none"))
        self.assertEqual(rc, 0, err)
        doc = env.read_json(self.P / "transcript.json")
        self.assertEqual([w["sp"] for w in doc["words"]], [None] * 4)
        self.assertIsNone(doc["speakers_vtt"])

    def test_caption_only(self):
        # The fixture audio is pink noise plus 10 ms clicks at 1.0, 2.5 and 4.0 s. "clip." (src
        # 0.58-0.98) reaches the 1.0 s click within the gate's 0.1 s margin; "Short" lies in
        # noise only, so the audio gate drops it as a hallucination (events kind no_audio).
        canned = [[" Short", 0.1, 0.4, 0.9], [" clip.", 0.4, 0.8, 0.9]]
        with fake_backend(canned):
            rc, out, err = self._run(self._args(file=str(self.media), clip="c09-short"))
        self.assertEqual(rc, 0, err)
        doc = env.read_json(self.P / "transcripts" / "c09-short.json")
        self.assertGreater(doc["audio_offset"], 0.1)
        self.assertEqual([w["w"] for w in doc["words"]], ["clip."])
        self.assertEqual(doc["words"][0]["s"], round(0.4 + doc["audio_offset"], 3))
        self.assertEqual([(e["kind"], e["s"]) for e in doc["events"]],
                         [("no_audio", round(0.1 + doc["audio_offset"], 3))])
        self.assertTrue((self.P / "transcripts" / "c09-short.md").is_file())
        with self.assertRaises(env.HCError):
            self._run(self._args(file=str(self.media), clip="Bad_ID"))
        with self.assertRaises(env.HCError):
            self._run(self._args(clip="c01"))


@needs_fixtures
class TestSlowASR(unittest.TestCase):
    @slow
    def test_auto_backend_call_path(self):
        """Real ASR on a 6 s speech-less fixture: proves the call path (zero words is fine).
        HOOKCUTTER_TEST_ASR_MODEL overrides the model (e.g. "tiny" for a quick run)."""
        with tempfile.TemporaryDirectory() as td:
            media = fixtures.make_av_fixture(Path(td), fps="25", dur=6, flashes=[1.0, 3.0])
            words = T.transcribe_media(media, "auto", os.environ.get("HOOKCUTTER_TEST_ASR_MODEL"),
                                       "en", ["Acme"])
        self.assertIsInstance(words, list)
        for k, w in enumerate(words):
            self.assertEqual(w["i"], k)
            self.assertLessEqual(w["s"], w["e"])
            self.assertTrue(-0.5 <= w["s"] <= 7.0, w)


if __name__ == "__main__":
    unittest.main()
