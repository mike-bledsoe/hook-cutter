"""Regression tests for the second round of QA and core fixes.

G10 sees a caption that lights under a breathy laugh or a click, G6 tells a final consonant
from a clipped vowel, G5 / G11 / G14 count inserted speech, G12 judges the top of the head
only when it was measured, G16 reports missing face data once per clip, `hc status` names
stale plans, renders and covers with the reason, and `hc all` says what it did per clip.

Everything is synthetic: tones, shaped noise and clicks written as small wav files, and
hand-written plan / captions / cover / shots json.
"""

from __future__ import annotations

import io
import os
import shutil
import sys
import tempfile
import types
import unittest
import wave
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "scripts"))

import fixtures as F  # noqa: E402
import qa_media as qm  # noqa: E402
from hookcutter import cli, env, qa  # noqa: E402
from hookcutter import plan as plan_mod  # noqa: E402
from hookcutter import render as render_mod  # noqa: E402
from hookcutter.audio import EnergyTrack  # noqa: E402
from hookcutter.project import Project  # noqa: E402

SR = 16000
TMP = None


def setUpModule():
    global TMP
    TMP = Path(tempfile.mkdtemp(prefix="hc-fix2-fd-"))


def tearDownModule():
    if TMP and TMP.exists():
        shutil.rmtree(str(TMP), ignore_errors=True)


def quiet(fn, *a, **kw):
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        rv = fn(*a, **kw)
    return rv, out.getvalue(), err.getvalue()


# ---------------------------------------------------------------- synthetic sound

def _ramp(x, ms=8.0):
    k = min(int(SR * ms / 1000.0), len(x) // 2)
    if k:
        x = x.copy()
        x[:k] *= np.linspace(0.0, 1.0, k)
        x[-k:] *= np.linspace(1.0, 0.0, k)
    return x


def vowel(dur, f0=125.0, level=0.2):
    """A pitched, formant-shaped sound: harmonics of f0 with bumps near 700 and 1200 Hz."""
    n = int(round(dur * SR))
    t = np.arange(n) / SR
    x = np.zeros(n)
    k = 1
    while k * f0 < 3400:
        f = k * f0
        gain = 1.0 / k + 0.8 * np.exp(-((f - 700.0) / 150.0) ** 2) + 0.5 * np.exp(-((f - 1200.0) / 200.0) ** 2)
        x += gain * np.sin(2 * np.pi * f * t + 0.7 * k)
        k += 1
    return _ramp(level * x / np.sqrt(np.mean(x * x)))


def _shaped_noise(dur, lo, hi, tilt, level, seed):
    n = int(round(dur * SR))
    rng = np.random.default_rng(seed)
    spec = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1.0 / SR)
    mask = ((f >= lo) & (f <= hi)).astype(float)
    shape = np.where(f > 0, np.maximum(f, 1.0) ** (-tilt / 2.0), 0.0)
    x = np.fft.irfft(spec * mask * shape, n)
    return _ramp(level * x / np.sqrt(np.mean(x * x)))


def breath(dur, level=0.2, seed=3):
    """Unpitched sound with its weight low in the speech band (a breathy laugh, an exhale)."""
    return _shaped_noise(dur, 150.0, 4000.0, 1.0, level, seed)


def hiss(dur, level=0.12, seed=5):
    """Unpitched frication high in the spectrum (an "s")."""
    return _shaped_noise(dur, 4000.0, 7500.0, 0.0, level, seed)


def click(level=0.6):
    x = np.zeros(int(0.005 * SR))
    x[::2] = level
    x[1::2] = -level
    return x


def room(dur, seed=11):
    return np.random.default_rng(seed).standard_normal(int(round(dur * SR))) * 10 ** (-70 / 20)


def put(x, t, sound):
    a = int(round(t * SR))
    x[a:a + len(sound)] += sound[:max(0, len(x) - a)]


def write_wav(path, x):
    pcm = np.clip(np.round(np.asarray(x) * 32767.0), -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes(pcm.tobytes())
    return Path(path)


def talk(dur=16.0, start=2.0, word=0.30, gap=0.08, pause=0.7, every=5):
    """(samples, [(s, e)] words): vowels on room tone, a pause after every few words."""
    x = room(dur)
    words, t, k = [], start, 0
    while t < dur - 1.5:
        put(x, t, vowel(word, f0=118.0 + 9.0 * (k % 4)))
        words.append((round(t, 3), round(t + word, 3)))
        k += 1
        t += word + (pause if k % every == 0 else gap)
    return x, words


def captions(words, lead=0.06, per_page=5, hook_end=1.5):
    pages = []
    for pi in range(0, len(words), per_page):
        toks = [{"text": "W%d" % (pi + j), "s_out": s, "e_out": e, "idx": [pi + j]}
                for j, (s, e) in enumerate(words[pi:pi + per_page])]
        pages.append({"i": pi // per_page, "start_out": toks[0]["s_out"] - lead,
                      "end_out": toks[-1]["e_out"], "tokens": toks})
    return {"lead_ms": lead * 1000, "pages": pages, "hook": {"end_out": hook_end}}


def ctx_for(project=None, spec=None, asr=False):
    return qa._Ctx(project, spec or {"clips": {}}, None, asr, False)


# ---------------------------------------------------------------- speech-likeness

class SpeechLikeness(unittest.TestCase):
    """The reference G6 and G10 use: sustained, pitched or formant-shaped sound in the speech band."""

    @classmethod
    def setUpClass(cls):
        from hookcutter import speechband
        cls.sb = speechband
        x = room(6.0)
        put(x, 1.00, vowel(0.40))                    # a word
        put(x, 2.00, breath(0.50))                   # a breath: loud, unpitched
        put(x, 3.00, click())                        # a click
        put(x, 4.00, hiss(0.12))                     # "s" ...
        put(x, 4.12, vowel(0.40))                    # ... running into its vowel
        cls.S = speechband.analyse(x, SR)

    def test_runs_are_the_two_words(self):
        runs = self.S.speech_runs()
        self.assertEqual(len(runs), 2, runs)
        self.assertAlmostEqual(runs[0][0], 1.00, delta=0.04)
        self.assertAlmostEqual(runs[0][1], 1.40, delta=0.04)

    def test_word_initial_consonant_belongs_to_the_word(self):
        on, _voiced, _end = self.S.runs()[1]
        self.assertAlmostEqual(self.S.time(on), 4.00, delta=0.04)
        self.assertAlmostEqual(self.S.time(_voiced), 4.12, delta=0.05)

    def test_breath_and_click_are_sound_but_not_speech(self):
        self.assertGreater(self.S.loud_seconds(2.0, 2.5), 0.4)
        self.assertEqual(self.S.onset_after(2.05, 2.45), None)
        self.assertEqual(self.S.onset_after(2.95, 3.20), None)
        self.assertAlmostEqual(self.S.onset_after(2.05), 4.00, delta=0.04)

    def test_inside_a_word_the_onset_is_now(self):
        self.assertEqual(self.S.onset_after(1.2), 1.2)

    def test_describe_names_the_sound(self):
        self.assertEqual(self.S.describe(1.10, 1.30)["kind"], "voiced")
        self.assertEqual(self.S.describe(4.02, 4.10)["kind"], "fricative")
        self.assertEqual(self.S.describe(2.10, 2.40)["kind"], "other")
        self.assertEqual(self.S.describe(5.20, 5.60)["kind"], "quiet")

    def test_empty_audio(self):
        S = self.sb.analyse(np.zeros(0), SR)
        self.assertEqual((S.n, S.speech_runs(), S.onset_after(0.0)), (0, [], None))


# ---------------------------------------------------------------- G10 (item 1)

class CaptionUnderALaugh(unittest.TestCase):
    """A caption that lights before its word while another sound fills the gap."""

    @classmethod
    def setUpClass(cls):
        cls.dir = Path(tempfile.mkdtemp(prefix="g10-", dir=str(TMP)))
        x, cls.words = talk()
        cls.k = 10                                    # the first word after the second pause
        s = cls.words[cls.k][0]
        put(x, s - 0.70, breath(0.70, level=0.2))    # a laugh fills the pause, right up to the word
        cls.laugh = write_wav(cls.dir / "laugh.wav", x)
        x2, _ = talk()
        cls.k2 = 5                                    # the first word after the first pause
        s2 = cls.words[cls.k2][0]
        put(x2, s2 - 0.22, click())                   # a lip click 0.22 s before the voice
        cls.clicky = write_wav(cls.dir / "click.wav", x2)
        x3, _ = talk()
        cls.plain = write_wav(cls.dir / "plain.wav", x3)
        cls.plan = {"duration_out": 16.0}

    def g10(self, wav, cap):
        return qa.g10_sync(ctx_for(), self.plan, cap, None, None, 0.06, "vertical/clean",
                           E=EnergyTrack.from_media(wav), render=wav)

    def test_on_time_captions_pass(self):
        g = self.g10(self.plain, captions(self.words))
        self.assertEqual(g["status"], "PASS", g["evidence"])
        self.assertIn("speech: 0 more token(s)", g["evidence"])

    def test_caption_early_under_a_laugh_warns(self):
        cap = captions(self.words)
        tok = cap["pages"][self.k // 5]["tokens"][self.k % 5]
        tok["s_out"] -= 0.35                          # stamped into the laugh
        cap["pages"][self.k // 5]["start_out"] -= 0.35
        g = self.g10(self.laugh, cap)
        self.assertEqual(g["status"], "WARN", g["evidence"])
        self.assertIn("speech: 1 more token(s)", g["evidence"])
        line = [d for d in g["details"] if '"W%d"' % self.k in d]
        self.assertEqual(len(line), 1, g["details"])
        self.assertIn("before the first speech-like sound", line[0])
        self.assertIn("of other sound in between", line[0])
        self.assertRegex(line[0], r"0\.(3[5-9]|4\d)s before")   # 0.35 s early plus the 60 ms lead, less the attack

    def test_the_same_laugh_with_the_caption_on_the_word_passes(self):
        g = self.g10(self.laugh, captions(self.words))
        self.assertEqual(g["status"], "PASS", g["evidence"])

    def test_caption_on_a_click_before_the_voice_warns(self):
        cap = captions(self.words)
        tok = cap["pages"][self.k2 // 5]["tokens"][self.k2 % 5]
        tok["s_out"] -= 0.22                          # the word start put on the click
        cap["pages"][self.k2 // 5]["start_out"] -= 0.22
        g = self.g10(self.clicky, cap)
        self.assertEqual(g["status"], "WARN", g["evidence"])
        line = [d for d in g["details"] if '"W%d"' % self.k2 in d]
        self.assertEqual(len(line), 1, g["details"])
        self.assertIn("the gap is quiet", line[0])

    def test_token_on_a_word_flagged_timing_uncertain_is_named(self):
        """A voiced laugh is a vowel to the audio check: the transcript flag carries the doubt."""
        cap = captions(self.words)
        g = qa.g10_sync(ctx_for(), self.plan, cap, None, None, 0.06, "vertical/clean",
                        E=EnergyTrack.from_media(self.plain), render=self.plain, uncertain={7})
        self.assertEqual(g["status"], "WARN", g["evidence"])
        self.assertIn("1 token(s) on words the transcript flags timing_uncertain", g["evidence"])
        self.assertEqual(len([d for d in g["details"] if '"W7"' in d and "timing_uncertain" in d]), 1)
        # under the hook card (ends at 1.5 s here: nothing is) or not displayed: not named
        g = qa.g10_sync(ctx_for(), self.plan, cap, None, None, 0.06, "vertical/clean",
                        E=EnergyTrack.from_media(self.plain), render=self.plain, uncertain={999})
        self.assertEqual(g["status"], "PASS", g["evidence"])

    def test_uncertain_words_come_from_the_transcript_flags(self):
        project = types.SimpleNamespace(transcript=lambda cid=None: {"flags": [
            {"kind": "timing_uncertain", "a": 4, "b": 5}, {"kind": "unworded_audio", "s": 1.0, "e": 2.0},
            {"kind": "timing_uncertain", "a": 9, "b": 9}, {"kind": "timing_uncertain"}]})
        self.assertEqual(qa.uncertain_words(project, "c1"), {4, 5, 9})
        broken = types.SimpleNamespace(transcript=mock.Mock(side_effect=env.HCError("no transcript")))
        self.assertEqual(qa.uncertain_words(broken), set())

    def test_without_the_file_only_the_energy_checks_run(self):
        g = qa.g10_sync(ctx_for(), self.plan, captions(self.words), None, None, 0.06, "v",
                        E=EnergyTrack.from_media(self.plain))
        self.assertEqual(g["status"], "PASS")
        self.assertNotIn("speech:", g["evidence"])

    def test_speech_timing_ignores_tokens_under_the_hook_card(self):
        S = qa.speech_track(self.laugh)
        toks = qa.caption_tokens(captions(self.words), 0.06)
        toks[0]["d"] -= 0.9                           # far before its word, but under the card
        early = qa.speech_timing(S, toks, 3.0, 16.0)
        self.assertEqual(early, [])


# ---------------------------------------------------------------- G6 (item 2)

class CloseOnAConsonant(unittest.TestCase):
    """A clip that ends on a finished final consonant is not "mid-sound"; a clipped vowel is."""

    @classmethod
    def setUpClass(cls):
        cls.dir = Path(tempfile.mkdtemp(prefix="g6-", dir=str(TMP)))

    def render(self, name, tail):
        """Room tone and words, then `tail(x, end)` writes the last sounds; 60 ms final fade."""
        dur = 6.0
        x = room(dur)
        for t in (1.0, 1.5, 2.6, 3.1):
            put(x, t, vowel(0.35))
        tail(x, dur)
        n = int(0.06 * SR)
        x[-n:] *= np.linspace(1.0, 0.0, n)
        return write_wav(self.dir / name, x)

    def g6(self, wav):
        return qa.g6_open_close({"lints": []}, wav, "vertical (audio)", {"segments": []}, None)

    def test_final_consonant_after_a_vowel_is_not_mid_sound(self):
        def tail(x, end):
            put(x, end - 0.50, vowel(0.36, level=0.2))
            put(x, end - 0.14, hiss(0.14, level=0.15))       # the "s" sounds 80 ms, then the fade
        g = self.g6(self.render("sense.wav", tail))
        self.assertEqual(g["status"], "PASS", g["details"])
        self.assertIn("ends on a finished consonant", g["evidence"])
        self.assertIn("not a clipped vowel", g["evidence"])

    def test_clipped_vowel_still_warns(self):
        def tail(x, end):
            a = int((end - 0.40) * SR)
            x[a:] += vowel(0.60, level=0.2)[int(0.1 * SR):int(0.1 * SR) + len(x) - a]   # full level to the end
        g = self.g6(self.render("clipped.wav", tail))
        self.assertEqual(g["status"], "WARN", g["details"])
        self.assertIn("ends mid-sound", g["evidence"])
        self.assertIn("a voiced sound", g["evidence"])

    def test_consonant_after_a_pause_is_the_next_word(self):
        def tail(x, end):
            put(x, end - 0.14, hiss(0.14, level=0.15))       # nothing voiced before it
        g = self.g6(self.render("stray.wav", tail))
        self.assertEqual(g["status"], "WARN", g["details"])
        self.assertIn("the next word is starting", g["evidence"])

    def test_quiet_close_is_untouched(self):
        g = self.g6(self.render("quiet.wav", lambda x, end: None))
        self.assertEqual(g["status"], "PASS", g["details"])
        self.assertNotIn("consonant", g["evidence"])


# ---------------------------------------------------------------- G5 / G11 / G14 (item 3)

def insert_plan():
    """Six transcript words, with four tokens of speech the transcript missed after word 12."""
    texts = ["pick", "every", "modality", "today", "choose", "wisely."]
    words = []
    for k, t in enumerate(texts):
        words.append({"i": 10 + k, "raw": t, "text": t, "s_out": 0.5 * k, "e_out": 0.5 * k + 0.4, "piece": 0})
    extra = ["forty", "strangers", "bought", "tickets."]
    ins = [{"i": None, "ins": 0, "raw": t, "text": t, "s_out": 1.42 + 0.02 * k, "e_out": 1.44 + 0.02 * k,
            "piece": 0} for k, t in enumerate(extra)]
    return {"words": words[:3] + ins + words[3:], "intent": [10, 11, 12, 13, 14, 15],
            "inserts": [{"n": 0, "after": 12, "tokens": extra, "src": [100.0, 100.8]}],
            "hook_card": {"lines": ["*FORTY* STRANGERS"]}, "duration_out": 3.0}


class InsertedSpeech(unittest.TestCase):
    def test_sequence_places_tokens_after_their_word(self):
        seq = qa.intent_sequence(insert_plan())
        self.assertEqual([t for _i, t in seq],
                         ["pick", "every", "modality", "forty", "strangers", "bought", "tickets.",
                          "today", "choose", "wisely."])
        self.assertEqual([i for i, _t in seq], [10, 11, 12, None, None, None, None, 13, 14, 15])

    def test_sequence_without_inserts_is_the_intent(self):
        plan = insert_plan()
        plan.pop("inserts")
        plan["words"] = [w for w in plan["words"] if w["i"] is not None]
        self.assertEqual([i for i, _t in qa.intent_sequence(plan)], plan["intent"])

    def test_after_word_that_is_not_intended_and_two_runs(self):
        plan = insert_plan()
        plan["intent"] = [10, 11, 13, 14, 15]            # word 12 is a filler: not intended
        plan["inserts"].append({"n": 1, "after": 12, "tokens": ["again"], "src": [100.9, 101.0]})
        texts = [t for _i, t in qa.intent_sequence(plan)]
        self.assertEqual(texts, ["pick", "every", "forty", "strangers", "bought", "tickets.", "again",
                                 "today", "choose", "wisely."])

    def _g5(self, plan, heard):
        hyp = [{"w": w, "s": 0.2 * k, "e": 0.2 * k + 0.15} for k, w in enumerate(heard.split())]
        project = types.SimpleNamespace(transcript=lambda cid=None: {"backend": "test"})
        with mock.patch.object(qa, "_asr_words", return_value=hyp), \
                mock.patch.object(qa, "_silence_oracle", return_value=None):
            return qa.g5_words(ctx_for(project, asr=True), "c1", plan, Path("render.mp4"), "vertical (audio)")

    def test_word_diff_counts_inserted_tokens(self):
        g, d, _hyp = self._g5(insert_plan(), "pick every modality forty strangers bought tickets today choose wisely")
        self.assertEqual(g["status"], "PASS", g["evidence"])
        self.assertIn("10 intended tokens (4 inserted)", g["evidence"])
        self.assertEqual(d["intent"][3:7], [None] * 4)

    def test_missing_inserted_speech_is_a_hard_miss(self):
        g, _d, _hyp = self._g5(insert_plan(), "pick every modality today choose wisely")
        self.assertEqual(g["status"], "FAIL", g["evidence"])
        self.assertIn('missing "strangers"', " ".join(g["details"]))

    def test_inserted_tokens_never_break_the_overlap_check(self):
        plan = insert_plan()
        plan["overlaps"] = [{"words": [12, 13], "pieces": [0, 1]}]
        g, _d, _hyp = self._g5(plan, "pick every modality forty strangers bought tickets today choose wisely")
        self.assertIn(g["status"], ("PASS", "WARN"), g["evidence"])

    def test_claims_and_accents_hear_inserted_speech(self):
        plan = insert_plan()
        clip = {"hook_card": plan["hook_card"]}
        ctx = ctx_for(types.SimpleNamespace(transcript=lambda cid=None: {"glossary": [], "words": []}))
        g = qa.g11_faith(ctx, "c1", clip, plan)
        self.assertEqual(g["status"], "PASS", g["evidence"])        # "forty" is only in the insert
        cap = {"accents": {"phrases": ["FORTY"], "matched": []}}
        with mock.patch("hookcutter.presets.load", return_value={"caption": {"emphasis": {"sources": ["hook"]}}}), \
                mock.patch("hookcutter.presets.for_format", side_effect=lambda p, f: p):
            g = qa.g14_accent(ctx, plan, clip, "clean", "vertical", cap, "vertical/clean")
        self.assertEqual(g["status"], "WARN")
        self.assertIn("spoken, but never shown highlighted", g["evidence"])


# ---------------------------------------------------------------- G12 (item 6)

class CrownOnlyWhenMeasured(unittest.TestCase):
    def setUp(self):
        from hookcutter import cover
        self.cover = cover
        self.tmp = Path(tempfile.mkdtemp(prefix="g12-", dir=str(TMP)))
        self.P = Project(self.tmp, create=True)
        self.ctx = qa._Ctx(self.P, {"clips": {"c1": {}}}, None, False, False)
        img = cover.Image.new("RGB", (1080, 1920), (90, 100, 120))
        cover.save_jpeg(img, self.P.cover_path("c1", "clean", "main"))
        for kind in ("grid", "mobile"):
            cover.save_jpeg(img.resize((270, 360)), self.P.cover_path("c1", "clean", kind))

    def g12(self, **kw):
        m = {"headline_box": [80, 1200, 900, 300], "text_box": [80, 1200, 900, 300],
             "face_box": [300, 400, 400, 560], "faces_found": 1, "band": [240, 1680],
             "shift": 0, "warnings": [], "spec_hash": "h"}
        m.update(kw)
        env.write_json(self.P.cover_path("c1", "clean", "json"), m)
        return qa.g12_cover(self.ctx, "c1", "clean", {"spec_hash": "h"})

    def test_estimated_crown_above_the_tile_is_not_a_finding(self):
        # face top 400, estimate 400 - 0.3 x 560 = 232: wrong for a close-cropped head
        warn = "close-up: the top of the head (y 232) is above the grid crop (240) even unshifted"
        for extra in ({}, {"crown_method": "estimate"}):
            with self.subTest(extra=extra):
                g = self.g12(crown_y=232, warnings=[warn], **extra)
                self.assertEqual(g["status"], "PASS", g["evidence"])

    def test_measured_crown_above_the_tile_warns(self):
        g = self.g12(crown_y=232, crown_method="image")
        self.assertEqual(g["status"], "WARN", g["evidence"])
        self.assertIn("the grid tile trims the top of the head", g["evidence"])

    def test_measured_crown_inside_the_tile_passes(self):
        g = self.g12(crown_y=370, crown_method="image")
        self.assertEqual(g["status"], "PASS", g["evidence"])

    def test_cut_face_box_fails_whatever_the_method(self):
        for extra in ({}, {"crown_method": "estimate"}, {"crown_method": "image"}):
            with self.subTest(extra=extra):
                g = self.g12(crown_y=60, face_box=[300, 200, 400, 560], **extra)
                self.assertEqual(g["status"], "FAIL", g["evidence"])
                self.assertIn("face cut by the grid tile", g["evidence"])

    def test_measured_crown_pushed_out_by_the_shift_fails(self):
        g = self.g12(crown_y=120, crown_method="image", shift=200, face_box=[300, 270, 400, 500])
        self.assertEqual(g["status"], "FAIL", g["evidence"])
        self.assertIn("head cut by the grid tile", g["evidence"])


# ---------------------------------------------------------------- G16 (item 5)

def _fake_build_plan(project, clip_id, **_kw):
    return env.read_json(project.plan_path(clip_id))


def _fake_expected(project, clip_id, fmt, style, **_kw):
    return {"plan": None, "render": "fresh"}


class FaceDataGapOncePerClip(unittest.TestCase):
    """Two styles, no shots json: one G16 evidence gap for the clip, not one per style."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="g16-", dir=str(TMP)))
        media = cls.tmp / "media"
        media.mkdir()
        good = qm.make_render(media / "good.mp4")
        source = qm.make_source(media / "source.mp4", dur=8.0)
        hook = ["*BIG* IDEA"]
        clips = {"c-two": {"title": "c-two", "pieces": [{"words": [0, 0]}], "hook_card": {"lines": hook},
                           "styles": ["clean", "pop"]}}
        P = qm.write_project(cls.tmp / "P", clips, {"c-two": "this is a big idea that matters"}, source)
        cls.project = Project(P)
        cls.plan = qm.plan_for("c-two", 0, env.read_json(P / "transcript.json"), hook)
        env.write_json(cls.project.plan_path("c-two"), cls.plan)
        for st in ("clean", "pop"):
            shutil.copy(good, cls.project.render_path("c-two", "vertical", st))
            env.write_json(cls.project.render_path("c-two", "vertical", st, ".captions.json"), qm.captions_for(hook))
            env.write_json(cls.project.render_path("c-two", "vertical", st, ".log.json"),
                           {"cmds": [], "warnings": [], "audio_md5": "same",
                            "hashes": {"plan": render_mod.plan_hash(cls.plan), "render": "fresh"},
                            "framing_windows": [{"seg": 0, "out": [0.0, 3.0], "mode": "crop", "lo": 0.2,
                                                 "hi": 0.5, "multi_face": False}]})

    def gates(self):
        with mock.patch.object(plan_mod, "build_plan", _fake_build_plan), \
                mock.patch.object(render_mod, "expected_hashes", _fake_expected):
            quiet(qa.run, self.project, None, None, asr=False)
        rep = env.read_json(self.project.qa_path("c-two", ".json"))
        return [g for g in rep["gates"] if g["id"] == "G16"]

    def test_no_shots_json_is_one_gap(self):
        sp = self.project.shots_path("c-two")
        if sp.exists():
            sp.unlink()
        g16 = self.gates()
        self.assertEqual(len(g16), 1, g16)
        self.assertEqual((g16[0]["status"], g16[0].get("gap"), g16[0]["target"]),
                         ("SKIPPED", True, "vertical (all styles)"))
        self.assertIn("no shots json", g16[0]["evidence"])
        self.assertIn("hc shots", g16[0]["evidence"])

    def test_with_shots_json_each_style_is_checked(self):
        env.write_json(self.project.shots_path("c-two"), {"clip": "c-two", "ranges": [[0.0, 4.0]], "shots": [
            {"start_src": 0.0, "end_src": 4.0, "kind": "single", "multi_face": False,
             "faces": [{"x": 0.35, "y": 0.4, "h": 0.4, "presence": 1.0}]}]})
        g16 = self.gates()
        self.assertEqual(sorted(g["target"] for g in g16), ["vertical/clean", "vertical/pop"])
        self.assertFalse(any(g.get("gap") for g in g16), g16)

    def test_shots_analysed_for_other_pieces_is_one_gap(self):
        env.write_json(self.project.shots_path("c-two"), {"clip": "c-two", "ranges": [[20.0, 24.0]], "shots": [
            {"start_src": 20.0, "end_src": 24.0, "kind": "single", "multi_face": False, "faces": []}]})
        g16 = self.gates()
        self.assertEqual(len(g16), 1, g16)
        self.assertTrue(g16[0].get("gap"))
        self.assertIn("analysed for other pieces", g16[0]["evidence"])

    def test_uncovered_seconds(self):
        plan = {"segments": [{"src": [10.0, 14.0], "out": [0.0, 4.0]}, {"src": [20.0, 22.0], "out": [4.0, 6.0]}]}
        self.assertEqual(qa.shots_uncovered(plan, {"ranges": [[9.0, 21.0]]}), (1.0, 6.0))
        self.assertEqual(qa.shots_uncovered(plan, {"shots": []}), None)
        miss, total = qa.shots_uncovered(plan, {"shots": [{"start_src": 10.0, "end_src": 22.0}]})
        self.assertEqual((miss, total), (0.0, 6.0))
        # only what a crop window shows needs face data: the second segment is letterboxed
        windows = [{"out": [0.0, 4.0], "mode": "crop"}, {"out": [4.0, 6.0], "mode": "letterbox"}]
        self.assertEqual(qa.shots_uncovered(plan, {"ranges": [[9.0, 15.0]]}, windows), (0.0, 4.0))
        self.assertEqual(qa.shots_uncovered(plan, {"ranges": [[12.0, 15.0]]}, windows), (2.0, 4.0))


# ---------------------------------------------------------------- hc status (item 4)

def av():
    return F.make_av_fixture(TMP / "fx", dur=6, flashes=[])


class StatusProject(unittest.TestCase):
    """One clip with a plan, a render with its log, and a cover with its json."""

    CLIP = {"pieces": [{"words": [0, 1]}], "hook_card": {"lines": ["HELLO *WORLD*"]},
            "cover": {"headline": "HELLO *WORLD*", "at_out": 0.2}}

    def project(self, name, clip=None, plan_extra=None):
        P = F.make_project(TMP / name, av(), [("Hello", 1.0, 1.4), ("world.", 1.5, 1.9)])
        env.write_json(P / "clips.json", {"version": 1, "clips": {"c01-a": clip or dict(self.CLIP)}})
        plan = {"kind": "cut", "segments": [{"frames": 10}], "lints": [], "duration_out": 0.4, "fps": [25, 1]}
        plan.update(plan_extra or {})
        env.write_json(P / "plans" / "c01-a.plan.json", plan)
        r = P / "renders" / "c01-a__vertical__clean.mp4"
        r.parent.mkdir(parents=True, exist_ok=True)
        r.write_bytes(b"not really a video")
        env.write_json(P / "renders" / "c01-a__vertical__clean.log.json",
                       {"hashes": {"plan": render_mod.plan_hash(plan), "render": "made-from"},
                        "created": "2026-01-01T00:00:00+0000"})
        (P / "covers").mkdir(exist_ok=True)
        (P / "covers" / "c01-a__clean.jpg").write_bytes(b"not really a picture")
        self.cover_json(P)
        return P

    def cover_json(self, P, **kw):
        m = {"at_out": 0.2, "manual": True, "headline": "HELLO *WORLD*", "shift": 0, "shift_manual": False,
             "created": "2026-01-01T00:05:00+00:00", "spec_hash": "old"}
        m.update(kw)
        env.write_json(P / "covers" / "c01-a__clean.json", m)

    def set_clip(self, P, **cover):
        clip = dict(self.CLIP)
        clip["cover"] = dict(clip["cover"], **cover)
        env.write_json(P / "clips.json", {"version": 1, "clips": {"c01-a": clip}})

    def line(self, P, expected=None):
        cur = env.read_json(P / "plans" / "c01-a.plan.json")
        with mock.patch.object(cli, "_current_plan", return_value=cur), \
                mock.patch.object(render_mod, "expected_hashes", return_value={"render": expected or "made-from"}):
            rc, out, _ = quiet(cli.main, ["status", str(P)])
        self.assertEqual(rc, 0)
        return [ln for ln in out.splitlines() if ln.startswith("c01-a")][0]

    def cover_part(self, line):
        return line.split(" cover ", 1)[1].split(" qa ", 1)[0].strip()

    def render_part(self, line):
        return line.split(" render ", 1)[1].split(" cover ", 1)[0].strip()


class StaleCovers(StatusProject):
    def test_current_cover_is_not_stale(self):
        P = self.project("cov-fresh")
        self.assertEqual(self.cover_part(self.line(P)), "1/1")

    def test_at_out_change_is_stale(self):
        P = self.project("cov-at")
        self.set_clip(P, at_out=0.32)
        self.assertEqual(self.cover_part(self.line(P)), "1/1 STALE(at_out)")

    def test_at_out_is_compared_on_its_frame(self):
        P = self.project("cov-frame")
        self.set_clip(P, at_out=0.23)                 # the same 25 fps frame as 0.2
        self.assertEqual(self.cover_part(self.line(P)), "1/1")

    def test_forced_frame_with_auto_in_the_spec_is_stale(self):
        P = self.project("cov-auto")
        self.set_clip(P, at_out="auto")
        self.assertEqual(self.cover_part(self.line(P)), "1/1 STALE(at_out)")
        self.cover_json(P, manual=False, at_out=0.12)  # picked by the tool: follows the spec
        self.assertEqual(self.cover_part(self.line(P)), "1/1")

    def test_headline_change_is_stale(self):
        P = self.project("cov-head")
        self.set_clip(P, headline="HELLO *THERE*")
        self.assertEqual(self.cover_part(self.line(P)), "1/1 STALE(headline)")

    def test_headline_falls_back_to_the_hook_card(self):
        clip = dict(self.CLIP, cover={"at_out": 0.2})
        P = self.project("cov-hook", clip=clip)
        self.assertEqual(self.cover_part(self.line(P)), "1/1")       # the hook card says the same
        clip["hook_card"] = {"lines": ["HELLO", "*YOU*"]}
        env.write_json(P / "clips.json", {"version": 1, "clips": {"c01-a": clip}})
        self.assertEqual(self.cover_part(self.line(P)), "1/1 STALE(headline)")

    def test_explicit_shift_is_part_of_the_cover(self):
        P = self.project("cov-shift")
        self.set_clip(P, shift=120)
        self.assertEqual(self.cover_part(self.line(P)), "1/1 STALE(shift)")
        self.cover_json(P, shift=120, shift_manual=True)
        self.assertEqual(self.cover_part(self.line(P)), "1/1")

    def test_cover_older_than_its_render_is_stale(self):
        P = self.project("cov-render")
        self.cover_json(P, created="2025-12-31T23:00:00+00:00")
        self.assertEqual(self.cover_part(self.line(P)), "1/1 STALE(render)")

    def test_cover_of_a_stale_render_is_stale(self):
        P = self.project("cov-chain")
        line = self.line(P, expected="another-key")
        self.assertEqual(self.render_part(line), "1/1 STALE(inputs)")
        self.assertEqual(self.cover_part(line), "1/1 STALE(render)")

    def test_cover_without_json_is_not_judged(self):
        P = self.project("cov-nojson")
        (P / "covers" / "c01-a__clean.json").unlink()
        self.assertEqual(self.cover_part(self.line(P)), "1/1")


class StaleRendersAndShots(StatusProject):
    def test_render_key_difference_is_stale(self):
        P = self.project("rnd-key")
        self.assertEqual(self.render_part(self.line(P)), "1/1")
        self.assertEqual(self.render_part(self.line(P, expected="another-key")), "1/1 STALE(inputs)")

    def test_shots_json_appearing_makes_plan_and_render_stale(self):
        P = self.project("rnd-shots", plan_extra={"shots_hash": None})
        self.assertNotIn("STALE", self.line(P).split(" render ")[0])
        (P / "shots").mkdir(exist_ok=True)
        env.write_json(P / "shots" / "c01-a.json", {"clip": "c01-a", "shots": []})
        line = self.line(P, expected="key-with-face-tracks")
        self.assertIn("plan STALE(shots)", line)
        self.assertEqual(self.render_part(line), "1/1 STALE(shots)")

    def test_plan_made_from_the_current_shots_is_not_stale(self):
        P = self.project("rnd-shots-ok")
        (P / "shots").mkdir(exist_ok=True)
        env.write_json(P / "shots" / "c01-a.json", {"clip": "c01-a", "shots": []})
        plan = env.read_json(P / "plans" / "c01-a.plan.json")
        plan["shots_hash"] = env.sha256_file(P / "shots" / "c01-a.json")
        env.write_json(P / "plans" / "c01-a.plan.json", plan)
        env.write_json(P / "renders" / "c01-a__vertical__clean.log.json",
                       {"hashes": {"plan": render_mod.plan_hash(plan), "render": "made-from"},
                        "created": "2026-01-01T00:00:00+0000"})
        self.assertNotIn("shots", self.line(P))

    def test_log_without_a_render_hash_is_never_stale_for_inputs(self):
        P = self.project("rnd-old")
        plan = env.read_json(P / "plans" / "c01-a.plan.json")
        env.write_json(P / "renders" / "c01-a__vertical__clean.log.json",
                       {"hashes": {"plan": render_mod.plan_hash(plan)}, "created": "2026-01-01T00:00:00+0000"})
        self.assertEqual(self.render_part(self.line(P, expected="another-key")), "1/1")


# ---------------------------------------------------------------- hc all (item 8)

class AllSaysWhatItDid(unittest.TestCase):
    def project(self, name):
        P = F.make_project(TMP / name, av(), [("Hello", 1.0, 1.4), ("world.", 1.5, 1.9)])
        env.write_json(P / "clips.json", {"version": 1, "clips": {"c01-a": {"pieces": [{"words": [0, 1]}]}}})
        return P

    def fake(self, P, analyse_shots):
        """Steps that write what the real ones write. `render` analyses the shots itself when
        there is no shots json, and re-plans with them, as the real one does."""
        proj = Project(P)
        log = []

        def write_plan(cid):
            sp = proj.shots_path(cid)
            env.write_json(proj.plan_path(cid), {"kind": "cut", "clip": cid,
                                                 "shots_hash": env.sha256_file(sp) if sp.exists() else None})

        def dispatch(a):
            log.append(a.cmd)
            if a.cmd == "plan":
                for cid in a.clips:
                    write_plan(cid)
            elif a.cmd == "render":
                cid = a.clips[0]
                if analyse_shots and not proj.shots_path(cid).exists():
                    env.write_json(proj.shots_path(cid), {"clip": cid, "shots": []})
                    write_plan(cid)
                proj.render_path(cid, "vertical", "clean").write_bytes(b"x")
            elif a.cmd == "cover":
                for cid in a.clips:
                    for k in ("main", "grid", "mobile", "json"):
                        proj.cover_path(cid, "clean", k).write_bytes(b"x")
            elif a.cmd == "qa":
                for cid in a.clips:
                    env.write_json(proj.qa_path(cid, ".json"), {"status": "PASS"})
            return 0
        return log, dispatch

    def run_all(self, P, fake):
        args = cli.build_parser().parse_args(["all", str(P)])
        with mock.patch.object(cli, "dispatch", side_effect=fake), \
                mock.patch.object(cli, "_render_code_hash", return_value="code"):
            rc, out, err = quiet(cli.cmd_all, args)
        self.assertEqual(rc, 0)
        return [ln for ln in (out + err).splitlines() if ln.startswith("[all] c01-a:")]

    def test_one_line_per_clip_names_every_step(self):
        P = self.project("all-line")
        log, fake = self.fake(P, analyse_shots=False)
        lines = self.run_all(P, fake)
        self.assertEqual(len(lines), 1, lines)
        self.assertEqual(lines[0], "[all] c01-a: rendered 1 file(s); cover made; qa PASS")
        del log[:]
        lines = self.run_all(P, fake)
        self.assertEqual(log, ["plan", "export"])
        self.assertEqual(lines[0], "[all] c01-a: render skipped (up to date); cover skipped (up to date); "
                                   "qa PASS (skipped, up to date)")

    def test_render_that_found_its_files_up_to_date_is_not_called_rendered(self):
        """No stamps yet (the project was built step by step): `all` asks render, which finds its
        files current. The line says so instead of claiming a render."""
        P = self.project("all-kept")
        proj = Project(P)
        log, fake = self.fake(P, analyse_shots=False)
        proj.render_path("c01-a", "vertical", "clean").write_bytes(b"x")
        env.write_json(proj.render_path("c01-a", "vertical", "clean", ".log.json"),
                       {"created": "2026-01-01T00:00:00+0000"})
        lines = self.run_all(P, fake)
        self.assertEqual(log[:2], ["plan", "render"])
        self.assertEqual(lines[0], "[all] c01-a: 1 render(s) already up to date; cover made; qa PASS")

    def test_shots_analysed_by_render_is_said_and_not_redone(self):
        P = self.project("all-shots")
        log, fake = self.fake(P, analyse_shots=True)
        lines = self.run_all(P, fake)
        self.assertEqual(len(lines), 1, lines)
        self.assertIn("shots analysed by render (the framing needed face data: look at "
                      "sheets/shots_c01-a.jpg)", lines[0])
        self.assertIn("rendered 1 file(s)", lines[0])
        del log[:]
        lines = self.run_all(P, fake)                 # the plan now carries the shots: nothing to redo
        self.assertEqual(log, ["plan", "export"])
        self.assertNotIn("shots analysed", lines[0])


# ---------------------------------------------------------------- docs (item 7)

class DocsSayWhatTheToolDoes(unittest.TestCase):
    """The lessons of the first long run, written where an editor will look for them."""

    @classmethod
    def setUpClass(cls):
        root = env.SKILL_ROOT
        cls.doc = {}
        for name in ("SKILL.md", "references/editing.md", "references/captions.md",
                     "references/qa.md", "references/spec.md"):
            cls.doc[name] = " ".join((root / name).read_text(encoding="utf-8").split())

    def has(self, name, *phrases):
        for ph in phrases:
            self.assertTrue(ph in self.doc[name], "%s should say: %s" % (name, ph))

    def test_skill_md_framing_needs_face_data(self):
        self.has("SKILL.md", "Framing needs face data", "LOOK at the sheet",
                 "analyse the shots themselves", "analysed shots for <clip> (framing needs face data)",
                 "`insert`", "STALE(at_out|headline|shift|render)")

    def test_words_under_the_hook_card(self):
        self.has("references/editing.md", "Words under the hook card are never captioned",
                 "keep the card short", "do not put it under the card")
        self.has("references/captions.md", "heard but never captioned", "its auto duration",
                 "do not put a line that needs captions under the card")

    def test_unworded_sound_can_be_checked_without_ears(self):
        self.has("references/qa.md", "You cannot listen", "hc transcribe P --file cache/scratch-a.wav --clip scratch-a",
                 "Caption it with the clip key `insert`")

    def test_piece_split_for_a_protected_pause(self):
        self.has("references/editing.md", "Split the piece in two around it")

    def test_edge_offsets_reach_sixty_frames(self):
        self.has("references/editing.md", "up to 60", "`out_offset` (whole frames, + is later) is the fine control")
        self.has("references/spec.md", "whole frames, -60 to 60", "to step over a false start")
        self.has("references/qa.md", "-60 to 60")
        for name in ("references/spec.md", "references/qa.md", "references/editing.md"):
            self.assertNotIn("-12 to 12. ", self.doc[name])
            self.assertNotIn("-12..12", self.doc[name])

    def test_spec_md_documents_the_new_keys_and_flags(self):
        self.has("references/spec.md", "### insert", "`after`", "| `caption_backing` |",
                 "the hook card follows it", "| `false_start` |", "| `timing_uncertain` |",
                 "possible false start or repeated attempt at src T (words a-b)",
                 "word timing near src T is uncertain (overlapping sound)",
                 "`transitions[]`", '"kind": "cut" or "dissolve"')

    def test_captions_md_pages_and_cues_stop_at_cuts(self):
        self.has("references/captions.md", "A page never straddles such a seam",
                 "Cues also break at every join of kind piece, omit or dip",
                 "none starts before 0 or ends after the clip's duration", '"caption_backing": "always"')

    def test_qa_md_describes_the_changed_gates(self):
        self.has("references/qa.md", "the last consonant of a word", "speech-like sound",
                 "`crown_method: image`", "one evidence gap for the clip", "plus the words of every `insert`")


if __name__ == "__main__":
    unittest.main()
