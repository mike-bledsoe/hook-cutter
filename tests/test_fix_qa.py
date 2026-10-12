"""Regression tests for the QA and export fixes: claims extractor (G11 / L13), G0 render
freshness, G5 duplicated and uncaptioned speech, G6 dead air, G8 flashes and dark bands, G9
caption issues, G10 energy timing and display lag, G12 against the fixed grid band, G13
near-duplicates, G16 framing attribution, INCOMPLETE status and exit codes, the summary that
covers every clip, EYES per style, the ASR cache key and the recomputed audio md5.

Everything is synthetic: an EnergyTrack built from a dB array, tiny ffmpeg renders, and
hand-written plan / captions / cover / shots json.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "scripts"))

import qa_media as qm  # noqa: E402
from hookcutter import claims, env, qa  # noqa: E402
from hookcutter import render as render_mod  # noqa: E402
from hookcutter.audio import EnergyTrack  # noqa: E402
from hookcutter.project import Project  # noqa: E402


# ---------------------------------------------------------------- helpers

def speech_track(dur=20.0, start=2.0, word=0.33, gap=0.07, pause=0.6, every=6, seed=1):
    """(EnergyTrack, [(s, e)] words): -60 dB room tone, -22 dB words, a pause every few words."""
    rng = np.random.default_rng(seed)
    n = int(dur / 0.01)
    db = -60 + rng.normal(0, 1.5, n)
    words, t, k = [], start, 0
    while t < dur - 1.0:
        s, e = t, t + word
        words.append((s, e))
        a, b = int(round(s / 0.01)), int(round(e / 0.01))
        db[a:b] = -22 + rng.normal(0, 2, b - a)
        k += 1
        t = e + (pause if k % every == 0 else gap)
    return EnergyTrack(db), words


def captions(words, shift=0.0, lead=0.06, per_page=4, hook_end=1.5):
    pages = []
    for pi in range(0, len(words), per_page):
        toks = [{"text": "WORD", "s_out": s + shift, "e_out": e + shift, "idx": [pi + j]}
                for j, (s, e) in enumerate(words[pi:pi + per_page])]
        pages.append({"i": pi // per_page, "start_out": toks[0]["s_out"] - lead,
                      "end_out": toks[-1]["e_out"], "tokens": toks})
    return {"lead_ms": lead * 1000, "pages": pages, "hook": {"end_out": hook_end}}


def ctx_for(project=None, spec=None, asr=False):
    return qa._Ctx(project, spec or {"clips": {}}, None, asr, False)


def ff(args):
    env.ff(list(args))


# ---------------------------------------------------------------- claims (I11)

class ClaimsRules(unittest.TestCase):
    def nums(self, surface, spoken, label="hook card"):
        return [c["raw"] for c in claims.unspoken_claims([(label, surface)], spoken.split())
                if c["kind"] == "number"]

    def test_multipliers_and_ordinals_are_numbers(self):
        self.assertEqual(self.nums("10X YOUR SALES", "ten times your sales"), [])
        self.assertEqual(self.nums("10X YOUR SALES", "two times your sales"), ["10X"])
        self.assertEqual(self.nums("×3 GROWTH", "three times the growth"), [])
        self.assertEqual(self.nums("1ST CLIENT", "my first client"), [])
        self.assertEqual(self.nums("3RD CLIENT", "my first client"), ["3RD"])

    def test_ranges_need_both_ends(self):
        self.assertEqual(self.nums("80-90% OF PEOPLE", "eighty to ninety percent of people"), [])
        self.assertEqual(self.nums("80-90% OF PEOPLE", "eighty percent of people"), ["80-90%"])

    def test_number_words_count_only_before_a_noun(self):
        self.assertEqual(self.nums("ONE PERSON", "nothing here"), ["ONE"])
        self.assertEqual(self.nums("THE ONE THING", "nothing here"), [])
        self.assertEqual(self.nums("A NORMAL ONE", "nothing here"), [])

    def test_spoken_forms_of_one_and_years(self):
        self.assertEqual(self.nums("1 YEAR BUILDING", "a whole year building"), [])
        self.assertEqual(self.nums("3 YEARS BUILDING", "a whole year building"), ["3"])
        self.assertEqual(self.nums("2026 PLAN", "twenty twenty six plan"), [])

    def test_all_caps_unspoken_words_warn(self):
        res = claims.unspoken_claims([("hook card", "JORVIK QUELL: SELL IT, THEN MAKE IT")],
                                     "sell it then make it".split())
        self.assertEqual(sorted((c["raw"], c["kind"], c["level"]) for c in res),
                         [("JORVIK", "word", "WARN"), ("QUELL", "word", "WARN")])
        # common words that the speaker did not say are a paraphrase, not a claim
        self.assertEqual(claims.unspoken_claims([("cover headline", "SELL IT FIRST")], "sell it".split()), [])

    def test_names_in_first_position_and_levels(self):
        res = claims.unspoken_claims([("post title", "Jorvik told her to sell to strangers first")],
                                     "sell to strangers".split())
        self.assertEqual([(c["kind"], c["raw"], c["level"]) for c in res], [("name", "Jorvik", "WARN")])
        res = claims.unspoken_claims([("post caption", "She sold to 900 strangers")], "she sold to strangers".split())
        self.assertEqual([(c["kind"], c["level"]) for c in res], [("number", "FAIL")])

    def test_exempt_names_only_on_post_surfaces(self):
        text = "Dana Quell runs a studio"
        spoken = "runs a studio".split()
        self.assertEqual(claims.unspoken_claims([("post caption", text)], spoken, exempt_names=["Dana Quell"]), [])
        hook = claims.unspoken_claims([("hook card", text)], spoken, exempt_names=["Dana Quell"])
        self.assertEqual({c["raw"] for c in hook}, {"Dana", "Quell"})

    def test_surface_claims_shape(self):
        got = claims.surface_claims("SELL TO 40 STRANGERS")
        self.assertEqual(got["numbers"], [(40.0, "40")])
        self.assertIn("strangers", got["content"])


# ---------------------------------------------------------------- G10

class CaptionTiming(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.E, cls.words = speech_track()
        cls.plan = {"duration_out": 20.0}

    def g10(self, cap, diff=None, hyp=None, asr=False):
        return qa.g10_sync(ctx_for(asr=asr), self.plan, cap, diff, hyp, 0.06, "vertical/clean", E=self.E)

    def test_on_time_captions_pass(self):
        g = self.g10(captions(self.words))
        self.assertEqual(g["status"], "PASS", g["evidence"])
        self.assertIn("deviation", g["evidence"])

    def test_shifted_captions_fail(self):
        for shift in (0.3, -0.3, 1.0):
            with self.subTest(shift=shift):
                g = self.g10(captions(self.words, shift=shift))
                self.assertEqual(g["status"], "FAIL", g["evidence"])

    def test_small_shift_warns(self):
        self.assertEqual(self.g10(captions(self.words, shift=0.15))["status"], "WARN")

    def test_token_over_a_pause_warns(self):
        cap = captions(self.words)
        tok = cap["pages"][1]["tokens"][1]            # word 6 ends right before the 0.6 s pause
        tok["e_out"] += 0.8                          # ... and now runs across it
        g = self.g10(cap)
        self.assertEqual(g["status"], "WARN", g["evidence"])
        self.assertIn("silence", g["evidence"])

    def test_preset_lead_outside_the_window_fails(self):
        g = self.g10(captions(self.words, lead=-0.3))
        self.assertEqual(g["status"], "FAIL")
        self.assertIn("lead_ms -300", g["evidence"])

    def test_uncaptioned_voice_warns_then_fails_with_asr_words(self):
        E, words = speech_track(pause=0.22)          # runs of speech merge across short dips
        cap = captions(words)
        for p in cap["pages"]:                        # drop the captions of 7-13 s
            p["tokens"] = [t for t in p["tokens"] if not 7.0 <= t["s_out"] < 13.0]
        cap["pages"] = [p for p in cap["pages"] if p["tokens"]]
        g = qa.g10_sync(ctx_for(), self.plan, cap, None, None, 0.06, "v", E=E)
        self.assertEqual(g["status"], "WARN", g["evidence"])
        self.assertIn("has no caption", " ".join(g["details"]))
        hyp = [{"w": "word", "s": s, "e": e} for s, e in words]
        extra = [j for j, (s, _e) in enumerate(words) if 7.0 <= s < 13.0]
        g = qa.g10_sync(ctx_for(asr=True), self.plan, cap, {"pairs": [], "intent": [], "extra": extra}, hyp,
                        0.06, "v", E=E)
        self.assertEqual(g["status"], "FAIL", g["evidence"])
        self.assertIn("uncaptioned speech", " ".join(g["details"]))

    def test_needs_captions_and_audio(self):
        self.assertTrue(qa.g10_sync(ctx_for(), self.plan, None, None, None, 0.06, "v", E=self.E)["gap"])
        g = qa.g10_sync(ctx_for(), self.plan, captions(self.words), None, None, 0.06, "v", E=None)
        self.assertEqual((g["status"], g.get("gap")), ("SKIPPED", True))


# ---------------------------------------------------------------- G5

class WordDiffFixes(unittest.TestCase):
    def test_missing_twin_of_an_overlap_is_a_duplicate(self):
        ref = "we pick the alpha program alpha program today".split()
        hyp = "we pick the alpha program today".split()
        d = qa.diff_words(ref, hyp, dup={k: "pieces 1/2" for k in range(3, 7)})
        self.assertEqual(len(d["fail"]), 2)
        self.assertTrue(all(f.startswith("duplicated at join (pieces 1/2") for f in d["fail"]))
        self.assertFalse(any("stutter" in w for w in d["warn"]))
        # the same words without the overlap tag read as a stutter / plain miss
        plain = qa.diff_words(ref, hyp)
        self.assertFalse(any("duplicated" in f for f in plain["fail"]))

    def test_dup_positions_from_plan_overlaps(self):
        plan = {"overlaps": [{"pieces": [0, 1], "words": [11, 12], "src": [1.0, 2.0], "seconds": 1.0}]}
        self.assertEqual(qa._dup_positions(plan, [10, 11, 12, 11, 12, 13]),
                         {1: "pieces 0/1", 2: "pieces 0/1", 3: "pieces 0/1", 4: "pieces 0/1"})

    def test_overlap_heard_twice_is_noted(self):
        tmp = Path(tempfile.mkdtemp(prefix="hc-fixqa-twice-"))
        try:
            P = Project(tmp, create=True)
            env.write_json(P.root / "transcript.json", {"backend": "test", "model": "none", "words": []})
            r = tmp / "r.mp4"
            r.write_bytes(b"x")
            words = [{"i": i, "raw": w} for i, w in enumerate("we sell first then build".split())]
            plan = {"words": words, "intent": [0, 1, 2, 1, 2, 3, 4],
                    "overlaps": [{"pieces": [0, 1], "words": [1, 2], "src": [1.0, 2.0], "seconds": 1.0}]}
            hyp = [{"w": w, "s": 0.3 * k, "e": 0.3 * k + 0.25}
                   for k, w in enumerate("we sell first sell first then build".split())]
            ctx = qa._Ctx(P, {"clips": {}}, None, True, False)
            with mock.patch.object(qa, "_asr_words", return_value=hyp), \
                    mock.patch.object(qa, "_silence_oracle", return_value=None):
                g, _d, _h = qa.g5_words(ctx, None, plan, r, "vertical (audio)")
            self.assertEqual(g["status"], "WARN")
            self.assertTrue(any("heard twice" in x for x in g["details"]), g)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_long_extra_speech_is_uncaptioned(self):
        ref = "we sell first".split()
        hyp = ("we sell first and then customers worry about crazy fees every single month "
               "because banks charge everything").split()
        d = qa.diff_words(ref, hyp, hyp_times=[0.5 * k for k in range(len(hyp))])
        self.assertTrue(any(f.startswith("uncaptioned speech:") and "at out 1.50s" in f for f in d["fail"]), d)
        self.assertEqual(d["extra"], list(range(3, len(hyp))))
        short = qa.diff_words(ref, "we sell first and then the market".split())
        self.assertEqual(short["fail"], [])

    def test_asr_backend_crash_is_a_tool_error(self):
        tmp = Path(tempfile.mkdtemp(prefix="hc-fixqa-asr-"))
        try:
            P = Project(tmp, create=True)
            env.write_json(P.root / "transcript.json", {"backend": "test", "model": "none", "words": []})
            r = tmp / "r.mp4"
            r.write_bytes(b"x")
            ctx = qa._Ctx(P, {"clips": {}}, None, True, False)
            with mock.patch.object(qa, "_asr_words", side_effect=RuntimeError("model repo not found")):
                g, d, hyp = qa.g5_words(ctx, None, {"words": [], "intent": []}, r, "vertical (audio)")
            self.assertEqual((g["status"], g.get("gap"), d, hyp), ("SKIPPED", True, None, None))
            self.assertEqual(len(ctx.tool_errors), 1)
            self.assertEqual(qa.exit_code([{"status": "FAIL", "gaps": 1}], ctx.tool_errors, False), 3)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------- status, exit codes

class StatusAndExit(unittest.TestCase):
    def test_pass_with_gaps_is_incomplete(self):
        res = [qa.gate("G2", "PASS", "ok"), qa.gate("G5", "SKIPPED", "no asr", gap=True)]
        self.assertEqual(qa.clip_status(res, 1), "INCOMPLETE")
        self.assertEqual(qa.clip_status(res, 0), "PASS")
        self.assertEqual(qa.clip_status(res + [qa.gate("G9", "WARN", "w")], 1), "WARN")

    def test_exit_code_ranks(self):
        reps = [{"status": "INCOMPLETE", "gaps": 2}]
        self.assertEqual(qa.exit_code(reps, [], False), 0)
        self.assertEqual(qa.exit_code(reps, [], True), 2)
        self.assertEqual(qa.exit_code(reps + [{"status": "FAIL", "gaps": 0}], [], True), 1)
        self.assertEqual(qa.exit_code(reps + [{"status": "FAIL", "gaps": 0}], ["crash"], True), 3)


# ---------------------------------------------------------------- G6 dead air

class DeadAir(unittest.TestCase):
    def test_silence_at_segment_starts(self):
        db = np.full(1000, -60.0)
        db[40:300] = -20.0          # out 0: voice after 0.40 s (> 0.25 s at the opening)
        db[350:600] = -20.0         # segment 1 at 3.00 s: voice after 0.50 s (> 0.35 s)
        db[700:1000] = -20.0        # segment 2 at 6.80 s: voice after 0.20 s (fine)
        E = EnergyTrack(db)
        plan = {"segments": [{"out": [0.0, 3.0]}, {"out": [3.0, 6.8]}, {"out": [6.8, 10.0]}]}
        got = [(k, round(t, 2), round(w, 2)) for k, t, w in qa.dead_air(plan, E)]
        self.assertEqual(got, [(0, 0.0, 0.4), (1, 3.0, 0.5)])
        g = qa.g6_open_close({"lints": []}, None, "vertical (audio)", plan, E)
        self.assertTrue(g.get("gap"))               # no render to measure the edges
        self.assertIn("dead air", " ".join(g["details"]))


# ---------------------------------------------------------------- G0

class RenderFresh(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="hc-fixqa-g0-"))
        self.P = Project(self.tmp, create=True)
        self.plan = {"clip": "c1", "segments": [{"frames": 25}], "frames_out": 25, "spec_hash": "a"}
        self.ctx = qa._Ctx(self.P, {"clips": {}}, None, False, False)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def log(self, **hashes):
        env.write_json(self.P.render_path("c1", "vertical", "clean", ".log.json"), {"hashes": hashes})

    def g0(self):
        with mock.patch.object(render_mod, "expected_hashes", return_value={"plan": None, "render": "now"}):
            return qa.g0_fresh(self.ctx, "c1", "vertical", "clean", self.plan, "vertical/clean")

    def test_plan_changed(self):
        self.log(plan="older", render="now")
        g = self.g0()
        self.assertEqual(g["status"], "FAIL")
        self.assertIn("render predates the plan: hc render", g["evidence"])
        self.assertIn("--styles clean", g["evidence"])

    def test_inputs_changed(self):
        self.log(plan=render_mod.plan_hash(self.plan), render="before")
        g = self.g0()
        self.assertEqual(g["status"], "FAIL")
        self.assertIn("render predates its inputs (clip entry, preset, fonts, safe zone or code)", g["evidence"])

    def test_fresh_render_passes(self):
        self.log(plan=render_mod.plan_hash(self.plan), render="now")
        self.assertEqual(self.g0()["status"], "PASS")

    def test_no_hashes_is_a_gap(self):
        env.write_json(self.P.render_path("c1", "vertical", "clean", ".log.json"), {"warnings": []})
        g = self.g0()
        self.assertEqual((g["status"], g.get("gap")), ("SKIPPED", True))


# ---------------------------------------------------------------- G12

class CoverChecks(unittest.TestCase):
    def setUp(self):
        from hookcutter import cover
        self.cover = cover
        self.tmp = Path(tempfile.mkdtemp(prefix="hc-fixqa-g12-"))
        self.P = Project(self.tmp, create=True)
        self.spec = {"clips": {"c1": {"speaker": "Speaker A"}}}
        self.ctx = qa._Ctx(self.P, self.spec, None, False, False)
        self.write_image((90, 100, 120))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def write_image(self, rgb, bar=0):
        img = self.cover.Image.new("RGB", (1080, 1920), rgb)
        if bar:
            from PIL import ImageDraw
            ImageDraw.Draw(img).rectangle([0, 0, 1079, bar - 1], fill=(0, 0, 0))
        self.cover.save_jpeg(img, self.P.cover_path("c1", "clean", "main"))
        for kind in ("grid", "mobile"):
            self.cover.save_jpeg(img.resize((270, 360)), self.P.cover_path("c1", "clean", kind))

    def meta(self, **kw):
        # crown_method "image": the top of the head was measured in the picture (an estimate
        # from the face box never decides a head-cut finding; see test_fix2_fd)
        m = {"headline_box": [80, 1200, 900, 300], "text_box": [80, 1200, 900, 300],
             "face_box": [300, 400, 400, 500], "faces_found": 1, "crown_y": 350, "crown_method": "image",
             "band": [40, 1900], "warnings": [], "spec_hash": "h"}
        m.update(kw)
        env.write_json(self.P.cover_path("c1", "clean", "json"), m)

    def g12(self, stale=False):
        return qa.g12_cover(self.ctx, "c1", "clean", {"spec_hash": "h"}, render_stale=stale)

    def test_clean_cover_passes(self):
        self.meta()
        g = self.g12()
        self.assertEqual(g["status"], "PASS", g["evidence"])
        self.assertIn("1 face(s) found", g["evidence"])

    def test_fixed_grid_band_not_the_preset_band(self):
        self.meta(headline_box=[40, 1536, 1000, 334], text_box=[40, 1536, 1000, 334])
        g = self.g12()
        self.assertEqual(g["status"], "FAIL")
        self.assertIn("240-1680 grid band", g["evidence"])
        self.meta(ink_box=[60, 1300, 960, 300], headline_box=[40, 1536, 1000, 334])
        self.assertEqual(self.g12()["status"], "PASS")   # the ink, not the layout box, decides

    def test_head_cut_by_the_tile(self):
        # the headline shift pushed a head that fitted (crown 320 unshifted) out of the tile
        self.meta(crown_y=120, shift=200, face_box=[300, 270, 400, 500])
        g = self.g12()
        self.assertEqual(g["status"], "FAIL")
        self.assertIn("head cut by the grid tile", g["evidence"])
        self.assertIn("shift pushed it out", g["evidence"])
        # the face box itself starts above the tile
        self.meta(crown_y=20, shift=0, face_box=[300, 170, 400, 500])
        g = self.g12()
        self.assertEqual(g["status"], "FAIL")
        self.assertIn("face cut by the grid tile", g["evidence"])

    def test_close_up_crown_above_the_tile_unshifted_warns(self):
        """A close-up whose crown starts above the tile at shift 0 cannot fit at any shift; with
        the forehead inside the tile it is a WARN, not a FAIL that holds the export."""
        self.meta(crown_y=100, shift=40, face_box=[300, 250, 400, 500],
                  warnings=["close-up: the top of the head (y 100) is above the grid crop (240) even "
                            "unshifted, so the profile tile trims it; the forehead stays inside"])
        g = self.g12()
        self.assertEqual(g["status"], "WARN", g["evidence"])
        self.assertIn("close-up: the grid tile trims the top of the head", g["evidence"])
        self.assertEqual(g["evidence"].count("close-up"), 1, "the cover warning is not repeated")
        # shifted beyond the close-up allowance: the tile cuts the head down to the forehead
        self.meta(crown_y=-12, shift=240, face_box=[300, 250, 400, 500])
        g = self.g12()
        self.assertEqual(g["status"], "FAIL")
        self.assertIn("close-up shifted 240 px", g["evidence"])

    def test_no_face_never_reads_face_clear(self):
        self.meta(faces_found=0, face_box=None, warnings=["no face found in the chosen frame"])
        g = self.g12()
        self.assertEqual(g["status"], "WARN")
        self.assertIn("no face found: face overlap unchecked", g["evidence"])
        self.assertNotIn("clear", g["evidence"])

    def test_warnings_fallback_font_and_speaker(self):
        self.meta(font_fallback=True, frame_speaker="Speaker B", warnings=["headline shrunk to fit"])
        g = self.g12()
        self.assertEqual(g["status"], "WARN")
        for part in ("font fell back", "not the clip speaker Speaker A", "cover: headline shrunk to fit"):
            self.assertIn(part, g["evidence"])

    def test_dark_band_on_the_cover(self):
        self.write_image((90, 100, 120), bar=200)
        self.meta()
        g = self.g12()
        self.assertEqual(g["status"], "FAIL")
        self.assertIn("dark band on the cover", g["evidence"])

    def test_stale_cover_and_render(self):
        self.meta(spec_hash="old")
        self.assertIn("re-run hc cover", self.g12()["evidence"])
        self.assertIn("re-render, then re-cover", self.g12(stale=True)["evidence"])


# ---------------------------------------------------------------- G16

class FramingAttribution(unittest.TestCase):
    SHOTS = {"shots": [{"start_src": 100.0, "end_src": 120.0, "kind": "two", "multi_face": True,
                        "faces": [{"id": "A", "x": 0.27, "y": 0.5, "h": 0.2, "presence": 1.0},
                                  {"id": "B", "x": 0.75, "y": 0.5, "h": 0.22, "presence": 1.0}]}]}

    def plan(self, sp=None, role="body"):
        words = [{"i": k, "s_out": 0.5 * k, "e_out": 0.5 * k + 0.4, "sp": sp, "piece": 0} for k in range(20)]
        return {"segments": [{"src": [100.0, 110.0], "out": [0.0, 10.0]}], "words": words,
                "pieces": [{"n": 0, "role": role}]}

    def window(self, lo, hi, mode="crop", **kw):
        return [dict({"seg": 0, "out": [0.0, 10.0], "mode": mode, "lo": lo, "hi": hi, "multi_face": True}, **kw)]

    def run16(self, plan, windows, speakers, clip=None):
        return qa.framing_attribution(plan, clip or {"speaker": "Speaker A"}, self.SHOTS, windows, speakers)

    def test_single_face_shot_is_not_applicable(self):
        shots = {"shots": [{"start_src": 100.0, "end_src": 120.0, "kind": "single",
                            "faces": [{"x": 0.5, "h": 0.4, "presence": 1.0}]}]}
        st, _n, _s = qa.framing_attribution(self.plan(), {}, shots,
                                            [dict(self.window(0.3, 0.6)[0], multi_face=False)], {})
        self.assertEqual(st, "SKIPPED")

    def test_letterbox_and_wide_crop_pass(self):
        self.assertEqual(self.run16(self.plan(), self.window(0.0, 1.0, mode="letterbox"), {})[0], "PASS")
        self.assertEqual(self.run16(self.plan(), self.window(0.05, 0.95), {})[0], "PASS")

    def test_no_label_and_no_x(self):
        st, notes, _ = self.run16(self.plan(), self.window(0.6, 0.9), {}, clip={"title": "no speaker"})
        self.assertEqual(st, "FAIL")
        self.assertIn("no speaker labels or clip speaker", notes[0])
        st, notes, _ = self.run16(self.plan(), self.window(0.6, 0.9), {})
        self.assertEqual(st, "FAIL")
        self.assertIn('set clips.json "speakers": {"Speaker A"', notes[0])

    def test_wrong_face_fails_right_face_passes(self):
        speakers = {"Speaker A": {"x": 0.27}, "Speaker B": {"x": 0.75}}
        st, notes, _ = self.run16(self.plan(), self.window(0.6, 0.9), speakers)
        self.assertEqual(st, "FAIL")
        self.assertIn("framed face is not Speaker A for", notes[0])
        self.assertEqual(self.run16(self.plan(), self.window(0.1, 0.4), speakers)[0], "PASS")
        self.assertEqual(self.run16(self.plan(sp="Speaker B"), self.window(0.6, 0.9), speakers)[0], "PASS")

    def test_reaction_and_short_spans_warn(self):
        speakers = {"Speaker A": {"x": 0.27}, "Speaker B": {"x": 0.75}}
        self.assertEqual(self.run16(self.plan(role="reaction"), self.window(0.6, 0.9), speakers)[0], "WARN")
        self.assertEqual(self.run16(self.plan(), self.window(0.6, 0.9, reaction=True), speakers)[0], "WARN")
        short = self.plan(sp="Speaker B")
        short["words"][3]["sp"] = "Speaker A"               # one 0.4 s word of the other speaker
        st, notes, _ = self.run16(short, self.window(0.6, 0.9), speakers)
        self.assertEqual(st, "WARN")
        self.assertIn("0.4s", notes[0])

    def test_gate_needs_shots_json(self):
        tmp = Path(tempfile.mkdtemp(prefix="hc-fixqa-g16-"))
        try:
            P = Project(tmp, create=True)
            g = qa.g16_framing(qa._Ctx(P, {"clips": {}}, None, False, False), "c1", {}, self.plan(),
                               "vertical", "clean", "vertical/clean")
            self.assertEqual((g["status"], g.get("gap")), ("SKIPPED", True))
            self.assertIn("hc shots", g["evidence"])
            # a caption-only clip is never reframed: not applicable, not an evidence gap
            g = qa.g16_framing(qa._Ctx(P, {"clips": {}}, None, False, False), "c2",
                               {"kind": "caption_only"}, dict(self.plan(), kind="caption_only"),
                               "vertical", "clean", "vertical/clean")
            self.assertEqual(g["status"], "SKIPPED")
            self.assertFalse(g.get("gap"))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------- G8

class PictureDefects(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="hc-fixqa-g8-"))
        base = ["-f", "lavfi", "-i", "testsrc2=size=1080x1920:rate=25:duration=2", "-pix_fmt", "yuv420p",
                "-c:v", "libx264", "-preset", "ultrafast"]
        cls.band = cls.tmp / "band.mp4"
        ff(base[:4] + ["-vf", "drawbox=x=0:y=1770:w=iw:h=150:color=black:t=fill"] + base[4:] + [str(cls.band)])
        cls.flash = cls.tmp / "flash.mp4"
        ff(base[:4] + ["-vf", "drawbox=x=0:y=0:w=iw:h=ih:color=white:t=fill:enable='eq(n,30)'"]
           + base[4:] + [str(cls.flash)])
        cls.plan = {"fps": [25, 1], "duration_out": 2.0, "joins": []}

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def project(self, active_area=None):
        P = Project(self.tmp / ("P-aa" if active_area else "P"), create=True)
        video = {"width": 1280, "height": 720}
        if active_area:
            video["active_area"] = active_area
        env.write_json(P.root / "source.json", {"video": video, "audio": {}})
        return P

    def test_black_bottom_band(self):
        ctx = qa._Ctx(self.project(), {"clips": {}}, None, False, False)
        g = qa.g8_black_freeze(self.plan, self.band, None, "vertical/clean", ctx, "c1")
        self.assertEqual(g["status"], "WARN", g["evidence"])
        self.assertIn("dark band", g["evidence"])
        ctx = qa._Ctx(self.project(active_area=[0, 180, 1280, 360]), {"clips": {}}, None, False, False)
        g = qa.g8_black_freeze(self.plan, self.band, None, "vertical/clean", ctx, "c1")
        self.assertEqual(g["status"], "FAIL", g["evidence"])

    def test_single_frame_flash_and_peek_sheet(self):
        P = self.project()
        ctx = qa._Ctx(P, {"clips": {}}, None, False, False)
        g = qa.g8_black_freeze(self.plan, self.flash, None, "vertical/clean", ctx, "c1", "vertical", "clean")
        self.assertEqual(g["status"], "WARN", g["evidence"])
        self.assertIn("single-frame flash at 1.200s", g["evidence"])
        sheet = ctx.flag_sheets.get(("c1", "vertical", "clean"))
        self.assertTrue(sheet and Path(sheet).is_file())
        eyes = qa._eyes(ctx, "c1", None, ["vertical"], ["clean"])
        self.assertIn("sheets/c1__vertical__clean.flags.jpg", [e["path"] for e in eyes])

    def test_find_flashes_rule(self):
        self.assertEqual(qa.find_flashes([100, 100, 200, 100, 100]), [2])
        self.assertEqual(qa.find_flashes([100, 100, 200, 200, 200]), [])    # a cut, not a flash
        self.assertEqual(qa.dark_bands(np.array([10] * 10 + [200] * 80 + [12] * 10)), [(0, 10), (90, 100)])
        self.assertEqual(qa.dark_bands(np.array([10] * 2 + [200] * 98)), [])     # under 3 %


# ---------------------------------------------------------------- G9, G13, G3

class SmallGates(unittest.TestCase):
    def test_caption_issues_surface_in_g9(self):
        tmp = Path(tempfile.mkdtemp(prefix="hc-fixqa-g9-"))
        try:
            P = Project(tmp, create=True)
            ctx = qa._Ctx(P, {"clips": {}}, None, False, False)
            cap = {"pages": [{"i": 0, "start_out": 1.0, "end_out": 2.0, "box": [200, 1150, 600, 110]}],
                   "issues": [{"code": "min_page", "msg": "page 'WORD' shows for 0.40 s", "at_out": 1.0},
                              {"code": "glyph_stripped", "msg": "a glyph was dropped"}]}
            g = qa.g9_layout(ctx, "c1", {"_eff": {"safezone": "organic"}}, {"duration_out": 3.0},
                             "vertical", "clean", cap, tmp / "c.json", "vertical/clean")
            self.assertEqual(g["status"], "WARN")
            self.assertIn("caption min_page", g["evidence"])
            self.assertIn("caption note glyph_stripped: a glyph was dropped", g["details"])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_near_duplicate_hooks(self):
        hooks = ["SELL IT, THEN *MAKE* IT", "40 STRANGERS. NOT YOUR *FRIENDS*.", "STOP *SELLING* TO FRIENDS",
                 "YOUR PRODUCT IS *NOT* THE PROBLEM", "THE *MARKET* WAS TOO SMALL", "HE BUILT FOR A *YEAR*",
                 "NOBODY WANTS YOUR *PRODUCT*", "*90 DAYS* TO SELL IT", "SELL BEFORE YOU *BUILD*",
                 "TOO MUCH *PRODUCT*", "A DEPOSIT BEATS A *LIKE*", "MAKE *WAY MORE* MONEY"]
        pairs = [(a, b) for i, a in enumerate(hooks) for b in hooks[i + 1:] if qa.near_duplicate(a, b)]
        self.assertEqual(pairs, [])
        self.assertTrue(qa.near_duplicate("SELL IT TO 40 STRANGERS FIRST", "SELL TO *40 STRANGERS* BEFORE IT'S DONE"))

    def test_surface_echo_needs_two_shared_words(self):
        self.assertIsNone(qa.surface_echo("*SELL* TO 40 STRANGERS", "SELL IT BEFORE *IT'S DONE*"))
        self.assertIsNotNone(qa.surface_echo("SELL TO 40 STRANGERS", "SELL IT TO *40 STRANGERS* FIRST"))

    def test_audio_identity_is_read_from_the_files(self):
        tmp = Path(tempfile.mkdtemp(prefix="hc-fixqa-md5-"))
        try:
            P = Project(tmp, create=True)
            a = qm.make_render(Path(P.render_path("c1", "vertical", "clean")), dur=1.0, spans=((0.2, 0.8),))
            b = qm.make_render(Path(P.render_path("c1", "vertical", "pop")), dur=1.0, spans=((0.2, 0.8),),
                               lufs=-20.0)
            for st in ("clean", "pop"):        # the logs agree: only the files can tell
                env.write_json(P.render_path("c1", "vertical", st, ".log.json"), {"audio_md5": "same"})
            ctx = qa._Ctx(P, {"clips": {}}, None, False, False)
            g = qa.g_styles_audio(ctx, "c1", "vertical", {"clean": a, "pop": b})
            self.assertEqual(g["status"], "FAIL")
            self.assertIn("styles diverged", g["evidence"])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------- reports

class ReportsAndEyes(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="hc-fixqa-sum-"))
        self.P = Project(self.tmp, create=True)
        self.spec = {"clips": {c: {"_eff": {"styles": ["clean", "pop"]}} for c in ("c1", "c2", "c3")}}

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def rep(self, cid, status="WARN", **kw):
        r = {"clip": cid, "status": status, "created": "2026-01-01T00:00:00+00:00", "gates": [], "eyes": [],
             "counts": {s: 0 for s in qa.STATUSES}, "gaps": 0,
             "checked": {"styles": ["clean"], "formats": ["vertical"]}, "plan_spec_hash": None,
             "qa_code": qa.qa_code()}                 # made by the checks as they are now
        r.update(kw)
        return r

    def test_summary_lists_every_clip(self):
        env.write_json(self.P.qa_path("c2", ".json"), self.rep("c2"))
        rp = Path(self.P.render_path("c2", "vertical", "clean"))
        rp.write_bytes(b"x")
        later = time.time() + 5
        os.utime(rp, (later, later))                 # re-rendered after its QA
        ctx = qa._Ctx(self.P, self.spec, None, False, False)
        md = qa.summary_markdown([self.rep("c1", "PASS")], 0, ctx)
        self.assertIn("| c1 | PASS |", md)
        self.assertIn("| c2 | WARN (not re-run: checked 2026-01-01T00:00:00+00:00; stale) |", md)
        self.assertIn("| c3 | not checked |", md)

    def test_staleness_reasons(self):
        rep = self.rep("c1", plan_spec_hash="old")
        env.write_json(self.P.qa_path("c1", ".json"), rep)
        with mock.patch.object(qa, "current_spec_hash", return_value="new"):
            why = qa.qa_staleness(self.P, self.spec, "c1", rep)
        self.assertEqual(why, ["the clips.json entry changed since QA"])
        self.assertEqual(qa.qa_staleness(self.P, self.spec, "c1", None), ["QA has not run"])

    def test_eyes_list_every_style_grid(self):
        ctx = qa._Ctx(self.P, self.spec, None, False, False)
        for st in ("clean", "pop"):
            Path(self.P.cover_path("c1", st, "grid")).write_bytes(b"g")
        Path(self.P.sheet_path("covers-grid__clean.jpg")).write_bytes(b"m")   # rebuilt after the covers
        old = Path(self.P.sheet_path("covers-grid__pop.jpg"))
        old.write_bytes(b"m")
        os.utime(old, (time.time() - 60, time.time() - 60))                   # older than this cover
        eyes = {e["path"]: e for e in qa._eyes(ctx, "c1", None, ["vertical"], ["clean", "pop"])}
        self.assertTrue(eyes["sheets/covers-grid__clean.jpg"]["exists"])
        self.assertFalse(eyes["sheets/covers-grid__pop.jpg"]["exists"])
        self.assertIn("does not show this clip", eyes["sheets/covers-grid__pop.jpg"]["note"])
        self.assertTrue(eyes["covers/c1__pop.grid.jpg"]["exists"])
        contact = eyes["sheets/c1__vertical__clean.contact.jpg"]
        self.assertIn("cell 1 is frame 0", contact["why"])


# ---------------------------------------------------------------- ASR cache key (I10)

class AsrCacheKey(unittest.TestCase):
    def test_key_follows_backend_material_and_post_version(self):
        from hookcutter import transcribe
        tmp = Path(tempfile.mkdtemp(prefix="hc-fixqa-key-"))
        try:
            P = Project(tmp, create=True)
            r = tmp / "r.mp4"
            r.write_bytes(b"render")
            calls = []

            def fake(path, **kw):
                calls.append(kw)
                return [{"w": "hello", "s": 0.1, "e": 0.4}]
            tr = {"backend": "test", "model": "m", "language": "en", "glossary": []}
            with mock.patch.object(transcribe, "transcribe_media", fake), \
                    mock.patch.object(transcribe, "asr_key_material", lambda *a: {"backend": "test", "v": 1}), \
                    mock.patch.object(transcribe, "POST_VERSION", 2, create=True):
                qa._asr_words(P, r, tr)
                qa._asr_words(P, r, tr)                       # cached
                self.assertEqual(len(calls), 1)
                with mock.patch.object(transcribe, "POST_VERSION", 3, create=True):
                    qa._asr_words(P, r, tr)                   # repair changed: re-run
                self.assertEqual(len(calls), 2)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
