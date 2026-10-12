"""Regression tests for the review fixes in plan.py, spec.py and audio.py.

Each test names the finding it pins. Synthetic inputs only: energy tracks built from arrays,
the shared lavfi fixtures for the few tests that need real project files, and a fake ffmpeg
(a shell script) for the decode test."""
from __future__ import annotations

import os
import stat
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "scripts"))

from fixtures import fake_transcript, make_av_fixture, make_project, make_speech_fixture  # noqa: E402
from hookcutter import audio, claims, env, presets  # noqa: E402
from hookcutter import plan as P  # noqa: E402
from hookcutter import spec as S  # noqa: E402
from test_plan import build, lints_of, mkspec, seq_words, source  # noqa: E402


# ---- helpers --------------------------------------------------------------------

def speech(text, fix=None, pieces=None, omit=None, **clip_keys):
    """Plan of a clip over the words of `text` (0.3 s words, 0.1 s gaps), tighten off."""
    toks = text.split()
    clip = {"pieces": pieces or [{"words": [0, len(toks) - 1]}], "tighten": False}
    if fix:
        clip["fix"] = fix
    if omit:
        clip["omit"] = omit
    clip.update(clip_keys)
    plan, _tr, _E = build(seq_words(len(toks), text=toks), {"c01": clip})
    return plan


def msgs(plan, lid, level=None):
    return [x["msg"] for x in lints_of(plan, lid, level)]


def l01(issues, level):
    return [x["msg"] for x in issues if x["id"] == "L01" and x["level"] == level]


def old_hash(spec, cid):
    """The clip hash as it was before FAIL issues could enter it (pins the formula)."""
    clip = spec["clips"][cid]
    return env.stable_hash({"clip": S.public_clip(clip), "eff": clip.get("_eff"),
                            "speakers": spec.get("speakers") or {}})


def covers_fail(plan):
    return [x for x in plan["lints"] if x["id"] == "L01" and x["level"] == "FAIL"
            and "covers the whole clip" in x["msg"]]


# ---- P1: hold_after and omitted words ---------------------------------------------

class TestHoldNeverPlaysOmittedWord(unittest.TestCase):
    WORDS = [("w0", 1.0, 1.3), ("w1", 2.0, 2.3), ("w2", 3.0, 3.3), ("w3", 4.0, 4.3),
             ("w4", 5.0, 5.3), ("w5.", 6.0, 6.3)]

    def _plan(self, omit):
        clip = {"pieces": [{"words": [0, 1], "hold_after": 3.4}, {"words": [3, 5]}]}
        if omit:
            clip["omit"] = omit
        return build(self.WORDS, {"c01": clip})[0]

    def test_hold_stops_before_the_omitted_word(self):
        plan = self._plan([3])
        for s in plan["segments"]:
            played = min(s["src"][1], 4.3) - max(s["src"][0], 4.0)
            self.assertLess(played, 0.02, s)
        p0 = plan["pieces"][0]
        self.assertEqual(p0["extra_words"], [2])
        self.assertEqual(p0["hold"]["why"], "omit")
        self.assertIn("omit", P.HOLD_WHY)
        text = " ".join(msgs(plan, "L16"))
        self.assertIn("stopped before word 3, which the clip omits", text)
        self.assertNotIn("pieces[None]", text)
        self.assertIn("omits", P.hold_line(p0))

    def test_control_without_omit_is_unchanged(self):
        plan = self._plan(None)
        p0 = plan["pieces"][0]
        self.assertEqual(p0["extra_words"], [2])
        self.assertEqual(p0["hold"]["frame"], 98)
        self.assertEqual(p0["hold"]["why"], "clamped")
        self.assertIn("clamped at word 3 where pieces[1] starts", " ".join(msgs(plan, "L16")))


# ---- P2 / V14-N1 / V09-N1: lints judge the words that are heard ----------------------

class TestHeardWordLints(unittest.TestCase):
    def test_hiding_an_opening_filler_keeps_the_fail(self):
        words = [("Done.", 0.5, 1.0), ("Um,", 2.0, 2.3), ("people", 2.42, 2.8),
                 ("learn", 2.9, 3.2), ("fast.", 3.3, 3.8), ("Next.", 6.0, 6.4)]
        plain = build(words, {"c01": {"pieces": [{"words": [1, 4]}]}})[0]
        hidden = build(words, {"c01": {"pieces": [{"words": [1, 4]}], "fix": {"1": ""}}})[0]
        self.assertEqual(len(lints_of(plain, "L04", "FAIL")), 1)
        self.assertEqual(len(lints_of(hidden, "L04", "FAIL")), 1)
        self.assertEqual([s["src"] for s in plain["segments"]],
                         [s["src"] for s in hidden["segments"]])

    def test_hidden_first_word_before_a_filler_is_not_a_filler_opener(self):
        words = [("Done.", 0.5, 1.0), ("Well", 2.0, 2.3), ("um", 2.42, 2.7), ("we", 2.8, 3.1),
                 ("sat", 3.2, 3.5), ("down.", 3.6, 4.0)]
        plan = build(words, {"c01": {"pieces": [{"words": [1, 5]}], "tighten": False,
                                     "fix": {"1": ""}}})[0]
        self.assertFalse(lints_of(plan, "L04", "FAIL"))

    def test_display_merge_on_the_last_two_words_has_no_closer_lint(self):
        toks = "We sat down out on the old back deck.".split()
        words = seq_words(len(toks), text=toks)
        plain = build(words, {"c01": {"pieces": [{"words": [0, 8]}]}})[0]
        merged = build(words, {"c01": {"pieces": [{"words": [0, 8]}],
                                       "fix": {"7": "back deck.", "8": ""}}})[0]
        self.assertFalse(lints_of(plain, "L05"))
        self.assertFalse(lints_of(merged, "L05"))

    def test_display_merge_on_the_first_two_words_has_no_opener_lint(self):
        toks = "Back deck was built last year.".split()
        words = [("Done.", 0.5, 1.0)] + seq_words(len(toks), start=3.0, text=toks)
        for fix in ({"1": "", "2": "Back deck"}, {"1": "Back deck", "2": ""}):
            plan = build(words, {"c01": {"pieces": [{"words": [1, 6]}], "fix": fix}})[0]
            self.assertFalse(lints_of(plan, "L04"), fix)

    def test_a_clipped_close_is_still_reported_under_a_merge(self):
        toks = "We sat down out on the old back deck and then".split()
        words = seq_words(len(toks), text=toks)
        plan = build(words, {"c01": {"pieces": [{"words": [0, 8]}],
                                     "fix": {"7": "back deck", "8": ""}}})[0]
        text = " ".join(msgs(plan, "L05"))
        self.assertIn("'deck'", text)            # judged on the last word heard, not "back"
        self.assertNotIn("'back'", text)


# ---- P5: L06 at a piece boundary ------------------------------------------------------

class TestL06AtPieceBoundary(unittest.TestCase):
    W = [("This", 1.0, 1.2), ("is", 1.3, 1.5), ("not", 1.6, 1.8), ("easy.", 1.9, 2.3),
         ("Next.", 5.0, 5.4)]

    def test_every_way_of_dropping_the_negation_is_named_once(self):
        cases = [
            {"pieces": [{"words": [0, 3]}], "omit": [2]},
            {"pieces": [{"words": [0, 1]}, {"words": [3, 3]}]},
            {"pieces": [{"words": [0, 2]}, {"words": [3, 3]}], "omit": [2]},
            {"pieces": [{"words": [0, 1]}, {"words": [2, 3]}], "omit": [2]},
        ]
        for clip in cases:
            got = msgs(build(self.W, {"c01": clip})[0], "L06", "WARN")
            self.assertEqual(len(got), 1, (clip, got))
            self.assertIn("'not' (word 2)", got[0])

    def test_hedge_omitted_as_the_first_word_of_the_next_piece(self):
        w = [("We", 1.0, 1.2), ("will", 1.3, 1.5), ("probably", 1.6, 2.0), ("ship", 2.1, 2.4),
             ("Friday.", 2.5, 3.0), ("Next.", 6.0, 6.4)]
        got = msgs(build(w, {"c01": {"pieces": [{"words": [0, 1]}, {"words": [2, 4]}],
                                     "omit": [2]}})[0], "L06")
        self.assertEqual(len(got), 1)
        self.assertIn("'probably' (word 2)", got[0])

    def test_restart_at_the_edge_is_not_reported(self):
        toks = "I just, I just think so.".split()
        plan = build(seq_words(len(toks), text=toks),
                     {"c01": {"pieces": [{"words": [0, 1]}, {"words": [2, 5]}], "omit": [1]}})[0]
        self.assertFalse(lints_of(plan, "L06"))

    def test_pieces_more_than_five_words_apart_are_not_scanned(self):
        toks = "We did it not once but twice over many long years there.".split()
        plan = build(seq_words(len(toks), text=toks),
                     {"c01": {"pieces": [{"words": [0, 3]}, {"words": [10, 11]}], "omit": [3]}})[0]
        self.assertFalse(lints_of(plan, "L06"))


# ---- P6: offsets at a shared cut ------------------------------------------------------

class TestOffsetsAtSharedCut(unittest.TestCase):
    def _one(self, d_out=0, d_in=0, gap=0.07, spans=None):
        words = seq_words(8, start=1.0, wlen=0.3, gap=gap)
        words[-1] = (words[-1][0] + ".", words[-1][1], words[-1][2])
        p0, p1 = {"words": [0, 3]}, {"words": [4, 7]}
        if d_out:
            p0["out_offset"] = d_out
        if d_in:
            p1["in_offset"] = d_in
        return build(words, {"c01": {"pieces": [p0, p1]}}, fps=(30, 1), spans=spans)[0]

    def test_trimming_out_offset_moves_frame_by_frame(self):
        self.assertEqual(self._one()["frames_out"], 101)
        for d, frames in ((-1, 100), (-2, 99), (-3, 98)):
            plan = self._one(d_out=d)
            e = plan["pieces"][0]["edge_out"]
            self.assertEqual(plan["frames_out"], frames, d)
            self.assertEqual(e.get("offset"), d)
            self.assertNotEqual(e.get("guard"), "shared")

    def test_trimming_in_offset_moves_frame_by_frame(self):
        for d, frames in ((1, 100), (2, 99), (3, 98)):
            plan = self._one(d_in=d)
            self.assertEqual(plan["frames_out"], frames, d)
            self.assertNotEqual(plan["pieces"][0]["edge_out"].get("guard"), "shared")
            self.assertEqual(plan["pieces"][1]["edge_in"].get("offset"), d)

    def test_other_offsets_and_no_offsets_stay_one_seamless_cut(self):
        for kw in ({}, {"d_out": 1}, {"d_out": 2}, {"d_in": -1}, {"d_in": -2}):
            plan = self._one(**kw)
            self.assertEqual(plan["frames_out"], 101, kw)
            self.assertEqual(plan["pieces"][0]["edge_out"].get("guard"), "shared", kw)
            self.assertTrue(plan["joins"][0]["seamless"], kw)

    def test_an_offset_the_overlap_absorbs_is_named(self):
        # continuous speech: the two snapped edges overlap, so out_offset -1 clears nothing
        base = self._one(gap=0.0, spans=[(1.0, 3.4)])
        plan = self._one(d_out=-1, gap=0.0, spans=[(1.0, 3.4)])
        self.assertEqual(plan["frames_out"], base["frames_out"])
        info = [m for m in msgs(plan, "L01", "INFO") if "removed 0 frames" in m]
        self.assertEqual(len(info), 1)
        self.assertIn("pieces[0] out_offset -1", info[0])
        self.assertFalse([m for m in msgs(base, "L01", "INFO") if "removed 0 frames" in m])
        self.assertLess(self._one(d_out=-2, gap=0.0, spans=[(1.0, 3.4)])["frames_out"],
                        base["frames_out"])


# ---- P3: hook card against the planned length -------------------------------------------

class TestHookCardCoversPlannedClip(unittest.TestCase):
    def _hook(self, dur):
        return {"lines": ["A LINE"], "dur": dur}

    def test_tighten_shrinks_the_clip_under_the_card(self):
        plan = build([("Hello", 1.0, 1.3), ("world.", 20.0, 20.3)],
                     {"c01": {"pieces": [{"words": [0, 1]}], "hook_card": self._hook(3)}})[0]
        self.assertLess(plan["duration_out"], 2.0)
        self.assertEqual(len(covers_fail(plan)), 1)

    def test_omit_shrinks_the_clip_under_the_card(self):
        plan = build(seq_words(12), {"c01": {"pieces": [{"words": [0, 11]}], "omit": [[1, 10]],
                                             "hook_card": self._hook(2.5)}})[0]
        self.assertEqual(len(covers_fail(plan)), 1)

    def test_caption_only_clip_is_checked(self):
        media = {"duration": 4.0, "video": {"fps": [25, 1], "start": 0.0, "width": 1080,
                                            "height": 1920, "frames": 100, "hdr": "none"}}
        tr = fake_transcript([("One", 0.5, 0.9), ("two", 1.0, 1.4), ("three", 1.5, 1.9),
                              ("four.", 2.0, 2.4)], media="done.mp4")
        for dur, want in ((30, 1), (2.0, 0), ("auto", 0)):
            sp = mkspec({"c01": {"kind": "caption_only", "source_file": "done.mp4",
                                 "hook_card": self._hook(dur)}})
            self.assertEqual(len(covers_fail(P.plan_caption_only(sp, "c01", tr, media))), want, dur)

    def test_long_clip_and_no_duplicate(self):
        long = build(seq_words(20, gap=2.0), {"c01": {"pieces": [{"words": [0, 19]}],
                                                      "hook_card": self._hook(3)}})[0]
        self.assertFalse(covers_fail(long))
        both = build(seq_words(8), {"c01": {"pieces": [{"words": [0, 7]}],
                                            "hook_card": self._hook(3)}})[0]
        self.assertEqual(len(covers_fail(both)), 1)    # the spec check fired: not said twice


# ---- P4: motion values ------------------------------------------------------------------

class TestMotionValues(unittest.TestCase):
    def _clip(self, motion):
        sp = mkspec({"c01": {"pieces": [{"words": [0, 7]}], "motion": motion}})
        return sp, sp["clips"]["c01"]

    def test_bad_values_fail_and_never_reach_framing(self):
        from hookcutter import framing, render_util
        preset = render_util.load_preset("clean", None, "vertical")
        for bad in ("off", [1.0], {"enabled": False}, True, float("nan"), -1):
            sp, clip = self._clip({"punch_in": bad})
            fails = l01(clip["_issues"], "FAIL")
            self.assertTrue(any("motion.punch_in" in m for m in fails), repr(bad))
            self.assertNotIn("punch_in", clip["motion"])
            plan = P.plan_clip(sp, "c01", fake_transcript(seq_words(8)), source(),
                               build(seq_words(8), {"c01": {"pieces": [{"words": [0, 7]}]}})[2])
            framing.build_tracks(plan, clip, None, source(), "vertical", preset)   # no raise
        for key, bad in (("min_gap", "2s"), ("max_upscale", "3x")):
            _sp, clip = self._clip({key: bad})
            self.assertTrue(any(f"motion.{key}" in m for m in l01(clip["_issues"], "FAIL")))
            self.assertNotIn(key, clip["motion"])

    def test_valid_motion_is_stored_as_written(self):
        for motion in ({"punch_in": 1.08, "_note": "x"}, {"punch_in": "1.0"}, {"min_gap": 0},
                       {"max_upscale": None}, {}):
            sp, clip = self._clip(dict(motion))
            self.assertEqual(clip["_issues"], [], motion)
            self.assertEqual(clip["motion"], motion)
            self.assertEqual(S.clip_hash(sp, "c01"), old_hash(sp, "c01"))

    def test_unknown_motion_key_warns_once(self):
        _sp, clip = self._clip({"punchin": 1.0})
        self.assertEqual(len([m for m in l01(clip["_issues"], "WARN") if "'punchin'" in m]), 1)
        self.assertFalse(l01(clip["_issues"], "FAIL"))
        _sp, clip = self._clip("fast")
        self.assertTrue(any("motion must be an object" in m for m in l01(clip["_issues"], "FAIL")))


# ---- P7 / V10-N1: comment keys and post ---------------------------------------------------

class TestCommentKeysAndPost(unittest.TestCase):
    def test_underscore_keys_are_comments_everywhere(self):
        sp = mkspec({"_note": "x", "_c02-off": {"pieces": [{"words": [0, 1]}]},
                     "a": {"pieces": [{"words": [0, 3]}], "fix": {"_note": "x", "3": "units"},
                           "motion": {"_note": "x", "punch_in": 1.0},
                           "post": {"_note": "x", "title": "T"}}},
                    defaults={"length": {"_note": "x", "vertical": [10, 20, 30, 40]}})
        self.assertEqual([x for x in S.validate(sp) if x["id"] == "L01"], [])
        self.assertEqual(list(sp["clips"]), ["a"])
        self.assertEqual(sp["clips"]["a"]["fix"], {"3": "units"})
        self.assertEqual(sp["defaults"]["length"]["vertical"], [10.0, 20.0, 30.0, 40.0])
        self.assertEqual(sp["clips"]["a"]["post"], {"_note": "x", "title": "T"})

    def test_unknown_post_key_warns_and_is_still_exported(self):
        post = {"title": "T", "description": "We made $99M", "captin": "t"}
        sp = mkspec({"a": {"pieces": [{"words": [0, 3]}], "post": dict(post)}})
        warns = l01(sp["clips"]["a"]["_issues"], "WARN")
        for key in ("description", "captin"):
            hit = [m for m in warns if repr(key) in m]
            self.assertEqual(len(hit), 1, key)
            self.assertIn("export manifest", hit[0])
            self.assertNotIn("(ignored)", hit[0])
        self.assertEqual(sp["clips"]["a"]["post"], post)
        self.assertEqual(S.clip_hash(sp, "a"), old_hash(sp, "a"))

    def test_post_title_and_caption_must_be_strings(self):
        sp = mkspec({"a": {"pieces": [{"words": [0, 3]}], "post": {"title": 7, "caption": ["x"]}}})
        fails = l01(sp["clips"]["a"]["_issues"], "FAIL")
        self.assertTrue(any("post.title" in m for m in fails))
        self.assertTrue(any("post.caption" in m for m in fails))
        ok = mkspec({"a": {"pieces": [{"words": [0, 3]}], "post": {"title": None, "caption": "c"}}})
        self.assertEqual(ok["clips"]["a"]["_issues"], [])


# ---- P8 / V10-N2: safezone ----------------------------------------------------------------

class TestSafezone(unittest.TestCase):
    EFF_KEYS = {"styles", "formats", "tighten", "safezone", "length"}

    def test_marker_only_when_the_file_names_a_safezone(self):
        sp = mkspec({"a": {"pieces": [{"words": [0, 3]}]}})
        self.assertEqual(set(sp["clips"]["a"]["_eff"]), self.EFF_KEYS)
        self.assertNotIn("safezone_set", sp["defaults"])
        self.assertEqual(S.clip_hash(sp, "a"), old_hash(sp, "a"))
        named = mkspec({"a": {"pieces": [{"words": [0, 3]}]}}, defaults={"safezone": "organic"})
        self.assertIs(named["clips"]["a"]["_eff"].get("safezone_set"), True)
        self.assertIs(named["defaults"].get("safezone_set"), True)
        self.assertEqual(named["clips"]["a"]["_eff"]["safezone"], "organic")
        bad = mkspec({"a": {"pieces": [{"words": [0, 3]}]}}, defaults={"safezone": 7})
        self.assertNotIn("safezone_set", bad["clips"]["a"]["_eff"])
        self.assertTrue(l01(bad["_issues"], "FAIL"))

    @unittest.skipUnless(hasattr(presets, "zone_name"), "presets.zone_name is not in this tree")
    def test_resolver_reads_the_marker(self):
        free = mkspec({"a": {"pieces": [{"words": [0, 3]}]}})["clips"]["a"]["_eff"]
        named = mkspec({"a": {"pieces": [{"words": [0, 3]}]}},
                       defaults={"safezone": "organic"})["clips"]["a"]["_eff"]
        self.assertEqual(presets.zone_name(free, {"safezone": "ads"}, "vertical"), "ads")
        self.assertEqual(presets.zone_name(named, {"safezone": "ads"}, "vertical"), "organic")

    def _project(self, tmp, defaults=None, zones=None, preset=None, styles=None):
        root = Path(tmp)
        raw = {"version": 1, "clips": {"a": {"pieces": [{"words": [0, 3]}]}}}
        if defaults is not None:
            raw["defaults"] = defaults
        if styles:
            raw["clips"]["a"]["styles"] = styles
        env.write_json(root / "clips.json", raw)
        if zones is not None or preset is not None:
            (root / "presets").mkdir(exist_ok=True)
        if zones is not None:
            env.write_json(root / "presets" / "safezones.json", zones)
        if preset is not None:
            env.write_json(root / "presets" / "tall.json", preset)
        return root

    def test_unknown_defaults_safezone_fails_at_plan_time(self):
        with tempfile.TemporaryDirectory() as tmp:
            sp = S.load_spec(self._project(tmp, {"safezone": "organc"}))
            fails = l01(sp["_issues"], "FAIL")
            self.assertEqual(len(fails), 1)
            self.assertIn("'organc'", fails[0])
            self.assertIn("organic", fails[0])                   # lists what exists
            self.assertNotEqual(S.clip_hash(sp, "a"), old_hash(sp, "a"))
        with tempfile.TemporaryDirectory() as tmp:
            zones = {"studio": {"top": 100, "bottom": 300, "left": 60, "right": 60}}
            sp = S.load_spec(self._project(tmp, {"safezone": "studio"}, zones=zones))
            self.assertFalse(l01(sp["_issues"], "FAIL"))
        with tempfile.TemporaryDirectory() as tmp:
            sp = S.load_spec(self._project(tmp))
            self.assertFalse(l01(sp["_issues"], "FAIL"))
            self.assertEqual(S.clip_hash(sp, "a"), old_hash(sp, "a"))

    def test_a_preset_naming_an_unknown_profile_fails_when_the_resolver_reads_it(self):
        def zone_name(eff, preset, fmt):
            if (eff or {}).get("safezone_set"):
                return eff["safezone"]
            return (preset or {}).get("safezone") or "organic"
        preset = {"extends": "clean", "safezone": "nowhere"}
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(presets, "zone_name", zone_name, create=True):
            sp = S.load_spec(self._project(tmp, preset=preset, styles=["tall"]))
            fails = l01(sp["clips"]["a"]["_issues"], "FAIL")
            self.assertEqual(len(fails), 1)
            self.assertIn("'nowhere'", fails[0])
            self.assertIn("'tall'", fails[0])
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(presets, "zone_name", zone_name, create=True):
            sp = S.load_spec(self._project(tmp, {"safezone": "organic"}, preset=preset,
                                           styles=["tall"]))
            self.assertFalse(l01(sp["clips"]["a"]["_issues"], "FAIL"))   # the file's name wins


# ---- C2: cover.shift ----------------------------------------------------------------------

class TestCoverShift(unittest.TestCase):
    def test_negative_shift_is_a_fail_and_stored_as_zero(self):
        for bad in (-300, "-40", -0.6):
            sink = []
            out = S._norm_cover({"shift": bad}, S._Issues(sink, "c"))
            self.assertEqual(out["shift"], 0, bad)
            self.assertEqual(len(sink), 1, bad)
            self.assertEqual((sink[0]["id"], sink[0]["level"]), ("L01", "FAIL"))
            self.assertIn("cover.shift", sink[0]["msg"])

    def test_zero_and_positive_shifts_are_untouched(self):
        for ok, want in ((0, 0), (120, 120), ("80", 80), (None, 0)):
            sink = []
            self.assertEqual(S._norm_cover({"shift": ok}, S._Issues(sink, "c"))["shift"], want)
            self.assertEqual(sink, [])
        sp = mkspec({"a": {"pieces": [{"words": [0, 3]}], "cover": {"shift": 120}}})
        self.assertEqual(S.clip_hash(sp, "a"), old_hash(sp, "a"))


# ---- S1: the pin of a caption-only clip ---------------------------------------------------

class TestCaptionOnlyPin(unittest.TestCase):
    WORDS = [("Already", 0.5, 0.9), ("edited", 1.0, 1.4), ("video", 1.5, 1.9), ("here.", 2.0, 2.4)]
    MEDIA = {"duration": 4.0, "video": {"fps": [25, 1], "start": 0.0, "width": 1080,
                                        "height": 1920, "frames": 100, "hdr": "none"}}

    def _l03(self, clip_keys, tr):
        clip = {"kind": "caption_only", "source_file": "done.mp4"}
        clip.update(clip_keys)
        sp = mkspec({"c09": clip})
        plan = P.plan_caption_only(sp, "c09", tr, self.MEDIA)
        return [x for x in plan["lints"] if x["id"] == "L03"], sp

    def test_changed_words_fail_until_the_clip_is_pinned(self):
        tr = fake_transcript(self.WORDS, media="done.mp4")
        tr["text_changed_from"] = "0" * 64
        for keys in ({"fix": {"1": "EDITED"}}, {"emphasis": [2]},
                     {"insert": [{"after": 1, "text": "by hand", "src": [1.42, 1.48]}]}):
            got, _sp = self._l03(keys, tr)
            self.assertEqual([(x["id"], x["level"]) for x in got], [("L03", "FAIL")], keys)
            self.assertIn("--clip c09", got[0]["msg"])
            self.assertIn(tr["sha256"], got[0]["msg"])
        pinned, _sp = self._l03({"fix": {"1": "EDITED"}, "transcript_sha": tr["sha256"]}, tr)
        self.assertEqual(pinned, [])
        no_index, _sp = self._l03({}, tr)
        self.assertEqual(no_index, [])

    def test_a_first_transcript_needs_no_pin(self):
        got, _sp = self._l03({"fix": {"1": "EDITED"}}, fake_transcript(self.WORDS, media="done.mp4"))
        self.assertEqual(got, [])

    def test_a_stale_pin_fails_and_a_retime_keeps_it(self):
        tr = fake_transcript(self.WORDS, media="done.mp4")
        old = "a" * 64
        got, _sp = self._l03({"fix": {"1": "EDITED"}, "transcript_sha": old}, tr)
        self.assertEqual([(x["id"], x["level"]) for x in got], [("L03", "FAIL")])
        retimed = dict(tr, text_compatible_shas=[old], text_changed_from="b" * 64)
        got, _sp = self._l03({"fix": {"1": "EDITED"}, "transcript_sha": old}, retimed)
        self.assertEqual(got, [])

    def test_the_pin_enters_the_clip_only_when_written(self):
        base = {"kind": "caption_only", "source_file": "done.mp4", "fix": {"1": "EDITED"}}
        sp = mkspec({"c09": dict(base)})
        self.assertNotIn("transcript_sha", sp["clips"]["c09"])
        self.assertEqual(S.clip_hash(sp, "c09"), old_hash(sp, "c09"))
        pinned = mkspec({"c09": dict(base, transcript_sha="a" * 64)})
        self.assertEqual(pinned["clips"]["c09"]["transcript_sha"], "a" * 64)
        self.assertEqual(pinned["clips"]["c09"]["_issues"], [])
        self.assertNotEqual(S.clip_hash(pinned, "c09"), S.clip_hash(sp, "c09"))
        bad = mkspec({"c09": dict(base, transcript_sha=12)})
        self.assertTrue(any("transcript_sha" in m for m in l01(bad["clips"]["c09"]["_issues"], "FAIL")))
        self.assertNotIn("transcript_sha", bad["clips"]["c09"])

    def test_on_a_cut_clip_the_key_is_a_warning_and_changes_nothing(self):
        plain = mkspec({"a": {"pieces": [{"words": [0, 3]}]}})
        cut = mkspec({"a": {"pieces": [{"words": [0, 3]}], "transcript_sha": "a" * 64}})
        self.assertNotIn("transcript_sha", cut["clips"]["a"])
        self.assertEqual(len(l01(cut["clips"]["a"]["_issues"], "WARN")), 1)
        self.assertEqual(S.clip_hash(cut, "a"), S.clip_hash(plain, "a"))

    def test_the_pin_is_not_a_render_input(self):
        from hookcutter import render
        self.assertIn("transcript_sha", render.CLIP_NOT_RENDERED)


# ---- C6: the written surfaces come from one place ------------------------------------------

class TestWrittenSurfaces(unittest.TestCase):
    TEXT = "We built it in a small garage last year."

    def test_plan_lints_what_claims_says_is_written(self):
        seen = {}

        def surfaces(clip, plan=None):
            seen["plan"] = plan
            return [("cover headline (the title)", "We made $99M")]
        with mock.patch.object(claims, "written_surfaces", surfaces, create=True):
            plan = speech(self.TEXT, title="We made $99M")
        self.assertIsNotNone(seen.get("plan"))
        got = msgs(plan, "L13", "WARN")
        self.assertEqual(len(got), 1)
        self.assertTrue(got[0].startswith("cover headline (the title)"), got[0])
        self.assertIn("$99M", got[0])

    @unittest.skipUnless(hasattr(claims, "written_surfaces"), "claims.written_surfaces is not in this tree")
    def test_a_title_printed_on_the_cover_is_checked(self):
        self.assertTrue(any("$99M" in m for m in msgs(speech(self.TEXT, title="We made $99M"), "L13")))
        honest = speech(self.TEXT, title="Built in a garage",
                        hook_card={"lines": ["A SMALL *GARAGE*"]})
        self.assertFalse(lints_of(honest, "L13"))

    def test_hook_card_cover_and_post_are_still_checked(self):
        plan = speech(self.TEXT, hook_card={"lines": ["WE MADE *$5M*"]},
                      cover={"headline": "IN 40 DAYS"}, post={"title": "A 12 step plan"})
        text = " ".join(msgs(plan, "L13"))
        for claim in ("$5M", "40", "12"):
            self.assertIn(claim, text)


# ---- Q1: fix text that contradicts the audio ----------------------------------------------

class TestFixTextLint(unittest.TestCase):
    T = "We do not have five clients right now."

    def _fix_lints(self, text, fix):
        return [m for m in msgs(speech(text, fix), "L06", "WARN") if m.startswith("fix ")]

    def test_hidden_negation_and_changed_number_warn(self):
        got = self._fix_lints(self.T, {"2": ""})
        self.assertEqual(len(got), 1)
        self.assertIn("fix hides 'not' (word 2)", got[0])
        got = self._fix_lints(self.T, {"4": "fifty"})
        self.assertEqual(len(got), 1)
        self.assertIn("fix changes a number 'five' -> 'fifty' (word 4)", got[0])
        self.assertEqual(len(self._fix_lints(self.T, {"4": ""})), 1)       # the number is hidden
        slip = self._fix_lints("It cost twenty five grand to build.", {"2": "$250K", "3": "", "4": ""})
        self.assertEqual(len(slip), 1)
        self.assertIn("words 2-4", slip[0])
        self.assertEqual(len(self._fix_lints("We will probably ship it Friday.", {"2": ""})), 1)

    def test_honest_fixes_are_quiet(self):
        quiet = [
            (self.T, {"4": "5"}),
            (self.T, {"1": "don't", "2": ""}),
            (self.T, {"0": "WE"}),
            ("We sat on the back deck all night.", {"4": "back deck", "5": ""}),
            ("That was some shit right there.", {"3": "s***"}),
            ("We run cooper netties in production.", {"2": "Kubernetes", "3": ""}),
            ("It cost twenty five grand to build.", {"2": "$25K", "3": "", "4": ""}),
            ("We start in twenty twenty six for real.", {"3": "2026", "4": "", "5": ""}),
            ("It cost 10 K to build.", {"2": "10K", "3": ""}),
            ("We grew 80 percent last year.", {"2": "80%", "3": ""}),
            ("We grew just a little there.", {"2": "Just"}),
        ]
        for text, fix in quiet:
            self.assertEqual(self._fix_lints(text, fix), [], (text, fix))
        self.assertEqual(self._fix_lints(self.T, None), [])

    def test_the_lint_changes_no_segment(self):
        self.assertEqual(speech(self.T)["segments"], speech(self.T, {"4": "fifty"})["segments"])

    def test_caption_only_clips_are_linted_too(self):
        toks = self.T.split()
        tr = fake_transcript(seq_words(len(toks), start=0.2, text=toks), media="done.mp4")
        media = {"duration": 5.0, "video": {"fps": [25, 1], "start": 0.0, "width": 1080,
                                            "height": 1920, "frames": 125, "hdr": "none"}}
        sp = mkspec({"c09": {"kind": "caption_only", "source_file": "done.mp4", "fix": {"2": ""}}})
        got = [x["msg"] for x in P.plan_caption_only(sp, "c09", tr, media)["lints"]
               if x["id"] == "L06"]
        self.assertEqual(len(got), 1)
        self.assertIn("fix hides 'not'", got[0])


# ---- V09-N2: a fix on a word the transcript wrote as a filler --------------------------------

class TestCorrectedFiller(unittest.TestCase):
    WORDS = [("I", 1.0, 1.2), ("was", 1.3, 1.6), ("um", 1.9, 2.2), ("there.", 2.5, 3.0)]

    def _plan(self, fix=None):
        clip = {"pieces": [{"words": [0, 3]}]}
        if fix:
            clip["fix"] = fix
        return build(self.WORDS, {"c01": clip})[0]

    def test_a_corrected_filler_is_kept_captioned_and_in_intent(self):
        plan = self._plan({"2": "a"})
        w = next(x for x in plan["words"] if x["i"] == 2)
        self.assertFalse(w["filler"] or w["removed"] or w["hidden"])
        self.assertEqual(w["text"], "a")
        self.assertIsNotNone(w["s_out"])
        self.assertIn(2, plan["intent"])
        self.assertFalse([r for r in plan["removals"] if r["kind"] == "filler"])

    def test_without_a_fix_or_with_a_filler_fix_nothing_changes(self):
        base = self._plan()
        for fix in (None, {"2": ""}, {"2": "uh"}):
            plan = self._plan(fix)
            w = next(x for x in plan["words"] if x["i"] == 2)
            self.assertTrue(w["filler"] and w["removed"], fix)
            self.assertNotIn(2, plan["intent"])
            self.assertEqual(plan["segments"], base["segments"], fix)

    def test_caption_only_words_follow_the_same_rule(self):
        tr = fake_transcript([("It", 0.5, 0.7), ("was", 0.8, 1.0), ("uh", 1.1, 1.3),
                              ("mess.", 1.4, 1.8)], media="done.mp4")
        media = {"duration": 3.0, "video": {"fps": [25, 1], "start": 0.0, "width": 1080,
                                            "height": 1920, "frames": 75, "hdr": "none"}}
        for fix, filler in (({"2": "a"}, False), ({}, True), ({"2": ""}, True)):
            sp = mkspec({"c09": {"kind": "caption_only", "source_file": "done.mp4", "fix": fix}})
            plan = P.plan_caption_only(sp, "c09", tr, media)
            self.assertEqual(plan["words"][2]["filler"], filler, fix)
            self.assertEqual(2 in plan["intent"], not filler, fix)


# ---- R8 / R7 / V06-N1 / R6: plan freshness against real project files ------------------------

class TestPlanFreshness(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.dir = Path(cls.tmp.name)
        text = ("People always ask me how to learn fast. The honest answer is that you have to "
                "look silly first. And nobody wants that.").split()
        words, t = [], 1.0
        for k, w in enumerate(text):
            words.append((w, round(t, 3), round(t + 0.3, 3), "Host" if k < 8 else "Guest"))
            t += 0.42 + (1.2 if w.endswith(".") else 0.0)
        cls.words = words
        cls.media = make_speech_fixture(cls.dir / "media", words, fps="25")
        cls.n = 0

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def _project(self, clips, **top):
        type(self).n += 1
        proj = make_project(self.dir / f"p{self.n}", self.media, self.words)
        self._write(proj, clips, **top)
        return proj

    @staticmethod
    def _write(proj, clips, **top):
        env.write_json(proj / "clips.json", dict({"version": 1, "clips": clips}, **top))

    @staticmethod
    def _edit(path, **keys):
        data = env.read_json(path)
        data.update(keys)
        env.write_json(path, data)
        return data

    # -- R8: an edit the normaliser rejects is still an edit

    def test_hash_is_unchanged_for_clean_and_warn_only_clips(self):
        clean = mkspec({"c01": {"pieces": [{"words": [0, 7], "hold_after": 0.4}], "fix": {"3": "ME"},
                                "omit": [5]}})
        self.assertEqual(S.clip_hash(clean, "c01"), old_hash(clean, "c01"))
        warn = mkspec({"c01": {"pieces": [{"words": [0, 7]}], "hold_afer": 1}})
        self.assertTrue(l01(warn["clips"]["c01"]["_issues"], "WARN"))
        self.assertEqual(S.clip_hash(warn, "c01"), old_hash(warn, "c01"))

    def test_a_rejected_edit_makes_the_plan_stale_and_heals_when_removed(self):
        good = {"c01": {"pieces": [{"words": [0, 7]}]}}
        for bad in ({"c01": {"pieces": [{"words": [0, 7], "out_offset": 61}]}},
                    {"c01": {"pieces": [{"words": [0, 7]}], "fix": {"5": None}}},
                    {"c01": {"pieces": [{"words": [0, 7], "hold_after": -1}]}}):
            proj = self._project(good)
            plan = P.build_plan(proj, "c01")
            self.assertFalse(P.fail_lints(plan))
            self._write(proj, bad)
            self.assertNotEqual(S.clip_hash(S.load_spec(proj), "c01"), plan["spec_hash"])
            with self.assertRaises(env.HCError) as cm:
                P.load_plan(proj, "c01", require_ok=True)
            self.assertIn("L01", cm.exception.msg)
            self._write(proj, good)
            again = P.load_plan(proj, "c01", require_ok=True)
            self.assertFalse(P.fail_lints(again))
            self.assertEqual(again["spec_hash"], plan["spec_hash"])

    def test_the_transcript_pin_verdict_is_followed(self):
        clips = {"c01": {"pieces": [{"words": [0, 7]}]}}
        proj = self._project(clips)
        sha = env.read_json(proj / "transcript.json")["sha256"]
        plan = P.build_plan(proj, "c01")
        path = proj / "plans" / "c01.plan.json"
        mtime = path.stat().st_mtime_ns
        time.sleep(0.01)
        self._write(proj, clips, transcript_sha=sha)             # a pin that holds: nothing to redo
        self.assertEqual(P.load_plan(proj, "c01", require_ok=True)["segments"], plan["segments"])
        self.assertEqual(path.stat().st_mtime_ns, mtime)
        self._write(proj, clips, transcript_sha="deadbeef" * 8)
        with self.assertRaises(env.HCError) as cm:
            P.load_plan(proj, "c01", require_ok=True)
        self.assertIn("L03", cm.exception.msg)
        self._write(proj, clips, transcript_sha=sha)             # corrected: the FAIL goes away
        self.assertFalse(P.fail_lints(P.load_plan(proj, "c01", require_ok=True)))

    # -- R7: the transcript must come from the source in the project

    def test_transcript_from_another_source_stops_planning(self):
        proj = self._project({"c01": {"pieces": [{"words": [0, 7]}]}})
        self.assertFalse(P.fail_lints(P.build_plan(proj, "c01")))
        real = env.read_json(proj / "transcript.json")["source_fingerprint"]
        self.assertEqual(real, env.read_json(proj / "source.json")["fingerprint"])
        self._edit(proj / "transcript.json", source_fingerprint="f" * 64)
        for call in (lambda: P.build_plan(proj, "c01"), lambda: P.load_plan(proj, "c01"),
                     lambda: P.current_plan(proj, "c01"),
                     lambda: P.cmd_plan(SimpleNamespace(project=str(proj), clips=[], quiet=True))):
            with self.assertRaises(env.HCError) as cm:
                call()
            self.assertEqual(cm.exception.code, 1)
            self.assertIn("transcript.json", cm.exception.msg)
            self.assertIn("hc transcribe", cm.exception.msg)
        # the transcript is re-made from the right file (same words): no stale FAIL remains
        self._edit(proj / "transcript.json", source_fingerprint=real)
        self.assertFalse(P.fail_lints(P.load_plan(proj, "c01", require_ok=True)))

    def test_a_cfr_re_encode_keeps_the_transcript_valid(self):
        proj = self._project({"c01": {"pieces": [{"words": [0, 7]}]}})
        self._edit(proj / "transcript.json", source_fingerprint="e" * 64)
        self._edit(proj / "source.json", ingest={"method": "file", "cfr": {
            "from_avg_fps": 29.5, "fps": [30, 1], "from_fingerprint": "e" * 64}})
        self.assertFalse(P.fail_lints(P.build_plan(proj, "c01")))
        # an older record without the key is trusted; a record naming a third file is not
        self._edit(proj / "source.json", ingest={"method": "file", "cfr": {"fps": [30, 1]}})
        self.assertFalse(P.fail_lints(P.load_plan(proj, "c01")))
        self._edit(proj / "source.json", ingest={"method": "file", "cfr": {
            "fps": [30, 1], "from_fingerprint": "d" * 64}})
        with self.assertRaises(env.HCError):
            P.load_plan(proj, "c01")

    def test_unknown_fingerprints_do_not_stop_planning(self):
        proj = self._project({"c01": {"pieces": [{"words": [0, 7]}]}})
        self._edit(proj / "transcript.json", source_fingerprint=None)
        self.assertFalse(P.fail_lints(P.build_plan(proj, "c01")))

    # -- V06-N1: the same for a caption-only clip's own file

    def test_caption_only_transcript_from_another_version_of_the_file(self):
        proj = self._project({"c03": {"kind": "caption_only", "source_file": "finished.mp4"}})
        av = make_av_fixture(self.dir / "co", fps="25", dur=4, flashes=[1.0])
        if not (proj / "finished.mp4").exists():
            os.symlink(av, proj / "finished.mp4")
        words = [("Already", 0.5, 0.9), ("edited", 1.0, 1.4), ("video.", 1.5, 2.0)]
        tpath = proj / "transcripts" / "c03.json"
        env.write_json(tpath, fake_transcript(words, media="finished.mp4", fingerprint="9" * 64))
        with self.assertRaises(env.HCError) as cm:
            P.build_plan(proj, "c03")
        self.assertEqual(cm.exception.code, 1)
        self.assertIn("transcripts/c03.json", cm.exception.msg)
        self.assertIn("--clip c03", cm.exception.msg)
        env.write_json(tpath, fake_transcript(words, media="finished.mp4",
                                              fingerprint=env.fingerprint(proj / "finished.mp4")))
        plan = P.build_plan(proj, "c03")
        self.assertFalse(P.fail_lints(plan))
        spec = S.load_spec(proj)
        self.assertFalse(P._stale(P._as_project(proj), spec, "c03", plan))
        # the file is replaced after the plan was made: the fresh-looking plan is rebuilt
        self._edit(tpath, source_fingerprint="8" * 64)
        self.assertTrue(P._stale(P._as_project(proj), spec, "c03", plan))
        with self.assertRaises(env.HCError):
            P.load_plan(proj, "c03")

    # -- R6: speaker labels are part of what a plan was made from

    def _relabel(self, proj, change):
        tr = env.read_json(proj / "transcript.json")
        sha = tr["sha256"]
        for w in tr["words"]:
            w["sp"] = change(w)
        env.write_json(proj / "transcript.json", tr)
        self.assertEqual(env.read_json(proj / "transcript.json")["sha256"], sha)
        return tr

    def test_relabelled_speakers_make_the_plan_stale(self):
        n = len(self.words)
        proj = self._project({"c01": {"pieces": [{"words": [0, n - 1]}]}},
                             speakers={"Host": {"x": 0.3}, "Guest": {"x": 0.7}})
        plan = P.build_plan(proj, "c01")
        path = proj / "plans" / "c01.plan.json"
        project, spec = P._as_project(proj), S.load_spec(proj)
        self.assertFalse(P._stale(project, spec, "c01", env.read_json(path)))
        self.assertFalse(P.transcript_changed(plan, env.read_json(proj / "transcript.json")))
        tr = self._relabel(proj, lambda w: {"Host": "Guest", "Guest": "Host"}[w["sp"]])
        self.assertTrue(P.transcript_changed(plan, tr))
        self.assertTrue(P._stale(project, spec, "c01", env.read_json(path)))
        fresh = P.load_plan(proj, "c01")
        for w in fresh["words"]:
            self.assertEqual(w["sp"], tr["words"][w["i"]]["sp"])
        self.assertEqual([w["sp"] for w in env.read_json(path)["words"]],
                         [w["sp"] for w in fresh["words"]])            # the file was rewritten
        self.assertFalse(P._stale(project, spec, "c01", env.read_json(path)))

    def test_a_label_outside_the_clip_changes_nothing(self):
        proj = self._project({"c01": {"pieces": [{"words": [0, 7]}]}})
        plan = P.build_plan(proj, "c01")
        path = proj / "plans" / "c01.plan.json"
        mtime = path.stat().st_mtime_ns
        time.sleep(0.01)
        tr = self._relabel(proj, lambda w: "Caller" if w["i"] == 16 else w["sp"])
        self.assertFalse(P.transcript_changed(plan, tr))
        self.assertFalse(P._stale(P._as_project(proj), S.load_spec(proj), "c01", plan))
        P.load_plan(proj, "c01")
        self.assertEqual(path.stat().st_mtime_ns, mtime)

    def test_transcript_changed_on_plain_dicts(self):
        tr = fake_transcript([("One", 0.5, 0.9, "A"), ("two.", 1.0, 1.4, "A")])
        plan = {"transcript_sha": tr["sha256"],
                "words": [{"i": 0, "sp": "A"}, {"i": None, "sp": None}, {"i": 1, "sp": "A"}]}
        self.assertFalse(P.transcript_changed(plan, tr))
        self.assertTrue(P.transcript_changed(dict(plan, transcript_sha="x"), tr))
        self.assertTrue(P.transcript_changed(plan, dict(tr, words=tr["words"][:1])))
        self.assertTrue(P.transcript_changed(plan, None))
        self.assertFalse(P.transcript_changed({"transcript_sha": None, "words": []}, None))


# ---- A3: a noisy decode must not hang ----------------------------------------------------

@unittest.skipIf(sys.platform.startswith("win"), "the fake ffmpeg is a shell script")
class TestDecodeEnergyStderr(unittest.TestCase):
    SCRIPT = """#!/bin/sh
i=0
while [ $i -lt 4000 ]; do
  echo "[dec @ 0x0] frame $i could not be decoded: error error error error error error error" >&2
  i=$((i+1))
done
head -c 128000 /dev/zero
exit %d
"""

    def _decode(self, tmp, code):
        fake = Path(tmp) / "fake-ffmpeg"
        fake.write_text(self.SCRIPT % code, encoding="utf-8")
        fake.chmod(fake.stat().st_mode | stat.S_IXUSR)
        procs, box = [], {}
        real_popen = subprocess.Popen

        def spy(*a, **kw):
            p = real_popen(*a, **kw)
            procs.append(p)
            return p

        def work():
            try:
                box["db"] = audio.decode_energy(Path(tmp) / "x.mp4")
            except Exception as exc:                   # handed to the test thread
                box["exc"] = exc

        with mock.patch.object(env, "ffmpeg", return_value=str(fake)), \
                mock.patch.object(audio.subprocess, "Popen", side_effect=spy), \
                mock.patch.object(env, "log"):
            th = threading.Thread(target=work, daemon=True)
            th.start()
            th.join(timeout=20)
            if th.is_alive():
                for p in procs:
                    p.kill()
                th.join(timeout=5)
                self.fail("decode_energy hung on a decode that writes much to stderr")
        return box

    def test_much_stderr_does_not_block_the_decode(self):
        with tempfile.TemporaryDirectory() as tmp:
            box = self._decode(tmp, 0)
        self.assertNotIn("exc", box)
        self.assertEqual(len(box["db"]), 100)          # one second of 10 ms hops

    def test_a_failed_decode_reports_the_stderr_tail(self):
        with tempfile.TemporaryDirectory() as tmp:
            box = self._decode(tmp, 1)
        self.assertIsInstance(box.get("exc"), env.HCError)
        self.assertEqual(box["exc"].code, 3)
        self.assertIn("frame 3999 could not be decoded", box["exc"].msg)


if __name__ == "__main__":
    unittest.main()
