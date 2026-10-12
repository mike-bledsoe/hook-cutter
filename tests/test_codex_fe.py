"""Regression tests for the QA-gate fixes of the outside review (cluster FE).

G12: a cover whose words or whose render are no longer the clip's is a FAIL, a negative
recorded shift is a FAIL, the frame speaker is matched the way the cover step matches it.
G11: the written surfaces come from one shared definition. G9: base frames per style, the
safe zone the render used, a face-tool failure is a tool error. G3: the audio checksum of a
single render, and a one-sample timestamp wobble is not a hole. G16: `rect` tiles, window
position, rows. Reports carry the identity of the checks that made them.

Everything is synthetic: hand-written plan / cover / shots / log json and tiny ffmpeg files.
"""

from __future__ import annotations

import contextlib
import io
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "scripts"))

import numpy as np  # noqa: E402

import qa_media as qm  # noqa: E402
from hookcutter import claims, env, presets, qa  # noqa: E402
from hookcutter import cover_frames as cf  # noqa: E402
from hookcutter import plan as plan_mod  # noqa: E402
from hookcutter import render as render_mod  # noqa: E402
from hookcutter.project import Project  # noqa: E402


def ctx_for(project=None, spec=None, asr=False):
    return qa._Ctx(project, spec or {"clips": {}}, None, asr, False)


# ---------------------------------------------------------------- G12

class CoverBase(unittest.TestCase):
    def setUp(self):
        from hookcutter import cover
        self.cover = cover
        self.tmp = Path(tempfile.mkdtemp(prefix="hc-codexfe-g12-"))
        self.P = Project(self.tmp, create=True)
        self.spec = {"clips": {"c1": {"speaker": "Speaker A"}}}
        self.ctx = qa._Ctx(self.P, self.spec, None, False, False)
        img = cover.Image.new("RGB", (1080, 1920), (90, 100, 120))
        cover.save_jpeg(img, self.P.cover_path("c1", "clean", "main"))
        for kind in ("grid", "mobile"):
            cover.save_jpeg(img.resize((270, 360)), self.P.cover_path("c1", "clean", kind))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def meta(self, **kw):
        m = {"headline_box": [80, 1200, 900, 300], "text_box": [80, 1200, 900, 300],
             "face_box": [300, 400, 400, 500], "faces_found": 1, "crown_y": 350, "crown_method": "image",
             "band": [40, 1900], "warnings": [], "spec_hash": "h"}
        m.update(kw)
        env.write_json(self.P.cover_path("c1", "clean", "json"), m)

    def render_log(self, created):
        env.write_json(self.P.render_path("c1", "vertical", "clean", ".log.json"), {"created": created})

    def g12(self, stale=False, plan=None):
        return qa.g12_cover(self.ctx, "c1", "clean", plan or {"spec_hash": "h"}, render_stale=stale)


class CoverIsTheClipsCover(CoverBase):
    """X3 / V01-N1: G12 FAILs a cover that is not the one `hc cover` would make now."""

    def test_headline_on_the_cover_differs_from_the_clip(self):
        self.spec["clips"]["c1"]["cover"] = {"headline": "NEW *WORDS*"}
        self.meta(headline="OLD *WORDS*")
        g = self.g12()
        self.assertEqual(g["status"], "FAIL", g["evidence"])
        self.assertIn("OLD *WORDS*", g["evidence"])
        self.assertIn("NEW *WORDS*", g["evidence"])
        self.assertIn("hc cover", g["evidence"])
        self.meta(headline="NEW *WORDS*")
        self.assertEqual(self.g12()["status"], "PASS", self.g12()["evidence"])

    def test_a_cover_json_without_a_headline_is_not_judged(self):
        self.spec["clips"]["c1"]["cover"] = {"headline": "NEW *WORDS*"}
        self.meta()
        self.assertEqual(self.g12()["status"], "PASS", self.g12()["evidence"])

    def test_hook_card_used_as_the_headline(self):
        plan = {"spec_hash": "h", "hook_card": {"lines": ["SELL IT", "*FIRST*"]}}
        self.meta(headline="SELL IT *LATER*")
        g = self.g12(plan=plan)
        self.assertEqual(g["status"], "FAIL", g["evidence"])
        self.assertIn("SELL IT *FIRST*", g["evidence"])
        self.meta(headline="SELL IT *FIRST*")
        self.assertEqual(self.g12(plan=plan)["status"], "PASS")

    def test_render_made_after_the_cover(self):
        # the two stamps come in different offset forms: cover "+00:00", render log "-0500"
        self.meta(created="2026-03-01T12:00:00+00:00")
        self.render_log("2026-03-01T08:00:00-0500")            # 13:00 UTC: one hour after the cover
        g = self.g12()
        self.assertEqual(g["status"], "FAIL", g["evidence"])
        self.assertIn("render", g["evidence"])
        self.assertIn("hc cover", g["evidence"])
        self.render_log("2026-03-01T06:00:00-0500")            # 11:00 UTC: before the cover
        self.assertEqual(self.g12()["status"], "PASS", self.g12()["evidence"])
        self.render_log("2026-03-01T07:00:00-0500")            # the same second: not later
        self.assertEqual(self.g12()["status"], "PASS")

    def test_unreadable_stamps_are_not_judged(self):
        self.meta(created="yesterday")
        self.render_log("2026-03-01T08:00:00-0500")
        self.assertEqual(self.g12()["status"], "PASS")
        self.meta(created="2026-03-01T12:00:00+00:00")
        self.render_log(None)
        self.assertEqual(self.g12()["status"], "PASS")

    def test_stale_render_says_render_first(self):
        self.spec["clips"]["c1"]["cover"] = {"headline": "NEW *WORDS*"}
        self.meta(headline="OLD *WORDS*")
        g = self.g12(stale=True)
        self.assertEqual(g["status"], "FAIL")
        self.assertIn("re-render, then re-cover", g["evidence"])
        self.assertLess(g["evidence"].index("hc render"), g["evidence"].rindex("hc cover"))

    def test_spec_hash_alone_stays_a_warn(self):
        self.meta(spec_hash="old")
        g = self.g12()
        self.assertEqual(g["status"], "WARN")
        self.assertIn("cover predates the current plan", g["evidence"])

    def test_stamp_parser(self):
        a = qa._stamp_time("2026-03-01T12:00:00+00:00")
        self.assertEqual(a, qa._stamp_time("2026-03-01T07:00:00-0500"))
        self.assertEqual(a, qa._stamp_time("2026-03-01T12:00:00Z"))
        self.assertEqual(a, qa._stamp_time("2026-03-01T12:00:00.000+0000"))
        for bad in (None, "", 12, "soon", "2026-03-01"):
            self.assertIsNone(qa._stamp_time(bad), bad)


class CoverShift(CoverBase):
    """C2: `cover.shift` only slides the photo up."""

    def test_no_smaller_shift_advice_at_shift_zero(self):
        self.meta(shift=0, crown_y=20, face_box=[300, 170, 400, 500])
        g = self.g12()
        self.assertEqual(g["status"], "FAIL")
        self.assertIn("face cut by the grid tile", g["evidence"])
        self.assertNotIn("smaller cover.shift", g["evidence"])
        self.assertIn("cover.at_out", g["evidence"])
        self.assertIn("only slides the photo up", g["evidence"])

    def test_smaller_shift_advice_when_there_is_a_shift(self):
        self.meta(shift=120, crown_y=20, face_box=[300, 170, 400, 500])
        g = self.g12()
        self.assertEqual(g["status"], "FAIL")
        self.assertIn("smaller cover.shift", g["evidence"])

    def test_negative_recorded_shift_fails(self):
        self.meta(shift=-300, face_box=[300, 400, 400, 500])
        g = self.g12()
        self.assertEqual(g["status"], "FAIL", g["evidence"])
        self.assertIn("negative shift", g["evidence"])
        self.assertIn("hc cover", g["evidence"])
        self.assertNotIn("-300 px headline shift pushed", g["evidence"])
        self.meta(shift=0, face_box=[300, 400, 400, 500])
        self.assertEqual(self.g12()["status"], "PASS")


class CoverSpeaker(CoverBase):
    """V04-N3: the frame speaker is matched the way `hc cover` picks the frame."""

    def test_prefix_and_case_match_like_the_cover_step(self):
        self.spec["clips"]["c1"]["speaker"] = "Sam Rivera"
        for shown in ("Sam", "sam rivera", "Sam Rivera"):
            self.meta(frame_speaker=shown)
            g = self.g12()
            self.assertEqual(g["status"], "PASS", (shown, g["evidence"]))

    def test_someone_else_still_warns(self):
        self.spec["clips"]["c1"]["speaker"] = "Sam Rivera"
        for shown in ("Alex", "Samantha"):
            self.meta(frame_speaker=shown)
            g = self.g12()
            self.assertEqual(g["status"], "WARN", shown)
            self.assertIn(f"cover frame shows {shown}, not the clip speaker Sam Rivera", g["evidence"])

    def test_names_without_latin_letters_compare_exactly(self):
        self.spec["clips"]["c1"]["speaker"] = "田中"
        self.meta(frame_speaker="田中")
        self.assertEqual(self.g12()["status"], "PASS")
        self.meta(frame_speaker="佐藤")
        self.assertEqual(self.g12()["status"], "WARN")


# ---------------------------------------------------------------- G16

class FramingBase(unittest.TestCase):
    SHOTS = {"shots": [{"start_src": 100.0, "end_src": 120.0, "kind": "two", "multi_face": True,
                        "faces": [{"id": "A", "x": 0.27, "y": 0.5, "h": 0.2, "presence": 1.0},
                                  {"id": "B", "x": 0.75, "y": 0.5, "h": 0.22, "presence": 1.0}]}]}
    CLOSE = {"shots": [{"start_src": 100.0, "end_src": 120.0, "kind": "two", "multi_face": True,
                        "faces": [{"id": "A", "x": 0.40, "y": 0.5, "h": 0.143, "presence": 1.0},
                                  {"id": "B", "x": 0.60, "y": 0.5, "h": 0.143, "presence": 1.0}]}]}
    WIDE = {"Speaker A": {"x": 0.27}, "Speaker B": {"x": 0.75}}
    NEAR = {"Speaker A": {"x": 0.40}, "Speaker B": {"x": 0.60}}

    def plan(self, sp=None, role="body"):
        words = [{"i": k, "s_out": 0.5 * k, "e_out": 0.5 * k + 0.4, "sp": sp, "piece": 0} for k in range(20)]
        return {"segments": [{"src": [100.0, 110.0], "out": [0.0, 10.0]}], "words": words,
                "pieces": [{"n": 0, "role": role}]}

    def window(self, lo, hi, mode="crop", **kw):
        return [dict({"seg": 0, "out": [0.0, 10.0], "mode": mode, "lo": lo, "hi": hi, "multi_face": True}, **kw)]

    def run16(self, plan, windows, speakers, clip=None, shots=None):
        return qa.framing_attribution(plan, clip or {"speaker": "Speaker A"}, shots or self.SHOTS, windows, speakers)


class RectTileIsChecked(FramingBase):
    """F1: a `letterbox` mark with `rect` that shows one tile of a multi-face shot."""

    def test_the_listeners_tile_fails(self):
        st, notes, _ = self.run16(self.plan(), self.window(0.5, 1.0, mode="rect"), self.WIDE)
        self.assertEqual(st, "FAIL")
        self.assertIn("framed face is not Speaker A", notes[0])

    def test_the_speakers_tile_passes(self):
        self.assertEqual(self.run16(self.plan(), self.window(0.0, 0.5, mode="rect"), self.WIDE)[0], "PASS")

    def test_reaction_on_the_wrong_tile_warns(self):
        st, _n, _s = self.run16(self.plan(), self.window(0.5, 1.0, mode="rect", reaction=True), self.WIDE)
        self.assertEqual(st, "WARN")

    def test_a_rect_holding_both_faces_needs_no_labels(self):
        self.assertEqual(self.run16(self.plan(), self.window(0.1, 0.9, mode="rect"), {})[0], "PASS")

    def test_an_unlabeled_tile_fails_like_an_unlabeled_crop(self):
        st, notes, _ = self.run16(self.plan(), self.window(0.5, 1.0, mode="rect"), {})
        self.assertEqual(st, "FAIL")
        self.assertIn('set clips.json "speakers"', notes[0])

    def test_plain_letterbox_and_full_are_never_checked(self):
        for mode in ("letterbox", "full"):
            self.assertEqual(self.run16(self.plan(), self.window(0.0, 1.0, mode=mode), {})[0], "PASS", mode)
            self.assertEqual(self.run16(self.plan(), self.window(0.5, 1.0, mode=mode), self.WIDE)[0], "PASS", mode)

    def test_through_the_framing_code(self):
        """build_tracks + windows label the mark "rect" with the tile's x range; G16 reads it."""
        from hookcutter import framing
        source = {"video": {"width": 1280, "height": 720, "sar": "1:1"}}
        shots = {"shots": [dict(self.SHOTS["shots"][0], start_src=0.0, end_src=60.0)], "ranges": [[0.0, 60.0]]}
        words = [{"i": k, "s_src": 0.5 * k, "e_src": 0.5 * k + 0.4, "s_out": 0.5 * k, "e_out": 0.5 * k + 0.4,
                  "sp": "Speaker A", "piece": 0, "seg": 0} for k in range(1, 16)]

        def g16(rect):
            plan = {"clip": "t", "kind": "cut", "fps": [25, 1], "pieces": [{"n": 0, "role": "body"}],
                    "segments": [{"k": 0, "src": [0.0, 8.0], "out": [0.0, 8.0], "frames": 200}],
                    "joins": [], "words": words, "speakers": self.WIDE,
                    "framing": [{"at_src": 0.0, "mode": "letterbox", "rect": rect}]}
            tracks = framing.build_tracks(plan, {}, shots, source, "vertical", {"motion": {"punch_in": 1.0}})
            wins = framing.windows(tracks, plan, source)
            return wins, qa.framing_attribution(plan, {}, shots, wins, self.WIDE, 1280 / 720.0)

        wins, (st, notes, _) = g16([0.5, 0.0, 0.5, 1.0])          # the right tile: the listener
        self.assertEqual((wins[0]["mode"], wins[0]["lo"], wins[0]["hi"]), ("rect", 0.5, 1.0))
        self.assertEqual(st, "FAIL", notes)
        self.assertIn("framed face is not Speaker A", notes[0])
        self.assertEqual(g16([0.0, 0.0, 0.5, 1.0])[1][0], "PASS")  # the left tile: the speaker

    def test_uncovered_rect_stretch_is_an_evidence_gap(self):
        plan = {"segments": [{"src": [10.0, 16.0], "out": [0.0, 6.0]}]}
        rect = [{"mode": "rect", "out": [0.0, 4.0]}, {"mode": "letterbox", "out": [4.0, 6.0]}]
        self.assertEqual(qa.shots_uncovered(plan, {"ranges": [[12.0, 15.0]]}, rect), (2.0, 4.0))
        crop = [{"mode": "crop", "out": [0.0, 4.0]}, {"mode": "letterbox", "out": [4.0, 6.0]}]
        self.assertEqual(qa.shots_uncovered(plan, {"ranges": [[12.0, 15.0]]}, crop), (2.0, 4.0))
        plain = [{"mode": "letterbox", "out": [0.0, 6.0]}]
        self.assertEqual(qa.shots_uncovered(plan, {"ranges": [[12.0, 15.0]]}, plain), (0.0, 0.0))


class WindowPositionNotWidth(FramingBase):
    """F2: a window wide enough to hold two close faces is judged by where it sits."""

    def test_wide_enough_window_on_the_wrong_person_fails(self):
        st, notes, _ = self.run16(self.plan(sp="Speaker B"), self.window(0.2422, 0.5578), self.NEAR, shots=self.CLOSE)
        self.assertEqual(st, "FAIL")
        self.assertIn("framed face is not Speaker B", notes[0])

    def test_same_window_with_the_framed_person_talking_passes(self):
        st, _n, _s = self.run16(self.plan(sp="Speaker A"), self.window(0.2422, 0.5578), self.NEAR, shots=self.CLOSE)
        self.assertEqual(st, "PASS")

    def test_a_fixed_crop_beside_both_faces_fails(self):
        st, notes, _ = self.run16(self.plan(sp="Speaker B"), self.window(0.10, 0.4156), self.NEAR, shots=self.CLOSE)
        self.assertEqual(st, "FAIL", notes)

    def test_both_faces_inside_needs_no_labels(self):
        self.assertEqual(self.run16(self.plan(), self.window(0.35, 0.65), {}, shots=self.CLOSE)[0], "PASS")
        self.assertEqual(self.run16(self.plan(), self.window(0.05, 0.95), {})[0], "PASS")
        # a face a hair outside the window still counts as shown (WINDOW_SLACK)
        self.assertEqual(self.run16(self.plan(), self.window(0.41, 0.65), {}, shots=self.CLOSE)[0], "PASS")

    def test_one_face_outside_without_labels_asks_for_speakers(self):
        st, notes, _ = self.run16(self.plan(), self.window(0.2422, 0.5578), {}, shots=self.CLOSE)
        self.assertEqual(st, "FAIL")
        self.assertIn('"speakers"', notes[0])

    def test_minor_faces_do_not_count(self):
        shots = {"shots": [dict(self.CLOSE["shots"][0], faces=self.CLOSE["shots"][0]["faces"] + [
            {"id": "C", "x": 0.95, "y": 0.5, "h": 0.05, "presence": 0.1, "minor": True}])]}
        self.assertEqual(self.run16(self.plan(), self.window(0.35, 0.65), {}, shots=shots)[0], "PASS")


class RowsWhenTheLogHasThem(FramingBase):
    """F3: a window that records its rows is also checked against the speaker's y."""
    STACK = {"Top": {"x": 0.27, "y": 0.25}, "Low": {"x": 0.27, "y": 0.75}}

    def test_speaker_below_the_window_fails(self):
        win = self.window(0.1, 0.4, y_lo=0.0, y_hi=0.5)
        st, notes, _ = self.run16(self.plan(), win, self.STACK, clip={"speaker": "Low"})
        self.assertEqual(st, "FAIL")
        self.assertIn("framed face is not Low", notes[0])
        self.assertEqual(self.run16(self.plan(), win, self.STACK, clip={"speaker": "Top"})[0], "PASS")

    def test_a_log_without_rows_is_judged_on_x(self):
        st, _n, _s = self.run16(self.plan(), self.window(0.1, 0.4), self.STACK, clip={"speaker": "Low"})
        self.assertEqual(st, "PASS")

    def test_a_speaker_without_y_is_judged_on_x(self):
        win = self.window(0.1, 0.4, y_lo=0.0, y_hi=0.5)
        self.assertEqual(self.run16(self.plan(), win, {"Low": {"x": 0.27}}, clip={"speaker": "Low"})[0], "PASS")

    def test_a_full_height_window_holds_every_row(self):
        win = self.window(0.1, 0.4, y_lo=0.0, y_hi=1.0)
        self.assertEqual(self.run16(self.plan(), win, self.STACK, clip={"speaker": "Low"})[0], "PASS")

    def test_a_rect_tile_above_the_speaker_fails(self):
        win = self.window(0.0, 0.5, mode="rect", y_lo=0.0, y_hi=0.5)
        self.assertEqual(self.run16(self.plan(), win, self.STACK, clip={"speaker": "Low"})[0], "FAIL")

    def test_unreadable_rows_are_ignored(self):
        win = self.window(0.1, 0.4, y_lo="top", y_hi=None)
        self.assertEqual(self.run16(self.plan(), win, self.STACK, clip={"speaker": "Low"})[0], "PASS")

    def test_a_crop_with_no_common_rows_is_judged_on_x(self):
        win = self.window(0.1, 0.4, y_lo=0.6, y_hi=0.4)      # it travelled: no row was on screen throughout
        self.assertEqual(self.run16(self.plan(), win, self.STACK, clip={"speaker": "Low"})[0], "PASS")
        self.assertEqual(self.run16(self.plan(), win, self.STACK, clip={"speaker": "Top"})[0], "PASS")


# ---------------------------------------------------------------- G11

def spoken_plan(text, hook=None):
    words = [{"i": k, "raw": w, "text": w} for k, w in enumerate(text.split())]
    plan = {"words": words, "intent": [w["i"] for w in words]}
    if hook:
        plan["hook_card"] = {"lines": list(hook)}
    return plan


class WrittenSurfaces(unittest.TestCase):
    """C6: the title printed on a cover is a written surface; drawn and checked text agree."""
    SAID = "we sold the first batch before we built anything at all"

    def clip(self, formats=("vertical",), **kw):
        return dict({"title": "We made $99M", "_eff": {"formats": list(formats)}}, **kw)

    def g11(self, clip, plan=None):
        return qa.g11_faith(ctx_for(), "c1", clip, plan or spoken_plan(self.SAID))

    def test_title_on_the_cover_is_checked(self):
        g = self.g11(self.clip())
        self.assertEqual(g["status"], "FAIL", g["evidence"])
        self.assertIn("$99M", g["evidence"])

    def test_hook_card_null_still_prints_the_title(self):
        g = self.g11(self.clip(hook_card=None))
        self.assertEqual(g["status"], "FAIL", g["evidence"])
        self.assertIn("$99M", g["evidence"])

    def test_an_honest_title_on_the_cover_passes(self):
        g = self.g11({"title": "sold the first batch", "_eff": {"formats": ["vertical"]}})
        self.assertEqual(g["status"], "PASS", g["evidence"])
        self.assertIn("cover headline (the title)", g["evidence"])

    def test_the_label_says_where_the_words_are_printed(self):
        labels = [lab for lab, _ in qa.claim_surfaces(self.clip(), None)]
        self.assertIn("cover headline (the title)", labels)

    def test_a_hook_card_or_headline_is_what_the_cover_prints(self):
        with_hook = qa.claim_surfaces(self.clip(), spoken_plan(self.SAID, hook=["SOLD IT", "*FIRST*"]))
        self.assertIn(("hook card", "SOLD IT *FIRST*"), with_hook)
        self.assertNotIn("cover headline (the title)", [lab for lab, _ in with_hook])
        with_head = qa.claim_surfaces(self.clip(cover={"headline": "SOLD *FIRST*"}), None)
        self.assertIn(("cover headline", "SOLD *FIRST*"), with_head)
        self.assertNotIn("cover headline (the title)", [lab for lab, _ in with_head])

    def test_no_cover_is_made_for_a_landscape_clip(self):
        labels = [lab for lab, _ in qa.claim_surfaces(self.clip(formats=("landscape",)), None)]
        self.assertNotIn("cover headline (the title)", labels)

    def test_whatever_the_cover_prints_is_checked(self):
        """Invariant: for a vertical clip, cover.headline_for's text is among the checked texts."""
        from hookcutter import cover
        hook_plan = spoken_plan(self.SAID, hook=["SOLD IT", "*FIRST*"])
        cases = [(self.clip(cover={"headline": "A *HEADLINE*"}), None),
                 (self.clip(), hook_plan),
                 (self.clip(hook_card={"lines": ["CLIP", "CARD"]}), None),
                 (self.clip(), None),
                 (self.clip(hook_card=None), None),
                 (self.clip(cover={"at_out": 2.0}), None),
                 (self.clip(formats=("vertical", "landscape")), None)]
        for clip, plan in cases:
            drawn = cover.headline_for(clip, plan)
            self.assertTrue(drawn)
            self.assertIn(drawn, [text for _lab, text in qa.claim_surfaces(clip, plan)], (clip, plan))

    def test_the_shared_definition_is_the_one_used(self):
        fake = mock.Mock(return_value=[("cover headline (the title)", "We made $77M")])
        with mock.patch.object(claims, "written_surfaces", fake, create=True):
            clip = self.clip()
            plan = spoken_plan(self.SAID)
            self.assertEqual(qa.claim_surfaces(clip, plan), [("cover headline (the title)", "We made $77M")])
            fake.assert_called_once_with(clip, plan)
            g = self.g11(clip, plan)
        self.assertEqual(g["status"], "FAIL")
        self.assertIn("$77M", g["evidence"])

    def test_nothing_written_is_not_applicable(self):
        with mock.patch.object(claims, "written_surfaces", mock.Mock(return_value=[]), create=True):
            g = self.g11({"_eff": {"formats": ["landscape"]}})
        self.assertEqual((g["status"], g.get("reason")), ("SKIPPED", "not applicable"))


# ---------------------------------------------------------------- G9

def page_caps(box):
    """captions json with one page box and no hook card."""
    return qm.captions_for(None, page_box=box)


class G9Base(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="hc-codexfe-g9-"))
        self.P = Project(self.tmp, create=True)
        (self.tmp / "presets").mkdir(exist_ok=True)
        env.write_json(self.tmp / "presets" / "adlook.json", {"extends": "clean", "safezone": "ads"})
        env.write_json(self.tmp / "presets" / "high.json", {"extends": "clean", "letterbox": {"y_center": 500}})
        self.ctx = ctx_for(self.P)
        self.plan = {"duration_out": 3.0}
        self.clip = {"_eff": {"safezone": "organic"}}

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def g9(self, style, cap=None, clip=None, fmt="vertical"):
        return qa.g9_layout(self.ctx, "c1", clip or self.clip, self.plan, fmt, style,
                            cap or page_caps((200, 1150, 600, 110)), self.tmp / "none.json", f"{fmt}/{style}")


class BaseFramesPerStyle(G9Base):
    """C1 / V03-N1: each style is judged on its own stage A picture."""

    def recorder(self, calls):
        def fake(project, clip_id, clip, plan, fmt, times_out, size, letterbox_y=None, style=None):
            calls.append({"style": style, "letterbox_y": letterbox_y, "times": list(times_out)})
            fill = {"clean": 10, "minimal": 60}.get(style, 0)
            return {t: np.full((size[1], size[0], 3), fill, np.uint8) for t in times_out}, "segments"
        return fake

    @staticmethod
    def faces(arr, min_score=0.6):
        return [{"box": [float(arr[0, 0, 0]), 0.0, 10.0, 10.0]}]

    def test_styles_do_not_share_faces(self):
        calls = []
        with mock.patch.object(cf, "base_frames", self.recorder(calls)), \
                mock.patch.object(cf, "detect_faces", self.faces):
            a = qa._base_faces(self.ctx, "c1", {}, self.plan, "vertical", "clean", [1.0])
            b = qa._base_faces(self.ctx, "c1", {}, self.plan, "vertical", "minimal", [1.0])
            again = qa._base_faces(self.ctx, "c1", {}, self.plan, "vertical", "clean", [1.0])
        self.assertEqual([c["style"] for c in calls], ["clean", "minimal"])      # cached per style
        self.assertEqual(a[1.0][0][0], 20.0)
        self.assertEqual(b[1.0][0][0], 120.0)
        self.assertEqual(again, a)

    def test_the_gate_passes_its_style(self):
        calls = []
        with mock.patch.object(cf, "base_frames", self.recorder(calls)), \
                mock.patch.object(cf, "detect_faces", return_value=[]):
            for st in ("clean", "minimal"):
                self.assertEqual(self.g9(st)["status"], "PASS")
        self.assertEqual([c["style"] for c in calls], ["clean", "minimal"])

    def test_letterbox_centre_comes_from_the_preset(self):
        calls = []
        with mock.patch.object(cf, "base_frames", self.recorder(calls)), \
                mock.patch.object(cf, "detect_faces", return_value=[]):
            qa._base_faces(self.ctx, "c1", {}, self.plan, "vertical", "clean", [1.0])
            qa._base_faces(self.ctx, "c1", {}, self.plan, "vertical", "high", [1.0])
            qa._base_faces(self.ctx, "c1", {}, self.plan, "landscape", "clean", [1.0])
        self.assertAlmostEqual(calls[0]["letterbox_y"], 820 / 1920)
        self.assertAlmostEqual(calls[1]["letterbox_y"], 500 / 1920)
        self.assertIsNone(calls[2]["letterbox_y"])

    def test_a_base_frames_without_the_style_argument_still_works(self):
        seen = []

        def older(project, clip_id, clip, plan, fmt, times_out, size, letterbox_y=None):
            seen.append(letterbox_y)
            return {t: np.zeros((size[1], size[0], 3), np.uint8) for t in times_out}, "source"
        with mock.patch.object(cf, "base_frames", older), mock.patch.object(cf, "detect_faces", return_value=[]):
            got = qa._base_faces(self.ctx, "c1", {}, self.plan, "vertical", "clean", [0.5])
        self.assertEqual(got, {0.5: []})
        self.assertEqual(len(seen), 1)


class SafeZoneOfTheRender(G9Base):
    """P8: G9 checks the profile the render of that style used (one resolver, presets.zone_name)."""
    BOX = (298, 1150, 600, 110)             # right edge 898: inside organic (930), outside ads (838)

    @staticmethod
    def resolver(eff, preset, fmt):
        if (eff or {}).get("safezone_set"):
            return eff["safezone"]
        return (preset or {}).get("safezone") or (eff or {}).get("safezone") or "organic"

    def test_a_presets_own_profile_is_used(self):
        fake = mock.Mock(side_effect=self.resolver)
        with mock.patch.object(presets, "zone_name", fake, create=True), \
                mock.patch.object(qa, "_base_faces", return_value={}):
            g = self.g9("adlook", page_caps(self.BOX))
            self.assertEqual(g["status"], "WARN", g["evidence"])
            self.assertIn("outside the safe area x 65-838", g["evidence"])
            eff, preset, fmt = fake.call_args[0]
            self.assertEqual((eff, preset.get("safezone"), fmt), (self.clip["_eff"], "ads", "vertical"))
            g = self.g9("clean", page_caps(self.BOX))
            self.assertEqual(g["status"], "PASS", g["evidence"])
            self.assertIn("'organic' safezone", g["evidence"])

    def test_an_explicit_defaults_safezone_wins(self):
        clip = {"_eff": {"safezone": "organic", "safezone_set": True}}
        with mock.patch.object(presets, "zone_name", mock.Mock(side_effect=self.resolver), create=True), \
                mock.patch.object(qa, "_base_faces", return_value={}):
            self.assertEqual(self.g9("adlook", page_caps(self.BOX), clip=clip)["status"], "PASS")

    def test_without_a_resolver_the_clip_profile_is_used(self):
        with mock.patch.object(presets, "zone_name", mock.Mock(side_effect=RuntimeError("no")), create=True), \
                mock.patch.object(qa, "_base_faces", return_value={}):
            g = self.g9("adlook", page_caps(self.BOX), clip={"_eff": {"safezone": "ads"}})
            self.assertEqual(g["status"], "WARN")
            self.assertEqual(qa._zone_name(self.ctx, {"_eff": {}}, "clean", "landscape"), "landscape")

    @unittest.skipUnless(hasattr(presets, "zone_name"), "presets.zone_name is another cluster's part")
    def test_with_the_real_resolver(self):
        with mock.patch.object(qa, "_base_faces", return_value={}):
            self.assertEqual(self.g9("adlook", page_caps(self.BOX), clip={"_eff": {}})["status"], "WARN")
            self.assertEqual(self.g9("clean", page_caps(self.BOX), clip={"_eff": {}})["status"], "PASS")
            wins = {"_eff": {"safezone": "organic", "safezone_set": True}}
            self.assertEqual(self.g9("adlook", page_caps(self.BOX), clip=wins)["status"], "PASS")


class FaceToolFailure(G9Base):
    """Q4: a face tool that breaks inside G9 is a tool error (exit 3), not only a gap."""

    def test_tool_failure_is_a_tool_error(self):
        with mock.patch.object(qa, "_base_faces", side_effect=env.HCError("face model missing", code=3)):
            g = self.g9("clean")
        self.assertTrue(g.get("gap"))
        self.assertIn("face check not run", g["evidence"])
        self.assertEqual(len(self.ctx.tool_errors), 1)
        self.assertIn("c1 G9", self.ctx.tool_errors[0])
        self.assertIn("face model missing", self.ctx.tool_errors[0])
        self.assertEqual(qa.exit_code([], self.ctx.tool_errors, False), 3)
        self.ctx.asr = True
        md = qa.summary_markdown([], 3, self.ctx)
        self.assertIn("## Tool errors", md)
        self.assertIn("G9", md)

    def test_other_failures_stay_a_gap(self):
        for exc in (ValueError("bad frame"), env.HCError("no source", code=2)):
            ctx = ctx_for(self.P)
            self.ctx = ctx
            with mock.patch.object(qa, "_base_faces", side_effect=exc):
                g = self.g9("clean")
            self.assertTrue(g.get("gap"))
            self.assertIn("face check not run", g["evidence"])
            self.assertEqual(ctx.tool_errors, [])


# ---------------------------------------------------------------- G3

class AudioTimeline(unittest.TestCase):
    """V15-N2: a one-sample packet wobble is not a timestamp hole."""
    A = {"time_base": "1/48000", "sample_rate": "48000", "frame_size": 1024}

    def gaps(self, durs=None, pts=None, a=None):
        if pts is None:
            pts, t = [], 0
            for d in durs:
                pts.append(t)
                t += d
        durs = durs or [1024] * len(pts)
        data = {"packets": [{"pts": p, "duration": d} for p, d in zip(pts, durs)]}
        with mock.patch.object(env, "ffprobe_json", return_value=data):
            return qa._audio_pts_gaps(Path("x.mp4"), a or self.A)

    def test_clean_grid(self):
        self.assertEqual(self.gaps([1024] * 50), (0, 0.0))

    def test_one_sample_wobble_as_an_mp4_stores_it(self):
        # packet 187 a sample short, packet 188 a sample long, then back on the 1024 grid
        self.assertEqual(self.gaps([1024] * 187 + [1023, 1025] + [1024] * 20), (0, 0.0))
        self.assertEqual(self.gaps([1024] * 5 + [1025, 1023] + [1024] * 5), (0, 0.0))

    def test_several_wobbles_do_not_add_up(self):
        durs = ([1024] * 30 + [1023, 1025]) * 6 + [1024] * 10
        self.assertEqual(self.gaps(durs), (0, 0.0))
        self.assertEqual(self.gaps([1024] * 10 + [1022, 1026] + [1024] * 10), (0, 0.0))

    def test_wobble_with_free_timestamps(self):
        pts = [k * 1024 for k in range(40)]
        pts[20] -= 1                                    # one packet a sample early, the next on the grid
        self.assertEqual(self.gaps(pts=pts), (0, 0.0))

    def test_a_timeline_that_keeps_slipping_is_reported(self):
        n, worst = self.gaps([1024] * 5 + [1025] * 12 + [1024] * 5)
        self.assertGreaterEqual(n, 1)
        self.assertGreater(worst, 2 / 48000)
        n, _ = self.gaps([1023] * 40 + [1025] * 12 + [1024] * 5)    # early packets buy no allowance
        self.assertGreaterEqual(n, 1)

    def test_a_real_hole_is_reported(self):
        n, worst = self.gaps([1024] * 10 + [1024 + 480] + [1024] * 10)
        self.assertEqual(n, 1)
        self.assertAlmostEqual(worst, 0.01)
        n, worst = self.gaps([1024] * 10 + [1027] + [1024] * 10)    # three samples: over the allowance
        self.assertEqual(n, 1)

    def test_a_short_packet_followed_directly_is_not_a_hole(self):
        self.assertEqual(self.gaps([1024] * 10 + [512] + [1024] * 10), (0, 0.0))

    def test_a_coarse_time_base_allows_nothing(self):
        a = {"time_base": "1/1000", "sample_rate": "48000", "frame_size": 1024}
        n, _ = self.gaps(pts=[0, 21, 43, 64, 85], durs=[21, 21, 21, 21, 21], a=a)
        self.assertEqual(n, 1)


class AudioOfOneRender(unittest.TestCase):
    """V05-N2: with a single style the file's audio is still compared with its render log."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="hc-codexfe-md5-"))
        cls.P = Project(cls.tmp, create=True)
        cls.render = qm.make_render(Path(cls.P.render_path("c1", "vertical", "clean")), dur=1.0, spans=((0.2, 0.8),))
        cls.md5 = render_mod.audio_md5(cls.render)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def gate(self, log):
        lp = Path(self.P.render_path("c1", "vertical", "clean", ".log.json"))
        if log is None:
            if lp.exists():
                lp.unlink()
        else:
            env.write_json(lp, log)
        return qa.g_styles_audio(ctx_for(self.P), "c1", "vertical", {"clean": self.render})

    def test_matching_log_passes(self):
        g = self.gate({"audio_md5": self.md5})
        self.assertEqual((g["id"], g["status"]), ("G3", "PASS"), g["evidence"])
        self.assertIn("what its render log recorded", g["evidence"])
        self.assertIn(self.md5[:12], g["evidence"])

    def test_audio_changed_after_the_render_fails(self):
        g = self.gate({"audio_md5": "0" * 32})
        self.assertEqual(g["status"], "FAIL")
        self.assertIn("is not what its render log recorded", g["evidence"])

    def test_no_logged_md5_is_a_gap(self):
        for log in ({"warnings": []}, None):
            g = self.gate(log)
            self.assertEqual((g["status"], g.get("gap")), ("SKIPPED", True))

    def test_no_render_no_row(self):
        self.assertIsNone(qa.g_styles_audio(ctx_for(self.P), "c1", "vertical", {}))


# ---------------------------------------------------------------- G0, EYES, report identity

class Wording(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="hc-codexfe-word-"))
        self.P = Project(self.tmp, create=True)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_g0_names_fonts_and_safe_zone(self):
        """R1: the render key now covers font files and safe-zone margins."""
        plan = {"clip": "c1", "segments": [{"frames": 25}], "frames_out": 25, "spec_hash": "a"}
        env.write_json(self.P.render_path("c1", "vertical", "clean", ".log.json"),
                       {"hashes": {"plan": render_mod.plan_hash(plan), "render": "before"}})
        with mock.patch.object(render_mod, "expected_hashes", return_value={"plan": None, "render": "now"}):
            g = qa.g0_fresh(ctx_for(self.P), "c1", "vertical", "clean", plan, "vertical/clean")
        self.assertEqual(g["status"], "FAIL")
        for word in ("clip entry", "preset", "fonts", "safe zone", "code", "hc render"):
            self.assertIn(word, g["evidence"])

    def test_seams_sheet_is_described_as_what_it_is(self):
        """Q2: the seam sheet shows the base picture, not the finished join."""
        eyes = qa._eyes(ctx_for(self.P), "c1", {"joins": [{"k": 1, "kind": "dip"}]}, ["vertical"], ["clean"])
        why = next(e["why"] for e in eyes if e["path"].endswith(".seams.jpg"))
        self.assertNotIn("exact frames", why)
        for part in ("base picture", "just before and just after every join", "before captions", "dip darkening",
                     "join kind"):
            self.assertIn(part, why)


class ReportIdentity(unittest.TestCase):
    """R4: a report made by other checks is stale."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="hc-codexfe-id-"))
        self.P = Project(self.tmp, create=True)
        self.spec = {"clips": {"c1": {}}}

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def stale(self, **kw):
        rep = dict({"clip": "c1", "status": "PASS", "gates": [], "eyes": [], "gaps": 0,
                    "checked": {"styles": ["clean"], "formats": ["vertical"]}, "plan_spec_hash": None}, **kw)
        env.write_json(self.P.qa_path("c1", ".json"), rep)
        return qa.qa_staleness(self.P, self.spec, "c1", rep)

    def test_staleness_follows_the_digest(self):
        with mock.patch.object(env, "package_digest", return_value="now", create=True):
            self.assertEqual(qa.qa_code(), "now")
            self.assertEqual(self.stale(qa_code="now"), [])
            self.assertEqual(self.stale(qa_code="before"), ["the QA checks changed since this report was made"])
            self.assertEqual(self.stale(), [qa.QA_CODE_CHANGED])       # a report from before the field

    def test_no_digest_no_verdict(self):
        with mock.patch.object(env, "package_digest", side_effect=RuntimeError("unreadable"), create=True):
            self.assertIsNone(qa.qa_code())
            self.assertEqual(self.stale(qa_code="before"), [])

    @unittest.skipUnless(hasattr(env, "package_digest"), "env.package_digest is another cluster's part")
    def test_the_real_digest_is_what_reports_carry(self):
        self.assertEqual(qa.qa_code(), env.package_digest())
        self.assertEqual(self.stale(qa_code=qa.qa_code()), [])


def _fake_build_plan(project, clip_id, **_kw):
    return env.read_json(project.plan_path(clip_id))


def _fake_expected(project, clip_id, fmt, style, **_kw):
    return {"plan": None, "render": "fresh"}


class WholeRun(unittest.TestCase):
    """qa.run over one hand-built clip in two styles: what the reports and the exit code carry."""
    HOOK = ["*BIG* IDEA"]

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="hc-codexfe-run-"))
        media = cls.tmp / "media"
        media.mkdir()
        good = qm.make_render(media / "good.mp4")
        source = qm.make_source(media / "source.mp4", dur=8.0)
        clips = {"c1": {"title": "a big idea", "pieces": [{"words": [0, 0]}], "hook_card": {"lines": cls.HOOK},
                        "styles": ["clean", "pop"]}}
        P = qm.write_project(cls.tmp / "P", clips, {"c1": "this is a big idea that matters"}, source)
        cls.project = Project(P)
        plan = qm.plan_for("c1", 0, env.read_json(P / "transcript.json"), cls.HOOK)
        env.write_json(cls.project.plan_path("c1"), plan)
        md5 = render_mod.audio_md5(good)
        for st in ("clean", "pop"):
            shutil.copy(good, cls.project.render_path("c1", "vertical", st))
            env.write_json(cls.project.render_path("c1", "vertical", st, ".captions.json"), qm.captions_for(cls.HOOK))
            env.write_json(cls.project.render_path("c1", "vertical", st, ".log.json"),
                           {"cmds": [], "warnings": [], "audio_md5": md5,
                            "hashes": {"plan": render_mod.plan_hash(plan), "render": "fresh"},
                            "framing_windows": []})

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def run_qa(self, **patches):
        out = io.StringIO()
        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.object(plan_mod, "build_plan", _fake_build_plan))
            stack.enter_context(mock.patch.object(render_mod, "expected_hashes", _fake_expected))
            stack.enter_context(mock.patch.object(env, "package_digest", return_value="checks-now", create=True))
            for target, attr, value in patches.values():
                stack.enter_context(mock.patch.object(target, attr, value))
            stack.enter_context(contextlib.redirect_stdout(out))
            stack.enter_context(contextlib.redirect_stderr(out))
            code = qa.run(self.project, ["c1"], None, asr=False)
        return code, out.getvalue(), env.read_json(self.project.qa_path("c1", ".json"))

    def test_base_frames_once_per_style_and_the_report_identity(self):
        calls = []

        def fake(project, clip_id, clip, plan, fmt, times_out, size, letterbox_y=None, style=None):
            calls.append(style)
            return {t: np.zeros((size[1], size[0], 3), np.uint8) for t in times_out}, "segments"
        code, _out, rep = self.run_qa(frames=(cf, "base_frames", fake),
                                      faces=(cf, "detect_faces", mock.Mock(return_value=[])))
        self.assertEqual(sorted(calls), ["clean", "pop"])
        self.assertEqual(rep["qa_code"], "checks-now")
        self.assertIn(code, (0, 1))
        self.assertEqual([g["status"] for g in rep["gates"] if g["id"] == "G9"], ["PASS", "PASS"])

    def test_face_tool_failure_exits_3(self):
        boom = mock.Mock(side_effect=env.HCError("face model missing", code=3))
        code, out, rep = self.run_qa(faces=(qa, "_base_faces", boom))
        self.assertEqual(code, 3)
        g9 = [g for g in rep["gates"] if g["id"] == "G9"]
        self.assertTrue(all(g.get("gap") and "face check not run" in g["evidence"] for g in g9), g9)
        self.assertIn("TOOL ERROR c1 G9", out)
        summary = Path(self.project.qa_path("summary", ".md")).read_text(encoding="utf-8")
        self.assertIn("## Tool errors", summary)
        self.assertIn("c1 G9", summary)

    def test_other_face_check_failures_do_not(self):
        code, out, rep = self.run_qa(faces=(qa, "_base_faces", mock.Mock(side_effect=ValueError("bad frame"))))
        self.assertIn(code, (0, 1))
        self.assertNotIn("TOOL ERROR", out)
        self.assertNotIn("## Tool errors", Path(self.project.qa_path("summary", ".md")).read_text(encoding="utf-8"))

    def test_a_run_that_cannot_check_still_says_which_checks(self):
        boom = mock.Mock(side_effect=env.HCError("clips.json is broken", code=1))
        code, _out, rep = self.run_qa(clip=(qa, "check_clip", boom))
        self.assertEqual((code, rep["status"], rep["qa_code"]), (1, "FAIL", "checks-now"))


if __name__ == "__main__":
    unittest.main()
