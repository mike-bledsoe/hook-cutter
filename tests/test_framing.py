"""framing.py: crop math, the hint-anchored tracker, SubSeg splits, punch-in rules, sendcmd."""

from __future__ import annotations

import sys
import unittest
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts"))

from hookcutter import framing  # noqa: E402

SRC_169 = {"video": {"width": 1920, "height": 1080, "sar": "1:1"}}
SRC_916 = {"video": {"width": 1080, "height": 1920, "sar": "1:1"}}
SRC_720 = {"video": {"width": 1280, "height": 720, "sar": "1:1"}}
PRESET = {"letterbox": {"y_center": 820, "blur": 28, "dim": 0.35},
          "motion": {"punch_in": 1.12, "min_gap": 2.5, "max_upscale": 3.0}}


def samples(fn, t0=0.0, t1=20.0, fps=4):
    """Face samples at fps; fn(t) → list of faces [cx, cy, w, h, score]."""
    out, n = [], int(round((t1 - t0) * fps))
    for k in range(n + 1):
        t = round(t0 + k / float(fps), 4)
        out.append({"t_src": t, "faces": fn(t)})
    return out


def face(x, y=0.4, h=0.3):
    return [x, y, h * 0.56, h, 0.9]


def make_plan(spans, joins=(), framing_marks=(), fps=(25, 1), kind="cut"):
    rate = Fraction(*fps)
    segs, cur = [], Fraction(0)
    for k, (s, e) in enumerate(spans):
        n = int(round((Fraction(e).limit_denominator(1000) - Fraction(s).limit_denominator(1000)) * rate))
        segs.append({"k": k, "src": [s, e], "out": [float(cur), float(cur + n / rate)], "frames": n})
        cur += n / rate
    js = []
    for k, kind_ in joins:
        js.append({"k": k, "out": segs[k]["out"][0], "kind": kind_,
                   "before_src": segs[k - 1]["src"][1], "after_src": segs[k]["src"][0]})
    return {"clip": "t", "kind": kind, "fps": list(fps), "segments": segs, "joins": js,
            "framing": list(framing_marks)}


class TestGeometry(unittest.TestCase):
    def test_max_rect_landscape_source(self):
        self.assertEqual(framing.max_rect(1920, 1080, "vertical"), (607.5, 1080.0))
        self.assertEqual(framing.crop_size(1920, 1080, "vertical"), (606, 1080))
        self.assertEqual(framing.crop_size(1920, 1080, "landscape"), (1920, 1080))
        self.assertAlmostEqual(framing.upscale(1920, 1080, "vertical"), 1920 / 1080.0)

    def test_max_rect_vertical_source(self):
        self.assertEqual(framing.max_rect(1080, 1920, "vertical"), (1080.0, 1920.0))
        self.assertEqual(framing.crop_size(1080, 1920, "vertical"), (1080, 1920))
        self.assertEqual(framing.max_rect(1080, 1920, "landscape"), (1080.0, 607.5))
        self.assertEqual(framing.crop_size(1080, 1920, "landscape"), (1080, 606))

    def test_crop_factor_and_zoom_are_even(self):
        cw, ch = framing.crop_size(1920, 1080, "vertical", 0.62)
        self.assertEqual((cw, ch), (376, 668))
        zw, zh = framing.crop_size(1920, 1080, "vertical", 1.0, 1 / 1.12)
        self.assertEqual((zw % 2, zh % 2), (0, 0))
        self.assertLess(zh, 1080)
        self.assertAlmostEqual(framing.upscale(1920, 1080, "vertical", 0.62), 1920 / 668.0)

    def test_crop_xy_places_face_and_clamps(self):
        cw, ch = framing.crop_size(1920, 1080, "vertical", 0.5)       # 302 x 540
        x, y = framing.crop_xy(0.5, 0.5, cw, ch, 1920, 1080, "track")
        self.assertEqual(x, int(round((960 - cw / 2) / 2)) * 2)
        self.assertEqual(y, int(round((540 - 0.38 * ch) / 2)) * 2)
        self.assertEqual(framing.crop_xy(0.0, 0.0, cw, ch, 1920, 1080, "track"), (0, 0))
        x, y = framing.crop_xy(1.0, 1.0, cw, ch, 1920, 1080, "track")
        self.assertEqual((x, y), (1920 - cw, 1080 - ch))
        # fixed centres the rect on (x, y); unknown face y centres vertically
        x, y = framing.crop_xy(0.5, 0.5, cw, ch, 1920, 1080, "fixed")
        self.assertEqual(y, (1080 - ch) // 2 // 2 * 2)
        self.assertEqual(framing.crop_xy(0.5, None, cw, ch, 1920, 1080, "track")[1], (1080 - ch) // 2)

    def test_display_size_applies_sar(self):
        self.assertEqual(framing.display_size({"video": {"width": 1440, "height": 1080, "sar": "4:3"}}),
                         (1920, 1080))
        self.assertEqual(framing.display_size(SRC_169), (1920, 1080))


class TestTracker(unittest.TestCase):
    def test_anchor_ignores_neighbour(self):
        sm = samples(lambda t: [face(0.30), face(0.62)])
        trk = framing.Tracker(sm, 0.31, 0, 20)
        for t in (0.0, 5.0, 19.9):
            self.assertAlmostEqual(trk.value(t)[0], 0.30, places=3)

    def test_hold_when_face_lost_does_not_creep(self):
        # the anchored face vanishes from 4 s to 6 s while a neighbour 0.2 away stays visible
        sm = samples(lambda t: ([face(0.50)] if 4.0 <= t < 6.0 else [face(0.30), face(0.50)]))
        trk = framing.Tracker(sm, 0.30, 0, 20)
        self.assertAlmostEqual(trk.value(5.0)[0], 0.30, places=3)
        self.assertFalse(trk.near_face(5.0))
        self.assertTrue(trk.near_face(3.9))

    def test_follows_motion_inside_anchor(self):
        sm = samples(lambda t: [face(0.30 + 0.01 * min(t, 10.0))])
        trk = framing.Tracker(sm, 0.32, 0, 20)
        self.assertAlmostEqual(trk.value(18.0)[0], 0.40, places=2)
        self.assertLess(abs(trk.value(5.0)[0] - 0.35), 0.01)     # forward-backward EMA: no lag

    def test_no_faces_returns_hint(self):
        trk = framing.Tracker(samples(lambda t: []), 0.42, 0, 20)
        self.assertFalse(trk.found)
        self.assertEqual(trk.value(3.0), (0.42, None))

    def test_dead_zone_absorbs_jitter(self):
        sm = samples(lambda t: [face(0.5 + (0.002 if int(t * 4) % 2 else -0.002))])
        shots = {"shots": [], "samples": sm}
        plan = make_plan([(2.0, 6.0)], framing_marks=[{"at_src": 0.0, "mode": "track", "x": 0.5}])
        subs = framing.build_tracks(plan, {}, shots, SRC_169, "vertical", PRESET)[0]
        self.assertEqual(len(subs), 1)
        self.assertEqual(len(subs[0]["cmds"]), 1, subs[0]["cmds"][:5])

    def test_track_moves_crop_with_face(self):
        sm = samples(lambda t: [face(0.40 + 0.005 * t)])
        plan = make_plan([(0.0, 8.0)], framing_marks=[{"at_src": 0.0, "mode": "track", "x": 0.42}])
        sub = framing.build_tracks(plan, {}, {"shots": [], "samples": sm}, SRC_169, "vertical", PRESET)[0][0]
        xs = [c[1] for c in sub["cmds"]]
        self.assertGreater(len(xs), 5)
        self.assertEqual(xs, sorted(xs))                   # monotonic, no jitter back and forth
        cw = sub["size"][0]
        self.assertAlmostEqual((xs[-1] + cw / 2) / 1920.0, 0.44, delta=0.01)
        times = [c[0] for c in sub["cmds"]]
        self.assertEqual(times, sorted(times))
        self.assertEqual(times[0], 0.0)


class TestBuildTracks(unittest.TestCase):
    def test_fixed_default_centre(self):
        plan = make_plan([(10.0, 12.0)])
        sub = framing.build_tracks(plan, {}, None, SRC_169, "vertical", PRESET)[0][0]
        self.assertEqual(sub["mode"], "crop")
        self.assertEqual(sub["size"], [606, 1080])
        self.assertEqual(sub["cmds"], [[0.0, 656, 0]])
        self.assertEqual(sub["frames"], 50)

    def test_default_never_picks_the_largest_face_in_a_two_shot(self):
        # The largest face in a two-shot or gallery is not the speaker (a 59 s solo clip of the
        # smaller tile rendered the other person), so the default there is letterbox.
        shots = {"shots": [{"start_src": 0.0, "end_src": 30.0, "kind": "two",
                            "faces": [{"x": 0.25, "y": 0.4, "h": 0.2, "presence": 0.9},
                                      {"x": 0.70, "y": 0.4, "h": 0.35, "presence": 0.8}]}],
                 "samples": samples(lambda t: [face(0.25, h=0.2), face(0.70, h=0.35)])}
        sub = framing.build_tracks(make_plan([(10.0, 12.0)]), {}, shots, SRC_169, "vertical", PRESET)[0][0]
        self.assertEqual(sub["mode"], "letterbox")
        self.assertTrue(sub["multi_face"])

    def test_default_tracks_the_face_of_a_single_shot(self):
        shots = {"shots": [{"start_src": 0.0, "end_src": 30.0, "kind": "single",
                            "faces": [{"x": 0.70, "y": 0.4, "h": 0.35, "presence": 0.8}]}],
                 "samples": samples(lambda t: [face(0.70, h=0.35)])}
        sub = framing.build_tracks(make_plan([(10.0, 12.0)]), {}, shots, SRC_169, "vertical", PRESET)[0][0]
        cx = (sub["cmds"][0][1] + sub["size"][0] / 2) / 1920.0
        self.assertAlmostEqual(cx, 0.70, delta=0.01)

    def test_mark_change_splits_frame_exactly(self):
        marks = [{"at_src": 0.0, "mode": "fixed", "x": 0.3},
                 {"at_src": 11.0, "mode": "letterbox"}]
        subs = framing.build_tracks(make_plan([(10.0, 12.0)], framing_marks=marks), {}, None, SRC_169,
                                    "vertical", PRESET)[0]
        self.assertEqual([s["mode"] for s in subs], ["crop", "letterbox"])
        self.assertEqual([s["frames"] for s in subs], [25, 25])
        self.assertEqual(subs[0]["src"], [10.0, 11.0])
        self.assertEqual(subs[1]["src"], [11.0, 12.0])
        self.assertEqual(subs[1]["y_center"], 820)

    def test_mark_off_grid_snaps_to_nearest_frame(self):
        for at, first in ((11.013, 25), (11.03, 26), (10.99, 25)):
            marks = [{"at_src": 0.0, "mode": "fixed", "x": 0.3}, {"at_src": at, "mode": "letterbox"}]
            subs = framing.build_tracks(make_plan([(10.0, 12.0)], framing_marks=marks), {}, None,
                                        SRC_169, "vertical", PRESET)[0]
            self.assertEqual(subs[0]["frames"], first, at)
            self.assertEqual(sum(s["frames"] for s in subs), 50)

    def test_ntsc_frames_sum(self):
        plan = make_plan([(1.001, 3.003), (5.005, 6.006)], fps=(30000, 1001),
                         framing_marks=[{"at_src": 0.0, "mode": "fixed", "x": 0.5},
                                        {"at_src": 2.0, "mode": "fixed", "x": 0.7}])
        tracks = framing.build_tracks(plan, {}, None, SRC_169, "vertical", PRESET)
        for subs, seg in zip(tracks, plan["segments"]):
            self.assertEqual(sum(s["frames"] for s in subs), seg["frames"])
        self.assertEqual(tracks[0][1]["src"][0], round(float(Fraction(60, 1) * Fraction(1001, 30000)), 6))

    def test_landscape_is_full_and_caption_only_is_centred_cover(self):
        plan = make_plan([(0.0, 2.0), (3.0, 4.0)],
                         framing_marks=[{"at_src": 0.0, "mode": "track", "x": 0.2, "crop": 0.6}])
        tracks = framing.build_tracks(plan, {}, None, SRC_169, "landscape", PRESET)
        self.assertTrue(all(s["mode"] == "full" for subs in tracks for s in subs))
        co = make_plan([(0.0, 5.0)], kind="caption_only")
        sub = framing.build_tracks(co, {}, None, SRC_916, "vertical", PRESET)[0][0]
        self.assertEqual((sub["mode"], sub["size"], sub["cmds"]), ("crop", [1080, 1920], [[0.0, 0, 0]]))
        sub = framing.build_tracks(co, {}, None, SRC_169, "vertical", PRESET)[0][0]
        self.assertEqual(sub["cmds"][0][1], (1920 - 606) // 4 * 2)     # centred, even


class TestPunchIn(unittest.TestCase):
    SPANS = [(0.0, 3.0), (5.0, 8.0), (10.0, 13.0), (15.0, 18.0)]

    def zooms(self, joins, marks=None, src=SRC_169, preset=PRESET, clip=None, shots=None):
        marks = marks or [{"at_src": 0.0, "mode": "fixed", "x": 0.5, "crop": 1.0}]
        plan = make_plan(self.SPANS, joins=joins, framing_marks=marks)
        tracks = framing.build_tracks(plan, clip or {}, shots, src, "vertical", preset)
        return [subs[0]["zoom"] for subs in tracks], tracks

    def test_alternates_at_same_shot_joins(self):
        z, tracks = self.zooms([(1, "tighten"), (2, "omit"), (3, "piece")])
        self.assertEqual(z, [1.0, round(1 / 1.12, 6), 1.0, round(1 / 1.12, 6)])
        self.assertLess(tracks[1][0]["size"][1], tracks[0][0]["size"][1])

    def test_min_gap(self):
        preset = dict(PRESET, motion={"punch_in": 1.12, "min_gap": 4.0, "max_upscale": 3.0})
        z, _ = self.zooms([(1, "tighten"), (2, "tighten"), (3, "tighten")], preset=preset)
        # joins at out 3, 6, 9: the first is < 4 s in, the second toggles, the third is 3 s later
        self.assertEqual(z, [1.0, 1.0, round(1 / 1.12, 6), round(1 / 1.12, 6)])

    def test_not_across_shot_boundary(self):
        shots = {"shots": [{"start_src": 0.0, "end_src": 4.0, "faces": []},
                           {"start_src": 4.0, "end_src": 30.0, "faces": []}], "samples": []}
        z, _ = self.zooms([(1, "tighten"), (2, "tighten")], shots=shots)
        self.assertEqual(z[:3], [1.0, 1.0, round(1 / 1.12, 6)])

    def test_upscale_budget(self):
        marks = [{"at_src": 0.0, "mode": "fixed", "x": 0.5, "crop": 0.62}]
        z, _ = self.zooms([(1, "tighten"), (2, "tighten")], marks=marks)
        self.assertEqual(z, [1.0, 1.0, 1.0, 1.0])

    def test_never_in_letterbox_dip_or_when_off(self):
        z, tracks = self.zooms([(1, "tighten")], marks=[{"at_src": 0.0, "mode": "letterbox"}])
        self.assertTrue(all(s["mode"] == "letterbox" and s["zoom"] == 1.0 for t in tracks for s in t))
        z, _ = self.zooms([(1, "dip"), (2, "cut")])
        self.assertEqual(z, [1.0] * 4)
        z, _ = self.zooms([(1, "tighten")], clip={"motion": {"punch_in": 1.0}})
        self.assertEqual(z, [1.0] * 4)

    def test_no_punch_when_a_mark_follows_within_a_second(self):
        marks = [{"at_src": 0.0, "mode": "fixed", "x": 0.5}, {"at_src": 5.5, "mode": "fixed", "x": 0.6}]
        z, tracks = self.zooms([(1, "tighten")], marks=marks)
        self.assertEqual([s["zoom"] for s in tracks[1]], [1.0, 1.0])
        self.assertEqual([s["frames"] for s in tracks[1]], [12, 63])

    def test_new_mark_resets_zoom(self):
        marks = [{"at_src": 0.0, "mode": "fixed", "x": 0.5}, {"at_src": 10.0, "mode": "fixed", "x": 0.6}]
        z, _ = self.zooms([(1, "tighten"), (2, "tighten"), (3, "tighten")], marks=marks)
        self.assertEqual(z, [1.0, round(1 / 1.12, 6), 1.0, round(1 / 1.12, 6)])


class TestSendcmd(unittest.TestCase):
    def test_text(self):
        sub = {"cmds": [[0.0, 10, 20], [0.04, 12, 20], [Fraction(1001, 30000) * 3, 14, 22]]}
        txt = framing.sendcmd_text(sub, "crop@c1", (25, 1))
        lines = txt.strip().splitlines()
        self.assertEqual(lines[0], "0.020000 crop@c1 x 12, crop@c1 y 20;")
        self.assertEqual(len(lines), 2)
        self.assertEqual(framing.sendcmd_text({"cmds": [[0.0, 1, 2]]}, "crop@c0", (25, 1)), "")
        txt = framing.sendcmd_text({"cmds": [[0, 0, 0], [round(1001 / 30000.0, 6), 2, 0]]}, "crop@c0",
                                   (30000, 1001))
        self.assertTrue(txt.startswith("%.6f " % (0.5 * 1001 / 30000.0)))


if __name__ == "__main__":
    unittest.main()
