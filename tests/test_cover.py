"""Cover composition, scoring and end-to-end make_cover (contract section 11)."""

from __future__ import annotations

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
from PIL import Image  # noqa: E402

import qa_media as qm  # noqa: E402
from hookcutter import cover, cover_frames, env, qa  # noqa: E402
from hookcutter import plan as plan_mod  # noqa: E402
from hookcutter.project import Project  # noqa: E402

CFG = dict(cover.DEFAULT_COVER)
ACCENT = (255, 216, 77)


def _photo(color=(40, 44, 52)):
    return Image.new("RGB", (1080, 1920), color)


def _near(arr, rgb, tol=40):
    d = np.abs(arr.astype(int) - np.array(rgb)[None, None, :]).sum(axis=2)
    return int((d <= tol).sum())


class Composition(unittest.TestCase):
    HEADLINES = ["NO", "AFRAID OF LOOKING *STUPID*", "THE *ONE* THING EVERY CREATOR GETS WRONG ABOUT PRICING THEIR WORK",
                 "SUPERCALIFRAGILISTICEXPIALIDOCIOUS IS LONG", "*HONEST* WORK WINS"]

    def test_text_stays_inside_the_band(self):
        for h in self.HEADLINES:
            with self.subTest(headline=h):
                _img, info = cover.compose(_photo(), [], h, CFG)
                x, y, w, hh = info["headline_box"]
                self.assertGreaterEqual(y, 240)
                self.assertLessEqual(y + hh, 1680)
                self.assertGreaterEqual(x, 0)
                self.assertLessEqual(x + w, 1080)
                self.assertLessEqual(len(info["lines"]), CFG["max_lines"])
                self.assertGreaterEqual(info["size"], 24)

    def test_block_bottom_anchors_above_lockup(self):
        cfg = dict(CFG, rule=True, lockup=["A SHOW NAME"])
        _img, info = cover.compose(_photo(), [], "BIG *IDEA*", cfg)
        hb, tb = info["headline_box"], info["text_box"]
        self.assertLess(hb[1] + hb[3], tb[1] + tb[3])
        self.assertLessEqual(tb[1] + tb[3], 1680 - cfg["margin"] + 2)

    def test_accent_runs_are_coloured(self):
        img, info = cover.compose(_photo(), [], "AFRAID OF LOOKING *STUPID*", CFG)
        arr = np.asarray(img)
        x, y, w, h = (int(v) for v in info["headline_box"])
        box = arr[y:y + h, x:x + w]
        self.assertGreater(_near(box, ACCENT), 500)
        self.assertGreater(_near(box, (255, 255, 255)), 500)
        img2, info2 = cover.compose(_photo(), [], "AFRAID OF LOOKING STUPID", CFG)
        x, y, w, h = (int(v) for v in info2["headline_box"])
        self.assertEqual(_near(np.asarray(img2)[y:y + h, x:x + w], ACCENT, tol=20), 0)
        self.assertIn("*STUPID*", " ".join(info["lines"]))

    def test_face_collision_raises_shift(self):
        _img, info = cover.compose(_photo(), [], "AFRAID OF LOOKING *STUPID*", CFG)
        tb = info["text_box"]
        face = {"box": [tb[0] + 50, tb[1] - 40, 300, 340], "score": 0.9, "landmarks": [[0, 0]] * 5}
        _img, info2 = cover.compose(_photo(), [face], "AFRAID OF LOOKING *STUPID*", CFG)
        self.assertGreater(info2["shift"], 0)
        self.assertEqual(info2["shift"] % cover.SHIFT_STEP, 0)
        fb = info2["face_box"]
        self.assertEqual(cover_frames.box_overlap(fb, info2["text_box"]), 0.0)
        self.assertFalse(any("covers a face" in w for w in info2["warnings"]))
        high = {"box": [400, 300, 280, 320], "score": 0.9, "landmarks": [[0, 0]] * 5}
        self.assertEqual(cover.compose(_photo(), [high], "AFRAID OF LOOKING *STUPID*", CFG)[1]["shift"], 0)

    def test_unresolvable_collision_warns(self):
        face = {"box": [0, 300, 1080, 1600], "score": 0.9, "landmarks": [[0, 0]] * 5}
        _img, info = cover.compose(_photo(), [face], "AFRAID OF LOOKING *STUPID*", CFG)
        self.assertTrue(any("covers a face" in w for w in info["warnings"]))
        self.assertLessEqual(info["shift"], cover.SHIFT_MAX)

    def test_shifted_photo_fades_into_tint(self):
        top = Image.new("RGB", (1080, 1920), (250, 250, 250))
        img = cover._photo_layer(top, 400, CFG)
        arr = np.asarray(img)
        self.assertGreater(arr[100, 540].mean(), 200)
        self.assertLess(arr[1900, 540].mean(), 25)
        self.assertLess(arr[1920 - 400 - 4, 540].mean(), 25)        # photo's own bottom edge faded out
        self.assertGreater(arr[1920 - 400 - 300, 540].mean(), 40)    # above the fade the photo shows

    def test_missing_glyph_fails_loud(self):
        with self.assertRaises(env.HCError):
            cover.compose(_photo(), [], "HELLO 世界", CFG)

    def test_balanced_wrap_avoids_widows(self):
        wrap = cover.balanced_wrap([100, 100, 100, 100, 100], 20, 360, 3)
        self.assertEqual([len(x) for x in wrap], [3, 2])
        self.assertIsNone(cover.balanced_wrap([500], 20, 400, 2))

    def test_grid_and_mobile_geometry(self):
        img, _ = cover.compose(_photo(), [], "BIG IDEA", CFG)
        g = cover.grid_crop(img)
        self.assertEqual(g.size, (1080, 1440))
        self.assertEqual(cover.mobile_preview(g).size, (240, 320))


class Scoring(unittest.TestCase):
    def _img(self):
        rng = np.random.default_rng(1)
        return (rng.random((640, 360, 3)) * 255).astype(np.uint8)

    def test_frontal_beats_profile_and_bigger_beats_smaller(self):
        img = self._img()
        frontal = {"box": [110, 150, 140, 170], "score": 0.9,
                   "landmarks": [[150, 210], [210, 210], [180, 240], [155, 280], [205, 280]]}
        profile = {"box": [110, 150, 140, 170], "score": 0.9,
                   "landmarks": [[150, 210], [170, 210], [200, 240], [160, 280], [175, 280]]}
        small = {"box": [150, 200, 60, 70], "score": 0.9,
                 "landmarks": [[167, 225], [193, 225], [180, 238], [169, 255], [191, 255]]}
        s_f = cover.score_frame(img, [frontal])[0]
        self.assertGreater(s_f, cover.score_frame(img, [profile])[0])
        self.assertGreater(s_f, cover.score_frame(img, [small])[0])
        self.assertGreater(s_f, cover.score_frame(img, [])[0])

    def test_candidate_times_on_grid(self):
        ts = cover.candidate_times(10.0, (30000, 1001))
        self.assertEqual(ts[0], 0.0)
        self.assertTrue(all(b > a for a, b in zip(ts, ts[1:])))
        self.assertLessEqual(len(cover.candidate_times(3600.0, (25, 1))), cover.MAX_CANDIDATES + 1)


class MakeCover(unittest.TestCase):
    """End to end on a synthetic project: source fallback, then real stage A segments."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="hc-b6-cover-"))
        source = qm.make_source(cls.tmp / "source.mp4", dur=12.0)
        sentences = {"c01": "afraid of looking stupid on camera", "c02": "second clip words here"}
        clips = {cid: {"title": cid, "pieces": [{"words": [0, 0]}], "hook_card": {"lines": ["AFRAID OF *STUPID*"]},
                       "cover": {"headline": "AFRAID OF LOOKING *STUPID*"} if cid == "c01" else None}
                 for cid in sentences}
        clips["c02"].pop("cover")
        P = qm.write_project(cls.tmp / "P", clips, sentences, source)
        cls.project = Project(P)
        tr = env.read_json(P / "transcript.json")
        cls.plans = {}
        for k, cid in enumerate(sentences):
            cls.plans[cid] = qm.plan_for(cid, k, tr, ["AFRAID OF *STUPID*"])
            env.write_json(cls.project.plan_path(cid), cls.plans[cid])

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _load(self, project, clip_id, require_ok=False):
        return self.plans[clip_id]

    def test_make_cover_outputs(self):
        with mock.patch.object(plan_mod, "load_plan", self._load):
            res = cover.make_cover(self.project, "c01", "clean")
            res2 = cover.make_cover(self.project, "c02", "clean", at_out=1.0)
            grid = cover.covers_grid(self.project, "clean", ["c01", "c02"])
        P = self.project
        with Image.open(P.cover_path("c01", "clean", "main")) as im:
            self.assertEqual(im.size, (1080, 1920))
        with Image.open(P.cover_path("c01", "clean", "grid")) as im:
            self.assertEqual(im.size, (1080, 1440))
        with Image.open(P.cover_path("c01", "clean", "mobile")) as im:
            self.assertEqual(im.size, (240, 320))
        meta = env.read_json(P.cover_path("c01", "clean", "json"))
        for key in ("at_out", "score", "face_box", "headline_box", "size", "lines", "shift", "warnings",
                    "crown_y", "crown_method"):
            self.assertIn(key, meta)
        self.assertEqual(meta["frame_source"], "source")
        self.assertTrue(any("approximated" in w for w in meta["warnings"]))
        self.assertEqual(res["headline"], "AFRAID OF LOOKING *STUPID*")
        self.assertEqual(res2["headline"], "AFRAID OF *STUPID*")
        self.assertAlmostEqual(res2["at_out"], 1.0)
        self.assertTrue(res2["manual"])
        self.assertTrue(Path(P.sheet_path("c01.cover-candidates.jpg")).is_file())
        with Image.open(grid) as im:
            self.assertEqual(im.size[0], 3 * cover.GRID_TILE_W + 2 * 3)
        self.assertEqual(qa.check_cover_file(Path(P.cover_path("c01", "clean", "main"))), [])

    def test_frames_from_stage_a_segments(self):
        plan = self.plans["c01"]
        segs = []
        for s in plan["segments"]:
            p = self.project.work_dir("c01", "vertical") / f"seg_test{s['k']}.mkv"
            env.ff(["-f", "lavfi", "-i", f"color=c=0x2266AA:s=1080x1920:r=25:d={s['frames'] / 25}",
                    "-frames:v", str(s["frames"]), "-c:v", "libx264", "-preset", "ultrafast", "-qp", "0", str(p)])
            segs.append({"k": s["k"], "path": self.project.rel(p), "frames": s["frames"]})
        env.write_json(self.project.render_path("c01", "vertical", "clean", ".log.json"), {"segments": segs})
        try:
            files = cover_frames.segment_files(self.project, "c01", "vertical", plan)
            self.assertEqual(len(files), 2)
            frames, origin = cover_frames.base_frames(self.project, "c01", {}, plan, "vertical", [0.5, 2.0], (360, 640))
            self.assertEqual(origin, "segments")
            self.assertEqual(frames[2.0].shape, (640, 360, 3))
            self.assertGreater(frames[2.0][..., 2].mean(), frames[2.0][..., 0].mean())
        finally:
            Path(self.project.render_path("c01", "vertical", "clean", ".log.json")).unlink()

    def test_truncated_cover_is_detected(self):
        p = self.tmp / "trunc.jpg"
        cover.save_jpeg(_photo(), p)
        self.assertEqual(qa.check_cover_file(p), [])
        data = p.read_bytes()
        p.write_bytes(data[: int(len(data) * 0.6)])
        probs = qa.check_cover_file(p)
        self.assertTrue(probs)
        self.assertTrue(any("truncated" in x or "decode" in x for x in probs))


if __name__ == "__main__":
    unittest.main()
