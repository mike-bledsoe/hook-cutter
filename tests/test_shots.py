"""shots.py and faces.py: cut detection (hard cut + dissolve), per-shot face summary, kinds,
proposals, the sheet and the detector's input handling."""

from __future__ import annotations

import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "scripts"))

import fixtures  # noqa: E402
from hookcutter import env, faces, framing, project, shots  # noqa: E402


def gray_seq(values, size=(36, 64), noise=0.0, seed=1):
    rng = np.random.default_rng(seed)
    out = []
    for v in values:
        f = np.full(size, float(v), dtype=np.float32)
        if noise:
            f += rng.normal(0, noise, size).astype(np.float32)
        out.append(f)
    return out


class TestCutSignal(unittest.TestCase):
    def test_hard_cut_found_between_samples(self):
        g = gray_seq([40] * 20 + [160] * 20, noise=1.0)
        times = [k * 0.25 for k in range(40)]
        cuts, thr = shots.detect_cuts(times, g)
        self.assertEqual(len(cuts), 1)
        self.assertAlmostEqual(cuts[0][0], (times[19] + times[20]) / 2)
        self.assertGreaterEqual(thr, shots.MIN_THRESHOLD)

    def test_quiet_footage_has_no_cuts(self):
        g = gray_seq([80] * 60, noise=2.0)
        cuts, _ = shots.detect_cuts([k * 0.25 for k in range(60)], g)
        self.assertEqual(cuts, [])

    def test_cuts_closer_than_min_gap_keep_the_stronger(self):
        vals = [40] * 10 + [200] * 3 + [90] * 20
        cuts, _ = shots.detect_cuts([k * 0.25 for k in range(len(vals))], gray_seq(vals))
        self.assertEqual(len(cuts), 1)
        self.assertAlmostEqual(cuts[0][0], (9 + 10) / 2 * 0.25)

    def test_slow_dissolve_registers(self):
        ramp = list(np.linspace(40, 200, 6))           # 1.5 s dissolve, ~27 per sample
        g = gray_seq([40] * 12 + ramp + [200] * 12)
        cuts, _ = shots.detect_cuts([k * 0.25 for k in range(len(g))], g)
        self.assertEqual(len(cuts), 1)
        self.assertTrue(12 * 0.25 <= cuts[0][0] <= 18 * 0.25)


class TestFaceSummary(unittest.TestCase):
    def test_cluster_presence_kind_and_proposal(self):
        smp = []
        for k in range(20):
            fl = [[0.30 + 0.01 * (k % 3), 0.40, 0.10, 0.30, 0.9]]
            if k % 2 == 0:
                fl.append([0.70, 0.42, 0.08, 0.22, 0.8])
            if k == 5:
                fl.append([0.95, 0.5, 0.02, 0.05, 0.7])     # one-off false positive
            smp.append({"t_src": k * 0.25, "faces": fl})
        fl = shots.shot_faces(smp)
        self.assertEqual([f["x"] for f in fl], [0.31, 0.7])
        self.assertEqual([f["presence"] for f in fl], [1.0, 0.5])
        self.assertEqual(shots.shot_kind(fl), "two")
        # Two people of similar size: never track the largest face (it is not the speaker, see
        # the multi-face fix); without a speakers map the proposal is letterbox with the faces.
        prop = shots.propose([{"start_src": 12.0, "end_src": 20.0, "kind": "two",
                               "faces": shots.label_faces(fl)}])
        self.assertEqual(prop, [{"at_src": 12.0, "mode": "letterbox", "multi_face": True,
                                 "faces": [{"id": "A", "x": 0.31, "h": 0.3},
                                           {"id": "B", "x": 0.7, "h": 0.22}]}])
        self.assertEqual(shots.paste_block(prop), [{"at_src": 12.0, "mode": "letterbox"}])
        single = shots.propose([{"start_src": 12.0, "end_src": 20.0, "kind": "single",
                                 "faces": fl[:1]}])
        self.assertEqual(single, [{"at_src": 12.0, "mode": "track", "x": 0.31, "crop": 1.0}])

    def test_kinds(self):
        f = lambda x, h, p=1.0: {"x": x, "y": 0.4, "h": h, "presence": p}  # noqa: E731
        self.assertEqual(shots.shot_kind([]), "none")
        self.assertEqual(shots.shot_kind([f(0.5, 0.08)]), "wide")
        self.assertEqual(shots.shot_kind([f(0.5, 0.3)]), "single")
        self.assertEqual(shots.shot_kind([f(0.2, 0.3), f(0.5, 0.3), f(0.8, 0.3)]), "group")
        self.assertEqual(shots.shot_kind([f(0.2, 0.3), f(0.5, 0.3, 0.15)]), "single")
        self.assertEqual(shots.propose([{"start_src": 1.0, "kind": "group", "faces": []}]),
                         [{"at_src": 1.0, "mode": "letterbox", "multi_face": True, "faces": []}])
        self.assertEqual(shots.propose([{"start_src": 1.0, "kind": "wide", "faces": []}]),
                         [{"at_src": 1.0, "mode": "letterbox"}])

    def test_merge_ranges(self):
        pad = shots.RANGE_PAD
        self.assertGreater(pad, framing.TRACK_MARGIN)      # room for edge moves without re-analysis
        self.assertEqual(shots.merge_ranges([(10, 20), (22, 30), (100, 110)], 200),
                         [(10.0 - pad, 30.0 + pad), (100.0 - pad, 110.0 + pad)])
        self.assertEqual(shots.merge_ranges([(0.2, 5)], 5.5), [(0.0, 5.5)])


class TestFaces(unittest.TestCase):
    def test_blank_and_odd_inputs(self):
        self.assertEqual(faces.detect(np.zeros((360, 640, 3), np.uint8)), [])
        self.assertEqual(faces.detect(np.zeros((360, 640), np.uint8)), [])
        self.assertEqual(faces.detect(np.zeros((360, 640, 4), np.uint8)), [])
        self.assertEqual(faces.detect(np.zeros((8, 8, 3), np.uint8)), [])
        self.assertEqual(faces.detect(np.zeros((2160, 3840, 3), np.uint8)), [])
        from PIL import Image
        self.assertEqual(faces.detect(Image.new("RGB", (320, 180))), [])

    def test_face_object(self):
        f = faces.Face([0.5, 0.4, 0.2, 0.3, 0.9], [(0.45, 0.35)] * 5)
        cx, cy, w, h, s = f
        self.assertEqual((cx, h, f.score), (0.5, 0.3, 0.9))
        self.assertEqual(f.box_px(100, 100), (40.0, 25.0, 20.0, 30.0))
        self.assertEqual(env.canonical_json(f), "[0.5,0.4,0.2,0.3,0.9]")
        self.assertEqual(len(f.as_dict()["landmarks"]), 5)
        self.assertIs(faces.largest([]), None)


class TestAnalyze(unittest.TestCase):
    """Real decode on a synthetic source: hard cut at 4.0 s and a 0.6 s dissolve from 7.0 s."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="hc-shots-"))
        src = cls.tmp / "cuts.mp4"
        graph = ("[2:v]trim=duration=4,setpts=PTS-STARTPTS,settb=1/25[m];"
                 "[0:v][1:v]concat=n=2:v=1:a=0,settb=1/25[ab];"
                 "[ab][m]xfade=transition=fade:duration=0.6:offset=7,format=yuv420p[v]")
        env.ff(["-f", "lavfi", "-i", "testsrc2=s=640x360:r=25:d=4",
                "-f", "lavfi", "-i", "smptebars=s=640x360:r=25:d=4",
                "-f", "lavfi", "-i", "mandelbrot=s=640x360:r=25",
                "-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo",
                "-filter_complex", graph, "-map", "[v]", "-map", "3:a", "-t", "11",
                "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", str(src)])
        cls.P = fixtures.make_project(cls.tmp, src)
        cls.res = shots.analyze(cls.P, range_src=["0", "0:11"])

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_boundaries(self):
        starts = [s["start_src"] for s in self.res["shots"]]
        self.assertEqual(len(starts), 3, self.res["cuts"])
        self.assertEqual(starts[0], 0.0)
        self.assertEqual(starts[1], 4.0)                 # refined onto the exact frame
        self.assertTrue(7.0 <= starts[2] <= 7.6)          # inside the dissolve
        self.assertEqual(self.res["shots"][-1]["end_src"], 11.0)
        cut, dis = self.res["transitions"]
        self.assertEqual((cut["kind"], cut["at_src"], cut["start_src"], cut["end_src"]),
                         ("cut", 4.0, 4.0, 4.0))
        self.assertEqual(dis["kind"], "dissolve")
        self.assertEqual(dis["at_src"], starts[2])        # the shot changes at the midpoint
        self.assertAlmostEqual(dis["start_src"], 7.0, delta=0.09)
        self.assertAlmostEqual(dis["end_src"], 7.6, delta=0.09)
        self.assertAlmostEqual(dis["at_src"], 7.3, delta=0.09)

    def test_json_schema_and_outputs(self):
        r = env.read_json(Path(self.P) / "shots" / "range-0-11.json")
        for k in ("clip", "range_src", "sample_fps", "analysis_width", "shots", "samples", "proposal",
                  "transitions", "grid", "version"):
            self.assertIn(k, r)
        self.assertEqual(r["sample_fps"], 4)
        self.assertEqual(r["grid"], "absolute")
        self.assertEqual([s["t_src"] for s in r["samples"]], [k / 4.0 for k in range(44)])
        self.assertTrue(all(s["kind"] == "none" for s in r["shots"]))
        self.assertTrue(all(p["mode"] == "letterbox" for p in r["proposal"]))
        self.assertEqual([p["at_src"] for p in r["proposal"]], [s["start_src"] for s in r["shots"]])
        self.assertTrue(Path(self.res["sheet"]).is_file())

    def test_clip_ranges_from_spec(self):
        words = [("w%d" % i, 1.0 + i, 1.5 + i) for i in range(8)]
        env.write_json(Path(self.P) / "transcript.json", fixtures.fake_transcript(words))
        env.write_json(Path(self.P) / "clips.json", {"version": 1, "clips": {
            "c1": {"pieces": [{"words": [0, 1]}, {"time_src": [9.0, 9.5]}]}}})
        rngs = shots.clip_ranges(project.Project(self.P), "c1", 11.0)
        self.assertEqual(rngs, [(0.0, 11.0)])            # pieces and plan segments, each +- RANGE_PAD


if __name__ == "__main__":
    unittest.main()
