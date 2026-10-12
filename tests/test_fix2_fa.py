"""Regression tests for the second round of cuts fixes (found on a first real 4K run):

  1  render gets its own face data, the render key follows it, an up-to-date render is untouched
  2  shots analysis samples on an absolute grid and framing reads fixed windows, so tracks do
     not depend on the analysed range
  3  in_offset / out_offset reach 60 frames (INFO beyond 12); a time_src opener and L04
  4  a final consonant that runs into the next word is not faded away
  5  clip key "insert": speech the transcript missed becomes plan words
  6  clip key "caption_backing"
  7  two small faces of similar size are a multi-face shot
  8  source cross dissolves: span, midpoint, and framing marks that switch there
  9  hold_after: the plan says which grid frame it landed on; out_offset applies after it
  10 (with 1) "up to date" touches nothing
  11 L15 messages for the false_start and timing_uncertain transcript flags

Every fixture is synthetic: drawn frames, tone-free audio, made-up words.
"""
from __future__ import annotations

import io
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "scripts"))

import numpy as np  # noqa: E402

import fixtures  # noqa: E402
from fixtures import fake_transcript  # noqa: E402
from hookcutter import env, faces, framing, project, render, shots  # noqa: E402
from hookcutter import plan as P  # noqa: E402
from hookcutter import spec as S  # noqa: E402
from hookcutter.audio import EnergyTrack  # noqa: E402

FLOOR = -60.0
VOICE = -20.0
PRESET = {"letterbox": {"y_center": 820, "blur": 28, "dim": 0.35},
          "motion": {"punch_in": 1.0, "min_gap": 3.5, "max_upscale": 3.0}}
SRC_169 = {"video": {"width": 1920, "height": 1080, "sar": "1:1"}}


# ---- helpers --------------------------------------------------------------------------------

def energy(dur, spans, floor=FLOOR, level=VOICE, hop=0.01):
    n = int(round(dur / hop))
    db = np.full(n, floor, dtype=np.float64)
    for sp in spans:
        lv = sp[2] if len(sp) > 2 else level
        db[int(round(sp[0] / hop)):int(round(sp[1] / hop))] = lv
    return EnergyTrack(db, hop=hop, t0=0.0)


def source(fps=(25, 1), dur=60.0, w=1920, h=1080):
    v = {"fps": list(fps), "start": 0.0, "width": w, "height": h,
         "frames": int(math.floor(dur * fps[0] / fps[1] + 1e-9)), "hdr": "none", "sar": "1:1"}
    return {"fingerprint": "fp-test", "duration": dur, "video": v,
            "audio": {"start": 0.0, "index": 1}}


def build(words, clips, spans=None, extra=(), fps=(25, 1), dur=None, flags=None, cid="c01",
          E=None):
    tr = fake_transcript(words)
    if flags:
        tr["flags"] = flags
    end = dur or (max(w[2] for w in words) + 5.0)
    if E is None:
        E = energy(end, list(spans if spans is not None else [(w[1], w[2]) for w in words])
                   + list(extra))
    spec = S.normalize_spec({"version": 1, "clips": clips})
    return P.plan_clip(spec, cid, tr, source(fps, end), E, None)


def grid_words(n, start=0.6, wl=0.3, gap=0.1, text=None):
    out, t = [], start
    for k in range(n):
        out.append((text[k] if text else "w%d" % k, round(t, 3), round(t + wl, 3)))
        t += wl + gap
    return out


def lints(plan, lid, level=None):
    return [x for x in plan["lints"] if x["id"] == lid and (level is None or x["level"] == level)]


def issues(clip):
    sp = S.normalize_spec({"version": 1, "clips": {"c01": clip}})
    return sp, [(x["id"], x["level"]) for x in sp["clips"]["c01"]["_issues"]]


def face(x, y=0.4, h=0.3):
    return [x, y, h * 0.56, h, 0.9]


def make_plan(spans, framing_marks=(), fps=(25, 1), joins=()):
    rate = Fraction(*fps)
    segs, cur = [], Fraction(0)
    for k, (s, e) in enumerate(spans):
        n = int(round((Fraction(e).limit_denominator(1000) - Fraction(s).limit_denominator(1000)) * rate))
        segs.append({"k": k, "src": [s, e], "out": [float(cur), float(cur + n / rate)], "frames": n})
        cur += n / rate
    js = [{"k": k, "out": segs[k]["out"][0], "kind": kind, "before_src": segs[k - 1]["src"][1],
           "after_src": segs[k]["src"][0]} for k, kind in joins]
    return {"clip": "t", "kind": "cut", "fps": list(fps), "segments": segs, "joins": js,
            "framing": list(framing_marks)}


def crop_xs(sub, fps=25):
    """Crop x of every frame of a crop SubSeg."""
    at = {int(round(c[0] * fps)): c[1] for c in sub["cmds"]}
    out, cur = [], None
    for n in range(sub["frames"]):
        cur = at.get(n, cur)
        out.append(cur)
    return out


def encode_gray(path, frames, fps=25, audio=False):
    """Encode uint8 gray frames (h, w) losslessly, optionally with a silent stereo track."""
    h, w = frames[0].shape
    cmd = [env.ffmpeg(), "-hide_banner", "-nostdin", "-loglevel", "error", "-y", "-f", "rawvideo",
           "-pix_fmt", "gray", "-s", "%dx%d" % (w, h), "-r", str(fps), "-i", "-"]
    if audio:
        cmd += ["-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo", "-shortest", "-c:a", "aac"]
    cmd += ["-c:v", "libx264", "-qp", "0", "-g", "25", "-pix_fmt", "yuv420p", str(path)]
    subprocess.run(cmd, input=b"".join(np.ascontiguousarray(f).tobytes() for f in frames),
                   check=True)
    return path


def disc_frames(n, w=640, h=360, x0=0.38, x1=0.56, r=42, fps=25):
    """A bright disc (the stand-in face) that drifts from x0 to x1 over a dark frame."""
    yy, xx = np.mgrid[0:h, 0:w]
    out = []
    for k in range(n):
        cx = (x0 + (x1 - x0) * k / float(max(1, n - 1))) * w
        f = np.full((h, w), 28, np.uint8)
        f[(xx - cx) ** 2 + (yy - 0.45 * h) ** 2 <= r * r] = 225
        out.append(f)
    return out


def disc_detect(image, *_a, **_k):
    """Stand-in for the face detector: the bright disc, as one face box."""
    arr = np.asarray(image)
    g = arr[..., 0] if arr.ndim == 3 else arr
    ys, xs = np.nonzero(g > 128)
    if len(xs) < 40:
        return []
    h, w = g.shape[:2]
    cx, cy = float(xs.mean()) / w, float(ys.mean()) / h
    bw, bh = (int(xs.max()) - int(xs.min()) + 1) / float(w), (int(ys.max()) - int(ys.min()) + 1) / float(h)
    return [faces.Face([cx, cy, bw, bh, 0.99], [(cx, cy)] * 5)]


def video_md5(path) -> str:
    res = env.ff(["-i", str(path), "-map", "0:v:0", "-f", "md5", "-"])
    return res.text.strip()


# ---- 1, 10: render and its face data ---------------------------------------------------------

class RenderGetsItsFaceData(unittest.TestCase):
    """A clip framed by a track mark, in a project where `shots` never ran."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="hc-fix2-fa-render-"))
        src = encode_gray(cls.tmp / "disc.mp4", disc_frames(200), audio=True)
        words = [("one", 1.2, 1.5), ("two", 1.7, 2.0), ("three", 2.4, 2.8), ("four", 3.4, 3.8)]
        cls.P = fixtures.make_project(cls.tmp, src, words)
        env.write_json(Path(cls.P) / "clips.json", {"version": 1, "clips": {"c01": {
            "pieces": [{"time_src": [1.0, 5.0]}], "captions": False, "styles": ["clean"],
            "framing": [{"at_src": 0.0, "mode": "track", "x": 0.45}]}}})
        cls.proj = project.Project(cls.P)
        cls.mp4 = cls.proj.render_path("c01", "vertical", "clean")
        cls.log_path = cls.proj.render_path("c01", "vertical", "clean", ".log.json")
        cls.shots_path = Path(cls.P) / "shots" / "c01.json"
        cls.existed_before = cls.shots_path.exists()
        cls.err = io.StringIO()
        with mock.patch.object(faces, "detect", disc_detect), mock.patch("sys.stderr", cls.err):
            cls.first = render.render_clip(cls.proj, "c01")
        cls.log1 = env.read_json(cls.log_path)
        cls.md5_1 = video_md5(cls.mp4)
        cls.shots_1 = cls.shots_path.read_bytes() if cls.shots_path.exists() else b""

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_render_analyses_shots_and_says_so(self):
        self.assertFalse(self.existed_before)
        self.assertTrue(self.shots_path.is_file(), "render left the track mark without face data")
        self.assertIn("[render] analysed shots for c01 (framing needs face data)", self.err.getvalue())
        data = json.loads(self.shots_1)
        self.assertEqual(data["grid"], "absolute")
        self.assertTrue(data["samples"] and all(s["faces"] for s in data["samples"]))
        self.assertTrue(self.log1["shots"]["analysed_by_render"])
        # the plan on disk was rebuilt with the new shots data, so nothing is stale afterwards
        plan = env.read_json(self.proj.plan_path("c01"))
        self.assertFalse(P.shots_stale(self.proj, "c01", plan))
        self.assertEqual(self.log1["hashes"]["plan"], render.plan_hash(plan))

    def test_the_crop_follows_the_face(self):
        sub = self.log1["framing"][0][0]          # [mode, frames, size, zoom, n_cmds]
        self.assertEqual(sub[0], "crop")
        self.assertGreater(sub[4], 3, "a static crop: the face moves but the window does not")
        win = self.log1["framing_windows"][0]
        self.assertGreater(win["hi_any"] - win["lo_any"], win["hi"] - win["lo"] + 0.02)

    def test_same_pixels_after_an_explicit_shots_run(self):
        with mock.patch.object(faces, "detect", disc_detect), mock.patch("sys.stderr", io.StringIO()) as err:
            shots.analyze(self.proj, clip_id="c01")                      # what `hc shots P c01` does
            render.render_clip(self.proj, "c01", force=True)
        self.assertNotIn("analysed shots", err.getvalue())
        self.assertEqual(video_md5(self.mp4), self.md5_1)
        self.assertEqual(env.read_json(self.log_path)["framing"], self.log1["framing"])

    def test_up_to_date_touches_nothing(self):
        with mock.patch.object(faces, "detect", disc_detect), mock.patch("sys.stderr", io.StringIO()):
            render.render_clip(self.proj, "c01")                         # settle on the current data
            old = 1_600_000_000
            watched = [self.mp4, self.log_path, self.shots_path, self.proj.plan_path("c01")]
            for f in watched:
                os.utime(str(f), (old, old))
            err = io.StringIO()
            with mock.patch("sys.stderr", err):
                render.render_clip(self.proj, "c01")
        self.assertIn("up to date", err.getvalue())
        for f in watched:
            self.assertEqual(int(Path(f).stat().st_mtime), old, "%s was touched" % Path(f).name)

    def test_render_key_follows_the_face_samples(self):
        with mock.patch.object(faces, "detect", disc_detect), mock.patch("sys.stderr", io.StringIO()):
            render.render_clip(self.proj, "c01")
        plan = env.read_json(self.proj.plan_path("c01"))
        log = env.read_json(self.log_path)
        clip = self.proj.spec()["clips"]["c01"]
        before = render.expected_hashes(self.proj, "c01", "vertical", "clean", plan=plan, clip=clip)
        self.assertEqual(before["render"], log["hashes"]["render"])
        keep = self.shots_path.read_bytes()
        try:
            data = json.loads(keep)
            # a face sample inside the clip changes by less than the dead zone: same crop moves,
            # other face data
            k = next(i for i, s in enumerate(data["samples"]) if 2.0 <= s["t_src"] <= 3.0)
            data["samples"][k]["faces"][0][4] = 0.5
            env.write_json(self.shots_path, data)
            after = render.expected_hashes(self.proj, "c01", "vertical", "clean", plan=plan, clip=clip)
            self.assertNotEqual(after["render"], before["render"])
            self.assertTrue(P.shots_stale(self.proj, "c01", plan))
            # a sample far outside every tracker window is not face data the render reads
            data = json.loads(keep)
            data["samples"].append({"t_src": 190.0, "faces": [face(0.5)]})
            env.write_json(self.shots_path, data)
            far = render.expected_hashes(self.proj, "c01", "vertical", "clean", plan=plan, clip=clip)
            self.assertEqual(far["render"], before["render"])
        finally:
            self.shots_path.write_bytes(keep)

    def test_a_failed_analysis_stops_the_render_with_the_remedy(self):
        marks = [{"at_src": 0.0, "mode": "track", "x": 0.45}]
        plan = dict(make_plan([(1.0, 2.0)], marks), frames_out=25)
        clip = {"kind": "cut", "captions": False, "framing": marks}
        boom = env.HCError("the face model is missing", code=3)
        with mock.patch.object(shots, "analyze", side_effect=boom), \
                mock.patch("sys.stderr", io.StringIO()):
            with self.assertRaises(env.HCError) as ctx:
                render.render_clip(self.proj, "c09", formats=["vertical"], styles=["clean"],
                                   plan=plan, clip=clip)
        self.assertEqual(ctx.exception.code, 3)
        self.assertIn("c09 is framed by faces", ctx.exception.msg)
        self.assertIn("the face model is missing", ctx.exception.msg)
        self.assertIn("fixed or letterbox marks", ctx.exception.msg)
        self.assertFalse(self.proj.render_path("c09", "vertical", "clean").exists())
        # fixed marks need no face data: the same call would not analyse at all
        fixed = [{"at_src": 0.0, "mode": "fixed", "x": 0.5, "y": 0.5}]
        self.assertFalse(render.needs_shots(self.proj, "c09", dict(clip, framing=fixed),
                                            dict(plan, framing=fixed), ["vertical", "landscape"]))
        self.assertTrue(render.needs_shots(self.proj, "c09", clip, plan, ["vertical"]))
        self.assertFalse(render.needs_shots(self.proj, "c09", clip, plan, ["landscape"]))

    def test_which_framing_needs_face_data(self):
        plan = make_plan([(1.0, 3.0)], [{"at_src": 0.0, "mode": "fixed", "x": 0.5, "y": 0.5}])
        self.assertEqual(framing.face_windows(plan, {}, "vertical"), [])
        plan = make_plan([(1.0, 3.0)], [{"at_src": 0.0, "mode": "letterbox"}])
        self.assertEqual(framing.face_windows(plan, {}, "vertical"), [])
        plan = make_plan([(1.0, 3.0)], [{"at_src": 0.0, "mode": "track", "x": 0.5}])
        self.assertEqual(framing.face_windows(plan, {}, "landscape"), [])
        (lo, hi), = framing.face_windows(plan, {}, "vertical")
        self.assertAlmostEqual(lo, 1.0 - framing.TRACK_MARGIN)
        self.assertAlmostEqual(hi, 3.0 - 0.04 + framing.TRACK_MARGIN)
        # no mark at all: the per-shot default follows a face, so it needs the data too
        self.assertEqual(len(framing.face_windows(make_plan([(1.0, 3.0)]), {}, "vertical")), 1)
        # a fixed mark that starts later leaves the first second to the default
        plan = make_plan([(1.0, 3.0)], [{"at_src": 2.0, "mode": "fixed", "x": 0.5, "y": 0.5}])
        (lo, hi), = framing.face_windows(plan, {}, "vertical")
        self.assertAlmostEqual(hi, 2.0 - 0.04 + framing.TRACK_MARGIN)
        self.assertEqual(framing.face_windows(dict(plan, kind="caption_only"), {}, "vertical"), [])

    def test_covers(self):
        src = {"fingerprint": "fp", "duration": 100.0}
        data = {"grid": "absolute", "sample_fps": 4, "source_fingerprint": "fp",
                "ranges": [[10.0, 20.0], [40.0, 50.0]]}
        self.assertTrue(shots.covers(data, [(10.1, 19.9), (41.0, 42.0)], src))
        self.assertFalse(shots.covers(data, [(9.0, 12.0)], src))
        self.assertFalse(shots.covers(data, [(19.0, 41.0)], src))            # spans the hole
        self.assertFalse(shots.covers(dict(data, grid=None), [(11.0, 12.0)], src))   # older sampler
        self.assertFalse(shots.covers(dict(data, source_fingerprint="other"), [(11.0, 12.0)], src))
        self.assertFalse(shots.covers(None, [(11.0, 12.0)], src))
        self.assertTrue(shots.covers(None, [], src) is False)
        end = {"grid": "absolute", "sample_fps": 4, "ranges": [[95.0, 100.0]]}
        self.assertTrue(shots.covers(end, [(96.0, 103.0)], src))             # clamped to the source


# ---- 2: range-independent tracks -------------------------------------------------------------

class TracksDoNotDependOnTheAnalysedRange(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="hc-fix2-fa-grid-"))
        cls.src = encode_gray(cls.tmp / "disc.mp4", disc_frames(250), audio=True)
        cls.P = fixtures.make_project(cls.tmp, cls.src)
        cls.info = project.Project(cls.P).source()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_sample_times_are_absolute(self):
        self.assertEqual(shots.sample_times(0.3, 1.1), [0.5, 0.75, 1.0])
        self.assertEqual(shots.sample_times(2.0, 2.5), [2.0, 2.25, 2.5])
        self.assertEqual(shots.sample_times(0.26, 0.49), [])

    def test_a_sample_shows_the_same_frame_whatever_the_range(self):
        with mock.patch.object(faces, "detect", disc_detect):
            ta, ga, fa = shots._decode_range(self.src, self.info, 0.3, 8.6)
            tb, gb, fb = shots._decode_range(self.src, self.info, 3.05, 6.9)
        self.assertEqual(ta[0], 0.5)
        self.assertEqual(tb[0], 3.25)
        self.assertEqual(tb[-1], 6.75)
        at = {t: k for k, t in enumerate(ta)}
        for k, t in enumerate(tb):
            self.assertTrue(np.array_equal(ga[at[t]], gb[k]), "sample %.2f differs" % t)
            self.assertEqual(list(fa[at[t]][0]), list(fb[k][0]))
        # the disc moves 0.18 of the width in 10 s: each sample is the frame on screen at its time
        x = fa[at[5.0]][0][0]
        self.assertAlmostEqual(x, 0.38 + 0.18 * (5.0 * 25) / 249.0, delta=0.002)

    def test_two_covering_analyses_give_identical_tracks(self):
        with mock.patch.object(faces, "detect", disc_detect):
            a = shots.analyze(self.P, range_src=["0.3", "8.6"])
            b = shots.analyze(self.P, range_src=["1.3", "9.7"])
        self.assertNotEqual(a["ranges"], b["ranges"])
        marks = [{"at_src": 0.0, "mode": "track", "x": 0.45}]
        plan = make_plan([(3.0, 4.6), (5.2, 6.0)], marks, joins=[(1, "piece")])
        ta = framing.build_tracks(plan, {}, a, self.info, "vertical", PRESET)
        tb = framing.build_tracks(plan, {}, b, self.info, "vertical", PRESET)
        self.assertEqual(ta, tb)
        self.assertGreater(len(ta[0][0]["cmds"]), 2)                     # and they do follow the disc
        # the default framing (no marks) tracks the same way from either analysis
        plan = make_plan([(3.0, 4.6)])
        da = framing.build_tracks(plan, {}, a, self.info, "vertical", PRESET)
        db = framing.build_tracks(plan, {}, b, self.info, "vertical", PRESET)
        self.assertEqual([framing.render_view(s) for s in da[0]], [framing.render_view(s) for s in db[0]])

    def test_a_tracker_reads_only_its_window(self):
        # Two faces near the mark's x: one sits still at 0.40, the other walks in from 0.62 and
        # stops at 0.47 long before the piece. One analysis holds all 60 s of samples, the other
        # the 8 s around the piece. A tracker that reads everything since the mark locks onto the
        # first face it saw and gives different tracks for the two; a tracker that reads only
        # the piece's window gives the same tracks for both.
        def at(t):
            return [face(0.40), face(max(0.47, 0.62 - 0.006 * t))]
        full = [{"t_src": k * 0.25, "faces": at(k * 0.25)} for k in range(241)]
        near = [s for s in full if 26.0 <= s["t_src"] <= 34.0]
        plan = make_plan([(28.0, 31.0)], [{"at_src": 0.0, "mode": "track", "x": 0.46},
                                           {"at_src": 50.0, "mode": "track", "x": 0.50}])
        ta = framing.build_tracks(plan, {}, {"shots": [], "samples": full}, SRC_169, "vertical", PRESET)
        tb = framing.build_tracks(plan, {}, {"shots": [], "samples": near}, SRC_169, "vertical", PRESET)
        self.assertEqual(ta, tb)
        cx = (ta[0][0]["cmds"][0][1] + ta[0][0]["size"][0] / 2) / 1920.0
        self.assertAlmostEqual(cx, 0.47, delta=0.01)             # the face nearest the mark's x
        read = {}
        framing.build_tracks(plan, {}, {"shots": [], "samples": full}, SRC_169, "vertical", PRESET, read)
        self.assertEqual(read["trackers"], 1)
        self.assertTrue(read["faces"])


# ---- 3: edge offsets and the time_src opener ---------------------------------------------------

class EdgeOffsets(unittest.TestCase):
    #          a false start folded into one long first word, then the sentence
    WORDS = [("lead", 0.3, 0.6), ("in.", 0.7, 1.0),
             ("after", 2.0, 2.2), ("about", 2.2, 3.0), ("a", 3.05, 3.15), ("decade", 3.2, 3.7),
             ("it", 3.8, 3.95), ("ends.", 4.0, 4.5)]
    SPANS = [(0.3, 0.6), (0.7, 1.0), (2.0, 2.42), (2.5, 3.0), (3.05, 3.15), (3.2, 3.7),
             (3.8, 3.95), (4.0, 4.5)]

    def test_bound_is_sixty_frames(self):
        self.assertEqual(S.EDGE_OFFSET_MAX, 60)
        for ok in (13, -13, 17, 60, -60):
            sp, iss = issues({"pieces": [{"words": [0, 3], "in_offset": ok, "out_offset": ok}]})
            self.assertEqual(iss, [], ok)
            self.assertEqual(sp["clips"]["c01"]["pieces"][0]["in_offset"], ok)
        for bad in (61, -61, 1.5, "x", True):
            _sp, iss = issues({"pieces": [{"words": [0, 3], "in_offset": bad}]})
            self.assertIn(("L01", "FAIL"), iss, bad)

    def test_a_long_in_offset_skips_the_false_start_and_is_named_in_seconds(self):
        base = build(self.WORDS, {"c01": {"pieces": [{"words": [3, 7]}]}}, spans=self.SPANS)
        plan = build(self.WORDS, {"c01": {"pieces": [{"words": [3, 7], "in_offset": 15}]}},
                     spans=self.SPANS)
        self.assertEqual(lints(plan, "L01", "FAIL"), [])
        k0, k1 = base["pieces"][0]["edge_in"]["k"], plan["pieces"][0]["edge_in"]["k"]
        self.assertEqual(k1 - k0, 15)
        self.assertGreater(plan["segments"][0]["src"][0], 2.42)            # past the first attempt
        note = lints(plan, "L01", "INFO")
        self.assertEqual(len(note), 1)
        self.assertIn("in_offset +15 moves the in-edge 0.60 s later", note[0]["msg"])
        # twelve frames or fewer: no note
        small = build(self.WORDS, {"c01": {"pieces": [{"words": [3, 7], "in_offset": 12}]}},
                      spans=self.SPANS)
        self.assertEqual(lints(small, "L01", "INFO"), [])

    def test_an_opener_stepped_past_a_false_start_is_not_called_mid_sentence(self):
        # "after" (2.0-2.2) is the abandoned attempt; the piece starts on "about", whose span
        # holds the real opening. Plain, it opens right after "after" with no pause: L04.
        def mid(plan):
            return [x["msg"] for x in lints(plan, "L04") if "not at a sentence start" in x["msg"]]
        plain = build(self.WORDS, {"c01": {"pieces": [{"words": [3, 7]}]}}, spans=self.SPANS)
        self.assertEqual(len(mid(plain)), 1)
        # in_offset 15 puts the in-edge past the end of "after": that word is not in the clip, so
        # it no longer says where the sentence starts (the L01 INFO names the shift instead)
        plan = build(self.WORDS, {"c01": {"pieces": [{"words": [3, 7], "in_offset": 15}]}},
                     spans=self.SPANS)
        self.assertGreater(plan["segments"][0]["src"][0], 2.2)
        self.assertEqual(mid(plan), [])
        self.assertEqual(len(lints(plan, "L01", "INFO")), 1)
        # a nudge of a few frames is not a hand-placed opening: still judged
        small = build(self.WORDS, {"c01": {"pieces": [{"words": [3, 7], "in_offset": 12}]}},
                      spans=self.SPANS)
        self.assertEqual(len(mid(small)), 1)
        # and so is a long offset on a piece that is not the opener
        late = build(self.WORDS, {"c01": {"pieces": [{"words": [0, 1]},
                                                     {"words": [3, 7], "in_offset": 15}]}},
                     spans=self.SPANS)
        self.assertEqual(mid(late), [])            # the clip opens on "lead", a sentence start
        other = [("so", 0.3, 0.5)] + [(w, s + 0.0, e) for w, s, e in self.WORDS[1:]]
        opener = build([("x", 0.05, 0.2)] + other,
                       {"c01": {"pieces": [{"words": [2, 2]}, {"words": [4, 8], "in_offset": 15}]}},
                       spans=[(0.05, 0.2)] + self.SPANS)
        self.assertEqual(len(mid(opener)), 1)      # judged on its own opener, "in." after "so"

    def test_a_clamped_offset_says_where_it_stopped(self):
        plan = build(self.WORDS, {"c01": {"pieces": [{"words": [3, 7], "in_offset": 40}]}},
                     spans=self.SPANS)
        msg = lints(plan, "L01", "INFO")[0]["msg"]
        self.assertIn("in_offset +40", msg)
        self.assertIn("held to", msg)
        self.assertLessEqual(plan["segments"][0]["src"][0], 3.0 + 1e-6)     # never past its first word

    def test_a_long_widening_offset_names_the_words_it_brings_in(self):
        words = grid_words(12)
        plan = build(words, {"c01": {"pieces": [{"words": [5, 9], "in_offset": -30,
                                                 "out_offset": 25}]}})
        notes = [x["msg"] for x in lints(plan, "L01", "INFO")]
        self.assertEqual(len(notes), 2)
        self.assertIn("in_offset -30 moves the in-edge 1.20 s earlier", notes[0])
        self.assertIn("brings in words 2-4 ('w2 w3 w4')", notes[0])
        self.assertIn("heard but not captioned", notes[0])
        self.assertIn("out_offset +25 moves the out-edge 1.00 s later", notes[1])
        self.assertIn("brings in words 10-11", notes[1])
        self.assertEqual(lints(plan, "L01", "FAIL"), [])
        # the words it brings in are in the plan as hidden edge words: audible, never captioned
        edge = [w["i"] for w in plan["words"] if w.get("edge") and w["s_out"] is not None]
        self.assertTrue({2, 3, 4, 10, 11} <= set(edge))

    def test_time_src_opener_is_not_judged_by_a_word_before_it(self):
        # the editor opens at 2.48 s (the second attempt). The transcript's "after" ends at 2.2,
        # before that, so "not at a sentence start (previous word 'after')" would be about audio
        # that is not in the clip.
        clip = {"pieces": [{"time_src": [2.48, 4.6]}]}
        plan = build(self.WORDS, {"c01": clip}, spans=self.SPANS)
        self.assertEqual(next(w["raw"] for w in plan["words"] if not w["hidden"]), "about")
        self.assertEqual([x["msg"] for x in lints(plan, "L04")], [])
        # a words piece that starts on the same word still gets the check
        plan = build(self.WORDS, {"c01": {"pieces": [{"words": [3, 7]}]}}, spans=self.SPANS)
        self.assertTrue(any("not at a sentence start" in x["msg"] for x in lints(plan, "L04")))
        # and a time piece that cuts INTO the previous word is still told
        clip = {"pieces": [{"time_src": [3.5, 4.6]}]}
        plan = build(self.WORDS, {"c01": clip}, spans=self.SPANS)
        self.assertTrue(any("not at a sentence start" in x["msg"] for x in lints(plan, "L04")),
                        [x["msg"] for x in plan["lints"]])


# ---- 4: the final consonant ------------------------------------------------------------------

class FinalConsonant(unittest.TestCase):
    """"... makes no sense. So you ..." with no pause: the s of "sense" runs into "So"."""
    WORDS = [("it", 1.0, 1.2), ("makes", 1.25, 1.6), ("no", 1.65, 1.9), ("sense.", 1.95, 2.46),
             ("So", 2.64, 2.8), ("you", 2.85, 3.1)]
    # vowel and nasal to 2.50, a dip, the s from 2.52 to 2.62, a dip, then the next word
    SOUND = [(1.0, 1.2), (1.25, 1.6), (1.65, 1.9), (1.95, 2.50, -22.0), (2.50, 2.52, -36.0),
             (2.52, 2.62, -24.0), (2.62, 2.63, -40.0), (2.63, 3.1, -18.0)]

    def plan(self, **piece):
        p = dict({"words": [0, 3]}, **piece)
        return build(self.WORDS, {"c01": {"pieces": [p]}}, spans=self.SOUND, dur=8.0)

    def test_the_fade_no_longer_covers_the_consonant(self):
        plan = self.plan()
        e = plan["pieces"][0]["edge_out"]
        self.assertEqual(e["guard"], "consonant")
        self.assertGreaterEqual(e["auto"], 1)
        end = plan["segments"][-1]["src"][1]
        fade_from = end - plan["segments"][-1]["fade_out"]
        self.assertGreaterEqual(fade_from, 2.52 + 0.04)        # the s is heard before the fade bites
        self.assertLessEqual(fade_from, 2.625 + 1e-6)          # and the fade starts by the dip
        self.assertLessEqual(end, 2.625 + 0.03 + 0.04)         # the next word stays under the fade
        info = [x for x in lints(plan, "L10", "INFO") if "final consonant" in x["msg"]]
        self.assertEqual(len(info), 1)
        self.assertIn("out_offset %d keeps this edge" % e["auto"], info[0]["msg"])
        self.assertEqual([x for x in lints(plan, "L10", "WARN") if "clip ends" in x["msg"]], [])

    def test_an_explicit_out_offset_replaces_the_guard(self):
        auto = self.plan()["pieces"][0]["edge_out"]
        manual = self.plan(out_offset=auto["auto"])["pieces"][0]["edge_out"]
        self.assertNotEqual(manual.get("guard"), "consonant")
        self.assertEqual(manual["k"], auto["k"])               # the same edge, now the editor's
        one = self.plan(out_offset=1)["pieces"][0]["edge_out"]
        self.assertEqual(one["k"], auto["k"] - auto["auto"] + 1)

    def test_a_pause_or_a_hold_needs_no_guard(self):
        words = self.WORDS[:4] + [("So", 3.2, 3.4), ("you", 3.45, 3.7)]
        sound = [(1.0, 1.2), (1.25, 1.6), (1.65, 1.9), (1.95, 2.46), (3.2, 3.7)]
        plan = build(words, {"c01": {"pieces": [{"words": [0, 3]}]}}, spans=sound, dur=8.0)
        self.assertNotEqual(plan["pieces"][0]["edge_out"].get("guard"), "consonant")
        held = self.plan(hold_after=0.3)
        self.assertNotEqual(held["pieces"][0]["edge_out"].get("guard"), "consonant")

    def test_only_the_last_edge_of_the_clip(self):
        clip = {"pieces": [{"words": [0, 3]}, {"words": [0, 1]}]}
        plan = build(self.WORDS, {"c01": clip}, spans=self.SOUND, dur=8.0)
        self.assertNotEqual(plan["pieces"][0]["edge_out"].get("guard"), "consonant")


# ---- 5: inserted speech ------------------------------------------------------------------------

class InsertedSpeech(unittest.TestCase):
    WORDS = [("this", 1.0, 1.3), ("moment.", 1.4, 1.8), ("Then", 3.4, 3.7), ("more.", 3.8, 4.2)]
    INS = {"after": 1, "text": "I use this method.", "src": [1.9, 2.7]}

    def plan(self, ins=None, spans=None, flags=None, **clip):
        c = dict({"pieces": [{"words": [0, 3]}], "insert": [ins or self.INS]}, **clip)
        return build(self.WORDS, {"c01": c}, spans=spans, flags=flags, dur=8.0)

    def test_spec_normalises_and_hashes_it(self):
        sp, iss = issues({"pieces": [{"words": [0, 3]}],
                          "insert": [{"after": 1, "text": "  I use   this method. ", "src": ["0:01.9", 2.7]}]})
        self.assertEqual(iss, [])
        self.assertEqual(sp["clips"]["c01"]["insert"], [self.INS])
        other, _ = issues({"pieces": [{"words": [0, 3]}]})
        self.assertEqual(other["clips"]["c01"]["insert"], [])
        self.assertNotEqual(S.clip_hash(sp, "c01"), S.clip_hash(other, "c01"))
        for bad in ({"after": -1, "text": "x", "src": [1, 2]}, {"after": 1, "text": " ", "src": [1, 2]},
                    {"after": 1, "text": "x", "src": [2, 1]}, {"after": 1, "text": "x"}, "x"):
            _sp, iss = issues({"pieces": [{"words": [0, 3]}], "insert": [bad]})
            self.assertIn(("L01", "FAIL"), iss, bad)

    def test_plan_words_and_inserts(self):
        plan = self.plan()
        self.assertEqual(lints(plan, "L01"), [])
        self.assertEqual(plan["inserts"], [{"n": 0, "after": 1, "tokens": ["I", "use", "this", "method."],
                                            "src": [1.9, 2.7], "piece": 0}])
        seq = [(w["i"], w["text"]) for w in plan["words"]]
        self.assertEqual(seq, [(0, "this"), (1, "moment."), (None, "I"), (None, "use"), (None, "this"),
                               (None, "method."), (2, "Then"), (3, "more.")])
        toks = [w for w in plan["words"] if w["i"] is None]
        self.assertTrue(all(w["ins"] == 0 and w["sp"] is None and not w["hidden"] and not w["filler"]
                            and not w["omitted"] and not w["removed"] and w["piece"] == 0 for w in toks))
        # times are spread over src by character length (1 + 3 + 4 + 7 = 15 characters)
        self.assertAlmostEqual(toks[0]["s_src"], 1.9, places=3)
        self.assertAlmostEqual(toks[1]["s_src"], 1.9 + 0.8 * 1 / 15, places=3)
        self.assertAlmostEqual(toks[3]["s_src"], 1.9 + 0.8 * 8 / 15, places=3)
        self.assertAlmostEqual(toks[3]["e_src"], 2.7, places=3)
        seg = plan["segments"][toks[0]["seg"]]
        for w in toks:
            self.assertAlmostEqual(w["s_out"], seg["out"][0] + (w["s_src"] - seg["src"][0]), places=3)
            self.assertLessEqual(w["s_out"], w["e_out"])
        self.assertEqual(plan["intent"], [0, 1, 2, 3])               # indices only: inserts ride apart

    def test_tighten_keeps_the_inserted_speech(self):
        # the 1.6 s between "moment." and "Then" is quiet on the energy track, so it would be
        # tightened; the insert marks 0.8 s of it as speech
        bare = build(self.WORDS, {"c01": {"pieces": [{"words": [0, 3]}]}}, dur=8.0)
        self.assertTrue(any(r["src"][0] < 2.7 and r["src"][1] > 1.9 for r in bare["removals"]))
        plan = self.plan()
        for r in plan["removals"]:
            self.assertFalse(r["src"][0] < 2.7 + 0.04 and r["src"][1] > 1.9 - 0.09, r)
        self.assertTrue(any(s["src"][0] <= 1.9 and s["src"][1] >= 2.7 for s in plan["segments"]))

    def test_src_must_be_audio_the_clip_plays(self):
        plan = self.plan(ins={"after": 1, "text": "gone", "src": [5.5, 5.9]})
        fails = lints(plan, "L01", "FAIL")
        self.assertEqual(len(fails), 1)
        self.assertIn("not inside the kept audio of pieces[0]", fails[0]["msg"])
        self.assertEqual([w for w in plan["words"] if w["i"] is None], [])
        # after must be a word of the clip, and src must not start before that word
        tr = fake_transcript(self.WORDS)
        sp = S.normalize_spec({"version": 1, "clips": {"c01": {
            "pieces": [{"words": [0, 1]}],
            "insert": [{"after": 3, "text": "x", "src": [4.3, 4.5]},
                       {"after": 9, "text": "x", "src": [4.3, 4.5]},
                       {"after": 1, "text": "x", "src": [1.1, 1.3]}]}}})
        msgs = [x["msg"] for x in S.validate(sp, tr, source()) if x["level"] == "FAIL"]
        self.assertEqual(len(msgs), 3)
        self.assertTrue(any("is not inside any piece" in m for m in msgs))
        self.assertTrue(any("out of range" in m for m in msgs))
        self.assertTrue(any("before word 1" in m for m in msgs))

    def test_lints_treat_it_as_worded_speech(self):
        loud = [(w[1], w[2]) for w in self.WORDS] + [(1.85, 3.35)]       # 1.5 s of speech, no words
        flag = [{"kind": "unworded_audio", "s": 1.85, "e": 3.35, "a": 1, "b": 2}]
        bare = build(self.WORDS, {"c01": {"pieces": [{"words": [0, 3]}]}}, spans=loud, flags=flag, dur=8.0)
        self.assertTrue(lints(bare, "L15"))
        self.assertTrue(lints(bare, "L19"))
        plan = self.plan(ins={"after": 1, "text": "I use this method.", "src": [1.9, 3.3]}, spans=loud,
                         flags=flag)
        self.assertEqual(lints(plan, "L15"), [])
        self.assertEqual(lints(plan, "L19"), [])

    def test_hook_words_said_only_in_the_insert_count_as_spoken(self):
        plan = self.plan(hook_card={"lines": ["USE THIS *METHOD*"]})
        self.assertEqual([x["msg"] for x in lints(plan, "L13")], [])
        self.assertEqual([x["msg"] for x in lints(plan, "L08") if "never spoken" in x["msg"]], [])

    def test_a_clip_can_close_on_inserted_words(self):
        plan = build(self.WORDS, {"c01": {"pieces": [{"words": [0, 1], "hold_after": 1.0}],
                                          "insert": [self.INS]}}, dur=8.0)
        self.assertEqual(lints(plan, "L01"), [])
        self.assertEqual(plan["words"][-1]["text"], "method.")
        self.assertEqual(lints(plan, "L05"), [])                           # a sentence end, then a pause

    def test_caption_only_clips_take_it_too(self):
        tr = fake_transcript(self.WORDS)
        spec = S.normalize_spec({"version": 1, "clips": {"c01": {
            "kind": "caption_only", "source_file": "v.mp4", "insert": [self.INS]}}})
        media = {"duration": 8.0, "video": {"fps": [25, 1], "start": 0.0, "width": 1080, "height": 1920,
                                            "frames": 200, "hdr": "none"}}
        plan = P.plan_caption_only(spec, "c01", tr, media)
        self.assertEqual([w["text"] for w in plan["words"] if w["i"] is None],
                         ["I", "use", "this", "method."])
        self.assertEqual(plan["inserts"][0]["tokens"], ["I", "use", "this", "method."])
        self.assertEqual(plan["intent"], [0, 1, 2, 3])

    def test_plan_table_names_the_insert(self):
        out = io.StringIO()
        P.print_plan(self.plan(), file=out)
        self.assertIn('insert 0: after word 1 "I use this method."', out.getvalue())


# ---- 6: caption_backing ------------------------------------------------------------------------

class CaptionBacking(unittest.TestCase):
    def test_values_and_hash(self):
        base, iss = issues({"pieces": [{"words": [0, 3]}]})
        self.assertEqual(iss, [])
        self.assertIsNone(base["clips"]["c01"]["caption_backing"])
        seen = {S.clip_hash(base, "c01")}
        for mode in ("auto", "always", "never"):
            sp, iss = issues({"pieces": [{"words": [0, 3]}], "caption_backing": mode})
            self.assertEqual(iss, [], mode)
            self.assertEqual(sp["clips"]["c01"]["caption_backing"], mode)
            seen.add(S.clip_hash(sp, "c01"))
        self.assertEqual(len(seen), 4)                       # each value is its own clip hash
        sp, iss = issues({"pieces": [{"words": [0, 3]}], "caption_backing": " Always "})
        self.assertEqual(sp["clips"]["c01"]["caption_backing"], "always")
        for bad in ("sometimes", True, 1, ["auto"]):
            sp, iss = issues({"pieces": [{"words": [0, 3]}], "caption_backing": bad})
            self.assertIn(("L01", "FAIL"), iss, bad)
            self.assertIsNone(sp["clips"]["c01"]["caption_backing"])

    def test_plan_and_render_keys_carry_it(self):
        words = grid_words(6)
        a = build(words, {"c01": {"pieces": [{"words": [0, 5]}]}})
        b = build(words, {"c01": {"pieces": [{"words": [0, 5]}], "caption_backing": "never"}})
        self.assertNotEqual(a["spec_hash"], b["spec_hash"])
        self.assertNotIn("caption_backing", render.CLIP_NOT_RENDERED)


# ---- 7: small faces of similar size --------------------------------------------------------------

class WideTwoShot(unittest.TestCase):
    def fl(self, *faces_):
        return [{"x": x, "y": 0.4, "h": h, "presence": p} for x, h, p in faces_]

    def test_two_small_similar_faces_are_multi_face(self):
        fl = self.fl((0.222, 0.097, 1.0), (0.801, 0.102, 1.0))
        self.assertEqual(shots.shot_kind(fl), "wide")                    # the kind stays what it was
        self.assertTrue(shots.faces_multi(fl, "wide"))
        self.assertTrue(shots.similar_faces(fl))
        self.assertFalse(shots.similar_faces(self.fl((0.3, 0.10, 1.0), (0.7, 0.05, 1.0))))   # 2x apart
        self.assertFalse(shots.similar_faces(self.fl((0.3, 0.10, 1.0), (0.7, 0.10, 0.4))))   # rarely seen
        self.assertFalse(shots.similar_faces(self.fl((0.3, 0.10, 1.0))))
        self.assertTrue(shots.similar_faces(self.fl((0.3, 0.16, 0.5), (0.7, 0.10, 0.5))))    # exactly 1.6
        self.assertFalse(shots.faces_multi(self.fl((0.5, 0.3, 1.0)), "single"))

    def test_a_shots_json_written_before_the_rule_is_judged_again(self):
        old = {"start_src": 0.0, "end_src": 30.0, "kind": "wide", "multi_face": False,
               "faces": self.fl((0.222, 0.097, 1.0), (0.801, 0.102, 1.0))}
        self.assertTrue(shots.is_multi(old))
        self.assertTrue(framing.shot_is_multi(old))
        sh = {"shots": [old], "samples": []}
        self.assertEqual(framing.default_mark("vertical", sh, 5.0)["mode"], "letterbox")
        prop = shots.propose([old])
        self.assertEqual((prop[0]["mode"], prop[0].get("multi_face")), ("letterbox", True))
        # a crop of one of them carries the flag QA G16 looks for
        plan = make_plan([(1.0, 2.0)], [{"at_src": 0.0, "mode": "track", "x": 0.22, "crop": 0.5}])
        sub = framing.build_tracks(plan, {}, sh, SRC_169, "vertical", PRESET)[0][0]
        self.assertTrue(sub["multi_face"])
        self.assertTrue(framing.windows([[sub]], plan, SRC_169)[0]["multi_face"])

    def test_analysis_writes_the_flag(self):
        tmp = Path(tempfile.mkdtemp(prefix="hc-fix2-fa-wide-"))
        try:
            h, w = 360, 640
            yy, xx = np.mgrid[0:h, 0:w]
            f = np.full((h, w), 28, np.uint8)
            for cx in (0.22, 0.80):
                f[(xx - cx * w) ** 2 + (yy - 0.45 * h) ** 2 <= 17 * 17] = 225    # h about 0.097
            src = encode_gray(tmp / "wide.mp4", [f] * 100, audio=True)
            P_ = fixtures.make_project(tmp, src)

            def two(image, *_a, **_k):
                g = np.asarray(image)[..., 0]
                out = []
                for lo, hi in ((0, g.shape[1] // 2), (g.shape[1] // 2, g.shape[1])):
                    ys, xs = np.nonzero(g[:, lo:hi] > 128)
                    if len(xs) > 20:
                        gh, gw = g.shape
                        out.append(faces.Face([(xs.mean() + lo) / gw, ys.mean() / gh,
                                               (xs.max() - xs.min() + 1) / gw,
                                               (ys.max() - ys.min() + 1) / gh, 0.9], [(0.5, 0.5)] * 5))
                return out
            with mock.patch.object(faces, "detect", two):
                res = shots.analyze(P_, range_src=["0.5", "3.5"])
            shot = res["shots"][0]
            self.assertEqual(shot["kind"], "wide")
            self.assertTrue(shot["multi_face"])
            self.assertTrue(res["multi_face"])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


# ---- 8: dissolves ------------------------------------------------------------------------------

class Dissolves(unittest.TestCase):
    """Shot A, a hard cut to shot B at frame 100, then an 8-frame cross dissolve to shot C."""
    FPS = 25
    CUT = 100
    DIS = (200, 208)          # frames 200..207 mix B and C; frame 208 is C alone

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="hc-fix2-fa-dissolve-"))
        h, w = 180, 320
        rng = np.random.default_rng(7)

        def shot(seed, level):
            f = np.full((h, w), level, np.float32)
            r = np.random.default_rng(seed)
            for _ in range(6):
                x, y = int(r.integers(0, w - 60)), int(r.integers(0, h - 40))
                f[y:y + 40, x:x + 60] = float(r.integers(40, 220))
            return f
        a, b, c = shot(1, 40), shot(2, 120), shot(3, 70)
        frames = []
        for n in range(325):
            if n < cls.CUT:
                f = a
            elif n < cls.DIS[0]:
                f = b
            elif n < cls.DIS[1]:
                mix = (n - cls.DIS[0] + 1) / float(cls.DIS[1] - cls.DIS[0] + 1)
                f = b * (1 - mix) + c * mix
            else:
                f = c
            frames.append(np.clip(f + rng.normal(0, 0.6, f.shape), 0, 255).astype(np.uint8))
        src = encode_gray(cls.tmp / "three.mp4", frames, fps=cls.FPS, audio=True)
        cls.P = fixtures.make_project(cls.tmp, src)
        cls.res = shots.analyze(cls.P, range_src=["0.5", "12.5"])

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def t(self, frame):
        return frame / float(self.FPS)

    def test_transitions_tell_a_cut_from_a_dissolve(self):
        tr = self.res["transitions"]
        self.assertEqual([x["kind"] for x in tr], ["cut", "dissolve"])
        cut, dis = tr
        self.assertAlmostEqual(cut["at_src"], self.t(self.CUT), places=3)
        self.assertEqual((cut["start_src"], cut["end_src"]), (cut["at_src"], cut["at_src"]))
        self.assertAlmostEqual(dis["start_src"], self.t(self.DIS[0]), places=3)   # first mixed frame
        self.assertAlmostEqual(dis["end_src"], self.t(self.DIS[1]), places=3)     # first clean frame
        self.assertAlmostEqual(dis["at_src"], self.t(204), delta=0.041)           # where C takes over
        self.assertTrue(dis["start_src"] < dis["at_src"] < dis["end_src"])
        # shot boundaries: the exact frame for the cut, the midpoint for the dissolve
        starts = [s["start_src"] for s in self.res["shots"]]
        self.assertEqual(starts[1:], [cut["at_src"], dis["at_src"]])
        on_disk = env.read_json(Path(self.P) / "shots" / "range-0-12.json")
        self.assertEqual(on_disk["transitions"], tr)
        self.assertEqual([c["kind"] for c in on_disk["cuts"]], ["cut", "dissolve"])

    def test_blend_shares_and_classification(self):
        ramp = [0.0] * 10 + [k / 9.0 for k in range(1, 9)] + [1.0] * 10
        self.assertEqual(shots.classify_transition(ramp), ("dissolve", 10, 14, 18))
        hard = [0.0] * 10 + [1.0] * 10
        self.assertEqual(shots.classify_transition(hard), ("cut", 10, 10, 10))
        blended_field = [0.0] * 10 + [0.5] + [1.0] * 10                   # one mixed frame: a cut
        self.assertEqual(shots.classify_transition(blended_field)[0], "cut")
        self.assertIsNone(shots.classify_transition([0.0] * 5 + [1.0] * 3 + [0.0] * 5 + [1.0] * 5))
        self.assertIsNone(shots.classify_transition([0.2 + 0.03 * k for k in range(21)]))  # no clean end
        self.assertIsNone(shots.classify_transition(None))
        flat = [np.full((4, 4), 50.0, np.float32)] * 12
        self.assertIsNone(shots.blend_shares(flat))                       # the two ends are the same

    def test_a_mark_inside_the_dissolve_switches_at_its_midpoint(self):
        dis = self.res["transitions"][1]
        late = dis["end_src"] + 0.04                       # where a mark pasted from old output sat
        marks = [{"at_src": 0.0, "mode": "fixed", "x": 0.3, "y": 0.5},
                 {"at_src": late, "mode": "fixed", "x": 0.7, "y": 0.5}]
        snapped = framing.snap_marks(framing.norm_marks(marks), self.res)
        self.assertEqual(snapped[1]["at_src"], dis["at_src"])
        self.assertEqual(snapped[0]["at_src"], 0.0)
        plan = make_plan([(7.0, 9.0)], marks)
        src = {"video": {"width": 1920, "height": 1080, "sar": "1:1"}}
        subs = framing.build_tracks(plan, {}, self.res, src, "vertical", PRESET)[0]
        self.assertEqual(len(subs), 2)
        self.assertEqual(subs[0]["frames"], int(round((dis["at_src"] - 7.0) * self.FPS)))
        # without transitions (an older shots json) the mark stays where it was written
        legacy = {k: v for k, v in self.res.items() if k != "transitions"}
        subs = framing.build_tracks(plan, {}, legacy, src, "vertical", PRESET)[0]
        self.assertEqual(subs[0]["frames"], int(round((late - 7.0) * self.FPS)))
        # a mark far from any dissolve, or near the hard cut, is left alone
        far = framing.snap_marks(framing.norm_marks([{"at_src": 4.1, "mode": "letterbox"},
                                                     {"at_src": 6.0, "mode": "letterbox"}]), self.res)
        self.assertEqual([m["at_src"] for m in far], [4.1, 6.0])


# ---- 9: hold_after and the frame grid -----------------------------------------------------------

class HoldLandsOnAFrame(unittest.TestCase):
    WORDS = [("so", 1.0, 1.3), ("there.", 1.4, 1.8), ("next", 6.0, 6.3), ("bit", 6.4, 6.7)]
    # after the last word: room tone, one quiet beat around 3.2 s, then a laugh that runs on
    SOUND = [(1.0, 1.3), (1.4, 1.8), (1.8, 3.16, -48.0), (3.24, 5.0, -30.0), (6.0, 6.7)]

    def plan(self, hold, **piece):
        p = dict({"words": [0, 1], "hold_after": hold}, **piece)
        return build(self.WORDS, {"c01": {"pieces": [p]}}, spans=self.SOUND, dur=10.0)

    def test_nearby_holds_share_a_frame_and_the_plan_says_which(self):
        a, b = self.plan(1.30), self.plan(1.42)
        ka, kb = a["pieces"][0]["edge_out"]["k"], b["pieces"][0]["edge_out"]["k"]
        self.assertEqual(ka, kb)
        h = a["pieces"][0]["hold"]
        self.assertEqual(h["asked"], 1.3)
        self.assertEqual(h["frame"], ka)
        self.assertAlmostEqual(h["frame_src"], ka / 25.0, places=4)
        self.assertAlmostEqual(h["target_src"], 1.8 + 0.4 + 1.30, places=3)
        self.assertEqual(h["why"], "quietest")
        self.assertAlmostEqual(h["got"], h["frame_src"] - 1.8, places=3)
        out = io.StringIO()
        P.print_plan(a, file=out)
        line = next(ln for ln in out.getvalue().splitlines() if "hold_after" in ln)
        self.assertIn("hold_after 1.3 s: target 0:03.50, out-edge on frame %d" % ka, line)
        self.assertIn("the quietest frame near the target", line)
        self.assertIn("out_offset moves it frame by frame", line)
        self.assertNotIn("hold", self.plan(0.0)["pieces"][0])

    def test_out_offset_applies_after_the_hold_snap(self):
        base = self.plan(1.30)["pieces"][0]["edge_out"]["k"]
        for d in (-2, 1, 3):
            plan = self.plan(1.30, out_offset=d)
            self.assertEqual(plan["pieces"][0]["edge_out"]["k"], base + d, d)
            h = plan["pieces"][0]["hold"]
            self.assertEqual((h["frame"], h["frame_out"]), (base, base + d))
        out = io.StringIO()
        P.print_plan(self.plan(1.30, out_offset=3), file=out)
        self.assertIn("out_offset +3 then put it on frame %d" % (base + 3), out.getvalue())

    def test_a_clamped_hold_says_so(self):
        clip = {"pieces": [{"words": [0, 1], "hold_after": 5.0}, {"words": [2, 3]}]}
        plan = build(self.WORDS, {"c01": clip}, spans=[(1.0, 1.3), (1.4, 1.8), (6.0, 6.7)], dur=10.0)
        h = plan["pieces"][0]["hold"]
        self.assertEqual(h["why"], "clamped")
        self.assertEqual(h["frame"], plan["pieces"][0]["edge_out"]["k"])


# ---- 11: L15 for the new transcript flags ---------------------------------------------------------

class NewTranscriptFlags(unittest.TestCase):
    WORDS = grid_words(10)

    def test_messages_and_scope(self):
        flags = [{"kind": "false_start", "a": 2, "b": 3, "s": 1.4, "e": 2.1},
                 {"kind": "timing_uncertain", "a": 5, "b": 5},
                 {"kind": "false_start", "a": 8, "b": 9, "s": 3.8, "e": 4.5},
                 {"kind": "timing_uncertain", "a": 9, "b": 9}]
        plan = build(self.WORDS, {"c01": {"pieces": [{"words": [1, 6]}]}}, flags=flags)
        msgs = [x["msg"] for x in lints(plan, "L15", "WARN")]
        self.assertEqual(sorted(msgs), sorted([
            "possible false start or repeated attempt at src 0:01.40 (words 2-3): check with hc "
            "words and a slice transcription before opening on it",
            "word timing near src 0:02.60 is uncertain (overlapping sound): check the caption "
            "onset in the render"]))
        # flags on words outside the clip say nothing
        self.assertEqual(len(lints(plan, "L15")), 2)

    def test_a_false_start_the_clip_already_cut_is_settled(self):
        # words 2-3 are flagged; the abandoned attempt is src 1.4-1.72 and the phrase restarts
        # there. A clip that still plays the attempt is warned ...
        flags = [{"kind": "false_start", "a": 2, "b": 2, "s": 1.4, "e": 1.72}]
        words = [("w0", 0.6, 0.9), ("w1", 1.0, 1.3), ("after", 1.4, 2.1), ("w3", 2.2, 2.5),
                 ("w4", 2.6, 2.9), ("w5", 3.0, 3.3)]
        spans = [(0.6, 0.9), (1.0, 1.3), (1.4, 1.62), (1.74, 2.1), (2.2, 2.5), (2.6, 2.9), (3.0, 3.3)]
        plan = build(words, {"c01": {"pieces": [{"words": [2, 5]}]}}, spans=spans, flags=flags)
        self.assertLess(plan["segments"][0]["src"][0], 1.5)
        self.assertEqual(len(lints(plan, "L15", "WARN")), 1)
        # ... one whose in-edge sits on the restart (a time piece, or in_offset) is not
        cut = build(words, {"c01": {"pieces": [{"time_src": [1.72, 3.4]}]}}, spans=spans, flags=flags)
        self.assertEqual(lints(cut, "L15"), [])
        # a frame of the attempt may remain (the edge lands on the frame grid) ...
        near = build(words, {"c01": {"pieces": [{"time_src": [1.70, 3.4]}]}}, spans=spans, flags=flags)
        self.assertEqual(lints(near, "L15"), [])
        # ... two frames of it is the attempt again
        more = build(words, {"c01": {"pieces": [{"time_src": [1.60, 3.4]}]}}, spans=spans, flags=flags)
        self.assertEqual(len(lints(more, "L15", "WARN")), 1)

    def test_caption_only_plans_use_the_same_words(self):
        tr = fake_transcript(self.WORDS)
        tr["flags"] = [{"kind": "timing_uncertain", "a": 5, "b": 5},
                       {"kind": "false_start", "a": 2, "b": 3, "s": 1.4, "e": 2.1}]
        spec = S.normalize_spec({"version": 1, "clips": {"c01": {"kind": "caption_only",
                                                                 "source_file": "v.mp4"}}})
        media = {"duration": 8.0, "video": {"fps": [25, 1], "start": 0.0, "width": 1080, "height": 1920,
                                            "frames": 200, "hdr": "none"}}
        msgs = [x["msg"] for x in P.plan_caption_only(spec, "c01", tr, media)["lints"] if x["id"] == "L15"]
        self.assertTrue(any(m.startswith("word timing near src 0:02.60 is uncertain") for m in msgs))
        self.assertTrue(any(m.startswith("possible false start or repeated attempt at src 0:01.40")
                            for m in msgs))


if __name__ == "__main__":
    unittest.main()
