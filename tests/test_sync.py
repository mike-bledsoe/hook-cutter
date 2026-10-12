"""A/V sync regression (contract 12): 30 one-word pieces cut around flash/click pairs.

Each piece holds exactly one white flash frame and one 1 kHz click at the same container time.
After render we find every flash (luma) and click (energy) in the output and require
|click - flash| <= 1 frame for every piece, with no accumulation across the 29 joins. Both
earlier pipelines drifted about 31 ms per join; this test fails long before that.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "scripts"))

import fixtures  # noqa: E402
from hookcutter import env, project  # noqa: E402
from hookcutter import render  # noqa: E402

N_PIECES = 30
SPACING = 1.13          # s between flashes in the source
FIRST = 0.40            # first flash (its piece starts at 0: exercises a late audio start)
SR = 48000
SPREAD_MAX = 0.005      # s: max spread of per-piece A-V offsets (no accumulation)


def flash_times(rate: Fraction) -> list:
    return [float(Fraction(int(round((FIRST + k * SPACING) * rate))) / rate) for k in range(N_PIECES)]


def hand_plan(src: dict, rate: Fraction, flash_frames: list) -> tuple:
    """plan.json dict (contract 8.1) with one segment per piece around each flash frame.

    Returns (plan, expected output frame of each flash)."""
    grid0 = Fraction(src["video"]["start"]).limit_denominator(10 ** 6)
    num, den = rate.numerator, rate.denominator
    segs, words, joins, expect = [], [], [], []
    cur = 0
    for k, fk in enumerate(flash_frames):
        a = fk if k == 0 else 6 + (k * 7) % 9          # frames kept before the flash
        b = 8 + (k * 5) % 11                           # frames kept after it
        f0, f1 = fk - a, fk + b
        s, e = grid0 + Fraction(f0) / rate, grid0 + Fraction(f1) / rate
        n = f1 - f0
        out0 = Fraction(cur) / rate
        segs.append({"k": k, "piece": k, "src": [float(s), float(e)],
                     "out": [float(out0), float(Fraction(cur + n) / rate)], "frames": n,
                     "samples": int(round(Fraction(n * SR * den, num))),
                     "join_before": "start" if k == 0 else "piece",
                     "fade_in": 0.008, "fade_out": 0.008, "gain_db": 0.0})
        if k:
            joins.append({"k": k, "out": float(out0), "kind": "piece",
                          "before_src": segs[k - 1]["src"][1], "after_src": float(s)})
        t = grid0 + Fraction(fk) / rate
        words.append({"i": k, "raw": "word", "text": "word", "s_src": float(t),
                      "e_src": float(t + Fraction(1, 5)), "s_out": float(out0 + Fraction(a) / rate),
                      "e_out": float(out0 + Fraction(a) / rate + Fraction(1, 5)), "piece": k,
                      "seg": k, "sp": None, "filler": False, "hidden": False, "omitted": False,
                      "removed": False})
        expect.append(cur + a)
        cur += n
    plan = {"version": 1, "clip": "sync", "kind": "cut", "fps": [num, den],
            "grid_offset": float(grid0), "sample_rate": SR,
            "source": {"width": src["video"]["width"], "height": src["video"]["height"],
                       "duration": src["duration"], "hdr": "none"},
            "pieces": [], "segments": segs, "removals": [], "joins": joins, "words": words,
            "intent": list(range(N_PIECES)), "duration_out": float(Fraction(cur) / rate),
            "frames_out": cur, "hook_card": None, "framing": [], "lints": []}
    return plan, expect


def detect_flashes(path) -> list:
    """Output frame indices whose mean luma is near white."""
    res = env.ff(["-i", str(path), "-map", "0:v:0", "-vf", "scale=27:48:flags=area,format=gray",
                  "-fps_mode", "passthrough", "-f", "rawvideo", "-"])
    lum = np.frombuffer(res.stdout, dtype=np.uint8).reshape(-1, 48 * 27).mean(axis=1)
    return [int(i) for i in np.flatnonzero(lum > 150)]


def click_onset(audio: np.ndarray, lo: float, hi: float) -> float:
    """Onset (s) of the click inside [lo, hi): first sample above 30 % of the window peak."""
    i0, i1 = max(0, int(lo * SR)), min(audio.size, int(hi * SR))
    t_guess = (lo + hi) / 2
    seg = np.abs(audio[i0:i1])
    peak = float(seg.max()) if seg.size else 0.0
    floor = float(np.median(seg)) if seg.size else 0.0
    if peak <= 0 or peak < 8 * max(floor, 1e-6):
        raise AssertionError("no click near %.3f s (peak %.4f, floor %.5f)" % (t_guess, peak, floor))
    return (i0 + int(np.flatnonzero(seg > 0.3 * peak)[0])) / float(SR)


def measure_sync(tc: unittest.TestCase, out_path, rate: Fraction, meta: dict, expect=None):
    """Assert flash/click alignment in a render; returns (flash frames, A-V offsets in s)."""
    frame = 1.0 / float(rate)
    flashes = detect_flashes(out_path)
    tc.assertEqual(len(flashes), N_PIECES, "flash frames found: %s" % flashes)
    if expect is not None:
        for k, (got, want) in enumerate(zip(flashes, expect)):
            tc.assertLessEqual(abs(got - want), 1, "piece %d flash at frame %d, plan says %d"
                               % (k, got, want))
    # The PCM decode below ignores timestamps, so check the container timeline first: a hole in
    # the audio packet timestamps plays every later sample late in a real player (loudnorm's
    # end-of-stream flush once left an 80 ms hole that a decode-based check could not see).
    from hookcutter import qa
    streams = env.ffprobe_json(out_path, "-show_streams").get("streams") or []
    a = next(s for s in streams if s.get("codec_type") == "audio")
    v = next(s for s in streams if s.get("codec_type") == "video")
    holes, biggest = qa._audio_pts_gaps(Path(out_path), a)
    tc.assertEqual(holes, 0, "audio timestamp holes: %d, largest %.1f ms" % (holes, biggest * 1000))
    tc.assertLessEqual(abs(float(a["duration"]) - float(v["duration"])), frame,
                       "audio %.3f s vs video %.3f s" % (float(a["duration"]), float(v["duration"])))
    audio = fixtures.decode_mono(out_path)
    times = [float(Fraction(f) / rate) for f in flashes]
    offsets = []
    for i, tf in enumerate(times):
        # search only this piece's neighbourhood: half way to the neighbouring flashes, max 0.4 s
        lo = tf - min(0.4, (tf - times[i - 1]) / 2 if i else 0.4)
        hi = tf + min(0.4, (times[i + 1] - tf) / 2 if i + 1 < len(times) else 0.4)
        offsets.append(click_onset(audio, lo, hi) - tf)
    src_off = float(np.median([c - f for c, f in zip(meta["click_times"], meta["flash_times"])]))
    rel = [o - src_off for o in offsets]
    worst = max(abs(o) for o in rel)
    tc.assertLessEqual(worst, frame, "A-V offsets (ms) vs source: %s"
                       % [round(o * 1000, 1) for o in rel])
    spread = max(rel) - min(rel)
    tc.assertLessEqual(spread, SPREAD_MAX, "offsets accumulate (ms): %s"
                       % [round(o * 1000, 1) for o in rel])
    slope = np.polyfit(np.arange(len(rel)), np.array(rel), 1)[0]
    tc.assertLess(abs(slope) * len(rel), SPREAD_MAX, "drift slope %.3f ms/join" % (slope * 1000))
    return flashes, rel


class SyncCase(object):
    fps = "25"
    audio_delay = 0.0

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="hc-sync-"))
        rate = fixtures._fps(cls.fps)
        cls.rate = rate
        src = fixtures.make_av_fixture(cls.tmp, fps=cls.fps, dur=FIRST + N_PIECES * SPACING + 2,
                                       flashes=flash_times(rate), audio_delay=cls.audio_delay)
        cls.meta = fixtures.fixture_meta(src)
        cls.P = fixtures.make_project(cls.tmp, src)
        cls.proj = project.Project(cls.P)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_hand_plan_sync(self):
        plan, expect = hand_plan(self.proj.source(), self.rate, self.meta["flash_frames"])
        clip = {"captions": False, "framing": [], "kind": "cut",
                "_eff": {"styles": ["minimal"], "formats": ["vertical"]}}
        res = render.render_clip(self.proj, "sync", formats=["vertical"], styles=["minimal"],
                                 plan=plan, clip=clip, force=True)
        self.assertEqual(len(res), 1)
        out = Path(res[0]["path"])
        self.assertEqual(res[0]["frames"], plan["frames_out"])
        flashes, _ = measure_sync(self, out, self.rate, self.meta, expect)
        # caption timing source: every visible word's s_out lands on its flash frame
        for w, fidx in zip(plan["words"], flashes):
            self.assertLessEqual(abs(w["s_out"] - float(Fraction(fidx) / self.rate)),
                                 1.0 / float(self.rate) + 1e-6)


class TestSync25(SyncCase, unittest.TestCase):
    fps = "25"


class TestSyncNTSC(SyncCase, unittest.TestCase):
    fps = "30000/1001"


class TestSyncAudioDelay(SyncCase, unittest.TestCase):
    fps = "25"
    audio_delay = 0.2

    def test_audio_starts_late_in_source(self):
        self.assertGreater(self.proj.source()["audio"]["start"], 0.15)


class TestSyncGrouped(SyncCase, unittest.TestCase):
    """More segments than CONCAT_GROUP: stage B reads lossless group intermediates."""
    fps = "30000/1001"

    def setUp(self):
        self._group = render.CONCAT_GROUP
        render.CONCAT_GROUP = 7

    def tearDown(self):
        render.CONCAT_GROUP = self._group

    def test_hand_plan_sync(self):
        super().test_hand_plan_sync()
        log = env.read_json(self.proj.render_path("sync", "vertical", "minimal", ".log.json"))
        self.assertEqual(len(log["inputs"]), 5)
        self.assertTrue(all("grp_" in p for p in log["inputs"]))


class TestSyncThroughPlanner(unittest.TestCase):
    """Same check with the real planner and captions (minimal style) when both are present."""

    @classmethod
    def setUpClass(cls):
        try:
            from hookcutter import plan as plan_mod  # noqa: F401
            from hookcutter import captions  # noqa: F401
        except ImportError as exc:  # pragma: no cover
            raise unittest.SkipTest("planner or captions not available: %s" % exc)
        cls.tmp = Path(tempfile.mkdtemp(prefix="hc-sync-plan-"))
        cls.rate = Fraction(25)
        times = flash_times(cls.rate)
        src = fixtures.make_av_fixture(cls.tmp, fps="25", dur=FIRST + N_PIECES * SPACING + 2,
                                       flashes=times)
        cls.meta = fixtures.fixture_meta(src)
        words = [("word%d." % k, t, t + 0.20) for k, t in enumerate(cls.meta["flash_times"])]
        cls.P = fixtures.make_project(cls.tmp, src, words)
        clips = {"version": 1, "defaults": {"styles": ["minimal"], "formats": ["vertical"],
                                            "tighten": {"enabled": False}},
                 "clips": {"sync": {"title": "Sync", "pieces": [{"words": [k, k]}
                                                                 for k in range(N_PIECES)],
                                    "captions": True}}}
        env.write_json(Path(cls.P) / "clips.json", clips)
        cls.proj = project.Project(cls.P)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_planner_render_sync(self):
        from hookcutter import plan as plan_mod
        plan = plan_mod.build_plan(self.proj, "sync")
        res = render.render_clip(self.proj, "sync", formats=["vertical"], styles=["minimal"],
                                 plan=plan, force=True)
        out = Path(res[0]["path"])
        flashes, _ = measure_sync(self, out, self.rate, self.meta)
        cap = env.read_json(self.proj.render_path("sync", "vertical", "minimal", ".captions.json"))
        starts = sorted(t["s_out"] for p in cap.get("pages") or [] for t in p.get("tokens") or [])
        self.assertEqual(len(starts), N_PIECES, "caption tokens: %d" % len(starts))
        frame = 1.0 / float(self.rate)
        for s, fidx in zip(starts, flashes):
            self.assertLessEqual(abs(s - float(Fraction(fidx) / self.rate)), frame + 1e-6,
                                 "caption word at %.3f vs flash frame %d" % (s, fidx))


if __name__ == "__main__":
    unittest.main()
