"""Render loudness on hot, clipped sources (QA G7: -14 LUFS +/- 1, true peak at most -1 dBTP).

A source whose peaks already sit at 0 dBFS cannot be lifted to -14 LUFS by one linear gain
without pushing them past loudnorm's -1.5 dBTP, so ffmpeg drops to its dynamic mode, which lands
short. On a 75-minute 720p remote interview every clip fell to dynamic mode: one of twelve landed
at -15.3 LUFS and was held at export, the other eleven passed G7 by 0.1 to 0.8 LU. The renderer
now fits a pre-gain and a limiter for such a source and keeps the linear step; a source that
already normalises in linear mode keeps the exact filter graph it had, and a source that would
need more limiting than LIMIT_MAX_DB keeps it too (G7 still fails on it).
"""

from __future__ import annotations

import re
import shutil
import sys
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path
from unittest import mock

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "scripts"))

import fixtures  # noqa: E402
from hookcutter import env, project, qa, render  # noqa: E402

SR = 48000
RATE = Fraction(25)
PIECES = [(0.24, 5.84), (6.0, 11.6)]


def cut_plan(clip_id: str, pieces=PIECES) -> dict:
    segs, joins, cur = [], [], 0
    for k, (s, e) in enumerate(pieces):
        n = int(round((Fraction(e).limit_denominator(1000) - Fraction(s).limit_denominator(1000)) * RATE))
        segs.append({"k": k, "piece": k, "src": [s, e], "out": [float(cur / RATE), float((cur + n) / RATE)],
                     "frames": n, "samples": int(round(Fraction(n * SR) / RATE)),
                     "join_before": "start" if k == 0 else "piece", "fade_in": 0.008, "fade_out": 0.008,
                     "gain_db": 0.0})
        if k:
            joins.append({"k": k, "out": float(cur / RATE), "kind": "piece",
                          "before_src": pieces[k - 1][1], "after_src": s})
        cur += n
    return {"version": 1, "clip": clip_id, "kind": "cut", "fps": [25, 1], "grid_offset": 0.0,
            "sample_rate": SR, "pieces": [], "segments": segs, "removals": [], "joins": joins,
            "words": [], "intent": [], "frames_out": cur, "duration_out": cur / 25.0,
            "hook_card": None, "framing": [], "lints": []}


CLIP = {"kind": "cut", "captions": False, "framing": [],
        "_eff": {"styles": ["minimal"], "formats": ["vertical"]}}


def ebur128(path) -> tuple:
    """Integrated loudness and true peak, measured as G7 measures them."""
    rc, _, err = qa._run([env.ffmpeg(), "-hide_banner", "-nostdin", "-nostats", "-loglevel", "info",
                          "-i", str(path), "-map", "0:a:0", "-af", "ebur128=peak=true", "-f", "null", "-"])
    return qa.parse_ebur128(err)


def stage_b_graph(log: dict) -> str:
    args = log["cmds"]["stage_b"]
    return args[args.index("-filter_complex") + 1]


def old_loudnorm(meas: dict) -> str:
    """The stage B loudnorm step exactly as the renderer wrote it before limiting existed."""
    return ("loudnorm=I=-14.0:TP=-1.5:LRA=11.0:measured_I=%.2f:measured_TP=%.2f:measured_LRA=%.2f:"
            "measured_thresh=%.2f:offset=%.2f:linear=true"
            % (meas["input_i"], meas["input_tp"], meas["input_lra"], meas["input_thresh"],
               meas["target_offset"]))


class RenderCase(object):
    clip_id = "case"
    fixture = None

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="hc-loud-"))
        src = cls.make_source(cls.tmp)
        cls.P = fixtures.make_project(cls.tmp, src)
        cls.proj = project.Project(cls.P)
        res = render.render_clip(cls.proj, cls.clip_id, formats=["vertical"], styles=["minimal"],
                                 plan=cut_plan(cls.clip_id), clip=dict(CLIP))
        cls.out = Path(res[0]["path"])
        cls.log = env.read_json(cls.proj.render_path(cls.clip_id, "vertical", "minimal", ".log.json"))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)


class TestClippedSourceReachesTarget(RenderCase, unittest.TestCase):
    """Peaks at 0 dBFS, about -19 LUFS: the shape of the interview clip that failed G7."""
    clip_id = "hot"

    @staticmethod
    def make_source(tmp):
        return fixtures.make_peaky_speech_fixture(tmp)

    def test_source_cannot_normalise_linearly(self):
        m = self.log["loudnorm"]
        self.assertGreater(m["input_tp"], -0.5)
        self.assertGreater(m["input_tp"] + (-14.0 - m["input_i"]), -1.5)

    def test_g7_passes(self):
        g = qa.g7_loudness(self.out, "vertical")
        self.assertEqual(g["status"], "PASS", g["evidence"])

    def test_lands_on_target(self):
        i, tp = ebur128(self.out)
        self.assertLessEqual(abs(i - qa.TARGET_LUFS), 0.5, "integrated %.1f LUFS" % i)
        self.assertLessEqual(tp, qa.MAX_TRUE_PEAK, "true peak %.1f dBTP" % tp)

    def test_limiter_in_graph_and_log(self):
        lim = self.log["loudnorm"].get("limiter")
        self.assertIsInstance(lim, dict, self.log["loudnorm"])
        self.assertGreater(lim["gain_db"], 0.0)
        self.assertLessEqual(lim["reduction_db"], render.LIMIT_MAX_DB)
        self.assertTrue(render.loudnorm_linear(lim["measured"]))
        graph = stage_b_graph(self.log)
        self.assertIn("alimiter=", graph)
        self.assertIn("latency=true", graph)
        self.assertIn(old_loudnorm(lim["measured"]), graph)


class TestOrdinarySourceUnchanged(RenderCase, unittest.TestCase):
    """Tone bursts at -14 dB RMS (peaks near -11 dBFS) normalise in linear mode: same graph."""
    clip_id = "plain"

    @staticmethod
    def make_source(tmp):
        words = [("w%d" % k, 0.4 + k * 0.9, 0.4 + k * 0.9 + 0.6) for k in range(12)]
        return fixtures.make_speech_fixture(tmp / "plain", words, dur=12.0)

    def test_graph_is_the_old_one(self):
        m = self.log["loudnorm"]
        self.assertNotIn("limiter", m)
        self.assertNotIn("limiter_skipped", m)
        graph = stage_b_graph(self.log)
        self.assertNotIn("alimiter", graph)
        self.assertIn("[ac]%s,aresample=48000,asetpts=N/SR/TB," % old_loudnorm(m), graph)

    def test_g7_passes(self):
        g = qa.g7_loudness(self.out, "vertical")
        self.assertEqual(g["status"], "PASS", g["evidence"])


class TestTooPeakyStillFails(RenderCase, unittest.TestCase):
    """Peaks that would need more than LIMIT_MAX_DB of limiting: the old graph, and G7 still
    reports the clip (the gate is not loosened; such a clip needs an ear, not a harder limiter)."""
    clip_id = "spiky"

    @staticmethod
    def make_source(tmp):
        return fixtures.make_peaky_speech_fixture(tmp, body_db=-25.0)

    def test_old_graph_and_a_reason(self):
        m = self.log["loudnorm"]
        self.assertNotIn("limiter", m)
        self.assertIn("limiting", m.get("limiter_skipped") or "")
        graph = stage_b_graph(self.log)
        self.assertNotIn("alimiter", graph)
        self.assertIn("[ac]%s,aresample=48000,asetpts=N/SR/TB," % old_loudnorm(m), graph)

    def test_g7_still_fails(self):
        g = qa.g7_loudness(self.out, "vertical")
        self.assertEqual(g["status"], "FAIL", g["evidence"])


class TestNearTheCapReachesTarget(RenderCase, unittest.TestCase):
    """A source whose fit needs about LIMIT_MAX_DB of limiting: the secant step from the first
    two measurements overshoots the cap, and the fit must try the cap itself before it gives up
    (it used to give up here, so loudnorm fell back to dynamic mode and G7 failed)."""
    clip_id = "nearcap"

    @staticmethod
    def make_source(tmp):
        return fixtures.make_peaky_speech_fixture(tmp, body_db=-15.0, seed=9)

    def test_limiter_within_the_cap(self):
        lim = self.log["loudnorm"].get("limiter")
        self.assertIsInstance(lim, dict, self.log["loudnorm"])
        self.assertLessEqual(lim["reduction_db"], render.LIMIT_MAX_DB)
        self.assertTrue(render.loudnorm_linear(lim["measured"]))
        self.assertIn("alimiter=", stage_b_graph(self.log))

    def test_g7_passes(self):
        g = qa.g7_loudness(self.out, "vertical")
        self.assertEqual(g["status"], "PASS", g["evidence"])


class TestPastTheCapStillFails(RenderCase, unittest.TestCase):
    """A source that needs a little more than LIMIT_MAX_DB: even at the cap the limited audio
    stays too quiet for the linear step, so it keeps the old graph and G7 still fails it."""
    clip_id = "pastcap"

    @staticmethod
    def make_source(tmp):
        return fixtures.make_peaky_speech_fixture(tmp, seed=5)

    def test_old_graph_and_a_reason(self):
        m = self.log["loudnorm"]
        self.assertNotIn("limiter", m)
        why = m.get("limiter_skipped") or ""
        self.assertIn("limiting", why)
        self.assertRegex(why, r"reaches -1[0-9]\.[0-9]+ LUFS")
        self.assertIn("[ac]%s,aresample=48000,asetpts=N/SR/TB," % old_loudnorm(m), stage_b_graph(self.log))

    def test_g7_still_fails(self):
        g = qa.g7_loudness(self.out, "vertical")
        self.assertEqual(g["status"], "FAIL", g["evidence"])


class TestFitUsesTheWholeCap(unittest.TestCase):
    """_fit_limiter on a replayed gain curve: loudnorm's measurement after the limiter, against
    pre-gain, taken from ffmpeg on a synthetic track (make_peaky_speech_fixture, seed 11; peaks
    -2.0 dBTP at every gain). It never asks the limiter for more than LIMIT_MAX_DB, and it calls
    the peaks too much for it only after the gain at the cap has been measured and failed."""

    FIRST = {"input_i": -19.81, "input_tp": 0.04, "input_lra": 1.8, "input_thresh": -29.9,
             "target_offset": 0.0}
    CURVE = ([5.81, 9.5, 10.0, 10.67, 11.0, 11.5, 12.0, 12.46, 12.95],
             [-18.86, -16.48, -16.12, -15.61, -15.35, -14.98, -14.59, -14.23, -13.86])

    def fit(self, first=None, shift=0.0):
        gains = []

        def fake_pass(ins, prefix, chain=""):
            g = float(re.match(r"volume=(-?[0-9.]+)dB,", chain).group(1))
            gains.append(g)
            i = float(np.interp(g, self.CURVE[0], self.CURVE[1])) + shift
            return {"input_i": round(i, 2), "input_tp": -2.0, "input_lra": 2.6,
                    "input_thresh": round(i - 10.0, 2), "target_offset": 0.0}

        with mock.patch.object(render, "_loudnorm_pass", fake_pass):
            res = render._fit_limiter([], "", dict(first or self.FIRST))
        return res, gains

    def cap(self, first=None):
        return render.LIMIT_MAX_DB + render.LIMIT_CEILING - (first or self.FIRST)["input_tp"]

    def test_secant_past_the_cap_measures_the_cap(self):
        res, gains = self.fit()
        self.assertNotIn("skipped", res, res)
        self.assertLessEqual(res["reduction_db"], render.LIMIT_MAX_DB)
        self.assertLessEqual(max(gains), self.cap() + 1e-9, gains)
        self.assertAlmostEqual(res["gain_db"], self.cap(), places=2)
        self.assertTrue(render.loudnorm_linear(res["measured"]))

    def test_skips_only_after_the_cap_fails(self):
        res, gains = self.fit(shift=-0.85)   # at the cap: -14.71 LUFS, too quiet for linear
        self.assertIn("skipped", res, res)
        self.assertIn("limiting", res["skipped"])
        self.assertIn("reaches -14.71 LUFS", res["skipped"])
        self.assertLessEqual(max(gains), self.cap() + 1e-9, gains)
        self.assertAlmostEqual(gains[-1], self.cap(), places=2)
        self.assertEqual(sum(1 for g in gains if abs(g - self.cap()) < 0.005), 1, gains)

    def test_first_gain_past_the_cap_measures_the_cap(self):
        quiet = dict(self.FIRST, input_i=-30.0)   # 16 dB to the target, more than the cap allows
        res, gains = self.fit(first=quiet, shift=quiet["input_i"] - self.FIRST["input_i"])
        self.assertIn("limiting", res.get("skipped") or "", res)
        self.assertEqual(len(gains), 1, gains)
        self.assertAlmostEqual(gains[0], self.cap(quiet), places=2)


class TestLimiterKeepsTiming(unittest.TestCase):
    """The limited chain moves no sample: same length, no lag (alimiter's look-ahead is
    compensated), so captions and cuts stay where the plan put them."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="hc-loud-sync-"))
        src = fixtures.make_peaky_speech_fixture(cls.tmp)
        cls.wav = cls.tmp / "src.wav"
        env.ff(["-i", str(src), "-map", "0:a:0", "-c:a", "pcm_s24le", "-ar", str(SR), str(cls.wav)])

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_no_lag(self):
        meas = render.measure_loudness([self.wav], self.tmp / "loudnorm.json")
        self.assertIn("limiter", meas)
        chain = render.loudnorm_filter(meas)
        self.assertIn("alimiter=", chain)
        res = env.ff(["-i", str(self.wav), "-af", chain + ",aresample=%d" % SR, "-ac", "1", "-f", "f32le", "-"])
        y = np.frombuffer(res.stdout, dtype="<f4").astype(np.float64)
        x = fixtures.decode_mono(self.wav).astype(np.float64)
        self.assertEqual(y.size, x.size)
        a, b = int(2.0 * SR), int(6.0 * SR)
        best = max(range(-480, 481), key=lambda lag: float(np.dot(x[a:b], y[a + lag:b + lag])))
        self.assertEqual(best, 0, "limited audio is %d samples late" % best)


class TestLinearCheck(unittest.TestCase):
    """loudnorm_linear mirrors ffmpeg's own test for linear mode."""

    def test_cases(self):
        hot = {"input_i": -20.47, "input_tp": 0.04, "input_lra": 5.5, "input_thresh": -31.29}
        self.assertFalse(render.loudnorm_linear(hot))
        self.assertTrue(render.loudnorm_linear(dict(hot, input_tp=-8.0)))
        self.assertTrue(render.loudnorm_linear(dict(hot, input_i=-14.0, input_tp=-1.5)))
        self.assertFalse(render.loudnorm_linear(dict(hot, input_i=-14.0, input_tp=-1.49)))
        self.assertFalse(render.loudnorm_linear(dict(hot, input_tp=-8.0, input_lra=16.0)))
        self.assertTrue(render.loudnorm_linear(dict(hot, input_i=-6.0)))    # turned down


if __name__ == "__main__":
    unittest.main()
