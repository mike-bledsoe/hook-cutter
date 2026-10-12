"""render.py smoke: a tiny 3-segment clip through stage A + stage B in both formats and two
styles; ffprobe checks of the delivery spec, caching, sheets and the caption_only path."""

from __future__ import annotations

import argparse
import contextlib
import io
import shutil
import sys
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "scripts"))

import fixtures  # noqa: E402
from hookcutter import env, project, render  # noqa: E402

SR = 48000


def seg(k, s, e, cur, rate):
    n = int(round((Fraction(e).limit_denominator(1000) - Fraction(s).limit_denominator(1000)) * rate))
    out = {"k": k, "piece": k, "src": [s, e], "out": [float(cur / rate), float((cur + n) / rate)],
           "frames": n, "samples": int(round(Fraction(n * SR) / rate)),
           "join_before": "start" if k == 0 else "piece", "fade_in": 0.008, "fade_out": 0.008,
           "gain_db": -3.0 if k == 1 else 0.0}
    return out, cur + n


def smoke_plan():
    rate = Fraction(25)
    segs, cur = [], 0
    for k, (s, e) in enumerate([(0.6, 1.8), (2.4, 3.6), (4.8, 6.4)]):
        sg, cur = seg(k, s, e, cur, rate)
        segs.append(sg)
    marks = [{"at_src": 0.0, "mode": "fixed", "x": 0.3, "y": 0.5, "crop": 0.8},
             {"at_src": 2.4, "mode": "letterbox"},
             {"at_src": 5.2, "mode": "track", "x": 0.6, "crop": 1.0}]
    joins = [{"k": 1, "out": segs[1]["out"][0], "kind": "piece", "before_src": 1.8, "after_src": 2.4},
             {"k": 2, "out": segs[2]["out"][0], "kind": "dip", "before_src": 3.6, "after_src": 4.8}]
    return {"version": 1, "clip": "smoke", "kind": "cut", "fps": [25, 1], "grid_offset": 0.0,
            "sample_rate": SR, "pieces": [], "segments": segs, "removals": [], "joins": joins,
            "words": [], "intent": [], "frames_out": cur, "duration_out": cur / 25.0,
            "hook_card": None, "framing": marks, "lints": []}


def streams(path):
    d = env.ffprobe_json(path, "-count_packets", "-show_streams", "-show_format")
    out = {}
    for s in d.get("streams") or []:
        out[s["codec_type"]] = s
    return out, d.get("format") or {}


class TestRenderSmoke(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="hc-render-"))
        src = fixtures.make_av_fixture(cls.tmp, fps="25", dur=8, flashes=[1.0, 3.0, 5.6])
        cls.P = fixtures.make_project(cls.tmp, src)
        cls.proj = project.Project(cls.P)
        samples = [{"t_src": round(k * 0.25, 2), "faces": [[0.6, 0.4, 0.1, 0.18, 0.9]]} for k in range(33)]
        env.write_json(cls.P / "shots" / "smoke.json",
                       {"clip": "smoke", "range_src": [0, 8], "sample_fps": 4, "analysis_width": 640,
                        "shots": [{"start_src": 0.0, "end_src": 8.0, "kind": "single",
                                   "faces": [{"x": 0.6, "y": 0.4, "h": 0.18, "presence": 1.0}]}],
                        "samples": samples, "proposal": []})
        cls.plan = smoke_plan()
        cls.clip = {"kind": "cut", "captions": False, "framing": cls.plan["framing"],
                    "_eff": {"styles": ["clean"], "formats": ["vertical"]}}
        cls.res = render.render_clip(cls.proj, "smoke", formats=["vertical", "landscape"],
                                     styles=["clean", "minimal"], plan=cls.plan, clip=cls.clip)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_outputs_meet_delivery_spec(self):
        self.assertEqual(len(self.res), 4)
        n = self.plan["frames_out"]
        for r in self.res:
            st, fmt = streams(r["path"])
            v, a = st["video"], st["audio"]
            want = (1080, 1920) if r["format"] == "vertical" else (1920, 1080)
            self.assertEqual((v["width"], v["height"]), want)
            self.assertEqual(v["codec_name"], "h264")
            self.assertEqual(v["profile"], "High")
            self.assertEqual(v["pix_fmt"], "yuv420p")
            self.assertEqual(v["r_frame_rate"], "25/1")
            self.assertEqual((v.get("color_primaries"), v.get("color_transfer"), v.get("color_space"),
                              v.get("color_range")), ("bt709", "bt709", "bt709", "tv"))
            self.assertEqual(int(v["nb_read_packets"]), n)
            self.assertEqual(r["frames"], n)
            self.assertEqual((a["codec_name"], int(a["sample_rate"]), int(a["channels"])), ("aac", 48000, 2))
            vd, ad = float(v["duration"]), float(a["duration"])
            self.assertAlmostEqual(vd, n / 25.0, places=3)
            self.assertLessEqual(abs(vd - ad), 1 / 25.0)
            self.assertIn("hook-cutter", (fmt.get("tags") or {}).get("comment", ""))

    def test_styles_share_audio(self):
        for fmt in ("vertical", "landscape"):
            md5s = {env.read_json(self.proj.render_path("smoke", fmt, s, ".log.json"))["audio_md5"]
                    for s in ("clean", "minimal")}
            self.assertEqual(len(md5s), 1, md5s)

    def test_logs_sheets_and_segment_paths(self):
        log = env.read_json(self.proj.render_path("smoke", "vertical", "clean", ".log.json"))
        self.assertEqual([s["frames"] for s in log["segments"]], [s["frames"] for s in self.plan["segments"]])
        self.assertEqual([s["samples"] for s in log["segments"]], [s["samples"] for s in self.plan["segments"]])
        self.assertEqual(log["warnings"], [])
        self.assertEqual(len(log["dips"]), 1)
        modes = [[x[0] for x in subs] for subs in log["framing"]]
        self.assertEqual(modes, [["crop"], ["letterbox"], ["letterbox", "crop"]])
        for name in ("smoke__vertical.seams.jpg", "smoke__landscape.seams.jpg",
                     "smoke__vertical__clean.contact.jpg", "smoke__landscape__minimal.contact.jpg"):
            self.assertTrue((self.P / "sheets" / name).is_file(), name)
        paths = render.segment_paths(self.proj, "smoke", "vertical")
        self.assertEqual(len(paths), 3)
        self.assertTrue(all(p.is_file() for p in paths))
        cap = env.read_json(self.proj.render_path("smoke", "vertical", "clean", ".captions.json"))
        self.assertTrue(cap.get("disabled"))

    def test_rerender_is_cached(self):
        out = self.proj.render_path("smoke", "vertical", "clean")
        before = env.sha256_file(out)
        log0 = env.read_json(self.proj.render_path("smoke", "vertical", "clean", ".log.json"))
        render.render_clip(self.proj, "smoke", formats=["vertical"], styles=["clean"],
                           plan=self.plan, clip=self.clip)
        self.assertEqual(env.sha256_file(out), before)
        log1 = env.read_json(self.proj.render_path("smoke", "vertical", "clean", ".log.json"))
        self.assertEqual(log0["created"], log1["created"])
        # a changed framing mark re-renders stage A for the affected segment only
        plan2 = dict(self.plan, framing=[dict(self.plan["framing"][0], x=0.35)] + self.plan["framing"][1:])
        clip2 = dict(self.clip, framing=plan2["framing"])
        render.render_clip(self.proj, "smoke", formats=["vertical"], styles=["clean"], plan=plan2, clip=clip2)
        log2 = env.read_json(self.proj.render_path("smoke", "vertical", "clean", ".log.json"))
        self.assertEqual([s["cached"] for s in log2["segments"]], [False, True, True])

    def test_prune_keeps_referenced_work(self):
        wd = self.P / "work" / "smoke" / "landscape"
        wd.mkdir(parents=True, exist_ok=True)
        log = env.read_json(self.proj.render_path("smoke", "landscape", "clean", ".log.json"))
        present = [self.P / p for p in log["work_files"] if (self.P / p).exists()]
        junk = wd / "seg_00000000000000000000.mkv"
        junk.write_bytes(b"x")
        (wd / "seg_1.tmp.mkv").write_bytes(b"x")
        self.assertEqual(render.prune_work(self.proj, "smoke", "landscape"), 2)
        self.assertFalse(junk.exists())
        self.assertTrue(all(p.exists() for p in present))

    def test_up_to_date_survives_a_cleaned_work_dir(self):
        out = self.proj.render_path("smoke", "landscape", "minimal")
        before = env.sha256_file(out)
        shutil.rmtree(self.P / "work" / "smoke" / "landscape")
        render.render_clip(self.proj, "smoke", formats=["landscape"], styles=["minimal"],
                           plan=self.plan, clip=self.clip)
        self.assertEqual(env.sha256_file(out), before)
        self.assertFalse(list((self.P / "work" / "smoke" / "landscape").glob("seg_*.mkv")))

    def test_ff_escape_with_hostile_path(self):
        """A sendcmd file under a folder name full of filtergraph syntax still loads and fires."""
        d = self.tmp / "it's, a [dir]; x:y"
        d.mkdir(exist_ok=True)
        cmds = d / "cmds.txt"
        cmds.write_text("0.500000 crop@c0 x 40;\n", encoding="utf-8")
        graph = ("testsrc=s=64x64:r=10:d=1,sendcmd=f=%s,crop@c0=w=16:h=16:x=0:y=0,format=gray"
                 % render._ff_escape(str(cmds)))
        res = env.ff(["-f", "lavfi", "-i", graph, "-f", "rawvideo", "-"])
        frames = [res.stdout[i * 256:(i + 1) * 256] for i in range(len(res.stdout) // 256)]
        self.assertEqual(len(frames), 10)
        self.assertEqual(frames[0], frames[1])          # testsrc is static at this crop
        self.assertNotEqual(frames[4], frames[5])       # the crop moved at 0.5 s


class TestCaptionOnly(unittest.TestCase):
    def test_vertical_source_fills_frame(self):
        tmp = Path(tempfile.mkdtemp(prefix="hc-co-"))
        try:
            src = fixtures.make_av_fixture(tmp, fps="30", dur=6, size="360x640", flashes=[1.0])
            other = fixtures.make_av_fixture(tmp / "main", fps="25", dur=4)
            P = fixtures.make_project(tmp, other)
            proj = project.Project(P)
            plan = {"version": 1, "clip": "co", "kind": "caption_only", "fps": [30, 1],
                    "segments": [{"k": 0, "src": [0.0, 4.0], "out": [0.0, 4.0], "frames": 120,
                                  "samples": 192000, "fade_in": 0.008, "fade_out": 0.008, "gain_db": 0}],
                    "joins": [], "words": [], "frames_out": 120, "duration_out": 4.0, "framing": []}
            clip = {"kind": "caption_only", "source_file": str(src), "captions": False}
            res = render.render_clip(proj, "co", formats=["vertical"], styles=["clean"], plan=plan, clip=clip)
            st, _ = streams(res[0]["path"])
            self.assertEqual((st["video"]["width"], st["video"]["height"]), (1080, 1920))
            self.assertEqual(st["video"]["r_frame_rate"], "30/1")
            self.assertEqual(int(st["video"]["nb_read_packets"]), 120)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class TestCmdRender(unittest.TestCase):
    """cmd_render through the real planner (skipped when plan.py is not importable)."""

    def test_cmd_render_end_to_end(self):
        try:
            from hookcutter import plan as _plan  # noqa: F401
        except ImportError as exc:  # pragma: no cover
            self.skipTest("planner not available: %s" % exc)
        tmp = Path(tempfile.mkdtemp(prefix="hc-cmd-"))
        try:
            words = [("So", 1.0, 1.2), ("this", 1.25, 1.5), ("works.", 1.55, 2.0),
                     ("And", 3.5, 3.7), ("then", 3.75, 4.0), ("ends.", 4.05, 4.5)]
            src = fixtures.make_speech_fixture(tmp, words, dur=7.0)
            P = fixtures.make_project(tmp, src, words)
            env.write_json(P / "clips.json", {"version": 1, "defaults": {"styles": ["clean"]},
                                              "clips": {"c1": {"title": "One",
                                                               "pieces": [{"words": [0, 2]},
                                                                          {"words": [3, 5]}],
                                                               "captions": False}}})
            ns = argparse.Namespace(project=str(P), clips=["c1"], styles=None, formats=None, force=False)
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                self.assertEqual(render.cmd_render(ns), 0)
            self.assertIn("renders/c1__vertical__clean.mp4", buf.getvalue())
            self.assertIn("sheets/c1__vertical.seams.jpg", buf.getvalue())
            out = P / "renders" / "c1__vertical__clean.mp4"
            self.assertTrue(out.is_file())
            plan = env.read_json(P / "plans" / "c1.plan.json")
            st, _ = streams(out)
            self.assertEqual(int(st["video"]["nb_read_packets"]), plan["frames_out"])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
