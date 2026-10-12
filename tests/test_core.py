"""Core tests: env, project, probe, sheets, presets, ingest, doctor, cli, launcher."""

import io
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from fractions import Fraction
from pathlib import Path
from unittest import mock

import fixtures as F
from hookcutter import cli, doctor, env, ingest, presets, probe, sheets
from hookcutter.project import Project

SK = env.SKILL_ROOT
HC = SK / "scripts" / "hc"
TMP = None  # module-level scratch dir (fixtures are cached inside it)
FLASHES = [1.0, 2.0, 7.5]


def setUpModule():
    global TMP
    TMP = Path(tempfile.mkdtemp(prefix="hc-core-"))


def tearDownModule():
    if TMP and TMP.exists():
        shutil.rmtree(str(TMP), ignore_errors=True)


def av(**kw):
    kw.setdefault("dur", 10)
    kw.setdefault("flashes", FLASHES)
    return F.make_av_fixture(TMP / "fx", **kw)


def luma(img):
    from PIL import ImageStat
    return ImageStat.Stat(img.convert("L")).mean[0]


def raw_starts(path):
    """Stream start minus format start straight from ffprobe (how source.json defines start)."""
    d = env.ffprobe_json(path, "-show_format", "-show_streams")
    f0 = float(d["format"].get("start_time") or 0)
    return {s["codec_type"]: float(s["start_time"]) - f0 for s in d["streams"]}


def quiet(fn, *a, **kw):
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        rv = fn(*a, **kw)
    return rv, out.getvalue(), err.getvalue()


def fake_ffmpeg(dirpath, stderr_lines, rc=0):
    """A stand-in ffmpeg that echoes its args to stdout and prints given stderr lines."""
    p = Path(dirpath) / "fake-ffmpeg"
    body = "#!/bin/sh\necho \"$@\"\n"
    for ln in stderr_lines:
        body += "echo '%s' 1>&2\n" % ln.replace("'", "")
    body += "cat >/dev/null 2>&1 || true\nexit %d\n" % rc
    p.write_text(body)
    p.chmod(p.stat().st_mode | stat.S_IEXEC)
    return str(p)


# ======================================================================= env

class EnvTests(unittest.TestCase):

    def test_parse_and_format_time(self):
        self.assertEqual(env.parse_time(681.5), 681.5)
        self.assertEqual(env.parse_time("681.5"), 681.5)
        self.assertEqual(env.parse_time("11:21"), 681.0)
        self.assertAlmostEqual(env.parse_time("11:21.40"), 681.4)
        self.assertAlmostEqual(env.parse_time("1:02:03.4"), 3723.4)
        for bad in ("abc", "1:75", "", "1:2:3:4", True):
            with self.assertRaises(env.HCError):
                env.parse_time(bad)
        self.assertEqual(env.fmt_time(681.44), "11:21.4")
        self.assertEqual(env.fmt_time(3723.4, 0), "1:02:03")
        self.assertEqual(env.fmt_time(59.96, 1), "01:00.0")

    def test_write_json_atomic_unicode(self):
        p = TMP / "j" / "a.json"
        env.write_json(p, {"b": 1, "a": "café", "f": Fraction(30000, 1001), "p": Path("x")})
        text = p.read_text(encoding="utf-8")
        self.assertTrue(text.endswith("\n"))
        self.assertIn("café", text)
        self.assertEqual(env.read_json(p)["f"], [30000, 1001])
        self.assertEqual([x.name for x in p.parent.iterdir()], ["a.json"])
        with self.assertRaises(env.HCError):
            env.read_json(TMP / "j" / "missing.json")

    def test_hashes(self):
        self.assertEqual(env.stable_hash({"a": 1, "b": [1, 2]}), env.stable_hash({"b": [1, 2], "a": 1}))
        self.assertNotEqual(env.stable_hash({"a": 1}), env.stable_hash({"a": 2}))
        self.assertEqual(env.sha256_bytes(b"abc"),
                         "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
        big = TMP / "big.bin"
        big.write_bytes(os.urandom(9 * 1024 * 1024))
        fp1 = env.fingerprint(big)
        with open(big, "r+b") as fh:  # change a byte in the tail chunk
            fh.seek(-10, 2)
            fh.write(b"0123456789")
        self.assertNotEqual(fp1, env.fingerprint(big))
        small = TMP / "small.bin"
        small.write_bytes(b"x" * 100)
        self.assertEqual(len(env.fingerprint(small)), 64)
        self.assertEqual(env.sha256_file(small), env.sha256_bytes(b"x" * 100))

    def test_data_dir_override(self):
        with mock.patch.dict(os.environ, {"HOOKCUTTER_HOME": str(TMP / "home")}):
            self.assertEqual(env.data_dir(), TMP / "home")
            self.assertEqual(env.hf_home(), TMP / "home" / "models" / "hf")
            self.assertTrue(str(env.venv_python()).startswith(str(TMP / "home" / "venv")))

    def test_ff_real_run(self):
        r = env.ff(["-f", "lavfi", "-i", "anullsrc=r=48000:cl=mono", "-t", "0.1", "-f", "s16le", "-"])
        self.assertEqual(r.returncode, 0)
        self.assertEqual(len(r.stdout), 4800 * 2)
        self.assertEqual(r.warnings, [])

    def test_ff_prepends_flags_and_denies(self):
        d = TMP / "fake1"
        d.mkdir(exist_ok=True)
        deny = "[mp4 @ 0x1] Application provided invalid, non monotonically increasing dts to muxer"
        warn = "Past duration 0.999 too large"
        exe = fake_ffmpeg(d, [deny, warn])
        with mock.patch.dict(os.environ, {"HOOKCUTTER_FFMPEG": exe}):
            with self.assertRaises(env.HCError) as cm:
                env.ff(["-i", "x", "out.mp4"])
            self.assertEqual(cm.exception.code, 3)
            self.assertIn("non monotonically increasing dts", cm.exception.msg)
            r = env.ff(["-i", "x", "out.mp4"], allow=("non monotonically increasing dts",))
            self.assertIn("-hide_banner -nostdin -loglevel warning -y -i x out.mp4", r.text)
            self.assertTrue(any("Past duration" in w for w in r.warnings))
            r = env.ff(["-loglevel", "info", "-i", "x", "o"], allow=("application provided invalid",))
            self.assertEqual(r.text.count("-loglevel"), 1)
            log = d / "step.log"
            env.ff(["-i", "x"], allow=("dts",), log_to=log)
            self.assertIn("Past duration", log.read_text())

    def test_ff_nonzero_exit(self):
        d = TMP / "fake2"
        d.mkdir(exist_ok=True)
        exe = fake_ffmpeg(d, ["Something broke"], rc=1)
        with mock.patch.dict(os.environ, {"HOOKCUTTER_FFMPEG": exe}):
            with self.assertRaises(env.HCError) as cm:
                env.ff(["-i", "x"])
            self.assertEqual(cm.exception.code, 3)
            self.assertIn("Something broke", cm.exception.msg)
            self.assertEqual(env.ff(["-i", "x"], check=False).returncode, 1)

    def test_ff_popen_pipe_and_wait(self):
        log = TMP / "pipe.log"
        proc = env.ff_popen(["-f", "rawvideo", "-pix_fmt", "gray", "-s", "16x16", "-framerate", "25",
                             "-i", "pipe:0", "-f", "null", "-"], log)
        for _ in range(10):
            proc.stdin.write(bytes(256))
        r = env.ff_wait(proc, log)
        self.assertEqual(r.returncode, 0)
        d = TMP / "fake3"
        d.mkdir(exist_ok=True)
        exe = fake_ffmpeg(d, ["Non-monotonous DTS in output stream 0:1"])
        with mock.patch.dict(os.environ, {"HOOKCUTTER_FFMPEG": exe}):
            proc = env.ff_popen(["-i", "pipe:0", "x.mp4"], TMP / "pipe2.log")
            with self.assertRaises(env.HCError):
                env.ff_wait(proc, TMP / "pipe2.log")

    def test_ffprobe_json_and_missing_tool(self):
        p = av()
        d = env.ffprobe_json(p, "-show_format")
        self.assertIn("format", d)
        with mock.patch.dict(os.environ, {"HOOKCUTTER_FFPROBE": str(TMP / "nope")}):
            with self.assertRaises(env.HCError) as cm:
                env.ffprobe_json(p)
            self.assertEqual(cm.exception.code, 3)


# ======================================================================= probe

class ProbeTests(unittest.TestCase):

    def test_25fps(self):
        p = av()
        s = probe.probe_source(p)
        v, a = s["video"], s["audio"]
        self.assertEqual(v["fps"], [25, 1])
        self.assertFalse(v["vfr"])
        self.assertEqual((v["width"], v["height"]), (640, 360))
        self.assertEqual(v["frames"], 250)
        self.assertEqual(v["start"], 0.0)
        self.assertEqual(a["start"], 0.0)
        self.assertEqual((a["sample_rate"], a["channels"]), (48000, 2))
        self.assertTrue(a["dup_mono"])
        self.assertFalse(a["fold_mono"])
        self.assertEqual(v["hdr"], "none")
        self.assertAlmostEqual(s["duration"], 10.0, delta=0.05)
        self.assertEqual(len(s["fingerprint"]), 64)
        self.assertTrue(any("upscales 5.3x" in w for w in s["warnings"]))
        self.assertTrue(any("only 10 s" in w for w in s["warnings"]))
        for w in s["warnings"]:
            self.assertNotIn(chr(0x2014), w)  # no em-dash in user-facing text

    def test_ntsc_rate(self):
        s = probe.probe_source(av(fps="30000/1001"))
        self.assertEqual(s["video"]["fps"], [30000, 1001])
        self.assertFalse(s["video"]["vfr"])
        self.assertEqual(s["video"]["frames"], 300)  # frames with pts < 10 s

    def test_audio_delay_start_is_relative_to_format_start(self):
        p = av(audio_delay=0.2)
        s = probe.probe_source(p)
        raw = raw_starts(p)
        self.assertAlmostEqual(s["video"]["start"], raw["video"], places=6)
        self.assertAlmostEqual(s["audio"]["start"], raw["audio"], places=6)
        self.assertEqual(s["video"]["start"], 0.0)
        # AAC priming: the stream starts one 1024-sample frame before the delayed content.
        self.assertGreater(s["audio"]["start"], 0.15)
        self.assertLessEqual(s["audio"]["start"], 0.2)
        self.assertTrue(any("audio stream starts" in w for w in s["warnings"]))
        # Semantics: decoded sample 0 sits at audio.start, so clicks land on the flashes.
        onsets = F.click_onsets(F.decode_mono(p))
        meta = F.fixture_meta(p)
        self.assertEqual(len(onsets), len(meta["flash_times"]))
        for t_flash, t_on in zip(meta["flash_times"], onsets):
            self.assertAlmostEqual(s["audio"]["start"] + t_on, t_flash, delta=0.0015)

    def test_video_delay(self):
        p = av(audio_delay=-0.3)
        s = probe.probe_source(p)
        self.assertAlmostEqual(s["video"]["start"], 0.3, delta=0.001)
        self.assertEqual(s["audio"]["start"], 0.0)
        t = F.fixture_meta(p)["flash_times"][0]
        self.assertAlmostEqual(t, 1.3, delta=0.001)
        self.assertGreater(luma(sheets.frame_at(p, t, width=64)), 200)

    def test_hdr_tagged(self):
        s = probe.probe_source(av(hdr=True, dur=4, flashes=[]))
        self.assertEqual(s["video"]["hdr"], "hlg")
        self.assertEqual(s["video"]["color_transfer"], "arib-std-b67")
        self.assertTrue(any("HDR (HLG)" in w for w in s["warnings"]))

    def test_rotation_upright(self):
        src = av()
        out = TMP / "rot90.mp4"
        if not out.exists():
            try:
                env.ff(["-display_rotation", "90", "-i", str(src), "-c", "copy", str(out)])
            except env.HCError:
                self.skipTest("ffmpeg lacks -display_rotation")
        s = probe.probe_source(out)
        self.assertIn(s["video"]["rotation"], (90, -90))
        self.assertEqual((s["video"]["width"], s["video"]["height"]), (360, 640))
        self.assertEqual((s["video"]["coded_width"], s["video"]["coded_height"]), (640, 360))
        self.assertTrue(any("rotation" in w for w in s["warnings"]))

    def test_dead_channel_folds(self):
        out = TMP / "deadch.mp4"
        if not out.exists():
            env.ff(["-f", "lavfi", "-i", "testsrc=size=320x240:rate=25:duration=3",
                    "-f", "lavfi", "-i", "sine=frequency=300:sample_rate=48000:duration=3",
                    "-f", "lavfi", "-i", "anullsrc=r=48000:cl=mono",
                    "-filter_complex", "[1:a][2:a]amerge=inputs=2[a]", "-map", "0:v", "-map", "[a]",
                    "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", "-shortest", str(out)])
        s = probe.probe_source(out)
        self.assertTrue(s["audio"]["fold_mono"])
        self.assertFalse(s["audio"]["dup_mono"])
        self.assertLess(s["audio"]["channel_db"][1], -100)
        self.assertTrue(any("folded to mono" in w for w in s["warnings"]))

    def test_vfr_detected(self):
        out = TMP / "vfr.mp4"
        if not out.exists():
            env.ff(["-f", "lavfi", "-i", "testsrc=size=320x240:rate=25:duration=6",
                    "-vf", "setpts='if(gte(N,50),(N+40)/25/TB,N/25/TB)'", "-fps_mode", "passthrough",
                    "-c:v", "libx264", "-preset", "ultrafast", "-an", str(out)])
        info = probe.probe_media(out)
        self.assertTrue(info["video"]["vfr"])

    def test_no_audio_fails(self):
        out = TMP / "noaudio.mp4"
        if not out.exists():
            env.ff(["-f", "lavfi", "-i", "testsrc=size=160x120:rate=25:duration=1",
                    "-c:v", "libx264", "-preset", "ultrafast", str(out)])
        with self.assertRaises(env.HCError) as cm:
            probe.probe_source(out)
        self.assertIn("no audio", cm.exception.msg)


# ======================================================================= sheets

class SheetsTests(unittest.TestCase):

    def _check_exact(self, p):
        meta = F.fixture_meta(p)
        fd = meta["fps"][1] / float(meta["fps"][0])
        for t in meta["flash_times"]:
            self.assertLess(luma(sheets.frame_at(p, t - fd, width=64)), 80, t)
            self.assertLess(luma(sheets.frame_at(p, t - 0.001, width=64)), 80, t)
            self.assertGreater(luma(sheets.frame_at(p, t, width=64)), 200, t)
            self.assertGreater(luma(sheets.frame_at(p, t + 0.5 * fd, width=64)), 200, t)
            self.assertLess(luma(sheets.frame_at(p, t + fd, width=64)), 80, t)

    def test_frame_at_exact_25(self):
        self._check_exact(av())

    def test_frame_at_exact_ntsc(self):
        self._check_exact(av(fps="30000/1001"))

    def test_frame_at_width_and_bounds(self):
        p = av()
        im = sheets.frame_at(p, 3.0, width=200)
        self.assertEqual(im.size, (200, 112))
        self.assertEqual(sheets.frame_at(p, 9.99).size, (640, 360))
        with self.assertRaises(env.HCError):
            sheets.frame_at(p, 99.0)

    def test_contact_sheet_and_seam_pair(self):
        from PIL import Image
        imgs = [(Image.new("RGB", (100, 50), (i * 40, 0, 0)), "f%d\nline2" % i) for i in range(5)]
        out = sheets.contact_sheet(imgs, 2, TMP / "sheets" / "cs.jpg", 100, title="test sheet")
        with Image.open(out) as im:
            self.assertEqual(im.width, 2 * 100 + 3 * 6)
            self.assertGreater(im.height, 3 * 50)
        pair = sheets.seam_pair(Image.new("RGB", (90, 160)), Image.new("RGB", (90, 160), (255, 255, 255)),
                                "before 00:05.28", "after 00:05.32", title="join 1  tighten")
        self.assertGreater(pair.width, 180)
        self.assertGreater(pair.height, 160)
        self.assertEqual(pair.getpixel((92, 80))[:2], sheets.SEAM[:2])
        with self.assertRaises(env.HCError):
            sheets.contact_sheet([], 2, TMP / "x.jpg", 100)

    def test_peek(self):
        P = F.make_project(TMP / "peekproj", av())
        out, stdout, _ = quiet(sheets.cmd_peek, cli.build_parser().parse_args(
            ["peek", str(P), "0:01", "0:08", "--n", "6"]))
        self.assertEqual(out, 0)
        path = Path(stdout.strip())
        self.assertTrue(path.exists())
        self.assertEqual(path.parent, P / "sheets")
        self.assertTrue(path.name.startswith("peek_1.0-8.0"))
        with self.assertRaises(env.HCError):
            sheets.peek(P, "0:08", "0:01")
        rp = sheets.peek(P, 0, 2, n=4, file=str(av(fps="30000/1001")))
        self.assertIn("peek_av_", rp.name)


# ======================================================================= project

class ProjectTests(unittest.TestCase):

    def test_paths(self):
        P = Project(TMP / "projpaths")
        rp = P.render_path("c01", "vertical", "clean")
        self.assertEqual(rp, P.root / "renders" / "c01__vertical__clean.mp4")
        self.assertTrue(rp.parent.is_dir())
        self.assertEqual(P.render_path("c01", "vertical", "clean", "srt").name, "c01__vertical__clean.srt")
        self.assertEqual(P.cover_path("c01", "pop", "grid").name, "c01__pop.grid.jpg")
        self.assertEqual(P.cover_path("c01", "pop").name, "c01__pop.jpg")
        self.assertEqual(P.cover_path("c01", "pop", "json").name, "c01__pop.json")
        with self.assertRaises(env.HCError):
            P.cover_path("c01", "pop", "huge")
        self.assertEqual(P.plan_path("c01"), P.root / "plans" / "c01.plan.json")
        self.assertEqual(P.shots_path("c01"), P.root / "shots" / "c01.json")
        self.assertTrue(P.work_dir("c01", "vertical").is_dir())
        self.assertEqual(P.sheet_path("survey").name, "survey.jpg")
        self.assertEqual(P.qa_path("c01", "md"), P.root / "qa" / "c01.md")
        self.assertEqual(P.qa_path("summary", ".md").name, "summary.md")
        self.assertTrue(P.cache_path("energy-x.npy").parent.is_dir())
        self.assertTrue(P.export_dir().is_dir())

    def test_source_and_transcripts(self):
        P = Project(F.make_project(TMP / "projsrc", av(), [("Hello", 1.0, 1.4), ("world.", 1.5, 1.9)]))
        self.assertEqual(P.source_path().name, "source.mp4")
        self.assertEqual(P.source()["video"]["fps"], [25, 1])
        self.assertEqual(len(P.transcript()["words"]), 2)
        env.write_json(P.clips_path(), {"version": 1, "clips": {
            "c02-b": {"kind": "caption_only", "source_file": "x.mp4"}, "c01-a": {"pieces": []}}})
        self.assertEqual(P.clip_ids(), ["c02-b", "c01-a"])
        self.assertEqual(P.transcript_path("c02-b"), P.root / "transcripts" / "c02-b.json")
        self.assertEqual(P.transcript_path("c01-a"), P.root / "transcript.json")
        with self.assertRaises(env.HCError):
            P.transcript("c02-b")
        env.write_json(P.root / "transcripts" / "c02-b.json", {"words": [1]})
        self.assertEqual(P.transcript("c02-b"), {"words": [1]})

    def test_broken_link_and_missing(self):
        with self.assertRaises(env.HCError):
            Project(TMP / "nothing-here").source_path()
        d = TMP / "broken"
        d.mkdir(exist_ok=True)
        link = d / "source.mp4"
        if not link.is_symlink():
            os.symlink(str(TMP / "gone.mp4"), str(link))
        with self.assertRaises(env.HCError) as cm:
            Project(d).source_path()
        self.assertIn("broken link", cm.exception.msg)


# ======================================================================= presets

class PresetsTests(unittest.TestCase):

    def setUp(self):
        self.home = Path(tempfile.mkdtemp(dir=str(TMP), prefix="home-"))
        self.proj = Path(tempfile.mkdtemp(dir=str(TMP), prefix="presetproj-"))
        self.patch = mock.patch.dict(os.environ, {"HOOKCUTTER_HOME": str(self.home)})
        self.patch.start()
        (self.home / "presets").mkdir(parents=True)
        (self.proj / "presets").mkdir()

    def tearDown(self):
        self.patch.stop()

    def write(self, where, name, data):
        env.write_json(where / "presets" / ("%s.json" % name), data)

    def test_extends_deep_merge(self):
        self.write(self.home, "zz-base", {"name": "zz-base", "label": "Base", "caption": {
            "size": 78, "color": "#FFFFFF", "layout": {"max_words": 4, "y": 1250},
            "shadow": [{"blur": 14}, {"blur": 3}]}, "safezone": "organic"})
        self.write(self.proj, "zz-child", {"extends": "zz-base", "caption": {
            "size": 60, "layout": {"y": 1300}, "shadow": [{"blur": 1}]}})
        p = presets.load("zz-child", Project(self.proj))
        self.assertEqual(p["name"], "zz-child")
        self.assertEqual(p["label"], "Zz Child")
        self.assertEqual(p["caption"]["size"], 60)
        self.assertEqual(p["caption"]["color"], "#FFFFFF")
        self.assertEqual(p["caption"]["layout"], {"max_words": 4, "y": 1300})
        self.assertEqual(p["caption"]["shadow"], [{"blur": 1}])
        self.assertEqual(p["_chain"], ["zz-child", "zz-base"])
        self.assertEqual(p["extends"], "zz-base")
        # the parent alone is unchanged
        self.assertEqual(presets.load("zz-base")["caption"]["size"], 78)

    def test_same_name_extends_next_in_search_path(self):
        self.write(self.home, "zz-house", {"name": "zz-house", "caption": {"size": 78, "font": "A.ttf"}})
        self.write(self.proj, "zz-house", {"extends": "zz-house", "caption": {"size": 90}})
        p = presets.load("zz-house", self.proj)
        self.assertEqual(p["caption"], {"size": 90, "font": "A.ttf"})
        self.assertEqual(p["_chain"], ["zz-house", "zz-house"])

    def test_cycle_and_missing(self):
        self.write(self.proj, "zz-a", {"extends": "zz-b"})
        self.write(self.proj, "zz-b", {"extends": "zz-a"})
        with self.assertRaises(env.HCError) as cm:
            presets.load("zz-a", self.proj)
        self.assertIn("cycle", cm.exception.msg)
        with self.assertRaises(env.HCError):
            presets.load("zz-none", self.proj)
        self.write(self.proj, "zz-orphan", {"extends": "zz-nowhere"})
        with self.assertRaises(env.HCError):
            presets.load("zz-orphan", self.proj)
        self.assertIn("zz-a", presets.available(self.proj))

    def test_for_format(self):
        p = {"name": "x", "caption": {"size": 78, "layout": {"y": 1250, "max_lines": 1}},
             "landscape": {"caption": {"size": 54, "layout": {"y": 930}}}, "safezone": "organic"}
        v = presets.for_format(p, "vertical")
        self.assertNotIn("landscape", v)
        self.assertEqual(v["caption"]["size"], 78)
        ls = presets.for_format(p, "landscape")
        self.assertEqual(ls["caption"]["size"], 54)
        self.assertEqual(ls["caption"]["layout"], {"y": 930, "max_lines": 1})
        self.assertEqual(ls["safezone"], "landscape")
        self.assertEqual(p["caption"]["size"], 78)  # input untouched
        with self.assertRaises(env.HCError):
            presets.for_format(p, "square")

    def test_safezone(self):
        env.write_json(self.home / "presets" / "safezones.json",
                       {"organic": {"top": 240, "bottom": 420, "left": 90, "right": 150},
                        "zz-tall": {"top": 1, "bottom": 2, "left": 3, "right": 4}})
        env.write_json(self.proj / "presets" / "safezones.json", {"organic": {"bottom": 500}})
        z = presets.safezone("organic", self.proj)
        self.assertEqual(z, {"top": 240, "bottom": 500, "left": 90, "right": 150})
        self.assertEqual(presets.safezone("zz-tall", self.proj)["right"], 4)
        self.assertEqual(presets.safezone({"top": 5}), {"top": 5, "bottom": 0, "left": 0, "right": 0})
        self.assertEqual(presets.safe_rect(z, 1080, 1920), [90, 240, 840, 1180])
        with self.assertRaises(env.HCError):
            presets.safezone("zz-unknown", self.proj)


# ======================================================================= ingest

class IngestTests(unittest.TestCase):

    def test_local_ingest_and_survey(self):
        P = TMP / "ingproj"
        src = av()
        meta, out, _ = quiet(ingest.ingest, P, str(src))
        self.assertTrue((P / "source.mp4").is_symlink())
        self.assertEqual(env.read_json(P / "source.json")["fingerprint"], meta["fingerprint"])
        self.assertEqual(meta["ingest"]["method"], "symlink")
        self.assertEqual(meta["original"], str(src.resolve()))
        survey = P / "sheets" / "survey.jpg"
        self.assertTrue(survey.exists())
        from PIL import Image
        with Image.open(survey) as im:
            self.assertEqual(im.width, 4 * 480 + 5 * 6)
        self.assertIn("warning: source is 640x360", out)
        self.assertIn("next: hc transcribe", out)
        # same source again: refresh, no error
        quiet(ingest.ingest, P, str(src))
        other = av(fps="30000/1001")
        with self.assertRaises(env.HCError):
            quiet(ingest.ingest, P, str(other))
        meta2, _, _ = quiet(ingest.ingest, P, str(other), force=True, copy=True)
        self.assertFalse((P / "source.mp4").is_symlink())
        self.assertEqual(meta2["video"]["fps"], [30000, 1001])
        self.assertEqual(meta2["ingest"]["method"], "copy")

    def test_section_rules(self):
        self.assertEqual(ingest.parse_section("10:00-25:30"), (600.0, 1530.0))
        self.assertEqual(ingest.parse_section("600-inf"), (600.0, None))
        for bad in ("abc", "20-10", "5"):
            with self.assertRaises(env.HCError):
                ingest.parse_section(bad)
        with self.assertRaises(env.HCError):
            quiet(ingest.ingest, TMP / "ingproj2", str(av()), section="0-5")
        self.assertTrue(ingest.is_url("https://youtu.be/x"))
        self.assertFalse(ingest.is_url("/a/b.mp4"))

    def test_signals_shift(self):
        info = {"duration": 100, "heatmap": [{"start_time": 0, "end_time": 10, "value": 0.2},
                                             {"start_time": 50, "end_time": 60, "value": 1.0},
                                             {"start_time": 95, "end_time": 100, "value": 0.5}],
                "chapters": [{"start_time": 40, "end_time": 70, "title": "Middle"}]}
        s = ingest._signals(info, 45.0, 30.0, True)
        self.assertEqual(s["heatmap"], [{"start_src": 5.0, "end_src": 15.0, "value": 1.0}])
        self.assertEqual(s["chapters"], [{"start_src": 0.0, "end_src": 25.0, "title": "Middle"}])
        self.assertEqual(s["section_offset"], 45.0)
        self.assertFalse(s["section_offset_exact"])

    def test_url_section_offset_and_modes(self):
        src = av(audio_delay=0.2)  # a late-starting audio stream, like a stream-copied section

        def fake_download(proj, url, sec):
            dest = proj.root / "source.mp4"
            shutil.copy2(str(src), str(dest))
            return dest, {"id": "x", "duration": 100, "dynamic_range": "SDR",
                          "heatmap": [{"start_time": 40, "end_time": 50, "value": 0.7}],
                          "chapters": [{"start_time": 0, "end_time": 100, "title": "All"}]}
        P = TMP / "urlsec"
        with mock.patch.object(ingest, "_download", side_effect=fake_download):
            meta, out, _ = quiet(ingest.ingest, P, "https://example.com/watch?v=x", section="45-60")
        sig = env.read_json(P / "signals.json")
        pre = meta["audio"]["start"]
        self.assertAlmostEqual(sig["section_offset"], round(45 - pre, 3), places=3)
        self.assertFalse(sig["section_offset_exact"])
        self.assertEqual(sig["heatmap"][0]["start_src"], 0.0)
        self.assertAlmostEqual(sig["heatmap"][0]["end_src"], round(50 - (45 - pre), 3), places=3)
        self.assertEqual(meta["ingest"]["section"], [45.0, 60.0])
        self.assertEqual(meta["url_info"]["id"], "x")
        self.assertIn("keyframe pre-roll", out)
        for f in ("source.json", "signals.json", "sheets/survey.jpg"):
            self.assertEqual(stat.S_IMODE(os.stat(str(P / f)).st_mode), env.file_mode(), f)

    def test_url_without_ytdlp(self):
        with mock.patch.object(ingest, "_ytdlp_available", return_value=False):
            with self.assertRaises(env.HCError) as cm:
                quiet(ingest.ingest, TMP / "urlproj", "https://example.com/v.mp4")
        self.assertEqual(cm.exception.code, 3)
        self.assertIn("--with url", cm.exception.msg)


# ======================================================================= doctor

class DoctorTests(unittest.TestCase):

    def test_parse_ffmpeg_version(self):
        self.assertEqual(doctor.parse_ffmpeg_version("ffmpeg version 8.1.2 Copyright"), (8, 1))
        self.assertEqual(doctor.parse_ffmpeg_version("ffmpeg version n6.1.1 Copyright"), (6, 1))
        self.assertEqual(doctor.parse_ffmpeg_version("ffmpeg version 6.1.1-3ubuntu5 Copyright"), (6, 1))
        self.assertEqual(doctor.parse_ffmpeg_version("ffmpeg version 7 Copyright"), (7, 0))
        self.assertIsNone(doctor.parse_ffmpeg_version("ffmpeg version N-112345-gabcdef Copyright"))

    def test_requirements_pins(self):
        core = doctor.parse_requirements(env.SCRIPTS_DIR / "requirements.txt")
        self.assertIn(("numpy", "==", "2.5.3"), core)
        self.assertIn(("faster-whisper", "==", "1.2.1"), core)
        self.assertEqual(len(core), 6)
        self.assertEqual(doctor.parse_requirements(env.SCRIPTS_DIR / "requirements-mlx.txt"),
                         [("mlx-whisper", "==", "0.4.3")])
        self.assertEqual(doctor.parse_requirements(env.SCRIPTS_DIR / "requirements-url.txt"),
                         [("yt-dlp", ">=", "2025.1.1")])

    def test_assets_manifest_matches(self):
        checks = doctor.check_assets()
        self.assertEqual(checks[0].status, doctor.PASS, checks[0].detail)
        manifest = json.loads((SK / "fonts" / "manifest.json").read_text())
        self.assertEqual(len(manifest), 4)

    def test_install_commands(self):
        with mock.patch.dict(os.environ, {"HOOKCUTTER_HOME": str(TMP / "insthome")}):
            cmds = doctor.install_commands(["url", "whispercpp"], uv="/x/uv", venv_exists=False)
            self.assertEqual(cmds[0][:4], ["/x/uv", "venv", "--python", "3.12"])
            joined = " ".join(cmds[1])
            self.assertIn("requirements.txt", joined)
            self.assertIn("requirements-url.txt", joined)
            self.assertNotIn("mlx", joined)
            cmds = doctor.install_commands([], uv="/x/uv", venv_exists=True)
            self.assertEqual(len(cmds), 1)

    def test_extras_and_ffmpeg_checks(self):
        self.assertEqual(doctor._extras("mlx, url,yt-dlp"), ["mlx", "url"])
        with self.assertRaises(env.HCError):
            doctor._extras("gpu")
        checks = {c.name: c for c in doctor.check_ffmpeg()}
        self.assertEqual(checks["ffmpeg filters"].status, doctor.PASS, checks["ffmpeg filters"].detail)
        self.assertEqual(checks["ffmpeg encoders"].status, doctor.PASS)
        self.assertEqual(checks["encode smoke"].status, doctor.PASS)
        self.assertIn(checks["filter zscale"].status, (doctor.PASS, doctor.WARN))


# ======================================================================= cli + launcher

class CliTests(unittest.TestCase):

    def test_arg_names(self):
        ps = cli.build_parser()
        a = ps.parse_args(["qa", "P", "c1", "c2", "--no-asr", "--require-coverage", "--styles", "a, b"])
        self.assertEqual((a.project, a.clips, a.no_asr, a.require_coverage, a.styles),
                         ("P", ["c1", "c2"], True, True, ["a", "b"]))
        a = ps.parse_args(["transcribe", "P", "--speakers-vtt", "s.vtt", "--file", "v.mp4", "--clip", "x"])
        self.assertEqual((a.speakers_vtt, a.file, a.clip, a.backend, a.language), ("s.vtt", "v.mp4", "x", "auto", "auto"))
        a = ps.parse_args(["shots", "P", "--range", "1:00", "2:00"])
        self.assertEqual((a.clips, a.range), ([], ["1:00", "2:00"]))
        a = ps.parse_args(["render", "P", "--formats", "vertical,landscape"])
        self.assertEqual((a.formats, a.styles, a.force), (["vertical", "landscape"], None, False))
        a = ps.parse_args(["words", "P", "10", "20"])
        self.assertEqual((a.a, a.b), (10, 20))
        a = ps.parse_args(["find", "P", "look stupid", "--near", "11:20"])
        self.assertEqual((a.phrase, a.near, a.max), ("look stupid", "11:20", 10))
        a = ps.parse_args(["doctor", "--with", "mlx,url", "--install"])
        self.assertEqual((a.extras, a.install), (["mlx", "url"], True))
        a = ps.parse_args(["cover", "P", "c1", "--at", "3.5"])
        self.assertEqual(a.at, 3.5)
        a = ps.parse_args(["all", "P", "--no-asr"])
        self.assertTrue(a.no_asr)
        for cmd in cli.COMMANDS:
            self.assertIn(cmd, ps._subparsers._group_actions[0].choices)

    def test_missing_module_is_tool_error(self):
        with mock.patch.dict(cli.COMMANDS, {"zz": ("zz_not_a_module", "cmd_zz")}):
            with self.assertRaises(env.HCError) as cm:
                cli._resolve("zz")
        self.assertEqual(cm.exception.code, 3)
        rc, _, err = quiet(cli.main, ["status", str(TMP / "does-not-exist")])
        self.assertEqual(rc, 1)
        self.assertIn("error: project folder not found", err)

    def test_status(self):
        P = F.make_project(TMP / "statusproj", av(), [("Hello", 1.0, 1.4), ("world.", 1.5, 1.9)])
        rc, out, _ = quiet(cli.main, ["status", str(P)])
        self.assertEqual(rc, 0)
        self.assertIn("source      source.mp4", out)
        self.assertIn("transcript  2 words", out)
        self.assertIn("clips.json  missing", out)
        env.write_json(P / "clips.json", {"version": 1, "clips": {"c01-a": {"pieces": [{"words": [0, 1]}]},
                                                                  "c02-b": {"pieces": [{"words": [1, 1]}]}}})
        env.write_json(P / "qa" / "c01-a.json", {"gates": [{"id": "G1", "status": "PASS"},
                                                           {"id": "G7", "status": "FAIL"}]})
        rc, out, _ = quiet(cli.main, ["status", str(P)])
        self.assertEqual(rc, 0)
        lines = [ln for ln in out.splitlines() if ln.startswith("c0")]
        self.assertEqual(len(lines), 2)
        self.assertIn("plan missing", lines[0])
        self.assertIn("qa FAIL", lines[0])

    def test_qa_status(self):
        self.assertEqual(cli.qa_status({"status": "WARN"}), "WARN")
        self.assertEqual(cli.qa_status({"formats": [{"gates": [{"id": "G3", "status": "PASS"},
                                                               {"id": "G5", "status": "SKIPPED(no asr)"}]}]}),
                         "SKIPPED")
        self.assertIsNone(cli.qa_status({}))


def _python39():
    for c in ("/usr/bin/python3", shutil.which("python3.9") or ""):
        if c and os.path.exists(c):
            return c
    return sys.executable


class LauncherTests(unittest.TestCase):

    def run_hc(self, args, home=None, python=None):
        e = dict(os.environ)
        e.pop("PYTHONPATH", None)
        if home:
            e["HOOKCUTTER_HOME"] = str(home)
        return subprocess.run([python or _python39(), str(HC)] + args, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, universal_newlines=True, env=e, timeout=300)

    def test_executable(self):
        self.assertTrue(os.access(str(HC), os.X_OK))
        self.assertTrue(HC.read_text().startswith("#!/usr/bin/env python3"))

    def test_reexec_into_venv(self):
        if not env.venv_python().exists():
            self.skipTest("no venv on this machine")
        r = self.run_hc(["--version"])
        self.assertEqual(r.returncode, 0, r.stderr)
        import hookcutter
        self.assertIn("hook-cutter %s" % hookcutter.__version__, r.stdout)
        self.assertIn("venv active", r.stdout)
        self.assertIn(str(env.venv_dir()), r.stdout)

    def test_missing_venv_message(self):
        home = TMP / "emptyhome"
        home.mkdir(exist_ok=True)
        r = self.run_hc(["status", str(TMP)], home=home)
        self.assertEqual(r.returncode, 3)
        self.assertIn("hook-cutter is not set up. Run: python3 %s doctor --install" % HC, r.stderr)

    def test_doctor_bootstraps_without_venv(self):
        home = TMP / "emptyhome2"
        r = self.run_hc(["doctor", "--json"], home=home)
        self.assertEqual(r.returncode, 3, r.stderr)
        data = json.loads(r.stdout)
        self.assertFalse(data["ready"])
        self.assertFalse(data["in_venv"])
        venv = [c for c in data["checks"] if c["name"] == "venv"][0]
        self.assertEqual(venv["status"], "FAIL")
        self.assertIn("doctor --install", venv["detail"])


if __name__ == "__main__":
    unittest.main()
