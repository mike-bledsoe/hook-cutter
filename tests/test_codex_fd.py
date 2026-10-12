"""Regression tests for the third fix pass (outside review), cluster FD: render, framing,
shots and probe.

A fold to mono is a sum of both channels (never a copy of the louder one), a surround source is
downmixed before any fold, an interrupted or failed render leaves no log that calls the file up
to date, stage A files are looked up by the plan and style they belong to, fonts and safe-zone
margins move the render key, a clip's transcript pin does not, people stacked in one column are
told apart by y, a later speaker mark changes the crop inside a turn, shot boundaries of a source
with missing frames land on the nominal grid, and a dip join's seam pair says the darkening is
not shown.

Everything is synthetic: lavfi pictures, tones and noise written here, hand-written plans.
"""

from __future__ import annotations

import io
import shutil
import sys
import tempfile
import unittest
import wave
from contextlib import redirect_stderr, redirect_stdout
from fractions import Fraction
from pathlib import Path
from unittest import mock

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "scripts"))

import fixtures as F  # noqa: E402
from hookcutter import env, framing, probe, render, shots  # noqa: E402
from hookcutter import render_util as ru  # noqa: E402
from hookcutter.project import Project  # noqa: E402

SR = 48000
TMP = None


def setUpModule():
    global TMP
    TMP = Path(tempfile.mkdtemp(prefix="hc-codex-fd-"))


def tearDownModule():
    if TMP and TMP.exists():
        shutil.rmtree(str(TMP), ignore_errors=True)


def quiet(fn, *a, **kw):
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        rv = fn(*a, **kw)
    return rv, out.getvalue(), err.getvalue()


def tone_db(x, hz, sr=SR):
    """Level (dB re full scale, peak) of the `hz` component of a mono float array."""
    x = np.asarray(x, dtype=np.float64)
    if x.size < 64:
        return -200.0
    w = np.hanning(x.size)
    sp = np.abs(np.fft.rfft(x * w))
    k = int(round(hz * x.size / float(sr)))
    amp = 2.0 * float(sp[max(0, k - 2):k + 3].max()) / float(w.sum())
    return 20.0 * np.log10(max(amp, 1e-10))


def decode(path, af=None, nch=2, ss=None, t=None):
    args = []
    if ss is not None:
        args += ["-ss", "%.6f" % ss]
    if t is not None:
        args += ["-t", "%.6f" % t]
    args += ["-i", str(path), "-map", "0:a:0"]
    if af:
        args += ["-af", af]
    args += ["-ar", str(SR), "-f", "f32le", "-"]
    data = np.frombuffer(env.ff(args).stdout, dtype="<f4")
    return data[:data.size - data.size % nch].reshape(-1, nch).astype(np.float64)


def write_stereo_wav(path, left, right):
    st = np.stack([left, right], axis=1)
    pcm = np.clip(np.round(st * 32767.0), -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes(pcm.tobytes())


def one_seg_plan(clip_id, s, e, fps=25, marks=None, joins=None, segs=None):
    """A hand-written plan (the shape test_render_smoke uses) over one or more src ranges."""
    rate = Fraction(fps)
    out, cur = [], 0
    for k, (a, b) in enumerate(segs or [(s, e)]):
        n = int(round((Fraction(b).limit_denominator(1000) - Fraction(a).limit_denominator(1000)) * rate))
        out.append({"k": k, "piece": k, "src": [a, b],
                    "out": [float(cur / rate), float((cur + n) / rate)], "frames": n,
                    "samples": int(round(Fraction(n * SR) / rate)),
                    "join_before": "start" if k == 0 else "piece", "fade_in": 0.008,
                    "fade_out": 0.008, "gain_db": 0.0})
        cur += n
    marks = marks or [{"at_src": 0.0, "mode": "fixed", "x": 0.5, "y": 0.5, "crop": 1.0}]
    return {"version": 1, "clip": clip_id, "kind": "cut", "fps": [fps, 1], "grid_offset": 0.0,
            "sample_rate": SR, "pieces": [], "segments": out, "removals": [],
            "joins": list(joins or []), "words": [], "intent": [], "frames_out": cur,
            "duration_out": cur / float(fps), "hook_card": None, "framing": marks, "lints": []}


def plain_clip(plan):
    return {"kind": "cut", "captions": False, "framing": plan["framing"],
            "_eff": {"styles": ["clean"], "formats": ["vertical"]}}


# ---------------------------------------------------------------- A1 / A2: the fold is a sum

class TestFoldIsASum(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d = TMP / "fold"
        cls.d.mkdir(parents=True, exist_ok=True)
        cls.two = cls.d / "two.wav"
        # 300 Hz in the left channel only, 1000 Hz in the right channel only
        env.ff(["-f", "lavfi", "-i", "sine=frequency=300:sample_rate=48000:duration=2",
                "-f", "lavfi", "-i", "sine=frequency=1000:sample_rate=48000:duration=2",
                "-filter_complex", "[0:a][1:a]join=inputs=2:channel_layout=stereo[a]",
                "-map", "[a]", "-c:a", "pcm_s16le", str(cls.two)])
        cls.six = cls.d / "six.wav"       # 300 Hz in FL only, 1000 Hz in FC only
        env.ff(["-i", str(cls.two), "-af", "pan=5.1|FL=c0|FC=c1", "-c:a", "pcm_s16le", str(cls.six)])
        cls.four = cls.d / "four.wav"     # 300 Hz in channel 0 only
        env.ff(["-i", str(cls.two), "-af", "pan=quad|c0=c0", "-c:a", "pcm_s16le", str(cls.four)])

    def test_unchanged_layouts(self):
        self.assertEqual(render._pan({"channels": 1}), "pan=stereo|c0=c0|c1=c0")
        self.assertEqual(render._pan({"channels": 1, "fold_mono": True}), "pan=stereo|c0=c0|c1=c0")
        self.assertEqual(render._pan({"channels": 2, "fold_mono": False}), "pan=stereo|c0=c0|c1=c1")
        self.assertEqual(render._pan({"channels": 2, "fold_mono": False, "channel_db": [-20, -62]}),
                         "pan=stereo|c0=c0|c1=c1")
        self.assertEqual(render._pan({"channels": 6, "fold_mono": False}),
                         "aformat=channel_layouts=stereo")

    def test_stereo_fold_keeps_both_channels(self):
        src = decode(self.two)
        l300, r1000 = tone_db(src[:, 0], 300), tone_db(src[:, 1], 1000)
        self.assertLess(tone_db(src[:, 0], 1000), l300 - 50)      # the fixture is what it says
        for dbs in ([-20, -62], [-62, -20]):
            pan = render._pan({"channels": 2, "fold_mono": True, "channel_db": dbs})
            out = decode(self.two, "aresample=48000," + pan)
            for hz, ref in ((300, l300), (1000, r1000)):
                a, b = tone_db(out[:, 0], hz), tone_db(out[:, 1], hz)
                self.assertLess(abs(a - b), 1.0, (hz, dbs))
                self.assertGreater(a, ref - 7.0, (hz, dbs, a, ref))
        # the half weights cannot clip: two full-scale channels sum to full scale
        self.assertLessEqual(float(np.abs(out).max()), 1.0)

    def test_surround_fold_downmixes_first(self):
        src = decode(self.six, nch=6)
        fc = tone_db(src[:, 2], 1000)
        pan = render._pan({"channels": 6, "fold_mono": True,
                           "channel_db": [-40, -100, -20, -120, -120, -120]})
        self.assertTrue(pan.startswith("aformat=channel_layouts=stereo,"), pan)
        out = decode(self.six, "aresample=48000," + pan)
        for c in (0, 1):
            self.assertGreater(tone_db(out[:, c], 1000), fc - 16.0)     # centre dialogue kept
            self.assertGreater(tone_db(out[:, c], 1000), -40.0)

    def test_four_channel_fold_reaches_both_ears(self):
        pan = render._pan({"channels": 4, "fold_mono": True, "channel_db": [-20, -120, -120, -120]})
        out = decode(self.four, "aresample=48000," + pan)
        a, b = tone_db(out[:, 0], 300), tone_db(out[:, 1], 300)
        self.assertGreater(a, -40.0)
        self.assertLess(abs(a - b), 1.0)

    def test_probe_warning_says_nothing_is_dropped(self):
        v = {"width": 640, "height": 360, "hdr": "none", "vfr": False, "rotation": 0, "sar": "1:1",
             "fps": [25, 1], "start": 0.0}
        a = {"start": 0.0, "fold_mono": True, "dup_mono": False, "channels": 2, "holes": [],
             "channel_db": [-20.0, -62.0]}
        text = " | ".join(probe._warnings({}, v, a))
        self.assertIn("folded to mono", text)
        self.assertIn("mixed", text)
        self.assertIn("channel 2 is 42 dB quieter than channel 1", text)


class TestFoldEndToEnd(unittest.TestCase):
    """One speaker per channel; the second one talks only outside the sampled minute."""

    @classmethod
    def setUpClass(cls):
        cls.d = TMP / "fold-e2e"
        cls.d.mkdir(parents=True, exist_ok=True)
        dur = 70.0
        n = int(dur * SR)
        t = np.arange(n) / float(SR)
        gate = ((t % 0.4) < 0.3).astype(np.float64)
        left = F._pink_noise(n, -60.0, 1) + 0.1 * np.sin(2 * np.pi * 300 * t) * gate * ((t >= 10) & (t < 60))
        right = F._pink_noise(n, -60.0, 2) + 0.1 * np.sin(2 * np.pi * 1000 * t) * gate * ((t >= 0.5) & (t < 4.5))
        wav = cls.d / "two.wav"
        write_stereo_wav(wav, left, right)
        vid = cls.d / "v.mp4"
        F._encode_video(vid, F._video_filter("320x180", Fraction(25), dur, [], False))
        src = cls.d / "talk.mp4"
        env.ff(["-i", str(vid), "-i", str(wav), "-map", "0:v", "-map", "1:a", "-c:v", "copy",
                "-c:a", "aac", "-b:a", "160k", "-ar", str(SR), "-ac", "2", "-f", "mp4", str(src)])
        cls.P = F.make_project(cls.d, src)
        cls.proj = Project(cls.P)

    def test_quiet_channel_speaker_is_heard(self):
        audio = self.proj.source()["audio"]
        self.assertTrue(audio["fold_mono"], audio)            # the precondition
        plan = one_seg_plan("fold", 1.0, 4.0)
        res, _, _ = quiet(render.render_clip, self.proj, "fold", formats=["vertical"],
                          styles=["clean"], plan=plan, clip=plain_clip(plan))
        out = decode(res[0]["path"])[:3 * SR]         # the AAC decode pads its last packet
        self.assertGreaterEqual(out.shape[0], 3 * SR - 1024)
        mono = out.mean(axis=1)
        rms = 10 * np.log10(max(float(np.mean(mono ** 2)), 1e-12))
        self.assertGreater(rms, -30.0)
        self.assertGreater(tone_db(mono, 1000), -30.0)
        self.assertLess(abs(tone_db(out[:, 0], 1000) - tone_db(out[:, 1], 1000)), 1.0)
        ref = decode(self.proj.source_path(), "pan=mono|c0=c1", nch=1, ss=1.0, t=3.0)[:, 0]
        best = 0.0
        for lag in range(-48, 49, 2):
            a = mono[max(0, lag):mono.size + min(0, lag)]
            b = ref[max(0, -lag):ref.size + min(0, -lag)]
            m = min(a.size, b.size)
            a, b = a[:m] - a[:m].mean(), b[:m] - b[:m].mean()
            den = float(np.sqrt(np.sum(a * a) * np.sum(b * b))) or 1.0
            best = max(best, abs(float(np.sum(a * b))) / den)
        self.assertGreater(best, 0.9)


# ---------------------------------------------------------------- F5 / V16-N1: missing frames

def _cut_source(path, dur=10, cut=5.0, drop=None, dissolve=False):
    """30 fps, 320x180, two still pictures under light grain: a hard cut at `cut`, or a 0.6 s
    dissolve from 7.0 s. drop=(n0, n1) removes those source frames and keeps the timestamps of
    the rest, as a phone or a screen recorder does (variable frame rate)."""
    if dissolve:
        ins = ["-f", "lavfi", "-i", "smptebars=s=320x180:r=30:d=7.6",
               "-f", "lavfi", "-i", "rgbtestsrc=s=320x180:r=30:d=%s" % (dur - 7.0)]
        graph = "[0:v]settb=1/30[a];[1:v]settb=1/30[b];[a][b]xfade=transition=fade:duration=0.6:offset=7"
    else:
        ins = ["-f", "lavfi", "-i", "smptebars=s=320x180:r=30:d=%s" % cut,
               "-f", "lavfi", "-i", "rgbtestsrc=s=320x180:r=30:d=%s" % (dur - cut)]
        graph = "[0:v][1:v]concat=n=2:v=1:a=0"
    graph += ",noise=alls=3:allf=t,format=yuv420p"
    if drop:
        graph += ",select='not(between(n\\,%d\\,%d))'" % drop
    env.ff(ins + ["-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo",
                  "-filter_complex", graph + "[v]", "-map", "[v]", "-map", "2:a", "-t", str(dur),
                  "-fps_mode", "passthrough", "-c:v", "libx264", "-preset", "ultrafast",
                  "-c:a", "aac", str(path)])
    return path


class TestShotsOnMissingFrames(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d = TMP / "vfr"
        cls.P = {}
        for name, kw in (("cfr_cut", {}), ("vfr_cut", {"drop": (139, 149)}),
                         ("cfr_dis", {"dissolve": True, "dur": 11}),
                         ("vfr_dis", {"dissolve": True, "dur": 11, "drop": (195, 205)}),
                         ("late_drop", {"dur": 24, "cut": 20.0, "drop": (597, 599)})):
            sub = cls.d / name
            sub.mkdir(parents=True, exist_ok=True)
            cls.P[name] = F.make_project(sub, _cut_source(sub / (name + ".mp4"), **kw))

    def _analyze(self, name, a, b):
        res, _, _ = quiet(shots.analyze, self.P[name], range_src=[str(a), str(b)])
        return res

    def test_fixture_flags(self):
        v = Project(self.P["cfr_cut"]).source()["video"]
        self.assertFalse(v["vfr"])
        self.assertFalse(probe.frames_missing(v))
        self.assertFalse(shots.off_grid(v))
        self.assertTrue(shots.off_grid(Project(self.P["vfr_cut"]).source()["video"]))
        late = Project(self.P["late_drop"]).source()
        # the drops sit after the packets ingest samples and move the average rate by under
        # half a percent: only the frame count shows them
        self.assertFalse(late["video"]["vfr"], late["video"])
        self.assertTrue(probe.frames_missing(late["video"]))
        self.assertTrue(shots.off_grid(late["video"]))
        self.assertTrue(any("frames are missing" in w for w in late["warnings"]))

    def test_frames_missing_helper(self):
        base = {"fps": [30, 1], "nb_frames": 300, "duration": 10.0}
        self.assertFalse(probe.frames_missing(base))
        self.assertFalse(probe.frames_missing(dict(base, nb_frames=299)))       # one frame: no
        self.assertTrue(probe.frames_missing(dict(base, nb_frames=297)))
        self.assertFalse(probe.frames_missing(dict(base, nb_frames=330)))       # an edit list
        self.assertFalse(probe.frames_missing(dict(base, nb_frames=None)))
        self.assertFalse(probe.frames_missing({}))
        self.assertFalse(probe.frames_missing(None))

    def test_hard_cut_lands_on_the_grid(self):
        for name, rng, want in (("cfr_cut", (0, 10), 5.0), ("vfr_cut", (0, 10), 5.0),
                                ("late_drop", (16, 24), 20.0)):
            res = self._analyze(name, *rng)
            cuts = [(c["kind"], c["start_src"], c["src"], c["end_src"]) for c in res["cuts"]]
            self.assertEqual(cuts, [("cut", want, want, want)], name)

    def test_dissolve_lands_on_the_grid(self):
        ref = self._analyze("cfr_dis", 0, 11)["cuts"]
        got = self._analyze("vfr_dis", 0, 11)["cuts"]
        self.assertEqual([c["kind"] for c in ref], ["dissolve"])
        self.assertEqual([c["kind"] for c in got], ["dissolve"])
        self.assertEqual(got[0]["src"], ref[0]["src"])
        self.assertEqual(got[0]["end_src"], ref[0]["end_src"])
        self.assertLessEqual(abs(got[0]["start_src"] - ref[0]["start_src"]), 1 / 30.0 + 1e-6)

    def test_constant_rate_decode_is_untouched(self):
        """The constant-rate path keeps its exact ffmpeg arguments and frame counts, so the
        shots json of such a source (and every plan and render made from it) is unchanged."""
        proj = Project(self.P["cfr_cut"])
        src, path = proj.source(), proj.source_path()
        seen = []
        real = shots._stream

        def spy(p, args, nbytes):
            seen.append([str(a) for a in args])
            return real(p, args, nbytes)

        with mock.patch.object(shots, "_stream", spy):
            for win, n in ((shots.REFINE_WIN, 24), (shots.TRANS_WIN[0], 54), (2 * shots.SPIKE_WIN, 66)):
                t0, frames = shots._native_frames(path, src, 5.0, win)
                self.assertEqual(len(frames), n)
                self.assertEqual(shots.native_frame_count(win, 30), n)
                self.assertEqual(t0, round((5.0 - win / 2) * 30) / 30.0)
        t0 = round((5.0 - shots.REFINE_WIN / 2) * 30) / 30.0
        self.assertEqual(seen[0], ["-ss", "%.6f" % (t0 - 0.5 / 30), "-t", "%.6f" % (shots.REFINE_WIN + 0.2),
                                   "-i", str(path), "-map", "0:v:0",
                                   "-vf", "scale=64:36:flags=area,format=gray",
                                   "-fps_mode", "passthrough", "-enc_time_base", "-1",
                                   "-f", "rawvideo", "-pix_fmt", "gray", "-"])
        # and the grid decode returns the same frames for it (count and pixels)
        _t, plain = shots._native_frames(path, src, 5.0, shots.TRANS_WIN[0])
        with mock.patch.object(shots, "off_grid", lambda v: True):
            _t, grid = shots._native_frames(path, src, 5.0, shots.TRANS_WIN[0])
        self.assertEqual(len(grid), len(plain))
        self.assertTrue(all(np.array_equal(a, b) for a, b in zip(grid, plain)))

    def test_missing_frames_are_held_in_their_slots(self):
        proj = Project(self.P["vfr_cut"])
        src, path = proj.source(), proj.source_path()
        t0, frames = shots._native_frames(path, src, 5.0, shots.REFINE_WIN)
        self.assertEqual(len(frames), 24)
        steps = [float(np.mean(np.abs(frames[j] - frames[j - 1]))) for j in range(1, len(frames))]
        j = 1 + int(np.argmax(steps))
        self.assertAlmostEqual(t0 + j / 30.0, 5.0, places=4)
        k0 = int(round(t0 * 30))
        held = [k - k0 for k in range(140, 150)]            # slots of the missing frames
        self.assertTrue(all(steps[h - 1] == 0.0 for h in held if 1 <= h < len(frames)))


# ---------------------------------------------------------------- R3: interrupted renders

LETTERBOX = [{"at_src": 0.0, "mode": "letterbox"}]
CROPPED = [{"at_src": 0.0, "mode": "fixed", "x": 0.25, "y": 0.5, "crop": 0.6}]


def video_md5(path):
    return env.ff(["-i", str(path), "-map", "0:v:0", "-f", "md5", "-"]).text.strip()


class TestInterruptedRender(unittest.TestCase):
    """A render that stops after the new file took the MP4's name must not leave the previous
    render's log next to it, and a render that fails its check must not replace the good file."""

    def setUp(self):
        self.d = Path(tempfile.mkdtemp(prefix="r3-", dir=str(TMP)))
        src = F.make_av_fixture(TMP / "av", fps="25", dur=4, flashes=[1.0])
        self.P = F.make_project(self.d, src)
        self.proj = Project(self.P)
        self.plan_a = one_seg_plan("c1", 0.6, 1.8, marks=LETTERBOX)
        self.plan_b = one_seg_plan("c1", 0.6, 1.8, marks=CROPPED)
        self.mp4 = self.proj.render_path("c1", "vertical", "clean")
        self.log = self.proj.render_path("c1", "vertical", "clean", ".log.json")
        self._render(self.plan_a)
        self.md5_a = video_md5(self.mp4)
        self.key_a = env.read_json(self.log)["hashes"]["render"]

    def _render(self, plan):
        return quiet(render.render_clip, self.proj, "c1", formats=["vertical"], styles=["clean"],
                     plan=plan, clip=plain_clip(plan))

    def test_interrupt_after_the_rename_leaves_no_log(self):
        with mock.patch.object(render, "audio_md5", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                self._render(self.plan_b)
        self.assertNotEqual(video_md5(self.mp4), self.md5_a)      # the new picture is on disk
        self.assertFalse(self.log.exists())                       # ... and nothing vouches for it
        _res, _out, err = self._render(self.plan_a)               # the spec goes back to A
        self.assertNotIn("up to date", err)
        self.assertEqual(video_md5(self.mp4), self.md5_a)
        self.assertEqual(env.read_json(self.log)["hashes"]["render"], self.key_a)

    def test_failed_check_keeps_the_previous_render(self):
        boom = env.HCError("final render has 29 frames, plan says 30", code=3)
        with mock.patch.object(render, "_verify_final", side_effect=boom) as vf:
            with self.assertRaises(env.HCError):
                self._render(self.plan_b)
        self.assertNotEqual(Path(vf.call_args[0][0]), self.mp4)   # the temp file was checked
        self.assertEqual(video_md5(self.mp4), self.md5_a)
        self.assertEqual(env.read_json(self.log)["hashes"]["render"], self.key_a)
        self.assertEqual(list((self.P / "renders").glob("*.tmp.*")), [])
        _res, _out, err = self._render(self.plan_a)
        self.assertIn("up to date", err)


# ---------------------------------------------------------------- C1 / Q2: stage A files, seams

THREE = [(0.6, 1.4), (1.8, 2.6), (3.0, 3.8)]


def three_seg_plan(clip_id, shift=0.0):
    plan = one_seg_plan(clip_id, 0, 0, marks=LETTERBOX,
                        segs=[(round(a + shift, 3), round(b + shift, 3)) for a, b in THREE])
    sg = plan["segments"]
    plan["joins"] = [
        {"k": 1, "out": sg[1]["out"][0], "kind": "piece", "before_src": sg[0]["src"][1], "after_src": sg[1]["src"][0]},
        {"k": 2, "out": sg[2]["out"][0], "kind": "dip", "before_src": sg[1]["src"][1], "after_src": sg[2]["src"][0]}]
    return plan


class TestStageAFiles(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d = TMP / "stage-a"
        cls.d.mkdir(parents=True, exist_ok=True)
        src = F.make_av_fixture(TMP / "av", fps="25", dur=4, flashes=[1.0])
        cls.P = F.make_project(cls.d, src)
        cls.proj = Project(cls.P)
        # a second style whose base picture differs: the letterboxed picture sits higher
        env.write_json(cls.P / "presets" / "high.json", {"extends": "clean", "letterbox": {"y_center": 500}})
        cls.plan = three_seg_plan("c1")
        cls.clip = plain_clip(cls.plan)
        quiet(render.render_clip, cls.proj, "c1", formats=["vertical"], styles=["clean", "high"],
              plan=cls.plan, clip=cls.clip)

    def _log(self, style):
        return env.read_json(self.proj.render_path("c1", "vertical", style, ".log.json"))

    def _logged(self, style):
        return [self.P / sg["path"] for sg in self._log(style)["segments"]]

    def test_files_of_exactly_this_plan_and_style(self):
        got = {}
        for style in ("clean", "high"):
            got[style] = render.segment_paths(self.proj, "c1", "vertical", style=style,
                                              plan=self.plan, clip=self.clip)
            self.assertEqual(got[style], self._logged(style), style)
            self.assertTrue(all(q.is_file() for q in got[style]))
        self.assertNotEqual(got["clean"], got["high"])           # each style has its own picture

    def test_another_plan_has_no_files(self):
        moved = three_seg_plan("c1", shift=0.08)                  # same segment count, other frames
        self.assertEqual(len(moved["segments"]), len(self.plan["segments"]))
        self.assertIsNone(render.segment_paths(self.proj, "c1", "vertical", style="clean",
                                               plan=moved, clip=plain_clip(moved)))
        # a style that was never rendered, or does not exist, has none either
        env.write_json(self.P / "presets" / "low.json", {"extends": "clean", "letterbox": {"y_center": 1300}})
        self.assertIsNone(render.segment_paths(self.proj, "c1", "vertical", style="low",
                                               plan=self.plan, clip=self.clip))
        self.assertIsNone(render.segment_paths(self.proj, "c1", "vertical", style="no-such-style",
                                               plan=self.plan, clip=self.clip))

    def test_lookup_without_plan_keeps_the_newest_log(self):
        newest = max(("clean", "high"),
                     key=lambda st: self.proj.render_path("c1", "vertical", st, ".log.json").stat().st_mtime)
        self.assertEqual(render.segment_paths(self.proj, "c1", "vertical"), self._logged(newest))
        self.assertEqual(render.segment_paths(self.proj, "c1", "vertical", style="clean"),
                         self._logged(newest))
        self.assertIsNone(render.segment_paths(self.proj, "nope", "vertical"))

    def test_dip_seam_says_the_darkening_is_not_shown(self):
        self.assertEqual(render.seam_label(2, 1.6, {"kind": "piece"}), "#2  %s  piece" % ru.tc(1.6))
        self.assertEqual(render.seam_label(1, 0.0, {}), "#1  %s  cut" % ru.tc(0.0))
        dip = render.seam_label(2, 1.6, {"kind": "dip"})
        self.assertIn("dip", dip)
        self.assertIn("not shown", dip)
        segs = [{"path": str(self.P / sg["path"]), "frames": sg["frames"]} for sg in self._log("clean")["segments"]]
        seen = []
        real = ru.contact_sheet

        def spy(items, cols, out_path, cell_w, title=None):
            seen.append([label for _img, label in items])
            return real(items, cols, out_path, cell_w, title=title)

        with mock.patch.object(ru, "contact_sheet", spy):
            render.seams_sheet(self.proj, "c1", "vertical", self.plan, segs)
        labels = seen[-1]
        self.assertEqual(len(labels), 2)
        self.assertIn("piece", labels[0])
        self.assertNotIn("not shown", labels[0])
        self.assertIn("dip (darkening not shown)", labels[1])


# ---------------------------------------------------------------- R1 / S1: the render key

class TestRenderKey(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d = TMP / "key"
        cls.d.mkdir(parents=True, exist_ok=True)
        src = F.make_av_fixture(TMP / "av", fps="25", dur=4, flashes=[1.0])
        cls.P = F.make_project(cls.d, src)
        cls.proj = Project(cls.P)
        cls.plan = one_seg_plan("c1", 0.6, 1.8, marks=LETTERBOX)
        cls.clip = plain_clip(cls.plan)

    def _key(self, style="clean", clip=None):
        return render.expected_hashes(self.proj, "c1", "vertical", style, plan=self.plan,
                                      clip=clip or self.clip)["render"]

    def test_assets_are_in_the_key(self):
        from hookcutter import captions
        calls = []

        def fake(value):
            def fn(preset, clip, fmt, root):
                calls.append((preset.get("name"), clip is self.clip, fmt, Path(root)))
                return value
            return fn

        a = {"fonts": {"Body.ttf": "aa"}, "safezone": {"top": 1, "bottom": 2, "left": 3, "right": 4}}
        with mock.patch.object(captions, "render_assets", fake(a), create=True):
            k_a = self._key()
            self.assertEqual(self._key(), k_a)
        self.assertEqual(calls[0][1:], (True, "vertical", self.P))
        for other in ({"fonts": {"Body.ttf": "bb"}, "safezone": a["safezone"]},
                      {"fonts": a["fonts"], "safezone": dict(a["safezone"], right=350)},
                      {"fonts": {"Body.ttf": "missing"}, "safezone": a["safezone"]}):
            with mock.patch.object(captions, "render_assets", fake(other), create=True):
                self.assertNotEqual(self._key(), k_a, other)

    def test_font_bytes_and_safe_zone_margins_move_the_key(self):
        from hookcutter import captions, typeset
        if not hasattr(captions, "render_assets"):
            self.skipTest("captions.render_assets is not in this build")

        def clear():
            for name in ("_load", "_cmap"):
                fn = getattr(typeset, name, None)
                if hasattr(fn, "cache_clear"):
                    fn.cache_clear()

        fonts = HERE.parent / "fonts"
        (self.P / "fonts").mkdir(exist_ok=True)
        body = self.P / "fonts" / "Body.ttf"
        shutil.copyfile(fonts / "Anton-Regular.ttf", body)
        env.write_json(self.P / "presets" / "own.json",
                       {"extends": "clean", "caption": {"font": "Body.ttf"}, "hook_card": {"font": "Body.ttf"}})
        clear()
        k0 = self._key("own")
        shutil.copyfile(fonts / "ArchivoBlack-Regular.ttf", body)
        clear()
        self.assertNotEqual(self._key("own"), k0)
        shutil.copyfile(fonts / "Anton-Regular.ttf", body)
        clear()
        self.assertEqual(self._key("own"), k0)
        zones = self.P / "presets" / "safezones.json"
        try:
            env.write_json(zones, {"ads": {"right": 350}})          # a profile this clip does not use
            self.assertEqual(self._key("own"), k0)
            env.write_json(zones, {"organic": {"right": 350}})
            self.assertNotEqual(self._key("own"), k0)
        finally:
            zones.unlink()
        self.assertEqual(self._key("own"), k0)
        # and the render follows the key: a swapped font file re-renders
        def go():
            return quiet(render.render_clip, self.proj, "c1", formats=["vertical"], styles=["own"],
                         plan=self.plan, clip=self.clip)[2]
        self.assertNotIn("up to date", go())
        self.assertIn("up to date", go())
        shutil.copyfile(fonts / "ArchivoBlack-Regular.ttf", body)
        clear()
        self.assertNotIn("up to date", go())

    def test_transcript_pin_is_not_rendered(self):
        self.assertIn("transcript_sha", render.CLIP_NOT_RENDERED)
        k0 = self._key()
        self.assertEqual(self._key(clip=dict(self.clip, transcript_sha="0" * 64)), k0)
        self.assertEqual(self._key(clip=dict(self.clip, title="Another title")), k0)
        self.assertNotEqual(self._key(clip=dict(self.clip, caption_y=0.4)), k0)


# ---------------------------------------------------------------- F3 / F6: who the crop holds

SRC_720 = {"video": {"width": 1280, "height": 720, "sar": "1:1"}}
NO_PUNCH = {"motion": {"punch_in": 1.0}}


def face_rows(faces, t1=10.0, wobble=0.0):
    """4 fps shots-json samples; `wobble` moves each face in x, alternately and in opposite
    directions, so the face nearest in x changes from sample to sample."""
    out = []
    for k in range(int(t1 * 4) + 1):
        sign = 1 if k % 2 else -1
        out.append({"t_src": round(k * 0.25, 2),
                    "faces": [[f[0] + sign * wobble * (1 if i % 2 else -1), f[1], f[2], f[3], f[4]]
                              for i, f in enumerate(faces)]})
    return out


def one_shot(faces, t1=10.0):
    smp = face_rows(faces, t1)
    fl = shots.label_faces(shots.shot_faces(smp))
    kind = shots.shot_kind(fl)
    return {"version": shots.SHOTS_VERSION, "grid": shots.GRID, "sample_fps": shots.SAMPLE_FPS,
            "ranges": [[0.0, t1]], "range_src": [0.0, t1],
            "shots": [{"start_src": 0.0, "end_src": t1, "kind": kind,
                       "multi_face": shots.faces_multi(fl, kind), "faces": fl}],
            "transitions": [], "samples": smp}


def talk(sp, t0, dur, wl=0.3, gap=0.1):
    out, t = [], t0
    while t + wl <= t0 + dur + 1e-9:
        out.append(("word", round(t, 3), round(t + wl, 3), sp))
        t += wl + gap
    return out


def turn_plan(words, speakers, marks, fps=25):
    """One segment from 0 s (source time = output time) with speaker-labelled words."""
    end = round(max(w[2] for w in words) + 0.5, 2)
    n = int(round(end * fps))
    ws = [{"i": i, "raw": t, "text": t, "s_src": a, "e_src": b, "s_out": a, "e_out": b, "sp": sp,
           "hidden": False, "omitted": False, "removed": False, "filler": False, "piece": 0, "seg": 0}
          for i, (t, a, b, sp) in enumerate(words)]
    return {"clip": "t", "kind": "cut", "fps": [fps, 1],
            "segments": [{"k": 0, "src": [0.0, n / float(fps)], "out": [0.0, n / float(fps)], "frames": n}],
            "pieces": [{"n": 0, "role": "body"}], "joins": [], "words": ws,
            "framing": list(marks), "speakers": speakers}


TOP = [0.25, 0.25, 0.12, 0.22, 0.9]         # the taller face
BOTTOM = [0.25, 0.75, 0.10, 0.18, 0.9]      # the same column, lower tile


class TestStackedFaces(unittest.TestCase):
    def test_hint_y_picks_within_the_column(self):
        for wobble in (0.0, 0.01):
            sm = face_rows([TOP, BOTTOM], 20.0, wobble)
            low = framing.Tracker(sm, 0.25, 0, 20, hint_y=0.75)
            high = framing.Tracker(sm, 0.25, 0, 20, hint_y=0.25)
            for k in range(81):
                t = k * 0.25
                self.assertAlmostEqual(low.value(t)[1], 0.75, delta=0.02, msg=(wobble, t))
                self.assertAlmostEqual(high.value(t)[1], 0.25, delta=0.02, msg=(wobble, t))

    def test_without_a_y_the_pick_is_unchanged(self):
        trk = framing.Tracker(face_rows([TOP, BOTTOM], 20.0), 0.25, 0, 20)
        for t in (0.0, 7.5, 19.9):
            self.assertAlmostEqual(trk.value(t)[1], 0.25, places=3)          # the taller face

    def test_x_stays_the_first_key(self):
        # a face further away in x than one column is not a candidate for the y choice
        away = [0.33, 0.75, 0.10, 0.18, 0.9]
        trk = framing.Tracker(face_rows([TOP, away], 20.0), 0.25, 0, 20, hint_y=0.75)
        self.assertAlmostEqual(trk.value(5.0)[0], 0.25, places=3)
        self.assertAlmostEqual(trk.value(5.0)[1], 0.25, places=3)
        # a single row of people: a known y changes nothing
        row = face_rows([[0.30, 0.4, 0.1, 0.2, 0.9], [0.62, 0.42, 0.1, 0.22, 0.9]], 20.0, 0.01)
        a = framing.Tracker(row, 0.31, 0, 20, hint_y=0.4)
        b = framing.Tracker(row, 0.31, 0, 20)
        self.assertEqual((a.xs, a.ys), (b.xs, b.ys))

    def test_half_height_window_shows_the_person_talking(self):
        gallery = [TOP, [0.75, 0.25, 0.11, 0.20, 0.9], BOTTOM, [0.75, 0.75, 0.11, 0.20, 0.9]]
        speakers = {"TL": {"x": 0.25, "y": 0.25}, "TR": {"x": 0.75, "y": 0.25},
                    "BL": {"x": 0.25, "y": 0.75}, "BR": {"x": 0.75, "y": 0.75}}
        marks = [{"at_src": 0.0, "mode": "speaker", "crop": 0.5}]
        for who in ("BL", "TL"):
            plan = turn_plan(talk(who, 0.5, 8.0), speakers, marks)
            tracks = framing.build_tracks(plan, {}, one_shot(gallery), SRC_720, "vertical", NO_PUNCH)
            wins = framing.windows(tracks, plan, SRC_720)
            self.assertEqual({w["speaker"] for w in wins}, {who})
            for w in wins:
                self.assertLess(w["y_hi"] - w["y_lo"], 0.6)                   # not full height
                self.assertTrue(w["y_lo"] <= speakers[who]["y"] <= w["y_hi"], (who, w))
                self.assertTrue(w["lo"] <= 0.25 <= w["hi"])

    def test_windows_carry_the_y_interval(self):
        plan = turn_plan(talk("A", 0.5, 2.0), {}, [])
        sub = {"src": [0.0, 1.0], "frames": 25, "mode": "crop", "size": [242, 432],
               "cmds": [[0.0, 100, 200], [0.4, 110, 220]], "zoom": 1.0, "mark": 0}
        w = framing.windows([[sub]], plan, SRC_720)[0]
        self.assertEqual((w["y_lo"], w["y_hi"]), (round(220 / 720.0, 4), round(632 / 720.0, 4)))
        self.assertEqual((w["lo"], w["hi"]), (round(110 / 1280.0, 4), round(342 / 1280.0, 4)))
        rect = {"src": [0.0, 1.0], "frames": 25, "mode": "letterbox", "size": [1280, 720], "cmds": [],
                "zoom": 1.0, "rect": [0, 0, 640, 360], "rect_mark": True}
        w = framing.windows([[rect]], plan, SRC_720)[0]
        self.assertEqual((w["mode"], w["y_lo"], w["y_hi"], w["lo"], w["hi"]), ("rect", 0.0, 0.5, 0.0, 0.5))
        self.assertNotIn("y_lo", framing.RENDER_KEYS)


class TestSpeakerMarkCropInsideATurn(unittest.TestCase):
    TWO = [[0.30, 0.4, 0.12, 0.22, 0.9], [0.70, 0.4, 0.12, 0.22, 0.9]]
    SP = {"Host": {"x": 0.30}, "Guest": {"x": 0.70}}

    def _subs(self, words, marks):
        plan = turn_plan(words, self.SP, marks)
        return framing.build_tracks(plan, {}, one_shot(self.TWO), SRC_720, "vertical", NO_PUNCH)[0]

    def setUp(self):
        act = framing.active_rect(SRC_720)
        self.full = list(framing.crop_size(1280, 720, "vertical", 1.0, 1.0, act))
        self.tight = list(framing.crop_size(1280, 720, "vertical", 0.6, 1.0, act))
        self.assertNotEqual(self.full, self.tight)

    def test_later_mark_changes_the_crop(self):
        subs = self._subs(talk("Host", 0.5, 8.0), [{"at_src": 0.0, "mode": "speaker"},
                                                    {"at_src": 4.0, "mode": "speaker", "crop": 0.6}])
        self.assertEqual([(x["src"][0], x["size"], x["speaker"]) for x in subs],
                         [(0.0, self.full, "Host"), (4.0, self.tight, "Host")])
        # one face track for the whole turn: the crop centre does not jump at the mark
        (ax, _ay), (bx, _by) = subs[0]["cmds"][-1][1:], subs[1]["cmds"][0][1:]
        self.assertAlmostEqual((ax + self.full[0] / 2.0) / 1280, (bx + self.tight[0] / 2.0) / 1280, delta=0.01)

    def test_reversed_order(self):
        subs = self._subs(talk("Host", 0.5, 8.0), [{"at_src": 0.0, "mode": "speaker", "crop": 0.6},
                                                    {"at_src": 4.0, "mode": "speaker", "crop": 1.0}])
        self.assertEqual([x["size"] for x in subs], [self.tight, self.full])

    def test_letterbox_between_two_speaker_marks(self):
        subs = self._subs(talk("Host", 0.5, 8.0), [{"at_src": 0.0, "mode": "speaker"},
                                                    {"at_src": 2.0, "mode": "letterbox"},
                                                    {"at_src": 4.0, "mode": "speaker", "crop": 0.6}])
        self.assertEqual([x["mode"] for x in subs], ["crop", "letterbox", "crop"])
        self.assertEqual((subs[0]["size"], subs[2]["size"]), (self.full, self.tight))

    def test_turn_that_starts_just_before_the_mark(self):
        # the guest's turn starts three frames before the second mark
        words = talk("Host", 0.5, 3.0) + talk("Guest", 4.03, 4.0)
        subs = self._subs(words, [{"at_src": 0.0, "mode": "speaker"},
                                  {"at_src": 4.0, "mode": "speaker", "crop": 0.6}])
        guest = [x for x in subs if x["speaker"] == "Guest"]
        self.assertEqual([(x["src"][0], x["size"]) for x in guest], [(3.88, self.full), (4.0, self.tight)])

    def test_same_crop_marks_are_one_picture(self):
        words = talk("Host", 0.5, 8.0)
        one = self._subs(words, [{"at_src": 0.0, "mode": "speaker", "crop": 0.6}])
        two = self._subs(words, [{"at_src": 0.0, "mode": "speaker", "crop": 0.6},
                                 {"at_src": 4.0, "mode": "speaker", "crop": 0.6}])
        self.assertEqual({tuple(x["size"]) for x in two}, {tuple(self.tight)})
        self.assertEqual(sum(x["frames"] for x in two), one[0]["frames"])


if __name__ == "__main__":
    unittest.main()
