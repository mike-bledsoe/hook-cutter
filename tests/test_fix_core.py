"""Regression tests for the core fixes (env, probe, ingest, doctor, cli, spec, presets,
project, launcher). Each test pins one behaviour described in SKILL.md or references/*.md:
picture-area detection, the download format filter, exit codes, the installer, `hc all`
and `status`, the clips.json keys and their L01 checks, style names and temp folders."""

import io
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import threading
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest import mock

import fixtures as F
from hookcutter import cli, doctor, env, ingest, presets, probe
from hookcutter import spec as specmod
from hookcutter.project import Project

SK = env.SKILL_ROOT
HC = SK / "scripts" / "hc"
TMP = None


def setUpModule():
    global TMP
    TMP = Path(tempfile.mkdtemp(prefix="hc-fixcore-"))


def tearDownModule():
    if TMP and TMP.exists():
        for root, dirs, _files in os.walk(str(TMP)):
            for d in dirs:
                try:
                    os.chmod(os.path.join(root, d), 0o755)
                except OSError:
                    pass
        shutil.rmtree(str(TMP), ignore_errors=True)


def quiet(fn, *a, **kw):
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        rv = fn(*a, **kw)
    return rv, out.getvalue(), err.getvalue()


def av(**kw):
    kw.setdefault("dur", 6)
    kw.setdefault("flashes", [])
    return F.make_av_fixture(TMP / "fx", **kw)


def lavfi_clip(name, video, dur=4):
    """A small h264/aac mp4 from a lavfi video graph (cached in TMP)."""
    out = TMP / name
    if not out.exists():
        env.ff(["-f", "lavfi", "-i", video, "-f", "lavfi", "-i", "sine=frequency=300:duration=%s" % dur,
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-shortest", str(out)])
    return out


def words_project(name, words, source=None):
    """P with source (optional) and a transcript of (text, s, e[, sp]) tuples."""
    if source is not None:
        return F.make_project(TMP / name, source, words)
    P = TMP / name / "project"
    P.mkdir(parents=True, exist_ok=True)
    env.write_json(P / "transcript.json", F.fake_transcript(words))
    return P


def lint_ids(lints, level=None):
    return [(x["id"], x["level"]) for x in lints if level is None or x["level"] == level]


# ======================================================================= 1. picture area

class ActiveAreaTests(unittest.TestCase):

    def _frames(self, top, bottom, value=120, bar=0, n=3, size=(640, 360)):
        import numpy as np
        w, h = size
        out = []
        rng = np.random.default_rng(7)
        for _ in range(n):
            f = np.full((h, w), bar, dtype=np.uint8)
            f[top:bottom, :] = rng.integers(value - 40, value + 40, size=(bottom - top, w))
            out.append(f)
        return out

    def test_bars_give_the_picture_box_in_display_pixels(self):
        area = probe.active_area(self._frames(90, 270), (1280, 720))
        self.assertEqual(area, [0, 180, 1280, 360])

    def test_dark_curtain_is_not_a_bar(self):
        # luma about 30 around the picture is a dark set, not a letterbox
        self.assertIsNone(probe.active_area(self._frames(90, 270, bar=30), (1280, 720)))

    def test_thin_border_is_ignored(self):
        # 4 black rows of 360 (1.1%) is a sliver, not a layout
        self.assertIsNone(probe.active_area(self._frames(4, 356), (1280, 720)))

    def test_any_bright_sample_keeps_the_row(self):
        frames = self._frames(90, 270)
        frames[1][0:90, :] = 200          # one sample shows picture in the "bar"
        self.assertEqual(probe.active_area(frames, (1280, 720)), [0, 0, 1280, 540])

    def test_display_size_applies_sar(self):
        self.assertEqual(probe.display_size({"width": 960, "height": 720, "sar": "4:3"}), (1280, 720))
        self.assertEqual(probe.display_size({"width": 1280, "height": 720, "sar": "1:1"}), (1280, 720))

    def test_warning_uses_the_picture_height(self):
        src = {"duration": 100.0, "video": {"width": 1280, "height": 720, "hdr": "none", "rotation": 0,
                                            "fps": [25, 1], "vfr": False, "field_order": "progressive",
                                            "sar": "1:1", "start": 0.0,
                                            "active_area": [0, 180, 1280, 360]},
               "audio": {"start": 0.0}}
        ws = probe.source_warnings(src)
        self.assertTrue(any(w.startswith("letterboxed/gallery layout: the picture is 1280x360 at (0,180)")
                            for w in ws), ws)
        self.assertTrue(any("upscales 5.3x" in w for w in ws), ws)

    def test_ingest_letterboxed_file_and_refresh(self):
        src = lavfi_clip("letterbox.mp4", "testsrc2=size=1280x360:rate=25:duration=4,"
                                          "pad=1280:720:0:180:black")
        P = TMP / "lbproj"
        meta, out, _ = quiet(ingest.ingest, P, str(src))
        self.assertEqual(meta["video"]["active_area"], [0, 180, 1280, 360])
        self.assertIn("letterboxed/gallery layout: the picture is 1280x360 at (0,180)", out)
        self.assertIn("upscales 5.3x", out)
        self.assertNotIn("source is 1280x720: a full-height", out)
        # a refresh recomputes the layout and never touches the transcript
        env.write_json(P / "transcript.json", F.fake_transcript([("Hello", 1.0, 1.4)]))
        before = (P / "transcript.json").read_bytes()
        sj = env.read_json(P / "source.json")
        sj["video"].pop("active_area", None)
        env.write_json(P / "source.json", sj)
        meta2, _, _ = quiet(ingest.ingest, P, str(src))
        self.assertEqual(meta2["video"]["active_area"], [0, 180, 1280, 360])
        self.assertEqual(env.read_json(P / "source.json")["video"]["active_area"], [0, 180, 1280, 360])
        self.assertEqual((P / "transcript.json").read_bytes(), before)

    def test_ingest_dark_curtain_is_full_frame(self):
        src = lavfi_clip("curtain.mp4", "testsrc2=size=1280x360:rate=25:duration=4,"
                                        "pad=1280:720:0:180:0x1e1e1e")
        meta, out, _ = quiet(ingest.ingest, TMP / "curtainproj", str(src))
        self.assertIsNone(meta["video"].get("active_area"))
        self.assertNotIn("letterboxed", out)


# ======================================================================= 2 + 8. downloads

class DownloadTests(unittest.TestCase):

    def _fake_call(self, info):
        src = av()
        calls = []

        def fake_call(cmd, **kw):
            calls.append(list(cmd))
            out = Path(cmd[cmd.index("-o") + 1]).parent
            shutil.copy2(str(src), str(out / "source.mp4"))
            env.write_json(out / "source.info.json", info)
            return 0
        return calls, fake_call

    def test_format_filter_and_quiet_flags(self):
        info = {"id": "x", "duration": 6, "dynamic_range": "SDR", "format_id": "136+140",
                "format_note": "720p", "format": "136 - 1280x720 (720p)+140 - audio only"}
        calls, fake = self._fake_call(info)
        with mock.patch.object(ingest, "_ytdlp_available", return_value=True), \
                mock.patch.object(ingest, "_ytdlp_help", return_value="--progress-delta --js-runtimes"), \
                mock.patch.object(ingest.subprocess, "call", side_effect=fake):
            meta, out, _ = quiet(ingest.ingest, TMP / "dlproj", "https://example.com/watch?v=x",
                                 section="1-5")
        cmd = calls[0]
        self.assertEqual(cmd[cmd.index("-f") + 1], "bv*[format_id!$=-sr]+ba/b")
        for flag in ("--quiet", "--no-warnings", "--progress"):
            self.assertIn(flag, cmd)
        self.assertEqual(cmd[cmd.index("--progress-delta") + 1], "5")
        self.assertEqual(cmd[cmd.index("--downloader-args") + 1],
                         "ffmpeg:-hide_banner -loglevel error -nostats")
        self.assertFalse(any("AI-upscaled" in w for w in meta["warnings"]))

    def test_upscaled_download_is_flagged(self):
        info = {"id": "x", "duration": 6, "dynamic_range": "SDR", "format_id": "248-sr+140",
                "format_note": "1080p, AI-upscaled"}
        calls, fake = self._fake_call(info)
        with mock.patch.object(ingest, "_ytdlp_available", return_value=True), \
                mock.patch.object(ingest, "_ytdlp_help", return_value=""), \
                mock.patch.object(ingest.subprocess, "call", side_effect=fake):
            meta, out, _ = quiet(ingest.ingest, TMP / "dlproj2", "https://example.com/watch?v=y")
        self.assertTrue(any("AI-upscaled" in w for w in meta["warnings"]), meta["warnings"])
        self.assertIn("AI-upscaled", out)

    def test_upscaled_format_rules(self):
        self.assertTrue(ingest.upscaled_format({"format_note": "1080p, AI-upscaled"}))
        self.assertTrue(ingest.upscaled_format({"format_id": "399-sr+140"}))
        self.assertFalse(ingest.upscaled_format({"format_id": "136+140", "format_note": "720p"}))

    def test_old_ytdlp_without_progress_delta(self):
        self.assertNotIn("--progress-delta", ingest.quiet_args("usage: yt-dlp --progress --quiet"))
        self.assertIn("--progress-delta", ingest.quiet_args(""))


# ======================================================================= 3. exit codes, stdio, launcher

class ExitCodeTests(unittest.TestCase):

    def test_usage_error_exits_1(self):
        with self.assertRaises(SystemExit) as cm:
            quiet(cli.main, ["words", "P", "1.5", "3"])
        self.assertEqual(cm.exception.code, 1)
        with self.assertRaises(SystemExit) as cm:
            quiet(cli.main, ["nosuchcommand"])
        self.assertEqual(cm.exception.code, 1)

    def test_os_error_is_exit_3_without_traceback(self):
        exc = PermissionError(13, "Permission denied", "/x/export/01 - Clip.mp4.tmp")
        with mock.patch.object(cli, "dispatch", side_effect=exc):
            rc, _, err = quiet(cli.main, ["status", str(TMP)])
        self.assertEqual(rc, 3)
        self.assertIn("error: Permission denied: /x/export/01 - Clip.mp4.tmp", err)
        self.assertNotIn("Traceback", err)

    @unittest.skipIf(hasattr(os, "geteuid") and os.geteuid() == 0, "root ignores permissions")
    def test_read_only_folder_is_exit_3(self):
        ro = TMP / "readonly"
        ro.mkdir(exist_ok=True)
        os.chmod(str(ro), 0o555)
        try:
            rc, _, err = quiet(cli.main, ["ingest", str(ro / "P"), str(av())])
        finally:
            os.chmod(str(ro), 0o755)
        self.assertEqual(rc, 3, err)
        self.assertTrue(err.startswith("error: "), err)
        self.assertIn(str(ro), err)
        self.assertNotIn("Traceback", err)

    def test_worst_exit(self):
        self.assertEqual(cli.worst_exit(0, 2), 2)
        self.assertEqual(cli.worst_exit(2, 1), 1)
        self.assertEqual(cli.worst_exit(1, 3), 3)
        self.assertEqual(cli.worst_exit(0, 0), 0)
        self.assertEqual(cli.worst_exit(), 0)

    def test_gitignore_ships(self):
        self.assertIn("__pycache__/", (SK / ".gitignore").read_text().splitlines())


def _run_launcher(script, args, **extra_env):
    e = dict(os.environ)
    e.pop("PYTHONPATH", None)
    e.pop("PYTHONPYCACHEPREFIX", None)
    e.pop("PYTHONDONTWRITEBYTECODE", None)
    e.update(extra_env)
    return subprocess.run([sys.executable, str(script)] + list(args), stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, env=e, timeout=300)


class LauncherTests(unittest.TestCase):

    def test_cp1252_pipe_prints_any_text(self):
        if not env.venv_python().exists():
            self.skipTest("no venv on this machine")
        P = words_project("encproj", [("Hello", 1.0, 1.4), ("była.", 1.5, 1.9)])
        r = _run_launcher(HC, ["words", str(P), "0", "1"], PYTHONIOENCODING="cp1252")
        self.assertEqual(r.returncode, 0, r.stderr.decode("utf-8", "replace"))
        self.assertIn("była.", r.stdout.decode("utf-8"))

    def test_no_bytecode_written_into_the_skill(self):
        if not env.venv_python().exists():
            self.skipTest("no venv on this machine")
        copy = TMP / "skillcopy"
        if copy.exists():
            shutil.rmtree(str(copy))
        shutil.copytree(str(SK / "scripts"), str(copy / "scripts"),
                        ignore=shutil.ignore_patterns("__pycache__"))
        r = _run_launcher(copy / "scripts" / "hc", ["--version"])
        self.assertEqual(r.returncode, 0, r.stderr.decode("utf-8", "replace"))
        r = _run_launcher(copy / "scripts" / "hc", ["status", str(TMP / "nothing-here")])
        self.assertEqual(r.returncode, 1)
        caches = [str(p) for p in copy.rglob("__pycache__")]
        self.assertEqual(caches, [])


# ======================================================================= 4. doctor and install

class DoctorTests(unittest.TestCase):

    def setUp(self):
        self.home = Path(tempfile.mkdtemp(prefix="home-", dir=str(TMP)))
        self.env_patch = mock.patch.dict(os.environ, {"HOOKCUTTER_HOME": str(self.home)})
        self.env_patch.start()

    def tearDown(self):
        self.env_patch.stop()

    def test_install_commands_never_build_from_source(self):
        cmds = doctor.install_commands(["url"], uv="/x/uv", venv_exists=False)
        self.assertIn("--seed", cmds[0])
        self.assertIn("--no-build", cmds[1])
        with mock.patch.object(doctor, "find_base_python", return_value=(["/x/python3.12"], (3, 12))):
            cmds = doctor.install_commands(["url"], uv=None, venv_exists=False)
        self.assertEqual(cmds[0][:3], ["/x/python3.12", "-m", "venv"])
        i = cmds[1].index("--only-binary")
        self.assertEqual(cmds[1][i + 1], ":all:")

    def _fake_venv(self, version=(3, 13)):
        vdir = env.venv_dir()
        (vdir / "bin").mkdir(parents=True)
        vpy = env.venv_python()
        vpy.parent.mkdir(parents=True, exist_ok=True)
        vpy.write_text("#!/bin/sh\n")
        os.chmod(str(vpy), 0o755)
        return vdir

    def _install(self, versions, **kw):
        ran = []

        def call(cmd, **_k):
            ran.append(list(cmd))
            if cmd[1:3] == ["-m", "venv"]:
                vpy = env.venv_python()
                vpy.parent.mkdir(parents=True, exist_ok=True)
                vpy.write_text("#!/bin/sh\n")
            return 0
        with mock.patch.object(doctor, "_py_version", side_effect=lambda c: versions.get(c[0])), \
                mock.patch.object(doctor, "find_base_python",
                                  return_value=(["/x/python3.12"], (3, 12))), \
                mock.patch.object(doctor.shutil, "which", return_value=None), \
                mock.patch.object(doctor, "_ensure_pip"), \
                mock.patch.object(doctor.subprocess, "call", side_effect=call):
            quiet(doctor.install, [], **kw)
        return ran

    def test_python_flag_rebuilds_a_different_version(self):
        vdir = self._fake_venv()
        ran = self._install({str(env.venv_python()): (3, 13)}, python="/x/python3.12")
        aside = [p.name for p in vdir.parent.iterdir() if p.name.startswith("venv.py313-")]
        self.assertEqual(len(aside), 1, list(vdir.parent.iterdir()))
        self.assertEqual(ran[0][:4], ["/x/python3.12", "-m", "venv", str(vdir)])

    def test_same_version_keeps_the_venv(self):
        vdir = self._fake_venv()
        ran = self._install({str(env.venv_python()): (3, 12)}, python="/x/python3.12")
        self.assertFalse(any(p.name.startswith("venv.") for p in vdir.parent.iterdir()))
        self.assertEqual(len(ran), 1)
        self.assertEqual(ran[0][1:4], ["-m", "pip", "install"])

    def test_broken_venv_is_reported_and_moved_aside(self):
        vdir = env.venv_dir()
        vpy = env.venv_python()
        vpy.parent.mkdir(parents=True)
        os.symlink(str(self.home / "gone" / "python3.12"), str(vpy))
        report = doctor.venv_report(["numpy"])
        self.assertTrue(report and report.get("broken"), report)
        chk = doctor.check_venv(report)[0]
        self.assertEqual(chk.status, doctor.FAIL)
        self.assertTrue(chk.detail.startswith("broken: "), chk.detail)
        self._install({})
        self.assertTrue(any(p.name.startswith("venv.broken-") for p in vdir.parent.iterdir()))

    def test_missing_pip_is_added_with_ensurepip(self):
        calls = []
        states = iter([False, True])
        with mock.patch.object(doctor, "_has_pip", side_effect=lambda v: next(states)), \
                mock.patch.object(doctor.subprocess, "call",
                                  side_effect=lambda cmd, **k: calls.append(cmd) or 0):
            quiet(doctor._ensure_pip, Path("/x/venv/bin/python"))
        self.assertEqual(calls, [["/x/venv/bin/python", "-m", "ensurepip", "--upgrade"]])

    def test_failed_pip_install_explains_missing_wheels(self):
        self._fake_venv()

        def call(cmd, **_k):
            return 1 if cmd[1:3] == ["-m", "pip"] else 0
        with mock.patch.object(doctor, "_py_version", return_value=(3, 12)), \
                mock.patch.object(doctor.shutil, "which", return_value=None), \
                mock.patch.object(doctor, "_ensure_pip"), \
                mock.patch.object(doctor.subprocess, "call", side_effect=call):
            with self.assertRaises(env.HCError) as cm:
                quiet(doctor.install, [])
        self.assertEqual(cm.exception.code, 3)
        self.assertIn("no prebuilt wheel for this OS", cm.exception.msg)

    def test_platform_hints(self):
        with mock.patch.object(doctor.sys, "platform", "linux"):
            hint = doctor.no_python_hint()
            ff = doctor.ffmpeg_missing_hint("ffmpeg not found on PATH")
        self.assertTrue(hint.split(". ", 1)[1].startswith("Install uv"), hint)
        self.assertIn("apt install ffmpeg", ff)
        self.assertIn("Ubuntu 24.04", ff)
        self.assertIn("static build", ff)
        with mock.patch.object(doctor.sys, "platform", "darwin"):
            ff = doctor.ffmpeg_missing_hint("ffmpeg not found on PATH. Run: hc doctor")
        self.assertIn("brew install ffmpeg", ff)
        self.assertIn("static build", ff)
        self.assertNotIn("Run: hc doctor", ff)
        self.assertIn("HOOKCUTTER_FFMPEG", ff)

    def test_cuda_libraries_check(self):
        self.assertEqual(doctor.check_cuda({"cuda": None}), [])
        warn = doctor.check_cuda({"cuda": {"devices": 1, "missing": ["cuBLAS", "cuDNN"]}})[0]
        self.assertEqual(warn.status, doctor.WARN)
        self.assertIn("GPU visible but cuBLAS/cuDNN missing: pip install nvidia-cublas-cu12 "
                      "nvidia-cudnn-cu12", warn.detail)
        ok = doctor.check_cuda({"cuda": {"devices": 2, "missing": []}})[0]
        self.assertEqual(ok.status, doctor.PASS)

    def test_interpreter_row(self):
        names = [c.name for c in doctor.check_platform()]
        self.assertIn("interpreter", names)
        self.assertNotIn("launcher python", names)

    def test_backend_flag_picks_the_faster_whisper_model(self):
        a = cli.build_parser().parse_args(["doctor", "--model", "large-v3-turbo", "--backend",
                                           "faster-whisper"])
        self.assertEqual((a.model, a.backend), ("large-v3-turbo", "faster-whisper"))
        got = []
        import faster_whisper.utils as fwu

        def fake_dl(name, cache_dir=None):
            got.append(name)
            return cache_dir
        from hookcutter import transcribe
        with mock.patch.object(transcribe, "resolve_backend", return_value="mlx"), \
                mock.patch.object(doctor, "_remote_size", return_value=None), \
                mock.patch.object(fwu, "download_model", side_effect=fake_dl):
            quiet(doctor.model_download_main, "large-v3-turbo", "faster-whisper")
        self.assertEqual(len(got), 1)
        self.assertFalse(doctor.is_mlx_model(got[0]), got)

    @unittest.skipUnless(sys.platform == "darwin", "macOS only")
    def test_sysctl_does_not_depend_on_path(self):
        seen = []

        def fake_out(cmd, **kw):
            seen.append(cmd[0])
            return 0, "1\n"
        with mock.patch.object(doctor, "_out", side_effect=fake_out), \
                mock.patch.dict(os.environ, {"PATH": "/usr/bin:/bin"}):
            self.assertEqual(doctor._sysctl("hw.optional.arm64"), "1")
        if os.path.exists("/usr/sbin/sysctl"):
            self.assertEqual(seen, ["/usr/sbin/sysctl"])

    def test_hints_quote_the_launcher_path(self):
        with mock.patch.object(env, "SCRIPTS_DIR", Path("/a/Bob's skills/scripts")):
            detail = doctor.check_venv(None)[0].detail
        self.assertIn("python3 '/a/Bob'\"'\"'s skills/scripts/hc' doctor --install", detail)


# ======================================================================= 5. hc all, status, stamps

def _status_project(name):
    P = F.make_project(TMP / name, av(), [("Hello", 1.0, 1.4), ("world.", 1.5, 1.9)])
    env.write_json(P / "clips.json", {"version": 1, "clips": {"c01-a": {"pieces": [{"words": [0, 1]}]}}})
    return P


class StampsTests(unittest.TestCase):

    def test_two_runs_keep_both_clips(self):
        P = _status_project("stampsproj")
        proj = Project(P)
        a, b = cli.Stamps(proj), cli.Stamps(proj)     # both loaded before either saves
        a.set("render", "c01-a", {"in": "k1", "out": {"renders/x.mp4": "f1"}})
        b.set("render", "c02-b", {"in": "k2", "out": {"renders/y.mp4": "f2"}})
        b.drop("qa", "c02-b")
        a.save()
        b.save()
        data = env.read_json(P / "cache" / "all-stamps.json")
        self.assertEqual(sorted(data["render"]), ["c01-a", "c02-b"])

    def test_parallel_saves_under_the_lock(self):
        P = _status_project("stampsproj2")
        proj = Project(P)
        ids = ["c%02d" % i for i in range(8)]
        start = threading.Barrier(len(ids))

        def run(cid):
            st = cli.Stamps(proj)
            start.wait()
            st.set("render", cid, {"in": cid, "out": {}})
            st.save()
        threads = [threading.Thread(target=run, args=(c,)) for c in ids]
        with redirect_stderr(io.StringIO()):          # "waiting for ..." lock notices
            for t in threads:
                t.start()
            for t in threads:
                t.join()
        data = env.read_json(P / "cache" / "all-stamps.json")
        self.assertEqual(sorted(data["render"]), ids)


class StatusTests(unittest.TestCase):

    def _render(self, P, plan, plan_hash):
        from hookcutter import render
        env.write_json(P / "plans" / "c01-a.plan.json", plan)
        r = P / "renders" / "c01-a__vertical__clean.mp4"
        r.parent.mkdir(parents=True, exist_ok=True)
        r.write_bytes(b"not really a video")
        env.write_json(P / "renders" / "c01-a__vertical__clean.log.json",
                       {"hashes": {"plan": plan_hash if plan_hash else render.plan_hash(plan)},
                        "created": "2026-01-01T00:00:00+0000"})
        old = 1_000_000_000
        for f in (r, P / "renders" / "c01-a__vertical__clean.log.json"):
            os.utime(str(f), (old, old))
        return r

    def _line(self, P):
        rc, out, _ = quiet(cli.main, ["status", str(P)])
        self.assertEqual(rc, 0)
        return [ln for ln in out.splitlines() if ln.startswith("c01-a")][0]

    def test_copied_project_is_not_stale(self):
        P = _status_project("statusfresh")
        plan = {"kind": "cut", "segments": [{"frames": 10}], "lints": [], "duration_out": 0.4}
        self._render(P, plan, None)          # render older than the plan file, same plan hash
        line = self._line(P)
        self.assertIn("render 1/1 ", line + " ")
        self.assertNotIn("STALE", line.split("render", 1)[1].split("cover")[0])

    def test_changed_plan_is_stale(self):
        P = _status_project("statusstale")
        plan = {"kind": "cut", "segments": [{"frames": 10}], "lints": [], "duration_out": 0.4}
        self._render(P, plan, "0" * 64)
        self.assertIn("1/1 STALE", self._line(P))

    def test_qa_freshness_uses_created_stamps(self):
        P = _status_project("statusqa")
        plan = {"kind": "cut", "segments": [{"frames": 10}], "lints": [], "duration_out": 0.4}
        self._render(P, plan, None)
        qp = P / "qa" / "c01-a.json"
        env.write_json(qp, {"status": "INCOMPLETE", "created": "2026-01-02T00:00:00+00:00"})
        os.utime(str(qp), (900_000_000, 900_000_000))     # mtime older than the render file
        line = self._line(P)
        self.assertIn("qa INCOMPLETE", line)
        self.assertNotIn("qa INCOMPLETE STALE", line)

    def test_qa_status_incomplete(self):
        self.assertEqual(cli.qa_status({"status": "INCOMPLETE"}), "INCOMPLETE")
        self.assertEqual(cli._worst(["PASS", "INCOMPLETE", "SKIPPED"]), "INCOMPLETE")
        self.assertEqual(cli._worst(["INCOMPLETE", "WARN"]), "WARN")

    def test_printed_commands_quote_paths(self):
        P = TMP / "two words" / "it's here"
        P.mkdir(parents=True, exist_ok=True)
        _, out, _ = quiet(cli.main, ["status", str(P)])
        self.assertIn("run: hc transcribe '%s'" % str(P).replace("'", "'\"'\"'"), out)


class AllTests(unittest.TestCase):

    def _fake_dispatch(self, P, qa_rc=1, export_rc=1):
        proj = Project(P)
        log = []

        def dispatch(a):
            log.append((a.cmd, a))
            if a.cmd == "plan":
                for cid in a.clips:
                    env.write_json(proj.plan_path(cid), {"kind": "cut", "clip": cid})
            elif a.cmd == "render":
                for f in (a.formats or ["vertical"]):
                    for s in (a.styles or ["clean"]):
                        proj.render_path(a.clips[0], f, s).write_bytes(b"x")
            elif a.cmd == "cover":
                for cid in a.clips:
                    for s in (a.styles or ["clean"]):
                        for k in ("main", "grid", "mobile", "json"):
                            proj.cover_path(cid, s, k).write_bytes(b"x")
            elif a.cmd == "qa":
                for cid in a.clips:
                    env.write_json(proj.qa_path(cid, ".json"), {"status": "FAIL" if qa_rc == 1 else "PASS"})
                return qa_rc
            elif a.cmd == "export":
                return export_rc
            return 0
        return log, dispatch

    def test_flags_reach_every_step_and_export_runs_after_qa_fail(self):
        P = _status_project("allproj")
        log, fake = self._fake_dispatch(P)
        args = cli.build_parser().parse_args(["all", str(P), "--styles", "clean", "--formats", "vertical"])
        with mock.patch.object(cli, "dispatch", side_effect=fake), \
                mock.patch.object(cli, "_render_code_hash", return_value="code"):
            rc, _, _ = quiet(cli.cmd_all, args)
        steps = [c for c, _ in log]
        self.assertEqual(steps, ["plan", "render", "cover", "qa", "export"])
        qa = [a for c, a in log if c == "qa"][0]
        self.assertTrue(qa.require_coverage)
        self.assertFalse(qa.no_asr)
        self.assertEqual((qa.styles, qa.formats), (["clean"], ["vertical"]))
        ex = [a for c, a in log if c == "export"][0]
        self.assertEqual((ex.styles, ex.formats), (["clean"], ["vertical"]))
        self.assertEqual(rc, 1)

    def test_no_asr_skips_require_coverage(self):
        P = _status_project("allproj2")
        log, fake = self._fake_dispatch(P, qa_rc=0, export_rc=0)
        args = cli.build_parser().parse_args(["all", str(P), "--no-asr"])
        with mock.patch.object(cli, "dispatch", side_effect=fake), \
                mock.patch.object(cli, "_render_code_hash", return_value="code"):
            rc, _, _ = quiet(cli.cmd_all, args)
        qa = [a for c, a in log if c == "qa"][0]
        self.assertTrue(qa.no_asr)
        self.assertFalse(qa.require_coverage)
        self.assertEqual(rc, 0)

    def test_tool_error_in_export_wins(self):
        P = _status_project("allproj3")
        log, fake = self._fake_dispatch(P, qa_rc=1, export_rc=3)
        args = cli.build_parser().parse_args(["all", str(P)])
        with mock.patch.object(cli, "dispatch", side_effect=fake), \
                mock.patch.object(cli, "_render_code_hash", return_value="code"):
            rc, _, _ = quiet(cli.cmd_all, args)
        self.assertEqual(rc, 3)

    def test_unknown_clip_suggests(self):
        P = _status_project("allproj4")
        rc, _, err = quiet(cli.main, ["all", str(P), "c01-b"])
        self.assertEqual(rc, 1)
        self.assertIn("did you mean 'c01-a'", err)

    def test_cli_flags(self):
        ps = cli.build_parser()
        a = ps.parse_args(["transcribe", "P", "--retime"])
        self.assertTrue(a.retime)
        a = ps.parse_args(["ingest", "P", "v.mp4", "--cfr"])
        self.assertTrue(a.cfr)
        a = ps.parse_args(["qa", "P", "--formats", "landscape"])
        self.assertEqual(a.formats, ["landscape"])
        a = ps.parse_args(["export", "P", "--formats", "vertical", "--force"])
        self.assertEqual((a.formats, a.force), (["vertical"], True))
        a = ps.parse_args(["doctor", "--python", "/x/python3.13"])
        self.assertEqual((a.python, a.backend), ("/x/python3.13", "auto"))


# ======================================================================= 6. spec keys and lints

def _spec(clip, **top):
    raw = {"version": 1, "clips": {"c01-a": clip}}
    raw.update(top)
    return specmod.normalize_spec(raw)


def _issues(sp, cid="c01-a"):
    return lint_ids(sp["clips"][cid]["_issues"]) + lint_ids(sp["_issues"])


WORDS = [("Hello", 1.0, 1.4, "Ann"), ("there", 1.5, 1.9, "Ann"), ("my", 2.0, 2.2, "Ann"),
         ("friend.", 2.3, 2.8, "Ann"), ("Yes.", 3.0, 3.4, "Ben"), ("Right", 3.5, 3.9, "Ben"),
         ("away.", 4.0, 4.6, "Ben")]


class SpecTests(unittest.TestCase):

    def test_caption_y_bounds(self):
        self.assertIn(("L01", "FAIL"), _issues(_spec({"pieces": [{"words": [0, 3]}], "caption_y": 2500})))
        self.assertIn(("L01", "FAIL"), _issues(_spec({"pieces": [{"words": [0, 3]}], "caption_y": -4})))
        sp = _spec({"pieces": [{"words": [0, 3]}], "caption_y": 1200})
        self.assertEqual(sp["clips"]["c01-a"]["caption_y"], 1200.0)
        self.assertNotIn(("L01", "FAIL"), _issues(sp))

    def test_unresolvable_style_fails_at_load(self):
        P = TMP / "styleproj"
        P.mkdir(parents=True, exist_ok=True)
        for bad, needle in (("nosuchstyle", "not found"), ("CLEAN", "did you mean 'clean'"),
                            ("safezones", "not a preset")):
            env.write_json(P / "clips.json", {"version": 1, "clips": {"c01-a": {
                "pieces": [{"words": [0, 3]}], "styles": [bad]}}})
            sp = specmod.load_spec(Project(P))
            fails = [x for x in sp["clips"]["c01-a"]["_issues"] if x["level"] == "FAIL"]
            self.assertEqual(len(fails), 1, bad)
            self.assertIn(needle, fails[0]["msg"])
        env.write_json(P / "clips.json", {"version": 1, "clips": {"c01-a": {
            "pieces": [{"words": [0, 3]}], "styles": ["clean", "pop"]}}})
        sp = specmod.load_spec(Project(P))
        self.assertEqual([x for x in sp["clips"]["c01-a"]["_issues"] if x["level"] == "FAIL"], [])

    def test_hook_card_duration(self):
        tr = F.fake_transcript(WORDS)
        sp = _spec({"pieces": [{"words": [0, 6]}], "hook_card": {"lines": ["HELLO"], "dur": 30}})
        self.assertIn(("L01", "WARN"), _issues(sp))
        lints = specmod.validate(sp, tr)
        self.assertTrue(any(x["id"] == "L01" and x["level"] == "FAIL" and "covers the whole clip"
                            in x["msg"] for x in lints), lints)
        sp = _spec({"pieces": [{"words": [0, 6]}], "hook_card": {"lines": ["HELLO"], "dur": 2.0}})
        self.assertEqual(_issues(sp), [])
        self.assertEqual([x for x in specmod.validate(sp, tr) if x["level"] == "FAIL"], [])
        sp = _spec({"pieces": [{"words": [0, 6]}], "hook_card": {"lines": ["HELLO"], "dur": "auto"}})
        self.assertEqual(_issues(sp), [])

    def test_transcript_sha_accepts_text_shas(self):
        tr = F.fake_transcript(WORDS)
        tr["text_sha"] = "t" * 64
        tr["text_compatible_shas"] = ["o" * 64]
        for sha, ok in ((tr["sha256"], True), ("t" * 64, True), ("o" * 64, True), ("x" * 64, False)):
            sp = _spec({"pieces": [{"words": [0, 3]}]}, transcript_sha=sha)
            got = [x for x in specmod.validate(sp, tr) if x["id"] == "L03"]
            self.assertEqual(bool(got), not ok, sha)

    def test_speakers_map(self):
        sp = _spec({"pieces": [{"words": [0, 3]}]}, speakers={"Ann": {"x": 0.27}, "Ben": {"x": 0.75, "y": 0.4}})
        self.assertEqual(sp["speakers"], {"Ann": {"x": 0.27}, "Ben": {"x": 0.75, "y": 0.4}})
        self.assertEqual(_issues(sp), [])
        self.assertEqual(_spec({"pieces": [{"words": [0, 3]}]})["speakers"], {})
        for bad in ({"Ann": {"x": 1.5}}, {"Ann": {"y": 0.5}}, {"Ann": 0.3}, ["Ann"]):
            self.assertIn(("L01", "FAIL"), _issues(_spec({"pieces": [{"words": [0, 3]}]}, speakers=bad)), bad)
        self.assertIn(("L01", "WARN"), _issues(_spec({"pieces": [{"words": [0, 3]}]},
                                                     speakers={"Ann": {"x": 0.2, "z": 1}})))
        # a label no word or clip uses
        sp = _spec({"pieces": [{"words": [0, 3]}]}, speakers={"Anne": {"x": 0.27}})
        lints = specmod.validate(sp, F.fake_transcript(WORDS))
        self.assertTrue(any(x["level"] == "WARN" and "'Anne'" in x["msg"] for x in lints), lints)

    def test_speakers_change_the_clip_hash(self):
        a = _spec({"pieces": [{"words": [0, 3]}]}, speakers={"Ann": {"x": 0.27}})
        b = _spec({"pieces": [{"words": [0, 3]}]}, speakers={"Ann": {"x": 0.30}})
        c = _spec({"pieces": [{"words": [0, 3]}]}, speakers={"Ann": {"x": 0.27}})
        self.assertNotEqual(specmod.clip_hash(a, "c01-a"), specmod.clip_hash(b, "c01-a"))
        self.assertEqual(specmod.clip_hash(a, "c01-a"), specmod.clip_hash(c, "c01-a"))

    def test_framing_keys(self):
        sp = _spec({"pieces": [{"words": [0, 3]}], "framing": [
            {"at_src": 1.0, "mode": "letterbox", "rect": [0.0, 0.25, 0.5, 0.5]},
            {"at_src": 2.0, "mode": "speaker"},
            {"at_src": 3.0, "mode": "track", "x": 0.3, "reaction": True}]})
        marks = sp["clips"]["c01-a"]["framing"]
        self.assertEqual(marks[0]["rect"], [0.0, 0.25, 0.5, 0.5])
        self.assertEqual((marks[1]["mode"], marks[1]["crop"]), ("speaker", 1.0))
        self.assertTrue(marks[2]["reaction"])
        self.assertNotIn("reaction", marks[0])
        self.assertEqual(_issues(sp), [])
        for bad in ({"at_src": 1.0, "mode": "letterbox", "rect": [0.6, 0, 0.5, 1]},
                    {"at_src": 1.0, "mode": "letterbox", "rect": [0, 0, 1]},
                    {"at_src": 1.0, "mode": "track", "reaction": "yes"}):
            self.assertIn(("L01", "FAIL"), _issues(_spec({"pieces": [{"words": [0, 3]}], "framing": [bad]})))
        for warn in ({"at_src": 1.0, "mode": "track", "rect": [0, 0, 0.5, 0.5]},
                     {"at_src": 1.0, "mode": "speaker", "x": 0.4}):
            self.assertIn(("L01", "WARN"), _issues(_spec({"pieces": [{"words": [0, 3]}], "framing": [warn]})))

    def test_piece_edge_offsets(self):
        sp = _spec({"pieces": [{"words": [0, 3], "in_offset": 2, "out_offset": -3}]})
        p = sp["clips"]["c01-a"]["pieces"][0]
        self.assertEqual((p["in_offset"], p["out_offset"]), (2, -3))
        p = _spec({"pieces": [{"words": [0, 3]}]})["clips"]["c01-a"]["pieces"][0]
        self.assertEqual((p["in_offset"], p["out_offset"]), (0, 0))
        # the bound is 60 output frames (it was 12: too short to step over a false start)
        for ok in (13, -13, 60, -60):
            self.assertNotIn(("L01", "FAIL"), _issues(_spec({"pieces": [{"words": [0, 3], "in_offset": ok}]})),
                             ok)
        for bad in (61, -61, 1.5, "x", True):
            self.assertIn(("L01", "FAIL"), _issues(_spec({"pieces": [{"words": [0, 3], "in_offset": bad}]})),
                          bad)

    def test_byte_order_marks(self):
        P = TMP / "bomproj"
        P.mkdir(parents=True, exist_ok=True)
        body = json.dumps({"version": 1, "clips": {"c01-a": {"pieces": [{"words": [0, 1]}]}}})
        (P / "clips.json").write_bytes(b"\xef\xbb\xbf" + body.encode("utf-8"))
        self.assertEqual(list(specmod.load_spec(Project(P))["clips"]), ["c01-a"])
        self.assertEqual(env.read_json(P / "clips.json")["version"], 1)

    def test_validate_names_the_closest_clip(self):
        sp = _spec({"pieces": [{"words": [0, 3]}]})
        lints = specmod.validate(sp, clip_ids=["c01-b"])
        self.assertTrue(any(x["id"] == "L01" and "did you mean 'c01-a'" in x["msg"] for x in lints),
                        lints)

    def test_unknown_clip_message(self):
        err = env.unknown_clip("c01-fird-big-client", ["c01-fired-big-client", "c02-other"])
        self.assertEqual(err.code, 1)
        self.assertIn("did you mean 'c01-fired-big-client'", err.msg)
        self.assertIn("clips.json has: c01-fired-big-client, c02-other", err.msg)
        self.assertIn("did you mean 'c02-other'", env.unknown_clip("c02", ["c01-a", "c02-other"]).msg)
        self.assertIn("no clips yet", env.unknown_clip("c01", []).msg)


# ======================================================================= 8. presets, temp folders, probe, cfr

class PresetNameTests(unittest.TestCase):

    def test_names_are_exact_slugs(self):
        self.assertEqual(presets.check_name("clean"), "clean")
        for bad in ("CLEAN", "Clean", "safezones", "my style", "../clean", "-x", ""):
            with self.assertRaises(env.HCError) as cm:
                presets.find(bad)
            self.assertEqual(cm.exception.code, 1, bad)
        with self.assertRaises(env.HCError) as cm:
            presets.load("CLEAN")
        self.assertIn("did you mean 'clean'", cm.exception.msg)
        self.assertNotIn("safezones", presets.available())
        self.assertIn("clean", presets.available())

    def test_lead_ms_range(self):
        P = TMP / "leadproj"
        (P / "presets").mkdir(parents=True, exist_ok=True)
        for value, ok in ((120, True), (0, True), (250, True), (300, False), (-10, False), ("80", False)):
            env.write_json(P / "presets" / "lead.json", {"extends": "clean", "caption": {"lead_ms": value}})
            if ok:
                self.assertEqual(presets.load("lead", P)["caption"]["lead_ms"], value)
            else:
                with self.assertRaises(env.HCError) as cm:
                    presets.load("lead", P)
                self.assertEqual(cm.exception.code, 1)
                self.assertIn("lead_ms", cm.exception.msg)
        env.write_json(P / "presets" / "lead.json", {"extends": "clean",
                                                     "landscape": {"caption": {"lead_ms": 400}}})
        with self.assertRaises(env.HCError):
            presets.load("lead", P)


class TempFolderTests(unittest.TestCase):

    def test_system_temp_dirs(self):
        self.assertTrue(Project(os.path.join(tempfile.gettempdir(), "p")).is_temp())
        self.assertTrue(Project("/tmp/p").is_temp())
        self.assertFalse(Project(os.path.join(os.path.expanduser("~"), "hc-projects", "p")).is_temp())

    def test_env_vars_and_case(self):
        base = Path(tempfile.mkdtemp(prefix="scratch-", dir=str(TMP)))
        other = os.path.join(os.path.expanduser("~"), "hc-not-temp-%d" % os.getpid())
        with mock.patch.dict(os.environ, {"TMPDIR": other, "TEMP": "", "TMP": ""}):
            self.assertTrue(Project(os.path.join(other, "p")).is_temp())
        if sys.platform == "darwin":
            upper = os.path.join(tempfile.gettempdir().upper(), "p")
            self.assertTrue(Project(upper).is_temp())
        self.assertTrue(base.exists())


class ProbeWarningTests(unittest.TestCase):

    def _src(self, **video):
        v = {"width": 1920, "height": 1080, "hdr": "none", "rotation": 0, "fps": [25, 1],
             "vfr": False, "field_order": "progressive", "start": 0.0, "duration": 100.0, "sar": "1:1",
             "nb_frames": 2500}
        v.update(video)
        a = {"start": 0.0, "fold_mono": False, "dup_mono": False, "channels": 2, "holes": []}
        return {"duration": 100.0, "video": v, "audio": a}

    def _warnings(self, src):
        try:
            return probe.source_warnings(src)
        except (KeyError, TypeError) as exc:
            self.fail("source_warnings needs %s" % exc)

    def test_frame_count_mismatch(self):
        self.assertFalse(any("frames" in w for w in self._warnings(self._src())))
        ws = self._warnings(self._src(nb_frames=2400))
        self.assertTrue(any("the video stream has 2400 frames" in w and "--cfr" in w for w in ws), ws)
        ws = self._warnings(self._src(nb_frames=2600))
        self.assertTrue(any("holds 2600 frames" in w and "edit list hides 4.00 s" in w for w in ws), ws)
        self.assertFalse(any("frames (" in w for w in self._warnings(self._src(nb_frames=2501))))

    def test_audio_holes_warning_and_parser(self):
        src = self._src()
        src["audio"]["holes"] = [[12.5, 0.23]]
        ws = self._warnings(src)
        self.assertTrue(any("timestamp hole" in w and "0.23 s" in w for w in ws), ws)
        csv = "0.000000,0.021333\n0.021333,0.021333\n0.042667,0.021333\n0.300000,0.021333\n" \
              "0.321333,0.200000\n"
        fake = subprocess.CompletedProcess([], 0, stdout=csv, stderr="")
        with mock.patch.object(probe.env, "run", return_value=fake):
            holes = probe.audio_holes("x.mp4", {"codec": "aac", "index": 1}, 0.04)
        self.assertEqual(holes, [(0.064, 0.236), (0.321, 0.2)])
        self.assertEqual(probe.audio_holes("x.mov", {"codec": "pcm_s16le", "index": 1}, 0.04), [])

    def test_cfr_transcode(self):
        vfr = TMP / "vfr_av.mp4"
        if not vfr.exists():
            env.ff(["-f", "lavfi", "-i", "testsrc=size=320x240:rate=25:duration=6",
                    "-f", "lavfi", "-i", "sine=frequency=440:duration=8",
                    "-vf", "setpts='if(gte(N,50),(N+40)/25/TB,N/25/TB)'", "-fps_mode", "passthrough",
                    "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", "-shortest", str(vfr)])
        P = TMP / "cfrproj"
        meta0 = probe.probe_source(vfr)
        self.assertTrue(meta0["video"]["vfr"])
        meta, out, _ = quiet(ingest.ingest, P, str(vfr), cfr=True)
        self.assertFalse(meta["video"]["vfr"], meta["video"])
        self.assertEqual(meta["video"]["fps"], [25, 1])
        self.assertEqual(meta["ingest"]["cfr"]["fps"], [25, 1])
        self.assertTrue(vfr.exists())
        self.assertFalse((P / "source.mp4").is_symlink())
        self.assertIn("re-encoded to constant frame rate", out)

        def audio_md5(p):
            return env.ff(["-i", str(p), "-map", "0:a:0", "-c", "copy", "-f", "md5", "-"]).stdout
        self.assertEqual(audio_md5(vfr), audio_md5(P / "source.mp4"))
        # a CFR source with --cfr is left alone
        meta2, out2, _ = quiet(ingest.ingest, TMP / "cfrproj2", str(av()), cfr=True)
        self.assertNotIn("cfr", meta2["ingest"])


if __name__ == "__main__":
    unittest.main()
