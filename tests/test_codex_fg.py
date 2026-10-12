"""Regression tests for the outside-review fixes in the entry point, ingest, status and install.

Evidence gaps reach the exit code and the status line, `hc cover --at` reads clock times,
`hc all` follows fonts, safe zones, source facts and the QA code, a changed source file stops
the pipeline until it is ingested again, `hc ingest --force` never deletes before the new
source is in place and never touches files it did not make, and the launcher ignores modules
in the folder it is started from.

Everything is synthetic: generated test videos, hand-written transcripts and QA reports.
"""

from __future__ import annotations

import importlib.util
import io
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
from contextlib import redirect_stderr, redirect_stdout
from importlib.machinery import SourceFileLoader
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "scripts"))

import fixtures as F  # noqa: E402
import qa_media as qm  # noqa: E402
from hookcutter import cli, doctor, env, ingest  # noqa: E402
from hookcutter import plan as plan_mod  # noqa: E402
from hookcutter.project import Project  # noqa: E402

SK = HERE.parent
TMP = None

WORDS = [("Learn", 1.0, 1.4), ("it", 1.5, 1.8), ("in", 1.9, 2.2), ("time.", 2.3, 2.8),
         ("Then", 3.6, 4.0), ("teach", 4.1, 4.5), ("it", 4.6, 4.8), ("well.", 4.9, 5.4)]
WORDS_B = [("Other", 1.2, 1.6), ("words", 1.7, 2.1), ("live", 2.2, 2.6), ("here.", 2.7, 3.2),
           ("And", 3.9, 4.2), ("more", 4.3, 4.7), ("of", 4.8, 5.0), ("them.", 5.1, 5.6)]


def setUpModule():
    global TMP
    TMP = Path(tempfile.mkdtemp(prefix="hc-codex-fg-"))


def tearDownModule():
    if TMP and TMP.exists():
        shutil.rmtree(str(TMP), ignore_errors=True)


def quiet(fn, *a, **kw):
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        rv = fn(*a, **kw)
    return rv, out.getvalue(), err.getvalue()


def hc(*argv):
    """cli.main with captured output -> (exit code, stdout, stderr)."""
    return quiet(cli.main, [str(a) for a in argv])


def av():
    return F.make_av_fixture(TMP / "fx", dur=6, flashes=[])


def speech(words=WORDS, name="fx"):
    return F.make_speech_fixture(TMP / name, words, dur=7)


def small_source(name, dur=2.0):
    p = TMP / "media" / name
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists():
        qm.make_source(p, dur=dur, size="320x180")
    return p


def media_in(P):
    """The source.* entries of a project folder that are not json (sorted names)."""
    return sorted(x.name for x in Path(P).iterdir()
                  if x.name.startswith("source.") and not x.name.endswith(".json"))


# ---------------------------------------------------------------- evidence gaps (X1, V01-N3)

class EvidenceGaps(unittest.TestCase):
    def project(self, name, report):
        P = F.make_project(TMP / name, av(), [("Hello", 1.0, 1.4), ("world.", 1.5, 1.9)])
        env.write_json(P / "clips.json", {"version": 1, "clips": {"c01-a": {"pieces": [{"words": [0, 1]}]}}})
        env.write_json(Project(P).qa_path("c01-a", ".json"), report)
        return P

    def test_exit_code_read_back_from_reports(self):
        cases = [({"status": "WARN", "complete": False, "gaps": 1}, 2, 0),
                 ({"status": "WARN", "complete": False}, 2, 0),
                 ({"status": "WARN", "gaps": 2}, 2, 0),
                 ({"status": "INCOMPLETE", "complete": False, "gaps": 1}, 2, 0),
                 ({"status": "WARN", "complete": True, "gaps": 0}, 0, 0),
                 ({"status": "PASS"}, 0, 0),
                 ({"status": "FAIL", "gaps": 1}, 1, 1)]
        for n, (report, with_cov, without) in enumerate(cases):
            proj = Project(self.project("gaps-exit-%d" % n, report))
            self.assertEqual(cli._qa_exit_from_reports(proj, ["c01-a"], True), with_cov, report)
            self.assertEqual(cli._qa_exit_from_reports(proj, ["c01-a"], False), without, report)

    def test_status_says_incomplete(self):
        P = self.project("gaps-status", {"status": "WARN", "complete": False, "gaps": 1})
        rc, out, _ = hc("status", P)
        self.assertEqual(rc, 0)
        row = [ln for ln in out.splitlines() if ln.startswith("c01-a")][0]
        self.assertIn("qa WARN (incomplete)", row)
        env.write_json(Project(P).qa_path("c01-a", ".json"), {"status": "WARN", "complete": True, "gaps": 0})
        rc, out, _ = hc("status", P)
        row = [ln for ln in out.splitlines() if ln.startswith("c01-a")][0]
        self.assertIn("qa WARN", row)
        self.assertNotIn("incomplete", row)

    def test_incomplete_status_is_not_said_twice(self):
        self.assertEqual(cli._qa_word({"status": "INCOMPLETE", "complete": False, "gaps": 3}), "INCOMPLETE")
        self.assertEqual(cli._qa_word({"status": "PASS"}), "PASS")
        self.assertEqual(cli._qa_word({"status": "FAIL", "gaps": 1}), "FAIL (incomplete)")


# ---------------------------------------------------------------- a changed source file (R2)

FRAMING = [{"at_src": 0.0, "mode": "fixed", "x": 0.5, "y": 0.5, "crop": 1.0}]


def two_piece_spec():
    return {"version": 1, "defaults": {"styles": ["clean"], "formats": ["vertical"],
                                       "length": {"vertical": [1, 1, 10, 60]}},
            "clips": {"c1": {"title": "One", "pieces": [{"words": [0, 3]}, {"words": [4, 7]}],
                             "framing": FRAMING}}}


class SourceChanged(unittest.TestCase):
    def linked(self, name):
        """A project whose source links to a file outside it, plus that file's path."""
        root = TMP / name
        ext = root / "ext" / "episode.mp4"
        ext.parent.mkdir(parents=True)
        shutil.copy2(str(speech(WORDS, "fxa")), str(ext))
        P = root / "P"
        quiet(ingest.ingest, P, str(ext))
        fp = env.read_json(P / "source.json")["fingerprint"]
        env.write_json(P / "transcript.json", F.fake_transcript(WORDS, fingerprint=fp))
        env.write_json(P / "clips.json", two_piece_spec())
        return P, ext

    def replace(self, ext, words=WORDS_B, name="fxb"):
        tmp = ext.with_name("new.mp4")
        shutil.copy2(str(speech(words, name)), str(tmp))
        os.replace(str(tmp), str(ext))

    def test_replaced_source_stops_the_pipeline_until_it_is_ingested_again(self):
        P, ext = self.linked("drift-run")
        rc, _, err = hc("render", P)
        self.assertEqual(rc, 0, err)
        proj = Project(P)
        log_path = proj.render_path("c1", "vertical", "clean", ".log.json")
        created = env.read_json(log_path)["created"]
        segs = sorted(x.name for x in proj.work_dir("c1", "vertical").iterdir())
        self.assertIsNone(proj.source_drift())

        self.replace(ext)
        why = Project(P).source_drift()
        self.assertIsNotNone(why)
        self.assertIn("hc ingest", why)
        for argv in (["plan", P], ["render", P], ["all", P, "--no-asr"], ["qa", P, "--no-asr"],
                     ["shots", P, "c1"], ["cover", P], ["transcribe", P], ["peek", P, "0", "1"]):
            rc, _, err = hc(*argv)
            self.assertEqual(rc, 1, argv)
            self.assertIn("changed since ingest", err, argv)
        self.assertEqual(env.read_json(log_path)["created"], created)
        self.assertEqual(sorted(x.name for x in proj.work_dir("c1", "vertical").iterdir()), segs)
        rc, out, _ = hc("status", P)
        self.assertEqual(rc, 0)
        self.assertIn("CHANGED since ingest", out)
        row = [ln for ln in out.splitlines() if ln.startswith("c1")][0]
        self.assertIn("plan STALE(source)", row)
        self.assertIn("render 1/1 STALE(source)", row)

        # the remedy the message names: ingest again (a refresh), transcribe, render
        quiet(ingest.ingest, P, str(ext))
        fp = env.read_json(P / "source.json")["fingerprint"]
        env.write_json(P / "transcript.json", F.fake_transcript(WORDS_B, fingerprint=fp))
        self.assertIsNone(Project(P).source_drift())
        rc, _, err = hc("render", P)
        self.assertEqual(rc, 0, err)
        log = env.read_json(log_path)
        self.assertNotEqual(log["created"], created)
        self.assertTrue(log["segments"])
        self.assertEqual([x["cached"] for x in log["segments"]], [False] * len(log["segments"]))

    def test_guard_scope(self):
        P, ext = self.linked("drift-scope")
        raw = env.read_json(P / "clips.json")
        raw["clips"]["cap1"] = {"kind": "caption_only", "source_file": "own.mp4"}
        env.write_json(P / "clips.json", raw)
        self.replace(ext)
        parse = cli.build_parser().parse_args
        for argv in (["render", str(P)], ["render", str(P), "c1"], ["render", str(P), "c1", "cap1"],
                     ["plan", str(P)], ["all", str(P)], ["qa", str(P)], ["shots", str(P), "c1"],
                     ["transcribe", str(P)], ["peek", str(P), "0", "1"]):
            with self.assertRaises(env.HCError) as cm:
                cli._require_source_current(parse(argv))
            self.assertEqual(cm.exception.code, 1, argv)
        # commands that never decode the source, a file of their own, caption-only clips
        for argv in (["status", str(P)], ["export", str(P)], ["find", str(P), "x"], ["words", str(P), "0", "1"],
                     ["ingest", str(P), str(ext)], ["render", str(P), "cap1"], ["qa", str(P), "cap1"],
                     ["peek", str(P), "0", "1", "--file", "x.mp4"],
                     ["transcribe", str(P), "--file", "own.mp4", "--clip", "cap1"]):
            cli._require_source_current(parse(argv))

    def test_nothing_to_compare_is_not_drift(self):
        # no source.json at all (a project of caption-only clips)
        bare = TMP / "drift-bare" / "P"
        bare.mkdir(parents=True)
        self.assertIsNone(Project(bare).source_drift())
        cli._require_source_current(cli.build_parser().parse_args(["render", str(bare)]))
        # a broken link has its own error
        P, ext = self.linked("drift-broken")
        ext.unlink()
        self.assertIsNone(Project(P).source_drift())
        cli._require_source_current(cli.build_parser().parse_args(["render", str(P)]))
        with self.assertRaises(env.HCError) as cm:
            Project(P).source_path()
        self.assertIn("broken link", cm.exception.msg)
        # a source.json without a fingerprint (hand-made projects)
        P2 = F.make_project(TMP / "drift-nofp", av())
        meta = env.read_json(P2 / "source.json")
        meta["fingerprint"] = None
        env.write_json(P2 / "source.json", meta)
        self.assertIsNone(Project(P2).source_drift())
        # and an untouched one is current
        self.assertIsNone(Project(F.make_project(TMP / "drift-same", av())).require_source_current().source_drift())


# ---------------------------------------------------------------- ingest --force (I2, I3, I4, I5, V18-N1)

URL = "https://example.com/watch?v=x"
SIDECARS = ("source.srt", "source.notes.md", "source.backup.zip", "source.wav")


class IngestReplace(unittest.TestCase):
    def copied(self, name):
        """A healthy project whose only copy of the source is P/source.mp4."""
        P = TMP / name / "P"
        quiet(ingest.ingest, P, str(small_source("a.mp4")), copy=True)
        self.assertEqual(media_in(P), ["source.mp4"])
        return P

    def intact(self, P):
        self.assertEqual(media_in(P), ["source.mp4"])
        self.assertFalse((P / "source.mp4").is_symlink())
        self.assertEqual((P / "source.mp4").stat().st_size, small_source("a.mp4").stat().st_size)
        self.assertEqual(Project(P).source_path(), P / "source.mp4")
        self.assertEqual(env.read_json(P / "source.json")["path"], "source.mp4")
        self.assertIsNone(Project(P).source_drift())

    def test_failed_download_keeps_the_old_source(self):
        P = self.copied("force-url")
        with mock.patch.object(ingest, "_ytdlp_available", return_value=False):
            with self.assertRaises(env.HCError) as cm:
                quiet(ingest.ingest, P, URL, force=True)
        self.assertEqual(cm.exception.code, 3)
        self.intact(P)
        with mock.patch.object(ingest, "_ytdlp_available", return_value=True), \
                mock.patch.object(ingest, "_ytdlp_help", return_value=""), \
                mock.patch.object(ingest.subprocess, "call", return_value=1):
            with self.assertRaises(env.HCError) as cm:
                quiet(ingest.ingest, P, URL, force=True)
        self.assertEqual(cm.exception.code, 3)
        self.intact(P)

    def test_download_without_sound_never_replaces_a_source(self):
        P = self.copied("force-url-mute")
        mute = TMP / "media" / "mute.mp4"
        if not mute.exists():
            env.ff(["-f", "lavfi", "-i", "testsrc2=size=320x180:rate=25:duration=1", "-an",
                    "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", str(mute)])

        def fake_call(cmd, **kw):
            out = cmd[cmd.index("-o") + 1].replace("%(ext)s", "mp4")
            shutil.copy2(str(mute), out)
            return 0
        with mock.patch.object(ingest, "_ytdlp_available", return_value=True), \
                mock.patch.object(ingest, "_ytdlp_help", return_value=""), \
                mock.patch.object(ingest.subprocess, "call", side_effect=fake_call):
            with self.assertRaises(env.HCError) as cm:
                quiet(ingest.ingest, P, URL, force=True)
        self.assertIn("no audio stream", cm.exception.msg)
        self.intact(P)

    def test_force_of_a_file_that_is_not_a_video_changes_nothing(self):
        P = self.copied("force-text")
        notes = TMP / "media" / "notes.txt"
        notes.write_text("not a video\n", encoding="utf-8")
        with self.assertRaises(env.HCError):
            quiet(ingest.ingest, P, str(notes), force=True)
        self.intact(P)
        self.assertFalse(os.path.lexists(str(P / "source.txt")))

    def test_failed_copy_keeps_the_old_source_and_leaves_no_temp_file(self):
        real = shutil.copy2

        def half_copy(src, dst, **kw):
            with open(str(dst), "wb") as fh:
                fh.write(b"half")
            raise OSError(28, "No space left on device", str(dst))
        for n, other in enumerate((small_source("b.mov"), small_source("c.mp4"))):
            P = self.copied("force-nospace-%d" % n)
            with mock.patch.object(shutil, "copy2", side_effect=half_copy):
                with self.assertRaises(OSError):
                    quiet(ingest.ingest, P, str(other), force=True, copy=True)
            self.assertIs(shutil.copy2, real)
            self.intact(P)
            self.assertEqual([x.name for x in P.iterdir() if x.name.endswith(".tmp")], [])

    def test_force_removes_the_recorded_source_and_nothing_else(self):
        P = self.copied("force-ok")
        for name in SIDECARS:
            (P / name).write_bytes(b"mine")
        meta, _, _ = quiet(ingest.ingest, P, str(small_source("b.mov")), force=True, copy=True)
        self.assertEqual(meta["path"], "source.mov")
        self.assertEqual(env.read_json(P / "source.json")["path"], "source.mov")
        self.assertFalse(os.path.lexists(str(P / "source.mp4")))
        self.assertTrue((P / "source.mov").is_file())
        for name in SIDECARS:
            self.assertEqual((P / name).read_bytes(), b"mine", name)
        self.assertEqual(Project(P).source_path(), P / "source.mov")

    def test_cfr_removes_the_file_it_transcoded_and_nothing_else(self):
        for n, copy in enumerate((True, False)):
            P = TMP / ("cfr-sidecars-%d" % n) / "P"
            src = small_source("a.mp4")
            meta, _, _ = quiet(ingest.ingest, P, str(src), copy=copy)
            for name in SIDECARS:
                (P / name).write_bytes(b"mine")
            size = src.stat().st_size
            out, _, _ = quiet(ingest.to_cfr, Project(P), P / "source.mp4", meta)
            self.assertEqual(out, P / "source.mp4")
            self.assertTrue(out.is_file() and not out.is_symlink())
            for name in SIDECARS:
                self.assertEqual((P / name).read_bytes(), b"mine", name)
            self.assertEqual(src.stat().st_size, size)      # a linked original is never written to

    def test_copy_never_writes_through_a_link_at_a_temp_name(self):
        P = TMP / "copy-planted" / "P"
        P.mkdir(parents=True)
        victim = TMP / "copy-planted" / "victim.txt"
        victim.write_text("keep me\n", encoding="utf-8")
        os.symlink(str(victim), str(P / "source.mp4.tmp"))
        src = small_source("a.mp4")
        meta, _, _ = quiet(ingest.ingest, P, str(src), copy=True)
        self.assertEqual(meta["ingest"]["method"], "copy")
        self.assertEqual(victim.read_text(encoding="utf-8"), "keep me\n")
        self.assertTrue((P / "source.mp4").is_file())
        self.assertFalse((P / "source.mp4").is_symlink())
        self.assertEqual((P / "source.mp4").stat().st_size, src.stat().st_size)
        self.assertEqual([x.name for x in P.iterdir() if x.name.endswith(".tmp")], [])

    def test_no_symlinks_falls_back_to_the_same_safe_copy(self):
        P = TMP / "copy-fallback" / "P"
        src = small_source("a.mp4")
        with mock.patch.object(os, "symlink", side_effect=OSError(1, "not permitted")):
            meta, _, _ = quiet(ingest.ingest, P, str(src))
        self.assertEqual(meta["ingest"]["method"], "copy")
        self.assertFalse((P / "source.mp4").is_symlink())
        self.assertEqual([x.name for x in P.iterdir() if x.name.endswith(".tmp")], [])

    def in_place(self, name):
        P = TMP / name / "P"
        P.mkdir(parents=True)
        shutil.copy2(str(small_source("a.mp4")), str(P / "source.mp4"))
        meta, _, _ = quiet(ingest.ingest, P, str(P / "source.mp4"))
        self.assertEqual(meta["ingest"]["method"], "in-place")
        return P

    def test_original_ingested_in_place_is_never_deleted(self):
        P = self.in_place("inplace-other-name")
        meta, _, err = quiet(ingest.ingest, P, str(small_source("b.mov")), force=True)
        self.assertEqual(meta["path"], "source.mov")
        self.assertIn("kept source.mp4", err)
        self.assertTrue((P / "source.mp4").is_file())
        self.assertEqual((P / "source.mp4").stat().st_size, small_source("a.mp4").stat().st_size)
        # later, another file of the same name: the original is no longer in source.json,
        # and is still not replaced
        with self.assertRaises(env.HCError) as cm:
            quiet(ingest.ingest, P, str(small_source("c.mp4")), force=True)
        self.assertEqual(cm.exception.code, 1)
        self.assertIn("may be your own file", cm.exception.msg)
        self.assertEqual((P / "source.mp4").stat().st_size, small_source("a.mp4").stat().st_size)
        self.assertEqual(env.read_json(P / "source.json")["path"], "source.mov")

    def test_original_ingested_in_place_is_not_overwritten_by_a_same_name_source(self):
        P = self.in_place("inplace-same-name")
        for kw in ({}, {"copy": True}):
            with self.assertRaises(env.HCError) as cm:
                quiet(ingest.ingest, P, str(small_source("c.mp4")), force=True, **kw)
            self.assertEqual(cm.exception.code, 1)
        with mock.patch.object(ingest, "_download", side_effect=AssertionError("must not download")):
            with self.assertRaises(env.HCError) as cm:
                quiet(ingest.ingest, P, URL, force=True)
        self.assertIn("ingested in place", cm.exception.msg)
        self.assertEqual((P / "source.mp4").stat().st_size, small_source("a.mp4").stat().st_size)
        self.assertEqual(env.read_json(P / "source.json")["ingest"]["method"], "in-place")
        # the same file again is still a refresh, forced or not
        quiet(ingest.ingest, P, str(P / "source.mp4"))
        quiet(ingest.ingest, P, str(P / "source.mp4"), force=True)
        self.assertTrue((P / "source.mp4").is_file())

    def test_unfinished_copy_of_the_same_file_is_replaced(self):
        """A copy whose ingest never wrote source.json (interrupted) does not block a re-run."""
        P = TMP / "copy-rerun" / "P"
        P.mkdir(parents=True)
        src = small_source("a.mp4")
        shutil.copy2(str(src), str(P / "source.mp4"))
        meta, _, _ = quiet(ingest.ingest, P, str(src))
        self.assertEqual(meta["ingest"]["method"], "symlink")

    def test_two_ingests_of_one_project_take_turns(self):
        P = TMP / "lock-same" / "P"
        src = small_source("a.mp4")
        gate, copying = threading.Event(), threading.Event()
        real_copy, real_finish = shutil.copy2, ingest._finish
        methods, errors, results = [], [], {}

        def slow_copy(a, b, **kw):
            rv = real_copy(a, b, **kw)
            copying.set()
            gate.wait(30)
            return rv

        def finish(proj, dest, original, old, method, url_bits, cfr=False):
            methods.append(method)
            return real_finish(proj, dest, original, old, method, url_bits, cfr=cfr)

        def run(tag):
            try:
                results[tag] = ingest.ingest(P, str(src), copy=True)
            except BaseException as exc:           # noqa: B902 (reported to the main thread)
                errors.append((tag, exc))

        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err), \
                mock.patch.object(shutil, "copy2", side_effect=slow_copy), \
                mock.patch.object(ingest, "_finish", side_effect=finish):
            a = threading.Thread(target=run, args=("a",))
            a.start()
            self.assertTrue(copying.wait(30))
            b = threading.Thread(target=run, args=("b",))
            b.start()
            b.join(1.0)
            self.assertTrue(b.is_alive(), "the second ingest must wait for the first")
            gate.set()
            a.join(60)
            b.join(60)
        self.assertEqual(errors, [])
        self.assertEqual(methods, ["copy", "refresh"])
        self.assertEqual(media_in(P), ["source.mp4"])
        self.assertEqual([x.name for x in P.iterdir() if x.name.endswith(".tmp")], [])
        self.assertTrue((P / "cache" / "ingest.lock").exists())

    def test_second_ingest_of_another_file_is_told_the_project_is_taken(self):
        P = TMP / "lock-other" / "P"
        gate, copying = threading.Event(), threading.Event()
        real_copy = shutil.copy2
        errors, results = {}, {}

        def slow_copy(a, b, **kw):
            rv = real_copy(a, b, **kw)
            if not copying.is_set():
                copying.set()
                gate.wait(30)
            return rv

        def run(tag, src):
            try:
                results[tag] = ingest.ingest(P, str(src), copy=True)
            except BaseException as exc:           # noqa: B902
                errors[tag] = exc

        out, err = io.StringIO(), io.StringIO()
        with redirect_stdout(out), redirect_stderr(err), \
                mock.patch.object(shutil, "copy2", side_effect=slow_copy):
            a = threading.Thread(target=run, args=("a", small_source("a.mp4")))
            a.start()
            self.assertTrue(copying.wait(30))
            b = threading.Thread(target=run, args=("b", small_source("b.mov")))
            b.start()
            b.join(1.0)
            gate.set()
            a.join(60)
            b.join(60)
        self.assertEqual(list(errors), ["b"])
        self.assertIsInstance(errors["b"], env.HCError)
        self.assertIn("already holds", errors["b"].msg)
        self.assertEqual(media_in(P), ["source.mp4"])
        self.assertEqual(env.read_json(P / "source.json")["path"], "source.mp4")


# ---------------------------------------------------------------- --cfr and the transcript (R7, V06-N2, V16-N1)

class CfrAndTranscript(unittest.TestCase):
    def test_cfr_records_the_file_it_was_made_from(self):
        vfr = TMP / "media" / "vfr.mp4"
        vfr.parent.mkdir(parents=True, exist_ok=True)
        if not vfr.exists():
            env.ff(["-f", "lavfi", "-i", "testsrc=size=320x240:rate=25:duration=6",
                    "-f", "lavfi", "-i", "sine=frequency=440:duration=8",
                    "-vf", "setpts='if(gte(N,50),(N+40)/25/TB,N/25/TB)'", "-fps_mode", "passthrough",
                    "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac", "-shortest", str(vfr)])
        P = TMP / "cfr-from" / "P"
        meta0, _, _ = quiet(ingest.ingest, P, str(vfr))
        self.assertTrue(meta0["video"]["vfr"])
        env.write_json(P / "transcript.json", F.fake_transcript(WORDS, fingerprint=meta0["fingerprint"]))
        meta, out, _ = quiet(ingest.ingest, P, str(vfr), cfr=True)
        rec = env.read_json(P / "source.json")["ingest"]["cfr"]
        self.assertEqual(rec["from_fingerprint"], meta0["fingerprint"])
        self.assertNotEqual(meta["fingerprint"], meta0["fingerprint"])
        self.assertEqual(rec["fps"], [25, 1])
        self.assertIn("transcript.json stays valid", out)
        self.assertNotIn("re-run transcribe", out)
        self.assertTrue(vfr.exists())
        # a later refresh keeps the record and still does not ask for a new transcript
        meta2, out2, _ = quiet(ingest.ingest, P, str(vfr))
        self.assertEqual(meta2["ingest"]["cfr"]["from_fingerprint"], meta0["fingerprint"])
        self.assertNotIn("re-run transcribe", out2)
        check = getattr(plan_mod, "_require_transcript_source", None)
        if callable(check):
            check(Project(P))

    def test_cfr_acts_on_a_file_whose_frames_go_missing_late(self):
        """Two frames dropped after the first 300 packets: the average rate moves 0.4 %, so the
        file does not read as variable frame rate, and the ingest warning sends it to --cfr."""
        late = TMP / "media" / "late_drop.mp4"
        late.parent.mkdir(parents=True, exist_ok=True)
        if not late.exists():
            env.ff(["-f", "lavfi", "-i", "testsrc=size=320x240:rate=25:duration=20",
                    "-f", "lavfi", "-i", "sine=frequency=440:duration=20",
                    "-vf", "select='not(between(n\\,350\\,351))'", "-fps_mode", "passthrough",
                    "-c:v", "libx264", "-preset", "ultrafast", "-g", "25", "-c:a", "aac", "-t", "20", str(late)])
        P = TMP / "cfr-late" / "P"
        meta0, out0, _ = quiet(ingest.ingest, P, str(late))
        self.assertFalse(meta0["video"]["vfr"], meta0["video"])
        self.assertIn("frames are missing", out0)
        self.assertIn("--cfr", out0)
        meta, out, err = quiet(ingest.ingest, P, str(late), cfr=True)
        self.assertNotIn("changes nothing", err)
        self.assertIn("cfr", meta["ingest"])
        v = meta["video"]
        self.assertEqual(v["nb_frames"], int(round(v["duration"] * v["fps"][0] / v["fps"][1])))
        self.assertFalse(any("frames are missing" in w for w in meta["warnings"]), meta["warnings"])
        # a constant-rate file is still left alone
        meta2, _, err2 = quiet(ingest.ingest, TMP / "cfr-none" / "P", str(small_source("a.mp4")), cfr=True)
        self.assertNotIn("cfr", meta2["ingest"])
        self.assertIn("changes nothing", err2)

    def test_transcript_of_another_file_is_said_on_every_ingest(self):
        a, b = small_source("a.mp4"), small_source("c.mp4", dur=3.0)
        P = TMP / "warn-keeps" / "P"
        meta_a, _, _ = quiet(ingest.ingest, P, str(a))
        env.write_json(P / "transcript.json", F.fake_transcript(WORDS, fingerprint=meta_a["fingerprint"]))
        _, out, _ = quiet(ingest.ingest, P, str(b), force=True)
        self.assertIn("re-run transcribe", out)
        meta_b, out, _ = quiet(ingest.ingest, P, str(b))                 # a refresh: still said
        self.assertIn("re-run transcribe", out)
        self.assertTrue(any("re-run transcribe" in w for w in meta_b["warnings"]))
        env.write_json(P / "transcript.json", F.fake_transcript(WORDS, fingerprint=meta_b["fingerprint"]))
        meta_b, out, _ = quiet(ingest.ingest, P, str(b))
        self.assertNotIn("re-run transcribe", out)
        self.assertFalse(any("re-run transcribe" in w for w in meta_b["warnings"]))


# ---------------------------------------------------------------- hc cover --at (C7)

class CoverAtArgument(unittest.TestCase):
    def test_clock_times_and_seconds(self):
        ps = cli.build_parser()
        for text, want in (("3.5", 3.5), ("0:12", 12.0), ("00:12.40", 12.4), ("1:02:03.4", 3723.4), ("12", 12.0)):
            self.assertAlmostEqual(ps.parse_args(["cover", "P", "c1", "--at", text]).at, want, places=6)
        self.assertIsNone(ps.parse_args(["cover", "P", "c1"]).at)

    def test_bad_time_is_a_usage_error(self):
        for text in ("abc", "-1", "1:2:3:4", ""):
            err = io.StringIO()
            with redirect_stderr(err), self.assertRaises(SystemExit) as cm:
                cli.build_parser().parse_args(["cover", "P", "c1", "--at", text])
            self.assertEqual(cm.exception.code, 1, text)
            self.assertIn("argument --at", err.getvalue())


# ---------------------------------------------------------------- cover frame index in status (C3)

class CoverFrameIndex(unittest.TestCase):
    def reasons(self, name, fps, clip_at, json_at):
        P = TMP / name
        proj = Project(P, create=True)
        env.write_json(proj.cover_path("c1", "clean", "json"), {"manual": True, "at_out": json_at})
        clip = {"cover": {"at_out": clip_at}}
        return cli.cover_stale_reasons(proj, clip, clip, "c1", "clean", {"fps": list(fps)}, None, False)

    def test_time_copied_from_a_cover_json_maps_back_to_its_frame(self):
        # frame 13 at 30 fps is stored as 0.433333, a hair under 13/30
        self.assertEqual(self.reasons("cfi-a", (30, 1), 0.433333, 0.433333), [])
        self.assertEqual(self.reasons("cfi-b", (30, 1), 0.45, 0.433333), [])
        self.assertEqual(self.reasons("cfi-c", (30000, 1001), 0.467133, 0.467133), [])
        self.assertEqual(self.reasons("cfi-d", (30000, 1001), 0.5, 0.467133), [])
        self.assertEqual(self.reasons("cfi-e", (25, 1), 0.44, 0.44), [])
        # a cover that was cut one frame early is not the cover of this time
        self.assertEqual(self.reasons("cfi-f", (30, 1), 0.433333, 0.4), ["at_out"])

    def test_same_expression_as_hc_cover(self):
        import math
        from hookcutter import cover_frames as cf
        self.assertEqual(cli.COVER_FRAME_TOL, cf.FRAME_TOL)
        for num, den in ((30, 1), (30000, 1001), (24, 1), (24000, 1001), (60, 1), (25, 1), (120, 1)):
            for n in range(0, 400):
                t = round(n * den / num, 6)
                mine = int(math.floor(t * num / den + cli.COVER_FRAME_TOL))
                self.assertEqual(mine, n, (num, den, n))
                self.assertEqual(mine, cf.frame_index(t, (num, den)))


# ---------------------------------------------------------------- hc all keys (R1, V05-N1, V08-N1, R4)

class AllFollowsItsInputs(unittest.TestCase):
    def project(self, name, style="clean", preset=None, font=None):
        P = F.make_project(TMP / name, av(), [("Hello", 1.0, 1.4), ("world.", 1.5, 1.9)])
        env.write_json(P / "clips.json", {"version": 1, "defaults": {"styles": [style]},
                                          "clips": {"c01-a": {"pieces": [{"words": [0, 1]}]}}})
        if preset:
            env.write_json(P / "presets" / ("%s.json" % style), preset)
        if font:
            (P / "fonts").mkdir(exist_ok=True)
            shutil.copy2(str(SK / "fonts" / font), str(P / "fonts" / "Brand.ttf"))
        return P

    def run_all(self, P, style="clean"):
        """One `hc all` with steps that write what the real ones write -> the steps it ran."""
        proj = Project(P)
        log = []

        def dispatch(a):
            log.append(a.cmd)
            if a.cmd == "plan":
                for cid in a.clips:
                    env.write_json(proj.plan_path(cid), {"kind": "cut", "clip": cid, "shots_hash": None})
            elif a.cmd == "render":
                proj.render_path(a.clips[0], "vertical", style).write_bytes(b"x")
            elif a.cmd == "cover":
                for cid in a.clips:
                    for k in ("main", "grid", "mobile", "json"):
                        proj.cover_path(cid, style, k).write_bytes(b"x")
            elif a.cmd == "qa":
                for cid in a.clips:
                    env.write_json(proj.qa_path(cid, ".json"), {"status": "PASS"})
            return 0
        args = cli.build_parser().parse_args(["all", str(P)])
        with mock.patch.object(cli, "dispatch", side_effect=dispatch), \
                mock.patch.object(cli, "_render_code_hash", return_value="code"):
            rc, _, _ = quiet(cli.cmd_all, args)
        self.assertEqual(rc, 0)
        return log

    def test_qa_runs_again_when_the_qa_code_changes(self):
        P = self.project("all-qa-code")
        self.assertEqual(self.run_all(P), ["plan", "render", "cover", "qa", "export"])
        self.assertEqual(self.run_all(P), ["plan", "export"])
        with mock.patch.object(env, "package_digest", return_value="other"):
            self.assertEqual(self.run_all(P), ["plan", "qa", "export"])
            self.assertEqual(self.run_all(P), ["plan", "export"])

    def test_render_is_asked_again_when_the_source_facts_change(self):
        P = self.project("all-facts")
        self.run_all(P)
        self.assertEqual(self.run_all(P), ["plan", "export"])
        meta = env.read_json(P / "source.json")
        meta["audio"]["fold_mono"] = not meta["audio"].get("fold_mono")
        env.write_json(P / "source.json", meta)             # same file, same fingerprint
        self.assertEqual(self.run_all(P)[:2], ["plan", "render"])
        self.assertEqual(self.run_all(P), ["plan", "export"])
        meta["video"]["sar"] = "4:3"
        env.write_json(P / "source.json", meta)
        self.assertEqual(self.run_all(P)[:2], ["plan", "render"])

    def test_render_is_asked_again_when_a_font_or_safe_zone_changes(self):
        from hookcutter import captions
        P = self.project("all-assets")
        assets = {"fonts": {"Brand.ttf": "aaa"}, "safezone": {"top": 1, "bottom": 2, "left": 3, "right": 4}}
        with mock.patch.object(captions, "render_assets", create=True, side_effect=lambda *a: dict(assets)) as fn:
            self.run_all(P)
            self.assertEqual(self.run_all(P), ["plan", "export"])
            preset, clip, fmt, root = fn.call_args[0]
            self.assertIn("caption", preset)
            self.assertIn("pieces", clip)
            self.assertEqual((fmt, Path(root)), ("vertical", Project(P).root))
            assets["fonts"] = {"Brand.ttf": "bbb"}
            self.assertEqual(self.run_all(P)[:2], ["plan", "render"])
            self.assertEqual(self.run_all(P), ["plan", "export"])
            assets["safezone"] = dict(assets["safezone"], right=350)
            self.assertEqual(self.run_all(P)[:2], ["plan", "render"])
        # assets that cannot be computed never break `all`: render decides for itself
        with mock.patch.object(captions, "render_assets", create=True, side_effect=RuntimeError("boom")):
            self.assertEqual(cli._asset_hashes(Project(P), {}, ["clean"], ["vertical"]), "unknown")
            self.assertEqual(self.run_all(P)[:2], ["plan", "render"])
            self.assertEqual(self.run_all(P), ["plan", "export"])

    def test_replaced_caption_font_file_is_seen(self):
        from hookcutter import captions, typeset
        if not callable(getattr(captions, "render_assets", None)):
            self.skipTest("captions.render_assets is not in this build")
        P = self.project("all-font-file", style="brand", font="Anton-Regular.ttf",
                         preset={"extends": "clean", "caption": {"font": "Brand.ttf"},
                                 "hook_card": {"font": "Brand.ttf"}})
        self.run_all(P, "brand")
        self.assertEqual(self.run_all(P, "brand"), ["plan", "export"])
        shutil.copy2(str(SK / "fonts" / "ArchivoBlack-Regular.ttf"), str(P / "fonts" / "Brand.ttf"))
        for cache in ("_load", "_cmap"):
            clear = getattr(getattr(typeset, cache, None), "cache_clear", None)
            if callable(clear):
                clear()
        self.assertEqual(self.run_all(P, "brand")[:2], ["plan", "render"])

    def test_cover_is_made_again_when_a_cover_font_file_is_replaced(self):
        P = self.project("all-cover-font", style="brand", font="Anton-Regular.ttf",
                         preset={"extends": "clean", "cover": {"headline_font": "Brand.ttf"}})
        proj = Project(P)
        before = cli._cover_font_hashes(proj, ["brand"])
        self.assertEqual(before["brand"]["Brand.ttf"], env.sha256_file(SK / "fonts" / "Anton-Regular.ttf"))
        self.assertEqual(len(before["brand"]), 2)            # the headline font and the lockup font
        self.run_all(P, "brand")
        self.assertEqual(self.run_all(P, "brand"), ["plan", "export"])
        shutil.copy2(str(SK / "fonts" / "ArchivoBlack-Regular.ttf"), str(P / "fonts" / "Brand.ttf"))
        self.assertNotEqual(cli._cover_font_hashes(proj, ["brand"]), before)
        self.assertEqual(self.run_all(P, "brand"), ["plan", "cover", "export"])
        self.assertEqual(self.run_all(P, "brand"), ["plan", "export"])
        (P / "fonts" / "Brand.ttf").unlink()
        self.assertEqual(cli._cover_font_hashes(proj, ["brand"])["brand"]["Brand.ttf"], "missing")
        self.assertEqual(cli._cover_font_hashes(proj, ["no-such-style"]), "unknown")


class PackageDigest(unittest.TestCase):
    def test_follows_every_module_and_the_version(self):
        copy = TMP / "pkg-copy"
        shutil.copytree(str(SK / "scripts" / "hookcutter"), str(copy),
                        ignore=shutil.ignore_patterns("__pycache__"))
        d0 = env.package_digest(copy)
        self.assertEqual(len(d0), 64)
        self.assertEqual(d0, env.package_digest(SK / "scripts" / "hookcutter"))
        self.assertEqual(env.package_digest(), env.package_digest(SK / "scripts" / "hookcutter"))
        for n, name in enumerate(("qa.py", "claims.py", "textnorm.py")):
            other = TMP / ("pkg-copy-%d" % n)
            shutil.copytree(str(copy), str(other))
            with open(str(other / name), "ab") as fh:
                fh.write(b"\n")
            self.assertNotEqual(env.package_digest(other), d0, name)
        renamed = TMP / "pkg-copy-renamed"
        shutil.copytree(str(copy), str(renamed))
        os.replace(str(renamed / "textnorm.py"), str(renamed / "textnorm2.py"))
        self.assertNotEqual(env.package_digest(renamed), d0)
        with mock.patch.object(env, "__version__", "9.9.9"):
            env._PACKAGE_DIGESTS.pop(str(copy), None)
            self.assertNotEqual(env.package_digest(copy), d0)
        env._PACKAGE_DIGESTS.pop(str(copy), None)


# ---------------------------------------------------------------- status: transcript and comment keys (R6, V07-N2, P7)

class StatusPlanState(unittest.TestCase):
    def project(self, name, clips, plans, tr_sha="T1"):
        P = F.make_project(TMP / name, av(), [("Hello", 1.0, 1.4), ("world.", 1.5, 1.9)])
        tr = env.read_json(P / "transcript.json")
        tr["sha256"] = tr_sha
        env.write_json(P / "transcript.json", tr)
        env.write_json(P / "clips.json", {"version": 1, "clips": clips})
        for cid, plan in plans.items():
            env.write_json(Project(P).plan_path(cid), plan)
        return P

    def state(self, P, cid):
        proj = Project(P)
        return cli._plan_state(proj, None, cid, proj.source(), proj.transcript())[0]

    def test_cut_plan_asks_the_plan_module_whether_the_transcript_changed(self):
        P = self.project("ps-cut", {"c1": {"pieces": [{"words": [0, 1]}]}},
                         {"c1": {"kind": "cut", "transcript_sha": "T1", "words": [{"i": 0, "sp": "Host"}]}})
        with mock.patch.object(plan_mod, "transcript_changed", create=True, return_value=False) as fn:
            self.assertFalse(self.state(P, "c1").startswith("STALE"))
            plan, tr = fn.call_args[0]
            self.assertEqual((plan["transcript_sha"], tr["sha256"]), ("T1", "T1"))
        # same sha, another speaker label on one of the plan's words: only the plan module can tell
        with mock.patch.object(plan_mod, "transcript_changed", create=True, return_value=True):
            self.assertTrue(self.state(P, "c1").startswith("STALE(transcript)"), self.state(P, "c1"))
        with mock.patch.object(plan_mod, "transcript_changed", create=True, side_effect=RuntimeError("x")):
            self.assertFalse(self.state(P, "c1").startswith("STALE"))   # falls back to the sha

    def test_caption_only_plan_is_compared_with_its_own_transcript(self):
        clips = {"cap1": {"kind": "caption_only", "source_file": "own.mp4"}}
        P = self.project("ps-cap", clips, {"cap1": {"kind": "caption_only", "transcript_sha": "C1"}}, tr_sha="T1")
        proj = Project(P)
        own = dict(env.read_json(P / "transcript.json"), sha256="C1")
        env.write_json(proj.root / "transcripts" / "cap1.json", own)
        self.assertEqual(proj.transcript_path("cap1"), proj.root / "transcripts" / "cap1.json")
        self.assertNotIn("transcript", self.state(P, "cap1"))      # the project transcript is not its own
        env.write_json(proj.root / "transcripts" / "cap1.json", dict(own, sha256="C2"))
        self.assertTrue(self.state(P, "cap1").startswith("STALE(transcript)"), self.state(P, "cap1"))
        rc, out, _ = hc("status", P)
        self.assertEqual(rc, 0)
        self.assertIn("STALE(transcript)", [ln for ln in out.splitlines() if ln.startswith("cap1")][0])

    def test_relabelled_speakers_make_the_plan_stale(self):
        if not callable(getattr(plan_mod, "transcript_changed", None)):
            self.skipTest("plan.transcript_changed is not in this build")
        words = [(w, s, e, "Host" if i < 4 else "Guest") for i, (w, s, e) in enumerate(WORDS)]
        P = F.make_project(TMP / "ps-labels", speech(WORDS, "fxa"), words)
        spec = two_piece_spec()
        spec["speakers"] = {"Host": {"x": 0.3}, "Guest": {"x": 0.7}}
        del spec["clips"]["c1"]["framing"]
        env.write_json(P / "clips.json", spec)
        plan_mod.build_plan(Project(P), "c1")
        self.assertFalse(self.state(P, "c1").startswith("STALE"), self.state(P, "c1"))
        tr = env.read_json(P / "transcript.json")
        sha = tr["sha256"]
        for w in tr["words"]:
            w["sp"] = "Guest" if w["sp"] == "Host" else "Host"
        env.write_json(P / "transcript.json", tr)
        self.assertEqual(env.read_json(P / "transcript.json")["sha256"], sha)
        self.assertTrue(self.state(P, "c1").startswith("STALE(transcript)"), self.state(P, "c1"))

    def test_comment_keys_are_not_clips(self):
        P = F.make_project(TMP / "ps-comment", av(), [("Hello", 1.0, 1.4), ("world.", 1.5, 1.9)])
        env.write_json(P / "clips.json", {"version": 1, "clips": {
            "_note": "parked for now", "c01": {"pieces": [{"words": [0, 1]}]},
            "_c02-off": {"pieces": [{"words": [0, 0]}]}}})
        self.assertEqual(Project(P).clip_ids(), ["c01"])
        rc, out, _ = hc("status", P)
        self.assertEqual(rc, 0)
        self.assertIn("clips.json  1 clips", out)
        self.assertFalse(any(ln.startswith("_") for ln in out.splitlines()), out)
        with self.assertRaises(env.HCError):
            quiet(cli.cmd_all, cli.build_parser().parse_args(["all", str(P), "_note"]))


# ---------------------------------------------------------------- launcher and install (I1, I6, V19-N1)

class LauncherSafePath(unittest.TestCase):
    HC = SK / "scripts" / "hc"

    def test_child_environment_turns_safe_path_on(self):
        loader = SourceFileLoader("hc_launcher_fg", str(self.HC))
        mod = importlib.util.module_from_spec(importlib.util.spec_from_loader(loader.name, loader))
        keep = sys.dont_write_bytecode
        try:
            loader.exec_module(mod)
        finally:
            sys.dont_write_bytecode = keep
        self.assertEqual(mod._child_env()["PYTHONSAFEPATH"], "1")
        self.assertEqual(doctor._sub_env()["PYTHONSAFEPATH"], "1")

    def test_modules_in_the_current_folder_are_not_imported(self):
        if not env.venv_python().exists():
            self.skipTest("no venv on this machine")
        cwd = TMP / "shadow-cwd"
        (cwd / "hookcutter").mkdir(parents=True)
        (cwd / "hookcutter" / "__init__.py").write_text("", encoding="utf-8")
        (cwd / "hookcutter" / "cli.py").write_text("print('HIJACKED')\nraise SystemExit(42)\n", encoding="utf-8")
        (cwd / "random.py").write_text("X = 1\n", encoding="utf-8")
        (cwd / "shlex.py").write_text("print('HIJACKED-STDLIB')\n", encoding="utf-8")
        e = dict(os.environ)
        for k in ("PYTHONPATH", "PYTHONSAFEPATH"):
            e.pop(k, None)
        e["PYTHONDONTWRITEBYTECODE"] = "1"
        py = "/usr/bin/python3" if os.path.exists("/usr/bin/python3") else sys.executable
        def run(argv):
            return subprocess.run([py, str(self.HC)] + argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                  universal_newlines=True, env=e, cwd=str(cwd), timeout=300)
        r = run(["--version"])
        self.assertNotIn("HIJACKED", r.stdout + r.stderr)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("hook-cutter", r.stdout)
        self.assertIn("venv active", r.stdout)
        r = run(["status", str(TMP / "shadow-none")])           # imports the stdlib modules too
        self.assertNotIn("HIJACKED", r.stdout + r.stderr)
        self.assertNotIn("ImportError", r.stderr)
        self.assertEqual(r.returncode, 1, r.stderr)
        self.assertIn("project folder not found", r.stderr)


class InstallCommands(unittest.TestCase):
    def setUp(self):
        self.patch = mock.patch.dict(os.environ, {"HOOKCUTTER_HOME": str(TMP / "inst-home")})
        self.patch.start()

    def tearDown(self):
        self.patch.stop()

    def pip_cmd(self, extras):
        with mock.patch.object(doctor, "find_base_python", return_value=(["/x/python3.12"], (3, 12))):
            return doctor.install_commands(extras, uv=None, venv_exists=True)[-1]

    def test_with_url_moves_yt_dlp_to_the_newest_release(self):
        cmd = doctor.install_commands(["url"], uv="/x/uv", venv_exists=True)[-1]
        i = cmd.index("--upgrade-package")
        self.assertEqual(cmd[i + 1], "yt-dlp")
        self.assertIn("--no-build", cmd)
        cmd = self.pip_cmd(["url"])
        self.assertIn("--upgrade", cmd)
        self.assertEqual(cmd[cmd.index("--only-binary") + 1], ":all:")
        cmd = doctor.install_commands(["mlx", "url"], uv="/x/uv", venv_exists=True)[-1]
        self.assertEqual(cmd[cmd.index("--upgrade-package") + 1], "yt-dlp")

    def test_a_plain_install_never_moves_anything(self):
        for extras in ([], ["mlx"], ["whispercpp"]):
            for cmd in (doctor.install_commands(extras, uv="/x/uv", venv_exists=True)[-1], self.pip_cmd(extras)):
                self.assertFalse([x for x in cmd if x.startswith("--upgrade")], cmd)

    def test_the_hub_library_the_code_imports_is_pinned(self):
        core = doctor.parse_requirements(env.SCRIPTS_DIR / "requirements.txt")
        pins = dict((n, (op, ver)) for n, op, ver in core)
        self.assertEqual(pins["huggingface-hub"][0], "==")
        self.assertTrue(all(op == "==" for op, _ in pins.values()), pins)
        have = dict((n, ver) for n, (_, ver) in pins.items())
        ok = doctor.check_packages({"packages": dict(have), "imports": {}}, [])[0]
        self.assertEqual(ok.status, doctor.PASS, ok.detail)
        other = doctor.check_packages({"packages": dict(have, **{"huggingface-hub": "99.0.0"}), "imports": {}}, [])[0]
        self.assertEqual(other.status, doctor.WARN)
        self.assertIn("huggingface-hub 99.0.0", other.detail)


if __name__ == "__main__":
    unittest.main()
