"""A new user's first hour: the by-ear checklist written at export, `hc cover --grid`, and
`hc all` keeping covers that are already current instead of making them again."""

from __future__ import annotations

import argparse
import contextlib
import io
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts"))
sys.path.insert(0, str(HERE))

import fixtures  # noqa: E402
from hookcutter import cli, cover, env, export  # noqa: E402
from hookcutter.project import Project  # noqa: E402


def _word(i, raw, s_out, e_out, **kw):
    return dict({"i": i, "raw": raw, "text": raw, "s_out": s_out, "e_out": e_out, "hidden": False,
                 "omitted": False, "removed": False}, **kw)


class ByEar(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="hc-byear-"))
        self.P = Project(self.tmp / "P", create=True)
        env.write_json(self.P.root / "clips.json", {"version": 1, "defaults": {"styles": ["clean"]}, "clips": {
            "c01": {"title": "Open strong", "pieces": [{"words": [5, 7]}, {"words": [0, 2]}]}}})
        words = [_word(5, "Nobody", 0.0, 0.4), _word(6, "watches", 0.4, 0.8), _word(7, "television.", 0.8, 1.5),
                 _word(0, "Most", 1.6, 1.9), _word(1, "people", 1.9, 2.3), _word(2, "think.", 2.3, 2.9),
                 _word(3, "um", 2.0, 2.1, removed=True)]
        plan = {"duration_out": 3.0, "frames_out": 75, "fps": [25, 1], "spec_hash": "h", "kind": "cut",
                "hook_card": {"lines": ["*HOOK*"], "dur": 1.5}, "words": words,
                "pieces": [{"n": 0, "out": [0, 1.55]}, {"n": 1, "out": [1.55, 3.0]}],
                "segments": [{"k": 0, "src": [10.0, 11.55], "out": [0.0, 1.55]},
                             {"k": 1, "src": [2.0, 3.45], "out": [1.55, 3.0]}],
                "joins": [{"k": 1, "out": 1.55, "kind": "piece"}]}
        env.write_json(self.P.plan_path("c01"), plan)
        env.write_json(self.P.root / "transcript.json", {"version": 1, "words": [], "flags": [
            {"kind": "timing_uncertain", "a": 1, "b": 1},
            {"kind": "unworded_audio", "s": 2.5, "e": 3.0},
            {"kind": "false_start", "a": 40, "b": 42}]})
        Path(self.P.render_path("c01", "vertical", "clean")).write_bytes(b"video")
        Path(self.P.cover_path("c01", "clean", "main")).write_bytes(b"\xff\xd8\xff cover")
        from hookcutter import qa
        rep = {"status": "WARN", "complete": True, "gaps": 0, "stale_renders": [],
               "checked": {"styles": ["clean"], "formats": ["vertical"]},
               "gates": [{"id": "G10", "status": "WARN", "target": "vertical/clean",
                          "evidence": "token 'people' lights 0.31 s early"},
                         {"id": "G12", "status": "WARN", "target": "cover/clean", "evidence": "no face"}]}
        stamp = getattr(qa, "qa_code", lambda: None)()
        if stamp:
            rep["qa_code"] = stamp
        env.write_json(self.P.qa_path("c01", ".json"), rep)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_checklist_lists_every_join_flag_and_sound_warning(self):
        export.export(self.P)
        md = (self.P.export_dir() / export.BY_EAR).read_text(encoding="utf-8")
        self.assertIn("## 01 Open strong (3.0 s)", md)
        self.assertIn("`export/Clean/01 - Open strong.mp4`", md)
        self.assertIn('0:00.0 the opening, "Nobody watches television.": is the first word whole', md)
        self.assertIn('0:01.6 join between pieces: "Nobody watches television." | "Most people think."', md)
        self.assertIn('0:02.9 the ending, "Most people think.": does it end clean', md)
        body = md.split("## 01")[1]
        times = [ln.split()[2] for ln in body.splitlines() if ln.startswith("- [ ] 0:")]
        self.assertEqual(times, sorted(times), "the lines are in the order they play")
        self.assertNotIn("um", md.split("## 01")[1].split("Captions")[0].replace("plumb", ""))
        self.assertIn('0:01.9 "people": the start time of this word is a guess', md)
        self.assertIn("0:02.0 sound with no words", md)          # src 2.5 is out 1.55 + 0.5
        self.assertNotIn("stumble", md)                          # flag on words outside the clip
        self.assertIn("QA G10 WARN: token 'people' lights 0.31 s early", md)
        self.assertNotIn("G12", md)                              # not a sound check

    def test_held_clip_is_not_listed(self):
        rep = env.read_json(self.P.qa_path("c01", ".json"))
        rep["status"] = "FAIL"
        env.write_json(self.P.qa_path("c01", ".json"), rep)
        export.export(self.P)
        md = (self.P.export_dir() / export.BY_EAR).read_text(encoding="utf-8")
        self.assertIn("Nothing is exported yet.", md)


class CoverGrid(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="hc-grid-"))
        self.P = Project(self.tmp / "P", create=True)
        env.write_json(self.P.root / "clips.json", {"version": 1, "defaults": {"styles": ["clean"]}, "clips": {
            "c01": {"title": "One", "pieces": [{"words": [0, 1]}]},
            "c02": {"title": "Two", "pieces": [{"words": [2, 3]}]}}})

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _ns(self, **kw):
        return argparse.Namespace(**dict({"project": str(self.P.root), "clips": [], "styles": None,
                                          "at": None, "grid": True}, **kw))

    def test_grid_only_rebuilds_from_covers_on_disk(self):
        from PIL import Image
        for cid in ("c01", "c02"):
            Image.new("RGB", (360, 640), (40, 80, 120)).save(str(self.P.cover_path(cid, "clean", "grid")))
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self.assertEqual(cover.cmd_cover(self._ns()), 0)
        self.assertTrue(Path(self.P.sheet_path("covers-grid.jpg")).is_file())
        self.assertIn("covers grid", buf.getvalue())

    def test_grid_needs_covers_and_takes_no_clip(self):
        with self.assertRaises(env.HCError):
            cover.cmd_cover(self._ns())
        with self.assertRaises(env.HCError):
            cover.cmd_cover(self._ns(clips=["c01"]))


class AllKeepsCurrentCovers(unittest.TestCase):
    """A cover made by `hc cover` (or copied in from a clone) has no `hc all` stamp. `hc all`
    keeps it when `hc status` would call it current, instead of making it again."""

    def test_unstamped_current_cover_is_kept(self):
        tmp = Path(tempfile.mkdtemp(prefix="hc-allcover-"))
        try:
            words = [("Every", 1.0, 1.3), ("cover", 1.35, 1.7), ("costs", 1.75, 2.1), ("minutes.", 2.15, 2.7),
                     ("So", 3.6, 3.8), ("keep", 3.85, 4.1), ("good", 4.15, 4.4), ("ones.", 4.45, 5.0)]
            src = fixtures.make_speech_fixture(tmp, words, dur=17.0)
            P = fixtures.make_project(tmp, src, words)
            env.write_json(P / "clips.json", {"version": 1, "defaults": {"styles": ["clean"]}, "clips": {
                "c1": {"title": "Keep good covers", "pieces": [{"words": [0, 7]}], "captions": False,
                       "hook_card": {"lines": ["KEEP *GOOD* ONES"], "dur": "auto"},
                       "cover": {"headline": "KEEP *GOOD* ONES"}}}})
            ns = argparse.Namespace(project=str(P), clips=[], styles=None, formats=None, no_asr=True, force=False)
            out = io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
                cli.cmd_all(ns)
            proj = Project(P)
            jpg = Path(proj.cover_path("c1", "clean", "main"))
            before = (jpg.stat().st_mtime_ns, env.fingerprint(jpg))
            stamps = env.read_json(P / "cache" / cli.STAMPS_FILE)
            stamps["cover"] = {}
            env.write_json(P / "cache" / cli.STAMPS_FILE, stamps)
            Path(proj.sheet_path("covers-grid.jpg")).unlink()
            err = io.StringIO()
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(err):
                cli.cmd_all(ns)
            self.assertIn("cover kept (current, made outside hc all)", err.getvalue())
            self.assertEqual((jpg.stat().st_mtime_ns, env.fingerprint(jpg)), before)
            self.assertTrue(Path(proj.sheet_path("covers-grid.jpg")).is_file())
            # a changed headline makes it stale again: then it is made
            spec = env.read_json(P / "clips.json")
            spec["clips"]["c1"]["cover"]["headline"] = "KEEP THE *GOOD* ONES"
            env.write_json(P / "clips.json", spec)
            err = io.StringIO()
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(err):
                cli.cmd_all(ns)
            self.assertIn("cover made", err.getvalue())
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class SubtitleBreaks(unittest.TestCase):
    """The .srt sidecar: no word alone in a cue, and no break inside a number phrase."""

    CASES = ("It takes about twelve to eighteen months before a coach is any good at this, honestly.",
             "We grew from 12 to 18 months of runway and then we raised again later that year.",
             "I spent three thousand hours learning this one skill before anybody would pay me a single "
             "dollar for it.",
             "He charged eighty-eight dollars a month and sixty-eight for students and it filled up in a week.")

    def _cues(self, text):
        import test_captions as TC
        import test_fix2_fb as F
        plan = F.seam_plan([("start", 400, F.spoken(text, start=0.2, wps=3.2))])
        return [c[2] for c in TC.build(plan).srt_cues()]

    def test_no_orphans_and_numbers_stay_together(self):
        from hookcutter import captions
        for text in self.CASES:
            cues = self._cues(text)
            self.assertEqual(" ".join(c.replace("\n", " ") for c in cues).split(), text.split(), text)
            for c in cues:
                self.assertGreater(len(c.split()), 1, (text, cues))
            for c in cues:          # every line and cue boundary
                lines = c.split("\n")
                for a, b in zip(lines, lines[1:]):
                    self.assertFalse(captions._splits_number(a.split()[-1], b.split()[0]), (text, cues))
            for a, b in zip(cues, cues[1:]):
                self.assertFalse(captions._splits_number(a.split()[-1], b.split()[0]), (text, cues))

    def test_number_phrase_detection(self):
        from hookcutter import captions
        self.assertTrue(captions._splits_number("eighteen", "months"))
        self.assertTrue(captions._splits_number("12", "to"))
        self.assertTrue(captions._splits_number("to", "18"))
        self.assertFalse(captions._splits_number("to", "buy"))
        self.assertFalse(captions._splits_number("19.", "I"))


class SmallFixes(unittest.TestCase):
    def test_peek_names_keep_sub_second_times_apart(self):
        from hookcutter import sheets
        self.assertEqual(sheets._fmt_name(1.0), "1.0")
        self.assertEqual(sheets._fmt_name(748.95), "748.95")
        self.assertNotEqual(sheets._fmt_name(748.95), sheets._fmt_name(749.0))
        self.assertEqual(sheets._fmt_name(-0.5), "m0.5")

    def test_words_refuses_an_unknown_clip(self):
        from hookcutter import find
        tmp = Path(tempfile.mkdtemp(prefix="hc-words-"))
        try:
            P = Project(tmp / "P", create=True)
            env.write_json(P.root / "clips.json", {"version": 1, "clips": {"c01": {"pieces": [{"words": [0, 1]}]}}})
            env.write_json(P.root / "transcript.json", fixtures.fake_transcript([("a", 0.1, 0.2), ("b", 0.3, 0.4)]))
            ns = argparse.Namespace(project=str(P.root), a=0, b=1, clip="scratch-does-not-exist")
            with self.assertRaises(env.HCError):
                find.cmd_words(ns)
            env.write_json(P.root / "transcripts" / "scratch-1.json", fixtures.fake_transcript([("c", 0.1, 0.2)]))
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(find.cmd_words(argparse.Namespace(project=str(P.root), a=0, b=0,
                                                                   clip="scratch-1")), 0)
                self.assertEqual(find.cmd_words(argparse.Namespace(project=str(P.root), a=0, b=1,
                                                                   clip="c01")), 0)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
