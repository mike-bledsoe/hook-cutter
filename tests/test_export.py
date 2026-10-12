"""Export folder naming, manifest and manifest-scoped cleanup (contract section 11)."""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts"))

from hookcutter import env, export  # noqa: E402
from hookcutter.project import Project  # noqa: E402


def _plan(dur: float, piece_outs) -> dict:
    return {"duration_out": dur, "frames_out": int(dur * 25), "fps": [25, 1], "spec_hash": "h",
            "hook_card": {"lines": ["*HOOK*"], "dur": 1.5},
            "pieces": [{"n": i, "out": [a, b]} for i, (a, b) in enumerate(piece_outs)]}


class Sanitize(unittest.TestCase):
    def test_more_reserved_names_and_joiners(self):
        for name in ("COM\u00b9", "LPT\u00b3", "CONIN$", "conout$"):
            self.assertTrue(export.sanitize(name).endswith("_"), name)
        zwj = "family \U0001F468\u200d\U0001F469"
        self.assertEqual(export.sanitize(zwj), zwj)                  # ZWJ keeps the emoji sequence
        self.assertEqual(export.sanitize("a\u200cb"), "a\u200cb")      # ZWNJ is part of the spelling
        self.assertEqual(export.sanitize("a\u202eb\u2066c"), "abc")     # bidi controls go

    def test_cross_platform_names(self):
        self.assertEqual(export.sanitize('Rule 1: Ship? <now>/later.'), "Rule 1 - Ship now-later")
        self.assertEqual(export.sanitize("CON"), "CON_")
        self.assertEqual(export.sanitize("lpt1.txt"), "lpt1.txt_")
        self.assertEqual(export.sanitize(" ..hidden.. "), "hidden")
        self.assertEqual(export.sanitize("tab\tand\nnewline"), "tab and newline")
        self.assertEqual(export.sanitize("", fallback="c01"), "c01")
        self.assertEqual(export.sanitize("école"), "école")
        self.assertLessEqual(len(export.sanitize("é" * 400).encode("utf-8")), export.MAX_NAME_BYTES)
        self.assertNotRegex(export.sanitize('a<b>c:d"e/f\\g|h?i*j'), r'[<>:"/\\|?*]')


class ExportFolder(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="hc-b6-export-"))
        self.P = Project(self.tmp / "P", create=True)
        self.clips = {
            "c01": {"title": "Learn it: in time", "speaker": "Speaker B", "pieces": [{"words": [0, 1]}],
                    "post": {"title": "t", "caption": "c"}, "cover": {"headline": "LEARN *IT*"}},
            "c02": {"title": "Long one", "formats": ["vertical", "landscape"],
                    "pieces": [{"words": [0, 1], "chapter": "Intro"}, {"words": [2, 3], "chapter": "Middle"},
                               {"words": [4, 5], "chapter": "End"}]},
            "c03": {"title": "Wide only", "formats": ["landscape"], "pieces": [{"words": [0, 1]}]},
        }
        self._write_clips()
        plans = {"c01": _plan(30.0, [(0, 30)]), "c02": _plan(65.0, [(0, 20), (20, 45), (45, 65)]),
                 "c03": _plan(40.0, [(0, 40)])}
        for cid, plan in plans.items():
            env.write_json(self.P.plan_path(cid), plan)
            fmts = self.clips[cid].get("formats") or ["vertical"]
            for fmt in fmts:
                Path(self.P.render_path(cid, fmt, "clean")).write_bytes(f"{cid}-{fmt}".encode())
            if "vertical" in fmts:
                Path(self.P.cover_path(cid, "clean", "main")).write_bytes(b"\xff\xd8\xff cover")
        for cid in plans:
            self._qa(cid, "PASS")

    def _qa(self, cid, status, complete=True, styles=("clean",), formats=None, **extra):
        """A recorded qa/<clip>.json (export only vouches for what QA checked)."""
        fmts = list(formats or self.clips[cid].get("formats") or ["vertical"])
        rep = {"status": status, "complete": complete, "gaps": 0 if complete else 2,
               "checked": {"styles": list(styles), "formats": fmts}, "stale_renders": [], **extra}
        from hookcutter import qa
        stamp = getattr(qa, "qa_code", lambda: None)()   # a report made by this code (else: stale)
        if stamp:
            rep.setdefault("qa_code", stamp)
        env.write_json(self.P.qa_path(cid, ".json"), rep)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _write_clips(self):
        env.write_json(self.P.root / "clips.json", {"version": 1, "defaults": {"styles": ["clean"]}, "clips": self.clips})

    def test_names_manifest_and_chapters(self):
        res = export.export(self.P)
        root = self.P.export_dir()
        expect = ["Clean/01 - Learn it - in time.mp4", "Clean/01 - Learn it - in time - Cover.jpg",
                  "Clean/02 - Long one.mp4", "Clean/02 - Long one - Landscape.mp4", "Clean/02 - Long one - Cover.jpg",
                  "Clean/03 - Wide only.mp4"]
        for rel in expect:
            self.assertTrue((root / rel).is_file(), rel)
        self.assertEqual((root / "Clean/03 - Wide only.mp4").read_bytes(), b"c03-landscape")
        man = env.read_json(root / "manifest.json")
        self.assertEqual([c["id"] for c in man["clips"]], ["c01", "c02", "c03"])
        c1, c2 = man["clips"][0], man["clips"][1]
        self.assertEqual(c1["thumb_offset_ms"], 0)
        self.assertEqual(c1["qa_status"], "PASS")
        self.assertTrue(c1["qa_complete"])
        self.assertEqual(c1["qa_checked"], {"styles": ["clean"], "formats": ["vertical"]})
        self.assertEqual(c2["qa_checked"]["formats"], ["vertical", "landscape"])
        self.assertEqual(res["held"], [])
        self.assertIsNone(c1["cover_headline"])        # no cover json yet: nothing was rendered to vouch for
        self.assertEqual(c1["speaker"], "Speaker B")
        self.assertEqual(c1["duration"], 30.0)
        self.assertEqual([c["timestamp"] for c in c2["chapters"]], ["00:00", "00:20", "00:45"])
        self.assertTrue(c2["chapters_youtube_ready"])
        self.assertEqual(c1["chapters"], [])
        self.assertEqual(res["missing"], [])

    def test_cleanup_is_manifest_scoped(self):
        export.export(self.P)
        root = self.P.export_dir()
        mine = root / "Clean" / "my notes.txt"
        mine.write_text("keep me")
        self.clips["c01"]["title"] = "Renamed"
        del self.clips["c03"]
        self._write_clips()
        res = export.export(self.P)
        self.assertFalse((root / "Clean/01 - Learn it - in time.mp4").exists())
        self.assertTrue((root / "Clean/01 - Renamed.mp4").is_file())
        self.assertFalse((root / "Clean/03 - Wide only.mp4").exists())
        self.assertTrue(mine.is_file())
        self.assertIn("Clean/03 - Wide only.mp4", res["deleted"])

    def test_subset_export_keeps_other_clips(self):
        export.export(self.P)
        root = self.P.export_dir()
        res = export.export(self.P, ["c02"])
        self.assertTrue((root / "Clean/01 - Learn it - in time.mp4").is_file())
        self.assertEqual(res["deleted"], [])
        man = env.read_json(root / "manifest.json")
        self.assertEqual([c["id"] for c in man["clips"]], ["c01", "c02", "c03"])

    def test_missing_qa_holds_the_clip(self):
        Path(self.P.qa_path("c02", ".json")).unlink()
        res = export.export(self.P)
        root = self.P.export_dir()
        self.assertFalse((root / "Clean/02 - Long one.mp4").exists())
        self.assertEqual({(h["clip"], h["format"]) for h in res["held"]}, {("c02", "vertical"), ("c02", "landscape")})
        self.assertTrue(all("QA has not run" in h["reason"] for h in res["held"]))
        man = env.read_json(root / "manifest.json")
        c2 = next(c for c in man["clips"] if c["id"] == "c02")
        self.assertEqual((c2["files"], c2["qa_status"]), ([], "not run"))
        res = export.export(self.P, force=True)
        self.assertTrue((root / "Clean/02 - Long one.mp4").is_file())
        self.assertTrue(res["forced"])

    def test_fail_and_incomplete_are_held(self):
        self._qa("c01", "FAIL")
        self._qa("c03", "INCOMPLETE", complete=False)
        res = export.export(self.P)
        held = {h["clip"]: h["reason"] for h in res["held"]}
        self.assertIn("QA FAIL", held["c01"])
        self.assertIn("INCOMPLETE", held["c03"])
        self.assertFalse((self.P.export_dir() / "Clean/01 - Learn it - in time.mp4").exists())
        self.assertFalse((self.P.export_dir() / "Clean/01 - Learn it - in time - Cover.jpg").exists())
        self.assertNotIn("c02", held)

    def test_unchecked_style_is_held(self):
        self.clips["c01"]["styles"] = ["clean", "pop"]
        self._write_clips()
        Path(self.P.render_path("c01", "vertical", "pop")).write_bytes(b"c01-pop")
        Path(self.P.cover_path("c01", "pop", "main")).write_bytes(b"\xff\xd8\xff pop")
        self._qa("c01", "WARN")                          # QA ran on clean only (and is complete)
        res = export.export(self.P)
        self.assertEqual([(h["clip"], h["style"]) for h in res["held"]], [("c01", "pop")])
        self.assertIn("did not check vertical/pop", res["held"][0]["reason"])
        man = env.read_json(self.P.export_dir() / "manifest.json")
        c1 = man["clips"][0]
        self.assertEqual((c1["qa_status"], c1["qa_complete"]), ("WARN", True))
        self.assertEqual({f["style"] for f in c1["files"]}, {"clean"})

    def test_stale_qa_is_held(self):
        import os
        import time
        r = Path(self.P.render_path("c01", "vertical", "clean"))
        later = time.time() + 5
        os.utime(r, (later, later))                      # re-rendered after QA ran
        res = export.export(self.P)
        self.assertIn("QA is stale", next(h["reason"] for h in res["held"] if h["clip"] == "c01"))
        man = env.read_json(self.P.export_dir() / "manifest.json")
        self.assertTrue(man["clips"][0]["qa_stale"])

    def test_render_older_than_plan_is_held(self):
        from hookcutter import render as render_mod
        env.write_json(self.P.render_path("c01", "vertical", "clean", ".log.json"),
                       {"hashes": {"plan": "an-older-plan", "render": "x"}})
        self._qa("c01", "PASS")
        res = export.export(self.P)
        self.assertIn("older than the plan", next(h["reason"] for h in res["held"] if h["clip"] == "c01"))
        plan = env.read_json(self.P.plan_path("c01"))
        env.write_json(self.P.render_path("c01", "vertical", "clean", ".log.json"),
                       {"hashes": {"plan": render_mod.plan_hash(plan), "render": "x"}})
        self._qa("c01", "PASS")
        self.assertEqual([h for h in export.export(self.P)["held"] if h["clip"] == "c01"], [])

    def test_manifest_records_what_was_rendered(self):
        env.write_json(self.P.render_path("c01", "vertical", "clean", ".captions.json"),
                       {"hook": {"lines": ["NOT YOUR *FRIENDS*"]}, "pages": []})
        env.write_json(self.P.cover_path("c01", "clean", "json"), {"headline": "SELL TO *STRANGERS*"})
        plan = env.read_json(self.P.plan_path("c01"))
        plan.update(words=[{"i": 0, "sp": "B"}, {"i": 1, "sp": "A"}, {"i": 2, "sp": "B"}], intent=[0, 1, 2])
        env.write_json(self.P.plan_path("c01"), plan)
        self._qa("c01", "PASS")
        export.export(self.P)
        c1 = env.read_json(self.P.export_dir() / "manifest.json")["clips"][0]
        self.assertEqual(c1["hook_card"], {"lines": ["NOT YOUR *FRIENDS*"]})
        self.assertEqual(c1["cover_headline"], "SELL TO *STRANGERS*")
        self.assertEqual((c1["speakers"], c1["speaker"]), (["B", "A"], "Speaker B"))

    def test_dropped_style_files_are_not_kept(self):
        self.clips["c01"]["styles"] = ["clean", "pop"]
        self._write_clips()
        Path(self.P.render_path("c01", "vertical", "pop")).write_bytes(b"c01-pop")
        Path(self.P.cover_path("c01", "pop", "main")).write_bytes(b"\xff\xd8\xff pop")
        self._qa("c01", "PASS", styles=("clean", "pop"))
        export.export(self.P)
        root = self.P.export_dir()
        self.assertTrue((root / "Pop/01 - Learn it - in time.mp4").is_file())
        # --styles clean restricts the run: the pop export stays
        self._qa("c01", "PASS", styles=("clean", "pop"))
        export.export(self.P, styles=["clean"])
        self.assertTrue((root / "Pop/01 - Learn it - in time.mp4").is_file())
        # the clip's own style list drops pop: its old export goes
        self.clips["c01"]["styles"] = ["clean"]
        self._write_clips()
        self._qa("c01", "PASS")
        res = export.export(self.P)
        self.assertFalse((root / "Pop/01 - Learn it - in time.mp4").exists())
        self.assertIn("Pop/01 - Learn it - in time.mp4", res["deleted"])

    def test_case_only_title_change_keeps_the_files(self):
        export.export(self.P)
        self.clips["c01"]["title"] = "LEARN IT: IN TIME"
        self._write_clips()
        self._qa("c01", "PASS")
        res = export.export(self.P)
        man = env.read_json(self.P.export_dir() / "manifest.json")
        for f in man["clips"][0]["files"]:
            self.assertTrue((self.P.export_dir() / f["path"]).is_file(), f["path"])
        self.assertEqual([m for m in res["missing"] if "manifest" in m], [])
        names = os.listdir(self.P.export_dir() / "Clean")
        self.assertIn("01 - LEARN IT - IN TIME.mp4", names)

    def test_formats_flag_exports_landscape_with_chapters(self):
        self._qa("c01", "PASS", formats=("landscape",))
        Path(self.P.render_path("c01", "landscape", "clean")).write_bytes(b"c01-landscape")
        self.clips["c01"]["pieces"] = [{"words": [0, 1], "chapter": "A"}]
        self._write_clips()
        self._qa("c01", "PASS", formats=("landscape",))
        self._qa("c01", "PASS")
        export.export(self.P)                             # the vertical export already exists
        self._qa("c01", "PASS", formats=("landscape",))
        res = export.export(self.P, ["c01"], formats=["landscape"])
        self.assertEqual(res["held"], [])
        c1 = next(c for c in res["clips"] if c["id"] == "c01")
        got = {(f["format"], f["kind"]): f["path"] for f in c1["files"]}
        self.assertEqual(got[("landscape", "video")], "Clean/01 - Learn it - in time - Landscape.mp4")
        # the kept vertical export is still the vertical video, not overwritten by the landscape one
        self.assertEqual((self.P.export_dir() / got[("vertical", "video")]).read_bytes(), b"c01-vertical")
        self.assertEqual([c["timestamp"] for c in c1["chapters"]], ["00:00"])

    def test_missing_render_is_reported(self):
        Path(self.P.render_path("c01", "vertical", "clean")).unlink()
        res = export.export(self.P)
        self.assertTrue(any("c01__vertical__clean.mp4" in m for m in res["missing"]))

    def test_unchanged_files_are_not_rewritten(self):
        export.export(self.P)
        res = export.export(self.P)
        self.assertEqual(res["written"], [])


if __name__ == "__main__":
    unittest.main()
