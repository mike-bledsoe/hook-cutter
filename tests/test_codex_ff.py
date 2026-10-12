"""Regression tests for the cover and export fixes of the third review pass.

Export: evidence gaps hold a clip, a render whose inputs changed is held, a clip-wide hold
also drops the variants a restricted run would keep, style folders do not depend on style
order, one path has one owner, and two exports at once both reach the manifest.
Covers: the temp file of a JPEG write is the writer's own, a negative shift is ignored, the
headline is the one the claim checks read, frames come from the stage A files of the cover's
own style and plan, a recorded time maps back to its own frame, the candidate cache follows
the letterbox position and the shots data, and the source fallback reads the render's frame
grid.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "scripts"))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from hookcutter import cover, cover_frames, env, export  # noqa: E402
from hookcutter.project import Project  # noqa: E402


def _plan(dur: float = 30.0) -> dict:
    return {"duration_out": dur, "frames_out": int(dur * 30), "fps": [30, 1], "spec_hash": "h",
            "hook_card": {"lines": ["*HOOK*"], "dur": 1.5}, "pieces": [{"n": 0, "out": [0, dur]}]}


def _write_clips(P, clips: dict, default_styles=("clean",)) -> None:
    env.write_json(P.root / "clips.json",
                   {"version": 1, "defaults": {"styles": list(default_styles)}, "clips": clips})


def _stamped(rep: dict) -> dict:
    """The report with the stamp a QA run of this code writes (without it a report is stale)."""
    from hookcutter import qa
    stamp = getattr(qa, "qa_code", lambda: None)()
    return {**rep, "qa_code": stamp} if stamp else rep


def _qa(P, cid: str, status="PASS", complete=True, styles=("clean",), formats=("vertical",), **extra) -> None:
    env.write_json(P.qa_path(cid, ".json"), _stamped(
        {"status": status, "complete": complete, "gaps": 0 if complete else 2,
         "checked": {"styles": list(styles), "formats": list(formats)},
         "stale_renders": [], **extra}))


def _stub(P, cid: str, styles=("clean",), formats=("vertical",)) -> None:
    """A byte-stub render and cover per style (the bytes name the variant), a plan, a PASS QA."""
    env.write_json(P.plan_path(cid), _plan())
    for st in styles:
        for fmt in formats:
            Path(P.render_path(cid, fmt, st)).write_bytes(f"VIDEO {cid} {fmt} {st}".encode())
        if "vertical" in formats:
            Path(P.cover_path(cid, st, "main")).write_bytes(b"\xff\xd8\xff COVER " + f"{cid} {st}".encode())
    _qa(P, cid, styles=styles, formats=formats)


def _preset(P, name: str, label=None, extends="clean", **body) -> None:
    d = P.root / "presets"
    d.mkdir(exist_ok=True)
    data = {"name": name, "extends": extends, **body}
    if label is not None:
        data["label"] = label
    (d / f"{name}.json").write_text(json.dumps(data, indent=2), encoding="utf-8")


def _video(res: dict, cid: str, **want) -> str:
    c = next(c for c in res["clips"] if c["id"] == cid)
    return next(f["path"] for f in c["files"] if f["kind"] == "video" and all(f[k] == v for k, v in want.items()))


def _unowned(root: Path) -> list:
    man = env.read_json(root / "manifest.json")
    own = {f["path"] for c in man["clips"] for f in c["files"]}
    return [str(p.relative_to(root)) for p in sorted(root.rglob("*"))
            if p.is_file() and p.name not in ("manifest.json", export.BY_EAR)
            and str(p.relative_to(root)) not in own]


class _Tmp(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="hc-codex-ff-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def project(self, clips: dict, name: str = "P"):
        P = Project(self.tmp / name, create=True)
        _write_clips(P, clips)
        return P


# ------------------------------------------------------------------ export holds (X1, X2, X6)

class ExportHolds(_Tmp):
    def one(self, styles=("clean",)):
        clip = {"title": "Learn it: in time", "pieces": [{"words": [0, 1]}]}
        if tuple(styles) != ("clean",):
            clip["styles"] = list(styles)
        P = self.project({"c01": clip})
        _stub(P, "c01", styles=styles)
        return P

    def test_a_warn_report_with_evidence_gaps_is_held(self):
        P = self.one()
        _qa(P, "c01", "WARN", complete=False)
        res = export.export(P)
        root = P.export_dir()
        held = {h["clip"]: h["reason"] for h in res["held"]}
        self.assertIn("evidence gaps", held.get("c01", ""))
        self.assertFalse((root / "Clean/01 - Learn it - in time.mp4").exists())
        self.assertFalse((root / "Clean/01 - Learn it - in time - Cover.jpg").exists())
        c1 = env.read_json(root / "manifest.json")["clips"][0]
        self.assertEqual((c1["files"], c1["qa_complete"]), ([], False))
        res = export.export(P, force=True)
        self.assertTrue((root / "Clean/01 - Learn it - in time.mp4").is_file())
        self.assertTrue((root / "Clean/01 - Learn it - in time - Cover.jpg").is_file())
        video = next(f for f in res["clips"][0]["files"] if f["kind"] == "video")
        self.assertIn("evidence gaps", video["forced"])

    def test_gaps_count_whatever_the_complete_flag_says(self):
        P = self.one()
        env.write_json(P.qa_path("c01", ".json"), _stamped(
            {"status": "PASS", "gaps": 1, "checked": {"styles": ["clean"], "formats": ["vertical"]}}))
        self.assertIn("evidence gaps", export.export(P)["held"][0]["reason"])

    def test_a_complete_warn_report_is_exported(self):
        P = self.one()
        _qa(P, "c01", "WARN")
        self.assertEqual(export.export(P)["held"], [])
        # a hand-written report without the two keys counts as complete
        env.write_json(P.qa_path("c01", ".json"), _stamped(
            {"status": "PASS", "checked": {"styles": ["clean"], "formats": ["vertical"]}}))
        self.assertEqual(export.export(P)["held"], [])
        for rep in ({"status": "PASS"}, {"gaps": "n/a"}, None, []):
            self.assertFalse(export._has_gaps(rep), rep)
        for rep in ({"complete": False}, {"gaps": 3}, {"gaps": "2"}):
            self.assertTrue(export._has_gaps(rep), rep)

    def _logged(self, P, recorded: str):
        from hookcutter import render as render_mod
        plan = env.read_json(P.plan_path("c01"))
        env.write_json(P.render_path("c01", "vertical", "clean", ".log.json"),
                       {"hashes": {"plan": render_mod.plan_hash(plan), "render": recorded}})
        _qa(P, "c01")
        return {"plan": render_mod.plan_hash(plan)}

    def test_a_render_that_predates_its_inputs_is_held(self):
        from hookcutter import render as render_mod
        P = self.one()
        base = self._logged(P, "then")
        with mock.patch.object(render_mod, "expected_hashes", return_value={**base, "render": "now"}):
            res = export.export(P)
        self.assertIn("predates its inputs", res["held"][0]["reason"])
        c1 = env.read_json(P.export_dir() / "manifest.json")["clips"][0]
        self.assertEqual((c1["files"], c1["render_stale"]), ([], True))
        with mock.patch.object(render_mod, "expected_hashes", return_value={**base, "render": "now"}):
            res = export.export(P, force=True)
        video = next(f for f in res["clips"][0]["files"] if f["kind"] == "video")
        self.assertTrue(video["render_stale"])
        self.assertIn("predates its inputs", video["forced"])

    def test_a_render_that_matches_its_inputs_is_exported(self):
        from hookcutter import render as render_mod
        P = self.one()
        base = self._logged(P, "then")
        with mock.patch.object(render_mod, "expected_hashes", return_value={**base, "render": "then"}):
            self.assertEqual(export.export(P)["held"], [])
        # inputs that cannot be computed never make a render stale
        with mock.patch.object(render_mod, "expected_hashes", side_effect=env.HCError("no source", code=2)):
            self.assertEqual(export.export(P)["held"], [])
        # and neither does a log without the hash
        env.write_json(P.render_path("c01", "vertical", "clean", ".log.json"), {"hashes": {}})
        _qa(P, "c01")
        with mock.patch.object(render_mod, "expected_hashes", return_value={"render": "now"}):
            self.assertEqual(export.export(P)["held"], [])

    def test_a_held_clip_loses_its_other_style_in_a_restricted_run(self):
        P = self.one(styles=("clean", "pop"))
        root = P.export_dir()
        export.export(P)
        pop = [root / "Pop/01 - Learn it - in time.mp4", root / "Pop/01 - Learn it - in time - Cover.jpg"]
        self.assertTrue(all(p.is_file() for p in pop))
        # a clip that is not held keeps its pop export under --styles clean
        _qa(P, "c01", styles=("clean", "pop"))
        export.export(P, styles=["clean"])
        self.assertTrue(all(p.is_file() for p in pop))
        # a forced run keeps it too
        _qa(P, "c01", "FAIL", styles=("clean", "pop"))
        export.export(P, styles=["clean"], force=True)
        self.assertTrue(all(p.is_file() for p in pop))
        # held: the files QA no longer vouches for go, whatever the run was restricted to
        _qa(P, "c01", "FAIL", styles=("clean", "pop"))
        res = export.export(P, styles=["clean"])
        self.assertFalse(any(p.exists() for p in pop))
        held = [(h["clip"], h["style"], h["format"]) for h in res["held"]]
        self.assertIn(("c01", "pop", "vertical"), held)
        self.assertEqual(len(held), len(set(held)), held)        # video and cover share one entry
        self.assertEqual(env.read_json(root / "manifest.json")["clips"][0]["files"], [])


# ------------------------------------------------------------------ export folders (X4, V02-N1)

class StyleFolders(_Tmp):
    def styled(self, labels: dict, name: str = "P"):
        styles = list(labels)
        P = self.project({"c01": {"title": "One", "styles": styles, "pieces": [{"words": [0, 1]}]}}, name)
        for st, label in labels.items():
            _preset(P, st, label)
        _stub(P, "c01", styles=styles)
        return P

    def distinct(self, labels: dict):
        self.assertEqual(len({export._fold(v) for v in labels.values()}), len(labels), labels)
        for v in labels.values():
            self.assertLessEqual(len(v.encode("utf-8")), export.MAX_NAME_BYTES)

    def test_labels_do_not_depend_on_style_order(self):
        P = self.styled({"aa": "Shared", "bb": "Shared"})
        a, b = export._style_labels(P, ["aa", "bb"]), export._style_labels(P, ["bb", "aa"])
        self.assertEqual(a, b)
        self.distinct(a)
        self.assertEqual(a, {"aa": "Shared (aa)", "bb": "Shared (bb)"})

    def test_a_label_nobody_shares_keeps_its_folder(self):
        P = self.styled({"aa": "Shared", "bb": "Shared", "cc": "Alone"})
        self.assertEqual(export._style_labels(P, ["cc", "bb", "aa"])["cc"], "Alone")
        Q = self.project({"c01": {"title": "One", "styles": ["clean", "pop", "minimal"],
                                  "pieces": [{"words": [0, 1]}]}}, "Q")
        self.assertEqual(export._style_labels(Q, ["clean", "pop", "minimal"]),
                         {"clean": "Clean", "pop": "Pop", "minimal": "Minimal"})

    def test_a_label_that_looks_like_a_suffix(self):
        P = self.styled({"aa": "Shared", "bb": "Shared (cc)", "cc": "Shared"})
        self.distinct(export._style_labels(P, ["aa", "bb", "cc"]))
        res = export.export(P)
        for st in ("aa", "bb", "cc"):
            self.assertEqual((P.export_dir() / _video(res, "c01", style=st)).read_bytes().decode(),
                             f"VIDEO c01 vertical {st}")

    def test_long_and_case_folded_duplicates(self):
        P = self.styled({"aa": "L" * 150, "bb": "L" * 150, "cc": "Straße", "dd": "STRASSE"})
        labels = export._style_labels(P, ["aa", "bb", "cc", "dd"])
        self.distinct(labels)
        self.assertTrue(labels["aa"].endswith(" (aa)") and labels["bb"].endswith(" (bb)"), labels)

    def test_a_restricted_export_leaves_the_other_style_alone(self):
        P = self.styled({"aa": "Shared", "bb": "Shared"})
        r1 = export.export(P)
        r2 = export.export(P, styles=["bb"])
        root = P.export_dir()
        for st in ("aa", "bb"):
            self.assertEqual(_video(r1, "c01", style=st), _video(r2, "c01", style=st))
            self.assertEqual((root / _video(r2, "c01", style=st)).read_bytes().decode(), f"VIDEO c01 vertical {st}")
        paths = [f["path"] for f in r2["clips"][0]["files"]]
        self.assertEqual(len(paths), len(set(paths)), paths)


class OnePathOneOwner(_Tmp):
    def test_the_vertical_of_a_landscape_only_clip(self):
        P = self.project({"c01": {"title": "Wide only", "formats": ["landscape"], "pieces": [{"words": [0, 1]}]}})
        _stub(P, "c01", formats=("landscape",))
        export.export(P)
        Path(P.render_path("c01", "vertical", "clean")).write_bytes(b"VIDEO c01 vertical clean")
        _qa(P, "c01", formats=("vertical", "landscape"))
        res = export.export(P, ["c01"], formats=["vertical"])
        root = P.export_dir()
        files = res["clips"][0]["files"]
        paths = [export._fold(f["path"]) for f in files]
        self.assertEqual(len(paths), len(set(paths)), paths)
        for f in files:
            if f["kind"] == "video":
                self.assertEqual((root / f["path"]).read_bytes().decode(), f"VIDEO c01 {f['format']} clean", f)


# ------------------------------------------------------------------ two exports at once (X5)

class ParallelExports(_Tmp):
    def test_two_clips_exported_at_once_both_reach_the_manifest(self):
        P = self.project({c: {"title": c, "pieces": [{"words": [0, 1]}]} for c in ("c01", "c02")})
        for c in ("c01", "c02"):
            _stub(P, c)
        real = export._place
        errs = []

        def slow(src, dest):                 # hold the read..write window open
            time.sleep(0.25)
            return real(src, dest)

        def run(cid):
            try:
                export.export(Project(P.root), [cid])
            except BaseException as exc:     # noqa: BLE001
                errs.append(repr(exc))
        with mock.patch.object(export, "_place", slow), mock.patch.object(env, "log", lambda *a, **k: None):
            a = threading.Thread(target=run, args=("c01",))
            b = threading.Thread(target=run, args=("c02",))
            a.start()
            time.sleep(0.05)
            b.start()
            a.join()
            b.join()
        self.assertEqual(errs, [])
        man = env.read_json(P.export_dir() / "manifest.json")
        self.assertEqual([c["id"] for c in man["clips"]], ["c01", "c02"])
        self.assertEqual(_unowned(P.export_dir()), [])

    def test_a_failed_copy_leaves_no_temp_file(self):
        P = self.project({"c01": {"title": "One", "pieces": [{"words": [0, 1]}]}})
        _stub(P, "c01")
        with mock.patch.object(export.os, "replace", side_effect=OSError("disk went away")):
            with self.assertRaises(OSError):
                export.export(P)
        left = [p.name for p in P.export_dir().rglob("*") if p.is_file()]
        self.assertEqual(left, [])


# ------------------------------------------------------------------ cover files (X5, V02-N2)

class CoverTemp(_Tmp):
    def test_save_jpeg_has_its_own_temp_name(self):
        dest = self.tmp / "covers-grid.jpg"
        (self.tmp / ".covers-grid.jpg.tmp").mkdir()          # another writer's temp, in the way
        cover.save_jpeg(Image.new("RGB", (64, 64), (200, 30, 30)), dest)
        self.assertTrue(dest.is_file())

    def test_two_writers_of_one_grid_both_finish(self):
        d = self.tmp / "sheets"
        dest = d / "covers-grid.jpg"
        img = Image.effect_noise((1080, 1440), 12).convert("RGB")
        errs = []

        def run():
            try:
                for _ in range(6):
                    cover.save_jpeg(img, dest, 88)
            except BaseException as exc:     # noqa: BLE001
                errs.append(repr(exc))
        ts = [threading.Thread(target=run) for _ in range(4)]
        for th in ts:
            th.start()
        for th in ts:
            th.join()
        self.assertEqual(errs, [])
        with Image.open(dest) as im:
            im.load()
        self.assertEqual([q.name for q in d.iterdir() if q.name != dest.name], [])

    def test_a_failed_write_leaves_no_temp_file(self):
        d = self.tmp / "covers"
        with mock.patch.object(cover.os, "replace", side_effect=OSError("disk went away")):
            with self.assertRaises(OSError):
                cover.save_jpeg(Image.new("RGB", (64, 64)), d / "c01__clean.jpg")
        self.assertEqual(list(d.iterdir()), [])

    def test_the_grid_is_rebuilt_under_a_lock(self):
        P = self.project({"c01": {"title": "One", "pieces": [{"words": [0, 1]}]}})
        cover.save_jpeg(Image.new("RGB", (1080, 1440), (90, 20, 20)), Path(P.cover_path("c01", "clean", "grid")))
        lock = Path(P.cache_dir()) / cover.GRID_LOCK
        seen = []
        real = cover.covers_grid

        def spy(*a, **k):
            from hookcutter import fsutil
            with open(lock, "a+b") as fh:            # a second holder cannot take it meanwhile
                seen.append(fsutil._try_lock(fh))
            return real(*a, **k)
        with mock.patch.object(cover, "covers_grid", spy):
            lines = cover.refresh_grids(P)
        self.assertEqual(seen, [False])
        self.assertTrue(Path(P.sheet_path("covers-grid.jpg")).is_file())
        self.assertEqual(len(lines), 1)


# ------------------------------------------------------------------ cover shift (C2)

FACE = {"box": [340, 100, 400, 480], "score": 0.9, "landmarks": [[0, 0]] * 5}


def _block_photo():
    arr = np.full((1920, 1080, 3), 30, np.uint8)
    arr[100:580] = 215                       # a bright band where the face is
    return Image.fromarray(arr)


class CoverShift(unittest.TestCase):
    CFG = dict(cover.DEFAULT_COVER)

    def test_a_negative_shift_is_ignored_and_the_geometry_stays_honest(self):
        zero, info0 = cover.compose(_block_photo(), [dict(FACE)], "BUILT IN A *GARAGE*", self.CFG,
                                    base_shift=0, shift_manual=True)
        for bad in (-300, -1, "-80"):
            with self.subTest(shift=bad):
                img, info = cover.compose(_block_photo(), [dict(FACE)], "BUILT IN A *GARAGE*", self.CFG,
                                          base_shift=bad, shift_manual=True)
                self.assertEqual(info["shift"], 0)
                self.assertEqual(info["face_box"][1], 100)
                self.assertEqual(info["faces"], info0["faces"])
                self.assertEqual(info["crown_y"], info0["crown_y"])
                self.assertTrue(any("above the grid crop" in w for w in info["warnings"]), info["warnings"])
                self.assertTrue(np.array_equal(np.asarray(img), np.asarray(zero)))

    def test_a_shift_of_zero_or_more_is_kept(self):
        for good in (0, 120, "80"):
            with self.subTest(shift=good):
                _img, info = cover.compose(_block_photo(), [dict(FACE)], "BUILT IN A *GARAGE*", self.CFG,
                                           base_shift=good, shift_manual=True)
                self.assertEqual(info["shift"], int(good))
                self.assertEqual(info["face_box"][1], 100 - int(good))


# ------------------------------------------------------------------ cover headline (C6)

class CoverHeadline(unittest.TestCase):
    PLAN = {"hook_card": {"lines": ["PLAN *CARD*"]}}

    def test_the_headline_comes_from_the_claims_definition(self):
        from hookcutter import claims
        calls = []

        def fake(clip, plan=None):
            calls.append((clip, plan))
            return "FROM *CLAIMS*", "cover"
        clip = {"title": "A title", "cover": {"headline": "IGNORED HERE"}}
        with mock.patch.object(claims, "effective_headline", fake, create=True):
            self.assertEqual(cover.headline_for(clip, self.PLAN), "FROM *CLAIMS*")
        self.assertEqual(calls, [(clip, self.PLAN)])

    def test_precedence_is_unchanged(self):
        hook = {"lines": ["CLIP", "*CARD*"]}
        cases = [
            ({"title": "T", "hook_card": hook, "cover": {"headline": "OWN *WORDS*"}}, self.PLAN, "OWN *WORDS*"),
            ({"title": "T", "hook_card": hook}, self.PLAN, "PLAN *CARD*"),
            ({"title": "T", "hook_card": hook}, None, "CLIP *CARD*"),
            ({"title": "T", "hook_card": hook, "cover": {"headline": ""}}, {}, "CLIP *CARD*"),
            ({"title": "T", "hook_card": None}, {"hook_card": None}, "T"),
            ({"title": "T"}, None, "T"),
            ({"pieces": []}, None, ""),
        ]
        for clip, plan, want in cases:
            with self.subTest(clip=clip):
                self.assertEqual(cover.headline_for(clip, plan), want)

    def test_a_drawn_headline_is_a_checked_surface(self):
        from hookcutter import claims
        if not hasattr(claims, "written_surfaces"):
            self.skipTest("claims.written_surfaces is not in this build")
        eff = {"formats": ["vertical"]}
        hook = {"lines": ["CLIP *CARD*"]}
        for clip, plan in (({"title": "We made a lot", "_eff": eff}, None),
                           ({"title": "We made a lot", "hook_card": None, "_eff": eff}, {"hook_card": None}),
                           ({"title": "T", "hook_card": hook, "_eff": eff}, None),
                           ({"title": "T", "hook_card": hook, "_eff": eff}, self.PLAN),
                           ({"title": "T", "cover": {"headline": "OWN *WORDS*"}, "_eff": eff}, self.PLAN)):
            with self.subTest(clip=clip):
                text = cover.headline_for(clip, plan)
                self.assertTrue(text)
                self.assertIn(text, [str(s[1]) for s in claims.written_surfaces(clip, plan)])


# ------------------------------------------------------------------ frame index (C3, V03-N2)

RATES = ((30, 1), (30000, 1001), (24, 1), (60, 1), (25, 1))


def _numbered(path: Path, num: int, den: int, frames: int, first: int = 0, step: int = 2, size: str = "32x32"):
    """A lossless clip whose frame N has luma 16 + step * (first + N)."""
    env.ff(["-f", "lavfi", "-i", f"color=c=black:s={size}:r={num}/{den}",
            "-vf", f"format=yuv420p,geq=lum='16+{step}*(N+{first})':cb=128:cr=128", "-frames:v", str(frames),
            "-c:v", "libx264", "-preset", "ultrafast", "-qp", "0", str(path)])
    return path


class FrameIndex(_Tmp):
    def test_a_time_rounded_to_six_decimals_is_its_own_frame(self):
        self.assertEqual(cover_frames.FRAME_TOL, 1e-3)
        for num, den in RATES + ((24000, 1001), (120, 1)):
            wrong = [n for n in range(7300)
                     if cover_frames.frame_index(round(n * den / num, 6), (num, den)) != n]
            self.assertEqual(wrong, [], (num, den))
            for part in (0.002, 0.5, 0.998):                 # a time inside a frame is that frame
                self.assertEqual(cover_frames.frame_index((10 + part) * den / num, (num, den)), 10)

    def test_candidate_times_are_frame_times(self):
        for num, den in RATES:
            for t in cover.candidate_times(20.0, (num, den)):
                self.assertEqual(round(cover_frames.frame_index(t, (num, den)) * den / num, 6), t)

    def test_base_frames_reads_the_frame_a_time_names(self):
        for num, den in RATES:
            with self.subTest(fps=(num, den)):
                d = self.tmp / f"r{num}_{den}"
                d.mkdir()
                files = [_numbered(d / "a.mkv", num, den, 45, 0), _numbered(d / "b.mkv", num, den, 45, 45)]
                ref = {}
                for k, f in enumerate(files):
                    for j, arr in cover_frames.grab_frames(f, range(45), size=(32, 32)).items():
                        ref[45 * k + j] = float(arr.mean())
                self.assertEqual(len({round(v, 3) for v in ref.values()}), 90)
                cut, end = round(45 * den / num, 6), round(90 * den / num, 6)
                plan = {"fps": [num, den], "segments": [
                    {"k": 0, "src": [0.0, cut], "out": [0.0, cut], "frames": 45},
                    {"k": 1, "src": [cut, end], "out": [cut, end], "frames": 45}]}
                times = [round(n * den / num, 6) for n in range(90)]
                with mock.patch.object(cover_frames, "segment_files", lambda *a, **k: files):
                    frames, origin = cover_frames.base_frames(None, "c01", {}, plan, "vertical", times, (32, 32))
                self.assertEqual(origin, "segments")
                wrong = [n for n, tt in enumerate(times) if abs(float(frames[tt].mean()) - ref[n]) > 1e-6]
                self.assertEqual(wrong, [])


class _MockedCover(_Tmp):
    """make_cover on a project whose frames are supplied by the test (nothing is decoded)."""

    def setUp(self):
        super().setUp()
        import qa_media as qm
        from hookcutter import plan as plan_mod
        (self.tmp / "src.mp4").write_bytes(b"")
        sentences = {"c01": "we sold it first and then we built it"}
        clips = {"c01": {"title": "c01", "pieces": [{"words": [0, 0]}], "cover": {"headline": "SOLD IT *FIRST*"}}}
        self.P = Project(qm.write_project(self.tmp / "P", clips, sentences, self.tmp / "src.mp4"))
        self.plan = qm.plan_for("c01", 0, env.read_json(self.P.root / "transcript.json"), [])
        self.plan_mod = plan_mod
        self.rate(25, 1)
        self.calls = []

    def rate(self, num: int, den: int, dur: float = 3.0):
        n = int(round(dur * num / den))
        self.plan["fps"] = [num, den]
        self.plan["segments"] = [dict(self.plan["segments"][0], src=[100.0, 100.0 + dur],
                                      out=[0.0, round(n * den / num, 6)], frames=n)]
        self.plan["joins"] = []
        self.plan["duration_out"], self.plan["frames_out"] = round(n * den / num, 6), n

    def patched(self, fn):
        def base_frames(project, clip_id, clip, plan, fmt, times, size, letterbox_y=None, style=None):
            self.calls.append({"times": [float(x) for x in times], "letterbox_y": letterbox_y, "style": style})
            shade = np.full((size[1], size[0], 3), 70, np.uint8)
            return {float(x): shade for x in times}, "segments"

        def segment_files(project, clip_id, fmt, plan, style=None):
            self.calls.append({"segment_files": style})
            return None
        with mock.patch.object(cover_frames, "base_frames", base_frames), \
                mock.patch.object(cover_frames, "segment_files", segment_files), \
                mock.patch.object(cover_frames, "detect_faces", lambda arr, min_score=0.6: []), \
                mock.patch.object(self.plan_mod, "load_plan", lambda *a, **k: self.plan):
            return fn()


class CoverTimeRoundTrip(_MockedCover):
    def test_a_recorded_time_is_kept_when_fed_back(self):
        for (num, den), given, frame_time in (((30000, 1001), 0.5, 0.467133), ((30, 1), 0.45, 0.433333),
                                              ((24000, 1001), 1.0, 0.959292)):
            with self.subTest(fps=(num, den)):
                self.rate(num, den)
                first = self.patched(lambda: cover.make_cover(self.P, "c01", "clean", at_out=given))
                self.assertEqual(first["at_out"], frame_time)
                again = self.patched(lambda: cover.make_cover(self.P, "c01", "clean", at_out=first["at_out"]))
                self.assertEqual(again["at_out"], first["at_out"])
                self.assertEqual(cover_frames.frame_index(first["at_out"], (num, den)),
                                 cover_frames.frame_index(given, (num, den)))


# ------------------------------------------------------------------ stage A of this plan and style (C1)

class SegmentLookup(_Tmp):
    def setUp(self):
        super().setUp()
        self.P = self.project({"c01": {"title": "One", "pieces": [{"words": [0, 1]}]}})
        wd = Path(self.P.work_dir("c01", "vertical"))
        self.old, self.own = wd / "seg_old.mkv", wd / "seg_own.mkv"
        for f in (self.old, self.own):
            f.write_bytes(b"x")
        # the newest render log names the stage A file of some other plan or style
        env.write_json(self.P.render_path("c01", "vertical", "clean", ".log.json"),
                       {"segments": [{"k": 0, "path": self.P.rel(self.old)}]})
        self.plan = {"segments": [{"k": 0, "src": [0.0, 1.0], "out": [0.0, 1.0], "frames": 25}]}

    def test_a_style_gets_the_files_of_exactly_that_plan_and_style(self):
        from hookcutter import render
        calls, answer = [], [[self.own]]

        def strict(project, clip_id, fmt, style=None, plan=None, clip=None):
            calls.append((clip_id, fmt, style, plan))
            return answer[0]
        with mock.patch.object(render, "segment_paths", strict):
            got = cover_frames.segment_files(self.P, "c01", "vertical", self.plan, style="pop")
            self.assertEqual(got, [self.own])
            self.assertEqual(calls, [("c01", "vertical", "pop", self.plan)])
            answer[0] = None                 # not rendered from this plan in this style
            self.assertIsNone(cover_frames.segment_files(self.P, "c01", "vertical", self.plan, style="pop"))
            answer[0] = [self.own, self.old]  # not this plan's segment count
            self.assertIsNone(cover_frames.segment_files(self.P, "c01", "vertical", self.plan, style="pop"))

    def test_without_a_style_the_newest_log_still_answers(self):
        got = cover_frames.segment_files(self.P, "c01", "vertical", self.plan)
        self.assertEqual([q.resolve() for q in got], [self.old.resolve()])

    def test_base_frames_passes_the_style_on(self):
        seen = []

        def spy(project, clip_id, fmt, plan, style=None):
            seen.append(style)
            return None
        with mock.patch.object(cover_frames, "segment_files", spy):
            with self.assertRaises(env.HCError):          # no source to fall back to: fine here
                cover_frames.base_frames(self.P, "c01", {}, self.plan, "vertical", [0.2], (90, 160), style="pop")
        self.assertEqual(seen, ["pop"])


class CoverUsesItsOwnStyle(_MockedCover):
    def test_every_frame_of_a_cover_is_asked_for_in_its_style(self):
        self.patched(lambda: cover.make_cover(self.P, "c01", "clean", at_out=1.04))      # not a candidate time
        frames = [c for c in self.calls if "times" in c]
        self.assertGreaterEqual(len(frames), 4)      # candidates, the cover frame, the manual check, the sheet
        self.assertEqual({c["style"] for c in frames}, {"clean"})
        self.assertEqual([c["segment_files"] for c in self.calls if "segment_files" in c], ["clean"])


# ------------------------------------------------------------------ candidate cache (C5)

class CandidateCache(_MockedCover):
    def test_the_key_follows_the_letterbox_position_and_the_shots_data(self):
        from hookcutter import presets
        preset = presets.load("clean", self.P)
        high = {**preset, "letterbox": {**(preset.get("letterbox") or {}), "y_center": 500}}
        a = self.patched(lambda: cover.find_candidates(self.P, "c01", {}, self.plan, preset))
        n = len(self.calls)
        again = self.patched(lambda: cover.find_candidates(self.P, "c01", {}, self.plan, preset))
        self.assertEqual(again["key"], a["key"])
        self.assertEqual(len([c for c in self.calls[n:] if "times" in c]), 0)       # read from the cache
        b = self.patched(lambda: cover.find_candidates(self.P, "c01", {}, self.plan, high))
        self.assertNotEqual(a["key"], b["key"])
        ys = [round(c["letterbox_y"] * 1920) for c in self.calls if "times" in c]
        self.assertEqual(ys, [820, 500])                                             # both were framed anew
        c = self.patched(lambda: cover.find_candidates(self.P, "c01", {}, dict(self.plan, shots_hash="x"), preset))
        self.assertNotEqual(a["key"], c["key"])
        self.assertEqual(len([x for x in self.calls if "times" in x]), 3)


# ------------------------------------------------------------------ source fallback on the frame grid (C4)

def _gap_source(path: Path, drop=None, dur: float = 8.0) -> Path:
    """30 fps, frame N has luma 16 + 10 * (N mod 20); `drop` (n0, n1) removes those source frames
    (their neighbours keep their times: a source with missing frames)."""
    sel = f",select='not(between(n,{drop[0]},{drop[1]}))'" if drop else ""
    env.ff(["-f", "lavfi", "-i", f"color=c=black:s=160x90:r=30:d={dur}",
            "-f", "lavfi", "-i", f"sine=frequency=440:sample_rate=48000:duration={dur}",
            "-vf", "format=yuv420p,geq=lum='16+10*mod(N,20)':cb=128:cr=128" + sel,
            "-fps_mode", "passthrough", "-c:v", "libx264", "-preset", "ultrafast", "-qp", "0",
            "-c:a", "aac", "-shortest", str(path)])
    return path


def _shade(arr) -> int:
    """The N mod 20 a frame of _gap_source carries."""
    return int(round(float(np.asarray(arr).mean()) * 219.0 / 255.0 / 10.0))


class SourceFallbackGrid(_Tmp):
    TIMES = {1.0: 120, 1.7: 138, 1.9: 138, 2.0: 150, 2.5: 165, 3.5: 195}   # out time: source frame on screen

    def plan(self):
        return {"version": 1, "clip": "c1", "kind": "cut", "fps": [30, 1], "grid_offset": 0.0,
                "sample_rate": 48000, "pieces": [], "removals": [], "joins": [], "words": [], "intent": [],
                "segments": [{"k": 0, "piece": 0, "src": [3.0, 7.0], "out": [0.0, 4.0], "frames": 120,
                              "samples": 192000, "join_before": "start", "fade_in": 0.008,
                              "fade_out": 0.008, "gain_db": 0.0}],
                "frames_out": 120, "duration_out": 4.0, "hook_card": None,
                "framing": [{"at_src": 0.0, "mode": "fixed", "x": 0.5, "y": 0.5}], "lints": []}

    def test_missing_frames_do_not_shift_the_fallback(self):
        import fixtures
        from hookcutter import render
        src = _gap_source(self.tmp / "gap.mp4", drop=(139, 149))
        proj = Project(fixtures.make_project(self.tmp, src))
        plan = self.plan()
        clip = {"kind": "cut", "captions": False, "framing": plan["framing"]}
        times = sorted(self.TIMES)
        want = {t: self.TIMES[t] % 20 for t in times}
        before, origin = cover_frames.base_frames(proj, "c1", clip, plan, "vertical", times, (90, 160))
        self.assertEqual(origin, "source")
        self.assertEqual({t: _shade(before[t]) for t in times}, want)
        with mock.patch.object(env, "log", lambda *a, **k: None):
            render.render_clip(proj, "c1", formats=["vertical"], styles=["clean"], plan=plan, clip=clip)
        after, origin = cover_frames.base_frames(proj, "c1", clip, plan, "vertical", times, (90, 160))
        self.assertEqual(origin, "segments")
        # what render shows: the same frame (one frame apart is 11 grey levels; render's own
        # colour path moves a flat grey by a few)
        apart = {t: round(abs(float(after[t].mean()) - float(before[t].mean())), 1) for t in times}
        self.assertLess(max(apart.values()), 6.0, apart)

    def test_a_constant_rate_source_reads_the_same_frames(self):
        import fixtures
        src = _gap_source(self.tmp / "even.mp4")
        proj = Project(fixtures.make_project(self.tmp, src))
        plan = self.plan()
        times = sorted(self.TIMES)
        frames, origin = cover_frames.base_frames(proj, "c1", {"kind": "cut"}, plan, "vertical", times, (90, 160))
        self.assertEqual(origin, "source")
        self.assertEqual({t: _shade(frames[t]) for t in times},
                         {t: int(round((3.0 + t) * 30)) % 20 for t in times})
        # index n is decoded frame n whether or not the grid is asked for
        plain = cover_frames.grab_frames(src, [0, 7, 31], size=(160, 90), ss=2.5)
        grid = cover_frames.grab_frames(src, [0, 7, 31], size=(160, 90), ss=2.5, grid=(30, 1))
        self.assertEqual(sorted(grid), [0, 7, 31])
        for n in plain:
            self.assertTrue(np.array_equal(plain[n], grid[n]), n)


# ------------------------------------------------------------------ real renders (C1, X2)

class RenderedStyles(unittest.TestCase):
    """One clip rendered in two styles whose stage A differs (the letterbox sits elsewhere)."""

    WORDS = [("So", 1.0, 1.2), ("this", 1.25, 1.5), ("works.", 1.55, 2.0),
             ("And", 3.5, 3.7), ("then", 3.75, 4.0), ("ends.", 4.05, 4.5)]

    @classmethod
    def setUpClass(cls):
        import argparse
        import fixtures
        from hookcutter import render
        cls.tmp = Path(tempfile.mkdtemp(prefix="hc-codex-ff-render-"))
        cls.src = fixtures.make_speech_fixture(cls.tmp, cls.WORDS, dur=7.0)
        cls.root = fixtures.make_project(cls.tmp, cls.src, cls.WORDS)
        cls.P = Project(cls.root)
        _preset(cls.P, "high", letterbox={"y_center": 500})
        env.write_json(cls.root / "clips.json", {"version": 1, "defaults": {"styles": ["clean"]}, "clips": {
            "c1": {"title": "One", "pieces": [{"words": [0, 2]}], "captions": False, "styles": ["clean", "high"],
                   "framing": [{"at_src": 0, "mode": "letterbox"}]}}})
        cls.render = staticmethod(lambda P, clips: cls._quiet(
            render.cmd_render, argparse.Namespace(project=str(P), clips=clips, styles=None, formats=None,
                                                  force=False)))
        cls.strict = cover_frames._strict_lookup(render.segment_paths)
        if cls.strict:
            assert cls.render(cls.root, ["c1"]) == 0

    @staticmethod
    def _quiet(fn, *a):
        import contextlib
        import io
        with contextlib.redirect_stdout(io.StringIO()), mock.patch.object(env, "log", lambda *x, **k: None):
            return fn(*a)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def setUp(self):
        if not self.strict:
            self.skipTest("render.segment_paths has no plan and style lookup in this build")

    def test_each_style_reads_its_own_stage_a(self):
        from hookcutter import plan as plan_mod
        plan = plan_mod.load_plan(self.P, "c1")
        t = 0.4
        n = cover_frames.frame_index(t, plan["fps"])
        got, files = {}, {}
        for st in ("clean", "high"):
            log = env.read_json(self.P.render_path("c1", "vertical", st, ".log.json"))
            logged = [(self.P.root / s["path"]).resolve() for s in log["segments"]]
            files[st] = cover_frames.segment_files(self.P, "c1", "vertical", plan, style=st)
            self.assertEqual([q.resolve() for q in files[st]], logged)
            frames, origin = cover_frames.base_frames(self.P, "c1", {}, plan, "vertical", [t], (270, 480), style=st)
            self.assertEqual(origin, "segments")
            own = cover_frames.grab_frames(files[st][0], [n], size=(270, 480))[n]
            self.assertTrue(np.array_equal(frames[t], own), st)
            got[st] = frames[t].astype(np.float32)
        self.assertNotEqual(files["clean"], files["high"])
        self.assertGreater(float(np.abs(got["clean"] - got["high"]).mean()), 5.0)

    def test_export_holds_a_render_made_before_its_preset_changed(self):
        _qa(self.P, "c1", styles=("clean", "high"))
        self.assertEqual(export.export(self.P)["held"], [])
        mine = self.P.root / "presets" / "clean.json"
        try:
            _preset(self.P, "clean", caption={"size": 90})
            res = export.export(Project(self.root))
        finally:
            mine.unlink()
        held = {h["style"]: h["reason"] for h in res["held"]}
        self.assertIn("predates its inputs", held.get("clean", ""))
        self.assertEqual(export.export(Project(self.root))["held"], [])     # the preset is back: current again

    def test_a_cover_made_before_the_render_says_so(self):
        import fixtures
        from hookcutter import plan as plan_mod
        root = fixtures.make_project(self.tmp / "b", self.src, self.WORDS)
        clip = {"title": "One", "pieces": [{"words": [0, 2]}], "captions": False,
                "cover": {"headline": "PLAIN WORDS"}}
        env.write_json(root / "clips.json", {"version": 1, "defaults": {"styles": ["clean"]}, "clips": {"c1": clip}})
        self.assertEqual(self.render(root, ["c1"]), 0)
        first = self._quiet(cover.make_cover, Project(root), "c1", "clean")
        self.assertEqual(first["frame_source"], "segments")
        clip["pieces"] = [{"words": [3, 5]}]              # another range, the same segment count
        env.write_json(root / "clips.json", {"version": 1, "defaults": {"styles": ["clean"]}, "clips": {"c1": clip}})
        P = Project(root)
        plan = plan_mod.load_plan(P, "c1")
        self.assertIsNone(cover_frames.segment_files(P, "c1", "vertical", plan, style="clean"))
        stale = self._quiet(cover.make_cover, P, "c1", "clean")
        self.assertEqual(stale["frame_source"], "source")
        self.assertTrue(any("approximated" in w for w in stale["warnings"]), stale["warnings"])
        self.assertEqual(self.render(root, ["c1"]), 0)
        healed = self._quiet(cover.make_cover, Project(root), "c1", "clean")
        self.assertEqual(healed["frame_source"], "segments")
        self.assertFalse(any("approximated" in w for w in healed["warnings"]), healed["warnings"])


if __name__ == "__main__":
    unittest.main()
