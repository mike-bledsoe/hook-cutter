"""QA gates against seeded defects (contract section 12) plus focused unit tests of qa.py.

Each seeded clip carries exactly one defect and must produce exactly the stated status on the
stated gate, with no other gate going WARN or FAIL; the clean control has no WARN or FAIL. Fast
runs skip the ASR word diff (G5), so the clean control is INCOMPLETE (a PASS with an evidence
gap is never reported as PASS); G10's energy checks run without ASR. The slow class
re-transcribes real speech (HC_SLOW=1).
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import unittest

import numpy as np
from fractions import Fraction
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "scripts"))

import qa_media as qm  # noqa: E402
from hookcutter import cover, env, qa  # noqa: E402
from hookcutter import render as render_mod  # noqa: E402
from hookcutter import plan as plan_mod  # noqa: E402
from hookcutter.project import Project  # noqa: E402

try:
    from fixtures import slow  # noqa: E402
except Exception:  # pragma: no cover
    slow = unittest.skipUnless(os.environ.get("HC_SLOW") == "1", "slow test (run: hc test --slow)")

CLIPS = {
    # id: (sentence spoken in the clip, hook lines, seeded defect)
    "c-clean": ("this is a big idea that matters", ["*BIG* IDEA"], None),
    "c-96k": ("small wins add up over time", ["*SMALL* WINS"], "96k"),
    "c-quiet": ("loud rooms hide weak arguments", ["LOUD *ROOMS*"], "quiet"),
    "c-black": ("keep the shot honest always", ["*HONEST* SHOT"], "black"),
    "c-cover": ("covers sell the click first", ["*COVERS* SELL"], "cover"),
    "c-safe": ("captions belong inside the frame", ["*CAPTIONS* BELONG"], "safe"),
    "c-num": ("there are rules for writing hooks", ["*3* RULES FOR HOOKS"], "number"),
}
SEEDED = {"c-96k": ("G3", "FAIL"), "c-quiet": ("G7", "FAIL"), "c-black": ("G8", "FAIL"),
          "c-cover": ("G12", "FAIL"), "c-safe": ("G9", "WARN"), "c-num": ("G11", "FAIL")}


def _fake_build_plan(project, clip_id, **_kw):
    return env.read_json(project.plan_path(clip_id))


def _fake_expected(project, clip_id, fmt, style, **_kw):
    """The render key a render made now would record: the seeded logs carry "fresh"."""
    return {"plan": None, "render": "fresh"}


class SeededDefects(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="hc-b6-qa-"))
        media = cls.tmp / "media"
        media.mkdir()
        good = qm.make_render(media / "good.mp4")
        variants = {
            None: good,
            "96k": qm.make_render(media / "96k.mp4", ar=96000),
            "quiet": qm.make_render(media / "quiet.mp4", lufs=-20.0),
            "black": qm.make_render(media / "black.mp4", black_frames=(37, 38, 39)),
        }
        md5 = {d: render_mod.audio_md5(p) for d, p in variants.items()}   # what a render log records
        source = qm.make_source(media / "source.mp4")
        sentences = {cid: v[0] for cid, v in CLIPS.items()}
        clips = {}
        for cid, (_s, hook, _d) in CLIPS.items():
            # the title is a written surface too (G11): one that states nothing, so that each
            # clip keeps exactly its one seeded defect ("c-96k" as a title reads as the number 96k)
            clips[cid] = {"title": "seeded clip", "pieces": [{"words": [0, 0]}], "hook_card": {"lines": hook}}
        P = qm.write_project(cls.tmp / "P", clips, sentences, source)
        cls.project = Project(P)
        tr = env.read_json(P / "transcript.json")
        photo = cover.Image.new("RGB", (1080, 1920), (60, 70, 90))
        cfg = cover.DEFAULT_COVER
        for k, (cid, (_s, hook, defect)) in enumerate(CLIPS.items()):
            plan = qm.plan_for(cid, k, tr, hook)
            env.write_json(cls.project.plan_path(cid), plan)
            which = defect if defect in variants else None
            render = variants.get(which)
            shutil.copy(render, cls.project.render_path(cid, "vertical", "clean"))
            cap = qm.captions_for(hook, page_box=(40, 1150, 600, 110) if defect == "safe" else (200, 1150, 600, 110))
            env.write_json(cls.project.render_path(cid, "vertical", "clean", ".captions.json"), cap)
            env.write_json(cls.project.render_path(cid, "vertical", "clean", ".log.json"),
                           {"cmds": [], "warnings": [], "audio_md5": md5[which],
                            "hashes": {"plan": render_mod.plan_hash(plan), "render": "fresh"},
                            "framing_windows": [{"seg": 0, "out": [0.0, 3.0], "mode": "crop", "lo": 0.2,
                                                 "hi": 0.5, "multi_face": False}]})
            env.write_json(cls.project.shots_path(cid), {"clip": cid, "shots": [
                {"start_src": 4.0 * k, "end_src": 4.0 * k + 4.0, "kind": "single",
                 "faces": [{"x": 0.35, "y": 0.4, "h": 0.4, "presence": 1.0}]}]})
            img, info = cover.compose(photo, [], " ".join(hook), cfg)
            main = cls.project.cover_path(cid, "clean", "main")
            cover.save_jpeg(img, main)
            grid = cover.grid_crop(img)
            cover.save_jpeg(grid, cls.project.cover_path(cid, "clean", "grid"))
            cover.save_jpeg(cover.mobile_preview(grid), cls.project.cover_path(cid, "clean", "mobile"))
            env.write_json(cls.project.cover_path(cid, "clean", "json"),
                           {"headline_box": info["headline_box"], "text_box": info["text_box"],
                            "face_box": [400, 300, 280, 320], "faces_found": 1, "crown_y": 300,
                            "band": [240, 1680], "spec_hash": plan["spec_hash"]})
            if defect == "cover":
                data = Path(main).read_bytes()
                Path(main).write_bytes(data[: len(data) // 2])
        with mock.patch.object(plan_mod, "build_plan", _fake_build_plan), \
                mock.patch.object(render_mod, "expected_hashes", _fake_expected):
            cls.code = qa.run(cls.project, None, None, asr=False)
            cls.reports = {cid: env.read_json(cls.project.qa_path(cid, ".json")) for cid in CLIPS}
            cls.summary = Path(cls.project.qa_path("summary", ".md")).read_text(encoding="utf-8")
            cls.code_clean = qa.run(cls.project, ["c-clean"], None, asr=False)
            cls.code_cov = qa.run(cls.project, ["c-clean"], None, asr=False, require_coverage=True)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _gates(self, cid, gid):
        return [g for g in self.reports[cid]["gates"] if g["id"] == gid]

    def test_clean_control_passes(self):
        rep = self.reports["c-clean"]
        bad = [(g["id"], g["status"], g["evidence"]) for g in rep["gates"] if g["status"] in ("WARN", "FAIL")]
        self.assertEqual(bad, [])
        # --no-asr leaves the word diff unchecked: a PASS with gaps is INCOMPLETE, never PASS
        self.assertEqual(rep["status"], "INCOMPLETE")
        self.assertFalse(rep["complete"])
        self.assertEqual(rep["checked"], {"styles": ["clean"], "formats": ["vertical"]})
        self.assertEqual(rep["stale_renders"], [])
        for gid in ("G0", "G1", "G2", "G3", "G6", "G7", "G8", "G9", "G10", "G11", "G12", "G13", "G14", "G15"):
            self.assertEqual({g["status"] for g in self._gates("c-clean", gid)}, {"PASS"}, gid)
        g = self._gates("c-clean", "G5")[0]
        self.assertEqual(g["status"], "SKIPPED")
        self.assertTrue(g.get("gap"))
        g16 = self._gates("c-clean", "G16")[0]
        self.assertEqual((g16["status"], g16.get("gap")), ("SKIPPED", None))   # single-face clip: not applicable

    def test_seeded_defects_exact_status(self):
        for cid, (gid, status) in SEEDED.items():
            with self.subTest(cid=cid):
                got = [g for g in self.reports[cid]["gates"] if g["status"] in ("WARN", "FAIL")]
                self.assertEqual([(g["id"], g["status"]) for g in got], [(gid, status)],
                                 [(g["id"], g["status"], g["evidence"]) for g in got])

    def test_evidence_names_the_defect(self):
        # two G3 rows per clip: the audio-against-its-log row (PASS here) and the streams row
        self.assertIn("96000", [g for g in self._gates("c-96k", "G3") if g["status"] == "FAIL"][0]["evidence"])
        self.assertIn("LUFS", self._gates("c-quiet", "G7")[0]["evidence"])
        self.assertIn("join", self._gates("c-black", "G8")[0]["evidence"])
        self.assertIn("decode", self._gates("c-cover", "G12")[0]["evidence"])
        self.assertIn("safe area", self._gates("c-safe", "G9")[0]["evidence"])
        self.assertIn('"3"', self._gates("c-num", "G11")[0]["evidence"])

    def test_exit_codes_and_reports(self):
        self.assertEqual(self.code, 1)
        self.assertEqual(self.code_clean, 0)
        self.assertEqual(self.code_cov, 2)
        md = Path(self.project.qa_path("c-clean", ".md")).read_text(encoding="utf-8")
        self.assertIn("## EYES", md)
        self.assertIn("covers-grid.jpg", md)
        self.assertIn("c-clean__clean.grid.jpg", md)
        self.assertIn("cell 1 is frame 0", md)
        summary = self.summary
        for cid in CLIPS:
            self.assertIn(cid, summary)
        # a later single-clip run keeps every clip in summary.md, marked as not re-run
        after = Path(self.project.qa_path("summary", ".md")).read_text(encoding="utf-8")
        for cid in CLIPS:
            self.assertIn(cid, after)
        self.assertIn("not re-run", after)
        self.assertNotIn(chr(0x2014), md + summary)


class StylesDiverged(unittest.TestCase):
    def test_audio_md5_mismatch_fails(self):
        tmp = Path(tempfile.mkdtemp(prefix="hc-b6-sty-"))
        try:
            P = Project(tmp)
            renders = {}
            for st, md5 in (("clean", "aaa"), ("pop", "bbb")):
                env.write_json(P.render_path("c1", "vertical", st, ".log.json"), {"audio_md5": md5})
                renders[st] = P.render_path("c1", "vertical", st)
            ctx = qa._Ctx(P, {"clips": {}}, None, False, False)
            g = qa.g_styles_audio(ctx, "c1", "vertical", renders)
            self.assertEqual(g["status"], "FAIL")
            self.assertIn("styles diverged", g["evidence"])
            env.write_json(P.render_path("c1", "vertical", "pop", ".log.json"), {"audio_md5": "aaa"})
            self.assertEqual(qa.g_styles_audio(ctx, "c1", "vertical", renders)["status"], "PASS")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class WordDiffRules(unittest.TestCase):
    def test_negation_and_content_are_hard(self):
        d = qa.diff_words("I do not like the product launch".split(), "I do like the launch".split())
        self.assertEqual(len(d["fail"]), 2)
        self.assertTrue(any("negation" in f for f in d["fail"]))
        self.assertTrue(any('"product"' in f for f in d["fail"]))

    def test_function_word_is_soft(self):
        d = qa.diff_words("so we built the thing".split(), "we built thing".split())
        self.assertEqual(d["fail"], [])
        self.assertEqual(len(d["warn"]), 2)

    def test_stutter_only_soft_when_twin_survives(self):
        self.assertEqual(qa.diff_words("shipping shipping beats waiting".split(), "shipping beats waiting".split())["fail"], [])
        self.assertTrue(qa.diff_words("shipping shipping beats waiting".split(), "beats waiting".split())["fail"])

    def test_numbers_and_contractions_normalize(self):
        d = qa.diff_words("it's twenty five dollars don't wait".split(), "it is $25 do not wait".split())
        self.assertEqual((d["fail"], d["warn"]), ([], []))

    def test_replace_losing_negation_fails(self):
        d = qa.diff_words("you should never quit".split(), "you should ever quit".split())
        self.assertTrue(any("negation" in f for f in d["fail"]))

    def test_replace_is_warn_and_extra_is_warn(self):
        d = qa.diff_words("the project shipped".split(), "the product shipped today".split())
        self.assertEqual(d["fail"], [])
        self.assertTrue(any("heard" in w for w in d["warn"]))
        self.assertTrue(any("extra" in w for w in d["warn"]))

    def test_words_heard_in_silence_count_as_missing(self):
        ref, hyp = "we never stopped shipping new features".split(), "we never stopped showing new features".split()
        self.assertEqual(qa.diff_words(ref, hyp)["fail"], [])
        d = qa.diff_words(ref, hyp, silent=lambda j: j == 3)
        self.assertTrue(any('"shipping"' in f for f in d["fail"]))
        self.assertEqual(qa.diff_words(ref, hyp, silent=lambda j: False)["fail"], [])

    def test_pairs_map_word_indices(self):
        d = qa.diff_words(["Hello,", "big", "world."], ["hello", "world"])
        self.assertIn((0, 0), d["pairs"])
        self.assertIn((2, 1), d["pairs"])


class Parsers(unittest.TestCase):
    def test_ebur128(self):
        err = ("[Parsed_ebur128_0 @ 0x1] t: 2.9 M: -14 S: -14 I: -30.0 LUFS\n"
               "[Parsed_ebur128_0 @ 0x1] Summary:\n\n  Integrated loudness:\n    I:         -14.3 LUFS\n"
               "  True peak:\n    Peak:       -1.6 dBFS\n")
        self.assertEqual(qa.parse_ebur128(err), (-14.3, -1.6))
        self.assertEqual(qa.parse_ebur128("nothing"), (None, None))

    def test_black_and_freeze(self):
        err = ("[blackdetect @ 0x] black_start:0 black_end:0.04 black_duration:0.04\n"
               "[blackdetect @ 0x] black_start:2.0 black_end:2.08 black_duration:0.08\n"
               "[freezedetect @ 0x] lavfi.freezedetect.freeze_start: 5.0\n"
               "[freezedetect @ 0x] lavfi.freezedetect.freeze_duration: 1.5\n"
               "[freezedetect @ 0x] lavfi.freezedetect.freeze_end: 6.5\n"
               "[freezedetect @ 0x] lavfi.freezedetect.freeze_start: 9.0\n")
        blacks, freezes = qa.parse_black_freeze(err, 10.0)
        self.assertEqual(blacks, [(0.0, 0.04), (2.0, 2.08)])
        self.assertEqual(freezes, [(5.0, 6.5), (9.0, 10.0)])
        st, notes = qa.judge_black_freeze(blacks, freezes, [{"out": 2.04, "kind": "tighten"}], Fraction(25))
        self.assertEqual(st, "FAIL")
        self.assertEqual(sum(n.startswith("FAIL") for n in notes), 2)
        st, _ = qa.judge_black_freeze([(2.0, 2.04)], [], [{"out": 2.04, "kind": "dip"}], Fraction(25))
        self.assertEqual(st, "WARN")
        # a single black frame away from any join is a visible flash: WARN (it used to need 0.08 s)
        st, notes = qa.judge_black_freeze([(5.0, 5.04)], [], [{"out": 2.0, "kind": "cut"}], Fraction(25))
        self.assertEqual(st, "WARN")
        self.assertIn("1 frame", notes[0])
        self.assertEqual(qa.judge_black_freeze([], [], [{"out": 2.0, "kind": "cut"}], Fraction(25))[0], "PASS")

    def test_theil_sen_drift(self):
        xs = [0.5 * i for i in range(40)]
        ys = [0.01 * x + (0.05 if i == 7 else 0.0) for i, x in enumerate(xs)]
        self.assertAlmostEqual(qa.theil_sen(xs, ys), 0.01, places=3)
        self.assertIsNone(qa.theil_sen([1.0], [0.0]))

    def test_layout_and_faces(self):
        zone = {"top": 240, "bottom": 420, "left": 90, "right": 150}
        self.assertEqual(qa.layout_issues([("p", [100, 300, 500, 100])], 1080, 1920, zone)[0], "PASS")
        self.assertEqual(qa.layout_issues([("p", [50, 300, 500, 100])], 1080, 1920, zone)[0], "WARN")
        self.assertEqual(qa.layout_issues([("p", [700, 300, 500, 100])], 1080, 1920, zone)[0], "FAIL")
        face = [400, 400, 200, 260]
        self.assertTrue(qa.face_hits([300, 600, 400, 80], [face]))
        self.assertFalse(qa.face_hits([300, 420, 400, 40], [face]))

    def test_faithfulness_helpers(self):
        self.assertEqual(qa.surface_numbers("THE ONE THING"), [])
        self.assertEqual([v for v, _ in qa.surface_numbers("*3* RULES for $8M")], [3.0, 8e6])
        self.assertIn(8e6, qa.spoken_numbers("we made eight million dollars".split()))
        ev = qa._proper_noun_evidence([{"w": w} for w in "I use Claude daily. we like Claude a lot".split()])
        self.assertEqual(qa.surface_names("WHY CLAUDE WINS", ev, []), ["CLAUDE"])
        self.assertEqual(qa.surface_names("Why I switched to Notion", {}, []), ["Notion"])

    def test_cover_file_check(self):
        tmp = Path(tempfile.mkdtemp(prefix="hc-b6-cf-"))
        try:
            p = tmp / "c.jpg"
            cover.save_jpeg(cover.Image.new("RGB", (1080, 1920), (20, 30, 40)), p)
            self.assertEqual(qa.check_cover_file(p), [])
            data = p.read_bytes()
            p.write_bytes(data[: len(data) // 2])
            self.assertTrue(qa.check_cover_file(p))
            p.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 16000)
            self.assertTrue(qa.check_cover_file(p))
            self.assertEqual(qa.check_cover_file(tmp / "none.jpg"), ["missing"])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


@slow
class AsrWordDiff(unittest.TestCase):
    """Real re-transcription: synthesized speech, one render intact, one with a content word muted."""

    def test_g5_g10_on_speech(self):
        say = shutil.which("say")
        if not say:
            self.skipTest("needs a speech synthesizer (macOS say)")
        from hookcutter import transcribe
        tmp = Path(tempfile.mkdtemp(prefix="hc-b6-asr-"))
        try:
            aiff = tmp / "speech.aiff"
            sentence = "Our revenue doubled because we never stopped shipping new features every single week."
            env.run([say, "-o", str(aiff), sentence])
            wav = tmp / "speech.wav"
            env.ff(["-i", str(aiff), "-af", "apad=pad_dur=0.6,adelay=400|400", "-ar", "48000", "-ac", "1", str(wav)])
            backend = "whispercpp" if (shutil.which("whisper-cli") and os.environ.get("HOOKCUTTER_WHISPERCPP_MODEL")) else "auto"
            words = transcribe.transcribe_media(wav, backend=backend)
            # A recogniser that hears next to nothing in clean speech is the fault this test
            # exists to catch: it fails, it does not skip (only a missing synthesiser skips).
            self.assertGreaterEqual(len(words), 8, "ASR returned too few words: %r" % [w.get("w") for w in words])
            dur = float(env.ffprobe_json(wav, "-show_format")["format"]["duration"])
            frames = int(dur * qm.FPS)
            dur = frames / qm.FPS
            P = Project(tmp / "P")
            env.write_json(P.root / "transcript.json", {"backend": backend, "model": None, "language": "en", "glossary": [],
                                                        "words": words})
            plan = {"fps": [qm.FPS, 1], "duration_out": dur, "frames_out": frames,
                    "words": [{"i": w["i"], "raw": w["w"], "text": w["w"], "s_out": w["s"], "e_out": w["e"]} for w in words],
                    "intent": [w["i"] for w in words], "joins": [], "segments": []}
            k = next(i for i, w in enumerate(words) if "shipping" in w["w"].lower())
            # Mute the word's real audio: ASR word times are approximate (mlx turbo put the edges
            # 20-40 ms inside the word, leaving fragments it heard as "shopping"), so take the
            # energy minimum near each stamped edge as the boundary.
            db = qa.hop_db(qa.decode_mono(wav))

            def dip(t, before, after):
                a, b = max(0, int((t - before) / 0.01)), min(len(db), int((t + after) / 0.01) + 1)
                return (a + int(np.argmin(db[a:b]))) * 0.01

            gap = (dip(words[k]["s"], 0.12, 0.06), dip(words[k]["e"], 0.06, 0.12) + 0.01)
            for name, mute in (("ok", None), ("cut", gap)):
                out = tmp / f"{name}.mp4"
                af = "anull" if mute is None else f"volume=enable='between(t,{mute[0]},{mute[1]})':volume=0"
                env.ff(["-f", "lavfi", "-i", f"color=c=gray:s=1080x1920:r={qm.FPS}:d={dur}", "-i", str(wav),
                        "-map", "0:v", "-map", "1:a", "-af", af, "-frames:v", str(frames), "-c:v", "libx264",
                        "-preset", "ultrafast", "-pix_fmt", "yuv420p", "-c:a", "aac", "-ar", "48000", "-ac", "2", str(out)])
                ctx = qa._Ctx(P, {"clips": {}}, None, True, False)
                g5, diff, hyp = qa.g5_words(ctx, None, plan, out, "vertical (audio)")
                if name == "ok":
                    self.assertIn(g5["status"], ("PASS", "WARN"), g5)
                    cap = {"lead_ms": 60, "pages": [{"i": 0, "start_out": max(0.0, words[0]["s"] - 0.06),
                                                     "end_out": words[-1]["e"],
                                                     "tokens": [{"text": w["w"], "s_out": w["s"], "e_out": w["e"],
                                                                 "idx": [w["i"]]} for w in words]}]}
                    g10 = qa.g10_sync(ctx, plan, cap, diff, hyp, 0.06, "vertical/clean", render=out)
                    self.assertEqual(g10["status"], "PASS", g10)
                else:
                    self.assertEqual(g5["status"], "FAIL", g5)
                    self.assertIn("shipping", g5["evidence"])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
