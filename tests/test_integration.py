"""Integration regressions: defects found by running the whole chain on a real two-person
recording (a 58-minute, 640x360 video call) and fixed across modules. Each test pins one fix."""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "scripts"))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from hookcutter import env, project, qa, shots, textnorm  # noqa: E402
from hookcutter import transcribe as T  # noqa: E402

import test_captions as TC  # noqa: E402  (helpers only: make_plan, build, style)
import test_plan as TP  # noqa: E402  (helpers only: build, seq-free word lists)


def W(w, s, e, sp=None, p=1.0):
    return {"w": w, "s": s, "e": e, "p": p, "sp": sp}


class TestRootOf(unittest.TestCase):
    def test_plain_path_is_not_slash(self):
        # pathlib.Path has its own .root ("/"): getattr(p, "root", p) silently read "/"
        d = Path(tempfile.mkdtemp(prefix="hc-root-"))
        try:
            self.assertEqual(project.root_of(d), d.absolute())
            self.assertEqual(project.root_of(str(d)), d.absolute())
            self.assertEqual(project.root_of(project.Project(d)), d.absolute())
            self.assertIsNone(project.root_of(None))
            from hookcutter import spec
            self.assertEqual(spec.spec_path(d), d.absolute() / "clips.json")
        finally:
            shutil.rmtree(d, ignore_errors=True)


class TestShotCuts(unittest.TestCase):
    def frames(self, n, change_at, kind):
        rng = np.random.default_rng(3)
        base = rng.uniform(40, 200, (36, 64)).astype(np.float32)
        other = rng.uniform(40, 200, (36, 64)).astype(np.float32)
        out = []
        for k in range(n):
            g = base.copy()
            if kind == "cut" and k >= change_at:
                g = other.copy()
            if kind == "gesture" and change_at <= k < change_at + 3:
                g[10:30, 5:25] = 255.0           # a hand sweeping through a fifth of the frame
            out.append(g)
        return out

    def test_gesture_is_not_a_cut(self):
        times = [k / 4.0 for k in range(40)]
        cuts, _ = shots.detect_cuts(times, self.frames(40, 20, "gesture"))
        self.assertEqual(cuts, [])

    def test_camera_change_is_a_cut(self):
        times = [k / 4.0 for k in range(40)]
        cuts, _ = shots.detect_cuts(times, self.frames(40, 20, "cut"))
        self.assertEqual(len(cuts), 1)
        self.assertAlmostEqual(cuts[0][0], (times[19] + times[20]) / 2, places=6)

    def test_background_face_is_minor(self):
        samples = [{"faces": [[0.54, 0.42, 0.25, 0.45, 0.9], [0.31, 0.55, 0.06, 0.10, 0.7]]}
                   for _ in range(10)]
        fl = shots.shot_faces(samples)
        self.assertEqual(shots.shot_kind(fl), "single")      # the shelf photo does not make "two"
        self.assertTrue(any(f.get("minor") for f in fl))
        prop = shots.propose([{"start_src": 0.0, "kind": "single", "faces": fl}])
        self.assertAlmostEqual(prop[0]["x"], 0.54, places=2)


class TestPlanFixes(unittest.TestCase):
    def test_late_asr_start_is_rescued(self):
        # ASR stamps "issue" 1.0-2.0 and "I" at 2.1; the audio says "issue" 1.0-1.5, a pause,
        # then "I had" from 1.75. The contract window (around 2.1) sits inside the voiced run.
        words = [("common", 0.4, 0.95), ("issue.", 1.0, 2.0), ("I", 2.1, 2.3), ("had", 2.3, 2.6),
                 ("a", 2.6, 2.7), ("client.", 2.7, 3.2)]
        spans = [(0.4, 0.95), (1.0, 1.5), (1.75, 3.2)]
        plan, _, _ = TP.build(words, {"c01": {"pieces": [{"words": [2, 5]}]}}, spans=spans,
                              defaults={"tighten": {"enabled": False}})
        e = plan["pieces"][0]["edge_in"]
        self.assertEqual(e["guard"], "late_asr")
        self.assertLess(e["src"], 1.75)
        self.assertGreaterEqual(e["src"], 1.5)
        first = next(w for w in plan["words"] if w["i"] == 2)
        self.assertLess(first["s_out"], 0.2)            # caption starts at the audible onset

    def test_early_stamped_previous_word_is_not_rescued(self):
        # Here the pause is BEFORE the previous word ("common [pause] issue I"): ASR stamped
        # "issue" early into the pause. Cutting at the pause would put "issue" in the clip.
        words = [("common", 0.4, 0.95), ("issue.", 1.0, 2.0), ("I", 2.1, 2.3), ("had", 2.3, 2.6),
                 ("a", 2.6, 2.7), ("client.", 2.7, 3.2)]
        spans = [(0.4, 1.15), (1.5, 3.2)]
        plan, _, _ = TP.build(words, {"c01": {"pieces": [{"words": [2, 5]}]}}, spans=spans,
                              defaults={"tighten": {"enabled": False}})
        e = plan["pieces"][0]["edge_in"]
        self.assertNotEqual(e["guard"], "late_asr")
        self.assertGreater(e["src"], 1.8)

    def test_l11_ignores_the_analysis_range_start(self):
        words = [("w%d" % k, 10.0 + k * 0.5, 10.3 + k * 0.5) for k in range(20)]
        shots_json = {"ranges": [[9.0, 21.0]], "cuts": [],
                      "shots": [{"start_src": 9.0, "end_src": 21.0, "kind": "single", "faces": []}]}
        clip = {"pieces": [{"words": [0, 19]}], "framing": [{"at_src": 0.0, "mode": "track", "x": 0.5}]}
        plan, _, _ = TP.build(words, {"c01": clip}, shots=shots_json)
        self.assertFalse([x for x in plan["lints"] if x["id"] == "L11"])
        # a real cut inside the clip still warns
        shots_json = {"ranges": [[9.0, 21.0]], "cuts": [{"src": 15.0}],
                      "shots": [{"start_src": 9.0, "end_src": 15.0, "kind": "single", "faces": []},
                                {"start_src": 15.0, "end_src": 21.0, "kind": "single", "faces": []}]}
        plan, _, _ = TP.build(words, {"c01": clip}, shots=shots_json)
        self.assertTrue([x for x in plan["lints"] if x["id"] == "L11"])

    def test_l08_accent_spoken_only_under_the_card(self):
        words = [("reach", 1.0, 1.3), ("forty", 1.3, 1.6), ("people.", 1.6, 2.0)] + \
            [("w%d" % k, 2.5 + k * 0.4, 2.8 + k * 0.4) for k in range(30)]
        clip = {"pieces": [{"words": [0, 32]}], "hook_card": {"lines": ["*40* PEOPLE"], "dur": 2.5}}
        plan, _, _ = TP.build(words, {"c01": clip})
        msgs = [x["msg"] for x in plan["lints"] if x["id"] == "L08"]
        self.assertTrue(any("only under the hook card" in m for m in msgs), msgs)

    def test_l08_matches_the_caption_hook_trim(self):
        """Captions drop a word whose midpoint is before the card end (less the 60 ms lead), so
        an accent starting 0.3 s before the card ends is never shown: L08 must say so. An accent
        whose midpoint is after the card end is captioned and gets no under-card warning."""
        def l08(start):
            words = [("so", 0.0, 0.3), ("we", 0.3, 0.6), ("reach", 0.6, 1.0),
                     ("out", 1.0, start), ("forty", start, start + 0.3),
                     ("people.", start + 0.3, start + 0.7)] + \
                [("w%d" % k, 4.0 + k * 0.4, 4.3 + k * 0.4) for k in range(30)]
            clip = {"pieces": [{"words": [0, 35]}], "hook_card": {"lines": ["*40* PEOPLE"], "dur": 2.5}}
            plan, _, _ = TP.build(words, {"c01": clip}, defaults={"tighten": {"enabled": False}})
            w = [x for x in plan["words"] if x["text"] == "forty"][0]
            self.assertAlmostEqual(w["s_out"], start, delta=0.05)
            return [x["msg"] for x in plan["lints"] if x["id"] == "L08"]
        self.assertTrue(any("only under the hook card" in m for m in l08(2.2)))
        self.assertFalse(any("only under the hook card" in m for m in l08(2.4)))


class TestTranscriptFixes(unittest.TestCase):
    def test_capitalized_ordinary_word_is_not_a_name(self):
        words = [W("cohort,", 1.0, 1.4), W("Everyone", 1.5, 1.8, p=0.02), W("would", 1.8, 2.0),
                 W("help", 2.0, 2.2), W("everyone.", 2.2, 2.6), W("Ask", 2.7, 2.9),
                 W("Zorblat", 2.9, 3.3, p=0.2)]
        flagged = [f["w"] for f in T.low_confidence(words)]
        self.assertNotIn("Everyone", flagged)
        self.assertIn("Zorblat", flagged)

    def test_gap_between_same_speaker_cues(self):
        cues = [{"s": 0.0, "e": 2.0, "sp": "A", "text": "one two three"},
                {"s": 2.6, "e": 4.0, "sp": "A", "text": "five six"},
                {"s": 5.0, "e": 6.0, "sp": "B", "text": "seven"}]
        words = [W("one", 0.1, 0.5), W("and", 2.2, 2.4), W("five", 2.7, 3.0), W("x", 4.5, 4.6),
                 W("seven", 5.1, 5.5)]
        T.assign_speakers(words, cues)
        self.assertEqual([w["sp"] for w in words], ["A", "A", "A", None, "B"])

    def test_crosstalk_blip_is_relabelled(self):
        cues = [{"s": 0.0, "e": 2.0, "sp": "A", "text": "he had"},
                {"s": 2.0, "e": 2.4, "sp": "B", "text": "job."},
                {"s": 2.4, "e": 4.0, "sp": "A", "text": "a whole year."}]
        words = [W("he", 1.4, 1.6), W("had", 1.6, 1.9), W("done", 2.05, 2.2), W("for", 2.2, 2.35),
                 W("a", 2.45, 2.5), W("whole", 2.5, 2.8), W("year.", 2.8, 3.2)]
        T.assign_speakers(words, cues)
        self.assertEqual({w["sp"] for w in words}, {"A"})

    def test_real_interjection_is_kept(self):
        cues = [{"s": 0.0, "e": 2.0, "sp": "A", "text": "so I went"},
                {"s": 2.0, "e": 2.4, "sp": "B", "text": "Wow"},
                {"s": 2.4, "e": 4.0, "sp": "A", "text": "and then"}]
        words = [W("so", 1.0, 1.2), W("I", 1.2, 1.3), W("went", 1.3, 1.8), W("Wow", 2.05, 2.3),
                 W("and", 2.5, 2.7), W("then", 2.7, 3.0)]
        T.assign_speakers(words, cues)
        self.assertEqual(words[3]["sp"], "B")


class TestCaptionFixes(unittest.TestCase):
    def test_auto_emphasis_does_not_repeat_a_word(self):
        words = []
        for k in range(4):
            words += ["the", "technology", "is", "done."]
        track = TC.build(TC.make_plan(words, gap=0.3), TC.load_preset("pop"))
        emph = [t["text"] for p in track.pages for t in p["tokens"] if t.get("emph")]
        self.assertLessEqual(emph.count("TECHNOLOGY"), 1, emph)

    def test_short_page_does_not_merge_across_a_sentence(self):
        # "People say," is boxed in (no gap to grow into), so min_page wants a merge; minimal
        # (up to 6 words) has room to merge it backwards across "service."
        words = [("technology", {"s": 0.5, "e": 0.8}), ("or", {"s": 0.8, "e": 0.95}),
                 ("service.", {"s": 0.95, "e": 1.3}), ("People", {"s": 1.32, "e": 1.42}),
                 ("say,", {"s": 1.42, "e": 1.5}), ("I", {"s": 1.52, "e": 1.6}),
                 ("can't", {"s": 1.6, "e": 1.8}), ("do", {"s": 1.8, "e": 1.9}),
                 ("that.", {"s": 1.9, "e": 2.4})]
        track = TC.build(TC.make_plan(words), TC.load_preset("minimal"))
        for p in track.pages:
            low = p["text"].lower()
            self.assertFalse("service" in low and "people" in low, TC.page_texts(track))

    def test_authored_hook_breaks_kept_near_full_size(self):
        # The author's breaks stand when the card sets near full size (and at least
        # hook_card.min_ratio x the caption size). A card that sets small is re-broken for
        # size: the frame-0 title must read bigger than the captions.
        plan = TC.make_plan(["sell", "it", "make", "it."],
                            hook={"lines": ["SELL IT,", "*MAKE* IT"], "dur": 1.5})
        self.assertEqual(TC.build(plan).hook["rendered_lines"], ["SELL IT,", "MAKE IT"])
        plan = TC.make_plan(["sell", "it", "then", "make", "it."],
                            hook={"lines": ["SELL IT,", "THEN *MAKE* IT"], "dur": 1.5})
        track = TC.build(plan)
        self.assertEqual(track.hook["rendered_lines"], ["SELL IT,", "THEN", "MAKE IT"])
        self.assertGreaterEqual(track.hook["size"], 1.3 * TC.CLEAN["caption"]["size"])


class TestQaFixes(unittest.TestCase):
    def test_audio_timestamp_hole_is_found(self):
        d = Path(tempfile.mkdtemp(prefix="hc-gap-"))
        try:
            ok, holed = d / "ok.mp4", d / "holed.mp4"
            base = [env.ffmpeg(), "-v", "error", "-y", "-f", "lavfi", "-i", "sine=f=440:d=3:r=48000"]
            subprocess.run(base + ["-c:a", "aac", str(ok)], check=True)
            # shift every packet after 1.5 s by 80 ms: the mp4 muxer stores the hole as one
            # over-long packet duration, exactly what loudnorm's EOF flush produced
            subprocess.run([env.ffmpeg(), "-v", "error", "-y", "-i", str(ok), "-c:a", "copy",
                            "-bsf:a", "setts=ts=TS+gte(TS\\,72000)*3840", str(holed)], check=True)
            a = {"time_base": "1/48000", "sample_rate": "48000"}
            self.assertEqual(qa._audio_pts_gaps(ok, a)[0], 0)
            n, worst = qa._audio_pts_gaps(holed, a)
            self.assertGreaterEqual(n, 1)
            self.assertGreater(worst, 0.05)
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_mouth_zone_uses_landmarks(self):
        # box 430-1268 (YuNet pads below the chin); eyes 761, nose 948, mouth 1055
        f = [235, 430, 637, 838, [[378, 761], [687, 761], [517, 948], [396, 1055], [675, 1056]]]
        self.assertFalse(qa.face_hits([200, 1183, 680, 134], [f]))   # caption on the beard: fine
        self.assertTrue(qa.face_hits([200, 1000, 680, 134], [f]))    # across the mouth: WARN
        self.assertTrue(qa.face_hits([200, 1183, 680, 134], [f[:4]]))  # no landmarks: contract rule


class TestCoverFixes(unittest.TestCase):
    def test_grid_always_shows_the_whole_set(self):
        from hookcutter import cover
        d = Path(tempfile.mkdtemp(prefix="hc-grid-"))
        try:
            proj = project.Project(d, create=True)
            spec = {"clips": {"c01": {"_eff": {"styles": ["clean", "pop"]}},
                              "c02": {"_eff": {"styles": ["minimal"]}}}}
            for cid, st, col in (("c01", "clean", (200, 0, 0)), ("c01", "pop", (0, 200, 0)),
                                 ("c02", "minimal", (0, 0, 200))):
                Image.new("RGB", (1080, 1440), col).save(proj.cover_path(cid, st, "grid"))
            lines = cover.refresh_grids(proj, spec)
            self.assertTrue(lines)
            g = Image.open(proj.sheet_path("covers-grid.jpg")).convert("RGB")
            tw = cover.GRID_TILE_W
            left = g.getpixel((tw // 2, 20))
            mid = g.getpixel((tw + 3 + tw // 2, 20))
            self.assertGreater(left[0], 150)          # c01 in its first style (clean, red)
            self.assertGreater(mid[2], 150)           # c02 in minimal (blue)
            self.assertTrue(Path(proj.sheet_path("covers-grid__pop.jpg")).is_file())
        finally:
            shutil.rmtree(d, ignore_errors=True)

    def test_headline_shrinks_before_the_head_leaves_the_tile(self):
        from hookcutter import cover
        photo = Image.new("RGB", (1080, 1920), (90, 90, 90))
        faces = [{"box": [240, 330, 600, 820], "score": 0.9, "landmarks": None}]
        cfg = dict(cover.DEFAULT_COVER, headline_font="Anton-Regular.ttf", size_range=[96, 168])
        _img, info = cover.compose(photo, faces, "HE SOLD IT FIRST AND MADE *WAY MORE MONEY*", cfg)
        self.assertGreaterEqual(info["face_box"][1], 240, info)
        self.assertLess(info["size"], 168)


if __name__ == "__main__":
    unittest.main()
