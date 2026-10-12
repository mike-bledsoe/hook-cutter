"""B3 planning tests: spec normalization, energy track, edge snapping, tighten, omit,
out-time map, intent and every plan lint (L01-L15). Synthetic energy tracks are built
from numpy arrays; the ffmpeg-backed tests use the shared lavfi fixtures."""
from __future__ import annotations

import contextlib
import io
import json
import math
import os
import sys
import tempfile
import time
import unittest
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "scripts"))

import numpy as np  # noqa: E402

from fixtures import fake_transcript, make_av_fixture, make_project, make_speech_fixture  # noqa: E402
from hookcutter import env  # noqa: E402
from hookcutter import plan as P  # noqa: E402
from hookcutter import spec as S  # noqa: E402
from hookcutter.audio import EnergyTrack, loudnorm_measure  # noqa: E402

FLOOR = -60.0
VOICE = -20.0


# ---- helpers --------------------------------------------------------------------

def energy(dur, spans, floor=FLOOR, level=VOICE, hop=0.01, t0=0.0):
    """EnergyTrack with `level` dB inside each (a, b[, level]) span and `floor` elsewhere."""
    n = int(round(dur / hop))
    db = np.full(n, floor, dtype=np.float64)
    for sp in spans:
        a, b = sp[0], sp[1]
        lv = sp[2] if len(sp) > 2 else level
        db[int(round((a - t0) / hop)):int(round((b - t0) / hop))] = lv
    return EnergyTrack(db, hop=hop, t0=t0)


def source(fps=(25, 1), dur=60.0, start=0.0, w=1920, h=1080):
    frames = int(math.floor((dur - start) * fps[0] / fps[1] + 1e-9))
    return {"fingerprint": "fp-test", "duration": dur,
            "video": {"fps": list(fps), "start": start, "width": w, "height": h,
                      "frames": frames, "hdr": "none"},
            "audio": {"start": 0.0, "index": 1}}


def mkspec(clips, defaults=None, tsha=None):
    raw = {"version": 1, "clips": clips}
    if defaults is not None:
        raw["defaults"] = defaults
    if tsha is not None:
        raw["transcript_sha"] = tsha
    return S.normalize_spec(raw)


def build(words, clips, spans=None, fps=(25, 1), defaults=None, dur=None, shots=None,
          cid="c01", tsha=None, start=0.0, size=(1920, 1080), flags=None, extra=()):
    tr = fake_transcript(words)
    if flags:
        tr["flags"] = flags
    end = dur or (max(w[2] for w in words) + 5.0)
    if spans is None:
        spans = [(w[1], w[2]) for w in words]
    E = energy(end, list(spans) + list(extra))
    src = source(fps, end, start, size[0], size[1])
    spec = mkspec(clips, defaults, tsha)
    return P.plan_clip(spec, cid, tr, src, E, shots), tr, E


def seq_words(n, start=1.0, wlen=0.3, gap=0.1, text=None, sentence_every=0):
    """n words of length wlen separated by gap; optional sentence-final period."""
    out, t = [], start
    for k in range(n):
        w = text[k] if text else f"word{k}"
        if sentence_every and (k + 1) % sentence_every == 0 and not w.endswith("."):
            w += "."
        out.append((w, round(t, 3), round(t + wlen, 3)))
        t += wlen + gap
    return out


def lints_of(plan, lid, level=None):
    return [x for x in plan["lints"] if x["id"] == lid and (level is None or x["level"] == level)]


def on_grid(plan, t):
    num, den = plan["fps"]
    x = (t - plan["grid_offset"]) * num / den
    return abs(x - round(x)) < 1e-4


# ---- spec -----------------------------------------------------------------------

class TestSpec(unittest.TestCase):
    def test_parse_time(self):
        self.assertEqual(S.parse_time(681.5), 681.5)
        self.assertEqual(S.parse_time("681.5"), 681.5)
        self.assertEqual(S.parse_time("11:21"), 681.0)
        self.assertAlmostEqual(S.parse_time("11:21.40"), 681.4)
        self.assertAlmostEqual(S.parse_time("1:02:03.4"), 3723.4)
        self.assertEqual(S.parse_time("75:10"), 4510.0)
        for bad in ("", "1:75", "1:02:75", "abc", "-3", None, True, float("nan"), "1::2"):
            with self.assertRaises(env.HCError, msg=repr(bad)):
                S.parse_time(bad)

    def test_legacy_hook_text_and_defaults(self):
        sp = mkspec({"a": {"pieces": [{"words": [0, 3]}], "hook_card": {"text": "LINE ONE\\N*TWO*"}},
                     "b": {"pieces": [[0, 3]], "hook_card": "ONE\nTWO\\nTHREE"}})
        self.assertEqual(sp["clips"]["a"]["hook_card"], {"lines": ["LINE ONE", "*TWO*"], "dur": "auto"})
        self.assertEqual(sp["clips"]["b"]["hook_card"]["lines"], ["ONE", "TWO", "THREE"])
        self.assertEqual(sp["clips"]["b"]["pieces"][0]["words"], [0, 3])
        eff = sp["clips"]["a"]["_eff"]
        self.assertEqual(eff["tighten"]["gap"], 0.6)
        self.assertEqual(eff["formats"], ["vertical"])
        self.assertEqual(eff["length"]["vertical"], [15.0, 30.0, 60.0, 90.0])

    def test_unknown_keys_warn_and_comments_ignored(self):
        sp = mkspec({"a": {"pieces": [{"words": [0, 3], "bogus": 1}], "colour": "red",
                           "_note": "agent comment"}})
        lints = S.validate(sp)
        msgs = [x["msg"] for x in lints if x["id"] == "L01" and x["level"] == "WARN"]
        self.assertTrue(any("'bogus'" in m for m in msgs))
        self.assertTrue(any("'colour'" in m for m in msgs))
        self.assertFalse(any("_note" in m for m in msgs))
        self.assertFalse([x for x in lints if x["level"] == "FAIL"])

    def test_schema_failures(self):
        sp = mkspec({"Bad_ID": {"pieces": [{"words": [5, 2]}, {"words": [1, 2], "time_src": [0, 1]},
                                           {"time_src": ["0:10", "0:05"]}, {"words": [0, 1],
                                                                             "gain_db": 40}]},
                     "cap": {"kind": "caption_only", "pieces": [{"words": [0, 1]}]},
                     "nopieces": {"title": "x"}})
        lints = [x for x in S.validate(sp) if x["level"] == "FAIL"]
        text = " | ".join(x["msg"] for x in lints)
        self.assertIn("clip id 'Bad_ID' is invalid", text)
        self.assertIn("a <= b", text)
        self.assertIn("exactly one of", text)
        self.assertIn("end must be after start", text)
        self.assertIn("gain_db", text)
        self.assertIn("source_file", text)
        self.assertIn("must not have \"pieces\"", text)
        self.assertIn("no pieces", text)
        self.assertFalse(sp["clips"]["cap"]["_eff"]["tighten"]["enabled"])

    def test_framing_sorted_and_defaults(self):
        sp = mkspec({"a": {"pieces": [{"words": [0, 1]}], "framing": [
            {"at_src": "1:00", "mode": "letterbox"}, {"at_src": 10, "mode": "fixed"},
            {"at_src": 5, "mode": "track", "x": 0.4, "crop": 0.6}]}})
        fr = sp["clips"]["a"]["framing"]
        self.assertEqual([m["at_src"] for m in fr], [5.0, 10.0, 60.0])
        self.assertEqual(fr[1], {"at_src": 10.0, "mode": "fixed", "x": 0.5, "y": 0.5, "crop": 1.0})

    def test_clip_hash(self):
        a = mkspec({"a": {"pieces": [{"words": [0, 3]}]}})
        b = mkspec({"a": {"pieces": [{"words": [0, 3]}]}})
        c = mkspec({"a": {"pieces": [{"words": [0, 3]}]}}, defaults={"tighten": {"gap": 0.9}})
        d = mkspec({"a": {"pieces": [{"words": [0, 4]}]}})
        self.assertEqual(S.clip_hash(a, "a"), S.clip_hash(b, "a"))
        self.assertNotEqual(S.clip_hash(a, "a"), S.clip_hash(c, "a"))
        self.assertNotEqual(S.clip_hash(a, "a"), S.clip_hash(d, "a"))

    def test_legacy_spans_and_omit_ranges(self):
        sp = mkspec({"a": {"spans": [[0, 3], [5, 9]], "omit": [[6, 7], 8], "verbatim": [1, 2]}})
        c = sp["clips"]["a"]
        self.assertEqual([p["words"] for p in c["pieces"]], [[0, 3], [5, 9]])
        self.assertEqual(c["omit"], [6, 7, 8])
        self.assertEqual(c["verbatim"], [[1, 2]])

    def test_load_spec_file(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(env.HCError):
                S.load_spec(d)
            Path(d, "clips.json").write_text("{not json")
            with self.assertRaises(env.HCError):
                S.load_spec(d)
            Path(d, "clips.json").write_text(json.dumps({"version": 1, "clips": {
                "a": {"pieces": [{"words": [0, 1]}]}}}))
            self.assertIn("a", S.load_spec(d)["clips"])


# ---- energy track ----------------------------------------------------------------

class TestEnergySynthetic(unittest.TestCase):
    def setUp(self):
        self.E = energy(20, [(2.0, 3.0), (3.0, 3.6, -45.0), (8.0, 9.0), (12.0, 14.0, -25.0)])

    def test_db_floor(self):
        E = self.E
        self.assertAlmostEqual(E.db_at(2.5), VOICE, places=6)
        self.assertAlmostEqual(E.db_at(5.0), FLOOR, places=6)
        self.assertAlmostEqual(E.floor(5.0), FLOOR, places=6)
        self.assertEqual(E.db_at(-5.0), -120.0)

    def test_onset_and_voiced_end(self):
        E = self.E
        self.assertAlmostEqual(E.onset(2.2, 1.7), 2.0)          # whisper late: run start
        self.assertAlmostEqual(E.onset(1.95, 1.7), 2.0)         # whisper early: next run
        self.assertIsNone(E.onset(2.5, 2.1))                    # continuous: no onset
        self.assertIsNone(E.onset(5.0, 4.5))                    # nothing voiced
        self.assertAlmostEqual(E.voiced_end(2.9, 5.0), 3.6)     # quiet tail still voiced
        self.assertEqual(E.voiced_end(5.0, 6.0), 5.0)

    def test_true_end_and_quiet_gap(self):
        E = self.E
        self.assertAlmostEqual(E.true_end({"s": 8.0, "e": 11.0}), 9.0)   # stretched end
        self.assertEqual(E.true_end({"s": 8.0, "e": 8.9}), 8.9)
        self.assertTrue(E.quiet_gap(9.0, 12.0))
        self.assertFalse(E.quiet_gap(11.5, 14.5))                  # laughter is not quiet
        self.assertTrue(E.quiet_gap(11.5, 14.5, exclude=[(12.0, 14.0)]))


class TestEnergyMedia(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.dir = Path(cls.tmp.name)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_decode_cache_and_onsets(self):
        words = [("Hello", 1.0, 1.4), ("world.", 1.6, 2.0), ("Again", 3.0, 3.5)]
        path = make_speech_fixture(self.dir / "sp", words, dur=5)
        cache = self.dir / "cache"
        E = EnergyTrack.from_media(path, cache_dir=cache, key="k1")
        self.assertTrue((cache / "energy-k1.npy").exists())
        self.assertGreater(E.db_at(1.2), E.floor(1.2) + 30)
        self.assertLess(E.db_at(2.5), E.floor(2.5) + 6)
        for _w, s, _e in words:
            on = E.onset(s + 0.05, s - 0.3)
            self.assertIsNotNone(on)
            self.assertLessEqual(abs(on - s), 0.02)
        E2 = EnergyTrack.from_media(path, cache_dir=cache, key="k1")
        self.assertTrue(np.array_equal(E.db, E2.db))

    def test_audio_start_offset(self):
        av = make_av_fixture(self.dir / "av", fps="25", dur=8, flashes=[2.0, 5.0], audio_delay=0.2)
        P_ = make_project(self.dir / "avp", av)
        from hookcutter.project import Project
        proj = Project(P_)
        self.assertGreater(proj.source()["audio"]["start"], 0.1)
        E = EnergyTrack.for_project(proj)
        for t in (2.0, 5.0):
            lo, hi = E.index(t - 0.5), E.index(t + 0.5)
            peak = E.time(lo + int(np.argmax(E.db[lo:hi])))
            self.assertLessEqual(abs(peak - t), 0.02, f"click at {t} found at {peak}")

    def test_loudnorm_measure(self):
        words = [("a", 0.5, 2.5), ("b", 3.0, 4.5)]
        path = make_speech_fixture(self.dir / "ln", words, dur=5)
        m = loudnorm_measure(["-i", str(path)])
        self.assertIsInstance(m["input_i"], float)
        self.assertTrue(-40 < m["input_i"] < 0)
        m2 = loudnorm_measure(["-i", str(path), "-i", str(path)],
                              "[0:a][1:a]concat=n=2:v=0:a=1[cat]")
        self.assertAlmostEqual(m2["input_i"], m["input_i"], delta=0.6)


# ---- edges -----------------------------------------------------------------------

class TestEdges(unittest.TestCase):
    def _gap_case(self, fps, start):
        words = seq_words(9, start=1.0, wlen=0.35, gap=0.32, sentence_every=3)
        plan, tr, E = build(words, {"c01": {"pieces": [{"words": [0, 2]}, {"words": [6, 8]}],
                                            "tighten": False}}, fps=fps, start=start)
        W = tr["words"]
        for p in plan["pieces"]:
            a, b = p["words"]
            t_in, t_out = p["src"]
            self.assertTrue(on_grid(plan, t_in) and on_grid(plan, t_out))
            self.assertLessEqual(t_in, W[a]["s"] - 0.02 + 1e-9)
            if a > 0:
                self.assertGreaterEqual(t_in, W[a - 1]["e"])
            self.assertGreaterEqual(t_out, W[b]["e"] + 0.02 - 1e-9)
            if b + 1 < len(W):
                self.assertLessEqual(t_out, W[b + 1]["s"])
            self.assertFalse(p["edge_in"]["tight"] or p["edge_out"]["tight"])
        num, den = plan["fps"]
        for s in plan["segments"]:
            k0, k1 = s["grid"]
            self.assertAlmostEqual(s["src"][0], start + k0 * den / num, places=5)
            self.assertAlmostEqual(s["src"][1], start + k1 * den / num, places=5)
            self.assertEqual(s["frames"], k1 - k0)
        self.assertFalse(lints_of(plan, "L10"))

    def test_edges_in_gaps_on_grid_25(self):
        self._gap_case((25, 1), 0.0)

    def test_edges_in_gaps_on_grid_2997_with_offset(self):
        self._gap_case((30000, 1001), 0.033367)

    def test_onset_guard_when_whisper_start_is_late(self):
        words = [("Before.", 8.0, 8.5), ("Real", 10.24, 10.6), ("words.", 10.7, 11.0)]
        spans = [(8.0, 8.5), (10.0, 10.6), (10.7, 11.0)]          # voice really starts at 10.00
        plan, _tr, _E = build(words, {"c01": {"pieces": [{"words": [1, 2]}]}}, spans=spans)
        e = plan["pieces"][0]["edge_in"]
        self.assertEqual(e["guard"], "onset")
        self.assertLessEqual(e["src"], 10.0 - 0.02 + 1e-9)
        self.assertFalse(e["tight"])

    def test_offset_guard_when_whisper_end_is_early(self):
        words = [("Start.", 1.0, 1.5), ("Long", 4.0, 4.6), ("tail", 4.7, 5.0), ("Next.", 6.5, 7.0)]
        spans = [(1.0, 1.5), (4.0, 4.6), (4.7, 5.0), (5.0, 5.4, -45.0), (6.5, 7.0)]
        plan, _tr, _E = build(words, {"c01": {"pieces": [{"words": [0, 0]}, {"words": [1, 2]},
                                                         {"words": [3, 3]}],
                                              "tighten": False}}, spans=spans)
        e = plan["pieces"][1]["edge_out"]
        self.assertEqual(e["guard"], "offset")
        self.assertGreaterEqual(e["src"], 5.3)
        self.assertLessEqual(e["src"], 6.5 - 0.03)

    def test_tight_on_continuous_speech(self):
        words = [(f"w{k}", 10.0 + 0.3 * k, 10.3 + 0.3 * k) for k in range(12)]
        plan, _tr, _E = build(words, {"c01": {"pieces": [{"words": [4, 7]}]}},
                              spans=[(9.9, 13.7)], dur=20)
        p = plan["pieces"][0]
        self.assertTrue(p["edge_in"]["tight"])
        self.assertTrue(p["edge_out"]["tight"])
        msgs = [x["msg"] for x in lints_of(plan, "L10", "WARN")]
        self.assertTrue(any("starts inside voiced audio" in m for m in msgs))
        self.assertTrue(any("ends inside voiced audio" in m for m in msgs))

    def test_last_piece_tail(self):
        words = [("One.", 1.0, 1.5), ("Two.", 3.0, 3.5), ("Three.", 5.0, 5.5), ("Far.", 9.0, 9.5)]
        plan, _tr, _E = build(words, {"c01": {"pieces": [{"words": [0, 0]}, {"words": [2, 2]}],
                                              "tighten": False}})
        first, last = plan["pieces"]
        self.assertAlmostEqual(first["src"][1], 1.5 + 0.10, delta=0.021)
        self.assertAlmostEqual(last["src"][1], 5.5 + 0.40, delta=0.021)
        self.assertEqual(plan["segments"][-1]["fade_out"], P.FADE_LAST)

    def test_hold_after_extends_over_reaction(self):
        words = [("That", 1.0, 1.4), ("was", 1.5, 1.8), ("funny.", 1.9, 2.4), ("Ha", 2.6, 2.9),
                 ("Okay.", 6.0, 6.4)]
        laugh = (2.6, 4.0, -25.0)
        spans = [(1.0, 1.4), (1.5, 1.8), (1.9, 2.4), laugh, (6.0, 6.4)]
        plain, _tr, _E = build(words, {"c01": {"pieces": [{"words": [0, 2]}]}}, spans=spans)
        self.assertLessEqual(plain["pieces"][0]["src"][1], 2.6 - 0.03 + 1e-9)
        held, _tr, _E = build(words, {"c01": {"pieces": [{"words": [0, 2], "hold_after": 0.8}]}},
                              spans=spans)
        p = held["pieces"][0]
        self.assertGreaterEqual(p["src"][1], 2.4 + 0.4 + 0.8 - 0.05)
        self.assertEqual(p["extra_words"], [3])
        # L16 (was INFO I01): the held words are heard and captioned, so it is a WARN now
        l16 = lints_of(held, "L16", "WARN")
        self.assertTrue(l16)
        self.assertIn("pulls in words 3-3 ('Ha')", l16[0]["msg"])
        self.assertIn(3, held["intent"])
        self.assertFalse([r for r in held["removals"] if r["src"][1] > 2.4])

    def test_adjacent_pieces_share_one_cut(self):
        clip = {"pieces": [{"words": [0, 3], "role": "setup"}, {"words": [4, 7], "role": "payoff"}],
                "tighten": False}
        close, _tr, _E = build(seq_words(8, gap=0.15, sentence_every=4), {"c01": clip})
        j = close["joins"][0]
        self.assertEqual(j["kind"], "piece")
        self.assertTrue(j["seamless"])
        self.assertEqual(j["before_src"], j["after_src"])
        self.assertEqual(close["segments"][0]["fade_out"], 0.0)
        self.assertEqual(close["segments"][1]["fade_in"], 0.0)
        far, _tr, _E = build(seq_words(8, gap=1.5, sentence_every=4), {"c01": clip})
        j = far["joins"][0]
        self.assertFalse(j["seamless"])
        self.assertGreater(j["after_src"] - j["before_src"], 1.0)   # the long pause is cut


# ---- tighten ---------------------------------------------------------------------

class TestTighten(unittest.TestCase):
    WORDS = [("We", 1.0, 1.3), ("waited.", 1.4, 2.0), ("Then", 3.5, 3.8), ("we", 3.9, 4.1),
             ("laughed", 4.2, 4.6), ("hard.", 4.7, 6.0), ("Really", 7.5, 7.9), ("hard.", 8.0, 8.5)]

    def test_quiet_gap_removed_loud_gap_kept(self):
        spans = [(w[1], w[2]) for w in self.WORDS] + [(6.0, 7.5, -25.0)]  # laughter after word 5
        plan, _tr, _E = build(self.WORDS, {"c01": {"pieces": [{"words": [0, 7]}]}}, spans=spans)
        gaps = [r for r in plan["removals"] if r["kind"] == "gap"]
        self.assertEqual(len(gaps), 1)
        self.assertEqual(gaps[0]["src"], [2.12, 3.32])                  # keep 0.4/0.6 of 0.3 s
        self.assertFalse([r for r in plan["removals"] if r["src"][0] < 7.5 and r["src"][1] > 6.0])
        self.assertEqual(plan["joins"][0]["kind"], "tighten")
        self.assertAlmostEqual(plan["tighten"]["removed_s"], 1.2, places=6)
        for r in plan["removals"]:
            self.assertTrue(on_grid(plan, r["src"][0]) and on_grid(plan, r["src"][1]))

    def test_verbatim_blocks_tighten(self):
        plan, _tr, _E = build(self.WORDS, {"c01": {"pieces": [{"words": [0, 7]}],
                                                   "verbatim": [[1, 2]]}})
        self.assertFalse([r for r in plan["removals"] if r["src"][1] <= 3.5])
        plan2, _tr, _E = build(self.WORDS, {"c01": {"pieces": [{"words": [0, 7], "tighten": False}]}})
        self.assertFalse(plan2["removals"])

    def test_min_cut(self):
        words = [("A", 1.0, 1.3), ("b.", 1.4, 2.0), ("C", 2.75, 3.0), ("d.", 3.1, 3.5)]
        clip = {"pieces": [{"words": [0, 3]}]}
        kept, _tr, _E = build(words, {"c01": dict(clip, tighten={"min_cut": 0.4})})
        self.assertEqual(len(kept["removals"]), 1)
        self.assertAlmostEqual(kept["removals"][0]["src"][1] - kept["removals"][0]["src"][0], 0.44,
                               places=6)
        dropped, _tr, _E = build(words, {"c01": dict(clip, tighten={"min_cut": 0.5})})
        self.assertFalse(dropped["removals"])

    def test_keep_interval_at_least_two_frames(self):
        words = [("A", 1.0, 1.3), ("b", 2.5, 2.8)]
        tr = fake_transcript(words)
        sp = mkspec({"c01": {"pieces": [{"words": [0, 1]}]}})
        G = P.Grid(25, 1)
        ctx = P._Ctx(sp["clips"]["c01"], tr["words"], G, energy(5, [(1.0, 1.3), (2.5, 2.8)]), 125)
        # gap removal raw = [1.42, 2.32]; an out edge 1 frame later leaves a 1-frame keep
        run = {"idxs": [0, 1], "k_in": 20, "k_out": G.floor(2.32) + 1, "first": False, "last": False}
        self.assertEqual(P.tighten_run(ctx, run, {"hold_after": 0}), [])
        run["k_out"] = G.floor(2.32) + 2
        self.assertEqual(len(P.tighten_run(ctx, run, {"hold_after": 0})), 1)

    def test_filler_removed_flagged_and_merged(self):
        words = [("I", 1.0, 1.2), ("was", 1.3, 1.6), ("um", 1.9, 2.2), ("there.", 2.5, 3.0)]
        plan, _tr, _E = build(words, {"c01": {"pieces": [{"words": [0, 3]}]}})
        rm = plan["removals"]
        self.assertEqual(len(rm), 1)
        self.assertEqual(rm[0]["kind"], "filler")
        um = next(w for w in plan["words"] if w["i"] == 2)
        self.assertTrue(um["filler"] and um["removed"])
        self.assertIsNone(um["s_out"])
        self.assertNotIn(2, plan["intent"])
        keep, _tr, _E = build(words, {"c01": {"pieces": [{"words": [0, 3]}],
                                              "tighten": {"fillers": False}}})
        self.assertFalse(keep["removals"])
        self.assertNotIn(2, keep["intent"])


    def test_glued_filler_is_kept_whole(self):
        words = [("I", 1.0, 1.2), ("um", 1.22, 1.5), ("uh", 1.52, 1.7), ("think", 1.72, 2.0),
                 ("so.", 2.1, 2.4), ("Uh", 2.8, 3.0), ("yes.", 3.4, 3.8)]
        plan, _tr, _E = build(words, {"c01": {"pieces": [{"words": [0, 6]}]}})
        removed = {w["i"] for w in plan["words"] if w["removed"]}
        self.assertEqual(removed, {5})                 # only the filler with quiet air around it
        for j in plan["joins"]:
            self.assertFalse(j["tight"])
        for r in plan["removals"]:
            self.assertGreaterEqual(r["src"][0], 2.4)


# ---- omit ------------------------------------------------------------------------

class TestOmit(unittest.TestCase):
    WORDS = [("You", 1.0, 1.2), ("should", 1.3, 1.6), ("not", 1.75, 2.0), ("ever", 2.15, 2.4),
             ("quit", 2.5, 2.8), ("this.", 2.9, 3.3), ("Next.", 6.0, 6.5)]

    def test_omit_splits_and_snaps(self):
        plan, tr, _E = build(self.WORDS, {"c01": {"pieces": [{"words": [0, 5]}], "omit": [2]}})
        W = tr["words"]
        om = [r for r in plan["removals"] if r["kind"] == "omit"]
        self.assertEqual(len(om), 1)
        a, b = om[0]["src"]
        self.assertLessEqual(a, W[2]["s"] - 0.03 + 1e-9)
        self.assertGreaterEqual(b, W[2]["e"])
        self.assertLessEqual(b, W[3]["s"] - 0.02 + 1e-9)
        self.assertTrue(on_grid(plan, a) and on_grid(plan, b))
        self.assertEqual([j["kind"] for j in plan["joins"]], ["omit"])
        w2 = next(w for w in plan["words"] if w["i"] == 2)
        self.assertTrue(w2["omitted"])
        self.assertIsNone(w2["s_out"])
        self.assertNotIn(2, plan["intent"])
        self.assertTrue(any("'not'" in x["msg"] for x in lints_of(plan, "L06", "WARN")))

    def test_omit_first_word_moves_edge(self):
        plan, tr, _E = build(self.WORDS, {"c01": {"pieces": [{"words": [0, 5]}], "omit": [0]}})
        self.assertFalse([r for r in plan["removals"] if r["kind"] == "omit"])
        self.assertGreaterEqual(plan["pieces"][0]["src"][0], tr["words"][0]["e"] - 0.03)
        self.assertLessEqual(plan["pieces"][0]["src"][0], tr["words"][1]["s"] - 0.02)


# ---- maps / intent ---------------------------------------------------------------

class TestMaps(unittest.TestCase):
    def test_exact_frame_accumulation_30_segments(self):
        words = seq_words(30, start=1.0, wlen=0.37, gap=1.03, sentence_every=1)
        pieces = [{"words": [k, k]} for k in range(30)]
        plan, tr, _E = build(words, {"c01": {"pieces": pieces}}, fps=(30000, 1001), start=0.0)
        segs = plan["segments"]
        self.assertEqual(len(segs), 30)
        fd = Fraction(1001, 30000)
        cum = 0
        for s in segs:
            self.assertAlmostEqual(s["out"][0], float(cum * fd), places=6)
            self.assertEqual(s["samples"], round(Fraction(s["frames"] * 48000 * 1001, 30000)))
            cum += s["frames"]
        self.assertEqual(plan["frames_out"], cum)
        self.assertAlmostEqual(plan["duration_out"], float(cum * fd), places=6)
        last_out = -1.0
        for s in segs:
            for f in (0.0, 0.25, 0.5, 0.75, 0.999):
                t = s["src"][0] + f * (s["src"][1] - s["src"][0])
                o = P.out_time(plan, t)
                self.assertGreater(o, last_out)
                last_out = o
                k, back = P.src_time(plan, o)
                self.assertEqual(k, s["k"])
                self.assertAlmostEqual(back, t, places=6)
        # caption word times: exact frame-accumulated offset, no drift
        for w in plan["words"]:
            if w.get("edge"):
                continue
            seg = segs[w["seg"]]
            exp = float(Fraction(seg["grid"][0] * 0 + sum(x["frames"] for x in segs[:seg["k"]]))
                        * fd) + (w["s_src"] - seg["src"][0])
            self.assertAlmostEqual(w["s_out"], exp, places=3)

    def test_src_time_clamps_and_out_time_none(self):
        words = [("A.", 1.0, 1.5), ("B.", 4.0, 4.5)]
        plan, _tr, _E = build(words, {"c01": {"pieces": [{"words": [0, 0]}, {"words": [1, 1]}]}})
        self.assertEqual(P.src_time(plan, -3)[1], plan["segments"][0]["src"][0])
        self.assertEqual(P.src_time(plan, 1e9), (1, plan["segments"][-1]["src"][1]))
        self.assertIsNone(P.out_time(plan, 3.0))

    def test_intent_never_from_tighten(self):
        words = [("So", 1.0, 1.2), ("um", 1.5, 1.8), ("this", 2.6, 2.9), ("is", 3.0, 3.2),
                 ("Cloud.", 3.3, 3.8), ("Uh", 4.0, 4.2), ("really", 5.5, 5.9), ("not", 6.0, 6.2),
                 ("good.", 6.3, 6.8)]
        clip = {"pieces": [{"words": [0, 8]}], "omit": [7], "fix": {"4": "Claude", "3": ""}}
        on, _tr, _E = build(words, {"c01": clip})
        off, _tr, _E = build(words, {"c01": dict(clip, tighten=False)})
        self.assertTrue(on["removals"])
        self.assertEqual(on["intent"], off["intent"])
        self.assertEqual(on["intent"], [0, 2, 3, 4, 6, 8])
        w4 = next(w for w in on["words"] if w["i"] == 4)
        self.assertEqual((w4["raw"], w4["text"]), ("Cloud.", "Claude"))
        w3 = next(w for w in on["words"] if w["i"] == 3)
        self.assertTrue(w3["hidden"])


# ---- lints -----------------------------------------------------------------------

class TestLints(unittest.TestCase):
    def test_l01_bounds(self):
        words = seq_words(10, sentence_every=5)
        plan, _tr, _E = build(words, {"c01": {"pieces": [{"words": [0, 4]}, {"words": [8, 40]}],
                                              "omit": [7], "fix": {"99": "x"}}})
        msgs = [x["msg"] for x in lints_of(plan, "L01", "FAIL")]
        self.assertTrue(any("out of range" in m and "[8, 40]" in m for m in msgs))
        self.assertTrue(any("omit index 7" in m for m in msgs))
        self.assertTrue(any("fix index 99" in m for m in msgs))
        self.assertEqual(len(plan["pieces"]), 1)

    def test_l02_expect(self):
        words = seq_words(10, text="I believe that people who look stupid learn it fast".split())
        good, _tr, _E = build(words, {"c01": {"pieces": [{"words": [0, 9],
                                                          "expect": ["I believe that", "learn it fast"]}]}})
        self.assertFalse(lints_of(good, "L02"))
        bad, _tr, _E = build(words, {"c01": {"pieces": [{"words": [1, 9],
                                                         "expect": ["I believe that", ""]}]}})
        self.assertTrue(lints_of(bad, "L02", "FAIL"))

    def test_l03_transcript_sha(self):
        words = seq_words(10, sentence_every=5)
        plan, _tr, _E = build(words, {"c01": {"pieces": [{"words": [0, 4]}]}}, tsha="deadbeef")
        self.assertTrue(lints_of(plan, "L03", "FAIL"))
        tr = fake_transcript(words)
        ok, _tr, _E = build(words, {"c01": {"pieces": [{"words": [0, 4]}]}}, tsha=tr["sha256"])
        self.assertFalse(lints_of(ok, "L03"))

    def test_l04_opener(self):
        words = [("Done.", 0.5, 1.0), ("Um", 2.0, 2.3), ("and", 2.6, 2.8), ("it", 2.9, 3.0),
                 ("works.", 3.1, 3.6)]
        plan, _tr, _E = build(words, {"c01": {"pieces": [{"words": [1, 4]}]}})
        self.assertTrue(lints_of(plan, "L04", "FAIL"))
        plan2, _tr, _E = build(words, {"c01": {"pieces": [{"words": [2, 4]}]}})
        warn = lints_of(plan2, "L04", "WARN")
        self.assertTrue(warn and "continuation word 'and'" in warn[0]["msg"])
        self.assertFalse(lints_of(plan2, "L04", "FAIL"))
        clean = [("Done.", 0.5, 1.0), ("People", 2.0, 2.4), ("learn.", 2.5, 3.0)]
        plan3, _tr, _E = build(clean, {"c01": {"pieces": [{"words": [1, 2]}]}})
        self.assertFalse(lints_of(plan3, "L04"))

    def test_l05_closer(self):
        words = [("We", 1.0, 1.2), ("tried", 1.3, 1.6), ("and", 1.65, 1.8), ("failed.", 1.85, 2.3)]
        plan, _tr, _E = build(words, {"c01": {"pieces": [{"words": [0, 1]}]}})
        self.assertTrue(lints_of(plan, "L05", "WARN"))
        plan2, _tr, _E = build(words, {"c01": {"pieces": [{"words": [0, 3]}]}})
        self.assertFalse(lints_of(plan2, "L05"))

    def test_l06_piece_gap_drops_negation(self):
        words = [("This", 1.0, 1.2), ("is", 1.3, 1.5), ("not", 1.6, 1.8), ("easy.", 1.9, 2.3)]
        plan, _tr, _E = build(words, {"c01": {"pieces": [{"words": [0, 1]}, {"words": [3, 3]}]}})
        self.assertTrue(any("'not'" in x["msg"] for x in lints_of(plan, "L06", "WARN")))

    def test_l07_l09_length_and_cut_rate(self):
        words = seq_words(12, wlen=0.3, gap=0.7, sentence_every=1)
        plan, _tr, _E = build(words, {"c01": {"pieces": [{"words": [0, 11]}],
                                              "tighten": {"gap": 0.3, "keep": 0.1}}})
        self.assertTrue(lints_of(plan, "L07", "WARN"))
        self.assertGreater(plan["cut_rate"]["per_sec"], 0.5)
        self.assertTrue(lints_of(plan, "L09", "WARN"))

    def test_l08_payoff_timing(self):
        words = seq_words(40, wlen=0.3, gap=0.1, sentence_every=8,
                          text=["filler"] * 30 + ["afraid", "of", "looking", "stupid"] + ["end"] * 6)
        never, _tr, _E = build(words, {"c01": {"pieces": [{"words": [0, 39]}],
                                               "hook_card": {"lines": ["*NEVER SAID*"]}}})
        self.assertTrue(any("never spoken" in x["msg"] for x in lints_of(never, "L08")))
        late, _tr, _E = build(words, {"c01": {"pieces": [{"words": [0, 39]}],
                                              "hook_card": {"lines": ["AFRAID OF", "*LOOKING STUPID*"]}}})
        x = lints_of(late, "L08", "WARN")
        self.assertTrue(x and x[0]["at_out"] > 8.0)
        early, _tr, _E = build(words, {"c01": {"pieces": [{"words": [28, 39]}],
                                               "hook_card": {"lines": ["*LOOKING STUPID*"]}}})
        self.assertFalse(lints_of(early, "L08"))

    def test_l11_shots_need_marks(self):
        words = seq_words(20, start=10.0, wlen=0.3, gap=0.1, sentence_every=5)
        shots = {"shots": [{"start_src": 0.0, "end_src": 12.0}, {"start_src": 12.0, "end_src": 30.0}]}
        clip = {"pieces": [{"words": [0, 19]}], "tighten": False}
        plan, _tr, _E = build(words, {"c01": clip}, shots=shots)
        self.assertTrue(lints_of(plan, "L11", "WARN"))
        marked, _tr, _E = build(words, {"c01": dict(clip, framing=[
            {"at_src": 12.0, "mode": "track", "x": 0.6}])}, shots=shots)
        self.assertFalse(lints_of(marked, "L11"))

    def test_l12_upscale(self):
        words = seq_words(10, sentence_every=5)
        plan, _tr, _E = build(words, {"c01": {"pieces": [{"words": [0, 4]}]}}, size=(640, 360))
        x = lints_of(plan, "L12", "WARN")
        self.assertTrue(x and "5.3x" in x[0]["msg"])
        hd, _tr, _E = build(words, {"c01": {"pieces": [{"words": [0, 4]}]}}, size=(3840, 2160))
        self.assertFalse(lints_of(hd, "L12"))

    def test_l13_facts(self):
        words = seq_words(6, text="We made seventy thousand dollars there".split(), sentence_every=6)
        ok, _tr, _E = build(words, {"c01": {"pieces": [{"words": [0, 5]}],
                                            "hook_card": {"lines": ["WE MADE *$70K*"]}}})
        self.assertFalse(lints_of(ok, "L13"))
        bad, _tr, _E = build(words, {"c01": {"pieces": [{"words": [0, 5]}],
                                             "hook_card": {"lines": ["WE MADE *$90K*"]},
                                             "cover": {"headline": "How we hired Alice"}}})
        msgs = [x["msg"] for x in lints_of(bad, "L13", "WARN")]
        # one extractor with QA G11 (claims.py): the surface's own spelling is quoted
        self.assertTrue(any("number '$90K'" in m for m in msgs), msgs)
        self.assertTrue(any("'Alice'" in m for m in msgs))

    def test_l14_overlap(self):
        words = seq_words(80, wlen=0.3, gap=0.1, sentence_every=10)
        clips = {"c01": {"pieces": [{"words": [0, 59]}]}, "c02": {"pieces": [{"words": [10, 79]}]},
                 "c03": {"pieces": [{"words": [70, 79]}]}}
        plan, _tr, _E = build(words, clips)
        x = lints_of(plan, "L14", "WARN")
        self.assertEqual(len(x), 1)
        self.assertIn("'c02'", x[0]["msg"])

    def test_l15_flags(self):
        words = seq_words(20, sentence_every=5)
        plan, _tr, _E = build(words, {"c01": {"pieces": [{"words": [0, 9]}]}},
                              flags=[{"a": 5, "b": 7, "kind": "repetition_loop"},
                                     {"a": 15, "b": 16, "kind": "repetition_loop"}])
        self.assertEqual(len(lints_of(plan, "L15", "WARN")), 1)


# ---- project integration (real files) --------------------------------------------

class TestProject(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.dir = Path(cls.tmp.name)
        words = []
        t = 1.0
        text = ("People always ask me how to learn fast. The honest answer is um you have to "
                "look stupid first. And nobody wants that.").split()
        for k, w in enumerate(text):
            words.append((w, round(t, 3), round(t + 0.3, 3)))
            t += 0.42 + (1.2 if w.endswith(".") else 0.0)
        cls.words = words
        cls.media = make_speech_fixture(cls.dir / "media", words, fps="30000/1001")
        cls.P = make_project(cls.dir / "proj", cls.media, words)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def _write_clips(self, clips):
        env.write_json(self.P / "clips.json", {"version": 1, "clips": clips})

    def test_build_load_and_cmd(self):
        self._write_clips({"c01": {"title": "Learn fast", "pieces": [
            {"words": [8, 18], "role": "cold_open"}, {"words": [0, 7], "role": "setup"}]}})
        args = SimpleNamespace(project=str(self.P), clips=[], quiet=False)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(P.cmd_plan(args), 0)
        self.assertIn("lints:", out.getvalue())
        path = self.P / "plans" / "c01.plan.json"
        plan = env.read_json(path)
        self.assertEqual(plan["fps"], [30000, 1001])
        for s in plan["segments"]:
            self.assertTrue(on_grid(plan, s["src"][0]) and on_grid(plan, s["src"][1]))
        for p in plan["pieces"]:
            self.assertFalse(p["edge_in"]["tight"] or p["edge_out"]["tight"])
        um = next(w for w in plan["words"] if w["raw"] == "um")
        self.assertTrue(um["removed"])
        mtime = path.stat().st_mtime_ns
        time.sleep(0.01)
        again = P.load_plan(str(self.P), "c01")
        self.assertEqual(path.stat().st_mtime_ns, mtime)
        self.assertEqual(again["segments"], plan["segments"])
        self._write_clips({"c01": {"title": "Learn fast", "pieces": [{"words": [0, 7]}]}})
        rebuilt = P.load_plan(str(self.P), "c01")
        self.assertEqual(len(rebuilt["pieces"]), 1)
        self.assertNotEqual(path.stat().st_mtime_ns, mtime)

    def test_fail_lints_exit_1(self):
        self._write_clips({"c02": {"pieces": [{"words": [0, 7], "expect": ["nothing like this", ""]}]}})
        args = SimpleNamespace(project=str(self.P), clips=["c02"], quiet=True)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(P.cmd_plan(args), 1)
        self.assertIn("FAIL L02", out.getvalue())
        with self.assertRaises(env.HCError):
            P.load_plan(str(self.P), "c02", require_ok=True)
        with self.assertRaises(env.HCError):
            P.cmd_plan(SimpleNamespace(project=str(self.P), clips=["nope"], quiet=True))

    def test_caption_only(self):
        av = make_av_fixture(self.dir / "co", fps="25", dur=6, flashes=[1.0])
        target = self.P / "finished.mp4"
        if not target.exists():
            os.symlink(av, target)
        words = [("Already", 0.5, 0.9), ("edited", 1.0, 1.4), ("video.", 1.5, 2.0)]
        env.write_json(self.P / "transcripts" / "c03.json", fake_transcript(words, media="finished.mp4"))
        self._write_clips({"c03": {"kind": "caption_only", "source_file": "finished.mp4",
                                   "fix": {"1": "EDITED"}}})
        plan = P.build_plan(str(self.P), "c03")
        self.assertEqual(plan["kind"], "caption_only")
        self.assertEqual(len(plan["segments"]), 1)
        self.assertEqual(plan["frames_out"], 150)
        self.assertEqual([w["s_out"] for w in plan["words"]], [0.5, 1.0, 1.5])
        self.assertEqual(plan["words"][1]["text"], "EDITED")
        self.assertFalse([x for x in plan["lints"] if x["level"] == "FAIL"])


if __name__ == "__main__":
    unittest.main()
