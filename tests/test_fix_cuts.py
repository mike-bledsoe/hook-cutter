"""Regression tests for the cuts fixes: multi-face framing and speaker turns, active picture
areas, VFR and edit-list grids, stage A on VFR sources, hold_after and overlapping pieces,
tighten around laughs, the audio clock, analysis pipes on MKV, concurrent renders, hard camera
switches with few changed cells, lints L04/L05/L06/L10/L12/L13/L16-L20, the in-edge dip and
edge offsets, the contact sheet's frame 0, the hook card without captions, fsutil."""
from __future__ import annotations

import math
import subprocess
import sys
import tempfile
import shutil
import threading
import types
import unittest
from fractions import Fraction
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "scripts"))

import numpy as np  # noqa: E402

import fixtures  # noqa: E402
from fixtures import fake_transcript  # noqa: E402
from hookcutter import env, framing, fsutil, project, render, shots  # noqa: E402
from hookcutter import plan as P  # noqa: E402
from hookcutter import spec as S  # noqa: E402
from hookcutter.audio import EnergyTrack  # noqa: E402

FLOOR = -60.0
VOICE = -20.0
PRESET = {"letterbox": {"y_center": 820, "blur": 28, "dim": 0.35},
          "motion": {"punch_in": 1.0, "min_gap": 3.5, "max_upscale": 3.0}}


# ---- helpers --------------------------------------------------------------------------------

def energy(dur, spans, floor=FLOOR, level=VOICE, hop=0.01, t0=0.0):
    n = int(round(dur / hop))
    db = np.full(n, floor, dtype=np.float64)
    for sp in spans:
        lv = sp[2] if len(sp) > 2 else level
        db[int(round((sp[0] - t0) / hop)):int(round((sp[1] - t0) / hop))] = lv
    return EnergyTrack(db, hop=hop, t0=t0)


def source(fps=(25, 1), dur=60.0, w=1920, h=1080, **video):
    v = {"fps": list(fps), "start": 0.0, "width": w, "height": h,
         "frames": int(math.floor(dur * fps[0] / fps[1] + 1e-9)), "hdr": "none", "sar": "1:1"}
    v.update(video)
    return {"fingerprint": "fp-test", "duration": dur, "video": v,
            "audio": {"start": 0.0, "index": 1}}


def mkspec(clips, **top):
    raw = {"version": 1, "clips": clips}
    raw.update(top)
    return S.normalize_spec(raw)


def build(words, clips, spans=None, extra=(), fps=(25, 1), dur=None, src=None, flags=None,
          cid="c01", E=None, **top):
    tr = fake_transcript(words)
    if flags:
        tr["flags"] = flags
    end = dur or (max(w[2] for w in words) + 5.0)
    if E is None:
        E = energy(end, list(spans if spans is not None else [(w[1], w[2]) for w in words])
                   + list(extra))
    spec = mkspec(clips, **top)
    return P.plan_clip(spec, cid, tr, src or source(fps, end), E, None)


def grid_words(n, start=0.6, wl=0.3, gap=0.1, sp=None, text=None):
    out, t = [], start
    for k in range(n):
        w = text[k] if text else "w%d" % k
        out.append((w, round(t, 3), round(t + wl, 3)) + ((sp,) if sp else ()))
        t += wl + gap
    return out


def lints(plan, lid, level=None):
    return [x for x in plan["lints"] if x["id"] == lid and (level is None or x["level"] == level)]


def ffmpeg(*args):
    subprocess.run([env.ffmpeg(), "-hide_banner", "-nostdin", "-loglevel", "error", "-y"]
                   + [str(a) for a in args], check=True)


def encode_gray(path, frames, fps=24, extra=()):
    """Encode uint8 gray frames (h, w) losslessly; returns path."""
    h, w = frames[0].shape
    cmd = [env.ffmpeg(), "-hide_banner", "-nostdin", "-loglevel", "error", "-y", "-f", "rawvideo",
           "-pix_fmt", "gray", "-s", "%dx%d" % (w, h), "-r", str(fps), "-i", "-",
           "-c:v", "libx264", "-qp", "0", "-pix_fmt", "yuv420p", *extra, str(path)]
    subprocess.run(cmd, input=b"".join(np.ascontiguousarray(f).tobytes() for f in frames),
                   check=True)
    return path


# ---- fsutil ---------------------------------------------------------------------------------

class TestFsutil(unittest.TestCase):
    def setUp(self):
        self.d = Path(tempfile.mkdtemp(prefix="hc-fs-"))

    def tearDown(self):
        shutil.rmtree(self.d, ignore_errors=True)

    def test_tmp_path_is_unique_in_the_same_dir(self):
        a, b = fsutil.tmp_path(self.d / "seg.mkv"), fsutil.tmp_path(self.d / "seg.mkv")
        self.assertNotEqual(a, b)
        self.assertEqual(a.parent, self.d)
        self.assertEqual(a.suffix, ".mkv")
        self.assertTrue(a.name.startswith("seg.") and ".tmp" in a.name)

    def test_atomic_writes(self):
        fsutil.atomic_write_text(self.d / "a" / "x.txt", "hello")
        fsutil.atomic_write_bytes(self.d / "y.bin", b"\x00\x01")
        self.assertEqual((self.d / "a" / "x.txt").read_text(), "hello")
        self.assertEqual((self.d / "y.bin").read_bytes(), b"\x00\x01")
        self.assertEqual(sorted(p.name for p in self.d.iterdir()), ["a", "y.bin"])

    def test_lock_waits_and_times_out(self):
        lock = self.d / ".lock"
        order = []
        with fsutil.file_lock(lock):
            with self.assertRaises(env.HCError) as cm:
                with fsutil.file_lock(lock, timeout=0.3, poll=0.05, what="test"):
                    pass
            self.assertEqual(cm.exception.code, 3)

            def other():
                with fsutil.file_lock(lock, poll=0.02):
                    order.append("other")
            th = threading.Thread(target=other)
            th.start()
            th.join(0.3)
            order.append("first")
        th.join(5)
        self.assertEqual(order, ["first", "other"])


# ---- multi-face framing and speaker turns ---------------------------------------------------

def face(x, h, y=0.4):
    return [x, y, h * 0.56, h, 0.9]


def two_shot(t1=60.0):
    fl = [{"x": 0.267, "y": 0.4, "h": 0.20, "presence": 1.0},
          {"x": 0.75, "y": 0.4, "h": 0.23, "presence": 1.0}]
    smp = [{"t_src": round(k * 0.25, 2), "faces": [face(0.267, 0.20), face(0.75, 0.23)]}
           for k in range(int(t1 * 4) + 1)]
    return {"shots": [{"start_src": 0.0, "end_src": t1, "kind": "two", "multi_face": True,
                       "faces": shots.label_faces(fl)}], "samples": smp}


def turn_plan(words, speakers=None, marks=()):
    """A one-segment plan over [0, end) whose words are (text, s, e, sp) in source = output time."""
    end = round(max(w[2] for w in words) + 0.5, 2)
    n = int(round(end * 25))
    ws = []
    for i, (t, s, e, sp) in enumerate(words):
        ws.append({"i": i, "raw": t, "text": t, "s_src": s, "e_src": e, "s_out": s, "e_out": e,
                   "sp": sp, "hidden": False, "omitted": False, "removed": False, "filler": False,
                   "piece": 0, "seg": 0})
    return {"clip": "t", "kind": "cut", "fps": [25, 1],
            "segments": [{"k": 0, "src": [0.0, n / 25.0], "out": [0.0, n / 25.0], "frames": n}],
            "joins": [], "words": ws, "framing": list(marks), "speakers": speakers or {}}


def talk(sp, t0, dur, wl=0.3, gap=0.1, text=None):
    out, t = [], t0
    while t + wl <= t0 + dur + 1e-9:
        out.append((text or "word", round(t, 3), round(t + wl, 3), sp))
        t += wl + gap
    return out


class TestMultiFace(unittest.TestCase):
    SRC = {"video": {"width": 1280, "height": 720, "sar": "1:1"}}
    MAP = {"Host": {"x": 0.267}, "Guest": {"x": 0.75}}

    def test_similar_faces_propose_and_default_letterbox(self):
        sh = two_shot()
        prop = shots.propose(sh["shots"])
        self.assertEqual(prop[0]["mode"], "letterbox")
        self.assertTrue(prop[0]["multi_face"])
        self.assertEqual([f["id"] for f in prop[0]["faces"]], ["A", "B"])
        self.assertNotIn("x", shots.paste_block(prop)[0])
        self.assertEqual(framing.default_mark("vertical", sh, 5.0)["mode"], "letterbox")
        sub = framing.build_tracks(turn_plan([("a", 1.0, 1.3, None)]), {}, sh, self.SRC,
                                   "vertical", PRESET)[0][0]
        self.assertEqual(sub["mode"], "letterbox")

    def test_single_face_shot_still_tracks(self):
        sh = {"shots": [{"start_src": 0.0, "end_src": 30.0, "kind": "single",
                         "faces": [{"x": 0.62, "y": 0.4, "h": 0.3, "presence": 1.0}]}],
              "samples": [{"t_src": k * 0.25, "faces": [face(0.62, 0.3)]} for k in range(121)]}
        self.assertEqual(shots.propose(sh["shots"])[0], {"at_src": 0.0, "mode": "track",
                                                         "x": 0.62, "crop": 1.0})
        sub = framing.build_tracks(turn_plan([("a", 1.0, 1.3, None)]), {}, sh, self.SRC,
                                   "vertical", PRESET)[0][0]
        cx = (sub["cmds"][0][1] + sub["size"][0] / 2) / 1280.0
        self.assertAlmostEqual(cx, 0.62, delta=0.01)

    def test_speaker_follow_switches_at_turns_not_interjections(self):
        words = (talk("Host", 0.5, 4.0) + talk("Guest", 4.9, 2.4)
                 + [("Yeah.", 7.5, 8.1, "Host")] + talk("Guest", 8.3, 2.4)
                 + talk("Host", 11.2, 3.0))
        plan = turn_plan(words, self.MAP)
        subs = framing.build_tracks(plan, {}, two_shot(), self.SRC, "vertical", PRESET)[0]
        self.assertEqual([s["speaker"] for s in subs], ["Host", "Guest", "Host"])
        for s in subs:
            want = self.MAP[s["speaker"]]["x"]
            for _t, x0, _y in s["cmds"]:
                self.assertAlmostEqual((x0 + s["size"][0] / 2) / 1280.0, want, delta=0.02)
        # the switch to the guest happens just before his first word, the "Yeah." does not switch
        t_switch = subs[0]["frames"] / 25.0
        self.assertAlmostEqual(t_switch, 4.9 - framing.TURN_LEAD, delta=0.041)
        win = framing.windows([subs], plan, self.SRC)
        self.assertEqual(len(win), 3)
        self.assertTrue(all(w["lo"] <= self.MAP[w["speaker"]]["x"] <= w["hi"] for w in win))
        self.assertTrue(all(w["multi_face"] for w in win))

    def test_speaker_follow_needs_every_label_mapped(self):
        words = talk("Host", 0.5, 4.0) + talk("Third", 4.9, 3.0)
        subs = framing.build_tracks(turn_plan(words, {"Host": {"x": 0.267}}), {}, two_shot(),
                                    self.SRC, "vertical", PRESET)[0]
        self.assertEqual([s["mode"] for s in subs], ["letterbox"])

    def test_track_mark_without_x_uses_the_speakers_map(self):
        words = talk("Guest", 0.5, 4.0)
        plan = turn_plan(words, self.MAP, marks=[{"at_src": 0.0, "mode": "track"}])
        sub = framing.build_tracks(plan, {}, two_shot(), self.SRC, "vertical", PRESET)[0][0]
        self.assertEqual(sub["speaker"], "Guest")
        self.assertAlmostEqual((sub["cmds"][0][1] + sub["size"][0] / 2) / 1280.0, 0.75, delta=0.02)
        plan = turn_plan(words, {}, marks=[{"at_src": 0.0, "mode": "track"}])
        sub = framing.build_tracks(plan, {}, two_shot(), self.SRC, "vertical", PRESET)[0][0]
        self.assertEqual(sub["mode"], "letterbox")

    def test_speaker_mark_follows_from_its_time(self):
        words = talk("Host", 0.5, 4.0) + talk("Guest", 4.9, 3.0)
        marks = [{"at_src": 0.0, "mode": "fixed", "x": 0.5, "y": 0.5},
                 {"at_src": 2.0, "mode": "speaker"}]
        subs = framing.build_tracks(turn_plan(words, self.MAP, marks), {}, two_shot(), self.SRC,
                                    "vertical", PRESET)[0]
        self.assertEqual(subs[0]["mark"], 0)
        self.assertEqual([s.get("speaker") for s in subs[1:]], ["Host", "Guest"])

    def test_norm_marks_keeps_reaction_and_rect(self):
        m = framing.norm_marks([{"at_src": 1, "mode": "letterbox", "rect": [0, 0.25, 0.5, 0.5],
                                 "reaction": True}])[0]
        self.assertTrue(m["reaction"])
        self.assertEqual(m["rect"], [0.0, 0.25, 0.5, 0.5])

    def test_proposals_follow_turns_with_a_map(self):
        sh = two_shot(20.0)
        words = [{"w": "x", "s": s, "e": e, "sp": sp} for _t, s, e, sp in
                 talk("Host", 0.5, 4.0) + talk("Guest", 4.9, 3.0)]
        prop = shots.propose(sh["shots"], self.MAP, words)
        self.assertEqual([(m["mode"], m["x"], m["speaker"]) for m in prop],
                         [("track", 0.267, "Host"), ("track", 0.75, "Guest")])
        self.assertAlmostEqual(prop[1]["at_src"], 4.9 - framing.TURN_LEAD, places=3)
        self.assertEqual(set(shots.paste_block(prop)[1]), {"at_src", "mode", "x", "crop"})
        turns = shots.speaker_turns(words, [(0.0, 20.0)])
        self.assertEqual([t["sp"] for t in turns], ["Host", "Guest"])
        res = {"shots": sh["shots"], "speaker_turns": turns, "proposal": shots.propose(sh["shots"])}
        line = shots.multi_face_lines(res)[0]
        self.assertTrue(line.startswith("MULTI-FACE shot"))
        self.assertIn("faces A x=0.27, B x=0.75", line)
        self.assertIn('"Host": {"x": ...}', line)

    def test_plan_carries_speakers(self):
        words = [(w, s, e, "Host") for w, s, e in grid_words(8)]
        plan = build(words, {"c01": {"pieces": [{"words": [0, 7]}]}},
                     speakers={"Host": {"x": 0.3}})
        self.assertEqual(plan["speakers"], {"Host": {"x": 0.3}})
        self.assertEqual(plan["pieces"][0]["speakers"], ["Host"])
        self.assertEqual((plan["pieces"][0]["in_offset"], plan["pieces"][0]["out_offset"]), (0, 0))


# ---- active picture area --------------------------------------------------------------------

class TestActiveArea(unittest.TestCase):
    SRC = {"video": {"width": 1280, "height": 720, "sar": "1:1", "active_area": [0, 180, 1280, 360]}}

    def test_crops_stay_inside_the_band(self):
        self.assertEqual(framing.active_rect(self.SRC), (0, 180, 1280, 360))
        self.assertEqual(framing.active_rect({"video": {"width": 1280, "height": 720}}),
                         (0, 0, 1280, 720))
        for mark in ({"at_src": 0, "mode": "fixed", "x": 0.5, "y": 0.5},
                     {"at_src": 0, "mode": "track", "x": 0.3}):
            plan = turn_plan([("a", 1.0, 1.3, None)], marks=[mark])
            sub = framing.build_tracks(plan, {}, None, self.SRC, "vertical", PRESET)[0][0]
            self.assertEqual(sub["size"][1], 360)
            self.assertTrue(all(c[2] == 180 for c in sub["cmds"]))
        self.assertAlmostEqual(framing.upscale(1280, 720, "vertical", rect=(0, 180, 1280, 360)),
                               1920 / 360.0)

    def test_l12_measures_the_active_picture(self):
        words = grid_words(10)
        src = source((25, 1), 10.0, 1280, 720, active_area=[0, 180, 1280, 360])
        plan = build(words, {"c01": {"pieces": [{"words": [0, 9]}]}}, src=src)
        msg = lints(plan, "L12", "WARN")[0]["msg"]
        self.assertIn("5.3x", msg)
        self.assertIn("1280x360", msg)

    def test_rect_mark_crops_the_rect(self):
        mark = {"at_src": 0, "mode": "letterbox", "rect": [0.5, 0.25, 0.5, 0.5]}
        plan = turn_plan([("a", 1.0, 1.3, None)], marks=[mark])
        src = {"video": {"width": 1280, "height": 720, "sar": "1:1"}}
        sub = framing.build_tracks(plan, {}, None, src, "vertical", PRESET)[0][0]
        self.assertEqual(sub["rect"], [640, 180, 640, 360])
        chain = render._sub_chain("[0:0]", "[v]", sub, 0, 0.0, Fraction(25), src["video"],
                                  (1080, 1920), None, False)
        self.assertIn("crop=640:360:640:180", chain)
        w = framing.windows([[sub]], plan, src)[0]
        self.assertEqual((w["mode"], w["lo"], w["hi"]), ("rect", 0.5, 1.0))
        # letterbox and the blurred fill come from the active picture, not the bars
        sub2 = framing.build_tracks(turn_plan([("a", 1.0, 1.3, None)],
                                              marks=[{"at_src": 0, "mode": "letterbox"}]),
                                    {}, None, self.SRC, "vertical", PRESET)[0][0]
        self.assertEqual(sub2["rect"], [0, 180, 1280, 360])
        chain = render._sub_chain("[0:0]", "[v]", sub2, 0, 0.0, Fraction(25), self.SRC["video"],
                                  (1080, 1920), None, False)
        self.assertIn("crop=1280:360:0:180", chain.split("split=2")[0])


# ---- grid end: VFR, edit lists, audio end ---------------------------------------------------

class TestGridEnd(unittest.TestCase):
    def test_vfr_grid_runs_to_the_timed_end(self):
        G = P.Grid(30, 1, 0.0)
        v = {"fps": [30, 1], "vfr": True, "frames": 1089, "nb_frames": 1089, "duration": 40.34}
        self.assertEqual(P._kmax(G, v, 40.34), 1210)
        words = grid_words(94, start=0.3)          # last word ends at 37.8
        self.assertLessEqual(words[-1][2], 37.84)
        src = source((30, 1), 40.34, vfr=True, frames=1089, nb_frames=1089, duration=40.34)
        plan = build(words, {"c01": {"pieces": [{"words": [88, 93]}]}}, src=src, fps=(30, 1))
        self.assertGreaterEqual(plan["segments"][-1]["src"][1], words[-1][2])
        self.assertTrue(all(not w["hidden"] for w in plan["words"] if w["i"] >= 88))
        co = P.plan_caption_only(mkspec({"v": {"kind": "caption_only", "source_file": "v.mp4"}}),
                                 "v", fake_transcript(grid_words(3)),
                                 {"duration": 40.34, "video": dict(src["video"])})
        self.assertEqual(co["frames_out"], 1210)

    def test_edit_listed_cfr_is_capped_at_its_timed_end(self):
        G = P.Grid(25, 1, 0.0)
        v = {"fps": [25, 1], "frames": 964, "nb_frames": 964, "duration": 37.26}
        self.assertEqual(P._kmax(G, v, 37.26), 931)
        words = grid_words(91, start=0.6)         # last word ends at 37.2
        src = source((25, 1), 37.26, frames=964, nb_frames=964, duration=37.26)
        plan = build(words, {"c01": {"pieces": [{"words": [85, 90]}]}}, src=src, dur=38.6)
        self.assertLessEqual(plan["segments"][-1]["src"][1], 37.24 + 1e-6)

    def test_out_edge_never_past_the_audio_end(self):
        words = grid_words(10)
        E = energy(4.4, [(w[1], w[2]) for w in words])        # audio ends at 4.4
        plan = build(words, {"c01": {"pieces": [{"words": [5, 9]}]}}, E=E, dur=8.0)
        self.assertLessEqual(plan["segments"][-1]["src"][1], 4.4 - 0.01 + 1e-9)


# ---- hold_after and overlapping pieces ------------------------------------------------------

class TestHoldAndOverlap(unittest.TestCase):
    def test_hold_never_replays_the_next_piece(self):
        words = grid_words(30, wl=0.30, gap=0.07)
        plan = build(words, {"c01": {"pieces": [{"words": [10, 14], "hold_after": 0.2},
                                                {"words": [15, 19]}]}}, fps=(30000, 1001))
        a, b = plan["segments"][0], plan["segments"][-1]
        self.assertLessEqual(a["src"][1], b["src"][0] + 1e-9)
        msg = lints(plan, "L16", "WARN")[0]["msg"]
        self.assertIn("clamped at word 15 where pieces[1] starts", msg)
        self.assertEqual(plan["overlaps"], [])
        self.assertFalse(lints(plan, "I01"))

    def test_shared_words_overlap_warns_and_cold_open_fails(self):
        words = grid_words(30)
        plan = build(words, {"c01": {"pieces": [{"words": [0, 5]}, {"words": [5, 9]}]}})
        x = lints(plan, "L17", "WARN")
        self.assertEqual(len(x), 1)
        self.assertIn("pieces[0] and pieces[1] share words 5-5 ('w5')", x[0]["msg"])
        self.assertEqual(plan["overlaps"][0]["pieces"], [0, 1])
        self.assertEqual(plan["overlaps"][0]["words"], [5, 5])
        plan = build(words, {"c01": {"pieces": [{"words": [10, 14], "role": "cold_open"},
                                                {"words": [0, 19]}]}})
        self.assertTrue(lints(plan, "L17", "FAIL"))
        self.assertGreater(plan["overlaps"][0]["seconds"], 1.5)
        self.assertTrue(P.fail_lints(plan))


# ---- tighten keeps laughs -------------------------------------------------------------------

class TestTightenLaughs(unittest.TestCase):
    SPEC = [(1.4, [(0.575, 0.25, -26)]),            # chuckle centred: was 100% removed
            (1.4, [(1.02, 0.30, -26)]),             # laugh ending 0.08 s before the word: 60% cut
            (1.4, [(0.05, 0.30, -26)]),             # laugh right after the word: 70% cut
            (1.4, [(0.40, 0.12, -24), (0.62, 0.12, -24), (0.84, 0.12, -24)]),   # ha ha ha
            (2.0, [(0.75, 0.50, -26)]),             # 0.5 s laugh in a 2 s pause
            (1.4, [(0.30, 0.80, -40)])]             # soft 0.8 s laugh

    def test_laughs_survive_and_quiet_parts_go(self):
        words, extra, gaps, t = [], [], [], 0.6
        for g, (glen, laughs) in enumerate(self.SPEC):
            for j in range(5):
                words.append(("g%dw%d%s" % (g, j, "." if j == 4 else ""), round(t, 3),
                              round(t + 0.3, 3)))
                t += 0.3 + (0.12 if j < 4 else 0)
            gaps.append((t, t + glen, [(t + a, t + a + ln) for a, ln, _ in laughs]))
            extra += [(t + a, t + a + ln, db) for a, ln, db in laughs]
            t += glen
        for j in range(5):
            words.append(("endw%d" % j, round(t, 3), round(t + 0.3, 3)))
            t += 0.42
        plan = build(words, {"c01": {"pieces": [{"words": [0, len(words) - 1]}]}}, extra=extra)
        rem = [r["src"] for r in plan["removals"] if r["kind"] != "omit"]
        for k, (gs, ge, laughs) in enumerate(gaps):
            for la, lb in laughs:
                cut = sum(max(0.0, min(lb, b) - max(la, a)) for a, b in rem)
                self.assertEqual(cut, 0.0, "gap %d laugh %.2f-%.2f" % (k, la, lb))
            removed = sum(max(0.0, min(ge, b) - max(gs, a)) for a, b in rem)
            if k < 3:
                self.assertGreater(removed, 0.3, k)        # the quiet parts still go
            else:
                self.assertEqual(removed, 0.0, k)           # unchanged: never tightened


# ---- in-edge dip and edge offsets -----------------------------------------------------------

class TestInEdge(unittest.TestCase):
    WORDS = [("Before", 0.4, 0.9), ("and", 1.30, 1.50), ("Then", 1.50, 1.90), ("we", 2.0, 2.2),
             ("went.", 2.3, 2.7), ("Next", 3.4, 3.7)]
    SPANS = [(0.4, 0.9), (1.30, 1.49), (1.49, 1.53, -34.0), (1.53, 1.90), (2.0, 2.2), (2.3, 2.7),
             (3.4, 3.7)]

    def test_cut_lands_in_the_dip_after_a_dropped_touching_word(self):
        plan = build(self.WORDS, {"c01": {"pieces": [{"words": [2, 4]}]}}, spans=self.SPANS)
        e = plan["pieces"][0]["edge_in"]
        self.assertEqual(e["guard"], "dip")
        self.assertAlmostEqual(e["src"], 1.52, places=6)
        plan = build(self.WORDS, {"c01": {"pieces": [{"words": [2, 4], "in_offset": 2,
                                                      "out_offset": -1}]}}, spans=self.SPANS)
        e = plan["pieces"][0]["edge_in"]
        self.assertAlmostEqual(e["src"], 1.60, places=6)
        self.assertEqual(e["offset"], 2)
        base = build(self.WORDS, {"c01": {"pieces": [{"words": [2, 4]}]}}, spans=self.SPANS)
        self.assertEqual(plan["pieces"][0]["edge_out"]["k"], base["pieces"][0]["edge_out"]["k"] - 1)
        self.assertEqual(plan["pieces"][0]["in_offset"], 2)

    def test_pick_in_the_silence_before_a_late_onset_stays(self):
        # the previous word's stamp runs to the start, but its audio ended earlier; the next word
        # really starts 0.1 s before its (late) stamp with a soft onset: the silence is the cut
        words = [("Before", 0.4, 0.9), ("and", 1.30, 1.66), ("I", 1.66, 1.84), ("we", 2.0, 2.2),
                 ("went.", 2.3, 2.7), ("Next", 3.4, 3.7)]
        spans = [(0.4, 0.9), (1.30, 1.48), (1.55, 1.57, -45.0), (1.57, 1.84), (2.0, 2.2),
                 (2.3, 2.7), (3.4, 3.7)]
        plan = build(words, {"c01": {"pieces": [{"words": [2, 4]}]}}, spans=spans)
        e = plan["pieces"][0]["edge_in"]
        self.assertNotEqual(e["guard"], "dip")
        self.assertTrue(1.48 <= e["src"] <= 1.56, e)

    def test_kept_previous_word_needs_no_dip(self):
        plan = build(self.WORDS, {"c01": {"pieces": [{"words": [0, 1]}, {"words": [2, 4]}]}},
                     spans=self.SPANS)
        self.assertNotEqual(plan["pieces"][1]["edge_in"].get("guard"), "dip")

    def test_offset_never_crosses_into_another_kept_word(self):
        words = grid_words(12)
        plan = build(words, {"c01": {"pieces": [{"words": [0, 4], "out_offset": 12},
                                                {"words": [5, 9]}]}})
        self.assertLessEqual(plan["pieces"][0]["src"][1], words[5][1] + 1e-9)


# ---- lints ----------------------------------------------------------------------------------

class TestLints(unittest.TestCase):
    def test_l19_speech_without_words(self):
        words = grid_words(6) + [("after", 7.6, 7.9), ("words.", 8.0, 8.3)]
        voiced = (3.2, 7.2)                                   # 4 s of sound, no words
        plan = build(words, {"c01": {"pieces": [{"words": [0, 7]}]}}, extra=[voiced])
        x = lints(plan, "L19", "FAIL")
        self.assertEqual(len(x), 1)
        self.assertIn("4.0 s of speech at out", x[0]["msg"])
        flag = {"kind": "unworded_audio", "s": 3.2, "e": 7.2, "a": 5, "b": 6}
        plan = build(words, {"c01": {"pieces": [{"words": [0, 7]}]}}, extra=[voiced], flags=[flag])
        self.assertFalse(lints(plan, "L19", "FAIL"))
        self.assertTrue(lints(plan, "L19", "WARN"))
        self.assertTrue(any("sound with no words" in m["msg"] for m in lints(plan, "L15")))
        resp = {"kind": "respliced", "s": 1.0, "e": 2.0, "a": 0, "b": 3, "n": 4}
        plan = build(words, {"c01": {"pieces": [{"words": [0, 7]}]}}, flags=[resp])
        self.assertFalse(lints(plan, "L15"))

    def test_l18_dead_air_at_the_start(self):
        words = grid_words(8, start=1.0)
        spans = [(w[1] + (0.6 if k == 0 else 0.0), w[2] + (0.6 if k == 0 else 0.0))
                 for k, w in enumerate(words)]
        spans[1] = (words[1][1] + 0.3, words[1][2])
        plan = build(words, {"c01": {"pieces": [{"words": [0, 7]}]}}, spans=spans)
        x = lints(plan, "L18", "WARN")
        self.assertTrue(x and "opens on 0.6" in x[0]["msg"], plan["lints"])
        ok = build(words, {"c01": {"pieces": [{"words": [0, 7]}]}})
        self.assertFalse(lints(ok, "L18"))

    def test_l10_levels(self):
        words = grid_words(12)
        clip = {"c01": {"pieces": [{"words": [2, 9]}], "omit": [5], "tighten": {"enabled": False}}}

        def run(level):
            speech = [(words[0][1], words[-1][2], level)]          # continuous speech, no pauses
            return build(words, clip, spans=speech, dur=30.0)
        mild = run(-40.0)               # 20 dB over the floor: edges WARN, the omit join INFO
        joins = [x for x in lints(mild, "L10") if x["msg"].startswith("omit join")]
        self.assertTrue(joins and all(x["level"] == "INFO" for x in joins), mild["lints"])
        self.assertTrue(any(x["msg"].startswith("clip starts") and x["level"] == "WARN"
                            for x in lints(mild, "L10")))
        soft = run(-45.0)               # 15 dB over: tight, but only INFO
        self.assertFalse(lints(soft, "L10", "WARN"), soft["lints"])
        self.assertTrue(lints(soft, "L10", "INFO"))
        loud = run(-20.0)               # 40 dB over: everything WARNs
        self.assertTrue(any(x["msg"].startswith("omit join") for x in lints(loud, "L10", "WARN")))

    def test_l04_determiner_is_not_a_bare_pronoun(self):
        txt = "That conversation changed everything for me today.".split()
        plan = build(grid_words(7, text=txt), {"c01": {"pieces": [{"words": [0, 6]}]}})
        self.assertFalse(any("bare pronoun" in x["msg"] for x in lints(plan, "L04")))
        txt = "That is why it changed for me.".split()
        plan = build(grid_words(7, text=txt), {"c01": {"pieces": [{"words": [0, 6]}]}})
        self.assertTrue(any("bare pronoun" in x["msg"] for x in lints(plan, "L04")))

    def test_l06_restart_is_not_a_lost_qualifier(self):
        txt = "We do it. I just I just think it works. Done here.".split()
        words = grid_words(len(txt), text=txt)
        plan = build(words, {"c01": {"pieces": [{"words": [3, 9]}], "omit": [4, 5]}})
        self.assertFalse(lints(plan, "L06"))
        txt = "We do it. I just think it works fine. Done here.".split()
        plan = build(grid_words(len(txt), text=txt),
                     {"c01": {"pieces": [{"words": [3, 9]}], "omit": [4]}})
        self.assertTrue(lints(plan, "L06"))

    def test_l05_sentence_end_with_speech_running_on(self):
        txt = "It all starts with. And so it is just".split()
        words, t = [], 0.6
        for k, w in enumerate(txt):
            words.append((w, round(t, 3), round(t + 0.3, 3)))
            t += 0.3 + (0.03 if k == 3 else 0.1)
        spans = [(words[0][1], words[-1][2])]                  # continuous speech
        plan = build(words, {"c01": {"pieces": [{"words": [0, 3]}]}}, spans=spans)
        x = lints(plan, "L05", "WARN")
        self.assertTrue(x and "speech continues immediately" in x[0]["msg"], plan["lints"])
        self.assertIn("'And so it is'", x[0]["msg"])
        # the same stamps with a real breath (0.13 s quiet) before "And": a clean close
        spans = [(words[0][1], words[3][2]), (words[4][1] + 0.10, words[-1][2])]
        clean = build(words, {"c01": {"pieces": [{"words": [0, 3]}]}}, spans=spans)
        self.assertFalse(lints(clean, "L05"), clean["lints"])

    def test_l13_one_extractor(self):
        # "A NORMAL ONE" is not a number claim (the old extractor read ONE as 1)
        txt = "my brain vs a normal person is wild".split()
        words = grid_words(len(txt), text=txt)
        plan = build(words, {"c01": {"pieces": [{"words": [0, 7]}],
                                     "hook_card": {"lines": ["MY BRAIN VS A *NORMAL* ONE"]}}})
        self.assertFalse(lints(plan, "L13"), plan["lints"])
        txt = "it made us two times faster this year".split()
        plan = build(grid_words(len(txt), text=txt),
                     {"c01": {"pieces": [{"words": [0, 7]}], "hook_card": {"lines": ["*10X* FASTER"]}}})
        self.assertTrue(any("number '10X'" in x["msg"] for x in lints(plan, "L13", "WARN")))

    def test_l20_hook_card_smaller_than_captions(self):
        clip = {"hook_card": {"lines": ["A LONG HOOK LINE HERE"]},
                "_eff": {"styles": ["clean"], "formats": ["vertical"]}}
        fake = types.SimpleNamespace(hook_fit=lambda c, p, f, r=None: {
            "hook_px": 70, "caption_px": 64, "ratio": 70 / 64.0, "lines": 1, "max_chars": 12})
        proj = types.SimpleNamespace(root=Path(tempfile.gettempdir()))
        import hookcutter
        from hookcutter import render_util as ru
        with mock.patch.dict(sys.modules, {"hookcutter.captions": fake}), \
                mock.patch.object(hookcutter, "captions", fake, create=True), \
                mock.patch.object(ru, "load_preset", lambda *a, **k: {}):
            out = P._lint_hook_fit(proj, clip, "c01")
        self.assertEqual(len(out), 1)
        self.assertEqual((out[0]["id"], out[0]["level"]), ("L20", "WARN"))
        self.assertIn("70 px against 64 px captions in clean: shorten lines to <= 12", out[0]["msg"])

    def test_unknown_clip_suggests(self):
        spec = mkspec({"c01-hello": {"pieces": [{"words": [0, 1]}]}})
        with self.assertRaises(env.HCError) as cm:
            P._check_id(spec, "c01")
        self.assertIn("did you mean 'c01-hello'", str(cm.exception))
        self.assertEqual(cm.exception.code, 1)


# ---- render: contact sheet, hook card without captions, stage A chain ------------------------

class TestRenderBits(unittest.TestCase):
    def test_contact_sheet_starts_at_frame_zero(self):
        idx = render.contact_frames(250)
        self.assertEqual(len(idx), render.CONTACT_FRAMES)
        self.assertEqual(idx[0], 0)
        self.assertEqual(idx[1], int(250 / 22))
        self.assertEqual(idx[-1], int(21 * 250 / 22))

    def test_captions_false_keeps_the_hook_card(self):
        seen = {}

        class Track:
            def __init__(self, pages, hook):
                self.band_h, self.pages, self.hook = 40, pages, hook

        def build_track(plan, clip, preset, fmt, size, fps, luma, project_root=None):
            seen["clip"] = clip
            return Track([], {"lines": ["HOOK"]} if clip.get("hook_card") else None)
        fake = types.SimpleNamespace(build_track=build_track)
        proj = types.SimpleNamespace(root=Path(tempfile.gettempdir()))
        import hookcutter
        with mock.patch.dict(sys.modules, {"hookcutter.captions": fake}), \
                mock.patch.object(hookcutter, "captions", fake, create=True):
            track, why = render._caption_track({}, {"captions": False, "hook_card": {"lines": ["HOOK"]}},
                                               {}, "vertical", Fraction(25), None, proj)
            self.assertIsNotNone(track)
            self.assertIsNone(why)
            track, why = render._caption_track({}, {"captions": False}, {}, "vertical",
                                               Fraction(25), None, proj)
            self.assertIsNone(track)
            self.assertEqual(why, "captions off for this clip")

    def test_stage_a_maps_slots_before_trimming(self):
        sub = {"src": [2.0, 3.0], "frames": 25, "mode": "crop", "size": [606, 1080],
               "cmds": [[0.0, 600, 0]]}
        video = {"width": 1920, "height": 1080, "sar": "1:1", "index": 0}
        chain = render._sub_chain("[0:0]", "[v]", sub, 0, 1.0, Fraction(25), video, (1080, 1920),
                                  None, True)
        self.assertLess(chain.index("fps=25/1:round=down"), chain.index("trim=start_frame=25"))
        self.assertIn("end_frame=50", chain)
        self.assertNotIn("tpad", chain)


# ---- media: energy clock, MKV analysis, VFR render, concurrency, hard switches ---------------

class TestMedia(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="hc-fixcuts-"))

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_energy_follows_the_wall_clock_across_an_audio_hole(self):
        d = self.tmp / "hole"
        d.mkdir()
        sr = 48000
        t = np.arange(int(6.0 * sr)) / sr
        x = (0.3 * np.sin(2 * np.pi * 440 * t) * ((t % 1.0) < 0.5)).astype("<f4")
        (d / "a.f32").write_bytes(x.tobytes())
        ffmpeg("-f", "f32le", "-ar", sr, "-ac", 1, "-i", d / "a.f32",
               "-af", "asetpts='if(gte(PTS*TB,3.0),PTS+0.23/TB,PTS)'", "-c:a", "aac", "-b:a", "128k",
               d / "hole.m4a")
        E = EnergyTrack.from_media(d / "hole.m4a")
        self.assertAlmostEqual(E.end, 6.23, delta=0.03)
        # the tone burst that starts at 4.0 s (after the hole) shows at 4.23 s on the wall clock
        self.assertLess(E.db_at(4.20, 0.02), E.db_at(4.30, 0.02) - 20)

    def test_mkv_seeks_do_not_trip_the_deny_list(self):
        d = self.tmp / "mkv"
        d.mkdir()
        mkv = d / "s24.mkv"
        ffmpeg("-f", "lavfi", "-i", "testsrc2=size=320x180:rate=24:duration=6", "-c:v", "libx264",
               "-g", "12", "-pix_fmt", "yuv420p", mkv)
        from hookcutter import probe
        src = {"video": probe.probe_media(mkv)["video"]}
        for k in range(15):
            t = (24 + 5 * k) / 24.0 - 0.5 / 24.0
            shots.refine_cut(mkv, src, t, 6.0)           # raises HCError on the deny list
            self.assertTrue(shots.spike_at(mkv, src, t, 6.0) in (True, False))

    def test_hard_switch_with_few_changed_cells_is_a_cut(self):
        # dark close-up (a big face-sized patch) -> dark wide (small figures): few cells move by
        # more than 20 levels, but the switch happens in one frame. A gesture sweeps over many.
        d = self.tmp / "switch"
        d.mkdir()
        h, w, fps = 180, 320, 24
        rng = np.random.default_rng(5)
        noise = rng.integers(0, 3, (fps * 10, h, w), dtype=np.uint8)
        frames = []
        for n in range(fps * 10):
            f = np.full((h, w), 24, np.uint8) + noise[n]
            if n < fps * 5:
                f[30:150, 100:220] = 70                    # close-up: one large, dim face
            else:
                f[80:110, 60:80] = 90                      # wide: small figures
                f[80:110, 240:260] = 90
            frames.append(f)
        src_path = encode_gray(d / "switch.mp4", frames, fps)
        gframes = []
        for n in range(fps * 10):
            f = np.full((h, w), 24, np.uint8) + noise[n]
            f[30:150, 100:220] = 70
            if fps * 4 <= n < fps * 6:                      # a hand crossing over 2 s
                x0 = int(20 + (n - fps * 4) * 6)
                f[60:140, x0:x0 + 60] = 120
            gframes.append(f)
        gest_path = encode_gray(d / "gesture.mp4", gframes, fps)
        from hookcutter import probe
        for path, want in ((src_path, 1), (gest_path, 0)):
            src = {"video": probe.probe_media(path)["video"]}
            times, grays, _ = shots._decode_range(path, src, 0.0, 10.0)
            g = [x for x in grays]
            changed = max(shots.changed_share(g[j], g[j - shots.LAG]) for j in range(shots.LAG, len(g)))
            self.assertLess(changed, shots.MIN_CHANGED)            # the cell floor alone misses it
            cuts, thr = shots.detect_cuts(times, g, spike=lambda tc, th: shots.spike_at(path, src, tc, th))
            self.assertEqual(len(cuts), want, (path.name, cuts, thr))
            if want:
                self.assertAlmostEqual(shots.refine_cut(path, src, cuts[0][0], thr), 5.0, delta=0.05)
            self.assertEqual(shots.detect_cuts(times, g)[0], [])   # no spike check: as before

    def test_vfr_gap_renders_exact_frames(self):
        d = self.tmp / "vfr"
        d.mkdir()
        src = d / "vfr.mp4"
        ffmpeg("-f", "lavfi", "-i", "testsrc2=size=320x180:rate=30:duration=8",
               "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=8",
               "-vf", "select='not(between(n,60,104))+not(mod(n,15))',setpts=N/30/TB*0+PTS",
               "-fps_mode", "passthrough", "-c:v", "libx264", "-pix_fmt", "yuv420p",
               "-c:a", "aac", "-shortest", src)
        P_ = fixtures.make_project(d, src)
        proj = project.Project(P_)
        info = proj.source()
        num, den = info["video"]["fps"]
        rate = Fraction(num, den)
        segs, cur = [], 0
        for k, (s, e) in enumerate([(1.0, 2.6), (3.2, 4.4)]):
            n = int(round((Fraction(e).limit_denominator(1000) - Fraction(s).limit_denominator(1000)) * rate))
            segs.append({"k": k, "piece": k, "src": [s, e], "out": [cur / float(rate), (cur + n) / float(rate)],
                         "frames": n, "samples": int(round(Fraction(n * 48000) / rate)),
                         "join_before": "start" if k == 0 else "piece", "fade_in": 0.008,
                         "fade_out": 0.008, "gain_db": 0.0})
            cur += n
        plan = {"version": 1, "clip": "vfr", "kind": "cut", "fps": [num, den], "grid_offset": 0.0,
                "sample_rate": 48000, "pieces": [], "segments": segs, "removals": [], "joins": [],
                "words": [], "intent": [], "frames_out": cur, "duration_out": cur / float(rate),
                "hook_card": None, "framing": [{"at_src": 0.0, "mode": "fixed", "x": 0.5, "y": 0.5}],
                "lints": []}
        clip = {"kind": "cut", "captions": False, "framing": plan["framing"]}
        res = render.render_clip(proj, "vfr", formats=["vertical"], styles=["clean"], plan=plan, clip=clip)
        self.assertEqual(res[0]["frames"], cur)


class TestConcurrentRender(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="hc-conc-"))
        src = fixtures.make_av_fixture(cls.tmp, fps="25", dur=4, flashes=[1.0, 2.6])
        cls.P = fixtures.make_project(cls.tmp, src)
        cls.proj = project.Project(cls.P)
        rate = Fraction(25)
        segs, cur = [], 0
        for k, (s, e) in enumerate([(0.4, 1.6), (2.0, 3.2)]):
            n = int(round((Fraction(e).limit_denominator(1000) - Fraction(s).limit_denominator(1000)) * rate))
            segs.append({"k": k, "piece": k, "src": [s, e], "out": [cur / 25.0, (cur + n) / 25.0],
                         "frames": n, "samples": int(round(Fraction(n * 48000) / rate)),
                         "join_before": "start" if k == 0 else "piece", "fade_in": 0.008,
                         "fade_out": 0.008, "gain_db": 0.0})
            cur += n
        cls.plan = {"version": 1, "clip": "conc", "kind": "cut", "fps": [25, 1], "grid_offset": 0.0,
                    "sample_rate": 48000, "pieces": [], "segments": segs, "removals": [],
                    "joins": [{"k": 1, "out": segs[1]["out"][0], "kind": "piece",
                               "before_src": 1.6, "after_src": 2.0}],
                    "words": [], "intent": [], "frames_out": cur, "duration_out": cur / 25.0,
                    "hook_card": None,
                    "framing": [{"at_src": 0.0, "mode": "fixed", "x": 0.4, "y": 0.5},
                                {"at_src": 2.4, "mode": "letterbox"}], "lints": []}
        cls.clip = {"kind": "cut", "captions": False, "framing": cls.plan["framing"]}

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_two_renders_of_one_clip_serialize(self):
        errs, done = [], []

        def go():
            try:
                done.append(render.render_clip(self.proj, "conc", formats=["vertical"],
                                               styles=["clean"], force=True, plan=self.plan,
                                               clip=self.clip))
            except Exception as exc:          # noqa: BLE001 (report any crash)
                errs.append(exc)
        ths = [threading.Thread(target=go) for _ in range(2)]
        for th in ths:
            th.start()
        for th in ths:
            th.join(300)
        self.assertEqual(errs, [])
        self.assertEqual(len(done), 2)
        out = self.proj.render_path("conc", "vertical", "clean")
        first = env.sha256_file(out)
        render.render_clip(self.proj, "conc", formats=["vertical"], styles=["clean"], force=True,
                           plan=self.plan, clip=self.clip)
        self.assertEqual(env.sha256_file(out), first)
        wd = self.proj.work_dir("conc", "vertical")
        self.assertFalse([p for p in wd.iterdir() if ".tmp" in p.name])
        log = env.read_json(self.proj.render_path("conc", "vertical", "clean", ".log.json"))
        # I2: the hashes a render records are reproducible without rendering
        self.assertEqual(render.plan_hash(self.plan), log["hashes"]["plan"])
        exp = render.expected_hashes(self.proj, "conc", "vertical", "clean", plan=self.plan,
                                     clip=self.clip)
        self.assertEqual(exp, {"plan": log["hashes"]["plan"], "render": log["hashes"]["render"]})
        # a new cover headline, post copy or title (spec_hash moves) does not make it stale
        plan2 = dict(self.plan, spec_hash="other", set_hash="other")
        clip2 = dict(self.clip, cover={"headline": "A NEW *COVER*"}, post={"title": "t"}, title="x")
        self.assertEqual(render.expected_hashes(self.proj, "conc", "vertical", "clean", plan=plan2,
                                                clip=clip2), exp)
        # a new hook card does
        clip3 = dict(self.clip, hook_card={"lines": ["NEW *HOOK*"], "dur": 1.5})
        self.assertNotEqual(render.expected_hashes(self.proj, "conc", "vertical", "clean",
                                                   plan=self.plan, clip=clip3)["render"], exp["render"])
        # I3: one framing window per SubSeg, letterbox spans the frame
        fw = log["framing_windows"]
        self.assertEqual(len(fw), sum(len(s) for s in log["framing"]))
        self.assertEqual([w["mode"] for w in fw], ["crop", "crop", "letterbox"])
        self.assertEqual((fw[-1]["lo"], fw[-1]["hi"]), (0.0, 1.0))
        self.assertLess(fw[0]["lo"], 0.4)
        self.assertGreater(fw[0]["hi"], 0.4)


if __name__ == "__main__":
    unittest.main()
