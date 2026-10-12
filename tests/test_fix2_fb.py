"""Regression tests for the second captions and covers fix round: pages and subtitle cues never
cross a cut, the per-clip backing override, inserted words, the build-reveal pill, a merged token
that wraps, and the cover's dissolve and crown handling.

Every test is synthetic (generated plans, words, luma arrays and drawn frames)."""
from __future__ import annotations

import math
import random
import re
import shutil
import sys
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "scripts"))

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from hookcutter import captions, cover, cover_frames  # noqa: E402

import test_captions as TC  # noqa: E402  (helpers: build, load_preset, CLEAN)

CLEAN = TC.CLEAN
POP = TC.load_preset("pop")
MINIMAL = TC.load_preset("minimal")
HARD = ("piece", "omit", "dip")


def seam_plan(segs, fps=(30, 1), hook=None):
    """A plan on the output frame grid, as plan.py writes it.

    `segs`: [(join kind before the segment ("start" for the first), frames, words)] with words
    [(text, start, end)] in seconds from the segment's first frame (an optional 4th item is a
    dict of extra word keys). Word times are clamped to the segment and rounded to 0.1 ms; a
    piece or dip join starts a new piece."""
    num, den = fps
    words, segments, joins = [], [], []
    cum, piece, i = 0, 0, 0
    for k, (kind, frames, seg_words) in enumerate(segs):
        if kind in ("piece", "dip"):
            piece += 1
        o0, o1 = float(Fraction(cum * den, num)), float(Fraction((cum + frames) * den, num))
        segments.append({"k": k, "piece": piece, "src": [100.0 * k, 100.0 * k + (o1 - o0)],
                         "out": [round(o0, 6), round(o1, 6)], "frames": frames,
                         "join_before": kind})
        if k:
            joins.append({"k": k, "out": round(o0, 6), "kind": kind})
        for row in seg_words:
            text, s, e = row[:3]
            extra = dict(row[3]) if len(row) > 3 else {}
            s_out = round(min(max(round(o0, 6) + s, round(o0, 6)), round(o1, 6)), 4)
            e_out = round(min(max(round(o0, 6) + e, s_out), round(o1, 6)), 4)
            word = {"i": i, "raw": text, "text": text, "s_src": 100.0 * k + s,
                    "e_src": 100.0 * k + e, "s_out": s_out, "e_out": e_out, "piece": piece,
                    "seg": k, "sp": None, "filler": False, "hidden": False, "omitted": False,
                    "removed": False}
            word.update(extra)
            words.append(word)
            if word["i"] is not None:
                i += 1
        cum += frames
    plan = {"version": 1, "clip": "c01", "fps": [num, den], "words": words, "joins": joins,
            "segments": segments, "frames_out": cum,
            "duration_out": round(float(Fraction(cum * den, num)), 6)}
    if hook:
        plan["hook_card"] = hook
    return plan


def spoken(text, start=0.1, wps=3.0, gap=0.0):
    """[(word, start, end)] evenly timed from `start`."""
    out, t, d = [], start, 1.0 / wps
    for w in text.split():
        out.append((w, round(t, 3), round(t + 0.85 * d, 3)))
        t += d + gap
    return out


def hard_joins(plan):
    """[(first frame of the segment after the join, exact seconds, segment k)] of hard joins."""
    num, den = plan["fps"]
    first, cum = {}, 0
    for sg in plan["segments"]:
        first[sg["k"]] = cum
        cum += sg["frames"]
    return [(first[j["k"]], Fraction(first[j["k"]] * den, num), j["k"])
            for j in plan["joins"] if j["kind"] in HARD]


def crossings(plan, track):
    """Pages that show on both sides of a hard join (by frames or by their written seconds)."""
    bad = []
    for p in track.pages:
        a, b = p["frames"]
        for f, t, k in hard_joins(plan):
            if a < f < b or (p["start_out"] < float(t) - 1e-9 and p["end_out"] > float(t) + 1e-9):
                bad.append((p["text"], p["frames"], p["start_out"], p["end_out"], f))
    return bad


def group_of_frame(plan, n):
    return sum(1 for f, _, _ in hard_joins(plan) if f <= n)


def group_of_page(plan, page):
    """The stretch between hard joins that holds the page's words (by their plan segment)."""
    seg = {w["i"]: w["seg"] for w in plan["words"] if w["i"] is not None}
    ks = sorted(k for _, _, k in hard_joins(plan))
    idx = [i for t in page["tokens"] for i in t["idx"]]
    if not idx:
        return None
    return {sum(1 for k in ks if k <= seg[i]) for i in idx}


def drawn_wrong_side(plan, track):
    """Frames whose drawn page belongs to the other side of a hard join (what the viewer sees)."""
    bad = []
    pages = track.pages
    for n in range(track.frames):
        key = track.frame_key(n)
        if key[0] != "page":
            continue
        groups = group_of_page(plan, pages[key[1]])
        if groups is not None and groups != {group_of_frame(plan, n)}:
            bad.append((n, pages[key[1]]["text"]))
    return bad


# ----------------------------------------------------------------------------- 1. pages and seams

class PagesStayOnOneSideOfACut(unittest.TestCase):
    def test_lead_does_not_light_a_page_before_the_cut(self):
        # The second piece opens on a word at the cut. With a 60 ms lead the page used to start
        # two frames early, over the last frames of the previous shot.
        plan = seam_plan([("start", 169, spoken("we shipped the product first.")),
                          ("piece", 120, spoken("Then nobody bought it.", start=0.0))])
        for preset in (CLEAN, POP):
            track = TC.build(plan, preset=preset)
            page = next(p for p in track.pages if p["text"].startswith("THEN"))
            self.assertEqual(page["frames"][0], 169, page)
            self.assertGreaterEqual(page["start_out"], 169 / 30.0 - 1e-9, page)
            self.assertEqual(crossings(plan, track), [])
            self.assertEqual(drawn_wrong_side(plan, track), [])

    def test_a_page_due_within_a_frame_of_the_cut_shows_on_the_cut_frame(self):
        # The first word after the cut is spoken 67 ms in. With the 60 ms lead its page is due
        # 7 ms after the cut, which rounded up to the next frame: the page before ends at the
        # cut, so the cut frame itself carried no caption (a one-frame blink).
        late = [("Then", 0.067, 0.3), ("nobody", 0.3, 0.6), ("bought", 0.6, 0.9), ("it.", 0.9, 1.2)]
        plan = seam_plan([("start", 290, spoken("we shipped the product first.", start=8.2)),
                          ("piece", 120, late)])
        track = TC.build(plan)
        before = [p for p in track.pages if p["frames"][0] < 290][-1]
        after = [p for p in track.pages if p["frames"][0] >= 290][0]
        self.assertEqual(before["frames"][1], 290, before)
        self.assertEqual(after["frames"][0], 290, after)
        self.assertEqual(track.frame_key(290)[0], "page")
        self.assertEqual(track.pages[track.frame_key(290)[1]]["text"], after["text"])
        self.assertGreater(after["start_out"], 290 / 30.0)          # the time is still the word's
        self.assertEqual(crossings(plan, track), [])
        self.assertEqual(drawn_wrong_side(plan, track), [])
        # a page due a frame or more after the cut starts when its word does
        later = [("Then", 0.2, 0.4), ("nobody", 0.4, 0.7), ("bought", 0.7, 1.0), ("it.", 1.0, 1.3)]
        plan = seam_plan([("start", 290, spoken("we shipped the product first.", start=8.2)),
                          ("piece", 120, later)])
        after = [p for p in TC.build(plan).pages if p["frames"][0] >= 290][0]
        self.assertEqual(after["frames"][0], 295, after)             # 0.2 s - 60 ms lead = 4.2 frames
        # and the hook card is never undercut: a cut that falls under the card pulls nothing
        plan = seam_plan([("start", 40, spoken("so here it is.", start=0.2)),
                          ("piece", 150, [("We", 0.18, 0.4), ("sold", 0.4, 0.7), ("it", 0.7, 0.9),
                                          ("first.", 0.9, 1.3)])],
                         hook={"lines": ["SOLD IT *FIRST*"], "dur": 1.5})
        track = TC.build(plan)
        self.assertGreaterEqual(track.pages[0]["frames"][0], 45, track.pages[0])

    def test_tail_stops_at_the_cut(self):
        # The last word ends 0.1 s before the cut and the next piece opens on a second of
        # silence: the 0.4 s tail used to run nine frames into the next shot.
        plan = seam_plan([("start", 169, [("we", 4.6, 4.8), ("shipped.", 4.8, 5.53)]),
                          ("piece", 150, spoken("Then nobody bought it.", start=1.0))])
        track = TC.build(plan)
        first = track.pages[0]
        self.assertEqual(first["text"], "WE SHIPPED")
        self.assertLessEqual(first["frames"][1], 169, first)
        self.assertLessEqual(first["end_out"], 169 / 30.0 + 1e-9, first)
        self.assertEqual(crossings(plan, track), [])

    def test_min_page_backfill_stops_at_the_cut(self):
        # "GO" shows for 0.4 s; min_page (0.8 s) used to back-fill it 0.36 s before the cut, so a
        # dimmed caption sat over the previous shot.
        plan = seam_plan([("start", 150, spoken("we shipped the product first.", wps=2.5)),
                          ("piece", 150, [("Go.", 0.1, 0.25), ("Now", 0.5, 0.7), ("we", 0.7, 0.85),
                                          ("sell", 0.85, 1.1), ("it.", 1.1, 1.4)])])
        track = TC.build(plan)
        page = next(p for p in track.pages if p["text"].startswith("GO"))
        self.assertGreaterEqual(page["frames"][0], 150, page)
        self.assertGreaterEqual(page["start_out"], 5.0 - 1e-9, page)
        self.assertEqual(crossings(plan, track), [])
        self.assertEqual(drawn_wrong_side(plan, track), [])

    def test_every_hard_join_kind_bounds_a_page_and_tighten_does_not(self):
        for kind in HARD:
            plan = seam_plan([("start", 100, [("alpha", 2.6, 2.9), ("beta", 2.9, 3.3)]),
                              (kind, 100, [("gamma", 0.0, 0.3), ("delta", 0.3, 0.6)])])
            for preset in (CLEAN, POP, MINIMAL):
                track = TC.build(plan, preset=preset)
                self.assertEqual(crossings(plan, track), [], (kind, preset["name"]))
                self.assertEqual(drawn_wrong_side(plan, track), [], (kind, preset["name"]))
                self.assertEqual(track.pages[0]["frames"][1], 100, (kind, preset["name"]))
                self.assertEqual(track.pages[1]["frames"][0], 100, (kind, preset["name"]))
        plan = seam_plan([("start", 100, [("alpha", 2.6, 2.9), ("beta", 2.9, 3.3)]),
                          ("tighten", 100, [("gamma", 0.0, 0.3), ("delta", 0.3, 0.6)])])
        page = TC.build(plan).pages[0]
        self.assertEqual(page["text"], "ALPHA BETA GAMMA DELTA")       # one page across a tighten
        self.assertLess(page["frames"][0], 100)
        self.assertGreater(page["frames"][1], 100)

    def test_join_frame_is_exact_at_a_third_of_a_second(self):
        # 308 frames at 30 fps is 10.266667 s once rounded: a hair past frame 308. The page after
        # the cut starts on frame 308 itself, and the page before it ends there.
        plan = seam_plan([("start", 308, spoken("we shipped the product first.", start=8.4)),
                          ("piece", 90, spoken("Then nobody bought it.", start=0.0))])
        track = TC.build(plan)
        before = [p for p in track.pages if p["frames"][0] < 308]
        after = [p for p in track.pages if p["frames"][0] >= 308]
        self.assertEqual(before[-1]["frames"][1], 308, before[-1])
        self.assertEqual(after[0]["frames"][0], 308, after[0])
        self.assertLessEqual(before[-1]["end_out"], 308 / 30.0)
        self.assertGreaterEqual(after[0]["start_out"], 308 / 30.0)

    def test_join_frames_without_segment_frame_counts_fall_back_to_out(self):
        plan = seam_plan([("start", 308, spoken("we shipped the product first.", start=8.4)),
                          ("piece", 90, spoken("Then nobody bought it.", start=0.0))])
        for sg in plan["segments"]:
            del sg["frames"]
        track = TC.build(plan)
        self.assertEqual(crossings(plan_with_frames(plan, [308, 90]), track), [])
        self.assertEqual([p["frames"][0] for p in track.pages if p["frames"][0] >= 308][0], 308)

    def test_a_suffix_glued_across_an_omit_join_stays_before_it(self):
        # "80" + "%" with a word omitted between them: the glued token belongs to the side it
        # starts on, so its page still ends at the join.
        plan = seam_plan([("start", 60, [("about", 1.2, 1.5), ("80", 1.5, 1.9)]),
                          ("omit", 90, [("%", 0.0, 0.2), ("of", 0.2, 0.4), ("them", 0.4, 0.8)])])
        track = TC.build(plan)
        self.assertEqual(track.pages[0]["text"], "ABOUT 80%")
        self.assertLessEqual(track.pages[0]["frames"][1], 60)
        self.assertGreaterEqual(track.pages[1]["frames"][0], 60)

    def test_hook_card_then_a_cut(self):
        plan = seam_plan([("start", 80, spoken("so here is the thing about it.", start=0.2)),
                          ("piece", 120, spoken("We sold it first and built it after.", start=0.0))],
                         hook={"lines": ["SOLD IT *FIRST*"], "dur": 1.5})
        for preset in (CLEAN, POP):
            track = TC.build(plan, preset=preset)
            self.assertEqual(crossings(plan, track), [])
            self.assertEqual(drawn_wrong_side(plan, track), [])

    def test_random_plans_never_cross_a_hard_join(self):
        text = ("we built the product for a whole year and then it did not sell so we went back "
                "and sold it first to forty strangers in ninety days. Then we built it. Right, "
                "that was the order, and nobody told us. So, here is what I would do now.").split()
        presets = (CLEAN, POP, MINIMAL)
        rates = ((30, 1), (30000, 1001), (25, 1), (60, 1))
        pages = crossed_tighten = 0
        for seed in range(60):
            rng = random.Random(seed)
            fps = rates[seed % len(rates)]
            spf = fps[1] / fps[0]
            segs, w = [], 0
            for k in range(rng.randint(2, 7)):
                frames = rng.randint(8, 150)
                dur = frames * spf
                rows, t = [], rng.choice([0.0, 0.0, 0.03, 0.2, 0.6])
                while t < dur - 0.05 and w < len(text) * 3:
                    d = rng.uniform(0.12, 0.45)
                    rows.append((text[w % len(text)], round(t, 3), round(min(t + d, dur), 3)))
                    w += 1
                    t += d + rng.choice([0.0, 0.0, 0.02, 0.1, 0.5])
                kind = "start" if k == 0 else rng.choice(HARD + ("tighten", "piece"))
                segs.append((kind, frames, rows))
            hook = {"lines": ["SOLD IT *FIRST*"], "dur": 1.2} if seed % 3 == 0 else None
            plan = seam_plan(segs, fps=fps, hook=hook)
            track = captions.build_track(plan, {}, presets[seed % 3], "vertical", (TC.W, TC.H), fps,
                                         None)
            self.assertEqual(crossings(plan, track), [], seed)
            self.assertEqual(drawn_wrong_side(plan, track), [], seed)
            pages += len(track.pages)
            first = {sg["k"]: sum(s["frames"] for s in plan["segments"][:n])
                     for n, sg in enumerate(plan["segments"])}
            for j in plan["joins"]:
                if j["kind"] == "tighten":
                    f = first[j["k"]]
                    crossed_tighten += sum(1 for p in track.pages
                                           if p["frames"][0] < f < p["frames"][1])
        self.assertGreater(pages, 300)
        self.assertGreater(crossed_tighten, 0)      # the rule is about cuts, not every join


def plan_with_frames(plan, frames):
    out = dict(plan, segments=[dict(sg, frames=f) for sg, f in zip(plan["segments"], frames)])
    return out


class PagingKnowsTheCut(unittest.TestCase):
    def test_paging_counts_screen_time_only_up_to_the_cut(self):
        # Six quick words, then a cut. Paged 4 + 2, "WHITE PINK" gets 0.46 s before the cut (its
        # 0.4 s tail is on the other side); 3 + 3 gives both pages about 0.66 s. Without the
        # cut the tail counts and 4 + 2 is right.
        rows = [(w, 2.0 + 0.2 * k, 2.2 + 0.2 * k)
                for k, w in enumerate("red green blue black white pink".split())]
        cut = seam_plan([("start", 96, rows), ("piece", 60, [("Then", 0.5, 0.8), ("grey.", 0.8, 1.2)])])
        free = seam_plan([("start", 156, rows + [("Then", 3.7, 4.0), ("grey.", 4.0, 4.4)])])
        self.assertEqual(TC.page_texts(TC.build(cut))[:2], ["RED GREEN BLUE", "BLACK WHITE PINK"])
        self.assertEqual(TC.page_texts(TC.build(free))[:2], ["RED GREEN BLUE BLACK", "WHITE PINK"])
        self.assertEqual(crossings(cut, TC.build(cut)), [])

    def test_schedule_without_seams_keeps_its_old_contract(self):
        # schedule() is still callable without seam data (no clamp at the joins).
        plan = seam_plan([("start", 90, spoken("alpha beta gamma delta."))])
        style = captions.resolve_style(CLEAN, "vertical")
        cap = style["caption"]
        setter = captions._Setter(cap, None)
        toks = captions.display_tokens(plan, {}, cap)
        pages = captions.paginate(toks, setter, 700.0, cap["layout"], [], 0.06)
        pages = captions.schedule(pages, setter, 700.0, cap["layout"], 0.06, 0.0, 3.0, [])
        self.assertEqual([" ".join(t.text for t in p.tokens) for p in pages],
                         ["ALPHA BETA GAMMA DELTA"])

    def test_seams_object(self):
        plan = seam_plan([("start", 169, []), ("piece", 58, []), ("tighten", 81, []),
                          ("omit", 90, []), ("dip", 30, [])])
        seams = captions._Seams(plan, 30, 1, 428, 428 / 30.0)
        self.assertEqual(seams.ks, [1, 3, 4])
        self.assertEqual(seams.span(0, 0), (0, 169))
        self.assertEqual(seams.span(1, 2), (169, 308))       # the tighten join is inside
        self.assertEqual(seams.span(2, 2), (169, 308))
        self.assertEqual(seams.span(3, 3), (308, 398))
        self.assertEqual(seams.span(4, 4), (398, 428))
        lo, hi = seams.bounds(1, 2)
        self.assertAlmostEqual(lo, 169 / 30.0, places=9)
        self.assertAlmostEqual(hi, 308 / 30.0, places=9)
        self.assertTrue(seams.between(0, 1))
        self.assertFalse(seams.between(1, 2))
        self.assertEqual(captions._frame_snap(10.266667, 30, 1), 308)
        self.assertEqual(captions._frame_ceil(10.266667, 30, 1), 309)   # why the snap exists
        self.assertEqual(captions._frame_snap(10.27, 30, 1), 309)


# ----------------------------------------------------------------------------- 2. SRT

def srt_stamps(text):
    """[(start ms, end ms)] of a written .srt."""
    out = []
    for a, b in re.findall(r"(\d\d:\d\d:\d\d,\d\d\d) --> (\d\d:\d\d:\d\d,\d\d\d)", text):
        def ms(s):
            h, m, rest = s.split(":")
            sec, milli = rest.split(",")
            return ((int(h) * 60 + int(m)) * 60 + int(sec)) * 1000 + int(milli)
        out.append((ms(a), ms(b)))
    return out


class SubtitleCuesAndCuts(unittest.TestCase):
    def test_cues_break_at_hard_joins(self):
        # One sentence runs across the cut with no pause: the cue used to read across it.
        for kind in HARD:
            plan = seam_plan([("start", 100, spoken("we sold it and", start=2.2, wps=4.0)),
                              (kind, 120, spoken("then we built it after that.", start=0.0, wps=4.0))])
            cues = TC.build(plan).srt_cues()
            texts = [c[2].replace("\n", " ") for c in cues]
            self.assertEqual(texts, ["we sold it and", "then we built it after that."], kind)
            cut = 100 / 30.0
            self.assertLessEqual(cues[0][1], cut + 1e-9, kind)
            self.assertGreaterEqual(cues[1][0], cut - 1e-9, kind)

    def test_a_tighten_join_does_not_break_a_cue(self):
        plan = seam_plan([("start", 100, spoken("we sold it and", start=2.2, wps=4.0)),
                          ("tighten", 120, spoken("then we built it after that.", start=0.0, wps=4.0))])
        texts = [c[2].replace("\n", " ") for c in TC.build(plan).srt_cues()]
        self.assertEqual(texts, ["we sold it and then we built it after that."])

    def test_no_cue_ends_after_the_clip(self):
        # The last word ends 0.1 s before the end of the file: the 0.4 s cue tail used to run past it.
        plan = seam_plan([("start", 88, [("it", 2.2, 2.4), ("makes", 2.4, 2.6), ("no", 2.6, 2.75),
                                         ("sense.", 2.75, 2.83)])])
        track = TC.build(plan)
        dur = 88 / 30.0
        cues = track.srt_cues()
        self.assertLessEqual(cues[-1][1], dur + 1e-9, cues)
        with tempfile.TemporaryDirectory() as d:
            text = track.write_srt(Path(d) / "c.srt").read_text(encoding="utf-8")
        stamps = srt_stamps(text)
        self.assertTrue(stamps)
        self.assertLessEqual(stamps[-1][1], math.floor(dur * 1000 + 1e-6), stamps)
        for a, b in stamps:
            self.assertLess(a, b)
            self.assertGreaterEqual(a, 0)

    def test_written_stamps_never_cross_a_join(self):
        # A cut at 9.666667 s: the cue before it is written 00:00:09,666 (not ,667).
        plan = seam_plan([("start", 290, spoken("the heavy lift got", start=8.6, wps=4.0)),
                          ("piece", 60, spoken("really light.", start=0.0, wps=4.0))])
        track = TC.build(plan)
        with tempfile.TemporaryDirectory() as d:
            stamps = srt_stamps(track.write_srt(Path(d) / "c.srt").read_text(encoding="utf-8"))
        self.assertEqual(len(stamps), 2)
        self.assertEqual(stamps[0][1], 9666)
        self.assertEqual(stamps[1][0], 9667)

    def test_random_plans(self):
        text = ("we built the product for a whole year and then it did not sell so we went back "
                "and sold it first to forty strangers in ninety days. Then we built it.").split()
        n_cues = 0
        for seed in range(40):
            rng = random.Random(1000 + seed)
            segs, w = [], 0
            for k in range(rng.randint(2, 6)):
                frames = rng.randint(10, 140)
                dur = frames / 30.0
                rows, t = [], rng.choice([0.0, 0.0, 0.05, 0.4])
                while t < dur - 0.05:
                    d = rng.uniform(0.12, 0.4)
                    rows.append((text[w % len(text)], round(t, 3), round(min(t + d, dur), 3)))
                    w += 1
                    t += d + rng.choice([0.0, 0.02, 0.3])
                segs.append(("start" if k == 0 else rng.choice(HARD + ("tighten",)), frames, rows))
            plan = seam_plan(segs)
            track = TC.build(plan)
            cues = track.srt_cues()
            n_cues += len(cues)
            dur = plan["frames_out"] / 30.0
            said = [wd["text"] for wd in plan["words"]]
            self.assertEqual(" ".join(c[2].replace("\n", " ") for c in cues).split(), said, seed)
            for k, (a, b, _) in enumerate(cues):
                self.assertGreaterEqual(a, 0.0, seed)
                self.assertLess(a, b, seed)
                self.assertLessEqual(b, dur + 1e-9, (seed, a, b, dur))
                if k + 1 < len(cues):
                    self.assertLessEqual(b, cues[k + 1][0] + 1e-9, seed)
                for _, t, _ in hard_joins(plan):
                    self.assertFalse(a < float(t) - 1e-9 and b > float(t) + 1e-9, (seed, a, b, float(t)))
        self.assertGreater(n_cues, 80)


# ----------------------------------------------------------------------------- 3. caption_backing

def flat_luma(plan, value):
    n = int(plan["duration_out"] * captions.LUMA_FPS) + 2
    return np.full((n, 480, 270), int(round(value * 255)), np.uint8)


def backing_plan():
    return seam_plan([("start", 150, spoken("sell it first.", start=2.2, wps=2.0))],
                     hook={"lines": ["SELL IT *FIRST*"], "dur": 1.5})


class ClipBackingOverride(unittest.TestCase):
    def test_always_pills_a_dark_clip_and_its_hook_card(self):
        plan = backing_plan()
        auto = TC.build(plan, luma=flat_luma(plan, 0.2))
        self.assertFalse(any(p["backing"] for p in auto.pages))
        self.assertFalse(auto.hook["backing"])
        track = TC.build(plan, clip={"caption_backing": "always"}, luma=flat_luma(plan, 0.2))
        self.assertTrue(track.pages)
        self.assertTrue(all(p["backing"] for p in track.pages), track.pages)
        self.assertEqual({p["backing_reason"] for p in track.pages}, {"always"})
        self.assertTrue(all(p["backing_box"] for p in track.pages))
        self.assertTrue(track.hook["backing"])
        self.assertEqual(track.to_json()["backing_mode"], "always")
        self.assertEqual(track.to_json()["backing_override"], "always")

    def test_never_bares_a_bright_clip_and_its_hook_card(self):
        plan = backing_plan()
        auto = TC.build(plan, luma=flat_luma(plan, 0.9))
        self.assertTrue(all(p["backing"] for p in auto.pages))
        self.assertTrue(auto.hook["backing"])
        track = TC.build(plan, clip={"caption_backing": "never"}, luma=flat_luma(plan, 0.9))
        self.assertFalse(any(p["backing"] for p in track.pages), track.pages)
        self.assertFalse(track.hook["backing"])
        self.assertTrue(all(p["backing_box"] is None for p in track.pages))
        # The override changes the pill and nothing else: same pages, lines and boxes.
        for a, b in zip(auto.pages, track.pages):
            for key in ("text", "lines", "box", "frames", "size"):
                self.assertEqual(a[key], b[key], key)
        self.assertEqual(auto.hook["box"], track.hook["box"])
        self.assertEqual(auto.to_json()["max_width"], track.to_json()["max_width"])

    def test_auto_overrides_a_preset_that_always_pills(self):
        plan = backing_plan()
        self.assertEqual(MINIMAL["caption"]["backing"]["mode"], "always")
        boxed = TC.build(plan, preset=MINIMAL, luma=flat_luma(plan, 0.2))
        self.assertTrue(all(p["backing"] for p in boxed.pages) and boxed.hook["backing"])
        track = TC.build(plan, preset=MINIMAL, clip={"caption_backing": "auto"},
                         luma=flat_luma(plan, 0.2))
        self.assertFalse(any(p["backing"] for p in track.pages))
        self.assertFalse(track.hook["backing"])
        bright = TC.build(plan, preset=MINIMAL, clip={"caption_backing": "auto"},
                          luma=flat_luma(plan, 0.9))
        self.assertTrue(all(p["backing"] for p in bright.pages) and bright.hook["backing"])

    def test_always_on_a_bare_preset_keeps_the_pill_in_the_safe_zone(self):
        bare = captions._deep_merge(CLEAN, {"caption": {"backing": {"mode": "never"}},
                                            "hook_card": {"backing": {"mode": "never"}}})
        plan = seam_plan([("start", 200, spoken("we shipped the whole product to everybody "
                                                "yesterday afternoon.", start=0.2, wps=2.5))])
        off = TC.build(plan, preset=bare, luma=flat_luma(plan, 0.9))
        self.assertFalse(any(p["backing"] for p in off.pages))
        track = TC.build(plan, preset=bare, clip={"caption_backing": "always"},
                         luma=flat_luma(plan, 0.2))
        self.assertTrue(all(p["backing"] for p in track.pages))
        zone = {"left": 90, "right": 150}                  # the organic profile
        for p in track.pages:
            x, _, w, _ = p["backing_box"]
            self.assertGreaterEqual(x, zone["left"], p)
            self.assertLessEqual(x + w, TC.W - zone["right"], p)
        self.assertLess(track.to_json()["max_width"], off.to_json()["max_width"])

    def test_the_value_rides_in_the_clip_or_its_effective_settings(self):
        self.assertEqual(captions.backing_override({"caption_backing": "Always"}), "always")
        self.assertEqual(captions.backing_override({"_eff": {"caption_backing": "never"}}), "never")
        self.assertEqual(captions.backing_override({"caption_backing": "never",
                                                    "_eff": {"caption_backing": "always"}}), "never")
        for junk in (None, "", "sometimes", True, 1):
            self.assertIsNone(captions.backing_override({"caption_backing": junk}), junk)
        self.assertIsNone(captions.backing_override(None))
        plan = backing_plan()
        track = TC.build(plan, clip={"caption_backing": "sometimes"}, luma=flat_luma(plan, 0.2))
        self.assertFalse(any(p["backing"] for p in track.pages))      # ignored: preset mode
        self.assertIsNone(track.to_json()["backing_override"])
        self.assertEqual(track.to_json()["backing_mode"], "auto")

    def test_hook_fit_reads_the_override(self):
        # A clip that turns the pill on for a bare preset gives the card the pill's room.
        bare = captions._deep_merge(CLEAN, {"caption": {"backing": {"mode": "never"}},
                                            "hook_card": {"backing": {"mode": "never"}}})
        style = captions.resolve_style(bare, "vertical")
        wide = captions._geometry(style, {}, "vertical", TC.W, TC.H)
        room = captions._geometry(style, {"caption_backing": "always"}, "vertical", TC.W, TC.H)
        self.assertLess(room["hk_max"], wide["hk_max"])
        self.assertEqual((wide["mode"], wide["hook_mode"]), ("never", "never"))
        self.assertEqual((room["mode"], room["hook_mode"]), ("always", "always"))


# ----------------------------------------------------------------------------- 4. inserted words

def inserted(n):
    return {"i": None, "ins": n}


def insert_plan(hook=None):
    """"...in this moment. [I use this method.] You are just there." with the bracketed words
    inserted by the editor (speech the transcript has no words for)."""
    plan = seam_plan([("start", 240, [
        ("method", 0.3, 0.7), ("in", 0.7, 0.85), ("this", 0.85, 1.05), ("moment.", 1.05, 1.5),
        ("I", 1.56, 1.61, inserted(0)), ("use", 1.61, 1.83, inserted(0)),
        ("this", 1.83, 2.02, inserted(0)), ("method.", 2.02, 2.30, inserted(0)),
        ("You", 2.6, 2.8), ("are", 2.8, 2.95), ("just", 2.95, 3.2), ("there.", 3.2, 3.6)])],
        hook=hook)
    plan["inserts"] = [{"n": 0, "after": 3, "tokens": ["I", "use", "this", "method."],
                        "src": [1.56, 2.30]}]
    return plan


class InsertedWords(unittest.TestCase):
    def test_inserted_words_are_caption_tokens_with_no_index(self):
        plan = insert_plan()
        for preset in (CLEAN, POP, MINIMAL):
            track = TC.build(plan, preset=preset)
            toks = [t for p in track.pages for t in p["tokens"]]
            self.assertEqual([t["text"].upper().strip(".") for t in toks],
                             "METHOD IN THIS MOMENT I USE THIS METHOD YOU ARE JUST THERE".split())
            ins = [t for t in toks if not t["idx"]]
            self.assertEqual([t["text"].upper().strip(".") for t in ins], ["I", "USE", "THIS", "METHOD"])
            self.assertEqual([t["idx"] for t in toks if t["idx"]], [[k] for k in range(8)])
            self.assertEqual(set(ins[0]), {"text", "s_out", "e_out", "emph", "source", "idx"})
            # they time like any word: on screen while they are spoken
            for t in ins:
                page = next(p for p in track.pages if t in p["tokens"])
                self.assertLessEqual(page["start_out"], t["s_out"] + 1e-6)
                self.assertGreaterEqual(page["end_out"], t["e_out"] - 1e-6)

    def test_inserted_words_page_and_reveal_like_any_word(self):
        plan = insert_plan()
        track = TC.build(plan, preset=POP)
        page_i = next(i for i, p in enumerate(track.pages)
                      if any(not t["idx"] for t in p["tokens"]))
        page = track.pages[page_i]
        j = next(k for k, t in enumerate(page["tokens"]) if not t["idx"])
        onset = page["tokens"][j]["s_out"] - 0.06
        before = track.frame_key(max(page["frames"][0], int(onset * 30) - 1))
        after = track.frame_key(min(page["frames"][1] - 1, int(onset * 30) + 6))
        if before[0] == "page" and before[1] == page_i:
            self.assertEqual(before[2][j], captions.INVISIBLE)        # build reveal: not yet
        self.assertEqual(after[:2], ("page", page_i))
        self.assertGreater(after[2][j][0], 200)                       # spoken: drawn
        for n in range(track.frames):                                 # every frame renders
            track.frame(n)

    def test_inserted_words_are_in_the_srt(self):
        text = " ".join(c[2].replace("\n", " ") for c in TC.build(insert_plan()).srt_cues())
        self.assertIn("in this moment. I use this method. You are just there.", text)

    def test_fix_and_emphasis_never_address_an_inserted_word(self):
        plan = insert_plan()
        # A fix keyed "None" (or by the insert number) and an emphasis on index 0 touch only
        # transcript words: the inserted text is drawn as typed and never accented by index.
        clip = {"fix": {"None": "WRONG", "0": "Method"}, "emphasis": [0]}
        for w in plan["words"]:
            if w["i"] == 0:
                w["text"] = "Method"                                   # as plan.py applies a fix
        track = TC.build(plan, clip=clip)
        toks = [t for p in track.pages for t in p["tokens"]]
        self.assertNotIn("WRONG", [t["text"] for t in toks])
        self.assertEqual([t["text"] for t in toks if t["emph"]], ["METHOD"])
        self.assertEqual([t["idx"] for t in toks if t["emph"]], [[0]])

    def test_hook_accent_can_land_on_an_inserted_word(self):
        plan = insert_plan(hook={"lines": ["I *USE* IT"], "dur": 0.2})
        track = TC.build(plan)
        hit = [t for p in track.pages for t in p["tokens"] if t["emph"]]
        self.assertEqual([(t["text"], t["source"], t["idx"]) for t in hit], [("USE", "hook", [])])
        self.assertEqual(track.to_json()["accents"]["matched"], ["USE"])

    def test_inserted_word_order_at_a_shared_start_and_glue(self):
        # An inserted word that starts with a transcript word sorts after the word it follows in
        # the plan; an inserted "%" glues to the number before it like any suffix.
        plan = seam_plan([("start", 120, [("about", 0.5, 0.9), ("80", 0.9, 1.2),
                                          ("%", 1.2, 1.3, inserted(0)), ("of", 1.3, 1.5),
                                          ("it", 1.3, 1.5, inserted(1)), ("all.", 1.6, 2.0)])])
        toks = captions.display_tokens(plan, {}, CLEAN["caption"])
        self.assertEqual([t.text for t in toks], ["ABOUT", "80%", "OF", "IT", "ALL"])
        self.assertEqual([t.idx for t in toks], [[0], [1], [2], [], [3]])

    def test_an_inserted_word_keeps_to_its_side_of_a_cut(self):
        plan = seam_plan([("start", 90, [("in", 2.0, 2.2), ("this", 2.2, 2.4), ("moment.", 2.4, 2.7),
                                         ("I", 2.75, 2.8, inserted(0)), ("do.", 2.8, 2.98, inserted(0))]),
                          ("piece", 90, spoken("You are just there.", start=0.0))])
        track = TC.build(plan)
        self.assertEqual(crossings(plan, track), [])
        page = next(p for p in track.pages if any(not t["idx"] for t in p["tokens"]))
        self.assertLessEqual(page["frames"][1], 90)


# ----------------------------------------------------------------------------- 5. build-reveal pill

class BuildRevealPill(unittest.TestCase):
    def setUp(self):
        self.pre = captions._deep_merge(POP, {"caption": {"backing": {"mode": "always"},
                                                          "emphasis": {"sources": []}}})
        self.plan = seam_plan([("start", 120, [("sell", 1.0, 1.3), ("it", 1.52, 1.7),
                                               ("first", 2.03, 2.4)])])
        self.track = TC.build(self.plan, preset=self.pre)
        self.page = self.track._pages[0]

    def alpha(self, n):
        t = self.track
        return np.frombuffer(t.frame(n), np.uint8).reshape(t.band_h, TC.W, 4)[..., 3]

    def test_the_pill_covers_a_word_from_the_first_frame_it_is_drawn(self):
        # Each word used to be drawn faint for a frame or two with no pill behind it: the pill
        # eased to it over the word's own fade (and faded in with the first word).
        t, page = self.track, self.page
        self.assertEqual(len(page.tokens), 3)
        pills = page.token_pills
        full = int(captions.parse_color(self.pre["caption"]["backing"]["color"])[3])
        first_seen = {}
        for n in range(page.start_f, page.end_f):
            key = t.frame_key(n)
            if key[0] != "page":
                continue
            for j, st in enumerate(key[2]):
                if st[0] > 0 and j not in first_seen:
                    first_seen[j] = (n, st[0])
        self.assertEqual(sorted(first_seen), [0, 1, 2])
        faint = 0
        for j, (n, a_word) in first_seen.items():
            faint += a_word < 255
            x, y, w, h = pills[j]
            row = self.alpha(n)[y + h // 2 - t.band_y]
            # the pill is whole (its own alpha) from its left edge to this word's right edge
            self.assertGreaterEqual(int(row[x + 4]), full - 2, (j, n))
            self.assertGreaterEqual(int(row[x + w - 5]), full - 2, (j, n))
            # and just above the word's cap height, inside the padding (never under a glyph)
            tx0, ty0, tx1, _ = page.token_typos[j]
            top = self.alpha(n)[int(ty0) - 8 - t.band_y]
            self.assertGreaterEqual(int(top[int(tx1) - 6]), full - 2, (j, n))
        self.assertGreater(faint, 0)            # the case that mattered: a word mid-fade

    def test_the_pill_is_never_ahead_of_the_newest_word(self):
        t, page = self.track, self.page
        pills = page.token_pills
        checked = 0
        for n in range(page.start_f, page.end_f):
            key = t.frame_key(n)
            if key[0] != "page":
                continue
            shown = [j for j, st in enumerate(key[2]) if st[0] > 0]
            if not shown or shown[-1] == len(pills) - 1:
                continue
            x, y, w, h = pills[shown[-1]]
            row = self.alpha(n)[y + h // 2 - t.band_y]
            self.assertEqual(int(row[x + w + 2:].max()), 0, n)
            checked += 1
        self.assertGreater(checked, 5)
        self.assertEqual(t.frame_key(page.start_f - 1), ("blank",))

    def test_fill_and_static_pills_are_whole_from_the_page_start(self):
        for name in ("clean", "minimal"):
            pre = captions._deep_merge(TC.load_preset(name), {"caption": {"backing": {"mode": "always"}}})
            track = TC.build(self.plan, preset=pre)
            page = track._pages[0]
            x, y, w, h = page.pill
            a = np.frombuffer(track.frame(page.start_f), np.uint8).reshape(track.band_h, TC.W, 4)[..., 3]
            row = a[y + h // 2 - track.band_y]
            self.assertGreater(int(row[x + 4]), 200, name)
            self.assertGreater(int(row[x + w - 5]), 200, name)


# ----------------------------------------------------------------------------- 8. merged token wraps

def merged_plan(text="Started dating", rest="my wife then."):
    """A display merge as `fix` makes it: word 0 carries the merged text, word 1 is hidden."""
    first, second = text.split()[0], " ".join(text.split()[1:])
    rows = [(text, 0.5, 0.9), (second, 0.9, 1.3, {"hidden": True})]
    t = 1.3
    for w in rest.split():
        rows.append((w, t, t + 0.25))
        t += 0.3
    plan = seam_plan([("start", 120, rows)])
    plan["words"][0]["raw"] = first
    return plan


class MergedTokenWraps(unittest.TestCase):
    def test_a_long_merge_takes_two_lines_at_full_size(self):
        # It used to be set as one unbreakable word and shrunk (73 px in the 78 px preset).
        for preset in (CLEAN, POP):
            track = TC.build(merged_plan(), preset=preset)
            page = track.pages[0]
            self.assertEqual(page["text"], "STARTED DATING")
            self.assertEqual(page["lines"], [["STARTED"], ["DATING"]], preset["name"])
            self.assertEqual(page["size"], preset["caption"]["size"], preset["name"])
            self.assertEqual(len(page["tokens"]), 1)                 # still one token, one index
            self.assertEqual(page["tokens"][0]["idx"], [0])
            self.assertNotIn("word_shrunk", [i["code"] for i in track.to_json()["issues"]])
            self.assertLessEqual(page["box"][2], track.to_json()["max_width"] + 1)

    def test_both_lines_are_drawn_and_the_token_reveals_as_one(self):
        track = TC.build(merged_plan(), preset=POP)
        page = track._pages[0]
        self.assertEqual(len(page.runs[0]), 2)
        (_, _, base0), (_, _, base1) = page.runs[0]
        self.assertGreater(base1, base0 + 40)
        n = page.end_f - 1
        key = track.frame_key(n)
        self.assertEqual(len(key[2]), 1)                             # one token state
        a = np.frombuffer(track.frame(n), np.uint8).reshape(track.band_h, TC.W, 4)[..., 3]
        for base in (base0, base1):
            band = a[int(base) - 40 - track.band_y:int(base) - 5 - track.band_y]
            self.assertGreater(int((band > 200).sum()), 500, base)   # ink on this line
        for n in range(page.start_f, page.end_f):                    # every frame renders
            track.frame(n)

    def test_a_wrapped_merge_with_emphasis_and_a_pill(self):
        pre = captions._deep_merge(POP, {"caption": {"backing": {"mode": "always"}}})
        track = TC.build(merged_plan(), preset=pre, clip={"emphasis": [0]})
        page = track._pages[0]
        self.assertTrue(page.tokens[0].emph)
        self.assertEqual(track.pages[0]["lines"], [["STARTED"], ["DATING"]])
        # the build pill grows to the whole token (both lines), not just its first line
        x, y, w, h = page.token_pills[0]
        self.assertGreaterEqual(y + h, int(page.runs[0][1][2]))
        for n in range(page.start_f, page.end_f):
            track.frame(n)

    def test_a_short_merge_stays_one_word_on_its_line(self):
        plan = merged_plan("what were", "you doing out there?")
        for preset in (CLEAN, POP, MINIMAL):
            track = TC.build(plan, preset=preset)
            runs = [r for p in track.pages for line in p["lines"] for r in line]
            self.assertIn("what were", [r.lower() for r in runs], preset["name"])
            page = track._pages[0]
            self.assertEqual(len(page.runs[0]), 1)
            self.assertEqual(page.size, preset["caption"]["size"])

    def test_a_merge_may_share_its_second_line_with_the_next_words(self):
        # Parts are line-break opportunities only: the token still pages as one unit.
        style = captions.resolve_style(CLEAN, "vertical")
        setter = captions._Setter(style["caption"], None)
        toks = captions.display_tokens(merged_plan("Started dating", "my"), {}, style["caption"])
        self.assertEqual([t.text for t in toks], ["STARTED DATING", "MY"])
        width = setter.run_width(toks[0], "STARTED DATING", setter.size)
        lines = setter.wrap(toks, setter.size, width - 20)            # the merge cannot sit whole
        self.assertIsNotNone(lines)
        self.assertEqual([[u.text for u in line] for line in lines], [["STARTED"], ["DATING", "MY"]])
        whole = setter.wrap(toks[:1], setter.size, width + 1)          # it fits: left whole
        self.assertEqual(whole, [[toks[0]]])
        self.assertIsNone(setter.wrap(toks, setter.size, 60.0))        # nothing fits 60 px


# ----------------------------------------------------------------------------- 7. the top of the head

BG = (96, 100, 108)
SKIN = (208, 152, 128)
HAIR = (44, 34, 30)


def head_photo(box, crown, hair=None, bg=BG, slats=False, glow=False, size=(1080, 1920)):
    """A picture with a head drawn around a face box [x, y, w, h]: a skin oval whose top is at
    `crown` (a bald head), or a `hair` coloured oval from `crown` with the face set into it.
    `slats`: a wall of vertical slats with dark gaps; `glow`: the wall is lit from the top."""
    W, H = size
    arr = np.empty((H, W, 3), np.float32)
    arr[:] = bg
    if slats:
        for x0 in range(0, W, 64):
            arr[:, x0:x0 + 10] = (30, 22, 20)
    if glow:
        arr *= np.linspace(1.35, 0.8, H, dtype=np.float32)[:, None, None]
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    d = ImageDraw.Draw(img)
    x, y, w, h = box
    cx, half, bottom = x + w / 2.0, 0.64 * w, y + 1.05 * h
    d.ellipse([cx - half, crown, cx + half, bottom], fill=hair or SKIN)
    if hair:
        d.ellipse([cx - 0.9 * half, y - 0.03 * h, cx + 0.9 * half, bottom], fill=SKIN)
    return img


def face(box):
    x, y, w, h = box
    return {"box": list(box), "score": 0.9,
            "landmarks": [[x + 0.3 * w, y + 0.4 * h], [x + 0.7 * w, y + 0.4 * h],
                          [x + 0.5 * w, y + 0.6 * h], [x + 0.35 * w, y + 0.8 * h],
                          [x + 0.65 * w, y + 0.8 * h]]}


BALD_BOX = [350, 403, 380, 565]         # the face box starts 47 px under a bald crown ...
BALD_CROWN = 356                        # ... where the old guess (403 - 0.3 x 565) said y 233


class HeadTop(unittest.TestCase):
    def test_a_bald_head_is_measured_not_guessed(self):
        self.assertLess(cover.crown_y(BALD_BOX), 240)                     # the guess: above the tile
        for kw in ({}, {"slats": True}, {"glow": True}, {"slats": True, "glow": True}):
            photo = head_photo(BALD_BOX, BALD_CROWN, bg=(150, 96, 70), **kw)
            y = cover_frames.crown_from_image(photo, BALD_BOX)
            self.assertIsNotNone(y, kw)
            self.assertLess(abs(y - BALD_CROWN), 20, (kw, y))
            self.assertEqual(cover.head_top(photo, BALD_BOX), (y, "image"))

    def test_hair_is_part_of_the_head(self):
        box = [350, 520, 380, 520]
        for crown in (330, 400, 470):
            for kw in ({}, {"slats": True, "glow": True}):
                photo = head_photo(box, crown, hair=HAIR, **kw)
                y = cover_frames.crown_from_image(photo, box)
                self.assertIsNotNone(y, (crown, kw))
                self.assertLess(abs(y - crown), 22, (crown, kw, y))

    def test_small_and_off_centre_faces(self):
        box = [640, 700, 150, 190]                                       # a face in a wide shot
        y = cover_frames.crown_from_image(head_photo(box, 650, hair=HAIR), box)
        self.assertLess(abs(y - 650), 12, y)
        edge = [900, 500, 170, 220]                                      # no room on the right
        y = cover_frames.crown_from_image(head_photo(edge, 470), edge)
        self.assertLess(abs(y - 470), 12, y)

    def test_no_measurement_without_a_clear_picture(self):
        flat = Image.new("RGB", (1080, 1920), BG)
        self.assertIsNone(cover_frames.crown_from_image(flat, BALD_BOX))         # no head to see
        self.assertEqual(cover.head_top(flat, BALD_BOX), (cover.crown_y(BALD_BOX), "estimate"))
        same = head_photo(BALD_BOX, BALD_CROWN, bg=SKIN)                         # skin-coloured wall
        self.assertIsNone(cover_frames.crown_from_image(same, BALD_BOX))
        top = [350, 60, 380, 565]                                                # head out of frame
        self.assertIsNone(cover_frames.crown_from_image(head_photo(top, -200), top))
        wide = [140, 500, 800, 900]                                              # no side bands
        self.assertIsNone(cover_frames.crown_from_image(head_photo(wide, 300), wide))
        self.assertIsNone(cover_frames.crown_from_image(flat, [0, 0, 4, 4]))
        self.assertIsNone(cover_frames.crown_from_image(flat, None))

    def test_the_cover_reports_a_measured_crown_and_no_false_cut(self):
        # The old guess put the top of this head at y 233, above the 240 grid line, and every
        # cover of a bald head warned that the profile tile trims it. The head is at y 356.
        photo = head_photo(BALD_BOX, BALD_CROWN, bg=(150, 96, 70), slats=True)
        _img, info = cover.compose(photo, [face(BALD_BOX)], "IT *WORKS*", dict(cover.DEFAULT_COVER))
        self.assertEqual(info["crown_method"], "image")
        self.assertEqual(info["shift"], 0)
        self.assertLess(abs(info["crown_y"] - BALD_CROWN), 20, info["crown_y"])
        self.assertGreaterEqual(info["crown_y"], 240)
        self.assertFalse([w for w in info["warnings"] if "head" in w or "tile" in w], info["warnings"])

    def test_a_guessed_crown_is_not_a_finding(self):
        # Nothing to measure (a flat picture): the crown is the guess, labelled as one, and the
        # cover does not claim the tile trims a head it cannot see.
        flat = Image.new("RGB", (1080, 1920), (90, 90, 90))
        _img, info = cover.compose(flat, [face(BALD_BOX)], "IT *WORKS*", dict(cover.DEFAULT_COVER))
        self.assertEqual(info["crown_method"], "estimate")
        self.assertAlmostEqual(info["crown_y"], round(cover.crown_y(BALD_BOX) - info["shift"], 1))
        self.assertLess(info["crown_y"], 240)
        self.assertFalse([w for w in info["warnings"] if "head" in w or "tile" in w], info["warnings"])
        _img, none = cover.compose(flat, [], "IT *WORKS*", dict(cover.DEFAULT_COVER))
        self.assertIsNone(none["crown_y"])
        self.assertIsNone(none["crown_method"])

    def test_a_measured_crown_above_the_tile_is_still_reported(self):
        box = [350, 330, 380, 520]
        photo = head_photo(box, 150, hair=HAIR)                           # tall hair: y 150
        _img, info = cover.compose(photo, [face(box)], "IT *WORKS*", dict(cover.DEFAULT_COVER))
        self.assertEqual(info["crown_method"], "image")
        self.assertLess(info["crown_y"], 240)
        self.assertTrue(any("top of the head" in w for w in info["warnings"]), info["warnings"])

    def test_a_face_box_above_the_tile_always_warns(self):
        flat = Image.new("RGB", (1080, 1920), (90, 90, 90))
        _img, info = cover.compose(flat, [face(BALD_BOX)], "IT *WORKS*", dict(cover.DEFAULT_COVER),
                                   base_shift=220, shift_manual=True)
        self.assertEqual(info["crown_method"], "estimate")
        self.assertLess(info["face_box"][1], 240)
        self.assertTrue(any("face top at y 183" in w and "cuts the head" in w
                            for w in info["warnings"]), info["warnings"])

    def test_the_layout_keeps_a_measured_head_in_the_tile(self):
        # A long headline wants the photo pushed up. The guess called this a close-up (crown
        # already out) and allowed 80 px blind; the measured crown allows what the head has.
        photo = head_photo(BALD_BOX, BALD_CROWN, bg=(150, 96, 70))
        cfg = dict(cover.DEFAULT_COVER, headline_font="Anton-Regular.ttf")
        headline = "USE AI TO DO 80-90% OF YOUR JOB AND YOU'RE *MORE VALUABLE*"
        _img, info = cover.compose(photo, [face(BALD_BOX)], headline, cfg)
        self.assertEqual(info["crown_method"], "image")
        self.assertGreaterEqual(info["crown_y"], 240, info)              # the head stays inside
        self.assertFalse(any("covers a face" in w for w in info["warnings"]), info["warnings"])
        self.assertFalse([w for w in info["warnings"] if "top of the head" in w], info["warnings"])


# ----------------------------------------------------------------------------- 6. cover candidates

def picture(seed, size=(360, 640)):
    """A busy still: blocks of random grey with a bright square (a different one per seed)."""
    rng = np.random.default_rng(seed)
    small = rng.integers(30, 200, (size[1] // 40, size[0] // 40)).astype(np.uint8)
    a = np.kron(small, np.ones((40, 40), np.uint8))
    return np.repeat(a[:, :, None], 3, axis=2).copy()


def moved(base, dx):
    """`base` with a bright block that sits `dx` px to the right (a hand moving)."""
    a = base.copy()
    a[260:420, 60 + dx:160 + dx] = 240
    return a


class DissolveFrames(unittest.TestCase):
    def test_a_frame_between_two_pictures_is_a_blend(self):
        a, b = picture(1).astype(np.float32), picture(2).astype(np.float32)
        for alpha in (0.3, 0.5, 0.7):                # the frames 2 either side: 4 frames of 8
            before = (a * (1 - alpha + 0.25) + b * (alpha - 0.25)).astype(np.uint8)
            frame = (a * (1 - alpha) + b * alpha).astype(np.uint8)
            after = (a * (1 - alpha - 0.25) + b * (alpha + 0.25)).astype(np.uint8)
            m = cover.blend_measure(before, frame, after)
            self.assertGreater(m["change"], cover.BLEND_MIN, m)
            self.assertLess(m["linear"], 0.1, m)
            self.assertTrue(cover.is_blend(m), m)

    def test_motion_and_stillness_are_not_blends(self):
        base = picture(3)
        still = cover.blend_measure(base, base, base)
        self.assertFalse(cover.is_blend(still), still)
        m = cover.blend_measure(moved(base, 0), moved(base, 60), moved(base, 120))    # a hand moving
        self.assertGreater(m["change"], cover.BLEND_MIN, m)
        self.assertGreater(m["linear"], cover.BLEND_LINEAR, m)
        self.assertFalse(cover.is_blend(m), m)
        cut = cover.blend_measure(picture(4), picture(4), picture(5))                 # a hard cut next door
        self.assertTrue(cover.is_blend(cut), cut)
        self.assertFalse(cover.is_blend(None))

    def test_two_overlapping_faces_are_two_pictures(self):
        self.assertTrue(cover.faces_superimposed([[100, 150, 140, 180], [130, 170, 120, 150]]))
        self.assertFalse(cover.faces_superimposed([[40, 150, 120, 150], [200, 160, 120, 150]]))
        self.assertFalse(cover.faces_superimposed([[100, 150, 140, 180]]))
        self.assertFalse(cover.faces_superimposed([]))

    def test_dissolve_spans_come_from_the_shots_transitions(self):
        root = Path(tempfile.mkdtemp(prefix="hc-fix2-shots-"))
        try:
            from hookcutter import env
            from hookcutter.project import Project
            proj = Project(root)
            self.assertEqual(cover.dissolve_spans(proj, "c01"), [])              # no shots json
            path = Path(proj.shots_path("c01"))
            path.parent.mkdir(parents=True, exist_ok=True)
            env.write_json(path, {"clip": "c01", "shots": [], "transitions": [
                {"at_src": 12.0, "kind": "cut", "start_src": 12.0, "end_src": 12.0},
                {"at_src": 20.13, "kind": "dissolve", "start_src": 20.0, "end_src": 20.27},
                {"at_src": 31.0, "kind": "dissolve"}, "junk"]})
            spans = cover.dissolve_spans(proj, "c01")
            self.assertEqual(spans, [(20.0, 20.27), (31.0, 31.0)])
            self.assertTrue(cover.in_dissolve(spans, 20.1))
            self.assertFalse(cover.in_dissolve(spans, 12.0))                     # a cut is no ghost
            self.assertFalse(cover.in_dissolve(spans, 19.95))
            self.assertTrue(cover.in_dissolve(spans, 19.95, margin=0.06))
            self.assertFalse(cover.in_dissolve(spans, None))
        finally:
            shutil.rmtree(root, ignore_errors=True)


class CandidatesSkipGhosts(unittest.TestCase):
    """find_candidates and make_cover on a project whose frames are supplied by the test: the
    best-scoring frame is the middle of a cross dissolve (it used to be the automatic pick)."""

    FACE = {"box": [110.0, 150.0, 140.0, 170.0], "score": 0.9,
            "landmarks": [[150, 210], [210, 210], [180, 240], [155, 280], [205, 280]]}
    SMALL = {"box": [130.0, 190.0, 90.0, 110.0], "score": 0.9,
             "landmarks": [[155, 230], [195, 230], [175, 250], [158, 275], [192, 275]]}

    def setUp(self):
        import qa_media as qm
        from hookcutter import env
        from hookcutter import plan as plan_mod
        from hookcutter.project import Project
        self.env, self.plan_mod = env, plan_mod
        self.tmp = Path(tempfile.mkdtemp(prefix="hc-fix2-cands-"))
        (self.tmp / "src.mp4").write_bytes(b"")                       # never decoded: frames are mocked
        sentences = {"c01": "we sold it first and then we built it"}
        clips = {"c01": {"title": "c01", "pieces": [{"words": [0, 0]}],
                         "cover": {"headline": "SOLD IT *FIRST*"}}}
        P = qm.write_project(self.tmp / "P", clips, sentences, self.tmp / "src.mp4")
        self.project = Project(P)
        self.plan = qm.plan_for("c01", 0, env.read_json(P / "transcript.json"), [])
        # one 3 s segment at 25 fps, source 100.0 .. 103.0; a dissolve from frame 46 to 54
        self.plan["segments"] = [dict(self.plan["segments"][0], src=[100.0, 103.0], out=[0.0, 3.0],
                                      frames=75)]
        self.plan["joins"] = []
        self.a, self.b = picture(11), picture(12)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def frame_at(self, t):
        n = int(round(t * 25))
        alpha = min(1.0, max(0.0, (n - 46) / 8.0))
        return (self.a.astype(np.float32) * (1 - alpha) + self.b.astype(np.float32) * alpha).astype(np.uint8)

    def patches(self, faces=None):
        def base_frames(project, clip_id, clip, plan, fmt, times, size, letterbox_y=None,
                        style=None):
            out = {}
            for t in times:
                img = Image.fromarray(self.frame_at(float(t))).resize(size, Image.BILINEAR)
                out[float(t)] = np.asarray(img)
            return out, "segments"

        def detect(arr, min_score=0.6):
            if faces is not None:
                return faces(arr)
            # the ghost frame "has" the biggest, most frontal face: it would win on score
            mid = abs(float(np.asarray(arr).mean()) - (float(self.a.mean()) + float(self.b.mean())) / 2) < 0.6
            return [dict(self.FACE)] if mid else [dict(self.SMALL)]

        return (mock.patch.object(cover_frames, "base_frames", base_frames),
                mock.patch.object(cover_frames, "detect_faces", detect),
                mock.patch.object(self.plan_mod, "load_plan", lambda *a, **k: self.plan))

    def run_patched(self, fn, faces=None):
        p1, p2, p3 = self.patches(faces)
        with p1, p2, p3:
            return fn()

    def test_the_dissolve_frame_is_set_aside(self):
        from hookcutter import presets
        preset = presets.load("clean", self.project)
        data = self.run_patched(lambda: cover.find_candidates(self.project, "c01", {}, self.plan, preset))
        by_t = {c["at_out"]: c for c in data["candidates"]}
        self.assertEqual(by_t[2.0].get("skip"), "dissolve", by_t[2.0])          # frame 50: mid-dissolve
        self.assertLess(by_t[2.0]["blend"]["linear"], cover.BLEND_LINEAR)
        self.assertEqual(max(data["candidates"], key=lambda c: c["score"])["at_out"], 2.0)
        self.assertEqual([c["at_out"] for c in data["candidates"] if c.get("skip")], [2.0])
        self.assertNotIn(2.0, [c["at_out"] for c in cover.usable(data["candidates"])])
        self.assertEqual(len(cover.usable(data["candidates"])), len(data["candidates"]) - 1)

    def test_the_automatic_pick_avoids_it_and_a_manual_pick_is_warned(self):
        res = self.run_patched(lambda: cover.make_cover(self.project, "c01", "clean"))
        self.assertNotEqual(res["at_out"], 2.0)
        self.assertFalse(res["manual"])
        self.assertNotIn(2.0, [c["at_out"] for c in res["candidates"]])
        self.assertFalse(any("dissolve" in w for w in res["warnings"]), res["warnings"])
        self.assertIn("crown_method", res)
        forced = self.run_patched(lambda: cover.make_cover(self.project, "c01", "clean", at_out=2.0))
        self.assertTrue(forced["manual"])
        self.assertEqual(forced["at_out"], 2.0)
        self.assertTrue(any("inside a dissolve" in w for w in forced["warnings"]), forced["warnings"])
        off_grid = self.run_patched(lambda: cover.make_cover(self.project, "c01", "clean", at_out=1.96))
        self.assertTrue(any("inside a dissolve" in w for w in off_grid["warnings"]), off_grid["warnings"])
        clear = self.run_patched(lambda: cover.make_cover(self.project, "c01", "clean", at_out=1.0))
        self.assertFalse(any("dissolve" in w for w in clear["warnings"]), clear["warnings"])

    def test_a_dissolve_named_by_the_shot_analysis_is_set_aside_too(self):
        # No blend to measure here (every frame is the same still): the shots json says where
        # the source dissolves, on the source clock.
        from hookcutter import presets
        self.b = self.a
        path = Path(self.project.shots_path("c01"))
        path.parent.mkdir(parents=True, exist_ok=True)
        self.env.write_json(path, {"clip": "c01", "shots": [], "transitions": [
            {"at_src": 101.0, "kind": "dissolve", "start_src": 100.9, "end_src": 101.1},
            {"at_src": 102.5, "kind": "cut", "start_src": 102.5, "end_src": 102.5}]})
        preset = presets.load("clean", self.project)
        data = self.run_patched(lambda: cover.find_candidates(self.project, "c01", {}, self.plan, preset),
                                faces=lambda arr: [dict(self.FACE)])
        flagged = {c["at_out"]: c["skip"] for c in data["candidates"] if c.get("skip")}
        self.assertEqual(flagged, {1.0: "dissolve"})

    def test_superimposed_faces_are_set_aside(self):
        from hookcutter import presets
        self.b = self.a
        two = [dict(self.FACE), dict(self.FACE, box=[130.0, 170.0, 120.0, 150.0])]
        preset = presets.load("clean", self.project)
        data = self.run_patched(lambda: cover.find_candidates(self.project, "c01", {}, self.plan, preset),
                                faces=lambda arr: two)
        self.assertTrue(data["candidates"])
        self.assertEqual({c.get("skip") for c in data["candidates"]}, {"superimposed"})
        # nothing else to pick from: a cover is still made, and says so
        res = self.run_patched(lambda: cover.make_cover(self.project, "c01", "clean"),
                               faces=lambda arr: two)
        self.assertTrue(any("every candidate frame" in w for w in res["warnings"]), res["warnings"])


if __name__ == "__main__":
    unittest.main()
