"""Regression tests for the captions and covers fixes: the pill rule, paging and min_page,
merges, the hook card, emphasis, number ranges, zero-width characters, the SRT sidecar,
landscape presets, cover tint and contrast, cover speaker, cover framing and HDR.

Every test is synthetic (generated words, luma arrays, frames and tiny lavfi media)."""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "scripts"))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from hookcutter import captions, cover, cover_frames, env, presets, typeset  # noqa: E402
from hookcutter.project import Project  # noqa: E402

import test_captions as TC  # noqa: E402  (helpers: make_plan, build, style, load_preset)

CLEAN = TC.CLEAN
W, H = TC.W, TC.H


def timed(text, wps=2.5, start=0.3, gap=0.0):
    """Evenly timed words: (text, {"s", "e"}) rows for TC.make_plan."""
    out, t, d = [], start, 1.0 / wps
    for w in text.split():
        out.append((w, {"s": round(t, 3), "e": round(t + d * 0.85, 3)}))
        t += d + gap
    return out


def luma_for(plan, value=None, fn=None):
    """A [n, 480, 270] luma array covering the plan: a constant or fn(n) -> array."""
    n = int(plan["duration_out"] * captions.LUMA_FPS) + 2
    if fn is not None:
        return fn(n)
    return np.full((n, 480, 270), int(round(value * 255)), np.uint8)


def busy_luma(n, seed=3, block=3, hi=140):
    rng = np.random.default_rng(seed)
    small = rng.integers(0, hi, (480 // block + 1, 270 // block + 1)).astype(np.uint8)
    frame = np.kron(small, np.ones((block, block), np.uint8))[:480, :270]
    return np.repeat(frame[None], n, axis=0)


def emph(track):
    return [(t["text"], t["source"]) for p in track.pages for t in p["tokens"] if t["emph"]]


def codes(track):
    return [i["code"] for i in track.to_json()["issues"]]


# ----------------------------------------------------------------------------- 1. the pill

class PillRule(unittest.TestCase):
    def test_grey_070_bare_085_pilled(self):
        plan = TC.make_plan(timed("we sold the product first", wps=2.0))
        bare = TC.build(plan, luma=luma_for(plan, 0.70))
        self.assertFalse(any(p["backing"] for p in bare.pages), bare.pages)
        lit = TC.build(plan, luma=luma_for(plan, 0.85))
        self.assertTrue(all(p["backing"] for p in lit.pages))
        self.assertEqual({p["backing_reason"] for p in lit.pages}, {"contrast"})

    def test_shadow_strength_of_the_clean_shadow(self):
        font = typeset.load_font(CLEAN["caption"]["font"], CLEAN["caption"]["size"])
        s = typeset.shadow_strength(font, CLEAN["caption"]["shadow"])
        self.assertGreater(s, 0.24)
        self.assertLess(s, 0.34)
        self.assertEqual(typeset.shadow_strength(font, []), 0.0)

    def test_busy_background_gets_a_texture_pill(self):
        plan = TC.make_plan(timed("we sold the product first", wps=2.0))
        luma = luma_for(plan, fn=busy_luma)
        track = TC.build(plan, luma=luma)
        page = track.pages[0]
        self.assertGreater(page["edge_density"], captions.TEXTURE_SHARE)
        self.assertTrue(page["backing"])
        self.assertEqual(page["backing_reason"], "texture")
        # same brightness without the texture: no pill
        flat = TC.build(plan, luma=luma_for(plan, float(np.percentile(luma[0], 90)) / 255.0))
        self.assertFalse(flat.pages[0]["backing"])

    def test_narrow_bright_strip_does_not_blink_one_page(self):
        # A 50 px bright strip in the centre of a dark frame: short centred pages see it in
        # their box, long ones do not. The decision is per shot, so no page is pilled alone.
        words = timed("Done. The product came first, then the build. Right. Then we shipped "
                      "the whole thing. Yes.", wps=2.2)
        plan = TC.make_plan(words)

        def strip(n):
            a = np.full((n, 480, 270), 70, np.uint8)
            a[:, :, 129:141] = 235
            return a

        track = TC.build(plan, luma=luma_for(plan, fn=strip))
        backing = [p["backing"] for p in track.pages]
        for i, b in enumerate(backing):
            near = [backing[j] for j in (i - 1, i + 1) if 0 <= j < len(backing)]
            self.assertFalse(b and near and not any(near), backing)
        self.assertEqual(len(set(backing)), 1, backing)

    def test_shot_bounds_from_marks_and_speaker_follow(self):
        # Joins inside one framing (piece at 2.0, dip at 6.0) are jump cuts on the same
        # background: only the framing change (the 113.5 mark, at out 4.5) bounds a shot.
        plan = {"joins": [{"k": 1, "out": 2.0, "kind": "piece"}, {"k": 2, "out": 4.0, "kind": "omit"},
                          {"k": 3, "out": 6.0, "kind": "dip"}, {"k": 4, "out": 7.0, "kind": "tighten"}],
                "segments": [{"src": [100.0, 102.0], "out": [0.0, 2.0]},
                             {"src": [110.0, 112.0], "out": [2.0, 4.0]},
                             {"src": [113.0, 115.0], "out": [4.0, 6.0]},
                             {"src": [120.0, 122.0], "out": [6.0, 8.0]}],
                "framing": [{"at_src": 100.0, "mode": "track", "x": 0.3},
                            {"at_src": 113.5, "mode": "track", "x": 0.7}]}
        self.assertEqual(captions._shot_bounds(plan), [4.5])
        plan["framing"].append({"at_src": 120.0, "mode": "track", "x": 0.3})
        self.assertEqual(captions._shot_bounds(plan), [4.5, 6.0])

    def test_one_marginal_page_does_not_pill_the_shot(self):
        """A shot is pilled when a quarter of its pages need it or one is severe; a single
        borderline page among many stays bare (it used to pill every page of its piece)."""
        words = timed("we built the product for a whole year and then it did not sell so we "
                      "went back and sold it first to forty strangers in ninety days", wps=2.2)
        plan = TC.make_plan(words)
        base = TC.build(plan, luma=None)
        n = int(plan["duration_out"] * captions.LUMA_FPS) + 2

        def one_bright(level):
            def fn(_n):
                a = np.full((n, 480, 270), 40, np.uint8)
                p = base.pages[3]
                a[int(p["start_out"] * captions.LUMA_FPS):int(p["end_out"] * captions.LUMA_FPS) + 1] = level
                return a
            return fn

        self.assertGreaterEqual(len(base.pages), 5)
        marginal = TC.build(plan, luma=luma_for(plan, fn=one_bright(214)))    # contrast ~2.5
        self.assertTrue(captions.SEVERE_CONTRAST <= marginal.pages[3]["contrast"] < 3.0,
                        marginal.pages[3]["contrast"])
        self.assertFalse(any(p["backing"] for p in marginal.pages),
                         [(p["text"], p["contrast"]) for p in marginal.pages])
        severe = TC.build(plan, luma=luma_for(plan, fn=one_bright(240)))      # contrast < 2.5
        self.assertTrue(all(p["backing"] for p in severe.pages))

    def test_hook_card_follows_the_first_shot(self):
        def plan():
            words = [("sell", {"s": 2.0, "e": 2.3}), ("it", {"s": 2.3, "e": 2.5}),
                     ("first.", {"s": 2.5, "e": 2.9})]
            return TC.make_plan(words, hook={"lines": ["SELL *FIRST*"], "dur": 1.5})

        def late_bright(n):                     # dark under the card, bright under the words
            a = np.full((n, 480, 270), 50, np.uint8)
            a[int(1.6 * captions.LUMA_FPS):] = 235
            return a

        track = TC.build(plan(), luma=luma_for(plan(), fn=late_bright))
        self.assertTrue(all(p["backing"] for p in track.pages))
        self.assertTrue(track.hook["backing"])          # one shot: the card matches its pages
        dark = TC.build(plan(), luma=luma_for(plan(), 0.2))
        self.assertFalse(dark.hook["backing"])

    def test_unspoken_alpha_raised(self):
        self.assertAlmostEqual(CLEAN["caption"]["unspoken_alpha"], 0.70)


# ----------------------------------------------------------------------------- 2-5. paging

class HookTrim(unittest.TestCase):
    def test_first_page_after_the_card_is_a_fresh_phrase(self):
        words = [("him,", {"s": 1.0, "e": 1.42}), ("let's", {"s": 1.44, "e": 1.6}),
                 ("sell", {"s": 1.6, "e": 1.8}), ("it.", {"s": 1.8, "e": 2.1})]
        plan = TC.make_plan(words, hook={"lines": ["SELL *IT*"], "dur": 1.5})
        track = TC.build(plan)
        self.assertEqual(TC.page_texts(track)[0], "LET'S SELL IT")
        self.assertIn("him,", "\n".join(c[2] for c in track.srt_cues()))     # still in the SRT


class ClauseRule(unittest.TestCase):
    def test_clause_tail_never_leads_the_next_page(self):
        plan = TC.make_plan(timed("you've never met, not your friends,", wps=4.0))
        self.assertEqual(TC.page_texts(TC.build(plan)), ["YOU'VE NEVER MET", "NOT YOUR FRIENDS"])
        plan = TC.make_plan(timed("reach out to 40 people you've never met, not your friends, "
                                  "not your family,", wps=4.0))
        pages = TC.page_texts(TC.build(plan))
        self.assertFalse(any(p.startswith("MET") for p in pages), pages)
        self.assertIn("NOT YOUR FRIENDS", pages)

    def test_one_word_whole_clause_may_lead(self):
        plan = TC.make_plan(timed("Well, they get them into it.", wps=3.0))
        self.assertTrue(TC.page_texts(TC.build(plan))[0].startswith("WELL THEY GET"))

    def test_clauses_ok_helper(self):
        toks = captions.display_tokens(TC.make_plan(["met,", "not", "your"]), {}, CLEAN["caption"])
        self.assertFalse(captions._clauses_ok(toks, False, 2))     # tail of a clause leads
        self.assertTrue(captions._clauses_ok(toks, True, 2))       # "WELL, THEY ..." shape


class MinPage(unittest.TestCase):
    TEXT = ("you've never met, not your friends, not your family, and sell them on your "
            "product, technology or service. People say, I can't do that. It's not done yet.")

    def durations(self, track):
        return [p["end_out"] - p["start_out"] for p in track.pages]

    def test_fast_speech_has_fewer_flash_pages(self):
        plan = TC.make_plan(timed(self.TEXT, wps=5.0))
        on = self.durations(TC.build(plan))
        off = self.durations(TC.build(plan, TC.style(layout={"min_page": 0})))
        self.assertEqual(sum(d < 0.5 for d in on), 0)
        self.assertLess(sum(d < 0.8 for d in on), sum(d < 0.8 for d in off))

    def test_hard_danglers_never_end_a_page_or_line(self):
        hard = {"A", "AN", "THE", "MY", "YOUR", "HIS", "HER", "OUR", "THEIR", "ITS", "OF", "TO"}
        texts = ("he spent a whole year building the product and then he sold it",
                 "he spent a whole year building the product and then he sold it to all of their "
                 "customers")
        for k, text in enumerate(texts):
            for gap in (0.02, 0.06, 0.2):
                plan = TC.make_plan(text.split(), gap=gap)
                for name in ("clean", "pop"):
                    track = TC.build(plan, TC.load_preset(name))
                    for p in track.pages[:-1]:
                        self.assertNotIn(p["lines"][-1][-1], hard, TC.page_texts(track))
                    if k:
                        continue        # "OF THEIR CUSTOMERS" cannot avoid a line end on one
                    for p in track.pages:
                        for line in p["lines"][:-1]:
                            self.assertNotIn(line[-1], hard, p["lines"])

    def test_short_gap_is_held_except_at_a_seam(self):
        def words():
            return [("we", {"s": 0.5, "e": 0.7}), ("shipped.", {"s": 0.7, "e": 1.1}),
                    ("then", {"s": 2.0, "e": 2.3, "seg": 1}), ("sold.", {"s": 2.3, "e": 2.8, "seg": 1})]
        held = TC.build(TC.make_plan(words()))
        self.assertAlmostEqual(held.pages[0]["end_out"], held.pages[1]["start_out"], places=3)
        cut = TC.build(TC.make_plan(words(), joins=[{"k": 1, "out": 1.9, "kind": "piece"}]))
        self.assertLess(cut.pages[0]["end_out"], cut.pages[1]["start_out"] - 0.1)

    def test_one_word_hesitation_joins_the_next_page(self):
        words = [("I", {"s": 0.5, "e": 0.7}), ("had", {"s": 1.3, "e": 1.5}),
                 ("a", {"s": 1.5, "e": 1.6}), ("client.", {"s": 1.6, "e": 2.0})]
        self.assertEqual(TC.page_texts(TC.build(TC.make_plan(words))), ["I HAD A CLIENT"])

    def test_page_left_under_min_page_is_an_issue(self):
        # "STOP GO" starts at 0 and is boxed in by a piece join: nothing can lengthen it.
        words = [("Stop.", {"s": 0.05, "e": 0.2}), ("Go.", {"s": 0.22, "e": 0.4}),
                 ("Now", {"s": 0.42, "e": 0.6, "seg": 1}), ("then.", {"s": 0.6, "e": 1.2, "seg": 1})]
        track = TC.build(TC.make_plan(words, joins=[{"k": 1, "out": 0.41, "kind": "piece"}]))
        issues = [i for i in track.to_json()["issues"] if i["code"] == "min_page"]
        self.assertTrue(issues)
        for i in issues:
            self.assertEqual(set(i), {"code", "msg", "at_out", "page"})
            self.assertIsInstance(i["page"], int)


class Merges(unittest.TestCase):
    def test_pause_does_not_hide_a_sentence_end(self):
        for gap in (0.30, 0.60):
            t0 = 0.7 + gap
            words = [("It", {"s": 0.2, "e": 0.4}), ("works.", {"s": 0.4, "e": 0.7}),
                     ("People", {"s": t0, "e": t0 + 0.15}), ("say,", {"s": t0 + 0.15, "e": t0 + 0.3}),
                     ("I", {"s": t0 + 0.32, "e": t0 + 0.4}), ("can't", {"s": t0 + 0.4, "e": t0 + 0.55}),
                     ("do", {"s": t0 + 0.55, "e": t0 + 0.62}), ("that.", {"s": t0 + 0.62, "e": t0 + 0.8})]
            pages = TC.page_texts(TC.build(TC.make_plan(words)))
            self.assertEqual(pages[0], "IT WORKS", pages)

    def test_one_word_sentence_is_not_glued_to_half_a_sentence(self):
        words = timed("Done. The product came first, then the build. Right. Then we shipped "
                      "the whole thing. Yes.", wps=2.2)
        pages = TC.page_texts(TC.build(TC.make_plan(words)))
        self.assertIn("DONE", pages)
        self.assertFalse(any(p.startswith("DONE ") or p.startswith("RIGHT ") for p in pages), pages)


# ----------------------------------------------------------------------------- 6. landscape

class LandscapeStyle(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="hc-land-"))
        (self.tmp / "presets").mkdir()
        house = {"name": "house", "extends": "clean",
                 "caption": {"size": 96, "layout": {"max_words": 3, "max_lines": 1, "y": 1392}},
                 "landscape": {"caption": {"size": 64,
                                           "layout": {"y": 960, "max_width": 1600, "max_lines": 1}}}}
        (self.tmp / "presets" / "house.json").write_text(json.dumps(house))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def resolved(self, name, raw=False):
        p = presets.load(name, self.tmp)
        if raw:
            return captions.resolve_style(p, "landscape")
        return captions.resolve_style(presets.for_format(p, "landscape"), "landscape")

    def test_landscape_overrides_survive(self):
        for raw in (False, True):
            self.assertEqual(self.resolved("minimal", raw)["caption"]["size"], 46)
            self.assertEqual(self.resolved("clean", raw)["caption"]["size"], 54)
            house = self.resolved("house", raw)["caption"]
            self.assertEqual((house["size"], house["layout"]["max_lines"], house["layout"]["y"]),
                             (64, 1, 960))

    def test_landscape_track_uses_the_preset_size(self):
        plan = TC.make_plan(["one", "two", "three"])
        pre = presets.for_format(presets.load("minimal", self.tmp), "landscape")
        track = TC.build(plan, pre, fmt="landscape")
        self.assertEqual(track.pages[0]["size"], 46)


# ----------------------------------------------------------------------------- 7. auto emphasis

class AutoEmphasis(unittest.TestCase):
    def setUp(self):
        self.pre = TC.style(emphasis={"sources": ["hook", "auto"], "per_page": 1})

    def test_only_numbers_currency_and_names(self):
        text = ("honestly the strategy worked. we started sweating bullets. Priya paid $40K. "
                "then it took us about 90 days. we met Dana later on.")
        track = TC.build(TC.make_plan(timed(text, wps=1.0)), self.pre)
        got = [t for t, _ in emph(track)]
        # "$40K" outranks "PRIYA" on its page; no long ordinary word (STRATEGY, SWEATING) is
        # accented, and "DANA" is a mid-sentence name 5 s after "90".
        self.assertEqual(got, ["$40K", "90", "DANA"])

    def test_gap_between_auto_accents(self):
        text = "we paid $10 then $20 then $30 then $40 then $50."
        track = TC.build(TC.make_plan(timed(text, wps=2.0)), self.pre)
        times = [t["s_out"] for p in track.pages for t in p["tokens"] if t["source"] == "auto"]
        self.assertTrue(times)
        for a, b in zip(times, times[1:]):
            self.assertGreaterEqual(b - a, captions.AUTO_GAP - 1e-6)

    def test_no_auto_next_to_a_hook_accent(self):
        text = "the deal closed. it paid $5K. that was all. then $9K came. done now."
        plan = TC.make_plan(timed(text, wps=1.0), hook={"lines": ["*DEAL* CLOSED"], "dur": 0.2})
        track = TC.build(plan, self.pre)
        hooked = {i for i, p in enumerate(track.pages) if any(t["source"] == "hook" for t in p["tokens"])}
        auto = {i for i, p in enumerate(track.pages) if any(t["source"] == "auto" for t in p["tokens"])}
        self.assertTrue(hooked)
        self.assertFalse({i for h in hooked for i in (h - 1, h, h + 1)} & auto, TC.page_texts(track))


# ----------------------------------------------------------------------------- 8. numbers

class NumberRanges(unittest.TestCase):
    def test_split_range_glues(self):
        toks = captions.display_tokens(TC.make_plan(["about", "80", "-90", "%", "of", "it"]), {},
                                       CLEAN["caption"])
        self.assertEqual([t.text for t in toks], ["ABOUT", "80-90%", "OF", "IT"])

    def test_number_list_stays_on_one_page(self):
        for gap in (0.02, 0.1):
            plan = TC.make_plan("it will replace 80, 90% of what you do every day".split(), gap=gap)
            pages = TC.page_texts(TC.build(plan))
            self.assertTrue(any("80, 90%" in p for p in pages), pages)
            self.assertFalse(any(p.endswith("80,") for p in pages), pages)


# ----------------------------------------------------------------------------- 9. zero-width

class ZeroWidth(unittest.TestCase):
    def test_caption_word_draws_like_the_clean_word(self):
        def plan(first):
            return TC.make_plan([(first, {"s": 0.5, "e": 0.8}), ("first", {"s": 0.85, "e": 1.2})])
        a = TC.build(plan("sell\u200b"))
        b = TC.build(plan("sell"))
        n = a.pages[0]["frames"][1] - 1
        self.assertEqual(a.frame(n), b.frame(n))
        self.assertIn("glyph_stripped", codes(a))
        self.assertNotIn("glyph_stripped", codes(b))

    def test_hook_card_draws_like_the_clean_line(self):
        def plan(line):
            return TC.make_plan([("go", {"s": 2.0, "e": 2.3})], hook={"lines": [line], "dur": 1.0})
        a = TC.build(plan("SELL\u200b *FIRST*"))
        b = TC.build(plan("SELL *FIRST*"))
        self.assertEqual(a.frame(0), b.frame(0))
        self.assertIn("glyph_stripped", codes(a))

    def test_other_format_characters_are_reported(self):
        path = typeset.resolve_font("ArchivoBlack-Regular.ttf")[0]
        self.assertEqual(typeset.missing_glyphs(path, "A\u200eB"), ["\u200e"])
        with self.assertRaises(env.HCError):
            TC.build(TC.make_plan(["left\u200e", "mark"]))

    def test_cover_headline_is_stripped(self):
        self.assertEqual(cover.parse_runs("SELL\u200b *FIRST\ufeff*"), [("SELL", False), ("FIRST", True)])
        # punctuation touching an accent run stays on its word (no "VALUE ," with a space)
        self.assertEqual(cover.parse_runs("*PRICE AND VALUE*, NOT IT"),
                         [("PRICE", True), ("AND", True), ("VALUE,", True), ("NOT", False), ("IT", False)])
        self.assertEqual(cover.parse_runs('"*WORD*" X'), [('"WORD"', True), ("X", False)])
        img, info = cover.compose(Image.new("RGB", (1080, 1920), (40, 40, 40)), [],
                                  "SELL\u2060 *FIRST*", dict(cover.DEFAULT_COVER))
        self.assertTrue(any("invisible" in w for w in info["warnings"]))


# ----------------------------------------------------------------------------- 10. SRT

class Sidecar(unittest.TestCase):
    def test_srt_has_every_spoken_word_and_no_headline(self):
        text = ("I need you to reach out to 40 people you've never met, not your friends, not your "
                "family, and sell them on your product. People say, I can't do that.")
        plan = TC.make_plan(timed(text, wps=3.5, start=0.3),
                            hook={"lines": ["40 STRANGERS.", "NOT YOUR *FRIENDS*."], "dur": 1.95})
        track = TC.build(plan)
        cues = track.srt_cues()
        self.assertTrue(cues[0][2].replace("\n", " ").startswith("I need you to reach out to"), cues[0])
        self.assertAlmostEqual(cues[0][0], 0.3, places=3)
        joined = " ".join(c[2].replace("\n", " ") for c in cues)
        self.assertEqual(joined.split(), text.split())
        for k, (a, b, body) in enumerate(cues):
            lines = body.split("\n")
            self.assertLessEqual(len(lines), captions.SRT_LINES)
            self.assertTrue(all(len(x) <= captions.SRT_LINE for x in lines), lines)
            self.assertLessEqual(b - a, captions.SRT_MAX_DUR + captions.SRT_TAIL + 1e-6)
            if k + 1 < len(cues):
                self.assertLessEqual(b, cues[k + 1][0] + 1e-9)
        self.assertNotIn("STRANGERS", joined)


# ----------------------------------------------------------------------------- 12. hook card

class HookCardSize(unittest.TestCase):
    def test_card_rebroken_to_reach_the_ratio(self):
        plan = TC.make_plan(["go"], start=3.0, hook={"lines": ["SELL IT,", "THEN *MAKE* IT"], "dur": 1.5})
        track = TC.build(plan)
        self.assertGreaterEqual(track.hook["size"], 1.3 * CLEAN["caption"]["size"])
        self.assertEqual(track.hook["caption_size"], CLEAN["caption"]["size"])
        self.assertNotIn("hook_small", codes(track))

    def test_card_that_cannot_reach_the_ratio_is_an_issue(self):
        plan = TC.make_plan(["go"], start=3.0, hook={"lines": ["INTERNATIONALIZATION"], "dur": 1.5})
        track = TC.build(plan)
        small = [i for i in track.to_json()["issues"] if i["code"] == "hook_small"]
        self.assertEqual(len(small), 1)
        self.assertIn("characters", small[0]["msg"])

    def test_hook_fit(self):
        clip = {"hook_card": {"lines": ["SELL IT,", "THEN *MAKE* IT"]}}
        fit = captions.hook_fit(clip, CLEAN, "vertical")
        self.assertEqual(set(fit), {"hook_px", "caption_px", "ratio", "lines", "max_chars"})
        self.assertEqual(fit["caption_px"], 78)
        self.assertGreaterEqual(fit["ratio"], 1.3)
        self.assertIsNone(captions.hook_fit({}, CLEAN, "vertical"))
        long = captions.hook_fit({"hook_card": {"lines": ["INTERNATIONALIZATION"]}}, CLEAN, "vertical")
        self.assertLess(long["ratio"], 1.3)
        self.assertGreater(long["max_chars"], 3)

    def test_card_ink_stays_above_the_bottom_ui_zone(self):
        """A 3-line card centred low (caption_y) is clamped by its ink, not just its baselines:
        round letters and commas dip below the baseline and used to cross the safe area."""
        style = captions.resolve_style(TC.load_preset("pop"), "vertical")
        warns = []
        for cy in (1300, 1343, 1400):
            hk = captions._hook_layout({"lines": ["REPLACE", "YOUR OWN", "*ROLE* FIRST,"]}, style, 724.0,
                                       540.0, float(cy), (240.0, 1500.0), None, warns, 80.0)
            self.assertLessEqual(hk["box"][1] + hk["box"][3], 1500, (cy, hk["box"]))

    def test_shipped_hook_max_size(self):
        self.assertEqual(TC.load_preset("clean")["hook_card"]["max_size"], 120)
        self.assertEqual(TC.load_preset("pop")["hook_card"]["max_size"], 120)


# ----------------------------------------------------------------------------- 17. polish

class Polish(unittest.TestCase):
    def test_build_pill_grows_word_by_word_and_never_ahead(self):
        # The pill of a build reveal reaches exactly as far as the newest word on screen: no
        # dark bar ahead of the speech, and (since the second fix round) no word outside it:
        # it covers a word from the first frame that word is drawn. See also
        # test_fix2_fb.BuildRevealPill.
        pre = TC.load_preset("pop")
        pre = captions._deep_merge(pre, {"caption": {"backing": {"mode": "always"},
                                                     "emphasis": {"sources": []}}})
        plan = TC.make_plan([("sell", {"s": 1.0, "e": 1.3}), ("first", {"s": 1.5, "e": 1.9})])
        track = TC.build(plan, pre)
        page = track._pages[0]
        x, y, w, h = page.token_pills[0]
        x1 = page.token_pills[1]
        row_y = y + h // 2 - track.band_y

        def right_edge(n):
            a = np.frombuffer(track.frame(n), np.uint8).reshape(track.band_h, W, 4)[..., 3]
            cols = np.nonzero(a[row_y] >= 200)[0]
            return int(cols.max()) if len(cols) else None

        seen_one = seen_two = False
        for n in range(page.start_f, page.end_f):
            st = track.frame_key(n)
            if st[0] != "page":
                continue
            one, two = st[2][0][0] > 0, st[2][1][0] > 0
            if one and not two:
                seen_one = True
                self.assertLessEqual(right_edge(n), x + w, n)          # never ahead of the speech
                self.assertGreaterEqual(right_edge(n), x + w - 3, n)   # and whole under word one
            if two:
                seen_two = True
                self.assertGreaterEqual(right_edge(n), x1[0] + x1[2] - 3, n)
        self.assertTrue(seen_one and seen_two)

    def test_build_reveal_anchors_the_first_line(self):
        pre = TC.load_preset("pop")
        plan = TC.make_plan(timed("so we went. we sold the whole product to everyone.", wps=2.0))
        track = TC.build(plan, pre)
        firsts = {round(p.pos[0][1]) for p in track._pages}
        lines = {len(p.lines) for p in track._pages}
        self.assertEqual(lines, {1, 2})
        self.assertEqual(len(firsts), 1, firsts)

    def test_caption_y_is_clamped_into_the_safe_zone(self):
        plan = TC.make_plan(["one", "two", "three"])
        track = TC.build(plan, clip={"caption_y": 1880})
        self.assertIn("caption_y_clamped", codes(track))
        for p in track.pages:
            self.assertLessEqual(p["box"][1] + p["box"][3], H - 420)
        land = TC.build(plan, clip={"caption_y": 300}, fmt="landscape")
        self.assertIn("caption_y_clamped", codes(land))
        self.assertGreater(land.pages[0]["box"][1], 700)

    def test_accent_matching_folds_diacritics_and_halves(self):
        self.assertEqual(captions.match_phrases(["JOSE"], ["call", "Jos\u00e9", "now"]), [[[1]]])
        self.assertEqual(captions.match_phrases(["2.5X"], ["grew", "two", "and", "a", "half", "times"]),
                         [[[1, 2, 3, 4, 5]]])
        self.assertEqual(captions.match_phrases(["SELL FIRST"], "sell it to them first".split()), [[]])
        self.assertEqual(captions.match_phrases(["SELL FIRST"], "sell it first".split()), [[[0, 2]]])

    def test_repeated_word_keeps_its_comma(self):
        toks = captions.display_tokens(TC.make_plan(["with,", "with", "the", "team,", "done"]), {},
                                       CLEAN["caption"])
        self.assertEqual([t.text for t in toks], ["WITH,", "WITH", "THE", "TEAM", "DONE"])

    def test_issues_for_shrunk_overflow_and_clamped_width(self):
        shrunk = TC.build(TC.make_plan(["internationalization"]))
        self.assertIn("word_shrunk", codes(shrunk))
        over = TC.build(TC.make_plan(["pneumonoultramicroscopicsilicovolcanoconiosis"]))
        self.assertIn("overflow", codes(over))
        wide = TC.build(TC.make_plan(["one", "two"]), TC.style(layout={"max_width": 1000}))
        self.assertIn("width_clamped", codes(wide))
        self.assertNotIn("width_clamped", codes(TC.build(TC.make_plan(["one", "two"]))))


# ----------------------------------------------------------------------------- covers

class CoverTint(unittest.TestCase):
    def test_headline_reads_on_a_white_frame(self):
        white = Image.new("RGB", (1080, 1920), (245, 245, 245))
        for headline in ("IT *WORKS*", "HE SOLD IT FIRST AND MADE *WAY MORE MONEY*"):
            for accent in ("#FFD84D", "#FF5A36"):
                cfg = dict(cover.DEFAULT_COVER, accent=accent)
                img, info = cover.compose(white, [], headline, cfg)
                self.assertTrue(info["line_contrast"])
                self.assertGreaterEqual(min(info["line_contrast"]), cover.LINE_CONTRAST - 0.05, info)
                self.assertFalse(any("contrast" in w for w in info["warnings"]))

    def test_tint_is_anchored_to_the_headline(self):
        a = cover._tint_alpha(1400.0, cover.DEFAULT_COVER)[:, 0]
        ta = cover.DEFAULT_COVER["tint_alpha"]
        self.assertEqual(float(a[1400 - cover.TINT_LEAD - 1]), 0.0)
        self.assertAlmostEqual(float(a[1400]), 0.8 * ta, places=3)
        self.assertAlmostEqual(float(a[1400 + cover.TINT_FULL + 5]), ta, places=3)

    def test_low_contrast_line_is_reported(self):
        cfg = dict(cover.DEFAULT_COVER, tint="#F0F0F0", tint_alpha=0.9)
        white = Image.new("RGB", (1080, 1920), (245, 245, 245))
        _img, info = cover.compose(white, [], "IT *WORKS*", cfg)
        self.assertTrue(any(w.startswith("cover headline contrast") for w in info["warnings"]))


class CoverCrownAndShift(unittest.TestCase):
    def test_crown_not_face_top_decides_the_head_cut(self):
        photo = Image.new("RGB", (1080, 1920), (90, 90, 90))
        cfg = dict(cover.DEFAULT_COVER, headline_font="Anton-Regular.ttf")
        headline = "HE SOLD IT FIRST AND MADE *WAY MORE MONEY* FAST"
        # The crown (420 - 0.3 x 600 = 240) sits on the band top. The 168 px headline needs a
        # 40 px shift, which keeps the face-box top in the band but cuts the scalp: the
        # headline steps down instead and the photo stays put.
        faces = [{"box": [240, 420, 600, 600], "score": 0.9, "landmarks": None}]
        self.assertAlmostEqual(cover.crown_y(faces[0]["box"]), 240.0)
        _img, info = cover.compose(photo, faces, headline, cfg)
        self.assertEqual(info["shift"], 0, info)
        self.assertEqual(info["crown_y"], 240.0)
        self.assertLess(info["size"], 168)
        self.assertFalse(any("top of the head" in w for w in info["warnings"]), info["warnings"])
        # A close-up whose crown is outside the band anyway protects the forehead instead.
        # On this flat picture the crown is only the guess from the face box (labelled
        # "estimate"), so nothing claims the tile trims the head: a warning needs a crown
        # measured on the picture (test_fix2_fb.HeadTop) or a face box above the band.
        close = [{"box": [186, 345, 654, 928], "score": 0.9, "landmarks": None}]
        _img, info = cover.compose(photo, close, headline, cfg)
        self.assertGreaterEqual(info["face_box"][1], 240, info)
        self.assertFalse(any("covers a face" in w for w in info["warnings"]), info["warnings"])
        self.assertEqual(info["crown_method"], "estimate")
        self.assertLess(info["crown_y"], 240)
        self.assertFalse(any("top of the head" in w for w in info["warnings"]), info["warnings"])

    def test_close_up_shift_is_limited(self):
        """A close-up whose crown starts just above the band: a long 4-line headline at the top
        size would push the photo up far (the tile cutting to the forehead). The headline
        shrinks until the shift is at most CLOSEUP_SHIFT_MAX, or to the size that shifts least."""
        photo = Image.new("RGB", (1080, 1920), (90, 90, 90))
        cfg = dict(cover.DEFAULT_COVER, headline_font="Anton-Regular.ttf")
        headline = "USE AI TO DO 80-90% OF YOUR JOB AND YOU'RE *MORE VALUABLE*"
        faces = [{"box": [230, 440, 620, 700], "score": 0.9, "landmarks": None}]
        self.assertLess(cover.crown_y(faces[0]["box"]), 240)
        _img, info = cover.compose(photo, faces, headline, cfg)
        self.assertLessEqual(info["shift"], cover.CLOSEUP_SHIFT_MAX, info)
        self.assertGreaterEqual(info["face_box"][1], 240, info)
        self.assertFalse(any("covers a face" in w for w in info["warnings"]), info["warnings"])

    def test_explicit_shift_zero_is_manual(self):
        photo = Image.new("RGB", (1080, 1920), (90, 90, 90))
        faces = [{"box": [340, 1200, 400, 420], "score": 0.9, "landmarks": None}]
        _img, auto = cover.compose(photo, faces, "IT *WORKS*", dict(cover.DEFAULT_COVER))
        self.assertGreater(auto["shift"], 0)
        _img, man = cover.compose(photo, faces, "IT *WORKS*", dict(cover.DEFAULT_COVER),
                                  base_shift=0, shift_manual=True)
        self.assertEqual(man["shift"], 0)
        self.assertTrue(man["shift_manual"])
        self.assertTrue(any("covers a face" in w for w in man["warnings"]))


class CoverExplicitShift(unittest.TestCase):
    def test_explicit_shift_is_read_from_the_file(self):
        root = Path(tempfile.mkdtemp(prefix="hc-shift-"))
        try:
            env.write_json(root / "clips.json", {"version": 1, "clips": {
                "a": {"cover": {"shift": 0}}, "b": {"cover": {"headline": "X"}}, "c": {}}})
            proj = Project(root)
            self.assertTrue(cover._explicit_shift(proj, "a"))
            self.assertFalse(cover._explicit_shift(proj, "b"))
            self.assertFalse(cover._explicit_shift(proj, "c"))
        finally:
            shutil.rmtree(root, ignore_errors=True)


class CoverSpeaker(unittest.TestCase):
    PLAN = {"words": [
        {"i": 0, "w": "so", "s_out": 0.0, "e_out": 0.4, "sp": "Sam Rivera"},
        {"i": 1, "w": "tell", "s_out": 0.5, "e_out": 2.0, "sp": "Sam Rivera"},
        {"i": 2, "w": "we", "s_out": 2.1, "e_out": 2.5, "sp": "Alex Kim"},
        {"i": 3, "w": "sold", "s_out": 2.5, "e_out": 4.9, "sp": "Alex Kim"},
        {"i": 4, "w": "right", "s_out": 5.0, "e_out": 6.0, "sp": "Sam Rivera"}]}

    def test_names_match_by_prefix(self):
        self.assertTrue(cover.same_speaker("Alex", "Alex Kim"))
        self.assertTrue(cover.same_speaker("alex kim", "Alex Kim"))
        self.assertFalse(cover.same_speaker("Alexander", "Alex Kim"))
        self.assertFalse(cover.same_speaker("", "Alex Kim"))

    def test_candidates_outside_the_owner_are_excluded(self):
        cands = [{"at_out": t, "score": s} for t, s in ((1.0, 0.9), (3.0, 0.5), (5.5, 0.95))]
        pool, note = cover.candidate_pool(cands, self.PLAN, "Alex")
        self.assertEqual([c["at_out"] for c in pool], [3.0])
        self.assertIsNone(note)
        self.assertEqual(cover.speaker_at(self.PLAN, 3.0), "Alex Kim")
        self.assertEqual(cover.speaker_at(self.PLAN, 1.0), "Sam Rivera")
        pool, note = cover.candidate_pool(cands, self.PLAN, "Jordan")
        self.assertEqual(pool, cands)
        self.assertIn("Jordan", note)
        self.assertEqual(cover.candidate_pool(cands, {"words": []}, "Alex"), (cands, None))


class CoverFrames(unittest.TestCase):
    """Source-fallback frames: framing parity with render (active area, default mark), the HDR
    tone-map rule and the rawvideo timestamp flag."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="hc-cf-"))
        cls.src = cls.tmp / "band.mp4"
        # 1280x720 with the picture in a 1280x360 band (black bars above and below)
        env.ff(["-f", "lavfi", "-i", "color=c=black:s=1280x720:r=25:d=3",
                "-f", "lavfi", "-i", "testsrc2=size=1280x360:rate=25:duration=3",
                "-filter_complex", "[0:v][1:v]overlay=0:180:shortest=1,format=yuv420p",
                "-c:v", "libx264", "-preset", "ultrafast", "-qp", "0", str(cls.src)])

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def project(self, hdr=None, active=True):
        root = Path(tempfile.mkdtemp(prefix="p", dir=self.tmp))
        (root / "source.mp4").symlink_to(self.src)
        video = {"width": 1280, "height": 720, "fps": [25, 1], "start": 0.0}
        if active:
            video["active_area"] = [0, 180, 1280, 360]
        if hdr:
            video["hdr"] = hdr
        env.write_json(root / "source.json", {"version": 1, "path": "source.mp4", "fingerprint": "f",
                                              "duration": 3.0, "format_start": 0.0, "video": video,
                                              "audio": {"start": 0.0}})
        env.write_json(root / "clips.json", {"version": 1, "clips": {"c01": {}}})
        return Project(root)

    PLAN = {"fps": [25, 1], "grid_offset": 0.0, "duration_out": 2.0, "frames_out": 50,
            "segments": [{"k": 0, "src": [0.5, 2.5], "out": [0.0, 2.0], "frames": 50}], "words": []}

    def test_source_frame_has_no_black_rows(self):
        for active, expect_clean in ((True, True), (False, False)):
            proj = self.project(active=active)
            frames, origin = cover_frames.base_frames(proj, "c01", {}, dict(self.PLAN), "vertical",
                                                      [1.0], (180, 320))
            self.assertEqual(origin, "source")
            rows = frames[1.0].max(axis=(1, 2))
            self.assertEqual(bool((rows > 20).all()), expect_clean, rows.min())

    def test_letterbox_uses_the_active_picture(self):
        img = Image.new("RGB", (1280, 720), (0, 0, 0))
        img.paste(Image.new("RGB", (1280, 360), (200, 120, 40)), (0, 180))
        out = cover_frames.fit_frame(img, {"mode": "letterbox"}, 360, 640, 0.43,
                                     active=(0, 180, 1280, 360))
        rows = np.asarray(out).max(axis=(1, 2))
        self.assertTrue((rows > 20).all(), rows.min())
        rect = cover_frames.fit_frame(img, {"mode": "letterbox", "rect": [0.5, 0.25, 0.5, 0.5]},
                                      360, 640, 0.43, active=(0, 180, 1280, 360))
        self.assertTrue((np.asarray(rect).max(axis=(1, 2)) > 20).all())

    def test_multi_face_shot_defaults_to_letterbox(self):
        proj = self.project()
        shot = {"start_src": 0.0, "end_src": 3.0, "kind": "two", "multi_face": True,
                "faces": [{"id": "A", "x": 0.27, "y": 0.4, "h": 0.2, "presence": 1.0},
                          {"id": "B", "x": 0.75, "y": 0.4, "h": 0.23, "presence": 1.0}]}
        Path(proj.shots_path("c01")).parent.mkdir(parents=True, exist_ok=True)
        env.write_json(proj.shots_path("c01"), {"version": 1, "multi_face": True, "shots": [shot],
                                                "samples": [], "proposal": [{"mode": "track", "x": 0.75}]})
        m = cover_frames.frame_mark(proj, "c01", {}, dict(self.PLAN), "vertical", 1.0)
        self.assertEqual(m["mode"], "letterbox")

    def test_speaker_follow_frames_the_speaking_person(self):
        proj = self.project()
        words = []
        for k, (sp, s) in enumerate((("A", 0.5), ("A", 0.7), ("A", 0.9), ("A", 1.1),
                                     ("B", 1.6), ("B", 1.8), ("B", 2.0), ("B", 2.2))):
            words.append({"i": k, "w": "w", "s_src": s, "e_src": s + 0.18, "s_out": s - 0.5,
                          "e_out": s - 0.32, "sp": sp})
        plan = dict(self.PLAN, words=words, framing=[],
                    speakers={"A": {"x": 0.25}, "B": {"x": 0.75}})
        a = cover_frames.frame_mark(proj, "c01", {}, plan, "vertical", 0.8)
        b = cover_frames.frame_mark(proj, "c01", {}, plan, "vertical", 2.1)
        self.assertEqual((a["mode"], a["x"]), ("track", 0.25))
        self.assertEqual((b["mode"], b["x"]), ("track", 0.75))

    def test_track_mark_without_y_is_framed(self):
        img = Image.new("RGB", (1280, 720), (90, 90, 90))
        out = cover_frames.fit_frame(img, {"mode": "track", "x": 0.7, "y": None}, 90, 160)
        self.assertEqual(out.size, (90, 160))

    def test_hdr_source_without_zscale_fails_like_render(self):
        proj = self.project(hdr="hlg")
        with mock.patch.object(env, "has_filter", return_value=False):
            with self.assertRaises(env.HCError) as ctx:
                cover_frames.base_frames(proj, "c01", {}, dict(self.PLAN), "vertical", [1.0], (180, 320))
        self.assertEqual(ctx.exception.code, 3)
        self.assertIn("zscale", str(ctx.exception))
        self.assertIn("hc doctor", str(ctx.exception))

    def test_rawvideo_pipe_keeps_decoder_timestamps(self):
        seen = []
        real = cover_frames.subprocess.run

        def spy(cmd, *a, **k):
            seen.append(list(cmd))
            return real(cmd, *a, **k)

        mkv = self.tmp / "t24.mkv"
        env.ff(["-f", "lavfi", "-i", "testsrc2=size=160x90:rate=24:duration=2", "-c:v", "libx264",
                "-preset", "ultrafast", str(mkv)])
        with mock.patch.object(cover_frames.subprocess, "run", spy):
            got = cover_frames.grab_frames(mkv, range(0, 30, 3), size=(160, 90), ss=0.5 - 0.5 / 24)
        self.assertEqual(len(got), 10)
        cmd = seen[0]
        i = cmd.index("-fps_mode")
        self.assertEqual(cmd[i + 2:i + 4], ["-enc_time_base", "-1"])


if __name__ == "__main__":
    unittest.main()
