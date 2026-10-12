"""Captions and typesetting (contract 9.3, 9.4): tokens, emphasis, paging, hook card, legibility,
glyphs, frame determinism, band, SRT. Plans are built by hand per contract 8.1."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

import numpy as np

SK = Path(__file__).resolve().parents[1]
if str(SK / "scripts") not in sys.path:
    sys.path.insert(0, str(SK / "scripts"))

from hookcutter import captions, env, typeset  # noqa: E402

W, H = 1080, 1920


def load_preset(name: str) -> dict:
    data = json.loads((SK / "presets" / f"{name}.json").read_text())
    parent = data.pop("extends", None)
    return captions._deep_merge(load_preset(parent), data) if parent else data


CLEAN = load_preset("clean")


def style(**caption) -> dict:
    """clean with caption overrides (layout keys go under caption.layout)."""
    layout = caption.pop("layout", {})
    over = {"caption": dict(caption, layout=layout)}
    return captions._deep_merge(CLEAN, over)


def make_plan(words, *, gap=0.06, start=0.5, joins=(), dur=None, hook=None, fps=30):
    """words: list of str or (text, dict-of-overrides). Times are generated unless given."""
    out, t = [], start
    for i, w in enumerate(words):
        text, extra = (w, {}) if isinstance(w, str) else w
        length = extra.pop("len", 0.14 + 0.03 * len(text))
        s = extra.pop("s", t)
        e = extra.pop("e", s + length)
        row = {"i": i, "raw": text, "text": text, "s_src": s, "e_src": e, "s_out": round(s, 3),
               "e_out": round(e, 3), "piece": 0, "seg": 0, "sp": None, "filler": False,
               "hidden": False, "omitted": False, "removed": False}
        row.update(extra)
        out.append(row)
        t = e + extra.get("after", gap)
    total = dur if dur is not None else t + 0.6
    plan = {"version": 1, "clip": "c01", "fps": [fps, 1], "words": out, "joins": list(joins),
            "frames_out": int(round(total * fps)), "duration_out": total}
    if hook:
        plan["hook_card"] = hook
    return plan


def build(plan, preset=None, clip=None, luma=None, fmt="vertical", fps=(30, 1), **kw):
    return captions.build_track(plan, clip or {}, preset or CLEAN, fmt,
                                (W, H) if fmt == "vertical" else (1920, 1080), fps, luma, **kw)


def page_texts(track):
    return [p["text"] for p in track.pages]


class TokenTests(unittest.TestCase):
    def tokens(self, words, **cap):
        return [t.text for t in captions.display_tokens(make_plan(words), {}, style(**cap)["caption"])]

    def test_glue_rules(self):
        self.assertEqual(self.tokens(["It", "cost", "$", "70", "and", "80", "%", "of", "it."]),
                         ["IT", "COST", "$70", "AND", "80%", "OF", "IT"])
        self.assertEqual(self.tokens(["a", "full", "-time", "job"]), ["A", "FULL-TIME", "JOB"])
        self.assertEqual(self.tokens(["I", "do", "n't", "know", "it", "'s", "fine"]),
                         ["I", "DON'T", "KNOW", "IT'S", "FINE"])
        self.assertEqual(self.tokens(["we", "'re", "at", "$2", ",400", "and", "3", ".5"]),
                         ["WE'RE", "AT", "$2,400", "AND", "3.5"])
        self.assertEqual(self.tokens(["post", "-COVID"]), ["POST", "-COVID"])   # capital: no glue
        self.assertEqual(self.tokens(["in", "the", "U.S.,", "mostly"]), ["IN", "THE", "U.S.", "MOSTLY"])
        plan = make_plan(["costs", ("$", {"piece": 0}), ("5", {"piece": 1})])
        self.assertEqual([t.text for t in captions.display_tokens(plan, {}, CLEAN["caption"])],
                         ["COSTS", "$", "5"])

    def test_glue_keeps_indices_and_times(self):
        plan = make_plan(["seven", "-figure", "deal"])
        toks = captions.display_tokens(plan, {}, CLEAN["caption"])
        self.assertEqual(toks[0].idx, [0, 1])
        self.assertEqual(toks[0].s, plan["words"][0]["s_out"])
        self.assertEqual(toks[0].e, plan["words"][1]["e_out"])

    def test_no_glue_across_pieces(self):
        plan = make_plan(["deal", ("'s", {"piece": 1})])
        self.assertEqual([t.text for t in captions.display_tokens(plan, {}, CLEAN["caption"])],
                         ["DEAL", "'S"])

    def test_visibility_and_strip(self):
        words = ["Well,", ("um", {"filler": True}), ("gone", {"removed": True, "s_out": None,
                                                              "e_out": None}),
                 ("x", {"hidden": True}), "the", "U.S.", "really?", "work."]
        self.assertEqual(self.tokens(words), ["WELL", "THE", "U.S.", "REALLY?", "WORK"])
        self.assertEqual(self.tokens(["Hello,", "world."], case="as_is", strip_punct=""),
                         ["Hello,", "world."])


class EmphasisTests(unittest.TestCase):
    def marked(self, words, phrases):
        toks = captions.display_tokens(make_plan(words), {}, CLEAN["caption"])
        captions.mark_emphasis(toks, phrases, set(), ["hook"])
        return [t.text for t in toks if t.emph]

    def test_apostrophes(self):
        self.assertEqual(self.marked(["you", "don't", "need", "it"], ["DON'T"]), ["DON'T"])
        self.assertEqual(self.marked(["you", "do", "n't", "need", "it"], ["don’t"]), ["DON'T"])
        self.assertEqual(self.marked(["it", "wouldn't", "work"], ["WOULDN'T"]), ["WOULDN'T"])

    def test_hyphens(self):
        self.assertEqual(self.marked(["seven-figure", "owners"], ["SEVEN-FIGURE"]), ["SEVEN-FIGURE"])
        self.assertEqual(self.marked(["seven", "figure", "owners"], ["SEVEN-FIGURE"]),
                         ["SEVEN", "FIGURE"])
        self.assertEqual(self.marked(["7-figure", "owners"], ["SEVEN-FIGURE"]), ["7-FIGURE"])

    def test_stopwords_not_accented(self):
        words = ["an", "idea", "in", "an", "afternoon", "and", "an", "hour"]
        self.assertEqual(self.marked(words, ["an afternoon"]), ["AFTERNOON"])

    def test_numbers(self):
        self.assertEqual(self.marked(["it", "cost", "seventy", "thousand", "dollars"], ["$70K"]),
                         ["SEVENTY", "THOUSAND"])
        self.assertEqual(self.marked(["it", "cost", "$70,000"], ["seventy thousand"]), ["$70,000"])
        self.assertEqual(self.marked(["eighty", "percent", "of", "it"], ["80%"]), ["EIGHTY", "PERCENT"])
        self.assertEqual(self.marked(["five", "six", "times"], ["11"]), [])   # no false sums

    def test_multiword_and_negation(self):
        self.assertEqual(self.marked(["afraid", "of", "looking", "stupid"], ["LOOKING STUPID"]),
                         ["LOOKING", "STUPID"])
        self.assertEqual(self.marked(["it", "is", "not", "about", "money"], ["NOT"]), ["NOT"])
        self.assertEqual(self.marked(["the", "one", "thing"], ["THE"]), ["THE"])   # literal fallback

    def test_accent_phrases_parse(self):
        self.assertEqual(captions.accent_phrases(["AFRAID OF *LOOKING STUPID*", "*$70K* *DONE*"]),
                         ["LOOKING STUPID", "$70K", "DONE"])

    def test_per_page_cap(self):
        plan = make_plan(["big", "and", "bold"], hook={"lines": ["*BIG* AND *BOLD*"], "dur": 0.2},
                         start=1.0)
        track = build(plan)
        self.assertEqual(page_texts(track), ["BIG AND BOLD"])
        self.assertEqual([t["text"] for t in track.pages[0]["tokens"] if t["emph"]], ["BIG"])
        self.assertEqual(track.to_json()["accents"]["matched"], ["BIG"])

    def test_cap_counts_runs_not_tokens(self):
        plan = make_plan(["seventy", "thousand"], hook={"lines": ["*$70K*"], "dur": 0.2}, start=1.0)
        track = build(plan)
        self.assertEqual(len(track.pages), 1)
        emph = [t["text"] for t in track.pages[0]["tokens"] if t["emph"]]
        self.assertEqual(emph, ["SEVENTY", "THOUSAND"])
        self.assertEqual(track.to_json()["accents"]["matched"], ["$70K"])

    def test_manual_emphasis(self):
        plan = make_plan(["we", "shipped", "it", "today"])
        track = build(plan, clip={"emphasis": [3]})
        self.assertEqual([t["text"] for p in track.pages for t in p["tokens"] if t["emph"]], ["TODAY"])


class PagingTests(unittest.TestCase):
    def test_pause_break(self):
        plan = make_plan(["one", ("two", {"after": 0.8}), "three", "four"])
        self.assertEqual(page_texts(build(plan)), ["ONE TWO", "THREE FOUR"])

    def test_sentence_break(self):
        plan = make_plan(["Go", "now.", "Then", "stop"])
        self.assertEqual(page_texts(build(plan)), ["GO NOW", "THEN STOP"])

    def test_comma_break_needs_min_words(self):
        plan = make_plan(["So,", "we", "won,", "then", "left"])
        self.assertEqual(page_texts(build(plan)), ["SO WE WON", "THEN LEFT"])

    def test_join_break(self):
        words = ["alpha", "beta", ("gamma", {"seg": 1}), ("delta", {"seg": 1})]
        tight = make_plan(words, joins=[{"k": 1, "out": 1.0, "kind": "tighten"}])
        self.assertEqual(page_texts(build(tight)), ["ALPHA BETA GAMMA DELTA"])
        for kind in ("piece", "omit", "dip"):
            plan = make_plan(words, joins=[{"k": 1, "out": 1.0, "kind": kind}])
            self.assertEqual(page_texts(build(plan)), ["ALPHA BETA", "GAMMA DELTA"], kind)

    def test_speaker_break(self):
        plan = make_plan([("yes", {"sp": "A"}), ("right", {"sp": "A"}), ("no", {"sp": "B"}),
                          ("way", {"sp": "B"})])
        self.assertEqual(page_texts(build(plan)), ["YES RIGHT", "NO WAY"])

    def test_max_words(self):
        plan = make_plan(["red", "green", "blue", "black", "white", "pink"])
        pages = page_texts(build(plan))
        self.assertTrue(all(len(p.split()) <= 4 for p in pages))
        self.assertEqual(" ".join(pages), "RED GREEN BLUE BLACK WHITE PINK")
        self.assertNotIn(1, [len(p.split()) for p in pages])         # no orphan

    def test_max_chars_and_lines(self):
        pre = style(layout={"max_chars": 9, "max_lines": 1, "max_words": 6})
        plan = make_plan(["tiny", "cat", "sat", "on", "a", "warm", "mat"])
        track = build(plan, pre)
        for p in track.pages:
            self.assertEqual(len(p["lines"]), 1)
            self.assertLessEqual(len(" ".join(p["lines"][0])), 9, p)

    def test_max_width_real_measurement(self):
        plan = make_plan(["extraordinary", "accomplishments", "everywhere"])
        track = build(plan)
        limit = track.to_json()["max_width"]
        self.assertLessEqual(limit, 780)                           # symmetric organic safe span
        font = typeset.load_font(CLEAN["caption"]["font"], CLEAN["caption"]["size"])
        for p in track.pages:
            for line in p["lines"]:
                if len(line) > 1:
                    self.assertLessEqual(typeset.text_width(font, " ".join(line)), limit + 1)
            self.assertLessEqual(len(p["lines"]), 2)

    def test_no_dangling_function_word(self):
        # At a natural pace (0.2 s between words) soft danglers are avoided. Paging also weighs
        # on-screen time now (min_page), so at a very fast pace a soft dangler such as "THAT"
        # may end a page rather than flash a 0.6 s page; hard danglers never do (test_fix).
        words = "I believe that if you're afraid of looking stupid".split()
        pages = page_texts(build(make_plan(words, gap=0.2)))
        self.assertEqual(" ".join(pages), "I BELIEVE THAT IF YOU'RE AFRAID OF LOOKING STUPID")
        for p in pages[:-1]:
            self.assertNotIn(p.split()[-1], {"THAT", "IF", "OF", "YOU'RE"}, pages)
        self.assertNotIn(1, [len(p.split()) for p in pages])
        fast = page_texts(build(make_plan(words, gap=0.02)))
        for p in fast[:-1]:
            self.assertNotIn(p.split()[-1], {"OF", "A", "THE", "TO"}, fast)

    def test_min_page_extends_into_gap(self):
        plan = make_plan(["Stop.", ("Now", {"s": 2.0, "len": 0.15}), ("then", {"s": 3.5})])
        track = build(plan)
        for p in track.pages:
            self.assertGreaterEqual(p["end_out"] - p["start_out"], 0.8 - 1e-6, p)

    def test_min_page_merges_short_neighbours(self):
        # "No." is boxed in by its neighbours: no gap on either side to extend into.
        plan = make_plan([("Run.", {"s": 0.06, "e": 0.9}), ("No.", {"s": 0.92, "e": 1.0}),
                          ("Yes", {"s": 1.02, "e": 1.4})])
        track = build(plan)
        self.assertEqual(page_texts(track), ["RUN NO", "YES"])
        for p in track.pages:
            self.assertGreaterEqual(p["end_out"] - p["start_out"], 0.8 - 1e-6, p)

    def test_pages_do_not_overlap(self):
        words = "we built the whole thing in a weekend and it worked better than the old one".split()
        track = build(make_plan(words, gap=0.02))
        spans = [(p["start_out"], p["end_out"]) for p in track.pages]
        for (a0, a1), (b0, b1) in zip(spans, spans[1:]):
            self.assertLessEqual(a1, b0 + 1e-9)
        frames = [p["frames"] for p in track.pages]
        for (a0, a1), (b0, b1) in zip(frames, frames[1:]):
            self.assertLessEqual(a1, b0)


class HookTests(unittest.TestCase):
    def plan(self):
        words = ["afraid", "of", "looking", "stupid", "is", "the", "real", "cost", "here", "today"]
        return make_plan(words, start=0.2, gap=0.12, hook={"lines": ["AFRAID OF", "*STUPID*"], "dur": 1.2})

    def test_full_opacity_on_frame_zero(self):
        track = build(self.plan())
        self.assertEqual(track.frame_key(0), ("hook", 255))
        img = np.frombuffer(track.frame(0), np.uint8).reshape(track.band_h, W, 4)
        self.assertEqual(int(img[..., 3].max()), 255)
        self.assertEqual(track.hook["start_out"], 0.0)
        self.assertEqual(track.to_json()["hook"]["frames"][0], 0)

    def test_fades_out_by_dur(self):
        track = build(self.plan())
        end_f = track.hook["frames"][1]
        alphas = [track.frame_key(n)[1] for n in range(end_f)]
        self.assertEqual(alphas[: end_f - 8], [255] * (end_f - 8))
        self.assertTrue(all(a >= b for a, b in zip(alphas, alphas[1:])))
        self.assertLess(alphas[-1], 255)
        self.assertNotEqual(track.frame_key(end_f)[0], "hook")

    def test_pages_trimmed_after_card(self):
        track = build(self.plan())
        self.assertTrue(track.pages)
        for p in track.pages:
            self.assertGreaterEqual(p["start_out"], 1.2 - 1e-9)
            self.assertGreaterEqual(p["frames"][0], track.hook["frames"][1])

    def test_short_remainder_dropped(self):
        plan = make_plan([("gone", {"s": 0.2, "e": 1.0}), ("next", {"s": 2.0})],
                         hook={"lines": ["HOOK"], "dur": 1.2})
        self.assertEqual(page_texts(build(plan)), ["NEXT"])

    def test_auto_duration_and_srt_first(self):
        plan = make_plan(["hello", "world"], start=3.5, hook={"lines": ["SAY *HELLO*"], "dur": "auto"})
        track = build(plan, clip={})
        self.assertAlmostEqual(track.hook["end_out"], 1.5, places=3)       # clamp(0.5 + 9/20, 1.5, 3)
        with tempfile.TemporaryDirectory() as d:
            text = track.write_srt(Path(d) / "x.srt").read_text()
        # The SRT carries what is said, not the card's headline: the first cue is the first
        # spoken word (the hook card is a title, not speech).
        self.assertTrue(text.startswith("1\n00:00:03,500 --> "), text)
        self.assertIn("hello world", text)
        self.assertNotIn("SAY", text)

    def test_long_hook_line_is_rebroken_bigger(self):
        plan = make_plan(["afraid", "of", "looking", "stupid"], start=3.0,
                         hook={"lines": ["AFRAID OF LOOKING", "*STUPID*"], "dur": 2.0})
        track = build(plan)
        self.assertEqual(track.hook["rendered_lines"], ["AFRAID OF", "LOOKING", "STUPID"])
        self.assertEqual(track.hook["size"], CLEAN["hook_card"]["max_size"])
        two = captions._deep_merge(CLEAN, {"hook_card": {"max_lines": 2}})
        self.assertEqual(build(plan, two).hook["rendered_lines"], ["AFRAID OF LOOKING", "STUPID"])

    def test_legacy_text_hook(self):
        plan = make_plan(["x", "y"], start=3.0)
        track = build(plan, clip={"hook_card": {"text": "TWO\\N*LINES*", "dur": 1.5}})
        self.assertEqual(track.hook["lines"], ["TWO", "*LINES*"])


class LegibilityTests(unittest.TestCase):
    """The pill: shadow-aware contrast of each page's own box, a texture trigger, and the
    decision held per shot (pages between piece/dip joins or framing changes), so it never
    blinks from page to page inside one picture."""

    def luma_track(self, values_by_page, words_per_page=2, cut_after=None):
        words, t = [], 0.5
        joins = []
        for k in range(len(values_by_page)):
            seg = 0 if cut_after is None or k <= cut_after else 1
            if cut_after is not None and k == cut_after + 1:
                joins.append({"k": 1, "out": round(t - 0.3, 3), "kind": "piece"})
            for j in range(words_per_page):
                text = ("word%s" % chr(97 + k)) if j == 0 else "end."
                words.append((text, {"s": t, "e": t + 0.3, "seg": seg}))
                t += 0.4
            t += 0.6
        plan = make_plan(words, dur=t + 1, joins=joins)
        if joins:
            # the join changes the framing (another mark): a new shot for the caption backing
            cut = joins[0]["out"]
            plan["segments"] = [{"src": [0.0, cut], "out": [0.0, cut]},
                                {"src": [100.0, 100.0 + t + 1 - cut], "out": [cut, t + 1]}]
            plan["framing"] = [{"at_src": 0.0, "mode": "track", "x": 0.3},
                               {"at_src": 100.0, "mode": "track", "x": 0.7}]
        n = int(plan["duration_out"] * 5) + 2
        luma = np.zeros((n, 480, 270), np.uint8)
        track0 = build(plan, luma=None)
        for p, v in zip(track0.pages, values_by_page):
            a, b = int(p["start_out"] * 5), int(p["end_out"] * 5) + 1
            luma[a:b] = v
        return build(plan, luma=luma)

    def test_bright_page_pills_its_whole_shot(self):
        track = self.luma_track([20, 230, 20, 20])
        self.assertEqual([p["backing"] for p in track.pages], [True] * 4)
        self.assertEqual([p["backing_forced"] for p in track.pages], [False, True, False, False])
        self.assertEqual(track.pages[1]["backing_reason"], "contrast")
        self.assertEqual(track.pages[0]["backing_reason"], "shot")
        lit = track.pages[1]
        bare = captions.contrast_ratio(captions.parse_color("#FFFFFF")[:3], lit["p90"])
        self.assertGreater(lit["contrast"], bare)          # the pill raised the contrast

    def test_shot_boundary_limits_the_pill(self):
        track = self.luma_track([20, 230, 20, 20], cut_after=1)
        self.assertEqual([p["backing"] for p in track.pages], [True, True, False, False])

    def test_light_grey_stays_bare(self):
        # p90 0.65 used to force a pill (bare contrast under 2.5:1); with the shadow it reads
        # at about 4:1, so no pill.
        track = self.luma_track([20, 166, 20, 20])
        self.assertEqual([p["backing"] for p in track.pages], [False] * 4)
        for p in track.pages:
            self.assertIsNotNone(p["p90"])
            self.assertIsNotNone(p["contrast"])

    def test_modes_and_no_luma(self):
        plan = make_plan(["a", "b", "c."])
        self.assertTrue(all(p["backing"] for p in build(plan, style(backing={"mode": "always"})).pages))
        self.assertFalse(any(p["backing"] for p in build(plan).pages))      # no luma: no evidence
        self.assertIsNone(build(plan).pages[0]["p90"])

    def test_pill_inside_safe_zone(self):
        plan = make_plan("unbelievably extraordinary accomplishments everywhere".split())
        track = build(plan, style(backing={"mode": "always"}))
        for p in track.pages:
            x, y, w, h = p["backing_box"]
            self.assertGreaterEqual(x, 90)
            self.assertLessEqual(x + w, W - 150)

    def test_contrast_ratio(self):
        self.assertAlmostEqual(captions.contrast_ratio((255, 255, 255), 0.0), 21.0, places=1)
        self.assertAlmostEqual(captions.contrast_ratio((255, 255, 255), 1.0), 1.0, places=2)


class StyleOptionTests(unittest.TestCase):
    def test_outline_draws_wider_ink(self):
        plan = make_plan(["outline", "test"], start=0.5)
        plain, lined = build(plan), build(plan, style(outline={"px": 5, "color": "#000000"}))
        n = plain.pages[0]["frames"][1] - 1

        def ink(track):
            a = np.frombuffer(track.frame(n), np.uint8).reshape(track.band_h, W, 4)[..., 3]
            return int((a > 200).sum())

        self.assertGreater(ink(lined), ink(plain))

    def test_overwide_word_shrinks_page(self):
        plan = make_plan(["internationalization"], start=0.5)
        track = build(plan)
        page = track.pages[0]
        self.assertLess(page["size"], CLEAN["caption"]["size"])
        self.assertLessEqual(page["box"][2], track.to_json()["max_width"] + 2)
        self.assertTrue(any("shrunk" in w for w in track.to_json()["warnings"]))

    def test_accent_reveal_page_start(self):
        plan = make_plan(["the", "big", "idea"], hook={"lines": ["*IDEA*"], "dur": 0.2}, start=1.0)
        accent = tuple(captions.parse_color(CLEAN["caption"]["accent"])[:3])
        on_spoken, early = build(plan), build(plan, style(accent_reveal="page_start"))
        n = on_spoken.pages[0]["frames"][0]
        self.assertNotEqual(tuple(on_spoken.frame_key(n)[2][2][1:4]), accent)
        self.assertEqual(tuple(early.frame_key(n)[2][2][1:4]), accent)
        end = on_spoken.pages[0]["frames"][1] - 1
        self.assertEqual(tuple(on_spoken.frame_key(end)[2][2][1:4]), accent)

    def test_auto_emphasis_ranking(self):
        # Auto accents only numbers, currency and mid-sentence names: a long ordinary word
        # ("STRATEGY") is not the claim, so it stays plain.
        pre = style(emphasis={"sources": ["auto"], "per_page": 1})
        pages = build(make_plan(["basically", "we", "earned", "$40K"]), pre).pages
        self.assertEqual([t["text"] for p in pages for t in p["tokens"] if t["emph"]], ["$40K"])
        pages = build(make_plan(["honestly", "the", "strategy", "worked"]), pre).pages
        self.assertEqual([t["text"] for p in pages for t in p["tokens"] if t["emph"]], [])
        pages = build(make_plan(["we", "met", "Priya", "there"]), pre).pages
        self.assertEqual([t["text"] for p in pages for t in p["tokens"] if t["emph"]], ["PRIYA"])

    def test_static_reveal_single_state(self):
        track = build(make_plan(["one", "two", "three"]), load_preset("minimal"))
        page = track.pages[0]
        keys = {track.frame_key(n) for n in range(*page["frames"])}
        self.assertEqual(len(keys), 1)


class GlyphTests(unittest.TestCase):
    def test_missing_glyph_raises(self):
        plan = make_plan(["price", "in", "漢字"])
        with self.assertRaises(env.HCError) as ctx:
            build(plan)
        self.assertIn("U+6F22", str(ctx.exception))

    def test_hook_glyph_checked(self):
        plan = make_plan(["fine"], start=3.0, hook={"lines": ["★ STAR"], "dur": 1.5})
        with self.assertRaises(env.HCError):
            build(plan)

    def test_missing_glyphs_helper(self):
        path = typeset.resolve_font("ArchivoBlack-Regular.ttf")[0]
        self.assertEqual(typeset.missing_glyphs(path, "HELLO WORLD’S €5"), [])
        self.assertEqual(typeset.missing_glyphs(path, "ok 漢 ok 漢"), ["漢"])


class FrameTests(unittest.TestCase):
    def setUp(self):
        words = "we doubled revenue in ninety days without hiring anyone at all".split()
        self.plan = make_plan(words, hook={"lines": ["*DOUBLED* IN 90 DAYS"], "dur": 0.4}, start=0.5)

    def test_frame_times_exact(self):
        track = build(self.plan, fps=(30000, 1001))
        self.assertEqual(track.time(30000), Fraction(1001))
        self.assertEqual(track.fps, (30000, 1001))
        plan = make_plan(["one", "two"], start=1.06)          # first page starts at exactly 1.0 s
        track = build(plan, fps=(25, 1))
        self.assertEqual(track.pages[0]["frames"][0], 25)

    def test_determinism_and_cache(self):
        a = build(self.plan, preset=load_preset("pop"))
        b = build(self.plan, preset=load_preset("pop"))
        keys = [a.frame_key(n) for n in range(a.frames)]
        frames = [a.frame(n) for n in range(a.frames)]
        for n in range(0, a.frames, 7):                         # fresh track, random order
            self.assertEqual(b.frame(a.frames - 1 - n), frames[a.frames - 1 - n])
        for n in range(1, a.frames):
            self.assertEqual(keys[n] == keys[n - 1], frames[n] == frames[n - 1], n)
        for k in keys:
            if k[0] == "page":
                self.assertTrue(all(v >= 0 for st in k[2] for v in st), k)

    def test_unspoken_state_is_fixed_sentinel(self):
        track = build(self.plan)
        page = track.pages[0]
        n0 = page["frames"][0]
        last_state = track.frame_key(n0)[2][-1]
        later = track.frame_key(n0 + 2)[2][-1]
        self.assertEqual(last_state, later)                    # same until spoken
        self.assertEqual(last_state[0], round(255 * CLEAN["caption"]["unspoken_alpha"]))

    def test_fill_reaches_full(self):
        track = build(self.plan)
        page = track.pages[0]
        settled = track.frame_key(page["frames"][1] - 1)[2]
        self.assertTrue(all(st[0] == 255 for st in settled))

    def test_iter_frames_and_size(self):
        track = build(self.plan, preset=load_preset("minimal"))
        count = 0
        for data in track.iter_frames():
            self.assertEqual(len(data), W * track.band_h * 4)
            count += 1
        self.assertEqual(count, self.plan["frames_out"])

    def test_band_inside_frame(self):
        for y in (1250, 1880, 40):
            track = build(self.plan, clip={"caption_y": y})
            self.assertEqual(track.band_y % 2, 0)
            self.assertEqual(track.band_h % 2, 0)
            self.assertGreaterEqual(track.band_y, 0)
            self.assertLessEqual(track.band_y + track.band_h, H)
            if y == 1250:
                for p in track.pages + [track.hook]:
                    x, by, w, h = p["box"]
                    self.assertGreaterEqual(by, track.band_y)
                    self.assertLessEqual(by + h, track.band_y + track.band_h)

    def test_landscape(self):
        track = build(self.plan, fmt="landscape")
        self.assertEqual(track.width, 1920)
        self.assertEqual(track.pages[0]["size"], CLEAN["landscape"]["caption"]["size"])
        self.assertLessEqual(track.band_y + track.band_h, 1080)

    def test_captions_off_keeps_hook_only(self):
        track = build(self.plan, clip={"captions": False, "hook_card": None})
        self.assertEqual(track.pages, [])
        self.assertIsNotNone(track.hook)


class FfmpegPipeTests(unittest.TestCase):
    """The strip as stage B consumes it: raw RGBA on stdin at the exact rational rate."""

    def test_overlay_through_ffmpeg(self):
        import subprocess
        fps = (25, 1)
        plan = make_plan([("we", {"s": 1.0}), ("shipped", {"s": 1.2}), ("it.", {"s": 1.5, "e": 1.8})],
                         hook={"lines": ["*SHIPPED*"], "dur": 0.6}, dur=2.4, fps=25)
        track = build(plan, fps=fps)
        strip = b"".join(track.iter_frames())
        bw, bh = W // 4, track.band_h // 4
        cmd = [env.ffmpeg(), "-hide_banner", "-nostdin", "-loglevel", "error",
               "-f", "lavfi", "-i", "color=c=0x404040:s=%dx%d:r=25:d=2.4" % (W, H),
               "-f", "rawvideo", "-pix_fmt", "rgba", "-s", "%dx%d" % (W, track.band_h),
               "-framerate", "25/1", "-i", "pipe:0",
               "-filter_complex",
               "[1:v]format=yuva420p[s];[0:v][s]overlay=0:%d:eof_action=pass,"
               "crop=%d:%d:0:%d,scale=%d:%d,format=gray[v]"
               % (track.band_y, W, track.band_h, track.band_y, bw, bh),
               "-map", "[v]", "-f", "rawvideo", "-pix_fmt", "gray", "pipe:1"]
        res = subprocess.run(cmd, input=strip, capture_output=True, check=True)
        frames = np.frombuffer(res.stdout, np.uint8).reshape(-1, bh, bw)
        self.assertEqual(len(frames), plan["frames_out"])
        base = 0x40
        bright = [int((f > base + 60).sum()) for f in frames]
        self.assertGreater(bright[0], 50)                       # hook card on frame 0
        hook_end = track.hook["frames"][1]
        page0 = track.pages[0]["frames"][0]
        self.assertGreater(page0, hook_end)
        self.assertTrue(all(b == 0 for b in bright[hook_end:page0]), bright[hook_end:page0])
        self.assertGreater(bright[page0 + 12], 50)              # spoken words fully lit
        self.assertTrue(all(b == 0 for b in bright[track.pages[-1]["frames"][1]:]))


class JsonSrtTests(unittest.TestCase):
    def test_to_json_shape(self):
        plan = make_plan(["hello", "there,", "friend"], start=2.0, hook={"lines": ["*HELLO*"], "dur": 1.5})
        data = build(plan).to_json()
        for key in ("band", "font", "pages", "hook", "accents"):
            self.assertIn(key, data)
        page = data["pages"][0]
        for key in ("i", "start_out", "end_out", "text", "lines", "tokens", "box", "backing", "p90",
                    "contrast"):
            self.assertIn(key, page)
        self.assertEqual(set(page["tokens"][0]), {"text", "s_out", "e_out", "emph", "source", "idx"})
        self.assertEqual(data["accents"], {"phrases": ["HELLO"], "matched": ["HELLO"]})
        json.dumps(data)

    def test_srt_format(self):
        plan = make_plan(["Hello", "there.", "General", "Kenobi!"], start=2.0,
                         hook={"lines": ["A *BOLD* ONE"], "dur": 1.5})
        track = build(plan)
        with tempfile.TemporaryDirectory() as d:
            text = track.write_srt(Path(d) / "c.srt").read_text(encoding="utf-8")
        cues = text.strip().split("\n\n")
        # Cues follow subtitle conventions over every spoken word (no hook-card cue).
        self.assertEqual(len(cues), len(track.srt_cues()))
        stamp = r"\d\d:\d\d:\d\d,\d\d\d"
        for k, cue in enumerate(cues, 1):
            lines = cue.split("\n")
            self.assertEqual(lines[0], str(k))
            self.assertRegex(lines[1], rf"^{stamp} --> {stamp}$")
            self.assertTrue(lines[2].strip())
        self.assertNotIn("BOLD", text)
        self.assertIn("Hello there.", text)                  # natural case and punctuation


class TypesetTests(unittest.TestCase):
    def test_variable_instance_and_errors(self):
        black = typeset.load_font("Montserrat[wght].ttf:Black", 60)
        thin = typeset.load_font("Montserrat[wght].ttf:Thin", 60)
        self.assertGreater(black.getlength("HELLO"), thin.getlength("HELLO"))
        self.assertIs(typeset.load_font("Montserrat[wght].ttf:Black", 60), black)    # cached
        with self.assertRaises(env.HCError):
            typeset.load_font("Montserrat[wght].ttf:Heavyish", 60)
        with self.assertRaises(env.HCError):
            typeset.load_font("NoSuchFont.ttf", 60)
        self.assertEqual(typeset.parse_font_spec(r"C:\fonts\x.ttf"), (r"C:\fonts\x.ttf", None))
        self.assertEqual(typeset.parse_font_spec("A.ttf:Bold"), ("A.ttf", "Bold"))
        mid = typeset.load_font("Montserrat[wght].ttf:wght=650", 60)
        self.assertLess(thin.getlength("HELLO"), mid.getlength("HELLO"))
        self.assertLess(mid.getlength("HELLO"), black.getlength("HELLO"))

    def test_project_font_dir_searched_first(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "fonts").mkdir()
            src = SK / "fonts" / "Anton-Regular.ttf"
            (Path(d) / "fonts" / "ArchivoBlack-Regular.ttf").write_bytes(src.read_bytes())
            path, _ = typeset.resolve_font("ArchivoBlack-Regular.ttf", d)
            self.assertEqual(path.parent, Path(d) / "fonts")

    def test_tracking_and_fit(self):
        f = typeset.load_font("ArchivoBlack-Regular.ttf", 80)
        self.assertAlmostEqual(typeset.text_width(f, "ABC", 100) - typeset.text_width(f, "ABC"),
                               2 * 8.0, places=4)
        size = typeset.fit_size(["AFRAID OF LOOKING"], "ArchivoBlack-Regular.ttf", 700, 40, 120)
        self.assertLessEqual(typeset.text_width(typeset.load_font("ArchivoBlack-Regular.ttf", size),
                                                "AFRAID OF LOOKING"), 700)
        self.assertGreater(typeset.text_width(typeset.load_font("ArchivoBlack-Regular.ttf", size + 1),
                                              "AFRAID OF LOOKING"), 700)
        self.assertEqual(typeset.fit_size(["X"], "ArchivoBlack-Regular.ttf", 5, 40, 120), 40)

    def test_layout_engine_pinned(self):
        self.assertEqual(typeset.load_font("Anton-Regular.ttf", 50).layout_engine,
                         typeset.ImageFont.Layout.BASIC)

    def test_balanced_wrap(self):
        measure = lambda ts: sum(len(t) for t in ts) * 10 + 10 * (len(ts) - 1)   # noqa: E731
        self.assertEqual(typeset.balanced_wrap(["aa", "bb"], measure, 100, 2), [["aa", "bb"]])
        lines = typeset.balanced_wrap(["aaaa", "bbbb", "cccc", "dddd"], measure, 120, 2)
        self.assertEqual(lines, [["aaaa", "bbbb"], ["cccc", "dddd"]])
        # widow avoided: [long long][x] loses to [long][long x]
        lines = typeset.balanced_wrap(["looooong", "looooong", "x"], measure, 170, 2)
        self.assertEqual(lines, [["looooong"], ["looooong", "x"]])
        self.assertIsNone(typeset.balanced_wrap(["toolongword"], measure, 50, 3))
        self.assertEqual(typeset.balanced_wrap([], measure, 50, 2), [])


class PresetFileTests(unittest.TestCase):
    def test_clean_defines_every_key(self):
        keys = {"name", "label", "extends", "description", "caption", "hook_card", "landscape",
                "letterbox", "motion", "cover", "safezone"}
        raw = json.loads((SK / "presets" / "clean.json").read_text())
        self.assertTrue(keys <= set(raw))
        self.assertIsNone(raw["extends"])
        cap_keys = {"font", "size", "tracking", "case", "strip_punct", "color", "unspoken_alpha",
                    "accent", "accent_reveal", "emphasis", "reveal", "anim", "shadow", "outline",
                    "backing", "layout", "lead_ms"}
        self.assertTrue(cap_keys <= set(CLEAN["caption"]))

    def test_shipped_presets_build(self):
        plan = make_plan("it took me seventy thousand dollars to learn this".split(), start=2.0,
                         hook={"lines": ["*$70K* LESSON"], "dur": 1.5})
        for name in ("clean", "pop", "minimal"):
            track = build(plan, load_preset(name))
            self.assertTrue(track.pages, name)
            track.frame(track.pages[0]["frames"][0] + 1)
        zones = json.loads((SK / "presets" / "safezones.json").read_text())
        self.assertEqual(zones["organic"], {"top": 240, "bottom": 420, "left": 90, "right": 150})


if __name__ == "__main__":
    unittest.main()
