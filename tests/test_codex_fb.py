"""Regression tests for the captions and presets fixes of the third fix pass: a display merge
keeps its hidden words' time and closing mark, the min_page back-fill is capped, the build
reveal pill follows the ink, a leading dot is not always a number suffix, the render assets and
the one safe-zone resolver, and preset type checks.

Every test is synthetic (generated words, plans, presets and fonts copied from the skill)."""
from __future__ import annotations

import io
import json
import shutil
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "scripts"))

import numpy as np  # noqa: E402

from hookcutter import captions, presets, qa, typeset  # noqa: E402
from hookcutter.env import HCError  # noqa: E402

import test_captions as TC  # noqa: E402  (helpers: build, load_preset, make_plan, CLEAN)
import test_plan as TP  # noqa: E402      (helpers: build, mkspec, energy)

SK = HERE.parent
CLEAN = TC.CLEAN
POP = TC.load_preset("pop")
MINIMAL = TC.load_preset("minimal")
FPS = (30, 1)
FRAME = 1.0 / 30


def planned(words, clip, defaults=None, preset=None, fps=FPS):
    """Real planner -> real caption track for one cut clip. Returns (plan, spec clip, track)."""
    defaults = defaults or {}
    with redirect_stderr(io.StringIO()):
        plan, _tr, _E = TP.build(words, {"c01": clip}, defaults=defaults, fps=fps)
        sclip = TP.mkspec({"c01": clip}, defaults)["clips"]["c01"]
        track = TC.build(plan, preset or CLEAN, clip=sclip, fps=fps)
    return plan, sclip, track


def word_out(plan, idx, key="e_out"):
    return next(float(w[key]) for w in plan["words"] if w.get("i") == idx and w.get(key) is not None)


def texts(track):
    return [p["text"] for p in track.pages]


def piece(words, **extra):
    return {"pieces": [dict({"words": [0, len(words) - 1]}, **extra)]}


# ----------------------------------------------------------------------------- display merges

DECK = [("We", 1.0, 1.12), ("sat", 1.12, 1.4), ("on", 1.44, 1.56), ("the", 1.6, 1.7),
        ("back", 1.74, 2.0), ("deck.", 2.04, 2.54)]
NIGHT = [("We", 1.0, 1.12), ("sat", 1.12, 1.4), ("on", 1.44, 1.56), ("the", 1.6, 1.7),
         ("back", 1.74, 2.0), ("deck", 2.04, 2.46), ("all", 2.5, 2.7), ("night", 2.74, 3.1),
         ("long.", 3.14, 3.6)]
RAIN = [("The", 1.0, 1.1), ("back", 1.1, 1.32), ("deck.", 1.34, 1.52), ("It", 1.7, 1.8),
        ("rained.", 1.8, 2.3), ("We", 2.5, 2.62), ("went", 2.62, 2.85), ("inside.", 2.85, 3.4)]


class MergeKeepsItsTime(unittest.TestCase):
    """`fix` {"i": "back deck", "i+1": ""}: the token lasts until the hidden word ends."""

    def test_last_merged_token_holds_until_the_hidden_word_ends(self):
        merged = dict(piece(DECK, hold_after=1.0), fix={"4": "back deck", "5": ""})
        plan, _c, track = planned(DECK, merged)
        _p, _c, control = planned(DECK, piece(DECK, hold_after=1.0))
        spoken = word_out(plan, 5)
        last = track.pages[-1]
        self.assertEqual(last["text"], "WE SAT ON THE BACK DECK")
        self.assertGreaterEqual(last["end_out"], spoken)       # it left 0.14 s before the word ended
        self.assertAlmostEqual(last["end_out"], control.pages[-1]["end_out"], places=6)
        self.assertGreaterEqual(track.srt_cues()[-1][1], spoken)
        tok = last["tokens"][-1]
        self.assertEqual(tok["idx"], [4])                      # the hidden index is not the token's
        self.assertAlmostEqual(tok["e_out"], spoken, places=3)

    def test_a_merge_inside_a_sentence_makes_no_pause_break(self):
        plan, _c, track = planned(NIGHT, dict(piece(NIGHT), fix={"4": "back deck", "5": ""}))
        self.assertNotIn("hard:pause", [p["break"] for p in track.pages])
        self.assertEqual(" ".join(texts(track)), "WE SAT ON THE BACK DECK ALL NIGHT LONG")

    def test_three_word_merge_chains(self):
        words = [("In", 1.0, 1.1), ("New", 1.14, 1.3), ("York", 1.34, 1.6), ("City.", 1.64, 2.2)]
        clip = dict(piece(words, hold_after=1.0), fix={"1": "New York City", "2": "", "3": ""})
        plan, sclip, track = planned(words, clip)
        toks = captions.display_tokens(plan, sclip, CLEAN["caption"])
        self.assertEqual([t.natural for t in toks], ["In", "New York City."])
        self.assertAlmostEqual(toks[-1].e, word_out(plan, 3), places=3)

    def test_reverse_idiom_never_pulls_the_start_back(self):
        # {"i": "", "i+1": "text"} discards an uncertain early start: the carrier starts on its own word.
        plan, sclip, _t = planned(NIGHT, dict(piece(NIGHT), fix={"4": "", "5": "back deck"}))
        toks = captions.display_tokens(plan, sclip, CLEAN["caption"])
        carrier = next(t for t in toks if t.text == "BACK DECK")
        self.assertAlmostEqual(carrier.s, word_out(plan, 5, "s_out"), places=4)
        before = next(t for t in toks if t.text == "THE")
        self.assertAlmostEqual(before.e, word_out(plan, 3), places=4)   # "the" did not absorb it

    def test_a_lone_hidden_word_changes_nothing(self):
        plan, sclip, _t = planned(NIGHT, dict(piece(NIGHT), fix={"5": ""}))
        toks = captions.display_tokens(plan, sclip, CLEAN["caption"])
        self.assertEqual([t.text for t in toks], ["WE", "SAT", "ON", "THE", "BACK", "ALL", "NIGHT", "LONG"])
        back = toks[4]
        self.assertAlmostEqual(back.e, word_out(plan, 4), places=4)
        self.assertEqual(back.natural, "back")

    def test_a_hidden_word_across_a_cut_is_not_absorbed(self):
        a = [("We", 1.0, 1.12), ("sat", 1.12, 1.4), ("back", 1.44, 1.7)]
        b = [("deck.", 9.0, 9.5), ("All", 9.54, 9.8), ("night.", 9.84, 10.3)]
        words = a + b
        clip = {"pieces": [{"words": [0, 2]}, {"words": [3, 5]}], "fix": {"2": "back deck", "3": ""}}
        plan, sclip, _t = planned(words, clip)
        toks = captions.display_tokens(plan, sclip, CLEAN["caption"])
        carrier = next(t for t in toks if t.text == "BACK DECK")
        self.assertAlmostEqual(carrier.e, word_out(plan, 2), places=4)
        self.assertEqual(carrier.natural, "back deck")
        self.assertEqual(carrier.seg0, carrier.seg1)

    def test_clip_without_a_merge_is_untouched(self):
        # The edge words outside a piece are hidden too: they never become rows.
        plan, sclip, track = planned(NIGHT, {"pieces": [{"words": [2, 7]}]})
        toks = captions.display_tokens(plan, sclip, CLEAN["caption"])
        self.assertEqual([t.text for t in toks], ["ON", "THE", "BACK", "DECK", "ALL", "NIGHT"])
        self.assertEqual([t.idx for t in toks], [[2], [3], [4], [5], [6], [7]])


class MergeKeepsTheClosingMark(unittest.TestCase):
    """The carrier takes the hidden word's sentence or clause end: paging and the .srt see it."""

    def test_sentence_end_survives_the_merge(self):
        plan, _c, track = planned(RAIN, dict(piece(RAIN), fix={"1": "back deck", "2": ""}))
        _p, _c, control = planned(RAIN, piece(RAIN))
        self.assertEqual(texts(track), ["THE BACK DECK", "IT RAINED", "WE WENT INSIDE"])
        self.assertEqual(texts(track), texts(control))
        self.assertEqual(track.pages[1]["break"], "hard:sentence")
        srt = " ".join(t for _a, _b, t in track.srt_cues()).replace("\n", " ")
        self.assertIn("back deck. It rained.", srt)

    def test_clause_mark_is_kept(self):
        words = [(("deck," if w == "deck." else w), s, e) for w, s, e in RAIN]
        plan, sclip, track = planned(words, dict(piece(words), fix={"1": "back deck", "2": ""}))
        toks = captions.display_tokens(plan, sclip, CLEAN["caption"])
        self.assertEqual(toks[1].natural, "back deck,")
        self.assertIn("back deck, It rained.", " ".join(t for _a, _b, t in track.srt_cues()).replace("\n", " "))

    def test_a_retyped_mark_is_never_doubled(self):
        for typed, want in (("back deck.", "back deck."), ("back deck!", "back deck!"),
                            ("back deck,", "back deck,"), ('back deck"', 'back deck."')):
            plan, sclip, _t = planned(RAIN, dict(piece(RAIN), fix={"1": typed, "2": ""}))
            toks = captions.display_tokens(plan, sclip, CLEAN["caption"])
            self.assertEqual(toks[1].natural, want, typed)

    def test_the_last_hidden_word_decides_the_mark(self):
        words = [("In", 1.0, 1.1), ("New", 1.14, 1.3), ("York,", 1.34, 1.6), ("City", 1.64, 2.0),
                 ("it", 2.04, 2.2), ("rained.", 2.24, 2.7)]
        clip = dict(piece(words), fix={"1": "New York City", "2": "", "3": ""})
        plan, sclip, _t = planned(words, clip)
        toks = captions.display_tokens(plan, sclip, CLEAN["caption"])
        self.assertEqual([t.natural for t in toks], ["In", "New York City", "it", "rained."])

    def test_abbreviation_is_not_a_sentence_end(self):
        words = [("Ask", 1.0, 1.2), ("the", 1.24, 1.34), ("Dr.", 1.38, 1.6), ("now.", 1.64, 2.1)]
        plan, sclip, _t = planned(words, dict(piece(words), fix={"1": "the doctor", "2": ""}))
        toks = captions.display_tokens(plan, sclip, CLEAN["caption"])
        self.assertEqual([t.natural for t in toks], ["Ask", "the doctor", "now."])

    def test_helpers(self):
        self.assertEqual(captions._closing_mark("deck."), ".")
        self.assertEqual(captions._closing_mark('deck?!"'), "?!")
        self.assertEqual(captions._closing_mark("deck"), "")
        self.assertEqual(captions._closing_mark("U.S."), "")
        self.assertEqual(captions._take_mark("back deck", "."), "back deck.")
        self.assertEqual(captions._take_mark("back deck)", ","), "back deck,)")
        self.assertEqual(captions._take_mark("back deck;", "."), "back deck;")
        self.assertEqual(captions._take_mark("back deck", ""), "back deck")


# ----------------------------------------------------------------------------- min_page back-fill

NO_TIGHTEN = {"tighten": {"enabled": False}}
LATE = [("We", 1.0, 1.15), ("shipped", 1.15, 1.5), ("it", 1.5, 1.6), ("on", 1.6, 1.72),
        ("time.", 1.72, 2.2), ("No.", 3.9, 4.06), ("We", 4.1, 4.25), ("shipped", 4.25, 4.6),
        ("it", 4.6, 4.7), ("three", 4.7, 5.0), ("weeks", 5.0, 5.3), ("late.", 5.3, 5.8)]


class BackfillIsCapped(unittest.TestCase):
    """min_page grows a short page backward by at most BACKFILL_MAX before its first word."""

    def short_page(self, preset):
        plan, _c, track = planned(LATE, piece(LATE), defaults=NO_TIGHTEN, preset=preset)
        cap = track.to_json()
        page = next(p for p in cap["pages"] if p["text"].upper().startswith("NO"))
        return plan, cap, page

    def test_page_never_shows_long_before_its_first_word(self):
        for preset in (CLEAN, MINIMAL, POP):
            plan, cap, page = self.short_page(preset)
            first = page["tokens"][0]["s_out"]
            early = first - page["start_out"]
            self.assertLessEqual(early, captions.BACKFILL_MAX + FRAME + 1e-6, preset["name"])   # was 0.66
            self.assertGreater(early, cap["lead_ms"] / 1000.0 + 0.1, preset["name"])           # still back-fills

    def test_the_gate_no_longer_fails_the_page(self):
        from hookcutter import qa as qamod
        plan, cap, page = self.short_page(CLEAN)
        spans = [(w["s_out"], w["e_out"]) for w in plan["words"]
                 if not w.get("edge") and w["s_out"] is not None]
        E = TP.energy(plan["duration_out"], spans)
        gate = qamod.g10_sync(qamod._Ctx(None, {"clips": {}}, None, False, False), plan, cap, None, None,
                              cap["lead_ms"] / 1000.0, "vertical/clean", E=E)
        self.assertNotEqual(gate["status"], "FAIL", gate)
        self.assertNotIn("the voice starts", " ".join(gate.get("details") or []))
        # the page that stays short is reported the documented way
        self.assertIn("min_page", [i["code"] for i in cap["issues"]])

    def test_scheduler_and_gate_cannot_drift_apart(self):
        from hookcutter import qa as qamod
        leads = [float(json.loads(f.read_text()).get("caption", {}).get("lead_ms", 0)) / 1000.0
                 for f in sorted((SK / "presets").glob("*.json")) if f.name != "safezones.json"]
        self.assertTrue(leads)
        self.assertLess(captions.BACKFILL_MAX + max(leads) + FRAME, qamod.G10_PAGE_EARLY)

    def test_a_short_back_fill_is_unchanged(self):
        # 0.3 s of free time before a 0.5 s page: the cap does not bite.
        words = [("We", 1.0, 1.15), ("shipped.", 1.15, 1.6), ("No.", 2.4, 2.56), ("We", 2.6, 2.75),
                 ("did", 2.75, 2.95), ("not.", 2.95, 3.4)]
        _p, _c, track = planned(words, piece(words), defaults=NO_TIGHTEN)
        page = next(p for p in track.pages if p["text"].startswith("NO"))
        self.assertAlmostEqual(page["start_out"], track.pages[page["i"] - 1]["end_out"], places=6)
        self.assertLess(page["tokens"][0]["s_out"] - page["start_out"], captions.BACKFILL_MAX)


# ----------------------------------------------------------------------------- build pill and ink

def timed(*words):
    """[(text, {"s", "e"})] at a steady pace, for TC.make_plan."""
    out, t = [], 1.0
    for w in words:
        out.append((w, {"s": round(t, 2), "e": round(t + 0.3, 2)}))
        t += 0.4
    return out


def pop_preset(**emphasis):
    over = {"caption": {"backing": {"mode": "always"}}}
    if emphasis:
        over["caption"]["emphasis"] = emphasis
    return captions._deep_merge(POP, over)


def quiet(plan, preset, clip=None):
    with redirect_stderr(io.StringIO()):
        return TC.build(plan, preset, clip=clip)


def ink_outside(track, thr=26):
    """Worst count of pixels (alpha >= 10 %) outside the pill drawn on the same frame, over
    every frame of every backed page. With a pill there is no shadow, so all of it is glyph ink."""
    worst = 0
    for page in track._pages:
        if not page.backing:
            continue
        for n in range(page.start_f, page.end_f):
            key = track.frame_key(n)
            if key[0] != "page":
                continue
            box = track._build_pill(page, key[2]) if track._reveal == "build" else page.pill
            if box is None:
                continue
            x, y, w, h = box
            a = np.frombuffer(track.frame(n), np.uint8).reshape(track.band_h, TC.W, 4)[..., 3].copy()
            y0 = y - track.band_y
            a[max(0, y0):y0 + h, x:x + w] = 0
            worst = max(worst, int((a >= thr).sum()))
    return worst


def contains(pill, box, margin=4):
    x, y, w, h = pill
    return (x <= box[0] - margin + 1e-6 and y <= box[1] - margin + 1e-6
            and x + w >= box[2] + margin - 1e-6 and y + h >= box[3] + margin - 1e-6)


class BuildPillFollowsTheInk(unittest.TestCase):
    """Build reveal: the pill that has grown to a token covers the ink of everything shown."""

    CASES = (
        ("number", lambda: pop_preset(), timed("we", "made", "$70,000"), None),
        ("big emphasis", lambda: pop_preset(size_mult=1.5), timed("HUGE", "win"), {"emphasis": [0]}),
        ("accent", lambda: pop_preset(sources=[]), timed("we", "love", "JOSÉ"), None),
        ("accent, emphasised", lambda: pop_preset(), timed("we", "love", "JOSÉ"), {"emphasis": [2]}),
    )

    def test_every_token_pill_holds_the_ink_shown_so_far(self):
        for label, preset, words, clip in self.CASES:
            track = quiet(TC.make_plan(words), preset(), clip)
            setter = track._setter
            for page in track._pages:
                seen = []
                for k, (tok, runs) in enumerate(zip(page.tokens, page.runs)):
                    font = setter.font(page.size, bool(tok.emph))
                    seen += [captions._ink_box(font, text, setter.tracking, x, b) for text, x, b in runs]
                    self.assertTrue(contains(page.token_pills[k], captions._union_box(seen)),
                                    (label, k, page.token_pills[k], captions._union_box(seen)))
                self.assertEqual(page.token_pills[-1], page.pill, label)   # what captions.json reports

    def test_no_glyph_pixel_outside_the_pill_on_any_frame(self):
        for label, preset, words, clip in self.CASES:
            track = quiet(TC.make_plan(words), preset(), clip)
            self.assertEqual(ink_outside(track), 0, label)       # the pop peak frame included

    def test_plain_capitals_keep_their_pills(self):
        track = quiet(TC.make_plan(timed("we", "made", "HUGE")), pop_preset(sources=[]))
        page = track._pages[0]
        self.assertEqual(page.token_pills, [(161, 1208, 207, 84), (161, 1208, 491, 84), (161, 1208, 758, 84)])
        self.assertEqual(page.pill, (161, 1208, 758, 84))

    def test_fill_reveal_is_untouched(self):
        # clean and minimal draw page.pill: same box as before, with or without a big accent.
        pre = captions._deep_merge(CLEAN, {"caption": {"backing": {"mode": "always"}}})
        track = quiet(TC.make_plan(timed("we", "love", "JOSÉ")), pre)
        page = track._pages[0]
        self.assertEqual(page.pill, captions._pill_box(page.typo, page.box, pre["caption"]["backing"]["pad"],
                                                       TC.W, TC.H))
        self.assertEqual(ink_outside(track), 0)

    def test_the_pill_stays_inside_the_band(self):
        for label, preset, words, clip in self.CASES:
            track = quiet(TC.make_plan(words), preset(), clip)
            for page in track._pages:
                for x, y, w, h in [b for b in page.token_pills if b]:
                    self.assertGreaterEqual(y, track.band_y, label)
                    self.assertLessEqual(y + h, track.band_y + track.band_h, label)


# ----------------------------------------------------------------------------- a leading dot

class LeadingDotWords(unittest.TestCase):
    """A word that starts with a dot glues only as the rest of a split number ("3" + ".5")."""

    def shown(self, words, clip=None, **cap):
        cap.setdefault("case", "sentence")
        cap.setdefault("strip_punct", "")
        plan = TC.make_plan(list(words))
        for w in plan["words"]:
            fixv = ((clip or {}).get("fix") or {}).get(str(w["i"]))
            if fixv is not None:
                w["text"], w["hidden"] = fixv, fixv == ""
        return [t.text for t in captions.display_tokens(plan, clip or {}, TC.style(**cap)["caption"])]

    def test_a_dot_word_after_a_word_stands_alone(self):
        for words, want in (
                (["Only", ".5%", "of", "founders"], ["Only", ".5%", "of", "founders"]),   # was "Only.5%"
                (["a", ".22", "rifle"], ["a", ".22", "rifle"]),
                (["bought", "a", ".com", "domain"], ["bought", "a", ".com", "domain"]),
                (["built", "on", ".NET", "back", "then"], ["built", "on", ".NET", "back", "then"])):
            self.assertEqual(self.shown(words), want)

    def test_split_numbers_and_punctuation_still_glue(self):
        for words, want in (
                (["It", "was", "3", ".5", "percent"], ["It", "was", "3.5", "percent"]),
                (["made", "$70", ",000", "a", "year"], ["made", "$70,000", "a", "year"]),
                (["we", "grew", "80", "%", "."], ["we", "grew", "80%."]),
                (["wait", "...", "what"], ["wait...", "what"]),
                (["wait", "...and", "then"], ["wait...and", "then"]),
                (['"3"', ".5", "x"], ['"3".5', "x"])):
            self.assertEqual(self.shown(words), want)

    def test_a_typed_fix_follows_the_same_rule(self):
        clip = {"fix": {"1": ".5%", "2": "", "3": ""}}
        self.assertEqual(self.shown(["Only", "half", "a", "percent"], clip), ["Only", ".5%"])

    def test_the_dot_survives_strip_punct(self):
        # clean strips "." and ",": "ONLY 5%" would be ten times the number that was said.
        self.assertEqual(self.shown(["Only", ".5%", "of", "them."], case="upper", strip_punct=".,"),
                         ["ONLY", ".5%", "OF", "THEM"])
        self.assertEqual(self.shown(["on", ".NET,", "then"], case="upper", strip_punct=".,"),
                         ["ON", ".NET", "THEN"])
        self.assertEqual(self.shown(["wait", "...and", "U.S.,", "end."], strip_punct=".,"),
                         ["wait...and", "U.S.", "end"])
        self.assertEqual(captions._strip_display("...and", ".,"), "and")
        self.assertEqual(captions._strip_display(".5.", ".,"), ".5")
        self.assertEqual(captions._strip_display(".", ".,"), "")

    def test_rule_matches_the_claim_reader(self):
        # textnorm reads numbers off the same words for the lints and G11: the two must agree
        # on when ".<digit>" continues a number.
        import inspect
        from hookcutter import textnorm
        if "prev" not in inspect.signature(textnorm._glues_prev).parameters:
            self.skipTest("textnorm._glues_prev takes no previous token here")
        for text, prev in ((".5", "3"), (".5%", "Only"), (".5", "3)"), (".22", "a")):
            self.assertEqual(captions._glues_to_prev(text, prev), textnorm._glues_prev(text, prev), (text, prev))


# ----------------------------------------------------------------------------- safe zone + assets

class _TmpProject(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="hc-codex-fb-")
        self.P = Path(self._tmp.name) / "project"
        (self.P / "presets").mkdir(parents=True)
        (self.P / "fonts").mkdir()
        self.addCleanup(self._tmp.cleanup)
        self.addCleanup(typeset._load.cache_clear)
        self.addCleanup(typeset._cmap.cache_clear)

    def preset(self, name, **body):
        (self.P / "presets" / f"{name}.json").write_text(json.dumps(dict({"extends": "clean"}, **body)))

    def zones(self, **profiles):
        (self.P / "presets" / "safezones.json").write_text(json.dumps(profiles))

    def loaded(self, name, fmt="vertical"):
        return presets.for_format(presets.load(name, self.P), fmt)

    def geometry(self, name, eff, fmt="vertical"):
        size = (1080, 1920) if fmt == "vertical" else (1920, 1080)
        style = captions.resolve_style(self.loaded(name, fmt), fmt)
        return captions._geometry(style, {"_eff": eff} if eff is not None else {}, fmt, size[0], size[1], self.P)


UNSET = {"safezone": "organic"}                            # the default, clips.json did not name one
SET_ORGANIC = {"safezone": "organic", "safezone_set": True}


class OneSafeZoneResolver(_TmpProject):
    """presets.zone_name: clips.json when it set one, else the preset's own, else organic."""

    def test_precedence(self):
        z = presets.zone_name
        ads = {"safezone": "ads"}
        self.assertEqual(z(UNSET, ads, "vertical"), "ads")                    # was organic: never read
        self.assertEqual(z(SET_ORGANIC, ads, "vertical"), "organic")
        self.assertEqual(z({"safezone": "ads", "safezone_set": True}, {"safezone": "organic"}, "vertical"), "ads")
        self.assertEqual(z(UNSET, {}, "vertical"), "organic")
        self.assertEqual(z(None, None, "vertical"), "organic")
        self.assertEqual(z({}, {"safezone": None}, "vertical"), "organic")
        self.assertEqual(z(UNSET, {"safezone": {"top": 1}}, "vertical"), "organic")   # a name, not margins
        self.assertEqual(z(UNSET, {"safezone": "organic"}, "landscape"), "landscape")
        self.assertEqual(z({"safezone": "ads", "safezone_set": True}, {"safezone": "landscape"}, "landscape"),
                         "landscape")
        self.assertEqual(z(UNSET, {"safezone": "tv"}, "landscape"), "tv")
        self.assertEqual(z({"safezone": "mine", "safezone_set": True}, {"safezone": "tv"}, "landscape"), "mine")

    def test_the_layout_uses_the_presets_own_profile(self):
        self.preset("adlook", safezone="ads")
        geo = self.geometry("adlook", UNSET)
        self.assertEqual(geo["zone_name"], "ads")
        self.assertEqual(geo["max_width"], 540.0)              # organic gives 724
        self.assertEqual(self.geometry("adlook", SET_ORGANIC)["zone_name"], "organic")
        self.assertEqual(self.geometry("adlook", None)["zone_name"], "ads")

    def test_shipped_presets_resolve_as_before(self):
        for name in ("clean", "pop", "minimal"):
            geo = self.geometry(name, UNSET)
            self.assertEqual((geo["zone_name"], geo["max_width"]), ("organic", 724.0), name)
            self.assertEqual(self.geometry(name, UNSET, "landscape")["zone_name"], "landscape", name)
        track = quiet(TC.make_plan(timed("we", "made", "it")), CLEAN, clip={"_eff": dict(UNSET)})
        self.assertEqual(track.to_json()["safezone"], "organic")

    def test_a_landscape_override_is_reachable(self):
        self.zones(tv={"top": 80, "bottom": 80, "left": 300, "right": 300})
        self.preset("wide", landscape={"safezone": "tv"})
        geo = self.geometry("wide", UNSET, "landscape")
        self.assertEqual(geo["zone_name"], "tv")
        self.assertEqual(geo["zone"]["left"], 300)
        self.assertEqual(self.geometry("wide", UNSET)["zone_name"], "organic")

    def test_a_spec_that_names_no_profile_keeps_its_keys(self):
        # Hash stability: `_eff` gains a key only when clips.json itself sets defaults.safezone.
        clip = TP.mkspec({"c01": {"pieces": [{"words": [0, 3]}]}})["clips"]["c01"]
        self.assertEqual(set(clip["_eff"]), {"styles", "formats", "tighten", "safezone", "length"})
        self.preset("adlook", safezone="ads")
        style = captions.resolve_style(self.loaded("adlook"), "vertical")
        self.assertEqual(captions._geometry(style, clip, "vertical", 1080, 1920, self.P)["zone_name"], "ads")
        named = TP.mkspec({"c01": {"pieces": [{"words": [0, 3]}]}}, {"safezone": "organic"})["clips"]["c01"]
        if not named["_eff"].get("safezone_set"):
            self.skipTest("this spec.py does not mark a safezone set in clips.json")
        self.assertEqual(captions._geometry(style, named, "vertical", 1080, 1920, self.P)["zone_name"], "organic")


class RenderAssets(_TmpProject):
    """captions.render_assets: the font bytes and safe-zone margins a render key must see."""

    def setUp(self):
        super().setUp()
        self.font = self.P / "fonts" / "Custom.ttf"
        shutil.copyfile(SK / "fonts" / "Anton-Regular.ttf", self.font)
        self.preset("custom", caption={"font": "Custom.ttf"}, hook_card={"font": "Custom.ttf"})

    def assets(self, name="custom", eff=UNSET, fmt="vertical"):
        return captions.render_assets(self.loaded(name, fmt), {"_eff": dict(eff)}, fmt, self.P)

    def test_shape(self):
        a = self.assets()
        self.assertEqual(set(a), {"fonts", "safezone"})
        self.assertEqual(list(a["fonts"]), ["Custom.ttf"])
        self.assertRegex(a["fonts"]["Custom.ttf"], r"^[0-9a-f]{64}$")
        self.assertEqual(a["safezone"], {"top": 240, "bottom": 420, "left": 90, "right": 150})
        json.dumps(a)
        self.assertEqual(self.assets(), a)                      # deterministic

    def test_swapping_the_font_file_changes_it(self):
        a = self.assets()
        shutil.copyfile(SK / "fonts" / "ArchivoBlack-Regular.ttf", self.font)
        typeset._load.cache_clear()
        typeset._cmap.cache_clear()
        b = self.assets()
        self.assertNotEqual(a["fonts"], b["fonts"])
        self.assertEqual(a["safezone"], b["safezone"])
        shutil.copyfile(SK / "fonts" / "Anton-Regular.ttf", self.font)
        self.assertEqual(self.assets(), a)

    def test_every_font_a_render_draws_with_is_listed(self):
        self.preset("three", caption={"font": "Custom.ttf", "emphasis": {"font": "Anton-Regular.ttf"}},
                    hook_card={"font": "ArchivoBlack-Regular.ttf"})
        self.assertEqual(sorted(self.assets("three")["fonts"]),
                         ["Anton-Regular.ttf", "ArchivoBlack-Regular.ttf", "Custom.ttf"])
        pop = self.assets("pop")["fonts"]
        self.assertEqual(list(pop), ["Montserrat[wght].ttf:Black"])

    def test_editing_the_margins_in_force_changes_it(self):
        a = self.assets()
        self.zones(organic={"right": 350})
        b = self.assets()
        self.assertEqual(b["safezone"], {"top": 240, "bottom": 420, "left": 90, "right": 350})
        self.assertNotEqual(a, b)

    def test_a_profile_not_in_use_changes_nothing(self):
        a = self.assets()
        self.zones(ads={"right": 400})
        self.assertEqual(self.assets(), a)
        self.assertEqual(self.assets(eff={"safezone": "ads", "safezone_set": True})["safezone"]["right"], 400)

    def test_it_follows_the_resolver(self):
        self.preset("adlook", safezone="ads")
        self.assertEqual(self.assets("adlook")["safezone"]["bottom"], 672)
        self.assertEqual(self.assets("adlook", eff=SET_ORGANIC)["safezone"]["bottom"], 420)
        self.assertEqual(self.assets("clean", fmt="landscape")["safezone"],
                         {"top": 54, "bottom": 54, "left": 96, "right": 96})

    def test_it_never_raises(self):
        self.font.unlink()
        a = self.assets()
        self.assertEqual(a["fonts"], {"Custom.ttf": "missing"})
        self.preset("lost", safezone="nowhere")
        self.assertEqual(self.assets("lost")["safezone"], "unknown")
        self.zones(organic=300)
        self.assertEqual(self.assets("clean")["safezone"], "unknown")
        for bad in (None, {}, {"caption": None}, {"caption": {"font": 5}}, "clean"):
            out = captions.render_assets(bad, None, "vertical", self.P)
            self.assertEqual(set(out), {"fonts", "safezone"}, bad)


# ----------------------------------------------------------------------------- preset types

class PresetTypes(_TmpProject):
    """presets.validate: a wrong type or an explicit null stops the preset loading."""

    REJECTED = (
        ({"caption": None}, "caption"),
        ({"hook_card": None}, "hook_card"),
        ({"caption": {"layout": None}}, "caption.layout"),
        ({"caption": {"anim": None}}, "caption.anim"),
        ({"caption": {"size": None}}, "caption.size"),
        ({"caption": {"size": "big"}}, "caption.size"),
        ({"caption": {"lead_ms": None}}, "caption.lead_ms"),
        ({"caption": {"lead_ms": False}}, "caption.lead_ms"),
        ({"caption": {"layout": {"max_words": "four"}}}, "caption.layout.max_words"),
        ({"caption": {"layout": {"max_width": "wide"}}}, "caption.layout.max_width"),
        ({"caption": {"layout": {"max_lines": True}}}, "caption.layout.max_lines"),
        ({"caption": {"backing": {"pad": 28}}}, "caption.backing.pad"),
        ({"hook_card": {"backing": {"pad": [28]}}}, "hook_card.backing.pad"),
        ({"hook_card": {"max_size": None}}, "hook_card.max_size"),
        ({"hook_card": {"fade_out_ms": None}}, "hook_card.fade_out_ms"),
        ({"landscape": "wide"}, "landscape"),
        ({"landscape": {"caption": None}}, "landscape.caption"),
        ({"landscape": {"caption": {"lead_ms": None}}}, "landscape.caption.lead_ms"),
        ({"landscape": {"caption": {"layout": {"max_width": "wide"}}}}, "landscape.caption.layout.max_width"),
        ({"motion": "off"}, "motion"),
        ({"motion": {"punch_in": "off"}}, "motion.punch_in"),
        ({"motion": {"punch_in": True}}, "motion.punch_in"),
        ({"motion": {"min_gap": [3]}}, "motion.min_gap"),
        ({"motion": {"max_upscale": float("nan")}}, "motion.max_upscale"),
        ({"letterbox": {"blur": "soft"}}, "letterbox.blur"),
        ({"letterbox": [820]}, "letterbox"),
        ({"landscape": {"motion": {"punch_in": "off"}}}, "landscape.motion.punch_in"),
    )
    ACCEPTED = (
        {"landscape": None}, {"motion": None}, {"letterbox": None}, {"cover": None},
        {"caption": {"backing": None}}, {"caption": {"shadow": None}}, {"caption": {"emphasis": None}},
        {"caption": {"outline": None}}, {"caption": {"layout": {"y": None}}},
        {"caption": {"layout": {"max_words": None}}}, {"caption": {"backing": {"pad": None}}},
        {"caption": {"backing": {"pad": [30, 16]}}}, {"motion": {"punch_in": None}},
        {"motion": {"punch_in": 1.08, "_note": "x"}}, {"letterbox": {"y_center": 800, "blur": 20.5}},
        {"caption": {"size": 72.0, "lead_ms": 0}}, {"safezone": None},
    )

    def test_wrong_types_stop_the_preset_loading(self):
        for over, key in self.REJECTED:
            self.preset("bad", **over)
            with self.assertRaises(HCError, msg=over) as cm:
                presets.load("bad", self.P)
            self.assertEqual(cm.exception.code, 1, over)
            self.assertIn(" %s must be " % key, cm.exception.msg, over)

    def test_null_names_the_way_out(self):
        self.preset("bad", caption=None)
        with self.assertRaises(HCError) as cm:
            presets.load("bad", self.P)
        self.assertIn("delete the key to inherit", cm.exception.msg)

    def test_what_the_renderer_tolerates_still_loads(self):
        for over in self.ACCEPTED:
            self.preset("fine", **over)
            presets.load("fine", self.P)
        (self.P / "presets" / "root.json").write_text(json.dumps({"label": "Root"}))   # no caption key at all
        self.assertEqual(presets.load("root", self.P)["name"], "root")
        for name in ("clean", "pop", "minimal"):
            presets.load(name, self.P)

    def test_validate_changes_nothing(self):
        before = presets.load("pop")
        after = json.loads(json.dumps(before))
        presets.validate(after, "pop")
        self.assertEqual(after, json.loads(json.dumps(before)))

    def test_rejected_presets_no_longer_reach_the_renderer(self):
        # Each of these used to load and then die inside the caption track with a traceback.
        plan = TC.make_plan(timed("we", "sold", "forty", "units"), hook={"lines": ["WE SOLD *FORTY*"], "dur": 1.5})
        for over in ({"caption": None}, {"hook_card": None}, {"caption": {"anim": None}},
                     {"caption": {"backing": {"pad": 28}}}, {"hook_card": {"fade_out_ms": None}}):
            self.preset("bad", **over)
            with self.assertRaises(HCError, msg=over):
                pre = presets.for_format(presets.load("bad", self.P), "vertical")
                with redirect_stderr(io.StringIO()):
                    captions.build_track(plan, {}, pre, "vertical", (1080, 1920), (30, 1), None,
                                         project_root=self.P)

    def test_a_bad_style_is_one_plan_time_failure(self):
        from hookcutter import spec as specmod
        self.preset("bad", caption=None)
        (self.P / "clips.json").write_text(json.dumps(
            {"version": 1, "defaults": {"styles": ["bad"]}, "clips": {"c01": {"pieces": [{"words": [0, 3]}]}}}))
        sp = specmod.load_spec(self.P)
        fails = [i for i in sp["clips"]["c01"].get("_issues") or [] if i["id"] == "L01" and i["level"] == "FAIL"]
        self.assertEqual(len(fails), 1, fails)
        self.assertIn("style 'bad' cannot be used", fails[0]["msg"])
        self.assertIn("caption must be an object", fails[0]["msg"])

    def test_a_safezone_profile_must_be_an_object(self):
        self.zones(organic=300)
        with self.assertRaises(HCError) as cm:
            presets.safezone("organic", self.P)
        self.assertEqual(cm.exception.code, 1)
        self.assertIn("must be an object", cm.exception.msg)
        with self.assertRaises(HCError):
            presets.safezone(["top"], self.P)
        self.zones(organic={"right": 200})
        self.assertEqual(presets.safezone("organic", self.P)["right"], 200)
        self.assertEqual(presets.safezone({"top": 5}, self.P), {"top": 5, "bottom": 0, "left": 0, "right": 0})


if __name__ == "__main__":
    unittest.main()
