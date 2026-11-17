"""Regression tests for caption defects found on a 75-minute 720p remote interview (host and guest
in two fixed tiles, a transcript with no speaker labels):

  1  a page that ends a frame or two before a cut, or before the next page, holds to it: no
     caption-free frame right before a seam
  2  a hook-card accent broken over two lines ("THE *BANK*" / "*ACCOUNT*") is one accent where
     the words are said together, so emphasis.per_page 1 lights both
  3  a word whose onset is under the hook card is never captioned (the .srt still carries it),
     and plan lint L08 judges an accent the same way
  4  an insert that starts at the same moment as the next transcript word plays before it, all
     of its words together
  5  the .srt capitalises a cue that opens a piece on a new sentence (never a word led by a digit)

Every test is synthetic (generated plans and words)."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "scripts"))

from hookcutter import captions  # noqa: E402

import test_captions as TC  # noqa: E402  (helpers: make_plan, build, CLEAN)
import test_fix2_fa as FA  # noqa: E402  (helpers: build (a real plan), lints)
import test_fix2_fb as FB  # noqa: E402  (helpers: seam_plan, spoken, crossings, drawn_wrong_side)

CLEAN = TC.CLEAN


def lit(track):
    return [t["text"] for p in track.pages for t in p["tokens"] if t["emph"]]


def codes(track):
    return [i["code"] for i in track.to_json()["issues"]]


# ----------------------------------------------------------------------------- 1. seam blink

class NoBlankFrameBeforeASeam(unittest.TestCase):
    def around(self, plan, track, cut):
        before = [p for p in track.pages if p["frames"][0] < cut][-1]
        after = [p for p in track.pages if p["frames"][0] >= cut][0]
        return before, after

    def test_a_page_ending_a_frame_before_the_cut_holds_to_it(self):
        # 24 fps: the last word before the cut ends at 8.45 s, so the 0.4 s tail ends the page at
        # 8.85 s (its last frame is 212) while the cut is on frame 214: frame 213 had no caption,
        # and the next page starts on the cut.
        rows = [("we", 7.5, 7.8), ("grow", 7.8, 8.1), ("through", 8.1, 8.25),
                ("practice.", 8.25, 8.45)]
        plan = FB.seam_plan([("start", 214, rows),
                             ("piece", 96, FB.spoken("Then it changed.", start=0.0))], fps=(24, 1))
        track = TC.build(plan, fps=(24, 1))
        before, after = self.around(plan, track, 214)
        self.assertEqual(before["frames"][1], 214, before)
        self.assertEqual(after["frames"][0], 214, after)
        self.assertEqual(track.frame_key(213)[0], "page")
        self.assertEqual(track.pages[track.frame_key(213)[1]]["text"], before["text"])
        self.assertLessEqual(before["end_out"], 214 / 24.0 + 1e-9, before)
        self.assertEqual(FB.crossings(plan, track), [])
        self.assertEqual(FB.drawn_wrong_side(plan, track), [])

    def test_two_blank_frames_are_held_three_are_a_real_gap(self):
        # 30 fps, cut on frame 270 (9.0 s). A tail ending at 8.93 s leaves frames 268 and 269
        # blank: held. A tail ending at 8.89 s leaves three blank frames: a gap, as before.
        for last_end, end_f in ((8.53, 270), (8.49, 267)):
            rows = [("we", 7.3, 7.6), ("grow", 7.6, 7.9), ("through", 7.9, 8.2),
                    ("practice.", 8.2, last_end)]
            plan = FB.seam_plan([("start", 270, rows),
                                 ("piece", 90, FB.spoken("Then it changed.", start=0.0))])
            track = TC.build(plan)
            before, after = self.around(plan, track, 270)
            self.assertEqual(before["frames"][1], end_f, (last_end, before))
            self.assertEqual(after["frames"][0], 270, after)
            self.assertEqual(FB.crossings(plan, track), [])
            self.assertEqual(FB.drawn_wrong_side(plan, track), [])

    def test_never_held_over_the_cut_into_the_next_shot(self):
        # The tail would run past the cut: it still stops on the cut frame.
        rows = [("we", 8.2, 8.5), ("shipped.", 8.5, 8.9)]
        plan = FB.seam_plan([("start", 270, rows),
                             ("piece", 90, FB.spoken("Then it changed.", start=0.0))])
        track = TC.build(plan)
        before, after = self.around(plan, track, 270)
        self.assertEqual(before["frames"][1], 270, before)
        self.assertEqual(FB.crossings(plan, track), [])
        self.assertEqual(FB.drawn_wrong_side(plan, track), [])

    def test_a_speaker_change_a_frame_later_is_held_too(self):
        # No cut, but a speaker change: the hold under HOLD_GAP stops there. A one- or two-frame
        # gap between the two pages is a blink; a longer one stays.
        def plan(gap):
            return TC.make_plan([("we", {"s": 0.5, "e": 0.8, "sp": "A"}),
                                 ("shipped.", {"s": 0.8, "e": 1.5, "sp": "A"}),
                                 ("Then", {"s": 1.96 + gap, "e": 2.3, "sp": "B"}),
                                 ("what?", {"s": 2.3 + gap, "e": 2.8 + gap, "sp": "B"})])
        track = TC.build(plan(0.04))
        a, b = track.pages
        self.assertEqual(a["frames"][1], b["frames"][0], (a, b))
        track = TC.build(plan(0.3))
        a, b = track.pages
        self.assertLess(a["frames"][1], b["frames"][0] - 2, (a, b))

    def test_the_last_page_holds_to_the_last_frame(self):
        plan = FB.seam_plan([("start", 60, [("we", 0.5, 0.9), ("shipped.", 0.9, 1.53)])])
        track = TC.build(plan)
        self.assertEqual(track.pages[-1]["frames"][1], 60, track.pages[-1])


# ----------------------------------------------------------------------------- 2. accent over a line break

class AccentBrokenOverTwoLines(unittest.TestCase):
    CARD = {"lines": ["THE *BANK*", "*ACCOUNT*"], "dur": 1.5}

    def test_links_join_a_run_that_closes_a_line_to_one_that_opens_the_next(self):
        self.assertEqual(captions.accent_links(["THE *BANK*", "*ACCOUNT*"]), {(0, 1)})
        self.assertEqual(captions.accent_links(["*A*", "*B*", "*C*"]), {(0, 1), (1, 2)})
        self.assertEqual(captions.accent_links(["AFRAID OF *LOOKING STUPID*", "*$70K* *DONE*"]),
                         {(0, 1)})
        self.assertEqual(captions.accent_links(["THE *BANK*,", "*ACCOUNT*"]), set())
        self.assertEqual(captions.accent_links(["*BIG* AND", "*BOLD*"]), set())
        self.assertEqual(captions.accent_links(["*BIG* AND *BOLD*"]), set())
        self.assertEqual(captions.accent_links(["NO ACCENT", "*HERE*"]), set())
        # the phrases themselves are unchanged: plan L08 and QA G14 count the same two
        self.assertEqual(captions.accent_phrases(["THE *BANK*", "*ACCOUNT*"]), ["BANK", "ACCOUNT"])

    def test_said_together_both_words_light_on_one_page(self):
        plan = TC.make_plan(["the", "bank", "account", "matters."], start=2.0, hook=self.CARD)
        track = TC.build(plan)
        self.assertEqual(TC.page_texts(track)[0], "THE BANK ACCOUNT")
        self.assertEqual(lit(track), ["BANK", "ACCOUNT"])
        self.assertEqual(track.to_json()["accents"]["matched"], ["BANK", "ACCOUNT"])
        self.assertNotIn("accent_unshown", codes(track))
        # a stopword or two between them still reads as one phrase ("bank, the account")
        plan = TC.make_plan(["bank,", "the", "account", "matters."], start=2.0, hook=self.CARD)
        self.assertEqual(lit(TC.build(plan)), ["BANK", "ACCOUNT"])

    def test_said_apart_they_are_two_accents_and_still_capped(self):
        # Not said together: two accents on one page, and per_page 1 keeps one. The unshown
        # accent is still reported.
        plan = TC.make_plan(["bank", "fees", "beat", "account"], start=2.0, hook=self.CARD)
        track = TC.build(plan)
        self.assertEqual(len(track.pages), 1)
        self.assertEqual(lit(track), ["BANK"])
        self.assertIn("accent_unshown", codes(track))
        # on one line, two runs stay two accents even when said together
        plan = TC.make_plan(["the", "bank", "account", "matters."], start=2.0,
                            hook={"lines": ["THE *BANK* *ACCOUNT*"], "dur": 1.5})
        self.assertEqual(lit(TC.build(plan)), ["BANK"])


# ----------------------------------------------------------------------------- 3. onset under the card

class WordStartingUnderTheCard(unittest.TestCase):
    def plan(self, and_s):
        # card 0.0-1.4 s; "And" starts at and_s and ends at 1.6, so its midpoint is after the card
        return TC.make_plan([("I", {"s": 0.3, "e": 0.5}), ("waited.", {"s": 0.5, "e": 1.0}),
                             ("And", {"s": and_s, "e": 1.6}), ("I", {"s": 1.6, "e": 1.7}),
                             ("spent", {"s": 1.7, "e": 2.0}), ("a", {"s": 2.0, "e": 2.1}),
                             ("whole", {"s": 2.1, "e": 2.4}), ("week.", {"s": 2.4, "e": 3.0})],
                            hook={"lines": ["A WHOLE *WEEK*"], "dur": 1.4})

    def test_not_captioned_even_when_most_of_it_is_after_the_card(self):
        track = TC.build(self.plan(1.33))
        shown = [t for p in track.pages for t in p["tokens"]]
        self.assertNotIn("AND", [t["text"] for t in shown], TC.page_texts(track))
        self.assertTrue(all(t["s_out"] >= 1.4 - 1e-9 for t in shown), shown)
        self.assertEqual(track.pages[0]["text"].split()[0], "I")
        # the sidecar still carries every word, the ones under the card included, with no fix
        with tempfile.TemporaryDirectory() as d:
            text = track.write_srt(Path(d) / "c.srt").read_text(encoding="utf-8")
        self.assertIn("I waited.", text)
        self.assertIn("And I spent", text)

    def test_a_word_starting_as_the_card_ends_is_captioned(self):
        track = TC.build(self.plan(1.4))
        self.assertEqual(track.pages[0]["text"].split()[0], "AND", TC.page_texts(track))

    def test_l08_says_so_for_an_accent_starting_under_the_card(self):
        def l08(start):
            words = [("so", 0.0, 0.3), ("we", 0.3, 0.6), ("reach", 0.6, 1.0),
                     ("out", 1.0, start), ("forty", start, start + 0.4),
                     ("people.", start + 0.4, start + 0.8)] + \
                [("w%d" % k, 4.0 + k * 0.4, 4.3 + k * 0.4) for k in range(30)]
            clip = {"pieces": [{"words": [0, 35]}],
                    "hook_card": {"lines": ["*40* PEOPLE"], "dur": 2.5}}
            plan = FA.build(words, {"c01": clip}, dur=20.0)
            w = [x for x in plan["words"] if x["text"] == "forty"][0]
            self.assertAlmostEqual(w["s_out"], start, delta=0.05)
            return [x["msg"] for x in FA.lints(plan, "L08")]
        # "forty" starts 0.1 s before the card ends; its midpoint is 0.1 s after
        self.assertTrue(any("only under the hook card" in m for m in l08(2.4)))
        self.assertFalse(any("only under the hook card" in m for m in l08(2.6)))


# ----------------------------------------------------------------------------- 4. insert on a tie

class InsertTiedWithTheNextWord(unittest.TestCase):
    # The transcript folded "like" into the start of "helping": both start at 1.6 s.
    WORDS = [("about", 1.0, 1.3), ("helping", 1.6, 2.0), ("people.", 2.05, 2.5),
             ("Then", 3.4, 3.7), ("more.", 3.8, 4.2)]

    def plan(self, src):
        clip = {"pieces": [{"words": [0, 4]}],
                "insert": [{"after": 0, "text": "like", "src": src}]}
        return FA.build(self.WORDS, {"c01": clip}, dur=8.0), clip

    def test_an_insert_plays_before_a_word_that_starts_with_it(self):
        plan, clip = self.plan([1.6, 1.75])
        self.assertEqual(FA.lints(plan, "L01"), [])
        self.assertEqual([w["text"] for w in plan["words"]],
                         ["about", "like", "helping", "people.", "Then", "more."])
        toks = captions.display_tokens(plan, clip, CLEAN["caption"])
        self.assertEqual([t.text for t in toks], ["ABOUT", "LIKE", "HELPING", "PEOPLE", "THEN", "MORE"])
        # "helping" is heard once "like" ends, so its caption starts there
        by = {w["text"]: w for w in plan["words"]}
        self.assertAlmostEqual(by["helping"]["s_out"], by["like"]["e_out"], places=4)
        self.assertGreaterEqual(by["helping"]["e_out"], by["helping"]["s_out"])

    def test_a_word_that_starts_first_still_plays_first(self):
        plan, clip = self.plan([1.65, 1.8])
        self.assertEqual([w["text"] for w in plan["words"]],
                         ["about", "helping", "like", "people.", "Then", "more."])
        by = {w["text"]: w for w in plan["words"]}
        self.assertLess(by["helping"]["s_out"], by["like"]["s_out"])   # its own start, unmoved

    def test_the_word_an_insert_follows_keeps_its_start(self):
        # An insert that starts with its own `after` word goes after it, and that word's caption
        # stays where the transcript put it.
        base = FA.build(self.WORDS, {"c01": {"pieces": [{"words": [0, 4]}]}}, dur=8.0)
        clip = {"pieces": [{"words": [0, 4]}],
                "insert": [{"after": 0, "text": "really", "src": [1.0, 1.2]}]}
        plan = FA.build(self.WORDS, {"c01": clip}, dur=8.0)
        self.assertEqual([w["text"] for w in plan["words"]][:3], ["about", "really", "helping"])
        self.assertEqual(plan["words"][0]["s_out"], base["words"][0]["s_out"])
        toks = captions.display_tokens(plan, clip, CLEAN["caption"])
        self.assertEqual([t.text for t in toks][:3], ["ABOUT", "REALLY", "HELPING"])

    def test_a_multi_word_insert_tied_with_the_next_word_stays_together(self):
        # Only the insert's first word starts with "helping"; the others start after it, and the
        # plan must not slip "helping" in between them (QA G5 expects the insert as one run).
        from hookcutter import qa
        for text, src, said in [
                ("kind of like", [1.6, 1.95], ["kind", "of", "like"]),
                ("you know", [1.6, 1.9], ["you", "know"])]:
            with self.subTest(text=text):
                clip = {"pieces": [{"words": [0, 4]}],
                        "insert": [{"after": 0, "text": text, "src": src}]}
                plan = FA.build(self.WORDS, {"c01": clip}, dur=8.0)
                self.assertEqual(FA.lints(plan, "L01"), [])
                want = ["about"] + said + ["helping", "people.", "Then", "more."]
                self.assertEqual([w["text"] for w in plan["words"]], want)
                toks = captions.display_tokens(plan, clip, CLEAN["caption"])
                self.assertEqual([t.text for t in toks],
                                 [w.upper().rstrip(".") for w in want])
                self.assertEqual([t for _i, t in qa.intent_sequence(plan)], want)
                self.assertFalse(any("_s0" in w or "_after" in w for w in plan["words"]))
                starts = [t.s for t in toks]
                self.assertEqual(starts, sorted(starts))
                by = {w["text"]: w for w in plan["words"]}
                self.assertAlmostEqual(by["helping"]["s_out"], by[said[-1]]["e_out"], places=4)

    def test_a_multi_word_insert_after_a_word_that_starts_first_stays_after_it(self):
        clip = {"pieces": [{"words": [0, 4]}],
                "insert": [{"after": 0, "text": "kind of like", "src": [1.65, 1.95]}]}
        plan = FA.build(self.WORDS, {"c01": clip}, dur=8.0)
        self.assertEqual([w["text"] for w in plan["words"]],
                         ["about", "helping", "kind", "of", "like", "people.", "Then", "more."])


# ----------------------------------------------------------------------------- 5. srt case at a piece start

class SrtCaseAtAPieceStart(unittest.TestCase):
    def cues(self, segs):
        return [c[2] for c in TC.build(FB.seam_plan(segs)).srt_cues()]

    def test_a_piece_opening_mid_sentence_starts_its_cue_with_a_capital(self):
        cues = self.cues([("start", 90, FB.spoken("we sold it first.")),
                          ("piece", 120, FB.spoken("the point is to sell.", start=0.2))])
        self.assertEqual(cues, ["We sold it first.", "The point is to sell."])
        cues = self.cues([("start", 90, FB.spoken("we sold it first.")),
                          ("piece", 120, FB.spoken("it's going to work.", start=0.2))])
        self.assertEqual(cues[1], "It's going to work.")

    def test_a_piece_that_continues_a_sentence_or_a_mixed_case_word_is_left_alone(self):
        cues = self.cues([("start", 90, FB.spoken("We sold it first,")),
                          ("piece", 120, FB.spoken("the point is to sell.", start=0.2))])
        self.assertEqual(cues[1], "the point is to sell.")
        cues = self.cues([("start", 90, FB.spoken("We sold it first.")),
                          ("piece", 120, FB.spoken("eBook sales went up.", start=0.2))])
        self.assertEqual(cues[1], "eBook sales went up.")

    def test_a_word_led_by_a_digit_is_left_alone(self):
        # "3pm" is not a word to capitalise: "3Pm" garbles it (likewise "10x", "2nd", "$5k").
        for text in ["3pm is the time.", "10x growth came next.", "2nd place won.",
                     "$5k a month is fine.", "40-year-old men run.", "90s music is back."]:
            with self.subTest(text=text):
                cues = self.cues([("start", 90, FB.spoken("We sold it first.")),
                                  ("piece", 120, FB.spoken(text, start=0.2))])
                self.assertEqual(cues[1], text)
                self.assertEqual(self.cues([("start", 120, FB.spoken(text))])[0], text)
        self.assertEqual(captions._cap_first("'cause"), "'Cause")
        self.assertEqual(captions._cap_first("...and"), "...And")


if __name__ == "__main__":
    unittest.main()
