"""Lint L21: speech the transcript may have folded into a neighbouring word or dropped.

Seen on a 75-minute 720p remote interview with no speaker labels: a restart folded into one long
word token ("I'm in a, I'm in a terminal" stamped as one 0.40 s 'a'), a filler with no token in a
voiced gap between two words ("CEO, like, I'm"), and short spoken stretches with no words, under
the 1.5 s that L15 and L19 look at. No lint or gate saw them; only slice transcriptions did. The
energy tracks here are synthetic: speech at -20 dB over a -60 dB floor."""
from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "scripts"))

import numpy as np  # noqa: E402

from fixtures import fake_transcript  # noqa: E402
from hookcutter import listen  # noqa: E402
from hookcutter import plan as P  # noqa: E402
from hookcutter import spec as S  # noqa: E402
from hookcutter.audio import EnergyTrack  # noqa: E402

FLOOR = -60.0
VOICE = -20.0
BREATH = -45.0           # 25 dB under the speech level: a breath, not a word


def energy(dur, spans, floor=FLOOR, level=VOICE, hop=0.01):
    n = int(round(dur / hop))
    db = np.full(n, floor, dtype=np.float64)
    for sp in spans:
        lv = sp[2] if len(sp) > 2 else level
        db[int(round(sp[0] / hop)):int(round(sp[1] / hop))] = lv
    return EnergyTrack(db, hop=hop)


def source(dur, fps=(25, 1)):
    return {"fingerprint": "fp-test", "duration": dur,
            "video": {"fps": list(fps), "start": 0.0, "width": 1920, "height": 1080,
                      "frames": int(math.floor(dur * fps[0] / fps[1] + 1e-9)), "hdr": "none",
                      "sar": "1:1"},
            "audio": {"start": 0.0, "index": 1}}


def build(words, clip, extra=(), flags=None, dur=None):
    tr = fake_transcript(words)
    if flags:
        tr["flags"] = flags
    end = dur or (max(w[2] for w in words) + 5.0)
    E = energy(end, [(w[1], w[2]) for w in words] + list(extra))
    spec = S.normalize_spec({"version": 1, "clips": {"c01": clip}})
    return P.plan_clip(spec, "c01", tr, source(end), E, None)


def seq(texts, start=0.6, wl=0.3, gap=0.1):
    out, t = [], start
    for w in texts:
        out.append((w, round(t, 3), round(t + wl, 3)))
        t += wl + gap
    return out


def l21(plan):
    return [x for x in plan["lints"] if x["id"] == "L21"]


LEAD = ["So", "when", "I'm", "on", "my", "laptop,", "I'm", "in"]       # words 0-7, ends 3.7 s
TAIL = ["and", "I", "type", "things", "out.", "That", "is", "how", "it", "goes."]


def folded(a_len=0.45, pause_after=0.0):
    """Word 8 'a' runs a_len seconds, then 'terminal' (word 9) follows after pause_after."""
    words = seq(LEAD)
    s = 3.8
    words.append(("a", s, round(s + a_len, 3)))
    t = round(s + a_len + pause_after, 3)
    words.append(("terminal", t, round(t + 0.45, 3)))
    words += seq(TAIL, start=round(t + 0.55, 3))
    return words


class TestFoldedWord(unittest.TestCase):
    """(a) a word token far longer than its text allows, running straight into the next word."""

    def test_overlong_word_running_into_the_next_warns(self):
        words = folded()
        plan = build(words, {"pieces": [{"words": [0, len(words) - 1]}]})
        x = l21(plan)
        self.assertEqual(len(x), 1, plan["lints"])
        self.assertEqual(x[0]["level"], "WARN")
        msg = x[0]["msg"]
        for part in ("'a'", "word 8", "0:03.80", "0.45 s", "'terminal'", "hc listen P",
                     "its caption"):
            self.assertIn(part, msg)
        self.assertAlmostEqual(x[0]["at_out"], P.out_time(plan, 3.8), places=2)

    def test_ordinary_or_drawn_words_do_not_warn(self):
        # an 'a' of ordinary length, and a long 'a' followed by a pause (a drawn hesitation)
        for words in (folded(a_len=0.20), folded(pause_after=0.40)):
            plan = build(words, {"pieces": [{"words": [0, len(words) - 1]}]})
            self.assertEqual(l21(plan), [], plan["lints"])

    def test_numbers_count_as_spoken_words(self):
        # "190" is said "one ninety": 0.6 s for three digits is not long
        words = seq(["I", "made", "about"]) + [("190", 1.8, 2.4), ("decisions", 2.4, 2.9)] + \
            seq(["that", "day."], start=3.0)
        plan = build(words, {"pieces": [{"words": [0, len(words) - 1]}]})
        self.assertEqual(l21(plan), [], plan["lints"])

    def test_a_fix_on_the_word_settles_it(self):
        words = folded()
        plan = build(words, {"pieces": [{"words": [0, len(words) - 1]}], "fix": {"8": "a"}})
        self.assertEqual(l21(plan), [], plan["lints"])

    def test_an_insert_over_the_word_settles_it(self):
        # the editor typed what the slice heard inside the word's time
        words = folded()
        piece = [{"words": [0, len(words) - 1]}]
        ins = {"after": 7, "text": "a, I'm in", "src": [3.8, 4.0]}
        plan = build(words, {"pieces": piece, "insert": [ins]})
        self.assertEqual(l21(plan), [], plan["lints"])
        # an insert somewhere else answers nothing here
        ins = {"after": 10, "text": "well,", "src": [5.1, 5.2]}
        plan = build(words, {"pieces": piece, "insert": [ins]})
        self.assertEqual(len(l21(plan)), 1, plan["lints"])

    def test_an_in_edge_past_the_first_attempt_still_needs_the_heard_text(self):
        # editing.md rule 6: open on the folded word and push the in-edge past the first
        # attempt. What is left can still hold more than the token's text (a restart stamped
        # 'built' as "I did a, I built": the edge past "I did a," leaves "I built"), so the
        # edge alone does not settle it; a fix saying what is heard from the edge on does.
        words = folded()
        last = len(words) - 1
        piece = [{"words": [8, last], "in_offset": 6}]
        plan = build(words, {"pieces": piece})
        self.assertGreater(plan["segments"][0]["src"][0], 3.8)
        self.assertEqual(len(l21(plan)), 1, plan["lints"])
        plan = build(words, {"pieces": piece, "fix": {"8": "I'm in a"}})
        self.assertEqual(l21(plan), [], plan["lints"])

    def test_a_flagged_word_is_still_checked(self):
        # a false_start flag can name one restart while the word holds more: L15 and L21 both
        words = folded()
        flag = {"kind": "false_start", "a": 8, "b": 8, "s": 3.8, "e": 4.0}
        plan = build(words, {"pieces": [{"words": [0, len(words) - 1]}]}, flags=[flag])
        self.assertTrue([y for y in plan["lints"] if y["id"] == "L15"], plan["lints"])
        self.assertEqual(len(l21(plan)), 1, plan["lints"])

    def test_word_just_outside_the_in_edge_is_checked(self):
        # the piece opens on 'terminal'; the folded 'a' right before the edge is still named
        words = folded()
        plan = build(words, {"pieces": [{"words": [9, len(words) - 1]}]})
        x = l21(plan)
        self.assertEqual(len(x), 1, plan["lints"])
        # the word is not in the clip, so it has no caption to get wrong
        self.assertNotIn("caption", x[0]["msg"])
        # nor has a word the clip omits
        plan = build(words, {"pieces": [{"words": [0, len(words) - 1]}], "omit": [8]})
        x = l21(plan)
        self.assertEqual(len(x), 1, plan["lints"])
        self.assertNotIn("caption", x[0]["msg"])
        # a folded word far from every edge is not this clip's business
        plan = build(words, {"pieces": [{"words": [12, len(words) - 1]}]})
        self.assertEqual(l21(plan), [], plan["lints"])


def voiced_gap(gap=0.26, level=-24.0):
    """'CEO.' ends at 2.0 s; the next word starts `gap` later; the gap is voiced at `level`."""
    words = seq(["I'm", "playing"], start=0.8) + [("CEO.", 1.6, 2.0)]
    t = round(2.0 + gap, 3)
    words += seq(["I'm", "an", "actor", "in", "a", "play.", "Then", "it", "ends."], start=t)
    return words, [(2.0, t, level)]


class TestVoicedGap(unittest.TestCase):
    """(b) an inter-word gap voiced at speech level across most of its length."""

    def test_voiced_gap_warns(self):
        words, extra = voiced_gap()
        plan = build(words, {"pieces": [{"words": [0, len(words) - 1]}]}, extra=extra)
        x = l21(plan)
        self.assertEqual(len(x), 1, plan["lints"])
        msg = x[0]["msg"]
        for part in ("0.26 s", "'CEO.'", "\"I'm\"", "0:02.00", "hc listen P"):
            self.assertIn(part, msg)

    def test_breath_and_short_gaps_do_not_warn(self):
        for gap, level in ((0.26, BREATH), (0.14, -24.0), (0.26, FLOOR)):
            words, extra = voiced_gap(gap, level)
            plan = build(words, {"pieces": [{"words": [0, len(words) - 1]}]}, extra=extra)
            self.assertEqual(l21(plan), [], (gap, level, plan["lints"]))

    def test_an_insert_or_a_fix_on_the_next_word_settles_it(self):
        words, extra = voiced_gap()
        piece = [{"words": [0, len(words) - 1]}]
        for key in ({"insert": [{"after": 2, "text": "like,", "src": [2.04, 2.2]}]},
                    {"fix": {"3": "like, I'm"}}):
            plan = build(words, dict({"pieces": piece}, **key), extra=extra)
            self.assertEqual(l21(plan), [], (key, plan["lints"]))
        # a fix on another word answers nothing here
        plan = build(words, {"pieces": piece, "fix": {"5": "actor"}}, extra=extra)
        self.assertEqual(len(l21(plan)), 1, plan["lints"])


def wordless(run=(3.62, 4.02), level=-26.0):
    """'I' ends at 3.5 s, 'expected' starts at 4.4 s: 0.9 s of gap with a run of sound in it."""
    words = seq(["It's", "not", "exactly", "what"], start=1.9) + [("I", 3.4, 3.5)]
    words += seq(["expected", "it", "to", "be.", "And", "then", "I", "moved", "on."], start=4.4)
    return words, [(run[0], run[1], level)]


class TestWordlessSound(unittest.TestCase):
    """(c) speech-level sound with no words, between 0.3 s and L19's 1.5 s."""

    def test_short_wordless_speech_warns(self):
        words, extra = wordless()
        plan = build(words, {"pieces": [{"words": [0, len(words) - 1]}]}, extra=extra)
        x = l21(plan)
        self.assertEqual(len(x), 1, plan["lints"])
        msg = x[0]["msg"]
        for part in ("0.4 s", "0:03.62", "'I'", "'expected'", "hc listen P"):
            self.assertIn(part, msg)
        self.assertFalse([y for y in plan["lints"] if y["id"] == "L19"])

    def test_breath_and_tiny_sounds_do_not_warn(self):
        for run, level in (((3.62, 4.02), BREATH), ((3.7, 3.9), -26.0)):
            words, extra = wordless(run, level)
            plan = build(words, {"pieces": [{"words": [0, len(words) - 1]}]}, extra=extra)
            self.assertEqual(l21(plan), [], (run, level, plan["lints"]))

    def test_long_wordless_speech_stays_with_l19(self):
        words = seq(["It's", "not", "exactly", "what"], start=1.9) + [("I", 3.4, 3.5)]
        words += seq(["expected", "it", "to", "be."], start=6.0)
        plan = build(words, {"pieces": [{"words": [0, len(words) - 1]}]},
                     extra=[(3.6, 5.8, -26.0)])
        self.assertTrue([y for y in plan["lints"] if y["id"] == "L19"], plan["lints"])
        self.assertEqual(l21(plan), [], plan["lints"])

    def test_settled_by_insert_or_fix_or_reported_by_l15(self):
        words, extra = wordless()
        clip = {"pieces": [{"words": [0, len(words) - 1]}],
                "insert": [{"after": 4, "text": "wanted", "src": [3.62, 4.02]}]}
        self.assertEqual(l21(build(words, clip, extra=extra)), [])
        clip = {"pieces": [{"words": [0, len(words) - 1]}], "fix": {"5": "wanted, expected"}}
        self.assertEqual(l21(build(words, clip, extra=extra)), [])
        flag = {"kind": "unworded_audio", "s": 3.62, "e": 4.02, "a": 4, "b": 5}
        plan = build(words, {"pieces": [{"words": [0, len(words) - 1]}]}, extra=extra,
                     flags=[flag])
        self.assertEqual(l21(plan), [], plan["lints"])
        self.assertTrue([y for y in plan["lints"] if y["id"] == "L15"])

    def test_sound_just_past_the_out_edge_is_checked(self):
        words = seq(["It's", "not", "exactly", "what", "I", "expected", "it", "to", "be."],
                    start=1.0)
        clip = {"pieces": [{"words": [0, 8]}]}
        end = build(words, clip)["segments"][-1]["src"][1]
        near = build(words, clip, extra=[(end + 0.05, end + 0.45, -26.0)])
        self.assertEqual(near["segments"][-1]["src"][1], end)
        x = l21(near)
        self.assertEqual(len(x), 1, near["lints"])
        # the slice it asks for holds the sound, though no word follows it
        self.assertIn(f"-{S.fmt_time(end + 0.45)} ", x[0]["msg"])
        far = build(words, clip, extra=[(end + 0.8, end + 1.2, -26.0)])
        self.assertEqual(l21(far), [], far["lints"])

    def test_a_long_stretch_partly_outside_the_clip_says_how_much_is_inside(self):
        # 1.6 s of sound, 1.3 s of it inside a time piece: L19 measures only the inside part
        words = seq(["It's", "not", "exactly", "what", "I", "expected", "it", "to", "be."],
                    start=1.0)
        plan = build(words, {"pieces": [{"time_src": [0.9, 6.0]}]}, extra=[(4.7, 6.3, -26.0)])
        self.assertEqual(plan["segments"][-1]["src"][1], 6.0)
        self.assertFalse([y for y in plan["lints"] if y["id"] == "L19"], plan["lints"])
        x = l21(plan)
        self.assertEqual(len(x), 1, plan["lints"])
        msg = x[0]["msg"]
        self.assertIn("1.6 s of sound", msg)
        self.assertIn("1.30 s of it inside the clip", msg)
        self.assertNotIn("L19 starts", msg)
        self.assertIn("0:04.20-0:06.30", msg)


class TestWhereItPoints(unittest.TestCase):
    """at_out for a stretch outside every segment: the nearest edge in the source, also when a
    later passage plays first (a cold open)."""

    WORDS = seq(["One", "two", "three", "four", "five.", "Six", "seven", "eight", "nine",
                 "ten."], start=2.0)
    COLD = {"pieces": [{"words": [5, 9]}, {"words": [0, 4]}]}

    def test_sound_before_the_in_edge_of_the_second_piece(self):
        plan = build(self.WORDS, self.COLD, extra=[(1.35, 1.8, -26.0)])
        x = l21(plan)
        self.assertEqual(len(x), 1, plan["lints"])
        second = plan["segments"][1]
        self.assertLess(second["src"][0], plan["segments"][0]["src"][0])
        self.assertAlmostEqual(x[0]["at_out"], second["out"][0], places=3)

    def test_sound_past_the_out_edge_of_the_cold_open(self):
        end = build(self.WORDS, self.COLD)["segments"][0]["src"][1]
        plan = build(self.WORDS, self.COLD, extra=[(end + 0.05, end + 0.45, -26.0)])
        x = l21(plan)
        self.assertEqual(len(x), 1, plan["lints"])
        first = plan["segments"][0]
        self.assertAlmostEqual(x[0]["at_out"], first["out"][1], places=3)
        self.assertLess(x[0]["at_out"], plan["duration_out"] - 1.0)


class TestPointsAtHcListen(unittest.TestCase):
    """Each WARN prints the `hc listen` command for its stretch (from the word before to the word
    after, 0.2 s of margin each side), the same check L15 names: the old recipe wrote a scratch
    transcript into P, and "slice-transcribe" named no command at all."""

    def check(self, plan, lo, hi):
        x = l21(plan)
        self.assertEqual(len(x), 1, plan["lints"])
        msg = x[0]["msg"]
        self.assertIn(f"hc listen P {lo} {hi}", msg)
        self.assertIn("writes nothing into P", msg)
        self.assertNotIn("slice-transcribe", msg)
        self.assertLessEqual(float(hi) - float(lo), listen.LISTEN_MAX + listen.LIMIT_EPS)

    def test_a_folded_word(self):
        words = folded()                 # 'in' 3.40 .. 'terminal' ends 4.70
        self.check(build(words, {"pieces": [{"words": [0, len(words) - 1]}]}), "3.20", "4.90")

    def test_a_voiced_gap(self):
        words, extra = voiced_gap()      # 'CEO.' 1.60 .. "I'm" ends 2.56
        plan = build(words, {"pieces": [{"words": [0, len(words) - 1]}]}, extra=extra)
        self.check(plan, "1.40", "2.76")

    def test_wordless_sound(self):
        words, extra = wordless()        # 'I' 3.40 .. 'expected' ends 4.70
        plan = build(words, {"pieces": [{"words": [0, len(words) - 1]}]}, extra=extra)
        self.check(plan, "3.20", "4.90")

    def test_the_start_is_never_before_the_file(self):
        words = [("a", 0.1, 0.55), ("terminal", 0.55, 1.0)] + seq(TAIL, start=1.1)
        self.check(build(words, {"pieces": [{"words": [0, len(words) - 1]}]}), "0.00", "1.20")


class TestCleanSpeech(unittest.TestCase):
    def test_clean_speech_has_no_l21(self):
        texts = ("This is how it works when nothing is wrong. Every word has its own sound, "
                 "the pauses are quiet, and none of it runs long.").split()
        plan = build(seq(texts), {"pieces": [{"words": [0, len(texts) - 1]}]})
        self.assertEqual(l21(plan), [], plan["lints"])


if __name__ == "__main__":
    unittest.main()
