"""find: fuzzy phrase search, hit formatting, the words table and both commands."""

from __future__ import annotations

import argparse
import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from hookcutter import env, find, transcribe as T  # noqa: E402

TEXT = ("Alright, so today we talk pricing. I believe that you look stupid when you start, "
        "and that is fine. Um, we made $70K in the first year with Greptile. "
        "Later I said it again: I believe that you look stupid when you start. "
        "We don't know yet. Thank you. the the the the end.")


def make_words(text=TEXT, step=0.4):
    words = []
    for k, t in enumerate(text.split()):
        words.append({"i": k, "w": t, "s": round(k * step, 3), "e": round(k * step + 0.3, 3),
                      "p": 0.9, "sp": "Speaker A" if k < 30 else "Speaker B"})
    return words


def idx_of(words, token, nth=0):
    hits = [w["i"] for w in words if w["w"] == token]
    return hits[nth]


class TestSearch(unittest.TestCase):
    def setUp(self):
        self.words = make_words()

    def test_exact_two_occurrences(self):
        hits = find.search(self.words, "I believe that you look stupid")
        self.assertEqual(len(hits), 2)
        first = idx_of(self.words, "believe") - 1
        self.assertEqual((hits[0]["a"], hits[0]["b"]), (first, first + 5))
        self.assertEqual(hits[0]["score"], 1.0)
        self.assertEqual(hits[0]["s"], self.words[first]["s"])
        self.assertEqual(hits[0]["e"], self.words[first + 5]["e"])

    def test_near_prefers_closer_occurrence(self):
        second = idx_of(self.words, "believe", 1) - 1
        hits = find.search(self.words, "I believe that you look stupid",
                           near=self.words[second]["s"])
        self.assertEqual(hits[0]["a"], second)

    def test_fuzzy_spelling_filler_numbers_contractions(self):
        hits = find.search(self.words, "with reptile")          # ASR misspelling
        g = idx_of(self.words, "Greptile.")
        self.assertEqual((hits[0]["a"], hits[0]["b"]), (g - 1, g))
        self.assertGreater(hits[0]["score"], 0.75)
        hits = find.search(self.words, "we made seventy thousand dollars")
        self.assertEqual(hits[0]["a"], idx_of(self.words, "made") - 1)
        self.assertEqual(hits[0]["b"], idx_of(self.words, "$70K"))
        self.assertEqual(hits[0]["score"], 1.0)
        hits = find.search(self.words, "fine we made")          # "Um," between is ignored
        self.assertEqual(hits[0]["score"], 1.0)
        hits = find.search(self.words, "we do not know")       # contraction expanded
        self.assertEqual(hits[0]["a"], idx_of(self.words, "don't") - 1)
        self.assertEqual(hits[0]["score"], 1.0)

    def test_split_asr_tokens(self):
        words = make_words("it cost $2 ,400 a month and 80 % of people paid")
        hits = find.search(words, "two thousand four hundred dollars a month")
        self.assertEqual((hits[0]["a"], hits[0]["b"], hits[0]["score"]), (2, 5, 1.0))
        hits = find.search(words, "80% of people")
        self.assertEqual((hits[0]["a"], hits[0]["b"], hits[0]["score"]), (7, 10, 1.0))

    def test_dropped_word_still_matches(self):
        hits = find.search(self.words, "so today we really talk pricing")
        self.assertEqual(hits[0]["a"], idx_of(self.words, "so"))
        self.assertLess(hits[0]["score"], 1.0)
        self.assertAlmostEqual(hits[0]["score"], (5 - find.GAP_PHRASE) / 6, places=3)

    def test_no_match_and_bad_phrase(self):
        self.assertEqual(find.search(self.words, "quantum chromodynamics lecture"), [])
        with self.assertRaises(env.HCError):
            find.search(self.words, "um ...")
        self.assertEqual(find.search([], "anything"), [])

    def test_max_hits(self):
        self.assertEqual(len(find.search(self.words, "you look stupid", max_hits=1)), 1)


class TestFormatting(unittest.TestCase):
    def test_format_hit(self):
        words = make_words()
        hit = find.search(words, "we made seventy thousand dollars")[0]
        line = find.format_hit(words, hit)
        self.assertRegex(line, r"^#\d+-\d+ \[\d\d:\d\d\.\d-\d\d:\d\d\.\d\] \(1\.00\) Speaker [AB]: ")
        self.assertIn(">>we made $70K<<", line)
        self.assertNotIn("\u2013", line)

    def test_word_rows(self):
        words = make_words()
        # a repetition loop needs evidence (one repeat with p < 0.5 or in silence): real speech
        # repeats words too, so a confident "the the the the" alone is not flagged
        words[idx_of(words, "the", 2)]["p"] = 0.3
        doc = {"words": words, "flags": T.find_loops(words)}
        um = idx_of(words, "Um,")
        rows = find.word_rows(doc, um - 1, um + 1)
        self.assertEqual(len(rows), 4)
        self.assertIn("speaker", rows[0])
        self.assertRegex(rows[2], r"^\s*%d\s+\S+\s+\S+\s+\S+\s+0\.100\s+0\.90\s+F\s" % um)
        loop_i = idx_of(words, "the", 1)
        self.assertRegex(find.word_rows(doc, loop_i, loop_i)[1], r"\sL\s")
        self.assertIn(" - ", find.word_rows(doc, 0, 0)[1])


class TestCommands(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.P = Path(self.tmp.name) / "project"
        self.P.mkdir()
        words = make_words()
        words[idx_of(words, "the", 2)]["p"] = 0.3    # evidence for the loop flag
        env.write_json(self.P / "transcript.json", {"version": 1, "words": words,
                                                    "lines": T.build_lines(words),
                                                    "flags": T.find_loops(words)})

    def tearDown(self):
        self.tmp.cleanup()

    def _run(self, fn, **kw):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = fn(argparse.Namespace(project=str(self.P), **kw))
        return rc, out.getvalue(), err.getvalue()

    def test_cmd_find(self):
        rc, out, _ = self._run(find.cmd_find, phrase="I believe that", near="00:25", max=None)
        self.assertEqual(rc, 0)
        lines = out.strip().splitlines()
        self.assertEqual(len(lines), 2)
        self.assertTrue(lines[0].startswith("#"))
        rc, out, err = self._run(find.cmd_find, phrase="zebra crossing xylophone", near=None, max=3)
        self.assertEqual(rc, 1)
        self.assertIn("no match", err)

    def test_cmd_words(self):
        rc, out, _ = self._run(find.cmd_words, a="10", b="#12")
        self.assertEqual(rc, 0)
        rows = out.strip().splitlines()
        self.assertEqual(len(rows), 4)
        # a reversed range is a user error (it used to be swapped silently)
        with self.assertRaises(env.HCError) as cm:
            self._run(find.cmd_words, a="12", b="#10")
        self.assertEqual(cm.exception.code, 1)
        rc, out, _ = self._run(find.cmd_words, a="0", b="99999")
        self.assertEqual(rc, 0)
        with self.assertRaises(env.HCError):
            self._run(find.cmd_words, a="5000", b="6000")
        with self.assertRaises(env.HCError):
            self._run(find.cmd_words, a="x", b="2")


if __name__ == "__main__":
    unittest.main()
