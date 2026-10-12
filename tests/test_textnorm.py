"""textnorm: normalization, numbers, contractions, sentence ends, glue, word lists."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from hookcutter import textnorm as tn  # noqa: E402


class TestNorm(unittest.TestCase):
    def test_norm_single_tokens(self):
        cases = {
            "Don't": "dont", "\u201cHello,\u201d": "hello", "It\u2019s": "its", "$70K": "70000",
            "70,000": "70000", "1.5M": "1500000", "80%": "80", "Seventy": "70",
            "thousand": "1000", "full-time": "fulltime", "...": "", "3rd": "third",
            "1st": "first", "Caf\u00e9": "caf\u00e9", "U.S.": "us", "2,400.": "2400",
        }
        for raw, want in cases.items():
            self.assertEqual(tn.norm(raw), want, raw)

    def test_norm_tokens(self):
        self.assertEqual(tn.norm_tokens("I've got $70K, full-time!"),
                         ["ive", "got", "70000", "dollars", "full", "time"])
        self.assertEqual(tn.norm_tokens("It grew 80% at 10:30"),
                         ["it", "grew", "80", "percent", "at", "10", "30"])
        self.assertEqual(tn.norm_tokens(["R&D", "&", "AI"]), ["r", "d", "and", "ai"])
        self.assertEqual(tn.norm_tokens("COVID-19 hit"), ["covid", "19", "hit"])
        self.assertEqual(tn.norm_tokens(""), [])

    def test_norm_tokens_expand(self):
        self.assertEqual(tn.norm_tokens("I'm gonna say it's John's, can't won't", expand=True),
                         ["i", "am", "going", "to", "say", "it", "is", "johns", "can", "not",
                          "will", "not"])
        # the same sentence written two ways normalizes identically
        self.assertEqual(tn.norm_tokens("We do not know", expand=True),
                         tn.norm_tokens("We don\u2019t know", expand=True))

    def test_spoken_numbers_collapse(self):
        cases = {
            "seventy thousand": ["70000"], "70 thousand": ["70000"],
            "one point five million": ["1500000"], "a hundred": ["100"],
            "one hundred and five": ["105"], "twenty five": ["25"],
            "two thousand twenty six": ["2026"], "twenty twenty six": ["20", "26"],
            "one two three": ["1", "2", "3"], "70 grand": ["70000"],
            "$2 million": ["2000000", "dollars"], "a grand piano": ["a", "grand", "piano"],
            "a dog": ["a", "dog"],
        }
        for text, want in cases.items():
            self.assertEqual(tn.norm_tokens(text), want, text)

    def test_norm_stream_backpointers(self):
        raws = ["It", "cost", "seventy", "thousand", "dollars.", "Wow"]
        stream = tn.norm_stream(raws)
        self.assertEqual(stream, [("it", 0, 0), ("cost", 1, 1), ("70000", 2, 3),
                                  ("dollars", 4, 4), ("wow", 5, 5)])
        stream = tn.norm_stream(["$2", "million", "now"])
        self.assertEqual(stream, [("2000000", 0, 1), ("dollars", 0, 1), ("now", 2, 2)])
        # ASR split tokens are glued before normalizing; the span covers every piece
        self.assertEqual(tn.norm_stream(["It", "cost", "$2", ",400", "or", "80", "%", "full",
                                         "-time", "$", "5"]),
                         [("it", 0, 0), ("cost", 1, 1), ("2400", 2, 3), ("dollars", 2, 3),
                          ("or", 4, 4), ("80", 5, 6), ("percent", 5, 6), ("full", 7, 8),
                          ("time", 7, 8), ("5", 9, 10), ("dollars", 9, 10)])
        self.assertEqual(tn.extract_numbers(["$2", ",400", "and", "80", "%"]), [2400.0, 80.0])
        # contraction expansion keeps the source index on every piece
        self.assertEqual(tn.norm_stream(["Don't", "go"], expand=True),
                         [("do", 0, 0), ("not", 0, 0), ("go", 1, 1)])


class TestNumbers(unittest.TestCase):
    def test_num_value_forms(self):
        cases = {
            "$70K": 70000, "70,000": 70000, "seventy thousand": 70000, "70 thousand": 70000,
            "1.5M": 1.5e6, "80%": 80, "$2,400": 2400, "one point five million": 1.5e6,
            "a hundred": 100, "70 grand": 70000, "$1.2B": 1.2e9, "2.5k": 2500,
            "I made $70K last year": 70000,
        }
        for text, want in cases.items():
            self.assertAlmostEqual(tn.num_value(text), want, msg=text)
        self.assertAlmostEqual(tn.num_value(["seventy", "thousand"]), 70000)
        self.assertIsNone(tn.num_value("nothing here"))
        self.assertIsNone(tn.num_value(""))

    def test_extract_numbers(self):
        self.assertEqual(tn.extract_numbers("$70K and 80% in 3 years"), [70000.0, 80.0, 3.0])

    def test_fmt_number(self):
        self.assertEqual(tn.fmt_number(70000.0), "70000")
        self.assertEqual(tn.fmt_number(1.5), "1.5")
        self.assertEqual(tn.fmt_number(0.25), "0.25")


class TestContractions(unittest.TestCase):
    def test_expand(self):
        self.assertEqual(tn.expand_contractions(["I'm", "gonna"]), ["i", "am", "going", "to"])
        self.assertEqual(tn.expand_contractions(["dont", "its", "Well", "I\u2019ll"]),
                         ["do", "not", "its", "well", "i", "will"])
        self.assertEqual(tn.expand_contractions("they're here, aren't they?"),
                         ["they", "are", "here", "are", "not", "they"])
        self.assertEqual(tn.expand_contractions(["can't"]), ["can", "not"])
        self.assertEqual(tn.expand_contractions(["cannot"]), ["can", "not"])
        self.assertEqual(tn.expand_contractions(["John's"]), ["johns"])
        self.assertEqual(tn.expand_contractions(["that's"]), ["that", "is"])


class TestPredicates(unittest.TestCase):
    def test_is_filler(self):
        for w in ("um", "Uh,", "hmmm", "Umm...", "er", "erm", "mhm", "Ahh"):
            self.assertTrue(tn.is_filler(w), w)
        for w in ("uh-huh", "oh", "like", "a", "hum", "um-um-um-yes", "", "em"):
            self.assertFalse(tn.is_filler(w), w)

    def test_is_sentence_end(self):
        yes = ("end.", "why?", "wow!)", '"done."', "3.", "no.", "it?!", "etc.")
        no = ("e.g.", "i.e.", "Mr.", "Dr.", "3.5", "so...", "so\u2026", "J.", "U.S.", "word",
              "list:", "", "word,")
        for w in yes:
            self.assertTrue(tn.is_sentence_end(w), w)
        for w in no:
            self.assertFalse(tn.is_sentence_end(w), w)
        self.assertFalse(tn.is_sentence_end("etc.", "and"))
        self.assertTrue(tn.is_sentence_end("etc.", "And"))

    def test_word_predicates(self):
        self.assertTrue(tn.is_negation("Don't"))
        self.assertTrue(tn.is_negation("never"))
        self.assertTrue(tn.is_negation("isn\u2019t"))
        self.assertFalse(tn.is_negation("know"))
        self.assertTrue(tn.is_hedge("probably"))
        self.assertTrue(tn.is_hedge("Almost,"))
        self.assertTrue(tn.is_continuation("And"))
        self.assertTrue(tn.is_continuation("so,"))
        self.assertFalse(tn.is_continuation("Today"))
        self.assertTrue(tn.is_pronoun_opener("That's"))
        self.assertTrue(tn.is_pronoun_opener("They"))
        self.assertFalse(tn.is_pronoun_opener("I"))
        self.assertTrue(tn.is_stopword("the"))
        self.assertFalse(tn.is_stopword("product"))

    def test_match_preamble(self):
        self.assertEqual(tn.match_preamble("Yeah, so the thing is"), "yeah so")
        self.assertEqual(tn.match_preamble("Great question. So"), "great question")
        self.assertEqual(tn.match_preamble("Um, you know what I think"), "you know what")
        self.assertEqual(tn.match_preamble(["Well,", "here", "goes"]), "well")
        self.assertIsNone(tn.match_preamble("Right now we are building"))
        self.assertIsNone(tn.match_preamble("Nobody tells you this"))
        self.assertIsNone(tn.match_preamble("Wellness is a market"))

    def test_word_lists(self):
        self.assertGreaterEqual(len(tn.STOPWORDS), 150)
        self.assertFalse(tn.STOPWORDS & tn.NEGATIONS, "negations must never be stopwords")
        for name in ("FILLERS", "STOPWORDS", "NEGATIONS", "HEDGES", "CONTINUATIONS",
                     "PRONOUN_OPENERS"):
            for w in getattr(tn, name):
                self.assertEqual(w, w.lower(), (name, w))
                self.assertNotIn("'", w, (name, w))
                self.assertEqual(tn.norm(w), w, (name, w))


class TestOverlap(unittest.TestCase):
    def test_overlap_coefficient(self):
        self.assertAlmostEqual(tn.overlap_coefficient("Afraid of looking stupid",
                                                      "Are you afraid to look stupid?"), 1.0)
        self.assertEqual(tn.overlap_coefficient("", "anything"), 0.0)
        self.assertEqual(tn.overlap_coefficient("the and of", "a an"), 0.0)
        self.assertAlmostEqual(tn.overlap_coefficient(["a", "b"], ["b", "c", "d"]), 0.5)
        self.assertLess(tn.overlap_coefficient("money buys freedom", "pricing is hard"), 0.5)

    def test_stem(self):
        self.assertEqual(tn.stem("looking"), "look")
        self.assertEqual(tn.stem("stopped"), "stop")
        self.assertEqual(tn.stem("costs"), "cost")
        self.assertEqual(tn.stem("glass"), "glass")
        self.assertEqual(tn.stem("70000"), "70000")


class TestGlue(unittest.TestCase):
    def _words(self, texts, key="w"):
        return [{key: t, "i": i, "s": float(i), "e": i + 0.5} for i, t in enumerate(texts)]

    def test_glue_rules(self):
        texts = ["It", "cost", "$", "2", ",400", "and", "80", "%", "full", "-time", "COVID",
                 "-19", "do", "n't", '"', "...and", "what", "?", "Google", ".com", "it", "'s"]
        got = [(g["text"], g["idx"]) for g in tn.glue_tokens(self._words(texts))]
        self.assertEqual(got, [
            ("It", [0]), ("cost", [1]), ("$2,400", [2, 3, 4]), ("and", [5]), ("80%", [6, 7]),
            ("full-time", [8, 9]), ("COVID-19", [10, 11]), ('don\'t"', [12, 13, 14]),
            ("...and", [15]), ("what?", [16, 17]), ("Google.com", [18, 19]), ("it's", [20, 21])])

    def test_glue_times_and_keys(self):
        words = [{"text": "$", "idx": [7], "s_out": 1.0, "e_out": 1.1},
                 {"text": "5", "idx": [8], "s_out": 1.1, "e_out": 1.3},
                 {"text": "%", "idx": [9], "s_out": 1.3, "e_out": 1.5},
                 {"text": "", "idx": [10], "s_out": 1.5, "e_out": 1.6},
                 {"text": "now", "idx": [11], "s_out": 1.6, "e_out": 1.9}]
        got = tn.glue_tokens(words)
        self.assertEqual([(g["text"], g["idx"], g["s"], g["e"]) for g in got],
                         [("$5%", [7, 8, 9], 1.0, 1.5), ("now", [11], 1.6, 1.9)])
        self.assertEqual(len(got[0]["parts"]), 3)

    def test_glue_edges(self):
        # a glue-prev token at the start stands alone; a trailing "$" stands alone
        got = [g["text"] for g in tn.glue_tokens(self._words([",so", "money", "$"]))]
        self.assertEqual(got, [",so", "money", "$"])
        # uppercase after a hyphen is a separate word (contract rule: "-" + lowercase)
        got = [g["text"] for g in tn.glue_tokens(self._words(["wait", "-Then"]))]
        self.assertEqual(got, ["wait", "-Then"])
        # explicit keys
        got = tn.glue_tokens([{"raw": "a", "i": 3, "t0": 1.0, "t1": 2.0}], text_key="raw",
                             time_keys=("t0", "t1"))
        self.assertEqual((got[0]["text"], got[0]["idx"], got[0]["s"], got[0]["e"]),
                         ("a", [3], 1.0, 2.0))


if __name__ == "__main__":
    unittest.main()
