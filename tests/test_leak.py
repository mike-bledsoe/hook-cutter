"""Leak test: the shipped skill carries no private brand, people or machine strings.

The skill is self-contained and generic (README "Privacy" and "Built by"): only LICENSE and the
README "Built by" footer may name the author. Scans every text file under the skill root
(contents and relative paths), except LICENSE, the README "Built by" footer, fonts/ and models/.

Every forbidden term is stored only as a salted SHA-256 prefix, so this file does not itself
ship the strings it guards against: identifying terms (people, private brands) in DENY_DIGESTS,
and non-identifying ones (private colours, machine paths, publishing endpoints) with a label in
PLAIN_DIGESTS. Matching runs on normalised word n-grams (lowercase ASCII words, up to four in a
row), so "Sample-Shop", "sample shop" and "SAMPLE_SHOP" are the same phrase, and "a.b/c" is the
phrase "a b c". To forbid another term, print its digest and add it to one of the tables:

    python3 tests/test_leak.py --hash "term to forbid"
"""
from __future__ import annotations

import hashlib
import re
import sys
import unicodedata
import unittest
from pathlib import Path

SK = Path(__file__).resolve().parents[1]
SALT = "hook-cutter/leak/v1:"
MAX_NGRAM = 4
MAX_BYTES = 4 * 1024 * 1024

# Salted digests of forbidden word n-grams (see module docstring).
DENY_DIGESTS = frozenset({
    "87288968dc9235a9", "280a8bd68abb9138", "5f018b4b3850941d", "8c9cc45e0bdab423",
    "652194cc1798e7af", "017617ad0e30e99d", "b15a9b83ae6f775e", "da501ee615e48d6b",
    "d7695013ac03ab10", "19cdccf5263cad81", "dead211478377f78", "01e26e4b5babe0c1",
    "64144da5d933fbad", "a54654a284d9f508", "cb85083d7891fe62", "064680e055164f0a",
    "5b02b05f3b5c1d4f", "3436c7f8faa7a9a3", "7070f880c1e21321", "81881c51ce73c46e",
    "4c914002d175fe66", "a13202cdec01197b", "aeff1bb2c564e76c", "3512d2e0a0534fbe",
    "4c0ecf50cf8c11a2", "a669d44c7b26d9d2", "e7058633d1f40866", "5e5fd497eedf748a", "a05b219e6fcab5fd", "88fa550d34cfe409",
    "4cee72ec1081d466", "ea7b3fe7887cc560", "dc737ffaadf92f5b",
    "7a980314d5f610a4", "bdc5a5a5ed4c411e", "97b7950d1882a303", "c00c0846b113f4ce",
    "a0e7bde5e1a9a216", "eac1af1ad0b86187", "d9fb7a38780af03f", "51a355690c4f67ec",
    "1f83fdcd5d55e57e", "03742b792390f6cd", "19a6f20a291634da", "0570ed9b22c225f0",
    "73df3db3bed06fb1", "16665559868e7f68", "1be4bb959e70798a", "d53cc6137f51a178",
    "dec51c1297e32fe7", "387b87408ac7960b", "cd47a806b581e702", "a5ee63db1bc95790",
    "94dbdbb55965ce50", "97d575bfc53d4f5f", "4888ceff159145db", "d53a680e506753b5",
})
# A few forbidden words must also be caught inside longer tokens (CamelCase, joined handles).
# Every substring of these lengths is hashed for long tokens.
SUBSTRING_LENGTHS = (7, 9)
SUBSTRING_DIGESTS = frozenset({"87288968dc9235a9", "017617ad0e30e99d"})

# Non-identifying terms with a label for the report: private colours, private machine paths,
# publishing endpoints (the skill never uploads anything). Digests as above.
PLAIN_DIGESTS = {
    "f12b82ffb4d5cd3d": "a private brand colour",
    "f1903bc2d7325ca7": "a private brand colour",
    "402671889eb65c5f": "a private brand colour",
    "27395d14be4d6938": "a private folder name",
    "48fb745f554c6127": "a machine-specific ffmpeg path",
    "3fc33b24f11bc719": "a private workspace path",
    "9671d2a3689a8b0e": "a private workspace path",
    "2c6598f062307665": "a private workspace path",
    "ea1de91acbb24fd2": "a publishing endpoint",
    "0f5183a38d133c55": "a publishing endpoint",
    "fc1befc8bbbf53d9": "a publishing endpoint",
    "dcf26878672022a3": "a publishing endpoint",
    "e10825e1372f2eaa": "a publishing endpoint",
    "fe37fe54d6de763f": "a publishing endpoint",
    "64adbdcfa072b962": "an upload bucket",
    "4a49fdb4744fbd21": "an upload bucket",
    "1df83760d3ae95e6": "a publishing service",
    "d9f7c0c6622bf9b5": "a private secrets file",
}
# Colours and the service name are also caught inside longer tokens (a colour with an alpha
# byte, "#RRGGBBAA", or a "0x" prefix).
PLAIN_SUBSTRING_LENGTHS = (6, 9)
PLAIN_SUBSTRING_DIGESTS = frozenset({"f12b82ffb4d5cd3d", "f1903bc2d7325ca7", "402671889eb65c5f",
                                     "1df83760d3ae95e6"})

SKIP_DIRS = {"fonts", "models", "__pycache__", ".git", ".venv", "venv", ".mypy_cache"}
BINARY_EXTS = {".ttf", ".otf", ".woff", ".woff2", ".onnx", ".bin", ".pyc", ".npy", ".npz",
               ".jpg", ".jpeg", ".png", ".gif", ".webp", ".mp4", ".mov", ".mkv", ".webm",
               ".m4a", ".aac", ".wav", ".mp3", ".zip", ".gz", ".tar"}
FULLY_EXEMPT = {"LICENSE"}
FOOTER_RE = re.compile(r"^#{1,6}\s+Built by\b", re.M)
SELF = "tests/test_leak.py"


def norm_words(text: str) -> list:
    """Lowercase ASCII words of `text` (accents folded, everything else a separator)."""
    folded = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    return re.findall(r"[a-z0-9]+", folded.lower())


def digest(phrase: str) -> str:
    """Salted digest of a phrase after normalisation."""
    return hashlib.sha256((SALT + " ".join(norm_words(phrase))).encode()).hexdigest()[:16]


def _hash(normed: str) -> str:
    return hashlib.sha256((SALT + normed).encode()).hexdigest()[:16]


def find_denied(text: str, deny=DENY_DIGESTS, substr=SUBSTRING_DIGESTS,
                lengths=SUBSTRING_LENGTHS) -> list:
    """[(line_no, matched_text)] for every forbidden n-gram in `text`."""
    hits = []
    for line_no, line in enumerate(text.splitlines(), 1):
        words = norm_words(line)
        for i in range(len(words)):
            for n in range(1, MAX_NGRAM + 1):
                if i + n > len(words):
                    break
                gram = " ".join(words[i:i + n])
                if _hash(gram) in deny:
                    hits.append((line_no, gram))
            w = words[i]
            for size in lengths:
                if len(w) <= size:
                    continue
                for k in range(len(w) - size + 1):
                    if _hash(w[k:k + size]) in substr:
                        hits.append((line_no, w))
                        break
    return hits


def scanned_files() -> list:
    """(relative path, text) for every text file the leak rules apply to."""
    out = []
    for path in sorted(SK.rglob("*")):
        rel = path.relative_to(SK)
        if not path.is_file() or any(part in SKIP_DIRS for part in rel.parts):
            continue
        if path.suffix.lower() in BINARY_EXTS or path.stat().st_size > MAX_BYTES:
            continue
        rel_s = rel.as_posix()
        if rel_s in FULLY_EXEMPT:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        if rel_s == "README.md":
            m = FOOTER_RE.search(text)
            if m:
                text = text[:m.start()]
        out.append((rel_s, text))
    return out


class LeakTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.files = scanned_files()

    def test_scan_sees_the_skill(self):
        names = {rel for rel, _ in self.files}
        for required in ("SKILL.md", "README.md", SELF):
            self.assertIn(required, names, f"leak scan did not read {required}")
        self.assertGreaterEqual(len(self.files), 5, "leak scan read almost nothing")

    def test_matcher_self_check(self):
        canary = {digest("leak canary phrase")}
        self.assertTrue(find_denied("x LEAK-Canary_phrase y", canary, frozenset()))
        self.assertFalse(find_denied("leak canaries phrase", canary, frozenset()))
        sub = {_hash("canarys")}
        self.assertTrue(find_denied("myCanarysHandle", frozenset(), sub))
        # path and domain separators are word breaks: a path pattern is a phrase
        path = {digest("canary/opt/leak-dir")}
        self.assertTrue(find_denied("see /x/Canary/opt/leak-dir/bin", path, frozenset()))
        hexsub = {_hash("c0ffee")}
        self.assertTrue(find_denied('"#C0FFEE80"', frozenset(), hexsub, (6,)))

    def test_no_private_terms(self):
        problems = []
        for rel, text in self.files:
            for line_no, what in find_denied(rel + "\n" + text):
                where = rel if line_no == 1 else f"{rel}:{line_no - 1}"
                problems.append(f"{where}: {what!r}")
        self.assertEqual(problems, [], "private brand, people or machine strings in the "
                         "skill (only LICENSE and the README 'Built by' footer may name the "
                         "author):\n" + "\n".join(problems))

    def test_no_plain_patterns(self):
        problems = []
        deny = frozenset(PLAIN_DIGESTS)
        for rel, text in self.files:
            for line_no, what in find_denied(rel + "\n" + text, deny, PLAIN_SUBSTRING_DIGESTS,
                                             PLAIN_SUBSTRING_LENGTHS):
                where = rel if line_no == 1 else f"{rel}:{line_no - 1}"
                label = PLAIN_DIGESTS.get(_hash(what)) or next(
                    (PLAIN_DIGESTS[_hash(what[k:k + n])] for n in PLAIN_SUBSTRING_LENGTHS
                     for k in range(len(what) - n + 1) if _hash(what[k:k + n]) in PLAIN_DIGESTS),
                    "a private string")
                problems.append(f"{where}: {label}: {what!r}")
        self.assertEqual(problems, [], "\n".join(problems))

    def test_this_file_ships_no_plain_strings(self):
        """The tables above hold digests only (the reason this test file is safe to ship)."""
        text = (SK / SELF).read_text(encoding="utf-8")
        for value in list(PLAIN_DIGESTS) + list(DENY_DIGESTS):
            self.assertRegex(value, r"^[0-9a-f]{16}$")
        self.assertFalse(find_denied(text, frozenset(PLAIN_DIGESTS), PLAIN_SUBSTRING_DIGESTS,
                                     PLAIN_SUBSTRING_LENGTHS))

    def test_footer_and_license_are_the_only_author_mentions(self):
        readme = (SK / "README.md").read_text(encoding="utf-8")
        m = FOOTER_RE.search(readme)
        self.assertIsNotNone(m, "README.md needs a final 'Built by' section")
        footer = readme[m.start():]
        lines = [ln for ln in footer.splitlines() if ln.strip()]
        self.assertLessEqual(len(lines), 6, "the README footer is exempt from the leak scan, "
                             "so it must stay a short final section")
        self.assertTrue(find_denied(footer), "footer should name the author (it is the "
                        "exempt place for it)")
        self.assertIn("https://", footer)
        lic = (SK / "LICENSE").read_text(encoding="utf-8")
        self.assertIn("MIT License", lic)
        self.assertRegex(lic, r"Copyright \(c\) 2026 \S")


def _main(argv: list) -> int:
    if len(argv) >= 2 and argv[0] == "--hash":
        print(digest(" ".join(argv[1:])))
        return 0
    unittest.main(argv=[sys.argv[0]] + argv)
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
