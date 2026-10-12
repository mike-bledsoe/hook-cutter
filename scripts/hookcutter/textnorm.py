"""Token normalization, word lists and number parsing.

Shared by transcribe, find, plan (lints), captions (glue, emphasis) and qa (word diff, faithfulness).
Everything here is pure Python with no third-party imports.

Normalized form ("norm"): lowercase, NFKC, curly quotes straightened, apostrophes removed, other
punctuation removed, numbers canonical ("$70K" -> "70000", "seventy thousand" -> "70000",
"1.5M" -> "1500000", "80%" -> "80" + "percent" in token streams). Hyphens split words in token
streams ("full-time" -> "full", "time") but are simply removed by `norm` on a single token.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Iterable, Sequence

# ---------------------------------------------------------------------------
# Word lists (normalized forms: lowercase, no apostrophes)
# ---------------------------------------------------------------------------

FILLERS = frozenset({
    "um", "umm", "uh", "uhh", "uhm", "er", "err", "erm", "eh", "ehm", "ah", "ahh",
    "hm", "hmm", "mm", "mmm", "mhm",
})
# Stretched variants ("ummmm", "uhhhh", "hmmmm"). Never matches "uh-huh" (normalizes to "uhhuh").
FILLER_RE = re.compile(r"^(?:u+h+|u+m+|u+h+m+|e+r+m*|e+h+m*|h+m+|m+h?m+|a+h+)$")

NEGATIONS = frozenset({
    "not", "no", "never", "none", "nobody", "nothing", "nowhere", "neither", "nor",
    "cannot", "cant", "dont", "doesnt", "didnt", "isnt", "arent", "wasnt", "werent",
    "wont", "wouldnt", "shouldnt", "couldnt", "hasnt", "havent", "hadnt", "aint",
    "mustnt", "neednt", "shant", "mightnt", "without", "nope", "nah", "noone",
})

# Common English function words and discourse markers. A word missing from a re-transcription
# that is in this set is usually ASR noise (unstressed, merged or dropped), not a cut defect.
# Negations are deliberately NOT stopwords: they always carry meaning.
STOPWORDS = frozenset({
    # articles, determiners, quantifiers
    "a", "an", "the", "this", "that", "these", "those", "some", "any", "each", "every",
    "both", "all", "such", "other", "another", "same", "own", "much", "more", "most",
    "many", "few", "less", "lot", "lots",
    # conjunctions and subordinators
    "and", "or", "but", "so", "yet", "if", "then", "than", "because", "as", "while",
    "though", "although", "whether", "since", "until", "till", "once",
    # prepositions
    "of", "at", "by", "for", "with", "about", "against", "between", "into", "onto",
    "upon", "through", "during", "before", "after", "above", "below", "to", "from",
    "up", "down", "in", "out", "on", "off", "over", "under", "via", "per", "within",
    "around", "across", "toward", "towards",
    # adverbs of place/time/degree
    "here", "there", "when", "where", "why", "how", "again", "further", "too", "very",
    "also", "just", "now", "even", "still", "ever",
    # pronouns
    "i", "me", "my", "myself", "mine", "we", "us", "our", "ours", "ourselves", "you",
    "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she",
    "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs",
    "themselves", "what", "which", "who", "whom", "whose", "something", "anything",
    "everything", "someone", "anyone", "everyone", "somebody", "anybody", "everybody",
    # auxiliaries and modals
    "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "having", "do", "does", "did", "doing", "done", "can", "could", "will", "would",
    "shall", "should", "may", "might", "must",
    # contractions (apostrophes removed)
    "im", "ive", "id", "ill", "youre", "youve", "youll", "youd", "hes", "shes", "weve",
    "theyre", "theyve", "theyll", "theyd", "thats", "whats", "theres", "heres", "lets",
    "itll",
    # discourse markers that ASR routinely drops or merges
    "like", "well", "really", "actually", "literally", "basically", "yeah", "yes",
    "oh", "okay", "ok", "right", "know", "mean", "kind", "sort", "thing", "things",
    "get", "got", "go", "going", "gonna", "wanna", "gotta", "kinda", "sorta",
}) - NEGATIONS

# Qualifiers whose removal changes a claim ("almost", "probably", "only", "always").
HEDGES = frozenset({
    "maybe", "perhaps", "probably", "possibly", "likely", "unlikely", "might", "may",
    "could", "apparently", "arguably", "seemingly", "supposedly", "allegedly",
    "presumably", "potentially", "theoretically", "roughly", "approximately",
    "almost", "nearly", "mostly", "partly", "partially", "somewhat", "fairly",
    "rather", "quite", "slightly", "relatively", "generally", "usually", "typically",
    "often", "sometimes", "occasionally", "rarely", "seldom", "hardly", "barely",
    "scarcely", "only", "mainly", "largely", "virtually", "practically", "essentially",
    "always", "kinda", "sorta", "suppose", "guess",
})
HEDGE_PHRASES = (
    ("kind", "of"), ("sort", "of"), ("i", "think"), ("i", "guess"), ("i", "suppose"),
    ("more", "or", "less"), ("a", "little"), ("a", "bit"), ("in", "a", "way"),
    ("to", "some", "extent"), ("not", "necessarily"),
)

# Opening words that signal the clip starts mid-thought.
CONTINUATIONS = frozenset({
    "and", "but", "so", "or", "nor", "because", "cause", "cuz", "then", "also", "plus",
    "which", "whereas", "though", "although", "yet", "anyway", "anyways", "besides",
    "otherwise", "therefore", "thus", "hence", "meanwhile", "however", "still", "like",
    "as", "too", "either", "instead", "again", "whereby",
})

# Bare pronoun / demonstrative openers whose referent likely sits before the clip.
PRONOUN_OPENERS = frozenset({
    "he", "she", "it", "they", "them", "him", "her", "his", "their", "theirs", "its",
    "this", "that", "these", "those", "hes", "shes", "theyre", "theyve", "theyd",
    "theyll", "thats", "thatll", "itll", "itd", "hed",
})

# Throat-clearing openers, as regexes over the normalized token string (tokens joined by single
# spaces). Each must match at the start and end on a token boundary.
PREAMBLE_PATTERNS = (
    r"yeah(?: yeah)*(?: so)?",
    r"yes(?: so)?",
    r"yep",
    r"okay(?: so)?",
    r"ok(?: so)?",
    r"alright(?: so)?",
    r"all right(?: so)?",
    r"right(?! now| here| there| after| before| away| when| from| at| in| on| off)(?: so)?",
    r"well(?! known| being| off| done)",
    r"so yeah",
    r"so basically",
    r"and so",
    r"you know(?: what)?",
    r"i mean",
    r"honestly",
    r"basically",
    r"actually",
    r"anyway(?:s)?",
    r"oh",
    r"sure",
    r"for sure",
    r"absolutely",
    r"totally",
    r"exactly",
    r"definitely",
    r"of course",
    r"(?:thats a |what a )?(?:really )?(?:good|great) question",
    r"thank you(?: so much)?",
    r"thanks",
    r"(?:like|as) i (?:said|mentioned|was saying)",
    r"to be honest",
    r"to answer your question",
    r"going back to",
    r"speaking of",
    r"on that note",
    r"the thing is",
    r"i would say",
    r"id say",
    r"let me think",
)
PREAMBLE_RE = re.compile(r"^(?:" + "|".join(PREAMBLE_PATTERNS) + r")(?= |$)")

# Abbreviations whose trailing period does not end a sentence.
ABBREVIATIONS = frozenset({
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "mt", "vs", "v", "eg", "ie",
    "cf", "al", "approx", "fig", "vol", "dept", "gen", "gov", "sen", "rep",
    "lt", "col", "sgt", "capt", "messrs",
})
# Abbreviations that often do end a sentence: they end one unless the next token is lowercase.
SOFT_ABBREVIATIONS = frozenset({"etc", "inc", "ltd", "co", "corp", "llc"})

# ---------------------------------------------------------------------------
# Character preparation
# ---------------------------------------------------------------------------

_APOSTROPHES = str.maketrans({"\u2019": "'", "\u2018": "'", "\u02bc": "'", "`": "'",
                              "\u00b4": "'", "\u201c": '"', "\u201d": '"',
                              "\u2013": "-", "\u2014": "-", "\u2212": "-"})
_EDGE_PUNCT = "\"'()[]{}<>.,;:!?\u2026*_~^|\\"
_CURRENCY = {"$": "dollars", "\u00a3": "pounds", "\u20ac": "euros"}
_DIGIT_NUM_RE = re.compile(
    r"^(?P<cur>[$\u00a3\u20ac])?"
    r"(?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?|\.\d+)"
    r"(?P<suf>k|m|mm|b|bn|t)?"
    r"(?P<pct>%)?$"
)
_ORDINAL_RE = re.compile(r"^(\d+)(st|nd|rd|th)$")
_ORDINAL_WORDS = {1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth", 6: "sixth",
                  7: "seventh", 8: "eighth", 9: "ninth", 10: "tenth"}
_SUFFIX_MULT = {"k": 1e3, "m": 1e6, "mm": 1e6, "b": 1e9, "bn": 1e9, "t": 1e12}
_SPLIT_RE = re.compile(r"[-/:+&=\s]+")
_NON_WORD_RE = re.compile(r"[\W_]+")
_SYMBOL_WORDS = {"&": "and", "+": "plus", "=": "equals"}
_CUR_MARK = "\x00cur:"  # internal marker: a currency word that follows the number run after it

UNIT_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
    "nineteen": 19,
}
TENS_WORDS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
              "seventy": 70, "eighty": 80, "ninety": 90}
SCALE_WORDS = {"hundred": 100, "thousand": 1e3, "million": 1e6, "billion": 1e9,
               "trillion": 1e12}


def _prep(raw: str) -> str:
    """NFKC, lowercase, straighten quotes and dashes, trim whitespace."""
    return unicodedata.normalize("NFKC", str(raw)).translate(_APOSTROPHES).lower().strip()


def _strip_edges(t: str) -> str:
    """Strip edge punctuation but keep a leading currency sign and a trailing percent sign, and
    the decimal point of a number written without its zero: ".5%", "(.5%)" and "*.5%*" are 0.5
    percent, not 5. A trail-off ("...5") is still 5."""
    s = t.strip(_EDGE_PUNCT)
    if s[:1].isdigit():
        k = len(t) - len(t.lstrip(_EDGE_PUNCT))
        if k >= 1 and t[k - 1] == "." and (k < 2 or t[k - 2] != "."):
            s = "." + s
    return s


def fmt_number(v: float) -> str:
    """Canonical string for a number: integers without decimals, otherwise trimmed decimals."""
    if abs(v - round(v)) < 1e-9:
        return str(int(round(v)))
    return ("%.6f" % v).rstrip("0").rstrip(".")


def _digit_number(t: str) -> tuple[float, str | None, bool] | None:
    """Parse "$70K", "70,000", "1.5M", "80%" -> (value, currency word, percent)."""
    m = _DIGIT_NUM_RE.match(t)
    if not m:
        return None
    try:
        v = float(m.group("num").replace(",", ""))
    except ValueError:
        return None
    if m.group("suf"):
        v *= _SUFFIX_MULT[m.group("suf")]
    cur = _CURRENCY.get(m.group("cur") or "")
    return v, cur, bool(m.group("pct"))


def _is_numeric(t: str) -> bool:
    return bool(re.fullmatch(r"\d+(?:\.\d+)?", t))


# ---------------------------------------------------------------------------
# Single-token normalization
# ---------------------------------------------------------------------------

def norm(token: str) -> str:
    """Normalize one token: lowercase, no apostrophes or punctuation, canonical numbers.

    norm("Don't") -> "dont"; norm("$70K") -> "70000"; norm("Seventy") -> "70";
    norm("full-time") -> "fulltime"; norm("...") -> "".
    """
    t = _strip_edges(_prep(token))
    if not t:
        return ""
    dn = _digit_number(t)
    if dn is not None:
        return fmt_number(dn[0])
    om = _ORDINAL_RE.match(t)
    if om and int(om.group(1)) in _ORDINAL_WORDS:
        return _ORDINAL_WORDS[int(om.group(1))]
    t = _NON_WORD_RE.sub("", t.replace("'", ""))
    if t in UNIT_WORDS:
        return str(UNIT_WORDS[t])
    if t in TENS_WORDS:
        return str(TENS_WORDS[t])
    if t in SCALE_WORDS:
        return fmt_number(SCALE_WORDS[t])
    return t


# ---------------------------------------------------------------------------
# Contractions
# ---------------------------------------------------------------------------

_IRREGULAR_CONTRACTIONS = {
    "can't": ["can", "not"], "cannot": ["can", "not"], "won't": ["will", "not"],
    "shan't": ["shall", "not"], "ain't": ["is", "not"], "y'all": ["you", "all"],
    "let's": ["let", "us"], "'cause": ["because"], "'em": ["them"], "ma'am": ["madam"],
    "o'clock": ["oclock"],
}
_S_IS_STEMS = frozenset({"he", "she", "it", "that", "what", "there", "here", "who", "where",
                         "how", "when", "why", "this", "everything", "nothing", "something",
                         "everyone", "someone", "nobody", "somebody", "everybody"})
# Bare (apostrophe-less) forms that are unambiguous. "its", "well", "ill", "id", "hell", "shell",
# "wed", "were", "lets", "cause" are ambiguous and left alone.
_BARE_CONTRACTIONS = {
    "dont": ["do", "not"], "doesnt": ["does", "not"], "didnt": ["did", "not"],
    "cant": ["can", "not"], "cannot": ["can", "not"], "wont": ["will", "not"],
    "isnt": ["is", "not"], "arent": ["are", "not"], "wasnt": ["was", "not"],
    "werent": ["were", "not"], "hasnt": ["has", "not"], "havent": ["have", "not"],
    "hadnt": ["had", "not"], "wouldnt": ["would", "not"], "couldnt": ["could", "not"],
    "shouldnt": ["should", "not"], "mustnt": ["must", "not"], "neednt": ["need", "not"],
    "aint": ["is", "not"], "im": ["i", "am"], "ive": ["i", "have"],
    "youre": ["you", "are"], "youve": ["you", "have"], "youll": ["you", "will"],
    "youd": ["you", "would"], "theyre": ["they", "are"], "theyve": ["they", "have"],
    "theyll": ["they", "will"], "theyd": ["they", "would"], "weve": ["we", "have"],
    "thats": ["that", "is"], "whats": ["what", "is"], "theres": ["there", "is"],
    "heres": ["here", "is"], "hes": ["he", "is"], "shes": ["she", "is"],
    "itll": ["it", "will"], "thatll": ["that", "will"], "gonna": ["going", "to"],
    "wanna": ["want", "to"], "gotta": ["got", "to"], "kinda": ["kind", "of"],
    "sorta": ["sort", "of"], "lemme": ["let", "me"], "gimme": ["give", "me"],
    "cuz": ["because"], "coz": ["because"], "yall": ["you", "all"],
}
_SUFFIX_EXPANSIONS = (("n't", "not"), ("'re", "are"), ("'ve", "have"), ("'ll", "will"),
                      ("'m", "am"), ("'d", "would"))


def _expand_one(piece: str) -> list[str]:
    """Expand one prepared (lowercase, straight-apostrophe) piece; returns raw-ish pieces."""
    core = piece.strip(_EDGE_PUNCT.replace("'", ""))
    bare = core.strip("'")
    if core in _IRREGULAR_CONTRACTIONS:
        return list(_IRREGULAR_CONTRACTIONS[core])
    if bare in _IRREGULAR_CONTRACTIONS:
        return list(_IRREGULAR_CONTRACTIONS[bare])
    if "'" in bare:
        for suf, word in _SUFFIX_EXPANSIONS:
            if bare.endswith(suf) and len(bare) > len(suf):
                return [bare[: -len(suf)], word]
        if bare.endswith("'s") and bare[:-2] in _S_IS_STEMS:
            return [bare[:-2], "is"]
        return [piece]
    key = _NON_WORD_RE.sub("", bare)
    if key in _BARE_CONTRACTIONS:
        return list(_BARE_CONTRACTIONS[key])
    return [piece]


def expand_contractions(tokens: Iterable[str] | str) -> list[str]:
    """Expand contractions and return normalized tokens.

    Works on raw tokens ("don't", "It's") and on already-normalized ones ("dont"); only
    unambiguous apostrophe-less forms are expanded ("its", "well", "ill" are left alone).
    expand_contractions(["I'm", "gonna"]) -> ["i", "am", "going", "to"].
    """
    if isinstance(tokens, str):
        tokens = tokens.split()
    out: list[str] = []
    for tok in tokens:
        for piece in _expand_one(_prep(tok)):
            n = norm(piece)
            if n:
                out.append(n)
    return out


# ---------------------------------------------------------------------------
# Token streams (with back-pointers to source word indices)
# ---------------------------------------------------------------------------

def _number_tokens(t: str) -> list[str] | None:
    """Tokens for a digit-form number: currency marker first, then the number, then "percent"."""
    dn = _digit_number(t)
    if dn is None:
        return None
    v, cur, pct = dn
    out = [_CUR_MARK + cur] if cur else []
    out.append(fmt_number(v))
    if pct:
        out.append("percent")
    return out


def _piece_tokens(piece: str) -> list[str]:
    """Split one prepared piece into normalized sub-tokens (no number-word collapsing)."""
    t = _strip_edges(piece)
    if not t:
        return []
    if t in _SYMBOL_WORDS:
        return [_SYMBOL_WORDS[t]]
    nt = _number_tokens(t)
    if nt is not None:
        return nt
    om = _ORDINAL_RE.match(t)
    if om and int(om.group(1)) in _ORDINAL_WORDS:
        return [_ORDINAL_WORDS[int(om.group(1))]]
    out: list[str] = []
    for sub in _SPLIT_RE.split(t):
        sub = _strip_edges(sub)
        if not sub:
            continue
        nt = _number_tokens(sub)
        if nt is not None:
            out.extend(nt)
            continue
        cleaned = _NON_WORD_RE.sub("", sub.replace("'", ""))
        if cleaned:
            out.append(cleaned)
    return out


def _collapse_numbers(stream: list[tuple[str, int, int]]) -> list[tuple[str, int, int]]:
    """Collapse spoken number runs ("seventy thousand", "70 thousand", "one point five million").

    A currency marker before a number ("$2" + "million") becomes a word after the whole run:
    "$2 million" -> "2000000", "dollars".
    """
    out: list[tuple[str, int, int]] = []
    toks = [t for t, _, _ in stream]
    i, n = 0, len(stream)
    while i < n:
        tok, a, b = stream[i]
        cur = None
        if tok.startswith(_CUR_MARK):
            cur = tok[len(_CUR_MARK):]
            if i + 1 >= n:
                out.append((cur, a, b))
                i += 1
                continue
            i += 1
        parsed = _parse_number_run(toks, i)
        if parsed is None:
            if cur:
                out.append((cur, a, b))
            out.append(stream[i])
            i += 1
            continue
        value, j = parsed  # stream[i:j] consumed
        first = a if cur else stream[i][1]
        out.append((fmt_number(value), first, stream[j - 1][2]))
        if cur:
            out.append((cur, first, stream[j - 1][2]))
        i = j
    return out


def _parse_number_run(toks: Sequence[str], i: int) -> tuple[float, int] | None:
    """Parse a number starting at toks[i]. Returns (value, end index exclusive) or None."""
    n = len(toks)
    t0 = toks[i]
    is_word = t0 in UNIT_WORDS or t0 in TENS_WORDS
    is_digit = _is_numeric(t0)
    starts_a = t0 == "a" and i + 1 < n and toks[i + 1] in SCALE_WORDS
    if not (is_word or is_digit or starts_a or t0 in SCALE_WORDS):
        return None
    total = 0.0
    current = 0.0
    last = None  # "unit" | "tens" | "teen" | "hundred" | "scale" | "digit"
    j = i
    consumed_any = False
    while j < n:
        t = toks[j]
        if t == "a" and j == i and starts_a:
            current, last, j = 1.0, "unit", j + 1
            consumed_any = True
            continue
        if _is_numeric(t):
            if last is not None:
                break
            current, last = float(t), "digit"
        elif t in UNIT_WORDS:
            u = UNIT_WORDS[t]
            if last in ("unit", "teen", "digit"):
                break
            if last == "tens" and u >= 10:
                break
            if last == "tens" and u == 0:
                break
            current += u
            last = "teen" if u >= 10 else "unit"
        elif t in TENS_WORDS:
            if last in ("unit", "teen", "tens", "digit"):
                break
            current += TENS_WORDS[t]
            last = "tens"
        elif t == "hundred":
            if last in ("hundred", "scale") and current == 0:
                break
            if last == "hundred":
                break
            current = (current or 1.0) * 100
            last = "hundred"
        elif t in SCALE_WORDS:
            scale = SCALE_WORDS[t]
            if last == "scale" and current == 0:
                break
            total += (current or 1.0) * scale
            current = 0.0
            last = "scale"
        elif t == "grand" and last in ("unit", "teen", "tens", "digit", "hundred"):
            total += current * 1000
            current = 0.0
            last = "scale"
        elif t == "and" and last in ("hundred", "scale") and j + 1 < n and (
                toks[j + 1] in UNIT_WORDS or toks[j + 1] in TENS_WORDS):
            j += 1
            continue
        elif t == "point" and last in ("unit", "teen", "tens", "digit", "hundred") and j + 1 < n and (
                toks[j + 1] in UNIT_WORDS and UNIT_WORDS[toks[j + 1]] < 10 or toks[j + 1].isdigit()):
            # decimal digits: "one point five", "2 point 25"
            k = j + 1
            digits = ""
            while k < n:
                if toks[k] in UNIT_WORDS and UNIT_WORDS[toks[k]] < 10:
                    digits += str(UNIT_WORDS[toks[k]])
                elif toks[k].isdigit() and not digits:
                    digits += toks[k]
                else:
                    break
                k += 1
            current = current + float("0." + digits)
            last = "digit"
            j = k
            consumed_any = True
            continue
        else:
            break
        consumed_any = True
        j += 1
    if not consumed_any or j == i:
        return None
    if j == i + 1 and t0 in SCALE_WORDS:
        # a bare scale word stays a number ("thousand" -> 1000)
        return SCALE_WORDS[t0], j
    return total + current, j


def _glue_raw(raws: Sequence[str]) -> list[tuple[str, int, int]]:
    """Group raw tokens by the glue rules ("$2" + ",400", "80" + "%") -> (text, first, last)."""
    out: list[list] = []
    pending: list | None = None
    for k, raw in enumerate(raws):
        text = str(raw).strip() if raw is not None else ""
        if not text:
            continue
        if pending is not None:
            pending = [pending[0] + text, pending[1], k]
            if _joins_next(pending[0]):
                continue
            out.append(pending)
            pending = None
            continue
        if out and _glues_prev(text, out[-1][0]):
            out[-1][0] += text
            out[-1][2] = k
            continue
        if _joins_next(text):
            pending = [text, k, k]
            continue
        out.append([text, k, k])
    if pending is not None:
        out.append(pending)
    return [(t, a, b) for t, a, b in out]


def norm_stream(raws: Sequence[str], expand: bool = False) -> list[tuple[str, int, int]]:
    """Normalize a sequence of raw tokens into (token, first_raw_index, last_raw_index) triples.

    ASR split tokens are glued first (glue_tokens rules), so "$2" + ",400" -> "2400", "dollars"
    with a span covering both raw tokens. Spoken numbers spanning several raw tokens collapse
    into one token whose span covers them.
    """
    stream: list[tuple[str, int, int]] = []
    for text, a, b in _glue_raw(raws):
        prepared = _prep(text)
        if not prepared:
            continue
        pieces = []
        for chunk in prepared.split():
            pieces.extend(_expand_one(chunk) if expand else [chunk])
        for piece in pieces:
            for tok in _piece_tokens(piece):
                stream.append((tok, a, b))
    return _collapse_numbers(stream)


def norm_tokens(text: str | Iterable[str], expand: bool = False) -> list[str]:
    """Normalize text (or a token list) to canonical tokens.

    norm_tokens("I've got $70K, full-time!") -> ["ive", "got", "70000", "dollars", "full", "time"]
    With expand=True contractions are expanded first ("i", "have", ...).
    """
    raws = text.split() if isinstance(text, str) else list(text)
    return [t for t, _, _ in norm_stream(raws, expand=expand)]


def content_tokens(text: str | Iterable[str]) -> list[str]:
    """Normalized tokens minus stopwords and fillers (negations are kept)."""
    return [t for t in norm_tokens(text) if t not in STOPWORDS and not _filler_norm(t)]


# ---------------------------------------------------------------------------
# Numbers
# ---------------------------------------------------------------------------

def extract_numbers(text: str | Iterable[str]) -> list[float]:
    """All number values in the text, in order ("$70K and 80%" -> [70000.0, 80.0])."""
    return [float(t) for t in norm_tokens(text) if _is_numeric(t)]


def num_value(x: str | Iterable[str]) -> float | None:
    """Value of the first number expression in a token, token list or text; None if none.

    Handles "$70K", "70,000", "seventy thousand", "70 thousand", "1.5M", "80%",
    "one point five million", "a hundred", "70 grand".
    """
    nums = extract_numbers(x)
    return nums[0] if nums else None


# ---------------------------------------------------------------------------
# Predicates
# ---------------------------------------------------------------------------

def _filler_norm(n: str) -> bool:
    return bool(n) and (n in FILLERS or bool(FILLER_RE.match(n)))


def is_filler(raw: str) -> bool:
    """True for vocal fillers ("um", "Uh,", "hmmm"); False for "uh-huh", "oh", "like"."""
    return _filler_norm(norm(raw))


def is_stopword(raw: str) -> bool:
    return norm(raw) in STOPWORDS


def is_negation(raw: str) -> bool:
    t = _prep(raw).strip(_EDGE_PUNCT)
    return norm(raw) in NEGATIONS or t.endswith("n't")


def is_hedge(raw: str) -> bool:
    return norm(raw) in HEDGES


def is_continuation(raw: str) -> bool:
    return norm(raw) in CONTINUATIONS


def is_pronoun_opener(raw: str) -> bool:
    return norm(raw) in PRONOUN_OPENERS


def match_preamble(text: str | Iterable[str]) -> str | None:
    """Return the preamble phrase a text opens with ("yeah so", "good question"), else None."""
    toks = norm_tokens(text)
    toks = [t for t in toks if not _filler_norm(t)]
    m = PREAMBLE_RE.match(" ".join(toks))
    return m.group(0) if m else None


def is_sentence_end(raw: str, next_raw: str | None = None) -> bool:
    """True when a raw token ends a sentence (".", "?", "!"), ignoring "e.g.", "Mr.", initials,
    ellipses and decimals. `next_raw`, when given, resolves "etc." / "Inc." style endings."""
    t = unicodedata.normalize("NFKC", str(raw)).translate(_APOSTROPHES).strip()
    t = t.rstrip("\"')]}")
    if not t:
        return False
    if t[-1] in "?!":
        return True
    if t[-1] != ".":
        return False
    if t.endswith("..") or t.endswith("\u2026"):
        return False
    core = t[:-1].lstrip("\"'([{").lower()
    if not core:
        return False
    if re.fullmatch(r"(?:[a-z]\.)+[a-z]", core):  # e.g, i.e, u.s, a.m
        return False
    if len(core) == 1 and core.isalpha():  # an initial: "J."
        return False
    key = core.replace(".", "")
    if key in ABBREVIATIONS:
        return False
    if key in SOFT_ABBREVIATIONS:
        if next_raw is None:
            return True
        nxt = str(next_raw).strip().lstrip("\"'([{")
        return bool(nxt) and nxt[0].isupper()
    return True


def stem(token: str) -> str:
    """Very light suffix stripping for set comparisons ("looking" -> "look", "costs" -> "cost")."""
    t = token
    if len(t) <= 4 or _is_numeric(t):
        return t
    for suf in ("ing", "ed", "ies", "ly", "s"):
        if t.endswith(suf) and len(t) - len(suf) >= 3:
            base = t[: -len(suf)]
            if suf == "ies":
                return base + "y"
            if suf in ("ing", "ed") and len(base) >= 4 and base[-1] == base[-2]:
                base = base[:-1]  # "stopped" -> "stop"
            if suf == "s" and base.endswith(("s", "u", "i")):
                return t  # "glass", "bus", "analysis"
            return base
    return t


def overlap_coefficient(a: str | Iterable[str], b: str | Iterable[str]) -> float:
    """|A & B| / min(|A|, |B|) over token sets. Strings become their stemmed content tokens
    (normalized, stopwords and fillers removed); iterables are used as given. Empty -> 0.0."""
    sa = {stem(t) for t in content_tokens(a)} if isinstance(a, str) else set(a)
    sb = {stem(t) for t in content_tokens(b)} if isinstance(b, str) else set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / min(len(sa), len(sb))


# ---------------------------------------------------------------------------
# Glue (display tokens)
# ---------------------------------------------------------------------------

_GLUE_SUFFIXES = ("'s", "'re", "'ll", "'t", "'ve", "'d", "'m", "n't")
_JOIN_NEXT = frozenset({"$", "\u00a3", "\u20ac", "#", "(", "[", "{", "\u201c", "\u2018",
                        "\u00bf", "\u00a1"})
_PUNCT_CHARS = set(".,;:!?%)]}\"'\u2026\u201d\u2019-\u2013\u2014*")


def _glues_prev(text: str, prev: str | None = None) -> bool:
    """Does `text` join the token before it? With `prev` (that token's text), a leading-decimal
    number (".5", ".5%") joins only a token that ends in a digit ("3" + ".5" is 3.5): after a
    word it is its own number, 0.5 ("ONLY .5%" is not "ONLY.5%"). Without `prev` it joins, as
    before."""
    t = text.translate(_APOSTROPHES)
    if not t:
        return False
    if t.lower() in _GLUE_SUFFIXES:
        return True
    if all(c in _PUNCT_CHARS for c in t):  # pure punctuation: "?", "...", "%", "-"
        return True
    if t.startswith("..."):  # "...and" continues after a trail-off; keep it separate
        return False
    if prev is not None and len(t) > 1 and t[0] == "." and t[1].isdigit():
        return str(prev).rstrip("\"')]}*")[-1:].isdigit()
    if t[0] in ",.%":
        return True
    if len(t) > 1 and t[0] == "-" and (t[1].islower() or t[1].isdigit()):
        return True
    return False


def _joins_next(text: str) -> bool:
    return text in _JOIN_NEXT


def _field(w: Any, keys: Sequence[str]) -> Any:
    if isinstance(w, dict):
        for k in keys:
            if k in w:
                return w[k]
        return None
    for k in keys:
        if hasattr(w, k):
            return getattr(w, k)
    return None


def glue_tokens(words: Iterable[Any], text_key: str | None = None,
                time_keys: tuple[str, str] | None = None) -> list[dict]:
    """Glue ASR split tokens into display tokens.

    A token that starts with "," "." "%" (not "..."), is "'s" "'re" "'ll" "'t" "'ve" "'d" "'m"
    "n't", is pure punctuation, or is "-" + lowercase/digit joins the previous token; a lone "$"
    (or other opening symbol) joins the next. One exception: "." + digit (".5", ".5%") joins only
    a token that ends in a digit ("3" + ".5" -> "3.5"); after a word it stays a token of its own
    (the number 0.5). Empty texts are dropped.

    `words`: dicts (or objects) with text in `text_key` (default: "text", "w", "raw"), index in
    "idx" (list) or "i", times in `time_keys` (default: ("s", "e"), ("s_out", "e_out") or
    ("s_src", "e_src")). Returns [{"text", "idx": [..], "s", "e", "parts": [word, ...]}].
    """
    out: list[dict] = []
    pending: dict | None = None
    for pos, w in enumerate(words):
        text = _field(w, [text_key] if text_key else ["text", "w", "raw"])
        text = "" if text is None else str(text).strip()
        if not text:
            continue
        idx = _field(w, ["idx"])
        if idx is None:
            i = _field(w, ["i"])
            idx = [pos if i is None else i]
        else:
            idx = list(idx)
        if time_keys:
            s, e = _field(w, [time_keys[0]]), _field(w, [time_keys[1]])
        else:
            s, e = None, None
            for ks, ke in (("s", "e"), ("s_out", "e_out"), ("s_src", "e_src")):
                if isinstance(w, dict) and (ks in w or ke in w):
                    s, e = w.get(ks), w.get(ke)
                    break
                if not isinstance(w, dict) and hasattr(w, ks):
                    s, e = getattr(w, ks), getattr(w, ke, None)
                    break
        tok = {"text": text, "idx": idx, "s": s, "e": e, "parts": [w]}
        if pending is not None:
            tok = {"text": pending["text"] + text, "idx": pending["idx"] + idx,
                   "s": pending["s"] if pending["s"] is not None else s,
                   "e": _max_time(pending["e"], e), "parts": pending["parts"] + [w]}
            pending = None
        elif out and _glues_prev(text, out[-1]["text"]):
            prev = out[-1]
            prev["text"] += text
            prev["idx"] += idx
            if prev["s"] is None:
                prev["s"] = s
            prev["e"] = _max_time(prev["e"], e)
            prev["parts"].append(w)
            continue
        if _joins_next(tok["text"]):
            pending = tok
            continue
        out.append(tok)
    if pending is not None:
        out.append(pending)
    return out


def _max_time(a: float | None, b: float | None) -> float | None:
    if a is None:
        return b
    if b is None:
        return a
    return max(a, b)
