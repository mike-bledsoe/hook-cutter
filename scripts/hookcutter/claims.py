"""Claims on written surfaces versus the words spoken in the clip.

Hard rule: every number and name on a written surface is said in the clip. The surfaces are
the hook card, the cover headline (cover.headline, else the hook card, else the clip title: the
text `hc cover` prints), the clip title (the exported file name and manifest title), the post
title and caption, and the chapter titles of a landscape clip. `written_surfaces` is the one
list of them and `effective_headline` the one definition of the cover text: plan lint L13
(always WARN), QA gate G11 and the cover drawer all read these, so what is drawn is what was
checked and plan time and QA agree.

Rules:
- Numbers: digits ("40", "$10K", "1.5M", "80-90%"), multipliers ("10X", "x3", "3×"), ordinals
  ("3RD", "first".."tenth") and number words. Number words and ordinal words only count before a
  noun or unit ("ONE PERSON", "FIRST CLIENT"), or when they carry a scale ("TWENTY-FIVE GRAND",
  "ONE MILLION", "FIVE HUNDRED THOUSAND": anywhere, also at the end of the surface);
  "THE ONE", "A NORMAL ONE" and "SELL IT FIRST" are not number claims. A surface "1 YEAR" matches
  a spoken "a year" or "a whole year"; "2026" matches a spoken "twenty twenty six". A surface
  "0 X" / "ZERO X" matches a spoken zero, or a spoken "no X" / "nothing on X" / "nobody X"
  (the negation must stand within three words before the noun the surface counts; a bare "No,"
  elsewhere in the clip states nothing). Unspoken numbers FAIL.
- Names: glossary terms, words the transcript only ever capitalizes mid-sentence, and in mixed
  case text any capitalized word (in first position too, unless it is a common word). A name
  matches only itself, its possessive or plural ("James", "James's"), never a word that merely
  shares its stem ("jam"). Unspoken names WARN.
- Words: all-caps and title-case surfaces cannot show which words are names, so every content
  word that is not spoken (after normalization and light stemming) and is not a common word
  (the built-in list, or a word the transcript uses in lowercase) WARNs.
- `exempt_names` (the clip speaker and the speaker labels inside the clip) only apply to the
  clip title and the post surfaces, where crediting a speaker by name is normal. A title that is
  printed on the cover is also checked as a cover headline, without that exemption.
"""

from __future__ import annotations

import re
from typing import Iterable, Sequence

from . import textnorm as tn

TITLE_ON_COVER = "cover headline (the title)"
LABELS = ("hook card", "cover headline", TITLE_ON_COVER, "title", "post title", "post caption",
          "chapter title")

ORDINAL_WORDS = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6,
                 "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10}
NUMBER_WORDS = (set(tn.UNIT_WORDS) | set(tn.TENS_WORDS) | set(tn.SCALE_WORDS)
                | {"dozen", "grand"})
# spoken forms that state a number without a digit or number word
SPOKEN_VALUES = {"double": 2.0, "twice": 2.0, "triple": 3.0, "thrice": 3.0, "quadruple": 4.0,
                 "dozen": 12.0, "half": 0.5, "zero": 0.0}
# A negation states "zero" only of what it negates ("no employees", "nothing on ads"): the
# content words within ZERO_NEAR tokens after it, unless its own token ends the clause ("No,").
ZERO_WORDS = frozenset({"no", "none", "nothing", "nobody", "noone"})
ZERO_NEAR = 3
# "no way churn stays low": in these set phrases the negation reaches only the word after it
_NO_IDIOMS = frozenset({"way", "matter", "wonder", "longer", "kidding"})
# S4: words that make a spelled-out amount a claim wherever it stands ("ONE MILLION")
_SCALE_RAW = frozenset(tn.SCALE_WORDS) | {"grand", "dozen"}
_MULT_POST = re.compile(r"^(\d+(?:\.\d+)?)[x×]$", re.I)
_MULT_PRE = re.compile(r"^[x×](\d+(?:\.\d+)?)$", re.I)
_ORD_DIGIT = re.compile(r"^(\d+)(?:st|nd|rd|th)$", re.I)
_NUMERIC = re.compile(r"^\d+(?:\.\d+)?$")
_EDGE = "\"'()[]{}.,;:!?*“”‘’"
_STOP_PUNCT = ".,;:!?"

# Common English words: a headline that uses one of these without the speaker saying it is a
# paraphrase, not a new claim. Names, brands and rare words are deliberately absent.
COMMON = frozenset("""
able about above accept access account across act action active actually add advice afford afraid
after again age agency agent ago agree ahead aim air all allow almost alone along already also
always amazing amount angry announce annual answer anyone anything anyway apart app apply approach
area argue argument around arrive art ask asleep attention audience avoid awake aware away awful bad
badly balance bank bar base basic battle bear beat beautiful become bed before begin beginning
behind believe benefit best better beyond big bill bit black blame block blow board body bold book
boom born borrow boss both bottom box brain brand brave break breath brief bright brilliant bring
broke broken brother brutal budget build builder building built burn burnout business busy buy
buyer call calm camera can capital car card care career careful carry case cash catch cause
celebrate center certain chain chance change channel charge cheap check child children choice
choose chose city claim class clean clear clever click client close closed coach code cold collect
come comfort comfortable coming common community company compare competition complete complex
confidence confident consistent constant content control cook cool copy core corner cost could
count country couple courage course cover crash crazy create creative creator credit crew crisis
critical cross crowd cry culture cure curious current customer cut cycle daily damage danger dark
data date daughter day dead deal dear death debt decide decision deep default degree deliver demand
deny depend design desire desk detail die different difficult dinner direct direction dirty
discover dive divide doctor dollar dollars door doubt dream dress drink drive drop dumb during duty
dying early earn easy eat economy edge effect effort either else email employee empty end enemy
energy engine enjoy enough enter entire entrepreneur equal error escape even evening event ever
every evidence exact exactly example except excited exciting excuse exist expect expensive
experience expert explain extra eye face fact fail failure fair faith fake fall false fame family
famous fan far fast fat father fault favorite fear fee feed feel feeling fell felt few field fight
figure fill film final finally find fine finish fire firm first fit fix flat flip floor flow fly
focus follow food fool for force forever forget forgot form forward found founder free freedom
fresh friend friends front full fun fund funny future gain game gap gave general genius gift girl
give glad goal gold golden gone good grab great greed green grind ground group grow growth guess
guest guy habit hack half hand handle happen happy hard harder hate head health hear heard heart
heavy held hell help hero hidden hide high hire history hit hold hole home honest hope horrible
hot hour house huge human hundred hungry hunt hurt idea ideal ignore ill image imagine impact
important impossible improve include income increase industry info inside insane instead interest
invest investor issue item job join joke journey judge jump keep kept key kid kids kill kind king
knew knock know knowledge lack land language large last late later laugh launch law lazy lead
leader learn least leave led left legal less lesson let letter level lie life lift light limit
line link list listen little live living load loan local lock long look loose lose loss lost loud
love low luck lucky machine mad made main major make maker man manage manager many map margin mark
market marketing master match matter maybe meet meeting member memory mental mess message method
middle might mind mindset minute miss mission mistake mistakes model mom moment money month mood
more morning most mother move movie much music must myth name natural near need negative network
never new news next nice night noise normal note nothing notice now number offer office often old
once one online only open opinion opportunity order ordinary other outside own owner pace page paid
pain paper parent part partner party pass past path pay peace people perfect period person phone
pick picture piece place plan play player point poor popular position positive possible post
pound power practice premium prepare present press pressure pretty price pride private prize
problem process produce product profit program project promise proof proud prove public pull push
put quality question quick quiet quit quite race raise ran random rate rather reach read ready real
reality realize reason receive recent record red reduce regret relationship release remember
remove rent repeat reply report rest result return revenue rich ride right ring rise risk road
rock role room root rough round rule rules run rush sad safe said sale sales same save saw say scale
scared school score secret see seem sell seller send sense serious serve service set settle sharp
shift ship shit shock shop short shot should show shut sick side sign signal silent simple single
sit size skill skip sleep slow small smart social sold solid solve someone son soon sorry sound
source space speak special speed spend spent spot stack staff stage stand start state stay step
stick still stock stop store story strange stranger strangers strategy street strength stress
strong struggle stuck student study stuff stupid style success successful sudden suffer sure
surprise survive switch system table take talent talk task taste tax teach team tell term test
text thank theory think thought thousand three throw ticket tight till time tiny tired today
together told tomorrow tool top total touch tough toward track trade train trap trick trip trouble
true trust truth try turn type ugly understand unit until upset usual value video view voice wait
wake walk wall want war warm warning waste watch water way weak wealth wear week weird well went
whole wife wild will win wind window winner wise wish woman women won wonder word work worker world
worry worse worst worth would write writing wrong year years yesterday young yourself zero
able ai answer apps ask asked asking bad bought built came cost died does doing done drove felt
fought gets give given goes got gotten grew hands helped hired holds kept kicked knows learned
lied liked lives looked lost loved made makes making meant met paid played quit ran said sat says
seen sent sets showed sold spent started stole stood stopped taught thinks told took tried turned
used wanted wants went won wrote
""".split())

_IRREGULAR = {"sold": "sell", "made": "make", "told": "tell", "bought": "buy", "built": "build",
              "paid": "pay", "said": "say", "went": "go", "took": "take", "gave": "give",
              "thought": "think", "found": "find", "felt": "feel", "kept": "keep", "left": "leave",
              "lost": "lose", "meant": "mean", "met": "meet", "ran": "run", "saw": "see",
              "sent": "send", "spent": "spend", "stood": "stand", "taught": "teach", "won": "win",
              "wrote": "write", "became": "become", "began": "begin", "broke": "break",
              "brought": "bring", "chose": "choose", "came": "come", "did": "do", "drove": "drive",
              "fell": "fall", "flew": "fly", "forgot": "forget", "grew": "grow", "heard": "hear",
              "held": "hold", "knew": "know", "led": "lead", "quit": "quit", "rode": "ride",
              "rose": "rise", "sat": "sit", "shot": "shoot", "slept": "sleep", "spoke": "speak",
              "stole": "steal", "stuck": "stick", "threw": "throw", "understood": "understand",
              "woke": "wake", "wore": "wear", "fought": "fight", "caught": "catch", "taken": "take",
              "given": "give", "gone": "go", "done": "do", "seen": "see", "written": "write",
              "known": "know", "grown": "grow", "shown": "show", "men": "man", "women": "woman",
              "children": "child", "people": "person", "better": "good", "best": "good",
              "worse": "bad", "worst": "bad"}


# ---------------------------------------------------------------- token helpers

def _variants(key: str) -> set:
    """A normalized word plus its light stems, so "selling", "sells" and "sold" meet "sell"."""
    out = {key}
    if not key:
        return out
    out.add(tn.stem(key))
    if key in _IRREGULAR:
        out.add(_IRREGULAR[key])
    for suf in ("s", "es", "ed", "d", "ing", "er", "ers", "est", "ly"):
        if key.endswith(suf) and len(key) - len(suf) >= 3:
            base = key[: -len(suf)]
            out.add(base)
            out.add(base + "e")
            if len(base) >= 4 and base[-1] == base[-2]:
                out.add(base[:-1])
    if key.endswith("ies") and len(key) > 4:
        out.add(key[:-3] + "y")
    return out


def _core(raw: str) -> str:
    core = str(raw).strip(_EDGE)
    low = core.lower()
    if low.endswith("'s") or low.endswith("’s"):
        core = core[:-2]
    return core


def _shown(raw: str) -> str:
    """A surface token as messages quote it: no edge punctuation, but a leading decimal point
    stays (".5%" is quoted as ".5%", the 0.5 it is read as, not as "5%")."""
    core = _core(raw)
    if core[:1].isdigit():
        k = str(raw).find(core)
        if k >= 1 and str(raw)[k - 1] == "." and (k < 2 or str(raw)[k - 2] != "."):
            return "." + core
    return core or str(raw)


def _is_num_tok(tok: str) -> bool:
    return bool(_NUMERIC.match(tok))


def _content(tokens: Iterable[str]) -> list[str]:
    """The content words among normalized tokens (no stopwords, negations or number words)."""
    return [t for t in tokens if t.isalpha() and len(t) > 1 and t not in tn.STOPWORDS
            and t not in tn.NEGATIONS and t not in NUMBER_WORDS]


def _same(a: float, b: float) -> bool:
    return abs(a - b) <= 1e-6 * max(1.0, abs(a), abs(b))


def _gloss_keys(glossary: Iterable[str]) -> set:
    keys = set()
    for g in glossary or ():
        for part in re.split(r"[\s\-/]+", str(g)):
            k = tn.norm(_core(part))
            if k and k not in tn.STOPWORDS and not _is_num_tok(k):
                keys.add(k)
    return keys


def _name_keys(names: Iterable[str]) -> set:
    return _gloss_keys(names)


# ---------------------------------------------------------------- surface parsing

def _prepare(text: str) -> tuple[list[str], dict]:
    """Raw tokens with multipliers rewritten to their digits; {raw index: "mult"}."""
    raws, kinds = [], {}
    for k, tok in enumerate(str(text or "").replace("*", " ").split()):
        core = tok.strip(_EDGE)
        m = _MULT_POST.match(core) or _MULT_PRE.match(core)
        if m:
            raws.append(tok.replace(core, m.group(1)))
            kinds[k] = "mult"
            continue
        raws.append(tok)
    return raws, kinds


def _mode(raws: Sequence[str]) -> str:
    """"caps", "title" or "mixed" (sentence case)."""
    letters = [w for w in raws if any(c.isalpha() for c in w)]
    if not letters:
        return "mixed"
    if all(w.upper() == w for w in letters):
        return "caps"
    content = [w for w in letters[1:] if tn.norm(_core(w)) not in tn.STOPWORDS]
    if content and sum(1 for w in content if _core(w)[:1].isupper()) >= 0.6 * len(content):
        return "title"
    return "mixed"


def _next_is_noun(stream, pos: int, raws: Sequence[str], b: int) -> tuple[bool, str | None]:
    """Does a noun or unit follow the number run that ends at raw index `b`?"""
    if raws[b].rstrip("\"')]}*")[-1:] in _STOP_PUNCT:
        return False, None
    if pos + 1 >= len(stream):
        return False, None
    tok = stream[pos + 1][0]
    if not tok.isalpha() or len(tok) < 2 or tok in tn.STOPWORDS or tok in tn.NEGATIONS:
        return False, None
    if tok in NUMBER_WORDS or tok in ORDINAL_WORDS:
        return False, None
    return True, tok


def _surface_numbers(raws: Sequence[str], kinds: dict) -> tuple[list[dict], set]:
    """[{"value", "raw", "unit", "near"}] and the raw indices they consumed. "near" are the
    content words around the number (what a surface zero counts)."""
    stream = tn.norm_stream(list(raws))
    out, used = [], set()
    for pos, (tok, a, b) in enumerate(stream):
        value = None
        span_raw = [str(raws[k]) for k in range(a, b + 1)]
        has_digit = any(any(c.isdigit() for c in r) for r in span_raw)
        # judged on the raw word: tn.norm("MILLION") is "1000000", never a scale word
        scaled = any(part in _SCALE_RAW for r in span_raw
                     for part in re.split(r"[-/]", _core(r).lower()))
        if _is_num_tok(tok):
            value = float(tok)
        elif tok in ORDINAL_WORDS:
            value = float(ORDINAL_WORDS[tok])
        elif _ORD_DIGIT.match(tok):
            value = float(_ORD_DIGIT.match(tok).group(1))
        elif tok == "dozen":
            value = 12.0
        if value is None:
            continue
        noun, unit = _next_is_noun(stream, pos, raws, b)
        if not has_digit and not scaled and not noun:
            continue                              # "THE ONE", "A NORMAL ONE", "SELL IT FIRST"
        if not has_digit and all(_core(r).lower() in ("a", "an") for r in span_raw):
            continue
        raw = " ".join(_shown(r) for r in span_raw)
        if any(kinds.get(k) == "mult" for k in range(a, b + 1)):
            raw += "X"
        near = (_content(t for t, _x, _y in stream[pos + 1:pos + 2 + ZERO_NEAR])
                + _content(t for t, _x, _y in stream[max(0, pos - 2):pos]))
        out.append({"value": value, "raw": raw, "unit": unit, "near": near})
        used.update(range(a, b + 1))
    return out, used


def _parse(text: str, glossary=(), evidence=None) -> dict:
    raws, kinds = _prepare(text)
    mode = _mode(raws)
    numbers, used = _surface_numbers(raws, kinds)
    gl = _gloss_keys(glossary)
    ev = evidence or {}
    names, words, seen = [], [], set()
    for i, raw in enumerate(raws):
        if i in used:
            continue
        core = _core(raw)
        if not core or not any(c.isalpha() for c in core):
            continue
        initial = i == 0 or str(raws[i - 1]).rstrip("\"')]}*")[-1:] in ".!?:"
        for key in [t for t, _a, _b in tn.norm_stream([core])]:
            if not key or not key.isalpha() or key in tn.STOPWORDS or key in seen:
                continue
            if key in ORDINAL_WORDS or key in NUMBER_WORDS:
                continue
            seen.add(key)
            rec = ev.get(key)
            cap_only = bool(rec and rec[0] > 0 and rec[1] == 0)
            if key in gl or (cap_only and (mode != "mixed" or core[:1].isupper())):
                names.append((core, key))
            elif mode == "mixed":
                if core[:1].isupper() and (not initial or not is_common(key, ev)):
                    names.append((core, key))
            else:
                words.append((core, key))
    return {"mode": mode, "numbers": numbers, "names": names, "words": words}


def is_common(key: str, evidence=None) -> bool:
    """A common English word, or one the transcript uses in lowercase mid-sentence."""
    var = _variants(key)
    if var & COMMON:
        return True
    rec = (evidence or {}).get(key)
    return bool(rec and rec[1] > 0)


def surface_claims(text: str) -> dict:
    """{"numbers": [(value, raw)], "names": [str], "content": [str], "mode"} for one surface.

    "names" are the capitalized name candidates of mixed-case text; "content" are the content
    words of an all-caps or title-case surface (whose unspoken ones G11 warns about)."""
    p = _parse(text)
    return {"numbers": [(n["value"], n["raw"]) for n in p["numbers"]],
            "names": [c for c, _k in p["names"]],
            "content": [k for _c, k in p["words"]], "mode": p["mode"]}


# ---------------------------------------------------------------- spoken side

class Spoken:
    """Token, stem and number sets of the words spoken in the clip."""

    def __init__(self, spoken: Sequence[str]):
        raws, _kinds = _prepare(" ".join(str(w) for w in spoken))
        self.tokens: set = set()
        self.values: list = []
        self.a_units: set = set()
        self.zero_units: set = set()
        for expand in (False, True):
            stream = tn.norm_stream(raws, expand=expand)
            for tok, _a, _b in stream:
                self.tokens.add(tok)
        stream = tn.norm_stream(raws)
        prev = None
        for pos, (tok, a, b) in enumerate(stream):
            if _is_num_tok(tok):
                v = float(tok)
                self.values.append(v)
                nxt = [t for t, _x, _y in stream[pos + 1:pos + 4]]
                if nxt[:3] == ["and", "a", "half"]:
                    self.values.append(v + 0.5)
                if prev is not None and prev[2] + 1 == a:
                    pv = prev[0]
                    if 10 <= pv <= 99 and pv == int(pv) and 0 <= v <= 99 and v == int(v):
                        self.values.append(pv * 100 + v)        # "twenty twenty six" -> 2026
                prev = (v, a, b)
                continue
            prev = None
            if tok in ORDINAL_WORDS:
                self.values.append(float(ORDINAL_WORDS[tok]))
            elif _ORD_DIGIT.match(tok):
                self.values.append(float(_ORD_DIGIT.match(tok).group(1)))
            if tok in SPOKEN_VALUES:
                self.values.append(SPOKEN_VALUES[tok])
            if tok in ZERO_WORDS and str(raws[b]).rstrip("\"')]}*")[-1:] not in _STOP_PUNCT:
                after = [t for t, _x, _y in stream[pos + 1:pos + 1 + ZERO_NEAR]]
                if after and after[0] in _NO_IDIOMS:
                    after = after[:1]
                for t in _content(after):
                    self.zero_units.update(_variants(t))
            if tok in ("a", "an"):
                for t, _x, _y in stream[pos + 1:pos + 3]:
                    if t.isalpha() and t not in tn.STOPWORDS:
                        self.a_units.update(_variants(t))
        self.variants: set = set()
        for t in self.tokens:
            self.variants.update(_variants(t))

    def has_word(self, key: str) -> bool:
        if key in self.tokens or key + "s" in self.tokens:
            return True
        return bool(_variants(key) & self.variants)

    def has_name(self, key: str) -> bool:
        """A name is spoken when the same word is: "james", or its possessive or plural
        ("James's" normalizes to "jamess"). Never a word that only shares a stem ("jam" for
        James, "parked" for Parker, "William" for Williams)."""
        return key in self.tokens or key + "s" in self.tokens

    def has_number(self, value: float, unit: str | None = None,
                   near: Sequence[str] = ()) -> bool:
        """Is `value` stated? `unit` is the noun after a surface 1 ("1 YEAR" / "a year"); `near`
        are the content words around a surface 0 ("0 EMPLOYEES" / "no employees")."""
        if any(_same(value, v) for v in self.values):
            return True
        if _same(value, 1.0) and unit and _variants(unit) & self.a_units:
            return True
        if _same(value, 0.0):
            return any(_variants(t) & self.zero_units for t in near or ())
        return False


# ---------------------------------------------------------------- what is written

def _hook_text(clip: dict, plan: dict | None = None) -> str:
    hc = (plan or {}).get("hook_card") or (clip or {}).get("hook_card") or {}
    if isinstance(hc, dict) and hc.get("lines"):
        return " ".join(str(x) for x in hc["lines"])
    return ""


def _formats(clip: dict) -> list:
    eff = (clip or {}).get("_eff")
    fmts = (eff.get("formats") if isinstance(eff, dict) else None) or (clip or {}).get("formats")
    if isinstance(fmts, str):
        fmts = [fmts]
    return list(fmts) if isinstance(fmts, (list, tuple)) and fmts else ["vertical"]


def effective_headline(clip: dict, plan: dict | None = None) -> tuple[str, str]:
    """(text, origin): the headline `hc cover` prints and where it comes from. origin is
    "cover" (clip cover.headline), "hook" (the hook card lines, the plan's before the clip's),
    "title" (the clip title, when there is neither) or "" (nothing to print)."""
    clip = clip if isinstance(clip, dict) else {}
    c = clip.get("cover") or {}
    if isinstance(c, dict) and c.get("headline"):
        return str(c["headline"]), "cover"
    hook = _hook_text(clip, plan)
    if hook:
        return hook, "hook"
    title = str(clip.get("title") or "")
    return title, ("title" if title else "")


def written_surfaces(clip: dict, plan: dict | None = None) -> list[tuple[str, str]]:
    """Every written surface of a clip as [(label, text)]: the list plan lint L13 and QA gate
    G11 check, in this order.

    "hook card" (the plan's, else the clip's); "cover headline" (cover.headline), or
    "cover headline (the title)" when a vertical clip has neither a cover headline nor a hook
    card and the cover therefore prints the title; "title" (the clip title: file name and
    manifest title); "post title"; "post caption"; one "chapter title" per piece `chapter` of a
    clip with a landscape format (pasted as YouTube chapters)."""
    clip = clip if isinstance(clip, dict) else {}
    out: list[tuple[str, str]] = []
    hook = _hook_text(clip, plan)
    if hook:
        out.append(("hook card", hook))
    head, origin = effective_headline(clip, plan)
    formats = _formats(clip)
    if origin == "cover":
        out.append(("cover headline", head))
    elif origin == "title" and "vertical" in formats:
        out.append((TITLE_ON_COVER, head))
    if clip.get("title"):
        out.append(("title", str(clip["title"])))
    post = clip.get("post") if isinstance(clip.get("post"), dict) else {}
    for key, label in (("title", "post title"), ("caption", "post caption")):
        if post.get(key):
            out.append((label, str(post[key])))
    if "landscape" in formats:
        for piece in clip.get("pieces") or []:
            if isinstance(piece, dict) and piece.get("chapter"):
                out.append(("chapter title", str(piece["chapter"])))
    return out


def _credits_speaker(label: str) -> bool:
    """Surfaces where naming the speaker is normal even when nobody says the name: the clip
    title and the post title and caption. Not a cover, a hook card or a chapter title."""
    label = str(label)
    return label == "title" or label.startswith("post")


# ---------------------------------------------------------------- the check

def unspoken_claims(surfaces, spoken, *, glossary=(), exempt_names=(), name_evidence=None) -> list:
    """Claims on `surfaces` ([(label, text)]) that the `spoken` words never state.

    Returns [{"surface", "kind": "number"|"name"|"word", "raw", "level", "msg"}]: numbers FAIL,
    names and words WARN. `exempt_names` apply only to the clip title and to surfaces whose label
    starts with "post". A text that stands on two surfaces (a title that is also printed on the
    cover) reports each claim once, under the first label.
    """
    sp = spoken if isinstance(spoken, Spoken) else Spoken(list(spoken or []))
    exempt = _name_keys(exempt_names)
    out = []
    told: dict = {}
    for label, text in surfaces or []:
        if not text:
            continue
        p = _parse(str(text), glossary, name_evidence)
        post = _credits_speaker(label)
        seen = told.setdefault(str(text), set())
        for n in p["numbers"]:
            if sp.has_number(n["value"], n["unit"], n["near"]):
                continue
            key = ("number", n["value"])
            if key in seen:
                continue
            seen.add(key)
            out.append({"surface": label, "kind": "number", "raw": n["raw"], "value": n["value"],
                        "level": "FAIL",
                        "msg": f'{label}: number "{n["raw"]}" ({tn.fmt_number(n["value"])}) is never spoken in the clip'})
        for core, key in p["names"]:
            if sp.has_name(key) or (post and key in exempt) or ("name", key) in seen:
                continue
            seen.add(("name", key))
            out.append({"surface": label, "kind": "name", "raw": core, "level": "WARN",
                        "msg": f'{label}: name "{core}" is not spoken in the clip'})
        for core, key in p["words"]:
            if (sp.has_word(key) or is_common(key, name_evidence) or (post and key in exempt)
                    or ("word", key) in seen):
                continue
            seen.add(("word", key))
            out.append({"surface": label, "kind": "word", "raw": core, "level": "WARN",
                        "msg": f'{label}: "{core}" is not spoken in the clip (a name, or a word the speaker did not use)'})
    return out


def spoken_claims_found(surfaces, spoken, *, glossary=(), name_evidence=None) -> list[str]:
    """Numbers and names on the surfaces that the clip does state (for evidence lines)."""
    sp = spoken if isinstance(spoken, Spoken) else Spoken(list(spoken or []))
    found = []
    texts = set()
    for label, text in surfaces or []:
        if not text or str(text) in texts:      # the same text on two surfaces is listed once
            continue
        texts.add(str(text))
        p = _parse(str(text), glossary, name_evidence)
        for n in p["numbers"]:
            if sp.has_number(n["value"], n["unit"], n["near"]):
                found.append(f"{label}: {n['raw']}")
        for core, key in p["names"]:
            if sp.has_name(key):
                found.append(f"{label}: {core}")
    return found
