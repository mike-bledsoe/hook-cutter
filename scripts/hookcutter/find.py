"""`find` (fuzzy phrase search over the transcript) and `words` (exact word window) commands.

Search normalizes both sides with textnorm (case, punctuation, contractions, numbers), ignores
fillers, and runs a semi-global alignment of the phrase against the word stream, so a phrase
still matches when ASR misspelled a word, inserted an "um", or dropped a small word.
"""

from __future__ import annotations

from typing import Any, Sequence

from . import env, textnorm

MATCH_MIN = 0.75     # character similarity for two different tokens to count as a match
MISMATCH = 0.6       # penalty: phrase token aligned to a different stream token
GAP_PHRASE = 0.6     # penalty: phrase token with no stream counterpart
GAP_STREAM = 0.4     # penalty: extra stream token inside the match
MIN_SCORE = 0.5      # hits below this normalized score are not reported
NEAR_WEIGHT = 0.3    # --near: score bonus range for proximity
NEAR_SCALE = 1800.0  # --near: seconds over which the proximity bonus fades out
DEFAULT_MAX = 8
CONTEXT_WORDS = 8
LOW_P = 0.5


def _bigrams(t: str, cache: dict) -> frozenset:
    g = cache.get(t)
    if g is None:
        g = frozenset(t[k:k + 2] for k in range(len(t) - 1))
        cache[t] = g
    return g


def _similarity(a: str, b: str, cache: dict) -> float:
    """1.0 for equal tokens; character-bigram Dice for near-misses ("reptile" ~ "greptile");
    0.0 below MATCH_MIN, for numbers that differ, or for very short tokens."""
    if a == b:
        return 1.0
    key = (a, b)
    if key in cache:
        return cache[key]
    sim = 0.0
    la, lb = len(a), len(b)
    if (min(la, lb) >= 3 and abs(la - lb) <= 0.4 * max(la, lb)
            and not (_is_num(a) or _is_num(b))):
        ga, gb = _bigrams(a, cache), _bigrams(b, cache)
        if ga and gb:
            d = 2.0 * len(ga & gb) / (len(ga) + len(gb))
            sim = d if d >= MATCH_MIN else 0.0
    cache[key] = sim
    return sim


def _is_num(t: str) -> bool:
    try:
        float(t)
        return True
    except ValueError:
        return False


def _stream(words: Sequence[dict]) -> list[tuple[str, int, int]]:
    """Normalized, filler-free token stream with word-index spans."""
    return [x for x in textnorm.norm_stream([w.get("w", "") for w in words], expand=True)
            if not textnorm.is_filler(x[0])]


def search(words: Sequence[dict], phrase: str, near: float | None = None,
           max_hits: int = DEFAULT_MAX, min_score: float = MIN_SCORE) -> list[dict]:
    """Ranked, non-overlapping fuzzy matches of `phrase` in transcript `words`.

    Returns [{"a", "b", "s", "e", "score", "rank"}] where a..b are inclusive word indices and
    score is in (0, 1] (1.0 = every phrase token matched exactly, nothing inserted).
    """
    ptoks = [t for t in textnorm.norm_tokens(phrase, expand=True)
             if not textnorm.is_filler(t)]
    if not ptoks:
        raise env.HCError("the search phrase has no words after normalization: %r" % phrase,
                          code=1)
    stream = _stream(words)
    stoks = [t for t, _, _ in stream]
    m, n = len(ptoks), len(stoks)
    if n == 0:
        return []
    cache: dict = {}
    prev = [0.0] * (n + 1)
    prev_start = list(range(n + 1))
    for i in range(1, m + 1):
        p = ptoks[i - 1]
        cur = [0.0] * (n + 1)
        cur_start = [0] * (n + 1)
        cur[0] = -GAP_PHRASE * i
        for j in range(1, n + 1):
            sim = _similarity(p, stoks[j - 1], cache)
            best = prev[j - 1] + (sim if sim > 0 else -MISMATCH)
            bstart = prev_start[j - 1]
            up = prev[j] - GAP_PHRASE
            if up > best:
                best, bstart = up, prev_start[j]
            left = cur[j - 1] - GAP_STREAM
            if left > best:
                best, bstart = left, cur_start[j - 1]
            cur[j] = best
            cur_start[j] = bstart
        prev, prev_start = cur, cur_start
    cands = []
    for j in range(1, n + 1):
        score = prev[j] / m
        start = prev_start[j]
        if score >= min_score and start < j:
            cands.append((score, start, j))
    hits: list[dict] = []
    taken: list[tuple[int, int]] = []
    taken_words: list[tuple[int, int]] = []
    for score, start, end in sorted(cands, key=lambda c: (-c[0], c[1])):
        if any(start < b and a < end for a, b in taken):
            continue
        # A hyphenated word ("bye-bye") is several tokens of ONE word: two token spans that do
        # not overlap can still be the same word range. Best score first, so the best hit of a
        # word range is the one kept, and a repeat cannot use up a --max slot.
        a, b = stream[start][1], stream[end - 1][2]
        if any(a <= tb and ta <= b for ta, tb in taken_words):
            continue
        taken.append((start, end))
        taken_words.append((a, b))
        hit = {"a": a, "b": b, "s": words[a]["s"], "e": words[b]["e"],
               "score": round(score, 3)}
        rank = score
        if near is not None:
            dist = abs(hit["s"] - near)
            rank += NEAR_WEIGHT * (1.0 - min(1.0, dist / NEAR_SCALE))
        hit["rank"] = round(rank, 4)
        hits.append(hit)
    hits.sort(key=lambda h: (-h["rank"], h["a"]))
    return hits[:max(1, int(max_hits))]


def _clock(t: float, hours: bool) -> str:
    from .transcribe import fmt_clock
    return fmt_clock(t, hours)


def format_hit(words: Sequence[dict], hit: dict, context: int = CONTEXT_WORDS,
               hours: bool = False) -> str:
    """`#a-b [t0-t1] (score) Speaker: ...before >>match<< after...`"""
    a, b = hit["a"], hit["b"]
    lo, hi = max(0, a - context), min(len(words), b + 1 + context)
    before = " ".join(w["w"] for w in words[lo:a])
    match = " ".join(w["w"] for w in words[a:b + 1])
    after = " ".join(w["w"] for w in words[b + 1:hi])
    sp = words[a].get("sp")
    text = ("..." if lo > 0 else "") + (before + " " if before else "") + ">>" + match + "<<" + \
        (" " + after if after else "") + ("..." if hi < len(words) else "")
    return "#%d-%d [%s-%s] (%.2f) %s%s" % (a, b, _clock(hit["s"], hours), _clock(hit["e"], hours),
                                           hit["score"], (sp + ": ") if sp else "", text)


def word_rows(transcript: dict, a: int, b: int) -> list[str]:
    """Table rows for words a..b: index, src times, clock, gap before, p, flags, speaker, word.

    Flags: F filler, ? low probability, L inside a repetition loop, ~ interpolated timing (a run
    of words that shared one start), r re-timed from the audio, ! other transcript flag on the
    word (no audible sound, uncertain speaker label, possible false start, uncertain start)."""
    words = transcript.get("words") or []
    marks: dict = {}

    def add(k0: int, k1: int, ch: str) -> None:
        for k in range(max(k0, a), min(k1, b) + 1):
            if ch not in marks.setdefault(k, ""):
                marks[k] += ch

    for f in transcript.get("flags") or []:
        if not isinstance(f.get("a"), int) or not isinstance(f.get("b"), int):
            continue
        if f.get("kind") == "repetition_loop":
            add(f["a"], f["b"], "L")
        elif f.get("kind") in ("no_audio", "speaker_uncertain", "false_start", "timing_uncertain"):
            add(f["a"], f["b"], "!")
    for r in transcript.get("timing_repairs") or []:
        if isinstance(r.get("i"), int):
            add(r["i"], r["i"], "r")
        elif isinstance(r.get("a"), int) and isinstance(r.get("b"), int):
            add(r["a"], r["b"], "~")
    hours = (words[-1]["e"] if words else 0) >= 3600
    rows = ["%6s  %9s  %9s  %10s  %6s  %5s  %-4s %-16s %s" % (
        "#", "s_src", "e_src", "at", "gap", "p", "flg", "speaker", "word")]
    for k in range(a, b + 1):
        w = words[k]
        gap = "-" if k == 0 else "%.3f" % (w["s"] - words[k - 1]["e"])
        p = w.get("p")
        mark = ""
        if textnorm.is_filler(w["w"]):
            mark += "F"
        if p is not None and p < LOW_P:
            mark += "?"
        mark += marks.get(k, "")
        rows.append("%6d  %9.3f  %9.3f  %10s  %6s  %5s  %-4s %-16s %s" % (
            k, w["s"], w["e"], _clock(w["s"], hours), gap, "-" if p is None else "%.2f" % p,
            mark, (w.get("sp") or "-")[:16], w["w"]))
    return rows


def _load(args: Any) -> dict:
    from .project import Project
    proj = Project(args.project).require()
    clip = getattr(args, "clip", None)
    if clip and proj.clip_kind(clip) is None and not (proj.root / "transcripts" / ("%s.json" % clip)).exists():
        # an unknown id must not quietly read the project transcript instead
        raise env.unknown_clip(clip, proj.clip_ids())
    return proj.transcript(clip)


def _parse_index(v: Any, name: str) -> int:
    s = str(v).strip().lstrip("#")
    try:
        return int(s)
    except ValueError:
        raise env.HCError("%s must be a word index (integer), got %r" % (name, v), code=1)


def cmd_find(args: Any) -> int:
    """hc find P "phrase" [--near MM:SS] [--max N]"""
    transcript = _load(args)
    words = transcript.get("words") or []
    near_arg = getattr(args, "near", None)
    near = env.parse_time(near_arg) if near_arg not in (None, "") else None
    max_hits = getattr(args, "max", None) or DEFAULT_MAX
    hits = search(words, args.phrase, near=near, max_hits=max_hits)
    if not hits:
        env.log("find", "no match for %r" % args.phrase)
        return 1
    hours = (words[-1]["e"] if words else 0) >= 3600
    for h in hits:
        print(format_hit(words, h, hours=hours))
    return 0


def cmd_words(args: Any) -> int:
    """hc words P A B: words A..B (inclusive) with times, gaps, filler flags and speakers."""
    transcript = _load(args)
    words = transcript.get("words") or []
    if not words:
        raise env.HCError("the transcript has no words", code=1)
    a = _parse_index(args.a, "A")
    b = _parse_index(args.b, "B")
    n = len(words)
    if a < 0 or b < 0:
        raise env.HCError("word indices start at 0, got %d %d (the transcript has 0-%d)"
                          % (a, b, n - 1), code=1)
    if a > b:
        raise env.HCError("A must not be after B, got %d %d (did you mean: hc words P %d %d?)"
                          % (a, b, b, a), code=1)
    if a >= n:
        raise env.HCError("word range %d-%d is outside the transcript (0-%d)" % (a, b, n - 1), code=1)
    b = min(n - 1, b)
    for row in word_rows(transcript, a, b):
        print(row)
    return 0
