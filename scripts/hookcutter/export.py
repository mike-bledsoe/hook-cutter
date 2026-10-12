"""Export folder + manifest (contract section 11).

export/<Style label>/NN - <Title>.mp4 and NN - <Title> - Cover.jpg, plus export/manifest.json.
Files are copied (never linked, so a later re-render cannot change an exported file in place).
Cleanup deletes only files the previous manifest listed that this export no longer produces;
nothing is ever glob-deleted.

Export only vouches for what QA checked: a clip whose QA is FAIL, INCOMPLETE, stale or missing
or has evidence gaps (checks that could not run), a style or format QA did not check, a render
older than its plan and a render that no longer matches its inputs (preset, transcript, shots,
source or render code changed since) are HELD (not copied, listed under "held") unless
force=True. A held clip keeps no exported file, also in a run restricted with --styles or
--formats. The manifest records, per clip, the QA status, whether it was complete, what it
checked, and the hook card and cover headline actually rendered.

One export per project at a time: the manifest is read, the files copied and the manifest
written back under a lock (cache/.export.lock), so two runs on different clips both land.
"""

from __future__ import annotations

import os
import re
import shutil
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

from . import env

MANIFEST = "manifest.json"
BY_EAR = "by-ear.md"
JOIN_LABEL = {"piece": "join between pieces", "omit": "omitted words", "tighten": "tightened pause",
              "filler": "removed filler"}
EAR_FLAGS = {"timing_uncertain": "the start time of this word is a guess: listen for a clipped "
                                 "start or a caption that lights early",
             "false_start": "two attempts at a phrase: listen for a stumble",
             "dropout": "the recogniser may have missed words here",
             "unworded_audio": "sound with no words: listen for a word the captions miss",
             "speaker_uncertain": "who is speaking is not certain",
             "no_audio": "a word with no audible sound"}
EAR_GATES = ("G5", "G6", "G7", "G10")
EXPORT_LOCK = ".export.lock"    # in the project's cache folder, never in the export folder
MAX_NAME_BYTES = 150
WIN_RESERVED = ({"CON", "PRN", "AUX", "NUL", "CONIN$", "CONOUT$"}
                | {f"COM{i}" for i in list(range(1, 10)) + ["\u00b9", "\u00b2", "\u00b3"]}
                | {f"LPT{i}" for i in list(range(1, 10)) + ["\u00b9", "\u00b2", "\u00b3"]})
_CHAR_MAP = {":": " -", "/": "-", "\\": "-", '"': "'", "<": "", ">": "", "|": "-", "?": "", "*": ""}
_KEEP_CF = {"\u200c", "\u200d"}       # ZWNJ / ZWJ: needed by emoji sequences and some scripts


def sanitize(name, fallback: str = "untitled", max_bytes: int = MAX_NAME_BYTES) -> str:
    """A file/folder name that is valid on Windows, macOS and Linux (NFC, no reserved names,
    no forbidden or control characters, no bidi or other format characters except ZWJ/ZWNJ,
    no trailing dot/space, bounded UTF-8 length)."""
    s = unicodedata.normalize("NFC", str(name or ""))
    s = "".join(" " if ch.isspace() else _CHAR_MAP.get(ch, ch) for ch in s)
    s = "".join(ch for ch in s if ch in _KEEP_CF or
                unicodedata.category(ch) not in ("Cc", "Cf", "Cs", "Zl", "Zp"))
    s = re.sub(r"\s+", " ", s).strip().strip(".").strip()
    if not s:
        s = fallback
    while len(s.encode("utf-8")) > max_bytes:
        s = s[:-1]
    s = s.rstrip(". \u200c\u200d") or fallback
    if s.split(".")[0].strip().upper() in WIN_RESERVED:
        s = s + "_"
    return s


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _place(src: Path, dest: Path) -> bool:
    """Copy src to dest atomically unless dest already holds that exact copy. True if written.
    On a case-insensitive file system an existing file whose name differs only in case is
    renamed to the new spelling first."""
    st = src.stat()
    if dest.is_file():
        try:
            names = os.listdir(dest.parent)
        except OSError:
            names = [dest.name]
        if dest.name not in names:
            twin = next((n for n in names if _same_name(n, dest.name)), None)
            if twin is not None:
                tmp = dest.with_name(f".{dest.name}.case.tmp")
                os.replace(dest.parent / twin, tmp)
                os.replace(tmp, dest)
        dt = dest.stat()
        if dt.st_size == st.st_size and dt.st_mtime_ns == st.st_mtime_ns:
            return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    # a fixed hidden temp name: export() lets one run at a time into the folder, and a re-run
    # reclaims the temp a killed run left behind
    tmp = dest.with_name(f".{dest.name}.tmp")
    try:
        shutil.copy2(src, tmp)
        os.replace(tmp, dest)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return True


def _fold(path: str) -> str:
    return unicodedata.normalize("NFC", str(path)).casefold()


def _same_name(a: str, b: str) -> bool:
    return _fold(a) == _fold(b)


def _mmss(t: float) -> str:
    t = max(0, int(round(t)))
    h, rem = divmod(t, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def chapters_for(clip: dict, plan: dict | None) -> list[dict]:
    """Chapters from pieces with a `chapter` title, out times from the plan."""
    out = []
    if not plan:
        return out
    by_n = {p.get("n"): p for p in plan.get("pieces") or []}
    for n, piece in enumerate(clip.get("pieces") or []):
        title = piece.get("chapter") if isinstance(piece, dict) else None
        pp = by_n.get(n)
        if not title or not pp or not pp.get("out"):
            continue
        t = float(pp["out"][0])
        out.append({"title": str(title), "start_out": round(t, 3), "timestamp": _mmss(t)})
    return out


def youtube_ready(chapters: list[dict], duration: float) -> bool:
    """YouTube's chapter rules: first at 0:00, at least 3, each at least 10 s long."""
    if len(chapters) < 3 or chapters[0]["start_out"] > 0.5:
        return False
    starts = [c["start_out"] for c in chapters] + [duration]
    return all(b - a >= 10.0 for a, b in zip(starts, starts[1:]))


def _suffixed(label: str, tag: str) -> str:
    """`label (tag)` within MAX_NAME_BYTES: the label gives up bytes, the tag never does."""
    tail = f" ({tag})"
    head = sanitize(label, max_bytes=max(1, MAX_NAME_BYTES - len(tail.encode("utf-8"))))
    return sanitize(head + tail)


def _style_labels(project, styles: list[str]) -> dict:
    """Export folder per style: the preset label, unique under the disk's case and normalization
    folding. Every style whose label another style shares gets " (<style>)", so the answer
    depends on the set of styles and never on their order (which --styles changes)."""
    base = {}
    for st in styles:
        try:
            from . import presets
            label = str(presets.load(st, project).get("label") or st)
        except Exception:
            label = st[:1].upper() + st[1:]
        base[st] = sanitize(label, fallback=sanitize(st))
    count = {}
    for label in base.values():
        count[_fold(label)] = count.get(_fold(label), 0) + 1
    labels, used = {}, set()
    for st in styles:                               # a label nobody shares keeps its folder
        if count[_fold(base[st])] == 1:
            labels[st] = base[st]
            used.add(_fold(base[st]))
    for st in sorted(s for s in base if s not in labels):
        label, n = _suffixed(base[st], st), 2
        while _fold(label) in used:                 # "Shared (cc)" is itself another style's label
            label, n = _suffixed(base[st], f"{st} {n}"), n + 1
        labels[st] = label
        used.add(_fold(label))
    return labels


def _read(path: Path):
    try:
        return env.read_json(path) if path.is_file() else None
    except env.HCError:
        return None


def _speakers(plan: dict | None) -> list[str]:
    """Speaker labels in order of first appearance among the plan's intended words."""
    out = []
    if not plan:
        return out
    by_i = {w.get("i"): w for w in plan.get("words") or []}
    for i in plan.get("intent") or []:
        w = by_i.get(i) or {}
        sp = w.get("sp")
        if sp and not w.get("omitted") and not w.get("removed") and sp not in out:
            out.append(sp)
    return out


def _rendered_hook(project, cid: str, styles: list, formats: list, plan: dict | None):
    """The hook card the render actually carries (captions.json hook.lines)."""
    for fmt in ["vertical"] + [f for f in formats if f != "vertical"]:
        for st in styles:
            cap = _read(Path(project.render_path(cid, fmt, st, ext=".captions.json")))
            if isinstance(cap, dict) and "hook" in cap:
                hook = cap.get("hook") or None
                return {"lines": list(hook.get("lines") or [])} if hook else None
    hc = (plan or {}).get("hook_card")
    return {"lines": list(hc.get("lines") or [])} if isinstance(hc, dict) and hc.get("lines") else None


def _has_gaps(rep) -> bool:
    """A qa/<clip>.json whose checks did not all run: WARN and FAIL reports can have evidence
    gaps too, and a WARN with gaps is not a clip QA vouched for. Missing keys count as complete."""
    if not isinstance(rep, dict):
        return False
    try:
        gaps = int(rep.get("gaps") or 0)
    except (TypeError, ValueError):
        gaps = 0
    return rep.get("complete") is False or gaps > 0


def _inputs_changed(project, cid: str, fmt: str, st: str) -> bool:
    """The render's log key differs from the key a render made now would record (preset,
    transcript, shots, source or render code changed since). A log without that key, or inputs
    that cannot be computed, never make a render stale (the rule `hc status` uses)."""
    log = _read(Path(project.render_path(cid, fmt, st, ext=".log.json"))) or {}
    have = (log.get("hashes") or {}).get("render") if isinstance(log, dict) else None
    if not have:
        return False
    try:
        from . import render as render_mod
        return have != render_mod.expected_hashes(project, cid, fmt, st)["render"]
    except Exception:
        return False


def _render_stale(project, cid: str, fmt: str, st: str, plan: dict | None, qa_rep: dict | None) -> bool:
    """The render was made from another plan than the current plan file, or QA found it stale."""
    rel = project.rel(project.render_path(cid, fmt, st))
    if qa_rep and rel in (qa_rep.get("stale_renders") or []):
        return True
    log = _read(Path(project.render_path(cid, fmt, st, ext=".log.json"))) or {}
    have = (log.get("hashes") or {}).get("plan")
    if not have or not plan:
        return False
    try:
        from . import render as render_mod
        return have != render_mod.plan_hash(plan)
    except Exception:
        return False


def export(project, clip_ids=None, styles=None, formats=None, force: bool = False) -> dict:
    """Write the export folder and manifest; returns the manifest plus "written", "deleted",
    "held" ([{"clip", "style", "format", "reason"}]) and "missing" (inputs, and manifest files
    that are not on disk). `styles` / `formats` restrict the run; `force` exports held files too.

    One export per project at a time: the manifest is read, the files copied and the manifest
    written back under a lock, so a run for one clip never drops the entry another run wrote."""
    from . import fsutil
    with fsutil.file_lock(Path(project.cache_dir()) / EXPORT_LOCK, what="the export folder"):
        return _export(project, clip_ids, styles, formats, force)


def _export(project, clip_ids, styles, formats, force: bool) -> dict:
    from . import qa
    spec = project.spec()
    clips = spec.get("clips") or {}
    all_ids = list(clips)
    ids = list(clip_ids or []) or all_ids
    for c in ids:
        if c not in clips:
            raise env.unknown_clip(c, all_ids)
    root = Path(project.export_dir())
    prev = _read(root / MANIFEST) or {}
    prev_clips = {c.get("id"): c for c in prev.get("clips") or [] if isinstance(c, dict)}
    width = max(2, len(str(len(all_ids))))
    style_union = []
    for cid in ids:
        for st in (styles or (clips[cid].get("_eff") or {}).get("styles") or ["clean"]):
            if st not in style_union:
                style_union.append(st)
    for pc in prev_clips.values():
        for f in pc.get("files") or []:
            if f.get("style") and f["style"] not in style_union:
                style_union.append(f["style"])
    labels = _style_labels(project, style_union)
    written, missing, entries, held = [], [], [], []
    for cid in all_ids:
        clip = clips[cid]
        if cid not in ids:
            if cid in prev_clips:
                entries.append(prev_clips[cid])
            continue
        eff = clip.get("_eff") or {}
        own_formats = list(eff.get("formats") or ["vertical"])
        own_styles = list(eff.get("styles") or ["clean"])
        cformats = list(formats or own_formats)
        cstyles = list(styles or own_styles)
        nn = str(all_ids.index(cid) + 1).zfill(width)
        title = sanitize(clip.get("title") or cid, fallback=cid)
        base = f"{nn} - {title}"
        qa_file = Path(project.qa_path(cid, ".json"))
        qa_rep = _read(qa_file)
        if not isinstance(qa_rep, dict):
            qa_rep = None
        plan = _read(Path(project.plan_path(cid)))
        qa_status = str(qa_rep.get("status") or "unknown") if qa_rep else "not run"
        stale_why = qa.qa_staleness(project, spec, cid, qa_rep, qa_file) if qa_rep else []
        checked = (qa_rep or {}).get("checked") or {"styles": (qa_rep or {}).get("styles") or [],
                                                    "formats": (qa_rep or {}).get("formats") or []}
        if qa_rep is None:
            clip_hold = "QA has not run: hc qa"
        elif qa_status == "FAIL":
            clip_hold = "QA FAIL (see qa/%s.md)" % cid
        elif qa_status == "INCOMPLETE":
            clip_hold = "QA INCOMPLETE: some checks could not run (see qa/%s.md)" % cid
        elif _has_gaps(qa_rep):
            clip_hold = ("QA %s with evidence gaps: some checks could not run (see qa/%s.md); "
                         "run hc qa --require-coverage" % (qa_status, cid))
        elif stale_why:
            clip_hold = "QA is stale: " + stale_why[0]
        else:
            clip_hold = None
        files, any_stale, held_here = [], False, []
        for st in cstyles:
            sdir = root / labels[st]
            vert_ok = None
            for fmt in cformats:
                src = Path(project.render_path(cid, fmt, st))
                # vertical never has a suffix; another format drops it only when it is the clip's
                # one format (with --formats, the clip's own formats still count: no name clash)
                suffix = "" if (fmt == "vertical" or set(own_formats) | set(cformats) == {fmt}) \
                    else f" - {fmt[:1].upper()}{fmt[1:]}"
                dest = sdir / f"{base}{suffix}.mp4"
                if not src.is_file():
                    missing.append(project.rel(src))
                    continue
                rstale = _render_stale(project, cid, fmt, st, plan, qa_rep)
                any_stale = any_stale or rstale
                reason = clip_hold
                if reason is None and (st not in (checked.get("styles") or []) or fmt not in (checked.get("formats") or [])):
                    reason = f"QA did not check {fmt}/{st}: hc qa --styles {st} --formats {fmt}"
                if reason is None and rstale:
                    reason = "render is older than the plan: hc render"
                if reason is None and _inputs_changed(project, cid, fmt, st):
                    rstale = any_stale = True
                    reason = ("render predates its inputs (preset, transcript, shots or code changed): "
                              "hc render, then hc qa")
                if fmt == "vertical":
                    vert_ok = reason is None or force
                if reason and not force:
                    held_here.append({"clip": cid, "style": st, "format": fmt, "reason": reason})
                    continue
                if _place(src, dest):
                    written.append(str(dest.relative_to(root)))
                f = {"style": st, "format": fmt, "kind": "video", "path": str(dest.relative_to(root)),
                     "bytes": dest.stat().st_size, "qa_checked": reason is None or not reason.startswith("QA did not"),
                     "render_stale": rstale}
                if reason:
                    f["forced"] = reason
                files.append(f)
            if "vertical" in cformats:
                csrc = Path(project.cover_path(cid, st, "main"))
                dest = sdir / f"{base} - Cover.jpg"
                if not csrc.is_file():
                    missing.append(project.rel(csrc))
                elif vert_ok:
                    if _place(csrc, dest):
                        written.append(str(dest.relative_to(root)))
                    files.append({"style": st, "format": "vertical", "kind": "cover",
                                  "path": str(dest.relative_to(root)), "bytes": dest.stat().st_size})
        old = prev_clips.get(cid) or {}
        for f in old.get("files") or []:
            restricted = ((styles and f.get("style") not in cstyles and f.get("style") in own_styles) or
                          (formats and f.get("format") not in cformats and f.get("format") in own_formats))
            if not restricted or not (root / f.get("path", "")).is_file():
                continue
            if _fold(f.get("path", "")) in {_fold(g["path"]) for g in files}:
                continue                    # this run just wrote that path: it is no longer that file
            if clip_hold and not force:
                # a clip-wide hold covers every variant: the kept one goes too (the cleanup
                # below removes it), as it does in a run that is not restricted
                pair = (f.get("style"), f.get("format"))
                if pair not in [(h["style"], h["format"]) for h in held_here]:
                    held_here.append({"clip": cid, "style": pair[0], "format": pair[1], "reason": clip_hold})
                continue
            files.append(f)
        held += held_here
        cover_meta = None
        for st in cstyles:
            cover_meta = cover_meta or _read(Path(project.cover_path(cid, st, "json")))
        duration = float(plan.get("duration_out")) if plan and plan.get("duration_out") is not None else None
        chapters = chapters_for(clip, plan) if "landscape" in cformats else []
        entry = {
            "id": cid, "n": int(nn), "title": clip.get("title") or cid,
            "speaker": clip.get("speaker"), "speakers": _speakers(plan),
            "files": files, "duration": round(duration, 3) if duration is not None else None,
            "hook_card": _rendered_hook(project, cid, cstyles, cformats, plan),
            "cover_headline": (cover_meta or {}).get("headline"),
            "post": clip.get("post"), "qa_status": qa_status,
            "qa_complete": bool((qa_rep or {}).get("complete")) if qa_rep else False,
            "qa_checked": {"styles": list(checked.get("styles") or []), "formats": list(checked.get("formats") or [])},
            "qa_stale": bool(stale_why), "qa_stale_reasons": stale_why, "render_stale": any_stale,
            "held": [{"style": h["style"], "format": h["format"], "reason": h["reason"]} for h in held_here],
            "thumb_offset_ms": 0, "chapters": chapters,
        }
        if chapters:
            entry["chapters_youtube_ready"] = youtube_ready(chapters, duration or 0.0)
        entries.append(entry)
    keep = {f["path"] for e in entries for f in e.get("files") or []}
    keep_fold = {}
    for k in keep:
        keep_fold.setdefault(_fold(k), []).append(k)
    deleted = []
    root_abs = os.path.normpath(str(root.absolute()))
    for pc in prev_clips.values():
        for f in pc.get("files") or []:
            rel = f.get("path")
            if not rel or rel in keep or os.path.isabs(rel):
                continue
            p = Path(os.path.normpath(os.path.join(root_abs, rel)))
            if not str(p).startswith(root_abs + os.sep) or p.name == MANIFEST:
                continue
            twins = [Path(root_abs) / k for k in keep_fold.get(_fold(rel), [])]
            if any(_samefile(p, t) for t in twins):
                continue                    # the same file under another case or normalization
            if p.is_file() or p.is_symlink():
                p.unlink()
                deleted.append(rel)
                if p.parent != Path(root_abs):
                    try:
                        p.parent.rmdir()
                    except OSError:
                        pass
    absent = sorted(k for k in keep if not (root / k).is_file())
    missing += [f"export/{k} (listed in the manifest but not on disk)" for k in absent]
    manifest = {
        "version": 2, "created": _now(), "code_version": env.code_version(),
        "styles": [{"style": s, "label": labels[s], "dir": labels[s]} for s in style_union
                   if any(f.get("style") == s for e in entries for f in e.get("files") or [])],
        "clips": entries, "missing": missing, "held": held, "forced": bool(force),
        "by_ear": BY_EAR,
    }
    env.write_json(root / MANIFEST, manifest)
    (root / BY_EAR).write_text(by_ear(project, entries), encoding="utf-8")
    return {**manifest, "written": written, "deleted": deleted}


def _clock(t: float) -> str:
    t = max(0.0, float(t))
    return "%d:%04.1f" % (int(t // 60), t - 60 * int(t // 60))


def _spoken(plan: dict) -> list:
    """The words heard in the clip, in output order (not hidden, omitted or removed)."""
    ws = [w for w in plan.get("words") or [] if isinstance(w.get("s_out"), (int, float))
          and not (w.get("hidden") or w.get("omitted") or w.get("removed"))]
    return sorted(ws, key=lambda w: w["s_out"])


def _around(words: list, t: float, n: int = 4) -> tuple[str, str]:
    before = [w["raw"] for w in words if w["e_out"] <= t + 0.02][-n:]
    after = [w["raw"] for w in words if w["s_out"] >= t - 0.02][:n]
    return " ".join(before), " ".join(after)


def _src_to_out(plan: dict, t: float):
    for seg in plan.get("segments") or []:
        a, b = seg["src"]
        if a - 1e-6 <= t <= b + 1e-6:
            return seg["out"][0] + (t - a)
    return None


def by_ear(project, entries: list) -> str:
    """export/by-ear.md: what to listen for in each exported clip. The agent cannot hear, so
    every join, the first and last words, the words the transcript flags and the sound-related
    QA warnings are listed with their time in the exported file, for a person to check."""
    lines = ["# By-ear checklist", "",
             "The agent that made these clips cannot listen. Play each one with the sound on and "
             "check every line. Send a problem back as one plain note with its time, for "
             "example \"clipped word at 0:03.0\" or \"the caption lights early at 0:11.2\".", ""]
    flags_by_word: dict = {}
    timed_flags: list = []
    try:
        tr = env.read_json(Path(project.transcript_path()))
        for f in tr.get("flags") or []:
            if isinstance(f.get("a"), int) and isinstance(f.get("b"), int):
                for i in range(f["a"], f["b"] + 1):
                    flags_by_word.setdefault(i, []).append(f.get("kind"))
            elif isinstance(f.get("s"), (int, float)):
                timed_flags.append(f)
    except env.HCError:
        pass
    shown = [e for e in entries if any(f.get("kind") == "video" for f in e.get("files") or [])]
    if not shown:
        lines.append("Nothing is exported yet.")
    for e in shown:
        cid = e["id"]
        plan = _read(Path(project.plan_path(cid))) or {}
        video = next(f["path"] for f in e["files"] if f.get("kind") == "video")
        dur = e.get("duration")
        lines += ["## %02d %s (%s)" % (e["n"], e["title"], ("%.1f s" % dur) if dur else "length unknown"), "",
                  "File: `export/%s`" % video, ""]
        words = _spoken(plan)
        if not words:
            lines += ["- [ ] Listen to the whole clip (no word timing in its plan).", ""]
            continue
        joins = sorted(plan.get("joins") or [], key=lambda j: j["out"])
        lo = joins[0]["out"] if joins else None
        hi = joins[-1]["out"] if joins else None
        first = [w for w in words if lo is None or w["e_out"] <= lo + 0.02][:4] or words[:4]
        last = [w for w in words if hi is None or w["s_out"] >= hi - 0.02][-4:] or words[-4:]
        items = [(first[0]["s_out"], "the opening, \"%s\": is the first word whole, with no sound "
                  "before it that does not belong?" % " ".join(w["raw"] for w in first))]
        for j in joins:
            b, a = _around(words, j["out"])
            what = JOIN_LABEL.get(j.get("kind"), j.get("kind") or "join")
            items.append((j["out"], "%s: \"%s\" | \"%s\": no clipped word, no stray syllable, "
                          "no click%s" % (what, b, a, ", the breath sounds natural"
                                          if j.get("kind") == "tighten" else "")))
        if plan.get("kind") != "caption_only":
            seen = set()
            for w in words:
                for kind in flags_by_word.get(w.get("i"), []):
                    if kind in EAR_FLAGS and (kind, w.get("i")) not in seen:
                        seen.add((kind, w.get("i")))
                        items.append((w["s_out"], "\"%s\": %s" % (w["raw"], EAR_FLAGS[kind])))
            for f in timed_flags:
                t = _src_to_out(plan, float(f["s"]))
                if t is not None and f.get("kind") in EAR_FLAGS:
                    items.append((t, EAR_FLAGS[f["kind"]]))
        items.append((last[-1]["e_out"], "the ending, \"%s\": does it end clean, not cut off "
                      "mid-word?" % " ".join(w["raw"] for w in last)))
        for t, text in sorted(items, key=lambda x: x[0]):
            lines.append("- [ ] %s %s" % (_clock(t), text))
        qa_rep = _read(Path(project.qa_path(cid, ".json"))) or {}
        for g in qa_rep.get("gates") or []:
            if g.get("id") in EAR_GATES and g.get("status") in ("WARN", "FAIL"):
                ev = str(g.get("evidence") or "")
                lines.append("- [ ] QA %s %s: %s" % (g["id"], g["status"], ev[:240] + ("..." if len(ev) > 240 else "")))
        lines.append("- [ ] Captions: each word lights as it is said, none early or late.")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _samefile(a: Path, b: Path) -> bool:
    try:
        return a.exists() and b.exists() and os.path.samefile(a, b)
    except OSError:
        return False


def cmd_export(args) -> int:
    """hc export P [CLIP...] [--styles ...] [--formats ...] [--force]"""
    from .project import Project
    project = Project(args.project).require()
    res = export(project, list(getattr(args, "clips", None) or []), getattr(args, "styles", None),
                 formats=getattr(args, "formats", None), force=bool(getattr(args, "force", False)))
    root = project.rel(project.export_dir())
    print(f"export: {len(res['written'])} file(s) written, {len(res['deleted'])} removed -> {root}/")
    print(f"by-ear checklist: {root}/{BY_EAR} (play every clip with the sound on and check each line)")
    for e in res["clips"]:
        flag = "" if e.get("qa_status") in ("PASS", "WARN") and not e.get("qa_stale") and e.get("qa_complete", True) else \
            f"  (QA {e.get('qa_status')}{', incomplete' if e.get('qa_complete') is False else ''}" \
            f"{', stale' if e.get('qa_stale') else ''})"
        print(f"  {e['n']:02d} {e['title']}: {len(e.get('files') or [])} file(s){flag}")
    rc = 0
    if res["held"]:
        print("held (not exported; fix and re-run QA, or export anyway with --force):")
        for h in res["held"]:
            print(f"  {h['clip']} {h.get('format', '')}/{h.get('style', '')}: {h['reason']}")
        rc = 1
    if res["missing"]:
        print("missing (render or cover first): " + ", ".join(res["missing"]))
        rc = 1
    return rc
