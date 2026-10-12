"""Docs stay true to the tool.

- SKILL.md frontmatter and size limits; every referenced file exists.
- Every `hc <command> --flag` written in the docs exists in the CLI (the documented command
  surface below, and the real argparse parser), and every flag the parser has is in the
  SKILL.md command reference.
- examples/clips.example.json, every ```json clips.json block in references/spec.md, and every
  clip fragment in a ```json block of the references validate with hookcutter.spec with zero
  lints (against a synthetic transcript that satisfies their `expect` anchors). Skipped with a
  clear message while spec.py does not exist.
- The preset values quoted in references/spec.md match presets/clean.json; the safe-zone table
  matches presets/safezones.json.
- No em-dashes in docs, JSON, or string literals of the shipped scripts (house style).
"""
from __future__ import annotations

import argparse
import copy
import io
import json
import re
import sys
import tokenize
import unittest
from pathlib import Path

SK = Path(__file__).resolve().parents[1]
SCRIPTS = SK / "scripts"
REFS = SK / "references"
DOCS = [SK / "SKILL.md", SK / "README.md"] + sorted(REFS.glob("*.md"))
REQUIRED_REFS = ("hooks", "editing", "captions", "packaging", "fidelity", "qa", "spec")
EM_DASH = chr(0x2014)
EM_DASH_OK = "em-dash ok"          # line marker for a deliberate, non user-facing use
# In Python string literals only prose use counts (an em-dash between words, optionally
# spaced); character classes and translation tables that normalise input text are fine.
PROSE_EM_DASH_RE = re.compile(r"[A-Za-z0-9,.;:)'\"]\s?" + EM_DASH + r"\s?[A-Za-z0-9('\"]")

# The documented command surface (SKILL.md "Command reference"). "--help" is always accepted.
DOCUMENTED_CLI = {
    "doctor": {"--install", "--with", "--model", "--backend", "--python", "--json"},
    "ingest": {"--copy", "--section", "--cfr", "--force"},
    "transcribe": {"--backend", "--model", "--language", "--glossary", "--speakers-vtt",
                   "--file", "--clip", "--retime", "--force"},
    "find": {"--near", "--max", "--clip"},
    "words": {"--clip"},
    "peek": {"--n", "--file"},
    "shots": {"--range"},
    "plan": {"--quiet"},
    "render": {"--styles", "--formats", "--force"},
    "cover": {"--styles", "--at", "--grid"},
    "qa": {"--styles", "--formats", "--no-asr", "--require-coverage"},
    "export": {"--styles", "--formats", "--force"},
    "all": {"--styles", "--formats", "--no-asr", "--force"},
    "status": set(),
    "test": {"--slow"},
    "demo": {"--force", "--open"},
}

FENCE_RE = re.compile(r"^```([^\n`]*)\n(.*?)^```[ \t]*$", re.M | re.S)
INLINE_RE = re.compile(r"`([^`\n]+)`")
HC_RE = re.compile(r"(?:^|[\s/(\[])hc\s+([a-z][a-z-]*)(.*)$")
FLAG_RE = re.compile(r"(?<![\w-])(--[a-z][a-z0-9-]*)")


# ---- helpers -------------------------------------------------------------------------------

def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def fenced_blocks(text: str) -> list:
    """[(info_string, body)] for every fenced code block."""
    return [(m.group(1).strip(), m.group(2)) for m in FENCE_RE.finditer(text)]


def code_snippets(text: str) -> list:
    """Lines of fenced blocks plus inline code spans outside them."""
    out = []
    for _, body in fenced_blocks(text):
        out.extend(body.splitlines())
    prose = FENCE_RE.sub("", text)
    out.extend(m.group(1) for m in INLINE_RE.finditer(prose))
    return out


def hc_commands(text: str) -> list:
    """[(command, {flags}, snippet)] for every `hc <command> ...` written in code."""
    found = []
    for snip in code_snippets(text):
        m = HC_RE.search(snip)
        if m:
            found.append((m.group(1), set(FLAG_RE.findall(m.group(2))), snip.strip()))
    return found


def real_cli():
    """{command: {--flags}} from hookcutter.cli's parser, or None if not available yet."""
    if not (SCRIPTS / "hookcutter" / "cli.py").exists():
        return None
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    from hookcutter import cli  # noqa: E402  (an import error here is a real failure)
    parser = None
    for name in ("build_parser", "make_parser", "get_parser", "parser"):
        obj = getattr(cli, name, None)
        if isinstance(obj, argparse.ArgumentParser):
            parser = obj
        elif callable(obj):
            try:
                cand = obj()
            except TypeError:
                continue
            if isinstance(cand, argparse.ArgumentParser):
                parser = cand
        if parser is not None:
            break
    if parser is None:
        return None
    cmds = {}
    for action in parser._actions:                      # argparse has no public walker
        if isinstance(action, argparse._SubParsersAction):
            for cname, sub in action.choices.items():
                cmds[cname] = {s for a in sub._actions for s in a.option_strings
                               if s.startswith("--")}
    return cmds or None


def spec_module():
    if not (SCRIPTS / "hookcutter" / "spec.py").exists():
        raise unittest.SkipTest("hookcutter/spec.py is missing: docs examples were not "
                                "validated")
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    import hookcutter.spec as spec  # noqa: E402
    return spec


def _place(tokens: dict, idx: int, word: str, label: str):
    if idx < 0:
        raise AssertionError(f"{label}: expect text is longer than its piece")
    prev = tokens.get(idx)
    if prev is not None and prev != word:
        raise AssertionError(f"{label}: conflicting expect anchors at word {idx} "
                             f"({prev!r} vs {word!r})")
    tokens[idx] = word


def _words(n: int, tokens: dict) -> list:
    return [{"i": i, "w": tokens.get(i, "word"), "s": round(i * 0.4, 3),
             "e": round(i * 0.4 + 0.3, 3), "p": 0.95, "sp": None} for i in range(n)]


def synthetic_inputs(raw: dict, label: str):
    """A transcript (and caption-only transcripts) whose words satisfy every expect anchor."""
    tokens: dict = {}
    top = 0
    clip_needs: dict = {}
    for cid, clip in (raw.get("clips") or {}).items():
        if not isinstance(clip, dict):
            continue
        idx = [int(k) for k in (clip.get("fix") or {})]
        for x in clip.get("emphasis") or []:
            idx.extend(x if isinstance(x, list) else [x])
        if clip.get("kind") == "caption_only":
            clip_needs[cid] = max(idx, default=0)
            continue
        for x in clip.get("omit") or []:
            idx.extend(x if isinstance(x, list) else [x])
        for r in clip.get("verbatim") or []:
            idx.extend(r)
        for p in clip.get("pieces") or []:
            if not isinstance(p, dict) or "words" not in p:
                continue
            a, b = p["words"]
            idx.append(b)
            first, last = (list(p.get("expect") or []) + ["", ""])[:2]
            for k, w in enumerate((first or "").split()):
                _place(tokens, a + k, w, f"{label}/{cid}")
            lw = (last or "").split()
            for k, w in enumerate(lw):
                _place(tokens, b - len(lw) + 1 + k, w, f"{label}/{cid}")
        top = max([top] + idx)
    words = _words(top + 60, tokens)
    transcript = {"version": 1, "words": words, "lines": [], "flags": [], "sha256": "synthetic"}
    source = {"duration": max(7200.0, words[-1]["e"] + 10.0)}
    clip_transcripts = {cid: {"version": 1, "words": _words(n + 60, {}), "lines": [],
                              "flags": [], "sha256": "synthetic"}
                        for cid, n in clip_needs.items()}
    return transcript, source, clip_transcripts


def validate_doc(spec, raw: dict, label: str) -> list:
    """Problems (strings) for one clips.json document: lints plus doc doctrine."""
    norm = spec.normalize_spec(copy.deepcopy(raw))
    transcript, source, clip_transcripts = synthetic_inputs(raw, label)
    lints = spec.validate(norm, transcript=transcript, source=source,
                          clip_transcripts=clip_transcripts)
    problems = [f"{label}: {lt.get('id')} {lt.get('level')} [{lt.get('clip')}] {lt.get('msg')}"
                for lt in lints]
    for cid, clip in (raw.get("clips") or {}).items():
        for n, p in enumerate((clip or {}).get("pieces") or []):
            if "words" in p:
                exp = (list(p.get("expect") or []) + ["", ""])[:2]
                if not (exp[0] and exp[1]):
                    problems.append(f"{label}: [{cid}] pieces[{n}] has no expect anchors "
                                    "(the docs promise every piece carries them)")
    return problems


def doc_json_blocks():
    """[(label, info, parsed_or_None, error)] for every json block in the references."""
    out = []
    for path in [SK / "SKILL.md", SK / "README.md"] + sorted(REFS.glob("*.md")):
        for k, (info, body) in enumerate(fenced_blocks(_read(path))):
            words = info.split()
            if not words or words[0] != "json":
                continue
            label = f"{path.relative_to(SK).as_posix()} block {k + 1}"
            try:
                out.append((label, info, json.loads(body), None))
                continue
            except ValueError as exc:
                err = str(exc)
            try:                                      # a clip fragment: "id": {...}
                out.append((label, info, json.loads("{" + body + "}"), None))
            except ValueError:
                out.append((label, info, None, err))
    return out


def _dig(obj, dotted: str):
    for part in dotted.split("."):
        obj = obj[part]
    return obj


# ---- tests ---------------------------------------------------------------------------------

class SkillDocTest(unittest.TestCase):

    def test_frontmatter(self):
        text = _read(SK / "SKILL.md")
        m = re.match(r"^---\n(.*?)\n---\n(.*)$", text, re.S)
        self.assertIsNotNone(m, "SKILL.md must start with a --- frontmatter block")
        front, body = m.group(1), m.group(2)
        fields = dict(line.split(":", 1) for line in front.splitlines() if ":" in line)
        self.assertEqual(fields.get("name", "").strip(), "hook-cutter")
        desc = fields.get("description", "").strip()
        self.assertTrue(desc, "description is empty")
        self.assertLessEqual(len(desc), 1024, f"description is {len(desc)} chars (max 1024)")
        for trigger in ("reels", "shorts", "hooks", "caption"):
            self.assertIn(trigger, desc.lower())
        self.assertLessEqual(len(body.splitlines()), 260, "SKILL.md body is over ~250 lines")

    def test_referenced_files_exist(self):
        for name in REQUIRED_REFS:
            self.assertTrue((REFS / f"{name}.md").is_file(), f"references/{name}.md is missing")
        for name in ("README.md", "LICENSE", "THIRD_PARTY.md", "examples/clips.example.json"):
            self.assertTrue((SK / name).is_file(), f"{name} is missing")
        for doc in DOCS:
            for ref in re.findall(r"(references/[a-z_-]+\.md|examples/[\w.-]+\.json)", _read(doc)):
                self.assertTrue((SK / ref).is_file(), f"{doc.name} points at missing {ref}")
            for ref in re.findall(r"`([a-z_-]+\.md)`", _read(doc)):
                if ref in {f"{n}.md" for n in REQUIRED_REFS}:
                    self.assertTrue((REFS / ref).is_file(), f"{doc.name} points at {ref}")

    def test_gates_a_to_e_are_in_the_workflow(self):
        text = _read(SK / "SKILL.md")
        for gate in "ABCDE":
            self.assertRegex(text, rf"\*\*Gate {gate} \(", f"SKILL.md lost gate {gate}")


class CliDocTest(unittest.TestCase):

    def _check(self, cli: dict, source: str):
        problems = []
        seen = 0
        for doc in DOCS:
            for cmd, flags, snip in hc_commands(_read(doc)):
                seen += 1
                if cmd not in cli:
                    problems.append(f"{doc.name}: unknown command {cmd!r} in `{snip}`")
                    continue
                bad = sorted(flags - cli[cmd] - {"--help"})
                if bad:
                    problems.append(f"{doc.name}: {cmd} has no {', '.join(bad)} ({source}) "
                                    f"in `{snip}`")
        self.assertGreater(seen, 20, "found almost no hc commands in the docs")
        self.assertEqual(problems, [], "\n".join(problems))

    def test_doc_commands_match_documented_surface(self):
        self._check(DOCUMENTED_CLI, "SKILL.md command reference")

    def test_doc_commands_match_real_parser(self):
        cli = real_cli()
        if cli is None:
            self.skipTest("hookcutter/cli.py has no parser to inspect")
        self._check(cli, "cli.py")

    def test_every_command_is_documented(self):
        skill = _read(SK / "SKILL.md")
        documented = {cmd for cmd, _, _ in hc_commands(skill)}
        self.assertEqual(sorted(set(DOCUMENTED_CLI) - documented), [],
                         "SKILL.md command reference is missing commands")

    def test_every_flag_is_documented(self):
        """The parser and the SKILL.md command reference list the same flags per command."""
        cli = real_cli()
        if cli is None:
            self.skipTest("hookcutter/cli.py has no parser to inspect")
        skill = _read(SK / "SKILL.md")
        table = skill[skill.index("## Command reference"):]
        in_table = {}
        for cmd, flags, _ in hc_commands(table):
            in_table.setdefault(cmd, set()).update(flags)
        problems = []
        for cmd, flags in sorted(cli.items()):
            missing = sorted(flags - in_table.get(cmd, set()) - {"--help"})
            if missing:
                problems.append(f"hc {cmd}: {', '.join(missing)} not in the SKILL.md command reference")
            extra = sorted(DOCUMENTED_CLI.get(cmd, set()) ^ (flags - {"--help"}))
            if extra:
                problems.append(f"hc {cmd}: DOCUMENTED_CLI differs from the parser: {', '.join(extra)}")
        self.assertEqual(problems, [], "\n".join(problems))


class ExampleSpecTest(unittest.TestCase):

    def test_example_file_validates(self):
        spec = spec_module()
        raw = json.loads(_read(SK / "examples" / "clips.example.json"))
        self.assertGreaterEqual(len(raw["clips"]), 2)
        problems = validate_doc(spec, raw, "examples/clips.example.json")
        self.assertEqual(problems, [], "\n".join(problems))

    def test_harness_catches_drift(self):
        """Negative control: a drifted anchor and an unknown key must not validate."""
        spec = spec_module()
        raw = json.loads(_read(SK / "examples" / "clips.example.json"))
        transcript, source, ctr = synthetic_inputs(raw, "control")
        bad = copy.deepcopy(raw)
        clip = next(iter(bad["clips"].values()))
        clip["pieces"][0]["words"] = [w + 3 for w in clip["pieces"][0]["words"]]
        clip["colour_grade"] = "warm"
        lints = spec.validate(spec.normalize_spec(bad), transcript=transcript, source=source,
                              clip_transcripts=ctr)
        ids = {(lt.get("id"), lt.get("level")) for lt in lints}
        self.assertIn(("L02", "FAIL"), ids, "a drifted expect anchor was not caught")
        self.assertIn(("L01", "WARN"), ids, "an unknown key was not reported")

    def test_example_file_shows_the_editing_controls(self):
        raw = json.loads(_read(SK / "examples" / "clips.example.json"))
        first = next(iter(raw["clips"].values()))
        roles = [p.get("role") for p in first["pieces"]]
        self.assertEqual(roles[0], "cold_open")
        for key in ("omit", "verbatim", "hook_card", "framing", "cover"):
            self.assertTrue(first.get(key), f"the cold-open example should show {key}")
        self.assertTrue(any(p.get("hold_after") for p in first["pieces"]))

    def test_spec_md_clips_blocks_validate(self):
        spec = spec_module()
        blocks = [(k, body) for k, (info, body) in enumerate(fenced_blocks(_read(REFS / "spec.md")))
                  if "clips.json" in info.split()]
        self.assertGreaterEqual(len(blocks), 3, "spec.md lost its validated clips.json examples")
        problems = []
        for k, body in blocks:
            label = f"references/spec.md block {k + 1}"
            try:
                raw = json.loads(body)
            except ValueError as exc:
                problems.append(f"{label}: not valid JSON: {exc}")
                continue
            problems.extend(validate_doc(spec, raw, label))
        self.assertEqual(problems, [], "\n".join(problems))

    def test_json_blocks_parse_and_clip_fragments_validate(self):
        problems = []
        fragments = []
        for label, info, parsed, err in doc_json_blocks():
            if parsed is None:
                problems.append(f"{label}: does not parse as JSON ({err})")
            elif "clips.json" in info.split():
                continue                      # covered by test_spec_md_clips_blocks_validate
            elif "clips" in parsed:
                fragments.append((label, parsed))
            elif len(parsed) == 1 and isinstance(next(iter(parsed.values())), dict) and \
                    "pieces" in next(iter(parsed.values())):
                fragments.append((label, {"version": 1, "clips": parsed}))
        self.assertEqual(problems, [], "\n".join(problems))
        if not fragments:
            return
        spec = spec_module()
        for label, raw in fragments:
            problems.extend(validate_doc(spec, raw, label))
        self.assertEqual(problems, [], "\n".join(problems))


class PresetDocTest(unittest.TestCase):

    def _clean(self):
        path = SK / "presets" / "clean.json"
        if not path.exists():
            self.skipTest("presets/clean.json is missing")
        return json.loads(_read(path))

    def test_quoted_clean_values_match_preset(self):
        clean = self._clean()
        text = _read(REFS / "spec.md")
        problems, checked = [], 0
        for section in re.split(r"^### ", text, flags=re.M)[1:]:
            head, _, rest = section.partition("\n")
            if "`clean` value" not in rest:
                continue
            group = head.strip()
            for key, value in re.findall(r"^\| `([a-z_.0-9]+)` \| `([^`]+)` \|", rest, re.M):
                try:
                    actual = _dig(clean[group], key)
                except (KeyError, TypeError):
                    problems.append(f"spec.md {group}.{key}: not in presets/clean.json")
                    continue
                try:
                    quoted = json.loads(value)
                except ValueError:
                    quoted = value
                checked += 1
                if quoted != actual:
                    problems.append(f"spec.md {group}.{key} says {value}, clean.json has "
                                    f"{json.dumps(actual)}")
        self.assertGreater(checked, 10, "found almost no preset values to check")
        self.assertEqual(problems, [], "\n".join(problems))

    def test_safezones_block_matches_file(self):
        path = SK / "presets" / "safezones.json"
        if not path.exists():
            self.skipTest("presets/safezones.json is missing")
        blocks = [json.loads(b) for info, b in fenced_blocks(_read(REFS / "spec.md"))
                  if info.split() == ["json"] and '"organic"' in b]
        blocks = [b for b in blocks if "organic" in b]
        self.assertEqual(len(blocks), 1, "spec.md should show presets/safezones.json once")
        self.assertEqual(blocks[0], json.loads(_read(path)))

    def test_custom_preset_example_uses_known_keys(self):
        clean = self._clean()

        def unknown(obj, ref, where):
            out = []
            for k, v in obj.items():
                if k not in ref:
                    out.append(f"{where}{k}")
                elif isinstance(v, dict) and isinstance(ref[k], dict):
                    out.extend(unknown(v, ref[k], f"{where}{k}."))
            return out

        blocks = [json.loads(b) for info, b in fenced_blocks(_read(REFS / "spec.md"))
                  if info.split()[:1] == ["json"] and '"extends"' in b]
        self.assertTrue(blocks, "spec.md lost its custom preset example")
        for preset in blocks:
            self.assertEqual(unknown(preset, clean, ""), [])


class HouseStyleTest(unittest.TestCase):

    def test_no_em_dashes(self):
        problems = []
        skip = {"fonts", "models", "__pycache__", "tests"}
        for path in sorted(SK.rglob("*")):
            rel = path.relative_to(SK)
            if not path.is_file() or any(p in skip for p in rel.parts[:-1]):
                continue
            if rel.parts[0] == "tests" or path.suffix.lower() not in {
                    ".md", ".json", ".txt", ".py", ""}:
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            if path.suffix == ".py" or (path.suffix == "" and text.startswith("#!")
                                        and "python" in text.splitlines()[0]):
                problems.extend(f"{rel}:{ln}" for ln in _py_string_em_dashes(text))
            else:
                problems.extend(f"{rel}:{n}" for n, line in enumerate(text.splitlines(), 1)
                                if EM_DASH in line and EM_DASH_OK not in line)
        self.assertEqual(problems, [], "em-dashes in user-facing text (use a comma, colon or "
                         "parentheses; mark a deliberate non user-facing use with "
                         f"'# {EM_DASH_OK}'):\n" + "\n".join(problems))


def _py_string_em_dashes(source: str) -> list:
    """Line numbers of string literals (docstrings included) that use an em-dash in prose."""
    lines = source.splitlines()
    hits = []
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(source).readline))
    except (tokenize.TokenError, SyntaxError):
        return [n for n, line in enumerate(lines, 1)
                if PROSE_EM_DASH_RE.search(line) and EM_DASH_OK not in line]
    string_types = {tokenize.STRING}
    for name in ("FSTRING_MIDDLE", "TSTRING_MIDDLE"):
        if hasattr(tokenize, name):
            string_types.add(getattr(tokenize, name))
    for tok in toks:
        if tok.type in string_types and EM_DASH in tok.string:
            for k, part in enumerate(tok.string.splitlines()):
                n = tok.start[0] + k
                if PROSE_EM_DASH_RE.search(part) and EM_DASH_OK not in lines[n - 1]:
                    hits.append(n)
    return hits


if __name__ == "__main__":
    unittest.main()
