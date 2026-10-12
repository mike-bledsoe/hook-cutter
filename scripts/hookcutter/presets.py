"""Preset and safezone loading (contract section 9.4).

Search order: ``P/presets/``, ``<data>/presets/``, ``SK/presets/``. A preset may
``"extends": "<name>"`` another; the child deep-merges over the parent (dicts merge,
lists and scalars replace). A preset may extend its own name: the parent is then
looked up further down the search path, so a project ``clean.json`` can tweak the
shipped one.
"""

from __future__ import annotations

import copy
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from . import env

SAFEZONES_FILE = "safezones.json"
FORMATS = ("vertical", "landscape")
# Style names become part of render and export file names, so they are plain lowercase slugs
# and match their file name exactly (a case-insensitive disk must not turn "CLEAN" into clean).
NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")
RESERVED = ("safezones",)
LEAD_MS_RANGE = (0.0, 250.0)


def search_path(project: Any = None) -> List[Path]:
    dirs = []
    if project is not None:
        # A Project, or a path to P (note: pathlib.Path has its own .root, so test the type).
        root = project if isinstance(project, (str, os.PathLike)) else project.root
        dirs.append(Path(root) / "presets")
    dirs.append(env.user_presets_dir())
    dirs.append(env.bundled_presets_dir())
    return dirs


def deep_merge(base: Any, over: Any) -> Any:
    """New object: ``over`` merged into ``base`` (dicts recurse; everything else replaces)."""
    if isinstance(base, dict) and isinstance(over, dict):
        out = {k: copy.deepcopy(v) for k, v in base.items()}
        for k, v in over.items():
            out[k] = deep_merge(out[k], v) if k in out else copy.deepcopy(v)
        return out
    return copy.deepcopy(over)


def check_name(name: Any, what: str = "style") -> str:
    """``name`` if it is a valid preset name, else HCError(code 1)."""
    s = str(name) if name is not None else ""
    if s in RESERVED:
        raise env.HCError("%r is not a preset (presets/%s.json holds the safe-zone profiles); "
                          "use a style name such as clean, pop or minimal" % (s, s), code=1)
    if not NAME_RE.match(s):
        low = s.lower()
        hint = (" (did you mean %r?)" % low) if low != s and NAME_RE.match(low) else ""
        raise env.HCError("invalid %s name %r%s: preset names are lowercase letters, digits and "
                          "hyphens, matching the file presets/<name>.json exactly"
                          % (what, s, hint), code=1)
    return s


def _exact_file(d: Path, fname: str) -> bool:
    """True when ``d`` holds a file named exactly ``fname`` (case-sensitive on every disk)."""
    try:
        return fname in os.listdir(str(d)) and (d / fname).is_file()
    except OSError:
        return False


def _candidates(name: str, project: Any) -> List[Path]:
    """Every file that could define preset ``name``, in search order."""
    fname = name if name.endswith(".json") else name + ".json"
    return [d / fname for d in search_path(project) if _exact_file(d, fname)]


def find(name: str, project: Any = None) -> Path:
    check_name(name[:-5] if str(name).endswith(".json") else name)
    cands = _candidates(name, project)
    if not cands:
        raise env.HCError("preset %r not found (searched %s). Available: %s"
                          % (name, ", ".join(str(d) for d in search_path(project)),
                             ", ".join(available(project)) or "none"), code=1)
    return cands[0]


def available(project: Any = None) -> List[str]:
    names = []
    for d in search_path(project):
        if d.is_dir():
            for f in sorted(d.glob("*.json")):
                if f.stem in RESERVED or not NAME_RE.match(f.stem):
                    continue
                if f.stem not in names:
                    names.append(f.stem)
    return names


def _read(path: Path) -> Dict[str, Any]:
    data = env.read_json(path)
    if not isinstance(data, dict):
        raise env.HCError("preset %s must be a JSON object" % path, code=1)
    return data


def _parent_candidates(path: Path, parent: str, project: Any) -> List[Path]:
    cands = _candidates(parent, project)
    if Path(parent).stem == path.stem:
        # Same name: take the next definition further down the search path.
        resolved = [c.resolve() for c in cands]
        here = path.resolve()
        cands = cands[resolved.index(here) + 1:] if here in resolved else []
    return cands


def _resolve(path: Path, project: Any, chain: List[str]) -> Tuple[Dict[str, Any], List[str]]:
    key = str(path.resolve())
    if key in chain:
        raise env.HCError("preset extends cycle: %s"
                          % " -> ".join(Path(c).stem for c in chain + [key]), code=1)
    data = _read(path)
    parent = data.get("extends")
    if not parent:
        return data, [path.stem]
    check_name(parent, "extends")
    cands = _parent_candidates(path, str(parent), project)
    if not cands:
        raise env.HCError("preset %s extends %r, which was not found" % (path, parent), code=1)
    base, names = _resolve(cands[0], project, chain + [key])
    return deep_merge(base, data), [path.stem] + names


def load(name: str, project: Any = None) -> Dict[str, Any]:
    """Fully resolved preset dict. ``name``: a preset name, i.e. the file ``<name>.json`` in
    one of the search_path folders (lowercase letters, digits and hyphens, matched exactly;
    a path is not accepted because the name becomes part of render and export file names).

    Adds ``_chain`` (names from this preset up to its root) for diagnostics. Raises
    HCError(code 1) when a value is out of range (see ``validate``).
    """
    path = find(name, project)
    data, names = _resolve(path, project, [])
    own = _read(path)  # name/label are never inherited from a parent
    data["name"] = own.get("name") or path.stem
    data["label"] = own.get("label") or str(data["name"]).replace("-", " ").replace("_", " ").title()
    data["_chain"] = names
    validate(data, path.stem)
    return data


# Numeric preset values the renderer converts without a guard. NUMBERS must be a number when
# the key is present (null is not "inherit": delete the key for that); NUMBERS_OR_NULL also
# take null, which the renderer reads as "use the built-in default".
NUMBERS = {"caption": ("size",),
           "caption.layout": ("pause_break", "comma_min_words"),
           "hook_card": ("min_size", "max_size", "fade_out_ms")}
NUMBERS_OR_NULL = {"caption.layout": ("max_words", "max_chars", "max_lines", "max_width",
                                      "line_gap", "min_page", "y"),
                   "hook_card": ("max_lines", "max_width", "min_ratio"),
                   "motion": ("punch_in", "min_gap", "max_upscale"),
                   "letterbox": ("y_center", "blur", "dim")}
OBJECTS = ("caption", "hook_card", "caption.layout", "caption.anim")    # an object when present
OBJECTS_OR_NULL = ("motion", "letterbox")
PADS = ("caption.backing", "hook_card.backing")                         # pad: [x, y] or null


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v))


def _at(block: Any, path: str) -> Tuple[bool, Any]:
    """(present, value) of a dotted key path inside ``block`` (absent when a parent is not an
    object: that parent is reported on its own)."""
    cur = block
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return False, None
        cur = cur[key]
    return True, cur


def _check_types(block: Dict[str, Any], pre: str, name: str) -> None:
    def bad(where: str, what: str, got: Any, hint: str = "") -> None:
        raise env.HCError("preset %s: %s%s must be %s (got %r)%s"
                          % (name, pre, where, what, got, hint), code=1)

    inherit = "; null does not mean inherit: delete the key to inherit it"
    for path in OBJECTS:
        there, v = _at(block, path)
        if there and not isinstance(v, dict):
            bad(path, "an object", v, inherit)
    for path in OBJECTS_OR_NULL:
        there, v = _at(block, path)
        if there and v is not None and not isinstance(v, dict):
            bad(path, "an object or null", v)
    for table, nullable in ((NUMBERS, False), (NUMBERS_OR_NULL, True)):
        for path, keys in table.items():
            there, obj = _at(block, path)
            if not there or not isinstance(obj, dict):
                continue
            for key in keys:
                if key not in obj or (nullable and obj[key] is None):
                    continue
                if not _is_number(obj[key]):
                    bad("%s.%s" % (path, key), "a number", obj[key],
                        inherit if obj[key] is None else "")
    for path in PADS:
        there, obj = _at(block, path)
        pad = obj.get("pad") if there and isinstance(obj, dict) else None
        if pad is not None and not (isinstance(pad, (list, tuple)) and len(pad) == 2
                                    and all(_is_number(x) for x in pad)):
            bad(path + ".pad", "two numbers [x, y]", pad)


def validate(preset: Dict[str, Any], name: str = "preset") -> None:
    """Type and value checks a resolved preset must pass (HCError code 1), for the base preset
    and for its ``landscape`` block: the values the renderer converts without a guard, so a
    wrong type or an explicit null stops the preset loading (an L01 FAIL at plan time) and
    never reaches a render. Nothing is changed in ``preset`` (its content is hashed into the
    render key)."""
    land = preset.get("landscape")
    if land is not None and not isinstance(land, dict):
        raise env.HCError("preset %s: landscape must be an object or null (got %r)" % (name, land),
                          code=1)
    _check_types(preset, "", name)
    if isinstance(land, dict):
        _check_types(land, "landscape.", name)
    blocks = [("caption", preset.get("caption"))]
    if isinstance(land, dict):
        blocks.append(("landscape.caption", land.get("caption")))
    lo, hi = LEAD_MS_RANGE
    for where, cap in blocks:
        if not isinstance(cap, dict) or "lead_ms" not in cap:
            continue
        v = cap["lead_ms"]
        ok = isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v)) \
            and lo <= float(v) <= hi
        if not ok:
            raise env.HCError("preset %s: %s.lead_ms must be a number from %d to %d ms (got %r); "
                              "captions lead the audio by this much" % (name, where, lo, hi, v),
                              code=1)


def zone_name(eff: Any, preset: Any, fmt: str = "vertical") -> str:
    """The safe-zone profile of one clip x style x format, the one rule the caption layout,
    the render key and QA share: clips.json ``defaults.safezone`` when the file set it
    (``eff["safezone_set"]``), else the preset's own ``safezone``, else the default in
    ``eff``, else "organic". A landscape render maps organic and ads to the "landscape"
    profile. ``eff`` is the clip's ``_eff`` (or None); ``preset`` the preset or resolved style
    for ``fmt``."""
    eff = eff if isinstance(eff, dict) else {}
    own = preset.get("safezone") if isinstance(preset, dict) else None
    named = eff.get("safezone")
    named = named if isinstance(named, str) and named else None
    if eff.get("safezone_set") and named:
        name = named
    elif isinstance(own, str) and own:
        name = own
    else:
        name = named or "organic"
    if fmt == "landscape" and name in ("organic", "ads"):
        name = "landscape"
    return name


def for_format(preset: Dict[str, Any], fmt: str) -> Dict[str, Any]:
    """The preset as used for ``fmt``: for landscape, ``preset["landscape"]`` deep-merges over
    the base and the safezone defaults to the "landscape" profile (unless the override sets one)."""
    if fmt not in FORMATS:
        raise env.HCError("unknown format %r (use %s)" % (fmt, ", ".join(FORMATS)), code=1)
    base = {k: v for k, v in preset.items() if k != "landscape"}
    if fmt != "landscape":
        return copy.deepcopy(base)
    over = preset.get("landscape") or {}
    out = deep_merge(base, over)
    if "safezone" not in over:
        out["safezone"] = "landscape"
    out["_format"] = "landscape"
    return out


def safezones(project: Any = None) -> Dict[str, Dict[str, Any]]:
    """All safezone profiles; earlier search-path entries override later ones by key."""
    merged = {}  # type: Dict[str, Dict[str, Any]]
    for d in reversed(search_path(project)):
        f = d / SAFEZONES_FILE
        if f.is_file():
            data = env.read_json(f)
            if not isinstance(data, dict):
                raise env.HCError("%s must be a JSON object" % f, code=1)
            for k, v in data.items():
                merged[k] = deep_merge(merged.get(k, {}), v) if isinstance(v, dict) else v
    return merged


def safezone(name: Any = "organic", project: Any = None) -> Dict[str, int]:
    """Safezone margins {top, bottom, left, right} in output px. ``name`` may be a dict."""
    if isinstance(name, dict):
        zone = name
    else:
        zones = safezones(project)
        zone = zones.get(str(name))
        if zone is None:
            raise env.HCError("unknown safezone %r (available: %s)"
                              % (name, ", ".join(sorted(zones)) or "none"), code=1)
    if not isinstance(zone, dict):
        raise env.HCError("safezone %r must be an object with top, bottom, left and right "
                          "(got %r)" % (name, zone), code=1)
    out = {}
    for k in ("top", "bottom", "left", "right"):
        try:
            out[k] = int(zone.get(k, 0))
        except (TypeError, ValueError):
            raise env.HCError("safezone %r: %s must be an integer" % (name, k), code=1)
    return out


def safe_rect(zone: Dict[str, int], width: int, height: int) -> List[int]:
    """[x, y, w, h] of the safe area for an output frame of width x height."""
    return [zone["left"], zone["top"], width - zone["left"] - zone["right"],
            height - zone["top"] - zone["bottom"]]
