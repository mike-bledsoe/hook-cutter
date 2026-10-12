"""``doctor``: check the environment, or set it up with ``--install``.

Runs under the system python too (stdlib only, Python 3.9 compatible), because it
bootstraps the venv. Anything that needs the venv's packages runs as a subprocess
of the venv python.
"""

from __future__ import annotations

import fnmatch
import json
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import __version__, env

PASS, WARN, FAIL, SKIP = "PASS", "WARN", "FAIL", "SKIPPED"

REQUIRED_FILTERS = ("concat", "loudnorm", "ebur128", "overlay", "trim", "atrim", "apad",
                    "aresample", "sendcmd", "crop", "scale", "boxblur", "blackdetect",
                    "freezedetect", "fps", "setsar", "eq", "setpts", "asetpts", "afade", "volume",
                    "pan", "astats")
OPTIONAL_FILTERS = (("zscale", "HDR (HLG/PQ) tone-mapping to SDR"),)
REQUIRED_ENCODERS = ("libx264", "aac", "pcm_s24le")
DECODER_GROUPS = (("H.264", ("h264",)), ("HEVC", ("hevc",)), ("VP9", ("vp9", "libvpx-vp9")),
                  ("AV1", ("libdav1d", "libaom-av1", "av1")))
MIN_FFMPEG = (6, 0)
PY_MIN, PY_MAX = (3, 12), (3, 13)   # numpy==2.5.3 (pinned) requires Python 3.12+
PREFERRED_PY = "3.12"
EXTRAS = ("mlx", "url", "whispercpp")
REQ_FILES = {"core": "requirements.txt", "mlx": "requirements-mlx.txt", "url": "requirements-url.txt"}
FW_ALIASES = {
    "large-v3-turbo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
    "turbo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",
    "large-v3": "Systran/faster-whisper-large-v3",
    "medium": "Systran/faster-whisper-medium",
    "small": "Systran/faster-whisper-small",
    "base": "Systran/faster-whisper-base",
}
FW_ALLOW = ("config.json", "preprocessor_config.json", "model.bin", "tokenizer.json", "vocabulary.*")
DEFAULT_FW_MODEL = "large-v3-turbo"
DEFAULT_MLX_MODEL = "mlx-community/whisper-large-v3-turbo"
MIN_FREE_BYTES = 5 * 10 ** 9
JS_RUNTIMES = ("deno", "node", "bun", "qjs")
IMPORT_PROBES = ("numpy", "PIL.Image", "cv2", "fontTools.ttLib", "ctranslate2")


class Check(object):
    def __init__(self, name: str, status: str, detail: str = ""):
        self.name, self.status, self.detail = name, status, detail

    def as_dict(self) -> Dict[str, str]:
        return {"name": self.name, "status": self.status, "detail": self.detail}


# ---------------------------------------------------------------- small helpers

def _out(cmd: Sequence[str], timeout: float = 60, env_vars: Optional[dict] = None) -> Tuple[int, str]:
    try:
        p = subprocess.run(list(cmd), stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, universal_newlines=True, timeout=timeout,
                           env=env_vars)
        return p.returncode, p.stdout or ""
    except (OSError, subprocess.TimeoutExpired) as exc:
        return 127, str(exc)


def _sysctl(name: str) -> Optional[str]:
    if sys.platform != "darwin":
        return None
    # /usr/sbin is often missing from an agent shell's PATH (PATH=/usr/bin:/bin): without the
    # absolute path an Apple Silicon Mac read as "not Apple Silicon" and the mlx extra was skipped
    exe = "/usr/sbin/sysctl" if os.path.exists("/usr/sbin/sysctl") else (shutil.which("sysctl") or "sysctl")
    rc, out = _out([exe, "-n", name], timeout=5)
    return out.strip() if rc == 0 else None


def is_apple_silicon() -> bool:
    return sys.platform == "darwin" and (_sysctl("hw.optional.arm64") == "1")


def rosetta_translated() -> bool:
    return _sysctl("sysctl.proc_translated") == "1"


def parse_requirements(path: Path) -> List[Tuple[str, str, str]]:
    """[(dist name, op, version)] from a requirements file (extras stripped)."""
    out = []
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        m = re.match(r"^([A-Za-z0-9_.\-]+)(\[[^\]]*\])?\s*(==|>=|~=|<=)?\s*([^\s;]*)", line)
        if m:
            out.append((m.group(1), m.group(3) or "", m.group(4) or ""))
    return out


def _vtuple(v: str) -> Tuple[int, ...]:
    parts = []
    for p in re.split(r"[.+-]", v or ""):
        if p.isdigit():
            parts.append(int(p))
        else:
            break
    return tuple(parts)


def _req_files(extras: Sequence[str]) -> List[Path]:
    files = [env.SCRIPTS_DIR / REQ_FILES["core"]]
    for x in extras:
        if x in REQ_FILES:
            files.append(env.SCRIPTS_DIR / REQ_FILES[x])
    return files


def _sub_env() -> Dict[str, str]:
    e = dict(os.environ)
    e["PYTHONNOUSERSITE"] = "1"
    e["PYTHONPATH"] = str(env.SCRIPTS_DIR)
    e["PYTHONUTF8"] = "1"
    e["PYTHONSAFEPATH"] = "1"     # `-m pip` / `-m venv` never import from the current folder
    e["PYTHONPYCACHEPREFIX"] = str(env.cache_dir() / "pyc")
    e["HF_HOME"] = str(env.hf_home())
    e.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    return e


_REPORT_SCRIPT = r"""
import importlib, json, platform, sys
try:
    from importlib import metadata
except ImportError:
    metadata = None
pk = {}
for name in sys.argv[1].split(","):
    if not name:
        continue
    try:
        pk[name] = metadata.version(name)
    except Exception:
        pk[name] = None
imp = {}
for mod in sys.argv[2].split(","):
    if not mod:
        continue
    try:
        importlib.import_module(mod)
        imp[mod] = None
    except Exception as exc:
        imp[mod] = "%s: %s" % (type(exc).__name__, exc)
cuda = None
try:
    import ctranslate2
    n = int(ctranslate2.get_cuda_device_count())
    if n > 0:
        import ctypes, os, glob
        win = sys.platform.startswith("win")
        libs = {"cuBLAS": ["cublas64_12.dll"] if win else ["libcublas.so.12"],
                "cuDNN": ["cudnn_ops64_9.dll", "cudnn64_9.dll"] if win else ["libcudnn_ops.so.9", "libcudnn.so.9"]}
        dirs = []
        for pkg in ("nvidia.cublas", "nvidia.cudnn"):
            try:
                mod = importlib.import_module(pkg)
                for base in list(getattr(mod, "__path__", [])):
                    dirs += [os.path.join(base, "lib"), os.path.join(base, "bin")]
            except Exception:
                pass
        missing = []
        for label, names in libs.items():
            ok = False
            for name in names:
                for cand in [name] + [os.path.join(d, name) for d in dirs]:
                    try:
                        ctypes.CDLL(cand)
                        ok = True
                        break
                    except OSError:
                        continue
                if ok:
                    break
            if not ok:
                missing.append(label)
        cuda = {"devices": n, "missing": missing}
except Exception:
    cuda = None
print(json.dumps({"python": list(sys.version_info[:3]), "machine": platform.machine(),
                  "executable": sys.executable, "packages": pk, "imports": imp, "cuda": cuda}))
"""


def venv_report(names: Sequence[str], imports: Sequence[str] = IMPORT_PROBES) -> Optional[Dict[str, Any]]:
    """Facts from inside the venv, None when there is no venv folder, or {"error", "broken"}
    when the folder exists but its interpreter is missing or does not run."""
    vpy = env.venv_python()
    if not vpy.exists():
        if env.venv_dir().exists() or vpy.is_symlink():
            what = ("%s points at a Python that no longer exists" % vpy) if vpy.is_symlink() else \
                ("%s has no interpreter" % env.venv_dir())
            return {"error": what, "broken": True}
        return None
    rc, out = _out([str(vpy), "-s", "-c", _REPORT_SCRIPT, ",".join(names), ",".join(imports)],
                   timeout=180, env_vars=_sub_env())
    if rc != 0:
        return {"error": out.strip()[-400:], "broken": True}
    try:
        return json.loads(out.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return {"error": out.strip()[-400:], "broken": True}


# ---------------------------------------------------------------- checks

def check_platform() -> List[Check]:
    out = []
    osname = {"darwin": "macOS", "win32": "Windows"}.get(sys.platform, platform.system() or sys.platform)
    ver = platform.mac_ver()[0] if sys.platform == "darwin" else platform.release()
    out.append(Check("platform", PASS, "%s %s %s" % (osname, ver, platform.machine())))
    if rosetta_translated():
        out.append(Check("rosetta", WARN, "this python runs under Rosetta (x86_64 on Apple Silicon): "
                                         "use a native arm64 python3"))
    py = sys.version_info
    status = PASS if py[:2] >= (3, 9) else FAIL
    # The launcher re-execs into the venv when it exists, so this is whichever runs doctor.
    out.append(Check("interpreter", status, "%d.%d.%d at %s%s" % (
        py[0], py[1], py[2], sys.executable, " (the venv)" if env.in_venv() else "")))
    return out


def check_data_dir() -> List[Check]:
    d = env.data_dir()
    try:
        d.mkdir(parents=True, exist_ok=True)
        probe_file = d / (".write-test-%d" % os.getpid())
        probe_file.write_text("ok")
        probe_file.unlink()
    except OSError as exc:
        return [Check("data dir", FAIL, "%s is not writable: %s" % (d, exc))]
    free = shutil.disk_usage(str(d)).free
    status = PASS if free >= MIN_FREE_BYTES else WARN
    detail = "%s (%s free)" % (d, env.human_bytes(free))
    if status == WARN:
        detail += ": models and renders need several GB"
    return [Check("data dir", status, detail)]


def hc_path() -> str:
    """The launcher path, quoted for a shell (skill folders can hold spaces or quotes)."""
    return shlex.quote(str(env.SCRIPTS_DIR / "hc"))


def check_venv(report: Optional[Dict[str, Any]]) -> List[Check]:
    vpy = env.venv_python()
    if report is None:
        return [Check("venv", FAIL, "missing at %s. Run: python3 %s doctor --install"
                      % (env.venv_dir(), hc_path()))]
    if "error" in report:
        return [Check("venv", FAIL, "broken: %s (its base Python moved or was removed?). Run: "
                                    "python3 %s doctor --install (it moves the broken venv aside "
                                    "and builds a new one)" % (report["error"], hc_path()))]
    py = tuple(report["python"])
    ok = PY_MIN <= py[:2] <= PY_MAX
    out = [Check("venv", PASS if ok else FAIL,
                 "Python %s (%s) at %s%s" % (".".join(str(x) for x in py), report.get("machine"),
                                             env.venv_dir(),
                                             "" if ok else ": needs Python %d.%d to %d.%d; delete it "
                                             "and run doctor --install" % (PY_MIN + PY_MAX)))]
    if is_apple_silicon() and report.get("machine") == "x86_64":
        out.append(Check("venv arch", WARN, "venv python is x86_64 under Rosetta: slower ASR and no "
                                           "mlx. Recreate it with an arm64 python"))
    out += check_cuda(report)
    return out


def check_cuda(report: Optional[Dict[str, Any]]) -> List[Check]:
    """A visible NVIDIA GPU without the CUDA libraries faster-whisper needs."""
    cuda = (report or {}).get("cuda") if report else None
    if not cuda or not cuda.get("devices"):
        return []
    missing = cuda.get("missing") or []
    if not missing:
        return [Check("gpu", PASS, "%d CUDA device(s); cuBLAS and cuDNN load" % cuda["devices"])]
    return [Check("gpu", WARN, "GPU visible but %s missing: pip install nvidia-cublas-cu12 "
                               "nvidia-cudnn-cu12 into the venv (%s -m pip install ...). Until then "
                               "transcribe runs on the CPU (or set HOOKCUTTER_DEVICE=cpu)"
                  % ("/".join(missing), env.venv_python()))]


def check_packages(report: Optional[Dict[str, Any]], extras: Sequence[str]) -> List[Check]:
    if not report or "error" in report:
        return [Check("python packages", SKIP, "no working venv")]
    out = []
    pk = report.get("packages") or {}
    core = parse_requirements(env.SCRIPTS_DIR / REQ_FILES["core"])
    bad, missing = [], []
    for name, op, ver in core:
        have = pk.get(name)
        if have is None:
            missing.append(name)
        elif op == "==" and have != ver:
            bad.append("%s %s (pinned %s)" % (name, have, ver))
    if missing:
        out.append(Check("python packages", FAIL, "missing: %s. Run: hc doctor --install" % ", ".join(missing)))
    elif bad:
        out.append(Check("python packages", WARN, "%s. Run: hc doctor --install" % "; ".join(bad)))
    else:
        out.append(Check("python packages", PASS, ", ".join("%s %s" % (n, pk.get(n)) for n, _, _ in core)))
    broken = {m: e for m, e in (report.get("imports") or {}).items() if e}
    if broken:
        out.append(Check("imports", FAIL, "; ".join("%s: %s" % (m, e) for m, e in broken.items())))
    elif not missing:
        out.append(Check("imports", PASS, ", ".join(IMPORT_PROBES)))
    for extra, dist in (("mlx", "mlx-whisper"), ("url", "yt-dlp")):
        reqs = parse_requirements(env.SCRIPTS_DIR / REQ_FILES[extra])
        want = reqs[0] if reqs else (dist, "", "")
        have = pk.get(dist)
        requested = extra in extras
        if extra == "mlx" and not is_apple_silicon():
            out.append(Check("extra: mlx", FAIL if requested else SKIP,
                             "mlx needs Apple Silicon; use the faster-whisper backend"))
            continue
        if have is None:
            out.append(Check("extra: %s" % extra, FAIL if requested else SKIP,
                             "%s not installed%s" % (dist, ". Run: hc doctor --install --with %s" % extra
                                                     if requested else " (optional: --with %s)" % extra)))
        elif want[1] == "==" and have != want[2]:
            out.append(Check("extra: %s" % extra, WARN, "%s %s (pinned %s)" % (dist, have, want[2])))
        elif want[1] == ">=" and _vtuple(have) < _vtuple(want[2]):
            out.append(Check("extra: %s" % extra, WARN, "%s %s is older than %s. Run: hc doctor "
                                                         "--install --with %s" % (dist, have, want[2], extra)))
        else:
            out.append(Check("extra: %s" % extra, PASS, "%s %s" % (dist, have)))
    return out


def parse_ffmpeg_version(text: str) -> Optional[Tuple[int, int]]:
    m = re.search(r"version\s+n?(\d+)\.(\d+)", text or "")
    if m:
        return int(m.group(1)), int(m.group(2))
    m = re.search(r"version\s+n?(\d+)(?:[^\d.]|$)", text or "")
    return (int(m.group(1)), 0) if m else None


def check_ffmpeg() -> List[Check]:
    out = []
    try:
        ff, fp = env.ffmpeg(), env.ffprobe()
    except env.HCError as exc:
        return [Check("ffmpeg", FAIL, ffmpeg_missing_hint(str(exc)))]
    rc, text = _out([ff, "-hide_banner", "-version"], timeout=20)
    first = (text.splitlines() or [""])[0]
    ver = parse_ffmpeg_version(first)
    if rc != 0:
        return [Check("ffmpeg", FAIL, "%s does not run: %s" % (ff, text.strip()[-200:]))]
    if ver is None:
        out.append(Check("ffmpeg", WARN, "%s (%s): version not parseable; relying on the capability "
                                         "checks below" % (first.strip(), ff)))
    elif ver < MIN_FFMPEG:
        out.append(Check("ffmpeg", FAIL, "%s is %d.%d; needs %d.%d or newer" % (ff, ver[0], ver[1],
                                                                                MIN_FFMPEG[0], MIN_FFMPEG[1])))
    else:
        out.append(Check("ffmpeg", PASS, "%d.%d at %s" % (ver[0], ver[1], ff)))
    rc, _ = _out([fp, "-version"], timeout=20)
    out.append(Check("ffprobe", PASS if rc == 0 else FAIL, fp))
    try:
        filters = env.ffmpeg_capabilities("filters")
        encoders = env.ffmpeg_capabilities("encoders")
        decoders = env.ffmpeg_capabilities("decoders")
    except env.HCError as exc:
        return out + [Check("ffmpeg capabilities", FAIL, str(exc))]
    miss = [f for f in REQUIRED_FILTERS if f not in filters]
    out.append(Check("ffmpeg filters", FAIL if miss else PASS,
                     ("missing: %s" % ", ".join(miss)) if miss else "%d required present" % len(REQUIRED_FILTERS)))
    for name, why in OPTIONAL_FILTERS:
        if name in filters:
            out.append(Check("filter %s" % name, PASS, why))
        else:
            out.append(Check("filter %s" % name, WARN,
                             "absent: HDR (HLG/PQ) sources cannot render (SDR is fine). For those, "
                             "install an ffmpeg built with libzimg and point HOOKCUTTER_FFMPEG and "
                             "HOOKCUTTER_FFPROBE at it"))
    miss = [e for e in REQUIRED_ENCODERS if e not in encoders]
    out.append(Check("ffmpeg encoders", FAIL if miss else PASS,
                     ("missing: %s" % ", ".join(miss)) if miss else ", ".join(REQUIRED_ENCODERS)))
    nodec = [label for label, names in DECODER_GROUPS if not any(n in decoders for n in names)]
    out.append(Check("ffmpeg decoders", WARN if nodec else PASS,
                     ("no %s decoder: such sources (often YouTube downloads) will not open" % ", ".join(nodec))
                     if nodec else ", ".join(label for label, _ in DECODER_GROUPS)))
    out.append(_encode_smoke(ff))
    return out


def ffmpeg_missing_hint(reason: str) -> str:
    """How to get an ffmpeg 6+ that works, for the platform doctor runs on."""
    reason = reason.split(". Run: hc doctor")[0].rstrip(".")
    if sys.platform == "darwin":
        how = ("`brew install ffmpeg` (Homebrew), or a static build (https://ffmpeg.org/download.html "
               "lists macOS builds)")
    elif sys.platform.startswith("win"):
        how = "`winget install Gyan.FFmpeg`, or a build from https://ffmpeg.org/download.html"
    else:
        how = ("`sudo apt install ffmpeg` on Ubuntu 24.04+ / Debian 13+ (older releases ship "
               "4.x/5.x); otherwise a static build from https://ffmpeg.org/download.html")
    return ("%s. Install ffmpeg %d or newer: %s; or point HOOKCUTTER_FFMPEG and HOOKCUTTER_FFPROBE "
            "at any ffmpeg/ffprobe binaries" % (reason, MIN_FFMPEG[0], how))


def _encode_smoke(ff: str) -> Check:
    with tempfile.TemporaryDirectory(prefix="hc-doctor-") as td:
        dst = os.path.join(td, "smoke.mp4")
        rc, text = _out([ff, "-hide_banner", "-nostdin", "-loglevel", "error", "-y",
                         "-f", "lavfi", "-i", "testsrc=size=320x240:rate=25:duration=0.4",
                         "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=0.4",
                         "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
                         "-c:a", "aac", "-shortest", "-movflags", "+faststart", dst], timeout=60)
        ok = rc == 0 and os.path.exists(dst) and os.path.getsize(dst) > 0
    return Check("encode smoke", PASS if ok else FAIL,
                 "libx264 + aac mp4 OK" if ok else "test encode failed: %s" % text.strip()[-300:])


def check_assets() -> List[Check]:
    mf = env.bundled_fonts_dir() / "manifest.json"
    if not mf.exists():
        return [Check("bundled assets", FAIL, "fonts/manifest.json missing")]
    try:
        manifest = json.loads(mf.read_text(encoding="utf-8"))
    except ValueError as exc:
        return [Check("bundled assets", FAIL, "fonts/manifest.json invalid: %s" % exc)]
    bad = []
    for rel, digest in sorted(manifest.items()):
        p = env.SKILL_ROOT / rel
        if not p.exists():
            bad.append("%s missing" % rel)
        elif env.sha256_file(p) != digest:
            bad.append("%s hash mismatch" % rel)
    if bad:
        return [Check("bundled assets", FAIL, "; ".join(bad) + " (reinstall the skill)")]
    return [Check("bundled assets", PASS, "%d fonts/models verified by sha256" % len(manifest))]


def _snapshot_has(cache: Path, repo: str, patterns: Sequence[str]) -> Optional[Path]:
    snaps = cache / ("models--" + repo.replace("/", "--")) / "snapshots"
    if not snaps.is_dir():
        return None
    for snap in sorted(snaps.iterdir()):
        names = [f.name for f in snap.iterdir()] if snap.is_dir() else []
        if all(any(fnmatch.fnmatch(n, pat) for n in names) for pat in patterns):
            return snap
    return None


def check_models() -> List[Check]:
    found = []
    fw = _snapshot_has(env.fw_models_dir(), FW_ALIASES[DEFAULT_FW_MODEL], ("model.bin", "config.json"))
    if fw:
        found.append("faster-whisper %s" % DEFAULT_FW_MODEL)
    mlx = _snapshot_has(env.hf_home() / "hub", DEFAULT_MLX_MODEL, ("config.json", "weights.*"))
    if mlx:
        found.append(DEFAULT_MLX_MODEL)
    if found:
        return [Check("asr model", PASS, ", ".join(found))]
    return [Check("asr model", WARN, "none downloaded yet: the first transcribe fetches about 1.6 GB "
                                     "(or run: hc doctor --model %s)" % DEFAULT_FW_MODEL)]


def check_url_runtime(report: Optional[Dict[str, Any]], extras: Sequence[str]) -> List[Check]:
    have = (report or {}).get("packages", {}).get("yt-dlp") if report else None
    if not have and "url" not in extras:
        return []
    rts = [r for r in JS_RUNTIMES if shutil.which(r)]
    if "deno" in rts:
        return [Check("js runtime", PASS, "deno (yt-dlp's default for YouTube)")]
    if rts:
        return [Check("js runtime", PASS, "%s (ingest passes --js-runtimes %s to yt-dlp)" % (rts[0], rts[0]))]
    return [Check("js runtime", WARN, "no deno/node found: YouTube downloads may fail or lose formats. "
                                      "Install deno (https://deno.com)")]


def check_whispercpp(extras: Sequence[str]) -> List[Check]:
    requested = "whispercpp" in extras
    cli = shutil.which("whisper-cli")
    model = os.environ.get("HOOKCUTTER_WHISPERCPP_MODEL")
    if not cli:
        if requested:
            return [Check("extra: whispercpp", FAIL, "whisper-cli not on PATH (install whisper.cpp, e.g. "
                                                    "brew install whisper-cpp)")]
        return []
    if model and Path(model).expanduser().exists():
        return [Check("extra: whispercpp", PASS, "%s, model %s" % (cli, model))]
    return [Check("extra: whispercpp", WARN if requested else SKIP,
                  "%s found; set HOOKCUTTER_WHISPERCPP_MODEL or pass --model to transcribe" % cli)]


def run_checks(extras: Sequence[str] = ()) -> List[Check]:
    checks = []  # type: List[Check]
    checks += check_platform()
    checks += check_data_dir()
    names = [n for f in _req_files(["mlx", "url"]) for n, _, _ in parse_requirements(f)]
    report = venv_report(names)
    checks += check_venv(report)
    checks += check_packages(report, extras)
    checks += check_ffmpeg()
    checks += check_assets()
    checks += check_models()
    checks += check_url_runtime(report, extras)
    checks += check_whispercpp(extras)
    checks.append(Check("HF_HOME", PASS, str(env.hf_home())))
    return checks


def print_table(checks: Sequence[Check], ready: bool) -> None:
    w = max(len(c.name) for c in checks) + 2
    print("hook-cutter doctor %s" % __version__)
    print("%-*s%-9s%s" % (w, "CHECK", "STATUS", "DETAIL"))
    for c in checks:
        print("%-*s%-9s%s" % (w, c.name, c.status, c.detail))
    nfail = sum(1 for c in checks if c.status == FAIL)
    print("READY" if ready else "NOT READY (%d blocking)" % nfail)
    if ready and not os.path.isdir(os.path.expanduser("~/hook-cutter-demo")):
        hc = str(env.SCRIPTS_DIR / "hc").replace(os.path.expanduser("~"), "~", 1)
        print("next: python3 %s demo   (one finished Short from a made-up recording, in about "
              "a minute)" % hc)


# ---------------------------------------------------------------- install

def _py_version(cmd: Sequence[str]) -> Optional[Tuple[int, int]]:
    rc, out = _out(list(cmd) + ["-c", "import sys;print('%d.%d' % sys.version_info[:2])"], timeout=30)
    m = re.search(r"(\d+)\.(\d+)", out) if rc == 0 else None
    return (int(m.group(1)), int(m.group(2))) if m else None


def find_base_python(explicit: Optional[str] = None) -> Tuple[List[str], Tuple[int, int]]:
    """An interpreter in the supported range to build the venv from."""
    cands = []  # type: List[List[str]]
    if explicit:
        cands.append([explicit])
    for minor in range(PY_MIN[1], PY_MAX[1] + 1):
        exe = shutil.which("python3.%d" % minor)
        if exe:
            cands.append([exe])
    if sys.platform.startswith("win") and shutil.which("py"):
        for minor in range(PY_MIN[1], PY_MAX[1] + 1):
            cands.append(["py", "-3.%d" % minor])
    cands.append([sys.executable])
    if shutil.which("python3"):
        cands.append([shutil.which("python3") or "python3"])
    for cmd in cands:
        ver = _py_version(cmd)
        if ver and PY_MIN <= ver <= PY_MAX:
            return cmd, ver
        if explicit and cmd == [explicit]:
            raise env.HCError("--python %s is %s; needs %d.%d to %d.%d"
                              % (explicit, "%d.%d" % ver if ver else "not runnable", PY_MIN[0], PY_MIN[1],
                                 PY_MAX[0], PY_MAX[1]), code=3)
    raise env.HCError(no_python_hint(), code=3)


def no_python_hint() -> str:
    """Where to get Python 3.12/3.13. uv first on Linux: Debian 12 and Ubuntu 22.04 ship 3.11/3.10
    and have no python3.12 packages."""
    uv = ("install uv (curl -LsSf https://astral.sh/uv/install.sh | sh), open a new shell so "
          "~/.local/bin is on PATH, then rerun doctor --install: uv fetches Python %s itself"
          % PREFERRED_PY)
    if sys.platform.startswith("linux"):
        how = "%s. Or install Python %s from your distribution (Ubuntu 24.04+ ships 3.12)" % (
            uv[0].upper() + uv[1:], PREFERRED_PY)
    elif sys.platform == "darwin":
        how = "`brew install python@%s` or python.org, or %s" % (PREFERRED_PY, uv)
    else:
        how = "python.org (or `winget install Python.Python.%s`), or %s" % (PREFERRED_PY, uv)
    return "no Python %d.%d to %d.%d found. %s" % (PY_MIN + PY_MAX + (how,))


def install_commands(extras: Sequence[str], *, python: Optional[str] = None,
                     uv: Optional[str] = None, venv_exists: bool = False) -> List[List[str]]:
    """The commands ``--install`` runs (venv creation unless it exists, then pip installs)."""
    vdir, vpy = env.venv_dir(), env.venv_python()
    reqs = []
    for f in _req_files([x for x in extras if x != "whispercpp"]):
        reqs += ["-r", str(f)]
    # `--with url` is also how yt-dlp is updated (its requirement is a floor, and sites change
    # under it): ask for the newest release, or an installed copy would never move
    url = "url" in list(extras or [])
    cmds = []
    if uv and not python:
        if not venv_exists:
            # --seed puts pip in the venv, so a later install without uv on PATH still works
            cmds.append([uv, "venv", "--python", PREFERRED_PY, "--seed", str(vdir)])
        cmds.append([uv, "pip", "install", "--no-build", "--python", str(vpy)] + reqs
                    + (["--upgrade-package", "yt-dlp"] if url else []))
        return cmds
    if not venv_exists:
        base, _ = find_base_python(python)
        cmds.append(base + ["-m", "venv", str(vdir)])
    # pip upgrades only what the requirement lines name ("only-if-needed"), and the core
    # lines are exact pins, so --upgrade moves yt-dlp (and its own matching parts) alone
    cmds.append([str(vpy), "-m", "pip", "install", "--disable-pip-version-check",
                 "--only-binary", ":all:"] + (["--upgrade"] if url else []) + reqs)
    return cmds


def _has_pip(vpy: Path) -> bool:
    rc, _ = _out([str(vpy), "-m", "pip", "--version"], timeout=60, env_vars=_sub_env())
    return rc == 0


def _ensure_pip(vpy: Path) -> None:
    """A uv-made venv has no pip unless seeded; add it before a plain pip install."""
    if _has_pip(vpy):
        return
    env.log("doctor", "the venv has no pip: adding it (python -m ensurepip --upgrade)")
    rc = subprocess.call([str(vpy), "-m", "ensurepip", "--upgrade"], stdin=subprocess.DEVNULL,
                         env=_sub_env())
    if rc != 0 or not _has_pip(vpy):
        raise env.HCError("the venv at %s has no pip and ensurepip failed (exit %d). Delete that "
                          "folder and rerun doctor --install" % (env.venv_dir(), rc), code=3)


NO_WHEEL_HINT = ("If the log above says a package has no matching distribution or wheel: a "
                 "pinned package has no prebuilt wheel for this OS (opencv-python-headless needs "
                 "macOS 13+ on Apple Silicon or 14+ on Intel; see README Requirements). Nothing "
                 "is compiled from source.")


def _move_aside(vdir: Path, why: str) -> None:
    aside = vdir.with_name("venv.%s-%d" % (why, int(time.time())))
    env.log("doctor", "moving the %s venv aside to %s" % (why, aside))
    os.replace(str(vdir), str(aside))


def install(extras: Sequence[str], python: Optional[str] = None) -> None:
    for d in (env.data_dir(), env.hf_home(), env.fw_models_dir(), env.user_presets_dir(),
              env.user_fonts_dir(), env.cache_dir()):
        d.mkdir(parents=True, exist_ok=True)
    extras = list(extras)
    if "mlx" in extras and not is_apple_silicon():
        env.log("doctor", "skipping the mlx extra: it needs Apple Silicon")
        extras.remove("mlx")
    vdir, vpy = env.venv_dir(), env.venv_python()
    exists = vpy.exists()
    ver = _py_version([str(vpy)]) if exists else None
    if vdir.exists() and (not exists or ver is None):
        _move_aside(vdir, "broken")          # folder without a working interpreter
        exists, ver = False, None
    if exists and python:
        want = find_base_python(python)[1]
        if ver != want:
            # --python asks for a different interpreter: rebuild instead of mixing versions
            env.log("doctor", "the venv is Python %d.%d; --python is %d.%d: rebuilding it"
                    % (ver + want))
            _move_aside(vdir, "py%d%d" % ver)
            exists, ver = False, None
    if exists and not (PY_MIN <= ver <= PY_MAX):
        raise env.HCError("the venv at %s uses Python %d.%d; needs %d.%d to %d.%d. Rerun with "
                          "doctor --install --python <python3.12> (or delete that folder)"
                          % ((vdir,) + ver + PY_MIN + PY_MAX), code=3)
    uv = shutil.which("uv")
    for cmd in install_commands(extras, python=python, uv=uv, venv_exists=exists):
        is_pip = cmd[1:3] == ["-m", "pip"]
        if is_pip:
            _ensure_pip(vpy)
        env.log("doctor", env.quote_cmd(cmd))
        rc = subprocess.call(cmd, stdin=subprocess.DEVNULL, env=_sub_env())
        if rc != 0:
            hint = ""
            if cmd[1:3] == ["-m", "venv"] and sys.platform.startswith("linux"):
                hint = " (on Debian/Ubuntu: apt install python3-venv, or install uv)"
            elif is_pip or "pip" in cmd[1:3]:
                hint = ". " + NO_WHEEL_HINT
            raise env.HCError("install step failed (exit %d): %s%s" % (rc, env.quote_cmd(cmd[:4]), hint),
                              code=3)


# ---------------------------------------------------------------- models

def is_mlx_model(name: str) -> bool:
    return name.startswith("mlx-community/") or "mlx" in name.lower()


def _dir_size(path: Path) -> int:
    total = 0
    for root, _dirs, files in os.walk(str(path), followlinks=True):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def _remote_size(repo: str, patterns: Optional[Sequence[str]]) -> Optional[int]:
    try:
        from huggingface_hub import HfApi
        info = HfApi().model_info(repo, files_metadata=True)
    except Exception:
        return None
    total = 0
    for s in info.siblings or []:
        if patterns and not any(fnmatch.fnmatch(s.rfilename, p) for p in patterns):
            continue
        total += int(getattr(s, "size", 0) or 0)
    return total or None


def model_download_main(name: str, backend: str = "auto") -> int:
    """Download an ASR model into the data dir (runs inside the venv).

    A short alias ("large-v3-turbo") follows ``backend``: "auto" is the backend
    `transcribe --backend auto` will use, so the pre-download is the model the first transcribe
    loads; "faster-whisper" or "mlx" fetch that backend's model explicitly."""
    env.setup_runtime_env()
    backend = backend or "auto"
    if "/" not in name and backend != "faster-whisper":
        try:
            from . import transcribe as _tr
            use = _tr.resolve_backend(backend) if backend == "auto" else backend
            if use == "mlx":
                name = _tr.resolve_model("mlx", name)
        except Exception:
            if backend == "mlx":
                name = DEFAULT_MLX_MODEL if name in ("large-v3-turbo", "turbo") else name
    if is_mlx_model(name):
        from huggingface_hub import snapshot_download
        try:
            from .transcribe import _hf_cached
            cached = _hf_cached(name)
        except Exception:
            cached = False
        if cached:
            print("checking the cached %s in %s" % (name, env.hf_home() / "hub"), flush=True)
        else:
            size = _remote_size(name, None)
            print("downloading %s (%s) into %s" % (name, env.human_bytes(size) if size else "size unknown",
                                                  env.hf_home() / "hub"), flush=True)
        path = Path(snapshot_download(name))
    else:
        try:
            from faster_whisper.utils import _MODELS as fw_models  # type: ignore
        except Exception:
            fw_models = {}
        repo = name if "/" in name else (fw_models.get(name) or FW_ALIASES.get(name))
        if not repo:
            raise env.HCError("unknown model %r (faster-whisper aliases: %s; or an HF repo id; or %s)"
                              % (name, ", ".join(sorted(FW_ALIASES)), DEFAULT_MLX_MODEL), code=1)
        size = _remote_size(repo, FW_ALLOW)
        print("downloading %s (%s) into %s" % (repo, env.human_bytes(size) if size else "size unknown",
                                              env.fw_models_dir()), flush=True)
        from faster_whisper.utils import download_model
        env.fw_models_dir().mkdir(parents=True, exist_ok=True)
        path = Path(download_model(name if "/" in name or name in fw_models else repo,
                                   cache_dir=str(env.fw_models_dir())))
    print("model ready: %s (%s on disk)" % (path, env.human_bytes(_dir_size(path))), flush=True)
    return 0


def download_model(name: str, backend: str = "auto") -> None:
    if env.in_venv():
        model_download_main(name, backend)
        return
    vpy = env.venv_python()
    if not vpy.exists():
        raise env.HCError("no venv yet: run doctor --install before --model", code=3)
    rc = subprocess.call([str(vpy), "-s", "-c",
                          "import sys; from hookcutter import doctor, env\n"
                          "try:\n    sys.exit(doctor.model_download_main(sys.argv[1], sys.argv[2]))\n"
                          "except env.HCError as e:\n    print('error: %s' % e, file=sys.stderr); sys.exit(e.code)",
                          name, backend or "auto"], stdin=subprocess.DEVNULL, env=_sub_env())
    if rc != 0:
        raise env.HCError("model download failed (exit %d)" % rc, code=3)


# ---------------------------------------------------------------- command

def _extras(value: Any) -> List[str]:
    if not value:
        return []
    items = value if isinstance(value, (list, tuple)) else str(value).split(",")
    out = []
    for x in items:
        x = str(x).strip().lower().replace("-", "")
        if not x:
            continue
        x = {"ytdlp": "url", "whisper.cpp": "whispercpp", "whispercpp": "whispercpp"}.get(x, x)
        if x not in EXTRAS:
            raise env.HCError("unknown extra %r (use %s)" % (x, ", ".join(EXTRAS)), code=1)
        if x not in out:
            out.append(x)
    return out


def cmd_doctor(args: Any) -> int:
    extras = _extras(getattr(args, "extras", None))
    if getattr(args, "install", False):
        install(extras, python=getattr(args, "python", None))
    if getattr(args, "model", None):
        download_model(args.model, getattr(args, "backend", None) or "auto")
    checks = run_checks(extras)
    ready = not any(c.status == FAIL for c in checks)
    if getattr(args, "json", False):
        print(json.dumps({"version": __version__, "ready": ready, "python": sys.executable,
                          "prefix": sys.prefix, "in_venv": env.in_venv(),
                          "data_dir": str(env.data_dir()),
                          "checks": [c.as_dict() for c in checks]}, indent=2))
    else:
        print_table(checks, ready)
    return 0 if ready else 3
