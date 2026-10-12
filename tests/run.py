#!/usr/bin/env python3
"""Run the hook-cutter test suite (stdlib unittest) under the skill venv.

    python3 tests/run.py [--slow] [-v] [--failfast] [PATTERN]

PATTERN selects test files by name (``core``, ``test_core``, ``test_c*``) and,
when no file matches, test names (unittest ``-k`` style). ``--slow`` sets
HC_SLOW=1, which enables ASR tests.
"""

import argparse
import fnmatch
import os
import shlex
import sys
import time
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
SK = os.path.dirname(HERE)
SCRIPTS = os.path.join(SK, "scripts")
for p in (HERE, SCRIPTS):
    if p not in sys.path:
        sys.path.insert(0, p)

sys.dont_write_bytecode = True   # until the cache prefix is known: keep the skill folder clean
from hookcutter import env  # noqa: E402

PYC_DIR = str(env.cache_dir() / "pyc")
sys.pycache_prefix = PYC_DIR      # bytecode for the tests and modules goes to the data dir
sys.dont_write_bytecode = False


def _reexec_in_venv():
    if env.in_venv():
        return
    vpy = env.venv_python()
    if not vpy.exists():
        sys.stderr.write("hook-cutter venv missing. Run: python3 %s doctor --install\n"
                         % shlex.quote(os.path.join(SCRIPTS, "hc")))
        sys.exit(3)
    e = dict(os.environ)
    e["PYTHONNOUSERSITE"] = "1"
    e["PYTHONPATH"] = SCRIPTS
    e["PYTHONUTF8"] = "1"
    e["PYTHONPYCACHEPREFIX"] = PYC_DIR
    e["HF_HOME"] = str(env.hf_home())
    cmd = [str(vpy), "-s", os.path.abspath(__file__)] + sys.argv[1:]
    if sys.platform.startswith("win"):
        import subprocess
        sys.exit(subprocess.call(cmd, env=e))
    os.execve(str(vpy), cmd, e)


def _select(pattern):
    files = sorted(f[:-3] for f in os.listdir(HERE) if f.startswith("test_") and f.endswith(".py"))
    if not pattern:
        return files, None
    pats = [pattern, "test_" + pattern, pattern + ".py"]
    chosen = [f for f in files if any(fnmatch.fnmatch(f, p) or fnmatch.fnmatch(f + ".py", p) for p in pats)]
    if not chosen:
        chosen = [f for f in files if pattern in f]
    if chosen:
        return chosen, None
    return files, ["*%s*" % pattern]


def main():
    ap = argparse.ArgumentParser(description="hook-cutter tests")
    ap.add_argument("--slow", action="store_true", help="include ASR tests (HC_SLOW=1)")
    ap.add_argument("-v", "--verbose", action="store_true")
    ap.add_argument("--failfast", action="store_true")
    ap.add_argument("pattern", nargs="?", default=None)
    args = ap.parse_args()
    _reexec_in_venv()
    if args.slow:
        os.environ["HC_SLOW"] = "1"
    env.setup_runtime_env()
    names, name_patterns = _select(args.pattern)
    loader = unittest.TestLoader()
    if name_patterns:
        loader.testNamePatterns = name_patterns
    suite = unittest.TestSuite()
    for name in names:
        suite.addTests(loader.loadTestsFromName(name))
    t0 = time.time()
    runner = unittest.TextTestRunner(verbosity=2 if args.verbose else 1, failfast=args.failfast)
    result = runner.run(suite)
    print("hook-cutter tests: %d run, %d failures, %d errors, %d skipped in %.1fs (%s)" % (
        result.testsRun, len(result.failures), len(result.errors), len(result.skipped),
        time.time() - t0, ", ".join(names) or "none"))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
