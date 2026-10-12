"""`hc demo`: the edit spec it writes plans cleanly, and (slow) the whole demo runs.

The fast test stands in for text-to-speech and the speech model: the demo script's words are
laid out with a tone burst each, pauses where the script marks them, and the spec the demo
writes must plan with no FAIL lint. The slow test runs the real command (it needs `say` or
espeak-ng, and downloads the tiny.en model on first use)."""

from __future__ import annotations

import argparse
import contextlib
import io
import os
import re
import shutil
import tempfile
import unittest
from pathlib import Path

import fixtures
from hookcutter import demo, env, plan


def script_words():
    """(text, s, e) for the demo script: 0.3 s a word, 0.6 s pauses at "|", 0.2 s after a
    sentence end."""
    out, t = [], 1.2
    for chunk in demo.SCRIPT.split("|"):
        for tok in chunk.split():
            out.append((tok.replace("thirty", "30"), t, t + 0.26))
            t += 0.3 + (0.2 if re.search(r"[.,]$", tok) else 0.0)
        t += 0.6
    return out


class TestDemoSpec(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="hc-demo-"))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_spec_plans_clean(self):
        words = script_words()
        src = fixtures.make_speech_fixture(self.tmp, words)
        P = fixtures.make_project(self.tmp, src, words)
        spec = demo.write_spec(P)
        clip = spec["clips"][demo.CLIP_ID]
        self.assertEqual([p["role"] for p in clip["pieces"]], ["cold_open", "setup", "close"])
        self.assertEqual(clip["pieces"][0]["expect"][0], "Nobody watches an")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = plan.cmd_plan(argparse.Namespace(project=str(P), clips=[], quiet=True))
        self.assertEqual(rc, 0, buf.getvalue())
        self.assertIn(" 0 FAIL ", buf.getvalue())

    def test_spec_refuses_a_missing_line(self):
        words = [w for w in script_words() if w[0].lower().strip(".,") not in ("nobody", "watches", "hour")]
        src = fixtures.make_speech_fixture(self.tmp, words)
        P = fixtures.make_project(self.tmp, src, words)
        with self.assertRaises(env.HCError) as cm:
            demo.write_spec(P)
        self.assertIn("cold_open", cm.exception.msg)

    def test_refuses_a_folder_that_is_not_a_demo(self):
        (self.tmp / "keep.txt").write_text("mine", encoding="utf-8")
        with self.assertRaises(env.HCError):
            demo._prepare(self.tmp, force=True)
        self.assertTrue((self.tmp / "keep.txt").is_file())


@fixtures.slow
class TestDemoRuns(unittest.TestCase):
    def test_demo_end_to_end(self):
        if not (shutil.which("say") or shutil.which("espeak-ng") or shutil.which("espeak")):
            self.skipTest("no text-to-speech voice on this machine")
        tmp = Path(tempfile.mkdtemp(prefix="hc-demo-run-"))
        try:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = demo.cmd_demo(argparse.Namespace(dir=str(tmp / "d"), force=False, open=False))
            self.assertEqual(rc, 0, buf.getvalue()[-3000:])
            P = tmp / "d" / "project"
            self.assertTrue(list((P / "export").rglob("*.mp4")))
            self.assertTrue(list((P / "export").rglob("* - Cover.jpg")))
            qa = env.read_json(P / "qa" / ("%s.json" % demo.CLIP_ID))
            self.assertNotEqual(qa.get("status"), "FAIL", qa)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
