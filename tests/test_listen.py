"""hc listen: transcribe one short stretch of the source, or of a clip's vertical render, on its
own. The slice is cut into a temporary folder outside P and transcribed with the transcript's
own ASR settings; the words come back on the chosen clock (source or output seconds), nothing
is written into P, and an edge that falls inside voiced audio is reported with the nearest quiet
edges. Canned ASR: a stand-in backend hears the fixture's tone bursts, so no model is loaded."""

from __future__ import annotations

import contextlib
import io
import json
import os
import re
import shutil
import stat
import sys
import tempfile
import unittest
import wave
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
SK = HERE.parent
sys.path.insert(0, str(SK / "scripts"))
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

import fixtures as F  # noqa: E402
from hookcutter import cli, env, transcribe as T  # noqa: E402
from hookcutter.audio import EnergyTrack  # noqa: E402

try:
    from hookcutter import listen as L  # noqa: E402
except ImportError:  # the command is missing: the tests below fail, the module still loads
    L = None

TMP = None
# The source: one tone burst per word on a -60 dB floor.
WORDS = [("Hello", 1.0, 1.6), ("there", 2.4, 3.0), ("again.", 4.0, 4.5)]
# The render has its one burst where the source has silence, so a slice cut from the wrong file
# shows up in the times.
RENDER_WORDS = [("Short", 0.5, 1.0)]
RENDER_NAME = "c01-test__vertical__clean.mp4"


def setUpModule():
    global TMP
    TMP = Path(tempfile.mkdtemp(prefix="hc-listen-test-"))


def tearDownModule():
    if TMP and TMP.exists():
        for dirpath, dirnames, _ in os.walk(str(TMP)):
            for name in dirnames:
                os.chmod(os.path.join(dirpath, name), 0o755)
        shutil.rmtree(str(TMP), ignore_errors=True)


class BurstRunner(object):
    """Stands in for an ASR backend: one word per tone burst in the wav it is handed, on that
    wav's own clock, so the times it returns show where the slice was cut."""

    def __init__(self, fail=None):
        self.calls = []
        self.fail = fail

    def __call__(self, wav, model, language, prompt, progress=True):
        with wave.open(str(wav), "rb") as w:
            sr = w.getframerate()
            x = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float64) / 32768.0
        self.calls.append({"wav": str(wav), "model": model, "language": language,
                           "prompt": prompt, "dur": round(len(x) / float(sr), 3)})
        if self.fail:
            raise env.HCError(self.fail, code=3)
        hop = int(sr * 0.01)
        n = len(x) // hop
        rms = np.sqrt(np.mean(x[:n * hop].reshape(n, hop) ** 2, axis=1)) if n else np.zeros(0)
        loud = rms > 0.02
        words, k = [], 0
        while k < n:
            if loud[k]:
                j = k
                while j < n and loud[j]:
                    j += 1
                words.append([" burst", round(k * 0.01, 3), round(j * 0.01, 3), 0.9])
                k = j
            else:
                k += 1
        return {"words": words, "language": language or "en",
                "options": T.runner_options("faster-whisper", model, prompt)}


@contextlib.contextmanager
def fake_backend(name="faster-whisper", fail=None):
    runner = BurstRunner(fail)
    old = T.RUNNERS[name]
    T.RUNNERS[name] = runner
    try:
        yield runner
    finally:
        T.RUNNERS[name] = old


def hc(*argv):
    """cli.main with captured output; a usage error's SystemExit becomes its exit code."""
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            rc = cli.main([str(a) for a in argv])
        except SystemExit as exc:
            rc = exc.code
    return rc, out.getvalue(), err.getvalue()


def tree(root):
    """Every path under root with its kind, size and mtime: a write anywhere in it changes this."""
    root = Path(root)
    out = {".": os.lstat(str(root)).st_mtime_ns}
    for dirpath, dirnames, filenames in os.walk(str(root)):
        for name in dirnames + filenames:
            p = Path(dirpath) / name
            st = os.lstat(str(p))
            out[p.relative_to(root).as_posix()] = (stat.S_IFMT(st.st_mode), st.st_size,
                                                   st.st_mtime_ns)
    return out


def track(segments=(), dur=10.0, floor=-60.0, hop=0.01):
    """EnergyTrack at `floor` dB with [(s, e, dB)] segments (t0 = 0)."""
    db = np.full(int(round(dur / hop)), floor, dtype=np.float64)
    for s, e, level in segments:
        db[int(round(s / hop)):int(round(e / hop))] = level
    return EnergyTrack(db, hop=hop, t0=0.0)


def make_listen_project(name):
    """A project whose transcript records real ASR settings, plus one clip's vertical render."""
    media = F.make_speech_fixture(TMP / "media", WORDS, dur=6.0)
    P = F.make_project(TMP / name, media, WORDS)
    doc = env.read_json(P / "transcript.json")
    doc.update(backend="faster-whisper", model="large-v3-turbo", language="en", glossary=["Acme"])
    env.write_json(P / "transcript.json", doc)
    render = F.make_speech_fixture(TMP / "render-media", RENDER_WORDS, dur=3.0)
    (P / "renders").mkdir(exist_ok=True)
    shutil.copy2(str(render), str(P / "renders" / RENDER_NAME))
    return P


class ListenCommandTest(unittest.TestCase):
    """The command end to end on a fixture project, through the real parser."""

    @classmethod
    def setUpClass(cls):
        cls.P = make_listen_project("listen")

    def listen(self, *argv, P=None):
        """Run hc listen with the temp folder pointed at an empty folder of its own; returns
        (rc, stdout, stderr, what was left in that temp folder)."""
        scratch = Path(tempfile.mkdtemp(dir=str(TMP), prefix="tmpdir-"))
        with mock.patch.object(tempfile, "tempdir", str(scratch)):
            rc, out, err = hc("listen", P or self.P, *argv)
        return rc, out, err, sorted(p.name for p in scratch.iterdir())

    def test_source_slice_writes_nothing_into_P(self):
        before = tree(self.P)
        with fake_backend() as runner:
            rc, out, err, left = self.listen("0.8", "3.2")
        self.assertEqual(rc, 0, err)
        self.assertEqual(tree(self.P), before, "hc listen wrote into P")
        self.assertFalse((self.P / "transcripts").exists())
        self.assertFalse((self.P / "cache").exists())
        self.assertEqual(left, [], "hc listen left temporary files behind")
        # the transcript's own ASR settings, so the slice is comparable with the transcript
        self.assertTrue(runner.calls, "the backend was never called")
        self.assertEqual(runner.calls[0]["model"], "large-v3-turbo")
        self.assertEqual(runner.calls[0]["language"], "en")
        self.assertEqual(runner.calls[0]["prompt"], "Acme.")
        self.assertAlmostEqual(runner.calls[0]["dur"], 2.4, delta=0.02)
        self.assertIn("s_src", out)
        self.assertNotIn("warning", err)

    def test_words_on_the_source_clock(self):
        with fake_backend():
            rc, out, err, left = self.listen("0.8", "3.2", "--json")
        self.assertEqual(rc, 0, err)
        self.assertEqual(left, [])
        data = json.loads(out)
        self.assertEqual((data["clock"], data["media"]), ("src", "source.mp4"))
        self.assertEqual((data["start"], data["end"]), (0.8, 3.2))
        self.assertEqual(data["backend"], "faster-whisper")
        got = [(w["s"], w["e"]) for w in data["words"]]
        self.assertEqual(len(got), 2, data["words"])
        for (s, e), (_, ts, te) in zip(got, WORDS[:2]):
            self.assertAlmostEqual(s, ts, delta=0.03)
            self.assertAlmostEqual(e, te, delta=0.03)
        self.assertEqual(data["warnings"], [])
        self.assertFalse(data["edges"]["start"]["voiced"])
        self.assertFalse(data["edges"]["end"]["voiced"])

    def test_edges_inside_voiced_audio_warn_with_quiet_edges(self):
        with fake_backend():
            rc, out, err, left = self.listen("1.3", "2.7", "--json")
        self.assertEqual(rc, 0, err)
        data = json.loads(out)
        st, en = data["edges"]["start"], data["edges"]["end"]
        self.assertTrue(st["voiced"])
        self.assertTrue(en["voiced"])
        # 0.2 s into the pause before the burst at 1.0, and after the one ending at 3.0
        self.assertAlmostEqual(st["quiet"], 0.8, delta=0.03)
        self.assertAlmostEqual(en["quiet"], 3.2, delta=0.03)
        self.assertEqual(len(data["warnings"]), 2, data["warnings"])
        with fake_backend():
            rc, out, err, left = self.listen("1.3", "2.7")
        self.assertEqual(rc, 0, err)
        self.assertRegex(err, r"START 1\.30 is inside voiced audio")
        self.assertRegex(err, r"END 2\.70 is inside voiced audio")
        self.assertRegex(err, r"hc listen \S+ 0\.80 3\.20")
        self.assertEqual(left, [])

    def test_render_slice_on_the_output_clock(self):
        before = tree(self.P)
        with fake_backend():
            rc, out, err, left = self.listen("0.2", "1.4", "--render", "c01-test", "--json")
        self.assertEqual(rc, 0, err)
        data = json.loads(out)
        self.assertEqual((data["clock"], data["media"]), ("out", "renders/" + RENDER_NAME))
        self.assertEqual(len(data["words"]), 1, data["words"])
        self.assertAlmostEqual(data["words"][0]["s"], 0.5, delta=0.03)
        self.assertAlmostEqual(data["words"][0]["e"], 1.0, delta=0.03)
        self.assertEqual(tree(self.P), before, "hc listen --render wrote into P")
        self.assertEqual(left, [])
        with fake_backend():
            rc, out, err, left = self.listen("0.2", "1.4", "--render", "c01-test")
        self.assertEqual(rc, 0, err)
        self.assertIn("s_out", out)

    def test_read_only_project(self):
        P = make_listen_project("listen-ro")
        dirs = [P] + [Path(d) / n for d, ns, _ in os.walk(str(P)) for n in ns]
        try:
            for d in dirs:
                os.chmod(str(d), 0o555)
            if os.access(str(P), os.W_OK):
                self.skipTest("this user can write into a read-only folder")
            with fake_backend():
                rc, out, err, left = self.listen("0.8", "3.2", P=P)
                self.assertEqual(rc, 0, err)
                rc, out, err, left = self.listen("0.2", "1.4", "--render", "c01-test", P=P)
                self.assertEqual(rc, 0, err)
        finally:
            for d in dirs:
                os.chmod(str(d), 0o755)

    def test_backend_error_leaves_nothing(self):
        before = tree(self.P)
        with fake_backend(fail="the model could not load"):
            rc, out, err, left = self.listen("0.8", "3.2")
        self.assertEqual(rc, 3, err)
        self.assertIn("the model could not load", err)
        self.assertEqual(tree(self.P), before)
        self.assertEqual(left, [])

    def test_usage_errors(self):
        before = tree(self.P)
        cases = [
            (("3", "2"), "END must be after START"),
            (("0", "200"), "at most"),
            (("7", "8"), "long"),
            (("abc", "2"), "bad time"),
            (("1", "2", "--render", "Bad_ID"), "invalid clip id"),
            (("1", "2", "--render", "c09-none"), "hc render"),
            (("1", "2", "--backend", "nope"), "invalid choice"),
        ]
        with fake_backend() as runner:
            for argv, needle in cases:
                rc, out, err, left = self.listen(*argv)
                self.assertEqual(rc, 1, (argv, err))
                self.assertIn(needle, err, argv)
                self.assertEqual(left, [], argv)
        self.assertEqual(runner.calls, [], "a usage error must not run the recogniser")
        self.assertEqual(tree(self.P), before)

    def test_short_slice_warns(self):
        with fake_backend():
            rc, out, err, left = self.listen("0.9", "1.5")
        self.assertEqual(rc, 0, err)
        self.assertIn("shorter than about a second", err)


class ListenArgsTest(unittest.TestCase):

    def parse(self, argv):
        try:
            return cli.build_parser().parse_args(argv)
        except SystemExit:
            self.fail("hc does not take: %s" % " ".join(argv))

    def test_parser_and_command_table(self):
        a = self.parse(["listen", "P", "1:02", "1:05.5", "--render", "c01", "--backend",
                        "faster-whisper", "--model", "tiny", "--json"])
        self.assertEqual((a.project, a.start, a.end, a.render, a.backend, a.model, a.json),
                         ("P", 62.0, 65.5, "c01", "faster-whisper", "tiny", True))
        a = self.parse(["listen", "P", "1", "2"])
        self.assertEqual((a.render, a.backend, a.model, a.json), (None, None, None, False))
        self.assertEqual(cli.COMMANDS.get("listen"), ("listen", "cmd_listen"))
        self.assertIn("listen", cli.SOURCE_BOUND)

    def test_source_drift_stops_source_slices_only(self):
        media = F.make_speech_fixture(TMP / "media", WORDS, dur=6.0)
        P = F.make_project(TMP / "listen-drift", media, WORDS)
        meta = env.read_json(P / "source.json")
        meta["fingerprint"] = "0" * 64
        env.write_json(P / "source.json", meta)
        with self.assertRaises(env.HCError) as cm:
            cli._require_source_current(self.parse(["listen", str(P), "1", "2"]))
        self.assertEqual(cm.exception.code, 1)
        cli._require_source_current(self.parse(["listen", str(P), "1", "2", "--render", "c01"]))


class ListenPartsTest(unittest.TestCase):

    def test_slice_sample_zero_is_start_with_delayed_audio(self):
        """The audio stream starts 0.2 s into the file: the slice is padded so that its
        sample 0 is START on the file's clock, the clock the transcript uses."""
        self.assertIsNotNone(L, "hookcutter/listen.py is missing")
        flashes = [1.0, 2.5, 4.0]
        media = F.make_av_fixture(TMP / "av", fps="25", dur=6, flashes=flashes, audio_delay=0.2)
        times = F.fixture_meta(media)["flash_times"]
        for a, b in ((0.0, 3.0), (2.0, 5.0)):
            out = TMP / ("slice-%g.wav" % a)
            L.cut_audio(media, a, b, out)
            onsets = F.click_onsets(np.asarray(T.load_wav(out)), sr=16000)
            want = [t for t in times if a < t < b]
            self.assertEqual(len(onsets), len(want), (a, onsets))
            for onset, t in zip(onsets, want):
                self.assertLess(abs(onset + a - t), 0.003, (a, onset, t))
            self.assertAlmostEqual(T.wav_duration(out), b - a, delta=0.01)

    def test_edge_report(self):
        self.assertIsNotNone(L, "hookcutter/listen.py is missing")
        # speech 1.0-1.6 and 1.7-3.0: the 0.1 s dip between them is no pause
        E = track([(1.0, 1.6, -20.0), (1.7, 3.0, -20.0)], dur=10.0)
        r = L.edge_report(E, 1.3, "start", 0.0, 10.0)
        self.assertTrue(r["voiced"])
        self.assertGreater(r["over_floor_db"], 30)
        self.assertAlmostEqual(r["quiet"], 0.8, delta=0.011)
        self.assertAlmostEqual(L.edge_report(E, 2.0, "start", 0.0, 10.0)["quiet"], 0.8, delta=0.011)
        self.assertAlmostEqual(L.edge_report(E, 2.0, "end", 0.0, 10.0)["quiet"], 3.2, delta=0.011)
        r = L.edge_report(E, 0.5, "start", 0.0, 10.0)
        self.assertFalse(r["voiced"])
        self.assertIsNone(r["quiet"])
        # a short pause sits at its middle, not 0.2 s in
        E = track([(1.0, 2.0, -20.0), (2.3, 3.0, -20.0)], dur=10.0)
        self.assertAlmostEqual(L.edge_report(E, 2.5, "start", 0.0, 10.0)["quiet"], 2.15, delta=0.011)
        # speech from the very start, and no pause within reach after: nothing to suggest
        E = track([(0.0, 9.0, -20.0)], dur=20.0)
        self.assertIsNone(L.edge_report(E, 4.0, "start", 0.0, 20.0)["quiet"])
        self.assertIsNone(L.edge_report(E, 3.0, "end", 0.0, 20.0)["quiet"])
        self.assertTrue(L.edge_report(E, 3.0, "end", 0.0, 20.0)["voiced"])


class ListenDocsTest(unittest.TestCase):

    def test_recipe_uses_hc_listen(self):
        qa = (SK / "references" / "qa.md").read_text(encoding="utf-8")
        sec = qa[qa.index("### Checking sound that has no words"):]
        sec = sec[:sec.index("\n## ")]
        self.assertIn("hc listen P", sec)
        self.assertNotIn("P/cache/scratch", sec, "the recipe still writes into P")
        self.assertNotIn("--clip scratch", sec, "the recipe still writes into P")
        self.assertRegex(sec, r"pause")
        skill = (SK / "SKILL.md").read_text(encoding="utf-8")
        table = skill[skill.index("## Command reference"):]
        self.assertTrue(re.search(r"`hc listen P START END", table), "no hc listen row")


if __name__ == "__main__":
    unittest.main()
