"""grab_frames stops decoding once its last selected frame is out (-frames:v).

Without the cap, ffmpeg decoded the rest of the input after the last selected frame: on a long
4K source a grab near the start decoded about an hour of video. These tests pin the cap to the
number of distinct frames each select expression emits, and check that the frames themselves
are unchanged.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "scripts"))

import numpy as np  # noqa: E402

import qa_media as qm  # noqa: E402
from hookcutter import cover_frames  # noqa: E402

SIZE = (64, 36)


class GrabFramesStopsEarly(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="hc-grab-frames-"))
        cls.src = qm.make_source(cls.tmp / "source.mp4", dur=6.0, size="320x180")   # 150 frames

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _grab(self, indices, **kw):
        """grab_frames, plus every ffmpeg command it ran."""
        real = subprocess.run
        cmds = []

        def spy(cmd, *a, **k):
            cmds.append(list(cmd))
            return real(cmd, *a, **k)

        with mock.patch.object(cover_frames.subprocess, "run", side_effect=spy):
            got = cover_frames.grab_frames(self.src, indices, size=SIZE, **kw)
        return got, cmds

    @staticmethod
    def _cap(cmd):
        return int(cmd[cmd.index("-frames:v") + 1])

    def test_cap_is_the_number_of_distinct_frames(self):
        got, cmds = self._grab([12, 3, 3, 40])
        self.assertEqual(sorted(got), [3, 12, 40])
        self.assertEqual(len(cmds), 1)
        self.assertEqual(self._cap(cmds[0]), 3)
        self.assertLess(cmds[0].index("-frames:v"), cmds[0].index("-f"))   # an output option

    def test_every_chunk_gets_its_own_cap(self):
        with mock.patch.object(cover_frames, "SELECT_CHUNK", 2):
            got, cmds = self._grab([5, 9, 30])
        self.assertEqual(sorted(got), [5, 9, 30])
        self.assertEqual([self._cap(c) for c in cmds], [2, 1])

    def test_frames_match_single_grabs(self):
        many, _ = self._grab([4, 20, 77])
        for i in (4, 20, 77):
            one, cmds = self._grab([i])
            self.assertEqual(self._cap(cmds[0]), 1)
            self.assertTrue(np.array_equal(many[i], one[i]), f"frame {i} differs")

    def test_index_past_the_end_is_a_short_read(self):
        got, cmds = self._grab([3, 100000])
        self.assertEqual(sorted(got), [3])
        self.assertEqual(self._cap(cmds[0]), 2)


if __name__ == "__main__":
    unittest.main()
