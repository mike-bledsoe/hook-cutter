"""Regression tests for four QA and cover false alarms found on letterboxed renders of a remote
interview (host and guest in two tiles, each clip framed with a letterbox `rect` of one tile).

- G8 and G12 judged the renderer's own letterbox fill (the blurred, dimmed copy of the tile
  above and below it) as black bars: a dark room above the tile read as a dark band.
- G8's freeze check ran on the whole frame, where the still fill outweighs a talking head in
  the tile, so a nearly still talker read as a frozen picture.
- The cover json called a headline set below the preset size range a typeface fallback, and
  G12 reported that one condition twice, once as a font that "fell back to another face".
- `cover.at_out` and `hc cover --at` floored a time to a frame, so a frame time written with
  three decimals named the frame before it; and a recorded time that lands on another source
  frame after an edge moved gave no warning.

The real defects stay caught: a black bar inside the picture is measured against the
picture's height (the share the fill used to show it at, so 2:1 and 2.39:1 source bars keep
their levels), and a cover cut from the source, a frozen tile and a stale render are judged as
before. Everything is synthetic: lavfi renders,
hand-written render logs, cover json and PIL covers.
"""

from __future__ import annotations

import contextlib
import io
import re
import shutil
import sys
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "scripts"))

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

import test_codex_ff as FF  # noqa: E402  (helpers: _MockedCover)
from hookcutter import cli, cover, cover_frames, env, framing, qa  # noqa: E402
from hookcutter import render as render_mod  # noqa: E402
from hookcutter.project import Project  # noqa: E402

# One tile of a 1280x720 two-up call: the right half, 90 % of the height (640x648 px).
# Render fits it to 1080x1094 and centres it on the clean preset's letterbox y_center 820:
# the picture holds output rows 272-1366, the blurred fill the rows above and below.
TILE = {"seg": 0, "mode": "rect", "lo": 0.5, "hi": 1.0, "lo_any": 0.5, "hi_any": 1.0,
        "y_lo": 0.0, "y_hi": 0.9, "mark": 0, "reaction": False, "multi_face": True}
PIC = (272, 1366)
DUR = 2.4
PLAN = {"fps": [25, 1], "duration_out": DUR, "joins": []}


def _ff(args):
    env.ff(list(args))


def _letterboxed(path: Path, *, top: str, picture: str, bar: int = 0, patch: int = 90, patch_y: int = 330) -> Path:
    """A 1080x1920 25 fps render laid out like a letterboxed tile: `top` fills rows 0-272
    ("dark": a smooth dark gradient, full-range luma about 6-19, like the blurred fill of a dark
    room; "grey": a still mid grey), the picture rows 272-1366 hold `picture` ("moving": a
    testsrc2 pattern; "talker": a still grey picture with one small moving patch, like a talking
    head, `patch` px square at picture row `patch_y`; "still": nothing moves), the rows below
    are a still grey fill. `bar` paints that many black rows at the top of the picture (a bar
    of the source inside the tile)."""
    h_top, h_pic = PIC[0], PIC[1] - PIC[0]
    srcs = ["-f", "lavfi", "-i", f"color=c=0x606060:s=1080x1920:r=25:d={DUR}"]
    if top == "dark":
        srcs += ["-f", "lavfi", "-i", f"color=c=black:s=1080x{h_top}:r=25:d={DUR},"
                 "format=yuv420p,geq=lum='21+11*X/W':cb=128:cr=128"]
    else:
        srcs += ["-f", "lavfi", "-i", f"color=c=0x505050:s=1080x{h_top}:r=25:d={DUR}"]
    if picture == "moving":
        srcs += ["-f", "lavfi", "-i", f"testsrc2=size=1080x{h_pic}:rate=25:duration={DUR}"]
    else:
        srcs += ["-f", "lavfi", "-i", f"color=c=0x707070:s=1080x{h_pic}:r=25:d={DUR}"]
    chain = "[2:v]null[p0]"
    if picture == "talker":
        srcs += ["-f", "lavfi", "-i", f"color=c=white:s={patch}x{patch}:r=25:d={DUR}"]
        chain = f"[2:v][3:v]overlay=x='100+4*n':y={patch_y}:eval=frame[p0]"
    if bar:
        chain += f";[p0]drawbox=x=0:y=0:w=iw:h={bar}:color=black:t=fill[p1]"
    else:
        chain += ";[p0]null[p1]"
    chain += (f";[0:v][1:v]overlay=x=0:y=0[b0];[b0][p1]overlay=x=0:y={PIC[0]},format=yuv420p[v]")
    _ff(srcs + ["-filter_complex", chain, "-map", "[v]", "-frames:v", str(int(DUR * 25)),
                "-c:v", "libx264", "-qp", "0", "-preset", "ultrafast", "-pix_fmt", "yuv420p", str(path)])
    return path


class _Project(unittest.TestCase):
    def project(self, name: str, windows=None, source=(1280, 720)) -> Project:
        P = Project(self.tmp / name, create=True)
        env.write_json(P.root / "source.json", {"video": {"width": source[0], "height": source[1]},
                                                "audio": {}})
        if windows is not None:
            env.write_json(P.render_path("c1", "vertical", "clean", ".log.json"),
                           {"version": 1, "clip": "c1", "format": "vertical", "style": "clean",
                            "size": [1080, 1920], "framing_windows": windows})
        return P

    def ctx(self, P):
        return qa._Ctx(P, {"clips": {"c1": {"speaker": "Speaker A"}}}, None, False, False)


# ---------------------------------------------------------------- the picture rows of a letterbox

class PictureRows(unittest.TestCase):
    """QA places the picture of a letterbox window where render puts it."""

    def test_rows_match_the_render_chain(self):
        cases = [((1280, 720), [640, 0, 640, 648], 820), ((1280, 720), [664, 38, 576, 588], 820),
                 ((1280, 720), [664, 38, 576, 588], 680), ((1920, 1080), [0, 0, 1920, 1080], 820),
                 ((1280, 720), [0, 0, 1280, 720], 300), ((1280, 720), [0, 0, 1280, 720], 1900)]
        for (sw, sh), rect, yc in cases:
            with self.subTest(source=(sw, sh), rect=rect, y_center=yc):
                sub = {"src": [0.0, 1.0], "frames": 25, "mode": "letterbox", "size": [sw, sh], "cmds": [],
                       "zoom": 1.0, "rect": rect, "rect_mark": True, "y_center": yc, "blur": 28, "dim": 0.35}
                chain = render_mod._sub_chain("[0:0]", "[v]", sub, 0, 0.0, Fraction(25), {"width": sw, "height": sh},
                                              (1080, 1920), None, False)
                fh = int(re.search(r"\[lbf0\]scale=1080:(\d+)", chain).group(1))
                fy = int(re.search(r"overlay=x=0:y=(\d+)", chain).group(1))
                win = framing.windows([[sub]], {"fps": [25, 1], "segments": [{"out": [0.0, 1.0]}]},
                                      {"width": sw, "height": sh})[0]
                self.assertEqual(qa.letterbox_picture_rows(win, (sw, sh), (1080, 1920), yc), (fy, fy + fh))

    def test_a_band_is_measured_against_the_rows_given(self):
        rows = np.array([10] * 2 + [200] * 98)
        self.assertEqual(qa.dark_bands(rows), [])                        # 2 % of 100 rows
        self.assertEqual(qa.dark_bands(rows, height=50), [(0, 2)])      # 4 % of a 50-row picture
        self.assertEqual(qa._band_text([(0, 2)], 100, 1920, 50), "y 0-38 (4% of the picture)")
        self.assertEqual(qa._band_text([(0, 4)], 100, 1920), "y 0-77 (4%)")

    def test_a_crop_or_a_full_height_picture_has_no_fill_rows(self):
        self.assertIsNone(qa.letterbox_picture_rows(dict(TILE, mode="crop"), (1280, 720), (1080, 1920), 820))
        tall = dict(TILE, lo=0.0, hi=0.25)                      # 320x648: taller than 9:16
        self.assertIsNone(qa.letterbox_picture_rows(tall, (1280, 720), (1080, 1920), 820))
        self.assertEqual(qa.letterbox_picture_rows(TILE, (1280, 720), (1080, 1920), 820), PIC)


# ---------------------------------------------------------------- G8 dark band and freeze

class LetterboxPicture(_Project):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="hc-qafa-g8-"))
        cls.dark_fill = _letterboxed(cls.tmp / "dark_fill.mp4", top="dark", picture="moving")
        cls.tile_bar = _letterboxed(cls.tmp / "tile_bar.mp4", top="dark", picture="moving", bar=128)
        cls.talker = _letterboxed(cls.tmp / "talker.mp4", top="grey", picture="talker")
        cls.small_talker = _letterboxed(cls.tmp / "small_talker.mp4", top="grey", picture="talker",
                                        patch=60, patch_y=100)
        cls.still = _letterboxed(cls.tmp / "still.mp4", top="grey", picture="still")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def g8(self, P, render, **kw):
        return qa.g8_black_freeze(PLAN, render, None, "vertical/clean", self.ctx(P), "c1", "vertical", "clean", **kw)

    def test_the_fill_of_a_dark_room_is_not_a_band(self):
        g = self.g8(self.project("fill", [dict(TILE, out=[0.0, DUR])]), self.dark_fill)
        self.assertEqual(g["status"], "PASS", g["evidence"])
        self.assertIn("letterbox fill", g["evidence"])

    def test_a_black_bar_inside_the_tile_is_still_a_band(self):
        # 128 rows of a 1094-row picture (12 %): judged against the picture, as the fill used to
        # show it magnified to the frame height, so it FAILs at the same level as before
        g = self.g8(self.project("bar", [dict(TILE, out=[0.0, DUR])]), self.tile_bar)
        self.assertEqual(g["status"], "FAIL", g["evidence"])
        self.assertIn("dark band y 264-400 (12% of the picture)", g["evidence"])
        P = self.project("bar-aa", [dict(TILE, out=[0.0, DUR])])
        env.write_json(P.root / "source.json", {"video": {"width": 1280, "height": 720,
                                                          "active_area": [0, 0, 1280, 720]}, "audio": {}})
        self.assertEqual(self.g8(P, self.tile_bar)["status"], "FAIL")

    def test_without_the_framing_the_fill_is_judged_as_before(self):
        for name, windows, kw in (("nolog", None, {}), ("crop", [dict(TILE, mode="crop", out=[0.0, DUR])], {}),
                                  ("stale", [dict(TILE, out=[0.0, DUR])], {"stale": True})):
            with self.subTest(name):
                g = self.g8(self.project(name, windows), self.dark_fill, **kw)
                self.assertEqual(g["status"], "FAIL", g["evidence"])
                self.assertIn("dark band y 0-272 (14%)", g["evidence"])

    def test_a_talking_tile_over_a_still_fill_is_not_frozen(self):
        g = self.g8(self.project("talker", [dict(TILE, out=[0.0, DUR])]), self.talker)
        self.assertEqual(g["status"], "PASS", g["evidence"])
        # judged on the whole frame (no framing data), the still fill outweighs the moving patch
        g = self.g8(self.project("talker-nolog"), self.talker)
        self.assertIn("frozen picture", g["evidence"])

    def test_the_caption_band_inside_the_tile_does_not_dilute_the_motion(self):
        # rows 800-1300 of the tile are masked for the captions; the moving patch sits above them
        P = self.project("band", [dict(TILE, out=[0.0, DUR])])
        g = qa.g8_black_freeze(PLAN, self.small_talker, [800, 500], "vertical/clean", self.ctx(P), "c1",
                               "vertical", "clean")
        self.assertEqual(g["status"], "PASS", g["evidence"])
        self.assertIn("594 rows the caption band leaves", " ".join(g["details"]))

    def test_a_frozen_tile_still_warns(self):
        g = self.g8(self.project("still", [dict(TILE, out=[0.0, DUR])]), self.still)
        self.assertEqual(g["status"], "WARN", g["evidence"])
        self.assertIn("frozen picture 0.00-2.40s", g["evidence"])
        # with the caption band masked over part of the tile, a still tile is still frozen
        P = self.project("still-band", [dict(TILE, out=[0.0, DUR])])
        g = qa.g8_black_freeze(PLAN, self.still, [800, 500], "vertical/clean", self.ctx(P), "c1", "vertical", "clean")
        self.assertIn("frozen picture 0.00-2.40s", g["evidence"])

    def test_a_clip_that_also_crops_checks_freezes_on_the_whole_frame(self):
        half = DUR / 2
        P = self.project("mixed", [dict(TILE, out=[0.0, half]), dict(TILE, mode="crop", out=[half, DUR])])
        self.assertIn("frozen picture", self.g8(P, self.talker)["evidence"])


# ---------------------------------------------------------------- G12 dark band on the cover

class LetterboxCover(_Project):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="hc-qafa-g12-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def cover(self, P, *, shift=0, bar=0, frame_source="segments"):
        """A letterboxed cover: a dark gradient fill above the tile (moved up by `shift` like
        the photo), a bright tile, `bar` black rows at the top of the tile."""
        top = PIC[0] - shift
        img = Image.new("RGB", (1080, 1920), (120, 130, 140))
        px = np.asarray(img).copy()
        ramp = (6 + 13 * np.arange(1080) / 1080.0).astype(np.uint8)
        px[:top, :, :] = ramp[None, :, None]
        if bar:
            px[top:top + bar, :, :] = 0
        img = Image.fromarray(px)
        ImageDraw.Draw(img).rectangle([80, 1300, 1000, 1500], fill=(255, 255, 255))
        cover.save_jpeg(img, P.cover_path("c1", "clean", "main"))
        for kind in ("grid", "mobile"):
            cover.save_jpeg(img.resize((270, 360)), P.cover_path("c1", "clean", kind))
        env.write_json(P.cover_path("c1", "clean", "json"),
                       {"headline_box": [80, 1300, 920, 200], "text_box": [80, 1300, 920, 200],
                        "face_box": [400, 500, 300, 360], "faces_found": 1, "crown_y": 420,
                        "crown_method": "image", "at_out": 1.0, "shift": shift, "frame_source": frame_source,
                        "band": [240, 1680], "warnings": [], "spec_hash": "h"})

    def g12(self, P, stale=False):
        return qa.g12_cover(self.ctx(P), "c1", "clean", {"spec_hash": "h"}, render_stale=stale)

    def test_the_fill_above_the_tile_is_not_a_band(self):
        P = self.project("P", [dict(TILE, out=[0.0, DUR])])
        self.cover(P)
        g = self.g12(P)
        self.assertEqual(g["status"], "PASS", g["evidence"])
        self.cover(P, shift=120)                  # the photo slid up: so did the fill rows
        self.assertEqual(self.g12(P)["status"], "PASS")

    def test_a_black_bar_inside_the_tile_still_fails(self):
        P = self.project("P", [dict(TILE, out=[0.0, DUR])])
        self.cover(P, bar=120)
        g = self.g12(P)
        self.assertEqual(g["status"], "FAIL", g["evidence"])
        self.assertIn("dark band on the cover: y 264-392 (12% of the picture)", g["evidence"])

    def test_a_thin_bar_inside_the_tile_fails_against_the_picture_height(self):
        # 48 rows: 3 % of the 1920-row cover would not report it, 4 % of the 1094-row picture does
        P = self.project("P", [dict(TILE, out=[0.0, DUR])])
        self.cover(P, bar=48)
        g = self.g12(P)
        self.assertEqual(g["status"], "FAIL", g["evidence"])
        self.assertIn("of the picture", g["evidence"])
        self.cover(P, bar=48, shift=120)          # the photo slid up: the bar is still judged on the picture
        self.assertEqual(self.g12(P)["status"], "FAIL")

    def test_a_cover_the_render_cannot_vouch_for_is_judged_as_before(self):
        for name, windows, src, stale in (("source", [dict(TILE, out=[0.0, DUR])], "source", False),
                                          ("nolog", None, "segments", False),
                                          ("stale", [dict(TILE, out=[0.0, DUR])], "segments", True),
                                          ("crop", [dict(TILE, mode="crop", out=[0.0, DUR])], "segments", False)):
            with self.subTest(name):
                P = self.project(name, windows)
                self.cover(P, frame_source=src)
                g = self.g12(P, stale=stale)
                self.assertEqual(g["status"], "FAIL", g["evidence"])
                self.assertIn("dark band on the cover: y 0-272", g["evidence"])


# ---------------------------------------------------------------- source bars through render's letterbox

class SourceBarsThroughTheLetterbox(_Project):
    """A 1280x720 source with letterbox bars baked into the picture and no measured
    `active_area`, framed `letterbox` on the whole frame and rendered with render's own chain.
    The fill is a blurred copy of the picture stretched to the frame height, so it used to show
    the bars magnified; with the fill left out, a bar is judged against the picture's own
    height, at the level the fill gave it: 2:1 content (bars 5.6 % of the source) WARNs in
    G8 and FAILs in G12, 2.39:1 (12.8 %) FAILs in both."""

    FPS, LEN, SW, SH = 25, 1.2, 1280, 720

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="hc-qafa-bars-"))
        cls.renders = {}
        n = int(cls.FPS * cls.LEN)
        for name, bar in (("none", 0), ("2to1", 40), ("239to1", 92)):
            src = cls.tmp / f"src-{name}.mp4"
            vf = f"testsrc2=size={cls.SW}x{cls.SH}:rate={cls.FPS}:duration={cls.LEN}"
            if bar:
                vf += (f",drawbox=x=0:y=0:w=iw:h={bar}:color=black:t=fill,"
                       f"drawbox=x=0:y={cls.SH - bar}:w=iw:h={bar}:color=black:t=fill")
            _ff(["-f", "lavfi", "-i", vf, "-frames:v", str(n), *render_mod.LOSSLESS_V, str(src)])
            sub = {"src": [0.0, cls.LEN], "frames": n, "mode": "letterbox", "size": [cls.SW, cls.SH],
                   "cmds": [], "zoom": 1.0, "rect": [0, 0, cls.SW, cls.SH], "y_center": 820,
                   "blur": 28, "dim": 0.35}
            video = {"width": cls.SW, "height": cls.SH}
            chain = render_mod._sub_chain("[0:v]", "[v]", sub, 0, 0.0, Fraction(cls.FPS), video,
                                          (1080, 1920), None, False)
            rp = cls.tmp / f"r-{name}.mp4"
            _ff(["-i", str(src), "-filter_complex", chain, "-map", "[v]", *render_mod.FINAL_V, str(rp)])
            win = framing.windows([[sub]], {"fps": [cls.FPS, 1], "segments": [{"out": [0.0, cls.LEN]}]}, video)
            cls.renders[name] = (rp, win)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def g8(self, name):
        rp, win = self.renders[name]
        P = self.project(f"g8-{name}", win)
        plan = {"fps": [self.FPS, 1], "duration_out": self.LEN, "joins": []}
        return qa.g8_black_freeze(plan, rp, None, "vertical/clean", self.ctx(P), "c1", "vertical", "clean")

    def g12(self, name):
        rp, win = self.renders[name]
        P = self.project(f"g12-{name}", win)
        png = self.tmp / f"frame-{name}.png"
        _ff(["-ss", "0.6", "-i", str(rp), "-frames:v", "1", str(png)])
        with Image.open(png) as im:
            img = im.convert("RGB")
        ImageDraw.Draw(img).rectangle([80, 1300, 1000, 1500], fill=(255, 255, 255))
        cover.save_jpeg(img, P.cover_path("c1", "clean", "main"))
        for kind in ("grid", "mobile"):
            cover.save_jpeg(img.resize((270, 360)), P.cover_path("c1", "clean", kind))
        env.write_json(P.cover_path("c1", "clean", "json"),
                       {"headline_box": [80, 1300, 920, 200], "text_box": [80, 1300, 920, 200],
                        "face_box": [400, 600, 300, 300], "faces_found": 1, "crown_y": 560,
                        "crown_method": "image", "at_out": 0.6, "shift": 0, "frame_source": "segments",
                        "band": [240, 1680], "warnings": [], "spec_hash": "h"})
        return qa.g12_cover(self.ctx(P), "c1", "clean", {"spec_hash": "h"})

    def test_g8_judges_a_source_bar_against_the_picture(self):
        for name, want in (("none", "PASS"), ("2to1", "WARN"), ("239to1", "FAIL")):
            with self.subTest(name):
                g = self.g8(name)
                self.assertEqual(g["status"], want, g["evidence"])
                if want != "PASS":
                    self.assertIn("of the picture", g["evidence"])

    def test_g12_judges_a_source_bar_against_the_picture(self):
        for name, want in (("none", "PASS"), ("2to1", "FAIL"), ("239to1", "FAIL")):
            with self.subTest(name):
                g = self.g12(name)
                self.assertEqual(g["status"], want, g["evidence"])
                if want == "PASS":
                    self.assertIn("letterbox fill not judged", g["evidence"])

# ---------------------------------------------------------------- size fallback vs typeface

class CoverSizeFallback(_Project):
    SIZE_MSG = "headline needed 86 px, below the preset minimum 96 px"

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="hc-qafa-fb-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.P = self.project("P")
        img = Image.new("RGB", (1080, 1920), (90, 100, 120))
        cover.save_jpeg(img, self.P.cover_path("c1", "clean", "main"))
        for kind in ("grid", "mobile"):
            cover.save_jpeg(img.resize((270, 360)), self.P.cover_path("c1", "clean", kind))

    def meta(self, **kw):
        m = {"headline_box": [80, 1200, 900, 300], "text_box": [80, 1200, 900, 300],
             "face_box": [300, 400, 400, 500], "faces_found": 1, "crown_y": 350, "crown_method": "image",
             "band": [240, 1680], "warnings": [], "spec_hash": "h"}
        m.update(kw)
        env.write_json(self.P.cover_path("c1", "clean", "json"), m)

    def g12(self):
        return qa.g12_cover(self.ctx(self.P), "c1", "clean", {"spec_hash": "h"})

    def test_compose_records_a_size_fallback_not_a_typeface_one(self):
        _img, info = cover.compose(Image.new("RGB", (1080, 1920), (40, 44, 52)), [],
                                   "SUPERCALIFRAGILISTICEXPIALIDOCIOUS IS LONG", dict(cover.DEFAULT_COVER))
        self.assertIn("below the preset minimum 96 px", info["size_fallback"])
        self.assertIs(info["font_fallback"], False)          # the cover is set in the preset's own face
        self.assertEqual(sum("below the preset minimum" in w for w in info["warnings"]), 1)
        _img, fits = cover.compose(Image.new("RGB", (1080, 1920), (40, 44, 52)), [], "BIG *IDEA*",
                                   dict(cover.DEFAULT_COVER))
        self.assertIsNone(fits["size_fallback"])

    def test_g12_reports_a_size_fallback_once_and_as_a_size(self):
        self.meta(size_fallback=self.SIZE_MSG, font_fallback=False, warnings=[self.SIZE_MSG])
        g = self.g12()
        self.assertEqual(g["status"], "WARN", g["evidence"])
        self.assertEqual(g["evidence"].count("86 px"), 1, g["evidence"])
        self.assertNotIn("another face", g["evidence"])

    def test_g12_reads_an_older_cover_json_the_same_way(self):
        # covers made before size_fallback was recorded set font_fallback from the size fit
        self.meta(font_fallback=True, warnings=[self.SIZE_MSG])
        g = self.g12()
        self.assertEqual(g["status"], "WARN", g["evidence"])
        self.assertEqual(g["evidence"].count("86 px"), 1, g["evidence"])
        self.assertNotIn("another face", g["evidence"])

    def test_a_typeface_fallback_still_warns(self):
        self.meta(font_fallback=True, size_fallback=None)
        g = self.g12()
        self.assertEqual(g["status"], "WARN", g["evidence"])
        self.assertIn("the headline font fell back to another face", g["evidence"])


# ---------------------------------------------------------------- cover.at_out names the nearest frame

class CoverFrameRounding(FF._MockedCover):
    def at(self, t):
        return self.patched(lambda: cover.make_cover(self.P, "c01", "clean", at_out=t))

    def test_a_time_names_its_nearest_frame(self):
        self.rate(24, 1, dur=40.0)
        for given, frame in ((35.458, 851), (24.333, 584), (24.29, 583), (24.31, 583), (35.4375, 851)):
            with self.subTest(given=given):
                self.assertEqual(self.at(given)["at_out"], round(frame / 24, 6))
        self.rate(25, 1)
        self.assertEqual(self.at(2.99)["at_out"], 2.96)          # the last frame, never past the end

    def test_hc_status_rounds_the_same_way(self):
        self.rate(24, 1, dur=40.0)
        self.at(24.333)
        clip = {"cover": {"headline": "SOLD IT *FIRST*", "at_out": 24.333}}
        self.assertNotIn("at_out", cli.cover_stale_reasons(self.P, clip, clip, "c01", "clean", self.plan, None, False))
        moved = {"cover": {"headline": "SOLD IT *FIRST*", "at_out": 24.4}}
        self.assertIn("at_out", cli.cover_stale_reasons(self.P, moved, moved, "c01", "clean", self.plan, None, False))

    def test_a_recorded_time_on_another_source_frame_warns(self):
        self.rate(25, 1)
        quiet = io.StringIO()
        with contextlib.redirect_stderr(quiet):
            first = self.at(1.2)
            self.at(1.2)
        self.assertNotIn("source frame", quiet.getvalue())        # same frame: nothing to say
        self.plan["segments"][0]["src"] = [100.2, 103.2]          # an edge before the frame moved
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            again = self.at(1.2)
        self.assertEqual(again["at_out"], first["at_out"])
        self.assertIn("101.200", err.getvalue())
        self.assertIn("101.400", err.getvalue())
        self.assertNotIn("source", " ".join(again["warnings"]))  # said when it happens, not stored


if __name__ == "__main__":
    unittest.main()
