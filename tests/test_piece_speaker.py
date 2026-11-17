"""Piece key "speaker": who speaks a piece when the transcript has no speaker labels.

Found on a remote interview recorded as a fixed two-tile split screen (the host's tile on the
left, the guest's on the right) and transcribed without speaker labels. Every unlabelled word
counted as the clip's `speaker` (the guest), so:

  1  a closing piece where the HOST speaks, framed correctly on the host's own tile, FAILed QA
     G16 ("framed face is not <guest>") and held the export; the only escape, a `reaction`
     mark, mislabels a speaking turn;
  2  `hc cover`'s automatic pick had no speaker to prefer and chose a frame of the host;
  3  speaker-follow framing had no turns to follow.

The fix: a piece can name its speaker. Its words take that label wherever the transcript gives
none (transcript labels still win), and once any piece of a clip names one, the clip's other
unlabelled words take the clip's `speaker`. G16, framing and the cover read the labels.

Every fixture is synthetic: made-up words, a made-up two-tile layout, no media files.
"""
from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "scripts"))

import numpy as np  # noqa: E402

from fixtures import fake_transcript  # noqa: E402
from hookcutter import cover, framing, qa  # noqa: E402
from hookcutter import plan as P  # noqa: E402
from hookcutter import spec as S  # noqa: E402
from hookcutter.audio import EnergyTrack  # noqa: E402

FLOOR = -60.0
VOICE = -20.0
W, H = 1280, 720                      # a 720p remote recording, two tiles side by side
HOST_TILE = [0.0, 0.0, 0.5, 1.0]      # left tile
GUEST_TILE = [0.5, 0.0, 0.5, 1.0]     # right tile
SPEAKERS = {"Guest": {"x": 0.75}, "Host": {"x": 0.25}}
SHOTS = {"shots": [{"start_src": 0.0, "end_src": 60.0, "kind": "two", "multi_face": True,
                    "faces": [{"id": "A", "x": 0.25, "y": 0.45, "h": 0.2, "presence": 1.0},
                              {"id": "B", "x": 0.75, "y": 0.45, "h": 0.2, "presence": 1.0}]}],
         "ranges": [[0.0, 60.0]]}
PRESET = {"letterbox": {"y_center": 820, "blur": 28, "dim": 0.35},
          "motion": {"punch_in": 1.0, "min_gap": 3.5, "max_upscale": 3.0}}
SOURCE = {"video": {"width": W, "height": H, "sar": "1:1"}}


def energy(dur, spans, hop=0.01):
    db = np.full(int(round(dur / hop)), FLOOR, dtype=np.float64)
    for a, b in spans:
        db[int(round(a / hop)):int(round(b / hop))] = VOICE
    return EnergyTrack(db, hop=hop, t0=0.0)


def source(dur, fps=(25, 1)):
    v = {"fps": list(fps), "start": 0.0, "width": W, "height": H,
         "frames": int(math.floor(dur * fps[0] / fps[1] + 1e-9)), "hdr": "none", "sar": "1:1"}
    return {"fingerprint": "fp-test", "duration": dur, "video": v, "audio": {"start": 0.0, "index": 1}}


def words(labels=None):
    """12 guest words (0-11), a 1 s pause, then 12 host words (12-23). No labels unless given
    ({index: label})."""
    out, t = [], 0.6
    for k in range(24):
        if k == 12:
            t += 1.0
        text = "word%d" % k + ("." if k in (11, 23) else "")
        out.append((text, round(t, 3), round(t + 0.3, 3), (labels or {}).get(k)))
        t += 0.4
    return out


HOST_FIRST = words()[12][1]           # source time the host's piece starts


def clip(host_key=True, framing_marks=None, speaker="Guest", **extra):
    host = {"words": [12, 23], "expect": ["word12", "word23."]}
    if host_key:
        host["speaker"] = "Host"
    c = {"speaker": speaker,
         "pieces": [{"words": [0, 11], "expect": ["word0", "word11."]}, host],
         "framing": framing_marks if framing_marks is not None else [
             {"at_src": 0.0, "mode": "letterbox", "rect": GUEST_TILE},
             {"at_src": round(HOST_FIRST - 0.05, 3), "mode": "letterbox", "rect": HOST_TILE}]}
    c.update(extra)
    return c


def build(c, labels=None, speakers=SPEAKERS):
    ws = words(labels)
    tr = fake_transcript(ws)
    end = ws[-1][2] + 5.0
    raw = {"version": 1, "clips": {"c01": c}}
    if speakers is not None:
        raw["speakers"] = speakers
    spec = S.normalize_spec(raw)
    plan = P.plan_clip(spec, "c01", tr, source(end), energy(end, [(w[1], w[2]) for w in ws]), None)
    return plan, spec, tr


def g16(plan, c):
    tracks = framing.build_tracks(plan, c, SHOTS, SOURCE, "vertical", PRESET)
    wins = framing.windows(tracks, plan, SOURCE)
    return qa.framing_attribution(plan, c, SHOTS, wins, plan.get("speakers") or {}, W / H)


def piece_out(plan, n):
    return next(p["out"] for p in plan["pieces"] if p["n"] == n)


def lints(plan, needle, level=None):
    return [x for x in plan["lints"] if needle in x["msg"] and (level is None or x["level"] == level)]


class HostPieceOnTheHostsTile(unittest.TestCase):
    """1: the host's closing piece on the host's tile passes G16 once the piece names him."""

    def test_without_the_key_the_correct_framing_fails(self):
        c = clip(host_key=False)
        st, notes, _ = g16(build(c)[0], c)
        self.assertEqual(st, "FAIL", notes)
        self.assertIn("framed face is not Guest", notes[0])

    def test_with_the_key_the_correct_framing_passes(self):
        c = clip()
        plan = build(c)[0]
        st, notes, summary = g16(plan, c)
        self.assertEqual(st, "PASS", notes)
        self.assertIn("24 word(s) checked", summary)
        host = [w for w in plan["words"] if w["piece"] == 1 and w["i"] is not None]
        self.assertTrue(host and all(w["sp"] == "Host" for w in host))
        self.assertEqual(plan["pieces"][1]["speakers"], ["Host"])

    def test_the_host_piece_on_the_guests_tile_still_fails(self):
        c = clip(framing_marks=[{"at_src": 0.0, "mode": "letterbox", "rect": GUEST_TILE}])
        st, notes, _ = g16(build(c)[0], c)
        self.assertEqual(st, "FAIL", notes)
        self.assertTrue(any("framed face is not Host" in n for n in notes), notes)

    def test_the_guest_pieces_on_the_hosts_tile_still_fail(self):
        """The other pieces' unlabelled words are the clip speaker's, as before."""
        c = clip(framing_marks=[{"at_src": 0.0, "mode": "letterbox", "rect": HOST_TILE}])
        st, notes, _ = g16(build(c)[0], c)
        self.assertEqual(st, "FAIL", notes)
        self.assertTrue(any("framed face is not Guest" in n for n in notes), notes)
        self.assertFalse(any("framed face is not Host" in n for n in notes), notes)

    def test_a_piece_speaker_with_no_x_in_the_map_fails(self):
        c = clip()
        plan = build(c, speakers={"Guest": {"x": 0.75}})[0]
        st, notes, _ = g16(plan, c)
        self.assertEqual(st, "FAIL", notes)
        self.assertTrue(any('"Host"' in n for n in notes), notes)

    def test_transcript_labels_still_win(self):
        """A labelled word keeps its label; the key only names the unlabelled ones."""
        c = clip()
        plan = build(c, labels={k: "Guest" for k in range(18, 24)})[0]
        sp = {w["i"]: w["sp"] for w in plan["words"] if w["i"] is not None}
        self.assertEqual([sp[k] for k in range(12, 18)], ["Host"] * 6)
        self.assertEqual([sp[k] for k in range(18, 24)], ["Guest"] * 6)
        self.assertTrue(lints(plan, "pieces[1].speaker", "WARN"), plan["lints"])
        st, notes, _ = g16(plan, c)          # 6 guest words on the host's tile
        self.assertEqual(st, "FAIL", notes)

    def test_an_insert_and_the_other_pieces_take_the_spec_labels(self):
        ws = words()
        c = clip(insert=[{"after": 14, "text": "really", "src": [ws[14][2] + 0.01, ws[15][1] - 0.01]}])
        plan = build(c)[0]
        ins = [w for w in plan["words"] if w.get("ins") is not None]
        self.assertEqual([(w["text"], w["sp"], w["sp_from"]) for w in ins], [("really", "Host", "piece")])
        guest = [w for w in plan["words"] if w["piece"] == 0 and w["i"] is not None and not w.get("edge")]
        self.assertTrue(guest and all((w["sp"], w["sp_from"]) == ("Guest", "clip") for w in guest))

    def test_an_insert_takes_the_transcript_label_of_the_word_it_follows(self):
        """Where the transcript labels the word an insert follows, the insert takes that label
        (the speech is that speaker's), not the piece key: no one-word turn mid-sentence."""
        ws = words()
        ins = [{"after": 14, "text": "really", "src": [ws[14][2] + 0.01, ws[15][1] - 0.01]}]
        plan = build(clip(insert=ins), labels={k: "Guest" for k in range(24)})[0]
        got = [(w["text"], w["sp"]) for w in plan["words"] if w.get("ins") is not None]
        self.assertEqual(got, [("really", "Guest")])
        self.assertEqual(plan["pieces"][1]["speakers"], ["Guest"])
        # with no key anywhere an insert has no label, as before
        plan = build(clip(host_key=False, insert=ins), labels={k: "Guest" for k in range(24)})[0]
        got = [(w["text"], w["sp"]) for w in plan["words"] if w.get("ins") is not None]
        self.assertEqual(got, [("really", None)])

    def test_a_mismatch_on_unlabelled_words_names_the_piece_key(self):
        c = clip(host_key=False)
        _st, notes, _ = g16(build(c)[0], c)
        self.assertIn('"speaker"', notes[0])


class CoverPrefersTheClipSpeaker(unittest.TestCase):
    """2: the automatic cover frame comes from the clip speaker's pieces, not the host's."""

    def cands(self, plan, best_in_host=True):
        g0, g1 = piece_out(plan, 0)
        h0, h1 = piece_out(plan, 1)
        out = [{"at_out": round(g0 + (g1 - g0) * f, 3), "score": 0.30} for f in (0.25, 0.5, 0.75)]
        out += [{"at_out": round(h0 + (h1 - h0) * f, 3), "score": 0.33 if best_in_host else 0.2}
                for f in (0.25, 0.5, 0.75)]
        return out

    def test_the_pick_avoids_the_host_piece(self):
        c = clip()
        plan = build(c)[0]
        pool, note = cover.candidate_pool(self.cands(plan), plan, c["speaker"])
        self.assertIsNone(note)
        best = max(pool, key=lambda x: (x["score"], -x["at_out"]))
        g0, g1 = piece_out(plan, 0)
        self.assertTrue(g0 <= best["at_out"] <= g1, (best, piece_out(plan, 0), piece_out(plan, 1)))
        self.assertEqual(cover.speaker_at(plan, best["at_out"]), "Guest")
        h0, h1 = piece_out(plan, 1)
        self.assertEqual(cover.speaker_at(plan, (h0 + h1) / 2), "Host")

    def test_without_the_key_the_whole_clip_is_used(self):
        c = clip(host_key=False)
        plan = build(c)[0]
        pool, note = cover.candidate_pool(self.cands(plan), plan, c["speaker"])
        self.assertEqual(len(pool), 6)
        self.assertIsNone(note)

    def test_only_host_frames_fall_back_to_the_whole_clip_with_a_warning(self):
        c = clip()
        plan = build(c)[0]
        h0, h1 = piece_out(plan, 1)
        only = [{"at_out": round((h0 + h1) / 2, 3), "score": 0.3}]
        pool, note = cover.candidate_pool(only, plan, c["speaker"])
        self.assertEqual(pool, only)
        self.assertIn("Guest", note)


class SpeakerFramingFollowsThePiece(unittest.TestCase):
    """3: a "speaker" mark follows the piece speakers (host's piece on the host's x)."""

    def test_turns_follow_the_piece_speakers(self):
        c = clip(framing_marks=[{"at_src": 0.0, "mode": "speaker"}])
        plan = build(c)[0]
        self.assertEqual([t["sp"] for t in framing.speaker_turns(plan)], ["Guest", "Host"])
        tracks = framing.build_tracks(plan, c, SHOTS, SOURCE, "vertical", PRESET)
        wins = framing.windows(tracks, plan, SOURCE)
        h0, h1 = piece_out(plan, 1)
        mid = (h0 + h1) / 2
        win = next(w for w in wins if w["out"][0] <= mid < w["out"][1])
        self.assertEqual(win["mode"], "crop")
        self.assertTrue(win["lo"] <= 0.25 <= win["hi"], win)
        self.assertEqual(win.get("speaker"), "Host")
        st, notes, _ = qa.framing_attribution(plan, c, SHOTS, wins, SPEAKERS, W / H)
        self.assertEqual(st, "PASS", notes)

    def test_an_insert_that_starts_with_the_next_word_frames(self):
        """An inserted word (no transcript index) that starts at the same output time as a
        transcript word: once the key labels it, ordering the labelled words must not compare
        its missing index with a number (hc render crashed with a TypeError)."""
        ws = words()
        for mode in ("speaker", None):
            marks = [{"at_src": 0.0, "mode": "speaker"}] if mode else None
            c = clip(framing_marks=marks,
                     insert=[{"after": 13, "text": "like", "src": [ws[14][1], ws[14][1] + 0.08]}])
            if marks is None:
                c.pop("framing")
            plan = build(c)[0]
            self.assertFalse(lints(plan, "", "FAIL"), plan["lints"])
            ins = next(w for w in plan["words"] if w.get("ins") is not None)
            tie = next(w for w in plan["words"] if w["i"] == 14)
            self.assertEqual((ins["s_out"], ins["sp"]), (tie["s_out"], "Host"))
            self.assertEqual(framing.clip_labels(plan), ["Guest", "Host"])
            self.assertTrue(framing.follow_ok(plan))
            turns = framing.speaker_turns(plan)
            self.assertEqual([t["sp"] for t in turns], ["Guest", "Host"])
            # an inserted word sorts right after the transcript word before it in the plan
            self.assertEqual(turns[1]["words"][:4], [12, 13, 14, None])
            tracks = framing.build_tracks(plan, c, SHOTS, SOURCE, "vertical", PRESET)
            wins = framing.windows(tracks, plan, SOURCE)
            st, notes, _ = qa.framing_attribution(plan, c, SHOTS, wins, SPEAKERS, W / H)
            self.assertEqual(st, "PASS", (mode, notes))

    def test_a_piece_speaker_without_a_clip_speaker_warns_and_still_fails(self):
        """With no clip "speaker" the other pieces' words stay unlabelled and speaker-follow
        framing follows the one label through them: L01 says so, and G16 still FAILs."""
        c = clip(speaker=None, framing_marks=[{"at_src": 0.0, "mode": "speaker"}])
        plan = build(c)[0]
        got = lints(plan, "has no \"speaker\"", "WARN")
        self.assertEqual(len(got), 1, plan["lints"])
        self.assertIn("pieces[1]", got[0]["msg"])
        st, notes, _ = g16(plan, c)
        self.assertEqual(st, "FAIL", notes)
        self.assertTrue(any("no speaker labels or clip speaker" in n for n in notes), notes)
        # a clip speaker, or no piece key, and there is no such warning
        self.assertFalse(lints(build(clip())[0], "has no \"speaker\""))
        self.assertFalse(lints(build(clip(speaker=None, host_key=False))[0], "has no \"speaker\""))
        one = clip(speaker=None)
        one["pieces"] = one["pieces"][1:]           # every piece names its speaker
        self.assertFalse(lints(build(one)[0], "has no \"speaker\""))

    def test_no_piece_speaker_keeps_the_letterbox(self):
        c = clip(host_key=False, framing_marks=[{"at_src": 0.0, "mode": "speaker"}])
        plan = build(c)[0]
        self.assertEqual(framing.speaker_turns(plan), [])
        self.assertTrue(all(w.get("sp") is None for w in plan["words"]))


class SpecAndStaleness(unittest.TestCase):

    def test_the_key_is_known_and_kept(self):
        spec = S.normalize_spec({"version": 1, "speakers": SPEAKERS, "clips": {"c01": clip()}})
        self.assertEqual(spec["clips"]["c01"]["pieces"][1]["speaker"], "Host")
        self.assertFalse([x for x in spec["clips"]["c01"]["_issues"] if "speaker" in x["msg"]])

    def test_a_piece_without_the_key_hashes_as_before(self):
        """No "speaker" entry is added to a piece that does not set one (the clip hash of
        every existing clip is unchanged)."""
        spec = S.normalize_spec({"version": 1, "clips": {"c01": clip(host_key=False)}})
        self.assertNotIn("speaker", spec["clips"]["c01"]["pieces"][1])

    def test_a_bad_value_fails(self):
        for bad in ("", "  ", 3, ["Host"]):
            c = clip()
            c["pieces"][1]["speaker"] = bad
            spec = S.normalize_spec({"version": 1, "clips": {"c01": c}})
            got = [x for x in spec["clips"]["c01"]["_issues"] if "pieces[1].speaker" in x["msg"]]
            self.assertEqual([x["level"] for x in got], ["FAIL"], repr(bad))
            self.assertNotIn("speaker", spec["clips"]["c01"]["pieces"][1])

    def test_a_label_missing_from_the_speakers_map_warns(self):
        spec = S.normalize_spec({"version": 1, "speakers": {"Guest": {"x": 0.75}},
                                 "clips": {"c01": clip()}})
        got = [x for x in spec["clips"]["c01"]["_issues"] if "pieces[1].speaker" in x["msg"]]
        self.assertEqual([x["level"] for x in got], ["WARN"])
        self.assertEqual(spec["clips"]["c01"]["pieces"][1]["speaker"], "Host")
        # with no map at all any label is accepted
        spec = S.normalize_spec({"version": 1, "clips": {"c01": clip()}})
        self.assertFalse([x for x in spec["clips"]["c01"]["_issues"] if "pieces[1].speaker" in x["msg"]])

    def test_a_map_label_used_only_by_a_piece_is_not_unused(self):
        plan = build(clip())[0]
        self.assertFalse(lints(plan, "matches no transcript speaker"), plan["lints"])
        plan = build(clip(host_key=False))[0]
        self.assertTrue(lints(plan, "'Host' matches no transcript speaker", "WARN"))

    def test_filled_labels_do_not_make_the_plan_stale(self):
        plan, _spec, tr = build(clip())
        self.assertTrue(any(w.get("sp") == "Host" for w in plan["words"]))
        self.assertFalse(P.transcript_changed(plan, tr))
        tr["words"][14]["sp"] = "Guest"       # the transcript now labels a word: re-plan
        self.assertTrue(P.transcript_changed(plan, tr))


if __name__ == "__main__":
    unittest.main()
