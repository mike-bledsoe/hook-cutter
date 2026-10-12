"""YuNet face detection (OpenCV FaceDetectorYN) with frame-normalized output.

`detect(image)` returns faces sorted largest first. Each face is a `Face`, a 5-item list
`[cx, cy, w, h, score]` normalized to the frame (so it unpacks and serializes exactly like
the face rows in shots json) that also carries `.landmarks`: five normalized (x, y) points in
YuNet order (right eye, left eye, nose tip, right mouth corner, left mouth corner).
"""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np

from . import env

MODEL_FILE = env.SKILL_ROOT / "models" / "face_detection_yunet_2023mar.onnx"
SCORE_THRESHOLD = 0.6
NMS_THRESHOLD = 0.3
TOP_K = 200
MAX_SIDE = 960          # frames larger than this are downscaled first (YuNet misses huge faces)
MIN_SIDE = 16           # smaller frames are skipped
MIN_BOX_PX = 4          # boxes smaller than this (in detector pixels) are noise

_LOCAL = threading.local()


class Face(list):
    """[cx, cy, w, h, score] normalized to the frame, plus `.landmarks`."""

    __slots__ = ("landmarks",)

    def __init__(self, values, landmarks=None):
        super().__init__(float(v) for v in values)
        self.landmarks = [tuple(p) for p in (landmarks or [])]

    cx = property(lambda self: self[0])
    cy = property(lambda self: self[1])
    w = property(lambda self: self[2])
    h = property(lambda self: self[3])
    score = property(lambda self: self[4])

    def box_px(self, width: int, height: int) -> tuple:
        """(x, y, w, h) in pixels of a width x height frame."""
        return ((self[0] - self[2] / 2) * width, (self[1] - self[3] / 2) * height,
                self[2] * width, self[3] * height)

    def landmarks_px(self, width: int, height: int) -> list:
        return [(x * width, y * height) for x, y in self.landmarks]

    def as_dict(self) -> dict:
        return {"cx": self[0], "cy": self[1], "w": self[2], "h": self[3], "score": self[4],
                "landmarks": [list(p) for p in self.landmarks]}

    def rounded(self, nd: int = 4) -> list:
        """Plain list rounded for JSON (the shots json sample row)."""
        return [round(v, nd) for v in self]


def model_path() -> Path:
    if not MODEL_FILE.is_file():
        raise env.HCError("face model missing: %s (reinstall the skill)" % MODEL_FILE, code=3)
    return MODEL_FILE


def _detector(w: int, h: int, score: float):
    import cv2
    det = getattr(_LOCAL, "det", None)
    if det is None or getattr(_LOCAL, "score", None) != score:
        if not hasattr(cv2, "FaceDetectorYN"):
            raise env.HCError("this OpenCV build has no FaceDetectorYN (need opencv >= 4.8). "
                              "Run: hc doctor --install", code=3)
        try:  # silence OpenCV's backend chatter on stderr
            cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
        except Exception:  # noqa: BLE001 (older builds lack the logging module)
            pass
        det = cv2.FaceDetectorYN.create(str(model_path()), "", (w, h), score, NMS_THRESHOLD, TOP_K)
        _LOCAL.det = det
        _LOCAL.score = score
        _LOCAL.size = None
    if _LOCAL.size != (w, h):
        det.setInputSize((w, h))
        _LOCAL.size = (w, h)
    return det


def _as_bgr(image) -> np.ndarray:
    """BGR uint8 HxWx3 from a numpy image (BGR, gray or BGRA) or a PIL image."""
    if not isinstance(image, np.ndarray):
        try:
            from PIL import Image
        except ImportError:  # pragma: no cover
            Image = None
        if Image is not None and isinstance(image, Image.Image):
            rgb = np.asarray(image.convert("RGB"))
            return np.ascontiguousarray(rgb[:, :, ::-1])
        image = np.asarray(image)
    arr = image
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        arr = np.repeat(arr[:, :, None], 3, axis=2)
    elif arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[:, :, :3]
    elif arr.ndim != 3 or arr.shape[2] != 3:
        raise env.HCError("faces.detect: expected an HxWx3 image, got shape %s" % (arr.shape,), code=3)
    return np.ascontiguousarray(arr)


def detect(image, score_threshold: float = SCORE_THRESHOLD, max_side: int = MAX_SIDE) -> list:
    """Faces in one frame (numpy BGR, gray, BGRA or a PIL image), largest first."""
    import cv2
    bgr = _as_bgr(image)
    h0, w0 = bgr.shape[:2]
    if min(h0, w0) < MIN_SIDE:
        return []
    scale = min(1.0, float(max_side) / max(h0, w0)) if max_side else 1.0
    if scale < 1.0:
        w, h = max(MIN_SIDE, int(round(w0 * scale))), max(MIN_SIDE, int(round(h0 * scale)))
        bgr = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA)
    h, w = bgr.shape[:2]
    _, rows = _detector(w, h, float(score_threshold)).detect(bgr)
    if rows is None:
        return []
    out = []
    for r in rows:
        score = float(r[14])
        if score < score_threshold:
            continue
        x, y, fw, fh = (float(v) for v in r[:4])
        x0, y0 = max(0.0, x), max(0.0, y)
        x1, y1 = min(float(w), x + fw), min(float(h), y + fh)
        if x1 - x0 < MIN_BOX_PX or y1 - y0 < MIN_BOX_PX:
            continue
        lms = [(min(1.0, max(0.0, float(r[4 + 2 * i]) / w)),
                min(1.0, max(0.0, float(r[5 + 2 * i]) / h))) for i in range(5)]
        out.append(Face([(x0 + x1) / 2 / w, (y0 + y1) / 2 / h, (x1 - x0) / w, (y1 - y0) / h, score],
                        lms))
    out.sort(key=lambda f: (f[3], f[4]), reverse=True)
    return out


def largest(faces: list):
    """The tallest face, or None."""
    return max(faces, key=lambda f: (f[3], f[4])) if faces else None
