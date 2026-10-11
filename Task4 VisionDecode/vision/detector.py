"""Path 2 - Object detection with cv2.dnn + MobileNet-SSD (slides 13-16)."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

import cv2
import numpy as np

from . import config


class ModelMissing(FileNotFoundError):
    """Raised when the MobileNet-SSD files are not on disk."""


@dataclass
class Detection:
    label: str
    confidence: float      # 0.0 - 1.0
    x: int                 # top-left origin
    y: int
    w: int
    h: int


@dataclass
class DetectionResult:
    detections: list[Detection]      # only those >= threshold
    dropped: int                     # candidates removed by the confidence gate
    threshold: float
    image_size: tuple                # (width, height)

    @property
    def passed(self) -> bool:
        return bool(self.detections) and all(d.confidence >= self.threshold for d in self.detections)

    @property
    def mean_confidence(self) -> float:
        return float(np.mean([d.confidence for d in self.detections])) if self.detections else 0.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["passed"] = self.passed
        d["mean_confidence"] = self.mean_confidence
        return d


class MobileNetSSDDetector:
    def __init__(self, prototxt: Path = config.PROTOTXT, caffemodel: Path = config.CAFFEMODEL):
        for p in (prototxt, caffemodel):
            if not Path(p).exists():
                raise ModelMissing(
                    f"Missing model file: {p}\nRun:  python scripts/download_models.py")
        if not hasattr(cv2.dnn, "readNetFromCaffe"):
            raise ModelMissing(
                f"Your OpenCV ({cv2.__version__}) has no Caffe model loader (removed in OpenCV 5).\n"
                "Fix: pip uninstall -y opencv-python opencv-contrib-python opencv-python-headless\n"
                '     pip install "opencv-python<5"')
        self.net = cv2.dnn.readNetFromCaffe(str(prototxt), str(caffemodel))

    def detect(self, image: np.ndarray, threshold: float = config.MIN_CONFIDENCE) -> DetectionResult:
        h, w = image.shape[:2]
        # Step 1 - blob construction: resize to 300x300 + mean subtraction + scaling
        blob = cv2.dnn.blobFromImage(cv2.resize(image, config.INPUT_SIZE),
                                     config.SCALE_FACTOR, config.INPUT_SIZE,
                                     config.MEAN_SUBTRACTION)
        self.net.setInput(blob)
        out = self.net.forward()                       # shape (1, 1, N, 7)

        kept, dropped = [], 0
        for i in range(out.shape[2]):
            conf = float(out[0, 0, i, 2])
            if conf < 0.05:                             # ignore pure noise entirely
                continue
            if conf < threshold:                        # the 80% gate
                dropped += 1
                continue
            cls = int(out[0, 0, i, 1])
            # Normalised coords -> real pixels (slide 14)
            x1, y1, x2, y2 = (out[0, 0, i, 3:7] * np.array([w, h, w, h])).astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)
            kept.append(Detection(config.CLASSES[cls], conf, int(x1), int(y1),
                                  int(x2 - x1), int(y2 - y1)))
        kept.sort(key=lambda d: d.confidence, reverse=True)
        return DetectionResult(kept, dropped, threshold, (w, h))


def _color_for(label: str) -> tuple:
    """A distinct, saturated BGR colour per class."""
    idx = config.CLASSES.index(label) if label in config.CLASSES else 0
    hsv = np.uint8([[[int(idx * 37) % 180, 230, 200]]])
    return tuple(int(c) for c in cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0])


def draw_detections(image: np.ndarray, result: DetectionResult) -> np.ndarray:
    """Visual confirmation: bounding boxes with 'label: NN%' tags."""
    canvas = image.copy()
    thick = max(2, int(round(min(canvas.shape[:2]) / 250)))
    scale = max(0.5, min(canvas.shape[:2]) / 900)
    for d in result.detections:
        color = _color_for(d.label)
        cv2.rectangle(canvas, (d.x, d.y), (d.x + d.w, d.y + d.h), color, thick)
        tag = f"{d.label}: {d.confidence * 100:.0f}%"
        (tw, th), base = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        top = max(th + base + 2, d.y)
        cv2.rectangle(canvas, (d.x, top - th - base - 2), (d.x + tw + 6, top), color, -1)
        cv2.putText(canvas, tag, (d.x + 3, top - base), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas