"""Central configuration for the VisionDecode pipeline."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
SAMPLES_DIR = ROOT / "samples"
OUTPUT_DIR = ROOT / "outputs"

# --- Gatekeeper rule: 80% is the absolute minimum standard (slide 16) ---
MIN_CONFIDENCE = 0.80

# --- MobileNet-SSD (Caffe, PASCAL VOC, 300x300 input) ---
PROTOTXT = MODELS_DIR / "MobileNetSSD_deploy.prototxt"
CAFFEMODEL = MODELS_DIR / "MobileNetSSD_deploy.caffemodel"
INPUT_SIZE = (300, 300)
SCALE_FACTOR = 0.007843      # 1 / 127.5
MEAN_SUBTRACTION = 127.5

CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus",
    "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike",
    "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
