"""VisionDecode - Streamlit web UI.   Run:  streamlit run app.py"""
import tempfile
from pathlib import Path

import cv2
import streamlit as st

from vision import config
from vision.detector import ModelMissing
from vision.pipeline import process_detection

st.set_page_config(page_title="VisionDecode", page_icon="👁️", layout="wide")


def show_image(img, caption=None):
    """Works on old and new Streamlit (use_container_width was replaced by width)."""
    try:
        st.image(img, caption=caption, width="stretch")
    except TypeError:
        st.image(img, caption=caption, use_container_width=True)


def show_table(rows):
    try:
        st.dataframe(rows, width="stretch")
    except TypeError:
        st.dataframe(rows, use_container_width=True)


def to_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


st.sidebar.title("👁️ VisionDecode")
st.sidebar.caption("Object detection · MobileNet-SSD")
threshold = st.sidebar.slider("Confidence gate", 0.80, 0.99, 0.80, 0.01,
                              help="80% is the project's absolute minimum standard.")
source = st.sidebar.radio("Input", ["Bundled sample", "Upload image"])

image_path = None
if source == "Bundled sample":
    samples = sorted(p.name for p in config.SAMPLES_DIR.iterdir()
                     if p.suffix.lower() in config.IMAGE_EXTENSIONS)
    image_path = config.SAMPLES_DIR / st.sidebar.selectbox("Sample", samples)
else:
    up = st.sidebar.file_uploader("Image", type=[e.strip(".") for e in config.IMAGE_EXTENSIONS])
    if up is not None:
        tmp = Path(tempfile.gettempdir()) / f"vd_upload_{up.name}"
        tmp.write_bytes(up.getbuffer())
        image_path = tmp

st.title("VisionDecode: Object Recognition")
st.caption("Pre-trained AI model -> measurable confidence -> clear visual output.")

if image_path is None:
    st.info("Pick a sample or upload an image in the sidebar.")
    st.stop()

out_dir = Path(tempfile.gettempdir()) / "visiondecode_out"
try:
    with st.spinner("Running MobileNet-SSD..."):
        result, annotated, checks = process_detection(image_path, out_dir, threshold=threshold)
    left, right = st.columns(2)
    with left:
        show_image(to_rgb(cv2.imread(str(image_path))), "Input")
    with right:
        show_image(to_rgb(annotated), "Detections")

    st.subheader("Detections")
    if result.detections:
        show_table([{"object": d.label, "confidence": f"{d.confidence:.1%}",
                     "x": d.x, "y": d.y, "w": d.w, "h": d.h} for d in result.detections])
    else:
        st.warning("Nothing met the confidence gate. This model only knows 20 object types "
                   "(person, car, dog, bicycle, bus, chair, bottle ...).")
    st.caption(f"{result.dropped} lower-confidence candidates were dropped by the gate.")
    ok, buf = cv2.imencode(".png", annotated)
    st.download_button("Download annotated image", buf.tobytes(), file_name=f"{image_path.stem}_detected.png")

    st.subheader("Gatekeeper validation")
    for col, c in zip(st.columns(4), checks):
        col.metric(c.name, "PASS" if c.passed else "FAIL")
        col.caption(c.detail)
except ModelMissing as exc:
    st.error(str(exc))
