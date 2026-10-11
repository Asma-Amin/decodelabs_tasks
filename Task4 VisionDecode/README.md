# VisionDecode — Object Recognition with MobileNet-SSD

A Python pipeline that takes a raw image, runs a **pre-trained MobileNet-SSD** model through
OpenCV's `cv2.dnn`, and draws labelled bounding boxes — keeping only detections the model is
**at least 80% confident** about.

Built for **DecodeLabs AI Engineering Internship (Batch 2026) — Project 4: Image or Text
Recognition (Basic)**, using the *Object Detection* path.

<p align="center">
  <img src="docs/result_dog_bike_car.png" width="48%" alt="Detection result: dog, bicycle, car">
  <img src="docs/result_person_dog.png" width="48%" alt="Detection result: person, dog">
</p>

## How it works

| Step | What happens |
|---|---|
| 1. Input | The image is loaded as an H × W × 3 array of 0–255 values |
| 2. Blob | `cv2.dnn.blobFromImage` resizes to 300×300, subtracts the mean (127.5) and scales by 1/127.5 |
| 3. Inference | MobileNet-SSD predicts class, confidence and a normalised box for each candidate |
| 4. Decode | Normalised coordinates × the real image size give pixel boxes `(x, y, w, h)` |
| 5. Gate | `confidence >= 0.80` → draw box and label; otherwise drop the detection |
| 6. Output | Annotated image, console table and a JSON report |

The 80% gate is the project minimum: the CLI and the app can raise it, but never go below it.
After each run the program prints four **Gatekeeper** checks (library integration,
pre-processing, ≥ 80% confidence, visual confirmation) as PASS / FAIL.

## Quick start

Requires Python 3.9+.

```bash
git clone https://github.com/Asma-Amin/decodelabs_tasks.git
cd decodelabs_tasks/<Task4 VisionDecode>

python -m venv venv
venv\Scripts\activate          # Windows  (macOS/Linux: source venv/bin/activate)
pip install -r requirements.txt
```

> `requirements.txt` pins OpenCV below version 5 on purpose — OpenCV 5 removed the Caffe
> model loader this project relies on.

The model files are in `models/`. If they are missing, run `python scripts/download_models.py`.

## Usage

```bash
python main.py demo                                      # run the bundled sample images
python main.py detect --image samples/dog_bike_car.jpg   # one image
python main.py detect --image "C:\photos\street.jpg" --threshold 0.9
python main.py detect --image my_folder\                 # a whole folder
streamlit run app.py                                     # web interface
pytest -q                                                # tests
```

Example output:

```
=== DETECT: dog_bike_car.jpg  (768x576) ===
  bicycle       99.8%   box(x=81, y=132, w=490, h=290)
  car           99.4%   box(x=460, y=71, w=227, h=101)
  dog           96.7%   box(x=138, y=209, w=188, h=328)
  [PASS] 1. Library Integration: cv2.dnn loaded MobileNet-SSD without errors
  [PASS] 3. Accuracy Benchmarking: 3 detections, mean confidence 98.6% (minimum 80%), 0 dropped
  ...
```

Results are saved to `outputs/` as `<name>_detected.png` and `<name>_detected.json`.

## Project structure

```
main.py              command-line interface (detect / demo)
app.py               Streamlit web interface
vision/
  config.py          thresholds, model paths, class names
  detector.py        model loading, blob construction, decoding, confidence gate, drawing
  pipeline.py        run + save outputs + Gatekeeper checks
models/              MobileNet-SSD prototxt and caffemodel
samples/             test photos
scripts/             download_models.py
tests/               automated tests
docs/                result images used in this README
```

## Limitations

- The model knows only the **20 PASCAL VOC classes**: person, bicycle, car, motorbike,
  aeroplane, bus, train, boat, bird, cat, dog, horse, sheep, cow, bottle, chair, sofa,
  pottedplant, diningtable, tvmonitor. Anything else (phones, laptops, ...) is not detected.
- A high confidence gate avoids false positives but can drop real objects that are small,
  blurry or partly hidden.

## Credits

- Model: MobileNet-SSD Caffe weights by [chuanqi305](https://github.com/chuanqi305/MobileNet-SSD),
  trained on PASCAL VOC.
- Sample photos: OpenCV and Darknet sample data.
