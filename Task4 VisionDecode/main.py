#!/usr/bin/env python3
"""VisionDecode - command line interface (object detection with MobileNet-SSD).

Examples
  python main.py detect --image samples/dog_bike_car.jpg
  python main.py detect --image "C:/photos/street.jpg" --threshold 0.9
  python main.py detect --image my_folder/        # batch a whole folder
  python main.py demo                             # run all bundled samples
"""
import argparse
import sys

from vision import config
from vision.detector import ModelMissing
from vision.pipeline import format_checks, list_images, process_detection


def run_detect_cmd(args) -> bool:
    ok = True
    for path in list_images(args.image):
        result, _, checks = process_detection(path, args.out, threshold=args.threshold)
        print(f"\n=== DETECT: {path.name}  ({result.image_size[0]}x{result.image_size[1]}) ===")
        for d in result.detections:
            print(f"  {d.label:<12} {d.confidence:6.1%}   box(x={d.x}, y={d.y}, w={d.w}, h={d.h})")
        if not result.detections:
            print("  (nothing above the confidence gate)")
        print(format_checks(checks))
        ok &= all(c.passed for c in checks)
    return ok


def run_demo(args) -> bool:
    ok = True
    for name in ("dog_bike_car.jpg", "person_dog.jpg", "people_football.jpg"):
        a = argparse.Namespace(**{**vars(args), "image": str(config.SAMPLES_DIR / name)})
        ok &= run_detect_cmd(a)
    print(f"\nAll outputs saved in: {args.out}")
    return ok


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="VisionDecode: object detection")
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp):
        sp.add_argument("--threshold", type=float, default=config.MIN_CONFIDENCE,
                        help="confidence gate, 0.80-1.0 (default 0.80 = project minimum)")
        sp.add_argument("--out", default=str(config.OUTPUT_DIR), help="output folder")

    d = sub.add_parser("detect", help="find objects in an image or folder")
    d.add_argument("--image", required=True, help="image file or folder")
    common(d)
    d.set_defaults(func=run_detect_cmd)

    m = sub.add_parser("demo", help="run all bundled sample images")
    common(m)
    m.set_defaults(func=run_demo)
    return p


def main() -> int:
    args = build_parser().parse_args()
    if args.threshold < config.MIN_CONFIDENCE:
        print(f"[!] Threshold {args.threshold:.0%} is below the 80% project minimum; using 80%.")
        args.threshold = config.MIN_CONFIDENCE
    try:
        ok = args.func(args)
    except (ModelMissing, FileNotFoundError) as exc:
        print(f"\n[ERROR] {exc}")
        return 2
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
