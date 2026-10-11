"""Download the MobileNet-SSD model files (only needed if models/ is empty)."""
import sys
import urllib.request
from pathlib import Path

BASE = "https://raw.githubusercontent.com/chuanqi305/MobileNet-SSD/master"
FILES = {
    "MobileNetSSD_deploy.prototxt": f"{BASE}/deploy.prototxt",
    "MobileNetSSD_deploy.caffemodel": f"{BASE}/mobilenet_iter_73000.caffemodel",
}
dest = Path(__file__).resolve().parent.parent / "models"
dest.mkdir(exist_ok=True)

for name, url in FILES.items():
    target = dest / name
    if target.exists() and target.stat().st_size > 1000:
        print(f"[ok] {name} already present")
        continue
    print(f"[..] downloading {name}")
    try:
        urllib.request.urlretrieve(url, target)
        print(f"[ok] saved {target} ({target.stat().st_size / 1e6:.1f} MB)")
    except Exception as exc:
        print(f"[!!] failed: {exc}\n     Download manually from {url} and save as {target}")
        sys.exit(1)
