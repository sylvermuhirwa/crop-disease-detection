"""Run the app engine on your own photos and print what the app would say.

Folders inside my_photos/:
  not_cassava/  other leaves, plants and objects
  cassava/      your own phone photos of cassava leaves
  quality/      blurry or dark photos
"""
import sys
from pathlib import Path
from PIL import Image

sys.path.insert(0, "app")
import engine

ROOT = Path("my_photos")
EXTENSIONS = {".jpg", ".jpeg", ".png"}

for group in ["not_cassava", "cassava", "quality"]:
    files = sorted(p for p in (ROOT / group).glob("*") if p.suffix.lower() in EXTENSIONS)
    print(f"\n=== {group}: {len(files)} photos ===")
    if not files:
        continue
    print(f"{'photo':30s} {'status':18s} {'label':8s} confidence")
    counts = {}
    for f in files:
        try:
            result = engine.analyze(Image.open(f))
        except Exception as error:
            print(f"{f.name[:30]:30s} could not open ({error})")
            continue
        counts[result["status"]] = counts.get(result["status"], 0) + 1
        label = result.get("label", "-")
        conf = f"{result['confidence']:.2f}" if "confidence" in result else "-"
        print(f"{f.name[:30]:30s} {result['status']:18s} {label:8s} {conf}")
    print("Summary:", counts)