"""Image-quality check: is the photo sharp enough and well lit?

The app calls check_quality() before running the model. Thresholds are
PROVISIONAL: chosen from 14 demo photos and must be re-checked with real
phone photos. Run this file directly (python app/quality.py) for a self-test.
"""
import glob
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance

MAX_SIDE = 512          # photos are shrunk to this size first, so cameras with
                        # different resolutions are measured fairly
BLUR_BELOW = 200        # sharpness under this = blurry
DARK_BELOW = 50         # brightness under this = too dark
BRIGHT_ABOVE = 220      # brightness over this = too bright


def to_gray(img):
    """Shrink the photo and convert it to grey levels (0-255)."""
    img = img.convert("L")
    img.thumbnail((MAX_SIDE, MAX_SIDE), Image.Resampling.LANCZOS)
    return np.asarray(img, dtype=np.float32)


def sharpness(gray):
    """Variance of the Laplacian: high = sharp edges, low = blurry."""
    lap = (-4 * gray[1:-1, 1:-1] + gray[:-2, 1:-1] + gray[2:, 1:-1]
           + gray[1:-1, :-2] + gray[1:-1, 2:])
    return float(lap.var())


def check_quality(img):
    """Return (ok, problem, sharp, bright). problem is None, 'too_dark', 'too_bright' or 'blurry'.

    Brightness is checked first, because a dark photo also scores low on sharpness.
    """
    gray = to_gray(img)
    sharp, bright = sharpness(gray), float(gray.mean())
    if bright < DARK_BELOW:
        return False, "too_dark", sharp, bright
    if bright > BRIGHT_ABOVE:
        return False, "too_bright", sharp, bright
    if sharp < BLUR_BELOW:
        return False, "blurry", sharp, bright
    return True, None, sharp, bright


if __name__ == "__main__":
    files = sorted(glob.glob("demo_photos/*.jpg"))
    conditions = {
        "original":      lambda im: im,
        "blur sigma 1":  lambda im: im.filter(ImageFilter.GaussianBlur(1)),
        "blur sigma 2":  lambda im: im.filter(ImageFilter.GaussianBlur(2)),
        "blur sigma 3":  lambda im: im.filter(ImageFilter.GaussianBlur(3)),
        "blur sigma 5":  lambda im: im.filter(ImageFilter.GaussianBlur(5)),
        "darker x0.3":   lambda im: ImageEnhance.Brightness(im).enhance(0.3),
        "brighter x1.8": lambda im: ImageEnhance.Brightness(im).enhance(1.8),
    }
    print(f"Self-test on {len(files)} demo photos "
          f"(blur below {BLUR_BELOW}, dark below {DARK_BELOW}, bright above {BRIGHT_ABOVE})\n")
    print(f"{'condition':15s} {'flagged':>8}   reasons")
    for name, change in conditions.items():
        reasons = {"too_dark": 0, "too_bright": 0, "blurry": 0}
        for f in files:
            ok, problem, _, _ = check_quality(change(Image.open(f).convert("RGB")))
            if problem:
                reasons[problem] += 1
        flagged = sum(reasons.values())
        detail = ", ".join(f"{k} {v}" for k, v in reasons.items() if v)
        print(f"{name:15s} {flagged:>5}/{len(files)}   {detail}")