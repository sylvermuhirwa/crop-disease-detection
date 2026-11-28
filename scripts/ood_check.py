"""Measure how 'strange' a photo is compared with real cassava photos.

Distance = Mahalanobis distance to the nearest disease-class average, in the
model's 1,280-number fingerprint space. The limit is set on the VALIDATION
photos only. The test set is not used.
"""
import sys
from pathlib import Path

import numpy as np
import timm
import torch
from PIL import Image

sys.path.insert(0, "app")
import engine

# 1. Load the fingerprints made on Kaggle
data = np.load("results/features.npz")
train, train_labels = data["train"].astype(np.float64), data["train_labels"]
val = data["val"].astype(np.float64)

# 2. Typical fingerprint of each class, and how much fingerprints normally vary
means = np.stack([train[train_labels == c].mean(0) for c in range(5)])
centered = train - means[train_labels]
S = centered.T @ centered / len(train)
shrink = 0.1                                   # stabilises the estimate (1,280 numbers, 3,000 photos)
S = (1 - shrink) * S + shrink * np.trace(S) / S.shape[0] * np.eye(S.shape[0])
P = np.linalg.inv(S)


def distance(feats):
    """Distance of each fingerprint from the nearest class average."""
    feats = np.atleast_2d(feats).astype(np.float64)
    d = [np.einsum("ij,jk,ik->i", feats - m, P, feats - m) for m in means]
    return np.min(d, axis=0)


# 3. Real cassava photos (validation set): where do their distances lie?
val_d = distance(val)
limits = {q: float(np.percentile(val_d, q)) for q in (95, 99, 99.5)}
print("Validation photos (real cassava), distance percentiles:")
print(f"  median {np.median(val_d):.0f} | 95th {limits[95]:.0f} | 99th {limits[99]:.0f} | 99.5th {limits[99.5]:.0f} | max {val_d.max():.0f}")
LIMIT = limits[99]
print(f"\nLimit used below = 99th percentile = {LIMIT:.0f}  (about 1 in 100 real cassava photos would be rejected)\n")

# 4. The model, to get the fingerprint of your own photos
model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=5)
model.load_state_dict(torch.load(engine.WEIGHTS_PATH, map_location="cpu", weights_only=True))
model.eval()


def fingerprint(img):
    _, x = engine.preprocess(engine.prepare_image(img))
    with torch.no_grad():
        return model.forward_head(model.forward_features(torch.from_numpy(x)), pre_logits=True).numpy()[0]


# 5. Your photos
EXT = {".jpg", ".jpeg", ".png"}
groups = {
    "not_cassava": sorted(Path("my_photos/not_cassava").glob("*")),
    "real cassava (demo, validation photos)": sorted(Path("demo_photos").glob("[0-9]*.jpg")),
}
caught = {}
for name, files in groups.items():
    files = [f for f in files if f.suffix.lower() in EXT]
    print(f"=== {name}: {len(files)} photos ===")
    print(f"{'photo':30s} {'distance':>9}   decision")
    rejected = 0
    for f in files:
        d = float(distance(fingerprint(Image.open(f)))[0])
        reject = d > LIMIT
        rejected += reject
        print(f"{f.name[:30]:30s} {d:>9.0f}   {'REJECT (not cassava)' if reject else 'accept'}")
    caught[name] = (rejected, len(files))
    print()

for name, (r, n) in caught.items():
    print(f"{name}: rejected {r} of {n}")