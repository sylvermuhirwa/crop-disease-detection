"""Grad-CAM demo: show where the model looked on the 14 demo photos."""
import json
import numpy as np
import torch
import timm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

NAMES = ["CBB", "CBSD", "CGM", "CMD", "Healthy"]
IMG_SIZE = 256
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
T = json.load(open("results/calibration.json"))["Run 2"]     # confidence calibration

model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=5)
model.load_state_dict(torch.load("models/run2_longer_best.pt", map_location="cpu", weights_only=True))
model.eval()


def preprocess(img):
    """Resize to 256x256 and normalise, exactly as in training."""
    small = img.convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.Resampling.BILINEAR)
    arr = np.asarray(small, dtype=np.float32) / 255.0
    x = (arr - MEAN) / STD
    return arr, torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0)


def explain(img):
    """Return the photo, the heatmap overlay, the predicted class and calibrated probabilities."""
    arr, x = preprocess(img)
    feats = model.forward_features(x)                  # [1, 1280, 8, 8] feature maps
    logits = model.forward_head(feats)                 # the 5 raw scores
    probs = torch.softmax(logits / T, dim=1)[0].detach().numpy()
    cls = int(probs.argmax())
    grads = torch.autograd.grad(logits[0, cls], feats)[0]        # how much each map pushes the answer
    weights = grads.mean(dim=(2, 3), keepdim=True)               # one weight per feature map
    cam = torch.relu((weights * feats).sum(dim=1, keepdim=True)) # weighted sum, keep positives only
    cam = torch.nn.functional.interpolate(cam, size=(IMG_SIZE, IMG_SIZE),
                                          mode="bilinear", align_corners=False)[0, 0].detach().numpy()
    cam = cam / cam.max() if cam.max() > 0 else cam             # scale to 0..1
    heat = matplotlib.colormaps["jet"](cam)[..., :3]            # colour it (red = important)
    overlay = np.clip(0.55 * arr + 0.45 * heat, 0, 1)
    return arr, overlay, cls, probs


if __name__ == "__main__":
    rows = json.load(open("results/demo_samples.json"))
    fig, axes = plt.subplots(7, 4, figsize=(12, 21))
    agree = 0
    for k, r in enumerate(rows):
        img = Image.open(f"demo_photos/{r['image_id']}")
        arr, overlay, cls, probs = explain(img)
        agree += NAMES[cls] == r["predicted"]
        row, col = divmod(k, 2)
        left, right = axes[row, col * 2], axes[row, col * 2 + 1]
        left.imshow(arr)
        left.set_title(f"True: {r['true']}", fontsize=9)
        right.imshow(overlay)
        right.set_title(f"Model: {NAMES[cls]} {probs[cls]:.0%}", fontsize=9)
        left.axis("off")
        right.axis("off")
        print(f"{r['image_id']:18s} true {r['true']:8s} model {NAMES[cls]:8s} {probs[cls]:.2f}")
    plt.tight_layout()
    plt.savefig("assets/gradcam_gallery.png", dpi=100)
    print(f"\nPredictions matching the earlier table: {agree}/{len(rows)}")
    print("Saved assets/gradcam_gallery.png")