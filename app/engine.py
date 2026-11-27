"""Core of the app: look at one photo and decide what to tell the farmer.

Steps: quality check -> ONNX prediction -> calibration -> decision.
The heatmap (Grad-CAM) is computed only when asked for.

Thresholds are PROVISIONAL (chosen on the validation set, not yet checked on
real phone photos): UNSURE_BELOW and HEALTHY_MIN.
Run this file (python app/engine.py) for a self-test on the demo photos.
"""
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageOps

try:
    from quality import check_quality          # when run as: python app/engine.py
except ImportError:
    from app.quality import check_quality      # when imported as a package

ROOT = Path(__file__).resolve().parent.parent
ONNX_PATH = ROOT / "models" / "run2_efficientnet_b0.onnx"
WEIGHTS_PATH = ROOT / "models" / "run2_longer_best.pt"
CALIBRATION_PATH = ROOT / "results" / "calibration.json"

NAMES = ["CBB", "CBSD", "CGM", "CMD", "Healthy"]
IMG_SIZE = 256
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

UNSURE_BELOW = 0.60     # below this confidence, say "not sure"
HEALTHY_MIN = 0.80      # "Healthy" needs at least this confidence

TEMPERATURE = json.load(open(CALIBRATION_PATH))["Run 2"]    # fitted on validation data
_session = None
_torch_model = None


def prepare_image(img):
    """Fix phone rotation and make sure the photo is RGB."""
    return ImageOps.exif_transpose(img).convert("RGB")


def preprocess(img):
    """Resize to 256x256 and normalise exactly as in training.

    Returns the photo as numbers 0-1 (for display) and the model input.
    """
    small = img.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.BILINEAR)
    arr = np.asarray(small, dtype=np.float32) / 255.0
    x = ((arr - MEAN) / STD).transpose(2, 0, 1)[np.newaxis]
    return arr, x.astype(np.float32)


def get_session():
    """Load the ONNX model once, the first time it is needed."""
    global _session
    if _session is None:
        _session = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    return _session


def analyze(img, healthy_min=HEALTHY_MIN, unsure_below=UNSURE_BELOW):
    """Look at one photo and return a dictionary describing the result.

    status is one of: 'retake', 'uncertain', 'uncertain_healthy', 'confident'.
    """
    img = prepare_image(img)
    ok, problem, sharp, bright = check_quality(img)
    if not ok:
        return {"status": "retake", "problem": problem,
                "sharpness": sharp, "brightness": bright}

    _, x = preprocess(img)
    logits = get_session().run(None, {"image": x})[0][0]
    z = logits / TEMPERATURE                      # calibration
    z = z - z.max()
    probs = np.exp(z) / np.exp(z).sum()
    k = int(probs.argmax())
    confidence = float(probs[k])
    label = NAMES[k]

    if label == "Healthy" and confidence < healthy_min:
        status = "uncertain_healthy"
    elif confidence < unsure_below:
        status = "uncertain"
    else:
        status = "confident"
    return {"status": status, "label": label, "confidence": confidence,
            "probs": {n: float(p) for n, p in zip(NAMES, probs)},
            "sharpness": sharp, "brightness": bright}


def heatmap(img):
    """Grad-CAM: return the photo with red/yellow marking where the model looked."""
    import torch
    import timm
    import matplotlib
    global _torch_model
    if _torch_model is None:
        model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=5)
        model.load_state_dict(torch.load(WEIGHTS_PATH, map_location="cpu", weights_only=True))
        model.eval()
        _torch_model = model

    arr, x = preprocess(prepare_image(img))
    feats = _torch_model.forward_features(torch.from_numpy(x))     # [1, 1280, 8, 8]
    logits = _torch_model.forward_head(feats)
    cls = int(logits.argmax(1))
    grads = torch.autograd.grad(logits[0, cls], feats)[0]          # push of each map on the answer
    weights = grads.mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * feats).sum(dim=1, keepdim=True))
    cam = torch.nn.functional.interpolate(cam, size=(IMG_SIZE, IMG_SIZE),
                                          mode="bilinear", align_corners=False)[0, 0].detach().numpy()
    cam = cam / cam.max() if cam.max() > 0 else cam
    heat = matplotlib.colormaps["jet"](cam)[..., :3]
    return np.clip(0.55 * arr + 0.45 * heat, 0, 1)


if __name__ == "__main__":
    rows = json.load(open(ROOT / "results" / "demo_samples.json"))
    print(f"{'photo':18s} {'true':8s} {'status':18s} {'label':8s} confidence")
    counts = {}
    for r in rows:
        result = analyze(Image.open(ROOT / "demo_photos" / r["image_id"]))
        counts[result["status"]] = counts.get(result["status"], 0) + 1
        label = result.get("label", "-")
        conf = f"{result['confidence']:.2f}" if "confidence" in result else "-"
        print(f"{r['image_id']:18s} {r['true']:8s} {result['status']:18s} {label:8s} {conf}")
    print("\nSummary:", counts)
    first = Image.open(ROOT / "demo_photos" / rows[0]["image_id"])
    print("Heatmap test, output shape:", heatmap(first).shape)