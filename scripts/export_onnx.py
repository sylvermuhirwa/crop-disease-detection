"""Export the chosen model (Run 2, EfficientNet-B0) to ONNX and check it.

Checks that ONNX Runtime gives the same answers as PyTorch, then measures
file size and CPU speed. The model is trained on 256x256 photos.
"""
import time
import numpy as np
import torch
import timm
import onnxruntime as ort

WEIGHTS = "models/run2_longer_best.pt"
ONNX_PATH = "models/run2_efficientnet_b0.onnx"
IMG_SIZE = 256

# 1. Load the trained model (same shape as in training: 5 outputs)
model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=5)
model.load_state_dict(torch.load(WEIGHTS, map_location="cpu", weights_only=True))
model.eval()

# 2. Export to ONNX. 'batch' may change size, so one or many photos both work.
dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
export_args = dict(
    input_names=["image"],
    output_names=["logits"],
    dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
)
try:
    torch.onnx.export(model, dummy, ONNX_PATH, dynamo=False, **export_args)
    print("Exported with the classic exporter.")
except Exception as error:
    print("Classic exporter failed:", str(error)[:200])
    print("Trying the default exporter...")
    torch.onnx.export(model, dummy, ONNX_PATH, **export_args)
    print("Exported with the default exporter.")

# 3. Check: do PyTorch and ONNX Runtime agree?
session = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
x = torch.randn(4, 3, IMG_SIZE, IMG_SIZE)
with torch.no_grad():
    torch_out = model(x).numpy()
onnx_out = session.run(None, {"image": x.numpy()})[0]
max_diff = float(np.abs(torch_out - onnx_out).max())
same_answers = (torch_out.argmax(1) == onnx_out.argmax(1)).all()
print(f"Largest difference in raw scores: {max_diff:.6f}")
print("Same predicted classes:", bool(same_answers))

# 4. File size and CPU speed (one photo at a time, like the app)
import os
size_mb = os.path.getsize(ONNX_PATH) / 1e6
one = np.random.randn(1, 3, IMG_SIZE, IMG_SIZE).astype(np.float32)
for _ in range(5):                       # warm-up runs, not counted
    session.run(None, {"image": one})
times = []
for _ in range(50):
    start = time.perf_counter()
    session.run(None, {"image": one})
    times.append((time.perf_counter() - start) * 1000)
print(f"ONNX file size: {size_mb:.1f} MB")
print(f"CPU time per photo: median {np.median(times):.0f} ms, slowest {max(times):.0f} ms")