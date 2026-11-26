"""Pick demo photos from the VALIDATION set (the test set is not used).

Writes results/kaggle_cell.txt: a ready-to-paste Kaggle cell that zips the photos.
Writes results/demo_samples.json: what each photo is and what the model says.
"""
import json
import numpy as np

HEALTHY = 4
NAMES = ["CBB", "CBSD", "CGM", "CMD", "Healthy"]

data = np.load("results/eval_data.npz")
ids, y = data["val_ids"], data["val_labels"]
T = json.load(open("results/calibration.json"))["Run 2"]

# Run 2 probabilities, calibrated with its temperature
logp = np.log(np.clip(data["val_run2"], 1e-9, 1)) / T
logp -= logp.max(1, keepdims=True)
p = np.exp(logp)
p /= p.sum(1, keepdims=True)
pred, conf = p.argmax(1), p.max(1)

rng = np.random.default_rng(0)
chosen = []
for c in range(5):                                   # 2 correct photos per class
    idx = np.where((y == c) & (pred == c))[0]
    chosen += list(rng.choice(idx, 2, replace=False))
sick_called_healthy = np.where((y != HEALTHY) & (pred == HEALTHY))[0]
healthy_called_sick = np.where((y == HEALTHY) & (pred != HEALTHY))[0]
chosen += list(rng.choice(sick_called_healthy, 2, replace=False))
chosen += list(rng.choice(healthy_called_sick, 2, replace=False))

rows = []
print(f"{'photo':20s} {'true':8s} {'model says':10s} confidence")
for i in chosen:
    rows.append({"image_id": str(ids[i]), "true": NAMES[y[i]],
                 "predicted": NAMES[pred[i]], "confidence": round(float(conf[i]), 3)})
    print(f"{ids[i]:20s} {NAMES[y[i]]:8s} {NAMES[pred[i]]:10s} {conf[i]:.2f}")

json.dump(rows, open("results/demo_samples.json", "w"), indent=2)

kaggle_code = f'''import os, shutil, zipfile
from IPython.display import FileLink, display

SRC = "/kaggle/input/competitions/cassava-leaf-disease-classification/train_images"
ids = {[r["image_id"] for r in rows]!r}

os.makedirs("/kaggle/working/demo_photos", exist_ok=True)
for name in ids:
    shutil.copy(f"{{SRC}}/{{name}}", "/kaggle/working/demo_photos/")
os.chdir("/kaggle/working")
with zipfile.ZipFile("demo_photos.zip", "w") as z:
    for name in ids:
        z.write(f"demo_photos/{{name}}", name)
print("Photos zipped:", len(ids))
display(FileLink("demo_photos.zip"))
'''
open("results/kaggle_cell.txt", "w").write(kaggle_code)
print("\\nWrote results/demo_samples.json and results/kaggle_cell.txt")