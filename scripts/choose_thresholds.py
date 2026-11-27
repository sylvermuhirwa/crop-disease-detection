"""Try different 'not sure' and 'Healthy' thresholds on the VALIDATION set (Run 2, calibrated).

For each pair we report what the app would do:
  confident   = % of photos where the app gives a confident answer
  accuracy    = % correct among those confident answers
  wrong/100   = wrong confident answers per 100 photos
  sick->Hlth  = % of sick leaves that get a confident 'Healthy' (the dangerous mistake)
The test set is NOT used here.
"""
import json
import numpy as np

data = np.load("results/eval_data.npz")
y = data["val_labels"]
T = json.load(open("results/calibration.json"))["Run 2"]

logp = np.log(np.clip(data["val_run2"], 1e-9, 1)) / T
logp -= logp.max(1, keepdims=True)
p = np.exp(logp)
p /= p.sum(1, keepdims=True)
pred, conf = p.argmax(1), p.max(1)
diseased = y != 4

print(f"{'not sure below':>15} {'Healthy needs':>14} {'confident':>10} {'accuracy':>9} {'wrong/100':>10} {'sick->Hlth':>11}")
for unsure in [0.40, 0.50, 0.60, 0.70]:
    for healthy_min in [0.70, 0.80, 0.90]:
        needed = np.where(pred == 4, np.maximum(unsure, healthy_min), unsure)
        confident = conf >= needed
        accuracy = (pred[confident] == y[confident]).mean() * 100
        wrong = (confident & (pred != y)).mean() * 100
        sick_healthy = (confident & (pred == 4) & diseased).sum() / diseased.sum() * 100
        print(f"{unsure:>15.2f} {healthy_min:>14.2f} {confident.mean() * 100:>9.1f}% "
              f"{accuracy:>8.1f}% {wrong:>10.1f} {sick_healthy:>10.1f}%")