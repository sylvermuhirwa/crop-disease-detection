"""Calibrate each model with temperature scaling, using the VALIDATION set only.

A calibrated model that says '90% sure' is right about 90% of the time.
The test set is NOT used here.
"""
import json
import numpy as np

HEALTHY = 4
data = np.load("results/eval_data.npz")
y = data["val_labels"]
models = {
    "Run 2": data["val_run2"],
    "Run 3": data["val_run3"],
    "Average": (data["val_run2"] + data["val_run3"]) / 2,
}
diseased = y != HEALTHY


def rescale(probs, T):
    """Apply temperature T to probabilities (same effect as scaling the model's raw scores)."""
    logp = np.log(np.clip(probs, 1e-9, 1)) / T
    logp -= logp.max(1, keepdims=True)
    p = np.exp(logp)
    return p / p.sum(1, keepdims=True)


def nll(probs):
    """Average 'surprise' at the true answer. Lower is better."""
    return -np.log(np.clip(probs[np.arange(len(y)), y], 1e-9, 1)).mean()


def ece(probs, bins=10):
    """Average gap (in %) between claimed confidence and real accuracy."""
    conf = probs.max(1)
    correct = probs.argmax(1) == y
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return total * 100


def apply_rule(probs, t, healthy_t):
    """Answer only if confidence >= t (and >= healthy_t when the answer is 'Healthy')."""
    pred = probs.argmax(1)
    conf = probs.max(1)
    needed = np.where(pred == HEALTHY, max(t, healthy_t), t)
    accepted = conf >= needed
    answered = accepted.mean() * 100
    accuracy = (pred[accepted] == y[accepted]).mean() * 100
    bad = (accepted & (pred == HEALTHY) & diseased).sum() / diseased.sum() * 100
    retake = (~accepted & diseased).sum() / diseased.sum() * 100
    return answered, accuracy, bad, retake


# 1. Find the best temperature for each model (lowest NLL on validation)
temps, calibrated = {}, {}
print(f"{'Model':8s} {'best T':>7} {'ECE before':>11} {'ECE after':>10} {'NLL before':>11} {'NLL after':>10}")
for name, probs in models.items():
    grid = np.arange(0.3, 3.01, 0.05)
    best_T = min(grid, key=lambda T: nll(rescale(probs, T)))
    calibrated[name] = rescale(probs, best_T)
    temps[name] = round(float(best_T), 2)
    print(f"{name:8s} {best_T:>7.2f} {ece(probs):>10.1f}% {ece(calibrated[name]):>9.1f}% "
          f"{nll(probs):>11.3f} {nll(calibrated[name]):>10.3f}")

# 2. Compare the 'retake the photo' rule on calibrated confidences
print("\nAfter calibration (validation set)")
print("healthy_t = confidence needed before the app may answer 'Healthy'\n")
for name, probs in calibrated.items():
    print(name)
    print(f"{'t':>5} {'healthy_t':>10} {'answered':>9} {'accuracy':>9} {'sick->Healthy':>14} {'sick retake':>12}")
    for t in [0.0, 0.5]:
        for healthy_t in [0.0, 0.7, 0.8, 0.9, 0.95]:
            a, acc, bad, rej = apply_rule(probs, t, healthy_t)
            print(f"{t:>5.2f} {healthy_t:>10.2f} {a:>8.1f}% {acc:>8.1f}% {bad:>13.1f}% {rej:>11.1f}%")
    print()

# 3. Save the temperatures for the app
with open("results/calibration.json", "w") as f:
    json.dump(temps, f, indent=2)
print("Saved temperatures to results/calibration.json:", temps)