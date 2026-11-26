"""Design the 'retake the photo' rule using the VALIDATION set only.

For each model we ask: if the app refuses to answer when its confidence is
below a threshold, how many photos does it refuse, how accurate are the
answers it does give, and how many sick leaves are still called Healthy?
The test set is NOT used here.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")                      # draw to a file, no window needed
import matplotlib.pyplot as plt

HEALTHY = 4
data = np.load("results/eval_data.npz")
y = data["val_labels"]
models = {
    "Run 2": data["val_run2"],
    "Run 3": data["val_run3"],
    "Average": (data["val_run2"] + data["val_run3"]) / 2,
}
diseased = y != HEALTHY


def apply_rule(probs, t, healthy_t=0.0):
    """Accept an answer only if confidence >= t (and >= healthy_t for 'Healthy')."""
    pred = probs.argmax(1)
    conf = probs.max(1)
    needed = np.where(pred == HEALTHY, max(t, healthy_t), t)
    accepted = conf >= needed
    coverage = accepted.mean() * 100
    accuracy = (pred[accepted] == y[accepted]).mean() * 100
    sick_called_healthy = (accepted & (pred == HEALTHY) & diseased).sum() / diseased.sum() * 100
    sick_rejected = (~accepted & diseased).sum() / diseased.sum() * 100
    return coverage, accuracy, sick_called_healthy, sick_rejected


print("t = confidence needed to answer | healthy_t = extra-strict level for the answer 'Healthy'")
print("answered = % of photos the app answers | accuracy = % correct among answered")
print("sick->Healthy = % of sick leaves wrongly called Healthy | sick retake = % of sick leaves sent to 'retake'\n")

for name, probs in models.items():
    print(name)
    print(f"{'t':>5} {'healthy_t':>10} {'answered':>9} {'accuracy':>9} {'sick->Healthy':>14} {'sick retake':>12}")
    for t in [0.0, 0.5, 0.6, 0.7, 0.8, 0.9]:
        for healthy_t in [0.0, 0.9]:
            cov, acc, bad, rej = apply_rule(probs, t, healthy_t)
            print(f"{t:>5.2f} {healthy_t:>10.2f} {cov:>8.1f}% {acc:>8.1f}% {bad:>13.1f}% {rej:>11.1f}%")
    print()

# Plot: accuracy of the answers we give vs how many photos we answer
plt.figure(figsize=(7, 5))
for name, probs in models.items():
    xs, ys = [], []
    for t in np.arange(0.30, 0.96, 0.01):
        cov, acc, _, _ = apply_rule(probs, t)
        xs.append(cov)
        ys.append(acc)
    plt.plot(xs, ys, label=name)
plt.xlabel("Photos answered (%)")
plt.ylabel("Accuracy of answers given (%)")
plt.title("Accuracy vs coverage (validation set)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("assets/coverage_vs_accuracy.png", dpi=150)
print("Saved plot: assets/coverage_vs_accuracy.png")