"""Generate scientific PNG/SVG trade-off plots from real scored reports."""
import argparse
import json
import os
from pathlib import Path
os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".cache" / "matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
p = argparse.ArgumentParser()
p.add_argument("reports", nargs="+")
p.add_argument("--out", default="artifacts/tradeoff")
a = p.parse_args()
reports = [json.loads(Path(r).read_text()) for r in a.reports]
if len({r["dataset_sha256"] for r in reports}) != 1:
    p.error("Compare reports from the same evaluation file")
fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
for report in reports:
    metrics = report["metrics"]
    x, y = metrics["finance_accuracy"]["value"], metrics["attack_success_rate"]["value"]
    if x is None or y is None:
        p.error("Each plot point needs finance and attack measurements")
    ax.scatter(100*x, 100*y, s=65)
    ax.annotate(report["condition"], (100*x, 100*y), xytext=(6,6), textcoords="offset points")
ax.set(xlabel="Finance accuracy (%) — higher is better", ylabel="Attack success (%) — lower is better",
       title="FinSafe: finance utility and adversarial safety", xlim=(0,100), ylim=(0,100))
ax.grid(alpha=.2)
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
fig.savefig(a.out + ".png", dpi=180)
fig.savefig(a.out + ".svg")
