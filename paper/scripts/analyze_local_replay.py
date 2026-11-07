"""Reconcile the three archived local package curves and execution receipts."""
from pathlib import Path
import json
import math

ROOT = Path(__file__).resolve().parents[1]
source = json.loads((ROOT / "data/local_replay_summary.json").read_text())
budgets = [0, 1, 3, 7, 15, 31]
weights = [.1, .2, .2, .2, .2, .1]
names = ["1033927_16-s44-rasch-offset", "1033935_18-s44-rasch-caimira", "1033965_27-s44-caimira-native"]
reconstructed = {}
for name in names:
    package = source["packages"][name]
    value = sum(w * package["brier_by_budget"][str(b)] for b, w in zip(budgets, weights))
    assert math.isclose(value, package["brier_alc"], abs_tol=1e-12)
    receipt = next(row for row in source["runs"] if row["run"] == name)
    assert math.isclose(value, receipt["panel_brier_alc"], abs_tol=1e-12)
    assert receipt["panel_sha256"] == source["panel_manifest"]["panel_sha256"]
    assert receipt["returncode"] == 0
    reconstructed[name] = value

manifest = json.loads((ROOT / "data/candidate_manifest.json").read_text())
blend_receipt = next(row for row in source["runs"] if row["run"] == names[1])
assert blend_receipt["package_sha256"] == manifest["sha256"]
r, blend, c = [reconstructed[name] for name in names]
result = {
    "source": "Archived local replay summaries and execution receipts",
    "reconstructed_brier_alc": reconstructed,
    "blend_minus_rasch": blend - r,
    "blend_minus_caimira": blend - c,
    "mean_component_loss_minus_blend": (r + c) / 2 - blend,
    "scope": "Recorded historical values; three source benchmarks covering 25 pairs share native tables with CAIMIRA training. Exact evaluated-row overlap and raw prediction replay remain unresolved.",
}
(ROOT / "data/local_replay_checks.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
