"""Codabench acquisition hook. Built by scripts/build_submission.py (omitted for official_random).

acquisition_function(input, prediction, labeled, context) -> bool  (True = reveal this label)
"""
import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from paiec.runtime import load_policy, load_predictor

_POLICY = load_policy(_ROOT).bind(load_predictor(_ROOT))


def acquisition_function(input, prediction, labeled, context):
    try:
        return bool(_POLICY(input, prediction, labeled, context))
    except Exception:  # noqa: BLE001 - never fail the run; fall back to accept-while-budget-remains
        return context.get("labels_remaining", 0) > 0
