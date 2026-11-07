"""Codabench entry point. Built by scripts/build_submission.py; do not edit the copy in dist/.

predict(input, labeled) -> float in [0, 1]
  input   = [subject: dict[str, str], item: dict[str, str]]
  labeled = [[[subject, item], 0 | 1], ...]   (empty at budget 0)
"""
import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from paiec.runtime import load_predictor, safe_predict

_PREDICTOR = load_predictor(_ROOT)


def predict(input, labeled):
    return safe_predict(_PREDICTOR, input, labeled or [])
