import warnings

from .base import Predictor
from .irt import IRT
from .irt_h import IRTH
from .irt_item import IRTItem, IRTItemLR
from .irt_llm import IRTItemLRLLM, IRTItemMixLLM
from .irt_lr import IRTLR
from .irt_mix import IRTItemMix
from .mirt import MIRT
from .ref_caimira import RefCaimira
from .ref_lex import RefLex
from .simple import BetaShrink, Constant, EmpiricalMean
from .stacked import StackedIRT
from .starter import Starter

PREDICTORS = {cls.name: cls for cls in (Constant, EmpiricalMean, BetaShrink, IRT, IRTH, IRTItem, IRTItemLR, IRTItemMix,
                                        IRTItemLRLLM, IRTItemMixLLM, IRTLR, MIRT, RefCaimira, RefLex, StackedIRT, Starter)}


def build_predictor(cfg: dict) -> Predictor:
    """Drop parameters that belong to *other* predictors (so a config can switch `name` over a base
    that sets e.g. `item_target`); unknown keys still raise."""
    cfg = dict(cfg)
    cls = PREDICTORS[cfg.pop("name")]
    others = {k for c in PREDICTORS.values() if c is not cls for c_ in c.__mro__ for k in vars(c_)
              if not k.startswith("_")}
    keep = {k: v for k, v in cfg.items() if hasattr(cls, k) or k not in others}
    if len(keep) < len(cfg):
        warnings.warn(f"{cls.name}: ignoring parameters of other predictors: {sorted(set(cfg) - set(keep))}", stacklevel=2)
    return cls(**keep)
