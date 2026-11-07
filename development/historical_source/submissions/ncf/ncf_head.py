"""Single source of truth for the NCF head architecture.

Why this file exists
--------------------
The trainer (``notebooks/train_ncf.py``) and the submission entry point
(``submission/model.py``) must agree on the model architecture. Pickling
the full ``nn.Module`` (via ``torch.save(model, path)``) embeds a class
reference like ``__main__.NCFHead`` into the saved file — and that
reference cannot be resolved inside the Codabench sandbox container, where
``__main__`` is the platform runner. The unpickle fails silently, the
submission's ``predict()`` falls back to 0.5 for every input, and the
trained model is wasted.

The fix is to:

1. Save the ``state_dict`` (a plain mapping of parameter names → tensors,
   no class metadata) via ``torch.save(model.state_dict(), path)``.
2. Define the architecture in ONE place that both the trainer and the
   submission can import without side effects.
3. Load with ``torch.load(path, weights_only=True)`` and apply via
   ``model.load_state_dict(state)`` — this both blocks arbitrary pickle
   execution and surfaces architecture drift as a loud shape-mismatch
   error.

Both the trainer and the submission import :class:`NCFHead` from this
module. Bumping ``NCF_ARCH_VERSION`` is the canonical way to signal
incompatibility — it lands in the training metadata JSON, and a future
``model.py`` could refuse to load mismatched checkpoints.

The current shipped artifact is tied to this exact ``nn.Sequential``
state_dict layout (``net.0.*``, ``net.2.*``, ``net.4.*``). It is NOT
directly loadable into a ``torch_measure`` NCF wrapper without an explicit
converter or adapter; any torch_measure/IRT experiment belongs to the
post-deadline framework track, not the deadline-fit submission package.
"""

from __future__ import annotations

import torch
from torch import nn

EMBED_DIM = 768
NCF_ARCH_VERSION = 1


class NCFHead(nn.Module):
    """2-hidden-layer MLP over concatenated subject+item embeddings.

    Architecture (version 1):
        Input  : (B, 2 * EMBED_DIM)         e.g. (B, 1536) for all-mpnet-base-v2
        Linear : 2*EMBED_DIM -> 256
        ReLU
        Linear : 256 -> 256
        ReLU
        Linear : 256 -> 1
        Output : (B,) raw logits — apply sigmoid for a probability.

    Bump ``NCF_ARCH_VERSION`` whenever the layer shapes change so old
    state_dicts can't silently load into a new architecture.
    """

    def __init__(self, embed_dim: int = EMBED_DIM, hidden: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * embed_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)
