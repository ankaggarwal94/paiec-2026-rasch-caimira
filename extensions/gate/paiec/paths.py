"""Where the full upstream measurement-db lives (only needed to rebuild the split or build new ones).

$PAIEC_MDB, else the lab's shared copy when it exists, else data/mdb in the repo.
"""
import os
from pathlib import Path

_LAB = Path("/fs/clip-scratch/mgor/measurement-db")
MDB_ROOT = os.environ.get("PAIEC_MDB") or (str(_LAB) if _LAB.is_dir() else "data/mdb")
