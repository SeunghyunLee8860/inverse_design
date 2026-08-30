from __future__ import annotations

from pathlib import Path
import os
import sys


NEW_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASELINE_ROOT = NEW_ROOT.parent / "pte_electrode_optimizer"
BASELINE_ROOT = Path(
    os.environ.get("TAIRTE4_PTE_BASELINE", str(DEFAULT_BASELINE_ROOT))
).expanduser().resolve()
if not (BASELINE_ROOT / "src" / "tairte4_pte" / "electrical.py").is_file():
    raise FileNotFoundError(
        "baseline pte_electrode_optimizer was not found; set "
        "TAIRTE4_PTE_BASELINE to its checkout path"
    )
if str(BASELINE_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(BASELINE_ROOT / "src"))

from tairte4_pte.config import load_config  # noqa: E402,F401
from tairte4_pte.electrical import (  # noqa: E402,F401
    Electrode,
    ElectricalModel,
)
def baseline_path(*parts: str) -> Path:
    return BASELINE_ROOT.joinpath(*parts)
