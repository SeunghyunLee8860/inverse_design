"""Importable facade for the numbered 4-um Lumerical Au Drude probe."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


_SOURCE = Path(__file__).resolve().with_name("22_probe_lumerical_4um_au_np_drude.py")
_SPEC = importlib.util.spec_from_file_location(
    "au_dualpol_4um_lumerical_np_drude_probe", _SOURCE
)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(_SOURCE)
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

DrudeCarrier = _MODULE.DrudeCarrier
fit_drude_carrier = _MODULE.fit_drude_carrier
installed_release = _MODULE.installed_release
load_frozen_au = _MODULE.load_frozen_au
