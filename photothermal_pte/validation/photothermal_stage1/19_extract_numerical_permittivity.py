#!/usr/bin/env python3
"""Query finite-dt numerical permittivity for each anisotropic component."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

import config_stage1 as config
from lumerical_api import select_installation, write_json


C0 = 299792458.0


def cj(value: complex) -> dict[str, float]:
    value = complex(value)
    return {"real": float(value.real), "imag": float(value.imag)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--lumerical-version", choices=("v251", "v261"), default="v261")
    parser.add_argument("--material-name", default="TaIrTe4_ani")
    args = parser.parse_args()
    project = Path(args.project).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    installation = select_installation(args.lumerical_version)
    os.environ["VC_LUMERICAL_ROOT"] = str(installation.root)
    os.environ["LUMERICAL_ROOT"] = str(installation.root)
    for path in (config.REPOSITORY_ROOT, config.REPOSITORY_ROOT / "bundle"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    import eqc_lib as runtime

    fdtd = runtime.open_control(project)
    try:
        frequency = C0 / 4e-6
        fmin = frequency * (1.0 - 1e-9)
        fmax = frequency * (1.0 + 1e-9)
        dt_sources: dict[str, object] = {}
        dt = None
        for label, getter in (
            ("getdata_FDTD_dt", lambda: fdtd.getdata("FDTD", "dt")),
            ("getnamed_FDTD_dt", lambda: fdtd.getnamed("FDTD", "dt")),
        ):
            try:
                value = np.asarray(getter()).reshape(-1)
                dt_sources[label] = value.tolist()
                if value.size == 1 and float(np.real(value[0])) > 0:
                    dt = float(np.real(value[0]))
                    break
            except Exception as exc:
                dt_sources[label] = f"{type(exc).__name__}: {exc}"
        if dt is None:
            raise RuntimeError(f"could not extract FDTD dt: {dt_sources}")
        axes = []
        for component, axis in zip((1, 2, 3), "xyz"):
            fitted_n = np.asarray(
                fdtd.getfdtdindex(
                    args.material_name,
                    np.asarray([frequency]),
                    fmin,
                    fmax,
                    component,
                )
            ).reshape(-1)[0]
            numerical = np.asarray(
                fdtd.getnumericalpermittivity(
                    args.material_name,
                    np.asarray([frequency]),
                    fmin,
                    fmax,
                    dt,
                    component,
                )
            ).reshape(-1)[0]
            axes.append(
                {
                    "axis": axis,
                    "fitted_epsilon_n_squared": cj(complex(fitted_n) ** 2),
                    "finite_dt_numerical_permittivity": cj(numerical),
                    "relative_difference": float(
                        abs(numerical - complex(fitted_n) ** 2)
                        / max(abs(complex(fitted_n) ** 2), np.finfo(float).tiny)
                    ),
                }
            )
    finally:
        fdtd.close()
    result = {
        "project": str(project),
        "material_name": args.material_name,
        "frequency_Hz": frequency,
        "dt_s": dt,
        "dt_sources": dt_sources,
        "axes": axes,
        "definition": (
            "getnumericalpermittivity returns the finite-dt D/E numerical "
            "permittivity used by the FDTD update"
        ),
        "heat_run": False,
        "clipping": False,
        "flux_gain": False,
    }
    write_json(output, result)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
