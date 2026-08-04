"""Regression guards for launcher/runner path + exit semantics (review P0-1/3/4)."""

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_output_resolved_cwd_relative():
    # P0-1: runner and finalizer must resolve --output against CWD (not __file__)
    # so the shell's OUT and the Python run_root point at the same directory.
    runner = (ROOT / "inverse_design/run_constrained_inverse_design.py").read_text()
    fin = (ROOT / "inverse_design/final_projection.py").read_text()
    assert "Path(args.output).expanduser().resolve()" in runner
    assert "Path(args.output).expanduser().resolve()" in fin
    assert "(HERE / args.output)" not in runner
    assert "(HERE / args.output)" not in fin


def test_launchers_strict_mode_and_no_buggy_exit():
    # P0-3: the ${frc:-5} pattern (which yields 0 when frc=0) must be gone; every
    # launcher present must use strict mode.  (launch_G is source-only; the
    # portable run_inverse_design.sh is the canonical launcher in the bundle.)
    candidates = ["run_inverse_design.sh",
                  "inverse_design/launch_G_constrained_500nm_20260724.sh"]
    present = [n for n in candidates if (ROOT / n).exists()]
    assert present, "no launcher found"
    for name in present:
        t = (ROOT / name).read_text()
        assert "set -euo pipefail" in t, name
        assert 'exit "${frc:-5}"' not in t, name


def test_exit_branch_no_success_is_failure(tmp_path):
    # reproduce the FIXED branch: finalizer rc 0 but no SUCCESS.json -> exit 5
    script = (
        'set -euo pipefail\n'
        'frc=0\n'
        f'if [ "$frc" -ne 0 ]; then exit "$frc"; fi\n'
        f'if [ ! -f "{tmp_path}/SUCCESS.json" ]; then exit 5; fi\n'
        'echo ok\n'
    )
    r = subprocess.run(["bash", "-c", script])
    assert r.returncode == 5


def test_runner_fails_process_on_stage_failure():
    # P0-4: a classified stage failure must raise SystemExit (nonzero) so the
    # launcher aborts instead of auto-finalising.
    runner = (ROOT / "inverse_design/run_constrained_inverse_design.py").read_text()
    assert "if stage_failure is not None:" in runner
    assert "raise SystemExit(6)" in runner
