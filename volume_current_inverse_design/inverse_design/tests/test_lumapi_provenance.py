"""Lumapi resolution must never reach the forbidden system install.

The production failure mode being guarded: `/opt/lumerical/v261`'s lumapi is
incompatible with this pipeline.  Pairing it with the r12 fdtd-engine does not
fail at startup -- it fails inside the FieldRegion adjoint at `importdataset`
("Failed to evaluate code"), minutes into a solve.  So resolution is pinned and
verified up front.

These tests spawn subprocesses because lumapi/eqc_lib import state is global and
`sys.modules` caching is exactly what is under test.
"""

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
BUNDLE = ROOT / "bundle"
FORBIDDEN = "/opt/lumerical/v261/api/python"
R12 = "/home/seunghyun/lumerical_r12/opt/lumerical/v261"


def _run(body: str, env_overrides: dict, expect_rc: int | None = None):
    """Run `body` in a fresh interpreter with a controlled environment."""
    import os

    env = {k: v for k, v in os.environ.items() if k not in {
        "PYTHONPATH", "VC_LUMERICAL_ROOT", "LUMERICAL_ROOT", "LUMERICAL_PYTHONPATH",
    }}
    env.update({k: v for k, v in env_overrides.items() if v is not None})
    script = textwrap.dedent(f"""
        import sys
        sys.path.insert(0, {str(ROOT)!r})
        sys.path.insert(0, {str(BUNDLE)!r})
    """) + textwrap.dedent(body)
    proc = subprocess.run([sys.executable, "-c", script], env=env,
                          capture_output=True, text=True)
    if expect_rc is not None:
        assert proc.returncode == expect_rc, (
            f"rc={proc.returncode}\nstdout={proc.stdout}\nstderr={proc.stderr}")
    return proc


requires_r12 = pytest.mark.skipif(
    not Path(R12, "api/python/lumapi.py").is_file(),
    reason="approved r12 tree not installed on this host",
)
requires_forbidden = pytest.mark.skipif(
    not Path(FORBIDDEN, "lumapi.py").is_file(),
    reason="system /opt/lumerical not installed; nothing to be shadowed by",
)


# --- resolution ---------------------------------------------------------
@requires_r12
@requires_forbidden
def test_polluted_pythonpath_still_resolves_r12():
    """The observed 4x-polluted PYTHONPATH must not win over the approved root."""
    proc = _run(
        """
        import eqc_lib as lib
        lib.bootstrap_env()
        import lumapi
        print("RESOLVED", lumapi.__file__)
        """,
        {"PYTHONPATH": ":".join([FORBIDDEN] * 4)},
        expect_rc=0,
    )
    resolved = proc.stdout.split("RESOLVED", 1)[1].strip()
    # NOTE: a plain `"/opt/lumerical" not in resolved` check is WRONG -- the
    # approved r12 path contains that text.  Anchor at the filesystem root.
    assert resolved.startswith(R12), resolved
    assert not resolved.startswith("/opt/lumerical"), resolved


@requires_r12
def test_explicit_forbidden_root_is_refused():
    """Pointing VC_LUMERICAL_ROOT at /opt must fail, not be honoured."""
    proc = _run("import eqc_lib", {"VC_LUMERICAL_ROOT": "/opt/lumerical/v261"})
    assert proc.returncode != 0
    assert "r12" in (proc.stdout + proc.stderr).lower()


@requires_r12
@requires_forbidden
def test_preloaded_forbidden_lumapi_fails_closed():
    """A wrong lumapi already in sys.modules must abort, never be hot-swapped."""
    proc = _run(
        """
        import lumapi                      # from the polluted PYTHONPATH
        import eqc_lib as lib
        lib.bootstrap_env()                # must raise
        print("REACHED-SOLVE")
        """,
        {"PYTHONPATH": FORBIDDEN, "VC_LUMERICAL_ROOT": R12},
    )
    assert proc.returncode != 0, proc.stdout
    assert "REACHED-SOLVE" not in proc.stdout
    assert "wrong lumapi loaded" in (proc.stdout + proc.stderr)


@requires_r12
def test_approved_lumapi_passes_assertion():
    proc = _run(
        """
        import eqc_lib as lib
        lib.bootstrap_env()
        import lumapi
        lib.assert_approved_lumapi(lumapi)
        p = lib.import_provenance()
        assert p["lumapi_approved"] is True, p
        assert p["lumapi_preloaded"] is True, p
        print("OK", p["lumapi_file"])
        """,
        {"VC_LUMERICAL_ROOT": R12},
        expect_rc=0,
    )
    assert proc.stdout.startswith("OK ") or "\nOK " in proc.stdout


@requires_r12
def test_bootstrap_purges_forbidden_pythonpath():
    proc = _run(
        """
        import os, sys, eqc_lib as lib
        lib.bootstrap_env()
        pp = os.environ.get("PYTHONPATH", "")
        # anchored: the approved r12 dir also contains the text /opt/lumerical
        assert not any(lib.is_forbidden_lumerical_path(p)
                       for p in pp.split(os.pathsep)), pp
        assert not [p for p in sys.path if lib.is_forbidden_lumerical_path(p)]
        assert lib.is_forbidden_lumerical_path("/opt/lumerical/v261/api/python")
        assert not lib.is_forbidden_lumerical_path(str(lib.R12_API))
        print("PURGED", repr(pp))
        """,
        {"PYTHONPATH": ":".join([FORBIDDEN, "/tmp/keepme"])},
        expect_rc=0,
    )
    assert "/tmp/keepme" in proc.stdout, proc.stdout


# --- source-level guarantees (no Lumerical needed) ----------------------
def test_msopt_has_no_opt_lumerical_glob():
    """The msopt helper must not rediscover /opt behind eqc_lib's back."""
    text = (BUNDLE / "msopt/Lumerical_utill.py").read_text()
    assert 'Path("/opt/lumerical").glob' not in text
    assert "_assert_approved_lumapi" in text


def test_eqc_lib_refuses_forbidden_root_in_source():
    text = (ROOT / "eqc_lib.py").read_text()
    assert "assert_approved_lumapi" in text
    assert "import_provenance" in text
    # the forbidden install must not be a discovery candidate
    assert '        Path("/opt/lumerical/v261"),' not in text


def test_launcher_purges_pythonpath():
    # The purge lives in env_production.sh (sourced by BOTH the production
    # launcher and the smoke wrapper), using an ANCHORED case filter -- a plain
    # substring grep would also drop the approved r12 tree.
    env = (ROOT / "env_production.sh").read_text()
    assert "LUMERICAL_PYTHONPATH" in env
    assert "/opt/lumerical|/opt/lumerical/*" in env
    for launcher in ("run_inverse_design.sh", "run_smoke.sh"):
        assert "env_production.sh" in (ROOT / launcher).read_text(), launcher


def test_launcher_purge_behavior():
    """Functional check: forbidden entries dropped, r12-lookalike kept."""
    import os
    import subprocess

    r12ish = "/home/seunghyun/lumerical_r12/opt/lumerical/v261/api/python"
    polluted = ":".join(["/opt/lumerical/v261/api/python"] * 4 + [r12ish, "/tmp/keep"])
    env = dict(os.environ)
    env.update({"PYTHONPATH": polluted, "GPU": "GPU 0"})
    env.pop("VC_LUMERICAL_ROOT", None)
    proc = subprocess.run(
        ["bash", "-c", f'. "{ROOT}/env_production.sh"; printf "%s" "${{PYTHONPATH:-}}"'],
        env=env, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    result = proc.stdout.splitlines()[-1] if proc.stdout else ""
    assert "/opt/lumerical/v261/api/python" not in result.split(":")
    assert r12ish in result.split(":")
    assert "/tmp/keep" in result.split(":")
