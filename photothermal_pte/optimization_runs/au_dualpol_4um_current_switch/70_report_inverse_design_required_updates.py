#!/usr/bin/env python3
"""Report required inverse-design updates since an older Git handoff.

Read-only with respect to source code and production artifacts.  The script
runs no Maxwell/PDE/optimizer calculation and writes only its requested report
folder.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any


HERE = Path(__file__).resolve().parent
REPOSITORY = Path(__file__).resolve().parents[3]
DEFAULT_BASELINE = "origin/agent/optimize-au-dualpol-4um-pte"

CHECKS = (
    {
        "id": "low_density_C1_material_path",
        "title_ko": "낮은 density의 Lumerical Yee 분기 회피",
        "file": "au_density_relaxation.py",
        "marker": "christiansen_nk_low_density_c1_then_square_v2",
        "required": True,
        "why_ko": (
            "기존 선형 n-k 경로가 매우 낮은 density에서 Lumerical의 "
            "Re(epsilon)=1 물질 분류 경계를 지나 성분별 Jacobian이 불연속적으로 "
            "바뀌던 문제를 rho<0.02 C1 전이로 제거한다."
        ),
        "restart_impact_ko": (
            "광학 물질 계약과 density hash가 바뀌므로 기존 optical cache와 해당 "
            "beta의 Jacobian/AD-FD 인증서는 재사용하지 않는다. 저장 density에서 "
            "새 commit·새 output으로 재개한다."
        ),
    },
    {
        "id": "target_cap_retention_stagnation",
        "title_ko": "목표 제약 후보를 이용한 불필요한 반복 종료",
        "file": "lumerical_4um_continuation.py",
        "marker": "target_cap_retention_progress",
        "required": True,
        "why_ko": (
            "같은 beta에서 최고 FOM이 오랫동안 개선되지 않아도 density 움직임이 "
            "남아 있으면 최대 평가 횟수까지 반복하던 문제를 막는다."
        ),
        "restart_impact_ko": (
            "FOM 하락만으로 멈추지 않는다. beta 최종 제약 만족, 최고 FOM의 90% "
            "이상 보존, 10개 고유 구조 동안 유의미한 최고값 개선 없음이 모두 "
            "확인되면 최종 cap 확인 단계로 이동한다."
        ),
    },
    {
        "id": "transient_resource_watchdog",
        "title_ko": "GPU 자원·라이선스 순간 충돌 자동 재시도",
        "file": "watch_lumerical_b200_direct_checkout.sh",
        "marker": "could not match resource name provided or the resource may not be active",
        "required": True,
        "why_ko": (
            "GPU가 순간 점유되거나 solver task가 부족할 때 전체 최적화를 끝내지 "
            "않고 저장 checkpoint에서 기다렸다 재시도한다."
        ),
        "restart_impact_ko": "CPU fallback 없이 같은 GPU/output/commit에서 재시도한다.",
    },
    {
        "id": "successful_evaluation_checkpoint",
        "title_ko": "매 성공 평가의 density checkpoint",
        "file": "41_optimize_lumerical_4um_dualpol_continuation.py",
        "marker": "latest_successful_state.npz",
        "required": True,
        "why_ko": "오류가 나도 마지막으로 검증된 물리 평가의 density를 보존한다.",
        "restart_impact_ko": (
            "MMA 내부 asymptote는 저장되지 않으므로 cross-commit 재개는 density를 "
            "시작점으로 쓰는 fresh MMA이다."
        ),
    },
    {
        "id": "density_preserving_beta_remap",
        "title_ko": "beta 변경 시 물리 density 보존 remap",
        "file": "lumerical_4um_continuation.py",
        "marker": "remap_latent_between_betas",
        "required": True,
        "why_ko": (
            "beta를 바꾸는 순간 같은 latent를 그대로 투영해 구조와 전류 부호가 "
            "갑자기 변하는 것을 막는다."
        ),
        "restart_impact_ko": "beta 전환마다 fresh current sign과 FOM 보존 gate를 통과해야 한다.",
    },
    {
        "id": "ADFD_cadence",
        "title_ko": "AD-FD 검사 횟수 제한",
        "file": "41_optimize_lumerical_4um_dualpol_continuation.py",
        "marker": "full_chain_current_AD_FD_cadence",
        "required": True,
        "why_ko": (
            "AD-FD는 시작 대표 구조, beta 변경, 최종 binary precursor에서만 수행하고 "
            "매 평가마다 반복하지 않는다."
        ),
        "restart_impact_ko": "gate 허용오차는 그대로 유지한다.",
    },
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-ref", default=DEFAULT_BASELINE)
    parser.add_argument("--run-manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args], cwd=REPOSITORY, text=True, capture_output=True, check=False
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ref_exists(ref: str) -> bool:
    return _git("rev-parse", "--verify", f"{ref}^{{commit}}").returncode == 0


def _source_at_ref(ref: str, relative: str) -> str | None:
    result = _git("show", f"{ref}:photothermal_pte/optimization_runs/au_dualpol_4um_current_switch/{relative}")
    return result.stdout if result.returncode == 0 else None


def _checks(baseline: str, baseline_exists: bool) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for check in CHECKS:
        current_path = HERE / str(check["file"])
        current_text = current_path.read_text(encoding="utf-8") if current_path.is_file() else ""
        old_text = _source_at_ref(baseline, str(check["file"])) if baseline_exists else None
        current_present = str(check["marker"]) in current_text
        baseline_present = None if old_text is None else str(check["marker"]) in old_text
        records.append(
            {
                **check,
                "current_present": current_present,
                "baseline_present": baseline_present,
                "update_required_from_baseline": bool(
                    check["required"] and current_present and baseline_present is not True
                ),
                "current_file": str(current_path),
                "current_file_sha256": _sha256(current_path) if current_path.is_file() else None,
            }
        )
    return records


def _active_history(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    active = manifest.get("active_stage")
    if isinstance(active, dict) and isinstance(active.get("callback_history"), list):
        return list(active["callback_history"])
    return []


def _run_audit(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    source = path.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    manifest = json.loads(source.read_text(encoding="utf-8"))
    history = _active_history(manifest)
    feasible = [row for row in history if row.get("design_feasible") is True]
    best_row = max(
        feasible,
        key=lambda row: float(row.get("balanced_utility_nA", float("-inf"))),
        default=None,
    )
    best_index = int(best_row["callback_index"]) if best_row else None
    newest_index = int(history[-1]["callback_index"]) if history else None
    evaluations_since_best = (
        newest_index - best_index
        if newest_index is not None and best_index is not None
        else None
    )
    latest = manifest.get("latest") if isinstance(manifest.get("latest"), dict) else {}
    beta = latest.get("beta")
    plan = None
    if beta is not None:
        plan = manifest.get("beta_constraint_homotopy_plans", {}).get(f"{float(beta):g}")
    target_candidate = None
    if best_row and isinstance(plan, dict):
        target_dfm = plan.get("target_DFM_caps", [])
        target_gray = float(plan.get("target_grayness_cap", float("inf")))
        floor = 0.9 * float(best_row["balanced_utility_nA"])
        candidates = []
        for row in feasible:
            raw = row.get("raw_DFM_values", [])
            if (
                len(raw) >= len(target_dfm)
                and all(float(v) <= float(c) * 1.001 for v, c in zip(raw, target_dfm))
                and float(row.get("grayness", float("inf"))) <= target_gray * 1.001
                and float(row.get("balanced_utility_nA", float("-inf"))) >= floor - 1.0e-6
                and float(row.get("current_Ea_nA", float("-inf"))) > 0.0
                and float(row.get("current_Eb_nA", float("inf"))) < 0.0
            ):
                candidates.append(row)
        if candidates:
            target_candidate = max(
                candidates, key=lambda row: float(row["balanced_utility_nA"])
            )
    error_text = str(manifest.get("error", ""))
    transient_resource_error = any(
        marker in error_text
        for marker in (
            "could not match resource name provided or the resource may not be active",
            "FlexNet Licensing error:-4,132",
            "Licensed number of users already reached",
        )
    )
    return {
        "path": str(source),
        "sha256": _sha256(source),
        "status": manifest.get("status"),
        "git_commit": manifest.get("git_commit"),
        "beta": beta,
        "cap_substage": latest.get("cap_substage"),
        "latest_iteration": latest.get("iteration"),
        "active_unique_callbacks": len(history),
        "best_feasible_callback_index": best_index,
        "best_feasible_FOM_nA": (
            float(best_row["balanced_utility_nA"]) if best_row else None
        ),
        "evaluations_since_best": evaluations_since_best,
        "target_cap_retention_candidate": target_candidate,
        "long_fixed_cap_repetition_detected": bool(
            evaluations_since_best is not None and evaluations_since_best >= 10
        ),
        "transient_GPU_or_license_resource_error_detected": transient_resource_error,
    }


def main() -> int:
    args = _args()
    baseline_exists = _ref_exists(args.baseline_ref)
    head = _git("rev-parse", "HEAD").stdout.strip()
    checks = _checks(args.baseline_ref, baseline_exists)
    missing_current = [row["id"] for row in checks if row["required"] and not row["current_present"]]
    updates = [row["id"] for row in checks if row["update_required_from_baseline"]]
    log = _git("log", "--oneline", f"{args.baseline_ref}..HEAD") if baseline_exists else None
    status = _git("status", "--short")
    run = _run_audit(args.run_manifest)
    report = {
        "schema": "inverse-design-post-handoff-update-report-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "read_only_source_and_production_audit": True,
        "solves_performed": {"Maxwell": 0, "thermal": 0, "electrical": 0, "optimizer": 0},
        "repository": str(REPOSITORY),
        "current_commit": head,
        "baseline_ref": args.baseline_ref,
        "baseline_exists": baseline_exists,
        "worktree_dirty": bool(status.stdout.strip()),
        "commits_after_baseline": log.stdout.splitlines() if log and log.returncode == 0 else [],
        "checks": checks,
        "updates_required_from_baseline": updates,
        "required_features_missing_from_current_checkout": missing_current,
        "run_audit": run,
        "safe_update_order": [
            "기존 실패·진행 artifact를 그대로 보존한다.",
            "현재 검증 branch를 새 worktree에 checkout하고 보고서의 누락 필수 수정만 반영한다.",
            "물질 계약이 바뀌었다면 기존 optical cache와 해당 beta의 Jacobian/AD-FD 인증서를 폐기한다.",
            "마지막 성공 density의 hash와 latent/projected 일치를 검증해 새 output root로 cross-commit density restart한다.",
            "같은 GPU UUID이면 검증된 source-only calibration을 재사용할 수 있고, GPU UUID가 바뀌면 4개만 다시 만든다.",
            "대표 Jacobian과 full-chain AD-FD gate를 통과한 뒤 fresh MMA를 시작한다.",
            "watchdog 아래에서 실행해 순간 GPU·license 자원 충돌은 기다렸다 자동 재시도한다.",
            "FOM 하락만으로 단계를 끝내거나 Jacobian/AD-FD 허용오차를 완화하지 않는다.",
        ],
    }
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "required_updates.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md = [
        "# Inverse-design 필수 업데이트 보고서",
        "",
        f"- 기준 ref: `{args.baseline_ref}`",
        f"- 현재 commit: `{head}`",
        f"- 기준 이후 필요한 업데이트: `{len(updates)}`개",
        f"- 현재 checkout 자체의 누락: `{len(missing_current)}`개",
        "- 이 script가 수행한 물리 계산: `0회`",
        "",
        "## 항목별 판정",
        "",
        "| 항목 | 예전 ref | 현재 | 업데이트 필요 |",
        "|---|:---:|:---:|:---:|",
    ]
    for row in checks:
        md.append(
            f"| {row['title_ko']} | {row['baseline_present']} | {row['current_present']} | {row['update_required_from_baseline']} |"
        )
    md.extend(["", "## 왜 필요한가", ""])
    for row in checks:
        md.extend(
            [
                f"### {row['title_ko']}",
                "",
                row["why_ko"],
                "",
                row["restart_impact_ko"],
                "",
            ]
        )
    if run is not None:
        md.extend(
            [
                "## Run 진단",
                "",
                f"- 상태: `{run['status']}`",
                f"- beta / cap / eval: `{run['beta']} / {run['cap_substage']} / {run['latest_iteration']}`",
                f"- 최고 FOM: `{run['best_feasible_FOM_nA']}` nA",
                f"- 최고값 이후 평가 수: `{run['evaluations_since_best']}`",
                f"- 불필요한 장기 반복 감지: `{run['long_fixed_cap_repetition_detected']}`",
                f"- 순간 GPU·license 자원 오류 감지: `{run['transient_GPU_or_license_resource_error_detected']}`",
                f"- 목표 제약·90% 보존 후보 존재: `{run['target_cap_retention_candidate'] is not None}`",
                "",
            ]
        )
    md.extend(["## 안전한 적용 순서", ""])
    md.extend(f"{i}. {step}" for i, step in enumerate(report["safe_update_order"], 1))
    md.extend(["", f"Raw JSON: `{json_path}`", ""])
    md_path = output / "REQUIRED_UPDATES.md"
    md_path.write_text("\n".join(md), encoding="utf-8")
    prompt_path = output / "CODEX_HANDOFF_PROMPT.txt"
    prompt_path.write_text(
        "70_report_inverse_design_required_updates.py를 현재 run manifest와 함께 실행하고 "
        "REQUIRED_UPDATES.md 및 required_updates.json을 읽어라. "
        "updates_required_from_baseline에 표시된 수정만 검증된 commit에서 가져오고, "
        "기존 artifact를 보존한 채 hash 검증 cross-commit density restart를 사용하라. "
        "FOM 하락만으로 종료하지 말고 Jacobian/AD-FD gate를 완화하지 마라. "
        "Maxwell은 Lumerical FDTD, thermal/electrical은 custom CUDA PDE만 사용하며 "
        "FDTDX와 Lumerical HEAT/CHARGE는 사용하지 마라.\n",
        encoding="utf-8",
    )
    print(json.dumps({"json": str(json_path), "markdown": str(md_path), "updates": updates}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
