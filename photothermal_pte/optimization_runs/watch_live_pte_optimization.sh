#!/usr/bin/env bash
set -u

EA_ROOT="/data/seunghyun/tairte4/artifacts/tairte4_contact_anchored/run044_ansys_dfm_Ea_20260811"
EB_ROOT="/data/seunghyun/tairte4/artifacts/tairte4_contact_anchored/run045_ansys_dfm_Eb_20260811"

print_run() {
    local label="$1"
    local session="$2"
    local root="$3"
    local history="${root}/history.json"
    local log="${root}/adaptive_recovery3_tmux.log"
    local session_state="STOPPED"
    local optimizer_state="STOPPED"
    local active_eval="none"

    if tmux has-session -t "${session}" 2>/dev/null; then
        session_state="RUNNING"
    fi

    if pgrep -f "run_ansys_dfm_ld_mma_optimization.*--raw-root ${root}" >/dev/null 2>&1; then
        optimizer_state="RUNNING"
    fi

    active_eval="$(
        ps -eo cmd \
        | rg "evaluate_objective_gradient.*--output-dir ${root}/evaluation_" \
        | rg -v "rg " \
        | sed -n 's#.*--output-dir [^ ]*/evaluation_\([^ ]*\).*#\1#p' \
        | head -n 1
    )"
    if [[ -z "${active_eval}" ]]; then
        active_eval="none"
    fi

    echo "${label}: tmux=${session_state}, optimizer=${optimizer_state}, active=${active_eval}"
    if [[ -f "${history}" ]]; then
        jq -r '
            .[-1]
            | "  completed_eval=\(.evaluation_id), beta=\(.beta), stage_eval=\(.stage_full_physics_evaluation), current_nA=\(.objective_at_reference_power_A * 1e9), gray=\(.gray_fraction_0p01_0p99), bad=\(.exact_bad_cells)"
        ' "${history}"
    else
        echo "  history missing: ${history}"
    fi

    if [[ -f "${log}" ]]; then
        local last_error
        last_error="$(rg 'Traceback|ERROR|license.*(fail|denied|unavailable)' "${log}" | tail -n 1 || true)"
        if [[ -n "${last_error}" ]]; then
            echo "  last_error=${last_error}"
        else
            echo "  last_error=none"
        fi
        echo "  log_updated=$(date -u -d "@$(stat -c %Y "${log}")" '+%Y-%m-%d %H:%M:%S UTC')"
    fi
}

echo "PTE optimization live status — $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
print_run "E||a" "tairte4_run044_Ea_adaptive3" "${EA_ROOT}"
print_run "E||b" "tairte4_run045_Eb_adaptive3" "${EB_ROOT}"

