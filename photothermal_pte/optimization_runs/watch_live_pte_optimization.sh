#!/usr/bin/env bash
set -u

EA_ROOT="/data/seunghyun/tairte4/artifacts/tairte4_contact_anchored/run044_ansys_dfm_Ea_20260811"
EB_ROOT="/data/seunghyun/tairte4/artifacts/tairte4_contact_anchored/run045_ansys_dfm_Eb_20260811"

print_run() {
    local label="$1"
    local session_prefix="$2"
    local root="$3"
    local history="${root}/history.json"
    local log=""
    local matched_sessions=""
    local session_state="STOPPED"
    local optimizer_state="STOPPED"
    local active_eval="none"

    matched_sessions="$(tmux list-sessions -F '#S' 2>/dev/null | rg "^${session_prefix}" || true)"
    if [[ -n "${matched_sessions}" ]]; then
        session_state="RUNNING"
    fi

    log="$(find "${root}" -maxdepth 1 -type f -name '*tmux.log' -printf '%T@ %p\n' 2>/dev/null | sort -nr | awk 'NR == 1 {print $2}')"

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
    if [[ -n "${matched_sessions}" ]]; then
        echo "  session=$(printf '%s' "${matched_sessions}" | paste -sd, -)"
    fi
    if [[ -f "${history}" ]]; then
        jq -r '
            .[-1]
            | "  completed_eval=\(.evaluation_id), beta=\(.beta), stage_eval=\(.stage_full_physics_evaluation), current_nA=\(.objective_at_reference_power_A * 1e9), gray=\(.gray_fraction_0p01_0p99), bad=\(.exact_bad_cells)"
        ' "${history}"
    else
        echo "  history missing: ${history}"
    fi

    if [[ -n "${log}" && -f "${log}" ]]; then
        local last_error
        last_error="$(rg 'Traceback|ERROR|license.*(fail|denied|unavailable)' "${log}" | tail -n 1 || true)"
        if [[ -n "${last_error}" ]]; then
            echo "  last_error=${last_error}"
        else
            echo "  last_error=none"
        fi
        echo "  log=$(basename "${log}")"
        echo "  log_updated=$(date -u -d "@$(stat -c %Y "${log}")" '+%Y-%m-%d %H:%M:%S UTC')"
    fi
}

echo "PTE optimization live status — $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
print_run "E||a" "tairte4_run044_Ea_" "${EA_ROOT}"
print_run "E||b" "tairte4_run045_Eb_" "${EB_ROOT}"
