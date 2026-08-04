#!/usr/bin/env bash
set -euo pipefail

ROLE="${1:?role is required}"
WAIT_PID="${2:-0}"
REPO=/home/seunghyun/tairte4/worktrees/device_a_waist_sweep
PY=/home/eidl/miniconda3/envs/EIDL-Lumapi/bin/python
CONTRACT=photothermal_pte/reports/paper_ir_device_a_inside_flake_center/device_a_nine_position_contract.json
GEOMETRY=photothermal_pte/reports/paper_ir_device_a_fig3h_registered_scan/device_a_fig3h_registered_geometry.json
OPTICAL=/data/seunghyun/tairte4/artifacts/paper_ir_device_a_nine_position_20260803
STAGING=/data/seunghyun/tairte4/artifacts/paper_ir_device_a_nine_position_gpu2_staging_20260803
THERMAL=/data/seunghyun/tairte4/artifacts/paper_ir_device_a_nine_position_thermal_20260803
REPORT=photothermal_pte/reports/paper_ir_device_a_nine_position_two_interface
MARKERS=/data/seunghyun/tairte4/artifacts/paper_ir_device_a_nine_position_pipeline_markers

mkdir -p "${MARKERS}"
cd "${REPO}"

if [[ "${WAIT_PID}" != 0 ]]; then
    while kill -0 "${WAIT_PID}" 2>/dev/null; do
        sleep 20
    done
fi

case "${ROLE}" in
    gpu0)
        "${PY}" photothermal_pte/validation/paper_ir_sanity/run_device_a_nine_position_optical_batch.py \
            --position-contract "${CONTRACT}" \
            --geometry-contract "${GEOMETRY}" \
            --output-root "${OPTICAL}" \
            --gpu-device "GPU 0" --threads 3
        touch "${MARKERS}/gpu0.complete"
        ;;
    gpu2)
        "${PY}" photothermal_pte/validation/paper_ir_sanity/run_device_a_nine_position_optical_batch.py \
            --position-contract "${CONTRACT}" \
            --geometry-contract "${GEOMETRY}" \
            --output-root "${STAGING}" \
            --gpu-device "GPU 2" --threads 3 \
            --only-label inside_top \
            --only-label inside_middle \
            --only-label inside_bottom
        touch "${MARKERS}/gpu2.complete"
        ;;
    finalize)
        while [[ ! -f "${MARKERS}/gpu0.complete" || ! -f "${MARKERS}/gpu2.complete" ]]; do
            sleep 20
        done
        # Let any incremental thermal workers finish before the all-case
        # resume starts; every output directory has one writer at a time.
        while pgrep -f "python.*run_device_a_nine_position_thermal_batch" >/dev/null; do
            sleep 20
        done
        for label in edge_bottom inside_top inside_middle inside_bottom; do
            if [[ ! -e "${OPTICAL}/${label}" ]]; then
                ln -s "${STAGING}/${label}" "${OPTICAL}/${label}"
            fi
        done
        "${PY}" photothermal_pte/validation/paper_ir_sanity/run_device_a_nine_position_optical_batch.py \
            --position-contract "${CONTRACT}" \
            --geometry-contract "${GEOMETRY}" \
            --output-root "${OPTICAL}" \
            --gpu-device "GPU 0" --threads 3
        "${PY}" -m photothermal_pte.validation.paper_ir_sanity.run_device_a_nine_position_thermal_batch \
            --position-contract "${CONTRACT}" \
            --geometry-contract "${GEOMETRY}" \
            --optical-root "${OPTICAL}" \
            --output-root "${THERMAL}"
        "${PY}" -m photothermal_pte.validation.paper_ir_sanity.summarize_device_a_nine_position_results \
            --position-contract "${CONTRACT}" \
            --optical-root "${OPTICAL}" \
            --thermal-root "${THERMAL}" \
            --output-dir "${REPORT}"
        touch "${MARKERS}/pipeline.complete"
        ;;
    *)
        echo "unknown role: ${ROLE}" >&2
        exit 2
        ;;
esac
