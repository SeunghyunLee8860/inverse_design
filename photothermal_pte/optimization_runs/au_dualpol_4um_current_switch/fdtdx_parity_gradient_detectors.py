"""Separate differentiated observables from nondifferentiated FDTD controls."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from photothermal_pte.optimization_runs.au_dualpol_4um_current_switch.fdtdx_parity_dynamic_checkpoint import (
    tree_array_bytes,
)


PRODUCTION_GRADIENT_DETECTORS = ("au_late", "tairte4_late")


def filter_gradient_detectors(
    arrays: Any,
    objects: Any,
    *,
    keep_names: Iterable[str],
    jax_module: Any,
) -> tuple[Any, Any, dict[str, Any]]:
    """Return aligned array/object containers containing only selected detectors.

    Material, source, boundary, and volume objects are retained bit-for-bit.  A
    detector is removed from both ``ObjectContainer.object_list`` and
    ``ArrayContainer.detector_states`` so FDTDX cannot update a missing state or
    silently checkpoint an unused one.
    """

    state_names = tuple(arrays.detector_states)
    object_names = tuple(detector.name for detector in objects.detectors)
    if len(state_names) != len(set(state_names)) or len(object_names) != len(
        set(object_names)
    ):
        raise ValueError("detector names must be unique")
    if set(state_names) != set(object_names):
        raise ValueError(
            "detector object/state names differ: "
            f"objects={object_names}, states={state_names}"
        )

    requested = tuple(str(name) for name in keep_names)
    if len(requested) != len(set(requested)):
        raise ValueError("keep_names contains duplicates")
    unknown = sorted(set(requested) - set(state_names))
    if unknown:
        raise ValueError(f"unknown gradient detector names: {unknown}")
    keep = set(requested)

    detector_indices = {
        obj.name: index
        for index, obj in enumerate(objects.object_list)
        if obj.name in set(object_names)
    }
    removed_names = tuple(name for name in state_names if name not in keep)
    if any(detector_indices[name] <= objects.volume_idx for name in removed_names):
        raise ValueError(
            "cannot remove a detector at/before volume_idx without remapping volume_idx"
        )

    retained_object_list = [
        obj
        for obj in objects.object_list
        if obj.name not in set(object_names) or obj.name in keep
    ]
    retained_states = {
        name: arrays.detector_states[name] for name in state_names if name in keep
    }
    filtered_objects = objects.aset("object_list", retained_object_list)
    filtered_arrays = arrays.aset("detector_states", retained_states)

    filtered_object_names = tuple(
        detector.name for detector in filtered_objects.detectors
    )
    filtered_state_names = tuple(filtered_arrays.detector_states)
    retained_names = tuple(name for name in state_names if name in keep)
    if filtered_object_names != retained_names or filtered_state_names != retained_names:
        raise RuntimeError(
            "filtered detector object/state ordering diverged: "
            f"objects={filtered_object_names}, states={filtered_state_names}, "
            f"expected={retained_names}"
        )
    if filtered_objects.volume_idx != objects.volume_idx:
        raise RuntimeError("detector filtering changed volume_idx")

    bytes_by_name = {
        name: tree_array_bytes(arrays.detector_states[name], jax_module=jax_module)
        for name in state_names
    }
    original_bytes = sum(bytes_by_name.values())
    retained_bytes = sum(bytes_by_name[name] for name in retained_names)
    audit = {
        "schema": "fdtdx_4um_parity_gradient_detector_filter_v1",
        "status": "PASS",
        "original_names": list(state_names),
        "retained_names": list(retained_names),
        "removed_names": list(removed_names),
        "state_bytes_by_name": bytes_by_name,
        "original_detector_state_bytes": original_bytes,
        "retained_detector_state_bytes": retained_bytes,
        "removed_detector_state_bytes": original_bytes - retained_bytes,
        "retained_over_original_fraction": (
            retained_bytes / original_bytes if original_bytes else 0.0
        ),
        "volume_index_unchanged": True,
        "detector_objects_and_states_filtered_together": True,
        "non_detector_object_count_unchanged": (
            len(objects.object_list) - len(objects.detectors)
            == len(filtered_objects.object_list) - len(filtered_objects.detectors)
        ),
    }
    if not audit["non_detector_object_count_unchanged"]:
        raise RuntimeError("detector filtering changed non-detector object count")
    return filtered_arrays, filtered_objects, audit
