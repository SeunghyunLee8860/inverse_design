"""Append-only optimization telemetry with per-iteration plots/checkpoints."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import numpy as np


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=float) + "\n")
    temporary.replace(path)


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


class OptimizationTelemetry:
    """Persist enough state to audit and resume every accepted iteration.

    This class does not calculate a FOM or gradient. The optimizer supplies
    literal scalar metrics and the latent/filtered/projected arrays. Every call
    appends one JSONL record and creates an immutable iteration directory.
    """

    def __init__(
        self,
        root: Path,
        *,
        extent_um: tuple[float, float, float, float],
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.extent_um = tuple(float(value) for value in extent_um)
        self.iterations = self.root / "iterations"
        self.live = self.root / "live_plots"
        self.checkpoints = self.root / "checkpoints"
        for directory in (self.root, self.iterations, self.live, self.checkpoints):
            directory.mkdir(parents=True, exist_ok=True)
        self.history_path = self.root / "history.jsonl"

    @staticmethod
    def _design_metrics(projected: np.ndarray) -> dict[str, float]:
        return {
            "grayness": float(np.mean(4.0 * projected * (1.0 - projected))),
            "solid_fraction_projected": float(np.mean(projected)),
            "solid_fraction_binary": float(np.mean(projected >= 0.5)),
        }

    def record(
        self,
        *,
        iteration: int,
        latent: np.ndarray,
        filtered: np.ndarray,
        projected: np.ndarray,
        metrics: Mapping[str, Any],
        constraint_fields: Mapping[str, np.ndarray] | None = None,
        physical_fields: Mapping[str, np.ndarray] | None = None,
    ) -> Path:
        if iteration < 0:
            raise ValueError("iteration must be nonnegative")
        arrays = {
            "latent": np.asarray(latent, float),
            "filtered": np.asarray(filtered, float),
            "projected": np.asarray(projected, float),
        }
        shape = arrays["latent"].shape
        if len(shape) != 2 or any(value.shape != shape for value in arrays.values()):
            raise ValueError("latent/filtered/projected must share one 2D shape")
        if any(not np.all(np.isfinite(value)) for value in arrays.values()):
            raise ValueError("design arrays must be finite")
        if np.min(arrays["projected"]) < 0.0 or np.max(arrays["projected"]) > 1.0:
            raise ValueError("projected density must remain within [0,1]")

        iteration_dir = self.iterations / f"iter_{iteration:06d}"
        iteration_dir.mkdir(exist_ok=False)
        record = {
            "iteration": int(iteration),
            **{key: value for key, value in metrics.items()},
            **self._design_metrics(arrays["projected"]),
            "design_shape": list(shape),
            "extent_um": list(self.extent_um),
        }
        with self.history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, default=float) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        _atomic_json(iteration_dir / "record.json", record)
        _atomic_json(self.root / "latest.json", record)
        _atomic_npz(
            iteration_dir / "design.npz",
            **arrays,
            binary=(arrays["projected"] >= 0.5).astype(np.uint8),
        )
        _atomic_npz(
            self.checkpoints / "latest_design.npz",
            **arrays,
            iteration=np.asarray(iteration),
        )
        self._plot_design(iteration_dir / "design.png", arrays, iteration)
        self._plot_history(self.live / "iteration_history.png")
        if constraint_fields:
            self._plot_fields(
                iteration_dir / "constraint_fields.png",
                constraint_fields,
                f"iteration {iteration}: constraint fields",
            )
        if physical_fields:
            self._plot_fields(
                iteration_dir / "physical_fields.png",
                physical_fields,
                f"iteration {iteration}: physical fields",
            )
        return iteration_dir

    def _plot_design(
        self, path: Path, arrays: Mapping[str, np.ndarray], iteration: int
    ) -> None:
        panels = {
            "latent": arrays["latent"],
            "filtered": arrays["filtered"],
            "projected physical density": arrays["projected"],
            "binary preview (threshold 0.5)": arrays["projected"] >= 0.5,
        }
        fig, axes = plt.subplots(1, 4, figsize=(15.5, 3.8), constrained_layout=True)
        for axis, (name, values) in zip(axes, panels.items()):
            image = axis.imshow(
                np.asarray(values).T,
                origin="lower",
                extent=self.extent_um,
                vmin=0.0,
                vmax=1.0,
                cmap="gray_r",
                interpolation="nearest",
                aspect="equal",
            )
            axis.set_title(name)
            axis.set_xlabel("x=b (µm)")
            axis.set_ylabel("y=a (µm)")
            fig.colorbar(image, ax=axis, fraction=0.046)
        fig.suptitle(f"Run 002 design state — iteration {iteration}")
        fig.savefig(path, dpi=160)
        plt.close(fig)

    def _history(self) -> list[dict[str, Any]]:
        return [
            json.loads(line)
            for line in self.history_path.read_text().splitlines()
            if line.strip()
        ]

    def _plot_history(self, path: Path) -> None:
        rows = self._history()
        iterations = np.asarray([row["iteration"] for row in rows], int)

        def series(key: str) -> np.ndarray:
            return np.asarray([row.get(key, np.nan) for row in rows], float)

        fig, axes = plt.subplots(2, 3, figsize=(14.2, 8.0), constrained_layout=True)
        objective = axes[0, 0]
        for key, label in (
            ("objective_scaled", "scaled FOM"),
            ("best_feasible_scaled", "best feasible"),
        ):
            values = series(key)
            if np.any(np.isfinite(values)):
                objective.plot(iterations, values, marker="o", label=label)
        objective.set_title("Iteration vs FOM")
        objective.legend()

        current = axes[0, 1]
        for key, label in (("current_A", "I"), ("current_per_power_A_W", "I/Pin")):
            values = series(key)
            if np.any(np.isfinite(values)):
                current.plot(iterations, values, marker="o", label=label)
        current.set_title("Signed objective components")
        current.legend()

        constraint = axes[0, 2]
        for key in ("g_solid", "g_void", "g_volume", "g_robust"):
            values = series(key)
            if np.any(np.isfinite(values)):
                constraint.plot(iterations, values, marker="o", label=key)
        constraint.axhline(0.0, color="k", ls="--", lw=1)
        constraint.set_title("Constraint residuals (≤0 feasible)")
        constraint.legend()

        grad = axes[1, 0]
        for key in ("physical_gradient_l2", "latent_gradient_l2", "mma_step_l2"):
            values = series(key)
            if np.any(np.isfinite(values)):
                grad.semilogy(iterations, np.maximum(np.abs(values), 1e-300), marker="o", label=key)
        grad.set_title("Gradient and accepted-step norms")
        grad.legend()

        design = axes[1, 1]
        for key in ("grayness", "solid_fraction_projected", "solid_fraction_binary"):
            design.plot(iterations, series(key), marker="o", label=key)
        design.set_ylim(-0.03, 1.03)
        design.set_title("Grayness and fill fractions")
        design.legend()

        gates = axes[1, 2]
        for key in (
            "optical_closure_relative",
            "Q_mapping_relative_error",
            "thermal_energy_balance_relative",
            "linear_residual_relative",
        ):
            values = series(key)
            if np.any(np.isfinite(values)):
                gates.semilogy(iterations, np.maximum(np.abs(values), 1e-300), marker="o", label=key)
        gates.set_title("Numerical gates")
        gates.legend(fontsize=8)

        for axis in axes.flat:
            axis.set_xlabel("iteration")
            axis.grid(alpha=0.25)
        fig.savefig(path, dpi=160)
        plt.close(fig)

    def _plot_fields(
        self,
        path: Path,
        fields: Mapping[str, np.ndarray],
        title: str,
    ) -> None:
        names = list(fields)
        columns = min(4, len(names))
        rows = int(np.ceil(len(names) / columns))
        fig, axes = plt.subplots(
            rows, columns, figsize=(4.0 * columns, 3.6 * rows), squeeze=False,
            constrained_layout=True,
        )
        for axis, name in zip(axes.flat, names):
            values = np.asarray(fields[name], float)
            if values.ndim != 2:
                raise ValueError(f"field {name} is not 2D")
            image = axis.imshow(
                values.T,
                origin="lower",
                extent=self.extent_um,
                cmap="coolwarm" if np.nanmin(values) < 0.0 else "magma",
                interpolation="nearest",
                aspect="equal",
            )
            axis.set_title(name)
            axis.set_xlabel("x=b (µm)")
            axis.set_ylabel("y=a (µm)")
            fig.colorbar(image, ax=axis, fraction=0.046)
        for axis in axes.flat[len(names):]:
            axis.set_visible(False)
        fig.suptitle(title)
        fig.savefig(path, dpi=160)
        plt.close(fig)
