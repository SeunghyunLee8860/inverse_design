"""Persistent float64 CUDA PCG for thermal forward and implicit adjoint.

The physical FVM operator is assembled by the existing validated code.  This
module changes neither coefficients nor boundary conditions: it stores one
literal CSR matrix on CUDA and solves arbitrary forward/adjoint right-hand
sides without differentiating through the Krylov iterations.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np
from scipy import sparse


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("PyTorch is required for CUDA thermal solves") from exc
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA thermal solve requested but PyTorch CUDA is unavailable")
    return torch


@dataclass(frozen=True)
class CudaPCGResult:
    solution: np.ndarray
    iterations: int
    explicit_relative_residual: float
    solve_seconds: float
    reliable_restarts: int


class PersistentCudaCSR:
    """One fixed SPD CSR operator reused for forward and adjoint RHS vectors."""

    def __init__(self, matrix: sparse.spmatrix, *, cuda_device: int = 0) -> None:
        torch = _torch()
        csr = sparse.csr_matrix(matrix, dtype=np.float64)
        if csr.shape[0] != csr.shape[1]:
            raise ValueError("thermal operator must be square")
        if not csr.has_sorted_indices:
            csr.sort_indices()
        diagonal_np = np.asarray(csr.diagonal(), dtype=np.float64)
        if diagonal_np.size != csr.shape[0] or np.any(diagonal_np <= 0.0):
            raise ValueError("Jacobi preconditioner requires a positive diagonal")
        self.torch = torch
        self.device = torch.device(f"cuda:{int(cuda_device)}")
        torch.cuda.set_device(self.device)
        self.shape = csr.shape
        self.crow = torch.as_tensor(csr.indptr, dtype=torch.int64, device=self.device)
        self.col = torch.as_tensor(csr.indices, dtype=torch.int64, device=self.device)
        self.values = torch.as_tensor(csr.data, dtype=torch.float64, device=self.device)
        self.operator = torch.sparse_csr_tensor(
            self.crow,
            self.col,
            self.values,
            size=csr.shape,
            dtype=torch.float64,
            device=self.device,
        )
        self.diagonal = torch.as_tensor(
            diagonal_np, dtype=torch.float64, device=self.device
        )

    def matvec(self, vector: Any) -> Any:
        return self.torch.sparse.mm(
            self.operator, vector.reshape(-1, 1)
        ).reshape(-1)

    def solve(
        self,
        rhs: np.ndarray,
        *,
        initial: np.ndarray | None = None,
        relative_tolerance: float = 1.0e-10,
        max_iterations: int = 20000,
        residual_check_interval: int = 25,
        reliable_restart_interval: int | None = None,
    ) -> CudaPCGResult:
        if relative_tolerance <= 0.0:
            raise ValueError("relative_tolerance must be positive")
        if max_iterations <= 0 or residual_check_interval <= 0:
            raise ValueError("iteration controls must be positive")
        if reliable_restart_interval is not None and (
            reliable_restart_interval <= 0
            or reliable_restart_interval % residual_check_interval != 0
        ):
            raise ValueError(
                "reliable_restart_interval must be a positive multiple of "
                "residual_check_interval"
            )
        torch = self.torch
        rhs_np = np.asarray(rhs, dtype=np.float64).reshape(-1)
        if rhs_np.size != self.shape[0] or not np.all(np.isfinite(rhs_np)):
            raise ValueError("invalid thermal RHS")
        b = torch.as_tensor(rhs_np, dtype=torch.float64, device=self.device)
        if initial is None:
            x = torch.zeros_like(b)
        else:
            initial_np = np.asarray(initial, dtype=np.float64).reshape(-1)
            if initial_np.shape != rhs_np.shape or not np.all(np.isfinite(initial_np)):
                raise ValueError("invalid initial thermal iterate")
            x = torch.as_tensor(
                initial_np, dtype=torch.float64, device=self.device
            ).clone()
        b_norm = torch.linalg.vector_norm(b)
        if float(b_norm.item()) == 0.0:
            return CudaPCGResult(
                solution=np.zeros_like(rhs_np),
                iterations=0,
                explicit_relative_residual=0.0,
                solve_seconds=0.0,
                reliable_restarts=0,
            )
        threshold = relative_tolerance * b_norm
        started = perf_counter()
        r = b - self.matvec(x)
        z = r / self.diagonal
        p = z.clone()
        rz = torch.dot(r, z)
        iterations = 0
        restarts = 0
        converged = False
        explicit_relative = float("inf")
        for iteration in range(1, max_iterations + 1):
            applied = self.matvec(p)
            denominator = torch.dot(p, applied)
            if not bool(torch.isfinite(denominator).item()) or float(denominator.item()) <= 0.0:
                # An exactly converged small system can produce p=0 before
                # the scheduled residual check.  Certify the literal
                # residual before classifying this as loss of SPD.
                explicit = b - self.matvec(x)
                explicit_norm = torch.linalg.vector_norm(explicit)
                explicit_relative = float((explicit_norm / b_norm).item())
                if bool(explicit_norm.le(threshold).item()):
                    r = explicit
                    converged = True
                    iterations = iteration - 1
                    break
                raise RuntimeError(
                    "CUDA PCG lost positive definiteness before convergence: "
                    f"explicit_relative_residual={explicit_relative:.6e}"
                )
            alpha = rz / denominator
            x.add_(p, alpha=alpha)
            r.add_(applied, alpha=-alpha)
            iterations = iteration
            if iteration % residual_check_interval == 0 or iteration == max_iterations:
                explicit = b - self.matvec(x)
                explicit_norm = torch.linalg.vector_norm(explicit)
                explicit_relative = float((explicit_norm / b_norm).item())
                if not np.isfinite(explicit_relative):
                    raise RuntimeError("CUDA PCG produced a non-finite residual")
                if bool(explicit_norm.le(threshold).item()):
                    r = explicit
                    converged = True
                    break
                if (
                    reliable_restart_interval is not None
                    and iteration % reliable_restart_interval == 0
                ):
                    # Replace the recursive residual with the literal b-Ax
                    # residual to recover from roundoff drift.  The requested
                    # convergence tolerance is unchanged.
                    r = explicit
                    z = r / self.diagonal
                    p = z.clone()
                    rz = torch.dot(r, z)
                    restarts += 1
                    continue
                recursive_norm = torch.linalg.vector_norm(r)
                if bool(recursive_norm.le(threshold).item()):
                    r = explicit
                    z = r / self.diagonal
                    p = z.clone()
                    rz = torch.dot(r, z)
                    restarts += 1
                    continue
            z = r / self.diagonal
            rz_new = torch.dot(r, z)
            beta = rz_new / rz
            p.mul_(beta).add_(z)
            rz = rz_new
        torch.cuda.synchronize(self.device)
        seconds = perf_counter() - started
        if not converged:
            raise RuntimeError(
                "CUDA PCG did not converge: "
                f"iterations={iterations}, explicit_relative_residual={explicit_relative:.6e}"
            )
        solution = x.detach().cpu().numpy()
        return CudaPCGResult(
            solution=solution,
            iterations=iterations,
            explicit_relative_residual=explicit_relative,
            solve_seconds=seconds,
            reliable_restarts=restarts,
        )


@dataclass(frozen=True)
class ThermalForwardAdjointPair:
    forward: CudaPCGResult
    adjoint: CudaPCGResult
    reciprocity_relative_error: float


def solve_forward_adjoint_cuda(
    matrix: sparse.spmatrix,
    forward_rhs: np.ndarray,
    adjoint_rhs: np.ndarray,
    *,
    cuda_device: int = 0,
    forward_initial: np.ndarray | None = None,
    adjoint_initial: np.ndarray | None = None,
    relative_tolerance: float = 1.0e-10,
    max_iterations: int = 20000,
) -> ThermalForwardAdjointPair:
    """Solve K T=b and K^T lambda=dF/dT using one symmetric operator.

    The assembled thermal operator is required to be symmetric to roundoff.
    This is checked before the CUDA transfer rather than silently replacing
    K^T by K for a future nonsymmetric discretization.
    """
    csr = sparse.csr_matrix(matrix, dtype=np.float64)
    difference = csr - csr.T
    scale = max(float(np.max(np.abs(csr.data))), np.finfo(float).tiny)
    asymmetry = (
        0.0 if difference.nnz == 0 else float(np.max(np.abs(difference.data))) / scale
    )
    if asymmetry > 1.0e-13:
        raise ValueError(f"thermal operator is not symmetric: {asymmetry:.6e}")
    operator = PersistentCudaCSR(csr, cuda_device=cuda_device)
    forward = operator.solve(
        forward_rhs,
        initial=forward_initial,
        relative_tolerance=relative_tolerance,
        max_iterations=max_iterations,
    )
    adjoint = operator.solve(
        adjoint_rhs,
        initial=adjoint_initial,
        relative_tolerance=relative_tolerance,
        max_iterations=max_iterations,
    )
    left = float(np.dot(np.asarray(adjoint_rhs), forward.solution))
    right = float(np.dot(adjoint.solution, np.asarray(forward_rhs)))
    reciprocity = abs(left - right) / max(abs(left), abs(right), np.finfo(float).tiny)
    return ThermalForwardAdjointPair(
        forward=forward,
        adjoint=adjoint,
        reciprocity_relative_error=reciprocity,
    )
