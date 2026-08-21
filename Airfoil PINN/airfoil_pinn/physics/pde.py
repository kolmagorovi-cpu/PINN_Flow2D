"""Dimensionless incompressible-flow PDE residuals.

The module operates exclusively on CFD-normalized quantities:

``x = x_dim / c``, ``u = u_dim / U_inf`` and
``p = (p_dim - p_inf) / (0.5 * rho * U_inf**2)``.

Consequently, the viscous coefficient is exactly ``1 / Re_c`` and the
pressure-gradient coefficient is exactly ``0.5``.  Density and dynamic
viscosity must not be supplied independently: Buckingham's Pi theorem reduces
their influence, together with chord and freestream speed, to the chord
Reynolds number.
"""

from __future__ import annotations

from typing import Callable, Dict

import torch

__all__ = ["PDEResiduals"]


class PDEResiduals:
    """Evaluate squared pointwise residuals of dimensionless flow equations."""

    @staticmethod
    def _inverse_reynolds_like(
        inverse_reynolds: torch.Tensor,
        reference: torch.Tensor,
    ) -> torch.Tensor:
        """Validate and reshape ``1/Re`` for safe pointwise broadcasting.

        A one-dimensional tensor with one value per point is converted from
        ``(N,)`` to ``(N, 1)``.  This prevents accidental ``(N, N)``
        broadcasting against column-vector fields.
        """
        coefficient = torch.as_tensor(
            inverse_reynolds,
            device=reference.device,
            dtype=reference.dtype,
        )
        if coefficient.ndim == 1 and reference.ndim > 1:
            if coefficient.shape[0] != reference.shape[0]:
                raise ValueError(
                    "inverse_reynolds length must equal the PDE batch size"
                )
            coefficient = coefficient.reshape(
                coefficient.shape[0],
                *((1,) * (reference.ndim - 1)),
            )
        try:
            torch.broadcast_shapes(coefficient.shape, reference.shape)
        except RuntimeError as exc:
            raise ValueError(
                "inverse_reynolds is not broadcastable to the flow fields: "
                f"{tuple(coefficient.shape)} vs {tuple(reference.shape)}"
            ) from exc
        detached = coefficient.detach()
        if not bool(torch.isfinite(detached).all()):
            raise ValueError("inverse_reynolds must contain only finite values")
        if bool((detached <= 0.0).any()):
            raise ValueError("inverse_reynolds must be strictly positive")
        return coefficient

    @staticmethod
    def compute_residuals_2d(
        u: torch.Tensor,
        v: torch.Tensor,
        p: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
        inverse_reynolds: torch.Tensor,
        grad_fn: Callable,
    ) -> Dict[str, torch.Tensor]:
        """Return planar steady Navier--Stokes residual squares.

        The dimensional similarity parameters are represented solely by the
        pointwise ``inverse_reynolds`` coefficient.  It may be scalar or
        contain one differentiable value per collocation point.
        """
        inv_re = PDEResiduals._inverse_reynolds_like(
            inverse_reynolds,
            u,
        )

        u_x = grad_fn(u, x, silent=True)
        u_y = grad_fn(u, y, silent=True)
        v_x = grad_fn(v, x, silent=True)
        v_y = grad_fn(v, y, silent=True)
        p_x = grad_fn(p, x, silent=True)
        p_y = grad_fn(p, y, silent=True)

        u_xx = grad_fn(u_x, x, silent=True)
        u_yy = grad_fn(u_y, y, silent=True)
        v_xx = grad_fn(v_x, x, silent=True)
        v_yy = grad_fn(v_y, y, silent=True)

        continuity = u_x + v_y
        momentum_x = (
            u * u_x
            + v * u_y
            + 0.5 * p_x
            - inv_re * (u_xx + u_yy)
        )
        momentum_y = (
            u * v_x
            + v * v_y
            + 0.5 * p_y
            - inv_re * (v_xx + v_yy)
        )
        return {
            "cont": continuity.square(),
            "mom_x": momentum_x.square(),
            "mom_y": momentum_y.square(),
        }

    @staticmethod
    def compute_residuals_euler_2d(
        u: torch.Tensor,
        v: torch.Tensor,
        p: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
        grad_fn: Callable,
    ) -> Dict[str, torch.Tensor]:
        """Return planar steady incompressible Euler residual squares."""
        u_x = grad_fn(u, x, silent=True)
        u_y = grad_fn(u, y, silent=True)
        v_x = grad_fn(v, x, silent=True)
        v_y = grad_fn(v, y, silent=True)
        p_x = grad_fn(p, x, silent=True)
        p_y = grad_fn(p, y, silent=True)

        continuity = u_x + v_y
        momentum_x = u * u_x + v * u_y + 0.5 * p_x
        momentum_y = u * v_x + v * v_y + 0.5 * p_y
        return {
            "cont": continuity.square(),
            "mom_x": momentum_x.square(),
            "mom_y": momentum_y.square(),
        }

    @staticmethod
    def compute_residuals_axisymmetric(
        u: torch.Tensor,
        v: torch.Tensor,
        p: torch.Tensor,
        r: torch.Tensor,
        z: torch.Tensor,
        inverse_reynolds: torch.Tensor,
        grad_fn: Callable,
        eps: float = 1e-12,
    ) -> Dict[str, torch.Tensor]:
        """Return no-swirl axisymmetric Navier--Stokes residual squares.

        ``u`` is radial velocity and ``v`` is axial velocity.  Coordinates and
        velocities use the same chord and freestream scales as the planar
        equations.
        """
        inv_re = PDEResiduals._inverse_reynolds_like(
            inverse_reynolds,
            u,
        )
        u_r = grad_fn(u, r, silent=True)
        u_z = grad_fn(u, z, silent=True)
        v_r = grad_fn(v, r, silent=True)
        v_z = grad_fn(v, z, silent=True)
        p_r = grad_fn(p, r, silent=True)
        p_z = grad_fn(p, z, silent=True)

        u_rr = grad_fn(u_r, r, silent=True)
        u_zz = grad_fn(u_z, z, silent=True)
        v_rr = grad_fn(v_r, r, silent=True)
        v_zz = grad_fn(v_z, z, silent=True)

        r_safe = r + torch.as_tensor(eps, device=r.device, dtype=r.dtype)
        continuity = u / r_safe + u_r + v_z
        laplacian_u = (
            u_rr + u_r / r_safe - u / r_safe.square() + u_zz
        )
        laplacian_v = v_rr + v_r / r_safe + v_zz
        momentum_r = (
            u * u_r + v * u_z + 0.5 * p_r - inv_re * laplacian_u
        )
        momentum_z = (
            u * v_r + v * v_z + 0.5 * p_z - inv_re * laplacian_v
        )
        return {
            "cont": continuity.square(),
            "mom_r": momentum_r.square(),
            "mom_z": momentum_z.square(),
        }

    @staticmethod
    def compute_residuals_euler_axisymmetric(
        u: torch.Tensor,
        v: torch.Tensor,
        p: torch.Tensor,
        r: torch.Tensor,
        z: torch.Tensor,
        grad_fn: Callable,
        eps: float = 1e-12,
    ) -> Dict[str, torch.Tensor]:
        """Return no-swirl axisymmetric incompressible Euler residuals."""
        u_r = grad_fn(u, r, silent=True)
        u_z = grad_fn(u, z, silent=True)
        v_r = grad_fn(v, r, silent=True)
        v_z = grad_fn(v, z, silent=True)
        p_r = grad_fn(p, r, silent=True)
        p_z = grad_fn(p, z, silent=True)

        r_safe = r + torch.as_tensor(eps, device=r.device, dtype=r.dtype)
        continuity = u / r_safe + u_r + v_z
        momentum_r = u * u_r + v * u_z + 0.5 * p_r
        momentum_z = u * v_r + v * v_z + 0.5 * p_z
        return {
            "cont": continuity.square(),
            "mom_r": momentum_r.square(),
            "mom_z": momentum_z.square(),
        }
