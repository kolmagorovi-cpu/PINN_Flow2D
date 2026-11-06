"""Float64 empirical kernels of signed physics residuals.

Each scalar residual is a Jacobian row; vector components are never summed
before differentiation. FVM rows include sqrt(V_i) for the spatial L2 measure.
Loss/SA weights and global loss normalizations are not included.
"""
from __future__ import annotations

import copy
import math
from typing import TYPE_CHECKING, Any, Iterable

import torch

from airfoil_pinn.data.batch import PINNBatch

if TYPE_CHECKING:
    from airfoil_pinn.training.components.loss_computer import LossComputer
    from airfoil_pinn.training.components.trainer import Trainer

Batch = PINNBatch | tuple[torch.Tensor, ...]

RELATIVE_CUTOFFS = (1e-12, 1e-10, 1e-8, 1e-6)


def residual_jacobian(
    residuals: torch.Tensor, parameters: Iterable[torch.nn.Parameter],
) -> torch.Tensor:
    """Differentiate in double precision without writing parameter.grad."""
    parameters = tuple(p for p in parameters if p.requires_grad)
    if residuals.dtype != torch.float64 or any(p.dtype != torch.float64 for p in parameters):
        raise TypeError("NTK residuals and model parameters must be float64")
    width = sum(p.numel() for p in parameters)
    rows = []
    for residual in residuals.reshape(-1):
        gradients = (
            torch.autograd.grad(residual, parameters, retain_graph=True, allow_unused=True)
            if parameters and residual.requires_grad else (None,) * len(parameters)
        )
        rows.append(torch.cat([
            (g.detach() if g is not None else torch.zeros_like(p)).reshape(-1)
            for p, g in zip(parameters, gradients)
        ]) if parameters else residual.new_empty(0))
    return torch.stack(rows) if rows else residuals.new_empty((0, width))


def kernel_spectrum(jacobian: torch.Tensor) -> dict[str, Any]:
    """Report raw eigenvalues, entropy effective rank and thresholded ranks.

    A cutoff keeps lambda > rtol * lambda_max. Condition numbers refer only
    to that retained subspace; full_condition is infinite for deficient rank
    at the float64 roundoff floor (n * eps).
    """
    if jacobian.dtype != torch.float64:
        raise TypeError("NTK Jacobian must be float64")
    if not bool(torch.isfinite(jacobian).all()):
        raise ValueError("NTK Jacobian contains NaN or Inf")
    kernel = jacobian @ jacobian.T
    if not bool(torch.isfinite(kernel).all()):
        raise ValueError("NTK matrix contains NaN or Inf")
    eigenvalues = torch.linalg.eigvalsh((kernel + kernel.T) * 0.5).cpu()
    size = eigenvalues.numel()
    maximum = max(0.0, float(eigenvalues[-1])) if size else 0.0
    floor = size * torch.finfo(torch.float64).eps

    def cutoff(rtol: float) -> dict[str, Any]:
        retained = eigenvalues[eigenvalues > rtol * maximum]
        return dict(relative_cutoff=rtol, absolute_cutoff=rtol * maximum,
                    rank=retained.numel(), discarded=size - retained.numel(),
                    condition=maximum / float(retained[0]) if retained.numel() else None)

    numerical = cutoff(floor)
    # Remove only roundoff modes before computing the entropy of the spectrum.
    positive = eigenvalues[eigenvalues > floor * maximum]
    probabilities = positive / maximum if positive.numel() else positive
    probabilities = probabilities / probabilities.sum() if positive.numel() else probabilities
    effective_rank = math.exp(float(-(probabilities * probabilities.log()).sum())) if positive.numel() else 0.0
    return dict(
        status="ok" if size else "empty", dtype="float64", rows=size,
        parameters=jacobian.shape[1], eigenvalues=eigenvalues.tolist(),
        lambda_min=float(eigenvalues[0]) if size else None,
        lambda_max=maximum if size else None, effective_rank=effective_rank,
        numerical_rank=numerical['rank'], roundoff_relative_cutoff=floor,
        negative_eigenvalues=int((eigenvalues < 0).sum()),
        significant_negative_eigenvalues=int((eigenvalues < -floor * maximum).sum()),
        full_condition=(numerical['condition'] if size and numerical['rank'] == size else None),
        cutoffs=[cutoff(rtol) for rtol in RELATIVE_CUTOFFS],
    )


def boundary_residuals(
    lc: LossComputer,
    u: torch.Tensor,
    v: torch.Tensor,
    p: torch.Tensor,
    bt: torch.Tensor | None,
    x: torch.Tensor,
    y: torch.Tensor,
    case_ids: torch.Tensor | None,
    normal_x: torch.Tensor | None,
    normal_y: torch.Tensor | None,
) -> torch.Tensor:
    """Signed components of the soft BC operators used by LossComputer.

    Auxiliary derivative operators are included when their weight is positive;
    their magnitude is unweighted, just like the value constraints.
    """
    rows = []
    if bt is None or lc.bc_type == 'hard':
        return u.new_empty(0)
    target_u, target_v = lc.get_freestream_targets(x, case_ids)
    for index, name in enumerate(lc.data_handler.boundary_type_names):
        mask = bt[:, index:index + 1] > 0.5
        if lc.is_hard_boundary(name) or not bool(mask.any()):
            continue
        if name in ('inlet', 'farfield', 'velocity_dirichlet'):
            rows.extend((u[mask] - target_u[mask], v[mask] - target_v[mask]))
            if name == 'farfield':
                rows.append(p[mask])
        elif name == 'outlet':
            rows.append(p[mask])
            if lc.loss_weights.get('outlet_grad', 0.0) > 0:
                nx, ny = lc._boundary_normals(normal_x, normal_y, mask, boundary_name=name)
                rows.extend(lc._normal_derivative(f, x, y, nx, ny, mask) for f in (u, v))
        elif name == 'wall':
            if lc.boundary_conditions['wall']['kind'] == 'no_slip':
                rows.extend((u[mask], v[mask]))
            else:
                nx, ny = lc.compute_wall_normals(x, y, mask)
                rows.append(u[mask] * nx + v[mask] * ny)
        elif name in ('slip', 'slip_wall', 'symmetry'):
            nx, ny = lc._boundary_normals(normal_x, normal_y, mask, boundary_name=name)
            rows.append(u[mask] * nx + v[mask] * ny)
            if name == 'symmetry' and lc.loss_weights.get('symmetry_grad', 10.0) > 0:
                # Normals are geometry constants, matching the training operator.
                nx_full, ny_full = torch.zeros_like(u), torch.zeros_like(v)
                nx_full[mask], ny_full[mask] = nx, ny
                tangent = -u * ny_full + v * nx_full
                rows.append(lc._normal_derivative(tangent, x, y, nx, ny, mask))
        elif name == 'axis_symmetry':
            rows.append(u[mask])
            if lc.loss_weights.get('axis_symmetry_grad', 10.0) > 0:
                rows.append(lc._grad(v, x, silent=False)[mask])
    return torch.cat([r.reshape(-1) for r in rows]) if rows else u.new_empty(0)


def _sample(indices: torch.Tensor, limit: int) -> torch.Tensor:
    if indices.numel() <= limit:
        return indices
    positions = torch.linspace(0, indices.numel() - 1, limit, device=indices.device).long()
    return indices[positions]


def _diagnostic_copy(trainer: Trainer) -> Trainer:
    """Copy only mutable numerical objects; keep the large dataset read-only."""
    diagnostic = copy.copy(trainer)
    model = getattr(trainer.model, '_orig_mod', trainer.model)
    diagnostic.model = copy.deepcopy(model).double().eval()
    diagnostic.dtype = torch.float64
    lc = diagnostic.loss_computer = copy.copy(trainer.loss_computer)
    lc.dtype = torch.float64
    # Copy both together to preserve their shared SDF module.
    lc.sdf_module, lc.boundary_transform = copy.deepcopy((lc.sdf_module, lc.boundary_transform))
    if lc.sdf_module is not None:
        lc.sdf_module.double()
    lc.boundary_transform.double()
    if lc.fvm is not None:
        lc.fvm = copy.copy(lc.fvm).to(trainer.device, torch.float64)
    diagnostic._fvm_quad_tensor_cache = None
    return diagnostic


def _point_jacobians(
    trainer: Trainer, batch: Batch, limit: int,
) -> dict[str, torch.Tensor]:
    lc = trainer.loss_computer
    bt = batch[4] if batch[4].shape[1] else None
    pde_mask = trainer._pde_loss_mask(bt, batch[1]).reshape(-1)
    bc_mask = torch.zeros_like(pde_mask)
    if bt is not None:
        for index, name in enumerate(trainer.data_handler.boundary_type_names):
            if name != 'interior' and not lc.is_hard_boundary(name):
                bc_mask |= bt[:, index] > 0.5
    selected = torch.unique(torch.cat([
        _sample(torch.where(mask)[0], limit) for mask in (pde_mask, bc_mask)
    ]))

    def take(value, *, integer=False):
        if value is None:
            return None
        return value[selected].detach().clone().to(
            trainer.device, dtype=torch.long if integer else torch.float64,
        )

    x, y, sdf = (take(batch[i]).reshape(-1, 1) for i in (1, 2, 3))
    params = tuple(trainer.model.parameters())
    if not selected.numel():
        empty = residual_jacobian(x.reshape(-1), params)
        return {'pde': empty, 'bc': empty}
    x.requires_grad_(True)
    y.requires_grad_(True)
    bt = take(bt)
    case_ids = take(getattr(batch, 'case_id', None), integer=True)
    nx, ny = (take(getattr(batch, name, None)) for name in ('normal_x', 'normal_y'))
    nx = nx.reshape(-1, 1) if nx is not None else None
    ny = ny.reshape(-1, 1) if ny is not None else None
    sdf, valid = lc.prepare_sdf_for_pde(x, y, sdf, bt)
    raw = trainer.model(lc.build_input_tensor(x, y, sdf, case_ids=case_ids))
    u, v, p = lc.transform_outputs(*raw.split(1, dim=1), x, y, sdf=sdf, case_ids=case_ids)
    if lc.bc_type == 'hard':
        masks = [take(batch[i]).bool().reshape(-1, 1) for i in (5, 6, 7)]
        targets = [take(batch[i]).reshape(-1, 1) for i in (8, 9, 10)]
        u, v, p = lc.apply_hard_constraints(u, v, p, *masks, *targets)
        u, v = lc.apply_velocity_dirichlet_constraint(u, v, bt, case_ids=case_ids)
    residuals = lc.compute_pde_residuals(u, v, p, x, y, case_ids=case_ids, squared=False)
    valid = valid & trainer._pde_loss_mask(bt, x)
    pde_indices = _sample(torch.where(valid.reshape(-1))[0], limit)
    pde = torch.cat([residuals[name][pde_indices].reshape(-1) for name in lc.pde_components])
    # Only the independently selected boundary sample contributes BC rows.
    if bt is not None:
        bc_indices = _sample(torch.where(bc_mask)[0], limit)
        keep = torch.isin(selected, bc_indices).to(bt.device)
        bc_types = bt * keep[:, None]
    else:
        bc_types = None
    bc = boundary_residuals(lc, u, v, p, bc_types, x, y, case_ids, nx, ny)
    return {'pde': residual_jacobian(pde, params), 'bc': residual_jacobian(bc, params)}


def _fvm_jacobian(trainer: Trainer, limit: int) -> torch.Tensor:
    """Differentiate sqrt(V_i) * divergence_i for each sampled cell and case.

    This gives K = sqrt(V) J_div J_div.T sqrt(V), with the spatial measure
    of sum(V_i * divergence_i**2). There is no normalization by total or
    sampled area, so sampling selects rows of the same full-mesh Jacobian.
    """
    lc = trainer.loss_computer
    params = tuple(trainer.model.parameters())
    if lc.fvm is None or lc.fvm_weight <= 0:
        empty = torch.empty(0, device=trainer.device, dtype=torch.float64)
        return residual_jacobian(empty, params)
    fvm = lc.fvm
    cells = _sample(torch.arange(fvm.N, device=trainer.device), limit)
    sqrt_volumes = fvm.V[cells].sqrt()
    # Evaluate only faces incident to sampled volumes; keep both sides of each
    # internal flux so this remains exactly the full-mesh divergence subvector.
    internal = torch.where(torch.isin(fvm.quad_i, cells) | torch.isin(fvm.quad_j, cells))[0]
    boundary = torch.where(torch.isin(fvm.bnd_node, cells))[0]
    x, y, sdf, bt, count = trainer._get_fvm_quadrature_tensors()
    selected = torch.cat((internal, boundary + count))
    x, y, sdf = (value[selected] for value in (x, y, sdf))
    bt = bt[selected] if bt is not None else None
    for name in ('quad_i', 'quad_j', 'quad_S'):
        setattr(fvm, name, getattr(fvm, name)[internal])
    for name in ('bnd_node', 'bnd_type', 'bnd_S', '_prescribed_mask', '_u_prescribed', '_zero_flux_mask'):
        setattr(fvm, name, getattr(fvm, name)[boundary])
    fvm.Q, fvm.B = internal.numel(), boundary.numel()
    rows = []
    for _, output, freestream in trainer._forward_fvm_cases(x, y, sdf, bt):
        rows.append(fvm.compute_residual_quadrature(
            output[:fvm.Q, :2], output[fvm.Q:, :2], freestream_velocity=freestream,
        )[cells] * sqrt_volumes)
    residuals = torch.cat(rows) if rows else x.new_empty(0)
    return residual_jacobian(residuals, params)


def compute_ntk_diagnostics(
    trainer: Trainer, batch: Batch, max_samples: int = 64,
) -> dict[str, Any]:
    """Use up to max_samples PDE/BC points and FVM cells per flow case."""
    if isinstance(max_samples, bool) or not isinstance(max_samples, int) or max_samples < 1:
        raise ValueError("ntk_sample_size must be a positive integer")
    diagnostic = _diagnostic_copy(trainer)
    with torch.enable_grad(), torch.autocast(device_type=trainer.device.type, enabled=False):
        jacobians = _point_jacobians(diagnostic, batch, max_samples)
        jacobians['fvm'] = _fvm_jacobian(diagnostic, max_samples)
        blocks = {name: kernel_spectrum(jacobian) for name, jacobian in jacobians.items()}
    blocks['fvm']['residual_weighting'] = 'sqrt_control_volume'
    blocks['fvm']['volume_normalization'] = 'none'
    if diagnostic.loss_computer.fvm is None:
        blocks['fvm']['status'] = 'unavailable: no mesh'
    elif diagnostic.loss_computer.fvm_weight <= 0:
        blocks['fvm']['status'] = 'disabled'
    # "weighted" refers to loss/SA weights; spatial FVM weights are reported
    # explicitly in the FVM block.
    return dict(dtype='float64', weighted=False, parameter_scope='model',
                max_samples=max_samples, blocks=blocks)


def format_ntk_diagnostics(report: dict[str, Any], epoch: int) -> str:
    lines = [f"[NTK] Epoch {epoch} | float64 | signed residuals, no loss/SA weights | K = J J^T",
             "effective_rank = exp(entropy); condition applies to retained eigenvalues; "
             "full_condition=inf means rank deficient, n/a means empty."]
    for name, block in report['blocks'].items():
        if name == 'fvm' and block.get('residual_weighting') == 'sqrt_control_volume':
            lines.append("  FVM measure: r_i = sqrt(V_i) * divergence_i; no total-area normalization.")
        full = block['full_condition']
        full_text = f"{full:.6e}" if full is not None else ('inf' if block['rows'] else 'n/a')
        lines.append(f"  {name.upper()}: {block['status']} | rows={block['rows']} | "
                     f"effective_rank={block['effective_rank']:.6g} | "
                     f"numerical_rank={block['numerical_rank']} | full_condition={full_text} | "
                     f"roundoff_rtol={block['roundoff_relative_cutoff']:.3e} | "
                     f"significant_negative={block['significant_negative_eigenvalues']}")
        lines.append("    eigenvalues (ascending): " + ', '.join(f"{v:.6e}" for v in block['eigenvalues']))
        for cut in block['cutoffs']:
            condition = cut['condition']
            condition_text = f"{condition:.6e}" if condition is not None else 'n/a'
            lines.append(f"    rtol={cut['relative_cutoff']:.1e}, "
                         f"atol=rtol*lambda_max={cut['absolute_cutoff']:.6e}: "
                         f"rank={cut['rank']}/{block['rows']}, "
                         f"discarded={cut['discarded']}, condition={condition_text}")
    return '\n'.join(lines)
