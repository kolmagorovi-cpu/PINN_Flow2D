#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Модуль loss_computer для проекта PINN Learning.
Содержит класс LossComputer, инкапсулирующий вычисления функции потерь:
- PDE невязки (через PDEResiduals)
- Граничные условия (soft/hard)
- Интегральное условие (баланс масс)
- Построение входного тензора для модели
- Применение hard constraints

Исправленная версия: метод compute_integral_loss переработан для корректного
учёта направления потока (нормали) и осесимметричной геометрии.
"""

import logging
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np
import torch

from airfoil_pinn.config.boundary import normalize_wall_condition
from airfoil_pinn.geometry.sdf import PolylineDistanceField
from airfoil_pinn.training.boundary import (
    BoundaryOutputTransform,
    resolve_boundary_conditions,
)
from airfoil_pinn.training.input_features import InputFeatureAssembler
from airfoil_pinn.utils import safe_grad
from airfoil_pinn.physics.pde import PDEResiduals

if TYPE_CHECKING:
    from airfoil_pinn.physics.fvm import FVMResiduals

logger = logging.getLogger(__name__)

__all__ = ["LossComputer"]


class LossComputer:
    """
    Вычисляет все компоненты функции потерь для PINN.
    Инкапсулирует логику, связанную с PDE, граничными условиями и интегральными условиями.
    """

    def __init__(
        self,
        config: Dict,
        data_handler,
        rho_norm: torch.Tensor,
        mu_norm: torch.Tensor,
        x_min: float,
        x_max: float,
        y_min: float,
        y_max: float,
        device: torch.device,
        dtype: torch.dtype
    ) -> None:
        """
        Args:
            config: полная конфигурация (словарь)
            data_handler: объект DataHandler с метаданными и информацией о границах
            rho_norm: устаревший аргумент совместимости; в безразмерной PDE
                плотность не является независимым коэффициентом
            mu_norm: устаревший аргумент совместимости; вязкий коэффициент
                вычисляется как ``1 / Re`` для каждой точки
            x_min, x_max, y_min, y_max: границы области (нормированные)
            device: устройство для вычислений
            dtype: тип данных
        """
        normalize_wall_condition(config)
        self.config = config
        self.data_handler = data_handler
        self.rho_norm = rho_norm
        self.mu_norm = mu_norm
        self.device = device
        self.dtype = dtype

        # Параметры из конфигурации
        self.flow_type = config['pde']['flow_type']
        self.pde_system = config['pde'].get('system', 'incompressible_navier_stokes')
        self.pde_on_boundary = bool(config['pde'].get('pde_on_boundary', False))
        boundary_section = config.get('boundary_conditions', {}) or {}
        self.bc_type = boundary_section.get('type', 'soft')
        self.boundary_conditions = resolve_boundary_conditions(config)
        wall_config = self.boundary_conditions['wall']
        self.wall_enforcement = str(
            wall_config.get('enforcement', 'soft')
        )
        self.pde_components = self._setup_pde_components()
        self.strict_autograd = bool(
            config.get('diagnostics', {}).get('strict_autograd', True)
        )

        # Веса потерь
        lw = config['loss_weights']
        self.loss_weights = lw['pde_weights'].copy()
        # A hard wall does not disable unrelated soft boundary operators.
        self.loss_weights.update(lw['bc_weights'])
        if 'slip_wall' in self.loss_weights:
            self.loss_weights.setdefault(
                'slip',
                self.loss_weights['slip_wall'],
            )
        elif 'slip' in self.loss_weights:
            self.loss_weights['slip_wall'] = self.loss_weights['slip']

        # SA-веса (флаг)
        self.use_sa_weights = lw['use_sa_weights']
        self.sa_apply_to_bc = lw['sa_apply_to_bc']

        # Адаптивная балансировка весов BC (опциональная функция, задаётся пользователем)
        self.adaptive_bc: bool = bool(lw.get('adaptive_bc', False))

        # Интегральное условие
        self.integral_condition = lw['integral_condition']
        self.integral_weight = lw['integral_weight']
        # Целевое значение интеграла (по умолчанию 0 - баланс масс)
        self.integral_target = lw.get('integral_target', 0.0)

        # FVM-потеря (закон сохранения массы на неструктурированной сетке)
        # Поддерживаемые форматы:
        #   dict с enabled/weight (новый): {'enabled': True, 'weight': 1.0}
        #   dict без enabled (старый):     {'weight': 0.5}  → вес берётся напрямую
        #   число/строка (legacy):         0.5
        fvm_cfg = lw.get('fvm', {})
        if isinstance(fvm_cfg, dict):
            if 'enabled' in fvm_cfg:
                fvm_on = bool(fvm_cfg['enabled'])
                self.fvm_weight: float = float(fvm_cfg.get('weight', 1.0)) if fvm_on else 0.0
            else:
                self.fvm_weight: float = float(fvm_cfg.get('weight', 0.0))
            self.fvm_batch_interval: int = max(1, int(fvm_cfg.get('batch_interval', 1)))
            self.fvm_schedule: str = str(
                fvm_cfg.get('schedule', 'every_optimizer_step')
            ).strip().lower().replace('-', '_')
            self.fvm_loss_reduction: str = str(
                fvm_cfg.get(
                    'reduction',
                    fvm_cfg.get('loss_reduction', 'divergence_l2_volume'),
                )
            )
            self.fvm_forward_chunk_size: int = max(
                1,
                int(fvm_cfg.get('forward_chunk_size', 32768)),
            )
            self.fvm_activation_checkpoint: bool = bool(
                fvm_cfg.get('activation_checkpoint', True)
            )
        else:
            self.fvm_weight: float = float(fvm_cfg)
            self.fvm_batch_interval: int = 1
            self.fvm_schedule: str = 'every_optimizer_step'
            self.fvm_loss_reduction: str = 'legacy'
            self.fvm_forward_chunk_size: int = 32768
            self.fvm_activation_checkpoint: bool = True
        # Заполняется из PINNSolver при наличии mesh_file в конфиге
        self.fvm: Optional["FVMResiduals"] = None

        # Границы области для интегрального условия
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.tol = 0.01 * max(x_max - x_min, y_max - y_min)

        # Входные параметры (для построения тензора)
        self.input_params = config['input_params']
        self.feature_assembler = InputFeatureAssembler(config)
        self.feature_manifest = self.feature_assembler.manifest
        logger.info(
            "Network input manifest: dim=%d, fingerprint=%s, features=%s",
            self.feature_manifest.input_dim,
            self.feature_manifest.fingerprint()[:12],
            ", ".join(
                feature.name for feature in self.feature_manifest.features
            ),
        )
        self.sdf_module: Optional[PolylineDistanceField] = None
        self.sdf_geometry_fingerprint: Optional[str] = None
        self.sdf_absolute_tie_tolerance = 1e-12
        self.sdf_relative_tie_tolerance = 1e-10
        self.sdf_float64_consistency_tolerance = float(
            (self.config.get('sdf', {}) or {}).get(
                'float64_consistency_tolerance',
                1e-12,
            )
        )
        if (
            not np.isfinite(self.sdf_float64_consistency_tolerance)
            or self.sdf_float64_consistency_tolerance < 0.0
        ):
            raise ValueError(
                "sdf.float64_consistency_tolerance must be finite and "
                "non-negative"
            )
        self._initialize_differentiable_sdf()
        self.synchronize_cached_training_sdf()
        self.boundary_transform = BoundaryOutputTransform(
            self.sdf_module,
            wall_kind=str(wall_config.get('kind', 'no_slip')),
            wall_enforcement=self.wall_enforcement,
            distance_scale=float(wall_config.get('distance_scale', 0.02)),
        ).to(device=self.device)
        self.boundary_enforcement_contract = {
            name: {
                "kind": str(values.get("kind", name)),
                "mode": str(values.get("enforcement", "soft")),
            }
            for name, values in self.boundary_conditions.items()
        }
        self.boundary_enforcement_contract["wall"] = (
            self.boundary_transform.contract()["wall"]
        )

        # Для axisymmetric требуется eps и множитель 2π
        self.eps_axi = 1e-12
        self.two_pi = 2.0 * torch.pi

    def get_flow_case_batch(
        self,
        reference: torch.Tensor,
        case_ids: Optional[torch.Tensor] = None,
    ):
        """Return the flow parameters associated with every point.

        The same helper is used by network input assembly, PDE coefficients
        and boundary targets.  This prevents a case identifier from selecting
        one operating point at the input while the loss silently uses another.
        """
        if case_ids is None:
            case_ids = torch.zeros(
                reference.shape[0],
                device=reference.device,
                dtype=torch.long,
            )
        if case_ids.numel() != reference.shape[0]:
            raise ValueError(
                "case_ids must contain exactly one identifier per point"
            )
        if not hasattr(self.data_handler, "get_flow_case_batch"):
            raise RuntimeError(
                "Parameterized physics requires "
                "DataHandler.get_flow_case_batch"
            )
        return self.data_handler.get_flow_case_batch(
            case_ids,
            device=reference.device,
            dtype=reference.dtype,
        )

    def get_freestream_targets(
        self,
        reference: torch.Tensor,
        case_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return dimensionless ``(u_inf, v_inf)`` for every point."""
        flow = self.get_flow_case_batch(reference, case_ids)
        return flow.cos_alpha, flow.sin_alpha

    def _initialize_differentiable_sdf(self) -> None:
        """Build the normalized segment SDF used by PDE autograd.

        Minimal unit-test handlers may not contain a geometry.  In that case
        input assembly with an explicitly supplied cached SDF remains
        available, but a training call requesting differentiable SDF fails
        with a descriptive error in :meth:`prepare_sdf_for_pde`.
        """
        if not self.feature_assembler.sdf_enabled:
            return

        polyline = getattr(self.data_handler, 'mesh_airfoil_coords', None)
        chord = float(getattr(self.data_handler, 'chord', 0.0))
        if polyline is None or len(polyline) < 2 or chord <= 0.0:
            logger.debug(
                "Differentiable SDF is not initialized: an ordered airfoil "
                "polyline and a positive chord are required."
            )
            return

        sdf_cfg = self.config.get('sdf', {}) or {}
        chunk_size = int(sdf_cfg.get('segment_chunk_size', 256))
        self.sdf_absolute_tie_tolerance = float(
            sdf_cfg.get('absolute_tie_tolerance', 1e-12)
        )
        self.sdf_relative_tie_tolerance = float(
            sdf_cfg.get('relative_tie_tolerance', 1e-10)
        )
        if (
            self.sdf_absolute_tie_tolerance < 0.0
            or self.sdf_relative_tie_tolerance < 0.0
        ):
            raise ValueError("SDF tie tolerances must be non-negative")

        normalized = np.asarray(polyline, dtype=np.float64) / chord
        sampler_metadata = getattr(self.data_handler, 'sampler_metadata', None)
        closed = (
            bool(sampler_metadata.get('foil_closed', True))
            if isinstance(sampler_metadata, dict)
            else len(normalized) >= 3
        )
        self.sdf_module = PolylineDistanceField(
            normalized,
            closed=closed,
            segment_chunk_size=chunk_size,
        ).to(device=self.device)
        self.sdf_geometry_fingerprint = (
            self.sdf_module.geometry_fingerprint
        )
        logger.info(
            "Differentiable SDF initialized: segments=%d, closed=%s, "
            "chunk=%d, geometry=%s",
            self.sdf_module.segment_count,
            closed,
            chunk_size,
            self.sdf_geometry_fingerprint[:12],
        )

    def evaluate_sdf_numpy(
        self,
        points: np.ndarray,
        *,
        point_chunk_size: int = 8192,
    ) -> np.ndarray:
        """Evaluate the canonical float64 SDF used by every non-AD path."""
        if self.sdf_module is None:
            raise RuntimeError(
                "Canonical SDF geometry is unavailable. Regenerate geometry "
                "and collocation artifacts."
            )
        return self.sdf_module.evaluate_numpy(
            np.asarray(points, dtype=np.float64),
            point_chunk_size=point_chunk_size,
        )

    @torch.no_grad()
    def evaluate_sdf_tensor(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        *,
        point_chunk_size: int = 8192,
    ) -> torch.Tensor:
        """Recompute inference/evaluation SDF by the canonical CPU engine."""
        if x.shape != y.shape:
            raise ValueError("x and y must have identical shapes")
        if x.ndim != 2 or x.shape[1] != 1:
            raise ValueError("x and y must have shape (N, 1)")
        points = torch.cat((x, y), dim=1).detach().cpu().double().numpy()
        values = self.evaluate_sdf_numpy(
            points,
            point_chunk_size=point_chunk_size,
        )
        return torch.as_tensor(
            values,
            device=x.device,
            dtype=x.dtype,
        ).view_as(x)

    def synchronize_cached_training_sdf(self) -> None:
        """Validate cached SDF provenance and replace it canonically in memory.

        The serialized cache is trusted only when its geometry fingerprint and
        method match the active segment geometry.  Irrespective of its storage
        dtype, the runtime cache is rebuilt with ``evaluate_numpy`` so training,
        RAR, FVM and inference share exactly one value engine.
        """
        if self.sdf_module is None:
            return
        dataset = getattr(self.data_handler, 'dataset', None)
        if dataset is None:
            return
        x = getattr(dataset, 'x', None)
        y = getattr(dataset, 'y', None)
        cached = getattr(dataset, 'sdf', None)
        if x is None or y is None or cached is None:
            return

        points = torch.cat((x, y), dim=1).detach().cpu().double().numpy()
        canonical = self.evaluate_sdf_numpy(points)
        cached_values = (
            cached.detach().cpu().double().numpy().reshape(-1)
        )
        if cached_values.shape != canonical.shape:
            max_error = float("inf")
        elif not np.isfinite(cached_values).all():
            max_error = float("inf")
        else:
            max_error = float(
                np.max(np.abs(cached_values - canonical), initial=0.0)
            )

        cached_fingerprint = getattr(
            self.data_handler,
            'cached_sdf_geometry_fingerprint',
            None,
        )
        cached_method = getattr(
            self.data_handler,
            'cached_sdf_method',
            None,
        )
        provenance_matches = (
            cached_fingerprint == self.sdf_geometry_fingerprint
            and cached_method == 'exact_polyline_segments'
        )
        if not provenance_matches:
            logger.warning(
                "Cached SDF provenance mismatch; rebuilding canonically: "
                "cached_geometry=%s, active_geometry=%s, method=%s",
                (
                    cached_fingerprint[:12]
                    if cached_fingerprint
                    else None
                ),
                self.sdf_geometry_fingerprint[:12],
                cached_method,
            )
        elif max_error > self.sdf_float64_consistency_tolerance:
            logger.info(
                "Cached SDF values differ from canonical float64 evaluation "
                "by %.3e; runtime cache is refreshed.",
                max_error,
            )

        dataset.sdf = torch.from_numpy(
            np.ascontiguousarray(canonical, dtype=np.float64)
        ).view(-1, 1)
        self.data_handler.cached_sdf_geometry_fingerprint = (
            self.sdf_geometry_fingerprint
        )
        self.data_handler.cached_sdf_method = 'exact_polyline_segments'
        self.data_handler.cached_sdf_max_abs_error = max_error
        metadata = getattr(self.data_handler, 'initial_metadata', None)
        if isinstance(metadata, dict):
            metadata['sdf_geometry_fingerprint'] = (
                self.sdf_geometry_fingerprint
            )
            metadata['sdf_method'] = 'exact_polyline_segments'

    def prepare_sdf_for_pde(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        cached_sdf: torch.Tensor,
        boundary_types: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compose SDF with coordinates and return its valid PDE mask.

        The exact minimum over segments is not differentiable where two
        segments are equally close.  Such tie points and wall points keep their
        cached value and are marked invalid for second-order PDE residuals.
        Interior points receive the online distance, preserving all chain-rule
        terms from the network's SDF input to ``x`` and ``y``.
        """
        if not self.feature_assembler.sdf_enabled:
            return cached_sdf, torch.ones_like(x, dtype=torch.bool)
        if self.sdf_module is None:
            raise RuntimeError(
                "SDF is enabled but no ordered airfoil polyline is available. "
                "Regenerate geometry/sampling artifacts before training."
            )
        if x.shape != y.shape or x.shape != cached_sdf.shape:
            raise ValueError("x, y and cached_sdf must have identical shapes")

        candidate_mask = torch.ones_like(x, dtype=torch.bool)
        if boundary_types is not None and boundary_types.shape[1] > 0:
            type_to_idx = getattr(
                self.data_handler.dataset,
                'type_to_idx',
                {},
            )
            if not self.pde_on_boundary and 'interior' in type_to_idx:
                interior_idx = type_to_idx['interior']
                candidate_mask = (
                    boundary_types[:, interior_idx].view(-1, 1) > 0.5
                )
            wall_idx = type_to_idx.get('wall')
            if wall_idx is not None and wall_idx < boundary_types.shape[1]:
                wall_mask = boundary_types[:, wall_idx].view(-1, 1) > 0.5
                candidate_mask = candidate_mask & ~wall_mask

        candidate_indices = torch.nonzero(
            candidate_mask.reshape(-1),
            as_tuple=False,
        ).reshape(-1)
        result = cached_sdf.clone()
        valid_mask = torch.zeros_like(x, dtype=torch.bool)
        if candidate_indices.numel() == 0:
            return result, valid_mask

        points = torch.cat((x, y), dim=1)
        candidate_points = points[candidate_indices]
        (
            nearest,
            second,
            nearest_projection,
            second_projection,
            nearest_gradient,
            second_gradient,
        ) = self.sdf_module.nearest_two_geometry(candidate_points)
        ties = self.sdf_module.tie_mask(
            nearest,
            second,
            absolute_tolerance=self.sdf_absolute_tie_tolerance,
            relative_tolerance=self.sdf_relative_tie_tolerance,
            nearest_projection=nearest_projection,
            second_projection=second_projection,
            nearest_gradient=nearest_gradient,
            second_gradient=second_gradient,
        ).detach().reshape(-1)
        finite = torch.isfinite(nearest.detach()).reshape(-1)
        safe_local = ~ties & finite
        safe_indices = candidate_indices[safe_local]

        if safe_indices.numel() > 0:
            flat_result = result.reshape(-1)
            flat_result[safe_indices] = nearest.reshape(-1)[safe_local]
            result = flat_result.reshape_as(cached_sdf)
            flat_valid = valid_mask.reshape(-1)
            flat_valid[safe_indices] = True
            valid_mask = flat_valid.reshape_as(x)

        return result, valid_mask

    def _grad(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        create_graph: bool = True,
        silent: bool = False,
    ) -> torch.Tensor:
        """Вычисляет dy/dx с учетом diagnostics.strict_autograd."""
        return safe_grad(
            y,
            x,
            create_graph=create_graph,
            silent=silent,
            strict=self.strict_autograd,
        )

    # ------------------------------------------------------------------
    # FVM-потеря (закон сохранения массы)
    # ------------------------------------------------------------------
    def compute_fvm_loss(
        self,
        u_fvm: torch.Tensor,
        u_boundary_fvm: Optional[torch.Tensor] = None,
        freestream_velocity: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Вычисляет взвешенную FVM-потерю закона сохранения массы.

        Args:
            u_fvm: ``(Q, 2)`` direct PINN velocities at internal dual-face
                segment midpoints. Legacy callers may still pass ``(N, 2)``
                nodal velocities when ``u_boundary_fvm`` is omitted.
            u_boundary_fvm: ``(B, 2)`` direct PINN velocities at physical
                boundary half-edge midpoints.
            freestream_velocity: case-local ``(cos(alpha), sin(alpha))``

        Returns:
            Configured FVM contribution ``fvm_weight * raw_fvm_loss``.
        """
        return self.compute_fvm_metrics(
            u_fvm,
            u_boundary_fvm=u_boundary_fvm,
            freestream_velocity=freestream_velocity,
        )["weighted"]

    def compute_fvm_metrics(
        self,
        u_fvm: torch.Tensor,
        u_boundary_fvm: Optional[torch.Tensor] = None,
        freestream_velocity: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Return the FVM mean-square objective and RMS diagnostic.

        ``raw`` is retained as a compatibility alias for ``mse``.  Every
        supported FVM reduction is quadratic; no square-rooted metric is ever
        fed into ``total_loss``.
        """
        if self.fvm is None:
            zero = torch.zeros((), device=self.device, dtype=self.dtype)
            return {"mse": zero, "raw": zero, "weighted": zero, "rms": zero}
        if u_boundary_fvm is None:
            # Compatibility path for analytical tests and old external
            # callers. Production training always supplies direct PINN
            # predictions for internal and boundary quadrature points.
            mse, rms = self.fvm.loss_and_divergence_rms(
                u_fvm,
                reduction=self.fvm_loss_reduction,
                freestream_velocity=freestream_velocity,
            )
        else:
            mse, rms = self.fvm.loss_and_divergence_rms_quadrature(
                u_fvm,
                u_boundary_fvm,
                reduction=self.fvm_loss_reduction,
                freestream_velocity=freestream_velocity,
            )
        return {
            "mse": mse,
            "raw": mse,
            "weighted": mse * self.fvm_weight,
            "rms": rms,
        }

    def _setup_pde_components(self) -> List[str]:
        """Возвращает список названий компонент PDE в зависимости от типа течения."""
        if self.flow_type == 'axisymmetric':
            return ['cont', 'mom_r', 'mom_z']
        else:
            return ['cont', 'mom_x', 'mom_y']

    # ------------------------------------------------------------------
    # Вычисление PDE невязок
    # ------------------------------------------------------------------
    def compute_pde_residuals(
        self,
        u: torch.Tensor,
        v: torch.Tensor,
        p: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
        case_ids: Optional[torch.Tensor] = None,
        *,
        squared: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Вычисляет невязки PDE (квадраты) для заданных полей.

        Args:
            u, v, p: поля скорости и давления (тензоры с requires_grad)
            x, y: координаты (тензоры с requires_grad)
            case_ids: идентификатор режима для каждой точки
            squared: False возвращает знаковые невязки для диагностики NTK

        Returns:
            словарь с ключами, соответствующими self.pde_components,
            значениями – тензоры невязок той же формы, что и входные поля.
        """
        if self.pde_system == 'euler':
            if self.flow_type == 'axisymmetric':
                return PDEResiduals.compute_residuals_euler_axisymmetric(
                    u, v, p, x, y, self._grad, self.eps_axi, squared=squared
                )
            return PDEResiduals.compute_residuals_euler_2d(
                u, v, p, x, y, self._grad, squared=squared
            )

        flow = self.get_flow_case_batch(x, case_ids)
        inverse_reynolds = flow.reynolds.reciprocal()
        if self.flow_type == 'axisymmetric':
            return PDEResiduals.compute_residuals_axisymmetric(
                u,
                v,
                p,
                x,
                y,
                inverse_reynolds,
                self._grad,
                self.eps_axi,
                squared=squared,
            )
        return PDEResiduals.compute_residuals_2d(
            u,
            v,
            p,
            x,
            y,
            inverse_reynolds,
            self._grad,
            squared=squared,
        )

    # ------------------------------------------------------------------
    # Граничные условия (только для soft)
    # ------------------------------------------------------------------
    def is_hard_boundary(self, boundary_name: str) -> bool:
        """Whether the resolved condition is enforced by an output transform."""
        return (
            self.boundary_conditions.get(boundary_name, {}).get(
                "enforcement", "soft"
            )
            == "hard"
        )

    def _freestream_velocity_loss(
        self,
        u: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor,
        target_u: torch.Tensor,
        target_v: torch.Tensor,
    ) -> torch.Tensor:
        return (
            torch.mean((u[mask] - target_u[mask]) ** 2)
            + torch.mean((v[mask] - target_v[mask]) ** 2)
        )

    def _boundary_normals(
        self,
        normal_x: Optional[torch.Tensor],
        normal_y: Optional[torch.Tensor],
        mask: torch.Tensor,
        *,
        boundary_name: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Validate and normalize stored normals on one boundary mask."""
        if normal_x is None or normal_y is None:
            raise RuntimeError(
                f"Boundary {boundary_name!r} requires stored normal_x and "
                "normal_y values. Regenerate collocation data with boundary "
                "normals."
            )
        if normal_x.shape != mask.shape or normal_y.shape != mask.shape:
            raise ValueError(
                "normal_x, normal_y and boundary mask must have identical "
                "shapes"
            )

        nx = normal_x[mask]
        ny = normal_y[mask]
        length = torch.sqrt(nx.square() + ny.square())
        valid = torch.isfinite(nx) & torch.isfinite(ny) & (length > 0.0)
        if not bool(valid.all()):
            invalid_count = int((~valid).sum().item())
            raise RuntimeError(
                f"Boundary {boundary_name!r} contains {invalid_count} point(s) "
                "without a finite non-zero normal. Regenerate collocation "
                "data with boundary normals."
            )
        return nx / length, ny / length

    def _normal_derivative(
        self,
        field: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
        nx: torch.Tensor,
        ny: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return ``grad(field) dot n`` at masked boundary points."""
        dfield_dx = self._grad(field, x, silent=False)[mask]
        dfield_dy = self._grad(field, y, silent=False)[mask]
        return dfield_dx * nx + dfield_dy * ny

    def _normal_velocity_gradient_loss(
        self,
        u: torch.Tensor,
        v: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
        mask: torch.Tensor,
        normal_x: Optional[torch.Tensor],
        normal_y: Optional[torch.Tensor],
        *,
        boundary_name: str,
    ) -> torch.Tensor:
        nx, ny = self._boundary_normals(
            normal_x,
            normal_y,
            mask,
            boundary_name=boundary_name,
        )
        du_dn = self._normal_derivative(u, x, y, nx, ny, mask)
        dv_dn = self._normal_derivative(v, x, y, nx, ny, mask)
        return torch.mean(du_dn.square() + dv_dn.square())

    def _slip_bc_loss(
        self,
        u: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor,
        normal_x: Optional[torch.Tensor],
        normal_y: Optional[torch.Tensor],
        *,
        boundary_name: str,
    ) -> torch.Tensor:
        nx, ny = self._boundary_normals(
            normal_x,
            normal_y,
            mask,
            boundary_name=boundary_name,
        )
        normal_velocity = u[mask] * nx + v[mask] * ny
        return torch.mean(normal_velocity.square())

    def _symmetry_bc_losses(
        self,
        u: torch.Tensor,
        v: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
        mask: torch.Tensor,
        normal_x: Optional[torch.Tensor],
        normal_y: Optional[torch.Tensor],
        *,
        boundary_name: str,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        nx, ny = self._boundary_normals(
            normal_x,
            normal_y,
            mask,
            boundary_name=boundary_name,
        )
        normal_velocity = u[mask] * nx + v[mask] * ny
        full_length = torch.sqrt(normal_x.square() + normal_y.square())
        safe_length = torch.where(
            mask,
            full_length,
            torch.ones_like(full_length),
        )
        nx_full = torch.where(mask, normal_x / safe_length, torch.zeros_like(u))
        ny_full = torch.where(mask, normal_y / safe_length, torch.zeros_like(v))
        tangential_velocity = -u * ny_full + v * nx_full
        dut_dn = self._normal_derivative(
            tangential_velocity,
            x,
            y,
            nx,
            ny,
            mask,
        )
        return torch.mean(normal_velocity.square()), torch.mean(dut_dn.square())

    def compute_wall_normals(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return unit wall normals as the one-sided gradient of the SDF.

        The exact unsigned SDF has two opposite gradients at the surface.
        ``PolylineDistanceField.surface_normals`` evaluates either one-sided
        limit from the nearest segment.  The orientation is immaterial for the
        squared slip-wall impermeability residual.
        """
        if self.sdf_module is None:
            raise RuntimeError(
                "Slip wall boundary conditions require an ordered airfoil "
                "polyline for SDF-based surface normals."
            )
        if x.shape != y.shape or x.shape != mask.shape:
            raise ValueError("x, y and wall mask must have identical shapes")

        wall_points = torch.stack((x[mask], y[mask]), dim=1)
        normals = self.sdf_module.surface_normals(wall_points)
        if not bool(torch.isfinite(normals).all()):
            raise RuntimeError("SDF-based wall normals contain non-finite values")
        return normals[:, 0], normals[:, 1]

    def compute_wall_point_loss(
        self,
        u: torch.Tensor,
        v: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """Return squared residuals of the resolved wall condition.

        Configuration normalization binds wall.kind to the PDE. All BC loss
        paths use that effective condition: no-slip penalizes both velocity
        components, while slip penalizes only penetration. Non-wall entries
        are zero; hard enforcement is excluded by the loss caller.
        """
        wall_kind = self.boundary_conditions['wall']['kind']
        point_loss = torch.zeros_like(u)
        if wall_kind == 'no_slip':
            point_loss[mask] = u[mask].square() + v[mask].square()
        elif wall_kind == 'slip_wall':
            nx, ny = self.compute_wall_normals(x, y, mask)
            point_loss[mask] = (
                u[mask] * nx + v[mask] * ny
            ).square()
        else:
            raise ValueError(f"Unsupported wall boundary kind: {wall_kind}")
        return point_loss

    def compute_bc_loss(
        self,
        u: torch.Tensor,
        v: torch.Tensor,
        p: torch.Tensor,
        boundary_types: Optional[torch.Tensor],
        x: torch.Tensor,
        y: torch.Tensor,
        case_ids: Optional[torch.Tensor] = None,
        normal_x: Optional[torch.Tensor] = None,
        normal_y: Optional[torch.Tensor] = None,
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Вычисляет потери на границах для soft boundary conditions.
        Давление Cp=0 задаётся только на outlet/farfield через mean(Cp²).
        Штрафа за среднее давление в области или на inlet нет.
        Возвращает словарь потерь по каждому типу границы и дополнительную
        потерю от градиентов на симметрии (symmetry_grad, axis_symmetry_grad).

        Args:
            u, v, p: поля скорости и давления
            boundary_types: one-hot тензор типов границ (или None)
            x, y: координаты
            case_ids: идентификатор режима для каждой точки
            normal_x, normal_y: единичные нормали граничных точек

        Returns:
            bc_comp: словарь {имя_границы: тензор потерь (скаляр)}
            extra_grad: дополнительная потеря от градиентов (скалярный тензор)
        """
        if boundary_types is None:
            return {}, torch.tensor(0.0, device=self.device, dtype=self.dtype)

        bc_comp = {name: torch.tensor(0.0, device=self.device, dtype=self.dtype)
                   for name in self.data_handler.boundary_type_names}
        extra_grad = torch.tensor(0.0, device=self.device, dtype=self.dtype)
        target_u, target_v = self.get_freestream_targets(x, case_ids)

        for idx, name in enumerate(self.data_handler.boundary_type_names):
            if self.is_hard_boundary(name):
                # The supported hard value constraint is already enforced by
                # BoundaryOutputTransform. Other boundaries remain soft.
                continue
            mask = boundary_types[:, idx].view(-1, 1) > 0.5
            if mask.sum() == 0:
                continue

            if name == 'inlet':
                bc_comp[name] += self._freestream_velocity_loss(
                    u,
                    v,
                    mask,
                    target_u,
                    target_v,
                )

            elif name in ('farfield', 'velocity_dirichlet'):
                bc_comp[name] += self._freestream_velocity_loss(
                    u,
                    v,
                    mask,
                    target_u,
                    target_v,
                )
                if name == 'farfield':
                    # Pressure is anchored only on the external boundary.
                    bc_comp[name] += p[mask].square().mean()

            elif name == 'outlet':
                p_out = p[mask]
                bc_comp[name] += torch.mean(p_out ** 2)
                outlet_grad_weight = self.loss_weights.get('outlet_grad', 0.0)
                if outlet_grad_weight > 0.0:
                    extra_grad += outlet_grad_weight * self._normal_velocity_gradient_loss(
                        u,
                        v,
                        x,
                        y,
                        mask,
                        normal_x,
                        normal_y,
                        boundary_name=name,
                    )

            elif name == 'wall':
                bc_comp[name] += torch.mean(
                    self.compute_wall_point_loss(
                        u,
                        v,
                        x,
                        y,
                        mask,
                    )[mask]
                )

            elif name in ('slip_wall', 'slip'):
                bc_comp[name] += self._slip_bc_loss(
                    u,
                    v,
                    mask,
                    normal_x,
                    normal_y,
                    boundary_name=name,
                )

            elif name == 'symmetry':
                normal_loss, tangent_grad_loss = self._symmetry_bc_losses(
                    u,
                    v,
                    x,
                    y,
                    mask,
                    normal_x,
                    normal_y,
                    boundary_name=name,
                )
                bc_comp[name] += normal_loss
                extra_grad += (
                    self.loss_weights.get('symmetry_grad', 10.0)
                    * tangent_grad_loss
                )

            elif name == 'axis_symmetry':
                bc_comp[name] += torch.mean(u[mask] ** 2)
                # dv/dr = 0 на оси: градиент по полному тензору, затем маска
                dv_dr = self._grad(v, x, silent=False)[mask]
                extra_grad += self.loss_weights.get('axis_symmetry_grad', 10.0) * torch.mean(dv_dr ** 2)

        return bc_comp, extra_grad

    # ------------------------------------------------------------------
    # Интегральное условие (баланс масс) – ИСПРАВЛЕННАЯ ВЕРСИЯ
    # ------------------------------------------------------------------
    def compute_mesh_integral_loss(
        self,
        velocity_nodes: torch.Tensor,
        bnd_node: torch.Tensor,
        bnd_area: torch.Tensor,
        bnd_type: torch.Tensor,
        node_x: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Интегральный баланс массы по FVM-граням.

        bnd_area хранит внешние векторы площадей/длин граней, поэтому сумма
        flux = u · S по всем открытым границам (inlet, outlet, farfield и
        velocity_dirichlet) является signed outflow. Для корректного баланса
        целевое значение по умолчанию равно 0.
        """
        if not self.integral_condition:
            return torch.tensor(0.0, device=self.device, dtype=self.dtype)

        if velocity_nodes.numel() == 0 or bnd_node.numel() == 0:
            return torch.tensor(0.0, device=self.device, dtype=self.dtype)

        try:
            from airfoil_pinn.physics.fvm import BC_CODES
            inlet_code = BC_CODES['inlet']
            outlet_code = BC_CODES['outlet']
            farfield_code = BC_CODES['farfield']
            velocity_code = BC_CODES['velocity_dirichlet']
        except Exception:
            inlet_code = 1
            outlet_code = 2
            farfield_code = 6
            velocity_code = 8

        device = velocity_nodes.device
        dtype = velocity_nodes.dtype
        bnd_node = bnd_node.to(device=device, dtype=torch.long)
        bnd_area = bnd_area.to(device=device, dtype=dtype)
        bnd_type = bnd_type.to(device=device, dtype=torch.long)

        mask = (
            (bnd_type == inlet_code)
            | (bnd_type == outlet_code)
            | (bnd_type == farfield_code)
            | (bnd_type == velocity_code)
        )
        if not bool(mask.any()):
            return torch.tensor(0.0, device=device, dtype=dtype)

        flux = (velocity_nodes[bnd_node] * bnd_area).sum(dim=1)
        if self.flow_type == 'axisymmetric' and node_x is not None:
            r = node_x.to(device=device, dtype=dtype).flatten()[bnd_node]
            two_pi = torch.as_tensor(self.two_pi, device=device, dtype=dtype)
            flux = flux * two_pi * torch.clamp(r, min=0.0)

        total_flux = flux[mask].sum()
        target = torch.as_tensor(self.integral_target, device=device, dtype=dtype)
        return (total_flux - target) ** 2

    def compute_boundary_quadrature_integral_loss(
        self,
        boundary_velocity: torch.Tensor,
        bnd_area: torch.Tensor,
        bnd_type: torch.Tensor,
        boundary_x: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Mass-balance loss from direct predictions at boundary midpoints."""
        if not self.integral_condition or boundary_velocity.numel() == 0:
            return torch.zeros((), device=self.device, dtype=self.dtype)
        try:
            from airfoil_pinn.physics.fvm import BC_CODES
            inlet_code = BC_CODES['inlet']
            outlet_code = BC_CODES['outlet']
            farfield_code = BC_CODES['farfield']
            velocity_code = BC_CODES['velocity_dirichlet']
        except Exception:
            inlet_code, outlet_code, farfield_code, velocity_code = 1, 2, 6, 8

        device = boundary_velocity.device
        dtype = boundary_velocity.dtype
        area = bnd_area.to(device=device, dtype=dtype)
        types = bnd_type.to(device=device, dtype=torch.long)
        if boundary_velocity.shape != area.shape:
            raise ValueError(
                "Boundary quadrature velocity and area vectors must have "
                f"the same shape, got {boundary_velocity.shape} and {area.shape}"
            )
        mask = (
            (types == inlet_code)
            | (types == outlet_code)
            | (types == farfield_code)
            | (types == velocity_code)
        )
        if not bool(mask.any()):
            return torch.zeros((), device=device, dtype=dtype)

        flux = (boundary_velocity * area).sum(dim=1)
        if self.flow_type == 'axisymmetric' and boundary_x is not None:
            radius = boundary_x.to(device=device, dtype=dtype).flatten()
            flux = flux * torch.as_tensor(
                self.two_pi,
                device=device,
                dtype=dtype,
            ) * torch.clamp(radius, min=0.0)
        total_flux = flux[mask].sum()
        target = torch.as_tensor(self.integral_target, device=device, dtype=dtype)
        return (total_flux - target).square()

    def compute_integral_loss(
        self,
        u: torch.Tensor,
        v: torch.Tensor,
        boundary_types: Optional[torch.Tensor],
        x: torch.Tensor,
        y: torch.Tensor
    ) -> torch.Tensor:
        """
        Вычисляет point-based интегральную потерю (баланс масс) по полному
        набору точек inlet/outlet. Метод не предназначен для случайного
        mini-batch: его вызывает Trainer только на полной boundary-выборке,
        если FVM-граней нет.

        Args:
            u, v: поля скорости
            boundary_types: one-hot тензор типов границ
            x, y: координаты

        Returns:
            скалярный тензор – квадрат отклонения суммарного потока от целевого значения
        """
        if not self.integral_condition or boundary_types is None:
            return torch.tensor(0.0, device=self.device, dtype=self.dtype)

        # Определяем индексы inlet и outlet
        if 'inlet' not in self.data_handler.boundary_type_names or 'outlet' not in self.data_handler.boundary_type_names:
            return torch.tensor(0.0, device=self.device, dtype=self.dtype)

        inlet_idx = self.data_handler.boundary_type_names.index('inlet')
        outlet_idx = self.data_handler.boundary_type_names.index('outlet')

        mask_inlet = boundary_types[:, inlet_idx].view(-1, 1) > 0.5
        mask_outlet = boundary_types[:, outlet_idx].view(-1, 1) > 0.5

        # Все точки, участвующие в интеграле
        mask_all = mask_inlet | mask_outlet
        if mask_all.sum() == 0:
            return torch.tensor(0.0, device=self.device, dtype=self.dtype)

        tol = self.tol
        x_min, x_max, y_min, y_max = self.x_min, self.x_max, self.y_min, self.y_max

        # Маски для четырёх сторон (с учётом допуска)
        # Разрешаем точкам принадлежать нескольким сторонам (углы будут учтены дважды,
        # что соответствует вкладу в поток через обе грани)
        left_mask = mask_all & (x < x_min + tol)
        right_mask = mask_all & (x > x_max - tol)
        bottom_mask = mask_all & (y < y_min + tol)
        top_mask = mask_all & (y > y_max - tol)

        x_flat = x.flatten()
        y_flat = y.flatten()
        u_flat = u.flatten()
        v_flat = v.flatten()
        total_flux = torch.tensor(0.0, device=self.device, dtype=self.dtype)

        def side_flux(mask: torch.Tensor, normal_velocity: torch.Tensor, tangent: torch.Tensor) -> torch.Tensor:
            flat_mask = mask.flatten()
            n = int(flat_mask.sum().item())
            if n == 0:
                return torch.tensor(0.0, device=self.device, dtype=self.dtype)

            vel = normal_velocity[flat_mask]
            tau = tangent[flat_mask]
            order = torch.argsort(tau)
            vel = vel[order]
            tau = tau[order]

            if n == 1:
                length = torch.as_tensor(
                    max(x_max - x_min, y_max - y_min),
                    device=self.device,
                    dtype=self.dtype,
                )
                weights = torch.ones_like(vel) * length
            else:
                dt = torch.clamp(tau[1:] - tau[:-1], min=0.0)
                weights = torch.zeros_like(vel)
                weights[0] = 0.5 * dt[0]
                weights[-1] = 0.5 * dt[-1]
                if n > 2:
                    weights[1:-1] = 0.5 * (dt[:-1] + dt[1:])

            if self.flow_type == 'axisymmetric':
                r = x_flat[flat_mask][order]
                two_pi = torch.as_tensor(self.two_pi, device=self.device, dtype=self.dtype)
                weights = weights * two_pi * torch.clamp(r, min=0.0)
            return torch.sum(vel * weights)

        total_flux = total_flux + side_flux(left_mask, -u_flat, y_flat)
        total_flux = total_flux + side_flux(right_mask, u_flat, y_flat)
        total_flux = total_flux + side_flux(bottom_mask, -v_flat, x_flat)
        total_flux = total_flux + side_flux(top_mask, v_flat, x_flat)

        # Потеря = (суммарный поток - целевой поток)^2
        target = torch.as_tensor(self.integral_target, device=self.device, dtype=self.dtype)
        loss = (total_flux - target) ** 2
        return loss

    # ------------------------------------------------------------------
    # Построение входного тензора для модели
    # ------------------------------------------------------------------
    def build_input_tensor(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        sdf: torch.Tensor,
        *,
        case_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Assemble the manifest-defined dimensionless network input.

        Args:
            x: Chord-normalized x coordinate.
            y: Chord-normalized y coordinate.
            sdf: Chord-normalized differentiable/cached distance.
            case_ids: Per-point flow-case identifiers.  When omitted, the
                legacy single case ``0`` is used.

        Returns:
            Tensor with exactly ``feature_manifest.input_dim`` columns.
        """
        flow = None
        if self.feature_assembler.requires_flow:
            flow = self.get_flow_case_batch(x, case_ids)
        return self.feature_assembler.build(
            x,
            y,
            sdf,
            flow,
        )

    def transform_outputs(
        self,
        raw_u: torch.Tensor,
        raw_v: torch.Tensor,
        raw_p: torch.Tensor,
        x: torch.Tensor,
        y: torch.Tensor,
        *,
        sdf: Optional[torch.Tensor] = None,
        case_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Apply graph-level hard constraints to raw model outputs."""
        flow = None
        if case_ids is not None:
            flow = self.get_flow_case_batch(x, case_ids)
        return self.boundary_transform(
            raw_u,
            raw_v,
            raw_p,
            x,
            y,
            sdf=sdf,
            flow=flow,
        )

    # ------------------------------------------------------------------
    # Применение hard граничных условий
    # ------------------------------------------------------------------
    def apply_hard_constraints(
        self,
        u_raw: torch.Tensor,
        v_raw: torch.Tensor,
        p_raw: torch.Tensor,
        mask_u: torch.Tensor,
        mask_v: torch.Tensor,
        mask_p: torch.Tensor,
        u_bc: torch.Tensor,
        v_bc: torch.Tensor,
        p_bc: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Legacy pointwise boundary override.

        This method is retained only for loading/debugging legacy runs. It is
        not a mathematical hard boundary condition and must not be used by the
        modern per-boundary training path.

        Args:
            u_raw, v_raw, p_raw: выход модели (без ограничений)
            mask_u, mask_v, mask_p: булевы маски, указывающие, где нужно заменить
            u_bc, v_bc, p_bc: значения, которые нужно подставить

        Returns:
            (u, v, p) с принудительно заданными граничными условиями
        """
        u_final = torch.where(mask_u, u_bc, u_raw)
        v_final = torch.where(mask_v, v_bc, v_raw)
        p_final = torch.where(mask_p, p_bc, p_raw)
        return u_final, v_final, p_final

    def apply_velocity_dirichlet_constraint(
        self,
        u_raw: torch.Tensor,
        v_raw: torch.Tensor,
        boundary_types: Optional[torch.Tensor],
        case_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Legacy pointwise velocity override.

        Per-boundary configurations keep ``velocity_dirichlet`` soft during
        the first hard-wall implementation. The override is retained only for
        old global-hard direct callers.
        """
        names = getattr(self.data_handler, 'boundary_type_names', [])
        if boundary_types is None or 'velocity_dirichlet' not in names:
            return u_raw, v_raw

        idx = names.index('velocity_dirichlet')
        if idx >= boundary_types.shape[1]:
            return u_raw, v_raw

        mask = boundary_types[:, idx].view(-1, 1) > 0.5
        if not bool(mask.any()):
            return u_raw, v_raw

        target_u, target_v = self.get_freestream_targets(u_raw, case_ids)
        u_final = torch.where(mask, target_u, u_raw)
        v_final = torch.where(mask, target_v, v_raw)
        return u_final, v_final

    # ------------------------------------------------------------------
    # Обновление адаптивных весов
    # ------------------------------------------------------------------
    def update_adaptive_weights(self, new_weights: Dict[str, float]) -> None:
        """
        Обновляет веса граничных условий в соответствии с адаптивной балансировкой.
        Вызывается из Trainer.update_adaptive_weights.

        Args:
            new_weights: словарь {имя_границы: новое значение веса}
        """
        for key, value in new_weights.items():
            if key in self.loss_weights:
                self.loss_weights[key] = value
            else:
                logger.warning(f"Попытка обновить неизвестный вес {key}")
        logger.debug(f"Адаптивные веса BC обновлены внутри LossComputer")
