#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fvm_residuals.py — FVM-вычислитель невязки уравнения неразрывности.

Реализует класс FVMResiduals: дискретизация закона сохранения массы для
несжимаемой жидкости методом FVM на медианной дуальной неструктурированной
сетке (mesh2d.Mesh2D). Production-обучение получает скорость прямым
предсказанием PINN в midpoint каждой прямой части медианной дуальной грани;
узловая центральная интерполяция сохранена только как legacy API;
все операции векторизованы на PyTorch без Python-циклов по граням.

Поддерживает два источника геометрии:
  - Mesh2D (mesh2d.py) — после build()
  - MeshData (data_handler.py) — загруженный из .npz файла

Пример:
    >>> mesh = Mesh2D(points, domain_coords, airfoil_coords)
    >>> mesh.build()
    >>> fvm = FVMResiduals(mesh).to('cuda')
    >>> loss = fvm.mass_loss_quadrature(u_internal_q, u_boundary_q)

Автор: Колмагоров И.Д.
Дата: 2026-05-21
"""

from __future__ import annotations

import hashlib
import logging
import warnings
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch

logger = logging.getLogger(__name__)

# ── Коды типов граничных условий ──────────────────────────────────────────────
# Используются для построения маски стенки внутри compute_residual.
# Расширьте словарь при добавлении новых типов ГУ.
BC_CODES: Dict[str, int] = {
    'wall':          0,
    'inlet':         1,
    'outlet':        2,
    'symmetry':      3,
    'axis_symmetry': 4,
    'interior':      5,
    'farfield':      6,
    'slip_wall':     7,
    'slip':          7,
    'velocity_dirichlet': 8,
}
_WALL_CODE: int          = BC_CODES['wall']
_SYMMETRY_CODE: int      = BC_CODES['symmetry']
_AXIS_SYMMETRY_CODE: int = BC_CODES['axis_symmetry']
_SLIP_CODE: int          = BC_CODES['slip']

# Нижняя граница объёма КО — защита от деления на ноль при вырожденной сетке.
# Значение много меньше любого физически реального объёма (< 1e-10 м²).
__all__ = [
    "BC_CODES",
    "FVMResiduals",
]

_GEOMETRY_CLOSURE_RTOL: float = 1e-8
_GEOMETRY_CLOSURE_ATOL: float = 1e-12


class FVMResiduals:
    """
    Вычислитель невязки R_i = Φ_i / V_i для уравнения неразрывности ∇·u = 0.

    Геометрические тензоры предобрабатываются однократно в конструкторе.
    На каждом шаге обучения вызывается только compute_residual / mass_loss.

    Аргументы:
        mesh: Mesh2D (mesh2d.py) или MeshData (data_handler.py)

    Атрибуты:
        N (int): число узлов
        M (int): число внутренних граней
        B (int): число граничных граней
        V        (N,):    объёмы КО [м²]
        edge_i   (M,):    индексы узлов i внутренних граней
        edge_j   (M,):    индексы узлов j внутренних граней
        edge_S   (M, 2):  векторы площадей S_ij (outward от i)
        bnd_node (B,):    индексы граничных узлов
        bnd_S    (B, 2):  векторы площадей граничных граней (outward)
        bnd_type (B,):    коды типов ГУ (0=wall, 1=inlet, ...)
    """

    def __init__(self, mesh) -> None:
        self._device = torch.device('cpu')
        self._dtype = torch.float64
        self._geometry_is_dimensionless = False
        self._geometry_length_scale = 1.0
        self._legacy_volume_reduction_warned = False
        self._build_from_mesh(mesh)

    # ── Инициализация ─────────────────────────────────────────────────────────

    def _build_from_mesh(self, mesh) -> None:
        cv, ei, ej, eS, bn, bS, bt_strs = (
            self._extract_mesh2d(mesh)
            if hasattr(mesh, 'internal_faces')
            else self._extract_mesh_data(mesh)
        )

        cv_array = np.asarray(cv, dtype=np.float64)
        edge_i_array = np.asarray(ei, dtype=np.int64)
        edge_j_array = np.asarray(ej, dtype=np.int64)
        edge_s_array = np.asarray(eS, dtype=np.float64)
        bnd_node_array = np.asarray(bn, dtype=np.int64)
        bnd_s_array = np.asarray(bS, dtype=np.float64)
        boundary_types = [str(value).strip().lower() for value in bt_strs]
        self._validate_geometry_arrays(
            cv_array,
            edge_i_array,
            edge_j_array,
            edge_s_array,
            bnd_node_array,
            bnd_s_array,
            boundary_types,
        )

        self.N = len(cv_array)
        self.M = len(edge_i_array)
        self.B = len(bnd_node_array)
        self._mesh_fingerprint = self._fingerprint_geometry(
            cv_array,
            edge_i_array,
            edge_j_array,
            edge_s_array,
            bnd_node_array,
            bnd_s_array,
            boundary_types,
        )

        self.V = torch.tensor(cv_array, dtype=self._dtype)
        self.edge_i = torch.tensor(edge_i_array, dtype=torch.long)
        self.edge_j = torch.tensor(edge_j_array, dtype=torch.long)
        self.edge_S = torch.tensor(edge_s_array, dtype=self._dtype)
        self.bnd_node = torch.tensor(bnd_node_array, dtype=torch.long)
        self.bnd_S = torch.tensor(bnd_s_array, dtype=self._dtype)
        bnd_codes = np.fromiter(
            (BC_CODES[t] for t in boundary_types),
            dtype=np.int64, count=self.B,
        )
        self.bnd_type = torch.tensor(bnd_codes, dtype=torch.long)

        if self.M > 0:
            quad_i, quad_j, quad_s, quad_xy, bnd_quad_xy = (
                self._extract_quadrature(mesh)
            )
        else:
            quad_i = np.empty(0, dtype=np.int64)
            quad_j = np.empty(0, dtype=np.int64)
            quad_s = np.empty((0, 2), dtype=np.float64)
            quad_xy = np.empty((0, 2), dtype=np.float64)
            bnd_quad_raw = getattr(mesh, "bnd_quad_coords", None)
            bnd_quad_xy = (
                np.asarray(bnd_quad_raw, dtype=np.float64).reshape(-1, 2)
                if bnd_quad_raw is not None
                else np.empty((0, 2), dtype=np.float64)
            )
        self._validate_quadrature_arrays(
            quad_i,
            quad_j,
            quad_s,
            edge_i_array,
            edge_j_array,
            edge_s_array,
        )
        if quad_xy.shape != (len(quad_i), 2):
            raise ValueError(
                "FVM internal quadrature coordinates do not match segment "
                f"count: {quad_xy.shape} != ({len(quad_i)}, 2)"
            )
        if bnd_quad_xy.size and bnd_quad_xy.shape != (self.B, 2):
            raise ValueError(
                "FVM boundary quadrature coordinates do not match boundary "
                f"face count: {bnd_quad_xy.shape} != ({self.B}, 2)"
            )
        if not np.isfinite(quad_xy).all() or not np.isfinite(bnd_quad_xy).all():
            raise ValueError("FVM quadrature coordinates contain NaN or Inf")
        self.Q = len(quad_i)
        self.quad_i = torch.tensor(quad_i, dtype=torch.long)
        self.quad_j = torch.tensor(quad_j, dtype=torch.long)
        self.quad_S = torch.tensor(quad_s, dtype=self._dtype)
        quadrature_digest = hashlib.sha256()
        quadrature_digest.update(self._mesh_fingerprint.encode("ascii"))
        for name, values, dtype in (
            ("quad_i", quad_i, "<i8"),
            ("quad_j", quad_j, "<i8"),
            ("quad_s", quad_s, "<f8"),
            ("quad_xy", quad_xy, "<f8"),
            ("bnd_quad_xy", bnd_quad_xy, "<f8"),
        ):
            array = np.ascontiguousarray(values, dtype=np.dtype(dtype))
            quadrature_digest.update(name.encode("ascii"))
            quadrature_digest.update(array.tobytes())
        self._mesh_fingerprint = quadrature_digest.hexdigest()

        # Предписанные скорости на гранях (hard BC).
        # По умолчанию — мягкое условие: production использует прямое
        # предсказание PINN в midpoint граничной полуграни.
        # Заполняются через set_bc_velocity() / set_inlet_velocity().
        self._u_prescribed    = torch.zeros(self.B, 2, dtype=self._dtype)
        self._prescribed_mask = torch.zeros(self.B, 1, dtype=torch.bool)

        # Маска граней с нулевым нормальным потоком: wall (no-slip) и
        # symmetry/axis_symmetry (непротекание u·n = 0). Предвычислена
        # однократно — используется в каждом вызове compute_residual.
        if self.B > 0:
            self._zero_flux_mask = (
                (self.bnd_type == _WALL_CODE)
                | (self.bnd_type == _SYMMETRY_CODE)
                | (self.bnd_type == _AXIS_SYMMETRY_CODE)
                | (self.bnd_type == _SLIP_CODE)
            ).unsqueeze(1)                                   # (B, 1)
        else:
            self._zero_flux_mask = torch.zeros(0, 1, dtype=torch.bool)

        logger.info(
            "FVMResiduals: N=%d, M=%d, Q=%d, B=%d",
            self.N,
            self.M,
            self.Q,
            self.B,
        )

    @staticmethod
    def _validate_quadrature_arrays(
        quad_i: np.ndarray,
        quad_j: np.ndarray,
        quad_s: np.ndarray,
        edge_i: np.ndarray,
        edge_j: np.ndarray,
        edge_s: np.ndarray,
    ) -> None:
        """Validate direct-PINN segment quadrature and its face sums."""
        if quad_i.ndim != 1 or quad_j.ndim != 1 or quad_i.shape != quad_j.shape:
            raise ValueError("FVM quadrature indices must be equal 1-D arrays")
        if quad_s.shape != (quad_i.size, 2):
            raise ValueError(
                "FVM quadrature area vectors must have shape "
                f"({quad_i.size}, 2), received {quad_s.shape}"
            )
        if edge_i.size and quad_i.size == 0:
            raise ValueError("FVM mesh must contain internal face quadrature")
        if not np.isfinite(quad_s).all():
            raise ValueError("FVM quadrature area vectors contain NaN or Inf")
        if np.any(np.linalg.norm(quad_s, axis=1) <= 0.0):
            raise ValueError("FVM quadrature area vectors must be non-zero")
        n_nodes = (
            int(max(edge_i.max(), edge_j.max())) + 1
            if edge_i.size else 0
        )
        if (
            np.any(quad_i < 0)
            or np.any(quad_j < 0)
            or np.any(quad_i >= n_nodes)
            or np.any(quad_j >= n_nodes)
        ):
            raise ValueError("FVM quadrature node indices are out of range")

        face_lookup = {
            (int(i), int(j)): face_id
            for face_id, (i, j) in enumerate(zip(edge_i, edge_j))
        }
        sums = np.zeros_like(edge_s)
        counts = np.zeros(len(edge_i), dtype=np.int64)
        for i, j, area in zip(quad_i, quad_j, quad_s):
            face_id = face_lookup.get((int(i), int(j)))
            if face_id is None:
                raise ValueError(
                    "FVM quadrature references an unknown oriented face "
                    f"({int(i)}, {int(j)})"
                )
            sums[face_id] += area
            counts[face_id] += 1
        if np.any((counts < 1) | (counts > 2)):
            raise ValueError(
                "Every FVM internal face must contain one or two quadrature "
                "segments"
            )
        if not edge_i.size:
            return
        scale = max(float(np.linalg.norm(edge_s, axis=1).max()), 1.0)
        error = float(np.linalg.norm(sums - edge_s, axis=1).max())
        if error > max(1.0e-12, 1.0e-9 * scale):
            raise ValueError(
                "FVM quadrature segment vectors do not reproduce aggregated "
                f"faces: max error={error:.6e}"
            )

    @staticmethod
    def _validate_geometry_arrays(
        volumes: np.ndarray,
        edge_i: np.ndarray,
        edge_j: np.ndarray,
        edge_s: np.ndarray,
        bnd_node: np.ndarray,
        bnd_s: np.ndarray,
        boundary_types: list[str],
    ) -> None:
        """Fail fast when a mesh cannot define a conservative FVM operator."""
        if volumes.ndim != 1 or volumes.size == 0:
            raise ValueError("FVM control volumes must be a non-empty 1-D array")
        if not np.isfinite(volumes).all():
            raise ValueError("FVM volumes contain NaN or Inf")
        if np.any(volumes <= 0.0):
            raise ValueError("FVM control volumes must be strictly positive")

        n_nodes = int(volumes.size)
        if edge_i.ndim != 1 or edge_j.ndim != 1:
            raise ValueError("FVM internal face indices must be 1-D arrays")
        if edge_i.shape != edge_j.shape:
            raise ValueError("FVM internal face index arrays have different lengths")
        if edge_s.shape != (edge_i.size, 2):
            raise ValueError(
                "FVM internal area vectors must have shape "
                f"({edge_i.size}, 2), received {edge_s.shape}"
            )
        if bnd_node.ndim != 1:
            raise ValueError("FVM boundary face indices must be a 1-D array")
        if bnd_s.shape != (bnd_node.size, 2):
            raise ValueError(
                "FVM boundary area vectors must have shape "
                f"({bnd_node.size}, 2), received {bnd_s.shape}"
            )
        if len(boundary_types) != bnd_node.size:
            raise ValueError(
                "FVM boundary type count does not match boundary face count"
            )
        if bnd_node.size == 0:
            raise ValueError("FVM mesh must contain boundary faces")

        for name, indices in (
            ("internal face i", edge_i),
            ("internal face j", edge_j),
            ("boundary face", bnd_node),
        ):
            if np.any(indices < 0) or np.any(indices >= n_nodes):
                raise ValueError(
                    f"FVM {name} indices must be in [0, {n_nodes})"
                )
        if np.any(edge_i == edge_j):
            raise ValueError("FVM internal faces cannot connect a node to itself")

        all_area_vectors = (
            np.vstack((edge_s, bnd_s))
            if edge_s.size
            else bnd_s
        )
        if not np.isfinite(all_area_vectors).all():
            raise ValueError("FVM area vectors contain NaN or Inf")
        if np.any(np.linalg.norm(all_area_vectors, axis=1) <= 0.0):
            raise ValueError("FVM area vectors must be non-zero")

        unknown_types = sorted(set(boundary_types) - set(BC_CODES))
        if unknown_types:
            raise ValueError(
                "FVM boundary faces contain unsupported types: "
                + ", ".join(unknown_types)
            )

        incident_faces = np.zeros(n_nodes, dtype=np.int64)
        np.add.at(incident_faces, edge_i, 1)
        np.add.at(incident_faces, edge_j, 1)
        np.add.at(incident_faces, bnd_node, 1)
        if np.any(incident_faces == 0):
            orphan_count = int(np.count_nonzero(incident_faces == 0))
            raise ValueError(
                f"FVM mesh contains {orphan_count} control volumes without faces"
            )

        closure = np.zeros((n_nodes, 2), dtype=np.float64)
        np.add.at(closure, edge_i, edge_s)
        np.add.at(closure, edge_j, -edge_s)
        np.add.at(closure, bnd_node, bnd_s)
        max_error = float(np.linalg.norm(closure, axis=1).max())
        area_scale = float(np.linalg.norm(all_area_vectors, axis=1).max())
        tolerance = max(
            _GEOMETRY_CLOSURE_ATOL,
            _GEOMETRY_CLOSURE_RTOL * area_scale,
        )
        if max_error > tolerance:
            raise ValueError(
                "FVM control-volume area vectors do not close: "
                f"max norm {max_error:.6e} exceeds {tolerance:.6e}"
            )

    @staticmethod
    def _fingerprint_geometry(
        volumes: np.ndarray,
        edge_i: np.ndarray,
        edge_j: np.ndarray,
        edge_s: np.ndarray,
        bnd_node: np.ndarray,
        bnd_s: np.ndarray,
        boundary_types: list[str],
    ) -> str:
        """Return a stable identity of the canonical dimensional mesh arrays."""
        digest = hashlib.sha256()
        for name, values, dtype in (
            ("volumes", volumes, "<f8"),
            ("edge_i", edge_i, "<i8"),
            ("edge_j", edge_j, "<i8"),
            ("edge_s", edge_s, "<f8"),
            ("bnd_node", bnd_node, "<i8"),
            ("bnd_s", bnd_s, "<f8"),
        ):
            array = np.ascontiguousarray(values, dtype=np.dtype(dtype))
            digest.update(name.encode("ascii"))
            digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
            digest.update(array.tobytes())
        digest.update("\0".join(boundary_types).encode("utf-8"))
        return digest.hexdigest()

    def _validate_velocity(self, u: torch.Tensor) -> None:
        """Проверяет скорость перед FVM-операциями, чтобы ошибки были явными."""
        if not torch.is_tensor(u):
            raise TypeError("FVMResiduals.compute_residual: u должен быть torch.Tensor")
        if u.ndim != 2 or u.shape != (self.N, 2):
            raise ValueError(
                "FVMResiduals.compute_residual: u должен иметь форму "
                f"({self.N}, 2), получено {tuple(u.shape)}"
            )
        geometry_device = self.V.device
        if u.device != geometry_device:
            raise ValueError(
                "FVMResiduals.compute_residual: device u не совпадает с геометрией "
                f"({u.device} != {geometry_device}). Вызовите fvm.to(u.device, u.dtype)."
            )
        if u.dtype != self._dtype:
            raise ValueError(
                "FVMResiduals.compute_residual: dtype u не совпадает с геометрией "
                f"({u.dtype} != {self._dtype}). Вызовите fvm.to(u.device, u.dtype)."
            )

    def _validate_tensor_geometry(self) -> None:
        """Reject invalid values introduced by dtype conversion or scaling."""
        if not torch.isfinite(self.V).all():
            raise ValueError("FVM volumes contain NaN or Inf")
        if torch.any(self.V <= 0):
            raise ValueError("FVM control volumes must be strictly positive")
        if (
            not torch.isfinite(self.edge_S).all()
            or not torch.isfinite(self.quad_S).all()
            or not torch.isfinite(self.bnd_S).all()
        ):
            raise ValueError("FVM area vectors contain NaN or Inf")
        if (
            (self.M > 0 and torch.any(self.edge_S.norm(dim=1) <= 0))
            or (self.Q > 0 and torch.any(self.quad_S.norm(dim=1) <= 0))
            or (self.B > 0 and torch.any(self.bnd_S.norm(dim=1) <= 0))
        ):
            raise ValueError("FVM area vectors must be non-zero")

    def _compute_residual(
        self,
        u: torch.Tensor,
        apply_boundary_conditions: bool,
        freestream_velocity: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Evaluate the legacy nodal finite-volume residual.

        ``freestream_velocity`` overrides stored values on ``inlet`` and
        ``velocity_dirichlet`` faces without mutating the FVM object.  Unlike
        :meth:`set_bc_velocity`, this path preserves autograd with respect to
        a continuous flow parameter.
        """
        self._validate_velocity(u)
        res = u.new_zeros(self.N)

        if self.M > 0:
            u_i = u[self.edge_i]
            u_j = u[self.edge_j]
            u_face = 0.5 * (u_i + u_j)
            flux = (u_face * self.edge_S).sum(dim=1)

            res = res.index_add(0, self.edge_i, flux)
            res = res.index_add(0, self.edge_j, -flux)

        if self.B > 0:
            u_b = u[self.bnd_node]
            if apply_boundary_conditions:
                u_face_b = torch.where(self._prescribed_mask, self._u_prescribed, u_b)
                if freestream_velocity is not None:
                    velocity = torch.as_tensor(
                        freestream_velocity,
                        device=u.device,
                        dtype=u.dtype,
                    )
                    if velocity.shape == (2,):
                        velocity = velocity.view(1, 2)
                    if velocity.shape != (1, 2):
                        raise ValueError(
                            "freestream_velocity must have shape (2,) or "
                            f"(1, 2), received {tuple(velocity.shape)}"
                        )
                    freestream_mask = (
                        (self.bnd_type == BC_CODES["inlet"])
                        | (
                            self.bnd_type
                            == BC_CODES["velocity_dirichlet"]
                        )
                    ).view(-1, 1)
                    u_face_b = torch.where(
                        freestream_mask,
                        velocity.expand(self.B, 2),
                        u_face_b,
                    )
                u_face_b = u_face_b.masked_fill(self._zero_flux_mask, 0.0)
            else:
                u_face_b = u_b
            flux_b = (u_face_b * self.bnd_S).sum(dim=1)
            res = res.index_add(0, self.bnd_node, flux_b)

        return res / self.V

    @staticmethod
    def _extract_mesh2d(mesh) -> Tuple:
        """Извлекает массивы из объекта Mesh2D."""
        cv = mesh.cv_volumes

        ifaces = mesh.internal_faces
        if ifaces:
            ei = np.array([f.node_i      for f in ifaces], dtype=np.int64)
            ej = np.array([f.node_j      for f in ifaces], dtype=np.int64)
            eS = np.array([f.area_vector for f in ifaces], dtype=np.float64)
        else:
            ei = np.empty(0, dtype=np.int64)
            ej = np.empty(0, dtype=np.int64)
            eS = np.empty((0, 2), dtype=np.float64)

        bfaces = mesh.boundary_faces
        if bfaces:
            bn      = np.array([f.node         for f in bfaces], dtype=np.int64)
            bS      = np.array([f.area_vector  for f in bfaces], dtype=np.float64)
            bt_strs = [f.boundary_type for f in bfaces]
        else:
            bn      = np.empty(0, dtype=np.int64)
            bS      = np.empty((0, 2), dtype=np.float64)
            bt_strs = []

        return cv, ei, ej, eS, bn, bS, bt_strs

    @staticmethod
    def _extract_mesh_data(md) -> Tuple:
        """Извлекает массивы из объекта MeshData (data_handler.py)."""
        return (
            md.cv_volumes,
            md.int_face_node_i.astype(np.int64),
            md.int_face_node_j.astype(np.int64),
            md.int_face_area.astype(np.float64),
            md.bnd_face_node.astype(np.int64),
            md.bnd_face_area.astype(np.float64),
            list(md.bnd_face_type.astype(str)),
        )

    @staticmethod
    def _extract_quadrature(
        mesh,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Extract or deterministically rebuild median-dual segment data."""
        quad_i = getattr(mesh, "int_quad_node_i", None)
        quad_j = getattr(mesh, "int_quad_node_j", None)
        quad_s = getattr(mesh, "int_quad_area", None)
        quad_xy = getattr(mesh, "int_quad_coords", None)
        bnd_quad_xy = getattr(mesh, "bnd_quad_coords", None)
        if (
            quad_i is not None
            and quad_j is not None
            and quad_s is not None
            and quad_xy is not None
            and bnd_quad_xy is not None
        ):
            return (
                np.asarray(quad_i, dtype=np.int64),
                np.asarray(quad_j, dtype=np.int64),
                np.asarray(quad_s, dtype=np.float64).reshape(-1, 2),
                np.asarray(quad_xy, dtype=np.float64).reshape(-1, 2),
                np.asarray(bnd_quad_xy, dtype=np.float64).reshape(-1, 2),
            )

        if not hasattr(mesh, "internal_faces"):
            raise ValueError(
                "FVM mesh does not contain direct-PINN face quadrature"
            )

        from airfoil_pinn.sampling.mesh.core import (
            build_dual_face_quadrature,
        )

        faces = mesh.internal_faces
        boundary_faces = mesh.boundary_faces
        quadrature = build_dual_face_quadrature(
            mesh.coords,
            mesh.cells,
            np.asarray([face.node_i for face in faces], dtype=np.int64),
            np.asarray([face.node_j for face in faces], dtype=np.int64),
            np.asarray(
                [face.area_vector for face in faces],
                dtype=np.float64,
            ).reshape(-1, 2),
            np.asarray(
                [face.node for face in boundary_faces],
                dtype=np.int64,
            ),
            np.asarray(
                [face.neighbor_node for face in boundary_faces],
                dtype=np.int64,
            ),
        )
        return (
            quadrature["int_quad_node_i"],
            quadrature["int_quad_node_j"],
            quadrature["int_quad_area"],
            quadrature["int_quad_coords"],
            quadrature["bnd_quad_coords"],
        )

    # ── Основные методы ───────────────────────────────────────────────────────

    def compute_residual(
        self,
        u: torch.Tensor,
        freestream_velocity: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Legacy-невязка по узловым скоростям R_i = Φ_i / V_i для каждого КО.

        Алгоритм:
          - Центральная схема на внутренних гранях: u_face = 0.5*(u_i + u_j).
          - Граничные грани по типу:
              wall           : flux = 0  (no-slip, u = 0)
              symmetry       : flux = 0  (непротекание u·n = 0; для уравнения
              axis_symmetry    неразрывности эквивалентно стенке)
              inlet          : flux = u_prescribed · S  (hard BC через set_inlet_velocity)
              outlet/interior: flux = u_узла · S  (мягкое условие из нейросети)
          - Нормировка на объём КО.

        Аргументы:
            u: скорость в узлах, форма (N, 2)

        Возвращает:
            R: тензор невязок (N,), [с⁻¹]
        """
        return self._compute_residual(
            u,
            apply_boundary_conditions=True,
            freestream_velocity=freestream_velocity,
        )

    def _validate_quadrature_velocity(
        self,
        internal_velocity: torch.Tensor,
        boundary_velocity: torch.Tensor,
    ) -> None:
        """Validate velocities predicted directly at face quadrature points."""
        for name, value, expected in (
            ("internal", internal_velocity, (self.Q, 2)),
            ("boundary", boundary_velocity, (self.B, 2)),
        ):
            if not torch.is_tensor(value):
                raise TypeError(f"FVM {name} quadrature velocity must be a tensor")
            if value.ndim != 2 or tuple(value.shape) != expected:
                raise ValueError(
                    f"FVM {name} quadrature velocity must have shape "
                    f"{expected}, received {tuple(value.shape)}"
                )
            if value.device != self.V.device or value.dtype != self._dtype:
                raise ValueError(
                    f"FVM {name} quadrature velocity device/dtype does not "
                    "match the operator; call fvm.to(device, dtype)"
                )

    def compute_residual_quadrature(
        self,
        internal_velocity: torch.Tensor,
        boundary_velocity: torch.Tensor,
        freestream_velocity: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute divergence from direct PINN predictions on face segments.

        No nodal interpolation is performed.  ``internal_velocity[q]`` is the
        network value at the midpoint of one straight median-dual segment.
        ``boundary_velocity[b]`` is the value at the midpoint of the physical
        boundary half-edge represented by boundary face ``b``.
        """
        self._validate_quadrature_velocity(
            internal_velocity,
            boundary_velocity,
        )
        res = internal_velocity.new_zeros(self.N)

        if self.Q > 0:
            flux = (internal_velocity * self.quad_S).sum(dim=1)
            res = res.index_add(0, self.quad_i, flux)
            res = res.index_add(0, self.quad_j, -flux)

        if self.B > 0:
            u_face_b = self.apply_boundary_velocity(
                boundary_velocity,
                freestream_velocity=freestream_velocity,
            )
            boundary_flux = (u_face_b * self.bnd_S).sum(dim=1)
            res = res.index_add(0, self.bnd_node, boundary_flux)

        return res / self.V

    def apply_boundary_velocity(
        self,
        predicted_velocity: torch.Tensor,
        freestream_velocity: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply the exact boundary-flux policy to midpoint predictions."""
        if predicted_velocity.shape != (self.B, 2):
            raise ValueError(
                "FVM boundary velocity must have shape "
                f"({self.B}, 2), received {tuple(predicted_velocity.shape)}"
            )
        u_face_b = torch.where(
            self._prescribed_mask,
            self._u_prescribed,
            predicted_velocity,
        )
        if freestream_velocity is not None:
            velocity = torch.as_tensor(
                freestream_velocity,
                device=predicted_velocity.device,
                dtype=predicted_velocity.dtype,
            )
            if velocity.shape == (2,):
                velocity = velocity.view(1, 2)
            if velocity.shape != (1, 2):
                raise ValueError(
                    "freestream_velocity must have shape (2,) or "
                    f"(1, 2), received {tuple(velocity.shape)}"
                )
            freestream_mask = (
                (self.bnd_type == BC_CODES["inlet"])
                | (self.bnd_type == BC_CODES["velocity_dirichlet"])
            ).view(-1, 1)
            u_face_b = torch.where(
                freestream_mask,
                velocity.expand(self.B, 2),
                u_face_b,
            )
        return u_face_b.masked_fill(self._zero_flux_mask, 0.0)

    def mass_loss_quadrature(
        self,
        internal_velocity: torch.Tensor,
        boundary_velocity: torch.Tensor,
        reduction: str = "divergence_l2_volume",
        freestream_velocity: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return the FVM loss using direct face-quadrature predictions."""
        residual = self.compute_residual_quadrature(
            internal_velocity,
            boundary_velocity,
            freestream_velocity=freestream_velocity,
        )
        return self._reduce_residual(residual, reduction)

    def mass_loss(
        self,
        u: torch.Tensor,
        reduction: str = "divergence_l2_volume",
        freestream_velocity: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Reduce the discrete mass residual to a mean-square objective.

        ``divergence_l2_volume`` evaluates
        ``sum(V_i * R_i**2) / sum(V_i)``. ``flux_balance_l2`` evaluates
        ``sum(Phi_i**2) / sum(V_i)``. The deprecated ``volume_weighted``
        name retains the latter formula for old experiment reproducibility.
        """
        residual = self.compute_residual(
            u,
            freestream_velocity=freestream_velocity,
        )
        return self._reduce_residual(residual, reduction)

    def _reduce_residual(
        self,
        residual: torch.Tensor,
        reduction: str,
    ) -> torch.Tensor:
        """Reduce one already-computed discrete divergence field."""
        mode = str(reduction or "legacy").strip().lower().replace("-", "_")

        volumes = self.V.to(device=residual.device, dtype=residual.dtype)
        denom = volumes.sum()
        flux_balance = residual * volumes

        if mode in {"legacy", "mean", "node_mean", "unweighted"}:
            return torch.mean(residual.square())

        if mode in {
            "divergence_l2_volume",
            "divergence_volume",
            "divergence_volume_weighted",
            "old_volume_weighted",
        }:
            return torch.sum(volumes * residual.square()) / denom

        if mode in {
            "volume",
            "volume_weighted",
            "cell_volume",
            "flux_balance_l2",
            "flux",
            "flux_volume",
            "flux_weighted",
            "balance",
        }:
            if (
                mode in {"volume", "volume_weighted", "cell_volume"}
                and not self._legacy_volume_reduction_warned
            ):
                warnings.warn(
                    "FVM reduction 'volume_weighted' is a deprecated legacy "
                    "flux-balance objective; use 'divergence_l2_volume' for "
                    "the physical divergence norm or 'flux_balance_l2' for "
                    "explicit diagnostics.",
                    DeprecationWarning,
                    stacklevel=2,
                )
                self._legacy_volume_reduction_warned = True
            return torch.sum(flux_balance.square()) / denom

        if mode in {"flux_mean", "mean_flux"}:
            return torch.mean(flux_balance.square())

        raise ValueError(
            "FVMResiduals.mass_loss: unknown reduction "
            f"{reduction!r}. Use 'divergence_l2_volume', "
            "'flux_balance_l2', 'volume_weighted' (legacy), or 'legacy'."
        )

    def loss_and_divergence_rms(
        self,
        u: torch.Tensor,
        reduction: str = "divergence_l2_volume",
        freestream_velocity: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute the selected mean-square loss and physical RMS diagnostic."""
        residual = self.compute_residual(
            u,
            freestream_velocity=freestream_velocity,
        )
        raw = self._reduce_residual(residual, reduction)
        divergence_l2 = self._reduce_residual(
            residual,
            "divergence_l2_volume",
        )
        return raw, torch.sqrt(divergence_l2.clamp_min(0.0))

    def loss_and_divergence_rms_quadrature(
        self,
        internal_velocity: torch.Tensor,
        boundary_velocity: torch.Tensor,
        reduction: str = "divergence_l2_volume",
        freestream_velocity: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return direct-quadrature FVM MSE and physical divergence RMS."""
        residual = self.compute_residual_quadrature(
            internal_velocity,
            boundary_velocity,
            freestream_velocity=freestream_velocity,
        )
        raw = self._reduce_residual(residual, reduction)
        divergence_l2 = self._reduce_residual(
            residual,
            "divergence_l2_volume",
        )
        return raw, torch.sqrt(divergence_l2.clamp_min(0.0))

    def divergence_rms(
        self,
        u: torch.Tensor,
        freestream_velocity: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return the volume-weighted RMS discrete divergence."""
        _, rms = self.loss_and_divergence_rms(
            u,
            reduction="divergence_l2_volume",
            freestream_velocity=freestream_velocity,
        )
        return rms

    # ── Вспомогательные методы ────────────────────────────────────────────────

    def check_geometry_conservativity(self, tol: float = 1e-8) -> bool:
        """
        Проверяет замыкание векторов площадей для каждого контрольного объёма.

        Этот тест не использует скорости и не применяет физические граничные
        условия. Он отвечает только на вопрос: сумма всех outward-векторов S
        вокруг каждого CV равна нулю?
        """
        flux_sum = torch.zeros(self.N, 2, device=self._device, dtype=self._dtype)
        if self.Q > 0:
            flux_sum = flux_sum.index_add(0, self.quad_i, self.quad_S)
            flux_sum = flux_sum.index_add(0, self.quad_j, -self.quad_S)
        if self.B > 0:
            flux_sum = flux_sum.index_add(0, self.bnd_node, self.bnd_S)

        max_norm = float(flux_sum.norm(dim=1).max()) if self.N else 0.0
        ok = max_norm < tol
        (logger.info if ok else logger.warning)(
            "FVMResiduals geometry conservativity %s: max||sum(S)||=%.3e (tol=%.3e)",
            "OK" if ok else "FAIL", max_norm, tol,
        )
        return ok

    def check_conservativity(
        self,
        u_const: Optional[torch.Tensor] = None,
        tol: float = 1e-6,
    ) -> bool:
        """
        Верификация геометрии через постоянное поле скорости.

        Физические BC намеренно не применяются: wall/symmetry/inlet меняют
        постоянное поле и превращают геометрическую проверку в проверку
        постановки ГУ. Для проверки рабочей FVM-невязки используйте
        compute_residual().

        Аргументы:
            u_const: тестовое поле (N, 2); None → однородный поток (1, 0)
            tol: допуск

        Возвращает:
            True если max|R_i| < tol
        """
        if u_const is None:
            u_const = torch.ones(self.N, 2, dtype=self._dtype, device=self._device)
        else:
            u_const = u_const.to(device=self._device, dtype=self._dtype)

        with torch.no_grad():
            R = self._compute_residual(u_const, apply_boundary_conditions=False)

        max_res = float(R.abs().max())
        ok = max_res < tol
        (logger.info if ok else logger.warning)(
            "FVMResiduals conservativity %s: max|R|=%.3e (tol=%.3e)",
            "OK" if ok else "FAIL", max_res, tol,
        )
        return ok

    def summary(self) -> Dict:
        """Статистика по сетке и потокам (не требует u)."""
        prescribed_types = []
        if self.B > 0:
            pm = self._prescribed_mask.squeeze(1)
            for name, code in BC_CODES.items():
                if (pm & (self.bnd_type == code)).any():
                    prescribed_types.append(name)
        return {
            'N': self.N,
            'M': self.M,
            'Q': self.Q,
            'B': self.B,
            'mesh_fingerprint': self._mesh_fingerprint,
            'device': str(self._device),
            'geometry_is_dimensionless': self._geometry_is_dimensionless,
            'geometry_length_scale': self._geometry_length_scale,
            'cv_volume_min':  float(self.V.min()),
            'cv_volume_max':  float(self.V.max()),
            'cv_volume_mean': float(self.V.mean()),
            'edge_S_norm_min':  float(self.edge_S.norm(dim=1).min()) if self.M > 0 else 0.0,
            'edge_S_norm_max':  float(self.edge_S.norm(dim=1).max()) if self.M > 0 else 0.0,
            'quad_S_norm_min':  float(self.quad_S.norm(dim=1).min()) if self.Q > 0 else 0.0,
            'quad_S_norm_max':  float(self.quad_S.norm(dim=1).max()) if self.Q > 0 else 0.0,
            'bnd_S_norm_min':   float(self.bnd_S.norm(dim=1).min()) if self.B > 0 else 0.0,
            'bnd_S_norm_max':   float(self.bnd_S.norm(dim=1).max()) if self.B > 0 else 0.0,
            'bnd_type_counts':  {
                name: int((self.bnd_type == code).sum())
                for name, code in BC_CODES.items()
                if int((self.bnd_type == code).sum()) > 0
            },
            'prescribed_bc_types': prescribed_types,
        }

    @property
    def mesh_fingerprint(self) -> str:
        """Stable SHA-256 identity of the dimensional mesh arrays."""
        return self._mesh_fingerprint

    # ── Предписанные граничные условия ────────────────────────────────────────

    def set_bc_velocity(
        self,
        boundary_type: str,
        u_bc: torch.Tensor,
    ) -> "FVMResiduals":
        """
        Устанавливает предписанные скорости для граней заданного типа ГУ.

        После вызова грани этого типа будут использовать u_bc вместо
        предсказания нейросети (hard BC).

        Аргументы:
            boundary_type: тип ГУ — 'inlet', 'outlet', 'symmetry', ...
            u_bc: (2,) — однородная скорость для всех граней типа,
                  (K, 2) — профиль, по одному вектору на каждую грань типа
                  (K = количество граней данного типа в bnd_type)

        Возвращает:
            self для цепочки: fvm.set_bc_velocity('inlet', u_inf).to('cuda')

        Пример:
            >>> fvm.set_bc_velocity('inlet', torch.tensor([10.0, 0.0]))
        """
        code = BC_CODES.get(boundary_type.lower())
        if code is None:
            raise ValueError(
                f"Неизвестный тип ГУ: {boundary_type!r}. "
                f"Допустимые: {list(BC_CODES)}"
            )

        type_mask = (self.bnd_type == code)           # (B,) bool
        n_faces = int(type_mask.sum())
        if n_faces == 0:
            logger.warning("set_bc_velocity: нет граней типа '%s'", boundary_type)
            return self

        u_bc = u_bc.detach().to(device=self._device, dtype=self._dtype)

        if u_bc.dim() == 1 and u_bc.shape[0] == 2:
            self._u_prescribed[type_mask] = u_bc.unsqueeze(0).expand(n_faces, 2)
        elif u_bc.dim() == 2 and u_bc.shape == (n_faces, 2):
            self._u_prescribed[type_mask] = u_bc
        else:
            raise ValueError(
                f"u_bc должен иметь форму (2,) или ({n_faces}, 2), "
                f"получено {tuple(u_bc.shape)}"
            )

        self._prescribed_mask[type_mask] = True
        logger.info(
            "set_bc_velocity: '%s' (%d граней), u_bc=%s",
            boundary_type, n_faces,
            u_bc.tolist() if u_bc.dim() == 1 else "<profile>",
        )
        return self

    def set_inlet_velocity(self, u_bc: torch.Tensor) -> "FVMResiduals":
        """
        Задаёт предписанный профиль скорости на inlet.

        Удобная обёртка над set_bc_velocity('inlet', u_bc).

        Аргументы:
            u_bc: (2,) — однородная скорость или (K, 2) — профиль
        """
        return self.set_bc_velocity('inlet', u_bc)

    def reset_prescribed(
        self,
        boundary_type: Optional[str] = None,
    ) -> "FVMResiduals":
        """
        Сбрасывает предписанные скорости: переходит к мягкому BC
        Production FVM продолжает получать скорость прямым предсказанием PINN
        в midpoint граничной полуграни.

        Аргументы:
            boundary_type: None — сброс для всех типов;
                           строка — сброс только для указанного типа

        Возвращает:
            self для цепочки вызовов
        """
        if boundary_type is None:
            self._u_prescribed.zero_()
            self._prescribed_mask.fill_(False)
            logger.info("reset_prescribed: сброшены все предписанные BC")
        else:
            code = BC_CODES.get(boundary_type.lower())
            if code is None:
                raise ValueError(f"Неизвестный тип ГУ: {boundary_type!r}")
            type_mask = (self.bnd_type == code)
            self._u_prescribed[type_mask] = 0.0
            self._prescribed_mask[type_mask] = False
            logger.info("reset_prescribed: сброшен тип '%s'", boundary_type)
        return self

    # ── Управление устройством ────────────────────────────────────────────────

    def to(
        self,
        device: Union[str, torch.device],
        dtype: Optional[torch.dtype] = None,
    ) -> "FVMResiduals":
        """
        Переносит все геометрические тензоры на device и, опционально, dtype.

        Возвращает self для цепочки вызовов:
            fvm = FVMResiduals(mesh).to('cuda', torch.float32)
        """
        target_device = torch.device(device)
        target_dtype = dtype if dtype is not None else self._dtype

        for attr in ('V', 'edge_S', 'quad_S', 'bnd_S', '_u_prescribed'):
            setattr(self, attr, getattr(self, attr).to(device=target_device, dtype=target_dtype))
        for attr in ('edge_i', 'edge_j', 'quad_i', 'quad_j', 'bnd_node', 'bnd_type'):
            setattr(self, attr, getattr(self, attr).to(device=target_device))
        for attr in ('_prescribed_mask', '_zero_flux_mask'):
            setattr(self, attr, getattr(self, attr).to(device=target_device))

        # torch.device("cuda") is an alias; tensors live on a concrete device
        # such as cuda:0. Store the concrete value to keep validation reliable.
        self._device = self.V.device
        self._dtype = self.V.dtype
        self._validate_tensor_geometry()
        return self

    def nondimensionalize_geometry(
        self,
        length_scale: float,
    ) -> "FVMResiduals":
        """Scale all planar FVM measures with one reference length.

        Face vectors are divided by ``length_scale`` and dual-cell areas by
        ``length_scale**2``.  Calling the method repeatedly with the same
        scale is idempotent; changing scales on an already normalized operator
        is rejected to avoid silent double normalization.
        """
        scale = float(length_scale)
        if not np.isfinite(scale) or scale <= 0.0:
            raise ValueError(
                "FVM geometry length_scale must be finite and positive"
            )
        if self._geometry_is_dimensionless:
            if not np.isclose(
                scale,
                self._geometry_length_scale,
                rtol=1e-12,
                atol=0.0,
            ):
                raise ValueError(
                    "FVM geometry is already normalized with a different "
                    f"length scale: {self._geometry_length_scale} != {scale}"
                )
            return self

        self.edge_S.div_(scale)
        self.quad_S.div_(scale)
        self.bnd_S.div_(scale)
        self.V.div_(scale ** 2)
        self._validate_tensor_geometry()
        self._geometry_is_dimensionless = True
        self._geometry_length_scale = scale
        logger.info(
            "FVM geometry normalized: S/c, V/c^2 with c=%.9g",
            scale,
        )
        return self

    def rebuild_from_mesh(self, new_mesh) -> None:
        """
        Пересчитывает геометрические тензоры из новой сетки (для RAR-адаптации).
        Сохраняет текущее устройство.

        Аргументы:
            new_mesh: Mesh2D или MeshData с обновлённой геометрией
        """
        device = self._device
        dtype = self._dtype
        was_dimensionless = self._geometry_is_dimensionless
        length_scale = self._geometry_length_scale
        self._build_from_mesh(new_mesh)
        self.to(device, dtype=dtype)
        self._geometry_is_dimensionless = False
        self._geometry_length_scale = 1.0
        if was_dimensionless:
            self.nondimensionalize_geometry(length_scale)
        logger.info(
            "FVMResiduals перестроен: N=%d, M=%d, Q=%d, B=%d",
            self.N,
            self.M,
            self.Q,
            self.B,
        )

    # ── Фабрики ───────────────────────────────────────────────────────────────

    @classmethod
    def from_mesh_data(cls, mesh_data) -> "FVMResiduals":
        """Создаёт из data_handler.MeshData (загруженного из .npz)."""
        return cls(mesh_data)

    def __repr__(self) -> str:
        return (f"FVMResiduals(N={self.N}, M={self.M}, Q={self.Q}, B={self.B}, "
                f"device={self._device})")
