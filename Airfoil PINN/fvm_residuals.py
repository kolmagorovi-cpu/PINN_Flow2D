#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fvm_residuals.py — FVM-вычислитель невязки уравнения неразрывности.

Реализует класс FVMResiduals: дискретизация закона сохранения массы для
несжимаемой жидкости методом FVM на медианной дуальной неструктурированной
сетке (mesh2d.Mesh2D). На внутренних гранях используется центральная
интерполяция скорости u_face = 0.5 * (u_i + u_j);
все операции векторизованы на PyTorch без Python-циклов по граням.

Поддерживает два источника геометрии:
  - Mesh2D (mesh2d.py) — после build()
  - MeshData (data_handler.py) — загруженный из .npz файла

Пример:
    >>> mesh = Mesh2D(points, domain_coords, airfoil_coords)
    >>> mesh.build()
    >>> fvm = FVMResiduals(mesh).to('cuda')
    >>> loss = fvm.mass_loss(u_pred)   # u_pred: (N, 2)

Автор: Колмагоров И.Д.
Дата: 2026-05-21
"""

from __future__ import annotations

import logging
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
}
_WALL_CODE: int          = BC_CODES['wall']
_SYMMETRY_CODE: int      = BC_CODES['symmetry']
_AXIS_SYMMETRY_CODE: int = BC_CODES['axis_symmetry']

# Нижняя граница объёма КО — защита от деления на ноль при вырожденной сетке.
# Значение много меньше любого физически реального объёма (< 1e-10 м²).
_V_MIN: float = 1e-30


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
        self._build_from_mesh(mesh)

    # ── Инициализация ─────────────────────────────────────────────────────────

    def _build_from_mesh(self, mesh) -> None:
        cv, ei, ej, eS, bn, bS, bt_strs = (
            self._extract_mesh2d(mesh)
            if hasattr(mesh, 'internal_faces')
            else self._extract_mesh_data(mesh)
        )

        self.N = len(cv)
        self.M = len(ei)
        self.B = len(bn)

        self.V = torch.tensor(np.asarray(cv, dtype=np.float64), dtype=self._dtype)
        n_bad = int((self.V <= 0).sum())
        if n_bad:
            logger.warning(
                "FVMResiduals: %d/%d CV-объёмов ≤ 0 (min=%.3e) — "
                "вырожденная сетка. Объёмы зажаты снизу до %.3e.",
                n_bad, self.N, float(self.V.min()), _V_MIN,
            )
            self.V = self.V.clamp(min=_V_MIN)

        self.edge_i = torch.tensor(np.asarray(ei, dtype=np.int64),    dtype=torch.long)
        self.edge_j = torch.tensor(np.asarray(ej, dtype=np.int64),    dtype=torch.long)
        self.edge_S = torch.tensor(np.asarray(eS, dtype=np.float64).reshape(-1, 2),
                                   dtype=self._dtype)
        self.bnd_node = torch.tensor(np.asarray(bn, dtype=np.int64),  dtype=torch.long)
        self.bnd_S    = torch.tensor(np.asarray(bS, dtype=np.float64).reshape(-1, 2),
                                     dtype=self._dtype)
        bnd_codes = np.fromiter(
            (BC_CODES.get(str(t).lower(), BC_CODES['interior']) for t in bt_strs),
            dtype=np.int64, count=self.B,
        )
        self.bnd_type = torch.tensor(bnd_codes, dtype=torch.long)

        # Предписанные скорости на гранях (hard BC).
        # По умолчанию — мягкое условие (u_face = u_узла из нейросети).
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
            ).unsqueeze(1)                                   # (B, 1)
        else:
            self._zero_flux_mask = torch.zeros(0, 1, dtype=torch.bool)

        logger.info("FVMResiduals: N=%d, M=%d, B=%d", self.N, self.M, self.B)

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

    def _compute_residual(self, u: torch.Tensor, apply_boundary_conditions: bool) -> torch.Tensor:
        """Внутренняя реализация FVM-невязки с опциональным применением BC."""
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

    # ── Основные методы ───────────────────────────────────────────────────────

    def compute_residual(self, u: torch.Tensor) -> torch.Tensor:
        """
        Невязка уравнения неразрывности R_i = Φ_i / V_i для каждого КО.

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
        return self._compute_residual(u, apply_boundary_conditions=True)

    def mass_loss(self, u: torch.Tensor) -> torch.Tensor:
        """
        Среднеквадратичная невязка (скаляр), готовая для сложения с total_loss.

        L_mass = (1/N) Σ R_i²

        Аргументы:
            u: скорость (N, 2)

        Возвращает:
            Скалярный тензор потерь
        """
        return torch.mean(self.compute_residual(u) ** 2)

    # ── Вспомогательные методы ────────────────────────────────────────────────

    def check_geometry_conservativity(self, tol: float = 1e-8) -> bool:
        """
        Проверяет замыкание векторов площадей для каждого контрольного объёма.

        Этот тест не использует скорости и не применяет физические граничные
        условия. Он отвечает только на вопрос: сумма всех outward-векторов S
        вокруг каждого CV равна нулю?
        """
        flux_sum = torch.zeros(self.N, 2, device=self._device, dtype=self._dtype)
        if self.M > 0:
            flux_sum = flux_sum.index_add(0, self.edge_i, self.edge_S)
            flux_sum = flux_sum.index_add(0, self.edge_j, -self.edge_S)
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
            'B': self.B,
            'device': str(self._device),
            'cv_volume_min':  float(self.V.min()),
            'cv_volume_max':  float(self.V.max()),
            'cv_volume_mean': float(self.V.mean()),
            'edge_S_norm_min':  float(self.edge_S.norm(dim=1).min()) if self.M > 0 else 0.0,
            'edge_S_norm_max':  float(self.edge_S.norm(dim=1).max()) if self.M > 0 else 0.0,
            'bnd_S_norm_min':   float(self.bnd_S.norm(dim=1).min()) if self.B > 0 else 0.0,
            'bnd_S_norm_max':   float(self.bnd_S.norm(dim=1).max()) if self.B > 0 else 0.0,
            'bnd_type_counts':  {
                name: int((self.bnd_type == code).sum())
                for name, code in BC_CODES.items()
                if int((self.bnd_type == code).sum()) > 0
            },
            'prescribed_bc_types': prescribed_types,
        }

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
        (u_face = предсказание нейросети в граничном узле).

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

        for attr in ('V', 'edge_S', 'bnd_S', '_u_prescribed'):
            setattr(self, attr, getattr(self, attr).to(device=target_device, dtype=target_dtype))
        for attr in ('edge_i', 'edge_j', 'bnd_node', 'bnd_type'):
            setattr(self, attr, getattr(self, attr).to(device=target_device))
        for attr in ('_prescribed_mask', '_zero_flux_mask'):
            setattr(self, attr, getattr(self, attr).to(device=target_device))

        # torch.device("cuda") is an alias; tensors live on a concrete device
        # such as cuda:0. Store the concrete value to keep validation reliable.
        self._device = self.V.device
        self._dtype = self.V.dtype
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
        self._build_from_mesh(new_mesh)
        self.to(device, dtype=dtype)
        logger.info("FVMResiduals перестроен: N=%d, M=%d, B=%d", self.N, self.M, self.B)

    # ── Фабрики ───────────────────────────────────────────────────────────────

    @classmethod
    def from_mesh_data(cls, mesh_data) -> "FVMResiduals":
        """Создаёт из data_handler.MeshData (загруженного из .npz)."""
        return cls(mesh_data)

    def __repr__(self) -> str:
        return (f"FVMResiduals(N={self.N}, M={self.M}, B={self.B}, "
                f"device={self._device})")
