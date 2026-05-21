#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fvm_residuals.py — FVM-вычислитель невязки уравнения неразрывности.

Реализует класс FVMResiduals: дискретизация закона сохранения массы для
несжимаемой жидкости методом FVM на медианной дуальной неструктурированной
сетке (mesh2d.Mesh2D). Противопоточная (upwind) схема 1-го порядка;
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
_WALL_CODE: int = BC_CODES['wall']

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

        self.V = torch.tensor(np.asarray(cv, dtype=np.float32), dtype=torch.float32)
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
        self.edge_S = torch.tensor(np.asarray(eS, dtype=np.float32).reshape(-1, 2),
                                   dtype=torch.float32)
        self.bnd_node = torch.tensor(np.asarray(bn, dtype=np.int64),  dtype=torch.long)
        self.bnd_S    = torch.tensor(np.asarray(bS, dtype=np.float32).reshape(-1, 2),
                                     dtype=torch.float32)
        bnd_codes = np.fromiter(
            (BC_CODES.get(str(t).lower(), BC_CODES['interior']) for t in bt_strs),
            dtype=np.int64, count=self.B,
        )
        self.bnd_type = torch.tensor(bnd_codes, dtype=torch.long)

        # Предписанные скорости на гранях (hard BC).
        # По умолчанию — мягкое условие (u_face = u_узла из нейросети).
        # Заполняются через set_bc_velocity() / set_inlet_velocity().
        self._u_prescribed    = torch.zeros(self.B, 2, dtype=torch.float32)
        self._prescribed_mask = torch.zeros(self.B, 1, dtype=torch.bool)

        logger.info("FVMResiduals: N=%d, M=%d, B=%d", self.N, self.M, self.B)

    @staticmethod
    def _extract_mesh2d(mesh) -> Tuple:
        """Извлекает массивы из объекта Mesh2D."""
        cv = mesh.cv_volumes

        ifaces = mesh.internal_faces
        if ifaces:
            ei = np.array([f.node_i      for f in ifaces], dtype=np.int64)
            ej = np.array([f.node_j      for f in ifaces], dtype=np.int64)
            eS = np.array([f.area_vector for f in ifaces], dtype=np.float32)
        else:
            ei = np.empty(0, dtype=np.int64)
            ej = np.empty(0, dtype=np.int64)
            eS = np.empty((0, 2), dtype=np.float32)

        bfaces = mesh.boundary_faces
        if bfaces:
            bn      = np.array([f.node         for f in bfaces], dtype=np.int64)
            bS      = np.array([f.area_vector  for f in bfaces], dtype=np.float32)
            bt_strs = [f.boundary_type for f in bfaces]
        else:
            bn      = np.empty(0, dtype=np.int64)
            bS      = np.empty((0, 2), dtype=np.float32)
            bt_strs = []

        return cv, ei, ej, eS, bn, bS, bt_strs

    @staticmethod
    def _extract_mesh_data(md) -> Tuple:
        """Извлекает массивы из объекта MeshData (data_handler.py)."""
        return (
            md.cv_volumes,
            md.int_face_node_i.astype(np.int64),
            md.int_face_node_j.astype(np.int64),
            md.int_face_area.astype(np.float32),
            md.bnd_face_node.astype(np.int64),
            md.bnd_face_area.astype(np.float32),
            list(md.bnd_face_type.astype(str)),
        )

    # ── Основные методы ───────────────────────────────────────────────────────

    def compute_residual(self, u: torch.Tensor) -> torch.Tensor:
        """
        Невязка уравнения неразрывности R_i = Φ_i / V_i для каждого КО.

        Алгоритм (см. §9 математического описания):
          - Upwind-интерполяция на внутренних гранях (направление по u_avg·S_ij).
          - Граничные грани: u_face = 0 на стенке, u_face = u_узла на inlet/outlet.
          - Нормировка на объём КО.

        Аргументы:
            u: скорость в узлах, форма (N, 2)

        Возвращает:
            R: тензор невязок (N,), [с⁻¹]
        """
        res = u.new_zeros(self.N)

        # ── Внутренние грани (векторизовано по всем M граням) ─────────────────
        if self.M > 0:
            u_i   = u[self.edge_i]                                # (M, 2)
            u_j   = u[self.edge_j]                                # (M, 2)
            u_avg = 0.5 * (u_i + u_j)                            # (M, 2)

            # Направление потока: δ_ij = u_avg · S_ij
            direction = (u_avg * self.edge_S).sum(dim=1)          # (M,)

            # Upwind-маска: 1.0 если поток идёт из i → j, 0.0 если j → i
            mask  = (direction >= 0.0).to(u.dtype).unsqueeze(1)   # (M, 1)
            u_face = u_i * mask + u_j * (1.0 - mask)             # (M, 2)
            flux   = (u_face * self.edge_S).sum(dim=1)            # (M,)

            # Аккумулирование: +flux для i, −flux для j
            res = res.index_add(0, self.edge_i,  flux)
            res = res.index_add(0, self.edge_j, -flux)

        # ── Граничные грани ───────────────────────────────────────────────────
        if self.B > 0:
            u_b = u[self.bnd_node]                                # (B, 2)

            # Prescribed BC: заменяет u_b там, где _prescribed_mask = True
            # (например, заданный профиль скорости на inlet).
            u_face_b = torch.where(
                self._prescribed_mask,
                self._u_prescribed.to(u.dtype),
                u_b,
            )
            # Стенка (no-slip): всегда ноль, перекрывает любые предписания
            wall_mask = (self.bnd_type == _WALL_CODE).unsqueeze(1)
            u_face_b  = u_face_b.masked_fill(wall_mask, 0.0)
            flux_b    = (u_face_b * self.bnd_S).sum(dim=1)       # (B,)

            res = res.index_add(0, self.bnd_node, flux_b)

        # ── Нормировка на объём КО ────────────────────────────────────────────
        return res / self.V.to(u.dtype)

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

    def check_conservativity(
        self,
        u_const: Optional[torch.Tensor] = None,
        tol: float = 1e-6,
    ) -> bool:
        """
        Верификация геометрии: для постоянного поля max|R_i| должен быть ≈ 0.

        Если mesh._verify_conservativity() прошла успешно, этот тест тоже
        должен проходить (R_i = (1/V_i) u·(Σ S_ij) = 0).

        Аргументы:
            u_const: тестовое поле (N, 2); None → однородный поток (1, 0)
            tol: допуск

        Возвращает:
            True если max|R_i| < tol
        """
        if u_const is None:
            u_const = torch.ones(self.N, 2, dtype=torch.float32, device=self._device)

        with torch.no_grad():
            R = self.compute_residual(u_const)

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

        u_bc = u_bc.detach().to(device=self._device, dtype=torch.float32)

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

    def to(self, device: Union[str, torch.device]) -> "FVMResiduals":
        """
        Переносит все геометрические тензоры на device.

        Возвращает self для цепочки вызовов:
            fvm = FVMResiduals(mesh).to('cuda')
        """
        self._device = torch.device(device)
        for attr in ('V', 'edge_i', 'edge_j', 'edge_S',
                     'bnd_node', 'bnd_S', 'bnd_type',
                     '_u_prescribed', '_prescribed_mask'):
            setattr(self, attr, getattr(self, attr).to(self._device))
        return self

    def rebuild_from_mesh(self, new_mesh) -> None:
        """
        Пересчитывает геометрические тензоры из новой сетки (для RAR-адаптации).
        Сохраняет текущее устройство.

        Аргументы:
            new_mesh: Mesh2D или MeshData с обновлённой геометрией
        """
        device = self._device
        self._build_from_mesh(new_mesh)
        self.to(device)
        logger.info("FVMResiduals перестроен: N=%d, M=%d, B=%d", self.N, self.M, self.B)

    # ── Фабрики ───────────────────────────────────────────────────────────────

    @classmethod
    def from_mesh_data(cls, mesh_data) -> "FVMResiduals":
        """Создаёт из data_handler.MeshData (загруженного из .npz)."""
        return cls(mesh_data)

    def __repr__(self) -> str:
        return (f"FVMResiduals(N={self.N}, M={self.M}, B={self.B}, "
                f"device={self._device})")
