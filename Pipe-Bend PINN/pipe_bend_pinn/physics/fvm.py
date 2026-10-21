#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fvm_residuals.py вЂ” FVM-РІС‹С‡РёСЃР»РёС‚РµР»СЊ РЅРµРІСЏР·РєРё СѓСЂР°РІРЅРµРЅРёСЏ РЅРµСЂР°Р·СЂС‹РІРЅРѕСЃС‚Рё.

РџРѕСЂС‚РёСЂРѕРІР°РЅ РёР· РїР°РєРµС‚Р° Airfoil PINN (``airfoil_pinn/physics/fvm.py``) РґР»СЏ
Pipe-Bend. Р РµР°Р»РёР·СѓРµС‚ РєР»Р°СЃСЃ FVMResiduals: РґРёСЃРєСЂРµС‚РёР·Р°С†РёСЏ Р·Р°РєРѕРЅР° СЃРѕС…СЂР°РЅРµРЅРёСЏ РјР°СЃСЃС‹
РґР»СЏ РЅРµСЃР¶РёРјР°РµРјРѕР№ Р¶РёРґРєРѕСЃС‚Рё РјРµС‚РѕРґРѕРј FVM РЅР° РјРµРґРёР°РЅРЅРѕР№ РґСѓР°Р»СЊРЅРѕР№ РЅРµСЃС‚СЂСѓРєС‚СѓСЂРёСЂРѕРІР°РЅРЅРѕР№
СЃРµС‚РєРµ (mesh2d.Mesh2D). РќР° РІРЅСѓС‚СЂРµРЅРЅРёС… РіСЂР°РЅСЏС… РёСЃРїРѕР»СЊР·СѓРµС‚СЃСЏ С†РµРЅС‚СЂР°Р»СЊРЅР°СЏ
РёРЅС‚РµСЂРїРѕР»СЏС†РёСЏ СЃРєРѕСЂРѕСЃС‚Рё u_face = 0.5 * (u_i + u_j);
РІСЃРµ РѕРїРµСЂР°С†РёРё РІРµРєС‚РѕСЂРёР·РѕРІР°РЅС‹ РЅР° PyTorch Р±РµР· Python-С†РёРєР»РѕРІ РїРѕ РіСЂР°РЅСЏРј.

РџРѕРґРґРµСЂР¶РёРІР°РµС‚ РґРІР° РёСЃС‚РѕС‡РЅРёРєР° РіРµРѕРјРµС‚СЂРёРё:
  - Mesh2D (mesh2d.py) вЂ” РїРѕСЃР»Рµ build()
  - MeshData (data_handler.py) вЂ” Р·Р°РіСЂСѓР¶РµРЅРЅС‹Р№ РёР· .npz С„Р°Р№Р»Р°

РџСЂРёРјРµСЂ:
    >>> mesh = Mesh2D(points, domain_coords, obstacle_coords=None)
    >>> mesh.build()
    >>> fvm = FVMResiduals(mesh).to('cuda')
    >>> loss = fvm.mass_loss(u_pred)   # u_pred: (N, 2)

РђРІС‚РѕСЂ: РљРѕР»РјР°РіРѕСЂРѕРІ Р.Р”. (РїРѕСЂС‚: Pipe-Bend PINN Team)
Р”Р°С‚Р°: 2026-05-21
"""

from __future__ import annotations

import hashlib
import logging
import warnings
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch

logger = logging.getLogger(__name__)

# в”Ђв”Ђ РљРѕРґС‹ С‚РёРїРѕРІ РіСЂР°РЅРёС‡РЅС‹С… СѓСЃР»РѕРІРёР№ в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ
# РСЃРїРѕР»СЊР·СѓСЋС‚СЃСЏ РґР»СЏ РїРѕСЃС‚СЂРѕРµРЅРёСЏ РјР°СЃРєРё СЃС‚РµРЅРєРё РІРЅСѓС‚СЂРё compute_residual.
# Р Р°СЃС€РёСЂСЊС‚Рµ СЃР»РѕРІР°СЂСЊ РїСЂРё РґРѕР±Р°РІР»РµРЅРёРё РЅРѕРІС‹С… С‚РёРїРѕРІ Р“РЈ.
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

# РќРёР¶РЅСЏСЏ РіСЂР°РЅРёС†Р° РѕР±СЉС‘РјР° РљРћ вЂ” Р·Р°С‰РёС‚Р° РѕС‚ РґРµР»РµРЅРёСЏ РЅР° РЅРѕР»СЊ РїСЂРё РІС‹СЂРѕР¶РґРµРЅРЅРѕР№ СЃРµС‚РєРµ.
# Р—РЅР°С‡РµРЅРёРµ РјРЅРѕРіРѕ РјРµРЅСЊС€Рµ Р»СЋР±РѕРіРѕ С„РёР·РёС‡РµСЃРєРё СЂРµР°Р»СЊРЅРѕРіРѕ РѕР±СЉС‘РјР° (< 1e-10 РјВІ).
__all__ = [
    "BC_CODES",
    "FVMResiduals",
    "evaluate_fvm_metrics",
]

_GEOMETRY_CLOSURE_RTOL: float = 1e-8
_GEOMETRY_CLOSURE_ATOL: float = 1e-12


class FVMResiduals:
    """
    Р’С‹С‡РёСЃР»РёС‚РµР»СЊ РЅРµРІСЏР·РєРё R_i = О¦_i / V_i РґР»СЏ СѓСЂР°РІРЅРµРЅРёСЏ РЅРµСЂР°Р·СЂС‹РІРЅРѕСЃС‚Рё в€‡В·u = 0.

    Р“РµРѕРјРµС‚СЂРёС‡РµСЃРєРёРµ С‚РµРЅР·РѕСЂС‹ РїСЂРµРґРѕР±СЂР°Р±Р°С‚С‹РІР°СЋС‚СЃСЏ РѕРґРЅРѕРєСЂР°С‚РЅРѕ РІ РєРѕРЅСЃС‚СЂСѓРєС‚РѕСЂРµ.
    РќР° РєР°Р¶РґРѕРј С€Р°РіРµ РѕР±СѓС‡РµРЅРёСЏ РІС‹Р·С‹РІР°РµС‚СЃСЏ С‚РѕР»СЊРєРѕ compute_residual / mass_loss.

    РђСЂРіСѓРјРµРЅС‚С‹:
        mesh: Mesh2D (mesh2d.py) РёР»Рё MeshData (data_handler.py)

    РђС‚СЂРёР±СѓС‚С‹:
        N (int): С‡РёСЃР»Рѕ СѓР·Р»РѕРІ
        M (int): С‡РёСЃР»Рѕ РІРЅСѓС‚СЂРµРЅРЅРёС… РіСЂР°РЅРµР№
        B (int): С‡РёСЃР»Рѕ РіСЂР°РЅРёС‡РЅС‹С… РіСЂР°РЅРµР№
        V        (N,):    РѕР±СЉС‘РјС‹ РљРћ [РјВІ]
        edge_i   (M,):    РёРЅРґРµРєСЃС‹ СѓР·Р»РѕРІ i РІРЅСѓС‚СЂРµРЅРЅРёС… РіСЂР°РЅРµР№
        edge_j   (M,):    РёРЅРґРµРєСЃС‹ СѓР·Р»РѕРІ j РІРЅСѓС‚СЂРµРЅРЅРёС… РіСЂР°РЅРµР№
        edge_S   (M, 2):  РІРµРєС‚РѕСЂС‹ РїР»РѕС‰Р°РґРµР№ S_ij (outward РѕС‚ i)
        bnd_node (B,):    РёРЅРґРµРєСЃС‹ РіСЂР°РЅРёС‡РЅС‹С… СѓР·Р»РѕРІ
        bnd_S    (B, 2):  РІРµРєС‚РѕСЂС‹ РїР»РѕС‰Р°РґРµР№ РіСЂР°РЅРёС‡РЅС‹С… РіСЂР°РЅРµР№ (outward)
        bnd_type (B,):    РєРѕРґС‹ С‚РёРїРѕРІ Р“РЈ (0=wall, 1=inlet, ...)
    """

    def __init__(self, mesh) -> None:
        self._device = torch.device('cpu')
        self._dtype = torch.float64
        self._geometry_is_dimensionless = False
        self._geometry_length_scale = 1.0
        self._legacy_volume_reduction_warned = False
        self._build_from_mesh(mesh)

    # в”Ђв”Ђ РРЅРёС†РёР°Р»РёР·Р°С†РёСЏ в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ

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

        # РџСЂРµРґРїРёСЃР°РЅРЅС‹Рµ СЃРєРѕСЂРѕСЃС‚Рё РЅР° РіСЂР°РЅСЏС… (hard BC).
        # РџРѕ СѓРјРѕР»С‡Р°РЅРёСЋ вЂ” РјСЏРіРєРѕРµ СѓСЃР»РѕРІРёРµ (u_face = u_СѓР·Р»Р° РёР· РЅРµР№СЂРѕСЃРµС‚Рё).
        # Р—Р°РїРѕР»РЅСЏСЋС‚СЃСЏ С‡РµСЂРµР· set_bc_velocity() / set_inlet_velocity().
        self._u_prescribed    = torch.zeros(self.B, 2, dtype=self._dtype)
        self._prescribed_mask = torch.zeros(self.B, 1, dtype=torch.bool)

        # РњР°СЃРєР° РіСЂР°РЅРµР№ СЃ РЅСѓР»РµРІС‹Рј РЅРѕСЂРјР°Р»СЊРЅС‹Рј РїРѕС‚РѕРєРѕРј: wall (no-slip) Рё
        # symmetry/axis_symmetry (РЅРµРїСЂРѕС‚РµРєР°РЅРёРµ uВ·n = 0). РџСЂРµРґРІС‹С‡РёСЃР»РµРЅР°
        # РѕРґРЅРѕРєСЂР°С‚РЅРѕ вЂ” РёСЃРїРѕР»СЊР·СѓРµС‚СЃСЏ РІ РєР°Р¶РґРѕРј РІС‹Р·РѕРІРµ compute_residual.
        if self.B > 0:
            self._zero_flux_mask = (
                (self.bnd_type == _WALL_CODE)
                | (self.bnd_type == _SYMMETRY_CODE)
                | (self.bnd_type == _AXIS_SYMMETRY_CODE)
                | (self.bnd_type == _SLIP_CODE)
            ).unsqueeze(1)                                   # (B, 1)
        else:
            self._zero_flux_mask = torch.zeros(0, 1, dtype=torch.bool)

        logger.info("FVMResiduals: N=%d, M=%d, B=%d", self.N, self.M, self.B)

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
        """РџСЂРѕРІРµСЂСЏРµС‚ СЃРєРѕСЂРѕСЃС‚СЊ РїРµСЂРµРґ FVM-РѕРїРµСЂР°С†РёСЏРјРё, С‡С‚РѕР±С‹ РѕС€РёР±РєРё Р±С‹Р»Рё СЏРІРЅС‹РјРё."""
        if not torch.is_tensor(u):
            raise TypeError("FVMResiduals.compute_residual: u РґРѕР»Р¶РµРЅ Р±С‹С‚СЊ torch.Tensor")
        if u.ndim != 2 or u.shape != (self.N, 2):
            raise ValueError(
                "FVMResiduals.compute_residual: u РґРѕР»Р¶РµРЅ РёРјРµС‚СЊ С„РѕСЂРјСѓ "
                f"({self.N}, 2), РїРѕР»СѓС‡РµРЅРѕ {tuple(u.shape)}"
            )
        geometry_device = self.V.device
        if u.device != geometry_device:
            raise ValueError(
                "FVMResiduals.compute_residual: device u РЅРµ СЃРѕРІРїР°РґР°РµС‚ СЃ РіРµРѕРјРµС‚СЂРёРµР№ "
                f"({u.device} != {geometry_device}). Р’С‹Р·РѕРІРёС‚Рµ fvm.to(u.device, u.dtype)."
            )
        if u.dtype != self._dtype:
            raise ValueError(
                "FVMResiduals.compute_residual: dtype u РЅРµ СЃРѕРІРїР°РґР°РµС‚ СЃ РіРµРѕРјРµС‚СЂРёРµР№ "
                f"({u.dtype} != {self._dtype}). Р’С‹Р·РѕРІРёС‚Рµ fvm.to(u.device, u.dtype)."
            )

    def _validate_tensor_geometry(self) -> None:
        """Reject invalid values introduced by dtype conversion or scaling."""
        if not torch.isfinite(self.V).all():
            raise ValueError("FVM volumes contain NaN or Inf")
        if torch.any(self.V <= 0):
            raise ValueError("FVM control volumes must be strictly positive")
        if (
            not torch.isfinite(self.edge_S).all()
            or not torch.isfinite(self.bnd_S).all()
        ):
            raise ValueError("FVM area vectors contain NaN or Inf")
        if (
            (self.M > 0 and torch.any(self.edge_S.norm(dim=1) <= 0))
            or (self.B > 0 and torch.any(self.bnd_S.norm(dim=1) <= 0))
        ):
            raise ValueError("FVM area vectors must be non-zero")

    def _compute_residual(
        self,
        u: torch.Tensor,
        apply_boundary_conditions: bool,
        freestream_velocity: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Evaluate a finite-volume residual with optional case-local BC.

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
        """РР·РІР»РµРєР°РµС‚ РјР°СЃСЃРёРІС‹ РёР· РѕР±СЉРµРєС‚Р° Mesh2D."""
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
        """РР·РІР»РµРєР°РµС‚ РјР°СЃСЃРёРІС‹ РёР· РѕР±СЉРµРєС‚Р° MeshData (data_handler.py)."""
        return (
            md.cv_volumes,
            md.int_face_node_i.astype(np.int64),
            md.int_face_node_j.astype(np.int64),
            md.int_face_area.astype(np.float64),
            md.bnd_face_node.astype(np.int64),
            md.bnd_face_area.astype(np.float64),
            list(md.bnd_face_type.astype(str)),
        )

    # в”Ђв”Ђ РћСЃРЅРѕРІРЅС‹Рµ РјРµС‚РѕРґС‹ в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ

    def compute_residual(
        self,
        u: torch.Tensor,
        freestream_velocity: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        РќРµРІСЏР·РєР° СѓСЂР°РІРЅРµРЅРёСЏ РЅРµСЂР°Р·СЂС‹РІРЅРѕСЃС‚Рё R_i = О¦_i / V_i РґР»СЏ РєР°Р¶РґРѕРіРѕ РљРћ.

        РђР»РіРѕСЂРёС‚Рј:
          - Р¦РµРЅС‚СЂР°Р»СЊРЅР°СЏ СЃС…РµРјР° РЅР° РІРЅСѓС‚СЂРµРЅРЅРёС… РіСЂР°РЅСЏС…: u_face = 0.5*(u_i + u_j).
          - Р“СЂР°РЅРёС‡РЅС‹Рµ РіСЂР°РЅРё РїРѕ С‚РёРїСѓ:
              wall           : flux = 0  (no-slip, u = 0)
              symmetry       : flux = 0  (РЅРµРїСЂРѕС‚РµРєР°РЅРёРµ uВ·n = 0; РґР»СЏ СѓСЂР°РІРЅРµРЅРёСЏ
              axis_symmetry    РЅРµСЂР°Р·СЂС‹РІРЅРѕСЃС‚Рё СЌРєРІРёРІР°Р»РµРЅС‚РЅРѕ СЃС‚РµРЅРєРµ)
              inlet          : flux = u_prescribed В· S  (hard BC С‡РµСЂРµР· set_inlet_velocity)
              outlet/interior: flux = u_СѓР·Р»Р° В· S  (РјСЏРіРєРѕРµ СѓСЃР»РѕРІРёРµ РёР· РЅРµР№СЂРѕСЃРµС‚Рё)
          - РќРѕСЂРјРёСЂРѕРІРєР° РЅР° РѕР±СЉС‘Рј РљРћ.

        РђСЂРіСѓРјРµРЅС‚С‹:
            u: СЃРєРѕСЂРѕСЃС‚СЊ РІ СѓР·Р»Р°С…, С„РѕСЂРјР° (N, 2)

        Р’РѕР·РІСЂР°С‰Р°РµС‚:
            R: С‚РµРЅР·РѕСЂ РЅРµРІСЏР·РѕРє (N,), [СЃвЃ»В№]
        """
        return self._compute_residual(
            u,
            apply_boundary_conditions=True,
            freestream_velocity=freestream_velocity,
        )

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

    # в”Ђв”Ђ Р’СЃРїРѕРјРѕРіР°С‚РµР»СЊРЅС‹Рµ РјРµС‚РѕРґС‹ в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ

    def check_geometry_conservativity(self, tol: float = 1e-8) -> bool:
        """
        РџСЂРѕРІРµСЂСЏРµС‚ Р·Р°РјС‹РєР°РЅРёРµ РІРµРєС‚РѕСЂРѕРІ РїР»РѕС‰Р°РґРµР№ РґР»СЏ РєР°Р¶РґРѕРіРѕ РєРѕРЅС‚СЂРѕР»СЊРЅРѕРіРѕ РѕР±СЉС‘РјР°.

        Р­С‚РѕС‚ С‚РµСЃС‚ РЅРµ РёСЃРїРѕР»СЊР·СѓРµС‚ СЃРєРѕСЂРѕСЃС‚Рё Рё РЅРµ РїСЂРёРјРµРЅСЏРµС‚ С„РёР·РёС‡РµСЃРєРёРµ РіСЂР°РЅРёС‡РЅС‹Рµ
        СѓСЃР»РѕРІРёСЏ. РћРЅ РѕС‚РІРµС‡Р°РµС‚ С‚РѕР»СЊРєРѕ РЅР° РІРѕРїСЂРѕСЃ: СЃСѓРјРјР° РІСЃРµС… outward-РІРµРєС‚РѕСЂРѕРІ S
        РІРѕРєСЂСѓРі РєР°Р¶РґРѕРіРѕ CV СЂР°РІРЅР° РЅСѓР»СЋ?
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
        Р’РµСЂРёС„РёРєР°С†РёСЏ РіРµРѕРјРµС‚СЂРёРё С‡РµСЂРµР· РїРѕСЃС‚РѕСЏРЅРЅРѕРµ РїРѕР»Рµ СЃРєРѕСЂРѕСЃС‚Рё.

        Р¤РёР·РёС‡РµСЃРєРёРµ BC РЅР°РјРµСЂРµРЅРЅРѕ РЅРµ РїСЂРёРјРµРЅСЏСЋС‚СЃСЏ: wall/symmetry/inlet РјРµРЅСЏСЋС‚
        РїРѕСЃС‚РѕСЏРЅРЅРѕРµ РїРѕР»Рµ Рё РїСЂРµРІСЂР°С‰Р°СЋС‚ РіРµРѕРјРµС‚СЂРёС‡РµСЃРєСѓСЋ РїСЂРѕРІРµСЂРєСѓ РІ РїСЂРѕРІРµСЂРєСѓ
        РїРѕСЃС‚Р°РЅРѕРІРєРё Р“РЈ. Р”Р»СЏ РїСЂРѕРІРµСЂРєРё СЂР°Р±РѕС‡РµР№ FVM-РЅРµРІСЏР·РєРё РёСЃРїРѕР»СЊР·СѓР№С‚Рµ
        compute_residual().

        РђСЂРіСѓРјРµРЅС‚С‹:
            u_const: С‚РµСЃС‚РѕРІРѕРµ РїРѕР»Рµ (N, 2); None в†’ РѕРґРЅРѕСЂРѕРґРЅС‹Р№ РїРѕС‚РѕРє (1, 0)
            tol: РґРѕРїСѓСЃРє

        Р’РѕР·РІСЂР°С‰Р°РµС‚:
            True РµСЃР»Рё max|R_i| < tol
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
        """РЎС‚Р°С‚РёСЃС‚РёРєР° РїРѕ СЃРµС‚РєРµ Рё РїРѕС‚РѕРєР°Рј (РЅРµ С‚СЂРµР±СѓРµС‚ u)."""
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
            'mesh_fingerprint': self._mesh_fingerprint,
            'device': str(self._device),
            'geometry_is_dimensionless': self._geometry_is_dimensionless,
            'geometry_length_scale': self._geometry_length_scale,
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

    @property
    def mesh_fingerprint(self) -> str:
        """Stable SHA-256 identity of the dimensional mesh arrays."""
        return self._mesh_fingerprint

    # в”Ђв”Ђ РџСЂРµРґРїРёСЃР°РЅРЅС‹Рµ РіСЂР°РЅРёС‡РЅС‹Рµ СѓСЃР»РѕРІРёСЏ в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ

    def set_bc_velocity(
        self,
        boundary_type: str,
        u_bc: torch.Tensor,
    ) -> "FVMResiduals":
        """
        РЈСЃС‚Р°РЅР°РІР»РёРІР°РµС‚ РїСЂРµРґРїРёСЃР°РЅРЅС‹Рµ СЃРєРѕСЂРѕСЃС‚Рё РґР»СЏ РіСЂР°РЅРµР№ Р·Р°РґР°РЅРЅРѕРіРѕ С‚РёРїР° Р“РЈ.

        РџРѕСЃР»Рµ РІС‹Р·РѕРІР° РіСЂР°РЅРё СЌС‚РѕРіРѕ С‚РёРїР° Р±СѓРґСѓС‚ РёСЃРїРѕР»СЊР·РѕРІР°С‚СЊ u_bc РІРјРµСЃС‚Рѕ
        РїСЂРµРґСЃРєР°Р·Р°РЅРёСЏ РЅРµР№СЂРѕСЃРµС‚Рё (hard BC).

        РђСЂРіСѓРјРµРЅС‚С‹:
            boundary_type: С‚РёРї Р“РЈ вЂ” 'inlet', 'outlet', 'symmetry', ...
            u_bc: (2,) вЂ” РѕРґРЅРѕСЂРѕРґРЅР°СЏ СЃРєРѕСЂРѕСЃС‚СЊ РґР»СЏ РІСЃРµС… РіСЂР°РЅРµР№ С‚РёРїР°,
                  (K, 2) вЂ” РїСЂРѕС„РёР»СЊ, РїРѕ РѕРґРЅРѕРјСѓ РІРµРєС‚РѕСЂСѓ РЅР° РєР°Р¶РґСѓСЋ РіСЂР°РЅСЊ С‚РёРїР°
                  (K = РєРѕР»РёС‡РµСЃС‚РІРѕ РіСЂР°РЅРµР№ РґР°РЅРЅРѕРіРѕ С‚РёРїР° РІ bnd_type)

        Р’РѕР·РІСЂР°С‰Р°РµС‚:
            self РґР»СЏ С†РµРїРѕС‡РєРё: fvm.set_bc_velocity('inlet', u_inf).to('cuda')

        РџСЂРёРјРµСЂ:
            >>> fvm.set_bc_velocity('inlet', torch.tensor([10.0, 0.0]))
        """
        code = BC_CODES.get(boundary_type.lower())
        if code is None:
            raise ValueError(
                f"РќРµРёР·РІРµСЃС‚РЅС‹Р№ С‚РёРї Р“РЈ: {boundary_type!r}. "
                f"Р”РѕРїСѓСЃС‚РёРјС‹Рµ: {list(BC_CODES)}"
            )

        type_mask = (self.bnd_type == code)           # (B,) bool
        n_faces = int(type_mask.sum())
        if n_faces == 0:
            logger.warning("set_bc_velocity: РЅРµС‚ РіСЂР°РЅРµР№ С‚РёРїР° '%s'", boundary_type)
            return self

        u_bc = u_bc.detach().to(device=self._device, dtype=self._dtype)

        if u_bc.dim() == 1 and u_bc.shape[0] == 2:
            self._u_prescribed[type_mask] = u_bc.unsqueeze(0).expand(n_faces, 2)
        elif u_bc.dim() == 2 and u_bc.shape == (n_faces, 2):
            self._u_prescribed[type_mask] = u_bc
        else:
            raise ValueError(
                f"u_bc РґРѕР»Р¶РµРЅ РёРјРµС‚СЊ С„РѕСЂРјСѓ (2,) РёР»Рё ({n_faces}, 2), "
                f"РїРѕР»СѓС‡РµРЅРѕ {tuple(u_bc.shape)}"
            )

        self._prescribed_mask[type_mask] = True
        logger.info(
            "set_bc_velocity: '%s' (%d РіСЂР°РЅРµР№), u_bc=%s",
            boundary_type, n_faces,
            u_bc.tolist() if u_bc.dim() == 1 else "<profile>",
        )
        return self

    def set_inlet_velocity(self, u_bc: torch.Tensor) -> "FVMResiduals":
        """
        Р—Р°РґР°С‘С‚ РїСЂРµРґРїРёСЃР°РЅРЅС‹Р№ РїСЂРѕС„РёР»СЊ СЃРєРѕСЂРѕСЃС‚Рё РЅР° inlet.

        РЈРґРѕР±РЅР°СЏ РѕР±С‘СЂС‚РєР° РЅР°Рґ set_bc_velocity('inlet', u_bc).

        РђСЂРіСѓРјРµРЅС‚С‹:
            u_bc: (2,) вЂ” РѕРґРЅРѕСЂРѕРґРЅР°СЏ СЃРєРѕСЂРѕСЃС‚СЊ РёР»Рё (K, 2) вЂ” РїСЂРѕС„РёР»СЊ
        """
        return self.set_bc_velocity('inlet', u_bc)

    def reset_prescribed(
        self,
        boundary_type: Optional[str] = None,
    ) -> "FVMResiduals":
        """
        РЎР±СЂР°СЃС‹РІР°РµС‚ РїСЂРµРґРїРёСЃР°РЅРЅС‹Рµ СЃРєРѕСЂРѕСЃС‚Рё: РїРµСЂРµС…РѕРґРёС‚ Рє РјСЏРіРєРѕРјСѓ BC
        (u_face = РїСЂРµРґСЃРєР°Р·Р°РЅРёРµ РЅРµР№СЂРѕСЃРµС‚Рё РІ РіСЂР°РЅРёС‡РЅРѕРј СѓР·Р»Рµ).

        РђСЂРіСѓРјРµРЅС‚С‹:
            boundary_type: None вЂ” СЃР±СЂРѕСЃ РґР»СЏ РІСЃРµС… С‚РёРїРѕРІ;
                           СЃС‚СЂРѕРєР° вЂ” СЃР±СЂРѕСЃ С‚РѕР»СЊРєРѕ РґР»СЏ СѓРєР°Р·Р°РЅРЅРѕРіРѕ С‚РёРїР°

        Р’РѕР·РІСЂР°С‰Р°РµС‚:
            self РґР»СЏ С†РµРїРѕС‡РєРё РІС‹Р·РѕРІРѕРІ
        """
        if boundary_type is None:
            self._u_prescribed.zero_()
            self._prescribed_mask.fill_(False)
            logger.info("reset_prescribed: СЃР±СЂРѕС€РµРЅС‹ РІСЃРµ РїСЂРµРґРїРёСЃР°РЅРЅС‹Рµ BC")
        else:
            code = BC_CODES.get(boundary_type.lower())
            if code is None:
                raise ValueError(f"РќРµРёР·РІРµСЃС‚РЅС‹Р№ С‚РёРї Р“РЈ: {boundary_type!r}")
            type_mask = (self.bnd_type == code)
            self._u_prescribed[type_mask] = 0.0
            self._prescribed_mask[type_mask] = False
            logger.info("reset_prescribed: СЃР±СЂРѕС€РµРЅ С‚РёРї '%s'", boundary_type)
        return self

    # в”Ђв”Ђ РЈРїСЂР°РІР»РµРЅРёРµ СѓСЃС‚СЂРѕР№СЃС‚РІРѕРј в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ

    def to(
        self,
        device: Union[str, torch.device],
        dtype: Optional[torch.dtype] = None,
    ) -> "FVMResiduals":
        """
        РџРµСЂРµРЅРѕСЃРёС‚ РІСЃРµ РіРµРѕРјРµС‚СЂРёС‡РµСЃРєРёРµ С‚РµРЅР·РѕСЂС‹ РЅР° device Рё, РѕРїС†РёРѕРЅР°Р»СЊРЅРѕ, dtype.

        Р’РѕР·РІСЂР°С‰Р°РµС‚ self РґР»СЏ С†РµРїРѕС‡РєРё РІС‹Р·РѕРІРѕРІ:
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
        РџРµСЂРµСЃС‡РёС‚С‹РІР°РµС‚ РіРµРѕРјРµС‚СЂРёС‡РµСЃРєРёРµ С‚РµРЅР·РѕСЂС‹ РёР· РЅРѕРІРѕР№ СЃРµС‚РєРё (РґР»СЏ RAR-Р°РґР°РїС‚Р°С†РёРё).
        РЎРѕС…СЂР°РЅСЏРµС‚ С‚РµРєСѓС‰РµРµ СѓСЃС‚СЂРѕР№СЃС‚РІРѕ.

        РђСЂРіСѓРјРµРЅС‚С‹:
            new_mesh: Mesh2D РёР»Рё MeshData СЃ РѕР±РЅРѕРІР»С‘РЅРЅРѕР№ РіРµРѕРјРµС‚СЂРёРµР№
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
        logger.info("FVMResiduals РїРµСЂРµСЃС‚СЂРѕРµРЅ: N=%d, M=%d, B=%d", self.N, self.M, self.B)

    # в”Ђв”Ђ Р¤Р°Р±СЂРёРєРё в”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђв”Ђ

    @classmethod
    def from_mesh_data(cls, mesh_data) -> "FVMResiduals":
        """РЎРѕР·РґР°С‘С‚ РёР· data_handler.MeshData (Р·Р°РіСЂСѓР¶РµРЅРЅРѕРіРѕ РёР· .npz)."""
        return cls(mesh_data)

    def __repr__(self) -> str:
        return (f"FVMResiduals(N={self.N}, M={self.M}, B={self.B}, "
                f"device={self._device})")


def evaluate_fvm_metrics(
    model,
    fvm: FVMResiduals,
    node_x: torch.Tensor,
    node_y: torch.Tensor,
    node_sdf: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    reduction: str = "divergence_l2_volume",
    inlet_velocity: Optional[torch.Tensor] = None,
    parametric: bool = False,
    param_value: float = 1.0,
) -> Tuple[float, float]:
    """РћС†РµРЅРёРІР°РµС‚ FVM-РЅРµРІСЏР·РєСѓ РЅРµСЂР°Р·СЂС‹РІРЅРѕСЃС‚Рё РґР»СЏ РјРѕРґРµР»Рё РЅР° СѓР·Р»Р°С… СЃРµС‚РєРё.

    Args:
        model: PINN-РјРѕРґРµР»СЊ, РІС‹С…РѕРґ вЂ” (N, 3): (u, v, p).
        fvm: РїРѕСЃС‚СЂРѕРµРЅРЅС‹Р№ ``FVMResiduals`` (РіРµРѕРјРµС‚СЂРёСЏ РґРѕР»Р¶РЅР° Р»РµР¶Р°С‚СЊ РЅР° device).
        node_x, node_y, node_sdf: РєРѕРѕСЂРґРёРЅР°С‚С‹ Рё SDF СѓР·Р»РѕРІ РІ РїРѕСЂСЏРґРєРµ СѓР·Р»РѕРІ Mesh2D.
        device, dtype: С†РµР»РµРІС‹Рµ device/dtype.
        reduction: СЃС…РµРјР° СЂРµРґСѓРєС†РёРё (РїРѕ СѓРјРѕР»С‡Р°РЅРёСЋ divergence_l2_volume).
        inlet_velocity: РѕРїС†РёРѕРЅР°Р»СЊРЅР°СЏ РїСЂРµРґРїРёСЃР°РЅРЅР°СЏ СЃРєРѕСЂРѕСЃС‚СЊ РЅР° РІС…РѕРґРµ
            (2,) РёР»Рё (K, 2); РµСЃР»Рё None вЂ” inlet СЃС‡РёС‚Р°РµС‚СЃСЏ РјСЏРіРєРёРј BC.

    Returns:
        ``(fvm_mse, fvm_rms)``.
    """
    x = torch.as_tensor(node_x, dtype=dtype, device=device).reshape(-1, 1)
    y = torch.as_tensor(node_y, dtype=dtype, device=device).reshape(-1, 1)
    s = torch.as_tensor(node_sdf, dtype=dtype, device=device).reshape(-1, 1)
    inp = torch.cat([x, y, s], dim=1)
    if parametric:
        inp = torch.cat([inp, torch.full_like(x, float(param_value))], dim=1)
    with torch.no_grad():
        out = model(inp)
        u = out[:, :2]
    if inlet_velocity is not None:
        fvm.set_inlet_velocity(torch.as_tensor(inlet_velocity, device=device, dtype=dtype))
    else:
        fvm.reset_prescribed('inlet')
    raw, rms = fvm.loss_and_divergence_rms(u, reduction=reduction)
    return float(raw.detach().cpu()), float(rms.detach().cpu())

