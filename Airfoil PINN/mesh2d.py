#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Модуль построения FVM-сетки на основе точек коллокации (mesh2d.py).

Ключевые свойства:
  - Конформная граница: рёбра сетки лежат точно на контуре профиля и внешней
    границе расчётной области (boundary-fitted mesh).
  - Строгая constrained Delaunay triangulation (CDT):
    · вставка граничных рёбер — cavity-retriangulation (Sloan 1993);
    · восстановление свойства Делоне для внутренних рёбер — алгоритм Лоусона
      (edge-flip до сходимости).
  - Удаление «дырок»: flood-fill от центроида профиля (первичный, топологически
    надёжный); Shapely point-in-polygon (страховка + внешняя область).
  - Медианное дуальное разбиение: объём КО узла i = (1/3)×Σ площадей
    примыкающих треугольников.
  - Вектор площади внутренней грани (i,j):
    S_ij = rotate90(G_T2 − G_T1), направлен outward от CV_i.
  - Граничные дуальные грани (BoundaryFace): для каждого граничного ребра (i,j)
    area_vector = 0.5 × rotate90(c[j]−c[i]), направлен наружу области.
  - Проверка геометрической консервативности: Σ S_ij + Σ S_bnd ≈ 0 для каждого узла.

Интерфейс:
    >>> mesh = Mesh2D(sampler.points,
    ...               domain_coords=sampler.domain_coords,
    ...               airfoil_coords=sampler.foil_coords)
    >>> mesh.build()
    >>> print(mesh.summary())

    >>> # или напрямую из результата generate_all_points:
    >>> mesh = Mesh2D.from_sampler(sampler)
    >>> mesh.build()

Автор: Колмагоров И.Д.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from scipy.spatial import Delaunay, cKDTree

logger = logging.getLogger(__name__)

# Допуск детерминанта в тесте описанной окружности: предотвращает
# случайные флипы кокруговых или почти-кокруговых рёбер.
_CIRCUMCIRCLE_EPS: float = 1e-10


# ──────────────────────────────────────────────────────────────────────────────
# Структуры данных
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class MeshNode:
    """
    Узел сетки.

    Attributes:
        index: порядковый индекс в массиве узлов (0-based)
        point_id: id из CollocationPoint (-1 если добавлен из геометрии)
        x: X-координата [м]
        y: Y-координата [м]
        node_type: 'interior' или 'boundary'
        boundary_type: строковое значение BoundaryType
        on_airfoil: на профиле ли точка
    """
    index: int
    point_id: int
    x: float
    y: float
    node_type: str
    boundary_type: str
    on_airfoil: bool


@dataclass
class InternalFace:
    """
    Внутренняя грань между двумя соседними контрольными объёмами.

    Attributes:
        node_i: индекс «исходного» узла
        node_j: индекс «целевого» узла
        area_vector: shape (2,), outward от CV_i (от i к j)
        length: длина грани [м]
    """
    node_i: int
    node_j: int
    area_vector: np.ndarray
    length: float = field(init=False)

    def __post_init__(self) -> None:
        self.length = float(np.linalg.norm(self.area_vector))


@dataclass
class BoundaryFace:
    """
    Граничная дуальная грань на физической границе области или профиля.

    Attributes:
        node: индекс узла, которому принадлежит грань
        neighbor_node: второй конец граничного ребра
        boundary_type: тип граничного условия ('wall', 'inlet', 'outlet', ...)
        area_vector: shape (2,), направлен наружу из расчётной области
        length: длина (норма) вектора площади [м]
    """
    node: int
    neighbor_node: int
    boundary_type: str
    area_vector: np.ndarray
    length: float = field(init=False)

    def __post_init__(self) -> None:
        self.length = float(np.linalg.norm(self.area_vector))


@dataclass
class _PointAdapter:
    """Обёртка для строки structured numpy array из .npy-файла."""
    id: int
    x: float
    y: float
    boundary_type: str
    on_airfoil: bool
    on_boundary: bool
    segment_id: int


# ──────────────────────────────────────────────────────────────────────────────
# Геометрические примитивы
# ──────────────────────────────────────────────────────────────────────────────

def _cross2d(o: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """Знаковая площадь треугольника (o,a,b): >0 — CCW, <0 — CW."""
    return float((a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]))


def _segs_cross_strict(p1: np.ndarray, p2: np.ndarray,
                       p3: np.ndarray, p4: np.ndarray) -> bool:
    """True если отрезки (p1,p2) и (p3,p4) строго пересекаются (не в концах)."""
    d1 = _cross2d(p1, p2, p3)
    d2 = _cross2d(p1, p2, p4)
    d3 = _cross2d(p3, p4, p1)
    d4 = _cross2d(p3, p4, p2)
    if d1 * d2 < 0 and d3 * d4 < 0:
        return True
    # Вырожденные случаи (коллинеарность) намеренно считаем False:
    # при достаточно плотной сетке не встречаются.
    return False


def _in_circumcircle(pa: np.ndarray, pb: np.ndarray,
                     pc: np.ndarray, pd: np.ndarray) -> bool:
    """
    True если точка pd строго внутри описанной окружности треугольника (pa,pb,pc).
    Треугольник должен быть CCW (знаковая площадь > 0) для корректного знака.
    """
    ax, ay = pa[0] - pd[0], pa[1] - pd[1]
    bx, by = pb[0] - pd[0], pb[1] - pd[1]
    cx, cy = pc[0] - pd[0], pc[1] - pd[1]
    det = (ax * (by * (cx**2 + cy**2) - cy * (bx**2 + by**2))
           - ay * (bx * (cx**2 + cy**2) - cx * (bx**2 + by**2))
           + (ax**2 + ay**2) * (bx * cy - by * cx))
    return det > _CIRCUMCIRCLE_EPS


def _ensure_ccw(tri: Tuple[int, int, int], coords: np.ndarray) -> Tuple[int, int, int]:
    a, b, c = tri
    if _cross2d(coords[a], coords[b], coords[c]) < 0:
        return (a, c, b)
    return (a, b, c)


def _point_in_tri(p: np.ndarray, pa: np.ndarray,
                  pb: np.ndarray, pc: np.ndarray) -> bool:
    """True если точка p находится внутри (или на границе) треугольника (pa,pb,pc)."""
    d1 = _cross2d(p, pa, pb)
    d2 = _cross2d(p, pb, pc)
    d3 = _cross2d(p, pc, pa)
    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
    return not (has_neg and has_pos)


# ──────────────────────────────────────────────────────────────────────────────
# Ядро CDT: вставка ограниченного ребра (Sloan 1993)
# ──────────────────────────────────────────────────────────────────────────────

def _triangulate_polygon_delaunay(polygon: List[int],
                                   coords: np.ndarray) -> List[Tuple[int, int, int]]:
    """
    Делоне-триангуляция простого многоугольника polygon (список глобальных индексов),
    где первый и последний элемент — вершины ограниченного ребра.
    Рекурсивный алгоритм: выбирает вершину k ∈ polygon[1:-1], для которой
    никакая другая вершина не лежит внутри описанной окружности (v0, vk, vn).
    """
    n = len(polygon)
    if n < 3:
        return []
    if n == 3:
        tri = _ensure_ccw((polygon[0], polygon[1], polygon[2]), coords)
        return [tri]

    v0, vn = polygon[0], polygon[-1]
    best_k = 1
    min_violations = n + 1  # любое значение > возможного максимума
    for k in range(1, n - 1):
        vk = polygon[k]
        pa = coords[v0]
        pb = coords[vk]
        pc = coords[vn]
        if _cross2d(pa, pb, pc) < 0:
            pa, pc = pc, pa
        violations = 0
        for j in range(1, n - 1):
            if j == k:
                continue
            if _in_circumcircle(pa, pb, pc, coords[polygon[j]]):
                violations += 1
        if violations == 0:
            best_k = k
            break
        if violations < min_violations:
            min_violations = violations
            best_k = k

    vk = polygon[best_k]
    tri = _ensure_ccw((v0, vk, vn), coords)
    result = [tri]

    left = polygon[:best_k + 1]   # [v0, ..., vk]
    right = polygon[best_k:]      # [vk, ..., vn]
    result.extend(_triangulate_polygon_delaunay(left, coords))
    result.extend(_triangulate_polygon_delaunay(right, coords))
    return result


def _insert_constrained_edge(
    s: int, t: int,
    tris: List[Tuple[int, int, int]],
    coords: np.ndarray,
) -> List[Tuple[int, int, int]]:
    """
    Вставляет ребро (s,t) в триангуляцию через cavity-retriangulation.

    Алгоритм:
    1. Находит все треугольники, пересечённые отрезком (s,t).
    2. Собирает вершины полостей по обе стороны (s,t).
    3. Делоне-триангулирует каждую из двух полигональных полостей.

    Возвращает обновлённый список треугольников.
    """
    ps, pt = coords[s], coords[t]

    # ── 1. Найти пересечённые треугольники ─────────────────────────────────
    crossed_idx: Set[int] = set()
    for ti, tri in enumerate(tris):
        if s in tri or t in tri:
            continue
        a, b, c = tri
        for u, v in ((a, b), (b, c), (c, a)):
            if _segs_cross_strict(ps, pt, coords[u], coords[v]):
                crossed_idx.add(ti)
                break

    if not crossed_idx:
        return tris  # ребро уже присутствует или нет пересечений

    # ── 2. Извлечь вершины полостей ─────────────────────────────────────────
    upper: List[int] = []  # слева от s→t
    lower: List[int] = []  # справа от s→t
    for ti in crossed_idx:
        for v in tris[ti]:
            if v == s or v == t:
                continue
            side = _cross2d(ps, pt, coords[v])
            if side > 0 and v not in upper:
                upper.append(v)
            elif side < 0 and v not in lower:
                lower.append(v)
            elif side == 0:
                logger.warning(
                    "_insert_constrained_edge: вершина %d коллинеарна с ребром (%d,%d)",
                    v, s, t,
                )

    def _param(v: int) -> float:
        d = pt - ps
        return float(np.dot(coords[v] - ps, d))

    upper.sort(key=_param)
    lower.sort(key=_param)

    # ── 3. Обновить триангуляцию ────────────────────────────────────────────
    kept = [tri for ti, tri in enumerate(tris) if ti not in crossed_idx]

    upper_poly = [s] + upper + [t]
    lower_poly = [s] + lower[::-1] + [t]

    kept.extend(_triangulate_polygon_delaunay(upper_poly, coords))
    kept.extend(_triangulate_polygon_delaunay(lower_poly, coords))
    return kept


# ──────────────────────────────────────────────────────────────────────────────
# CDT: восстановление свойства Делоне — алгоритм Лоусона
# ──────────────────────────────────────────────────────────────────────────────

def _flip_to_delaunay(
    tris: List[Tuple[int, int, int]],
    coords: np.ndarray,
    constrained_edges: Set[Tuple[int, int]],
    verbose: bool = False,
) -> List[Tuple[int, int, int]]:
    """
    Алгоритм Лоусона: переворачивает все нарушающие критерий Делоне
    некonstrained рёбра до достижения полного CDT-состояния.

    Гарантирует, что для каждого внутреннего ребра (a,b), смежного с
    треугольниками T1=(a,b,c) и T2=(a,b,d), вершина d не лежит строго
    внутри описанной окружности T1. Ограниченные рёбра не переворачиваются.
    """
    tris = list(tris)
    n_flips = 0
    max_iter = len(tris) * len(tris) + 10

    # Строим edge_map один раз; обновляем инкрементально после каждого флипа.
    edge_map: Dict[Tuple[int, int], List[int]] = {}
    for ti, (a, b, c) in enumerate(tris):
        for u, v in ((a, b), (b, c), (c, a)):
            edge_map.setdefault((min(u, v), max(u, v)), []).append(ti)

    try:
        from tqdm import tqdm as _tqdm
        _have_tqdm = True
    except ImportError:
        _have_tqdm = False

    _pbar = (_tqdm(desc="    edge-flip", unit="flip", leave=False)
             if verbose and _have_tqdm else None)

    _it = 0
    for _it in range(max_iter):
        flipped = False
        for (vi, vj), adj in list(edge_map.items()):
            if len(adj) != 2 or (vi, vj) in constrained_edges:
                continue
            t1i, t2i = adj
            t1, t2 = tris[t1i], tris[t2i]
            c_v = next(v for v in t1 if v != vi and v != vj)
            d_v = next(v for v in t2 if v != vi and v != vj)

            pa, pb, pc_pt = coords[vi], coords[vj], coords[c_v]
            if _cross2d(pa, pb, pc_pt) < 0:
                pa, pb = pb, pa

            if _in_circumcircle(pa, pb, pc_pt, coords[d_v]):
                new_t1 = _ensure_ccw((vi, c_v, d_v), coords)
                new_t2 = _ensure_ccw((vj, d_v, c_v), coords)

                # Инкрементальное обновление edge_map:
                # удаляем t1i и t2i из всех рёбер старых треугольников
                for old_tri_idx, old_tri in ((t1i, t1), (t2i, t2)):
                    a2, b2, c2 = old_tri
                    for u2, v2 in ((a2, b2), (b2, c2), (c2, a2)):
                        k2 = (min(u2, v2), max(u2, v2))
                        if k2 in edge_map and old_tri_idx in edge_map[k2]:
                            edge_map[k2].remove(old_tri_idx)
                            if not edge_map[k2]:
                                del edge_map[k2]

                tris[t1i] = new_t1
                tris[t2i] = new_t2

                # добавляем t1i и t2i в рёбра новых треугольников
                for new_tri_idx, new_tri in ((t1i, new_t1), (t2i, new_t2)):
                    a2, b2, c2 = new_tri
                    for u2, v2 in ((a2, b2), (b2, c2), (c2, a2)):
                        k2 = (min(u2, v2), max(u2, v2))
                        edge_map.setdefault(k2, [])
                        if new_tri_idx not in edge_map[k2]:
                            edge_map[k2].append(new_tri_idx)

                n_flips += 1
                if _pbar is not None:
                    _pbar.update(1)
                    _pbar.set_postfix_str(f"iter {_it + 1}")
                flipped = True
                break  # перезапускаем обход edge_map с обновлёнными данными

        if not flipped:
            break

    if _pbar is not None:
        _pbar.close()
    logger.info("CDT edge-flip: %d переворотов за %d итераций", n_flips, _it + 1)
    return tris


# ──────────────────────────────────────────────────────────────────────────────
# Flood-fill удаление треугольников-«дырок»
# ──────────────────────────────────────────────────────────────────────────────

def _flood_fill_remove(
    tris: List[Tuple[int, int, int]],
    coords: np.ndarray,
    seed_points: List[np.ndarray],
    constrained_edges: Set[Tuple[int, int]],
) -> List[Tuple[int, int, int]]:
    """
    Удаляет все треугольники, достижимые из seed через adjacency-граф,
    не пересекая constrained_edges.

    Args:
        tris: список треугольников (a,b,c)
        coords: координаты узлов
        seed_points: список 2D-точек внутри удаляемых областей
        constrained_edges: рёбра, через которые нельзя «перейти»
    """
    n = len(tris)
    if n == 0:
        return []

    # Adjacency: для каждого ребра → список индексов треугольников
    edge_to_tris: Dict[Tuple[int, int], List[int]] = {}
    for ti, (a, b, c) in enumerate(tris):
        for u, v in ((a, b), (b, c), (c, a)):
            key = (min(u, v), max(u, v))
            edge_to_tris.setdefault(key, []).append(ti)

    # Центроиды для быстрого поиска кандидатов
    centroids = np.array([(coords[a] + coords[b] + coords[c]) / 3.0
                          for a, b, c in tris])

    # Найти стартовые треугольники: seed-точка должна лежать ВНУТРИ треугольника.
    # Берём k ближайших кандидатов по центроиду и проверяем PIT.
    tree = cKDTree(centroids)
    k_search = min(30, len(tris))
    seed_tris: Set[int] = set()
    for sp in seed_points:
        _, idxs = tree.query(sp, k=k_search)
        idxs_iter = [idxs] if np.ndim(idxs) == 0 else idxs
        found = False
        for idx in idxs_iter:
            a, b, c = tris[int(idx)]
            if _point_in_tri(sp, coords[a], coords[b], coords[c]):
                seed_tris.add(int(idx))
                found = True
                break
        if not found:
            # seed вне всех треугольников — пропускаем этот seed
            pass

    # DFS (обход в глубину; для задачи flood-fill даёт тот же результат, что BFS)
    visited: Set[int] = set()
    queue = list(seed_tris)
    while queue:
        ti = queue.pop()
        if ti in visited:
            continue
        visited.add(ti)
        a, b, c = tris[ti]
        for u, v in ((a, b), (b, c), (c, a)):
            edge_key = (min(u, v), max(u, v))
            if edge_key in constrained_edges:
                continue  # граничное ребро — не переходим
            for tj in edge_to_tris.get(edge_key, []):
                if tj not in visited:
                    queue.append(tj)

    return [tri for ti, tri in enumerate(tris) if ti not in visited]


# ──────────────────────────────────────────────────────────────────────────────
# Основной класс сетки
# ──────────────────────────────────────────────────────────────────────────────

class Mesh2D:
    """
    Конформная FVM-сетка на основе точек коллокации.

    Строит constrained Delaunay triangulation (CDT): граничные рёбра домена
    и профиля принудительно включаются в сетку (алгоритм Слоана).
    Удаление «дырок» производится тестом point-in-polygon (Shapely).

    Аргументы:
        points: List[CollocationPoint] из Collocation.py
        domain_coords: ndarray (Nd, 2) — упорядоченный контур внешней границы
                       (sampler.domain_coords). ОБЯЗАТЕЛЕН для CDT.
        airfoil_coords: ndarray (Na, 2) — контур профиля (sampler.foil_coords).
                        Если None — тело профиля не удаляется.
    """

    def __init__(
        self,
        points,
        domain_coords: Optional[np.ndarray] = None,
        airfoil_coords: Optional[np.ndarray] = None,
    ) -> None:
        self._raw_points = points
        self._domain_coords = (np.asarray(domain_coords, dtype=np.float64)
                               if domain_coords is not None else None)
        self._airfoil_coords = (np.asarray(airfoil_coords, dtype=np.float64)
                                if airfoil_coords is not None else None)

        self._nodes: List[MeshNode] = []
        self._coords: Optional[np.ndarray] = None
        self._triangles: Optional[np.ndarray] = None
        self._cv_volumes: Optional[np.ndarray] = None
        self._faces: List[InternalFace] = []
        self._boundary_faces: List[BoundaryFace] = []
        self._constrained_edges: Set[Tuple[int, int]] = set()
        self._built = False

    # ── Фабрики ───────────────────────────────────────────────────────────────

    @classmethod
    def from_sampler(cls, sampler, **kwargs) -> "Mesh2D":
        """
        Создаёт сетку напрямую из объекта CollocationPoints.

        Args:
            sampler: экземпляр CollocationPoints после generate_all_points()
        """
        return cls(
            sampler.points,
            domain_coords=getattr(sampler, "domain_coords", None),
            airfoil_coords=getattr(sampler, "foil_coords", None),
            **kwargs,
        )

    @classmethod
    def from_collocation_result(
        cls,
        result: dict,
        domain_coords: Optional[np.ndarray] = None,
        airfoil_coords: Optional[np.ndarray] = None,
    ) -> "Mesh2D":
        """
        Создаёт сетку из dict generate_all_points().

        Args:
            result: dict с ключом 'points'
            domain_coords: контур внешней границы
            airfoil_coords: контур профиля
        """
        return cls(result["points"],
                   domain_coords=domain_coords,
                   airfoil_coords=airfoil_coords)

    @classmethod
    def from_npy(
        cls,
        path: str,
        domain_coords: Optional[np.ndarray] = None,
        airfoil_coords: Optional[np.ndarray] = None,
    ) -> "Mesh2D":
        """Загружает точки из .npy structured array."""
        arr = np.load(path, allow_pickle=True)
        points = [
            _PointAdapter(
                id=int(row["id"]),
                x=float(row["x"]),
                y=float(row["y"]),
                boundary_type=str(row["boundary_type"]),
                on_airfoil=bool(row["on_airfoil"]),
                on_boundary=bool(row["on_boundary"]),
                segment_id=int(row["segment_id"]),
            )
            for row in arr
        ]
        return cls(points, domain_coords=domain_coords, airfoil_coords=airfoil_coords)

    # ── Публичный API ─────────────────────────────────────────────────────────

    def build(self, verbose: bool = True) -> None:
        """
        Строит CDT-сетку:
          1. Подготовка вершин (коллокация + геометрия)
          2. Начальная триангуляция Делоне (scipy)
          3. Вставка ограниченных граничных рёбер (CDT)
          4. Удаление «дырок» через Shapely point-in-polygon
          5. Вычисление объёмов КО (медианное дуальное разбиение)
          6. Вычисление внутренних граней (S_ij)
          7. Вычисление граничных дуальных граней (BoundaryFace)
          8. Проверка геометрической консервативности

        Args:
            verbose: показывать tqdm-прогресс-бар (по умолчанию True).
        """
        self._verbose = verbose

        try:
            from tqdm import tqdm as _tqdm
            _have_tqdm = True
        except ImportError:
            _have_tqdm = False

        step_names = [
            "подготовка вершин",
            "триангуляция Делоне (scipy)",
            "вставка CDT-рёбер",
            "удаление дырок",
            "объёмы КО",
            "внутренние грани",
            "граничные грани",
            "консервативность",
        ]
        step_fns = [
            self._prepare_vertices,
            self._initial_delaunay,
            self._insert_all_constrained_edges,
            self._remove_holes,
            self._compute_cv_volumes,
            self._compute_internal_faces,
            self._compute_boundary_faces,
            self._verify_conservativity,
        ]

        def _step_result(i: int) -> str:
            if i == 0:
                return f"{len(self._nodes)} узлов"
            if i == 1:
                return f"{len(self._tris_list)} треуг."
            if i == 2:
                return f"{len(self._triangles)} треуг. после CDT"
            if i == 3:
                return f"{len(self._triangles)} треуг. после удаления дырок"
            if i == 4:
                return "КО готовы"
            if i == 5:
                return f"{len(self._faces)} внутр. граней"
            if i == 6:
                return f"{len(self._boundary_faces)} граничн. граней"
            return "OK"

        if verbose and _have_tqdm:
            bar = _tqdm(
                total=len(step_fns),
                desc="Building mesh",
                unit="шаг",
                bar_format="{desc}: {n_fmt}/{total_fmt} [{bar}] {percentage:3.0f}%"
                           "  {elapsed}<{remaining}  {postfix}",
            )
            for i, (name, fn) in enumerate(zip(step_names, step_fns)):
                bar.set_postfix_str(f"{name}…")
                fn()
                bar.set_postfix_str(f"{name} → {_step_result(i)}")
                bar.update(1)
            bar.close()
        else:
            for fn in step_fns:
                fn()

        self._built = True
        logger.info(
            "Mesh2D (CDT): %d узлов, %d треугольников, %d внутр. граней, %d граничн. граней",
            len(self._nodes), len(self._triangles), len(self._faces),
            len(self._boundary_faces),
        )

    # ── Свойства ──────────────────────────────────────────────────────────────

    @property
    def nodes(self) -> List[MeshNode]:
        self._assert_built(); return self._nodes

    @property
    def coords(self) -> np.ndarray:
        self._assert_built(); return self._coords

    @property
    def triangles(self) -> np.ndarray:
        self._assert_built(); return self._triangles

    @property
    def cv_volumes(self) -> np.ndarray:
        self._assert_built(); return self._cv_volumes

    @property
    def internal_faces(self) -> List[InternalFace]:
        self._assert_built(); return self._faces

    @property
    def boundary_faces(self) -> List[BoundaryFace]:
        self._assert_built(); return self._boundary_faces

    # ── Вспомогательные методы ────────────────────────────────────────────────

    def get_node_faces(
        self, include_boundary: bool = True
    ) -> Dict[int, List[Tuple[int, np.ndarray]]]:
        """
        Для каждого узла — список (сосед, вектор_площади_грани outward).

        Args:
            include_boundary: если True, добавляет граничные дуальные грани
                              (neighbor_node — реальный индекс смежного граничного узла).
        """
        self._assert_built()
        result: Dict[int, List[Tuple[int, np.ndarray]]] = {
            i: [] for i in range(len(self._nodes))
        }
        for face in self._faces:
            result[face.node_i].append((face.node_j, face.area_vector.copy()))
            result[face.node_j].append((face.node_i, -face.area_vector))
        if include_boundary:
            for bface in self._boundary_faces:
                result[bface.node].append((bface.neighbor_node, bface.area_vector.copy()))
        return result

    def get_boundary_nodes(self, boundary_type: Optional[str] = None) -> List[MeshNode]:
        """Граничные узлы, опционально отфильтрованные по типу ГУ."""
        self._assert_built()
        if boundary_type is None:
            return [n for n in self._nodes if n.node_type == "boundary"]
        return [n for n in self._nodes
                if n.node_type == "boundary" and n.boundary_type == boundary_type]

    def summary(self) -> Dict:
        """Сводная статистика по сетке."""
        self._assert_built()
        bt_counts: Dict[str, int] = {}
        n_int = n_bnd = 0
        for nd in self._nodes:
            if nd.node_type == "interior":
                n_int += 1
            else:
                n_bnd += 1
                bt_counts[nd.boundary_type] = bt_counts.get(nd.boundary_type, 0) + 1
        return {
            "num_nodes": len(self._nodes),
            "num_interior_nodes": n_int,
            "num_boundary_nodes": n_bnd,
            "boundary_type_counts": bt_counts,
            "num_triangles": int(len(self._triangles)),
            "num_internal_faces": len(self._faces),
            "num_boundary_faces": len(self._boundary_faces),
            "cv_volume_min": float(self._cv_volumes.min()),
            "cv_volume_max": float(self._cv_volumes.max()),
            "cv_volume_mean": float(self._cv_volumes.mean()),
            "cv_volume_total": float(self._cv_volumes.sum()),
        }

    def save(self, path: str) -> None:
        """
        Сохраняет сетку в файл .npz (numpy compressed archive).

        Массивы в файле:
            coords            (N, 2)  — физические координаты узлов [м]
            triangles         (T, 3)  — индексы вершин треугольников
            cv_volumes        (N,)    — объёмы КО (медианное дуальное)
            int_face_node_i   (F,)    — узел i внутренней грани
            int_face_node_j   (F,)    — узел j внутренней грани
            int_face_area     (F, 2)  — вектор площади S_ij (outward от i)
            bnd_face_node     (Fb,)   — узел граничной грани
            bnd_face_neighbor (Fb,)   — смежный граничный узел
            bnd_face_type     (Fb,)   — тип ГУ ('wall', 'inlet', ...)
            bnd_face_area     (Fb, 2) — вектор площади (outward)
            node_type         (N,)    — 'interior' или 'boundary'
            node_bnd_type     (N,)    — строка типа ГУ узла

        Args:
            path: путь к выходному файлу (рекомендуется расширение .npz)
        """
        self._assert_built()

        int_area = (np.array([f.area_vector for f in self._faces], dtype=np.float64)
                    if self._faces else np.empty((0, 2), dtype=np.float64))
        bnd_area = (np.array([f.area_vector for f in self._boundary_faces], dtype=np.float64)
                    if self._boundary_faces else np.empty((0, 2), dtype=np.float64))

        np.savez_compressed(
            path,
            coords=self._coords,
            triangles=self._triangles,
            cv_volumes=self._cv_volumes,
            int_face_node_i=np.array([f.node_i for f in self._faces], dtype=np.int32),
            int_face_node_j=np.array([f.node_j for f in self._faces], dtype=np.int32),
            int_face_area=int_area,
            bnd_face_node=np.array([f.node for f in self._boundary_faces], dtype=np.int32),
            bnd_face_neighbor=np.array([f.neighbor_node for f in self._boundary_faces], dtype=np.int32),
            bnd_face_type=np.array([f.boundary_type for f in self._boundary_faces], dtype=object),
            bnd_face_area=bnd_area,
            node_type=np.array([nd.node_type for nd in self._nodes], dtype=object),
            node_bnd_type=np.array([nd.boundary_type for nd in self._nodes], dtype=object),
        )
        logger.info(
            "Mesh2D сохранена: %s  (%d узлов, %d треуг., %d вн. граней, %d гран. граней)",
            path, len(self._nodes), len(self._triangles),
            len(self._faces), len(self._boundary_faces),
        )

    def plot(
        self,
        show_triangulation: bool = True,
        show_cv_volumes: bool = True,
        show_face_vectors: bool = False,
        face_vector_scale: float = 1.0,
        save_path: Optional[str] = None,
        dpi: int = 150,
        show: bool = True,
    ) -> None:
        """Визуализация сетки.

        Args:
            show: вызывать plt.show() (False — только сохранить в файл).
        """
        import matplotlib.pyplot as plt
        from matplotlib.collections import PolyCollection
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize

        fig, ax = plt.subplots(figsize=(14, 9))
        c = self._coords

        if show_triangulation and show_cv_volumes:
            vols = self._cv_volumes
            norm = Normalize(vmin=vols.min(), vmax=vols.max())
            cmap = plt.cm.viridis
            polys = c[self._triangles]
            avg_vol = vols[self._triangles].mean(axis=1)
            pc = PolyCollection(polys, array=avg_vol, cmap=cmap, norm=norm,
                                edgecolors="grey", linewidths=0.3, alpha=0.8)
            ax.add_collection(pc)
            fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                         label="Ср. CV [м²]")
        elif show_triangulation:
            pc = PolyCollection(c[self._triangles], facecolors="none",
                                edgecolors="steelblue", linewidths=0.4)
            ax.add_collection(pc)

        if show_face_vectors:
            for face in self._faces:
                mid = 0.5 * (c[face.node_i] + c[face.node_j])
                sv = face.area_vector * face_vector_scale
                ax.annotate("", xy=mid + sv, xytext=mid,
                            arrowprops=dict(arrowstyle="->", color="darkorange", lw=0.6))

        _bc_colors = {
            "inlet": "blue", "outlet": "cyan", "wall": "red",
            "symmetry": "purple", "axis_symmetry": "magenta", "interior": "green",
        }
        by_key: Dict[str, list] = {}
        for nd in self._nodes:
            key = nd.boundary_type if nd.node_type == "boundary" else "interior"
            by_key.setdefault(key, []).append([nd.x, nd.y])

        for key, pts in by_key.items():
            arr = np.array(pts)
            color = _bc_colors.get(key, "grey")
            size = 2 if key == "interior" else 14
            ax.scatter(arr[:, 0], arr[:, 1], c=color, s=size, zorder=5,
                       label=f"{key} ({len(pts)})")

        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
        ax.set_xlabel("X, м"); ax.set_ylabel("Y, м")
        ax.set_title(
            f"CDT FVM-сетка: {len(self._nodes)} узлов, "
            f"{len(self._triangles)} треуг., {len(self._faces)} граней"
        )
        ax.legend(loc="upper right", fontsize=8, ncol=2)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
            logger.info("График сохранён: %s", save_path)
        if show:
            plt.show()
        else:
            plt.close()

    # ── Внутренние шаги построения ────────────────────────────────────────────

    def _prepare_vertices(self) -> None:
        """
        Строит единый пул вершин: все точки коллокации + недостающие
        вершины контуров (domain_coords, airfoil_coords), дедуплицированные
        по расстоянию < EPS.
        """
        EPS = 1e-9

        # ─ Начальный пул: коллокационные точки ─
        coords: List[List[float]] = []
        nodes: List[MeshNode] = []
        for idx, p in enumerate(self._raw_points):
            bt = p.boundary_type.value if hasattr(p.boundary_type, "value") else str(p.boundary_type)
            nt = "boundary" if p.on_boundary else "interior"
            coords.append([float(p.x), float(p.y)])
            nodes.append(MeshNode(
                index=idx, point_id=int(p.id),
                x=float(p.x), y=float(p.y),
                node_type=nt, boundary_type=bt,
                on_airfoil=bool(p.on_airfoil),
            ))

        coords_arr = (np.array(coords, dtype=np.float64).reshape(-1, 2)
                      if coords else np.empty((0, 2), dtype=np.float64))

        def _add_geom_point(xy: np.ndarray, nt: str, bt: str, on_airfoil: bool) -> int:
            """Добавляет вершину если её ещё нет, возвращает индекс."""
            nonlocal coords_arr, nodes
            if len(coords_arr) > 0:
                dists = np.linalg.norm(coords_arr - xy, axis=1)
                nearest = int(np.argmin(dists))
                if dists[nearest] < EPS:
                    return nearest
            idx = len(nodes)
            nodes.append(MeshNode(
                index=idx, point_id=-1,
                x=float(xy[0]), y=float(xy[1]),
                node_type=nt, boundary_type=bt,
                on_airfoil=on_airfoil,
            ))
            coords_arr = np.vstack([coords_arr, xy[None, :]])
            return idx

        # ─ Индексы геометрических контуров (для построения сегментов) ─
        self._domain_idx: List[int] = []
        self._foil_idx: List[int] = []

        if self._domain_coords is not None:
            for xy in self._domain_coords:
                # BC для вершин домена определим позже через KDTree по коллокации
                idx = _add_geom_point(xy, "boundary", "_domain", False)
                self._domain_idx.append(idx)

        if self._airfoil_coords is not None:
            for xy in self._airfoil_coords:
                idx = _add_geom_point(xy, "boundary", "wall", True)
                self._foil_idx.append(idx)

        self._coords = coords_arr
        self._nodes = nodes

        # ─ Уточнить BC геометрических вершин домена через ближайшую коллокацию ─
        if self._domain_coords is not None:
            bnd_pts = np.array([[p.x, p.y] for p in self._raw_points if p.on_boundary and not p.on_airfoil])
            bnd_bt = [p.boundary_type.value if hasattr(p.boundary_type, "value") else str(p.boundary_type)
                      for p in self._raw_points if p.on_boundary and not p.on_airfoil]
            if len(bnd_pts) > 0:
                tree = cKDTree(bnd_pts)
                for vi in self._domain_idx:
                    nd = self._nodes[vi]
                    if nd.boundary_type == "_domain":
                        _, nn = tree.query(self._coords[vi])
                        nd.boundary_type = bnd_bt[nn]

    def _initial_delaunay(self) -> None:
        """Начальная триангуляция Делоне (scipy) по всему пулу вершин."""
        tri = Delaunay(self._coords)
        self._tris_list: List[Tuple[int, int, int]] = [
            tuple(int(x) for x in row) for row in tri.simplices
        ]

    def _order_boundary_chain(self, vertex_indices: List[int],
                               ref_coords: Optional[np.ndarray] = None) -> List[int]:
        """
        Упорядочивает список вершин-индексов в замкнутую цепочку.

        Если ref_coords задан (упорядоченный контур геометрии), использует его
        для построения цепочки: для каждого ребра (P_k, P_{k+1}) находит все
        вершины вблизи него и сортирует по параметру вдоль ребра.
        Иначе — сортирует по углу от центроида (работает для выпуклых форм).
        """
        if len(vertex_indices) < 2:
            return vertex_indices

        c = self._coords

        if ref_coords is not None and len(ref_coords) >= 2:
            # Оцениваем типичный шаг между граничными точками коллокации
            # для задания перпендикулярного допуска
            if len(vertex_indices) >= 2:
                _pts = c[vertex_indices]
                _tree = cKDTree(_pts)
                _dists, _ = _tree.query(_pts, k=min(2, len(_pts)))
                if _dists.ndim > 1:
                    _step = float(np.median(_dists[:, -1]))
                else:
                    _step = float(np.median(_dists))
                perp_tol = max(_step * 0.05, 1e-9)
            else:
                perp_tol = 1e-9

            # Проецируем каждую вершину на ближайший сегмент ref_coords.
            # Ключевой критерий: ПЕРПЕНДИКУЛЯРНОЕ расстояние до прямой < perp_tol.
            ordered: List[Tuple[float, int]] = []
            cumlen = 0.0
            for k in range(len(ref_coords)):
                A = ref_coords[k]
                B = ref_coords[(k + 1) % len(ref_coords)]
                seg = B - A
                seg_len = float(np.linalg.norm(seg))
                if seg_len < 1e-12:
                    continue
                seg_unit = seg / seg_len
                for vi in vertex_indices:
                    dv = c[vi] - A
                    proj = float(np.dot(dv, seg_unit))       # вдоль сегмента
                    perp = abs(float(                         # перпендикулярно
                        dv[0] * seg_unit[1] - dv[1] * seg_unit[0]))
                    if perp < perp_tol and -perp_tol <= proj <= seg_len + perp_tol:
                        ordered.append((cumlen + max(0.0, proj), vi))
                cumlen += seg_len
            if ordered:
                # Дедуплицируем: оставляем первое вхождение по vertex index
                seen: Set[int] = set()
                dedup: List[Tuple[float, int]] = []
                for param, vi in sorted(ordered):
                    if vi not in seen:
                        dedup.append((param, vi))
                        seen.add(vi)
                return [vi for _, vi in dedup]

        # Fallback: сортировка по углу от центроида
        pts = c[vertex_indices]
        cx, cy = pts.mean(axis=0)
        angles = np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)
        order = np.argsort(angles)
        return [vertex_indices[i] for i in order]

    def _chain_to_edges(self, chain: List[int]) -> List[Tuple[int, int]]:
        """Замкнутая цепочка → список рёбер."""
        edges = []
        for i in range(len(chain)):
            a, b = chain[i], chain[(i + 1) % len(chain)]
            if a != b:
                edges.append((a, b))
        return edges

    def _build_constrained_edge_set(self) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        """
        Возвращает (domain_edges, foil_edges) — полные упорядоченные цепочки
        граничных рёбер по ВСЕМ граничным точкам коллокации.
        """
        # ─ Профиль ─
        foil_verts = [nd.index for nd in self._nodes if nd.on_airfoil]
        foil_chain = self._order_boundary_chain(foil_verts, self._airfoil_coords)
        foil_edges = self._chain_to_edges(foil_chain) if len(foil_chain) >= 3 else []

        # ─ Домен ─
        dom_verts = [nd.index for nd in self._nodes
                     if nd.node_type == "boundary" and not nd.on_airfoil]
        dom_chain = self._order_boundary_chain(dom_verts, self._domain_coords)
        dom_edges = self._chain_to_edges(dom_chain) if len(dom_chain) >= 3 else []

        return dom_edges, foil_edges

    def _insert_all_constrained_edges(self) -> None:
        """Строит полные цепочки граничных рёбер и вставляет пропущенные в CDT."""
        has_domain = any(nd.node_type == "boundary" and not nd.on_airfoil
                         for nd in self._nodes)
        has_foil = any(nd.on_airfoil for nd in self._nodes)

        if not has_domain and not has_foil:
            logger.warning("domain_coords не задан — CDT невозможен.")
            self._triangles = np.array(self._tris_list, dtype=np.int32)
            self._constrained_edges = set()
            self._foil_constrained_edges: Set[Tuple[int, int]] = set()
            return

        dom_edges, foil_edges = self._build_constrained_edge_set()
        required = dom_edges + foil_edges

        # Текущие рёбра триангуляции
        current_edges: Set[Tuple[int, int]] = set()
        for a, b, c in self._tris_list:
            for u, v in ((a, b), (b, c), (c, a)):
                current_edges.add((min(u, v), max(u, v)))

        missing = [(s, t) for s, t in required
                   if (min(s, t), max(s, t)) not in current_edges]

        if missing:
            logger.info("CDT: вставка %d пропущенных граничных рёбер из %d",
                        len(missing), len(required))
        else:
            logger.info("CDT: все %d граничных рёбер уже в триангуляции", len(required))

        _verbose = getattr(self, "_verbose", False)
        try:
            from tqdm import tqdm as _tqdm
            _have_tqdm = True
        except ImportError:
            _have_tqdm = False

        tris = list(self._tris_list)
        _edge_iter = (
            _tqdm(missing, desc="    вставка рёбер", unit="ребро",
                  leave=False, total=len(missing))
            if _verbose and _have_tqdm and missing else missing
        )
        for s, t in _edge_iter:
            tris = _insert_constrained_edge(s, t, tris, self._coords)

        self._constrained_edges = {(min(s, t), max(s, t)) for s, t in required}
        self._foil_constrained_edges = {(min(s, t), max(s, t)) for s, t in foil_edges}
        self._dom_constrained_edges = {(min(s, t), max(s, t)) for s, t in dom_edges}

        tris = _flip_to_delaunay(tris, self._coords, self._constrained_edges,
                                  verbose=_verbose)

        self._tris_list = tris
        self._triangles = np.array(tris, dtype=np.int32)

    def _remove_holes(self) -> None:
        """
        Удаляет треугольники «дырок»:

        1. Профиль (первичный метод): flood-fill от центроида профиля,
           останавливается на рёбрах _foil_constrained_edges.
        2. Страховка + внешняя область: Shapely point-in-polygon.
           Убирает любые оставшиеся треугольники внутри профиля и все
           треугольники вне расчётной области (нужно для невыпуклых доменов).
        3. Fallback при отсутствии Shapely: KDTree-эвристика для профиля;
           треугольники вне домена не удаляются (предупреждение в логе).
        """
        if not self._tris_list:
            return

        tris: List[Tuple[int, int, int]] = list(self._tris_list)
        n_before = len(tris)

        # ── 1. Flood-fill: профиль ───────────────────────────────────────────
        foil_ce: Set[Tuple[int, int]] = getattr(self, "_foil_constrained_edges", set())
        if self._airfoil_coords is not None and len(self._airfoil_coords) >= 3 and foil_ce:
            try:
                from shapely.geometry import Polygon as _SPolyFF
                foil_seed = np.array(
                    _SPolyFF(self._airfoil_coords).representative_point().coords[0],
                    dtype=np.float64,
                )
            except Exception:
                foil_seed = np.mean(np.asarray(self._airfoil_coords, dtype=np.float64), axis=0)
            tris = _flood_fill_remove(tris, self._coords, [foil_seed], foil_ce)
            logger.info("Hole removal (flood-fill, профиль): удалено %d треугольников",
                        n_before - len(tris))

        # ── 2. Shapely: страховка для профиля + вся внешняя область ─────────
        try:
            from shapely.geometry import Point as _SPt, Polygon as _SPoly

            tris_arr = np.array(tris, dtype=np.int32)
            if len(tris_arr) == 0:
                self._tris_list = []
                self._triangles = np.empty((0, 3), dtype=np.int32)
                return

            centroids = self._coords[tris_arr].mean(axis=1)
            mask = np.ones(len(tris_arr), dtype=bool)

            _verbose = getattr(self, "_verbose", False)
            try:
                from tqdm import tqdm as _tqdm
                _have_tqdm = True
            except ImportError:
                _have_tqdm = False

            if self._airfoil_coords is not None and len(self._airfoil_coords) >= 3:
                foil_poly = _SPoly(self._airfoil_coords)
                if not foil_poly.is_valid:
                    foil_poly = foil_poly.buffer(0)
                with ThreadPoolExecutor() as ex:
                    _it = ex.map(lambda pt: foil_poly.contains(_SPt(pt[0], pt[1])), centroids)
                    if _verbose and _have_tqdm:
                        _it = _tqdm(_it, total=len(centroids),
                                    desc="    профиль (Shapely)", unit="треуг.", leave=False)
                    foil_inside = np.fromiter(_it, dtype=bool, count=len(centroids))
                mask &= ~foil_inside

            if self._domain_coords is not None and len(self._domain_coords) >= 3:
                dom_poly = _SPoly(self._domain_coords)
                if not dom_poly.is_valid:
                    dom_poly = dom_poly.buffer(0)
                with ThreadPoolExecutor() as ex:
                    _it = ex.map(lambda pt: dom_poly.contains(_SPt(pt[0], pt[1])), centroids)
                    if _verbose and _have_tqdm:
                        _it = _tqdm(_it, total=len(centroids),
                                    desc="    домен (Shapely)", unit="треуг.", leave=False)
                    dom_inside = np.fromiter(_it, dtype=bool, count=len(centroids))
                mask &= dom_inside

            removed_shp = int((~mask).sum())
            if removed_shp:
                logger.info("Hole removal (Shapely): удалено ещё %d треугольников", removed_shp)
            tris = [tuple(int(x) for x in tris_arr[i])
                    for i in range(len(tris_arr)) if mask[i]]

        except ImportError:
            # Fallback: KDTree-эвристика только для профиля
            if self._airfoil_coords is not None and len(tris) == n_before:
                foil_center = np.asarray(self._airfoil_coords).mean(axis=0)
                foil_r = np.max(np.linalg.norm(
                    np.asarray(self._airfoil_coords) - foil_center, axis=1))
                tris_arr = np.array(tris, dtype=np.int32)
                centroids = self._coords[tris_arr].mean(axis=1)
                keep = [i for i, cen in enumerate(centroids)
                        if np.linalg.norm(cen - foil_center) >= foil_r * 0.9]
                removed_kt = len(tris) - len(keep)
                if removed_kt:
                    logger.info("Hole removal (KDTree fallback): удалено %d треугольников",
                                removed_kt)
                tris = [tuple(int(x) for x in tris_arr[i]) for i in keep]
            logger.warning("Shapely недоступна: треугольники вне домена не удаляются")

        logger.info("Hole removal итого: удалено %d/%d треугольников",
                    n_before - len(tris), n_before)
        self._tris_list = tris
        self._triangles = np.array(tris, dtype=np.int32) if tris else np.empty((0, 3), dtype=np.int32)

    def _compute_cv_volumes(self) -> None:
        """CV_area(i) = (1/3) × Σ area(T) по треугольникам, содержащим i."""
        c = self._coords
        tri = self._triangles
        a, b, k = tri[:, 0], tri[:, 1], tri[:, 2]
        areas = 0.5 * np.abs(
            (c[b, 0] - c[a, 0]) * (c[k, 1] - c[a, 1])
            - (c[k, 0] - c[a, 0]) * (c[b, 1] - c[a, 1])
        )
        cv = np.zeros(len(self._nodes), dtype=np.float64)
        np.add.at(cv, a, areas / 3.0)
        np.add.at(cv, b, areas / 3.0)
        np.add.at(cv, k, areas / 3.0)
        self._cv_volumes = cv

    def _compute_internal_faces(self) -> None:
        """
        Вычисляет два типа InternalFace:

        1. Внутреннее ребро (смежно с 2 треугольниками T1, T2):
           S_ij = rotate90(G_T2 − G_T1), outward от CV_i.

        2. Граничное ребро (смежно с 1 треугольником T):
           Дуальная грань M_ij → G_T делит CV_i и CV_j внутри T.
           S_ij = rotate90(G_T − M_ij), outward от CV_i.
           Без этой ветки геометрическая консервативность нарушается
           для всех граничных узлов.
        """
        coords = self._coords
        triangles = self._triangles
        centroids = coords[triangles].mean(axis=1)

        edge_map: Dict[Tuple[int, int], List[Tuple[int, int, int]]] = {}
        for t_idx, (a, b, c) in enumerate(triangles):
            for u, v in ((a, b), (b, c), (c, a)):
                key = (min(u, v), max(u, v))
                edge_map.setdefault(key, []).append((t_idx, u, v))

        faces: List[InternalFace] = []
        for (vi, vj), adj in edge_map.items():
            if len(adj) == 2:
                t1_idx, a1, b1 = adj[0]
                t2_idx, _, _ = adj[1]

                G1 = centroids[t1_idx]
                G2 = centroids[t2_idx]
                dG = G2 - G1
                S = np.array([-dG[1], dG[0]], dtype=np.float64)

                i, j = a1, b1
                if np.dot(S, coords[j] - coords[i]) < 0:
                    i, j = j, i
                    S = -S

                faces.append(InternalFace(node_i=i, node_j=j, area_vector=S))

            elif len(adj) == 1:
                # Граничное ребро: дуальная грань M_ij → G_T
                # Важно: используем vi/vj (канонические), а не a1/b1.
                # Флип (i,j)=(j,i); S=-S знако-инвариантен: dot(-S,c[a]-c[b])=dot(S,c[b]-c[a]).
                # Проверяем ориентацию через vi/vj напрямую.
                t1_idx, _, _ = adj[0]
                t1_tri = triangles[t1_idx]
                k_v = next(v for v in t1_tri if v != vi and v != vj)

                M_ij = 0.5 * (coords[vi] + coords[vj])
                G_T = (coords[vi] + coords[vj] + coords[k_v]) / 3.0
                dG = G_T - M_ij
                S = np.array([-dG[1], dG[0]], dtype=np.float64)

                if np.dot(S, coords[vj] - coords[vi]) < 0:
                    S = -S

                faces.append(InternalFace(node_i=vi, node_j=vj, area_vector=S))

        self._faces = faces

    def _compute_boundary_faces(self) -> None:
        """
        Строит граничные дуальные грани для каждого физического граничного ребра.

        Для ребра (i,j) принадлежащего ровно одному треугольнику T=(i,j,k):
            area_vector = 0.5 × rotate90(c[j]−c[i]), направлен outward (от k).
        Каждое ребро порождает по одному BoundaryFace для обоих его концов.
        """
        coords = self._coords
        triangles = self._triangles

        edge_map: Dict[Tuple[int, int], List[Tuple[int, Tuple[int, int, int]]]] = {}
        for t_idx, tri in enumerate(triangles):
            a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
            for u, v in ((a, b), (b, c), (c, a)):
                key = (min(u, v), max(u, v))
                edge_map.setdefault(key, []).append((t_idx, (a, b, c)))

        node_bt = [nd.boundary_type for nd in self._nodes]

        bfaces: List[BoundaryFace] = []
        for (vi, vj), adj in edge_map.items():
            if len(adj) != 1:
                continue  # внутреннее ребро

            _, (a, b, c) = adj[0]
            # k — третья вершина (не vi, не vj)
            k = next(v for v in (a, b, c) if v != vi and v != vj)

            edge_vec = coords[vj] - coords[vi]
            S_raw = np.array([-edge_vec[1], edge_vec[0]], dtype=np.float64)
            # Направить outward: от k наружу
            if np.dot(S_raw, coords[vi] - coords[k]) < 0:
                S_raw = -S_raw

            area_vec = 0.5 * S_raw

            bfaces.append(BoundaryFace(
                node=vi, neighbor_node=vj,
                boundary_type=node_bt[vi],
                area_vector=area_vec.copy(),
            ))
            bfaces.append(BoundaryFace(
                node=vj, neighbor_node=vi,
                boundary_type=node_bt[vj],
                area_vector=area_vec.copy(),
            ))

        self._boundary_faces = bfaces
        logger.info("BoundaryFaces: %d (по %d граничным рёбрам)",
                    len(bfaces), len(bfaces) // 2)

    def _verify_conservativity(self, tol_factor: float = 1e-8) -> bool:
        """
        Проверяет геометрическую консервативность для каждого узла:
            ||Σ S_ij (внутр.) + Σ S_bnd (граничн.)|| < tol

        Args:
            tol_factor: допуск задаётся как tol_factor × char_length.

        Returns:
            True если все узлы прошли проверку, False при нарушении.
        """
        n = len(self._nodes)
        flux_sum = np.zeros((n, 2), dtype=np.float64)

        for face in self._faces:
            flux_sum[face.node_i] += face.area_vector
            flux_sum[face.node_j] -= face.area_vector

        for bface in self._boundary_faces:
            flux_sum[bface.node] += bface.area_vector

        norms = np.linalg.norm(flux_sum, axis=1)

        bbox = self._coords.max(axis=0) - self._coords.min(axis=0)
        char_len = float(np.max(bbox))
        tol = tol_factor * char_len

        violations = np.where(norms > tol)[0]
        if len(violations) > 0:
            worst = violations[int(np.argmax(norms[violations]))]
            logger.warning(
                "Геометрическая консервативность нарушена: %d/%d узлов, "
                "макс. остаток %.3e (узел %d, tol=%.3e)",
                len(violations), n, float(norms[worst]), int(worst), tol,
            )
            return False

        logger.info("Геометрическая консервативность OK (tol=%.2e, char_len=%.4f)",
                    tol, char_len)
        return True

    def _assert_built(self) -> None:
        if not self._built:
            raise RuntimeError("Вызовите build() перед обращением к результатам.")
