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
from collections import Counter, deque
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
    s: int,
    t: int,
    tris: Dict[int, Tuple[int, int, int]],
    coords: np.ndarray,
    edge_map: Dict[Tuple[int, int], Set[int]],
    vertex_to_tris: Dict[int, Set[int]],
    next_id: int,
) -> Tuple[int, List[int]]:
    """
    Вставляет ребро (s,t) в триангуляцию через локальный обход коридора.

    Сложность O(deg(s) + k·log k) вместо O(T):
    1. Веерный поиск: находит стартовый треугольник за O(deg(s)).
    2. Обход коридора: идёт по смежным треугольникам через exit-ребро, O(k).
    3. Сбор и сортировка полостей: O(k·log k).
    4. Инкрементальное обновление tris/edge_map/vertex_to_tris: O(k).

    Изменяет tris, edge_map, vertex_to_tris in-place.
    Возвращает (next_id, collinear_verts).
    collinear_verts непуст, если ребро (s,t) требует разбиения: вершины лежат
    строго между s и t на прямой s→t. Caller должен заменить (s,t) на
    (s,v0),(v0,v1),...,(vn,t) в порядке возрастания параметра вдоль s→t.
    В этом случае триангуляция НЕ изменяется (caller переставляет подрёбра).
    """
    # Ребро уже в триангуляции?
    key_st = (min(s, t), max(s, t))
    if key_st in edge_map and edge_map[key_st]:
        return next_id, []

    ps, pt = coords[s], coords[t]
    d = pt - ps

    # ── Фаза 1: Веерный поиск стартового треугольника ───────────────────────
    # Треугольники вокруг s образуют угловой веер.
    # Ищем T=(s,p,q) CCW, в угловой сектор которого попадает d = pt−ps:
    #   dp×d ≥ 0  AND  dq×d ≤ 0,
    #   где dp = coords[p]−coords[s], dq = coords[q]−coords[s].
    start_tid: Optional[int] = None
    for tid in list(vertex_to_tris.get(s, set())):
        a, b, c = tris[tid]
        others = [v for v in (a, b, c) if v != s]
        if len(others) != 2:
            continue
        p, q = others[0], others[1]

        # Прямой сосед — ребро (s,t) уже есть как сторона треугольника
        if t in (p, q):
            return next_id, []

        # Привести (s, p, q) к CCW
        if _cross2d(coords[s], coords[p], coords[q]) < 0:
            p, q = q, p

        dp = coords[p] - coords[s]
        dq = coords[q] - coords[s]
        c1 = dp[0] * d[1] - dp[1] * d[0]   # dp × d
        c2 = dq[0] * d[1] - dq[1] * d[0]   # dq × d

        if c1 >= -1e-12 and c2 <= 1e-12:    # d попадает в угловой сектор
            start_tid = tid
            break

    # Fallback для граничных вершин (неполный веер): ищем в веере ребро
    # противоположной стороны, которое строго пересекает (ps, pt).
    if start_tid is None:
        for tid in list(vertex_to_tris.get(s, set())):
            a, b, c = tris[tid]
            others = [v for v in (a, b, c) if v != s]
            if len(others) != 2:
                continue
            p, q = others[0], others[1]
            if _cross2d(ps, pt, coords[p]) * _cross2d(ps, pt, coords[q]) < 0:
                start_tid = tid
                break

    # ── Fallback 2: глобальный поиск O(T) для «зазорных» рёбер ────────────────
    # Веерный поиск проваливается для граничных вершин, у которых d направлен
    # вдоль границы (хвостовая кромка, почти коллинеарные точки). В этом случае
    # ни один сектор веера не содержит d. Глобальный fallback воспроизводит
    # поведение старого O(M·T) алгоритма для редких аварийных случаев.
    _direct_crossed: Optional[Set[int]] = None
    if start_tid is None:
        _fc: Set[int] = set()
        for _tid, (_a, _b, _c) in tris.items():
            if s in (_a, _b, _c) or t in (_a, _b, _c):
                continue
            for _u, _v in ((_a, _b), (_b, _c), (_c, _a)):
                if _segs_cross_strict(ps, pt, coords[_u], coords[_v]):
                    _fc.add(_tid)
                    break
        if _fc:
            _direct_crossed = _fc
        else:
            # Ребро лежит полностью в зазоре (внутри профиля / за пределами
            # домена): пересекаемых треугольников нет. Flood-fill справится
            # через Shapely; предупреждение не нужно.
            logger.debug(
                "_insert_constrained_edge: ребро (%d,%d) в зазоре веера, "
                "пересечений нет — пропуск.", s, t,
            )
            return next_id, []

    crossed_ids: Set[int]
    if _direct_crossed is not None:
        # Глобальный fallback: все пересечённые треугольники уже собраны.
        crossed_ids = _direct_crossed
    else:
        # ── Фаза 1б: Выход из start_tid — start_tid НЕ входит в crossed_ids ──
        # start_tid содержит вершину s. Удалять его нельзя: это создаёт замещающие
        # треугольники (s,p,t)/(s,q,t) с вершинами p,q у которых _param<0
        # («позади» s). При граничном exit-ребре corridor обрывается, giant-
        # треугольники перекрывают существующие → V_i=0 → зажим 1e-30 → R_i≈10³⁰.
        # Используем start_tid только чтобы найти first_exit_edge коридора.
        a, b, c = tris[start_tid]
        first_exit_edge: Optional[Tuple[int, int]] = None
        for u, v in ((a, b), (b, c), (c, a)):
            key = (min(u, v), max(u, v))
            d1 = _cross2d(ps, pt, coords[u])
            d2 = _cross2d(ps, pt, coords[v])
            if d1 * d2 < 0:
                first_exit_edge = key
                break

        if first_exit_edge is None:
            return next_id, []  # t внутри start_tid или ребро (s,t) уже корректно

        first_nbrs = edge_map.get(first_exit_edge, set()) - {start_tid}
        if not first_nbrs:
            return next_id, []  # exit-ребро граничное, соседнего треугольника нет

        # ── Фаза 2: Обход коридора ───────────────────────────────────────────
        # Шагаем по смежным треугольникам вдоль сегмента (s,t).
        # Тест строгого пересечения (ps,pt) с ребром (u,v):
        #   d1 = cross2d(ps,pt, coords[u]),  d2 = cross2d(ps,pt, coords[v])
        #   d1·d2 < 0  ↔  u и v лежат по разные стороны прямой s→t.
        crossed_ids = set()
        current_tid = next(iter(first_nbrs))
        entry_edge: Optional[Tuple[int, int]] = first_exit_edge
        max_steps = len(tris) + 10

        for _ in range(max_steps):
            a, b, c = tris[current_tid]

            if t in (a, b, c):
                break   # достигли треугольника, содержащего вершину t

            crossed_ids.add(current_tid)

            exit_edge: Optional[Tuple[int, int]] = None
            for u, v in ((a, b), (b, c), (c, a)):
                key = (min(u, v), max(u, v))
                if key == entry_edge:
                    continue  # ребро входа не рассматриваем
                d1 = _cross2d(ps, pt, coords[u])
                d2 = _cross2d(ps, pt, coords[v])
                if d1 * d2 < 0:
                    exit_edge = key
                    break

            if exit_edge is None:
                break   # t находится внутри текущего треугольника

            neighbors = edge_map.get(exit_edge, set()) - {current_tid}
            if not neighbors:
                break   # граничное ребро — нет смежного треугольника

            entry_edge = exit_edge
            current_tid = next(iter(neighbors))

        if not crossed_ids:
            return next_id, []   # коридор пуст — ребро уже присутствует

    # ── Фаза 3: Сбор вершин полостей ────────────────────────────────────────
    upper: List[int] = []     # слева от s→t
    lower: List[int] = []     # справа от s→t
    collinear: List[int] = [] # строго между s и t на прямой s→t
    param_t2 = float(np.dot(d, d))  # |st|²

    for tid in crossed_ids:
        for v in tris[tid]:
            if v == s or v == t:
                continue
            side = _cross2d(ps, pt, coords[v])
            if side > 0 and v not in upper:
                upper.append(v)
            elif side < 0 and v not in lower:
                lower.append(v)
            elif side == 0 and v not in collinear:
                # Проверяем, лежит ли v строго между s и t
                param_v = float(np.dot(coords[v] - ps, d))
                if param_t2 > 0 and 1e-10 * param_t2 < param_v < param_t2 * (1.0 - 1e-10):
                    # v на отрезке (s,t) — требуется разбиение ребра.
                    # Возвращаем v в caller без изменения триангуляции.
                    collinear.append(v)
                    logger.warning(
                        "_insert_constrained_edge: вершина %d коллинеарна с (%d,%d)"
                        " → разбиение ребра", v, s, t,
                    )
                else:
                    # v на продолжении прямой, но вне отрезка — jitter как fallback
                    d_len = float(np.linalg.norm(d))
                    if d_len > 0:
                        perp = np.array([-d[1], d[0]], dtype=np.float64) * (1e-9 / d_len)
                        coords[v] = coords[v] + perp
                        side = _cross2d(ps, pt, coords[v])
                    if side > 0 and v not in upper:
                        upper.append(v)
                    elif side < 0 and v not in lower:
                        lower.append(v)

    if collinear:
        # Триангуляция НЕ изменялась — возвращаем коллинеарные вершины caller-у.
        return next_id, collinear

    def _param(v: int) -> float:
        return float(np.dot(coords[v] - ps, pt - ps))

    upper.sort(key=_param)
    lower.sort(key=_param)

    # ── Фаза 4: Инкрементальное обновление структур ──────────────────────────
    # 4а — удалить пересечённые треугольники из tris, edge_map, vertex_to_tris
    for tid in crossed_ids:
        a, b, c = tris[tid]
        for u, v in ((a, b), (b, c), (c, a)):
            key = (min(u, v), max(u, v))
            if key in edge_map:
                edge_map[key].discard(tid)
                if not edge_map[key]:
                    del edge_map[key]
        for w in (a, b, c):
            if w in vertex_to_tris:
                vertex_to_tris[w].discard(tid)
        del tris[tid]

    # 4б — триангуляция полостей (алгоритм Sloan 1993, без изменений)
    upper_poly = [s] + upper + [t]
    lower_poly = [s] + lower + [t]
    new_triangles = (
        _triangulate_polygon_delaunay(upper_poly, coords)
        + _triangulate_polygon_delaunay(lower_poly, coords)
    )

    # 4в — добавить новые треугольники с монотонно возрастающими ID
    for tri in new_triangles:
        tid = next_id
        next_id += 1
        tris[tid] = tri
        a, b, c = tri
        for u, v in ((a, b), (b, c), (c, a)):
            key = (min(u, v), max(u, v))
            edge_map.setdefault(key, set()).add(tid)
        for w in (a, b, c):
            vertex_to_tris.setdefault(w, set()).add(tid)

    return next_id, []


# ──────────────────────────────────────────────────────────────────────────────
# CDT: восстановление свойства Делоне — алгоритм Лоусона
# ──────────────────────────────────────────────────────────────────────────────

def _flip_to_delaunay(
    tris: Dict[int, Tuple[int, int, int]],
    coords: np.ndarray,
    constrained_edges: Set[Tuple[int, int]],
    edge_map: Dict[Tuple[int, int], Set[int]],
    verbose: bool = False,
) -> None:
    """
    Алгоритм Лоусона: переворачивает все нарушающие критерий Делоне
    некonstrained рёбра до достижения полного CDT-состояния.

    Принимает актуальный edge_map извне (не перестраивает с нуля).
    Работает in-place по tris и edge_map: возвращает None.

    Использует Set вместо List в edge_map: discard() — O(1) вместо remove() — O(n).
    """
    n_flips = 0
    max_iter = len(tris) * len(tris) + 10

    _it = 0
    for _it in range(max_iter):
        flipped = False
        for (vi, vj), adj in list(edge_map.items()):
            if len(adj) != 2 or (vi, vj) in constrained_edges:
                continue
            t1i, t2i = tuple(adj)
            if t1i not in tris or t2i not in tris:
                continue
            t1, t2 = tris[t1i], tris[t2i]
            c_v = next(v for v in t1 if v != vi and v != vj)
            d_v = next(v for v in t2 if v != vi and v != vj)

            pa, pb, pc_pt = coords[vi], coords[vj], coords[c_v]
            if _cross2d(pa, pb, pc_pt) < 0:
                pa, pb = pb, pa

            if _in_circumcircle(pa, pb, pc_pt, coords[d_v]):
                new_t1 = _ensure_ccw((vi, c_v, d_v), coords)
                new_t2 = _ensure_ccw((vj, d_v, c_v), coords)

                # Инкрементальное обновление edge_map (Set-версия):
                # удаляем t1i и t2i из рёбер старых треугольников
                for old_tid, old_tri in ((t1i, t1), (t2i, t2)):
                    a2, b2, c2 = old_tri
                    for u2, v2 in ((a2, b2), (b2, c2), (c2, a2)):
                        k2 = (min(u2, v2), max(u2, v2))
                        if k2 in edge_map:
                            edge_map[k2].discard(old_tid)
                            if not edge_map[k2]:
                                del edge_map[k2]

                tris[t1i] = new_t1
                tris[t2i] = new_t2

                # добавляем t1i и t2i в рёбра новых треугольников
                for new_tid, new_tri in ((t1i, new_t1), (t2i, new_t2)):
                    a2, b2, c2 = new_tri
                    for u2, v2 in ((a2, b2), (b2, c2), (c2, a2)):
                        k2 = (min(u2, v2), max(u2, v2))
                        edge_map.setdefault(k2, set()).add(new_tid)

                n_flips += 1
                if verbose:
                    print(f"\r    edge-flip: {n_flips} flip, iter {_it + 1}   ",
                          end="", flush=True)
                flipped = True
                break  # перезапускаем обход edge_map с обновлёнными данными

        if not flipped:
            break

    if verbose and n_flips > 0:
        print(f"\r    edge-flip: {n_flips} flip за {_it + 1} итераций — готово        ")
    logger.info("CDT edge-flip: %d переворотов за %d итераций", n_flips, _it + 1)


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
        self._conservativity_ok: Optional[bool] = None

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
            if i == 7:
                if self._conservativity_ok is None:
                    return "не запускалась"
                return "OK" if self._conservativity_ok else "НАРУШЕНА ⚠"
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
            domain_coords=(self._domain_coords if self._domain_coords is not None
                           else np.empty((0, 2), dtype=np.float64)),
            airfoil_coords=(self._airfoil_coords if self._airfoil_coords is not None
                            else np.empty((0, 2), dtype=np.float64)),
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

    def plot_cv_volume_map(
        self,
        save_path: Optional[str] = None,
        dpi: int = 150,
        show: bool = True,
        log_scale: bool = True,
    ) -> None:
        """Карта объёмов контрольных объёмов (CV) по узлам сетки.

        Каждый узел окрашен своим значением CV-объёма. Цветовая шкала
        логарифмическая (при log_scale=True), что делает видимыми как
        мелкие (граничные) CV, так и крупные (интерьерные).
        Узел с минимальным объёмом помечается красным крестом.

        Args:
            save_path: путь для сохранения PNG (None — не сохранять).
            dpi:       разрешение при сохранении.
            show:      вызывать plt.show() (False — только файл).
            log_scale: логарифмическая цветовая шкала (рекомендуется).
        """
        self._assert_built()
        import matplotlib.pyplot as plt
        import matplotlib.tri as mtri
        from matplotlib.colors import LogNorm, Normalize

        vols = self._cv_volumes          # (N,) numpy float64
        coords = self._coords            # (N, 2)
        tris = np.array(self._triangles) # (T, 3)

        min_vol = float(vols.min())
        min_idx = int(vols.argmin())
        max_vol = float(vols.max())

        fig, ax = plt.subplots(figsize=(14, 8))

        triang = mtri.Triangulation(coords[:, 0], coords[:, 1], tris)

        if log_scale and min_vol > 0:
            norm = LogNorm(vmin=max(min_vol, max_vol * 1e-8), vmax=max_vol)
        else:
            norm = Normalize(vmin=min_vol, vmax=max_vol)

        tcf = ax.tricontourf(triang, vols, levels=64, cmap="plasma", norm=norm)
        fig.colorbar(tcf, ax=ax, label="Объём CV [м²]",
                     format="%.2e" if not log_scale else None)

        # Граничные узлы поверх контуров
        bnd_mask = np.array([nd.node_type == "boundary" for nd in self._nodes])
        if bnd_mask.any():
            ax.scatter(coords[bnd_mask, 0], coords[bnd_mask, 1],
                       c="white", s=6, zorder=4, linewidths=0)

        # Узел с минимальным объёмом
        ax.scatter(coords[min_idx, 0], coords[min_idx, 1],
                   c="red", s=80, marker="x", linewidths=1.5,
                   zorder=6, label=f"min CV={min_vol:.2e} (узел {min_idx})")

        ax.set_aspect("equal")
        ax.set_xlabel("X, м")
        ax.set_ylabel("Y, м")
        scale_str = " (лог. шкала)" if log_scale else ""
        ax.set_title(
            f"Карта объёмов CV{scale_str}:  min={min_vol:.2e}  max={max_vol:.2e}  "
            f"узлов={len(self._nodes)}"
        )
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.2)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
            logger.info("Карта CV сохранена: %s", save_path)
        if show:
            plt.show()
        else:
            plt.close()

    def plot_cv_volume_hist(
        self,
        bins: int = 100,
        save_path: Optional[str] = None,
        dpi: int = 150,
        show: bool = True,
    ) -> None:
        """Гистограмма распределения объёмов CV по 100 бинам.

        Ось X — объём в логарифмической шкале (бины равномерны в log-пространстве),
        ось Y — количество узлов (лог. шкала). Вертикальные линии отмечают
        минимум, медиану и максимум.

        Args:
            bins:      количество бинов (по умолчанию 100).
            save_path: путь для сохранения PNG (None — не сохранять).
            dpi:       разрешение при сохранении.
            show:      вызывать plt.show() (False — только файл).
        """
        self._assert_built()
        import matplotlib.pyplot as plt

        vols = self._cv_volumes   # numpy (N,)
        min_vol = float(vols.min())
        max_vol = float(vols.max())
        med_vol = float(np.median(vols))
        mean_vol = float(vols.mean())

        fig, ax = plt.subplots(figsize=(10, 5))

        # Логарифмические бины: корректно распределяют плотность по диапазону
        if min_vol > 0:
            log_bins = np.logspace(np.log10(max(min_vol, max_vol * 1e-10)),
                                   np.log10(max_vol), bins + 1)
            ax.set_xscale("log")
        else:
            log_bins = bins

        ax.hist(vols, bins=log_bins, color="steelblue", edgecolor="none", alpha=0.85)
        ax.set_yscale("log")

        # Вертикальные маркеры
        ax.axvline(min_vol,  color="red",    lw=1.2, ls="--",
                   label=f"min  = {min_vol:.2e}")
        ax.axvline(med_vol,  color="orange", lw=1.2, ls="-.",
                   label=f"медиана = {med_vol:.2e}")
        ax.axvline(mean_vol, color="green",  lw=1.2, ls=":",
                   label=f"среднее = {mean_vol:.2e}")
        ax.axvline(max_vol,  color="purple", lw=1.2, ls="--",
                   label=f"max  = {max_vol:.2e}")

        ax.set_xlabel("Объём CV [м²]")
        ax.set_ylabel("Количество узлов")
        ax.set_title(
            f"Распределение объёмов CV  ({len(vols)} узлов,  {bins} бинов)"
        )
        ax.legend(fontsize=8)
        ax.grid(True, which="both", alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
            logger.info("Гистограмма CV сохранена: %s", save_path)
        if show:
            plt.show()
        else:
            plt.close()

    @staticmethod
    def _boundary_rank(boundary_type: str) -> int:
        """Priority used when coincident points carry different labels."""
        ranks = {
            "interior": 0,
            "_domain": 1,
            "outlet": 2,
            "inlet": 2,
            "symmetry": 2,
            "axis_symmetry": 2,
            "wall": 3,
        }
        return ranks.get(str(boundary_type).lower(), 1)

    @classmethod
    def _prefer_boundary_type(cls, current: str, candidate: str) -> str:
        if cls._boundary_rank(candidate) > cls._boundary_rank(current):
            return candidate
        return current

    @classmethod
    def _merge_node_metadata(
        cls,
        node: MeshNode,
        node_type: str,
        boundary_type: str,
        on_airfoil: bool,
    ) -> None:
        if node_type == "boundary":
            node.node_type = "boundary"
        node.boundary_type = cls._prefer_boundary_type(node.boundary_type, boundary_type)
        node.on_airfoil = bool(node.on_airfoil or on_airfoil)
        if node.on_airfoil:
            node.node_type = "boundary"
            node.boundary_type = cls._prefer_boundary_type(node.boundary_type, "wall")

    @staticmethod
    def _clean_reference_chain(
        ref_coords: np.ndarray,
        eps: float = 1e-9,
    ) -> Tuple[np.ndarray, bool]:
        """Remove repeated closure/consecutive duplicates and report closedness."""
        ref = np.asarray(ref_coords, dtype=np.float64).reshape(-1, 2)
        if len(ref) == 0:
            return ref, False

        closed = bool(len(ref) > 2 and np.linalg.norm(ref[0] - ref[-1]) <= eps)
        cleaned = [ref[0]]
        for xy in ref[1:]:
            if np.linalg.norm(xy - cleaned[-1]) > eps:
                cleaned.append(xy)
        if len(cleaned) > 1 and np.linalg.norm(cleaned[0] - cleaned[-1]) <= eps:
            cleaned.pop()
            closed = True
        return np.asarray(cleaned, dtype=np.float64), closed

    # ── Внутренние шаги построения ────────────────────────────────────────────

    def _prepare_vertices(self) -> None:
        """
        Строит единый пул вершин: все точки коллокации + недостающие
        вершины контуров (domain_coords, airfoil_coords), дедуплицированные
        по расстоянию < EPS.
        """
        EPS = 1e-9

        nodes: List[MeshNode] = []
        coords_list: List[np.ndarray] = []
        spatial_hash: Dict[Tuple[int, int], List[int]] = {}

        def _bucket(xy: np.ndarray) -> Tuple[int, int]:
            return (int(np.floor(float(xy[0]) / EPS)),
                    int(np.floor(float(xy[1]) / EPS)))

        def _find_existing(xy: np.ndarray) -> Optional[int]:
            bx, by = _bucket(xy)
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for idx_existing in spatial_hash.get((bx + dx, by + dy), []):
                        if np.linalg.norm(coords_list[idx_existing] - xy) < EPS:
                            return idx_existing
            return None

        def _register(idx_new: int, xy: np.ndarray) -> None:
            spatial_hash.setdefault(_bucket(xy), []).append(idx_new)

        def _add_point(
            xy: np.ndarray,
            nt: str,
            bt: str,
            on_airfoil: bool,
            point_id: int = -1,
        ) -> int:
            xy = np.asarray(xy, dtype=np.float64)
            nearest = _find_existing(xy)
            if nearest is not None:
                self._merge_node_metadata(nodes[nearest], nt, bt, on_airfoil)
                return nearest

            idx_new = len(nodes)
            coords_list.append(xy.copy())
            nodes.append(MeshNode(
                index=idx_new, point_id=int(point_id),
                x=float(xy[0]), y=float(xy[1]),
                node_type=nt, boundary_type=bt,
                on_airfoil=bool(on_airfoil),
            ))
            if on_airfoil:
                self._merge_node_metadata(nodes[idx_new], "boundary", "wall", True)
            _register(idx_new, xy)
            return idx_new

        boundary_refs: List[np.ndarray] = []
        if self._domain_coords is not None and len(self._domain_coords) >= 2:
            boundary_refs.append(np.asarray(self._domain_coords, dtype=np.float64))
        if self._airfoil_coords is not None and len(self._airfoil_coords) >= 2:
            boundary_refs.append(np.asarray(self._airfoil_coords, dtype=np.float64))

        raw_boundary = np.array(
            [[float(p.x), float(p.y)] for p in self._raw_points if p.on_boundary],
            dtype=np.float64,
        )
        if len(raw_boundary) >= 2:
            raw_boundary = np.unique(np.round(raw_boundary, decimals=12), axis=0)
        if len(raw_boundary) >= 2:
            _btree = cKDTree(raw_boundary)
            _bdist, _ = _btree.query(raw_boundary, k=2)
            boundary_step = float(np.median(_bdist[:, 1]))
        else:
            boundary_step = 1.0
        airfoil_ref = (
            np.asarray(self._airfoil_coords, dtype=np.float64)
            if self._airfoil_coords is not None and len(self._airfoil_coords) >= 2
            else None
        )
        raw_airfoil = np.array(
            [[float(p.x), float(p.y)] for p in self._raw_points if p.on_airfoil],
            dtype=np.float64,
        )
        if len(raw_airfoil) >= 2:
            raw_airfoil = np.unique(np.round(raw_airfoil, decimals=12), axis=0)
        if len(raw_airfoil) >= 2:
            _atree = cKDTree(raw_airfoil)
            _adist, _ = _atree.query(raw_airfoil, k=2)
            airfoil_step = float(np.median(_adist[:, 1]))
        elif airfoil_ref is not None and len(airfoil_ref) >= 2:
            _seg = np.linalg.norm(np.diff(airfoil_ref, axis=0), axis=1)
            _seg = _seg[_seg > EPS]
            airfoil_step = float(np.median(_seg)) if len(_seg) else boundary_step
        else:
            airfoil_step = boundary_step

        near_boundary_tol = max(EPS * 10.0, boundary_step * 0.02)
        near_airfoil_tol = max(EPS * 10.0, airfoil_step * 0.15)

        def _point_to_chain_distance(xy: np.ndarray, ref: np.ndarray) -> float:
            best = float("inf")
            for i_ref in range(len(ref) - 1):
                a_ref = ref[i_ref]
                b_ref = ref[i_ref + 1]
                seg_ref = b_ref - a_ref
                denom_ref = float(np.dot(seg_ref, seg_ref))
                if denom_ref <= 1e-30:
                    dist_ref = float(np.linalg.norm(xy - a_ref))
                else:
                    t_ref = float(np.dot(xy - a_ref, seg_ref) / denom_ref)
                    t_ref = max(0.0, min(1.0, t_ref))
                    dist_ref = float(np.linalg.norm(xy - (a_ref + t_ref * seg_ref)))
                if dist_ref < best:
                    best = dist_ref
            return best

        def _point_to_ref_distance(xy: np.ndarray) -> float:
            best = float("inf")
            for ref in boundary_refs:
                best = min(best, _point_to_chain_distance(xy, ref))
            return best

        n_merged = 0
        n_skipped_near_boundary = 0
        n_skipped_near_airfoil = 0
        for p in self._raw_points:
            bt = p.boundary_type.value if hasattr(p.boundary_type, "value") else str(p.boundary_type)
            nt = "boundary" if p.on_boundary else "interior"
            xy = np.array([float(p.x), float(p.y)], dtype=np.float64)
            if (
                nt == "interior"
                and airfoil_ref is not None
                and _point_to_chain_distance(xy, airfoil_ref) <= near_airfoil_tol
            ):
                n_skipped_near_airfoil += 1
                continue
            if (
                nt == "interior"
                and boundary_refs
                and _point_to_ref_distance(xy) <= near_boundary_tol
            ):
                n_skipped_near_boundary += 1
                continue
            before = len(nodes)
            _add_point(
                xy,
                nt,
                bt,
                bool(p.on_airfoil),
                int(p.id),
            )
            if len(nodes) == before:
                n_merged += 1

        if n_merged:
            logger.info("Mesh2D: merged %d duplicate input points before Delaunay", n_merged)
        if n_skipped_near_boundary:
            logger.info(
                "Mesh2D: skipped %d interior points closer than %.3e m to a boundary",
                n_skipped_near_boundary, near_boundary_tol,
            )
        if n_skipped_near_airfoil:
            logger.info(
                "Mesh2D: skipped %d interior points closer than %.3e m to the airfoil",
                n_skipped_near_airfoil, near_airfoil_tol,
            )

        self._domain_idx = []
        self._foil_idx = []

        if self._domain_coords is not None:
            for xy in self._domain_coords:
                idx = _add_point(xy, "boundary", "_domain", False)
                self._domain_idx.append(idx)

        if self._airfoil_coords is not None:
            for xy in self._airfoil_coords:
                idx = _add_point(xy, "boundary", "wall", True)
                self._foil_idx.append(idx)

        self._coords = (np.vstack(coords_list).astype(np.float64)
                        if coords_list else np.empty((0, 2), dtype=np.float64))
        self._nodes = nodes

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
            for vi in self._domain_idx:
                if self._nodes[vi].boundary_type == "_domain":
                    self._nodes[vi].boundary_type = "interior"

        return

        coords_arr = np.empty((0, 2), dtype=np.float64)

        # ─ Начальный пул: коллокационные точки ─
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
            ref_chain, ref_closed = self._clean_reference_chain(ref_coords)
            if len(ref_chain) < 2:
                return vertex_indices
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
            seg_count = len(ref_chain) if ref_closed else len(ref_chain) - 1
            for k in range(seg_count):
                A = ref_chain[k]
                B = ref_chain[(k + 1) % len(ref_chain)]
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

    def _find_boundary_components(self, verts: List[int]) -> List[List[int]]:
        """
        Разбивает список вершин на связные компоненты по близости.

        Две вершины принадлежат одной компоненте, если расстояние между ними
        не превышает 3× медианного шага между соседними вершинами.
        Компоненты возвращаются в порядке убывания числа вершин.
        """
        if not verts:
            return []
        if len(verts) == 1:
            return [verts]

        c = self._coords[verts]
        tree = cKDTree(c)

        k = min(3, len(verts))
        dists, _ = tree.query(c, k=k)
        if dists.ndim > 1 and dists.shape[1] > 1:
            typical = float(np.median(dists[:, 1:][dists[:, 1:] > 0]))
        else:
            typical = float(np.median(dists[dists > 0])) if np.any(dists > 0) else 1e-3
        gap = max(typical * 3.0, 1e-9)

        # Union-Find
        parent = list(range(len(verts)))

        def _find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for i, j in tree.query_pairs(gap):
            ri, rj = _find(i), _find(j)
            if ri != rj:
                parent[ri] = rj

        groups: Dict[int, List[int]] = {}
        for i, vi in enumerate(verts):
            root = _find(i)
            groups.setdefault(root, []).append(vi)

        comps = sorted(groups.values(), key=len, reverse=True)
        if len(comps) > 1:
            logger.info("CDT: %d вершин домена разбиты на %d компонент(ы) границы",
                        len(verts), len(comps))
        return comps

    def _build_constrained_edge_set(self) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        """
        Возвращает (domain_edges, foil_edges) — полные упорядоченные цепочки
        граничных рёбер по ВСЕМ граничным точкам коллокации.

        Домен разбивается на связные компоненты; для каждой строится отдельная
        цепочка (замкнутая или открытая в зависимости от геометрии).
        """
        # ─ Профиль ─
        foil_verts = [nd.index for nd in self._nodes if nd.on_airfoil]
        foil_chain = self._order_boundary_chain(foil_verts, self._airfoil_coords)
        if len(foil_chain) >= 2:
            _c0, _c1 = self._coords[foil_chain[0]], self._coords[foil_chain[-1]]
            if len(foil_verts) >= 2:
                _fp = self._coords[foil_verts]
                _ft = cKDTree(_fp)
                _fd, _ = _ft.query(_fp, k=min(2, len(foil_verts)))
                _foil_step = float(np.median(_fd[:, -1])) if _fd.ndim > 1 else float(np.median(_fd))
            else:
                _foil_step = 1e-3
            if self._airfoil_coords is not None:
                _, foil_is_closed = self._clean_reference_chain(self._airfoil_coords)
            else:
                foil_is_closed = float(np.linalg.norm(_c1 - _c0)) < _foil_step * 2.5
            if foil_is_closed and len(foil_chain) >= 3:
                foil_edges = self._chain_to_edges(foil_chain)
            else:
                foil_edges = [(foil_chain[i], foil_chain[i + 1])
                              for i in range(len(foil_chain) - 1)
                              if foil_chain[i] != foil_chain[i + 1]]
            if not foil_is_closed:
                logger.info("CDT: профиль незамкнут — цепочка не закрывается (%d рёбер)", len(foil_edges))
        else:
            foil_edges = []

        # ─ Домен: одна или несколько связных компонент ─
        if self._domain_coords is not None and len(self._domain_coords) >= 2:
            dom_verts_ref = [nd.index for nd in self._nodes if nd.node_type == "boundary"]
            dom_chain = self._order_boundary_chain(dom_verts_ref, self._domain_coords)
            _, dom_is_closed = self._clean_reference_chain(self._domain_coords)

            if dom_is_closed and len(dom_chain) >= 3:
                dom_edges = self._chain_to_edges(dom_chain)
            else:
                dom_edges = [
                    (dom_chain[i], dom_chain[i + 1])
                    for i in range(len(dom_chain) - 1)
                    if dom_chain[i] != dom_chain[i + 1]
                ]

            logger.info(
                "CDT: domain reference chain -> %d vertices, %d edges",
                len(dom_chain), len(dom_edges),
            )
            return dom_edges, foil_edges

        dom_verts = [nd.index for nd in self._nodes
                     if nd.node_type == "boundary" and not nd.on_airfoil]

        # Исключить узлы домена, геометрически дублирующие границу профиля.
        # Это происходит, когда domain_coords включает сегменты профиля как часть
        # внешнего контура (например, пластина-вырез): внешний контур порождает
        # ~N точек с bt="interior" поверх N точек профиля → пересекающиеся CDT-рёбра.
        # Критерий: bt=="interior" И расстояние до ближайшего foil-узла < 1.5·h_foil.
        if foil_verts and dom_verts:
            _fp = self._coords[foil_verts]
            _ft = cKDTree(_fp)
            _fd2, _ = _ft.query(_fp, k=min(2, len(foil_verts)))
            _h_foil = (float(np.median(_fd2[:, -1])) if _fd2.ndim > 1 else float(np.median(_fd2)))
            _h_foil = max(_h_foil, 1e-9)
            _prox = _h_foil * 1.5
            _dp = self._coords[dom_verts]
            _dd, _ = _ft.query(_dp)
            _prev_n = len(dom_verts)
            dom_verts = [
                vi for vi, d in zip(dom_verts, _dd)
                if not (d < _prox and self._nodes[vi].boundary_type == "interior")
            ]
            n_excl = _prev_n - len(dom_verts)
            if n_excl:
                logger.info(
                    "CDT: %d вершин домена исключены (bt=interior, близко к профилю)", n_excl
                )

        dom_edges: List[Tuple[int, int]] = []

        if dom_verts:
            components = self._find_boundary_components(dom_verts)
            # Наибольшая компонента — внешний контур, использует domain_coords как референс.
            # Малые компоненты упорядочиваются по углу от своего центроида.
            for idx, comp in enumerate(components):
                ref = self._domain_coords if idx == 0 else None
                chain = self._order_boundary_chain(comp, ref)
                if len(chain) < 2:
                    continue
                # Определяем, замкнута ли компонента: расстояние между первой и
                # последней вершиной не больше типичного шага между соседями.
                c0, c1 = self._coords[chain[0]], self._coords[chain[-1]]
                comp_pts = self._coords[comp]
                tree_c = cKDTree(comp_pts)
                _d, _ = tree_c.query(comp_pts, k=min(2, len(comp)))
                typical_step = float(np.median(_d[:, -1])) if _d.ndim > 1 else float(np.median(_d))
                is_closed = float(np.linalg.norm(c1 - c0)) < typical_step * 2.5
                edges = self._chain_to_edges(chain) if is_closed and len(chain) >= 3 else [
                    (chain[i], chain[i + 1]) for i in range(len(chain) - 1)
                    if chain[i] != chain[i + 1]
                ]
                dom_edges.extend(edges)

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

        # ── Конвертация: List → Dict (O(T), один раз) ────────────────────────
        # Dict[tid → (a,b,c)] позволяет удалять/добавлять треугольники без
        # сдвига индексов — необходимо для инкрементального обновления edge_map.
        tris: Dict[int, Tuple[int, int, int]] = {
            i: (int(a), int(b), int(c))
            for i, (a, b, c) in enumerate(self._tris_list)
        }
        next_id: int = len(tris)

        # ── Построение edge_map и vertex_to_tris (O(T), один раз) ───────────
        # Эти структуры поддерживаются инкрементально через все вставки и флипы.
        # Set[int] вместо List[int]: discard() — O(1) вместо remove() — O(n).
        edge_map: Dict[Tuple[int, int], Set[int]] = {}
        vertex_to_tris: Dict[int, Set[int]] = {}
        for tid, (a, b, c) in tris.items():
            for u, v in ((a, b), (b, c), (c, a)):
                edge_map.setdefault((min(u, v), max(u, v)), set()).add(tid)
            for w in (a, b, c):
                vertex_to_tris.setdefault(w, set()).add(tid)

        # ── Проверка missing через edge_map (O(M), без O(T·3) прохода) ───────
        missing = [(s, t) for s, t in required
                   if (min(s, t), max(s, t)) not in edge_map]

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

        # ── Вставка рёбер через deque с поддержкой разбиения ─────────────────
        # Когда _insert_constrained_edge возвращает коллинеарные вершины,
        # исходное ребро (s,t) заменяется подрёбрами (s,v0),(v0,v1),...,(vn,t).
        # Подрёбра помещаются в начало очереди (они короче и вставляются первыми).
        foil_keys_orig: Set[Tuple[int, int]] = {(min(s, t), max(s, t)) for s, t in foil_edges}
        dom_keys_orig: Set[Tuple[int, int]] = {(min(s, t), max(s, t)) for s, t in dom_edges}
        all_constrained_keys: Set[Tuple[int, int]] = foil_keys_orig | dom_keys_orig
        foil_constrained_keys: Set[Tuple[int, int]] = set(foil_keys_orig)
        dom_constrained_keys: Set[Tuple[int, int]] = set(dom_keys_orig)

        queue: deque = deque(missing)
        n_total = len(missing)
        n_done = 0
        n_splits = 0

        _pbar = (
            _tqdm(total=n_total, desc="    вставка рёбер", unit="ребро", leave=False)
            if _verbose and _have_tqdm and missing else None
        )

        while queue:
            s, t = queue.popleft()
            next_id, collinear = _insert_constrained_edge(
                s, t, tris, self._coords, edge_map, vertex_to_tris, next_id
            )
            if collinear:
                # Разбиваем (s,t) → (s,v0,...,vn,t) по порядку вдоль ребра
                d_vec = self._coords[t] - self._coords[s]
                collinear.sort(
                    key=lambda v: float(np.dot(self._coords[v] - self._coords[s], d_vec))
                )
                orig_key = (min(s, t), max(s, t))
                is_foil = orig_key in foil_constrained_keys
                is_dom = orig_key in dom_constrained_keys
                all_constrained_keys.discard(orig_key)
                foil_constrained_keys.discard(orig_key)
                dom_constrained_keys.discard(orig_key)

                chain = [s] + collinear + [t]
                sub_edges: List[Tuple[int, int]] = []
                for i in range(len(chain) - 1):
                    a_v, b_v = chain[i], chain[i + 1]
                    sub_key = (min(a_v, b_v), max(a_v, b_v))
                    all_constrained_keys.add(sub_key)
                    if is_foil:
                        foil_constrained_keys.add(sub_key)
                    if is_dom:
                        dom_constrained_keys.add(sub_key)
                    if sub_key not in edge_map or not edge_map[sub_key]:
                        sub_edges.append((a_v, b_v))
                # Вставляем подрёбра в начало очереди (они короче)
                for se in reversed(sub_edges):
                    queue.appendleft(se)
                n_splits += 1
            else:
                n_done += 1
                if _pbar is not None:
                    _pbar.update(1)

        if _pbar is not None:
            _pbar.close()
        if n_splits:
            logger.info("CDT: %d рёбер разбито на подрёбра (коллинеарные вершины)", n_splits)

        self._constrained_edges = all_constrained_keys
        self._foil_constrained_edges = foil_constrained_keys
        self._dom_constrained_edges = dom_constrained_keys

        # ── Flip: передаём готовый edge_map (не перестраивается с нуля) ──────
        _flip_to_delaunay(tris, self._coords, self._constrained_edges,
                          edge_map, verbose=_verbose)

        # ── Конвертация: Dict → List для остального пайплайна ────────────────
        self._tris_list = list(tris.values())
        self._triangles = np.array(self._tris_list, dtype=np.int32)

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
        foil_is_closed = (
            self._airfoil_coords is not None
            and len(self._airfoil_coords) >= 3
            and self._clean_reference_chain(self._airfoil_coords)[1]
        )
        if foil_is_closed and foil_ce:
            try:
                from shapely.geometry import Polygon as _SPolyFF
                foil_seed = np.array(
                    _SPolyFF(self._airfoil_coords).representative_point().coords[0],
                    dtype=np.float64,
                )
            except Exception:
                foil_seed = np.mean(np.asarray(self._airfoil_coords, dtype=np.float64), axis=0)
            tris_ff = _flood_fill_remove(tris, self._coords, [foil_seed], foil_ce)
            if len(tris_ff) == 0 and len(tris) > 0:
                logger.warning(
                    "Hole removal (flood-fill, профиль): удалил все треугольники; "
                    "оставляю исходную триангуляцию для Shapely-фильтрации"
                )
            else:
                tris = tris_ff
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

            if foil_is_closed:
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
        foil_edges: Set[Tuple[int, int]] = getattr(self, "_foil_constrained_edges", set())

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
            edge_key = (min(vi, vj), max(vi, vj))
            touches_wall = (
                edge_key in foil_edges
                or self._nodes[vi].on_airfoil
                or self._nodes[vj].on_airfoil
                or str(node_bt[vi]).lower() == "wall"
                or str(node_bt[vj]).lower() == "wall"
            )
            if touches_wall:
                bt_i = "wall"
                bt_j = "wall"
            else:
                bt_i = node_bt[vi]
                bt_j = node_bt[vj]

            bfaces.append(BoundaryFace(
                node=vi, neighbor_node=vj,
                boundary_type=bt_i,
                area_vector=area_vec.copy(),
            ))
            bfaces.append(BoundaryFace(
                node=vj, neighbor_node=vi,
                boundary_type=bt_j,
                area_vector=area_vec.copy(),
            ))

        self._boundary_faces = bfaces
        logger.info("BoundaryFaces: %d (по %d граничным рёбрам)",
                    len(bfaces), len(bfaces) // 2)

    def _validate_topology(self) -> None:
        """Fail fast on mesh states that make FVM volumes/faces invalid."""
        coords = self._coords
        triangles = self._triangles
        cv = self._cv_volumes
        errors: List[str] = []

        if coords is None or triangles is None or cv is None:
            raise RuntimeError("Mesh topology validation called before mesh arrays are ready")

        if len(coords) == 0:
            errors.append("no nodes")
        if len(triangles) == 0:
            errors.append("no triangles")

        rounded = np.round(coords, decimals=12)
        _, counts = np.unique(rounded, axis=0, return_counts=True)
        duplicate_groups = int(np.sum(counts > 1))
        if duplicate_groups:
            errors.append(f"{duplicate_groups} duplicate coordinate groups")

        repeated = int(sum(len(set(map(int, tri))) < 3 for tri in triangles))
        if repeated:
            errors.append(f"{repeated} triangles with repeated vertices")

        if len(triangles):
            a, b, k = triangles[:, 0], triangles[:, 1], triangles[:, 2]
            areas = 0.5 * np.abs(
                (coords[b, 0] - coords[a, 0]) * (coords[k, 1] - coords[a, 1])
                - (coords[k, 0] - coords[a, 0]) * (coords[b, 1] - coords[a, 1])
            )
            bbox = coords.max(axis=0) - coords.min(axis=0)
            area_tol = max(1e-20, float(np.prod(bbox)) * 1e-18)
            degenerate = int(np.sum(areas <= area_tol))
            if degenerate:
                errors.append(f"{degenerate} degenerate triangles (area <= {area_tol:.3e})")

        edge_counts: Counter = Counter()
        used_nodes: Set[int] = set()
        for a_i, b_i, c_i in triangles:
            ia, ib, ic = int(a_i), int(b_i), int(c_i)
            used_nodes.update((ia, ib, ic))
            for u, v in ((ia, ib), (ib, ic), (ic, ia)):
                if u > v:
                    u, v = v, u
                edge_counts[(u, v)] += 1

        nonmanifold = sum(1 for n_adj in edge_counts.values() if n_adj > 2)
        if nonmanifold:
            errors.append(f"{nonmanifold} non-manifold edges with more than two adjacent triangles")

        unused = len(coords) - len(used_nodes)
        if unused:
            errors.append(f"{unused} nodes are not referenced by any triangle")

        zero_cv = int(np.sum(cv <= 0.0))
        if zero_cv:
            errors.append(f"{zero_cv} control volumes are zero or negative")

        interior_bfaces = sum(
            1 for face in self._boundary_faces
            if str(face.boundary_type).lower() == "interior"
        )
        if interior_bfaces:
            errors.append(f"{interior_bfaces} boundary faces have boundary_type='interior'")

        if errors:
            raise ValueError("Invalid Mesh2D topology: " + "; ".join(errors))

        logger.info("Mesh2D topology OK")

    def _verify_conservativity(self, tol_factor: float = 1e-8) -> bool:
        """
        Проверяет геометрическую консервативность для каждого узла:
            ||Σ S_ij (внутр.) + Σ S_bnd (граничн.)|| < tol

        Args:
            tol_factor: допуск задаётся как tol_factor × char_length.

        Returns:
            True если все узлы прошли проверку, False при нарушении.
        """
        self._validate_topology()

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
        ok = len(violations) == 0
        if not ok:
            worst = violations[int(np.argmax(norms[violations]))]
            logger.error(
                "Геометрическая консервативность НАРУШЕНА: %d/%d узлов, "
                "макс. остаток %.3e (узел %d, tol=%.3e). "
                "Вероятная причина: некорректная CDT-триангуляция (дыры, "
                "перекрывающиеся треугольники). FVM-невязка будет ~1e30.",
                len(violations), n, float(norms[worst]), int(worst), tol,
            )
        else:
            logger.info("Геометрическая консервативность OK (tol=%.2e, char_len=%.4f)",
                        tol, char_len)
        self._conservativity_ok = ok
        return ok

    def _assert_built(self) -> None:
        if not self._built:
            raise RuntimeError("Вызовите build() перед обращением к результатам.")
