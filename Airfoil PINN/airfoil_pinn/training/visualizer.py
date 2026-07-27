# Исправленный файл: visualizer.py

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Визуализация для проекта PINN Learning.
Содержит класс Visualizer для построения графиков в реальном времени и сохранения полей.
Поддерживает два отдельных окна: для PDE и для граничных условий.
Добавлены методы для фиксации количественных и качественных метрик,
а также для записи метрик в CSV.
Расширен для поддержки диагностических графиков:
- профили скорости в нескольких сечениях
- двумерный спектр (2D FFT)
- гистограмма RAR
- раздельные нормы градиентов
- карта невязки
- и др.

Исправленная версия:
- динамические цветовые шкалы для скорости (не фиксированные 0..1)
- улучшенное выделение граничных точек
- добавлено логирование диапазонов полей
"""

import os
import csv
import logging
from typing import Dict, Optional, List, Union, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch
import torch.nn as nn
import numpy as np

try:
    from scipy.interpolate import griddata
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logging.warning("scipy не установлен. 2D спектр будет недоступен.")

logger = logging.getLogger(__name__)

__all__ = ["Visualizer"]


class Visualizer:
    """
    Построение графиков в реальном времени и финальных полей.
    Поддерживает два отдельных окна: для PDE и для граничных условий.
    Также фиксирует количественные и качественные метрики.
    """

    def __init__(self, plot_frequency: Optional[int] = None, config_name: str = "unnamed", config: Optional[Dict] = None):
        """
        Args:
            plot_frequency: частота обновления графиков (эпох). Если None, графики отключены.
            config_name: название конфигурации для заголовков.
            config: полная конфигурация (для coordinate_scaling)
        """
        self.plot_frequency = plot_frequency
        self.config_name = config_name
        self.config = config  # Сохраняем конфигурацию для обратного масштабирования
        self.fig_pde = None
        self.fig_bc = None
        self.fig_diag = None
        self.axes_pde = None
        self.axes_bc = None
        self.axes_diag = None
        self.lines_pde = {}
        self.lines_bc = {}
        self.lines_diag = {}
        self.enabled = plot_frequency is not None

    def create_live_plot(self) -> None:
        """Создаёт три окна: для PDE, граничных условий и диагностики."""
        if not self.enabled:
            return
        try:
            # Окно для PDE (4 графика: cont, mom_x/r, mom_y/z, total_loss)
            self.fig_pde, self.axes_pde = plt.subplots(1, 4, figsize=(20, 4), constrained_layout=True)
            flow_type = (self.config or {}).get('pde', {}).get('flow_type', '2d')
            if flow_type == 'axisymmetric':
                pde_items = [
                    ('cont', 'Continuity'),
                    ('mom_r', 'Momentum R'),
                    ('mom_z', 'Momentum Z'),
                    ('total', 'Total Loss')
                ]
            else:
                pde_items = [
                    ('cont', 'Continuity'),
                    ('mom_x', 'Momentum X'),
                    ('mom_y', 'Momentum Y'),
                    ('total', 'Total Loss')
                ]
            self.lines_pde = {}
            for ax, (key, title) in zip(self.axes_pde, pde_items):
                line, = ax.plot([], [])
                self.lines_pde[key] = line
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Value')
                ax.set_title(title)
                ax.set_yscale('log', nonpositive='clip')
                ax.grid(True)
            self.fig_pde.suptitle(f'PDE Residuals & Total Loss - {self.config_name}')

            # Окно для граничных условий (6 графиков: Inlet, Outlet, Wall, Symmetry, Axis Symmetry, Wall Speed)
            self.fig_bc, self.axes_bc = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
            self.axes_bc = self.axes_bc.flatten()
            bc_titles = ['Inlet', 'Outlet', 'Wall', 'Symmetry', 'Axis Symmetry', 'Wall Speed']
            self.lines_bc = {}
            for ax, title in zip(self.axes_bc, bc_titles):
                line, = ax.plot([], [])
                self.lines_bc[title.lower().replace(' ', '_')] = (ax, line)
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Value')
                ax.set_title(title)
                ax.set_yscale('log', nonpositive='clip')
                ax.grid(True)
            self.fig_bc.suptitle(f'Boundary Conditions - {self.config_name}')

            # Окно для диагностики (3 графика: градиенты, средние скорости, ошибка масс)
            self.fig_diag, self.axes_diag = plt.subplots(1, 3, figsize=(18, 4), constrained_layout=True)
            diag_titles = ['Gradient Norms', 'Avg Inlet/Outlet Velocity', 'Mass Balance Error']
            diag_keys = ['grad_norms', 'avg_vel', 'mass_balance']
            for ax, title, key in zip(self.axes_diag, diag_titles, diag_keys):
                if key == 'grad_norms':
                    line1, = ax.plot([], [], label='PDE grad')
                    line2, = ax.plot([], [], label='BC grad')
                    line3, = ax.plot([], [], label='cosine')
                    self.lines_diag[key] = (line1, line2, line3)
                elif key == 'avg_vel':
                    line1, = ax.plot([], [], label='avg u_in')
                    line2, = ax.plot([], [], label='avg u_out')
                    self.lines_diag[key] = (line1, line2)
                else:
                    line, = ax.plot([], [])
                    self.lines_diag[key] = line
                ax.set_xlabel('Epoch')
                ax.set_title(title)
                ax.grid(True)
                if key in ['grad_norms', 'avg_vel']:
                    ax.legend()
                if key == 'mass_balance':
                    ax.set_yscale('log', nonpositive='clip')
            self.fig_diag.suptitle(f'Diagnostics - {self.config_name}')

            plt.show(block=False)
        except Exception as e:
            logger.warning(f"Не удалось создать мультиплот: {e}")
            self.enabled = False

    def update_plot(self, epoch: int, history: Dict) -> None:
        """Обновляет данные на графиках. При ошибке отключает live plot."""
        if not self.enabled or self.fig_pde is None or self.fig_bc is None or self.fig_diag is None:
            return
        try:
            epochs_range = list(range(len(history['loss'])))

            # Обновление PDE графиков
            if plt.fignum_exists(self.fig_pde.number):
                for key, line in self.lines_pde.items():
                    if key == 'total':
                        line.set_data(epochs_range, history['loss'])
                    else:
                        if key in history.get('pde', {}):
                            line.set_data(epochs_range, history['pde'][key])
                for ax in self.axes_pde:
                    ax.relim()
                    ax.autoscale_view()
                self.fig_pde.canvas.draw()
                self.fig_pde.canvas.flush_events()
            else:
                logger.info("Окно PDE графика закрыто, отключаю live plot")
                self.enabled = False
                return

            # Обновление BC графиков - с проверкой наличия данных
            if plt.fignum_exists(self.fig_bc.number):
                bc_data = history.get('bc', {})
                for name, (ax, line) in self.lines_bc.items():
                    if name in bc_data and len(bc_data[name]) > 0:
                        line.set_data(epochs_range, bc_data[name])
                    elif name == 'wall_speed' and 'wall_speed' in history and len(history['wall_speed']) > 0:
                        line.set_data(epochs_range, history['wall_speed'])
                    else:
                        line.set_data([], [])
                    ax.relim()
                    ax.autoscale_view()
                self.fig_bc.canvas.draw()
                self.fig_bc.canvas.flush_events()
            else:
                logger.info("Окно BC графика закрыто, отключаю live plot")
                self.enabled = False
                return

            # Обновление диагностических графиков
            if plt.fignum_exists(self.fig_diag.number):
                # Градиенты (заполняются реже, чем history['loss'] — используем длину серии)
                if 'grad_norms' in self.lines_diag:
                    line1, line2, line3 = self.lines_diag['grad_norms']
                    if 'grad_norm_pde' in history and len(history['grad_norm_pde']) > 0:
                        s = history['grad_norm_pde']
                        line1.set_data(list(range(len(s))), s)
                    if 'grad_norm_bc' in history and len(history['grad_norm_bc']) > 0:
                        s = history['grad_norm_bc']
                        line2.set_data(list(range(len(s))), s)
                    if 'grad_cosine' in history and len(history['grad_cosine']) > 0:
                        s = history['grad_cosine']
                        line3.set_data(list(range(len(s))), s)
                # Средние скорости (заполняются реже — используем длину серии)
                if 'avg_vel' in self.lines_diag:
                    line1, line2 = self.lines_diag['avg_vel']
                    if 'avg_u_in' in history and len(history['avg_u_in']) > 0:
                        s = history['avg_u_in']
                        line1.set_data(list(range(len(s))), s)
                    if 'avg_u_out' in history and len(history['avg_u_out']) > 0:
                        s = history['avg_u_out']
                        line2.set_data(list(range(len(s))), s)
                # Ошибка баланса масс (заполняется реже — используем длину серии)
                if 'mass_balance' in self.lines_diag and 'mass_balance_error' in history and len(history['mass_balance_error']) > 0:
                    line = self.lines_diag['mass_balance']
                    s = history['mass_balance_error']
                    line.set_data(list(range(len(s))), s)

                for ax in self.axes_diag:
                    ax.relim()
                    ax.autoscale_view()
                self.fig_diag.canvas.draw()
                self.fig_diag.canvas.flush_events()
            else:
                logger.info("Окно диагностики закрыто, отключаю live plot")
                self.enabled = False
                return

            plt.pause(0.1)
        except Exception as e:
            logger.warning(f"Ошибка обновления графика: {e}. Отключаю live plot.")
            self.enabled = False
            self.close()

    def save_plots(self, savedir: str, epoch: Optional[Union[int, str]] = None) -> None:
        """
        Сохраняет текущие графики в файлы.

        Args:
            savedir: директория для сохранения
            epoch: номер эпохи или строка-идентификатор (например, 'final')
        """
        if not self.enabled:
            return
        os.makedirs(savedir, exist_ok=True)
        suffix = f"_epoch_{epoch}" if epoch is not None else "_final"
        try:
            if self.fig_pde is not None and plt.fignum_exists(self.fig_pde.number):
                self.fig_pde.savefig(os.path.join(savedir, f"pde_loss{suffix}.png"), dpi=150)
                logger.info(f"График PDE сохранён: pde_loss{suffix}.png")
            if self.fig_bc is not None and plt.fignum_exists(self.fig_bc.number):
                self.fig_bc.savefig(os.path.join(savedir, f"bc_loss{suffix}.png"), dpi=150)
                logger.info(f"График BC сохранён: bc_loss{suffix}.png")
            if self.fig_diag is not None and plt.fignum_exists(self.fig_diag.number):
                self.fig_diag.savefig(os.path.join(savedir, f"diagnostics{suffix}.png"), dpi=150)
                logger.info(f"График диагностики сохранён: diagnostics{suffix}.png")
        except Exception as e:
            logger.warning(f"Не удалось сохранить графики: {e}")

    def save_final_report(self, savedir: str, metrics: Dict) -> None:
        """
        Сохраняет финальный отчёт с количественными метриками в текстовый файл.

        Args:
            savedir: директория для сохранения
            metrics: словарь с метриками
        """
        os.makedirs(savedir, exist_ok=True)
        report_path = os.path.join(savedir, "final_report.txt")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(f"Конфигурация: {metrics.get('config_name', self.config_name)}\n")
            f.write("=" * 50 + "\n")
            f.write("Количественные метрики:\n")
            for key, value in metrics.items():
                if key != 'config_name':
                    f.write(f"  {key}: {value}\n")
            f.write("=" * 50 + "\n")
        logger.info(f"Финальный отчёт сохранён: {report_path}")

    def append_metrics_to_csv(self, metrics: Dict, csv_path: str) -> None:
        """
        Добавляет строку метрик в CSV-файл. Если файл не существует, создаёт его с заголовками.

        Args:
            metrics: словарь с метриками (ключи становятся названиями колонок)
            csv_path: путь к CSV-файлу
        """
        if 'experiment' not in metrics:
            metrics['experiment'] = self.config_name

        file_exists = os.path.exists(csv_path)
        with open(csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=metrics.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(metrics)
        logger.debug(f"Метрики добавлены в {csv_path}")

    @staticmethod
    def _deduplicate_field_points(
        x_np: np.ndarray,
        y_np: np.ndarray,
        values: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Убирает совпадающие координаты, усредняя значения поля."""
        xy = np.column_stack([x_np, y_np])
        rounded = np.round(xy, decimals=12)
        uniq, inverse = np.unique(rounded, axis=0, return_inverse=True)
        counts = np.bincount(inverse).astype(np.float64)
        val_sum = np.bincount(inverse, weights=values.astype(np.float64))
        return uniq[:, 0], uniq[:, 1], val_sum / np.maximum(counts, 1.0)

    def _plot_smooth_field_map(
        self,
        x_np: np.ndarray,
        y_np: np.ndarray,
        values: np.ndarray,
        savedir: str,
        filename: str,
        title: str,
        colorbar_label: str,
        cmap: str = 'jet',
        x_label: str = 'x, м',
        y_label: str = 'y, м',
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        boundary_masks: Optional[Dict[str, np.ndarray]] = None,
        clip_polygon: Optional[np.ndarray] = None,
        mask_polygon: Optional[np.ndarray] = None,
        outline_polylines: Optional[List[np.ndarray]] = None,
        grid_size: int = 420,
        levels: int = 96,
    ) -> None:
        """Рисует гладкое поле вместо точечной карты.

        boundary_masks оставлен для обратной совместимости, но сами точки
        границ здесь не рисуются: они закрывают поле при плотных сетках.
        """
        os.makedirs(savedir, exist_ok=True)

        x0 = np.asarray(x_np, dtype=np.float64).flatten()
        y0 = np.asarray(y_np, dtype=np.float64).flatten()
        z0 = np.asarray(values, dtype=np.float64).flatten()
        finite = np.isfinite(x0) & np.isfinite(y0) & np.isfinite(z0)
        x0, y0, z0 = x0[finite], y0[finite], z0[finite]
        if len(z0) == 0:
            logger.warning("Нет конечных значений для построения поля %s", filename)
            return

        x, y, z = self._deduplicate_field_points(x0, y0, z0)
        if len(z) < 3:
            logger.warning("Недостаточно точек для гладкой карты %s", filename)
            return

        data_min = float(np.nanmin(z))
        data_max = float(np.nanmax(z))
        if vmin is None:
            vmin = data_min
        if vmax is None:
            vmax = data_max
        if not np.isfinite(vmin) or not np.isfinite(vmax) or abs(vmax - vmin) < 1e-30:
            pad = max(abs(data_max), 1.0) * 1e-6
            vmin = data_min - pad
            vmax = data_max + pad
        level_values = np.linspace(float(vmin), float(vmax), levels)

        fig, ax = plt.subplots(figsize=(10, 8))
        artist = None
        clip_poly = self._clip_polygon(clip_polygon)
        mask_poly = self._closed_mask_polygon(mask_polygon)

        if SCIPY_AVAILABLE and len(z) >= 4:
            xi = np.linspace(float(x.min()), float(x.max()), grid_size)
            yi = np.linspace(float(y.min()), float(y.max()), grid_size)
            Xi, Yi = np.meshgrid(xi, yi)
            method = 'cubic' if len(z) >= 16 else 'linear'
            try:
                Zi = griddata((x, y), z, (Xi, Yi), method=method)
                if np.all(~np.isfinite(Zi)):
                    Zi = griddata((x, y), z, (Xi, Yi), method='linear')
                if np.any(np.isfinite(Zi)):
                    Zi = self._apply_grid_masks(Xi, Yi, Zi, clip_poly, mask_poly)
                    artist = ax.contourf(Xi, Yi, Zi, levels=level_values,
                                         cmap=cmap, extend='both')
            except Exception as exc:
                logger.debug("griddata для %s не сработал: %s", filename, exc)

        if artist is None:
            try:
                import matplotlib.tri as mtri
                triang = mtri.Triangulation(x, y)
                tri_mask = np.zeros(len(triang.triangles), dtype=bool)
                if clip_poly is not None or mask_poly is not None:
                    centroids_x = x[triang.triangles].mean(axis=1)
                    centroids_y = y[triang.triangles].mean(axis=1)
                    if clip_poly is not None:
                        tri_mask |= ~self._points_inside_polygon(centroids_x, centroids_y, clip_poly)
                    if mask_poly is not None:
                        tri_mask |= self._points_inside_polygon(centroids_x, centroids_y, mask_poly)
                    triang.set_mask(tri_mask)
                artist = ax.tricontourf(triang, z, levels=level_values,
                                        cmap=cmap, extend='both')
            except Exception as exc:
                logger.warning("Не удалось построить гладкое поле %s: %s", filename, exc)
                plt.close(fig)
                return

        if clip_poly is not None:
            ax.plot(clip_poly[:, 0], clip_poly[:, 1], color='black', linewidth=0.9)
        if outline_polylines:
            for line in outline_polylines:
                line_arr = self._polyline(line)
                if line_arr is not None:
                    ax.plot(line_arr[:, 0], line_arr[:, 1], color='black', linewidth=1.2)
        elif mask_polygon is not None:
            line_arr = self._polyline(mask_polygon)
            if line_arr is not None:
                ax.plot(line_arr[:, 0], line_arr[:, 1], color='black', linewidth=1.2)

        fig.colorbar(artist, ax=ax, label=colorbar_label)
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_aspect('equal', adjustable='box')
        ax.grid(alpha=0.15)
        fig.tight_layout()
        fig.savefig(os.path.join(savedir, filename), dpi=150, bbox_inches='tight')
        plt.close(fig)

    @staticmethod
    def _clean_polyline(polyline: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if polyline is None:
            return None
        line = np.asarray(polyline, dtype=np.float64).reshape(-1, 2)
        finite = np.isfinite(line).all(axis=1)
        line = line[finite]
        if len(line) < 2:
            return None
        keep = np.ones(len(line), dtype=bool)
        keep[1:] = np.linalg.norm(np.diff(line, axis=0), axis=1) > 1e-12
        line = line[keep]
        return line if len(line) >= 2 else None

    @classmethod
    def _clip_polygon(cls, clip_polygon: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """Возвращает замкнутый полигон расчётного домена или None."""
        poly = cls._clean_polyline(clip_polygon)
        if poly is None or len(poly) < 3:
            return None
        if not np.allclose(poly[0], poly[-1], atol=1e-12):
            poly = np.vstack([poly, poly[0]])
        return poly

    @classmethod
    def _closed_mask_polygon(cls, mask_polygon: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """Возвращает замкнутый полигон тела или None."""
        poly = cls._clean_polyline(mask_polygon)
        if poly is None or len(poly) < 3:
            return None
        if not np.allclose(poly[0], poly[-1], atol=1e-12):
            return None
        return poly

    @classmethod
    def _polyline(cls, polyline: Optional[np.ndarray]) -> Optional[np.ndarray]:
        return cls._clean_polyline(polyline)

    @staticmethod
    def _scale_geometry_for_plot(polyline: Optional[np.ndarray], scale: float) -> Optional[np.ndarray]:
        if polyline is None:
            return None
        arr = np.asarray(polyline, dtype=np.float64).reshape(-1, 2)
        return arr * float(scale)

    @staticmethod
    def _points_inside_polygon(x: np.ndarray, y: np.ndarray, poly: np.ndarray) -> np.ndarray:
        pts = np.column_stack([np.asarray(x).ravel(), np.asarray(y).ravel()])
        try:
            from matplotlib.path import Path
            return Path(poly).contains_points(pts).reshape(np.shape(x))
        except Exception:
            return np.zeros(np.shape(x), dtype=bool)

    @classmethod
    def _apply_grid_masks(
        cls,
        Xi: np.ndarray,
        Yi: np.ndarray,
        Zi: np.ndarray,
        clip_poly: Optional[np.ndarray],
        mask_poly: Optional[np.ndarray],
    ) -> np.ndarray:
        masked = np.array(Zi, copy=True)
        if clip_poly is not None:
            outside_domain = ~cls._points_inside_polygon(Xi, Yi, clip_poly)
            masked[outside_domain] = np.nan
        if mask_poly is not None:
            inside_body = cls._points_inside_polygon(Xi, Yi, mask_poly)
            masked[inside_body] = np.nan
        return masked

    def plot_residual_map(self, x_np: np.ndarray, y_np: np.ndarray,
                          residual: np.ndarray, savedir: str,
                          epoch: Optional[Union[int, str]] = None,
                          clip_polygon: Optional[np.ndarray] = None,
                          mask_polygon: Optional[np.ndarray] = None,
                          outline_polylines: Optional[List[np.ndarray]] = None) -> None:
        """
        Строит карту невязки PDE по области (координаты в физических единицах).

        Args:
            x_np, y_np: координаты точек (физические)
            residual: значение невязки (суммарной или по компоненте)
            savedir: директория для сохранения
            epoch: идентификатор эпохи
        """
        os.makedirs(savedir, exist_ok=True)
        suffix = f"_epoch_{epoch}" if epoch is not None else "_final"

        self._plot_smooth_field_map(
            x_np, y_np, residual, savedir,
            filename=f"residual_map{suffix}.png",
            title=f'Карта невязки PDE - {self.config_name}',
            colorbar_label='Невязка',
            cmap='hot',
            clip_polygon=clip_polygon,
            mask_polygon=mask_polygon,
            outline_polylines=outline_polylines,
        )
        logger.info(f"Карта невязки сохранена: residual_map{suffix}.png")

    def plot_rar_error_map(self, x_np: np.ndarray, y_np: np.ndarray,
                           error: np.ndarray, savedir: str,
                           epoch: Optional[Union[int, str]] = None,
                           clip_polygon: Optional[np.ndarray] = None,
                           mask_polygon: Optional[np.ndarray] = None,
                           outline_polylines: Optional[List[np.ndarray]] = None) -> None:
        """
        Строит карту полной локальной ошибки, которую использует adaptive RAR.

        Обычно это local total-loss score: взвешенная PDE-часть + точечные BC.
        """
        os.makedirs(savedir, exist_ok=True)
        suffix = f"_epoch_{epoch}" if epoch is not None else "_final"

        self._plot_smooth_field_map(
            x_np, y_np, error, savedir,
            filename=f"rar_error_map{suffix}.png",
            title=f'Поле полной ошибки RAR - {self.config_name}',
            colorbar_label='RAR score / local total loss',
            cmap='magma',
            clip_polygon=clip_polygon,
            mask_polygon=mask_polygon,
            outline_polylines=outline_polylines,
        )
        logger.info(f"Карта полной ошибки RAR сохранена: rar_error_map{suffix}.png")

    def plot_gradient_norm_history(self, grad_norm_history: List[float],
                                   savedir: str, epoch: Optional[Union[int, str]] = None) -> None:
        """
        Строит историю нормы градиентов.

        Args:
            grad_norm_history: список значений нормы градиентов по эпохам
            savedir: директория для сохранения
            epoch: идентификатор эпохи
        """
        os.makedirs(savedir, exist_ok=True)
        suffix = f"_epoch_{epoch}" if epoch is not None else "_final"

        plt.figure(figsize=(10, 6))
        plt.plot(grad_norm_history, 'g-')
        plt.xlabel('Эпоха')
        plt.ylabel('Норма градиента')
        plt.title(f'История нормы градиентов - {self.config_name}')
        plt.yscale('log', nonpositive='clip')
        plt.grid(True)
        plt.savefig(os.path.join(savedir, f"grad_norm{suffix}.png"), dpi=150)
        plt.close()
        logger.info(f"История нормы градиентов сохранена: grad_norm{suffix}.png")

    def save_fields_as_npy(self,
                           u_norm: torch.Tensor,
                           v_norm: torch.Tensor,
                           p_norm: torch.Tensor,
                           x_norm: torch.Tensor,
                           y_norm: torch.Tensor,
                           data_handler,
                           savedir: str,
                           epoch: Optional[Union[int, str]] = None,
                           boundary_types: Optional[torch.Tensor] = None,
                           boundary_type_names: Optional[List[str]] = None) -> None:
        """
        Сохраняет поля в формате .npy (структурированный массив) вместе с координатами и типами границ.
        Сохраняются нормализованные поля (u_norm, v_norm, p_norm) и физические (если нужно).

        Args:
            u_norm, v_norm, p_norm: нормализованные поля
            x_norm, y_norm: нормализованные координаты
            data_handler: объект DataHandler с метаданными
            savedir: директория для сохранения
            epoch: идентификатор эпохи
            boundary_types: one-hot тензор типов границ
            boundary_type_names: список имён типов границ
        """
        os.makedirs(savedir, exist_ok=True)
        suffix = f"_epoch_{epoch}" if epoch is not None else "_final"

        # Преобразуем в numpy
        x_np = x_norm.cpu().numpy().flatten()
        y_np = y_norm.cpu().numpy().flatten()
        u_np = u_norm.cpu().numpy().flatten()
        v_np = v_norm.cpu().numpy().flatten()
        p_np = p_norm.cpu().numpy().flatten()

        # Определяем dtype для структурированного массива (нормализованные)
        dtype = [('x_norm', 'f8'), ('y_norm', 'f8'), ('u_norm', 'f8'), ('v_norm', 'f8'), ('p_norm', 'f8')]

        # Добавляем информацию о типах границ, если есть
        if boundary_types is not None and boundary_type_names is not None:
            bt_np = boundary_types.cpu().numpy()
            for i, name in enumerate(boundary_type_names):
                if bt_np.shape[1] > i:
                    dtype.append((name, 'bool'))

        # Создаём структурированный массив (нормализованные)
        arr_norm = np.zeros(len(x_np), dtype=dtype)
        arr_norm['x_norm'] = x_np
        arr_norm['y_norm'] = y_np
        arr_norm['u_norm'] = u_np
        arr_norm['v_norm'] = v_np
        arr_norm['p_norm'] = p_np

        if boundary_types is not None and boundary_type_names is not None:
            for i, name in enumerate(boundary_type_names):
                if bt_np.shape[1] > i:
                    arr_norm[name] = bt_np[:, i].astype(bool)

        # Сохраняем нормализованные поля
        filename_norm = os.path.join(savedir, f"fields_norm{suffix}.npy")
        np.save(filename_norm, arr_norm)
        logger.info(f"Нормализованные поля сохранены в {filename_norm}")

        # Сохраняем также отдельно физические величины (для удобства)
        u_phys = u_np * data_handler.U_inf
        v_phys = v_np * data_handler.U_inf
        p_dyn = 0.5 * data_handler.rho * data_handler.U_inf ** 2
        p_phys = p_np * p_dyn + data_handler.p_ref
        x_phys = x_np * data_handler.chord
        y_phys = y_np * data_handler.chord

        phys_arr = np.zeros(len(x_np), dtype=[('x', 'f8'), ('y', 'f8'),
                                               ('u', 'f8'), ('v', 'f8'), ('p', 'f8')])
        phys_arr['x'] = x_phys
        phys_arr['y'] = y_phys
        phys_arr['u'] = u_phys
        phys_arr['v'] = v_phys
        phys_arr['p'] = p_phys

        phys_filename = os.path.join(savedir, f"fields_phys{suffix}.npy")
        np.save(phys_filename, phys_arr)
        logger.info(f"Физические поля сохранены в {phys_filename}")

    def plot_final_fields_from_tensors(self,
                                       u_norm: torch.Tensor,
                                       v_norm: torch.Tensor,
                                       p_norm: torch.Tensor,
                                       x_norm: torch.Tensor,
                                       y_norm: torch.Tensor,
                                       data_handler,
                                       savedir: str,
                                       epoch: Optional[Union[int, str]] = None,
                                       boundary_types: Optional[torch.Tensor] = None,
                                       boundary_type_names: Optional[List[str]] = None,
                                       save_npy: bool = True,
                                       plot_physical: bool = False,
                                       config: Optional[Dict] = None) -> None:
        """
        Строит и сохраняет поля скорости и давления из готовых тензоров.
        Если plot_physical=True, отображаются физические величины (с размерностью),
        иначе – нормализованные (безразмерные). По умолчанию нормализованные.

        Для нормализованной скорости цветовая шкала подстраивается динамически:
        vmin=0, vmax = max(1.0, u_norm.max()) – чтобы учесть возможные значения >1.
        Для физической скорости vmax = max(U_inf, u_phys.max()).

        Args:
            u_norm, v_norm, p_norm: нормализованные поля
            x_norm, y_norm: нормализованные координаты
            data_handler: объект DataHandler с метаданными
            savedir: директория для сохранения
            epoch: номер эпохи (для имени файла)
            boundary_types: one-hot тензор типов границ
            boundary_type_names: список имён типов границ
            save_npy: сохранить ли как .npy файлы
            plot_physical: отображать физические или нормализованные величины
            config: конфигурация (для coordinate_scaling)
        """
        # Сохраняем в .npy, если требуется
        if save_npy:
            self.save_fields_as_npy(u_norm, v_norm, p_norm, x_norm, y_norm,
                                    data_handler, savedir, epoch,
                                    boundary_types, boundary_type_names)

        # Определяем, какие величины отображать
        if plot_physical:
            u_plot = u_norm * data_handler.U_inf
            v_plot = v_norm * data_handler.U_inf
            p_dyn = 0.5 * data_handler.rho * data_handler.U_inf ** 2
            p_plot = p_norm * p_dyn + data_handler.p_ref
            x_plot = x_norm * data_handler.chord
            y_plot = y_norm * data_handler.chord
            u_label = 'Скорость U, м/с'
            v_label = 'Скорость V, м/с'
            p_label = 'Давление, Па'
            x_label = 'x, м'
            y_label = 'y, м'
            title_suffix = ' (physical)'
            # u: динамически от реального минимума до максимума (с учётом U_inf как нижней планки)
            u_vmin = min(0.0, u_plot.min().item())
            u_vmax = max(data_handler.U_inf, u_plot.max().item())
            # v: симметричная шкала вокруг нуля (v может быть отрицательной)
            v_abs = max(abs(v_plot.min().item()), abs(v_plot.max().item()), data_handler.U_inf * 0.01)
            v_vmin, v_vmax = -v_abs, v_abs
            # p: динамически
            p_vmin, p_vmax = p_plot.min().item(), p_plot.max().item()
        else:
            u_plot = u_norm
            v_plot = v_norm
            p_plot = p_norm
            x_plot = x_norm
            y_plot = y_norm
            u_label = 'U (норм.)'
            v_label = 'V (норм.)'
            p_label = 'P (норм.)'
            x_label = 'x (норм.)'
            y_label = 'y (норм.)'
            title_suffix = ' (normalized)'
            # u: динамически от реального минимума (с учётом нижней планки 0)
            u_vmin = min(0.0, u_plot.min().item())
            u_vmax = max(1.0, u_plot.max().item())
            # v: симметричная шкала вокруг нуля
            v_abs = max(abs(v_plot.min().item()), abs(v_plot.max().item()), 0.01)
            v_vmin, v_vmax = -v_abs, v_abs
            # p: динамически
            p_vmin, p_vmax = p_plot.min().item(), p_plot.max().item()

        x_np = x_plot.cpu().numpy().flatten()
        y_np = y_plot.cpu().numpy().flatten()
        u_np = u_plot.cpu().numpy().flatten()
        v_np = v_plot.cpu().numpy().flatten()
        p_np = p_plot.cpu().numpy().flatten()

        if (
            getattr(data_handler, 'mesh_domain_coords', None) is None
            and hasattr(data_handler, 'load_mesh_geometry_from_geom_config')
        ):
            data_handler.load_mesh_geometry_from_geom_config()
        geom_scale = 1.0 if plot_physical else 1.0 / float(getattr(data_handler, 'chord', 1.0) or 1.0)
        clip_polygon = self._scale_geometry_for_plot(
            getattr(data_handler, 'mesh_domain_coords', None),
            geom_scale,
        )
        airfoil_polyline = self._scale_geometry_for_plot(
            getattr(data_handler, 'mesh_airfoil_coords', None),
            geom_scale,
        )
        outline_polylines = [airfoil_polyline] if airfoil_polyline is not None else None

        # Создаём маски для граничных точек
        boundary_masks = {}
        if boundary_types is not None and boundary_type_names is not None:
            bt_np = boundary_types.cpu().numpy()
            for i, name in enumerate(boundary_type_names):
                if bt_np.shape[1] > i:
                    mask = bt_np[:, i] > 0.5
                    boundary_masks[name] = mask

        fields = [('u', u_np, u_label, u_vmin, u_vmax),
                  ('v', v_np, v_label, v_vmin, v_vmax),
                  ('p', p_np, p_label, p_vmin, p_vmax)]

        os.makedirs(savedir, exist_ok=True)
        suffix = f"_epoch_{epoch}" if epoch is not None else "_final"

        logger.info(f"Поле u: min={u_np.min():.4f}, max={u_np.max():.4f} (vmax={u_vmax:.4f})")
        if boundary_masks and 'inlet' in boundary_masks:
            inlet_vals = u_np[boundary_masks['inlet']]
            logger.info(f"  inlet: count={len(inlet_vals)}, mean={inlet_vals.mean():.4f}, min={inlet_vals.min():.4f}, max={inlet_vals.max():.4f}")

        for name, val, label, vmin, vmax in fields:
            try:
                title = (
                    f'Predicted {name} (epoch {epoch}){title_suffix} - {self.config_name}'
                    if epoch else
                    f'Predicted {name}{title_suffix} - {self.config_name}'
                )
                self._plot_smooth_field_map(
                    x_np, y_np, val, savedir,
                    filename=f'field_{name}{suffix}.png',
                    title=title,
                    colorbar_label=label,
                    cmap='jet',
                    x_label=x_label,
                    y_label=y_label,
                    vmin=vmin,
                    vmax=vmax,
                    boundary_masks=boundary_masks,
                    clip_polygon=clip_polygon,
                    mask_polygon=airfoil_polyline,
                    outline_polylines=outline_polylines,
                )
                logger.info(f"Поле сохранено: field_{name}{suffix}.png")
            except Exception as e:
                logger.warning(f"Не удалось построить график для {name}: {e}")

    # ------------------------------------------------------------------
    # Диагностические методы
    # ------------------------------------------------------------------

    def plot_velocity_profiles_multiple(self,
                                        profiles: Dict[float, Tuple[np.ndarray, np.ndarray]],
                                        savedir: str,
                                        epoch: Optional[Union[int, str]] = None) -> None:
        """
        Строит профили скорости u(y) для нескольких сечений x.

        Args:
            profiles: словарь {x: (y_vals, u_vals)} – значения в физических единицах
            savedir: директория для сохранения
            epoch: идентификатор эпохи
        """
        os.makedirs(savedir, exist_ok=True)
        suffix = f"_epoch_{epoch}" if epoch is not None else "_final"

        plt.figure(figsize=(10, 6))
        for x, (y, u) in profiles.items():
            plt.plot(u, y, label=f'x = {x:.3f} м')
        plt.xlabel('u, м/с')
        plt.ylabel('y, м')
        plt.title(f'Профили скорости в сечениях - {self.config_name}')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(savedir, f"profiles_multiple{suffix}.png"), dpi=150)
        plt.close()
        logger.info(f"Множественные профили сохранены: profiles_multiple{suffix}.png")

    def plot_1d_spectrum(self,
                         y_np: np.ndarray,
                         u_np: np.ndarray,
                         savedir: str,
                         epoch: Optional[Union[int, str]] = None,
                         x_section: Optional[float] = None) -> None:
        """
        Строит одномерный спектр профиля u(y) в заданном сечении.
        """
        if len(y_np) < 2 or len(u_np) < 2:
            logger.warning("1D спектр пропущен: недостаточно точек.")
            return

        os.makedirs(savedir, exist_ok=True)
        suffix = f"_epoch_{epoch}" if epoch is not None else "_final"

        sort_idx = np.argsort(y_np)
        y_sorted = y_np[sort_idx]
        u_sorted = u_np[sort_idx]
        dy = float(np.mean(np.diff(y_sorted)))
        if dy <= 0:
            logger.warning("1D спектр пропущен: некорректная сетка по y.")
            return

        signal = u_sorted - np.mean(u_sorted)
        spectrum = np.abs(np.fft.rfft(signal))
        k = np.fft.rfftfreq(len(signal), d=dy) * 2 * np.pi

        plt.figure(figsize=(10, 6))
        plt.loglog(k[1:], spectrum[1:] + 1e-30, 'b-')
        plt.xlabel(r'$k_y$, рад/м')
        plt.ylabel('|FFT(u)|')
        title = '1D spectrum of u'
        if x_section is not None:
            title += f' at x={x_section:.3f} м'
        if epoch is not None:
            title += f' (epoch {epoch})'
        plt.title(title)
        plt.grid(True)
        plt.savefig(os.path.join(savedir, f"spectrum_1d{suffix}.png"), dpi=150)
        plt.close()
        logger.info(f"1D спектр сохранён: spectrum_1d{suffix}.png")

    def plot_2d_spectrum(self, x_np: np.ndarray, y_np: np.ndarray, u_np: np.ndarray,
                         savedir: str, epoch: Optional[Union[int, str]] = None,
                         nx: int = 256, ny: int = 256,
                         vmin: Optional[float] = None, vmax: Optional[float] = None) -> None:
        """
        Строит двумерный спектр поля u (или другого) по всей области.

        Args:
            x_np, y_np: координаты точек (физические, одномерные массивы)
            u_np: значения поля (одномерный массив)
            savedir: директория для сохранения
            epoch: номер эпохи
            nx, ny: размеры равномерной сетки для интерполяции
            vmin, vmax: пределы цветовой шкалы для спектра (логарифмической)
        """
        if not SCIPY_AVAILABLE:
            logger.warning("scipy не установлен. 2D спектр недоступен.")
            return

        os.makedirs(savedir, exist_ok=True)
        suffix = f"_epoch_{epoch}" if epoch is not None else "_final"

        # Определяем границы области
        x_min, x_max = x_np.min(), x_np.max()
        y_min, y_max = y_np.min(), y_np.max()

        # Создаём равномерную сетку
        xi = np.linspace(x_min, x_max, nx)
        yi = np.linspace(y_min, y_max, ny)
        Xi, Yi = np.meshgrid(xi, yi)

        # Интерполяция
        from scipy.interpolate import griddata
        Ui = griddata((x_np, y_np), u_np, (Xi, Yi), method='linear', fill_value=0.0)

        # Вычисление двумерного спектра
        F = np.fft.fft2(Ui)
        F_shifted = np.fft.fftshift(F)
        amplitude = np.abs(F_shifted)

        # Соответствующие волновые числа (циклические)
        kx = np.fft.fftshift(np.fft.fftfreq(nx, d=(x_max-x_min)/nx)) * 2*np.pi
        ky = np.fft.fftshift(np.fft.fftfreq(ny, d=(y_max-y_min)/ny)) * 2*np.pi

        # Визуализация спектра (логарифмическая шкала)
        plt.figure(figsize=(10, 8))
        plt.imshow(np.log10(amplitude + 1e-12), extent=[kx[0], kx[-1], ky[0], ky[-1]],
                   origin='lower', cmap='hot', aspect='auto', vmin=vmin, vmax=vmax)
        plt.colorbar(label='log10(|FFT|)')
        plt.xlabel(r'$k_x$, рад/м')
        plt.ylabel(r'$k_y$, рад/м')
        plt.title(f'2D spectrum of u (epoch {epoch})' if epoch else '2D spectrum of u')
        plt.savefig(os.path.join(savedir, f'spectrum_2d{suffix}.png'), dpi=150)
        plt.close()

        # Радиальное усреднение
        # amplitude имеет форму (ny, nx) — meshgrid(xi, yi) строки по y, столбцы по x
        KX, KY = np.meshgrid(kx, ky)          # оба (ny, nx), совпадают с amplitude
        k_radial = np.sqrt(KX**2 + KY**2).flatten()
        amp_flat = amplitude.flatten()

        # Сортируем по радиальному волновому числу и усредняем по бинам
        sort_idx = np.argsort(k_radial)
        k_sorted = k_radial[sort_idx]
        amp_sorted = amp_flat[sort_idx]

        n_bins = 100
        k_bins = np.linspace(0, k_sorted[-1], n_bins+1)
        k_center = []
        amp_binned = []
        for i in range(n_bins):
            mask = (k_sorted >= k_bins[i]) & (k_sorted < k_bins[i+1])
            if np.any(mask):
                amp_binned.append(amp_sorted[mask].mean())
                k_center.append((k_bins[i] + k_bins[i+1]) / 2)
            else:
                amp_binned.append(0)
                k_center.append((k_bins[i] + k_bins[i+1]) / 2)

        plt.figure(figsize=(10, 6))
        plt.loglog(k_center, amp_binned, 'b-')
        plt.xlabel(r'$k$, рад/м')
        plt.ylabel('Средняя амплитуда')
        plt.title(f'Radially averaged spectrum of u (epoch {epoch})' if epoch else 'Radially averaged spectrum of u')
        plt.grid(True)
        plt.savefig(os.path.join(savedir, f'spectrum_radial{suffix}.png'), dpi=150)
        plt.close()
        logger.info(f"2D спектр сохранён: spectrum_2d{suffix}.png, спектр с радиальным усреднением: spectrum_radial{suffix}.png")

    def plot_rar_histogram(self,
                           rar_history: List[Dict[str, int]],
                           boundary_type_names: List[str],
                           savedir: str,
                           epoch: Optional[Union[int, str]] = None) -> None:
        """
        Строит столбчатую диаграмму, показывающую, сколько точек какого типа
        было добавлено на каждом шаге RAR.

        Args:
            rar_history: список словарей {тип: количество} для каждого применения RAR
            boundary_type_names: список имён типов границ
            savedir: директория для сохранения
            epoch: идентификатор эпохи (для имени файла)
        """
        os.makedirs(savedir, exist_ok=True)
        suffix = f"_epoch_{epoch}" if epoch is not None else "_final"

        n_steps = len(rar_history)
        if n_steps == 0:
            return

        data = np.zeros((n_steps, len(boundary_type_names)))
        for i, step in enumerate(rar_history):
            for j, name in enumerate(boundary_type_names):
                data[i, j] = step.get(name, 0)

        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(n_steps)
        width = 0.8 / len(boundary_type_names)

        for j, name in enumerate(boundary_type_names):
            ax.bar(x + j*width, data[:, j], width, label=name)

        ax.set_xlabel('Шаг RAR')
        ax.set_ylabel('Количество добавленных точек')
        ax.set_title(f'Статистика добавления точек RAR - {self.config_name}')
        ax.set_xticks(x + width * (len(boundary_type_names)-1)/2)
        ax.set_xticklabels([f'{i+1}' for i in range(n_steps)])
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(savedir, f"rar_histogram{suffix}.png"), dpi=150)
        plt.close()
        logger.info(f"Гистограмма RAR сохранена: rar_histogram{suffix}.png")

    def plot_gradient_norms_separate(self,
                                     grad_norm_pde: List[float],
                                     grad_norm_bc: List[float],
                                     grad_cosine: List[float],
                                     savedir: str,
                                     epoch: Optional[Union[int, str]] = None) -> None:
        """
        Строит графики раздельных норм градиентов и косинуса угла между ними.

        Args:
            grad_norm_pde: история нормы градиентов PDE
            grad_norm_bc: история нормы градиентов BC
            grad_cosine: история косинуса угла
            savedir: директория для сохранения
            epoch: идентификатор эпохи
        """
        os.makedirs(savedir, exist_ok=True)
        suffix = f"_epoch_{epoch}" if epoch is not None else "_final"

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        epochs = list(range(len(grad_norm_pde)))
        ax1.semilogy(epochs, grad_norm_pde, 'r-', label='PDE grad')
        ax1.semilogy(epochs, grad_norm_bc, 'b-', label='BC grad')
        ax1.set_ylabel('Норма градиента')
        ax1.legend()
        ax1.grid(True)

        ax2.plot(epochs, grad_cosine, 'g-')
        ax2.set_xlabel('Эпоха')
        ax2.set_ylabel('Косинус угла')
        ax2.grid(True)

        plt.suptitle(f'Раздельные нормы градиентов - {self.config_name}')
        plt.tight_layout()
        plt.savefig(os.path.join(savedir, f"grad_norms_separate{suffix}.png"), dpi=150)
        plt.close()
        logger.info(f"График раздельных норм градиентов сохранён: grad_norms_separate{suffix}.png")

    def plot_final_fields(self, model: nn.Module, data_handler,
                          device: torch.device, savedir: str,
                          epoch: Optional[Union[int, str]] = None,
                          plot_physical: bool = False,
                          loss_computer=None) -> None:
        """
        Устаревший метод – сохранён для совместимости.
        Рекомендуется использовать plot_final_fields_from_tensors.
        """
        logger.warning("Метод plot_final_fields устарел. Используйте plot_final_fields_from_tensors.")
        if loss_computer is None:
            raise RuntimeError(
                "plot_final_fields requires LossComputer so evaluation SDF "
                "is recomputed by the canonical geometry engine"
            )
        model.eval()
        dataset = data_handler.dataset
        x_norm = dataset.x.to(device)
        y_norm = dataset.y.to(device)
        sdf_norm = loss_computer.evaluate_sdf_tensor(x_norm, y_norm)
        bt = dataset.boundary_types.to(device) if dataset.boundary_types is not None else None

        with torch.no_grad():
            input_tensor = loss_computer.build_input_tensor(
                x_norm,
                y_norm,
                sdf_norm,
            )
            output = model(input_tensor)
            u_norm, v_norm, p_norm = output[:, 0:1], output[:, 1:2], output[:, 2:3]
            u_norm, v_norm, p_norm = loss_computer.transform_outputs(
                u_norm,
                v_norm,
                p_norm,
                x_norm,
                y_norm,
                sdf=sdf_norm,
                case_ids=getattr(dataset, "case_id", None),
            )

        self.plot_final_fields_from_tensors(u_norm, v_norm, p_norm, x_norm, y_norm,
                                            data_handler, savedir, epoch,
                                            boundary_types=bt,
                                            boundary_type_names=data_handler.boundary_type_names,
                                            plot_physical=plot_physical,
                                            config=self.config if hasattr(self, 'config') else None)

    def close(self) -> None:
        """Закрывает все окна matplotlib."""
        try:
            if self.fig_pde is not None:
                plt.close(self.fig_pde)
                self.fig_pde = None
            if self.fig_bc is not None:
                plt.close(self.fig_bc)
                self.fig_bc = None
            if self.fig_diag is not None:
                plt.close(self.fig_diag)
                self.fig_diag = None
        except Exception:
            pass
