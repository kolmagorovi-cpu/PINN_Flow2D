"""
Визуализация для Pipe-Bend PINN.

Автор: Адаптировано из Airfoil PINN
Версия: 1.0
"""

import os
import logging
from typing import Dict, Optional, List, Union, Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np

logger = logging.getLogger(__name__)


class Visualizer:
    """Построение графиков в реальном времени и финальных полей."""

    def __init__(self, plot_frequency: Optional[int] = None, config_name: str = "unnamed",
                 config: Optional[Dict] = None):
        self.plot_frequency = plot_frequency
        self.config_name = config_name
        self.config = config
        self.fig_pde = None
        self.fig_bc = None
        self.axes_pde = None
        self.axes_bc = None
        self.lines_pde = {}
        self.lines_bc = {}
        self.enabled = plot_frequency is not None

    def create_live_plot(self) -> None:
        if not self.enabled:
            return
        try:
            self.fig_pde, self.axes_pde = plt.subplots(1, 4, figsize=(20, 4), constrained_layout=True)
            pde_items = [('cont', 'Continuity'), ('mom_x', 'Momentum X'),
                         ('mom_y', 'Momentum Y'), ('total', 'Total Loss')]
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

            self.fig_bc, self.axes_bc = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
            bc_titles = ['Inlet', 'Outlet', 'Wall']
            self.lines_bc = {}
            for ax, title in zip(self.axes_bc, bc_titles):
                line, = ax.plot([], [])
                self.lines_bc[title.lower().replace(' ', '_')] = line
                ax.set_xlabel('Epoch')
                ax.set_ylabel('Value')
                ax.set_title(title)
                ax.set_yscale('log', nonpositive='clip')
                ax.grid(True)
            self.fig_bc.suptitle(f'Boundary Conditions - {self.config_name}')
            plt.show(block=False)
        except Exception as e:
            logger.warning(f"Не удалось создать мультиплот: {e}")
            self.enabled = False

    def update_plot(self, epoch: int, history: Dict) -> None:
        if not self.enabled or self.fig_pde is None or self.fig_bc is None:
            return
        try:
            epochs_range = list(range(len(history['loss'])))
            if plt.fignum_exists(self.fig_pde.number):
                for key, line in self.lines_pde.items():
                    if key == 'total':
                        line.set_data(epochs_range, history['loss'])
                    elif f'pde_{key}' in history:
                        line.set_data(epochs_range, history[f'pde_{key}'])
                for ax in self.axes_pde:
                    ax.relim()
                    ax.autoscale_view()
                self.fig_pde.canvas.draw()
                self.fig_pde.canvas.flush_events()
            if plt.fignum_exists(self.fig_bc.number):
                for name, line in self.lines_bc.items():
                    bc_key = f'bc_{name}'
                    if bc_key in history and len(history[bc_key]) > 0:
                        line.set_data(epochs_range, history[bc_key])
                    ax_bc = self.axes_bc[list(self.lines_bc.keys()).index(name)] if isinstance(self.axes_bc, np.ndarray) else self.axes_bc
                for ax in self.axes_bc:
                    ax.relim()
                    ax.autoscale_view()
                self.fig_bc.canvas.draw()
                self.fig_bc.canvas.flush_events()
            plt.pause(0.1)
        except Exception:
            self.enabled = False
            self.close()

    def save_plots(self, savedir: str, epoch: Optional[Union[int, str]] = None) -> None:
        if not self.enabled:
            return
        os.makedirs(savedir, exist_ok=True)
        suffix = f"_epoch_{epoch}" if epoch is not None else "_final"
        try:
            if self.fig_pde is not None and plt.fignum_exists(self.fig_pde.number):
                self.fig_pde.savefig(os.path.join(savedir, f"pde_loss{suffix}.png"), dpi=150)
            if self.fig_bc is not None and plt.fignum_exists(self.fig_bc.number):
                self.fig_bc.savefig(os.path.join(savedir, f"bc_loss{suffix}.png"), dpi=150)
        except Exception:
            pass

    def plot_residual_map(self, x_np, y_np, residual, savedir, epoch=None):
        os.makedirs(savedir, exist_ok=True)
        suffix = f"_epoch_{epoch}" if epoch is not None else "_final"
        plt.figure(figsize=(10, 8))
        plt.scatter(x_np, y_np, c=residual, s=1, cmap='hot', alpha=0.6)
        plt.colorbar(label='Невязка')
        plt.title(f'Карта невязки - {self.config_name}')
        plt.xlabel('x, м')
        plt.ylabel('y, м')
        plt.axis('equal')
        plt.savefig(os.path.join(savedir, f"residual_map{suffix}.png"), dpi=150)
        plt.close()

    def plot_velocity_profiles_multiple(self, profiles, savedir, epoch=None):
        os.makedirs(savedir, exist_ok=True)
        suffix = f"_epoch_{epoch}" if epoch is not None else "_final"
        plt.figure(figsize=(10, 6))
        for x, (y, u) in profiles.items():
            plt.plot(u, y, label=f'x = {x:.3f} м')
        plt.xlabel('u, м/с')
        plt.ylabel('y, м')
        plt.title(f'Профили скорости - {self.config_name}')
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(savedir, f"profiles{suffix}.png"), dpi=150)
        plt.close()

    def save_fields_as_npy(self, u_norm, v_norm, p_norm, x_norm, y_norm,
                           data_handler, savedir, epoch=None, **kwargs):
        os.makedirs(savedir, exist_ok=True)
        suffix = f"_epoch_{epoch}" if epoch is not None else "_final"
        x_np = x_norm.cpu().numpy().flatten()
        y_np = y_norm.cpu().numpy().flatten()
        u_np = u_norm.cpu().numpy().flatten()
        v_np = v_norm.cpu().numpy().flatten()
        p_np = p_norm.cpu().numpy().flatten()
        arr = np.zeros(len(x_np), dtype=[('x_norm', 'f8'), ('y_norm', 'f8'),
                                          ('u_norm', 'f8'), ('v_norm', 'f8'), ('p_norm', 'f8')])
        arr['x_norm'] = x_np
        arr['y_norm'] = y_np
        arr['u_norm'] = u_np
        arr['v_norm'] = v_np
        arr['p_norm'] = p_np
        np.save(os.path.join(savedir, f"fields_norm{suffix}.npy"), arr)
        logger.info(f"Поля сохранены: fields_norm{suffix}.npy")

    def plot_final_fields_from_tensors(self, u_norm, v_norm, p_norm, x_norm, y_norm,
                                        data_handler, savedir, epoch=None, **kwargs):
        self.save_fields_as_npy(u_norm, v_norm, p_norm, x_norm, y_norm,
                                data_handler, savedir, epoch=epoch)
        suffix = f"_epoch_{epoch}" if epoch is not None else "_final"
        x_np = x_norm.cpu().numpy().flatten()
        y_np = y_norm.cpu().numpy().flatten()
        for name, val, label in [('u', u_norm.cpu().numpy().flatten(), 'U (норм.)'),
                                  ('v', v_norm.cpu().numpy().flatten(), 'V (норм.)'),
                                  ('p', p_norm.cpu().numpy().flatten(), 'P (норм.)')]:
            plt.figure(figsize=(10, 8))
            plt.scatter(x_np, y_np, c=val, s=1, cmap='jet', alpha=0.6)
            plt.colorbar(label=label)
            plt.title(f'Predicted {name}{suffix} - {self.config_name}')
            plt.xlabel('x (норм.)')
            plt.ylabel('y (норм.)')
            plt.axis('equal')
            plt.savefig(os.path.join(savedir, f"field_{name}{suffix}.png"), dpi=150)
            plt.close()

    def close(self) -> None:
        try:
            if self.fig_pde is not None:
                plt.close(self.fig_pde)
                self.fig_pde = None
            if self.fig_bc is not None:
                plt.close(self.fig_bc)
                self.fig_bc = None
        except Exception:
            pass