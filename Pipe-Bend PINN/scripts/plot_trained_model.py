#!/usr/bin/env python3
"""
Загружает обученный чекпоинт, строит графики полей U, V, P, скорость
и графики обучения (Total Loss, PDE, BC).

Ищет чекпоинты в:
1. Pipe-Bend PINN/Checkpoints/
2. PINN_Flow2D/Checkpoints/
3. --checkpoint путь

Использование:
    python plot_trained_model.py
    python plot_trained_model.py --checkpoint "Checkpoints/.../pinn_final.pt"
"""

import os
import sys
import argparse
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # корень проекта
PARENT_DIR = os.path.dirname(PROJECT_DIR)  # PINN_Flow2D
sys.path.insert(0, PROJECT_DIR)


def find_latest_checkpoint() -> str:
    """Находит последнюю папку с чекпоинтом в нескольких местах."""
    possible_dirs = [
        os.path.join(PROJECT_DIR, "workspace", "experiments"),
        os.path.join(PROJECT_DIR, "Checkpoints"),
    ]
    for checkpoints_dir in possible_dirs:
        if not os.path.exists(checkpoints_dir):
            continue
        try:
            entries = os.listdir(checkpoints_dir)
        except PermissionError:
            continue
        if not entries:
            continue
        folders = [f for f in entries if os.path.isdir(os.path.join(checkpoints_dir, f))]
        if not folders:
            continue
        folders.sort(reverse=True)
        for folder in folders:
            chk_path = os.path.join(checkpoints_dir, folder, "checkpoints", "pinn_final.pt")
            if os.path.exists(chk_path):
                return chk_path
    return None


def plot_training_history(checkpoint_path: str, output_dir: str, checkpoint_data: dict = None) -> None:
    """
    Строит 3 графика обучения из чекпоинта (history) или logs/metrics.csv:
    - training_loss.png  (Total Loss)
    - training_pde.png   (PDE Residuals)
    - training_bc.png    (BC Losses)
    """
    history = None
    if checkpoint_data and 'history' in checkpoint_data:
        history = checkpoint_data['history']
    
    # Fallback: читаем из CSV
    if not history or not history.get('loss'):
        run_dir = os.path.dirname(os.path.dirname(checkpoint_path))
        csv_path = os.path.join(run_dir, "logs", "metrics.csv")
        if os.path.exists(csv_path):
            rows = []
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    rows.append(row)
            if rows:
                history = {}
                for r in rows:
                    for k, v in r.items():
                        history.setdefault(k, []).append(float(v))
    
    if not history or not history.get('loss') or len(history['loss']) < 2:
        print("   ⚠️ Недостаточно данных для графика обучения (нужно минимум 2 эпохи)")
        return
    
    print("   📈 Построение графиков обучения...")
    epochs = list(range(1, len(history['loss']) + 1))
    
    # 1. Total Loss
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(epochs, history['loss'], 'b-', linewidth=1.5)
    ax.set_xlabel('Эпоха'); ax.set_ylabel('Total Loss')
    ax.set_title('Total Loss'); ax.grid(True); ax.set_yscale('log')
    fig.savefig(os.path.join(output_dir, "training_loss.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"   ✅ training_loss.png")
    
    # 2. PDE Residuals
    fig, ax = plt.subplots(figsize=(12, 6))
    for key in ['pde_cont', 'pde_mom_x', 'pde_mom_y', 'cont', 'mom_x', 'mom_y']:
        if key in history and len(history[key]) > 0:
            ax.plot(epochs, history[key], label=key.replace('pde_', ''), linewidth=1.5)
    ax.set_xlabel('Эпоха'); ax.set_ylabel('PDE Residual')
    ax.legend(); ax.set_title('PDE Residuals'); ax.grid(True); ax.set_yscale('log')
    fig.savefig(os.path.join(output_dir, "training_pde.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"   ✅ training_pde.png")
    
    # 3. BC Losses
    fig, ax = plt.subplots(figsize=(12, 6))
    for key in ['bc_inlet', 'bc_outlet', 'bc_wall', 'inlet', 'outlet', 'wall']:
        if key in history and len(history[key]) > 0:
            label = key.replace('bc_', '')
            ax.plot(epochs, history[key], label=label, linewidth=1.5)
    ax.set_xlabel('Эпоха'); ax.set_ylabel('BC Loss')
    ax.legend(); ax.set_title('Boundary Conditions Loss'); ax.grid(True); ax.set_yscale('log')
    fig.savefig(os.path.join(output_dir, "training_bc.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"   ✅ training_bc.png")
    
    print(f"   📊 Графики обучения сохранены")


def plot_from_checkpoint(checkpoint_path: str, output_dir: str = None) -> None:
    """Загружает чекпоинт и строит графики полей."""
    print(f"📂 Загрузка чекпоинта: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    config = checkpoint.get('config', {})
    # Сохраняем checkpoint_data для plot_training_history
    chk_data = dict(checkpoint)
    
    points_file = config.get('data', {}).get('points_file')
    if not points_file or not os.path.exists(points_file):
        fallback = os.path.join(PROJECT_DIR, "workspace", "initial", "fields", "initial.npy")
        if os.path.exists(fallback):
            points_file = fallback
        else:
            raise FileNotFoundError(f"Файл точек не найден ни в конфиге, ни в workspace/initial/fields/")
    print(f"📊 Файл точек: {points_file}")
    
    from pipe_bend_pinn.data.handler import DataHandler
    data_handler = DataHandler(points_file, batch_size=1000, shuffle=False)
    from pipe_bend_pinn.training.models import PINNIncompressible
    from pipe_bend_pinn.training.components.loss_computer import LossComputer
    
    device = torch.device('cpu')
    dtype = torch.float32
    rho_norm = torch.tensor(1.0)
    mu_norm = torch.tensor(1.0 / data_handler.Re) if data_handler.Re > 0 else torch.tensor(1.0)
    loss_computer = LossComputer(config=config, data_handler=data_handler, rho_norm=rho_norm, mu_norm=mu_norm,
                                  x_min=data_handler.x_min, x_max=data_handler.x_max,
                                  y_min=data_handler.y_min, y_max=data_handler.y_max, device=device, dtype=dtype)
    
    sample_batch = next(iter(data_handler.get_dataloader()))
    _, x_s, y_s, sdf_s, bt_s, _, _, _, _, _, _ = sample_batch
    input_tensor = loss_computer.build_input_tensor(x_s.to(device), y_s.to(device), sdf_s.to(device),
                                                     bt_s.to(device) if bt_s.shape[1] > 0 else None)
    actual_input_dim = input_tensor.shape[1]
    arch = config.get('architecture', {})
    model = PINNIncompressible(input_dim=actual_input_dim, layers=arch.get('hidden_layers', [50,50,50,50]),
                                activation=arch.get('activation', 'tanh'), output_dim=arch.get('output_dim', 3),
                                output_bias=arch.get('output_bias', [0.0,0.0,0.0]))
    model.eval()
    
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        if all(k.startswith('module.') for k in state_dict):
            state_dict = {k[len('module.'):]: v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        print("✅ Веса модели загружены")
    
    dataset = data_handler.dataset
    print(f"🔮 Выполнение предсказания на {len(dataset.x)} точках...")
    with torch.no_grad():
        x = dataset.x.to(device); y = dataset.y.to(device); sdf = dataset.sdf.to(device)
        bt = dataset.boundary_types.to(device) if dataset.boundary_types is not None else None
        bs = 2000; all_u, all_v, all_p = [], [], []
        for i in range(0, len(x), bs):
            be = min(i+bs, len(x))
            inp = loss_computer.build_input_tensor(x[i:be], y[i:be], sdf[i:be], bt[i:be] if bt is not None else None)
            out = model(inp)
            all_u.append(out[:,0].cpu().numpy().flatten())
            all_v.append(out[:,1].cpu().numpy().flatten())
            all_p.append(out[:,2].cpu().numpy().flatten())
    u_pred = np.concatenate(all_u); v_pred = np.concatenate(all_v); p_pred = np.concatenate(all_p)
    
    cv = float(data_handler.chord or 1.0); uv = float(data_handler.U_inf or 1.0)
    rv = float(data_handler.rho or 1.0); pv = float(data_handler.p_ref or 0.0)
    xp = dataset.x.cpu().numpy().flatten() * cv; yp = dataset.y.cpu().numpy().flatten() * cv
    up = u_pred * uv; vp = v_pred * uv
    pp = p_pred * (rv * uv**2) + pv; sp = np.sqrt(up**2 + vp**2)
    
    if output_dir is None:
        output_dir = os.path.join(PROJECT_DIR, "workspace", "experiments", "checkpoint_plots")
    os.makedirs(output_dir, exist_ok=True)
    
    domain = None
    dp = os.path.join(PROJECT_DIR, "workspace", "geometry", "domains", "pipe_bend_domain.npy")
    if os.path.exists(dp):
        domain = np.load(dp)
    
    print(f"💾 Сохранение графиков в {output_dir}")
    for fname, vals, label in [('U', up, 'U, м/с'), ('V', vp, 'V, м/с'), ('P', pp, 'P, Па'), ('Speed', sp, 'Скорость, м/с')]:
        fig, ax = plt.subplots(figsize=(12,9))
        sc = ax.scatter(xp, yp, c=vals, s=10, cmap='jet', alpha=0.8, edgecolors='none')
        plt.colorbar(sc, ax=ax, label=label)
        ax.set_xlabel('x, м'); ax.set_ylabel('y, м'); ax.set_title(label)
        ax.axis('equal'); ax.grid(alpha=0.3)
        if domain is not None:
            ax.plot(domain[:,0], domain[:,1], 'k-', linewidth=1.5, alpha=0.5)
        fig.savefig(os.path.join(output_dir, f"field_{fname.lower()}.png"), dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"   ✅ field_{fname.lower()}.png")
    
    print("   ⏳ Построение линий тока...")
    try:
        from scipy.interpolate import griddata
        x0, x1 = (domain[:,0].min(), domain[:,0].max()) if domain is not None else (xp.min(), xp.max())
        y0, y1 = (domain[:,1].min(), domain[:,1].max()) if domain is not None else (yp.min(), yp.max())
        mg = 0.05*max(x1-x0, y1-y0)
        nx, ny = 100, 100
        xi = np.linspace(x0-mg, x1+mg, nx); yi = np.linspace(y0-mg, y1+mg, ny)
        Xg, Yg = np.meshgrid(xi, yi)
        Ug = griddata((xp,yp), up, (Xg,Yg), method='linear', fill_value=0)
        Vg = griddata((xp,yp), vp, (Xg,Yg), method='linear', fill_value=0)
        sg = np.sqrt(Ug**2+Vg**2)
        if domain is not None:
            import matplotlib.path as mplP
            m = mplP.Path(domain[:,:2])
            inside = m.contains_points(np.column_stack((Xg.ravel(), Yg.ravel()))).reshape(Xg.shape)
            Ug[~inside]=0; Vg[~inside]=0; sg[~inside]=0
        fig, ax = plt.subplots(figsize=(12,9))
        ax.pcolormesh(Xg, Yg, sg, cmap='jet', shading='auto', alpha=0.6)
        ax.quiver(Xg[::5,::5], Yg[::5,::5], Ug[::5,::5], Vg[::5,::5],
                   color='white', scale=30, width=0.003, alpha=0.9)
        if domain is not None:
            ax.plot(domain[:,0], domain[:,1], 'k-', linewidth=2, alpha=0.8)
        ax.set_xlabel('x, м'); ax.set_ylabel('y, м'); ax.set_title('Поле скорости (векторы)')
        ax.axis('equal'); ax.grid(alpha=0.3)
        from matplotlib.colors import Normalize
        sm = plt.cm.ScalarMappable(cmap='jet', norm=Normalize(vmin=sg.min(), vmax=sg.max()))
        sm.set_array([]); plt.colorbar(sm, ax=ax, label='Скорость, м/с')
        fig.savefig(os.path.join(output_dir, "field_streamlines.png"), dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"   ✅ field_streamlines.png")
    except ImportError:
        print("   ⚠️ scipy не установлен")
    except Exception as e:
        print(f"   ⚠️ Ошибка линий тока: {e}")
    
    print(f"\n📂 Все графики полей сохранены в: {output_dir}")
    # Графики обучения
    plot_training_history(checkpoint_path, output_dir, checkpoint_data=chk_data)


def main():
    parser = argparse.ArgumentParser(description="Построение графиков по чекпоинту")
    parser.add_argument('--checkpoint', type=str, default=None, help='Путь к чекпоинту (pinn_final.pt)')
    parser.add_argument('--output', type=str, default=None, help='Директория для сохранения PNG')
    args = parser.parse_args()
    
    if args.checkpoint:
        chk_path = args.checkpoint
        if not os.path.exists(chk_path):
            print(f"❌ Указанный чекпоинт не найден: {chk_path}")
            return 1
    else:
        chk_path = find_latest_checkpoint()
        if chk_path is None:
            print("\n⚠️ Нет сохранённых чекпоинтов.")
            print(f"\nИскать в:\n  • {os.path.join(PROJECT_DIR, 'workspace', 'experiments')}")
            print("\nЗапустите: python pipe_bend_main.py --config workspace/configs/config.yaml")
            return 0
    
    print(f"🔍 Чекпоинт: {chk_path}")
    out = args.output or os.path.join(PROJECT_DIR, "workspace", "experiments", "checkpoint_plots")
    plot_from_checkpoint(chk_path, out)
    return 0


if __name__ == '__main__':
    sys.exit(main())