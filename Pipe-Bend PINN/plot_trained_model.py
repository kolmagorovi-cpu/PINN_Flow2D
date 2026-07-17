#!/usr/bin/env python3
"""
Загружает обученный чекпоинт и строит графики полей U, V, P, скорость.

Находит последнюю папку в Checkpoints/, загружает pinn_final.pt,
выполняет предсказание на всех точках и сохраняет PNG в plots/checkpoint_plots/

Использование:
    python plot_trained_model.py
    python plot_trained_model.py --checkpoint "Checkpoints/.../pinn_final.pt"
"""

import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)


def find_latest_checkpoint() -> str:
    """Находит последнюю папку в Checkpoints и возвращает путь к pinn_final.pt."""
    checkpoints_dir = os.path.join(PROJECT_DIR, "Checkpoints")
    if not os.path.exists(checkpoints_dir):
        raise FileNotFoundError(f"Папка Checkpoints не найдена: {checkpoints_dir}")
    
    folders = [f for f in os.listdir(checkpoints_dir) 
               if os.path.isdir(os.path.join(checkpoints_dir, f))]
    if not folders:
        raise FileNotFoundError("В папке Checkpoints нет подпапок с обучением")
    
    # Сортируем по имени (дата в имени папки YYYYMMDD_HHMMSS)
    folders.sort(reverse=True)
    latest = folders[0]
    chk_path = os.path.join(checkpoints_dir, latest, "checkpoints", "pinn_final.pt")
    if not os.path.exists(chk_path):
        raise FileNotFoundError(f"Чекпоинт не найден: {chk_path}")
    
    return chk_path


def plot_from_checkpoint(checkpoint_path: str, output_dir: str = None) -> None:
    """Загружает чекпоинт и строит графики полей."""
    print(f"📂 Загрузка чекпоинта: {checkpoint_path}")
    
    # Загружаем чекпоинт
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    config = checkpoint.get('config', {})
    
    # Путь к файлу точек
    points_file = config.get('data', {}).get('points_file')
    if not points_file or not os.path.exists(points_file):
        # Ищем в Initial/Data/
        fallback = os.path.join(PROJECT_DIR, "Initial", "Data", "initial.npy")
        if os.path.exists(fallback):
            points_file = fallback
        else:
            raise FileNotFoundError(f"Файл точек не найден ни в конфиге, ни в Initial/Data/")
    
    print(f"📊 Файл точек: {points_file}")
    
    # Создаём DataHandler
    from data_handler import DataHandler
    data_handler = DataHandler(points_file, batch_size=1000, shuffle=False)
    
    # Строим модель
    from models import PINNIncompressible
    from solver_components.loss_computer import LossComputer
    
    device = torch.device('cpu')
    dtype = torch.float32
    rho_norm = torch.tensor(1.0)
    mu_norm = torch.tensor(1.0 / data_handler.Re) if data_handler.Re > 0 else torch.tensor(1.0)
    
    loss_computer = LossComputer(
        config=config, data_handler=data_handler,
        rho_norm=rho_norm, mu_norm=mu_norm,
        x_min=data_handler.x_min, x_max=data_handler.x_max,
        y_min=data_handler.y_min, y_max=data_handler.y_max,
        device=device, dtype=dtype
    )
    
    # Определяем входной размер
    sample_batch = next(iter(data_handler.get_dataloader()))
    _, x_s, y_s, sdf_s, bt_s, _, _, _, _, _, _ = sample_batch
    input_tensor = loss_computer.build_input_tensor(x_s.to(device), y_s.to(device), sdf_s.to(device), 
                                                     bt_s.to(device) if bt_s.shape[1] > 0 else None)
    actual_input_dim = input_tensor.shape[1]
    
    arch = config.get('architecture', {})
    model = PINNIncompressible(
        input_dim=actual_input_dim,
        layers=arch.get('hidden_layers', [50, 50, 50, 50]),
        activation=arch.get('activation', 'tanh'),
        output_dim=arch.get('output_dim', 3),
        output_bias=arch.get('output_bias', [0.0, 0.0, 0.0])
    )
    model.eval()
    
    # Загружаем веса
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        # Убираем префикс module. если он есть (DataParallel)
        if all(k.startswith('module.') for k in state_dict):
            state_dict = {k[len('module.'):]: v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
        print("✅ Веса модели загружены")
    
    # Выполняем предсказание на всех точках
    dataset = data_handler.dataset
    print(f"🔮 Выполнение предсказания на {len(dataset.x)} точках...")
    
    with torch.no_grad():
        x = dataset.x.to(device)
        y = dataset.y.to(device)
        sdf = dataset.sdf.to(device)
        bt = dataset.boundary_types.to(device) if dataset.boundary_types is not None else None
        
        # Предсказываем батчами
        batch_size = 2000
        all_u, all_v, all_p = [], [], []
        for i in range(0, len(x), batch_size):
            batch_end = min(i + batch_size, len(x))
            x_batch = x[i:batch_end]
            y_batch = y[i:batch_end]
            sdf_batch = sdf[i:batch_end]
            bt_batch = bt[i:batch_end] if bt is not None else None
            
            inp = loss_computer.build_input_tensor(x_batch, y_batch, sdf_batch, bt_batch)
            out = model(inp)
            all_u.append(out[:, 0].cpu().numpy().flatten())
            all_v.append(out[:, 1].cpu().numpy().flatten())
            all_p.append(out[:, 2].cpu().numpy().flatten())
    
    u_pred = np.concatenate(all_u)
    v_pred = np.concatenate(all_v)
    p_pred = np.concatenate(all_p)
    
    # Переводим в физические величины
    chord_val = float(data_handler.chord or 1.0)
    u_inf_val = float(data_handler.U_inf or 1.0)
    rho_val = float(data_handler.rho or 1.0)
    p_ref_val = float(data_handler.p_ref or 0.0)
    
    x_phys = dataset.x.cpu().numpy().flatten() * chord_val
    y_phys = dataset.y.cpu().numpy().flatten() * chord_val
    u_phys = u_pred * u_inf_val
    v_phys = v_pred * u_inf_val
    p_phys = p_pred * (rho_val * u_inf_val ** 2) + p_ref_val
    speed_phys = np.sqrt(u_phys**2 + v_phys**2)
    
    # Определяем выходную директорию
    if output_dir is None:
        output_dir = os.path.join(PROJECT_DIR, "Checkpoints", "checkpoint_plots")
    os.makedirs(output_dir, exist_ok=True)
    
    # Загружаем контур отвода (для красоты)
    domain = None
    domain_path = os.path.join(PROJECT_DIR, "Geometry", "Data", "pipe_bend_domain.npy")
    if os.path.exists(domain_path):
        domain = np.load(domain_path)
    
    # Строим 4 графика
    fields = [
        ('U', u_phys, 'U, м/с'),
        ('V', v_phys, 'V, м/с'),
        ('P', p_phys, 'P, Па'),
        ('Speed', speed_phys, 'Скорость, м/с'),
    ]
    
    print(f"💾 Сохранение графиков в {output_dir}")
    for fname, vals, label in fields:
        fig, ax = plt.subplots(figsize=(12, 9))
        sc = ax.scatter(x_phys, y_phys, c=vals, s=10, cmap='jet', alpha=0.8, edgecolors='none')
        cbar = plt.colorbar(sc, ax=ax, label=label)
        ax.set_xlabel('x, м')
        ax.set_ylabel('y, м')
        ax.set_title(f'{label}')
        ax.axis('equal')
        ax.grid(alpha=0.3)
        if domain is not None:
            ax.plot(domain[:, 0], domain[:, 1], 'k-', linewidth=1.5, alpha=0.5)
        save_path = os.path.join(output_dir, f"field_{fname.lower()}.png")
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"   ✅ {save_path}")
    
    # График 5: линии тока (streamlines + quiver для стрелок)
    print("   ⏳ Построение линий тока...")
    try:
        from scipy.interpolate import griddata
        
        # Определяем границы сетки по контуру отвода
        if domain is not None:
            x_min, x_max = domain[:, 0].min(), domain[:, 0].max()
            y_min, y_max = domain[:, 1].min(), domain[:, 1].max()
        else:
            x_min, x_max = x_phys.min(), x_phys.max()
            y_min, y_max = y_phys.min(), y_phys.max()
        margin = 0.05 * max(x_max - x_min, y_max - y_min)
        
        # Регулярная сетка
        nx, ny = 100, 100
        xi = np.linspace(x_min - margin, x_max + margin, nx)
        yi = np.linspace(y_min - margin, y_max + margin, ny)
        X_grid, Y_grid = np.meshgrid(xi, yi)
        
        # Интерполяция U,V на сетку
        U_grid = griddata((x_phys, y_phys), u_phys, (X_grid, Y_grid), method='linear', fill_value=0)
        V_grid = griddata((x_phys, y_phys), v_phys, (X_grid, Y_grid), method='linear', fill_value=0)
        speed_grid = np.sqrt(U_grid**2 + V_grid**2)
        
        # Маска: обнуляем снаружи контура
        if domain is not None:
            import matplotlib.path as mplPath
            mask = mplPath.Path(domain[:, :2])
            pts = np.column_stack((X_grid.ravel(), Y_grid.ravel()))
            inside = mask.contains_points(pts).reshape(X_grid.shape)
            U_grid[~inside] = 0
            V_grid[~inside] = 0
            speed_grid[~inside] = 0
        
        fig, ax = plt.subplots(figsize=(12, 9))
        # Фон — модуль скорости
        ax.pcolormesh(X_grid, Y_grid, speed_grid, cmap='jet', shading='auto', alpha=0.6)
        # Векторное поле (quiver) — прореженная сетка
        skip = 5  # каждую 5-ю точку
        ax.quiver(X_grid[::skip, ::skip], Y_grid[::skip, ::skip],
                   U_grid[::skip, ::skip], V_grid[::skip, ::skip],
                   color='white', scale=30, width=0.003, alpha=0.9)
        # Контур отвода
        if domain is not None:
            ax.plot(domain[:, 0], domain[:, 1], 'k-', linewidth=2, alpha=0.8)
        ax.set_xlabel('x, м')
        ax.set_ylabel('y, м')
        ax.set_title('Поле скорости (векторы)')
        ax.axis('equal')
        ax.grid(alpha=0.3)
        # Colorbar
        from matplotlib.colors import Normalize
        sm = plt.cm.ScalarMappable(cmap='jet', norm=Normalize(vmin=speed_grid.min(), vmax=speed_grid.max()))
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label='Скорость, м/с')
        save_path = os.path.join(output_dir, "field_streamlines.png")
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"   ✅ {save_path}")
    except ImportError:
        print("   ⚠️ scipy не установлен. Пропущено: pip install scipy")
    except Exception as e:
        print(f"   ⚠️ Ошибка линий тока: {e}")
    
    print(f"\n📂 Все графики сохранены в: {output_dir}")
    print("Откройте папку и дважды кликните по PNG для просмотра.")


def main():
    parser = argparse.ArgumentParser(description="Построение графиков по чекпоинту")
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Путь к чекпоинту (pinn_final.pt). Если не указан — ищет последний.')
    parser.add_argument('--output', type=str, default=None,
                        help='Директория для сохранения PNG. По умолчанию Checkpoints/checkpoint_plots/')
    args = parser.parse_args()
    
    try:
        if args.checkpoint:
            chk_path = args.checkpoint
        else:
            chk_path = find_latest_checkpoint()
            print(f"🔍 Найден последний чекпоинт: {chk_path}")
        
        plot_from_checkpoint(chk_path, args.output)
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())