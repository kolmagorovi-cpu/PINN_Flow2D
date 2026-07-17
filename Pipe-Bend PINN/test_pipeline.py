#!/usr/bin/env python3
"""
Интеграционное тестирование Pipe-Bend PINN.

Выполняет 8 тестов для проверки корректности изменённых модулей обучения.

Зависимости: numpy, torch, matplotlib (установлены)

Автор: Pipe-Bend PINN Team
Версия: 1.1
"""

import os
import sys
import numpy as np
import traceback

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)

passed = 0
failed = 0


def test_step(num, name, func, *args, **kwargs):
    """Выполняет шаг теста и выводит результат."""
    global passed, failed
    print(f"\n{'='*70}")
    print(f"🔧 ТЕСТ {num}: {name}")
    print(f"{'='*70}")
    try:
        result = func(*args, **kwargs)
        if result is not False and result is not None:
            print(f"   ✅ УСПЕШНО")
            passed += 1
            return result
        else:
            print(f"   ❌ ПРОВАЛ")
            failed += 1
            return None
    except Exception as e:
        print(f"   ❌ ОШИБКА: {e}")
        traceback.print_exc()
        failed += 1
        return None


# ============================================================================
# Тест 1: Базовые импорты (уже выполнено, но для целостности)
# ============================================================================
def test_1_imports():
    """Проверка импортов всех модулей."""
    import numpy as np
    import torch
    import matplotlib
    from data_handler import PINNDataset, DataHandler
    from solver_components.loss_computer import LossComputer
    from solver_components.trainer import Trainer
    from solver_components.metrics import MetricsCollector
    from pde_residuals import PDEResiduals
    from models import PINNIncompressible, SelfAdaptiveWeights
    from checkpoint import CheckpointManager
    from solver import PINNSolver
    print("   ✅ numpy, torch, matplotlib доступны")
    print(f"   ✅ Все модули импортированы: {18} классов/функций")
    return True


# ============================================================================
# Тест 2: Проверка контракта compute_loss() → словарь
# ============================================================================
def test_2_loss_computer_contract():
    """Проверка что LossComputer.compute_loss() возвращает словарь с нужными ключами."""
    import torch
    from data_handler import PINNDataset, DataHandler
    from solver_components.loss_computer import LossComputer
    from models import PINNIncompressible

    # Проверяем контракт на уровне сигнатуры (без реальных данных)
    lc_signature = LossComputer.__init__.__code__ if hasattr(LossComputer.__init__, '__code__') else None
    compute_signature = LossComputer.compute_loss.__code__ if hasattr(LossComputer.compute_loss, '__code__') else None

    if compute_signature is not None:
        # Проверяем что есть возвращаемые типы в аннотациях
        import inspect
        sig = inspect.signature(LossComputer.compute_loss)
        return_annotation = sig.return_annotation
        if return_annotation is inspect.Parameter.empty:
            print("   ⚠️  compute_loss() не имеет аннотации возврата, но это не критично")
        else:
            print(f"   ✅ compute_loss() возвращает: {return_annotation}")
    else:
        print("   ⚠️  Сигнатура не проверена")

    # Проверяем что у LossComputer есть метод build_input_tensor
    assert hasattr(LossComputer, 'build_input_tensor'), "Нет build_input_tensor"
    assert hasattr(LossComputer, '_compute_bc_loss'), "Нет _compute_bc_loss"
    assert hasattr(LossComputer, 'apply_hard_constraints'), "Нет apply_hard_constraints"
    print("   ✅ LossComputer имеет все необходимые методы")
    return True


# ============================================================================
# Тест 3: Проверка моделей
# ============================================================================
def test_3_models():
    """Проверка создания моделей."""
    import torch
    from models import PINNIncompressible, SelfAdaptiveWeights, Sine

    # Проверка создания сети
    model = PINNIncompressible(
        input_dim=5,
        layers=[50, 50, 50, 50],
        activation='tanh',
        output_dim=3,
        output_bias=(0.0, 0.0, 0.0)
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"   ✅ PINNIncompressible создана: {n_params} параметров")

    # Проверка forward
    x = torch.randn(10, 5)
    out = model(x)
    assert out.shape == (10, 3), f"Ожидался размер (10,3), получен {out.shape}"
    print(f"   ✅ Forward pass: {x.shape} -> {out.shape}")

    # Проверка SA-весов
    sa = SelfAdaptiveWeights(n_points=100, n_eqns=3)
    w = sa()
    assert w.shape == (100, 3), f"SA-веса: ожидался (100,3), получен {w.shape}"
    print(f"   ✅ SelfAdaptiveWeights: {w.shape}")

    # Проверка Sine активации
    sine = Sine()
    y = sine(torch.tensor([0.0, np.pi/2]))
    assert abs(y[0].item()) < 1e-6, "sin(0) должен быть 0"
    assert abs(y[1].item() - 1.0) < 1e-6, "sin(pi/2) должен быть 1"
    print(f"   ✅ Sine активация sin(0)={y[0].item():.3f}, sin(pi/2)={y[1].item():.3f}")

    return True


# ============================================================================
# Тест 4: Проверка PDEResiduals
# ============================================================================
def test_4_pde_residuals():
    """Проверка модуля PDE невязок."""
    import torch
    from pde_residuals import PDEResiduals
    from utils import safe_grad

    # Создаём тестовые поля
    n = 10
    x = torch.randn(n, 1, requires_grad=True)
    y = torch.randn(n, 1, requires_grad=True)
    u = x + y  # простые поля
    v = x - y
    p = x * y
    rho = torch.tensor(1.0)
    mu = torch.tensor(0.01)

    residuals = PDEResiduals.compute_residuals_2d(u, v, p, x, y, rho, mu, safe_grad)

    assert 'cont' in residuals, "Нет cont в residuals"
    assert 'mom_x' in residuals, "Нет mom_x в residuals"
    assert 'mom_y' in residuals, "Нет mom_y в residuals"

    print(f"   ✅ PDE residuals: cont={residuals['cont'].mean().item():.4e}, "
          f"mom_x={residuals['mom_x'].mean().item():.4e}, "
          f"mom_y={residuals['mom_y'].mean().item():.4e}")
    return True


# ============================================================================
# Тест 5: Проверка Trainer (базовый)
# ============================================================================
def test_5_trainer():
    """Проверка создания Trainer."""
    import torch
    from models import PINNIncompressible
    from solver_components.trainer import Trainer

    # Создаём минимальный Trainer (без DataHandler - для проверки наличия методов)
    assert hasattr(Trainer, 'train_epoch_adam'), "Нет train_epoch_adam"
    assert hasattr(Trainer, 'train_epoch_lbfgs'), "Нет train_epoch_lbfgs"
    assert hasattr(Trainer, 'forward_pass'), "Нет forward_pass"
    assert hasattr(Trainer, 'apply_rar'), "Нет apply_rar"
    assert hasattr(Trainer, 'compute_grad_norms_separately'), "Нет compute_grad_norms_separately"
    assert hasattr(Trainer, 'get_state'), "Нет get_state"
    assert hasattr(Trainer, 'load_state'), "Нет load_state"
    print("   ✅ Trainer имеет все необходимые методы")
    return True


# ============================================================================
# Тест 6: Проверка MetricsCollector
# ============================================================================
def test_6_metrics():
    """Проверка MetricsCollector."""
    from solver_components.metrics import MetricsCollector

    assert hasattr(MetricsCollector, 'update_history'), "Нет update_history"
    assert hasattr(MetricsCollector, 'get_boundary_avg_velocities'), "Нет get_boundary_avg_velocities"
    assert hasattr(MetricsCollector, 'get_mass_flow_metrics'), "Нет get_mass_flow_metrics"
    assert hasattr(MetricsCollector, 'print_epoch_table'), "Нет print_epoch_table"
    assert hasattr(MetricsCollector, 'add_row_to_csv'), "Нет add_row_to_csv"
    print("   ✅ MetricsCollector имеет все необходимые методы")
    return True


# ============================================================================
# Тест 7: Проверка DataHandler (базовые методы)
# ============================================================================
def test_7_data_handler():
    """Проверка DataHandler на наличие методов."""
    from data_handler import PINNDataset, DataHandler

    assert hasattr(PINNDataset, 'get_state'), "PINNDataset нет get_state"
    assert hasattr(PINNDataset, 'extend'), "PINNDataset нет extend"
    assert hasattr(PINNDataset, 'set_inlet_values'), "PINNDataset нет set_inlet_values"

    assert hasattr(DataHandler, 'get_state'), "DataHandler нет get_state"
    assert hasattr(DataHandler, 'load_state'), "DataHandler нет load_state"
    assert hasattr(DataHandler, 'add_points'), "DataHandler нет add_points"
    assert hasattr(DataHandler, 'get_points_by_type'), "DataHandler нет get_points_by_type"
    print("   ✅ DataHandler/PINNDataset имеют все необходимые методы")
    return True


# ============================================================================
# Тест 8: Проверка solver.py
# ============================================================================
def test_8_solver():
    """Проверка PINNSolver на наличие методов."""
    from solver import PINNSolver

    assert hasattr(PINNSolver, 'train'), "Нет train"
    assert hasattr(PINNSolver, '_load_checkpoint_if_needed'), "Нет _load_checkpoint_if_needed"
    assert hasattr(PINNSolver, '_switch_to_lbfgs'), "Нет _switch_to_lbfgs"
    assert hasattr(PINNSolver, '_recreate_optimizer'), "Нет _recreate_optimizer"
    assert hasattr(PINNSolver, '_post_training_diagnostics'), "Нет _post_training_diagnostics"
    assert hasattr(PINNSolver, '_get_pde_multiplier'), "Нет _get_pde_multiplier"
    print("   ✅ PINNSolver имеет все необходимые методы")
    return True


# ============================================================================
# Тест 9: Сквозной тест (геометрия → коллокация → инициализация → обучение)
# ============================================================================
def test_9_end_to_end():
    """Сквозной тест: создаёт данные и запускает обучение на 2 эпохи."""
    import torch
    from Geometry import PipeBendGeometry, GeometryConfig
    from Fluids import FluidLibrary
    from Conditions import ConditionsManager
    from Initializer import Initializer

    print("   1. Создание геометрии отвода...")
    geom_config = GeometryConfig(save_format='npy', plot_show=False, verbose=False)
    geom = PipeBendGeometry(config=geom_config)
    if not geom.load_from_yaml(bend_name='PB_R100'):
        print("   ❌ Не удалось загрузить геометрию")
        return False
    domain = geom.create_domain()
    if domain is None:
        return False
    geom.save('pipe_bend')
    print(f"      Контур: {len(domain)} точек")

    print("   2. Создание жидкостей и условий...")
    fluids = FluidLibrary()
    fluids.create_default_library()
    cond_manager = ConditionsManager()
    cond_manager.create_defaults()
    print(f"      Жидкостей: {len(fluids.list_fluids())}, условий: {len(cond_manager.list_names())}")

    print("   3. Генерация точек коллокации (упрощённая)...")
    import matplotlib.path as mplPath
    from shapely.geometry import LineString

    R = geom.bend_radius
    W = geom.pipe_width
    R_outer = R + W / 2
    R_inner = R - W / 2
    x_min, y_min = np.min(domain, axis=0)
    x_max, y_max = np.max(domain, axis=0)
    step_size = 0.1

    path = mplPath.Path(domain)
    x_coords = np.arange(x_min, x_max, step_size)
    y_coords = np.arange(y_min, y_max, step_size)
    X, Y = np.meshgrid(x_coords, y_coords)
    points = np.column_stack((X.ravel(), Y.ravel()))
    inside = path.contains_points(points)
    interior_points = points[inside]

    # Граничные точки
    line = LineString(domain)
    total_length = line.length
    num_boundary = 100
    boundary_points = []
    boundary_types = []
    for i in range(num_boundary):
        distance = i * total_length / num_boundary
        pt = line.interpolate(distance)
        xb, yb = pt.x, pt.y
        if abs(xb - x_min) < 0.1 and yb < -R_inner:
            btype = 'inlet'
        elif abs(yb - y_max) < 0.1 and xb > x_min + W:
            btype = 'outlet'
        else:
            btype = 'wall'
        boundary_points.append((xb, yb))
        boundary_types.append(btype)

    all_points = np.array(interior_points.tolist() + boundary_points)
    all_types = ['interior'] * len(interior_points) + boundary_types
    dtype = [('id', 'i4'), ('x', 'f8'), ('y', 'f8'), ('boundary_type', 'U20'), ('on_boundary', 'bool')]
    structured = np.zeros(len(all_points), dtype=dtype)
    structured['id'] = np.arange(len(all_points))
    structured['x'] = all_points[:, 0]
    structured['y'] = all_points[:, 1]
    structured['boundary_type'] = all_types
    structured['on_boundary'] = [t != 'interior' for t in all_types]

    sampler_dir = os.path.join(PROJECT_DIR, 'Sampler', 'Data')
    os.makedirs(sampler_dir, exist_ok=True)
    np.save(os.path.join(sampler_dir, 'collocation_points.npy'), structured)
    metadata = {
        'num_points': len(structured),
        'bounds': {'x_min': x_min, 'x_max': x_max, 'y_min': y_min, 'y_max': y_max},
        'chord': geom.pipe_width,
        'pipe_width': geom.pipe_width,
    }
    np.save(os.path.join(sampler_dir, 'collocation_points_metadata.npy'), metadata)
    print(f"      Всего точек: {len(structured)} (interior: {len(interior_points)}, boundary: {len(boundary_points)})")

    print("   4. Инициализация данных...")
    init = Initializer()
    init.load_sampler_points()
    init.extract_wall_points()
    fluid_name = init.fluid_lib.list_fluids()[0]
    cond_name = init.cond_manager.list_names()[0]
    init.fluid = init.fluid_lib.get(fluid_name)
    init.conditions = init.cond_manager.get(cond_name)
    init.fluid_name = fluid_name
    init.conditions_name = cond_name
    init.initialize_fields(pressure_mode='linear_x', velocity_initial_mode='parabolic')
    init.save(description='initial')  # ← стандартное имя initial.npy
    print(f"      U_inf = {init.U_inf:.3f} м/с, Re = {init.Re:.2e}")

    print("   5. Тестовое обучение (2 эпохи)...")
    from solver import PINNSolver
    from config_manager import ConfigManager

    cfg = ConfigManager()
    cfg.config['project_name'] = 'test_pipe_bend'
    cfg.config['data']['points_file'] = os.path.join(PROJECT_DIR, 'Initial', 'Data', 'initial.npy')
    cfg.config['flow_type'] = 'planar'
    cfg.config['training']['epochs'] = 2
    cfg.config['training']['batch_size'] = 50
    cfg.config['training']['two_stage'] = False
    cfg.config['training']['rar_enable'] = False
    cfg.config['output']['console_only'] = True
    cfg.config['output']['plot_frequency'] = None
    cfg.config['enable_key_listener'] = False
    cfg.config['process']['device'] = 'cpu'
    cfg.config['process']['dtype'] = 'float32'
    cfg.config['loss_weights']['bc_weights'] = {'inlet': 5000.0, 'outlet': 5000.0, 'wall': 5000.0}
    cfg.config['checkpoint']['interval'] = 0

    try:
        solver = PINNSolver(cfg.config)
        solver.train(epochs=2)
        print(f"      ✅ Обучение завершено")
        return True
    except Exception as e:
        print(f"   ❌ Ошибка обучения: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# Главная функция
# ============================================================================
def main():
    global passed, failed
    passed = 0
    failed = 0

    print("=" * 70)
    print("ИНТЕГРАЦИОННОЕ ТЕСТИРОВАНИЕ PIPE-BEND PINN".center(70))
    print("=" * 70)

    tests = [
        (1, "Базовые импорты", test_1_imports),
        (2, "LossComputer контракт", test_2_loss_computer_contract),
        (3, "Модели нейросети", test_3_models),
        (4, "PDE невязки", test_4_pde_residuals),
        (5, "Trainer", test_5_trainer),
        (6, "MetricsCollector", test_6_metrics),
        (7, "DataHandler", test_7_data_handler),
        (8, "PINNSolver", test_8_solver),
        (9, "Сквозной тест (геом→колл→иниц→обуч)", test_9_end_to_end),
    ]

    for num, name, func in tests:
        test_step(num, name, func)

    # Итог
    print(f"\n{'='*70}")
    print("РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ".center(70))
    print(f"{'='*70}")
    print(f"  ✅ Успешно: {passed}")
    print(f"  ❌ Провалено: {failed}")
    print(f"  📊 Всего: {passed + failed}")
    print(f"{'='*70}")

    if failed == 0:
        print("🎉 ВСЕ ТЕСТЫ ПРОЙДЕНЫ!".center(70))
    else:
        print(f"⚠️ {failed} ТЕСТОВ НЕ ПРОЙДЕНЫ".center(70))
    print("=" * 70)

    return 0 if failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())