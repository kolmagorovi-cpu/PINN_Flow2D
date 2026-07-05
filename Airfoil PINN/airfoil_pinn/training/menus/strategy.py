#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Меню стратегии обучения PINN.

Здесь собраны интерактивные пункты, которые меняют раздел ``training``
конфигурации: этапы, LR, RAR, L-BFGS, размеры батчей и early stopping.
Главный ``learning_cli`` остаётся маршрутизатором и не хранит детали этого
подменю.
"""

from airfoil_pinn.config.manager import ConfigManager, yes_no
from airfoil_pinn.training.menus.common import (
    print_submenu_footer,
    print_submenu_header,
)
from airfoil_pinn.utils import clear_screen

__all__ = ["training_strategy_menu"]


def training_strategy_menu(cfg: ConfigManager) -> None:
    """Меню настройки стратегии обучения."""
    while True:
        clear_screen()
        print_submenu_header("Настройка стратегии обучения")
        tr = cfg.config['training']
        print("1. Двухэтапное обучение")
        print("2. Изменяющийся LR")
        print("3. Период сброса оптимизатора")
        print("4. Отсечение градиентов")
        print("5. RAR")
        print("6. Переключение Adam → L-BFGS")
        print("7. Размер батча и градиентное накопление")
        print("8. Количество эпох")
        print("9. Early stopping")
        print("10. Вывод текущей стратегии")
        print("0. Выход")
        sub = input("Выберите пункт: ").strip()

        if sub == '1':
            tr['two_stage'] = yes_no("Двухэтапное обучение", tr['two_stage'])
            if tr['two_stage']:
                tr['stage1_epochs'] = int(input(f"Эпох только BC [{tr['stage1_epochs']}]: ") or tr['stage1_epochs'])
                tr['stage2_pde_weight_start'] = float(input(f"Начальный вес PDE [{tr['stage2_pde_weight_start']}]: ") or tr['stage2_pde_weight_start'])
                tr['stage2_pde_weight_end'] = float(input(f"Конечный вес PDE [{tr['stage2_pde_weight_end']}]: ") or tr['stage2_pde_weight_end'])
                tr['stage2_ramp_epochs'] = int(input(f"Эпох нарастания [{tr['stage2_ramp_epochs']}]: ") or tr['stage2_ramp_epochs'])
        elif sub == '2':
            print("Тип изменения LR: 0 - постоянный, 1 - периодический, 2 - по детектору")
            lr_type = input("Выберите (0/1/2): ").strip()
            if lr_type == '1':
                tr['lr_schedule'] = 'cyclic'
                tr['lr_cyclic_period'] = int(input(f"Период (эпох) [{tr['lr_cyclic_period']}]: ") or tr['lr_cyclic_period'])
                tr['lr_cyclic_amplitude'] = float(input(f"Амплитуда [{tr['lr_cyclic_amplitude']}]: ") or tr['lr_cyclic_amplitude'])
            elif lr_type == '2':
                tr['lr_schedule'] = 'detector'
                tr['lr_detector_factor'] = float(input(f"Коэффициент увеличения [{tr['lr_detector_factor']}]: ") or tr['lr_detector_factor'])
                tr['lr_detector_patience'] = int(input(f"Терпение (эпох) [{tr['lr_detector_patience']}]: ") or tr['lr_detector_patience'])
                tr['lr_detector_recovery'] = int(input(f"Эпох восстановления [{tr['lr_detector_recovery']}]: ") or tr['lr_detector_recovery'])
            else:
                tr['lr_schedule'] = 'constant'
        elif sub == '3':
            tr['reset_interval'] = int(input(f"Интервал сброса оптимизатора (0 - выкл) [{tr['reset_interval']}]: ") or tr['reset_interval'])
            if tr['reset_interval'] > 0:
                tr['reset_noise'] = float(input(f"Амплитуда шума [{tr['reset_noise']}]: ") or tr['reset_noise'])
        elif sub == '4':
            clip = input(f"Градиентное отсечение (Enter - выкл, текущ: {tr['grad_clip']}): ").strip()
            tr['grad_clip'] = float(clip) if clip else None
        elif sub == '5':
            tr['rar_enable'] = yes_no("RAR", tr['rar_enable'])
            if tr['rar_enable']:
                tr.setdefault('rar_mode', 'adaptive')
                tr.setdefault('rar_score_mode', 'total_loss')
                tr.setdefault('rar_error_quantile', 0.75)
                tr.setdefault('rar_jitter_scale', 0.85)
                tr.setdefault('rar_min_distance_factor', 0.35)
                tr.setdefault('rar_rebuild_fvm', True)
                tr.setdefault('rar_fvm_rebuild_every', 1)
                tr.setdefault('rar_fvm_min_new_points', 1)
                tr.setdefault('rar_save_mesh_volume_plot', True)
                tr.setdefault('rar_mesh_plot_dpi', 150)
                tr['rar_interval'] = int(input(f"Интервал RAR [{tr['rar_interval']}]: ") or tr['rar_interval'])
                tr['rar_num_points'] = int(input(f"Количество точек [{tr['rar_num_points']}]: ") or tr['rar_num_points'])
                mode = input(f"Режим RAR (adaptive/legacy) [{tr['rar_mode']}]: ").strip().lower()
                if mode in ('adaptive', 'legacy'):
                    tr['rar_mode'] = mode
                if tr['rar_mode'] == 'adaptive':
                    score_mode = input(
                        f"Score для выбора RAR-точек (total_loss/pde) [{tr['rar_score_mode']}]: "
                    ).strip().lower()
                    if score_mode in ('total_loss', 'pde'):
                        tr['rar_score_mode'] = score_mode
                    tr['rar_error_quantile'] = float(input(
                        f"Квантиль ошибки для источников [{tr['rar_error_quantile']}]: "
                    ) or tr['rar_error_quantile'])
                    tr['rar_jitter_scale'] = float(input(
                        f"Радиус генерации в локальных шагах [{tr['rar_jitter_scale']}]: "
                    ) or tr['rar_jitter_scale'])
                    tr['rar_min_distance_factor'] = float(input(
                        f"Мин. расстояние до соседей в локальных шагах [{tr['rar_min_distance_factor']}]: "
                    ) or tr['rar_min_distance_factor'])
                    flow_rar = tr.setdefault('flow_rar', {})
                    flow_rar.setdefault('enabled', True)
                    flow_rar.setdefault('num_points', None)
                    flow_rar.setdefault('exclude_sdf_delta', None)
                    boundary_rar = tr.setdefault('boundary_rar', {})
                    boundary_rar.setdefault('enabled', False)
                    boundary_rar.setdefault('num_points', None)
                    boundary_rar.setdefault('fraction', 0.4)
                    boundary_rar.setdefault('delta_mode', 'estimated')
                    boundary_rar.setdefault('delta_max', 5.0)
                    boundary_rar.setdefault(
                        'eta_levels',
                        [0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0],
                    )
                    boundary_rar.setdefault('contour_samples', 400)
                    boundary_rar.setdefault('alpha', 1.5)
                    boundary_rar.setdefault('uniform_contour_weight', 0.5)
                    boundary_rar.setdefault('min_distance_factor', 0.02)
                    boundary_rar.setdefault('score_batch_size', 2048)
                    boundary_rar.setdefault(
                        'error_weights',
                        {'cont': 1.0, 'mom_x': 3.0, 'mom_y': 3.0, 'mom_r': 3.0, 'mom_z': 3.0},
                    )

                    flow_rar['enabled'] = yes_no("Flow RAR для свободного потока", flow_rar.get('enabled', True))
                    excl_current = flow_rar.get('exclude_sdf_delta', None)
                    excl = input(
                        f"Исключать из Flow RAR зону sdf/delta < value (Enter - авто/выкл, текущ: {excl_current}): "
                    ).strip()
                    if excl:
                        flow_rar['exclude_sdf_delta'] = float(excl)

                    boundary_rar['enabled'] = yes_no(
                        "Boundary-Layer RAR для пограничного слоя",
                        boundary_rar.get('enabled', False),
                    )
                    if boundary_rar['enabled']:
                        n_boundary = input(
                            f"Boundary RAR точек (Enter - fraction от rar_num_points, текущ: {boundary_rar.get('num_points')}): "
                        ).strip()
                        boundary_rar['num_points'] = int(n_boundary) if n_boundary else None
                        if boundary_rar['num_points'] is None:
                            boundary_rar['fraction'] = float(input(
                                f"Boundary RAR fraction [{boundary_rar.get('fraction', 0.4)}]: "
                            ) or boundary_rar.get('fraction', 0.4))
                        boundary_rar['delta_max'] = float(input(
                            f"Максимальная высота Boundary RAR sdf/delta [{boundary_rar.get('delta_max', 5.0)}]: "
                        ) or boundary_rar.get('delta_max', 5.0))
                        boundary_rar['contour_samples'] = int(input(
                            f"Сэмплов по контуру профиля [{boundary_rar.get('contour_samples', 400)}]: "
                        ) or boundary_rar.get('contour_samples', 400))
                        boundary_rar['alpha'] = float(input(
                            f"Степень вероятности по ошибке alpha [{boundary_rar.get('alpha', 1.5)}]: "
                        ) or boundary_rar.get('alpha', 1.5))
                        boundary_rar['uniform_contour_weight'] = float(input(
                            "Доля равномерности по контуру "
                            f"[{boundary_rar.get('uniform_contour_weight', 0.5)}]: "
                        ) or boundary_rar.get('uniform_contour_weight', 0.5))
                        boundary_rar['min_distance_factor'] = float(input(
                            "Мин. расстояние Boundary RAR в долях delta "
                            f"[{boundary_rar.get('min_distance_factor', 0.02)}]: "
                        ) or boundary_rar.get('min_distance_factor', 0.02))
                    tr['rar_rebuild_fvm'] = yes_no("Перестраивать FVM после adaptive RAR", tr['rar_rebuild_fvm'])
                    if tr['rar_rebuild_fvm']:
                        tr['rar_fvm_rebuild_every'] = int(input(
                            f"FVM rebuild раз в N срабатываний RAR [{tr['rar_fvm_rebuild_every']}]: "
                        ) or tr['rar_fvm_rebuild_every'])
                        tr['rar_fvm_min_new_points'] = int(input(
                            f"Мин. новых точек для FVM rebuild [{tr['rar_fvm_min_new_points']}]: "
                        ) or tr['rar_fvm_min_new_points'])
                    tr['rar_save_mesh_volume_plot'] = yes_no(
                        "Сохранять график CV-объёмов сетки после RAR",
                        tr['rar_save_mesh_volume_plot'],
                    )
                    if tr['rar_save_mesh_volume_plot']:
                        tr['rar_mesh_plot_dpi'] = int(input(
                            f"DPI графика CV-объёмов [{tr['rar_mesh_plot_dpi']}]: "
                        ) or tr['rar_mesh_plot_dpi'])
        elif sub == '6':
            tr['switch_to_lbfgs_epoch'] = int(input(f"Переключение на L-BFGS (эпоха, 0 - выкл) [{tr['switch_to_lbfgs_epoch']}]: ") or tr['switch_to_lbfgs_epoch'])
            if tr['switch_to_lbfgs_epoch'] > 0:
                tr['lbfgs_max_iter'] = int(input(f"max_iter L-BFGS [{tr['lbfgs_max_iter']}]: ") or tr['lbfgs_max_iter'])
                tr['lbfgs_history_size'] = int(input(f"history_size [{tr['lbfgs_history_size']}]: ") or tr['lbfgs_history_size'])
        elif sub == '7':
            tr['batch_size'] = int(input(f"Размер батча [{tr['batch_size']}]: ") or tr['batch_size'])
            tr['gradient_accumulation_steps'] = int(input(f"Шаги градиентного накопления [{tr['gradient_accumulation_steps']}]: ") or tr['gradient_accumulation_steps'])
        elif sub == '8':
            tr['epochs'] = int(input(f"Количество эпох [{tr['epochs']}]: ") or tr['epochs'])
        elif sub == '9':
            es = tr.setdefault('early_stopping', {
                'enabled': False,
                'patience': 100,
                'min_delta': 1e-6,
                'restore_best': True,
            })
            es['enabled'] = yes_no("Early stopping", es.get('enabled', False))
            if es['enabled']:
                es['patience'] = int(input(f"Patience [{es.get('patience', 100)}]: ") or es.get('patience', 100))
                es['min_delta'] = float(input(f"Min delta [{es.get('min_delta', 1e-6)}]: ") or es.get('min_delta', 1e-6))
                es['restore_best'] = yes_no("Восстановить лучшую модель", es.get('restore_best', True))
        elif sub == '10':
            print("\n--- ТЕКУЩАЯ СТРАТЕГИЯ ОБУЧЕНИЯ ---")
            print(f"Двухэтапное обучение: {tr['two_stage']}")
            if tr['two_stage']:
                print(f"  Stage1 эпох: {tr['stage1_epochs']}")
                print(f"  Stage2 вес PDE от {tr['stage2_pde_weight_start']} до {tr['stage2_pde_weight_end']} за {tr['stage2_ramp_epochs']} эпох")
            print(f"LR schedule: {tr['lr_schedule']}")
            print(f"Интервал сброса: {tr['reset_interval']} (шум {tr['reset_noise']})")
            print(f"Градиентное отсечение: {tr['grad_clip']}")
            print(f"RAR: {tr['rar_enable']}")
            if tr['rar_enable']:
                print(f"  режим: {tr.get('rar_mode', 'adaptive')}, "
                      f"интервал: {tr['rar_interval']}, точек: {tr['rar_num_points']}")
                if tr.get('rar_mode', 'adaptive') == 'adaptive':
                    print(f"  score: {tr.get('rar_score_mode', 'total_loss')}, "
                          f"error_quantile: {tr.get('rar_error_quantile', 0.75)}, "
                          f"FVM rebuild: {tr.get('rar_rebuild_fvm', True)}")
                    print(f"  mesh CV plot: {tr.get('rar_save_mesh_volume_plot', True)}, "
                          f"dpi={tr.get('rar_mesh_plot_dpi', 150)}")
                    flow_rar = tr.get('flow_rar', {}) or {}
                    boundary_rar = tr.get('boundary_rar', {}) or {}
                    print(
                        "  Flow RAR: "
                        f"enabled={flow_rar.get('enabled', True)}, "
                        f"num_points={flow_rar.get('num_points', None)}, "
                        f"exclude_sdf_delta={flow_rar.get('exclude_sdf_delta', None)}"
                    )
                    print(
                        "  Boundary RAR: "
                        f"enabled={boundary_rar.get('enabled', False)}, "
                        f"num_points={boundary_rar.get('num_points', None)}, "
                        f"fraction={boundary_rar.get('fraction', 0.4)}, "
                        f"delta_max={boundary_rar.get('delta_max', 5.0)}, "
                        f"min_distance_factor={boundary_rar.get('min_distance_factor', 0.02)}"
                    )
            print(f"Переключение на L-BFGS: эпоха {tr['switch_to_lbfgs_epoch']}")
            print(f"Батч: {tr['batch_size']}, накопление: {tr['gradient_accumulation_steps']}")
            print(f"Эпох: {tr['epochs']}")
            es = tr.get('early_stopping', {})
            print(f"Early stopping: {es.get('enabled', False)} "
                  f"(patience={es.get('patience', 100)}, min_delta={es.get('min_delta', 1e-6)}, "
                  f"restore_best={es.get('restore_best', True)})")
            input("Нажмите Enter...")
        elif sub == '0':
            print_submenu_footer()
            break
