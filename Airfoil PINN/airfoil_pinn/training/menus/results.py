#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Меню вывода, чекпоинтов и диагностики PINN.

Модуль собирает экраны, отвечающие за наблюдаемость и сохранение
результатов обучения. Основной CLI только вызывает эти функции и
не хранит детали настроек ``output``, ``checkpoint`` и ``diagnostics``.
"""

from airfoil_pinn.config.manager import ConfigManager, yes_no
from airfoil_pinn.training.menus.common import (
    print_submenu_footer,
    print_submenu_header,
)
from airfoil_pinn.utils import clear_screen

__all__ = ["checkpoint_menu", "diagnostics_menu", "output_menu"]


def output_menu(cfg: ConfigManager) -> None:
    """Меню настройки вывода."""
    while True:
        clear_screen()
        print_submenu_header("Настройки вывода")
        out = cfg.config['output']
        print("1. Только консоль / графическое окно")
        print("2. Частота обновления графиков (эпох)")
        print("3. Показать настройки вывода")
        print("0. Выход")
        sub = input("Выберите пункт: ").strip()

        if sub == '1':
            out['console_only'] = yes_no("Только консоль (иначе графические окна)", out['console_only'])
            if not out['console_only']:
                gw = out['graphical_window']
                gw['loss'] = yes_no("Показывать общую потерю (Loss)", gw['loss'])
                gw['pde'] = yes_no("Показывать PDE (каждая невязка)", gw['pde'])
                gw['bc'] = yes_no("Показывать BC (каждая невязка)", gw['bc'])
        elif sub == '2':
            out['plot_frequency'] = int(input(f"Частота обновления графиков (эпох) [{out['plot_frequency']}]: ") or out['plot_frequency'])
        elif sub == '3':
            print("\n--- ТЕКУЩИЕ НАСТРОЙКИ ВЫВОДА ---")
            print(f"Только консоль: {out['console_only']}")
            if not out['console_only']:
                print(f"  Loss: {out['graphical_window']['loss']}")
                print(f"  PDE: {out['graphical_window']['pde']}")
                print(f"  BC: {out['graphical_window']['bc']}")
            print(f"Частота обновления графиков: {out['plot_frequency']}")
            input("Нажмите Enter...")
        elif sub == '0':
            print_submenu_footer()
            break


def checkpoint_menu(cfg: ConfigManager) -> None:
    """Меню настройки сохранения."""
    while True:
        clear_screen()
        print_submenu_header("Настройки сохранения")
        ck = cfg.config['checkpoint']
        print("1. Период сохранения контрольных точек")
        print("2. Имя для сохранения контрольных точек")
        print("3. Имя нейронной сети при сохранении")
        print("4. Показать настройки сохранения")
        print("0. Выход")
        sub = input("Выберите пункт: ").strip()

        if sub == '1':
            ck['interval'] = int(input(f"Период сохранения чекпоинтов (0 - выкл) [{ck['interval']}]: ") or ck['interval'])
        elif sub == '2':
            ck['name_pattern'] = input(f"Шаблон имени чекпоинта (используйте {{project_name}}, {{timestamp}}) [{ck['name_pattern']}]: ").strip() or ck['name_pattern']
        elif sub == '3':
            ck['model_name_pattern'] = input(f"Шаблон имени модели [{ck['model_name_pattern']}]: ").strip() or ck['model_name_pattern']
        elif sub == '4':
            print("\n--- ТЕКУЩИЕ НАСТРОЙКИ СОХРАНЕНИЯ ---")
            print(f"Период чекпоинтов: {ck['interval']}")
            print(f"Шаблон чекпоинта: {ck['name_pattern']}")
            print(f"Шаблон модели: {ck['model_name_pattern']}")
            input("Нажмите Enter...")
        elif sub == '0':
            print_submenu_footer()
            break


def diagnostics_menu(cfg: ConfigManager) -> None:
    """Меню настройки диагностики."""
    while True:
        clear_screen()
        print_submenu_header("Настройки диагностики")
        diag = cfg.config.get('diagnostics', {})
        if not diag:
            diag = cfg.config['diagnostics'] = {}
        print("1. Интервал диагностики (эпох)")
        print("2. Построение карты невязки")
        print("3. Построение карты полной ошибки RAR")
        print("4. Построение профилей скорости")
        print("5. Сечения для профилей (через запятую)")
        print("6. Построение спектра (1D)")
        print("7. Координата X для спектра (1D)")
        print("8. Построение 2D спектра")
        print("9. Разрешение 2D спектра")
        print("10. Вычисление раздельных градиентов")
        print("11. Вычисление NTK и вывод в консоль")
        print("12. Сохранение истории RAR")
        print("13. Интервал inlet/outlet метрик")
        print("14. Показать текущие настройки")
        print(f"15. Подробные градиенты: каждые {diag.get('gradient_report_interval', 50)} эпох (0 — отключено)")
        state = 'включены' if diag.get('gradient_contributions_enabled', False) else 'выключены'
        print(f"16. Градиентные вклады PDE / BC / FVM / Integral: {state}")
        print(f"17. Интервал вывода градиентных вкладов: {diag.get('gradient_contributions_interval', 50)} эпох")
        cosine_state = 'включён' if diag.get('compute_fvm_pde_cosine', False) else 'выключен'
        print(f"18. Вычисление и вывод cosine(FVM, PDE), cosine(FVM, BC), cosine(PDE, BC): {cosine_state}")
        print("0. Выход")
        sub = input("Выберите пункт: ").strip()

        if sub == '1':
            diag['interval'] = int(input(f"Интервал (эпох) [{diag.get('interval', 50)}]: ") or diag.get('interval', 50))
        elif sub == '2':
            diag['plot_residual_map'] = yes_no("Строить карту невязки", diag.get('plot_residual_map', True))
        elif sub == '3':
            diag['plot_rar_error_map'] = yes_no("Строить карту полной ошибки RAR", diag.get('plot_rar_error_map', True))
        elif sub == '4':
            diag['plot_profiles'] = yes_no("Строить профили скорости", diag.get('plot_profiles', True))
        elif sub == '5':
            current = diag.get('profile_sections', [0.2, 0.5, 0.8])
            sections_str = input(f"Сечения (x, м) через запятую [{', '.join(map(str, current))}]: ").strip()
            if sections_str:
                try:
                    diag['profile_sections'] = [float(x.strip()) for x in sections_str.split(',')]
                except ValueError:
                    print("Неверный формат, оставлено текущее значение.")
        elif sub == '6':
            diag['plot_spectrum'] = yes_no("Строить 1D спектр", diag.get('plot_spectrum', True))
        elif sub == '7':
            diag['spectrum_section_x'] = float(input(f"X для 1D спектра (м) [{diag.get('spectrum_section_x', 0.5)}]: ") or diag.get('spectrum_section_x', 0.5))
        elif sub == '8':
            diag['plot_2d_spectrum'] = yes_no("Строить 2D спектр", diag.get('plot_2d_spectrum', False))
        elif sub == '9':
            diag['spectrum_2d_resolution'] = int(input(f"Разрешение 2D спектра [{diag.get('spectrum_2d_resolution', 256)}]: ") or diag.get('spectrum_2d_resolution', 256))
        elif sub == '10':
            diag['compute_grad_cosine'] = yes_no("Вычислять раздельные градиенты", diag.get('compute_grad_cosine', False))
        elif sub == '11':
            print("\nNTK: отдельные K = J J^T для знаковых невязок PDE, BC и FVM.")
            print("Модель, невязки, якобианы и спектры вычисляются в float64 без AMP.")
            print("Веса loss и SA не включаются; hard BC исключены из блока BC.")
            print("FVM: r_i = sqrt(V_i) * divergence_i; без нормировки на суммарную площадь.")
            print("Выводятся полный спектр и эффективный ранг exp(entropy).")
            print("Пороги: lambda > rtol * lambda_max, rtol = 1e-12, 1e-10, 1e-8, 1e-6.")
            print("Для каждого порога: численный ранг, число отсечений и condition.")
            print("Condition относится к оставленному подпространству; полный ранг проверяется отдельно.")
            print("ntk_condition в истории — PDE condition при rtol=1e-12.")
            print("ntk_sample_size ограничивает точки PDE/BC и объёмы FVM на каждый режим.")
            print("Пустой блок означает отсутствие выбранных невязок; FVM без сетки недоступен.")
            print("Расчёт и вывод выполняются с интервалом диагностики, кроме этапа L-BFGS.")
            print("Расчёт NTK требует дополнительных времени и памяти.")
            diag['compute_ntk_condition'] = yes_no(
                "Вычислять NTK и выводить в консоль",
                diag.get('compute_ntk_condition', False),
            )
        elif sub == '12':
            diag['save_rar_history'] = yes_no("Сохранять историю RAR", diag.get('save_rar_history', True))
        elif sub == '13':
            diag['mass_metrics_interval'] = int(
                input(f"Интервал inlet/outlet метрик [{diag.get('mass_metrics_interval', 1)}]: ")
                or diag.get('mass_metrics_interval', 1)
            )
            diag['mass_metrics_interval'] = max(1, diag['mass_metrics_interval'])
        elif sub == '15':
            current = diag.get('gradient_report_interval', 50)
            try:
                value = int(input(f"Интервал вывода градиентов в эпохах (0 — отключить) [{current}]: ") or current)
                if value < 0:
                    raise ValueError
                diag['gradient_report_interval'] = value
            except ValueError:
                print("Введите целое число >= 0. Текущее значение сохранено.")
                input("Нажмите Enter...")
        elif sub == '16':
            print("L2-нормы взвешенных градиентов по параметрам модели до clipping.")
            print("Вывод в консоль, текстовый лог и epochs.jsonl. Расчёт требует дополнительного времени и памяти.")
            diag['gradient_contributions_enabled'] = yes_no(
                "Измерять и выводить градиентные вклады PDE / BC / FVM / Integral",
                diag.get('gradient_contributions_enabled', False),
            )
        elif sub == '17':
            current = diag.get('gradient_contributions_interval', 50)
            try:
                value = int(input(f"Интервал вывода вкладов (эпох, >= 1) [{current}]: ") or current)
                if value < 1:
                    raise ValueError
                diag['gradient_contributions_interval'] = value
            except ValueError:
                print("Введите целое число >= 1. Текущее значение сохранено.")
                input("Нажмите Enter...")
        elif sub == '18':
            print("Косинусы между накопленными градиентами FVM, PDE и BC до clipping.")
            print("Работает независимо от пункта 16; периодичность задаётся пунктом 17.")
            diag['compute_fvm_pde_cosine'] = yes_no(
                "Вычислять и выводить cosine(FVM, PDE), cosine(FVM, BC), cosine(PDE, BC)",
                diag.get('compute_fvm_pde_cosine', False),
            )
        elif sub == '14':
            print(f"cosine(FVM, PDE), cosine(FVM, BC), cosine(PDE, BC): {cosine_state}")
            print(f"Градиентные вклады PDE / BC / FVM / Integral: {state}")
            print(f"Интервал вывода вкладов: {diag.get('gradient_contributions_interval', 50)} эпох")
            print(f"Интервал подробного вывода градиентов: {diag.get('gradient_report_interval', 50)} (0 — отключено)")
            print("\n--- ТЕКУЩИЕ НАСТРОЙКИ ДИАГНОСТИКИ ---")
            print(f"Интервал: {diag.get('interval', 50)}")
            print(f"Карта невязки: {diag.get('plot_residual_map', True)}")
            print(f"Карта полной ошибки RAR: {diag.get('plot_rar_error_map', True)}")
            print(f"Профили: {diag.get('plot_profiles', True)}")
            print(f"Сечения: {diag.get('profile_sections', [0.2, 0.5, 0.8])}")
            print(f"1D спектр: {diag.get('plot_spectrum', True)}")
            print(f"X 1D спектра: {diag.get('spectrum_section_x', 0.5)}")
            print(f"2D спектр: {diag.get('plot_2d_spectrum', False)}")
            print(f"Разрешение 2D спектра: {diag.get('spectrum_2d_resolution', 256)}")
            print(f"Раздельные градиенты: {diag.get('compute_grad_cosine', False)}")
            print(f"NTK (расчёт и вывод в консоль): {diag.get('compute_ntk_condition', False)}")
            print(f"История RAR: {diag.get('save_rar_history', True)}")
            print(f"Интервал inlet/outlet метрик: {diag.get('mass_metrics_interval', 1)}")
            input("Нажмите Enter...")
        elif sub == '0':
            print_submenu_footer()
            break
