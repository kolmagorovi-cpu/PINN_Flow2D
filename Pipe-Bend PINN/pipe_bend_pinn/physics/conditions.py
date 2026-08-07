from pipe_bend_pinn.app.paths import (
    PROJECT_ROOT, GEOMETRY_DOMAINS_DIR, GEOMETRY_METADATA_DIR,
    SAMPLING_COLLOCATION_DIR, INITIAL_FIELDS_DIR, INITIAL_PLOTS_DIR,
    FLUIDS_DIR, CONDITIONS_DIR, EXPERIMENTS_DIR, GEOMS_LIB_PATH,
    CONFIGS_DIR, GEOM_CONFIG_PATH,
)

"""
Модуль граничных условий для Pipe-Bend PINN.

Поддерживает параболический профиль скорости (Пуазейль) для ламинарного течения,
а также равномерный профиль для турбулентного течения.

Автор: Адаптировано из Airfoil PINN
Версия: 1.0
"""

import os
import json
import logging
import numpy as np
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass

from pipe_bend_pinn.physics.fluids import FluidLibrary

logger = logging.getLogger(__name__)


@dataclass
class BoundaryConditions:
    """
    Класс граничных условий для задачи течения в отводе.

    Attributes:
        fluid_name: Название жидкости из библиотеки
        inlet_pressure: Давление на входе [Па]
        outlet_pressure: Давление на выходе [Па]
        inlet_u: Компонента U скорости на входе [м/с]
        inlet_v: Компонента V скорости на входе [м/с]
        velocity_profile_mode: 'uniform' или 'parabolic' (Пуазейль)
        pipe_width: Ширина канала [м] (для профиля Пуазейля)
        description: Описание набора условий
    """
    fluid_name: str
    inlet_pressure: float
    outlet_pressure: float
    inlet_u: float = 1.0
    inlet_v: float = 0.0
    velocity_profile_mode: str = 'uniform'
    pipe_width: float = 0.5
    symmetry_type: Optional[str] = None
    description: str = ""

    def __post_init__(self) -> None:
        if self.velocity_profile_mode not in ['uniform', 'parabolic']:
            raise ValueError("velocity_profile_mode должен быть 'uniform' или 'parabolic'")

    def get_u_profile(self, y_local: np.ndarray) -> np.ndarray:
        """
        Возвращает профиль скорости U(y) на входе.

        Args:
            y_local: Локальные Y-координаты поперёк канала [м]
                     (0 = нижняя стенка, pipe_width = верхняя стенка)

        Returns:
            Профиль скорости U(y) [м/с]
        """
        if self.velocity_profile_mode == 'uniform':
            return np.full_like(y_local, self.inlet_u)
        else:  # parabolic (Пуазейль)
            # U(y) = U_max * (1 - ((2y/D - 1)^2))
            # где D = pipe_width, y от 0 до D
            y_norm = y_local / self.pipe_width  # 0..1
            return self.inlet_u * (1 - (2 * y_norm - 1) ** 2)

    def to_dict(self) -> dict:
        return dict(
            fluid_name=self.fluid_name,
            inlet_pressure=self.inlet_pressure,
            outlet_pressure=self.outlet_pressure,
            inlet_u=self.inlet_u,
            inlet_v=self.inlet_v,
            velocity_profile_mode=self.velocity_profile_mode,
            pipe_width=self.pipe_width,
            symmetry_type=self.symmetry_type,
            description=self.description
        )

    @classmethod
    def from_dict(cls, data: dict) -> 'BoundaryConditions':
        return cls(**data)

    def display(self, fluid_lib: FluidLibrary) -> str:
        fluid = fluid_lib.get(self.fluid_name)
        fluid_str = fluid.display() if fluid else f"Жидкость '{self.fluid_name}' (не найдена)"
        lines = [
            f"Граничные условия: {self.description}",
            f"  Жидкость: {fluid_str}",
            f"  Давление: вход {self.inlet_pressure} Па, выход {self.outlet_pressure} Па",
            f"  Скорость на входе: U = {self.inlet_u:.3f} м/с, V = {self.inlet_v:.3f} м/с",
            f"  Профиль скорости: {self.velocity_profile_mode}",
        ]
        if self.velocity_profile_mode == 'parabolic':
            lines.append(f"  Ширина канала: {self.pipe_width} м")
        return "\n".join(lines)


class ConditionsManager:
    """Менеджер для управления наборами граничных условий."""

    def __init__(self, directory: str = CONDITIONS_DIR) -> None:
        self.directory = directory
        self.conditions: Dict[str, BoundaryConditions] = {}
        self.fluid_lib = FluidLibrary()
        os.makedirs(self.directory, exist_ok=True)
        self._load_all()

    def _load_all(self) -> None:
        self.conditions.clear()
        if not os.path.exists(self.directory):
            return
        for filename in os.listdir(self.directory):
            if filename.endswith(".json"):
                path = os.path.join(self.directory, filename)
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    cond = BoundaryConditions.from_dict(data)
                    key = filename[:-5]
                    self.conditions[key] = cond
                    logger.info(f"Загружены условия: {key}")
                except Exception as e:
                    logger.error(f"Ошибка загрузки {filename}: {e}")

    def save(self, cond: BoundaryConditions, name: str) -> None:
        path = os.path.join(self.directory, f"{name}.json")
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(cond.to_dict(), f, indent=4, ensure_ascii=False)
        self.conditions[name] = cond
        logger.info(f"Сохранены условия: {name}")

    def delete(self, name: str) -> None:
        if name in self.conditions:
            for filename in os.listdir(self.directory):
                if filename.endswith(".json") and name in filename:
                    os.remove(os.path.join(self.directory, filename))
                    break
            del self.conditions[name]

    def list_names(self) -> List[str]:
        return list(self.conditions.keys())

    def get(self, name: str) -> Optional[BoundaryConditions]:
        return self.conditions.get(name)

    def create_defaults(self) -> None:
        fluid_names = self.fluid_lib.list_fluids()
        if not fluid_names:
            self.fluid_lib.create_default_library()
            fluid_names = self.fluid_lib.list_fluids()
        if not fluid_names:
            logger.error("Нет доступных жидкостей")
            return
        default_fluid = fluid_names[0]

        defaults = [
            BoundaryConditions(
                fluid_name=default_fluid,
                inlet_pressure=101325.0,
                outlet_pressure=100000.0,
                inlet_u=1.0,
                inlet_v=0.0,
                velocity_profile_mode='parabolic',
                pipe_width=0.5,
                description="Ламинарное течение в отводе, профиль Пуазейля"
            ),
            BoundaryConditions(
                fluid_name=default_fluid,
                inlet_pressure=101325.0,
                outlet_pressure=101000.0,
                inlet_u=0.5,
                inlet_v=0.0,
                velocity_profile_mode='uniform',
                pipe_width=0.5,
                description="Равномерный профиль, малый перепад давления"
            ),
        ]
        for i, cond in enumerate(defaults, 1):
            self.save(cond, f"Пример_{i}")


def select_fluid(fluid_lib: FluidLibrary) -> Optional[str]:
    """Интерактивный выбор жидкости."""
    fluids = fluid_lib.list_fluids()
    if not fluids:
        print("Библиотека жидкостей пуста.")
        return None
    print("\nДоступные жидкости:")
    for i, name in enumerate(fluids, 1):
        fluid = fluid_lib.get(name)
        print(f"{i}. {fluid.display()}")
    while True:
        try:
            choice = input("Выберите номер жидкости (0 для отмены): ").strip()
            if choice == '0':
                return None
            idx = int(choice) - 1
            if 0 <= idx < len(fluids):
                return fluids[idx]
            print("Неверный номер.")
        except ValueError:
            print("Введите число.")


def conditions_menu() -> None:
    """Интерактивное меню для управления граничными условиями."""
    manager = ConditionsManager()
    if not manager.list_names():
        print("\n📂 Библиотека условий пуста.")
        choice = input("Создать примеры условий? (Y/n): ").strip().lower()
        if choice == '' or choice in ('y', 'yes'):
            manager.create_defaults()
            print("✅ Примеры созданы.")

    while True:
        print("\n" + "="*60)
        print("ГРАНИЧНЫЕ УСЛОВИЯ".center(60))
        print("="*60)
        print("1. Показать список условий")
        print("2. Создать новые условия")
        print("3. Редактировать условия")
        print("4. Удалить условия")
        print("5. Создать примеры (сбросить)")
        print("0. Вернуться в главное меню")

        choice = input("\nВыберите действие (0-5): ").strip()
        if choice == '0':
            break
        elif choice == '1':
            names = manager.list_names()
            if not names:
                print("Библиотека пуста.")
            else:
                print("\nСохранённые наборы условий:")
                for i, name in enumerate(names, 1):
                    cond = manager.get(name)
                    print(f"{i}. {name}")
                    if cond:
                        print(f"   {cond.display(manager.fluid_lib)}")
                    print()
        elif choice == '2':
            create_conditions(manager)
        elif choice == '3':
            edit_conditions(manager)
        elif choice == '4':
            delete_conditions(manager)
        elif choice == '5':
            confirm = input("Это удалит все текущие условия. Продолжить? (y/N): ").strip().lower()
            if confirm in ('y', 'yes'):
                for f in os.listdir(manager.directory):
                    os.remove(os.path.join(manager.directory, f))
                manager.create_defaults()
                print("✅ Примеры созданы.")
            else:
                print("Отменено.")
        else:
            print("Неверный выбор.")


def create_conditions(manager: ConditionsManager) -> None:
    """Создаёт новые граничные условия."""
    print("\nВведите параметры новых граничных условий.")
    try:
        fluid_name = select_fluid(manager.fluid_lib)
        if fluid_name is None:
            return

        name = input("Название набора: ").strip()
        if not name:
            print("Название не может быть пустым.")
            return
        inlet_p = float(input("Давление на входе [Па]: "))
        outlet_p = float(input("Давление на выходе [Па]: "))
        u = float(input("U на входе [м/с]: "))
        v = float(input("V на входе [м/с]: "))

        print("Профиль скорости: 1 - равномерный, 2 - параболический (Пуазейль)")
        profile_choice = input("Выберите (1/2): ").strip()
        profile_mode = 'uniform' if profile_choice == '1' else 'parabolic'

        pipe_width = 0.5
        if profile_mode == 'parabolic':
            pipe_width = float(input("Ширина канала [м]: "))

        desc = input("Описание: ").strip()
        cond = BoundaryConditions(
            fluid_name=fluid_name,
            inlet_pressure=inlet_p,
            outlet_pressure=outlet_p,
            inlet_u=u,
            inlet_v=v,
            velocity_profile_mode=profile_mode,
            pipe_width=pipe_width,
            description=desc
        )
        manager.save(cond, name)
        print(f"✅ Условия '{name}' сохранены.")
    except ValueError as e:
        print(f"❌ Ошибка ввода: {e}")


def edit_conditions(manager: ConditionsManager) -> None:
    """Редактирование набора граничных условий."""
    names = manager.list_names()
    if not names:
        print("Библиотека пуста.")
        return
    print("Выберите набор для редактирования:")
    for i, name in enumerate(names, 1):
        print(f"{i}. {name}")
    try:
        idx = int(input("Номер: ")) - 1
        if idx < 0 or idx >= len(names):
            print("Неверный номер.")
            return
        old_name = names[idx]
        cond = manager.get(old_name)
        if cond is None:
            print("Условия не найдены.")
            return

        print(f"\nРедактирование '{old_name}'. Enter - без изменений.")
        new_fluid = input(f"Жидкость [{cond.fluid_name}]: ").strip() or cond.fluid_name
        new_name = input(f"Новое название [{old_name}]: ").strip() or old_name
        new_inlet_p = float(input(f"Давление на входе [{cond.inlet_pressure}]: ").strip() or cond.inlet_pressure)
        new_outlet_p = float(input(f"Давление на выходе [{cond.outlet_pressure}]: ").strip() or cond.outlet_pressure)
        new_u = float(input(f"U [{cond.inlet_u}]: ").strip() or cond.inlet_u)
        new_v = float(input(f"V [{cond.inlet_v}]: ").strip() or cond.inlet_v)

        print(f"Профиль скорости [{cond.velocity_profile_mode}]:")
        profile_choice = input("1 - равномерный, 2 - параболический: ").strip()
        if profile_choice == '1':
            profile_mode = 'uniform'
        elif profile_choice == '2':
            profile_mode = 'parabolic'
        else:
            profile_mode = cond.velocity_profile_mode

        new_pipe_width = cond.pipe_width
        if profile_mode == 'parabolic':
            new_pipe_width = float(input(f"Ширина канала [{cond.pipe_width}]: ").strip() or cond.pipe_width)

        new_desc = input(f"Описание [{cond.description}]: ").strip() or cond.description

        updated = BoundaryConditions(
            fluid_name=new_fluid,
            inlet_pressure=new_inlet_p,
            outlet_pressure=new_outlet_p,
            inlet_u=new_u,
            inlet_v=new_v,
            velocity_profile_mode=profile_mode,
            pipe_width=new_pipe_width,
            description=new_desc
        )
        if new_name != old_name:
            manager.delete(old_name)
        manager.save(updated, new_name)
        print(f"✅ Условия обновлены.")
    except (ValueError, IndexError) as e:
        print(f"❌ Ошибка: {e}")


def delete_conditions(manager: ConditionsManager) -> None:
    """Удаление набора граничных условий."""
    names = manager.list_names()
    if not names:
        print("Библиотека пуста.")
        return
    print("Выберите набор для удаления:")
    for i, name in enumerate(names, 1):
        print(f"{i}. {name}")
    try:
        idx = int(input("Номер: ")) - 1
        if idx < 0 or idx >= len(names):
            print("Неверный номер.")
            return
        name = names[idx]
        confirm = input(f"Удалить '{name}'? (y/N): ").strip().lower()
        if confirm in ('y', 'yes'):
            manager.delete(name)
            print(f"✅ Удалено.")
        else:
            print("Отменено.")
    except ValueError:
        print("❌ Неверный ввод.")


if __name__ == "__main__":
    conditions_menu()