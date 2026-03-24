#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Модуль граничных условий Conditions.

Позволяет задавать давление на входе/выходе и скорость на входе.
Поддерживает ввод скорости как (U, V) или как (скорость, угол атаки).
При наличии симметрии автоматически обнуляет вертикальную компоненту/угол.
Сохраняет настройки в JSON, включая имя выбранной жидкости.

Example:
    >>> cond = BoundaryConditions(
    ...     fluid_name="Воздух (20°C)",
    ...     inlet_pressure=101325.0,
    ...     outlet_pressure=101325.0,
    ...     inlet_velocity_mode='components',
    ...     inlet_u=10.0,
    ...     inlet_v=0.0,
    ...     description="Обтекание при нулевом угле атаки"
    ... )
    >>> manager = ConditionsManager()
    >>> manager.save(cond, "my_conditions")

Автор: Колмагоров И.Д.
Версия: 3.260226 (исправленная)
"""

import os
import json
import logging
import numpy as np
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, asdict

from Fluids import FluidLibrary
from utils import safe_input_number, input_with_default, confirm_action
from type_hints import BoundaryConditionsDict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class BoundaryConditions:
    """
    Класс граничных условий для задачи обтекания профиля.

    Attributes:
        fluid_name: Название жидкости из библиотеки.
        inlet_pressure: Давление на входе [Па].
        outlet_pressure: Давление на выходе [Па].
        inlet_velocity_mode: 'components' или 'magnitude_angle'.
        inlet_u: Компонента скорости U на входе [м/с] (если mode='components').
        inlet_v: Компонента скорости V на входе [м/с] (если mode='components').
        inlet_speed: Модуль скорости [м/с] (если mode='magnitude_angle').
        inlet_angle: Угол атаки [градусы] (если mode='magnitude_angle').
        symmetry_type: Тип симметрии ('symmetry', 'axis_symmetry' или None).
        description: Описание набора условий.
    """
    fluid_name: str
    inlet_pressure: float
    outlet_pressure: float
    inlet_velocity_mode: str = 'components'
    inlet_u: float = 0.0
    inlet_v: float = 0.0
    inlet_speed: float = 0.0
    inlet_angle: float = 0.0
    symmetry_type: Optional[str] = None
    description: str = ""

    def __post_init__(self) -> None:
        """
        Проверка и инициализация после создания dataclass.

        Raises:
            ValueError: Если inlet_velocity_mode некорректен
        """
        if self.inlet_velocity_mode not in ['components', 'magnitude_angle']:
            raise ValueError("inlet_velocity_mode должен быть 'components' или 'magnitude_angle'")
        # Принудительно обнуляем вертикальную составляющую при симметрии
        if self.symmetry_type in ['symmetry', 'axis_symmetry']:
            if self.inlet_velocity_mode == 'components' and abs(self.inlet_v) > 1e-12:
                logger.warning("При симметрии V принудительно установлена в 0.")
                self.inlet_v = 0.0
            elif self.inlet_velocity_mode == 'magnitude_angle' and abs(self.inlet_angle) > 1e-12:
                logger.warning("При симметрии угол атаки принудительно установлен в 0.")
                self.inlet_angle = 0.0

    def to_dict(self) -> BoundaryConditionsDict:
        """Преобразует граничные условия в словарь."""
        return BoundaryConditionsDict(
            fluid_name=self.fluid_name,
            inlet_pressure=self.inlet_pressure,
            outlet_pressure=self.outlet_pressure,
            inlet_velocity_mode=self.inlet_velocity_mode,  # type: ignore
            inlet_u=self.inlet_u,
            inlet_v=self.inlet_v,
            inlet_speed=self.inlet_speed,
            inlet_angle=self.inlet_angle,
            symmetry_type=self.symmetry_type,
            description=self.description
        )

    @property
    def u(self) -> float:
        """
        Возвращает компоненту U скорости на входе.

        Returns:
            Компонента U скорости [м/с]
        """
        if self.inlet_velocity_mode == 'components':
            return self.inlet_u
        else:
            return self.inlet_speed * np.cos(np.radians(self.inlet_angle))

    @property
    def v(self) -> float:
        """
        Возвращает компоненту V скорости на входе.

        Returns:
            Компонента V скорости [м/с]
        """
        if self.inlet_velocity_mode == 'components':
            return self.inlet_v
        else:
            return self.inlet_speed * np.sin(np.radians(self.inlet_angle))

    def to_dict(self) -> BoundaryConditionsDict:
        """Преобразует граничные условия в словарь."""
        return BoundaryConditionsDict(
            fluid_name=self.fluid_name,
            inlet_pressure=self.inlet_pressure,
            outlet_pressure=self.outlet_pressure,
            inlet_velocity_mode=self.inlet_velocity_mode,  # type: ignore
            inlet_u=self.inlet_u,
            inlet_v=self.inlet_v,
            inlet_speed=self.inlet_speed,
            inlet_angle=self.inlet_angle,
            symmetry_type=self.symmetry_type,
            description=self.description
        )

    @classmethod
    def from_dict(cls, data: dict) -> 'BoundaryConditions':
        """
        Создаёт объект из словаря.

        Args:
            data: Словарь с данными граничных условий

        Returns:
            Объект BoundaryConditions
        """
        return cls(**data)

    def display(self, fluid_lib: FluidLibrary) -> str:
        """
        Возвращает строковое представление граничных условий.

        Args:
            fluid_lib: Библиотека жидкостей для получения информации о жидкости

        Returns:
            Строка с описанием граничных условий
        """
        fluid = fluid_lib.get(self.fluid_name)
        fluid_str = fluid.display() if fluid else f"Жидкость '{self.fluid_name}' (не найдена)"
        lines = [
            f"Граничные условия: {self.description}",
            f"  Жидкость: {fluid_str}",
            f"  Давление: вход {self.inlet_pressure} Па, выход {self.outlet_pressure} Па",
        ]
        if self.inlet_velocity_mode == 'components':
            lines.append(f"  Скорость на входе: U = {self.inlet_u:.3f} м/с, V = {self.inlet_v:.3f} м/с")
        else:
            lines.append(f"  Скорость на входе: |V| = {self.inlet_speed:.3f} м/с, угол = {self.inlet_angle:.2f}°")
        if self.symmetry_type:
            lines.append(f"  Симметрия: {self.symmetry_type}")
        return "\n".join(lines)


class ConditionsManager:
    """
    Менеджер для управления наборами граничных условий.

    Загружает, сохраняет и предоставляет доступ к условиям через JSON-файлы.

    Attributes:
        directory: Директория для хранения файлов условий
        conditions: Словарь с загруженными условиями
        fluid_lib: Библиотека жидкостей
    """

    def __init__(self, directory: str = "Conditions") -> None:
        """
        Инициализирует менеджер граничных условий.

        Args:
            directory: Директория для хранения файлов условий
        """
        self.directory = directory
        self.conditions: Dict[str, BoundaryConditions] = {}
        self.fluid_lib = FluidLibrary()
        os.makedirs(self.directory, exist_ok=True)
        self._load_all()

    def _load_all(self) -> None:
        """
        Загружает все условия из директории.

        Загружает JSON-файлы и создаёт объекты BoundaryConditions.
        """
        self.conditions.clear()
        if not os.path.exists(self.directory):
            return
        for filename in os.listdir(self.directory):
            if filename.endswith(".json"):
                path = os.path.join(self.directory, filename)
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    if 'fluid_name' not in data:
                        logger.error(f"Файл {filename} устаревший (нет fluid_name). Пропускаю.")
                        continue
                    cond = BoundaryConditions.from_dict(data)
                    if not self.fluid_lib.get(cond.fluid_name):
                        logger.warning(f"Жидкость '{cond.fluid_name}' не найдена в библиотеке для условий {filename}")
                    key = filename[:-5]
                    self.conditions[key] = cond
                    logger.info(f"Загружены условия: {key}")
                except Exception as e:
                    logger.error(f"Ошибка загрузки {filename}: {e}")

    def save(self, cond: BoundaryConditions, name: str) -> None:
        """
        Сохраняет граничные условия в JSON файл.

        Args:
            cond: Объект граничных условий
            name: Имя для сохранения
        """
        from utils import save_json
        save_json(cond.to_dict(), self.directory, name)
        self.conditions[name] = cond
        logger.info(f"Сохранены условия: {name}")

    def delete(self, name: str) -> None:
        """
        Удаляет условия из библиотеки и файловой системы.

        Args:
            name: Имя условий для удаления
        """
        if name in self.conditions:
            for filename in os.listdir(self.directory):
                if filename.endswith(".json") and name in filename:
                    os.remove(os.path.join(self.directory, filename))
                    break
            del self.conditions[name]
            logger.info(f"Удалены условия: {name}")

    def list_names(self) -> List[str]:
        """
        Возвращает список имён условий.

        Returns:
            Список имён сохранённых условий
        """
        return list(self.conditions.keys())

    def get(self, name: str) -> Optional[BoundaryConditions]:
        """
        Возвращает условия по имени.

        Args:
            name: Имя условий

        Returns:
            Объект BoundaryConditions или None
        """
        return self.conditions.get(name)

    def create_defaults(self) -> None:
        """
        Создаёт примеры наборов граничных условий.

        Использует первую доступную жидкость из библиотеки.
        """
        fluid_names = self.fluid_lib.list_fluids()
        if not fluid_names:
            self.fluid_lib.create_default_library()
            fluid_names = self.fluid_lib.list_fluids()
        if not fluid_names:
            logger.error("Нет доступных жидкостей для создания примеров.")
            return
        default_fluid = fluid_names[0]

        defaults = [
            BoundaryConditions(
                fluid_name=default_fluid,
                inlet_pressure=101325.0,
                outlet_pressure=101325.0,
                inlet_velocity_mode='components',
                inlet_u=10.0,
                inlet_v=0.0,
                description="Обтекание при нулевом угле атаки, U=10 м/с"
            ),
            BoundaryConditions(
                fluid_name=default_fluid,
                inlet_pressure=101325.0,
                outlet_pressure=101325.0,
                inlet_velocity_mode='magnitude_angle',
                inlet_speed=20.0,
                inlet_angle=5.0,
                description="Обтекание под углом 5°, скорость 20 м/с"
            ),
            BoundaryConditions(
                fluid_name=default_fluid,
                inlet_pressure=101325.0,
                outlet_pressure=95000.0,
                inlet_velocity_mode='components',
                inlet_u=15.0,
                inlet_v=0.0,
                symmetry_type='symmetry',
                description="Симметричное обтекание с перепадом давления"
            ),
        ]
        for i, cond in enumerate(defaults, 1):
            self.save(cond, f"Пример_{i}")


def select_fluid(fluid_lib: FluidLibrary) -> Optional[str]:
    """
    Интерактивный выбор жидкости из библиотеки.

    Args:
        fluid_lib: Библиотека жидкостей

    Returns:
        Название выбранной жидкости или None
    """
    fluids = fluid_lib.list_fluids()
    if not fluids:
        print("Библиотека жидкостей пуста. Сначала создайте жидкость в модуле Fluids.")
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
            else:
                print("Неверный номер.")
        except ValueError:
            print("Введите число.")


def conditions_menu() -> None:
    """
    Интерактивное меню для управления граничными условиями.

    Example:
        >>> if __name__ == "__main__":
        ...     conditions_menu()
    """
    manager = ConditionsManager()

    if not manager.list_names():
        print("\n📂 Библиотека условий пуста.")
        choice = input("Создать примеры условий? (Y/n): ").strip().lower()
        if choice == '' or choice == 'y' or choice == 'yes':
            manager.create_defaults()
            print("✅ Примеры созданы.")

    while True:
        print("\n" + "="*60)
        print("ГРАНИЧНЫЕ УСЛОВИЯ (CONDITIONS)".center(60))
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
                    print(f"   {cond.display(manager.fluid_lib)}")
                    print()
        elif choice == '2':
            create_conditions(manager)
        elif choice == '3':
            edit_conditions(manager)
        elif choice == '4':
            delete_conditions(manager)
        elif choice == '5':
            confirm = input("Это удалит все текущие условия и создаст примеры. Продолжить? (y/N): ").strip().lower()
            if confirm in ['y', 'yes']:
                for f in os.listdir(manager.directory):
                    os.remove(os.path.join(manager.directory, f))
                manager.create_defaults()
                print("✅ Примеры созданы.")
            else:
                print("Отменено.")
        else:
            print("Неверный выбор.")


def create_conditions(manager: ConditionsManager) -> None:
    """
    Создаёт новые граничные условия через интерактивный ввод.

    Args:
        manager: Менеджер условий для сохранения
    """
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

        # Симметрия
        symm = input("Есть симметрия? (symmetry/axis_symmetry/нет) [нет]: ").strip().lower()
        if symm in ['symmetry', 'axis_symmetry']:
            symmetry_type = symm
            print(f"Выбрана симметрия: {symmetry_type}")
        else:
            symmetry_type = None

        # Режим ввода скорости
        print("Режим ввода скорости на входе:")
        print("   1 - компоненты U, V")
        print("   2 - модуль скорости и угол атаки")
        mode_choice = input("Выберите (1/2): ").strip()
        if mode_choice == '1':
            mode = 'components'
            u = float(input("U [м/с]: "))
            if symmetry_type:
                v = 0.0
                print("V автоматически установлена в 0 (симметрия)")
            else:
                v = float(input("V [м/с]: "))
            cond = BoundaryConditions(
                fluid_name=fluid_name,
                inlet_pressure=inlet_p,
                outlet_pressure=outlet_p,
                inlet_velocity_mode=mode,
                inlet_u=u,
                inlet_v=v,
                symmetry_type=symmetry_type
            )
        elif mode_choice == '2':
            mode = 'magnitude_angle'
            speed = float(input("Модуль скорости [м/с]: "))
            if symmetry_type:
                angle = 0.0
                print("Угол атаки автоматически установлен в 0 (симметрия)")
            else:
                angle = float(input("Угол атаки [градусы]: "))
            cond = BoundaryConditions(
                fluid_name=fluid_name,
                inlet_pressure=inlet_p,
                outlet_pressure=outlet_p,
                inlet_velocity_mode=mode,
                inlet_speed=speed,
                inlet_angle=angle,
                symmetry_type=symmetry_type
            )
        else:
            print("Неверный выбор режима.")
            return

        desc = input("Описание (необязательно): ").strip()
        cond.description = desc
        manager.save(cond, name)
        print(f"✅ Условия '{name}' сохранены.")
    except ValueError as e:
        print(f"❌ Ошибка ввода: {e}")


def _get_symmetry_selection(current_sym: Optional[str]) -> Optional[str]:
    """
    Запрашивает тип симметрии у пользователя.

    Args:
        current_sym: Текущий тип симметрии

    Returns:
        Выбранный тип симметрии или None
    """
    current_display = current_sym if current_sym else "нет"
    symm_input = input(f"Симметрия (symmetry/axis_symmetry/нет) [{current_display}]: ").strip().lower()
    
    if symm_input in ['symmetry', 'axis_symmetry']:
        return symm_input
    elif symm_input == '':
        return current_sym
    else:
        return None


def _get_velocity_input_components(symmetry_type: Optional[str]) -> Tuple[float, float]:
    """
    Запрашивает скорость в режиме компонент (U, V).

    Args:
        symmetry_type: Тип симметрии (для обнуления V)

    Returns:
        Кортеж (U, V)
    """
    u = safe_input_number("U [м/с]: ", allow_empty=False)
    
    if symmetry_type:
        v = 0.0
        print("V автоматически 0 (симметрия)")
    else:
        v = safe_input_number("V [м/с]: ", allow_empty=False)
    
    return u, v


def _get_velocity_input_magnitude_angle(symmetry_type: Optional[str]) -> Tuple[float, float]:
    """
    Запрашивает скорость в режиме модуль+угол.

    Args:
        symmetry_type: Тип симметрии (для обнуления угла)

    Returns:
        Кортеж (скорость, угол)
    """
    speed = safe_input_number("Модуль скорости [м/с]: ", allow_empty=False)
    
    if symmetry_type:
        angle = 0.0
        print("Угол=0 (симметрия)")
    else:
        angle = safe_input_number("Угол атаки [градусы]: ", allow_empty=False)
    
    return speed, angle


def _update_velocity_data(
    cond: BoundaryConditions,
    new_fluid: str,
    new_inlet_p: float,
    new_outlet_p: float,
    new_sym: Optional[str],
    new_desc: str
) -> BoundaryConditions:
    """
    Обновляет данные о скорости в граничных условиях.

    Args:
        cond: Текущие условия
        new_fluid: Новая жидкость
        new_inlet_p: Давление на входе
        new_outlet_p: Давление на выходе
        new_sym: Тип симметрии
        new_desc: Описание

    Returns:
        Обновлённые граничные условия
    """
    print("Текущий режим скорости: ", "компоненты" if cond.inlet_velocity_mode == 'components' else "скорость+угол")
    change_mode = input("Изменить режим? (y/N): ").strip().lower()
    
    # Определяем новый режим
    if change_mode in ['y', 'yes']:
        new_mode = 'magnitude_angle' if cond.inlet_velocity_mode == 'components' else 'components'
        print("Ввод в режиме", "скорость+угол" if new_mode == 'magnitude_angle' else "компоненты U, V")
    else:
        new_mode = cond.inlet_velocity_mode
    
    # Получаем данные в зависимости от режима
    if new_mode == 'magnitude_angle':
        speed, angle = _get_velocity_input_magnitude_angle(new_sym)
        return BoundaryConditions(
            fluid_name=new_fluid,
            inlet_pressure=new_inlet_p,
            outlet_pressure=new_outlet_p,
            inlet_velocity_mode=new_mode,
            inlet_speed=speed,
            inlet_angle=angle if not new_sym else 0.0,
            symmetry_type=new_sym,
            description=new_desc
        )
    else:  # components
        u, v = _get_velocity_input_components(new_sym)
        return BoundaryConditions(
            fluid_name=new_fluid,
            inlet_pressure=new_inlet_p,
            outlet_pressure=new_outlet_p,
            inlet_velocity_mode=new_mode,
            inlet_u=u,
            inlet_v=v if not new_sym else 0.0,
            symmetry_type=new_sym,
            description=new_desc
        )


def edit_conditions(manager: ConditionsManager) -> None:
    """
    Редактирование набора граничных условий.

    Args:
        manager: Менеджер условий
    """
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
        print(f"\nРедактирование '{old_name}'. Оставьте поле пустым, чтобы оставить без изменений.\n")

        # Смена жидкости
        print(f"Текущая жидкость: {cond.fluid_name}")
        change_fluid = input("Сменить жидкость? (y/N): ").strip().lower()
        if change_fluid in ['y', 'yes']:
            new_fluid = select_fluid(manager.fluid_lib)
            if new_fluid is None:
                new_fluid = cond.fluid_name
        else:
            new_fluid = cond.fluid_name

        # Новые названия и давления
        new_name = input_with_default("Новое название", old_name)
        new_inlet_p = safe_input_number("Давление на входе [Па]: ", default=cond.inlet_pressure)
        new_outlet_p = safe_input_number("Давление на выходе [Па]: ", default=cond.outlet_pressure)

        # Симметрия
        new_sym = _get_symmetry_selection(cond.symmetry_type)

        # Обновление данных о скорости
        updated = _update_velocity_data(cond, new_fluid, new_inlet_p, new_outlet_p, new_sym, cond.description)

        # Описание
        new_desc = input_with_default("Описание", cond.description)
        updated.description = new_desc

        # Сохранение
        if new_name and new_name != old_name:
            manager.delete(old_name)
            manager.save(updated, new_name)
        else:
            manager.save(updated, old_name)
        
        print(f"✅ Условия обновлены.")
        
    except ValueError as e:
        print(f"❌ Ошибка ввода: {e}")


def delete_conditions(manager: ConditionsManager) -> None:
    """
    Удаление набора граничных условий.

    Args:
        manager: Менеджер условий
    """
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
        if confirm in ['y', 'yes']:
            manager.delete(name)
            print(f"✅ Удалено.")
        else:
            print("Отменено.")
    except ValueError:
        print("❌ Неверный ввод.")


if __name__ == "__main__":
    conditions_menu()