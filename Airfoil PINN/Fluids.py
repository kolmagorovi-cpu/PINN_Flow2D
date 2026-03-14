#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Модуль управления параметрами жидкостей для PINNs решателя.
Хранит данные о плотности и вязкости, сохраняет в JSON.
"""

import os
import json
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class Fluid:
    """
    Класс, описывающий жидкость/газ.
    
    Attributes:
        name (str): Название жидкости.
        density (float): Плотность [кг/м³].
        kinematic_viscosity (float): Кинематическая вязкость [м²/с].
        dynamic_viscosity (float): Динамическая вязкость [Па·с] (опционально).
        description (str): Краткое описание (температура, условия).
    """
    name: str
    density: float
    kinematic_viscosity: float
    dynamic_viscosity: Optional[float] = None
    description: str = ""

    def __post_init__(self):
        # Если динамическая вязкость не задана, вычисляем через плотность
        if self.dynamic_viscosity is None:
            self.dynamic_viscosity = self.kinematic_viscosity * self.density
        # Если задана только динамическая, можно вычислить кинематическую, но мы ожидаем обе или хотя бы одну
        # Для удобства проверим соответствие
        if abs(self.dynamic_viscosity - self.kinematic_viscosity * self.density) > 1e-12:
            logger.warning(f"Несоответствие между кинематической и динамической вязкостью для {self.name}")

    def to_dict(self) -> dict:
        """Преобразование в словарь для JSON."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> 'Fluid':
        """Создание объекта из словаря."""
        return cls(**data)

    def display(self) -> str:
        """Краткое отображение параметров."""
        return (f"{self.name}: ρ={self.density:.3f} кг/м³, "
                f"ν={self.kinematic_viscosity:.3e} м²/с, "
                f"μ={self.dynamic_viscosity:.3e} Па·с")


class FluidLibrary:
    """
    Библиотека жидкостей, загружаемых из JSON-файлов.
    """

    def __init__(self, directory: str = "Fluids"):
        self.directory = directory
        self.fluids: Dict[str, Fluid] = {}
        os.makedirs(self.directory, exist_ok=True)
        self._load_all()

    def _load_all(self):
        """Загружает все JSON-файлы из директории."""
        self.fluids.clear()
        if not os.path.exists(self.directory):
            return
        for filename in os.listdir(self.directory):
            if filename.endswith(".json"):
                path = os.path.join(self.directory, filename)
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    fluid = Fluid.from_dict(data)
                    self.fluids[fluid.name] = fluid
                    logger.info(f"Загружена жидкость: {fluid.name}")
                except Exception as e:
                    logger.error(f"Ошибка загрузки {filename}: {e}")

    def save(self, fluid: Fluid, filename: Optional[str] = None):
        """Сохраняет жидкость в JSON-файл."""
        if filename is None:
            # Используем название, заменяя недопустимые символы
            filename = fluid.name.replace(" ", "_").replace("/", "_") + ".json"
        path = os.path.join(self.directory, filename)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(fluid.to_dict(), f, indent=4, ensure_ascii=False)
        self.fluids[fluid.name] = fluid
        logger.info(f"Сохранена жидкость: {fluid.name} -> {filename}")

    def delete(self, name: str):
        """Удаляет жидкость из библиотеки и файл."""
        if name in self.fluids:
            # Удаляем файл
            for filename in os.listdir(self.directory):
                if filename.endswith(".json") and name in filename:  # простое приближение
                    os.remove(os.path.join(self.directory, filename))
                    break
            del self.fluids[name]
            logger.info(f"Удалена жидкость: {name}")
        else:
            logger.warning(f"Жидкость '{name}' не найдена")

    def list_fluids(self) -> List[str]:
        """Возвращает список названий жидкостей."""
        return list(self.fluids.keys())

    def get(self, name: str) -> Optional[Fluid]:
        """Возвращает жидкость по имени."""
        return self.fluids.get(name)

    def create_default_library(self):
        """Создаёт несколько примеров жидкостей."""
        default_fluids = [
            Fluid(
                name="Воздух (20°C)",
                density=1.204,
                kinematic_viscosity=1.516e-5,
                description="Сухой воздух при 20°C и атмосферном давлении"
            ),
            Fluid(
                name="Вода (20°C)",
                density=998.0,
                kinematic_viscosity=1.004e-6,
                description="Пресная вода при 20°C"
            ),
            Fluid(
                name="Масло гидравлическое",
                density=870.0,
                kinematic_viscosity=1e-4,
                description="Типичное гидравлическое масло"
            ),
            Fluid(
                name="Глицерин (20°C)",
                density=1260.0,
                kinematic_viscosity=1.185e-3,
                description="Чистый глицерин при 20°C"
            ),
            Fluid(
                name="Воздух (0°C)",
                density=1.293,
                kinematic_viscosity=1.33e-5,
                description="Сухой воздух при 0°C"
            ),
        ]
        for fl in default_fluids:
            self.save(fl)


def fluids_menu():
    """Интерактивное меню для управления жидкостями."""
    lib = FluidLibrary()
    # Если библиотека пуста, предложим создать примеры
    if not lib.list_fluids():
        print("\n📂 Библиотека жидкостей пуста.")
        choice = input("Создать примеры жидкостей? (Y/n): ").strip().lower()
        if choice == '' or choice == 'y' or choice == 'yes':
            lib.create_default_library()
            print("✅ Примеры созданы.")
        else:
            print("Продолжаем с пустой библиотекой.")

    while True:
        print("\n" + "="*60)
        print("УПРАВЛЕНИЕ ЖИДКОСТЯМИ".center(60))
        print("="*60)
        print("1. Показать список жидкостей")
        print("2. Добавить новую жидкость")
        print("3. Редактировать жидкость")
        print("4. Удалить жидкость")
        print("5. Создать примеры (сбросить библиотеку)")
        print("0. Вернуться в главное меню")

        choice = input("\nВыберите действие (0-5): ").strip()
        if choice == '0':
            break
        elif choice == '1':
            fluids = lib.list_fluids()
            if not fluids:
                print("Библиотека пуста.")
            else:
                print("\nДоступные жидкости:")
                for i, name in enumerate(fluids, 1):
                    fluid = lib.get(name)
                    print(f"{i}. {fluid.display()}")
        elif choice == '2':
            # Добавление новой жидкости
            name = input("Название жидкости: ").strip()
            if not name:
                print("Название не может быть пустым.")
                continue
            try:
                density = float(input("Плотность [кг/м³]: "))
                kin_visc = float(input("Кинематическая вязкость [м²/с]: "))
                # опционально динамическая
                dyn_visc_input = input("Динамическая вязкость [Па·с] (Enter для автоматического расчёта): ").strip()
                dyn_visc = float(dyn_visc_input) if dyn_visc_input else None
                desc = input("Краткое описание (необязательно): ").strip()
                fluid = Fluid(
                    name=name,
                    density=density,
                    kinematic_viscosity=kin_visc,
                    dynamic_viscosity=dyn_visc,
                    description=desc
                )
                lib.save(fluid)
                print(f"✅ Жидкость '{name}' сохранена.")
            except ValueError as e:
                print(f"❌ Ошибка ввода: {e}")
        elif choice == '3':
            # Редактирование
            fluids = lib.list_fluids()
            if not fluids:
                print("Библиотека пуста.")
                continue
            print("Выберите жидкость для редактирования:")
            for i, name in enumerate(fluids, 1):
                print(f"{i}. {name}")
            try:
                idx = int(input("Номер: ")) - 1
                if idx < 0 or idx >= len(fluids):
                    print("Неверный номер.")
                    continue
                old_name = fluids[idx]
                fluid = lib.get(old_name)
                print(f"Редактирование '{old_name}'. Оставьте поле пустым, чтобы оставить без изменений.")
                new_name = input(f"Название [{fluid.name}]: ").strip()
                new_density_str = input(f"Плотность [{fluid.density}] кг/м³: ").strip()
                new_kin_str = input(f"Кинематическая вязкость [{fluid.kinematic_viscosity}] м²/с: ").strip()
                new_dyn_str = input(f"Динамическая вязкость [{fluid.dynamic_viscosity}] Па·с: ").strip()
                new_desc = input(f"Описание [{fluid.description}]: ").strip()

                # Создаём обновлённый объект
                updated = Fluid(
                    name=new_name if new_name else fluid.name,
                    density=float(new_density_str) if new_density_str else fluid.density,
                    kinematic_viscosity=float(new_kin_str) if new_kin_str else fluid.kinematic_viscosity,
                    dynamic_viscosity=float(new_dyn_str) if new_dyn_str else fluid.dynamic_viscosity,
                    description=new_desc if new_desc else fluid.description
                )
                # Если имя изменилось, удалим старый файл
                if updated.name != fluid.name:
                    lib.delete(fluid.name)
                lib.save(updated)
                print(f"✅ Жидкость обновлена.")
            except ValueError as e:
                print(f"❌ Ошибка ввода: {e}")
        elif choice == '4':
            # Удаление
            fluids = lib.list_fluids()
            if not fluids:
                print("Библиотека пуста.")
                continue
            print("Выберите жидкость для удаления:")
            for i, name in enumerate(fluids, 1):
                print(f"{i}. {name}")
            try:
                idx = int(input("Номер: ")) - 1
                if idx < 0 or idx >= len(fluids):
                    print("Неверный номер.")
                    continue
                name = fluids[idx]
                confirm = input(f"Удалить '{name}'? (y/N): ").strip().lower()
                if confirm == 'y' or confirm == 'yes':
                    lib.delete(name)
                    print(f"✅ Удалено.")
                else:
                    print("Отменено.")
            except ValueError:
                print("❌ Неверный ввод.")
        elif choice == '5':
            # Создать примеры (очистив старую библиотеку)
            confirm = input("Это удалит все текущие жидкости и создаст примеры. Продолжить? (y/N): ").strip().lower()
            if confirm == 'y' or confirm == 'yes':
                # Очистим директорию
                for f in os.listdir(lib.directory):
                    os.remove(os.path.join(lib.directory, f))
                lib.create_default_library()
                print("✅ Библиотека примеров создана.")
            else:
                print("Отменено.")
        else:
            print("Неверный выбор.")


if __name__ == "__main__":
    fluids_menu()