from pipe_bend_pinn.app.paths import (
    PROJECT_ROOT, GEOMETRY_DOMAINS_DIR, GEOMETRY_METADATA_DIR,
    SAMPLING_COLLOCATION_DIR, INITIAL_FIELDS_DIR, INITIAL_PLOTS_DIR,
    FLUIDS_DIR, CONDITIONS_DIR, EXPERIMENTS_DIR, GEOMS_LIB_PATH,
    CONFIGS_DIR, GEOM_CONFIG_PATH,
)

"""
Модуль управления параметрами жидкостей для Pipe-Bend PINN.

Автор: Адаптировано из Airfoil PINN
Версия: 1.0
"""

import os
import json
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class Fluid:
    """Класс, описывающий жидкость/газ."""
    name: str
    density: float
    kinematic_viscosity: float
    dynamic_viscosity: Optional[float] = None
    description: str = ""

    def __post_init__(self) -> None:
        if self.dynamic_viscosity is None:
            self.dynamic_viscosity = self.kinematic_viscosity * self.density

    def to_dict(self) -> dict:
        return dict(
            name=self.name,
            density=self.density,
            kinematic_viscosity=self.kinematic_viscosity,
            dynamic_viscosity=self.dynamic_viscosity,
            description=self.description
        )

    @classmethod
    def from_dict(cls, data: dict) -> 'Fluid':
        return cls(**data)

    def display(self) -> str:
        return (f"{self.name}: ρ={self.density:.3f} кг/м³, "
                f"ν={self.kinematic_viscosity:.3e} м²/с, "
                f"μ={self.dynamic_viscosity:.3e} Па·с")


class FluidLibrary:
    """Библиотека жидкостей, загружаемых из JSON-файлов."""

    def __init__(self, directory: str = FLUIDS_DIR) -> None:
        self.directory = directory
        self.fluids: Dict[str, Fluid] = {}
        os.makedirs(self.directory, exist_ok=True)
        self._load_all()

    def _load_all(self) -> None:
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

    def save(self, fluid: Fluid, filename: Optional[str] = None) -> None:
        if filename is None:
            filename = fluid.name.replace(" ", "_").replace("/", "_") + ".json"
        path = os.path.join(self.directory, filename)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(fluid.to_dict(), f, indent=4, ensure_ascii=False)
        self.fluids[fluid.name] = fluid
        logger.info(f"Сохранена жидкость: {fluid.name}")

    def delete(self, name: str) -> None:
        if name in self.fluids:
            for filename in os.listdir(self.directory):
                if filename.endswith(".json") and name in filename:
                    os.remove(os.path.join(self.directory, filename))
                    break
            del self.fluids[name]

    def list_fluids(self) -> List[str]:
        return list(self.fluids.keys())

    def get(self, name: str) -> Optional[Fluid]:
        return self.fluids.get(name)

    def create_default_library(self) -> None:
        defaults = [
            Fluid(name="Вода (20°C)", density=998.0, kinematic_viscosity=1.004e-6,
                  description="Пресная вода при 20°C"),
            Fluid(name="Воздух (20°C)", density=1.204, kinematic_viscosity=1.516e-5,
                  description="Сухой воздух при 20°C"),
            Fluid(name="Масло гидравлическое", density=870.0, kinematic_viscosity=1e-4,
                  description="Типичное гидравлическое масло"),
            Fluid(name="Глицерин (20°C)", density=1260.0, kinematic_viscosity=1.185e-3,
                  description="Чистый глицерин при 20°C"),
        ]
        for fl in defaults:
            self.save(fl)


def fluids_menu() -> None:
    """Интерактивное меню для управления жидкостями."""
    lib = FluidLibrary()
    if not lib.list_fluids():
        print("\n📂 Библиотека жидкостей пуста.")
        choice = input("Создать примеры жидкостей? (Y/n): ").strip().lower()
        if choice == '' or choice in ('y', 'yes'):
            lib.create_default_library()
            print("✅ Примеры созданы.")

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
            name = input("Название жидкости: ").strip()
            if not name:
                print("Название не может быть пустым.")
                continue
            try:
                density = float(input("Плотность [кг/м³]: "))
                kin_visc = float(input("Кинематическая вязкость [м²/с]: "))
                dyn_visc_input = input("Динамическая вязкость [Па·с] (Enter - авто): ").strip()
                dyn_visc = float(dyn_visc_input) if dyn_visc_input else None
                desc = input("Описание: ").strip()
                fluid = Fluid(name=name, density=density, kinematic_viscosity=kin_visc,
                             dynamic_viscosity=dyn_visc, description=desc)
                lib.save(fluid)
                print(f"✅ Жидкость '{name}' сохранена.")
            except ValueError as e:
                print(f"❌ Ошибка ввода: {e}")
        elif choice == '3':
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
                print(f"Редактирование '{old_name}'. Enter - без изменений.")
                new_name = input(f"Название [{fluid.name}]: ").strip()
                new_density = input(f"Плотность [{fluid.density}]: ").strip()
                new_kin = input(f"Кин. вязкость [{fluid.kinematic_viscosity}]: ").strip()
                new_dyn = input(f"Дин. вязкость [{fluid.dynamic_viscosity}]: ").strip()
                new_desc = input(f"Описание [{fluid.description}]: ").strip()

                updated = Fluid(
                    name=new_name if new_name else fluid.name,
                    density=float(new_density) if new_density else fluid.density,
                    kinematic_viscosity=float(new_kin) if new_kin else fluid.kinematic_viscosity,
                    dynamic_viscosity=float(new_dyn) if new_dyn else fluid.dynamic_viscosity,
                    description=new_desc if new_desc else fluid.description
                )
                if updated.name != fluid.name:
                    lib.delete(fluid.name)
                lib.save(updated)
                print(f"✅ Жидкость обновлена.")
            except ValueError as e:
                print(f"❌ Ошибка ввода: {e}")
        elif choice == '4':
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
                if confirm in ('y', 'yes'):
                    lib.delete(name)
                    print(f"✅ Удалено.")
                else:
                    print("Отменено.")
            except ValueError:
                print("❌ Неверный ввод.")
        elif choice == '5':
            confirm = input("Это удалит все текущие жидкости. Продолжить? (y/N): ").strip().lower()
            if confirm in ('y', 'yes'):
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