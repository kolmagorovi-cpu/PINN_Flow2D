#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Реестр консольных модулей Airfoil PINN.

Главное меню не должно хранить в себе сведения о расчетных модулях. Этот файл
явно описывает доступные блоки программы и связывает человекочитаемое имя с
импортируемой функцией старого или нового модуля.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModuleDescriptor:
    """
    Описание пункта меню и команды CLI.

    ``module`` и ``function`` пока указывают на старые модули верхнего уровня.
    По мере переноса кода в пакет здесь будут меняться только пути импорта, а
    меню и командный интерфейс останутся стабильными.
    """

    code: str
    command: str
    name: str
    module: str
    function: str
    status: str
    description: str

    def as_legacy_dict(self) -> dict[str, str]:
        """
        Возвращает словарь в старом формате.

        Такой адаптер нужен на переходном этапе, пока часть функций меню еще
        удобно работает со словарями.
        """
        return {
            "code": self.code,
            "command": self.command,
            "name": self.name,
            "module": self.module,
            "func": self.function,
            "status": self.status,
            "description": self.description,
        }


def default_modules() -> tuple[ModuleDescriptor, ...]:
    """Возвращает список модулей в порядке отображения в главном меню."""
    return (
        ModuleDescriptor(
            code="G2D",
            command="geometry",
            name="Геометрия (2D)",
            module="airfoil_pinn.geometry.cli",
            function="g2d",
            status="готов",
            description="Загрузка контура профиля и построение расчетной области",
        ),
        ModuleDescriptor(
            code="S2DR",
            command="sampling",
            name="Дискретизация",
            module="Collocation",
            function="S2DR",
            status="готов",
            description="Генерация точек коллокации с классификацией границ",
        ),
        ModuleDescriptor(
            code="Fluids",
            command="fluids",
            name="Свойства жидкости",
            module="Fluids",
            function="fluids_menu",
            status="готов",
            description="Задание плотности и вязкости",
        ),
        ModuleDescriptor(
            code="Conditions",
            command="conditions",
            name="Граничные условия",
            module="Conditions",
            function="conditions_menu",
            status="готов",
            description="Давление на входе/выходе, скорость на входе",
        ),
        ModuleDescriptor(
            code="Initializer",
            command="init",
            name="Инициализация",
            module="Initializer",
            function="initialize_menu",
            status="готов",
            description="Нормализация данных и вычисление SDF",
        ),
        ModuleDescriptor(
            code="PINNLearning",
            command="train",
            name="Обучение PINN",
            module="PINN_Learning",
            function="main_menu",
            status="готов",
            description="Настройка конфигурации и запуск обучения",
        ),
    )


def modules_as_legacy_dicts() -> list[dict[str, str]]:
    """Возвращает реестр в формате, совместимом со старым меню."""
    return [module.as_legacy_dict() for module in default_modules()]


def find_by_command(command: str) -> ModuleDescriptor | None:
    """Ищет модуль по имени команды пакетного CLI."""
    normalized = command.strip().lower()
    for module in default_modules():
        if module.command == normalized:
            return module
    return None
