#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Утилиты для пользовательского интерфейса.

Этот модуль содержит общие функции для работы с консольным меню,
устраняя дублирование кода в различных модулях проекта.

Example:
    >>> from utils import get_menu_choice, safe_input_number
    >>> choice = get_menu_choice("Выберите действие", valid_range=(1, 5))
    >>> value = safe_input_number("Введите число: ", default=0.0)
"""

import sys
import os
from typing import Optional, Tuple, Union, List, Dict, Callable, Any
from abc import ABC, abstractmethod

from constants import (
    MENU_LINE_WIDTH,
    MENU_SEPARATOR_CHAR,
    IS_WINDOWS,
)


def clear_screen() -> None:
    """
    Очищает экран консоли.
    
    Работает на Windows и Unix-системах.
    """
    os.system('cls' if IS_WINDOWS else 'clear')
    print('\033c', end='')


def print_separator(char: str = MENU_SEPARATOR_CHAR, width: int = MENU_LINE_WIDTH) -> None:
    """
    Выводит строку-разделитель.
    
    Args:
        char: Символ разделителя
        width: Ширина разделителя
    """
    print(char * width)


def get_menu_choice(
    prompt: str = "Выберите действие",
    valid_range: Optional[Tuple[int, int]] = None,
    allow_zero: bool = True,
    zero_action: str = "Назад"
) -> str:
    """
    Запрашивает выбор пользователя с валидацией.
    
    Args:
        prompt: Текст приглашения к вводу
        valid_range: Диапазон допустимых значений (min, max)
        allow_zero: Разрешить ввод 0 для выхода
        zero_action: Описание действия для 0
        
    Returns:
        Строка с выбором пользователя
        
    Example:
        >>> choice = get_menu_choice("Выберите модуль", valid_range=(1, 6))
        >>> choice = get_menu_choice("Действие", allow_zero=True, zero_action="Выход")
    """
    while True:
        try:
            if allow_zero:
                print(f"\n{prompt} (0 - {zero_action}): ", end="")
            else:
                print(f"\n{prompt}: ", end="")
            
            choice = input().strip()
            
            if choice == "0" and allow_zero:
                return choice
            
            if valid_range:
                choice_int = int(choice)
                min_val, max_val = valid_range
                if min_val <= choice_int <= max_val:
                    return choice
                else:
                    print(f"❌ Введите число от {min_val} до {max_val}")
                    continue
            else:
                return choice
                
        except ValueError:
            print("❌ Ошибка: введите число!")
        except KeyboardInterrupt:
            print("\n⚠️  Ввод прерван")
            return "0"


def safe_input_number(
    prompt: str,
    default: Optional[float] = None,
    min_value: Optional[float] = None,
    max_value: Optional[float] = None,
    allow_empty: bool = True
) -> Optional[float]:
    """
    Безопасный ввод числа с проверкой диапазона.
    
    Args:
        prompt: Текст приглашения к вводу
        default: Значение по умолчанию (если пустой ввод)
        min_value: Минимальное допустимое значение
        max_value: Максимальное допустимое значение
        allow_empty: Разрешить пустой ввод (использовать default)
        
    Returns:
        Введённое число или None
        
    Example:
        >>> speed = safe_input_number("Скорость [м/с]: ", default=10.0)
        >>> angle = safe_input_number("Угол [град]: ", min_value=-90, max_value=90)
    """
    while True:
        try:
            display_prompt = prompt
            if default is not None and allow_empty:
                display_prompt = f"{prompt} [{default}]: "
            elif default is not None:
                display_prompt = f"{prompt} [{default}]: "
            else:
                display_prompt = prompt
            
            user_input = input(display_prompt).strip()
            
            if user_input == "":
                if allow_empty and default is not None:
                    return default
                elif not allow_empty:
                    print("❌ Введите значение")
                    continue
                else:
                    return None
            
            value = float(user_input)
            
            if min_value is not None and value < min_value:
                print(f"❌ Значение должно быть не меньше {min_value}")
                continue
                
            if max_value is not None and value > max_value:
                print(f"❌ Значение должно быть не больше {max_value}")
                continue
            
            return value
            
        except ValueError:
            print("❌ Ошибка: введите число!")
        except KeyboardInterrupt:
            print("\n⚠️  Ввод прерван")
            return default


def confirm_action(prompt: str, default: bool = False) -> bool:
    """
    Запрашивает подтверждение действия (yes/no).
    
    Args:
        prompt: Текст вопроса
        default: Значение по умолчанию (True = yes, False = no)
        
    Returns:
        True если подтверждено, False если отменено
        
    Example:
        >>> if confirm_action("Удалить файл?", default=False):
        ...     os.remove(filename)
    """
    default_str = "[Y/n]" if default else "[y/N]"
    
    while True:
        try:
            response = input(f"{prompt} {default_str}: ").strip().lower()
            
            if response == "":
                return default
            
            if response in ['y', 'yes', 'да', 'д']:
                return True
            elif response in ['n', 'no', 'нет', 'н']:
                return False
            else:
                print("❌ Введите 'y' (да) или 'n' (нет)")
                
        except KeyboardInterrupt:
            print("\n⚠️  Ввод прерван")
            return False


def input_with_default(
    prompt: str,
    default: str,
    required: bool = False
) -> str:
    """
    Ввод строки с значением по умолчанию.
    
    Args:
        prompt: Текст приглашения к вводу
        default: Значение по умолчанию
        required: Обязательно ли значение (не разрешать пустой ввод)
        
    Returns:
        Введённая строка или значение по умолчанию
        
    Example:
        >>> name = input_with_default("Имя проекта: ", "default_project")
        >>> filename = input_with_default("Файл: ", "config.yaml", required=True)
    """
    while True:
        try:
            user_input = input(f"{prompt} [{default}]: ").strip()
            
            if user_input == "":
                if required:
                    print("❌ Введите значение")
                    continue
                return default
            
            return user_input
            
        except KeyboardInterrupt:
            print("\n⚠️  Ввод прерван")
            return default


def pause_and_continue(message: str = "Нажмите Enter для продолжения...") -> None:
    """
    Приостанавливает выполнение до нажатия Enter.
    
    Args:
        message: Сообщение для вывода
    """
    try:
        input(f"\n{message}")
    except KeyboardInterrupt:
        pass


def print_header(title: str, width: int = MENU_LINE_WIDTH) -> None:
    """
    Выводит заголовок раздела.
    
    Args:
        title: Текст заголовка
        width: Ширина заголовка
    """
    print(f"\n{'=' * width}")
    print(f"{title:^{width}}")
    print(f"{'=' * width}\n")


def print_list(
    items: List[Union[str, dict]],
    start_index: int = 1,
    show_index: bool = True
) -> None:
    """
    Выводит нумерованный список.
    
    Args:
        items: Список элементов для вывода
        start_index: Начальный индекс нумерации
        show_index: Показывать ли индексы
        
    Example:
        >>> print_list(["Элемент 1", "Элемент 2"])
        >>> print_list([{"name": "Модуль 1", "status": "готов"}], show_index=True)
    """
    for i, item in enumerate(items, start_index):
        if isinstance(item, dict):
            if 'name' in item:
                name = item['name']
                if show_index:
                    print(f"  {i}. {name}")
                else:
                    print(f"  {name}")
                
                if 'description' in item:
                    print(f"     ↳ {item['description']}")
                if 'status' in item:
                    print(f"     Статус: {item['status']}")
        else:
            if show_index:
                print(f"  {i}. {item}")
            else:
                print(f"  {item}")


class BaseMenu(ABC):
    """
    Базовый класс для консольных меню.
    
    Предоставляет основную логику отображения и обработки выбора,
    устраняя дублирование кода в различных меню проекта.
    
    Example:
        >>> class SettingsMenu(BaseMenu):
        ...     def __init__(self):
        ...         super().__init__("Настройки", [
        ...             {"id": "color", "label": "Цветной вывод"},
        ...             {"id": "verbose", "label": "Подробный вывод"},
        ...         ])
        ...     
        ...     def handle_choice(self, choice: str) -> bool:
        ...         if choice == "1":
        ...             self.toggle_color()
        ...         return True
        >>> 
        >>> menu = SettingsMenu()
        >>> menu.run()
    """
    
    def __init__(
        self,
        title: str,
        items: List[Union[str, Dict[str, str]]],
        show_zero_exit: bool = True,
        zero_exit_label: str = "Назад"
    ):
        """
        Инициализирует меню.
        
        Args:
            title: Заголовок меню
            items: Элементы меню (строки или словари с label/description)
            show_zero_exit: Показывать ли пункт 0 для выхода
            zero_exit_label: Текст для пункта выхода
        """
        self.title = title
        self.items = items
        self.show_zero_exit = show_zero_exit
        self.zero_exit_label = zero_exit_label
        self.running = False
    
    def display(self) -> None:
        """Отображает меню."""
        clear_screen()
        print_header(self.title)
        
        for i, item in enumerate(self.items, 1):
            if isinstance(item, dict):
                label = item.get('label', item.get('name', f'Пункт {i}'))
                description = item.get('description', '')
                
                print(f"  {Colors.BOLD}[{i}]{Colors.RESET} {label}")
                if description:
                    print(f"      {Colors.CYAN}↳ {description}{Colors.RESET}")
            else:
                print(f"  {Colors.BOLD}[{i}]{Colors.RESET} {item}")
        
        if self.show_zero_exit:
            print(f"  {Colors.BOLD}{Colors.RED}[0]{Colors.RESET} {Colors.RED}{self.zero_exit_label}{Colors.RESET}")
        
        print()
    
    def get_choice(self) -> str:
        """
        Запрашивает выбор пользователя.
        
        Returns:
            Строка с выбором пользователя
        """
        max_idx = len(self.items)
        prompt = f"Выберите действие (0-{max_idx})" if self.show_zero_exit else f"Выберите действие (1-{max_idx})"
        return get_menu_choice(prompt, valid_range=(0 if self.show_zero_exit else 1, max_idx), allow_zero=self.show_zero_exit)
    
    @abstractmethod
    def handle_choice(self, choice: str) -> bool:
        """
        Обрабатывает выбор пользователя.
        
        Args:
            choice: Выбор пользователя (номер пункта)
            
        Returns:
            True если продолжить выполнение меню, False для выхода
        """
        pass
    
    def before_loop(self) -> None:
        """Вызывается перед основным циклом меню."""
        pass
    
    def after_loop(self) -> None:
        """Вызывается после выхода из цикла меню."""
        pass
    
    def run(self) -> Optional[str]:
        """
        Запускает цикл меню.
        
        Returns:
            Результат выполнения (если есть)
        """
        self.running = True
        self.before_loop()
        
        try:
            while self.running:
                self.display()
                choice = self.get_choice()
                
                if choice == "0" and self.show_zero_exit:
                    self.running = False
                    break
                
                if not self.handle_choice(choice):
                    self.running = False
                    break
        except KeyboardInterrupt:
            print("\n⚠️  Меню прервано")
            self.running = False
        finally:
            self.after_loop()
        
        return None
    
    def stop(self) -> None:
        """Останавливает цикл меню."""
        self.running = False


# Для совместимости импортируем Colors из Airfoil_PINN если нужно
class Colors:
    """Цвета для консольного вывода (совместимость)."""
    GREEN = '\033[92m'
    BLUE = '\033[94m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    CYAN = '\033[96m'
    MAGENTA = '\033[95m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    RESET = '\033[0m'

    if IS_WINDOWS:
        GREEN = BLUE = YELLOW = RED = CYAN = MAGENTA = BOLD = UNDERLINE = RESET = ''
