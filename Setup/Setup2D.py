import numpy as np
import warnings
import os
import json
from scipy.spatial import distance, KDTree
import matplotlib.pyplot as plt
import time

class Fluid:
    """
    Класс с описанием свойств жидкости/газа
    """
    def __init__(self, 
                 name, 
                 density, 
                 dynamic_viscosity, 
                 kinematic_viscosity, 
                 temperature, 
                 pressure):
        """
        Конструктор класса свойств жидкости (по умолячанию воздух).
        
        Аргументы:
            name: название жидкости/газа (по умолчанию "Air" - воздух)
            density: плотность (кг/м³) - определяет инерционные свойства
            dynamic_viscosity: динамическая вязкость (Па·с) - определяет внутреннее трение
            kinematic_viscosity: кинематическая вязкость (м²/с) - если не задана, вычисляется
            temperature: температура (К) - влияет на свойства для сжимаемых течений
            pressure: давление (Па) - базовое давление в области
        """
        self.name = name; self.density = density; self.dynamic_viscosity = dynamic_viscosity
        self.kinematic_viscosity = kinematic_viscosity; self.temperature = temperature
        self.pressure = pressure;


    def get_properties(filename = "C:/Users/Ivan/source/repos/PINN_Flow2D/Setup/Data/fluids.json", fluid_key = None):
        try:
            with open(filename, 'r', encoding='utf-8') as file:
                data = json.load(file)
            print(f"Выберите газ/жидкость: ")
            print(r"""
            0) Указать путь к пользовательской жидкости;
            1) Air20C;
            2) Water;
            3) Oil.
             
            """)
            while fluid_key == None:
                fluid_map = ['user', 'air20C', 'water', 'oil']
                fluid_index = int(input("Номер жидкости "))
                try:
                    fluid_key = fluid_map[fluid_index]
                except Exception as e:
                    print(f"Ошибка при выборе")
                    fluid_key = None
            print(f"Выбрана жидкость: {fluid_key}")
        except FileNotFoundError:
            print(f"Ошибка: Файл '{filename}' не найден")
            return None
        except json.JSONDecodeError as e:
            print(f"Ошибка: Неверный формат JSON: {e}")
            return None
        except Exception as e:
            print(f"Неожиданная ошибка: {e}")
            return None
        if fluid_key in data:
            fluid_data = data[fluid_key]
            # Создаем и возвращаем объект Fluid
            return Fluid(
                name=fluid_key,
                density=fluid_data.get('density'),
                dynamic_viscosity=fluid_data.get('dynamic_viscosity'),
                kinematic_viscosity=fluid_data.get('kinematic_viscosity'),
                temperature=fluid_data.get('temperature'),
                pressure=fluid_data.get('pressure')
            )
        else:
            print(f"Ошибка: Жидкость '{fluid_key}' не найдена в базе данных")
            return None

def main():
    """
    Главная функция для тестирования класса Fluid
    """
    print("\n" + "="*60)
    print("ТЕСТИРОВАНИЕ КЛАССА Fluid")
    print("="*60)
    
    # Тест 1: Вызов get_properties
    print("\n[ТЕСТ 1] Загрузка свойств из JSON")
    print("-" * 40)
    print("Запускается метод get_properties()...")
    print("(следуйте инструкциям в консоли)")
    print("-" * 40)
    
    # Вызываем метод get_properties
    result = Fluid.get_properties()
    print ("Резульатат загрузки: ")
    print (f"""
        Жидкость: {result.name}
        Плотность: {result.density} км/м3 
        Динамическая вязкость:  {result.dynamic_viscosity} Па*с
        Кинематическая вязкость: {result.kinematic_viscosity} м*с2
        Температура : {result.temperature} К 
        Давление: {result.pressure} Па

    
    """)
    
    # Тест 2: Создание объекта Fluid вручную
    print("\n[ТЕСТ 2] Создание объекта Fluid вручную")
    print("-" * 40)
    
    # Создаем объект жидкости
    water = Fluid(
        name="Water",
        density=998,
        dynamic_viscosity=0.001002,
        kinematic_viscosity=1.004e-6,
        temperature=293.15,
        pressure=101325
    )
    
    # Выводим свойства
    print("Свойства воды:")
    print(f"  Название: {water.name}")
    print(f"  Плотность: {water.density} кг/м³")
    print(f"  Динамическая вязкость: {water.dynamic_viscosity} Па·с")
    print(f"  Кинематическая вязкость: {water.kinematic_viscosity} м²/с")
    print(f"  Температура: {water.temperature} K")
    print(f"  Давление: {water.pressure} Па")
  
    
    # Тест 3: Создание газа
    print("\n[ТЕСТ 3] Создание объекта Fluid (воздух)")
    print("-" * 40)
    
    air = Fluid(
        name="Air",
        density=1.204,
        dynamic_viscosity=1.82e-5,
        kinematic_viscosity=1.51e-5,
        temperature=293.15,
        pressure=101325
    )
    
    print("Свойства воздуха:")
    print(f"  Название: {air.name}")
    print(f"  Плотность: {air.density} кг/м³")
    print(f"  Динамическая вязкость: {air.dynamic_viscosity} Па·с")
    print(f"  Кинематическая вязкость: {air.kinematic_viscosity} м²/с")
    print(f"  Температура: {air.temperature} K")
    
    # Тест 4: Сравнение свойств
    print("\n[ТЕСТ 4] Сравнение свойств")
    print("-" * 40)
    
    print(f"{'Свойство':<30} {'Вода':<15} {'Воздух':<15}")
    print("-" * 60)
    print(f"{'Плотность (кг/м³)':<30} {water.density:<15.6g} {air.density:<15.6g}")
    print(f"{'Динамическая вязкость (Па·с)':<30} {water.dynamic_viscosity:<15.6g} {air.dynamic_viscosity:<15.6g}")
    print(f"{'Кинематическая вязкость (м²/с)':<30} {water.kinematic_viscosity:<15.6g} {air.kinematic_viscosity:<15.6g}")
    print(f"{'Температура (K)':<30} {water.temperature:<15.2f} {air.temperature:<15.2f}")
 
    
    # Тест 5: Проверка всех атрибутов
    print("\n[ТЕСТ 5] Проверка атрибутов объекта")
    print("-" * 40)
    
    print("Все атрибуты объекта water:")
    for attr, value in water.__dict__.items():
        print(f"  {attr}: {value}")
    
    # Тест 6: Проверка типа данных
    print("\n[ТЕСТ 6] Проверка типов данных")
    print("-" * 40)
    
    print(f"Тип name: {type(water.name)}")
    print(f"Тип density: {type(water.density)}")
    print(f"Тип dynamic_viscosity: {type(water.dynamic_viscosity)}")
    print(f"Тип temperature: {type(water.temperature)}")
    
    print("\n" + "="*60)
    print("ТЕСТИРОВАНИЕ ЗАВЕРШЕНО")
    print("="*60)


if __name__ == "__main__":
    main()