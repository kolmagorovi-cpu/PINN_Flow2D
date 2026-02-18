import numpy as np
import warnings
import os
import json
from scipy.spatial import distance, KDTree
import matplotlib.pyplot as plt
import time


class FluidProperties:
    """
    Класс для хранения свойств жидкости/газа.
    Инкапсулирует все физические свойства рабочей среды для CFD расчета.
    """
    
    def __init__(self, name="Air", density=1.225, dynamic_viscosity=1.8e-5, 
                 kinematic_viscosity=None, temperature=293.15, pressure=101325.0):
        """
        Конструктор класса свойств жидкости.
        
        Аргументы:
            name: название жидкости/газа (по умолчанию "Air" - воздух)
            density: плотность (кг/м³) - определяет инерционные свойства
            dynamic_viscosity: динамическая вязкость (Па·с) - определяет внутреннее трение
            kinematic_viscosity: кинематическая вязкость (м²/с) - если не задана, вычисляется
            temperature: температура (К) - влияет на свойства для сжимаемых течений
            pressure: давление (Па) - базовое давление в области
        """
        self.name = name  # Сохраняем название жидкости
        self.density = density  # Сохраняем плотность
        self.dynamic_viscosity = dynamic_viscosity  # Сохраняем динамическую вязкость
        
        # Вычисляем кинематическую вязкость, если она не задана явно
        if kinematic_viscosity is not None:
            self.kinematic_viscosity = kinematic_viscosity
        else:
            # ν = μ / ρ - стандартная формула для несжимаемой жидкости
            self.kinematic_viscosity = dynamic_viscosity / density if density > 0 else 0
        
        self.temperature = temperature  # Сохраняем температуру
        self.pressure = pressure  # Сохраняем давление
    
    def __str__(self):
        """Строковое представление для вывода в консоль."""
        return (f"{self.name}:\n"
                f"  Плотность: {self.density:.6f} кг/м³\n"
                f"  Динамическая вязкость: {self.dynamic_viscosity:.6e} Па·с\n"
                f"  Кинематическая вязкость: {self.kinematic_viscosity:.6e} м²/с\n"
                f"  Температура: {self.temperature:.2f} К\n"
                f"  Давление: {self.pressure:.2f} Па")
    
    def to_dict(self):
        """Преобразует свойства в словарь для сохранения в файл."""
        return {
            'name': self.name,
            'density': self.density,
            'dynamic_viscosity': self.dynamic_viscosity,
            'kinematic_viscosity': self.kinematic_viscosity,
            'temperature': self.temperature,
            'pressure': self.pressure
        }
    
    @staticmethod
    def from_dict(data):
        """Создает объект из словаря (для загрузки из файла)."""
        return FluidProperties(
            name=data.get('name', 'Air'),
            density=data.get('density', 1.225),
            dynamic_viscosity=data.get('dynamic_viscosity', 1.8e-5),
            kinematic_viscosity=data.get('kinematic_viscosity'),
            temperature=data.get('temperature', 293.15),
            pressure=data.get('pressure', 101325.0)
        )


class InletBoundary:
    """
    Граничное условие на входе потока.
    Для входа обычно задаются скорость и давление.
    """
    
    def __init__(self, velocity_x=1.0, velocity_y=0.0, pressure=101325.0, 
                 turbulence_intensity=0.05, name="Вход"):
        """
        Конструктор ГУ для входа.
        
        Аргументы:
            velocity_x: компонента скорости по X (м/с)
            velocity_y: компонента скорости по Y (м/с)
            pressure: давление на входе (Па)
            turbulence_intensity: интенсивность турбулентности (0-1)
            name: название границы для отображения
        """
        self.velocity_x = velocity_x  # Сохраняем скорость по X
        self.velocity_y = velocity_y  # Сохраняем скорость по Y
        self.pressure = pressure  # Сохраняем давление
        self.turbulence_intensity = turbulence_intensity  # Сохраняем турбулентность
        self.name = name  # Сохраняем название
    
    def __str__(self):
        """Строковое представление для вывода в консоль."""
        return (f"{self.name}:\n"
                f"  Скорость: ({self.velocity_x:.3f}, {self.velocity_y:.3f}) м/с\n"
                f"  Давление: {self.pressure:.2f} Па\n"
                f"  Интенсивность турбулентности: {self.turbulence_intensity*100:.1f}%")


class OutletBoundary:
    """
    Граничное условие на выходе потока.
    Для выхода обычно задается давление (условие мягкой границы).
    """
    
    def __init__(self, pressure=101325.0, backflow_prevention=True, name="Выход"):
        """
        Конструктор ГУ для выхода.
        
        Аргументы:
            pressure: давление на выходе (Па)
            backflow_prevention: предотвращение обратного потока
            name: название границы для отображения
        """
        self.pressure = pressure  # Сохраняем давление на выходе
        self.backflow_prevention = backflow_prevention  # Флаг предотвращения обратного потока
        self.name = name  # Сохраняем название
    
    def __str__(self):
        """Строковое представление для вывода в консоль."""
        return (f"{self.name}:\n"
                f"  Давление: {self.pressure:.2f} Па\n"
                f"  Предотвращение обратного потока: {self.backflow_prevention}")


class WallBoundary:
    """
    Граничное условие на твердой стенке.
    Может быть различных типов в зависимости от физической модели.
    """
    
    def __init__(self, wall_type="no_slip", roughness=0.0, temperature=None, name="Стенка"):
        """
        Конструктор ГУ для стенки.
        
        Аргументы:
            wall_type: тип стенки ("no_slip", "slip", "moving")
            roughness: шероховатость стенки (м)
            temperature: температура стенки (К)
            name: название границы для отображения
        """
        self.wall_type = wall_type  # Сохраняем тип стенки
        self.roughness = roughness  # Сохраняем шероховатость
        self.temperature = temperature  # Сохраняем температуру
        self.name = name  # Сохраняем название
    
    def __str__(self):
        """Строковое представление для вывода в консоль."""
        result = f"{self.name}:\n  Тип: {self.wall_type}"
        if self.roughness > 0:
            result += f"\n  Шероховатость: {self.roughness:.6f} м"
        if self.temperature:
            result += f"\n  Температура: {self.temperature:.2f} К"
        return result


class SymmetryBoundary:
    """
    Граничное условие симметрии.
    Используется для уменьшения размерности задачи при наличии симметрии.
    """
    
    def __init__(self, name="Симметрия"):
        """
        Конструктор ГУ симметрии.
        
        Аргументы:
            name: название границы для отображения
        """
        self.name = name  # Сохраняем название
    
    def __str__(self):
        """Строковое представление для вывода в консоль."""
        return f"{self.name}: плоскость симметрии"


class BoundaryCondition:
    """
    Общий класс-обертка для хранения граничного условия.
    Объединяет тип ГУ и его параметры в единую структуру.
    """
    
    def __init__(self, bc_type: str, params: dict, name: str = ""):
        """
        Конструктор общего ГУ.
        
        Аргументы:
            bc_type: тип граничного условия ('inlet', 'outlet', 'wall', 'symmetry')
            params: словарь с параметрами для соответствующего класса
            name: название границы для отображения
        """
        self.bc_type = bc_type  # Сохраняем тип ГУ
        self.params = params  # Сохраняем параметры
        self.name = name  # Сохраняем название
        
        # Создаем соответствующий специализированный объект в зависимости от типа
        if bc_type == 'inlet':
            self.bc_object = InletBoundary(**params)
        elif bc_type == 'outlet':
            self.bc_object = OutletBoundary(**params)
        elif bc_type == 'wall':
            self.bc_object = WallBoundary(**params)
        elif bc_type == 'symmetry':
            self.bc_object = SymmetryBoundary(**params)
        else:
            self.bc_object = None
    
    def __str__(self):
        """Строковое представление для вывода в консоль."""
        if self.name:
            return f"{self.name}: {self.bc_object}"
        return str(self.bc_object)
    
    def get_value(self):
        """
        Возвращает числовое значение для хранения в массиве points.
        """
        if self.bc_type == 'inlet':
            return self.bc_object.velocity_x  # Для входа возвращаем скорость
        elif self.bc_type == 'outlet':
            return self.bc_object.pressure  # Для выхода возвращаем давление
        else:
            return 0.0  # Для остальных типов возвращаем 0


class MeshManager:
    """
    Класс для управления расчетной сеткой в задачах вычислительной гидродинамики (CFD).
    Хранит точки сетки с их физическими свойствами и граничными условиями.
    Является ядром программы, обеспечивающим все операции с данными.
    """
    
    def __init__(self, dtype=np.float32):
        """
        Конструктор класса. Инициализирует менеджер сетки с заданной точностью данных.
        
        Аргументы:
            dtype: тип данных для хранения чисел с плавающей точкой
        """
        self.dtype = dtype  # Сохраняем тип данных
        self.current_id = 0  # Счетчик для генерации уникальных ID
        self.fluid_properties = None  # Свойства жидкости (будут заданы позже)
        self.contour_points = None  # Точки контура (будут заданы позже)
        
        # Создаем структурированный массив для хранения всех данных о точках
        # Используем структурированный массив для компактного хранения разнородных данных
        self.points = np.empty(0, dtype=[
            ('id', np.int32),           # Уникальный идентификатор точки
            ('x', dtype),                # Координата X
            ('y', dtype),                # Координата Y
            ('bc_type', np.int8),        # Тип граничного условия (код)
            ('bc_value', dtype),         # Числовое значение ГУ
            ('density', dtype),           # Плотность
            ('dynamic_viscosity', dtype), # Динамическая вязкость
            ('velocity_x', dtype),        # Скорость по X
            ('velocity_y', dtype),        # Скорость по Y
            ('pressure', dtype),          # Давление
            ('sdf', dtype)                # Расстояние до ближайшей стенки
        ])
        
        # Словарь для преобразования текстовых названий ГУ в числовые коды
        self.bc_mapping = {
            'inlet': 0,           # Вход потока
            'outlet': 1,          # Выход потока
            'wall': 2,            # Непроницаемая стенка
            'symmetry': 3,        # Плоскость симметрии
            'axis_symmetry': 4,   # Осевая симметрия
            'domain': 5           # Внутренние точки (не граница)
        }
        
        # Обратное отображение (из кода в название)
        self.bc_names = {v: k for k, v in self.bc_mapping.items()}
        
        # Значения физических параметров по умолчанию
        self.default_params = {
            'density': 1.225,
            'dynamic_viscosity': 1.8e-5,
            'velocity_x': 1.0,
            'velocity_y': 0.0,
            'pressure': 101325.0,
            'sdf': 0.0
        }

    def add_points(self, coordinates: np.ndarray):
        """
        Добавляет новые точки в расчетную сетку.
        
        Аргументы:
            coordinates: массив координат формы (N, 2)
        """
        # Проверяем тип входных данных
        if not isinstance(coordinates, np.ndarray):
            raise TypeError("Координаты должны быть numpy массивом")
        
        # Приводим координаты к нужному типу данных
        if coordinates.dtype != self.dtype:
            coordinates = coordinates.astype(self.dtype)
        
        # Проверяем размерность массива
        if coordinates.ndim != 2 or coordinates.shape[1] != 2:
            raise ValueError("Координаты должны быть двумерным массивом с формой (N, 2)")
        
        num_points = coordinates.shape[0]  # Количество добавляемых точек
        
        # Определяем начальный ID для новых точек
        start_id = len(self.points) if len(self.points) > 0 else self.current_id
        
        # Создаем новый массив для новых точек
        new_points = np.zeros(num_points, dtype=self.points.dtype)
        
        # Заполняем координаты
        new_points['x'] = coordinates[:, 0]  # Координаты X
        new_points['y'] = coordinates[:, 1]  # Координаты Y
        
        # Генерируем уникальные ID
        new_points['id'] = np.arange(start_id, start_id + num_points, dtype=np.int32)
        self.current_id = start_id + num_points  # Обновляем счетчик ID
        
        # Устанавливаем свойства жидкости или значения по умолчанию
        if self.fluid_properties:
            # Используем свойства из объекта жидкости
            for param in ['density', 'dynamic_viscosity', 'pressure']:
                if hasattr(self.fluid_properties, param):
                    new_points[param] = getattr(self.fluid_properties, param)
        else:
            # Используем значения по умолчанию
            for param, value in self.default_params.items():
                new_points[param] = value
        
        # По умолчанию все новые точки - внутренние (domain)
        new_points['bc_type'] = self.bc_mapping['domain']
        
        # Добавляем новые точки к существующим
        if len(self.points) == 0:
            self.points = new_points
        else:
            self.points = np.append(self.points, new_points)

    def set_fluid_properties(self, fluid: FluidProperties):
        """
        Устанавливает свойства жидкости для всей расчетной области.
        
        Аргументы:
            fluid: объект с свойствами жидкости
        """
        self.fluid_properties = fluid  # Сохраняем объект свойств
        
        # Если точки уже есть, обновляем их свойства
        if len(self.points) > 0:
            self.points['density'] = fluid.density
            self.points['dynamic_viscosity'] = fluid.dynamic_viscosity
            self.points['pressure'] = fluid.pressure
        
        print(f"✅ Свойства жидкости установлены:\n{fluid}")

    def set_boundary(self, indices: np.ndarray, bc: BoundaryCondition):
        """
        Устанавливает граничные условия для указанных точек.
        
        Аргументы:
            indices: индексы точек в массиве points
            bc: объект граничного условия
        """
        # Проверяем корректность типа ГУ
        if bc.bc_type not in self.bc_mapping:
            raise ValueError(f"Неверный тип граничного условия: {bc.bc_type}")
        
        # Проверяем тип индексов
        if not isinstance(indices, np.ndarray):
            raise TypeError("Индексы должны быть numpy массивом")
        
        bc_code = self.bc_mapping[bc.bc_type]  # Получаем числовой код ГУ
        
        # Устанавливаем тип ГУ для выбранных точек
        self.points[indices]['bc_type'] = bc_code
        # Устанавливаем числовое значение ГУ
        self.points[indices]['bc_value'] = bc.get_value()
        
        # Устанавливаем физические параметры в зависимости от типа ГУ
        if bc.bc_type == 'inlet':
            # Для входа устанавливаем скорость и давление
            self.points[indices]['velocity_x'] = bc.bc_object.velocity_x
            self.points[indices]['velocity_y'] = bc.bc_object.velocity_y
            self.points[indices]['pressure'] = bc.bc_object.pressure
        elif bc.bc_type == 'outlet':
            # Для выхода устанавливаем давление
            self.points[indices]['pressure'] = bc.bc_object.pressure
        elif bc.bc_type == 'wall':
            # Для стенок обнуляем скорость (условие прилипания)
            self.points[indices]['velocity_x'] = 0.0
            self.points[indices]['velocity_y'] = 0.0
        # Для остальных типов (symmetry) ничего не делаем

    def set_boundary_conditions(self, boundaries: dict, tolerance: float = 1e-2):
        """
        Устанавливает граничные условия для всех границ расчетной области.
        
        Аргументы:
            boundaries: словарь с ГУ для каждой границы
            tolerance: допуск для определения граничных точек
        """
        # Проверяем, есть ли точки в сетке
        if len(self.points) == 0:
            print("⚠️ Нет точек для установки граничных условий")
            return
        
        print("\n" + "=" * 60)
        print("УСТАНОВКА ГРАНИЧНЫХ УСЛОВИЙ")
        print("=" * 60)
        
        # Определяем границы области по минимальным и максимальным координатам
        x_min, x_max = np.min(self.points['x']), np.max(self.points['x'])
        y_min, y_max = np.min(self.points['y']), np.max(self.points['y'])
        
        print(f"Границы области:")
        print(f"  X: [{x_min:.3f}, {x_max:.3f}]")
        print(f"  Y: [{y_min:.3f}, {y_max:.3f}]")
        
        # ВАЖНО: Сначала сбрасываем все точки в domain
        # Это гарантирует, что не останется старых ГУ
        self.points['bc_type'] = self.bc_mapping['domain']
        print("\n🔄 Сброшены все предыдущие граничные условия")
        
        # Обрабатываем каждую границу из словаря
        for boundary_name, bc in boundaries.items():
            if bc is None:  # Пропускаем незаданные границы
                continue
            
            if boundary_name == 'left':
                # Находим точки на левой границе (x близко к x_min)
                indices = np.where(np.abs(self.points['x'] - x_min) < tolerance)[0]
                if len(indices) > 0:
                    self.set_boundary(indices, bc)
                    print(f"  ✅ {bc.name}: {len(indices)} точек")
                else:
                    print(f"  ⚠️ {bc.name}: точек не найдено")
            
            elif boundary_name == 'right':
                # Находим точки на правой границе (x близко к x_max)
                indices = np.where(np.abs(self.points['x'] - x_max) < tolerance)[0]
                if len(indices) > 0:
                    self.set_boundary(indices, bc)
                    print(f"  ✅ {bc.name}: {len(indices)} точек")
                else:
                    print(f"  ⚠️ {bc.name}: точек не найдено")
            
            elif boundary_name == 'bottom':
                # Находим точки на нижней границе (y близко к y_min)
                indices = np.where(np.abs(self.points['y'] - y_min) < tolerance)[0]
                if len(indices) > 0:
                    self.set_boundary(indices, bc)
                    print(f"  ✅ {bc.name}: {len(indices)} точек")
                else:
                    print(f"  ⚠️ {bc.name}: точек не найдено")
            
            elif boundary_name == 'top':
                # Находим точки на верхней границе (y близко к y_max)
                indices = np.where(np.abs(self.points['y'] - y_max) < tolerance)[0]
                if len(indices) > 0:
                    self.set_boundary(indices, bc)
                    print(f"  ✅ {bc.name}: {len(indices)} точек")
                else:
                    print(f"  ⚠️ {bc.name}: точек не найдено")
            
            elif boundary_name == 'contour':
                # Для контура используем специальный метод поиска
                if hasattr(self, 'contour_points') and self.contour_points is not None and len(self.contour_points) > 0:
                    print(f"\n  🔍 Поиск точек на контуре для {bc.name}...")
                    indices = self.find_points_on_contour(self.contour_points, tolerance)
                    if len(indices) > 0:
                        self.set_boundary(indices, bc)
                        print(f"  ✅ {bc.name}: {len(indices)} точек")
                        # Выводим первые несколько найденных точек для проверки
                        print(f"     Примеры точек на контуре (первые 5):")
                        for i, idx in enumerate(indices[:5]):
                            print(f"       Точка {i+1}: ({self.points[idx]['x']:.6f}, {self.points[idx]['y']:.6f})")
                    else:
                        print(f"  ❌ {bc.name}: НЕ НАЙДЕНО НИ ОДНОЙ ТОЧКИ! Проверьте:")
                        print(f"       - Совпадают ли координаты контура с точками сетки?")
                        print(f"       - Достаточен ли tolerance ({tolerance})?")
                        print(f"       - Загружен ли контур правильно?")
                else:
                    print(f"  ⚠️ {bc.name}: контур не задан или пуст")
        
        print("=" * 60)
        self.print_boundary_statistics()

    def print_boundary_statistics(self):
        """Выводит статистику по граничным условиям для проверки."""
        print("\n📊 СТАТИСТИКА ПО ГРАНИЧНЫМ УСЛОВИЯМ:")
        wall_count = 0
        for bc_name in self.bc_mapping.keys():
            points = self.get_boundary_points(bc_name)
            if len(points) > 0:
                print(f"  {bc_name}: {len(points)} точек")
                if bc_name == 'wall':
                    wall_count = len(points)
        
        domain_points = self.get_boundary_points('domain')
        print(f"\n  Внутренних точек (domain): {len(domain_points)}")
        
        # Проверяем, есть ли точки стенок для SDF
        if wall_count == 0:
            print("\n⚠️  ВНИМАНИЕ: Нет точек с типом 'wall'! SDF будет равен 0 для всех точек.")
            print("   Убедитесь, что контур правильно настроен как стенка.")
        else:
            print(f"\n✅ Найдено {wall_count} точек стенок - SDF будет корректно вычислено")

    def find_points_on_contour(self, contour_points: np.ndarray, tolerance: float = 1e-2):
        """
        Находит индексы точек сетки, которые лежат на заданном контуре.
        Использует KD-дерево для эффективного поиска.
        
        Аргументы:
            contour_points: массив координат точек контура
            tolerance: допуск для определения совпадения точек
            
        Возвращает:
            numpy.ndarray: индексы точек сетки на контуре
        """
        if len(self.points) == 0:
            print("  ⚠️ Нет точек сетки для поиска на контуре")
            return np.array([], dtype=np.int32)
        
        if len(contour_points) == 0:
            print("  ⚠️ Нет точек контура для поиска")
            return np.array([], dtype=np.int32)
        
        print(f"  🔍 Параметры поиска:")
        print(f"     Точек сетки: {len(self.points)}")
        print(f"     Точек контура: {len(contour_points)}")
        print(f"     Tolerance: {tolerance}")
        
        # Создаем массив координат всех точек сетки
        mesh_points = np.column_stack((self.points['x'], self.points['y']))
        
        # Строим KD-дерево для быстрого поиска ближайших соседей
        tree = KDTree(mesh_points)
        
        contour_indices = []  # Список для накопления индексов
        
        # Для каждой точки контура находим ближайшие точки сетки
        for i, contour_point in enumerate(contour_points):
            indices = tree.query_ball_point(contour_point, tolerance)
            if len(indices) > 0:
                contour_indices.extend(indices)
                # Для отладки выведем информацию о первых нескольких точках
                if i < 3 and len(indices) > 0:
                    print(f"     Точка контура {i}: ({contour_point[0]:.6f}, {contour_point[1]:.6f}) -> {len(indices)} точек сетки")
        
        # Убираем дубликаты
        unique_indices = np.unique(contour_indices)
        
        print(f"     Найдено уникальных точек на контуре: {len(unique_indices)}")
        
        return unique_indices.astype(np.int32)

    def set_contour(self, contour_points: np.ndarray):
        """
        Сохраняет точки контура для последующего использования.
        
        Аргументы:
            contour_points: массив координат точек контура
        """
        self.contour_points = contour_points
        print(f"✅ Контур сохранен: {len(contour_points)} точек")
        
        # Выводим информацию о контуре для отладки
        print(f"   Диапазон X контура: {np.min(contour_points[:, 0]):.6f} - {np.max(contour_points[:, 0]):.6f}")
        print(f"   Диапазон Y контура: {np.min(contour_points[:, 1]):.6f} - {np.max(contour_points[:, 1]):.6f}")
        print(f"   Первые 5 точек контура:")
        for i in range(min(5, len(contour_points))):
            print(f"     Точка {i}: ({contour_points[i, 0]:.6f}, {contour_points[i, 1]:.6f})")

    def calculate_sdf(self):
        """
        Вычисляет Signed Distance Function (SDF) - расстояние от каждой точки до ближайшей стенки.
        SDF важна для определения близости к границам и учета пристеночных эффектов.
        """
        # Находим индексы всех точек, которые являются стенками
        wall_indices = np.where(self.points['bc_type'] == self.bc_mapping['wall'])[0]
        
        if len(wall_indices) == 0:  # Если нет точек стенок
            warnings.warn("Нет точек стенок для расчёта SDF. Все расстояния будут равны 0.")
            self.points['sdf'] = 0.0  # Устанавливаем SDF в 0
            return
        
        print(f"\n📐 Вычисление SDF:")
        print(f"   Найдено {len(wall_indices)} точек стенок")
        
        # Формируем массив координат точек стенок
        wall_points = np.column_stack((self.points['x'][wall_indices], 
                                       self.points['y'][wall_indices]))
        
        # Формируем массив координат всех точек
        all_points = np.column_stack((self.points['x'], self.points['y']))
        
        # Вычисляем матрицу расстояний между всеми точками и точками стенок
        # Используем scipy.spatial.distance.cdist для эффективного вычисления
        dist_matrix = distance.cdist(all_points, wall_points, metric='euclidean')
        
        # Для каждой точки находим минимальное расстояние до стенок
        min_distances = np.min(dist_matrix, axis=1)
        
        # Сохраняем вычисленные расстояния в поле SDF
        self.points['sdf'] = min_distances.astype(self.dtype)
        
        # Для точек стенок устанавливаем SDF = 0 (расстояние до самой себя)
        self.points['sdf'][self.points['bc_type'] == self.bc_mapping['wall']] = 0.0
        
        print(f"   SDF вычислено. Диапазон расстояний: [{np.min(min_distances):.6f}, {np.max(min_distances):.6f}]")

    def check_precision(self):
        """
        Проверяет корректность данных в массиве points.
        Выявляет NaN, Inf и некорректные физические значения.
        """
        if np.isnan(self.points['x']).any() or np.isnan(self.points['y']).any():
            warnings.warn("Обнаружены NaN в координатах")
        
        if np.isinf(self.points['x']).any() or np.isinf(self.points['y']).any():
            warnings.warn("Обнаружены Inf в координатах")
        
        if (self.points['dynamic_viscosity'] <= 0).any():
            warnings.warn("Обнаружены некорректные значения вязкости")
        
        if (self.points['density'] <= 0).any():
            warnings.warn("Обнаружены некорректные значения плотности")

    def save_to_file(self, filename: str):
        """
        Сохраняет все данные сетки в файл формата .npz.
        
        Аргументы:
            filename: имя файла для сохранения
        """
        if not filename.endswith('.npz'):
            filename += '.npz'
        
        try:
            # Создаем словарь для сохранения
            save_dict = {
                'points': self.points,
                'bc_mapping': self.bc_mapping,
                'dtype': self.dtype,
                'default_params': self.default_params,
                'current_id': self.current_id
            }
            
            # Добавляем контур, если он есть
            if hasattr(self, 'contour_points') and self.contour_points is not None:
                save_dict['contour_points'] = self.contour_points
            
            # Добавляем свойства жидкости, если они есть
            if self.fluid_properties:
                save_dict['fluid_properties'] = self.fluid_properties.to_dict()
            
            np.savez(filename, **save_dict)
            print(f"✅ Данные успешно сохранены в {filename}")
        except Exception as e:
            warnings.warn(f"Ошибка при сохранении: {e}")

    @staticmethod
    def load_from_file(filename: str):
        """
        Статический метод для загрузки данных сетки из файла.
        
        Аргументы:
            filename: имя файла для загрузки
            
        Возвращает:
            MeshManager: восстановленный экземпляр класса
        """
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Файл {filename} не найден")
        
        data = np.load(filename, allow_pickle=True)
        
        # Создаем новый экземпляр с сохраненным типом данных
        manager = MeshManager(dtype=data['dtype'].item())
        
        # Восстанавливаем все поля
        manager.points = data['points']
        manager.bc_mapping = data['bc_mapping'].item()
        manager.bc_names = {v: k for k, v in manager.bc_mapping.items()}
        manager.default_params = data['default_params'].item()
        manager.current_id = data['current_id'].item() if 'current_id' in data else len(manager.points)
        
        # Загружаем контур, если он есть
        if 'contour_points' in data:
            manager.contour_points = data['contour_points']
        
        # Загружаем свойства жидкости, если они есть
        if 'fluid_properties' in data:
            manager.fluid_properties = FluidProperties.from_dict(data['fluid_properties'].item())
        
        return manager

    def get_sdf_stats(self):
        """
        Вычисляет статистические характеристики поля SDF.
        
        Возвращает:
            dict: словарь со статистиками
        """
        if 'sdf' not in self.points.dtype.fields:
            raise ValueError("Поле SDF не найдено в данных")
        
        sdf_values = self.points['sdf']
        return {
            'min': np.min(sdf_values),
            'max': np.max(sdf_values),
            'mean': np.mean(sdf_values),
            'std': np.std(sdf_values)
        }

    def visualize_bc(self, highlight_walls=True):
        """
        Визуализирует граничные условия.
        Каждый тип ГУ отображается своим цветом.
        
        Аргументы:
            highlight_walls: выделять ли точки стенок
        """
        try:
            plt.figure(figsize=(14, 10))
            
            # Цвета для разных типов ГУ
            bc_colors = {
                'inlet': 'blue',
                'outlet': 'red',
                'wall': 'black',
                'symmetry': 'green',
                'axis_symmetry': 'purple',
                'domain': 'gray'
            }
            
            # Сначала отображаем все внутренние точки серым цветом
            domain_points = self.get_boundary_points('domain')
            if len(domain_points) > 0:
                plt.scatter(
                    domain_points['x'],
                    domain_points['y'],
                    c='lightgray',
                    s=5,
                    alpha=0.5,
                    label='Внутренние точки'
                )
            
            # Отображаем точки с разными ГУ
            wall_count = 0
            for bc_name, bc_color in bc_colors.items():
                if bc_name == 'domain':
                    continue
                points = self.get_boundary_points(bc_name)
                if len(points) > 0:
                    plt.scatter(
                        points['x'],
                        points['y'],
                        c=bc_color,
                        s=20,
                        alpha=0.8,
                        label=f'{bc_name} ({len(points)})',
                        edgecolors='white',
                        linewidths=0.5
                    )
                    if bc_name == 'wall':
                        wall_count = len(points)
            
            # Если есть контур, отображаем его линией и точками
            if hasattr(self, 'contour_points') and self.contour_points is not None and len(self.contour_points) > 0:
                plt.plot(self.contour_points[:, 0], self.contour_points[:, 1], 
                        'k-', linewidth=2, alpha=0.7, label='Контур профиля')
                # Также отметим точки контура
                plt.scatter(
                    self.contour_points[:, 0],
                    self.contour_points[:, 1],
                    c='orange',
                    s=30,
                    marker='o',
                    alpha=0.5,
                    label='Точки контура (исходные)',
                    zorder=4
                )
            
            plt.title('Визуализация граничных условий')
            plt.xlabel('Координата X (м)')
            plt.ylabel('Координата Y (м)')
            plt.grid(True, alpha=0.3)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            
            # Добавляем информацию о количестве стенок
            if wall_count > 0:
                plt.figtext(0.02, 0.02, f'✅ Найдено {wall_count} точек стенок', 
                          fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
            else:
                plt.figtext(0.02, 0.02, '❌ НЕТ ТОЧЕК СТЕНОК!', 
                          fontsize=12, color='red', bbox=dict(facecolor='yellow', alpha=0.8))
            
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            warnings.warn("Для визуализации требуется установить matplotlib")
        except Exception as e:
            warnings.warn(f"Ошибка при визуализации: {e}")

    def visualize_sdf(self, highlight_walls=True):
        """
        Визуализирует поле SDF с помощью цветовой карты.
        
        Аргументы:
            highlight_walls: выделять ли точки стенок
        """
        try:
            plt.figure(figsize=(14, 10))
            
            # Проверяем, есть ли ненулевые значения SDF
            if np.all(self.points['sdf'] == 0):
                print("⚠️ Все значения SDF равны 0! Нет точек стенок для расчета расстояний.")
                # Создаем простой график
                plt.scatter(
                    self.points['x'],
                    self.points['y'],
                    c='gray',
                    s=10,
                    alpha=0.7,
                    label='Точки сетки (SDF=0)'
                )
            else:
                # Создаем scatter plot с цветовой картой для SDF
                scatter = plt.scatter(
                    self.points['x'],
                    self.points['y'],
                    c=self.points['sdf'],
                    cmap='coolwarm',
                    s=10,
                    alpha=0.7,
                    label='Точки сетки'
                )
                plt.colorbar(scatter, label='Расстояние до стенки (SDF), м')
            
            # Если нужно выделить точки стенок
            if highlight_walls:
                wall_points = self.get_boundary_points('wall')
                if len(wall_points) > 0:
                    plt.scatter(
                        wall_points['x'],
                        wall_points['y'],
                        c='black',
                        s=30,
                        marker='s',
                        edgecolors='white',
                        linewidths=1,
                        label='Стенка (wall)',
                        zorder=5
                    )
            
            # Если есть контур, отображаем его линией
            if hasattr(self, 'contour_points') and self.contour_points is not None and len(self.contour_points) > 0:
                plt.plot(self.contour_points[:, 0], self.contour_points[:, 1], 
                        'k-', linewidth=2, alpha=0.7, label='Контур профиля')
            
            plt.title('Визуализация SDF с выделением границ')
            plt.xlabel('Координата X (м)')
            plt.ylabel('Координата Y (м)')
            plt.grid(True, alpha=0.3)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            warnings.warn("Для визуализации требуется установить matplotlib")
        except Exception as e:
            warnings.warn(f"Ошибка при визуализации: {e}")

    def get_boundary_points(self, bc_type: str):
        """
        Возвращает все точки с заданным типом граничного условия.
        
        Аргументы:
            bc_type: строковое название граничного условия
            
        Возвращает:
            numpy.ndarray: массив точек с указанным ГУ
        """
        if bc_type not in self.bc_mapping:
            raise ValueError(f"Неверный тип граничного условия: {bc_type}")
        
        bc_code = self.bc_mapping[bc_type]
        return self.points[self.points['bc_type'] == bc_code]

    def get_boundary_summary(self):
        """
        Возвращает сводку по граничным условиям.
        
        Возвращает:
            dict: словарь с информацией о каждом типе ГУ
        """
        summary = {}
        for bc_name in self.bc_mapping.keys():
            points = self.get_boundary_points(bc_name)
            if len(points) > 0:
                summary[bc_name] = {
                    'count': len(points),
                    'value_range': (np.min(points['bc_value']), np.max(points['bc_value']))
                }
        return summary

    def update_physical_properties(self, indices: np.ndarray, **kwargs):
        """
        Обновляет физические свойства для указанных точек.
        
        Аргументы:
            indices: индексы точек для обновления
            **kwargs: пары "имя_поля=значение" для обновления
        """
        for key, value in kwargs.items():
            if key not in self.points.dtype.fields:
                raise ValueError(f"Неверное поле: {key}")
            
            if isinstance(value, (int, float)):
                self.points[indices][key] = value
            elif isinstance(value, np.ndarray):
                if len(value) != len(indices):
                    raise ValueError("Размер массива значений не соответствует количеству индексов")
                self.points[indices][key] = value
            else:
                raise TypeError(f"Неверный тип значения для {key}")


def load_points_from_file(file_path: str, file_type: str = "mesh"):
    """
    Загружает координаты точек из файла.
    Поддерживает различные форматы разделителей.
    
    Аргументы:
        file_path: путь к файлу
        file_type: тип файла для вывода
        
    Возвращает:
        numpy.ndarray: массив координат точек
    """
    print("=" * 50)
    print(f"ЗАГРУЗКА {file_type.upper()} ИЗ ФАЙЛА")
    print("=" * 50)
    
    print(f"1. Проверяем файл: {file_path}")
    
    if not os.path.exists(file_path):
        print(f"   ❌ ОШИБКА: Файл не найден!")
        return np.array([])
    
    print(f"   ✅ Файл найден")
    print(f"   Размер файла: {os.path.getsize(file_path)} байт")
    
    points = []
    
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
            print(f"2. Прочитано {len(lines)} строк из файла")
            
            for line_number, line in enumerate(lines, start=1):
                line = line.strip()
                
                if not line or line.startswith('#'):
                    continue
                
                try:
                    # Пробуем разные разделители
                    if '\t' in line:
                        parts = line.split('\t')
                    elif ' ' in line:
                        parts = line.split()
                    elif ',' in line:
                        parts = line.split(',')
                    else:
                        continue
                    
                    if len(parts) >= 2:
                        x = float(parts[0].strip())
                        y = float(parts[1].strip())
                        points.append((x, y))
                
                except (ValueError, IndexError):
                    pass  # Пропускаем строки с ошибками
                    
    except Exception as e:
        print(f"❌ ОШИБКА при чтении файла: {str(e)}")
        return np.array([])
    
    if len(points) == 0:
        print(f"❌ Не загружено ни одной точки!")
        return np.array([])
    
    points_array = np.array(points)
    print(f"3. ✅ Загружено {len(points_array)} точек")
    print(f"   Диапазон X: {np.min(points_array[:, 0]):.6f} - {np.max(points_array[:, 0]):.6f}")
    print(f"   Диапазон Y: {np.min(points_array[:, 1]):.6f} - {np.max(points_array[:, 1]):.6f}")
    print("=" * 50)
    
    return points_array


def test_file_loading():
    """
    Тестовая функция для проверки загрузки файла с точками.
    """
    print("\n" + "=" * 50)
    print("ТЕСТ ЗАГРУЗКИ ФАЙЛА")
    print("=" * 50)
    
    test_path = r"C:\Users\Ivan\source\repos\PINN_Flow2D\Samplers\Data\points.txt"
    
    print(f"Проверяем путь: {test_path}")
    
    if os.path.exists(test_path):
        print(f"✅ Файл существует")
        print(f"   Размер: {os.path.getsize(test_path)} байт")
        
        try:
            with open(test_path, 'r') as f:
                lines = f.readlines()
                print(f"   Всего строк: {len(lines)}")
                print(f"\n   Первые 5 строк файла:")
                for i, line in enumerate(lines[:5]):
                    print(f"   {i+1}: {line.strip()}")
                
                # Проверяем формат данных
                print(f"\n   Проверка формата данных:")
                valid_lines = 0
                for i, line in enumerate(lines[:10]):
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    try:
                        if '\t' in line:
                            parts = line.split('\t')
                        else:
                            parts = line.split()
                        if len(parts) >= 2:
                            x = float(parts[0])
                            y = float(parts[1])
                            print(f"   Строка {i+1}: X={x:.6f}, Y={y:.6f} ✓")
                            valid_lines += 1
                        else:
                            print(f"   Строка {i+1}: недостаточно данных ✗")
                    except:
                        print(f"   Строка {i+1}: ошибка преобразования ✗")
                
                if valid_lines > 0:
                    print(f"\n   ✅ Формат данных корректен")
                else:
                    print(f"\n   ❌ Формат данных некорректен")
                    
        except Exception as e:
            print(f"❌ Ошибка чтения файла: {e}")
    else:
        print(f"❌ Файл НЕ существует")
        print(f"   Текущая директория: {os.getcwd()}")
        
        parent_dir = os.path.dirname(test_path)
        if os.path.exists(parent_dir):
            print(f"   ✅ Родительская директория существует: {parent_dir}")
            print(f"   Содержимое директории:")
            files = os.listdir(parent_dir)
            for f in files[:10]:
                print(f"      - {f}")
            if len(files) > 10:
                print(f"      ... и еще {len(files) - 10} файлов")
        else:
            print(f"   ❌ Родительская директория не существует: {parent_dir}")
            
            # Проверяем путь по частям
            print(f"\n   Проверка пути по частям:")
            parts = test_path.split('\\')
            current_path = parts[0] + '\\'
            for i, part in enumerate(parts[1:], 1):
                current_path = os.path.join(current_path, part)
                exists = os.path.exists(current_path)
                status = "✅" if exists else "❌"
                print(f"   {status} {current_path}")


def create_fluid_from_input():
    """
    Создает свойства жидкости на основе пользовательского ввода.
    
    Возвращает:
        FluidProperties: объект со свойствами жидкости
    """
    print("\n" + "=" * 50)
    print("НАСТРОЙКА СВОЙСТВ ЖИДКОСТИ/ГАЗА")
    print("=" * 50)
    
    print("\nВыберите способ задания:")
    print("1. Ввести вручную")
    print("2. Загрузить из файла")
    print("3. Использовать стандартные (воздух)")
    
    choice = input("Ваш выбор (1/2/3): ").strip()
    
    if choice == "2":
        # Загрузка из JSON-файла
        file_path = input("Путь к файлу с описанием жидкости: ").strip()
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                fluid = FluidProperties.from_dict(data)
                print(f"\n✅ Загружены свойства:")
                print(fluid)
                return fluid
            except:
                print("❌ Ошибка загрузки файла. Используем стандартные.")
                return FluidProperties()
        else:
            print("❌ Файл не найден. Используем стандартные.")
            return FluidProperties()
    
    elif choice == "1":
        # Ручной ввод
        print("\nВведите свойства жидкости/газа:")
        
        name = input("Название [Воздух]: ").strip() or "Воздух"
        
        try:
            density = float(input("Плотность (кг/м³) [1.225]: ").strip() or "1.225")
        except:
            density = 1.225
        
        try:
            dyn_visc = float(input("Динамическая вязкость (Па·с) [1.8e-5]: ").strip() or "1.8e-5")
        except:
            dyn_visc = 1.8e-5
        
        try:
            temperature = float(input("Температура (К) [293.15]: ").strip() or "293.15")
        except:
            temperature = 293.15
        
        try:
            pressure = float(input("Давление (Па) [101325]: ").strip() or "101325")
        except:
            pressure = 101325
        
        fluid = FluidProperties(
            name=name,
            density=density,
            dynamic_viscosity=dyn_visc,
            temperature=temperature,
            pressure=pressure
        )
        
        print(f"\n✅ Созданы свойства:")
        print(fluid)
        return fluid
    
    else:
        print("\n✅ Используются стандартные свойства воздуха")
        return FluidProperties()


def create_inlet_from_input(boundary_name="Левая граница"):
    """
    Создает ГУ для входа на основе пользовательского ввода.
    
    Аргументы:
        boundary_name: название границы
        
    Возвращает:
        InletBoundary: объект ГУ
    """
    print(f"\nНастройка ГУ для {boundary_name} (ВХОД):")
    
    try:
        vx = float(input("  Скорость по X (м/с) [10.0]: ").strip() or "10.0")
    except:
        vx = 10.0
    
    try:
        vy = float(input("  Скорость по Y (м/с) [0.0]: ").strip() or "0.0")
    except:
        vy = 0.0
    
    try:
        pressure = float(input("  Давление на входе (Па) [101325]: ").strip() or "101325")
    except:
        pressure = 101325
    
    try:
        turbulence = float(input("  Интенсивность турбулентности (0-1) [0.05]: ").strip() or "0.05")
    except:
        turbulence = 0.05
    
    return InletBoundary(
        velocity_x=vx,
        velocity_y=vy,
        pressure=pressure,
        turbulence_intensity=turbulence,
        name=boundary_name
    )


def create_outlet_from_input(boundary_name="Правая граница"):
    """
    Создает ГУ для выхода на основе пользовательского ввода.
    
    Аргументы:
        boundary_name: название границы
        
    Возвращает:
        OutletBoundary: объект ГУ
    """
    print(f"\nНастройка ГУ для {boundary_name} (ВЫХОД):")
    
    try:
        pressure = float(input("  Давление на выходе (Па) [101325]: ").strip() or "101325")
    except:
        pressure = 101325
    
    backflow = input("  Предотвращать обратный поток? (y/n) [y]: ").strip().lower()
    backflow_prevention = backflow != 'n'
    
    return OutletBoundary(
        pressure=pressure,
        backflow_prevention=backflow_prevention,
        name=boundary_name
    )


def create_wall_from_input(boundary_name="Стенка"):
    """
    Создает ГУ для стенки на основе пользовательского ввода.
    По умолчанию создает стенку без проскальзывания (no_slip) и гладкую (roughness=0).
    
    Аргументы:
        boundary_name: название границы
        
    Возвращает:
        WallBoundary: объект ГУ
    """
    print(f"\nНастройка ГУ для {boundary_name} (СТЕНКА):")
    print("  По умолчанию: no_slip (без проскальзывания), гладкая (шероховатость=0)")
    
    # По умолчанию используем no_slip (условие прилипания)
    wall_type = "no_slip"
    roughness = 0.0
    temperature = None
    
    # Спрашиваем, нужно ли изменить параметры
    change = input("  Изменить параметры? (y/n) [n]: ").strip().lower()
    
    if change == 'y':
        print("  Выберите тип стенки:")
        print("    1. no_slip - условие прилипания (скорость = 0)")
        print("    2. slip - условие скольжения")
        print("    3. moving - движущаяся стенка")
        
        wall_type_choice = input("  Ваш выбор (1/2/3) [1]: ").strip() or "1"
        
        if wall_type_choice == "2":
            wall_type = "slip"
        elif wall_type_choice == "3":
            wall_type = "moving"
        
        try:
            roughness = float(input("  Шероховатость стенки (м) [0.0]: ").strip() or "0.0")
        except:
            roughness = 0.0
        
        temp_input = input("  Температура стенки (К) [оставить пустым, если не задана]: ").strip()
        temperature = float(temp_input) if temp_input else None
    
    return WallBoundary(
        wall_type=wall_type,
        roughness=roughness,
        temperature=temperature,
        name=boundary_name
    )


def create_symmetry_from_input(boundary_name="Симметрия"):
    """
    Создает ГУ симметрии.
    
    Аргументы:
        boundary_name: название границы
        
    Возвращает:
        SymmetryBoundary: объект ГУ
    """
    print(f"\n{boundary_name}: плоскость симметрии")
    return SymmetryBoundary(name=boundary_name)


def interactive_boundary_setup():
    """
    Интерактивная настройка граничных условий для всех границ.
    
    Возвращает:
        dict: словарь с ГУ для каждой границы
    """
    print("\n" + "=" * 60)
    print("ИНТЕРАКТИВНАЯ НАСТРОЙКА ГРАНИЧНЫХ УСЛОВИЙ")
    print("=" * 60)
    print("⚠️  ВАЖНО: Граничные условия будут применены ТОЛЬКО к точкам на границах области")
    print("   Внутренние точки останутся с типом 'domain'")
    print("   Границы определяются по минимальным/максимальным координатам X и Y")
    
    boundaries = {}
    
    # ЛЕВАЯ ГРАНИЦА
    print("\n" + "-" * 40)
    print("НАСТРОЙКА ЛЕВОЙ ГРАНИЦЫ (X = min)")
    print("-" * 40)
    print("Доступные типы ГУ:")
    print("  1. inlet - вход")
    print("  2. outlet - выход")
    print("  3. wall - стенка")
    print("  4. symmetry - симметрия")
    print("  5. none - не задавать")
    
    choice = input("Выберите тип ГУ (1-5) [1]: ").strip() or "1"
    
    if choice == "1":
        bc_object = create_inlet_from_input("Левая граница")
        boundaries['left'] = BoundaryCondition('inlet', bc_object.__dict__, "Левая граница")
    elif choice == "2":
        bc_object = create_outlet_from_input("Левая граница")
        boundaries['left'] = BoundaryCondition('outlet', bc_object.__dict__, "Левая граница")
    elif choice == "3":
        bc_object = create_wall_from_input("Левая граница")
        boundaries['left'] = BoundaryCondition('wall', bc_object.__dict__, "Левая граница")
    elif choice == "4":
        bc_object = create_symmetry_from_input("Левая граница")
        boundaries['left'] = BoundaryCondition('symmetry', {}, "Левая граница")
    # choice 5 - none, ничего не добавляем
    
    # ПРАВАЯ ГРАНИЦА
    print("\n" + "-" * 40)
    print("НАСТРОЙКА ПРАВОЙ ГРАНИЦЫ (X = max)")
    print("-" * 40)
    print("Доступные типы ГУ:")
    print("  1. inlet - вход")
    print("  2. outlet - выход")
    print("  3. wall - стенка")
    print("  4. symmetry - симметрия")
    print("  5. none - не задавать")
    
    choice = input("Выберите тип ГУ (1-5) [2]: ").strip() or "2"
    
    if choice == "1":
        bc_object = create_inlet_from_input("Правая граница")
        boundaries['right'] = BoundaryCondition('inlet', bc_object.__dict__, "Правая граница")
    elif choice == "2":
        bc_object = create_outlet_from_input("Правая граница")
        boundaries['right'] = BoundaryCondition('outlet', bc_object.__dict__, "Правая граница")
    elif choice == "3":
        bc_object = create_wall_from_input("Правая граница")
        boundaries['right'] = BoundaryCondition('wall', bc_object.__dict__, "Правая граница")
    elif choice == "4":
        bc_object = create_symmetry_from_input("Правая граница")
        boundaries['right'] = BoundaryCondition('symmetry', {}, "Правая граница")
    
    # НИЖНЯЯ ГРАНИЦА
    print("\n" + "-" * 40)
    print("НАСТРОЙКА НИЖНЕЙ ГРАНИЦЫ (Y = min)")
    print("-" * 40)
    print("Доступные типы ГУ:")
    print("  1. inlet - вход")
    print("  2. outlet - выход")
    print("  3. wall - стенка")
    print("  4. symmetry - симметрия")
    print("  5. none - не задавать")
    
    choice = input("Выберите тип ГУ (1-5) [3]: ").strip() or "3"
    
    if choice == "1":
        bc_object = create_inlet_from_input("Нижняя граница")
        boundaries['bottom'] = BoundaryCondition('inlet', bc_object.__dict__, "Нижняя граница")
    elif choice == "2":
        bc_object = create_outlet_from_input("Нижняя граница")
        boundaries['bottom'] = BoundaryCondition('outlet', bc_object.__dict__, "Нижняя граница")
    elif choice == "3":
        bc_object = create_wall_from_input("Нижняя граница")
        boundaries['bottom'] = BoundaryCondition('wall', bc_object.__dict__, "Нижняя граница")
    elif choice == "4":
        bc_object = create_symmetry_from_input("Нижняя граница")
        boundaries['bottom'] = BoundaryCondition('symmetry', {}, "Нижняя граница")
    
    # ВЕРХНЯЯ ГРАНИЦА
    print("\n" + "-" * 40)
    print("НАСТРОЙКА ВЕРХНЕЙ ГРАНИЦЫ (Y = max)")
    print("-" * 40)
    print("Доступные типы ГУ:")
    print("  1. inlet - вход")
    print("  2. outlet - выход")
    print("  3. wall - стенка")
    print("  4. symmetry - симметрия")
    print("  5. none - не задавать")
    
    choice = input("Выберите тип ГУ (1-5) [4]: ").strip() or "4"
    
    if choice == "1":
        bc_object = create_inlet_from_input("Верхняя граница")
        boundaries['top'] = BoundaryCondition('inlet', bc_object.__dict__, "Верхняя граница")
    elif choice == "2":
        bc_object = create_outlet_from_input("Верхняя граница")
        boundaries['top'] = BoundaryCondition('outlet', bc_object.__dict__, "Верхняя граница")
    elif choice == "3":
        bc_object = create_wall_from_input("Верхняя граница")
        boundaries['top'] = BoundaryCondition('wall', bc_object.__dict__, "Верхняя граница")
    elif choice == "4":
        bc_object = create_symmetry_from_input("Верхняя граница")
        boundaries['top'] = BoundaryCondition('symmetry', {}, "Верхняя граница")
    
    # КОНТУР ПРОФИЛЯ
    contour_path = r"C:\Users\Ivan\source\repos\PINN_Flow2D\Geometry\Data\foil.txt"
    if os.path.exists(contour_path):
        print("\n" + "-" * 40)
        print("НАСТРОЙКА КОНТУРА ПРОФИЛЯ")
        print("-" * 40)
        print("⚠️  Для корректного расчета SDF контур ДОЛЖЕН быть стенкой!")
        print("   Рекомендуется использовать тип 'wall' (стенка) без проскальзывания.")
        
        print("\nДоступные типы ГУ:")
        print("  1. wall - стенка (рекомендуется)")
        print("  2. symmetry - симметрия")
        print("  3. none - не задавать")
        
        choice = input("Выберите тип ГУ (1-3) [1]: ").strip() or "1"
        
        if choice == "1":
            # Создаем стенку с параметрами по умолчанию (no_slip, гладкая)
            bc_object = WallBoundary(
                wall_type="no_slip",
                roughness=0.0,
                temperature=None,
                name="Контур профиля"
            )
            boundaries['contour'] = BoundaryCondition('wall', bc_object.__dict__, "Контур профиля")
            print("  ✅ Контур настроен как стенка без проскальзывания, гладкая")
        elif choice == "2":
            bc_object = create_symmetry_from_input("Контур профиля")
            boundaries['contour'] = BoundaryCondition('symmetry', {}, "Контур профиля")
    else:
        print(f"\n⚠️ Файл контура не найден: {contour_path}")
        print("   Продолжаем без контура")
    
    return boundaries


def main():
    """
    Главная функция программы.
    Оркестрирует весь процесс: загрузка, настройка, расчет и визуализация.
    """
    print("\n" + "=" * 70)
    print("ПРОГРАММА НАСТРОЙКИ РАСЧЕТНОЙ СЕТКИ ДЛЯ CFD")
    print("=" * 70)
    
    # Тест загрузки файла
    test_file_loading()
    
    # ЭТАП 1: ЗАГРУЗКА ГЕОМЕТРИИ
    print("\n" + "=" * 50)
    print("ЭТАП 1: ЗАГРУЗКА ГЕОМЕТРИИ")
    print("=" * 50)
    
    # Пути к файлам
    mesh_path = r"C:\Users\Ivan\source\repos\PINN_Flow2D\Samplers\Data\points.txt"
    contour_path = r"C:\Users\Ivan\source\repos\PINN_Flow2D\Geometry\Data\foil.txt"
    
    print("\n📦 Загрузка точек расчетной сетки...")
    print(f"   Путь к файлу: {mesh_path}")
    mesh_points = load_points_from_file(mesh_path, "MESH")
    
    if len(mesh_points) == 0:
        print("\n❌ КРИТИЧЕСКАЯ ОШИБКА: Не удалось загрузить точки сетки!")
        print("   Убедитесь, что файл существует по указанному пути.")
        print("\n   Возможные решения:")
        print("   1. Проверьте правильность пути к файлу")
        print("   2. Убедитесь, что файл не пустой")
        print("   3. Проверьте формат данных (должны быть пары чисел X Y)")
        print("\n   Программа будет завершена.")
        return
    
    # Загружаем контур профиля
    if os.path.exists(contour_path):
        contour_points = load_points_from_file(contour_path, "CONTOUR")
    else:
        print(f"\n⚠️ Файл контура не найден: {contour_path}")
        print("   Продолжаем работу без контура")
        contour_points = np.array([])
    
    # ЭТАП 2: НАСТРОЙКА СВОЙСТВ ЖИДКОСТИ
    print("\n" + "=" * 50)
    print("ЭТАП 2: НАСТРОЙКА СВОЙСТВ ЖИДКОСТИ")
    print("=" * 50)
    
    fluid = create_fluid_from_input()
    
    # ЭТАП 3: СОЗДАНИЕ РАСЧЕТНОЙ СЕТКИ
    print("\n" + "=" * 50)
    print("ЭТАП 3: СОЗДАНИЕ РАСЧЕТНОЙ СЕТКИ")
    print("=" * 50)
    
    mesh = MeshManager(dtype=np.float32)
    mesh.set_fluid_properties(fluid)
    mesh.add_points(mesh_points)
    print(f"✅ Создана сетка с {len(mesh.points)} точками")
    
    if len(contour_points) > 0:
        mesh.set_contour(contour_points)
        print(f"✅ Добавлен контур с {len(contour_points)} точками")
    
    # ЭТАП 4: НАСТРОЙКА ГРАНИЧНЫХ УСЛОВИЙ
    print("\n" + "=" * 50)
    print("ЭТАП 4: НАСТРОЙКА ГРАНИЧНЫХ УСЛОВИЙ")
    print("=" * 50)
    print("⚠️  Для корректного расчета SDF необходимо задать стенки на контуре!")
    
    boundaries = interactive_boundary_setup()
    
    if boundaries:
        mesh.set_boundary_conditions(boundaries, tolerance=1e-2)  # Увеличиваем tolerance для контура
    else:
        print("⚠️ Не задано ни одного граничного условия!")
        print("   Все точки останутся с типом 'domain'")
    
    # ЭТАП 5: ВЫЧИСЛЕНИЕ SDF
    print("\n" + "=" * 50)
    print("ЭТАП 5: ВЫЧИСЛЕНИЕ SDF")
    print("=" * 50)
    print("📐 SDF вычисляется как расстояние до ближайшей точки с типом 'wall'")
    print("   Поэтому важно правильно задать стенки на контуре!")
    
    mesh.calculate_sdf()
    
    # ЭТАП 6: ПРОВЕРКА ДАННЫХ
    print("\n" + "=" * 50)
    print("ЭТАП 6: ПРОВЕРКА ДАННЫХ")
    print("=" * 50)
    
    mesh.check_precision()
    
    # ЭТАП 7: СТАТИСТИКА
    print("\n" + "=" * 50)
    print("ЭТАП 7: СТАТИСТИКА")
    print("=" * 50)
    
    print(f"\n📊 Общая статистика:")
    print(f"   Всего точек: {len(mesh.points)}")
    print(f"   Диапазон X: {np.min(mesh.points['x']):.6f} - {np.max(mesh.points['x']):.6f}")
    print(f"   Диапазон Y: {np.min(mesh.points['y']):.6f} - {np.max(mesh.points['y']):.6f}")
    
    summary = mesh.get_boundary_summary()
    if summary:
        print(f"\n📊 Распределение по граничным условиям:")
        for bc_name, stats in summary.items():
            print(f"   {bc_name}: {stats['count']} точек")
    
    try:
        sdf_stats = mesh.get_sdf_stats()
        print(f"\n📊 Статистика SDF:")
        print(f"   Минимальное: {sdf_stats['min']:.6f} м")
        print(f"   Максимальное: {sdf_stats['max']:.6f} м")
        print(f"   Среднее: {sdf_stats['mean']:.6f} м")
        print(f"   Стандартное отклонение: {sdf_stats['std']:.6f} м")
    except Exception as e:
        print(f"❌ Ошибка при получении статистики SDF: {e}")
    
    # ЭТАП 8: ВИЗУАЛИЗАЦИЯ
    print("\n" + "=" * 50)
    print("ЭТАП 8: ВИЗУАЛИЗАЦИЯ")
    print("=" * 50)
    print("🖼️  На графике ГУ: синий - вход, красный - выход, черный - стенка, зеленый - симметрия")
    print("🖼️  На графике SDF: цвет показывает расстояние до стенки")
    
    try:
        print("🖼️  Визуализация граничных условий...")
        mesh.visualize_bc(highlight_walls=True)
        
        print("🖼️  Визуализация SDF...")
        mesh.visualize_sdf(highlight_walls=True)
    except Exception as e:
        print(f"❌ Ошибка при визуализации: {e}")
    
    # ЭТАП 9: СОХРАНЕНИЕ РЕЗУЛЬТАТОВ
    print("\n" + "=" * 50)
    print("ЭТАП 9: СОХРАНЕНИЕ РЕЗУЛЬТАТОВ")
    print("=" * 50)
    
    mesh.save_to_file('cfd_mesh')
    time.sleep(590)
    print("\n" + "=" * 70)
    print("🎉 РАБОТА ПРОГРАММЫ УСПЕШНО ЗАВЕРШЕНА")
    print("=" * 70)
    print(f"\n📁 Результаты сохранены в файл: cfd_mesh.npz")
    print(f"📊 Всего обработано точек: {len(mesh.points)}")


if __name__ == "__main__":
    main()