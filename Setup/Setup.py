import numpy as np  # Импортируем NumPy для работы с массивами и математическими операциями
import warnings     # Импортируем модуль для управления предупреждениями
import os          # Импортируем модуль для работы с операционной системой (файлы, пути)
from scipy.spatial import distance  # Импортируем функцию для вычисления расстояний между точками
import matplotlib.pyplot as plt      # Импортируем модуль для визуализации данных

class MeshManager:
    """
    Класс для управления расчетной сеткой в задачах вычислительной гидродинамики (CFD).
    Хранит точки сетки с их физическими свойствами и граничными условиями.
    """
    
    def __init__(self, dtype=np.float32):
        """
        Конструктор класса. Инициализирует менеджер сетки с заданной точностью данных.
        
        Аргументы:
            dtype: тип данных для хранения чисел с плавающей точкой (по умолчанию float32 для экономии памяти)
        """
        self.dtype = dtype  # Сохраняем тип данных для всех числовых полей
        
        self.current_id = 0  # Счетчик для генерации уникальных идентификаторов точек
        # Используется для присвоения ID новым точкам и отслеживания максимального ID
        
        # Создаем структурированный массив NumPy для хранения всех данных о точках
        # Используем структурированный массив вместо отдельных массивов для:
        # 1) Удобства доступа к данным по именам полей
        # 2) Эффективного хранения разнородных данных в одной структуре
        # 3) Возможности быстрой фильтрации по любому полю
        self.points = np.empty(0, dtype=[
            ('id', np.int32),           # Уникальный идентификатор точки (целое число)
            ('x', dtype),                # Координата X точки в пространстве
            ('y', dtype),                # Координата Y точки в пространстве
            ('bc_type', np.int8),        # Тип граничного условия (кодируется целым числом)
            ('bc_value', dtype),         # Числовое значение граничного условия
            ('density', dtype),           # Плотность жидкости/газа в точке (кг/м³)
            ('dynamic_viscosity', dtype), # Динамическая вязкость (Па·с)
            ('velocity_x', dtype),        # Компонента скорости по оси X (м/с)
            ('velocity_y', dtype),        # Компонента скорости по оси Y (м/с)
            ('pressure', dtype),          # Давление в точке (Па)
            ('sdf', dtype)                # Signed Distance Function - расстояние до ближайшей стенки (м)
        ])
        
        # Словарь для преобразования текстовых названий граничных условий в числовые коды
        # Используется для: 
        # 1) Удобства работы с граничными условиями через понятные имена
        # 2) Эффективного хранения в памяти (числа занимают меньше места)
        # 3) Быстрого поиска точек по типу ГУ
        self.bc_mapping = {
            'inlet': 0,           # Вход потока - задаются скорость, давление и т.д.
            'outlet': 1,          # Выход потока - обычно задается давление
            'wall': 2,            # Непроницаемая стенка - скорость = 0
            'symmetry': 3,        # Плоскость симметрии - нулевой поток через границу
            'axis_symmetry': 4,   # Осевая симметрия - для осесимметричных задач
            'domain': 5           # Внутренние точки расчетной области (не граница)
        }
        
        # Значения физических параметров по умолчанию для новых точек
        # Используются для инициализации, чтобы избежать неопределенных значений
        self.default_params = {
            'density': 1.225,              # Плотность воздуха при нормальных условиях (кг/м³)
            'dynamic_viscosity': 1.8e-5,   # Вязкость воздуха при 20°C (Па·с)
            'velocity_x': 1.0,              # Начальная скорость по X (м/с)
            'velocity_y': 0.0,              # Начальная скорость по Y (м/с)
            'pressure': 101325.0,            # Атмосферное давление (Па)
            'sdf': 0.0                       # Начальное значение SDF (будет пересчитано)
        }

    def add_points(self, coordinates: np.ndarray):
        """
        Добавляет новые точки в расчетную сетку.
        
        Аргументы:
            coordinates: массив координат формы (N, 2), где N - количество точек
        """
        # Проверяем тип входных данных для предотвращения ошибок
        if not isinstance(coordinates, np.ndarray):
            raise TypeError("Координаты должны быть numpy массивом")
            
        # Приводим координаты к нужному типу данных для единообразия
        if coordinates.dtype != self.dtype:
            coordinates = coordinates.astype(self.dtype)
            
        # Проверяем размерность массива (должен быть двумерным с двумя колонками)
        if coordinates.ndim != 2 or coordinates.shape[1] != 2:
            raise ValueError("Координаты должны быть двумерным массивом с формой (N, 2)")
        
        num_points = coordinates.shape[0]  # Получаем количество добавляемых точек
        
        # Определяем начальный ID для новых точек
        # Если массив points пуст, используем current_id, иначе берем длину массива
        start_id = len(self.points) if len(self.points) > 0 else self.current_id
        
        # Создаем новый массив структурированного типа для новых точек
        # Используем np.zeros для инициализации нулями (быстрее, чем создание пустого массива)
        new_points = np.zeros(num_points, dtype=self.points.dtype)
        
        # Заполняем координаты из входного массива
        new_points['x'] = coordinates[:, 0]  # Первая колонка - координаты X
        new_points['y'] = coordinates[:, 1]  # Вторая колонка - координаты Y
        
        # Генерируем уникальные ID для новых точек
        # Используем arange для создания последовательности ID
        new_points['id'] = np.arange(start_id, start_id + num_points, dtype=np.int32)
        self.current_id = start_id + num_points  # Обновляем счетчик ID
        
        # Устанавливаем значения физических параметров по умолчанию для всех новых точек
        # Это гарантирует, что у всех точек будут корректные начальные значения
        for param, value in self.default_params.items():
            new_points[param] = value
            
        # Добавляем новые точки к существующим
        if len(self.points) == 0:  # Если массив пуст
            self.points = new_points  # Просто присваиваем новый массив
        else:  # Если уже есть точки
            self.points = np.append(self.points, new_points)  # Конкатенируем массивы

    def set_boundary(self, indices: np.ndarray, bc_type: str, value: float):
        """
        Устанавливает граничные условия для указанных точек.
        
        Аргументы:
            indices: индексы точек в массиве points
            bc_type: строковое название граничного условия
            value: числовое значение граничного условия
        """
        # Проверяем корректность типа граничного условия
        if bc_type not in self.bc_mapping:
            raise ValueError(f"Неверный тип граничного условия: {bc_type}")
            
        # Проверяем тип индексов (должны быть массивом NumPy)
        if not isinstance(indices, np.ndarray):
            raise TypeError("Индексы должны быть numpy массивом")
            
        bc_code = self.bc_mapping[bc_type]  # Получаем числовой код ГУ
        
        # Устанавливаем тип ГУ для выбранных точек
        self.points[indices]['bc_type'] = bc_code
        # Устанавливаем значение ГУ (например, скорость на входе или давление на выходе)
        self.points[indices]['bc_value'] = value

    def calculate_sdf(self):
        """
        Вычисляет Signed Distance Function (SDF) - расстояние от каждой точки до ближайшей стенки.
        SDF важна для:
        1) Определения близости к границам
        2) Адаптации размера ячеек сетки
        3) Учета пристеночных эффектов в турбулентных моделях
        """
        # Находим индексы всех точек, которые являются стенками
        wall_indices = np.where(self.points['bc_type'] == self.bc_mapping['wall'])[0]
        
        if len(wall_indices) == 0:  # Если нет точек стенок
            warnings.warn("Нет точек стенок для расчёта SDF")
            return
        
        # Формируем массив координат точек стенок
        # Используем column_stack для объединения двух одномерных массивов в двумерный
        wall_points = np.column_stack((self.points['x'][wall_indices], 
                                       self.points['y'][wall_indices]))
        
        # Формируем массив координат всех точек
        all_points = np.column_stack((self.points['x'], self.points['y']))
        
        # Вычисляем матрицу расстояний между всеми точками и точками стенок
        # Используем scipy.spatial.distance.cdist для эффективного вычисления всех попарных расстояний
        dist_matrix = distance.cdist(all_points, wall_points, metric='euclidean')
        
        # Для каждой точки находим минимальное расстояние до стенок
        min_distances = np.min(dist_matrix, axis=1)
        
        # Сохраняем вычисленные расстояния в поле SDF
        self.points['sdf'] = min_distances.astype(self.dtype)
        
        # Для точек стенок устанавливаем SDF = 0 (расстояние до самой себя)
        self.points['sdf'][self.points['bc_type'] == self.bc_mapping['wall']] = 0.0

    def check_precision(self):
        """
        Проверяет корректность данных в массиве points.
        Выявляет потенциальные проблемы:
        - NaN значения (результат некорректных вычислений)
        - Inf значения (переполнение)
        - Некорректные физические значения (отрицательная вязкость или плотность)
        """
        # Проверяем наличие NaN в координатах (могут возникнуть при некорректных вычислениях)
        if np.isnan(self.points['x']).any() or np.isnan(self.points['y']).any():
            warnings.warn("Обнаружены NaN в координатах")
            
        # Проверяем наличие бесконечных значений (переполнение)
        if np.isinf(self.points['x']).any() or np.isinf(self.points['y']).any():
            warnings.warn("Обнаружены Inf в координатах")
            
        # Проверяем вязкость (должна быть положительной)
        if (self.points['dynamic_viscosity'] <= 0).any():
            warnings.warn("Обнаружены некорректные значения вязкости")
            
        # Проверяем плотность (должна быть положительной)
        if (self.points['density'] <= 0).any():
            warnings.warn("Обнаружены некорректные значения плотности")

    def save_to_file(self, filename: str):
        """
        Сохраняет все данные сетки в файл формата .npz.
        Использование npz позволяет сохранить несколько массивов в одном файле.
        
        Аргументы:
            filename: имя файла для сохранения
        """
        # Добавляем расширение .npz, если его нет
        if not filename.endswith('.npz'):
            filename += '.npz'
            
        try:
            # Сохраняем все необходимые данные для восстановления состояния
            np.savez(
                filename,
                points=self.points,              # Основные данные о точках
                bc_mapping=self.bc_mapping,      # Словарь типов ГУ
                dtype=self.dtype,                  # Тип данных
                default_params=self.default_params, # Параметры по умолчанию
                current_id=self.current_id         # Текущий ID для продолжения нумерации
            )
            print(f"Данные успешно сохранены в {filename}")
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
        # Проверяем существование файла
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Файл {filename} не найден")
            
        # Загружаем данные из файла
        data = np.load(filename, allow_pickle=True)
        
        # Создаем новый экземпляр с сохраненным типом данных
        manager = MeshManager(dtype=data['dtype'].item())
        
        # Восстанавливаем все поля
        manager.points = data['points']
        manager.bc_mapping = data['bc_mapping'].item()
        manager.default_params = data['default_params'].item()
        # Загружаем current_id или вычисляем из длины массива для обратной совместимости
        manager.current_id = data['current_id'].item() if 'current_id' in data else len(manager.points)
        
        return manager

    def get_sdf_stats(self):
        """
        Вычисляет статистические характеристики поля SDF.
        Полезно для анализа геометрии расчетной области.
        
        Возвращает:
            dict: словарь со статистиками
        """
        # Проверяем наличие поля SDF
        if 'sdf' not in self.points.dtype.fields:
            raise ValueError("Поле SDF не найдено в данных")
            
        sdf_values = self.points['sdf']  # Получаем все значения SDF
        
        # Возвращаем основные статистические показатели
        return {
            'min': np.min(sdf_values),      # Минимальное расстояние до стенки
            'max': np.max(sdf_values),      # Максимальное расстояние
            'mean': np.mean(sdf_values),    # Среднее расстояние
            'std': np.std(sdf_values)       # Стандартное отклонение (мера разброса)
        }

    def visualize_sdf(self):
        """
        Визуализирует поле SDF с помощью цветовой карты.
        Позволяет наглядно оценить распределение расстояний до стенок.
        """
        try:
            # Создаем новую фигуру с заданным размером
            plt.figure(figsize=(10, 8))
            
            # Создаем scatter plot (точечную диаграмму)
            # Каждая точка окрашивается в соответствии со значением SDF
            scatter = plt.scatter(
                self.points['x'],           # Координаты X
                self.points['y'],           # Координаты Y
                c=self.points['sdf'],       # Цвет в зависимости от SDF
                cmap='coolwarm',             # Цветовая карта (синий-красный)
                s=10,                         # Размер точек
                alpha=0.7                     # Прозрачность для лучшей видимости
            )
            
            # Добавляем цветовую шкалу
            plt.colorbar(scatter, label='Расстояние до стенки (м)')
            
            # Добавляем заголовок и подписи осей
            plt.title('Визуализация SDF (Signed Distance Function)')
            plt.xlabel('Координата X (м)')
            plt.ylabel('Координата Y (м)')
            
            # Добавляем сетку для лучшей читаемости
            plt.grid(True, alpha=0.3)
            
            # Автоматически подгоняем расположение элементов
            plt.tight_layout()
            
            # Отображаем график
            plt.show()
            
        except ImportError:
            warnings.warn("Для визуализации требуется установить matplotlib")
        except Exception as e:
            warnings.warn(f"Ошибка при визуализации: {e}")

    def get_boundary_points(self, bc_type: str):
        """
        Возвращает все точки с заданным типом граничного условия.
        Полезно для извлечения граничных узлов для задания условий.
        
        Аргументы:
            bc_type: строковое название граничного условия
            
        Возвращает:
            numpy.ndarray: структурированный массив точек с указанным ГУ
        """
        # Проверяем корректность типа ГУ
        if bc_type not in self.bc_mapping:
            raise ValueError(f"Неверный тип граничного условия: {bc_type}")
            
        bc_code = self.bc_mapping[bc_type]  # Получаем числовой код
        
        # Возвращаем отфильтрованный массив точек
        return self.points[self.points['bc_type'] == bc_code]

    def update_physical_properties(self, indices: np.ndarray, **kwargs):
        """
        Обновляет физические свойства для указанных точек.
        Использует именованные аргументы для гибкого обновления любых полей.
        
        Аргументы:
            indices: индексы точек для обновления
            **kwargs: пары "имя_поля=значение" для обновления
        """
        for key, value in kwargs.items():  # Перебираем все переданные параметры
            # Проверяем существование поля
            if key not in self.points.dtype.fields:
                raise ValueError(f"Неверное поле: {key}")
                
            # Обрабатываем разные типы значений
            if isinstance(value, (int, float)):
                # Скалярное значение - присваиваем всем указанным точкам
                self.points[indices][key] = value
            elif isinstance(value, np.ndarray):
                # Массив значений - проверяем размер и присваиваем поэлементно
                if len(value) != len(indices):
                    raise ValueError("Размер массива значений не соответствует количеству индексов")
                self.points[indices][key] = value
            else:
                raise TypeError(f"Неверный тип значения для {key}")

def Loader():
    """
    Загружает координаты точек из файла.
    Формат файла: каждая строка содержит X и Y координаты,
    разделенные табуляцией, пробелом или запятой.
    
    Возвращает:
        numpy.ndarray: массив координат формы (N, 2)
    """
    try:
        # Читаем путь к файлу с координатами из конфигурационного файла
        # Используем абсолютный путь для надежности
        with open("C:/Users/Ivan/source/repos/PINN_Flow2D/Config/paths.txt", 'r') as paths_file:
            lines = paths_file.readlines()  # Читаем все строки
            if len(lines) < 3:  # Проверяем, что файл содержит минимум 3 строки
                print("Ошибка: файл paths.txt содержит недостаточно строк")
                return np.array([])
            path = lines[2].strip()  # Третья строка содержит путь к файлу с координатами
            coordinates = []  # Список для накопления координат
    except FileNotFoundError:
        print("Ошибка: файл .../Config/paths.txt не найден")
        return np.array([])
    except Exception as e:
        print(f"Произошла ошибка при чтении файла paths.txt: {str(e)}")
        return np.array([])
    
    try:
        # Открываем файл с координатами
        with open(path, 'r') as file:
            # Читаем файл построчно
            for line_number, line in enumerate(file, start=0):
                line = line.strip()  # Удаляем пробельные символы в начале и конце
                if not line:  # Пропускаем пустые строки
                    continue
                    
                try:
                    # Пробуем разные возможные разделители координат
                    if '\t' in line:  # Разделитель - табуляция
                        x_str, y_str = line.split('\t')
                    elif ' ' in line:  # Разделитель - пробел
                        parts = line.split()
                        if len(parts) >= 2:
                            x_str, y_str = parts[0], parts[1]
                        else:
                            raise ValueError("Недостаточно данных в строке")
                    elif ',' in line:  # Разделитель - запятая
                        x_str, y_str = line.split(',')
                    else:
                        raise ValueError("Неизвестный формат разделителя")
                        
                    # Преобразуем строки в числа с плавающей точкой
                    x = float(x_str)
                    y = float(y_str)
                    
                    # Добавляем координаты в список
                    coordinates.append((x, y))
                    
                except ValueError as e:
                    print(f"Ошибка в строке {line_number}: некорректный формат данных - {e}")
                    
    except FileNotFoundError:
        print(f"Ошибка: файл по пути {path} не найден")
        return np.array([])
    except Exception as e:
        print(f"Произошла ошибка при чтении файла данных: {str(e)}")
        return np.array([])
    
    # Проверяем, что удалось загрузить хотя бы одну точку
    if len(coordinates) == 0:
        print("Предупреждение: не загружено ни одной точки")
        return np.array([])
    
    # Преобразуем список в массив NumPy для дальнейшей обработки
    return np.array(coordinates)

def setup_2d_mesh():
    """
    Основная функция настройки двумерной расчетной сетки.
    Демонстрирует полный цикл работы с MeshManager:
    1. Создание менеджера
    2. Загрузка/генерация координат
    3. Добавление точек
    4. Задание граничных условий
    5. Установка физических свойств
    6. Вычисление SDF
    7. Проверка и визуализация
    8. Сохранение/загрузка
    """
    # Создаем менеджер сетки с 32-битной точностью (экономия памяти)
    mesh = MeshManager(dtype=np.float32)
    
    # Загружаем координаты из файла
    coord = Loader()
    if len(coord) == 0:  # Если загрузка не удалась
        print("Ошибка: не удалось загрузить координаты. Использую тестовые данные.")
        # Генерируем тестовые данные для демонстрации
        x = np.random.rand(1000)  # 1000 случайных X от 0 до 1
        y = np.random.rand(1000)  # 1000 случайных Y от 0 до 1
        coord = np.column_stack((x, y))  # Объединяем в массив (1000, 2)
    
    print(f"Загружено {len(coord)} точек")
    
    # Добавляем точки в расчетную сетку
    mesh.add_points(coord)
    
    # Устанавливаем граничные условия
    # Левая граница (x < 0.1) - стенка
    left_wall = np.where(mesh.points['x'] < 0.1)[0]  # Находим индексы точек слева
    if len(left_wall) > 0:
        mesh.set_boundary(left_wall, 'wall', 0.0)  # Устанавливаем тип "стенка"
    
    # Правая граница (x > 0.9) - выход
    right_wall = np.where(mesh.points['x'] > 0.9)[0]
    if len(right_wall) > 0:
        mesh.set_boundary(right_wall, 'outlet', 0.0)  # Устанавливаем тип "выход"
    
    # Верхняя граница (y > 0.9) - симметрия
    top_wall = np.where(mesh.points['y'] > 0.9)[0]
    # Нижняя граница (y < 0.1) - симметрия
    bottom_wall = np.where(mesh.points['y'] < 0.1)[0]
    
    if len(top_wall) > 0:
        mesh.set_boundary(top_wall, 'symmetry', 0.0)  # Устанавливаем тип "симметрия"
    if len(bottom_wall) > 0:
        mesh.set_boundary(bottom_wall, 'symmetry', 0.0)
    
    # Обновляем физические свойства для внутренних точек (не на границах)
    internal_points = np.where(mesh.points['bc_type'] == mesh.bc_mapping['domain'])[0]
    if len(internal_points) > 0:
        mesh.update_physical_properties(
            internal_points,
            velocity_x=10.0,        # Задаем скорость потока 10 м/с
            pressure=101325.0,      # Атмосферное давление
            density=1.225            # Плотность воздуха
        )
    
    # Вычисляем расстояния до стенок (SDF)
    mesh.calculate_sdf()
    
    # Проверяем корректность данных (NaN, Inf, отрицательные значения)
    mesh.check_precision()
    
    # Получаем и выводим статистику SDF
    try:
        sdf_stats = mesh.get_sdf_stats()
        print("\nСтатистика SDF:")
        print(f"Минимальное: {sdf_stats['min']:.4f} м")
        print(f"Максимальное: {sdf_stats['max']:.4f} м")
        print(f"Среднее: {sdf_stats['mean']:.4f} м")
        print(f"Стандартное отклонение: {sdf_stats['std']:.4f} м")
    except Exception as e:
        print(f"Ошибка при получении статистики SDF: {e}")
    
    # Визуализируем результаты
    try:
        mesh.visualize_sdf()
    except Exception as e:
        print(f"Ошибка при визуализации: {e}")
    
    # Сохраняем результаты в файл
    mesh.save_to_file('mesh_data')
    
    # Загружаем данные обратно для проверки работоспособности
    try:
        loaded_mesh = MeshManager.load_from_file('mesh_data.npz')
        print("Данные успешно загружены из файла")
        print(f"Загружено {len(loaded_mesh.points)} точек")
    except Exception as e:
        print(f"Ошибка при загрузке данных: {e}")

# Точка входа в программу
if __name__ == "__main__":
    # Запускаем основную функцию при прямом вызове скрипта
    # Это позволяет использовать модуль как библиотеку (импортировать классы)
    # и как исполняемый скрипт
    setup_2d_mesh()