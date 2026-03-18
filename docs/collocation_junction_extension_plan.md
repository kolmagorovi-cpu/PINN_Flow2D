# План расширения модуля Collocation.py для поддержки геометрии тройника

**Версия документа:** 1.1  
**Дата:** 2026-03-18  
**Автор:** BelchikovDA  
**Статус:** На утверждении

---

## 1. Обоснование расширения

### 1.1. Контекст

В рамках проекта Airfoil PINN возникла необходимость поддержки дополнительных типов геометрии:
- **Т-образный тройник** (плоский, бесконечный по оси Z)
- В перспективе: Y-образные тройники, крестообразные соединения

Текущая реализация модуля `Collocation.py` заточена под геометрию профиля крыла с внутренним телом (профилем). Геометрия тройника имеет принципиальные отличия:
- Отсутствие внутреннего тела (все стены — на внешней границе)
- Несколько границ одного типа (1 inlet + 2 outlet или наоборот)
- Прямолинейные границы (упрощает классификацию точек)

### 1.2. Существующая конфигурация

В проекте уже имеется файл конфигурации геометрии:

**Файл:** `Airfoil PINN/geom_config.yaml`

```yaml
geom_type: 'junction'
geom_name: 'VMFL010'
```

Параметр `geom_type` уже используется для выбора типа геометрии:
- `'junction'` — геометрия тройника
- `'airfoil'` — геометрия профиля крыла (по умолчанию)

Параметр `geom_name` ссылается на конкретную конфигурацию в файле `junction_geoms.yaml`.

### 1.3. Анализ альтернатив

| Критерий | Расширение Collocation.py | Новый модуль junction_collocation.py |
|----------|---------------------------|--------------------------------------|
| Переиспользование кода | 80% общего кода | Дублирование 80% кода |
| Срок реализации | 1-2 дня | 3-4 дня |
| Поддержка | Единая кодовая база | Два модуля для поддержки |
| Масштабируемость | Ограничена сложностью | Полная независимость |
| **Рекомендация** | ✅ **Предпочтительно** | ⚠️ При планах на 3D |

**Решение:** Расширить `Collocation.py` с разделением логики через параметр `geom_type` из `geom_config.yaml`.

---

## 2. Архитектурные изменения

### 2.1. Новая структура данных

#### 2.1.1. Расширение `BoundaryType`

```python
class BoundaryType(Enum):
    # Существующие типы
    INLET = "inlet"
    OUTLET = "outlet"
    WALL = "wall"
    SYMMETRY = "symmetry"
    AXIS_SYMMETRY = "axis_symmetry"
    INTERIOR = "interior"
    
    # Новые типы для тройника
    INLET_1 = "inlet_1"           # Нижний канал (основной inlet)
    OUTLET_SIDE = "outlet_side"   # Боковой outlet
    OUTLET_TOP = "outlet_top"     # Верхний outlet
```

#### 2.1.2. Расширение `SamplerConfig`

```python
@dataclass
class SamplerConfig:
    # ... существующие поля ...
    
    # Параметр загружается из geom_config.yaml
    geom_type: str = "airfoil"    # "airfoil" или "junction"
    geom_name: str = None         # Имя конфигурации (например, 'VMFL010')
    
    # Для тройника: именованные границы
    junction_config_path: str = None      # Путь к junction_geoms.yaml
    junction_params: Dict = None          # Загруженные параметры тройника
    
    # Назначение границ для тройника (автоматически определяется)
    junction_inlet_boundary: str = "bottom"   # "bottom", "left", "right", "top"
    junction_outlet_boundaries: List[str] = None  # ["top", "right"]
    
    # Флаг для отключения логики профиля
    has_internal_body: bool = True  # False для тройника
```

#### 2.1.3. Расширение структуры выходных данных

```python
# Для тройника добавляются поля:
dtype = [('id', 'i4'), ('x', 'f8'), ('y', 'f8'),
         ('boundary_type', 'U20'), 
         ('on_airfoil', 'bool'),      # Всегда False для тройника
         ('on_boundary', 'bool'),
         ('segment_id', 'i4'),
         ('channel_id', 'U20')]       # NEW: идентификатор канала
```

### 2.2. Загрузка конфигурации из geom_config.yaml

#### 2.2.1. Новый метод загрузки конфигурации

```python
def load_geometry_config(self, config_path: str = None) -> None:
    """
    Загрузка конфигурации геометрии из YAML-файла.
    
    Args:
        config_path: Путь к geom_config.yaml (по умолчанию: 
                     'Airfoil PINN/geom_config.yaml')
    """
    if config_path is None:
        config_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "Airfoil PINN", "geom_config.yaml"
        )
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            geom_config = yaml.safe_load(f)
        
        self.config.geom_type = geom_config.get('geom_type', 'airfoil')
        self.config.geom_name = geom_config.get('geom_name')
        
        logger.info(f"✅ Конфигурация геометрии загружена: {config_path}")
        logger.info(f"   geom_type: {self.config.geom_type}")
        logger.info(f"   geom_name: {self.config.geom_name}")
        
        # Для тройника — загрузка дополнительных параметров
        if self.config.geom_type == 'junction':
            self._load_junction_params()
        
    except Exception as e:
        logger.warning(f"⚠️ Не удалось загрузить geom_config.yaml: {e}")
        logger.info("   Используется конфигурация по умолчанию (airfoil)")

def _load_junction_params(self) -> None:
    """Загрузка параметров тройника из junction_geoms.yaml"""
    junctions_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "Airfoil PINN", "Geometry", "junction_geoms.yaml"
    )
    
    try:
        with open(junctions_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        # Поиск конфигурации по имени
        junctions = {j['name']: j for j in data.get('junctions', [])}
        
        if self.config.geom_name in junctions:
            self.config.junction_params = junctions[self.config.geom_name]
            self.channel_width = self.config.junction_params.get('chanel_width')
            logger.info(f"✅ Параметры тройника загружены: {self.config.geom_name}")
        else:
            logger.error(f"❌ Конфигурация '{self.config.geom_name}' не найдена")
            
    except Exception as e:
        logger.error(f"❌ Ошибка загрузки junction_geoms.yaml: {e}")
```

---

## 3. План модификации методов

### 3.0. Загрузка конфигурации в __init__()

**Новый метод:**

```python
def __init__(self, config: SamplerConfig):
    self.config = config
    self.points: List[CollocationPoint] = []
    self.points_interior: List[CollocationPoint] = []
    self.points_boundary: List[CollocationPoint] = []
    self.metadata = {}
    self.foil_metadata = {}
    self.next_id = 0
    self.foil_coords = None
    self.domain_coords = None
    self.foil_is_closed = False
    
    # Для тройника
    self.junction_params = None
    self.channel_width = None
    
    # Загрузка конфигурации геометрии
    self.load_geometry_config()
    
    os.makedirs(config.output_dir, exist_ok=True)
```

### 3.1. Метод `load_geometries()`

**Текущая логика:**
- Загрузка профиля и области из файлов
- Проверка замкнутости профиля

**Изменения:**

```python
def load_geometries(self) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Загрузка геометрии в зависимости от типа"""
    if self.config.geom_type == "junction":
        return self._load_junction_geometry()
    else:
        return self._load_airfoil_geometry()

def _load_junction_geometry(self) -> Tuple[np.ndarray, None]:
    """Загрузка геометрии тройника"""
    print("\n📂 Загрузка геометрии тройника...")
    
    # Поиск файла области
    domain_file = self._find_junction_domain_file()
    if domain_file is None:
        raise FileNotFoundError("Файл области тройника не найден")
    
    # Загрузка области
    domain = self._load_coords(domain_file)
    
    # Загрузка метаданных
    meta_file = self._find_junction_metadata_file()
    if meta_file:
        self.junction_params = np.load(meta_file, allow_pickle=True).item()
        self.channel_width = self.junction_params.get('chanel_width')
    
    # Для тройника профиль отсутствует
    self.foil_coords = None
    self.foil_is_closed = False
    
    self.output_base = os.path.join(self.config.output_dir, "collocation_points")
    return domain, None

def _load_airfoil_geometry(self) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Существующая логика для профиля (вынести в отдельный метод)"""
    # ... текущий код load_geometries() ...
```

### 3.2. Метод `sample_boundary_external()`

**Изменения:**

```python
def sample_boundary_external(self, contour: np.ndarray, step: float,
                              bounds: Dict) -> List[CollocationPoint]:
    """Генерация точек на внешнем контуре"""
    if self.config.geom_type == "junction":
        return self._sample_boundary_junction(contour, step)
    else:
        return self._sample_boundary_airfoil(contour, step, bounds)

def _sample_boundary_junction(self, contour: np.ndarray, 
                               step: float) -> List[CollocationPoint]:
    """
    Генерация точек на границах тройника.
    Все границы прямолинейные → равномерное распределение.
    """
    line = LineString(contour[:-1])  # Без замыкающей точки
    length = line.length
    num_points = int(length / step) + 1
    
    points = []
    for i in range(num_points):
        distance = i * step
        if distance <= length:
            pt = line.interpolate(distance)
            x, y = pt.x, pt.y
            
            # Классификация для тройника
            btype = self._classify_junction_boundary(x, y, contour)
            
            points.append(CollocationPoint(
                id=self._get_next_id(),
                x=x, y=y,
                boundary_type=btype,
                on_airfoil=False,
                on_boundary=True,
                segment_id=self._get_segment_id(btype)
            ))
    
    # Статистика
    type_counts = {}
    for p in points:
        type_counts[p.boundary_type] = type_counts.get(p.boundary_type, 0) + 1
    
    logger.info(f"✅ Границы тройника: {len(points)} точек")
    for bt, cnt in type_counts.items():
        logger.info(f"   {bt.value}: {cnt}")
    
    return points

def _classify_junction_boundary(self, x: float, y: float, 
                                 contour: np.ndarray) -> BoundaryType:
    """
    Определение типа границы для тройника.
    
    Логика для стандартного T-тройника:
    - Bottom edge (y = y_min) → INLET_1
    - Top edge (y = y_max) → OUTLET_TOP
    - Right edge (x = x_max) → OUTLET_SIDE
    - Остальные → WALL
    """
    # Получаем границы контура
    y_min = np.min(contour[:, 1])
    y_max = np.max(contour[:, 1])
    x_max = np.max(contour[:, 0])
    
    tol = 0.001  # Допуск для классификации
    
    if abs(y - y_min) < tol:
        return BoundaryType.INLET_1
    elif abs(y - y_max) < tol:
        return BoundaryType.OUTLET_TOP
    elif abs(x - x_max) < tol:
        return BoundaryType.OUTLET_SIDE
    else:
        return BoundaryType.WALL

def _sample_boundary_airfoil(self, contour: np.ndarray, step: float,
                              bounds: Dict) -> List[CollocationPoint]:
    """Существующая логика для профиля"""
    # ... текущий код sample_boundary_external() ...
```

### 3.3. Метод `sample_boundary_foil()`

**Изменения:**

```python
def sample_boundary_foil(self, foil: np.ndarray, step: float) -> List[CollocationPoint]:
    """Генерация точек на профиле"""
    if self.config.geom_type == "junction":
        # Для тройника профиль отсутствует
        logger.info("ℹ️ Профиль отсутствует (геометрия тройника)")
        return []
    else:
        # Существующая логика
        return self._sample_boundary_foil_airfoil(foil, step)

def _sample_boundary_foil_airfoil(self, foil: np.ndarray, 
                                   step: float) -> List[CollocationPoint]:
    """Существующая логика для профиля (вынести в отдельный метод)"""
    # ... текущий код ...
```

### 3.4. Метод `sample_bottom_edge()`

**Изменения:**

```python
def sample_bottom_edge(self, x_min_dom: float, x_min_foil: float,
                       y_min_dom: float, step: float) -> List[CollocationPoint]:
    """Генерация точек на нижней границе слева от профиля"""
    if self.config.geom_type == "junction":
        # Для тройника не применяется
        return []
    else:
        # Существующая логика
        return self._sample_bottom_edge_airfoil(
            x_min_dom, x_min_foil, y_min_dom, step)
```

### 3.5. Методы генерации внутренних точек

#### 3.5.1. `sample_interior_regular_*()`

**Изменения:** Минимальные — логика универсальна.

```python
def sample_interior_regular_sequential(self, domain: np.ndarray, 
                                        step: float) -> List[CollocationPoint]:
    """Генерация регулярных внутренних точек"""
    # Существующая логика работает для обеих геометрий
    # Единственное отличие: для тройника нет профиля для исключения
    return self._sample_interior_regular_generic(domain, step)
```

#### 3.5.2. Отключение `adaptive_foil`

```python
def generate(self, method: str = None, **kwargs) -> Dict[str, np.ndarray]:
    """Генерация всех точек коллокации"""
    method = method or self.config.sampling_method
    
    # Проверка совместимости метода с геометрией
    if self.config.geom_type == "junction" and method == "adaptive_foil":
        logger.warning("⚠️ Метод 'adaptive_foil' недоступен для тройника. "
                      "Используется 'adaptive_all'.")
        method = "adaptive_all"
    
    # ... остальная логика ...
```

### 3.6. Новый метод `_get_segment_id()`

```python
def _get_segment_id(self, boundary_type: BoundaryType) -> int:
    """
    Возвращает идентификатор сегмента для типа границы.
    
    Сегменты для тройника:
    0 → INLET_1 (нижний канал)
    1 → OUTLET_TOP (верхний канал)
    2 → OUTLET_SIDE (боковой канал)
    3+ → WALL (стены)
    """
    segment_map = {
        BoundaryType.INLET_1: 0,
        BoundaryType.OUTLET_TOP: 1,
        BoundaryType.OUTLET_SIDE: 2,
        BoundaryType.WALL: 3,
    }
    return segment_map.get(boundary_type, 99)
```

### 3.7. Метод `generate()` — главная координация

**Изменения:**

```python
def generate(self, method: str = None, **kwargs) -> Dict[str, np.ndarray]:
    """
    Генерация всех точек коллокации.
    
    Args:
        method: метод генерации ('regular', 'irregular', 'adaptive_foil', 'adaptive_all')
        **kwargs: дополнительные параметры
    
    Returns:
        Dict с массивами точек
    """
    print("\n" + "="*60)
    print("ГЕНЕРАЦИЯ ТОЧЕК КОЛЛОКАЦИИ".center(60))
    print("="*60)
    print(f"Тип геометрии: {self.config.geom_type}")
    
    # Загрузка геометрии
    domain, foil = self.load_geometries()
    
    # Вычисление шага
    step = self._compute_step(domain)
    bounds = self._compute_bounds(domain)
    
    # Генерация граничных точек
    if self.config.geom_type == "junction":
        boundary_points = self.sample_boundary_junction(domain, step)
        foil_points = []
        bottom_points = []
    else:
        boundary_points = self.sample_boundary_external(domain, step, bounds)
        foil_points = self.sample_boundary_foil(foil, step) if foil is not None else []
        bottom_points = self.sample_bottom_edge(...) if needed else []
    
    # Генерация внутренних точек
    interior_points = self._generate_interior(domain, method, step, foil)
    
    # Объединение и сохранение
    return self._save_all_points(boundary_points, foil_points, 
                                  interior_points, bottom_points)
```

### 3.8. Метод `_save_all_points()` и метаданные

**Изменения:**

```python
def _save_all_points(self, boundary_points, foil_points, 
                      interior_points, bottom_points) -> Dict[str, np.ndarray]:
    """Сохранение всех точек с метаданными"""
    
    # ... существующая логика ...
    
    # Сохранение метаданных
    metadata = {
        'geom_type': self.config.geom_type,
        'sampling_method': self.config.sampling_method,
        'total_points': total_count,
        'boundary_points': len(boundary_points),
        'interior_points': len(interior_points),
        'step_size': step,
        # Специфично для тройника
        'junction_params': self.junction_params,
        'channel_width': self.channel_width,
        'inlet_points': len([p for p in boundary_points 
                            if p.boundary_type == BoundaryType.INLET_1]),
        'outlet_points': len([p for p in boundary_points 
                             if p.boundary_type in (BoundaryType.OUTLET_TOP, 
                                                    BoundaryType.OUTLET_SIDE)]),
        'wall_points': len([p for p in boundary_points 
                           if p.boundary_type == BoundaryType.WALL]),
    }
    
    meta_path = f"{self.output_base}_metadata.npy"
    np.save(meta_path, metadata)
    logger.info(f"✅ Метаданные сохранены: {meta_path}")
```

---

## 4. Изменение главной функции `s2dr()`

### 4.1. Автоматическая загрузка из geom_config.yaml

```python
def s2dr() -> None:
    """Главная функция модуля дискретизации"""
    print('\033c')
    print("="*60)
    print("МОДУЛЬ ДИСКРЕТИЗАЦИИ S2DR".center(60))
    print("="*60)
    
    # Конфигурация
    config = SamplerConfig(
        sampling_method="regular",
        use_multiprocessing=True,
        num_processes=4,
        # ... остальные параметры ...
    )
    
    sampler = CollocationPoints(config)
    
    # Конфигурация загружается автоматически из geom_config.yaml
    # в __init__() через load_geometry_config()
    
    print(f"\n📐 Тип геометрии: {sampler.config.geom_type}")
    print(f"   Конфигурация: {sampler.config.geom_name or 'default'}")
    
    # Для тройника — отображение параметров
    if sampler.config.geom_type == "junction":
        print(f"\n📐 Параметры тройника:")
        print(f"   Ширина канала: {sampler.channel_width} м")
        # ... другие параметры ...
    
    # Генерация
    results = sampler.generate()
    
    # Визуализация
    if config.plot_results:
        sampler.plot_results(results)
    
    print("\n" + "="*60)
    input("Нажмите Enter для выхода...")
```

### 4.2. Опциональный интерактивный выбор

```python
# Если требуется переопределить геометрию из конфига:
print("\n📐 Использовать геометрию из geom_config.yaml?")
print("   1. Да (по умолчанию)")
print("   2. Нет, выбрать вручную")

choice = input("\nВаш выбор: ").strip()
if choice != "1":
    print("\nВыберите тип геометрии:")
    print("   1. Профиль крыла (airfoil)")
    print("   2. Т-образный тройник (junction)")
    
    geom_choice = input("\nВаш выбор: ").strip()
    config.geom_type = "airfoil" if geom_choice == "1" else "junction"
```

---

## 5. Визуализация

### 5.1. Расширение `plot_results()`

```python
def plot_results(self, results: Dict[str, np.ndarray]) -> None:
    """Визуализация точек коллокации"""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # ... существующая логика ...

    # Подписи границ для тройника
    if self.config.geom_type == "junction":
        self._annotate_junction_boundaries(axes[0, 0])

    plt.tight_layout()
    plt.show()

def _annotate_junction_boundaries(self, ax) -> None:
    """Аннотации границ тройника"""
    ax.text(..., "INLET", color='green', fontsize=10)
    ax.text(..., "OUTLET (top)", color='orange', fontsize=10)
    ax.text(..., "OUTLET (side)", color='orange', fontsize=10)
    ax.text(..., "WALL", color='red', fontsize=10)
```

---

## 6. Тестирование

### 6.1. Юнит-тесты

| Тест | Описание |
|------|----------|
| `test_load_geometry_config()` | Загрузка конфигурации из geom_config.yaml |
| `test_load_junction_params()` | Загрузка параметров тройника |
| `test_load_junction_geometry()` | Загрузка геометрии тройника |
| `test_classify_junction_boundary()` | Классификация границ тройника |
| `test_sample_boundary_junction()` | Генерация граничных точек |
| `test_generate_junction_regular()` | Полная генерация (regular) |
| `test_generate_junction_irregular()` | Полная генерация (irregular) |
| `test_adaptive_foil_disabled()` | Проверка отключения adaptive_foil |

### 6.2. Интеграционные тесты

1. **Полный цикл для тройника:**
   - Geometry → Collocation → Initializer
   - Проверка корректности SDF
   - Проверка one-hot кодирования границ

2. **Совместимость с solver.py:**
   - Загрузка точек в `PINNDataset`
   - Корректность масок для BC
   - Обучение на геометрии тройника

### 6.3. Тестовые конфигурации

```yaml
# geom_config.yaml
geom_type: 'junction'
geom_name: 'VMFL010'

# test_junction_config.yaml
project_name: "T_junction_test"
geom_type: "junction"
geom_name: "VMFL010"
sampling:
  method: "regular"
  step_size: 0.01
  boundary_density: 1.0
  interior_density: 1.0
```

---

## 7. Этапы реализации

### Этап 1: Подготовка (0.5 дня)

- [ ] Создать копию `Collocation.py` для резервирования
- [ ] Изучить текущую структуру кода
- [ ] Подготовить тестовые данные (геометрия тройника)
- [ ] Проверить наличие `geom_config.yaml` и `junction_geoms.yaml`

### Этап 2: Рефакторинг (1 день)

- [ ] Выделить `_load_airfoil_geometry()` из `load_geometries()`
- [ ] Выделить `_sample_boundary_airfoil()` из `sample_boundary_external()`
- [ ] Вынести `_sample_boundary_foil_airfoil()` из `sample_boundary_foil()`
- [ ] Добавить параметры `geom_type`, `geom_name` в `SamplerConfig`
- [ ] Реализовать метод `load_geometry_config()`
- [ ] Реализовать метод `_load_junction_params()`

### Этап 3: Реализация для тройника (1 день)

- [ ] Реализовать `_load_junction_geometry()`
- [ ] Реализовать `_sample_boundary_junction()`
- [ ] Реализовать `_classify_junction_boundary()`
- [ ] Реализовать `_get_segment_id()`
- [ ] Обновить `generate()` для поддержки тройника
- [ ] Обновить метаданные

### Этап 4: Интеграция и тестирование (0.5 дня)

- [ ] Обновить `s2dr()` для автоматической загрузки конфигурации
- [ ] Обновить визуализацию
- [ ] Провести юнит-тесты
- [ ] Провести интеграционный тест

### Этап 5: Документирование (0.5 дня)

- [ ] Обновить docstring методов
- [ ] Добавить примеры использования
- [ ] Обновить `context.md`

**Итого:** 3.5 рабочих дня

---

## 8. Риски и меры mitigation

| Риск | Вероятность | Влияние | Меры |
|------|-------------|---------|------|
| Нарушение обратной совместимости | Средняя | Высокое | Тщательное тестирование airfoil-режима |
| Ошибки классификации границ | Средняя | Среднее | Визуальная верификация, юнит-тесты |
| Проблемы с многопроцессорностью | Низкая | Высокое | Тестирование на Windows/Linux |
| Несовместимость с Initializer | Средняя | Высокое | Интеграционное тестирование |

---

## 9. Перспективы расширения

### 9.1. Будущие геометрии

На основе созданной архитектуры легко добавить:

1. **Y-образные тройники**
   - Добавить типы границ `OUTLET_BRANCH_1`, `OUTLET_BRANCH_2`
   - Расширить `_classify_junction_boundary()`

2. **Крестообразные соединения**
   - 4 границы inlet/outlet
   - Сложная логика классификации

3. **Криволинейные каналы**
   - Использовать `adaptive_all` с кривыми расстояниями

### 9.2. 3D-расширение

При планах на 3D-тройники рассмотреть создание отдельного модуля `Collocation3D.py`.

---

## 10. Приложения

### Приложение A. Существующие конфигурационные файлы

**geom_config.yaml** (основной файл конфигурации геометрии):

```yaml
geom_type: 'junction'
geom_name: 'VMFL010'
```

**junction_geoms.yaml** (библиотека конфигураций тройников):

```yaml
junctions:
  - name: "VMFL010"
    chanel_width: 0.1        # м
    down_length: 0.5         # м (inlet)
    side_length: 0.3         # м (outlet)
    top_length: 0.4          # м (outlet)
    down2side_angl: 90       # град
    side2top_angl: 90        # град
    
  - name: "T_junction_01"
    chanel_width: 0.05
    down_length: 0.3
    side_length: 0.2
    top_length: 0.25
    down2side_angl: 90
    side2top_angl: 90
```

### Приложение B. Пример вызова

```python
# Автоматическая загрузка из geom_config.yaml
config = SamplerConfig(
    sampling_method="regular",
    step_size=0.005,
    boundary_point_density=1.5,
    interior_point_density=1.0
)

sampler = CollocationPoints(config)
# Конфигурация загружается автоматически в __init__()

results = sampler.generate()

# Выходные файлы:
# - collocation_points.npy
# - collocation_points_boundary.npy
# - collocation_points_interior.npy
# - collocation_points_metadata.npy
```

### Приложение C. Диаграмма классов

```
┌─────────────────────────────────────────────────────────┐
│                  CollocationPoints                      │
├─────────────────────────────────────────────────────────┤
│ - config: SamplerConfig                                 │
│ - geom_type: str (из geom_config.yaml)                  │
│ - geom_name: str (из geom_config.yaml)                  │
│ - junction_params: Dict (из junction_geoms.yaml)        │
│ - points: List[CollocationPoint]                        │
├─────────────────────────────────────────────────────────┤
│ + __init__()                                            │
│   └─ load_geometry_config()                             │
│       ├─ Загрузка geom_config.yaml                      │
│       └─ _load_junction_params()                        │
│ + load_geometries()                                     │
│   ├─ _load_airfoil_geometry()                           │
│   └─ _load_junction_geometry()                          │
│ + generate()                                            │
│ + sample_boundary_external()                            │
│   ├─ _sample_boundary_airfoil()                         │
│   └─ _sample_boundary_junction()                        │
│ + sample_boundary_foil()                                │
│ + sample_interior_*()                                   │
└─────────────────────────────────────────────────────────┘
```

---

## 11. История изменений

| Версия | Дата | Автор | Изменения |
|--------|------|-------|-----------|
| 1.0 | 2026-03-18 | BelchikovDA | Первоначальная версия |
| 1.1 | 2026-03-18 | BelchikovDA | Интеграция с существующим geom_config.yaml |

---

**Документ утверждён:**

_____________________ / _________________ / 2026

(подпись) (ФИО)
