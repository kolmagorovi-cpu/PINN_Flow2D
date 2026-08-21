# junction_geom.py — Модуль геометрии тройников

**Версия:** 3.260226  
**Автор:** Колмагоров И.Д.  
**Назначение:** Построение расчётной области для плоских T-образных и Y-образных тройников

---

## Описание

Модуль `junction_geom.py` предоставляет функционал для работы с геометрией плоских тройников (бесконечных по оси Z). Модуль позволяет задавать конфигурацию тройника через YAML-файлы, строить расчётную область и визуализировать геометрию.

Все координаты хранятся и обрабатываются в метрах без нормализации.

---

## Область применения

Модуль предназначен для моделирования течений в:
- T-образных тройниках (углы 90°/90°)
- Y-образных тройниках (углы 120°/120°)
- Других конфигурациях с тремя каналами, сходящимися в одной точке

---

## Классы

### GeometryConfig

Класс конфигурации для геометрических параметров тройника.

| Атрибут | Тип | По умолчанию | Описание |
|---------|-----|--------------|----------|
| `enable_input` | `bool` | `False` | Включить интерактивный ввод путей |
| `save_format` | `str` | `'npy'` | Формат сохранения |
| `plot_enabled` | `bool` | `True` | Включена ли визуализация |
| `show_geom` | `bool` | `True` | Показывать ли график сразу |
| `plot_dpi` | `int` | `300` | DPI для сохранения графиков |
| `output_dir` | `str` | `"Geometry/Data"` | Директория для выходных файлов |

---

### JunctionGeometry

Основной класс для работы с геометрией тройника.

#### Методы

##### `load_junction(geom_config_path, junctions_config_path) → bool`

Загружает конфигурацию тройника из YAML-файлов.

**Структура файлов:**

`geom_config.yaml`:
```yaml
geom_type: 'junction'
geom_name: 'VMFL010'
```

`junction_geoms.yaml`:
```yaml
junctions:
  - name: 'VMFL010'
    chanel_width: 1
    down_length: 2
    side_length: 3
    top_length: 3
    top2side_angl: 90
    side2down_angl: 90
```

---

##### `create_domain() → None`

Создаёт замкнутый контур расчётной области.

**Алгоритм построения (обход против часовой стрелки):**

| № | Точка | Координаты |
|---|-------|------------|
| 1 | inlet_1 | `(-w/2, -(w/2 + L_down))` |
| 2 | inlet_2 | `(w/2, -(w/2 + L_down))` |
| 3 | wall_1 | `(w/2, -w/2)` |
| 4 | outlet_side_1 | `(w/2 + L_side, -w/2)` |
| 5 | outlet_side_2 | `(w/2 + L_side, w/2)` |
| 6 | wall_2 | `(w/2, w/2)` |
| 7 | outlet_top_1 | `(w/2, w/2 + L_top)` |
| 8 | outlet_top_2 | `(-w/2, w/2 + L_top)` |
| 9 | замыкание | `inlet_1` |

*Обозначения: w = chanel_width, L_down = down_length, L_side = side_length, L_top = top_length*

---

##### `show_domain(show_plot, title, save_path) → None`

Визуализирует расчётную область.

---

##### `save(filename, format) → None`

Сохраняет геометрию в файлы:
- `{filename}_domain.{fmt}` — координаты области
- `{filename}_metadata.npy` — метаданные

---

##### `print_stats() → None`

Выводит информацию о контуре в консоль.

---

## Примеры конфигураций

### T-образный тройник VMFL010

```yaml
- name: 'VMFL010'
  chanel_width: 1
  down_length: 2
  side_length: 3
  top_length: 3
  top2side_angl: 90
  side2down_angl: 90
```

### Y-образный тройник Y-60

```yaml
- name: 'Y-60'
  chanel_width: 1
  down_length: 2
  side_length: 2
  top_length: 2
  top2side_angl: 120
  side2down_angl: 120
```

---

## Выходные файлы

| Файл | Формат | Описание |
|------|--------|----------|
| `junction_domain.npy` | `(N, 2)` | Координаты вершин области |
| `junction_metadata.npy` | `dict` | `num_domain_points` |

---

## Пример использования

```python
from junction_geom import JunctionGeometry, GeometryConfig

config = GeometryConfig(show_geom=True)
geom = JunctionGeometry(config)

geom.load_junction('geom_config.yaml')
geom.create_domain()
geom.show_domain(title="T-образный тройник")
geom.save('junction_VMFL010')
```

---

## Ограничения

1. Только 2D плоские тройники
2. Все каналы одинаковой ширины
3. Углы не влияют на построение (зарезервировано)

---

## См. также

- `Geometry.py` — модуль геометрии профилей
- `docs/Geometry_module.md` — документация Geometry.py
- `docs/project_context.md` — общая документация
