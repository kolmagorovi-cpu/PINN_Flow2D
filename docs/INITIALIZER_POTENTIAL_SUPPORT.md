# Этап 1.1: Поддержка потенциала скорости в Initializer

## Дата: 10 апреля 2026

## Резюме изменений

### Файл: `Initializer.py`

Добавлена полная поддержка инициализации потенциала скорости для PDE-системы `potential`.

---

## 1. Обновление структуры данных

### Изменение `point_dtype`

**Добавлено поле:**
```python
point_dtype = np.dtype([
    ('id', 'i4'),
    ('x_norm', 'f8'),
    ('y_norm', 'f8'),
    ('sdf_norm', 'f8'),
    ('boundary_type', 'U20'),
    ('on_boundary', 'bool'),
    ('p_norm', 'f8'),
    ('u_norm', 'f8'),
    ('v_norm', 'f8'),
    ('phi_norm', 'f8')  # НОВОЕ: Потенциал скорости
])
```

**Назначение:**
- Хранение нормализованного потенциала φ в выходном `.npy` файле
- Совместимость с существующим кодом (поле всегда присутствует)

---

## 2. Обновление `initialize_fields()`

### Новый параметр

```python
def initialize_fields(
    ...
    pde_system: str = 'incompressible_navier_stokes'  # НОВЫЙ параметр
) -> np.ndarray:
```

**Возможные значения:**
- `'incompressible_navier_stokes'` - стандартный режим (u, v, p)
- `'potential'` - режим потенциала скорости (φ)

---

## 3. Инициализация для потенциала

### Формула инициализации

Для режима `pde_system='potential'`:

```python
# Базовое поле: φ = U_inf_x * x
phi_phys = self.conditions.u * x

# Добавляем шум (10% от характерного значения)
phi_amp = 0.1 * self.U_inf * self.chord
phi_phys += phi_amp * (2 * np.random.random(N) - 1)

# Нормализация
phi_norm = phi_phys / (self.U_inf * self.chord)
```

### Физическое обоснование

1. **Базовое решение**: φ = U_inf * x соответствует однородному потоку
   - u = ∂φ/∂x = U_inf
   - v = ∂φ/∂y = 0

2. **Шум**: Небольшое возмущение (10%) помогает сети избежать локальных минимумов

3. **Нормализация**: Приведение к безразмерному виду для лучшей обусловленности

### Скорости и давление

Для совместимости с существующим кодом:

```python
# Скорости (приближённо, будут вычисляться как градиенты φ в PINN)
u_phys = U_inf * np.ones(N)
v_phys = V_inf * np.ones(N)

# Давление (не используется для потенциала)
p_phys = np.zeros(N)
```

---

## 4. Обновление метаданных

### Добавленные поля

```python
meta = {
    ...
    'pde_system': 'potential',  # Тип PDE системы
    'has_potential': True,       # Флаг наличия потенциала
}
```

**Назначение:**
- Позволяет downstream коду определить тип инициализации
- Автоматическая загрузка правильного типа модели

---

## 5. Примеры использования

### 5.1. Стандартный режим (Навье-Стокс)

```python
init = Initializer()
init.load_sampler_points()
init.select_fluid()
init.select_conditions()
init.initialize_fields(
    pde_system='incompressible_navier_stokes',  # по умолчанию
    velocity_initial_mode='constant'
)
init.save('initial_ns')
```

**Результат:**
- `u_norm = U_inf / U_inf = 1.0`
- `v_norm = 0.0`
- `p_norm = (p_inlet - p_ref) / (ρ * U_inf²)`
- `phi_norm = 0.0` (не используется)

### 5.2. Режим потенциала скорости

```python
init = Initializer()
init.load_sampler_points()
init.select_fluid()
init.select_conditions()
init.initialize_fields(
    pde_system='potential',  # Включаем режим потенциала
    velocity_perturbation_amplitude=0.1  # 10% шум
)
init.save('initial_potential')
```

**Результат:**
- `phi_norm = (U_inf * x + noise) / (U_inf * chord) = x/chord + noise`
- `u_norm = 1.0` (для совместимости)
- `v_norm = 0.0` (для совместимости)
- `p_norm = 0.0` (не используется)

---

## 6. Структура выходного файла

### Для потенциала

```
initial_potential.npy
├── id: int32              # Идентификатор точки
├── x_norm: float64        # x / chord
├── y_norm: float64        # y / chord
├── sdf_norm: float64      # SDF / chord
├── boundary_type: str     # 'inlet', 'wall', 'outlet', 'symmetry'
├── on_boundary: bool      # Флаг границы
├── p_norm: float64        # 0.0 (не используется)
├── u_norm: float64        # 1.0 (для совместимости)
├── v_norm: float64        # 0.0 (для совместимости)
└── phi_norm: float64      # Потенциал (x/chord + noise)
```

### Метаданные

```
initial_potential_metadata.npy
{
    'description': 'initial_potential',
    'pde_system': 'potential',
    'has_potential': True,
    'chord': 1.0,
    'U_inf': 1.0,
    'inlet_u_norm': 1.0,
    'inlet_v_norm': 0.0,
    ...
}
```

---

## 7. Рекомендации по использованию

### 7.1. Мягкие vs жёсткие BC

**Рекомендация:** Используйте мягкие BC для потенциала.

**Причина:**
- Условие ∂φ/∂n = 0 на стенке сложно реализовать жёстко
- Требует знания нормалей на этапе инициализации
- Мягкие BC гибче и проще в реализации

### 7.2. Уровень шума

**Рекомендуемые значения:**
- `velocity_perturbation_amplitude = 0.1` (10%) - стандартный
- `0.05` (5%) - для более гладкой инициализации
- `0.2` (20%) - для более разнообразного начального поля

### 7.3. Проверка качества

После инициализации рекомендуется:

```python
# Загрузить сохранённые данные
data = np.load('Initial/Data/initial_potential.npy')
meta = np.load('Initial/Data/initial_potential_metadata.npy', allow_pickle=True).item()

# Проверить наличие поля phi_norm
assert 'phi_norm' in data.dtype.names

# Проверить метаданные
assert meta['pde_system'] == 'potential'
assert meta['has_potential'] == True

# Визуализировать поле phi_norm
import matplotlib.pyplot as plt
plt.scatter(data['x_norm'], data['y_norm'], c=data['phi_norm'], s=1)
plt.colorbar(label='φ_norm')
plt.title('Начальное поле потенциала')
plt.show()
```

---

## 8. Известные ограничения

1. **phi_norm не используется для жёстких BC**
   - Если в будущем будут добавлены hard BC для потенциала, потребуется модификация

2. **Совместимость со старыми файлами**
   - Файлы, созданные до этого обновления, не содержат `phi_norm`
   - Код должен проверять наличие поля перед использованием

3. **Давление для потенциала**
   - `p_norm = 0` всегда, так как давление не входит в уравнение Лапласа

---

## 9. Интеграция с PINN

### Загрузка данных в PINN

```python
# В DataHandler или аналогичном компоненте
data = np.load('Initial/Data/initial_potential.npy')

if 'phi_norm' in data.dtype.names:
    # Режим потенциала
    phi_init = data['phi_norm']
    model.initialize_with_potential(phi_init)
else:
    # Стандартный режим
    u_init = data['u_norm']
    v_init = data['v_norm']
    p_init = data['p_norm']
    model.initialize_with_fields(u_init, v_init, p_init)
```

### Использование в trainer

```python
# В forward_pass trainer.py
if self.loss_computer.use_potential:
    phi = model(input_tensor)
    # phi уже содержит потенциал, скорости вычисляются через градиенты
else:
    out = model(input_tensor)
    u, v, p = out[:, 0], out[:, 1], out[:, 2]
```

---

## 10. Тестирование

### Тест 1: Проверка инициализации

```python
def test_potential_initialization():
    init = Initializer()
    init.load_sampler_points()
    init.select_fluid()
    init.select_conditions()
    
    result = init.initialize_fields(pde_system='potential')
    
    # Проверка наличия поля
    assert 'phi_norm' in result.dtype.names
    
    # Проверка значений (должно быть примерно x/chord)
    x_norm = result['x_norm']
    phi_norm = result['phi_norm']
    
    # Среднее отклонение должно быть < 0.1 (из-за шума)
    avg_diff = np.mean(np.abs(phi_norm - x_norm))
    assert avg_diff < 0.15  # 10% шум + запас
    
    print("✓ Тест пройден")
```

### Тест 2: Визуализация

```python
def visualize_potential_field():
    data = np.load('Initial/Data/initial_potential.npy')
    
    plt.figure(figsize=(10, 8))
    plt.scatter(data['x_norm'], data['y_norm'], 
                c=data['phi_norm'], s=5, cmap='viridis')
    plt.colorbar(label='φ (норм.)')
    plt.xlabel('x (норм.)')
    plt.ylabel('y (норм.)')
    plt.title('Начальное поле потенциала скорости')
    plt.axis('equal')
    plt.show()
```

---

## Авторы

- **Реализация**: Колмагоров И.Д.
- **Версия**: 1.1 (Этап 1.1 - Initializer)
- **Дата**: 10 апреля 2026
- **Базовая версия**: Airfoil PINN v3.260226
