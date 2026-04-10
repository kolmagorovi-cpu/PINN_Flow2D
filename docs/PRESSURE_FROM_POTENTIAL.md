# Этап 1.1: Давление из потенциала (уравнение Бернулли)

## Дата: 10 апреля 2026

## Резюме

Добавлено вычисление давления из потенциала скорости по уравнению Бернулли как постпроцессинг для визуализации. **Давление не входит в функцию потерь на этапе 1.1.**

---

## 1. Уравнение Бернулли

### Для несжимаемой жидкости

```
p = p_inf + 0.5 * ρ * (U_inf² - |V|²)
```

где:
- `p` — статическое давление [Па]
- `p_inf` — давление на бесконечности [Па]
- `ρ` — плотность жидкости [кг/м³]
- `U_inf` — скорость на бесконечности [м/с]
- `|V|² = u² + v²` — квадрат модуля скорости
- `u = ∂φ/∂x`, `v = ∂φ/∂y` — скорости из потенциала

### Коэффициент давления Cp

```
Cp = (p - p_inf) / (0.5 * ρ * U_inf²) = 1 - |V|²/U_inf²
```

**Характерные значения:**
- `Cp = 1` в точке торможения (u=v=0)
- `Cp = 0` в свободном потоке (|V|=U_inf)
- `Cp < 0` в зонах ускорения (|V|>U_inf)

---

## 2. Реализация в `pde_residuals.py`

### Функция `compute_pressure_from_potential()`

```python
@staticmethod
def compute_pressure_from_potential(
    phi: torch.Tensor,         # Потенциал скорости
    x: torch.Tensor,           # Координата x
    y: torch.Tensor,           # Координата y
    U_inf: float,              # Скорость на бесконечности
    p_inf: float,              # Давление на бесконечности
    rho: float,                # Плотность
    create_graph: bool = False # Для обратного распространения?
) -> dict:
    """
    Возвращает:
        'pressure': p [Па]
        'u': скорость u [м/с]
        'v': скорость v [м/с]
        'velocity_squared': |V|²
        'dynamic_pressure': q = 0.5 * ρ * |V|²
        'pressure_coefficient': Cp
    """
```

### Особенности
- ✅ Использует `torch.autograd.grad` для точного вычисления скоростей
- ✅ Параметр `create_graph` для возможного использования в loss (этапы 1.2–1.3)
- ✅ Для визуализации: `create_graph=False` (быстрее, меньше памяти)

---

## 3. Визуализация в `visualizer.py`

### Метод `plot_pressure_from_potential()`

```python
def plot_pressure_from_potential(
    pressure_norm: torch.Tensor,  # Давление
    cp_norm: torch.Tensor,        # Коэффициент Cp
    x_norm: torch.Tensor,         # Координаты
    y_norm: torch.Tensor,
    data_handler,                 # Метаданные
    savedir: str,
    epoch: Optional[int] = None,
    plot_physical: bool = True,
    config: Optional[Dict] = None
) -> None:
```

### Выходные файлы
1. **PNG**: `pressure_from_potential_epoch_N.png`
   - 2 субплота: давление p и коэффициент Cp
   - Colormap `RdBu_r` (расходящаяся)
   
2. **NPY**: `pressure_from_potential_epoch_N.npy`
   - Словарь с данными полей

---

## 4. Примеры использования

### 4.1. Вычисление давления

```python
from pde_residuals import PDEResiduals

# После обучения
phi = model(input_tensor)  # Потенциал

# Вычисляем давление (только для визуализации)
pressure_data = PDEResiduals.compute_pressure_from_potential(
    phi=phi,
    x=x,
    y=y,
    U_inf=1.0,      # м/с
    p_inf=101325.0, # Па
    rho=1.225,      # кг/м³ (воздух)
    create_graph=False  # Не нужно для визуализации
)

# Доступ к результатам
p = pressure_data['pressure']           # Па
cp = pressure_data['pressure_coefficient']
u = pressure_data['u']                  # м/с
v = pressure_data['v']                  # м/с
```

### 4.2. Визуализация

```python
from visualizer import Visualizer

vis = Visualizer(config_name='potential_flow')

vis.plot_pressure_from_potential(
    pressure_norm=pressure_data['pressure'],
    cp_norm=pressure_data['pressure_coefficient'],
    x_norm=x_norm,
    y_norm=y_norm,
    data_handler=data_handler,
    savedir='results/potential',
    epoch=1000,
    plot_physical=True
)
```

### 4.3. В диагностическом цикле solver.py

```python
# В _run_diagnostics()
if self.config['pde']['system'] == 'potential':
    # Получаем предсказания
    with torch.no_grad():
        phi = self.model(input_tensor)
        
        # Вычисляем давление
        pressure_data = PDEResiduals.compute_pressure_from_potential(
            phi=phi,
            x=x,
            y=y,
            U_inf=self.data_handler.U_inf,
            p_inf=self.conditions.outlet_pressure,
            rho=self.data_handler.rho,
            create_graph=False
        )
        
        # Визуализируем
        self.visualizer.plot_pressure_from_potential(
            pressure_norm=pressure_data['pressure'],
            cp_norm=pressure_data['pressure_coefficient'],
            x_norm=x_norm,
            y_norm=y_norm,
            data_handler=self.data_handler,
            savedir=self._get_plots_dir(),
            epoch=epoch
        )
```

---

## 5. Физическая интерпретация

### Зоны на графике давления

1. **Область высокого давления (Cp > 0)**
   - Передняя кромка профиля (точка торможения)
   - Cp ≈ 1 в критической точке

2. **Область низкого давления (Cp < 0)**
   - Верхняя поверхность профиля (ускорение потока)
   - Cp может достигать -2...-3

3. **Свободный поток (Cp ≈ 0)**
   - Вдали от профиля
   - |V| ≈ U_inf

### Проверка качества решения

- **Симметрия**: При α=0 давление должно быть симметричным
- **Асимптотика**: Вдали от профиля Cp → 0
- **Точка торможения**: Cp ≈ 1 на передней кромке

---

## 6. Производительность

### Время вычисления
- O(N) для N точек коллокации
- Быстрее, чем вычисление невязок Лапласа (нет вторых производных)

### Память
- `create_graph=False`: минимальная (нет графа)
- `create_graph=True`: +O(N) для хранения графа

### Рекомендация
- Для визуализации: вызывать каждые N эпох (например, 50-100)
- Не вызывать на каждой эпохе (замедляет обучение)

---

## 7. Интеграция с существующим кодом

### Совместимость
- ✅ Не влияет на функцию потерь
- ✅ Не требует изменений в trainer.py
- ✅ Вызывается только при необходимости

### Зависимости
- ✅ `torch.autograd.grad` (уже используется)
- ✅ `matplotlib` (уже есть)
- ✅ `numpy` (уже есть)

---

## 8. Будущие этапы

### Этап 1.2–1.3: Давление в функции потерь

Возможное использование:

```python
# Регуляризация Бернулли
loss_bernoulli = mean((p_computed - p_target)²)

# Или через невязки от CFD (Fluent)
loss_cfd = mean((p_PINN - p_Fluent)²)
```

### Параметр create_graph

```python
# Для обучения (нужны градиенты)
pressure_data = compute_pressure_from_potential(
    phi, x, y, U_inf, p_inf, rho,
    create_graph=True  # Важно!
)

# Теперь pressure_data['pressure'] можно использовать в loss
loss_pressure = mean((pressure_data['pressure'] - p_target)²)
```

---

## 9. Тестирование

### Тест 1: Проверка уравнения Бернулли

```python
def test_bernoulli_pressure():
    # Однородный поток: φ = U_inf * x
    x = torch.linspace(0, 1, 100, requires_grad=True).view(-1, 1)
    y = torch.zeros(100, 1, requires_grad=True)
    phi = 1.0 * x  # U_inf = 1.0
    
    result = PDEResiduals.compute_pressure_from_potential(
        phi, x, y,
        U_inf=1.0, p_inf=101325.0, rho=1.225,
        create_graph=False
    )
    
    # В однородном потоке Cp = 0
    cp = result['pressure_coefficient']
    assert torch.allclose(cp, torch.zeros_like(cp), atol=1e-6)
    
    print("✓ Тест пройден: Cp = 0 в однородном потоке")
```

### Тест 2: Точка торможения

```python
def test_stagnation_point():
    # В точке торможения u=v=0, Cp=1
    # Создадим искусственную точку
    x = torch.tensor([[0.0]], requires_grad=True)
    y = torch.tensor([[0.0]], requires_grad=True)
    phi = torch.tensor([[0.0]], requires_grad=True)  # const
    
    # Градиенты будут нулевыми
    result = PDEResiduals.compute_pressure_from_potential(
        phi, x, y,
        U_inf=1.0, p_inf=101325.0, rho=1.225,
        create_graph=False
    )
    
    # Cp должен быть ≈ 1
    cp = result['pressure_coefficient'].item()
    assert abs(cp - 1.0) < 1e-6
    
    print(f"✓ Тест пройден: Cp = {cp:.6f} в точке торможения")
```

---

## 10. Ограничения

### Текущие
1. **Только несжимаемая жидкость**
   - Не поддерживает изэнтропическое соотношение для газа
   - Потребуется доработка для сжимаемых течений

2. **Не входит в loss**
   - На этапе 1.1 только для визуализации
   - Не влияет на обучение

3. **Требует requires_grad**
   - x и y должны иметь `requires_grad=True`
   - Иначе autograd.grad вернёт None

### Будущие улучшения
- [ ] Поддержка сжимаемого газа (изэнтропическое соотношение)
- [ ] Включение в функцию потерь (этап 1.3)
- [ ] Сравнение с CFD (Fluent) данными
- [ ] Регуляризация Бернулли

---

## Авторы

- **Реализация**: Команда Airfoil PINN
- **Версия**: 1.1 (Этап 1.1 - Давление)
- **Дата**: 10 апреля 2026
- **Базовая версия**: Airfoil PINN v3.260226
