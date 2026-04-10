# Этап 1.1: Визуализация потенциала скорости

## Дата: 10 апреля 2026

## Резюме изменений

### Файл: `visualizer.py`

Добавлены три новых метода для визуализации потенциала скорости и связанных полей.

---

## 1. `plot_potential_fields()`

### Назначение
Строит и сохраняет поля потенциала φ и скоростей (u, v), вычисленных из φ.

### Параметры
```python
def plot_potential_fields(
    phi_norm: torch.Tensor,      # Нормализованный потенциал
    u_norm: torch.Tensor,        # u = ∂φ/∂x
    v_norm: torch.Tensor,        # v = ∂φ/∂y
    x_norm: torch.Tensor,        # Координаты x
    y_norm: torch.Tensor,        # Координаты y
    data_handler,                # DataHandler с метаданными
    savedir: str,                # Директория для сохранения
    epoch: Optional[int] = None, # Номер эпохи
    plot_physical: bool = True,  # Физические или нормализованные величины
    config: Optional[Dict] = None
) -> None
```

### Выходные файлы
1. **PNG**: `potential_fields_epoch_N.png`
   - 3 субплота: φ, u, v
   - Цветовые шкалы для каждого поля
   
2. **NPY**: `potential_fields_epoch_N.npy`
   - Словарь с данными полей

### Особенности
- **φ**: colormap `viridis` (последовательная)
- **u, v**: colormap `RdBu_r` (расходящаяся, подходит для скоростей с разными знаками)
- Поддержка физических и нормализованных единиц
- Автоматический подбор цветовой шкалы

---

## 2. `plot_potential_loss_history()`

### Назначение
Строит графики сходимости потерь для потенциала скорости.

### Параметры
```python
def plot_potential_loss_history(
    history: Dict[str, List[float]],  # История потерь
    savedir: str,                     # Директория для сохранения
    epoch: Optional[int] = None       # Номер эпохи
) -> None
```

### Отображаемые компоненты
- `laplace`: уравнение Лапласа ∇²φ
- `inlet_phi`: граничное условие на входе
- `wall_phi`: граничное условие на стенке
- `outlet_phi`: граничное условие на выходе
- `symmetry_phi`: граничное условие на симметрии
- `uniform_reg`: регуляризация однородного потока
- `total_loss`: полная потеря

### Выходные файлы
- **PNG**: `potential_loss_history_epoch_N.png`
  - Логарифмическая шкала для всех графиков
  - Автоматическая подстройка количества субплотов

### Особенности
- Показывает только доступные компоненты
- Скрывает пустые субплоты
- Поддержка частичной истории (если какие-то компоненты отсутствуют)

---

## 3. `plot_velocity_magnitude_from_potential()`

### Назначение
Строит модуль скорости |V| = sqrt(u² + v²) из потенциала.

### Параметры
```python
def plot_velocity_magnitude_from_potential(
    u_norm: torch.Tensor,        # u = ∂φ/∂x
    v_norm: torch.Tensor,        # v = ∂φ/∂y
    x_norm: torch.Tensor,        # Координаты x
    y_norm: torch.Tensor,        # Координаты y
    data_handler,                # DataHandler с метаданными
    savedir: str,                # Директория для сохранения
    epoch: Optional[int] = None, # Номер эпохи
    plot_physical: bool = True,  # Физические или нормализованные величины
    config: Optional[Dict] = None
) -> None
```

### Выходные файлы
- **PNG**: `velocity_magnitude_epoch_N.png`
  - Модуль скорости |V|
  - Colormap `jet` (классический для CFD)

### Особенности
- Полезно для визуализации обтекания профиля
- Показывает зоны ускорения/замедления потока
- Автоматический подбор vmax для лучшей видимости

---

## Примеры использования

### 1. Визуализация полей потенциала

```python
from visualizer import Visualizer

vis = Visualizer(config_name='potential_flow')

# После обучения
predictions = trainer.get_predictions_on_dataset()

vis.plot_potential_fields(
    phi_norm=torch.from_numpy(predictions['phi_norm']),
    u_norm=torch.from_numpy(predictions['u_norm']),
    v_norm=torch.from_numpy(predictions['v_norm']),
    x_norm=torch.from_numpy(predictions['x_norm']),
    y_norm=torch.from_numpy(predictions['y_norm']),
    data_handler=data_handler,
    savedir='results/potential',
    epoch=1000,
    plot_physical=True
)
```

**Результат:**
- `results/potential/potential_fields_epoch_1000.png`
- `results/potential/potential_fields_epoch_1000.npy`

### 2. График сходимости потерь

```python
# history заполняется во время обучения
history = {
    'total_loss': [...],
    'laplace': [...],
    'inlet_phi': [...],
    'wall_phi': [...],
    # ...
}

vis.plot_potential_loss_history(
    history=history,
    savedir='results/potential',
    epoch=1000
)
```

**Результат:**
- `results/potential/potential_loss_history_epoch_1000.png`

### 3. Модуль скорости

```python
vis.plot_velocity_magnitude_from_potential(
    u_norm=torch.from_numpy(predictions['u_norm']),
    v_norm=torch.from_numpy(predictions['v_norm']),
    x_norm=torch.from_numpy(predictions['x_norm']),
    y_norm=torch.from_numpy(predictions['y_norm']),
    data_handler=data_handler,
    savedir='results/potential',
    epoch=1000,
    plot_physical=True
)
```

**Результат:**
- `results/potential/velocity_magnitude_epoch_1000.png`

---

## Интеграция с solver.py

### В методе диагностики

```python
# В solver.py::_run_diagnostics()
if self.config['pde']['system'] == 'potential':
    # Получаем предсказания
    pred = self.trainer.get_predictions_on_dataset()
    
    # Визуализация полей потенциала
    self.visualizer.plot_potential_fields(
        phi_norm=torch.from_numpy(pred['phi_norm']),
        u_norm=torch.from_numpy(pred['u_norm']),
        v_norm=torch.from_numpy(pred['v_norm']),
        x_norm=torch.from_numpy(pred['x_norm']),
        y_norm=torch.from_numpy(pred['y_norm']),
        data_handler=self.data_handler,
        savedir=self._get_plots_dir(),
        epoch=epoch,
        plot_physical=True,
        config=self.config
    )
    
    # Модуль скорости
    self.visualizer.plot_velocity_magnitude_from_potential(
        u_norm=torch.from_numpy(pred['u_norm']),
        v_norm=torch.from_numpy(pred['v_norm']),
        x_norm=torch.from_numpy(pred['x_norm']),
        y_norm=torch.from_numpy(pred['y_norm']),
        data_handler=self.data_handler,
        savedir=self._get_plots_dir(),
        epoch=epoch
    )
    
    # График сходимости
    self.visualizer.plot_potential_loss_history(
        history=self.metrics.history,
        savedir=self._get_plots_dir(),
        epoch=epoch
    )
```

---

## Цветовые схемы

### Потенциал φ: `viridis`
- Последовательная цветовая схема
- Хорошо подходит для скалярных полей
- Восприятию не мешает дальтонизм

### Скорости u, v: `RdBu_r`
- Расходящаяся схема (красный-синий)
- Подчёркивает положительные и отрицательные значения
- Интуитивно: красный = вперёд, синий = назад

### Модуль скорости: `jet`
- Классическая для CFD визуализаций
- Хорошо показывает градиенты
- Внимание: может вводить в заблуждение для дальтоников

---

## Физические vs нормализованные единицы

### Нормализованные (plot_physical=False)
```
φ_norm = φ / (U_inf * chord)  # [1]
u_norm = u / U_inf            # [1]
v_norm = v / U_inf            # [1]
x_norm = x / chord            # [1]
y_norm = y / chord            # [1]
```

### Физические (plot_physical=True)
```
φ = φ_norm * U_inf * chord    # [м²/с]
u = u_norm * U_inf            # [м/с]
v = v_norm * U_inf            # [м/с]
x = x_norm * chord            # [м]
y = y_norm * chord            # [м]
```

---

## Совместимость

### С existing кодом
- ✅ Методы не конфликтуют с существующими функциями
- ✅ Можно использовать параллельно для Навье-Стокса и потенциала
- ✅ Автоматическое определение доступных компонент потерь

### Требования
- ✅ matplotlib (уже есть в зависимостях)
- ✅ numpy (уже есть в зависимостях)
- ✅ torch (уже есть в зависимостях)
- ⚪ scipy (опционально, для interpolation в будущем)

---

## Рекомендуемый рабочий процесс

### 1. Во время обучения (каждые N эпох)
```python
# В solver.py
if epoch % diagnostics_interval == 0:
    # Визуализация полей
    vis.plot_potential_fields(...)
    vis.plot_velocity_magnitude_from_potential(...)
    
    # График сходимости
    vis.plot_potential_loss_history(...)
```

### 2. После обучения (финальная визуализация)
```python
# Финальные поля
vis.plot_potential_fields(..., epoch='final')
vis.plot_velocity_magnitude_from_potential(..., epoch='final')

# Финальная сходимость
vis.plot_potential_loss_history(..., epoch='final')
```

### 3. Сравнение эпох
```python
# Сравнить epoch 1000, 2000, 3000
for epoch in [1000, 2000, 3000]:
    vis.plot_potential_fields(..., epoch=epoch)
```

---

## Будущие улучшения (Этап 1.2+)

- [ ] Контурные графики (contour plots) вместо scatter
- [ ] Векторные поля (quiver plots) для (u, v)
- [ ] Линии тока (streamlines) из потенциала
- [ ] Анимация сходимости (video)
- [ ] Сравнение с аналитическим решением
- [ ] Интерактивные графики (plotly)

---

## Авторы

- **Реализация**: Команда Airfoil PINN
- **Версия**: 1.1 (Этап 1.1 - Визуализация)
- **Дата**: 10 апреля 2026
- **Базовая версия**: Airfoil PINN v3.260226
