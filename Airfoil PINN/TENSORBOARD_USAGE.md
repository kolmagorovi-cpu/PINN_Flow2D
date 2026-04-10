# TensorBoard Integration for Airfoil PINN

## Обзор

Визуализатор `visualizer.py` теперь поддерживает **TensorBoard** для асинхронного логирования метрик обучения. Это решает проблему лагов GUI при использовании matplotlib во время обучения на GPU.

## Преимущества

- ✅ **Асинхронное логирование** - запись чисел в файл занимает микросекунды
- ✅ **Нет нагрузки на GUI** - отрисовка происходит в отдельном веб-интерфейсе
- ✅ **История всех запусков** - каждый запуск сохраняется в отдельной директории
- ✅ **Богатая визуализация** - графики, гистограммы, изображения

## Быстрый старт

### 1. Включение TensorBoard в конфигурации

При создании `Visualizer` передайте параметры:

```python
from visualizer import Visualizer

visualizer = Visualizer(
    plot_frequency=10,  # Частота обновления matplotlib (можно 0 для отключения)
    config_name="my_experiment",
    config=config,
    use_tensorboard=True,  # Включить TensorBoard
    log_dir="logs"  # Директория для логов (по умолчанию "logs")
)
```

### 2. Использование в цикле обучения

В вашем цикле обучения (например, в `trainer.py` или `solver.py`):

```python
# На каждой эпохе
for epoch in range(num_epochs):
    # ... код обучения ...
    
    # Обновление графиков и логирование в TensorBoard
    loss_components = {
        'laplace': loss_laplace.item(),
        'inlet_phi': loss_inlet.item(),
        'wall_phi': loss_wall.item()
    }
    
    visualizer.update_plot(epoch, history, loss_components)
    
    # Опционально: логирование весов модели (раз в N эпох)
    if epoch % 100 == 0:
        visualizer.log_model_weights(model, epoch)
```

### 3. Запуск TensorBoard

После запуска обучения откройте терминал и выполните:

```bash
tensorboard --logdir=logs
```

Перейдите по ссылке: **http://localhost:6006**

## Доступные метрики

TensorBoard автоматически логирует:

### Потери (Loss)
- `Loss/total` - полная ошибка
- `PDE/{key}` - компоненты PDE (cont, mom_x, mom_y, laplace, и т.д.)
- `BC/{key}` - компоненты граничных условий (inlet, outlet, wall, и т.д.)
- `Components/{key}` - дополнительные компоненты из `loss_components`

### Градиенты
- `Gradients/PDE_norm` - норма градиентов PDE
- `Gradients/BC_norm` - норма градиентов BC
- `Gradients/cosine` - косинус угла между градиентами

### Физика
- `Mass/balance_error` - ошибка баланса масс
- `Velocity/avg_u_in` - средняя скорость на входе
- `Velocity/avg_u_out` - средняя скорость на выходе

### Веса модели (опционально)
- `Weights/{layer_name}` - гистограммы весов
- `Gradients/{layer_name}` - гистограммы градиентов

## Дополнительные методы

### Логирование изображений

```python
# Логирует поле скорости как изображение
# image_tensor должен быть формы (C, H, W) в диапазоне [0, 1]
visualizer.log_image('Fields/velocity_u', image_tensor, epoch)
```

### Логирование гистограмм

```python
# Логирует гистограмму значений
visualizer.log_histogram('Distributions/residuals', residuals_tensor, epoch)
```

### Логирование весов модели

```python
# Логирует все веса и градиенты модели
visualizer.log_model_weights(model, epoch)
```

## Отключение matplotlib

Если вы хотите **полностью отключить matplotlib** и использовать только TensorBoard:

```python
visualizer = Visualizer(
    plot_frequency=None,  # Отключает matplotlib
    config_name="my_experiment",
    config=config,
    use_tensorboard=True,
    log_dir="logs"
)
```

Тогда `visualizer.update_plot()` будет только логировать в TensorBoard без создания окон.

## Структура логов

```
logs/
├── my_experiment_20260410_143022/
│   ├── events.out.tfevents.1234567890
│   └── ...
├── my_experiment_20260410_150045/
│   └── ...
└── ...
```

Каждый запуск получает уникальное имя с timestamp.

## Сравнение с matplotlib

| Характеристика | matplotlib | TensorBoard |
|----------------|------------|-------------|
| Нагрузка на GUI | Высокая | Нет |
| Лаги при 100% GPU | Да | Нет |
| История запусков | Нет | Да |
| Веб-интерфейс | Нет | Да |
| Экспорт графиков | Да | Да |
| Делиться результатами | Сложнее | Легко (ссылка) |

## Рекомендуемая конфигурация

Для продуктивной работы:

```python
visualizer = Visualizer(
    plot_frequency=None,  # Отключаем matplotlib
    config_name=config.get('project_name', 'unnamed'),
    config=config,
    use_tensorboard=True,
    log_dir="logs"
)
```

Для отладки (оба режима):

```python
visualizer = Visualizer(
    plot_frequency=10,  # Обновление каждые 10 эпох
    config_name=config.get('project_name', 'unnamed'),
    config=config,
    use_tensorboard=True,
    log_dir="logs"
)
```

## Устранение проблем

### TensorBoard не запускается

Убедитесь, что установлен tensorboard:
```bash
pip install tensorboard
```

### Логи не появляются

Проверьте:
1. Параметр `use_tensorboard=True` при инициализации
2. Вызов `visualizer.update_plot()` или `visualizer.log_scalars()` в цикле
3. Правильность пути `log_dir`

### Медленная запись логов

TensorBoard использует асинхронную запись, поэтому это маловероятно. Если проблема persists, попробуйте:
- Уменьшить частоту логирования весов модели
- Логировать веса раз в 100-500 эпох вместо каждой

## Примеры использования

### Пример 1: Базовое логирование

```python
visualizer = Visualizer(
    plot_frequency=None,
    config_name="airfoil_test",
    use_tensorboard=True
)

for epoch in range(10000):
    loss = train_step()
    history['loss'].append(loss)
    
    visualizer.log_scalars(epoch, history)
```

### Пример 2: Полная конфигурация с полями

```python
visualizer = Visualizer(
    plot_frequency=100,
    config_name="airfoil_full",
    config=config,
    use_tensorboard=True,
    log_dir="runs"
)

for epoch in range(10000):
    loss, loss_components = train_step()
    history['loss'].append(loss)
    history['pde']['laplace'].append(loss_components['laplace'])
    
    # Обновляем графики и логируем
    visualizer.update_plot(epoch, history, loss_components)
    
    # Раз в 500 эпох логируем веса и поля
    if epoch % 500 == 0:
        visualizer.log_model_weights(model, epoch)
        # Здесь можно добавить логирование изображений полей
        # visualizer.log_image('Fields/pressure', pressure_image, epoch)
```

## Дополнительные ресурсы

- [TensorBoard документация](https://www.tensorflow.org/tensorboard)
- [PyTorch TensorBoard Tutorial](https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html)
