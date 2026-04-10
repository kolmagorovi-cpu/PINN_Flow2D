# Отключение live matplotlib окон - Документация

## ✅ Готово! Matplotlib окна больше не лагают

### Что было сделано

#### 1. **Отключены live matplotlib окна по умолчанию**

В `config_manager.py` изменены настройки:

```yaml
output:
  plot_frequency: null  # Было: 10 (теперь None = без окон)
  enable_matplotlib_live: false  # Новый параметр (по умолчанию выкл)
```

#### 2. **Добавлен новый метод сохранения графиков**

Создан `save_final_plots()` в `visualizer.py`:
- ✅ Работает БЕЗ live matplotlib окон
- ✅ Создает графики из истории потерь
- ✅ Вызывается автоматически на чекпоинтах

#### 3. **Интеграция в solver.py**

Графики сохраняются:
- На каждом чекпоинте (какие N эпох)
- При прерывании обучения (Ctrl+C)
- При завершении обучения (финальные графики)

### Как это работает теперь

#### Раньше (с лагами):
```
Каждые 10 эпох → Обновление matplotlib окон → Лаги GUI
```

#### Теперь (без лагов):
```
Каждая эпоха → TensorBoard (веб-интерфейс, без лагов)
Каждый чекпоинт → Сохранение графиков в PNG (без окон)
```

### Что видите вы

#### При запуске обучения:
```
📊 Live matplotlib окна отключены (используется TensorBoard)
📈 Графики сохраняются только на чекпоинтах
🚀 TensorBoard запускается в фоновом режиме...
✅ TensorBoard доступен: http://localhost:6006
```

#### Во время обучения:
- **Никаких окон** - всё работает в фоне
- **TensorBoard в браузере** - http://localhost:6006
- **Графики в папке** - `Pictures/pde_loss_epoch_500.png`, и т.д.

#### На чекпоинтах:
```
График PDE сохранён: pde_loss_epoch_500.png
График BC сохранён: bc_loss_epoch_500.png
График диагностики сохранён: diagnostics_epoch_500.png
```

### Настройка

#### Включить live окна (если нужно для отладки):

В конфигурации YAML:
```yaml
output:
  plot_frequency: 10  # Частота обновления (эпох)
  enable_matplotlib_live: true  # Включить live окна
```

#### Отключить обратно:
```yaml
output:
  plot_frequency: null  # Или любое число
  enable_matplotlib_live: false
```

### Какие графики сохраняются

#### 1. PDE Loss (4 графика в одном файле)
- Continuity
- Momentum X
- Momentum Y
- Total Loss

#### 2. BC Loss (6 графиков в одном файле)
- Inlet
- Outlet
- Wall
- Symmetry
- Axis Symmetry
- Wall Speed

#### 3. Diagnostics (3 графика в одном файле)
- Gradient Norms (PDE, BC, cosine)
- Avg Inlet/Outlet Velocity
- Mass Balance Error

### Где искать файлы

```
Pictures/
├── pde_loss_epoch_500.png
├── bc_loss_epoch_500.png
├── diagnostics_epoch_500.png
├── pde_loss_final.png
├── bc_loss_final.png
└── diagnostics_final.png
```

### Сравнение режимов

| Характеристика | Live matplotlib | TensorBoard | PNG на чекпоинтах |
|----------------|-----------------|-------------|-------------------|
| Лаги GUI | ❌ Да | ✅ Нет | ✅ Нет |
| Нагрузка на GPU | ❌ Высокая | ✅ Нет | ✅ Нет |
| Реальное время | ✅ Да | ✅ Да | ❌ Только на чекпоинтах |
| История запусков | ❌ Нет | ✅ Да | ✅ Да (файлы) |
| Мобильность | ❌ Только локально | ✅ Веб | ✅ Файлы |
| Экспорт | ⚠️ Сложнее | ✅ Легко | ✅ Готовые PNG |

### Рекомендуемая конфигурация

**Для продуктивной работы:**
```yaml
output:
  plot_frequency: null
  enable_matplotlib_live: false
  use_tensorboard: true
  tensorboard_auto_launch: true
```

**Для отладки (если нужны окна):**
```yaml
output:
  plot_frequency: 50
  enable_matplotlib_live: true
  use_tensorboard: true
  tensorboard_auto_launch: true
```

### Преимущества

✅ **Нет лагов** - matplotlib окна не создаются  
✅ **Нет нагрузки на GPU** - только TensorBoard  
✅ **Графики сохраняются** - на чекпоинтах и в конце  
✅ **История доступна** - в браузере через TensorBoard  
✅ **Файлы для отчета** - PNG в папке Pictures  

### Как смотреть обучение

#### Способ 1: TensorBoard (рекомендуется)
```bash
# Автоматически запускается
# Откройте браузер:
http://localhost:6006
```

#### Способ 2: Просмотр PNG файлов
```bash
# После чекпоинта откройте файлы:
Pictures/pde_loss_epoch_500.png
Pictures/diagnostics_epoch_500.png
```

### Устранение проблем

#### Графики не сохраняются на чекпоинтах

**Проверьте:**
1. `checkpoint.interval > 0` в конфигурации
2. Директория `Pictures` существует и доступна
3. В логах есть: "График PDE сохранён: ..."

#### Ошибка при сохранении

**Сообщение**: "Нет истории для сохранения графиков"

**Решение**: Убедитесь что `set_history()` вызывается в solver.py:
```python
self.visualizer.set_history(self.metrics.history)
```

#### Хочу live окна обратно

**Решение**:
```yaml
output:
  enable_matplotlib_live: true
  plot_frequency: 10
```

### Технические детали

#### Как работает `save_final_plots`

1. Создает новые фигуры matplotlib (не использует live окна)
2. Заполняет данными из `self._history_for_save`
3. Сохраняет в PNG
4. Закрывает фигуры (`plt.close`)

**Преимущество**: Не требует открытых окон matplotlib, работает в headless режиме.

#### Почему не использовать `save_plots`

`save_plots` проверяет `self.enabled` и существует ли окно. Если окна нет - не сохраняет.

`save_final_plots` создает окна временно, только для сохранения, и сразу закрывает их.

### Итоговая схема работы

```
Запуск обучения
    ↓
TensorBoard запускается (фон)
    ↓
Обучение идет (каждая эпоха)
    ↓
TensorBoard логируется (каждая эпоха)
    ↓
Чекпоинт (какие N эпох)
    ↓
Сохраняются PNG графики
    ↓
Продолжение обучения
    ↓
Завершение/Прерывание
    ↓
Финальные PNG графики
    ↓
TensorBoard останавливается
```

## Готово! 🎉

Теперь обучение работает без лагов, а графики сохраняются автоматически на чекпоинтах!
