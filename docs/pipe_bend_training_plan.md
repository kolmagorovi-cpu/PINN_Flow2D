# План доработки модулей обучения Pipe-Bend PINN

**Дата:** 16.07.2026
**Задача:** Исправление и доработка модулей обучения нейросети в Pipe-Bend PINN на основе анализа эталонной реализации в Airfoil PINN

---

## 1. Краткий анализ архитектуры обучения в Airfoil PINN

**Airfoil PINN** имеет зрелую, многослойную архитектуру обучения:

| Компонент | Назначение | Ключевые возможности |
|-----------|-----------|---------------------|
| `solver.py` (PINNSolver) | Дирижёр обучения | Двухэтапное обучение, переключение Adam→L-BFGS, RAR, early stopping, планировщики LR, детектор плато, чекпоинты по интервалу, загрузка чекпоинтов с восстановлением состояния датасета |
| `loss_computer.py` | Функция потерь | Soft/hard BC, SA-weights, интегральное условие (FVM), адаптивная балансировка BC, `apply_velocity_dirichlet_constraint()` |
| `trainer.py` | Прямой проход + оптимизация | `forward_pass()` → словарь с loss/pde_raw/bc_comp; градиентное накопление, AMP, RAR (через `RARManager`), раздельные нормы градиентов PDE/BC, FVM-loss, NTK condition |
| `metrics.py` | Сбор метрик | CSV-логирование, `save_final_fields()`, `save_data()`, `get_boundary_avg_velocities()` с кэшированием, `get_mass_flow_metrics()`, `collect_epoch_metrics()` |
| `optimizers.py` | LR-утилиты | `LocalMinimaDetector` (с factor/patience/recovery), `CyclicLR` (через `torch.optim.lr_scheduler`) |

Ключевой контракт: **`LossComputer.compute_loss()` → словарь**, а **`Trainer.train_epoch_adam()` → кортеж из 9 элементов**.

---

## 2. Критические дефекты в текущей реализации Pipe-Bend PINN

### 2.1 `solver.py` — 5 критических ошибок

| # | Проблема | Последствие |
|---|----------|------------|
| 1 | **Строка 38**: `self.loss_computer = LossComputer(..., x_min=0, x_max=1, y_min=0, y_max=1, ...)` — жёстко закодированные границы вместо `self.data_handler.x_min/max/y_min/max` | Граничные условия `tol` в интегральном условии = 0.01, SDF нормализация сломана |
| 2 | **Строка 86-87**: `self.rho_norm = torch.tensor(1.0)` и `self.mu_norm = torch.tensor(1.0 / self.data_handler.Re)` — НО mu_norm вычисляется до проверки Re | Если Re=0, mu_norm = inf |
| 3 | **Строка 36-38**: `DataHandler.__init__` не принимает `device` — все тензоры датасета на CPU, переносятся на GPU каждый батч заново | Медленная загрузка данных |
| 4 | **Нет методов**: `_load_checkpoint_if_needed()`, `_switch_to_lbfgs()`, `_post_training_diagnostics()`, `_recreate_optimizer()` | Невозможно продолжить обучение, использовать L-BFGS, сохранить финальные поля |
| 5 | **В `train()` нет**: обработки `scheduler.step()`, `checkpoint_interval`, `rar_enable`, `minima_detector`, `early_stopping`, `reset_interval` | Обучение работает в урезанном режиме |

### 2.2 `solver_components/loss_computer.py` — несовместимый интерфейс

| # | Проблема |
|---|----------|
| 1 | **`compute_loss()` возвращает `tuple`** из 5 элементов, а `trainer.py` ожидает **словарь** с ключами `'loss'`, `'pde_raw'`, `'bc_comp'`, `'integral_loss'`, `'wall_speed'`, `'pde_max'` |
| 2 | **Метод `_compute_bc_loss()`**: inlet определяется как `u_bc_f > 0.5` (эвристика по значению скорости), а не через `boundary_types` one-hot encoding. Для pipe bend с нулевой скоростью на стенке это НЕ сработает |
| 3 | **Нет поддержки SA-weights** внутри `compute_loss` — веса просто игнорируются |
| 4 | **Нет методов**: `apply_velocity_dirichlet_constraint()`, `compute_integral_loss()` |
| 5 | **`compute_wall_speed()`** использует хардкод `'wall'` вместо параметризации |

### 2.3 `solver_components/trainer.py` — заглушки вместо реализации

| # | Проблема |
|---|----------|
| 1 | **Строка 59**: `result = self.loss_computer.compute_loss(batch, self.model, pde_mult)` — возвращает кортеж, **но дальше обращается `result['loss']`** → 🐞 `TypeError: tuple indices must be integers...` |
| 2 | **`apply_rar()`**: residual считается как `sqrt(u²+v²)` (скорость), а не как PDE невязка — **некорректный критерий** для RAR |
| 3 | **`compute_grad_norms_separately()`** — заглушка, всегда возвращает нули |
| 4 | **Нет `forward_pass()`** метода — обучение не может быть разбито на прямой/обратный проход |
| 5 | **Нет `_get_grad_norm()`** вызова с проверкой `error_if_nonfinite` |

### 2.4 `solver_components/metrics.py` — неполная реализация

| # | Проблема |
|---|----------|
| 1 | **Нет методов**: `save_final_fields()`, `save_data()`, `close_csv_logger()`, `collect_epoch_metrics()`, `get_mass_flow_metrics()`, `invalidate_boundary_cache()` |
| 2 | **`_init_csv()`** использует жёстко закодированные колонки `['inlet', 'outlet', 'wall']` — не адаптируется под реальные `boundary_type_names` |

### 2.5 `data_handler.py` — маски границ заполняются некорректно

| # | Проблема |
|---|----------|
| 1 | **Маски** `mask_u_fixed`, `mask_v_fixed`, `mask_p_fixed` заполняются только если в данных есть поле `'boundary_type'`. Если нет — **все точки считаются interior**, и BC-потери не вычисляются |
| 2 | **Нет метода `get_state()` / `load_state()`** — невозможно сохранить/восстановить состояние датасета после RAR |

---

## 3. План доработки (в порядке приоритета)

### Этап 1: Исправление интерфейса LossComputer (базовый контракт)

**Файл:** `Pipe-Bend PINN/solver_components/loss_computer.py`

- [ ] **Переписать `compute_loss()`**: возвращать **словарь** с ключами `loss`, `pde_raw`, `bc_comp`, `integral_loss`, `wall_speed`, `pde_max`
- [ ] **Переписать `_compute_bc_loss()`**: использовать `boundary_types` one-hot тензор для определения типа границы, а не эвристики по значениям скорости
- [ ] **Добавить `apply_velocity_dirichlet_constraint()`** для hard BC режима

### Этап 2: Приведение trainer.py в рабочее состояние

**Файл:** `Pipe-Bend PINN/solver_components/trainer.py`

- [ ] **Исправить `train_epoch_adam()`**: распаковывать словарь, а не кортеж
- [ ] **Добавить `forward_pass()`** метод, вызывающий `LossComputer.compute_loss()` и возвращающий структурированный словарь
- [ ] **Исправить `apply_rar()`**: использовать PDE невязки из `point_pde_residual`, а не скорость
- [ ] **Добавить `compute_grad_norms_separately()`** с реальным вычислением градиентов PDE и BC отдельно

### Этап 3: Доработка solver.py до production-уровня

**Файл:** `Pipe-Bend PINN/solver.py`

- [ ] **Исправить инициализацию границ**: `x_min=self.data_handler.x_min` и т.д.
- [ ] **Добавить**: `_load_checkpoint_if_needed()`, `_switch_to_lbfgs()`, `_post_training_diagnostics()`, `_recreate_optimizer()`
- [ ] **Добавить в `train()`**: scheduler.step(), checkpoint по интервалу, RAR, детектор минимумов, early stopping, сброс оптимизатора с шумом
- [ ] **Добавить `_validate_physics_metadata()`** для проверки согласованности Re/rho/mu

### Этап 4: Расширение metrics.py

**Файл:** `Pipe-Bend PINN/solver_components/metrics.py`

- [ ] **Добавить `save_final_fields()`** — сохранение предсказанных полей в `.npy`
- [ ] **Добавить `collect_epoch_metrics()`** — сбор расширенной диагностики
- [ ] **Добавить `get_mass_flow_metrics()`** — баланс масс через inlet/outlet
- [ ] **Сделать CSV-колонки динамическими** на основе `boundary_type_names`

### Этап 5: Исправление data_handler.py

**Файл:** `Pipe-Bend PINN/data_handler.py`

- [ ] **Добавить `get_state()` / `load_state()`** для сериализации датасета
- [ ] **Исправить инициализацию масок**: для interior точек маски должны быть `False`, для inlet/outlet/wall — соответствующие `True`
- [ ] **Добавить `device` параметр** в DataHandler для переноса тензоров на GPU при загрузке

### Этап 6: Интеграционное тестирование

**Файл:** `Pipe-Bend PINN/test_pipeline.py`

- [ ] **Обновить `test_pipeline.py`**: добавить проверку всех новых методов
- [ ] **Проверить сквозной прогон**: геометрия → коллокация → инициализация → обучение (10 эпох) → сохранение чекпоинта → загрузка чекпоинта → продолжение обучения

---

## 4. Оценка трудозатрат

| Этап | Файлы | Часы |
|------|-------|------|
| 1. LossComputer | `loss_computer.py` | 3-4 |
| 2. Trainer | `trainer.py` | 3-4 |
| 3. Solver | `solver.py` | 4-5 |
| 4. Metrics | `metrics.py` | 2-3 |
| 5. DataHandler | `data_handler.py` | 2-3 |
| 6. Тестирование | `test_pipeline.py` | 2 |

**Итого: ~16-21 часов** (2.5-3 рабочих дня)

---

## 5. Стратегия реализации

Рекомендуемый **поэтапный подход с верификацией каждого этапа**:

1. Сначала исправить `loss_computer.py` и `trainer.py` (контракт "словарь")
2. Убедиться что `test_pipeline.py` (2 эпохи) проходит без ошибок
3. Затем доработать `solver.py` (L-BFGS, чекпоинты, RAR)
4. Расширить `metrics.py`
5. Исправить `data_handler.py`
6. Финальное интеграционное тестирование

---

*План составлен на основе анализа эталонной кодовой базы `Airfoil PINN/airfoil_pinn/training/` и текущей реализации `Pipe-Bend PINN/solver_components/`.*