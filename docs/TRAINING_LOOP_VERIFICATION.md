# Этап 1.1: Проверка реализации цикла обучения для потенциала

## Дата: 10 апреля 2026

## Резюме проверки

Проверена корректность реализации цикла обучения для потенциала скорости в модулях `solver.py` и `trainer.py`.

---

## ✅ 1. Ветвление по типу PDE системы

### Расположение: `trainer.py`, метод `forward_pass()`

**Строка 159-169:**
```python
# Обработка выхода модели в зависимости от типа PDE
if self.loss_computer.use_potential:
    # Для потенциала: out[:, 0:1] - это φ
    # Скорости вычисляются через градиенты потенциала
    phi = out[:, 0:1]
    # Вычисляем скорости как градиенты потенциала
    u = safe_grad(phi, x)  # u = ∂φ/∂x
    v = safe_grad(phi, y)  # v = ∂φ/∂y
    # Для совместимости создаём фиктивное давление
    p = torch.zeros_like(phi)
    u_raw, v_raw, p_raw = u, v, p
else:
    u_raw, v_raw, p_raw = out[:, 0:1], out[:, 1:2], out[:, 2:3]
    # ... стандартная логика для Навье-Стокса
```

**Статус:** ✅ Реализовано корректно

---

## ✅ 2. Вычисление невязок PDE

### Расположение: `trainer.py`, строка 181

```python
res = self.loss_computer.compute_pde_residuals(u, v, p, x, y)
```

Этот метод внутри `loss_computer.py` автоматически выбирает нужную логику:

```python
# В loss_computer.py::compute_pde_residuals()
if self.use_potential:
    return PDEResiduals.compute_residuals_potential(u, x, y, safe_grad)
elif self.flow_type == 'axisymmetric':
    return PDEResiduals.compute_residuals_axisymmetric(...)
else:
    return PDEResiduals.compute_residuals_2d(...)
```

**Статус:** ✅ Реализовано корректно

---

## ✅ 3. Ветвление для потерь PDE

### Расположение: `trainer.py`, строки 183-221

```python
if self.loss_computer.use_potential:
    # Для потенциала только одна компонента - laplace
    laplace = res['laplace']
    
    # ... вычисление loss_laplace
    
    # PDE потеря с учётом весов из конфигурации
    pde_loss = pde_mult * (
        self.loss_computer.loss_weights.get('laplace_weight', 1.0) * loss_laplace
    )
else:
    # Оригинальная логика для Навье-Стокса
    # ... cont, mom1, mom2
```

**Статус:** ✅ Реализовано корректно

---

## ✅ 4. Граничные условия с SDF

### Расположение: `trainer.py`, строка 280

```python
bc_comp, extra_grad = self.loss_computer.compute_bc_loss(u, v, p, bt, x, y, sdf)
```

**Передача sdf:** ✅ sdf передаётся в функцию потерь

### Расположение: `trainer.py`, строка 138

```python
sdf = sdf.to(self.device, dtype=self.dtype).requires_grad_(True)  # Нужны градиенты для нормалей
```

**requires_grad для SDF:** ✅ Установлено корректно

---

## ✅ 5. Регуляризация uniform_reg

### Расположение: `loss_computer.py`, строки 379-391

```python
# Регуляризация: стремление к однородному потоку вдали от стенки
# Используется маска по SDF: точки с sdf > boundary_layer_thickness
if sdf is not None and self.uniform_reg_weight > 0:
    # Создаём маску для точек вдали от пограничного слоя
    bl_mask = sdf.view(-1) > self.boundary_layer_thickness
    if bl_mask.sum() > 0:
        u_far = phi_x[bl_mask]
        v_far = phi_y[bl_mask]
        uniform_loss = (
            torch.mean((u_far - self.U_inf_x) ** 2) +
            torch.mean((v_far - self.U_inf_y) ** 2)
        )
        extra_grad += self.uniform_reg_weight * uniform_loss
```

**Статус:** ✅ Реализовано корректно

### Примечание о detach

В текущей реализации `phi_x` и `phi_y` вычисляются через `safe_grad(phi, x)` и `safe_grad(phi, y)`, 
что создаёт градиенты в графе вычислений. Это **правильно**, так как:

1. Регуляризация должна влиять на градиенты модели
2. `detach` не нужен, так как мы хотим оптимизировать φ для уменьшения uniform_loss
3. Граф вычислений создаётся один раз и используется корректно

---

## ✅ 6. Параметры обучения

### learning rate

**Расположение:** `config_manager.py` или конфигурационный файл

Стандартные параметры (для потенциала можно использовать те же):
```yaml
training:
  lr_schedule: constant
  optimizer_params:
    lr: 1e-3  # Стандартный learning rate
```

**Статус:** ✅ Параметры совместимы с потенциалом

### epochs

```yaml
training:
  epochs: 5000  # Рекомендуется больше для потенциала
```

**Статус:** ✅ Можно настроить отдельно

---

## ✅ 7. Модель выдаёт φ, а не (u,v,p)

### Расположение: `solver.py`, метод `_build_model()`

```python
if pde_system == 'potential':
    # Для потенциала используем новый класс PotentialPINN
    output_bias = arch.get('potential_output_bias', [0.0])[0]
    return PotentialPINN(
        input_dim=actual_input_dim,
        layers=arch['hidden_layers'],
        activation=arch.get('activation', 'tanh'),
        output_bias=output_bias,
        coordinate_scaling=coord_scaling
    )
```

**PotentialPINN** возвращает один выход (φ):
```python
# В models.py::PotentialPINN
self.output_layer = nn.Linear(prev, 1)  # Один выход: φ
```

**Статус:** ✅ Реализовано корректно

---

## ✅ 8. Вычисление скоростей из φ

### Во время обучения (trainer.py)

```python
phi = out[:, 0:1]
u = safe_grad(phi, x)  # u = ∂φ/∂x
v = safe_grad(phi, y)  # v = ∂φ/∂y
```

### Во время inference (solver.py::_get_predictions_norm)

```python
# Численное вычисление градиентов (для inference без autograd)
eps = 1e-6

# du/dx через конечные разности
x_plus = x + eps
x_minus = x - eps
input_plus = self.loss_computer.build_input_tensor(x_plus, y, sdf, bt)
input_minus = self.loss_computer.build_input_tensor(x_minus, y, sdf, bt)
phi_plus = self.model(input_plus)
phi_minus = self.model(input_minus)
u = (phi_plus - phi_minus) / (2 * eps)
```

**Статус:** ✅ Реализовано корректно (оба режима)

---

## 📊 Итоговая функция потерь

Для режима `potential`:

```python
loss = λ_laplace * mean((∇²φ)²)                              # PDE
     + λ_inlet_phi * [(∂φ/∂x - U_inf_x)² + (∂φ/∂y - U_inf_y)²]  # Inlet BC
     + λ_wall_phi * (∇φ·n)²                                    # Wall BC
     + λ_outlet_phi * (∂φ/∂n)²                                 # Outlet BC
     + λ_symmetry_phi * (∂φ/∂y)²                               # Symmetry BC
     + λ_uniform_reg * [(∂φ/∂x - U_inf_x)² + (∂φ/∂y - U_inf_y)²]  # Regularization
```

Все компоненты вычисляются корректно с использованием градиентов φ.

---

## 🎯 Выводы

### Все требования из пункта 6 реализованы:

| Требование | Статус | Расположение |
|------------|--------|--------------|
| Ветвление по `config.pde.system == "potential"` | ✅ | `trainer.py:159-169` |
| Модель выдаёт φ, а не (u,v,p) | ✅ | `solver.py::_build_model()` |
| Вычисление градиентов для регуляризации | ✅ | `loss_computer.py:379-391` |
| Параметры обучения (lr, epochs) стандартные | ✅ | Конфигурация |
| sdf имеет requires_grad=True | ✅ | `trainer.py:138` |
| Нормали вычисляются через SDF | ✅ | `loss_computer.py:318-336` |

### Дополнительных изменений не требуется! ✅

Текущая реализация полностью удовлетворяет всем требованиям из пункта 6.

---

## 📝 Рекомендации по настройке параметров обучения

### Для потенциала скорости (этап 1.1):

```yaml
training:
  epochs: 5000                    # Больше эпох для сходимости
  batch_size: 1000                # Больший размер батча
  lr_schedule: constant
  optimizer_params:
    lr: 1e-3                      # Стандартный learning rate

loss_weights:
  laplace_weight: 1.0             # Вес уравнения Лапласа
  bc_weights:
    inlet_phi: 100.0              # Сильный вес для входа
    wall_phi: 100.0               # Сильный вес для стенки
    outlet_phi: 10.0              # Умеренный вес для выхода
    symmetry_phi: 10.0            # Умеренный вес для симметрии
  uniform_reg_weight: 0.01        # Слабая регуляризация
```

### Обоснование:

1. **Больше эпох**: Потенциал требует больше итераций для сходимости ∇²φ → 0
2. **Больший batch_size**: Уменьшает шум градиента для стабильности
3. **Высокие веса BC**: Граничные условия критичны для физического решения
4. **Слабая регуляризация**: Не должна доминировать над основными потерями

---

## Авторы

- **Проверка**: Команда Airfoil PINN
- **Версия**: 1.1 (Этап 1.1 - Проверка цикла обучения)
- **Дата**: 10 апреля 2026
- **Базовая версия**: Airfoil PINN v3.260226
