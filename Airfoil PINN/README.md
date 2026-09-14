# Airfoil PINN

Проект обучает параметрическую PINN для стационарного двумерного
несжимаемого обтекания профиля. Сеть получает безразмерные координаты,
обязательный дифференцируемый SDF и настраиваемые параметры режима; выдаёт
`u/U∞`, `v/U∞` и коэффициент давления `Cp`.

Каноническое описание физического и программного контракта находится в
[docs/PINN_CFD_CONTRACT.md](docs/PINN_CFD_CONTRACT.md). В нём приведены:

- нормировка по хорде, скорости набегающего потока и динамическому давлению;
- порядок входных признаков, смысл `cos(alpha)` и `sin(alpha)`;
- граф autodiff, PDE/BC/FVM/RAR/SA составляющие loss;
- требования к конфигурациям и совместимости checkpoint;
- команды приёмочных тестов и известные границы применимости.

Отдельный математический и конфигурационный контракт FVM-loss описан в
[docs/FVM_OBJECTIVE.md](docs/FVM_OBJECTIVE.md).

## Возможности текущей версии

### Инициализация обучения
- Двухстадийное обучение с конфигурируемым ramp-ом веса PDE
  (`training.two_stage`, `stage1_epochs`, `stage2_pde_weight_*`,
  `stage2_ramp_epochs`): стадия 1 — только BC, стадия 2 — плавное включение PDE.
- Переход Adam → L-BFGS (`switch_to_lbfgs_epoch`) с фиксацией SA-весов,
  снапшотом обучаемого состояния и автоматическим откатом на Adam при NaN
  (`training.solver.PINNSolver`).
- Воспроизводимый resume: захват/восстановление состояния ГСЧ
  (Python, NumPy, torch CPU/CUDA) в чекпоинте
  (`training.reproducibility`), единый seed (`set_seed`).
- Контрактная проверка конфигурации и checkpoint (`config.validation`,
  `training.contract`): строгий `resume` против намеренного `finetune`,
  отпечатки SDF/mesh/нормировки.
- `DataHandler` с `mesh_file`, `num_workers`, `pin_memory`,
  `persistent_workers`, `prefetch_factor`, `seed`, `pde_system`; AMP
  (`GradScaler`), опциональный `torch.compile`.

### Первичное поле
- Начальное поле скорости задаётся алгебраически режимами
  `constant | linear_x | perturbed`, давление — `constant | linear_x`
  (+ относительный шум к скорости/давлению).
- **Функция тока (stream function) не используется**: поле генерируется
  напрямую, без представления через ψ (u = ∂ψ/∂y, v = −∂ψ/∂x).
- Нормировка — единая система масштабов через `ReferenceScales`
  (`data.normalization`): хорда, U∞, динамическое давление, Re по хорде.
  SDF вычисляется точно по сегментам и дифференцируем
  (`geometry.sdf.PolylineDistanceField`) с диагностикой пограничного слоя
  (`data.sdf_diagnostics`).

### Генерация сетки
- Расчётная область — `geometry.core.AirfoilGeometry.create_domain`
  (закрытый/открытый варианты).
- Коллокация — `sampling.core`: равномерная (regular), случайная
  (irregular), адаптивная (adaptive, сгущение к границам), пристеночный
  normal-layer (`sampling.normal_layer`); классификация граничных точек
  (INLET/OUTLET/WALL/SYMMETRY), отдельная выборка внешней границы/профиля.
- FVM-сетка — `sampling.mesh.core.Mesh2D`: boundary-fitted триангуляция
  (constrained Delaunay: cavity-retriangulation + edge-flip), медианные
  контрольные объёмы, векторы площадей граней, проверка геометрической
  консервативности.
- FVM-loss — `physics.fvm.FVMResiduals`: контрольно-объёмная невязка
  неразрывности (MSE), fail-fast валидация сетки, контракт в чекпоинте.

## Быстрая проверка

```powershell
$env:PYTHONPATH = (Get-Location).Path
Get-ChildItem tests -Filter '*.py' |
    Sort-Object Name |
    ForEach-Object { .\venv\Scripts\python.exe $_.FullName }
```

Основные пользовательские конфигурации находятся в `workspace/configs`.
Обучение следует запускать штатным CLI проекта; на Windows при
`data.num_workers > 0` точка входа обязана находиться под
`if __name__ == "__main__":`.
