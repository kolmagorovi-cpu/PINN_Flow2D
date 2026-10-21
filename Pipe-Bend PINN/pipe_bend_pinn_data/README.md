# pipe_bend_pinn_data

Изолированный пакет подготовки CFD-данных для параметрической Pipe-Bend PINN.
Работает с экспортом Fluent/CFX (сечение `Plane 1`) и не зависит от
обучающего ядра `pipe_bend_pinn`.

## Что делает

1. Разбирает CFD-экспорт с секциями `[Name]` / `[Data]` (заголовки сопоставляются
   по именам: `X`, `Y`, `Z`, `Pressure`, `Velocity`, `Velocity u`, `Velocity v`).
2. Читает манифест `CSV -> U_in` и единые `ref_scales`.
3. Обезразмеривает поля: `x/W`, `y/W`, `u/U_ref`, `v/U_ref`,
   `Cp = (p - p_ref) / (0.5·ρ·U_ref²)`, `Re = ρ·U_in·W/μ`, `re_norm = Re/Re_ref`.
4. Считает SDF до контура стенки (в нормированных единицах).
5. Валидирует данные, включая **привязку к геометрии** (доля точек внутри домена).
6. Собирает `torch.utils.data.Dataset` / `DataLoader` с полями
   `(x, y, sdf, re_norm) -> (u, v, p)`.

## Быстрый старт

```python
import sys
sys.path.insert(0, r"...\Pipe-Bend PINN")

from pipe_bend_pinn_data import CFDataHandler

handler = CFDataHandler("workspace/cfd/cfd_manifest.yaml").load()
print(handler.summary())

dataset = handler.dataset            # ParametricCFDDataset
loader = handler.dataloader(batch_size=4096)
batch = next(iter(loader))           # {'x': (B,1), ..., 're_norm': (B,1), 'case_id': (B,)}
handler.save_npy("workspace/initial/fields/cfd_points.npy")
```

CLI:

```powershell
python -m pipe_bend_pinn_data --manifest workspace/cfd/cfd_manifest.example.yaml --no-strict
```

## Контракт CSV

```text
[Name]
Plane 1

[Data]
X [ m ], Y [ m ], Z [ m ], Pressure [ Pa ], Velocity [ m s^-1 ], Velocity u [ m s^-1 ], Velocity v [ m s^-1 ]
...
```

Обязательные колонки: `X`, `Y`, `Pressure`, `Velocity u`, `Velocity v`.
`Z` (плоское сечение) и `Velocity` (модуль, для проверки) — опциональны.
Поддерживается и обычный CSV с заголовком `x,y,p,u,v`.

## Манифест

```yaml
ref_scales:
  U_ref: 1.0
  pipe_width: 0.5
  density: 998.0
  dynamic_viscosity: 0.998
  static_pressure: 100000.0
geometry:
  domain_file: workspace/geometry/domains/pipe_bend_domain.npy
  closed: true
cases:
  - file: workspace/cfd/case_U0.5.csv
    U_in: 0.5
  - file: workspace/cfd/case_U1.0.csv
    U_in: 1.0
```

Допускается краткая форма `cases: {file: U_in}`. Относительные пути
разрешаются от каталога манифеста, затем от корня проекта.

## Важно: единая шкала

`U_ref` задаётся **один раз** на весь диапазон. Покейсовая нормировка на
собственный `U_in` уничтожила бы зависимость решения от Re.

## Валидация привязки к геометрии

`validate_alignment` считает долю CFD-точек внутри контура. Если CSV
экспортирован с другой геометрии, покрытие будет ~0 и появится ошибка
`outside_domain`. Пример `docs/export.csv` в репозитории относится к другой
геометрии — это ожидаемо.

## Тесты

```powershell
python pipe_bend_pinn_data/tests/run_all.py
```
