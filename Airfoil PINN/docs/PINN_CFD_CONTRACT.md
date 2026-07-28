# Физический и программный контракт Airfoil PINN

## 1. Область применимости

Реализованная модель аппроксимирует стационарные поля двумерного
несжимаемого течения:

\[
\mathcal N_\theta:
(\mathbf x^*, d^*, \boldsymbol\pi)
\longmapsto (u^*, v^*, C_p).
\]

Здесь \(\mathbf x^*=(x/c,y/c)\), \(d^*=\mathrm{SDF}/c\),
\(\boldsymbol\pi\) — независимые безразмерные параметры режима и геометрии.
Тип границы не является входом сети: его one-hot разметка используется
только при построении граничных потерь, масок и жёстких ограничений.

Текущая вязкая PDE — ламинарные incompressible Navier–Stokes. Выбор
`turbulent` в оценке толщины пограничного слоя меняет только sampling и
диагностику, но не добавляет RANS/LES или модель перехода. Для режимов, где
турбулентность существенна, одного текущего PDE-контракта недостаточно.

## 2. Единственная система масштабов

Повторяющиеся величины П-теоремы выбраны как хорда \(c\), модуль скорости
набегающего потока \(U_\infty\) и плотность \(\rho_\infty\). В проекте
разрешена только следующая система:

\[
x^*=\frac{x}{c},\qquad
y^*=\frac{y}{c},\qquad
d^*=\frac{d}{c},
\]

\[
u^*=\frac{u}{U_\infty},\qquad
v^*=\frac{v}{U_\infty},
\]

\[
C_p=\frac{p-p_\infty}
{\tfrac12\rho_\infty U_\infty^2},\qquad
Re_c=\frac{\rho_\infty U_\infty c}{\mu}.
\]

Плотность, вязкость, скорость и хорда не подаются в сеть как независимые
размерные признаки. Для несжимаемой жидкости с постоянными свойствами их
влияние на безразмерную PDE сводится к \(Re_c\). Это предотвращает
дублирование зависимых входов и смешение единиц.

`ReferenceScales` в `airfoil_pinn/data/normalization.py` является единственным
источником этих преобразований. Метаданные содержат schema/version и SHA-256
fingerprint. Координаты, SDF, FVM-векторы граней и площади контрольных
объёмов приводятся соответственно к \(c\), \(c\), \(c\) и \(c^2\).

При выбранном определении \(C_p\) безразмерные уравнения имеют вид

\[
\partial_{x^*}u^*+\partial_{y^*}v^*=0,
\]

\[
u^*\partial_{x^*}u^*+v^*\partial_{y^*}u^*
+\frac12\partial_{x^*}C_p
-\frac1{Re_c}\nabla^{*2}u^*=0,
\]

\[
u^*\partial_{x^*}v^*+v^*\partial_{y^*}v^*
+\frac12\partial_{y^*}C_p
-\frac1{Re_c}\nabla^{*2}v^*=0.
\]

Коэффициент `0.5` перед градиентом `Cp` обязателен: он следует из масштаба
давления \(\tfrac12\rho U_\infty^2\). Одновременное использование
\(p/(\rho U_\infty^2)\) и `Cp` в разных частях loss запрещено.

## 3. Входы и выходы сети

`InputFeatureAssembler` строит один неизменяемый manifest. Порядок столбцов:

1. `coordinates`: \(x/c,\ y/c\), всегда;
2. `sdf`: \(\mathrm{SDF}/c\), всегда;
3. `reynolds`: настраиваемый `log10(Re)` с min-max отображением в `[-1, 1]`;
4. `angle_of_attack`: \(\cos\alpha,\ \sin\alpha\);
5. `pressure_drop_coefficient`: \(\Delta p/q_\infty\);
6. `geometry_parameters`: именованный список безразмерных отношений.

`cos(alpha)` и `sin(alpha)` — компоненты единичного направления набегающего
потока:

\[
\frac{u_\infty}{U_\infty}=\cos\alpha,\qquad
\frac{v_\infty}{U_\infty}=\sin\alpha.
\]

Такая пара не имеет искусственного разрыва между углами \(180^\circ\) и
\(-180^\circ\), в отличие от одного скаляра угла. Она же непосредственно
используется в inlet/farfield BC. Угол в градусах сначала переводится в
радианы; тригонометрические функции не получают градусы.

Выход имеет ровно три столбца:

1. \(u/U_\infty\);
2. \(v/U_\infty\);
3. \(C_p\).

Начальные поля в collocation-артефакте имеют роль `initial_guess_only`. Они
не являются supervised labels и не должны неявно добавляться в loss.

Пример современного раздела признаков:

```yaml
input_features:
  coordinates:
    enabled: true
  sdf:
    enabled: true
  reynolds:
    enabled: true
    range: [10000.0, 1000000.0]
  angle_of_attack:
    enabled: true
  pressure_drop_coefficient:
    enabled: false
  geometry_parameters:
    enabled: true
    names: [thickness_over_chord, camber_over_chord]
```

Если в таблице случаев меняется `Re`, угол, перепад давления или геометрия,
соответствующий признак обязан быть включён. Runtime validation останавливает
запуск, если два физических случая отображались бы в один и тот же вход сети.

## 4. Дифференцируемый SDF

SDF обязателен. Загрузка заранее вычисленного столбца с последующим
`requires_grad=True` некорректна: такой столбец становится независимой
переменной и цепное правило

\[
\frac{d\mathcal N}{dx}
=\frac{\partial\mathcal N}{\partial x}
+\frac{\partial\mathcal N}{\partial d}\frac{\partial d}{\partial x}
\]

теряет второй член.

`PolylineDistanceField` вычисляет расстояние до нормированных сегментов
профиля прямо из тех же `x` и `y`, по которым берутся PDE-производные.
Координаты сегментов — buffers, а не обучаемые параметры. Для внешней
области используется положительная ветвь расстояния.

Минимум по сегментам кусочно гладок, но не имеет единственной второй
производной на medial axis и в местах смены ближайшего сегмента.
`nearest_two_geometry()` и `tie_mask(points)` сначала сравнивают расстояния,
а затем ближайшие проекции и градиенты. Поэтому из вязкой PDE исключается
только настоящая смена ветви/medial axis; соседние сегменты, спроецированные
в одну общую вершину с одинаковым градиентом, сохраняются в PDE. Кэшированный
SDF остаётся допустимым для I/O, контроля
геометрии и точек, где PDE не вычисляется. Fingerprint геометрии SDF входит
в training/checkpoint contract.

### 4.1. Единый численный путь SDF

Training cache, RAR, FVM и inference используют только
`PolylineDistanceField` с расстоянием до отрезков упорядоченного контура.
Для RAR запрещено подменять это расстояние запросом к KD-tree по вершинам.
Все неавтодифференцируемые пути вызывают `evaluate_numpy` в `float64`.
Геометрические buffers SDF сохраняются в `float64`, даже если сеть обучается
в `float32`; при PDE-вызове сегменты приводятся к dtype координат без изменения
канонической геометрии.

Cached SDF считается производным артефактом, а не источником геометрии.
При загрузке сравниваются `sdf_geometry_fingerprint` и
`sdf_method=exact_polyline_segments`. При несовпадении fingerprint, метода
или значений cache пересчитывается по активному контуру. То же выполняется
после восстановления RAR-точек из checkpoint. FVM хранит канонические
координаты и SDF в `float64`, а приводит их к dtype сети только перед forward.

Численный контракт задаётся
`sdf.float64_consistency_tolerance` (по умолчанию `1e-12`):

\[
\max_i\left|\phi_{\mathrm{training},i}^{64}
-\phi_{\mathrm{inference},i}^{64}\right|
\le \varepsilon_{\mathrm{SDF}}.
\]

Метод, fingerprint геометрии и выбранный допуск входят в checkpoint training
contract.

## 5. Граф autodiff и функция потерь

Для каждого батча выполняется следующий граф:

```text
(x/c, y/c, case_id)
        |
        +--> differentiable SDF(x/c, y/c)
        |
        +--> FlowCase: Re_c, cos(alpha), sin(alpha), geometry
        |
        +--> ordered input manifest --> network --> (u*, v*, Cp)
                                              |
                     +------------------------+----------------------+
                     |                        |                      |
              first/second AD             soft/hard BC       integral/FVM
                     |                        |                      |
           NS/Euler residuals         case-specific targets   case-specific
                     +------------------------+----------------------+
                                              |
                                           total loss
```

Концептуально:

\[
\mathcal L =
\sum_k \lambda_{\mathrm{PDE},k}\mathcal L_{\mathrm{PDE},k}
+\sum_j \lambda_{\mathrm{BC},j}\mathcal L_{\mathrm{BC},j}
+\lambda_g\frac{1}{N_{\mathrm{case}}}
\sum_c\left(\operatorname{mean}_{i:\,case_i=c} C_{p,i}\right)^2
+\lambda_I\mathcal L_I
+\lambda_{\mathrm{FVM}}\mathcal L_{\mathrm{FVM}}
+\mathcal L_{\mathrm{extra}}.
\]

- Все оптимизируемые компоненты имеют квадратичную семантику: это средний
  квадрат скалярной невязки либо средний квадрат евклидовой нормы векторной
  невязки. RMS вычисляется только как диагностический `sqrt(MSE)` и не входит
  в `total loss`.
- PDE использует pointwise \(1/Re_c\) того же `case_id`, что и вход сети.
- Pressure gauge вычисляется отдельно для каждого `case_id`, затем
  усредняется между случаями. `pressure_gauge_weight` строго положителен,
  а `architecture.output_bias[2]` равен нулю.
- Inlet/farfield targets используют те же \(\cos\alpha,\sin\alpha\).
- Hard BC применяются к выходам до вычисления остальных членов.
- Integral mass loss группируется по случаю, затем усредняется между
  случаями; потоки разных режимов не компенсируют друг друга.
- Одна глобальная FVM-сетка вычисляется отдельно для каждого `FlowCase`;
  case-loss усредняются. Планарный FVM запрещён для axisymmetric PDE.
- RAR сохраняет `case_id`, оценивает кандидаты по соответствующей физике и
  не смешивает параметры режимов.
- SA-PINN weights положительны и обучаются как max-переменные через reversal
  градиента, пока параметры сети минимизируют loss. Перед L-BFGS они
  замораживаются.

Ни один loss не может использовать размерное давление вместе с
безразмерными скоростями или физическую вязкость вместе с нормированными
координатами.

## 6. Гладкость модели и численная устойчивость

Вязкие Navier–Stokes содержат вторые производные, поэтому разрешены гладкие
активации `tanh`, `sigmoid`, `swish` и `siren`. `relu`, `leaky_relu` и `elu`
отклоняются для вязкой системы; для Euler первого порядка они остаются
доступны.

SIREN использует:

- первый слой: \(W\sim U(-1/n_{in},1/n_{in})\);
- остальные и выходной слои:
  \(W\sim U(-\sqrt{6/n_{in}}/\omega_0,
  \sqrt{6/n_{in}}/\omega_0)\);
- активацию \(\sin(\omega_0 z)\);
- настраиваемые `siren_first_omega_0` и `siren_hidden_omega_0`.

Прямой `float16` запрещён. Допустимы `float32`, `float64` и AMP только при
CUDA + `float32`. До optimizer step проверяются конечность total loss и всех
градиентов; ошибка содержит стадию и компоненты loss. Gradient norm при AMP
считается после `unscale_`, поэтому telemetry не содержит масштабированную
фиктивную норму.

## 7. Конфигурация, telemetry и checkpoint

Граничные условия конфигурируются по физическим границам. На первом этапе
hard enforcement разрешён только для неподвижной no-slip стенки вязкой
несжимаемой Navier--Stokes:

```yaml
boundary_conditions:
  wall:
    kind: no_slip
    enforcement: hard
    distance_scale: 0.02
  slip_wall:
    kind: slip_wall
    enforcement: soft
  symmetry:
    kind: symmetry
    enforcement: soft
  farfield:
    kind: freestream
    enforcement: soft
  outlet:
    kind: pressure_outlet
    enforcement: soft
```

Скорость преобразуется внутри autograd-графа как
\(u=M(\Phi_w)u_{raw}\), \(v=M(\Phi_w)v_{raw}\), где
\(M(\Phi)=\Phi/(\Phi+\ell)\). Давление не преобразуется. PDE вычисляется по
преобразованным полям, а все неподдерживаемые hard-операторы остаются soft.
Глобальный `boundary_conditions.type=hard` запрещён.

Каждая collocation-точка границы хранит `normal_x`, `normal_y`. Soft-условия
не выводят направление из `left/right/top/bottom`:

- `slip_wall`: \(u_n=u n_x+v n_y=0\);
- `symmetry`: \(u_n=0\) и
  \(\partial u_t/\partial n=0\), где \(u_t=-u n_y+v n_x\);
- `farfield`: заданный вектор скорости набегающего потока.

Файл с `slip_wall` или `symmetry` без конечной ненулевой нормали считается
несовместимым и должен быть пересоздан Sampler/Initializer.

Статическая validation выполняется после YAML merge и перед сохранением.
Runtime validation после загрузки данных проверяет:

- обязательные координаты/SDF и размер выхода;
- диапазоны параметров и физические reference scales;
- соответствие `case_id` таблице случаев;
- видимость каждого изменяющегося параметра во входном manifest;
- совпадение SDF/FVM geometry fingerprint;
- совместимость PDE и FVM.

Каждый checkpoint записывается атомарно через временный файл и содержит:

- versioned checkpoint schema;
- model/optimizer/scheduler/SA states;
- training history, Trainer и DataHandler/RAR states;
- training contract с SHA-256;
- Python, NumPy, Torch CPU/CUDA и DataLoader RNG states.

По умолчанию несовпадение контракта блокирует resume до изменения состояния
модели. Контракт включает kind/mode каждой границы, тип distance transform и
`distance_scale`. Смена boundary transform всегда запрещает точный resume.
`checkpoint.load_mode: finetune` загружает только веса модели и намеренно
игнорирует optimizer/scheduler, DataHandler/RAR, history, epoch и RNG.
Старый checkpoint без контракта допускается только при
`checkpoint.allow_legacy_without_contract: true` с предупреждением. Для
строгой эксплуатации следует установить его в `false`.

В `logs/run_manifest.json` сохраняются эффективный контракт, версии runtime,
устройство, dtype, размеры модели и DataLoader. `logs/epochs.jsonl` содержит
строгий JSON по каждой эпохе: раздельные `*_mse` и `*_rms` для PDE, BC,
pressure gauge, integral/FVM, нормы градиента, mass/FVM-метрики, learning rate
и времена. Запись `NaN`/`Inf` в telemetry запрещена.

`output.progress_bar` включает tqdm в интерактивном терминале;
`progress_bar_force` предназначен для явного включения вне TTY. Sampling и
mesh routines используют собственные progress bars. `data.num_workers`
остаётся настраиваемым: in-memory датасет не всегда ускоряется от процессов.
На Windows используется воспроизводимый top-level worker initializer и
обязателен `__main__` guard.

## 8. Проверки

Полный smoke-набор:

```powershell
$env:PYTHONPATH = (Get-Location).Path
$failed = @()
Get-ChildItem tests -Filter '*.py' |
    Sort-Object Name |
    ForEach-Object {
        .\venv\Scripts\python.exe $_.FullName
        if ($LASTEXITCODE -ne 0) { $failed += $_.Name }
    }
if ($failed) { throw "Failed: $($failed -join ', ')" }
.\venv\Scripts\python.exe -m compileall -q airfoil_pinn tests
git diff --check
```

Набор покрывает:

- прямые и обратные преобразования нормировки;
- градиент и гессиан segment SDF, включая цепное правило сети;
- порядок, значения и градиенты параметрических входов;
- коэффициенты NS/Cp и pointwise производную по `Re`;
- case-aware BC, integral loss, RAR, SA-PINN;
- case-aware FVM и безразмерную геометрию;
- validation и повреждение/несовместимость checkpoint contract;
- SIREN bounds и вторые производные;
- NaN/Inf guards, telemetry, Windows `spawn`;
- end-to-end Adam save/resume с точным восстановлением модели, optimizer,
  dataset и всех RNG.

## 9. Научные основания

- M. Raissi, P. Perdikaris, G. E. Karniadakis,
  [Physics-informed neural networks](https://doi.org/10.1016/j.jcp.2018.10.045),
  Journal of Computational Physics 378 (2019).
- M. Naderibeni et al.,
  [Learning solutions of parametric Navier–Stokes with PINNs](https://arxiv.org/abs/2402.03153),
  2024.
- L. McClenny, U. Braga-Neto,
  [Self-Adaptive PINNs using a Soft Attention Mechanism](https://arxiv.org/abs/2009.04544),
  2020.
- V. Sitzmann et al.,
  [Implicit Neural Representations with Periodic Activation Functions](https://arxiv.org/abs/2006.09661),
  NeurIPS 2020.
- R. Gladstone et al.,
  [FO-PINNs](https://arxiv.org/abs/2210.14320), 2022 — ограничения distance
  functions при высоких порядках PDE.
- B. J. Kirby,
  [Nondimensionalization and Characteristic Parameters](https://doi.org/10.1017/CBO9780511760723.024),
  Cambridge University Press, 2010.
- NASA Glenn:
  [Reynolds number and boundary layer](https://www1.grc.nasa.gov/beginners-guide-to-aeronautics/boundary-layer/),
  [dynamic pressure](https://www.grc.nasa.gov/www/k-12/BGP/dynpress.html).

Документ описывает реализованный контракт, но не является доказательством
сходимости PINN к единственному физическому решению. Такая гарантия требует
отдельных исследований аппроксимации, оптимизации, sampling error и
сопоставления с независимым CFD/экспериментом.
