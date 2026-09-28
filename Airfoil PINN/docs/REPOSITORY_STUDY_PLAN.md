**Подробный план изучения Airfoil PINN — редакция 2**

Дата инвентаризации: 2026-09-16. Это план и карта охвата, а не отчёт о завершённом построчном разборе или успешном прохождении тестов.

Полнота проверяется по [индексу исходников и определений](REPOSITORY_SYMBOL_INDEX.md). В область входят Airfoil_PINN.py, airfoil_pinn, ui и tests, а также связанные конфигурации, документация и форматы workspace. Зависимости venv, .git и сгенерированные результаты не входят в построчный разбор исходного кода приложения.

Выявленные недостатки первой редакции: недостаточно выделены серии экспериментов, управление во время обучения, служебные типы/утилиты/реэкспорты, алгоритмы интерполяции сечений, NTK/FFT/пристенная диагностика и статус UI-возможностей. Здесь они имеют отдельные пункты.

Приоритет имеют фактический код и условия вызова. Наличие функции или настройки не доказывает её доступность из основного сценария. Статьи служат основанием метода, но не доказательством совпадения локальной реализации с публикацией.

<a id="stage-1"></a>

1. **Границы проекта и проверяемая полнота разбора.**

   - **1.1.** Отделить Git-корень, корень Python-приложения, workspace, локальное окружение и генерируемые результаты.
   - **1.2.** Составить индекс всех классов, функций, методов, вложенных определений, property/setter и lambda; каждому назначить этап разбора.
   - **1.3.** Для каждого модуля включить код вне функций: импорты, константы, поля классов, регистрации, обработчики и блок запуска.
   - **1.4.** Зафиксировать зависимости из ../requirements.txt, requirements_ui.txt, ui/requirements.txt и runtime; сверить с установленными версиями.
   - **1.5.** Учесть README, оба математических контракта, YAML-конфигурации, JSON-библиотеки, координаты профилей и локальные NPY/NPZ-артефакты.
   - **1.6.** Различать состояние «включено в план», «код разобран», «проверено тестом» и «проверено сквозным запуском»; не считать AST-инвентаризацию доказательством работоспособности.

   Источники: [D1: Python: ast](https://docs.python.org/3/library/ast.html).

<a id="stage-2"></a>

2. **Точки входа, окружение, пути и консольное приложение.**

   Основные файлы: [Airfoil_PINN.py](<../Airfoil_PINN.py>), [airfoil_pinn/__main__.py](<../airfoil_pinn/__main__.py>), [airfoil_pinn/app/cli.py](<../airfoil_pinn/app/cli.py>), [airfoil_pinn/app/runtime.py](<../airfoil_pinn/app/runtime.py>), [airfoil_pinn/app/paths.py](<../airfoil_pinn/app/paths.py>), [airfoil_pinn/app/module_registry.py](<../airfoil_pinn/app/module_registry.py>), [airfoil_pinn/training/learning_cli.py](<../airfoil_pinn/training/learning_cli.py>).

   - **2.1.** Проследить run.bat, запуск скрипта, python -m airfoil_pinn и отдельный CLI обучения.
   - **2.2.** Разобрать UTF-8, sys.path, поиск и создание venv, поиск зависимостей, установку и перезапуск процесса; установить побочные эффекты bootstrap.
   - **2.3.** Разобрать WorkspacePaths, создание каталогов, абсолютные и относительные пути, зависимость от рабочей директории.
   - **2.4.** Изучить ModuleDescriptor, реестр команд, динамический импорт и совместимость со старым форматом меню.
   - **2.5.** Разобрать главное меню, меню модулей, настройки, справку, аргументы train и передачу управления runner.
   - **2.6.** Проследить обработку неверной команды, отсутствующего файла, исключения, Ctrl+C и коды завершения.

<a id="stage-3"></a>

3. **Служебные типы, импорты и общие утилиты.**

   Основные файлы: [airfoil_pinn/core/constants.py](<../airfoil_pinn/core/constants.py>), [airfoil_pinn/core/type_hints.py](<../airfoil_pinn/core/type_hints.py>), [airfoil_pinn/utils/auxiliary_utils.py](<../airfoil_pinn/utils/auxiliary_utils.py>), [airfoil_pinn/utils/io_utils.py](<../airfoil_pinn/utils/io_utils.py>), [airfoil_pinn/utils/ui_utils.py](<../airfoil_pinn/utils/ui_utils.py>), [airfoil_pinn/training/components/utils.py](<../airfoil_pinn/training/components/utils.py>).

   - **3.1.** Разобрать все TypedDict-схемы и сопоставить их с реальными словарями и структурированными NumPy-массивами.
   - **3.2.** Проследить использование констант, значений по умолчанию и флагов наличия необязательных библиотек.
   - **3.3.** Разобрать __init__.py, __all__, ленивые __getattr__, реэкспорты и защиту от циклических импортов.
   - **3.4.** Объяснить sanitize_filename, каталоги, JSON/text I/O, поиск файлов и поведение при ошибках.
   - **3.5.** Разобрать BaseMenu, ввод чисел, значения по умолчанию, подтверждения, форматирование и Colors.
   - **3.6.** Проследить set_seed, контроль RAM/CUDA-памяти, safe_grad и обработку отсутствующих градиентов; связать с этапами 14, 18 и 21.

   Источники: [D6: NumPy 2.4: Broadcasting](https://numpy.org/doc/2.4/user/basics.broadcasting.html); [D9: PyTorch 2.11: torch.autograd.grad](https://docs.pytorch.org/docs/2.11/generated/torch.autograd.grad.html).

<a id="stage-4"></a>

4. **Конфигурация и все меню настройки обучения.**

   Основные файлы: [airfoil_pinn/config/manager.py](<../airfoil_pinn/config/manager.py>), [airfoil_pinn/config/validation.py](<../airfoil_pinn/config/validation.py>), [airfoil_pinn/training/menus/architecture.py](<../airfoil_pinn/training/menus/architecture.py>), [airfoil_pinn/training/menus/common.py](<../airfoil_pinn/training/menus/common.py>), [airfoil_pinn/training/menus/configuration.py](<../airfoil_pinn/training/menus/configuration.py>), [airfoil_pinn/training/menus/loss.py](<../airfoil_pinn/training/menus/loss.py>), [airfoil_pinn/training/menus/process.py](<../airfoil_pinn/training/menus/process.py>), [airfoil_pinn/training/menus/results.py](<../airfoil_pinn/training/menus/results.py>), [airfoil_pinn/training/menus/strategy.py](<../airfoil_pinn/training/menus/strategy.py>).

   - **4.1.** Разобрать создание, загрузку, редактирование, сохранение и сброс ConfigManager, рекурсивное объединение и миграцию старых полей.
   - **4.2.** Сопоставить каждый параметр fvm1/fvm2/fvm3 с потребителем; установить различия конфигураций и эффективные defaults.
   - **4.3.** Разобрать меню входных признаков, слоёв, активации, масштабирования координат и выходов.
   - **4.4.** Разобрать меню PDE, boundary conditions, весов loss, FVM, SA и интегрального условия.
   - **4.5.** Разобрать меню стратегии обучения, устройства, dtype, вывода, checkpoint и диагностики.
   - **4.6.** Разделить статическую validation, runtime-validation по данным и проверку checkpoint; пояснить каждое ограничение и сообщение.
   - **4.7.** Проверить неиспользуемые, устаревшие и недоступные через меню параметры; не приписывать настройке эффект без пути до потребителя.

   Источники: [D2: PyYAML Documentation](https://pyyaml.org/wiki/PyYAMLDocumentation).

<a id="stage-5"></a>

5. **Библиотеки жидкостей и условий течения.**

   Основные файлы: [airfoil_pinn/physics/fluids.py](<../airfoil_pinn/physics/fluids.py>), [airfoil_pinn/physics/fluids_cli.py](<../airfoil_pinn/physics/fluids_cli.py>), [airfoil_pinn/physics/conditions.py](<../airfoil_pinn/physics/conditions.py>), [airfoil_pinn/physics/conditions_cli.py](<../airfoil_pinn/physics/conditions_cli.py>).

   - **5.1.** Разобрать Fluid и BoundaryConditions: поля, проверки, свойства, to_dict/from_dict и отображение.
   - **5.2.** Проследить создание, чтение, сохранение, выбор, редактирование и удаление записей FluidLibrary и ConditionsManager.
   - **5.3.** Объяснить плотность, динамическую/кинематическую вязкость и связь размерных величин с Re.
   - **5.4.** Разобрать ввод скорости компонентами и через модуль/угол, градусы/радианы, синхронизацию представлений.
   - **5.5.** Проследить давление, симметрию, выбор жидкости и передачу условий в Initializer.
   - **5.6.** Описать все JSON-поля workspace/libraries, стандартные записи и поведение при некорректных данных.

<a id="stage-6"></a>

6. **Геометрия вычислительного пакета.**

   Основные файлы: [airfoil_pinn/geometry/core.py](<../airfoil_pinn/geometry/core.py>), [airfoil_pinn/geometry/cli.py](<../airfoil_pinn/geometry/cli.py>).

   - **6.1.** Разобрать GeometryConfig, AirfoilGeometry и Geometry2D, включая функции выбора файла и g2d.
   - **6.2.** Проследить загрузку контура из файлов и YAML-библиотеки, преобразование и проверку координат.
   - **6.3.** Объяснить определение замкнутости, создание областей для открытого и замкнутого профиля.
   - **6.4.** Разобрать front/rear/top/bottom, интерполяцию y по x, геометрические метаданные и масштаб хорды.
   - **6.5.** Проследить plot, print_stats, save и все поддерживаемые форматы сохранения.
   - **6.6.** Сопоставить геометрию backend и самостоятельную UI-модель из этапа 26.

   Источники: [D3: Shapely 2.1.2 User Manual](https://shapely.readthedocs.io/en/2.1.2/manual.html).

<a id="stage-7"></a>

7. **Коллокационные точки и параллельный sampling.**

   Основные файлы: [airfoil_pinn/sampling/core.py](<../airfoil_pinn/sampling/core.py>), [airfoil_pinn/sampling/cli.py](<../airfoil_pinn/sampling/cli.py>).

   - **7.1.** Разобрать BoundaryType, CollocationPoint, SamplerConfig, идентификаторы и сериализацию.
   - **7.2.** Проследить поиск геометрии, загрузку контура и области, выбор и предварительный просмотр.
   - **7.3.** Разобрать внешние границы, поверхность профиля, нижний край, классификацию, segment_id и нормали.
   - **7.4.** Пояснить регулярный, случайный и адаптивный sampling, последовательные и многопроцессные ветви и worker-функции.
   - **7.5.** Разобрать исключение внутренности профиля, пристенного диапазона, ограничение попыток и геометрические допуски.
   - **7.6.** Проследить generate_all_points, объединение результатов, прогресс, фильтрацию, визуализацию и save_points.
   - **7.7.** Описать связь точек с Initializer, сеткой и последующим RAR.

   Источники: [D3: Shapely 2.1.2 User Manual](https://shapely.readthedocs.io/en/2.1.2/manual.html).

<a id="stage-8"></a>

8. **Пристенное размещение точек и предварительная SDF-диагностика.**

   Основные файлы: [airfoil_pinn/sampling/normal_layer.py](<../airfoil_pinn/sampling/normal_layer.py>), [airfoil_pinn/data/sdf_diagnostics.py](<../airfoil_pinn/data/sdf_diagnostics.py>).

   - **8.1.** Разобрать NormalLayerConfig, NormalLayerResult и NormalLayerColumnsResult.
   - **8.2.** Проследить перепараметризацию поверхности, касательные, выбор внешних нормалей и верхней/нижней стороны.
   - **8.3.** Объяснить первую высоту, коэффициент роста, уровни eta, предел расстояния и оценки на основе y+.
   - **8.4.** Разобрать оценки Cf и толщины пограничного слоя, ламинарную/турбулентную ветви и их границы применимости.
   - **8.5.** Проследить геометрическую фильтрацию колонок, удаление дублей, минимальный шаг и лимит числа точек.
   - **8.6.** Разобрать nearest-wall расстояния, сводную SDF-статистику, интервалы и форматирование диагностик.
   - **8.7.** Отдельно показать, что выбор turbulent для размещения/оценки толщины сам по себе не добавляет турбулентную PDE-модель.

<a id="stage-9"></a>

9. **Триангуляция, дуальная сетка и контрольные объёмы.**

   Основные файлы: [airfoil_pinn/sampling/mesh/core.py](<../airfoil_pinn/sampling/mesh/core.py>).

   - **9.1.** Разобрать MeshNode, InternalFace, BoundaryFace, адаптеры и способы создания Mesh2D.
   - **9.2.** Пояснить геометрические предикаты: ориентацию, пересечение отрезков, окружность и принадлежность треугольнику.
   - **9.3.** Проследить подготовку вершин, объединение метаданных, восстановление цепочек и приоритет типов границ.
   - **9.4.** Разобрать начальную Delaunay-триангуляцию, вставку ограничивающих рёбер, перестройки и flood fill для удаления областей.
   - **9.5.** Вывести площади контрольных объёмов и векторы внутренних/граничных граней; проследить ориентацию знаков.
   - **9.6.** Разобрать проверку топологии и геометрической консервативности, NPZ-сохранение, свойства и поисковые методы.
   - **9.7.** Объяснить графики сетки, карты и гистограммы площадей; связать структуру с FVMResiduals.

   Источники: [D4: SciPy 1.17: Delaunay](https://docs.scipy.org/doc/scipy-1.17.0/reference/generated/scipy.spatial.Delaunay.html); [S8: Eymard, Gallouët, Herbin (2000), Finite Volume Methods; библиография автора](https://www.i2m.univ-amu.fr/perso/thierry.gallouet/publi.d/publi03.html).

<a id="stage-10"></a>

10. **Физическая нормировка и подготовка начальных артефактов.**

   Основные файлы: [airfoil_pinn/data/normalization.py](<../airfoil_pinn/data/normalization.py>), [airfoil_pinn/data/initializer.py](<../airfoil_pinn/data/initializer.py>), [airfoil_pinn/data/initializer_cli.py](<../airfoil_pinn/data/initializer_cli.py>).

   - **10.1.** Вывести систему масштабов c, U∞, rho, p∞, Cp и Re; разобрать ReferenceScales и прямые/обратные преобразования.
   - **10.2.** Проследить восстановление хорды и ссылок на геометрию, companion metadata и fingerprint.
   - **10.3.** Разобрать загрузку sampler-точек, выделение стенки, выбор жидкости/условий и восстановление области.
   - **10.4.** Разобрать constant/linear_x/perturbed режимы начальных полей и независимые возмущения скорости/давления.
   - **10.5.** Проследить структурированный dtype, нормировку координат/SDF/полей, сохранение нормалей, начальный FlowCase и initial_guess_only.
   - **10.6.** Объяснить _compute_sdf_chunk, совместимость старых аргументов и фактическое вычисление SDF.
   - **10.7.** Разобрать NPY и метаданные, графики полей, интерполяцию, маски профиля/области и initialize_menu.

   Источники: [S1: Raissi, Perdikaris, Karniadakis (2019), Physics-informed neural networks; авторская страница](https://maziarraissi.github.io/PINNs/); [D5: SciPy 1.17: griddata](https://docs.scipy.org/doc/scipy-1.17.0/reference/generated/scipy.interpolate.griddata.html).

<a id="stage-11"></a>

11. **Режимы течения, Dataset, батчи и состояние данных.**

   Основные файлы: [airfoil_pinn/data/flow.py](<../airfoil_pinn/data/flow.py>), [airfoil_pinn/data/batch.py](<../airfoil_pinn/data/batch.py>), [airfoil_pinn/data/handler.py](<../airfoil_pinn/data/handler.py>).

   - **11.1.** Разобрать FlowCase/FlowCaseBatch, проверки, угловые свойства, metadata и fingerprint.
   - **11.2.** Проследить case_id от файла до параметров PDE/BC/FVM; проверить представимость различных режимов входами сети.
   - **11.3.** Разобрать PINNDataset, реальные роли полей, legacy target aliases, inlet targets и маски.
   - **11.4.** Объяснить PINNBatch, legacy_tuple, итерацию, индексацию, collate_pinn_batches и pin_memory.
   - **11.5.** Проследить DataLoader для обучения, общей оценки и границ; seed worker, shuffle, workers и Windows spawn.
   - **11.6.** Разобрать согласование нормировки геометрии, загрузку MeshData, преобразование в Torch и FVM-тензоры.
   - **11.7.** Разобрать добавление/дублирование точек, сохранение/восстановление состояния и обновление кэшей после RAR.
   - **11.8.** Все методы интерполяции профилей и потоков отдельно детализировать в этапе 22.

   Источники: [S6: Naderibeni et al. (2024), Learning solutions of parametric Navier–Stokes with PINNs](https://arxiv.org/abs/2402.03153); [D8: PyTorch 2.11: Dataset/DataLoader](https://docs.pytorch.org/docs/2.11/data.html).

<a id="stage-12"></a>

12. **Дифференцируемое поле расстояний.**

   Основные файлы: [airfoil_pinn/geometry/sdf.py](<../airfoil_pinn/geometry/sdf.py>).

   - **12.1.** Разобрать DistanceFieldDiagnostics и PolylineDistanceField: очистку сегментов, buffers, fingerprint и dtype.
   - **12.2.** Вывести расстояние до отрезка, ограничение параметра проекции и chunked вычисление.
   - **12.3.** Проследить forward, wall_distance, evaluate_numpy и единый путь для training, RAR, FVM и inference.
   - **12.4.** Разобрать ближайшие две проекции, нормали поверхности, tie_mask и исключение негладких ветвей из вязкой PDE.
   - **12.5.** Показать цепное правило через SDF, первые/вторые производные и влияние clamp/min.
   - **12.6.** Разобрать диагностику, допуски float64, проверку/пересчёт кэша и differentiable_sdf_for_interior.

   Источники: [S5: Sukumar, Srivastava, Exact imposition of boundary conditions with distance functions](https://arxiv.org/abs/2104.08426); [D9: PyTorch 2.11: torch.autograd.grad](https://docs.pytorch.org/docs/2.11/generated/torch.autograd.grad.html).

<a id="stage-13"></a>

13. **Входные признаки и нейронная сеть.**

   Основные файлы: [airfoil_pinn/training/input_features.py](<../airfoil_pinn/training/input_features.py>), [airfoil_pinn/training/models.py](<../airfoil_pinn/training/models.py>), [airfoil_pinn/training/network_diagram.py](<../airfoil_pinn/training/network_diagram.py>).

   - **13.1.** Разобрать FeatureSlice, InputFeatureManifest, порядок столбцов, input_dim и fingerprint.
   - **13.2.** Пояснить координаты, SDF, log10(Re), cos/sin угла, коэффициент перепада давления и именованные параметры геометрии.
   - **13.3.** Проследить приоритет современных/legacy-настроек, обязательные признаки и проверки форм тензоров.
   - **13.4.** Разобрать PINNIncompressible, ModuleList/Linear, число параметров, активации и масштабирование только координат.
   - **13.5.** Объяснить Sine/SIREN: частоты, инициализацию скрытых/выходного слоёв и гладкость для PDE.
   - **13.6.** Разобрать SelfAdaptiveWeights, softplus, gradient hook, ограничение весов и расширение после RAR; min–max роль связать с этапом 16.
   - **13.7.** Проследить подписи входов и visualize_network, соответствие рисунка фактической модели.

   Источники: [S2: Sitzmann et al. (2020), Implicit Neural Representations with Periodic Activation Functions](https://arxiv.org/abs/2006.09661); [S3: McClenny, Braga-Neto (2020), Self-Adaptive PINNs](https://arxiv.org/abs/2009.04544); [S6: Naderibeni et al. (2024), Learning solutions of parametric Navier–Stokes with PINNs](https://arxiv.org/abs/2402.03153).

<a id="stage-14"></a>

14. **PDE и автоматическое дифференцирование.**

   Основные файлы: [airfoil_pinn/physics/pde.py](<../airfoil_pinn/physics/pde.py>).

   - **14.1.** Вывести стационарные безразмерные уравнения неразрывности и импульса с коэффициентами 0.5 и 1/Re.
   - **14.2.** Построчно разобрать compute_residuals_2d и compute_residuals_euler_2d.
   - **14.3.** Отдельно разобрать обе осесимметричные ветви, координаты, геометрические члены и поведение возле оси.
   - **14.4.** Проследить grad_fn/safe_grad, requires_grad, create_graph, retain_graph, detach и обработку отсутствующих производных.
   - **14.5.** Разобрать pointwise inverse_reynolds, broadcast (N,) против (N,1) и проверки конечности.
   - **14.6.** Установить, где возвращается невязка, а где её квадрат; связать с масками и редукциями Trainer.

   Источники: [S1: Raissi, Perdikaris, Karniadakis (2019), Physics-informed neural networks; авторская страница](https://maziarraissi.github.io/PINNs/); [D9: PyTorch 2.11: torch.autograd.grad](https://docs.pytorch.org/docs/2.11/generated/torch.autograd.grad.html).

<a id="stage-15"></a>

15. **Граничные условия и преобразование выходов.**

   Основные файлы: [airfoil_pinn/training/boundary.py](<../airfoil_pinn/training/boundary.py>).

   - **15.1.** Разобрать resolve_boundary_conditions, defaults, имена границ и legacy-алиасы.
   - **15.2.** Объяснить no-slip, slip, symmetry, axis_symmetry, freestream, velocity_dirichlet и pressure outlet по фактическим ветвям LossComputer.
   - **15.3.** Вывести нормальную/касательную скорость и производную по нормали; показать использование normal_x/normal_y.
   - **15.4.** Разобрать BoundaryOutputTransform и distance_mask, допустимые hard-условия, distance_scale и контракт.
   - **15.5.** Проследить применение преобразования до PDE, в FVM, метриках и inference.
   - **15.6.** Пояснить различия Euler/NS, ограничения неподвижной стенки и недоступные режимы; старые pointwise overrides пометить отдельно.

   Источники: [S5: Sukumar, Srivastava, Exact imposition of boundary conditions with distance functions](https://arxiv.org/abs/2104.08426).

<a id="stage-16"></a>

16. **Сборка функции потерь, маски и балансировка.**

   Основные файлы: [airfoil_pinn/training/components/loss_computer.py](<../airfoil_pinn/training/components/loss_computer.py>).

   - **16.1.** Разобрать инициализацию LossComputer, выбор PDE-компонентов и связи с DataHandler/InputFeatureAssembler/SDF.
   - **16.2.** Проследить маски interior/boundary/SDF-tie, пустые выборки, числители и знаменатели усреднения.
   - **16.3.** Разобрать отдельно каждую BC-потерю и нормальные производные, затем их агрегирование.
   - **16.4.** Объяснить pressure gauge по case_id и согласование с заданным давлением.
   - **16.5.** Разобрать глобальное интегральное условие, сеточный и доступные резервные пути, группировку по режимам.
   - **16.6.** Показать итоговый loss, статические коэффициенты, staged PDE multiplier, adaptive BC и SA min–max.
   - **16.7.** Разделить оптимизируемые MSE, диагностические RMS, local RAR score и PDE-карты.
   - **16.8.** Проследить обновление адаптивных весов и фактическое наличие потребителя каждого параметра loss.

   Источники: [S1: Raissi, Perdikaris, Karniadakis (2019), Physics-informed neural networks; авторская страница](https://maziarraissi.github.io/PINNs/); [S3: McClenny, Braga-Neto (2020), Self-Adaptive PINNs](https://arxiv.org/abs/2009.04544).

<a id="stage-17"></a>

17. **FVM-оператор и расписание глобальной потери.**

   Основные файлы: [airfoil_pinn/physics/fvm.py](<../airfoil_pinn/physics/fvm.py>).

   - **17.1.** Разобрать адаптеры Mesh2D/MeshData, геометрические тензоры, fingerprint и проверки размеров/индексов.
   - **17.2.** Вывести гранные потоки, значения скорости, prescribed BC и накопление балансов по контрольным объёмам.
   - **17.3.** Разобрать set_bc_velocity, set_inlet_velocity, reset_prescribed, перенос на устройство и нормировку геометрии.
   - **17.4.** Пояснить все reduction-ветви, divergence_l2_volume, flux_balance_l2, legacy-названия, MSE и RMS.
   - **17.5.** Проследить вычисления для каждого FlowCase и усреднение без компенсации потоков разных режимов.
   - **17.6.** Разобрать каждый FVM schedule, масштабирование при пропусках и gradient accumulation.
   - **17.7.** Проследить проверки консервативности и перестройку после RAR; пояснить несовместимость планарной сетки с осесимметричной PDE.

   Источники: [S8: Eymard, Gallouët, Herbin (2000), Finite Volume Methods; библиография автора](https://www.i2m.univ-amu.fr/perso/thierry.gallouet/publi.d/publi03.html).

<a id="stage-18"></a>

18. **Оркестрация, оптимизаторы, численная устойчивость и ускорение.**

   Основные файлы: [airfoil_pinn/training/runner.py](<../airfoil_pinn/training/runner.py>), [airfoil_pinn/training/solver.py](<../airfoil_pinn/training/solver.py>), [airfoil_pinn/training/components/trainer.py](<../airfoil_pinn/training/components/trainer.py>), [airfoil_pinn/training/components/optimizers.py](<../airfoil_pinn/training/components/optimizers.py>), [airfoil_pinn/training/numerics.py](<../airfoil_pinn/training/numerics.py>), [airfoil_pinn/training/progress.py](<../airfoil_pinn/training/progress.py>).

   - **18.1.** Проследить runner → PINNSolver.__init__ → train → finalize, порядок зависимостей и время создания контрактов.
   - **18.2.** Разобрать forward_pass: перенос батча, SDF, входы, сеть, boundary transform, loss и служебные выходы.
   - **18.3.** Проследить train_batch_adam/train_epoch_adam: zero_grad, backward, accumulation, clipping и optimizer.step.
   - **18.4.** Разобрать L-BFGS, closure, повторные вычисления, замораживание SA, историю и переключение оптимизатора.
   - **18.5.** Объяснить двухстадийную стратегию, PDE ramp, scheduler, LocalMinimaDetector, CyclicLR и reset с шумом.
   - **18.6.** Разобрать early stopping, сохранение/возврат лучшего trainable state и отличие от полного checkpoint.
   - **18.7.** Проследить CPU/CUDA, float32/float64, AMP/GradScaler, DataParallel и torch.compile; установить реальные проверки и fallback.
   - **18.8.** Разобрать NumericalTrainingError, ensure_finite_loss, GradientHealth, проверки градиентов, контроль памяти и tqdm.
   - **18.9.** Описать остановку, исключения, завершение файлов/графиков/слушателей и итоговые артефакты.

   Источники: [D9: PyTorch 2.11: torch.autograd.grad](https://docs.pytorch.org/docs/2.11/generated/torch.autograd.grad.html); [D10: PyTorch 2.11: LBFGS](https://docs.pytorch.org/docs/2.11/generated/torch.optim.LBFGS.html); [D11: PyTorch 2.11: Automatic Mixed Precision examples](https://docs.pytorch.org/docs/2.11/notes/amp_examples.html); [D12: PyTorch 2.11: DataParallel](https://docs.pytorch.org/docs/2.11/generated/torch.nn.DataParallel.html).

<a id="stage-19"></a>

19. **Адаптивное добавление точек: RAR.**

   Основные файлы: [airfoil_pinn/training/components/rar.py](<../airfoil_pinn/training/components/rar.py>).

   - **19.1.** Разобрать RARGenerationResult, RARManager, FlowRAR и BoundaryLayerRAR.
   - **19.2.** Проследить источники кандидатов, score, квантиль, степень невязки, распределение количества между ветвями и режимами.
   - **19.3.** Разобрать геометрическую фильтрацию, jitter, минимальные расстояния, исключение пристенного слоя и дубликаты.
   - **19.4.** Для пристенной ветви объяснить контур, нормали, eta, delta и пакетную оценку кандидатов.
   - **19.5.** Сравнить современный adaptive и legacy-пути в Trainer, сохраняя фактические различия.
   - **19.6.** Проследить изменение Dataset/DataLoader, SA-параметров, optimizer и FVM-сетки после добавления.
   - **19.7.** Разобрать сохранение RAR-состояния, SDF-согласованность, историю, карты и гистограммы.

   Источники: [S4: Wu et al., A comprehensive study of non-adaptive and residual-based adaptive sampling for PINNs](https://arxiv.org/abs/2207.10289).

<a id="stage-20"></a>

20. **Интерактивное управление работающим обучением.**

   Основные файлы: [airfoil_pinn/training/key_listener.py](<../airfoil_pinn/training/key_listener.py>), [airfoil_pinn/training/components/interactive.py](<../airfoil_pinn/training/components/interactive.py>).

   - **20.1.** Разобрать KeyListener, отдельный поток, регистрацию keyboard hotkeys и флаги запросов.
   - **20.2.** Проследить Ctrl+P/Ctrl+W/Ctrl+R/Ctrl+S и Ctrl+L/Ctrl+K/Ctrl+U до конкретного изменения состояния.
   - **20.3.** Разобрать ручное сохранение полей/графиков и checkpoint, редактирование весов, изменение LR и reset optimizer.
   - **20.4.** Проверить синхронизацию конфигурации, активных весов, optimizer/scheduler и training contract после ручных изменений.
   - **20.5.** Разобрать ограничения L-BFGS, необязательность keyboard, stop и снятие обработчиков.
   - **20.6.** Установить точку цикла обучения, где запрос реально исполняется, и поведение при отмене ввода.

<a id="stage-21"></a>

21. **Checkpoint, контракты и воспроизводимость.**

   Основные файлы: [airfoil_pinn/training/checkpoint.py](<../airfoil_pinn/training/checkpoint.py>), [airfoil_pinn/training/contract.py](<../airfoil_pinn/training/contract.py>), [airfoil_pinn/training/reproducibility.py](<../airfoil_pinn/training/reproducibility.py>).

   - **21.1.** Разобрать CheckpointManager, атомарную запись, unwrap модели и формат состояния.
   - **21.2.** Перечислить model/optimizer/scheduler/SA, epoch/history, Trainer/DataHandler/RAR и RNG-состояния.
   - **21.3.** Разобрать TrainingContract, fingerprint, перечень сравниваемых полей и сообщения о различиях.
   - **21.4.** Проследить строгий resume, finetune и legacy-checkpoint, порядок проверки до изменения состояния.
   - **21.5.** Разобрать Python/NumPy/Torch CPU/CUDA/DataLoader RNG и ограничения воспроизводимости.
   - **21.6.** Сопоставить автоматическое и ручное сохранение, восстановление после RAR и изменение boundary/FVM/input contracts.

   Источники: [D13: PyTorch 2.11: Serialization semantics](https://docs.pytorch.org/docs/2.11/notes/serialization.html).

<a id="stage-22"></a>

22. **Метрики, интерполяция сечений и независимый эталон.**

   Основные файлы: [airfoil_pinn/training/components/metrics.py](<../airfoil_pinn/training/components/metrics.py>), [airfoil_pinn/training/reference.py](<../airfoil_pinn/training/reference.py>).

   - **22.1.** Разобрать историю, схему ключей, средние скорости на границах, кэш и его инвалидирование.
   - **22.2.** Проследить расход по FVM-граням и по граничным точкам, знаки, нормировку и погрешность баланса массы.
   - **22.3.** Подробно разобрать DataHandler.interpolate_profile/interpolate_fvm_profile, barycentric weights, stencil-кэши и fallback без сетки.
   - **22.4.** Вывести пересечение вертикального сечения с треугольниками и integrate_fvm_vertical_flux; проверить участки, занятые профилем.
   - **22.5.** Разобрать load_reference, CSV-колонки, пропуски, griddata, nearest fallback, NPY import/export и абсолютную/относительную ошибку.
   - **22.6.** Проследить согласование единиц, Cp/p и case_id при сравнении с эталоном.
   - **22.7.** Разобрать compute_grad_norms_separately, grad cosine и compute_ntk_condition; уточнить, какая именно приближённая величина вычисляется.
   - **22.8.** Проследить print_epoch_table, rates, save_log, compute_final_metrics и отличие малой невязки от ошибки решения.

   Источники: [D4: SciPy 1.17: Delaunay](https://docs.scipy.org/doc/scipy-1.17.0/reference/generated/scipy.spatial.Delaunay.html); [D5: SciPy 1.17: griddata](https://docs.scipy.org/doc/scipy-1.17.0/reference/generated/scipy.interpolate.griddata.html); [S7: Wang, Yu, Perdikaris, When and why PINNs fail to train: A neural tangent kernel perspective](https://arxiv.org/abs/2007.14527).

<a id="stage-23"></a>

23. **Физическая диагностика пограничного слоя.**

   Основные файлы: [airfoil_pinn/training/boundary_layer_diagnostics.py](<../airfoil_pinn/training/boundary_layer_diagnostics.py>).

   - **23.1.** Разобрать конфигурацию диагностики, выбор участков/сторон поверхности и создание колонок по нормали.
   - **23.2.** Проследить независимую оценку сети на новых точках, SDF, параметры режима и boundary transform.
   - **23.3.** Объяснить переход к касательной скорости, определение U_edge и пересечение порога delta99.
   - **23.4.** Разобрать стеночный наклон, Cf, Cp, displacement thickness, momentum thickness и shape factor.
   - **23.5.** Пояснить признаки разрешённости первого слоя, конечность значений и обработку профилей с недостаточными данными.
   - **23.6.** Разобрать сводные показатели, CSV/NPZ и графики Cp/Cf/delta99/дефицита скорости; отделить диагностическую оценку от турбулентной модели.

<a id="stage-24"></a>

24. **Визуализация, спектры, экспорт и telemetry.**

   Основные файлы: [airfoil_pinn/training/visualizer.py](<../airfoil_pinn/training/visualizer.py>), [airfoil_pinn/training/telemetry.py](<../airfoil_pinn/training/telemetry.py>).

   - **24.1.** Разобрать live plot, историю loss/градиентов, обновление окон и close.
   - **24.2.** Проследить карты u/v/p, физические/безразмерные координаты, интерполяцию, удаление дублей и геометрические маски.
   - **24.3.** Разобрать карты PDE residual и RAR score, гистограммы RAR и графики отдельных норм градиентов.
   - **24.4.** Объяснить профили скорости, 1D FFT, 2D FFT, радиальное усреднение, шаг сетки, частоты и влияние интерполяции/маскирования.
   - **24.5.** Разобрать save_fields_as_npy, plot_final_fields, CSV и final_report.txt, имена файлов и структуру каталогов.
   - **24.6.** Проследить RunTelemetry: преобразование NumPy/Torch в JSON, run_manifest, epochs.jsonl, запрет NaN/Inf и атомарность.
   - **24.7.** Сопоставить производители артефактов с UI и batch collector; проверить пути и схемы файлов.

   Источники: [D7: NumPy 2.4: Discrete Fourier Transform](https://numpy.org/doc/2.4/reference/routines.fft.html); [D14: Matplotlib 3.10.8: Backends](https://matplotlib.org/3.10.8/users/explain/figure/backends.html).

<a id="stage-25"></a>

25. **Серии экспериментов и перебор конфигураций.**

   Основные файлы: [airfoil_pinn/training/batch.py](<../airfoil_pinn/training/batch.py>).

   - **25.1.** Разобрать load_base_config и load_experiment_list, явный список экспериментов и --grid.
   - **25.2.** Проследить декартово произведение параметров, точечные пути, deep_update и имена экспериментов.
   - **25.3.** Разобрать каталоги, сохранение индивидуального YAML и запуск каждого эксперимента через subprocess.
   - **25.4.** Проследить stdout/stderr, коды завершения, подсчёт успехов/ошибок и частичный сбой серии.
   - **25.5.** Разобрать collect_metrics, ожидаемый metrics.csv и summary.csv; проверить совместимость с текущими экспортами.
   - **25.6.** Зафиксировать, что --parallel разбирается, но текущий цикл запускает эксперименты последовательно; проверить и другие расхождения интерфейса и реализации.

   Источники: [D2: PyYAML Documentation](https://pyyaml.org/wiki/PyYAMLDocumentation).

<a id="stage-26"></a>

26. **UI: геометрия, редактор, виджеты и демонстрационные элементы.**

   Основные файлы: [ui/geometry_frontend.py](<../ui/geometry_frontend.py>), [ui/theme.py](<../ui/theme.py>), [ui/mock_data.py](<../ui/mock_data.py>), [ui/widgets/geometry_canvas.py](<../ui/widgets/geometry_canvas.py>), [ui/widgets/log_panel.py](<../ui/widgets/log_panel.py>), [ui/widgets/metric_plot.py](<../ui/widgets/metric_plot.py>), [ui/widgets/result_canvas.py](<../ui/widgets/result_canvas.py>).

   - **26.1.** Разобрать GeometryFrontendSession, Point/GeometryStats/GeometryArtifacts, загрузку файлов, библиотеки, вставку координат и drag-and-drop.
   - **26.2.** Проследить встроенные генераторы NACA и аппроксимацию пресета RAE2822, домен, редактирование контура и экспорт.
   - **26.3.** Разобрать GeometryCanvas: мировые/экранные координаты, zoom/pan, выбор и перетаскивание вершины, события мыши.
   - **26.4.** Проследить отрисовку профиля, области, сетки, точек, подписей, таймер и сокращение числа отображаемых точек.
   - **26.5.** Отдельно пометить mock collocation и формульную анимацию streamlines, не связанную с предсказанным полем.
   - **26.6.** Разобрать LogPanel, MetricPlot, ResultImageCanvas, Palette и stylesheet.
   - **26.7.** Сопоставить геометрические артефакты UI с backend и проверить совместимость форматов.

<a id="stage-27"></a>

27. **UI: приложение, обучение, процессы и результаты.**

   Основные файлы: [ui/app.py](<../ui/app.py>), [ui/main_window.py](<../ui/main_window.py>), [ui/workers.py](<../ui/workers.py>), [ui/mock_backend.py](<../ui/mock_backend.py>).

   - **27.1.** Разобрать QApplication/MainWindow, меню, toolbar, навигацию и все панели: geometry/collocation/fluid/boundaries/initialization/training/results.
   - **27.2.** Для каждого элемента определить статус: подключён к backend, используется только в UI, mock или задел.
   - **27.3.** Проследить сбор UI-конфигурации, JSON save/load, QSettings, base YAML и создание backend_config.yaml.
   - **27.4.** Разобрать BackendWorker/QThread, сигналы, subprocess.Popen, чтение потоков, buffering и парсер метрик/прогресса.
   - **27.5.** Проследить запуск, остановку дерева процессов Windows/POSIX, ошибки, cleanup и закрытие окна.
   - **27.6.** Разобрать ResultsScanWorker, поиск картинок/отчёта, просмотр и открытие каталога.
   - **27.7.** Разобрать mock_backend, искусственные метрики, генерацию графиков и резервную запись PNG.
   - **27.8.** Проверить эффект UI-параметров profile/point count/data_weight и связь с реальными points_file/mesh_file.

   Источники: [D2: PyYAML Documentation](https://pyyaml.org/wiki/PyYAMLDocumentation); [D15: Qt for Python: QThread](https://doc.qt.io/qtforpython-6/PySide6/QtCore/QThread.html).

<a id="stage-28"></a>

28. **Все тестовые сценарии и подтверждение поведения.**

   - **28.1.** Разобрать каждый файл tests и ui/smoke_tests.py, включая fixtures, вспомогательные классы и вложенные функции.
   - **28.2.** Сгруппировать проверки нормировки, геометрии/SDF, входов/case_id, PDE/BC, hard constraints и Euler.
   - **28.3.** Отдельно разобрать проверки FVM objective/schedule/geometry, расхода по сечениям и parametric FVM.
   - **28.4.** Разобрать проверки SIREN, finite guards, telemetry, DataLoader spawn и checkpoint contracts.
   - **28.5.** Проследить end-to-end save/resume, RNG и новые точки поверхности при hard wall.
   - **28.6.** Установить зависимости тестов от локальных workspace-данных, временных каталогов и Qt; различить test scripts и helper fvm_config.py.
   - **28.7.** Запустить релевантные существующие проверки после разбора prerequisites; сообщать фактические passed/failed/skipped и непроверенные ветви.

<a id="stage-29"></a>

29. **Сквозные пайплайны, построчный разбор и критерии завершения.**

   - **29.1.** Построить трассу подготовки задачи: профиль → домен → точки/сетка → жидкость/условия → нормированные артефакты.
   - **29.2.** Построить трассу обучения: конфигурация → solver → batch → SDF/features → сеть → PDE/BC/FVM → backward → update.
   - **29.3.** Отдельно детализировать параметрические режимы, normal-layer sampling, RAR, SA и перестройку сетки.
   - **29.4.** Построить трассы resume/finetune, ручного управления, независимой диагностики/эталона и экспорта.
   - **29.5.** Построить трассы серии экспериментов, UI-геометрии, реального UI-обучения и mock-режима.
   - **29.6.** Для каждой трассы указать функции, файлы, условия ветвления, формы/единицы данных, состояние, артефакты и ошибки.
   - **29.7.** Для каждого определения из индекса объяснить все строки, вызывающий контекст, библиотечный API, математический смысл и проверки.
   - **29.8.** Сверить индекс повторно; отдельно зарегистрировать недоступные ветви, заглушки, legacy-код, несоответствия документов и фактически проверенные результаты.

**Паспорт построчного объяснения.**

Для каждой функции/метода: назначение; полный путь вызова; входы/выходы; форма, dtype, device и единицы; ветви конфигурации; построчное действие; математическая формула; связь с autograd; изменения состояния/I/O; исключения; вызываемые helper-функции; конкретная документация API; существующий тест или явно отмеченный пробел проверки.

Вложенные closure, callbacks, property/setter, lambda и код вне функций входят в разбор. Повторяющиеся конструкции можно объяснять через ссылки на уже разобранный механизм, но их конкретную роль в текущем месте нужно указывать.

Для каждого этапа результатом являются карта вызовов, паспорт данных/артефактов и заметки о фактической доступности ветвей. Для научных методов добавляются вывод формул и сопоставление с первоисточником. Для каждой проверки фиксируются предпосылки и фактический результат, без подмены проверки наличием тестового файла.

**Основания и документация.**

| Код | Где используется | Источник |
|---|---|---|
| S1 | Основы PINN; этапы 10, 14–18 | [Raissi, Perdikaris, Karniadakis (2019), Physics-informed neural networks; авторская страница](https://maziarraissi.github.io/PINNs/) |
| S2 | SIREN; этап 13 | [Sitzmann et al. (2020), Implicit Neural Representations with Periodic Activation Functions](https://arxiv.org/abs/2006.09661) |
| S3 | SA-PINN; этапы 13, 16, 18 | [McClenny, Braga-Neto (2020), Self-Adaptive PINNs](https://arxiv.org/abs/2009.04544) |
| S4 | RAR; этап 19 | [Wu et al., A comprehensive study of non-adaptive and residual-based adaptive sampling for PINNs](https://arxiv.org/abs/2207.10289) |
| S5 | Distance-based hard BC; этапы 12, 15 | [Sukumar, Srivastava, Exact imposition of boundary conditions with distance functions](https://arxiv.org/abs/2104.08426) |
| S6 | Параметрические входы; этапы 11, 13 | [Naderibeni et al. (2024), Learning solutions of parametric Navier–Stokes with PINNs](https://arxiv.org/abs/2402.03153) |
| S7 | NTK-диагностика; этап 22 | [Wang, Yu, Perdikaris, When and why PINNs fail to train: A neural tangent kernel perspective](https://arxiv.org/abs/2007.14527) |
| S8 | FVM; этапы 9, 17 | [Eymard, Gallouët, Herbin (2000), Finite Volume Methods; библиография автора](https://www.i2m.univ-amu.fr/perso/thierry.gallouet/publi.d/publi03.html) |
| D1 | Инвентаризация; этап 1 | [Python: ast](https://docs.python.org/3/library/ast.html) |
| D2 | YAML; этапы 4, 25, 27 | [PyYAML Documentation](https://pyyaml.org/wiki/PyYAMLDocumentation) |
| D3 | Геометрия; этапы 6–9 | [Shapely 2.1.2 User Manual](https://shapely.readthedocs.io/en/2.1.2/manual.html) |
| D4 | Триангуляция; этапы 9, 22 | [SciPy 1.17: Delaunay](https://docs.scipy.org/doc/scipy-1.17.0/reference/generated/scipy.spatial.Delaunay.html) |
| D5 | Интерполяция; этапы 10, 22, 24 | [SciPy 1.17: griddata](https://docs.scipy.org/doc/scipy-1.17.0/reference/generated/scipy.interpolate.griddata.html) |
| D6 | Формы массивов; этапы 3, 11, 13–17 | [NumPy 2.4: Broadcasting](https://numpy.org/doc/2.4/user/basics.broadcasting.html) |
| D7 | Спектры; этап 24 | [NumPy 2.4: Discrete Fourier Transform](https://numpy.org/doc/2.4/reference/routines.fft.html) |
| D8 | Батчи; этап 11 | [PyTorch 2.11: Dataset/DataLoader](https://docs.pytorch.org/docs/2.11/data.html) |
| D9 | Производные; этапы 12–18, 22 | [PyTorch 2.11: torch.autograd.grad](https://docs.pytorch.org/docs/2.11/generated/torch.autograd.grad.html) |
| D10 | Оптимизация; этап 18 | [PyTorch 2.11: LBFGS](https://docs.pytorch.org/docs/2.11/generated/torch.optim.LBFGS.html) |
| D11 | AMP; этап 18 | [PyTorch 2.11: Automatic Mixed Precision examples](https://docs.pytorch.org/docs/2.11/notes/amp_examples.html) |
| D12 | Несколько GPU; этап 18 | [PyTorch 2.11: DataParallel](https://docs.pytorch.org/docs/2.11/generated/torch.nn.DataParallel.html) |
| D13 | Checkpoint; этап 21 | [PyTorch 2.11: Serialization semantics](https://docs.pytorch.org/docs/2.11/notes/serialization.html) |
| D14 | Графики; этап 24 | [Matplotlib 3.10.8: Backends](https://matplotlib.org/3.10.8/users/explain/figure/backends.html) |
| D15 | Потоки UI; этап 27 | [Qt for Python: QThread](https://doc.qt.io/qtforpython-6/PySide6/QtCore/QThread.html) |

Документация SciPy выбрана для линии 1.17; локально обнаружена 1.17.1. Для отличающихся версий при разборе API дополнительно проверяется установленная реализация. В научной части ограничения модели и различия с публикациями описываются явно. Исходные контракты проекта: [PINN_CFD_CONTRACT.md](PINN_CFD_CONTRACT.md) и [FVM_OBJECTIVE.md](FVM_OBJECTIVE.md).
