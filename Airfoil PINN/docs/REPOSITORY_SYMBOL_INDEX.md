**Source and symbol coverage index for the study plan**

Snapshot: 2026-09-16. Scope: Airfoil_PINN.py, airfoil_pinn/, ui/, tests/.

This is a static inventory and a mapping to planned review stages, not execution coverage or evidence that the code has already been explained or tested.
Every definition, including private methods, property setters, nested functions and lambdas, is listed. Counts represent definitions, not unique callable names or product capabilities.
Imports, module/class attributes, decorators, docstrings and executable module bodies are also in scope of the line-by-line review. Generated dataclass methods and imported library implementations are not counted as local source definitions.

[Detailed study plan](REPOSITORY_STUDY_PLAN.md)

Files: **111**. Class definitions: **156**. Function/method definitions: **1201**. Lambdas: **31**.

| Group | Files | Classes | Function/method definitions | Lambdas |
|---|---:|---:|---:|---:|
| application | 78 | 103 | 780 | 24 |
| ui | 13 | 15 | 225 | 4 |
| tests | 20 | 38 | 196 | 3 |

The tests group includes ui/smoke_tests.py. The application group includes Airfoil_PINN.py.
Each file has a primary stage. Cross-cutting methods in DataHandler, Trainer, Solver and LossComputer are additionally covered by the related stages explicitly named in the plan.

| Source file | Primary stage | Lines | Classes | Functions/methods | Lambdas |
|---|---|---:|---:|---:|---:|
| [Airfoil_PINN.py](<../Airfoil_PINN.py>) | [2](REPOSITORY_STUDY_PLAN.md#stage-2) | 33 | 0 | 1 | 0 |
| [airfoil_pinn/__init__.py](<../airfoil_pinn/__init__.py>) | [3](REPOSITORY_STUDY_PLAN.md#stage-3) | 14 | 0 | 0 | 0 |
| [airfoil_pinn/__main__.py](<../airfoil_pinn/__main__.py>) | [2](REPOSITORY_STUDY_PLAN.md#stage-2) | 28 | 0 | 1 | 0 |
| [airfoil_pinn/app/__init__.py](<../airfoil_pinn/app/__init__.py>) | [3](REPOSITORY_STUDY_PLAN.md#stage-3) | 5 | 0 | 0 | 0 |
| [airfoil_pinn/app/cli.py](<../airfoil_pinn/app/cli.py>) | [2](REPOSITORY_STUDY_PLAN.md#stage-2) | 507 | 0 | 16 | 0 |
| [airfoil_pinn/app/module_registry.py](<../airfoil_pinn/app/module_registry.py>) | [2](REPOSITORY_STUDY_PLAN.md#stage-2) | 123 | 1 | 4 | 0 |
| [airfoil_pinn/app/paths.py](<../airfoil_pinn/app/paths.py>) | [2](REPOSITORY_STUDY_PLAN.md#stage-2) | 145 | 1 | 4 | 0 |
| [airfoil_pinn/app/runtime.py](<../airfoil_pinn/app/runtime.py>) | [2](REPOSITORY_STUDY_PLAN.md#stage-2) | 268 | 0 | 12 | 0 |
| [airfoil_pinn/config/__init__.py](<../airfoil_pinn/config/__init__.py>) | [3](REPOSITORY_STUDY_PLAN.md#stage-3) | 13 | 0 | 0 | 0 |
| [airfoil_pinn/config/manager.py](<../airfoil_pinn/config/manager.py>) | [4](REPOSITORY_STUDY_PLAN.md#stage-4) | 769 | 1 | 13 | 0 |
| [airfoil_pinn/config/validation.py](<../airfoil_pinn/config/validation.py>) | [4](REPOSITORY_STUDY_PLAN.md#stage-4) | 641 | 1 | 6 | 0 |
| [airfoil_pinn/core/__init__.py](<../airfoil_pinn/core/__init__.py>) | [3](REPOSITORY_STUDY_PLAN.md#stage-3) | 12 | 0 | 0 | 0 |
| [airfoil_pinn/core/constants.py](<../airfoil_pinn/core/constants.py>) | [3](REPOSITORY_STUDY_PLAN.md#stage-3) | 249 | 0 | 0 | 0 |
| [airfoil_pinn/core/type_hints.py](<../airfoil_pinn/core/type_hints.py>) | [3](REPOSITORY_STUDY_PLAN.md#stage-3) | 713 | 38 | 0 | 0 |
| [airfoil_pinn/data/__init__.py](<../airfoil_pinn/data/__init__.py>) | [3](REPOSITORY_STUDY_PLAN.md#stage-3) | 71 | 0 | 1 | 0 |
| [airfoil_pinn/data/batch.py](<../airfoil_pinn/data/batch.py>) | [11](REPOSITORY_STUDY_PLAN.md#stage-11) | 89 | 1 | 6 | 0 |
| [airfoil_pinn/data/flow.py](<../airfoil_pinn/data/flow.py>) | [11](REPOSITORY_STUDY_PLAN.md#stage-11) | 224 | 2 | 10 | 0 |
| [airfoil_pinn/data/handler.py](<../airfoil_pinn/data/handler.py>) | [11](REPOSITORY_STUDY_PLAN.md#stage-11) | 2320 | 4 | 51 | 0 |
| [airfoil_pinn/data/initializer.py](<../airfoil_pinn/data/initializer.py>) | [10](REPOSITORY_STUDY_PLAN.md#stage-10) | 1406 | 1 | 29 | 0 |
| [airfoil_pinn/data/initializer_cli.py](<../airfoil_pinn/data/initializer_cli.py>) | [10](REPOSITORY_STUDY_PLAN.md#stage-10) | 7 | 0 | 0 | 0 |
| [airfoil_pinn/data/normalization.py](<../airfoil_pinn/data/normalization.py>) | [10](REPOSITORY_STUDY_PLAN.md#stage-10) | 358 | 1 | 20 | 0 |
| [airfoil_pinn/data/sdf_diagnostics.py](<../airfoil_pinn/data/sdf_diagnostics.py>) | [8](REPOSITORY_STUDY_PLAN.md#stage-8) | 290 | 0 | 12 | 0 |
| [airfoil_pinn/geometry/__init__.py](<../airfoil_pinn/geometry/__init__.py>) | [3](REPOSITORY_STUDY_PLAN.md#stage-3) | 32 | 0 | 0 | 0 |
| [airfoil_pinn/geometry/cli.py](<../airfoil_pinn/geometry/cli.py>) | [6](REPOSITORY_STUDY_PLAN.md#stage-6) | 13 | 0 | 0 | 0 |
| [airfoil_pinn/geometry/core.py](<../airfoil_pinn/geometry/core.py>) | [6](REPOSITORY_STUDY_PLAN.md#stage-6) | 828 | 3 | 23 | 0 |
| [airfoil_pinn/geometry/sdf.py](<../airfoil_pinn/geometry/sdf.py>) | [12](REPOSITORY_STUDY_PLAN.md#stage-12) | 625 | 2 | 17 | 0 |
| [airfoil_pinn/physics/__init__.py](<../airfoil_pinn/physics/__init__.py>) | [3](REPOSITORY_STUDY_PLAN.md#stage-3) | 12 | 0 | 0 | 0 |
| [airfoil_pinn/physics/conditions.py](<../airfoil_pinn/physics/conditions.py>) | [5](REPOSITORY_STUDY_PLAN.md#stage-5) | 719 | 2 | 22 | 0 |
| [airfoil_pinn/physics/conditions_cli.py](<../airfoil_pinn/physics/conditions_cli.py>) | [5](REPOSITORY_STUDY_PLAN.md#stage-5) | 7 | 0 | 0 | 0 |
| [airfoil_pinn/physics/fluids.py](<../airfoil_pinn/physics/fluids.py>) | [5](REPOSITORY_STUDY_PLAN.md#stage-5) | 397 | 2 | 12 | 0 |
| [airfoil_pinn/physics/fluids_cli.py](<../airfoil_pinn/physics/fluids_cli.py>) | [5](REPOSITORY_STUDY_PLAN.md#stage-5) | 7 | 0 | 0 | 0 |
| [airfoil_pinn/physics/fvm.py](<../airfoil_pinn/physics/fvm.py>) | [17](REPOSITORY_STUDY_PLAN.md#stage-17) | 870 | 1 | 26 | 0 |
| [airfoil_pinn/physics/pde.py](<../airfoil_pinn/physics/pde.py>) | [14](REPOSITORY_STUDY_PLAN.md#stage-14) | 221 | 1 | 5 | 0 |
| [airfoil_pinn/sampling/__init__.py](<../airfoil_pinn/sampling/__init__.py>) | [3](REPOSITORY_STUDY_PLAN.md#stage-3) | 37 | 0 | 0 | 0 |
| [airfoil_pinn/sampling/cli.py](<../airfoil_pinn/sampling/cli.py>) | [7](REPOSITORY_STUDY_PLAN.md#stage-7) | 12 | 0 | 0 | 0 |
| [airfoil_pinn/sampling/core.py](<../airfoil_pinn/sampling/core.py>) | [7](REPOSITORY_STUDY_PLAN.md#stage-7) | 2510 | 4 | 45 | 3 |
| [airfoil_pinn/sampling/mesh/__init__.py](<../airfoil_pinn/sampling/mesh/__init__.py>) | [3](REPOSITORY_STUDY_PLAN.md#stage-3) | 17 | 0 | 0 | 0 |
| [airfoil_pinn/sampling/mesh/core.py](<../airfoil_pinn/sampling/mesh/core.py>) | [9](REPOSITORY_STUDY_PLAN.md#stage-9) | 2255 | 5 | 57 | 3 |
| [airfoil_pinn/sampling/normal_layer.py](<../airfoil_pinn/sampling/normal_layer.py>) | [8](REPOSITORY_STUDY_PLAN.md#stage-8) | 645 | 3 | 21 | 13 |
| [airfoil_pinn/training/__init__.py](<../airfoil_pinn/training/__init__.py>) | [3](REPOSITORY_STUDY_PLAN.md#stage-3) | 25 | 0 | 0 | 0 |
| [airfoil_pinn/training/batch.py](<../airfoil_pinn/training/batch.py>) | [25](REPOSITORY_STUDY_PLAN.md#stage-25) | 297 | 0 | 9 | 0 |
| [airfoil_pinn/training/boundary.py](<../airfoil_pinn/training/boundary.py>) | [15](REPOSITORY_STUDY_PLAN.md#stage-15) | 167 | 1 | 7 | 0 |
| [airfoil_pinn/training/boundary_layer_diagnostics.py](<../airfoil_pinn/training/boundary_layer_diagnostics.py>) | [23](REPOSITORY_STUDY_PLAN.md#stage-23) | 888 | 2 | 29 | 1 |
| [airfoil_pinn/training/checkpoint.py](<../airfoil_pinn/training/checkpoint.py>) | [21](REPOSITORY_STUDY_PLAN.md#stage-21) | 128 | 1 | 4 | 0 |
| [airfoil_pinn/training/components/__init__.py](<../airfoil_pinn/training/components/__init__.py>) | [3](REPOSITORY_STUDY_PLAN.md#stage-3) | 63 | 0 | 1 | 0 |
| [airfoil_pinn/training/components/interactive.py](<../airfoil_pinn/training/components/interactive.py>) | [20](REPOSITORY_STUDY_PLAN.md#stage-20) | 327 | 1 | 8 | 0 |
| [airfoil_pinn/training/components/loss_computer.py](<../airfoil_pinn/training/components/loss_computer.py>) | [16](REPOSITORY_STUDY_PLAN.md#stage-16) | 1320 | 1 | 31 | 0 |
| [airfoil_pinn/training/components/metrics.py](<../airfoil_pinn/training/components/metrics.py>) | [22](REPOSITORY_STUDY_PLAN.md#stage-22) | 1714 | 1 | 30 | 0 |
| [airfoil_pinn/training/components/optimizers.py](<../airfoil_pinn/training/components/optimizers.py>) | [18](REPOSITORY_STUDY_PLAN.md#stage-18) | 102 | 2 | 5 | 0 |
| [airfoil_pinn/training/components/rar.py](<../airfoil_pinn/training/components/rar.py>) | [19](REPOSITORY_STUDY_PLAN.md#stage-19) | 1030 | 4 | 28 | 4 |
| [airfoil_pinn/training/components/trainer.py](<../airfoil_pinn/training/components/trainer.py>) | [18](REPOSITORY_STUDY_PLAN.md#stage-18) | 2631 | 1 | 42 | 0 |
| [airfoil_pinn/training/components/utils.py](<../airfoil_pinn/training/components/utils.py>) | [3](REPOSITORY_STUDY_PLAN.md#stage-3) | 0 | 0 | 0 | 0 |
| [airfoil_pinn/training/contract.py](<../airfoil_pinn/training/contract.py>) | [21](REPOSITORY_STUDY_PLAN.md#stage-21) | 265 | 1 | 6 | 0 |
| [airfoil_pinn/training/input_features.py](<../airfoil_pinn/training/input_features.py>) | [13](REPOSITORY_STUDY_PLAN.md#stage-13) | 354 | 3 | 14 | 0 |
| [airfoil_pinn/training/key_listener.py](<../airfoil_pinn/training/key_listener.py>) | [20](REPOSITORY_STUDY_PLAN.md#stage-20) | 123 | 1 | 11 | 0 |
| [airfoil_pinn/training/learning_cli.py](<../airfoil_pinn/training/learning_cli.py>) | [2](REPOSITORY_STUDY_PLAN.md#stage-2) | 62 | 0 | 1 | 0 |
| [airfoil_pinn/training/menus/__init__.py](<../airfoil_pinn/training/menus/__init__.py>) | [3](REPOSITORY_STUDY_PLAN.md#stage-3) | 98 | 0 | 1 | 0 |
| [airfoil_pinn/training/menus/architecture.py](<../airfoil_pinn/training/menus/architecture.py>) | [4](REPOSITORY_STUDY_PLAN.md#stage-4) | 255 | 0 | 6 | 0 |
| [airfoil_pinn/training/menus/common.py](<../airfoil_pinn/training/menus/common.py>) | [4](REPOSITORY_STUDY_PLAN.md#stage-4) | 20 | 0 | 2 | 0 |
| [airfoil_pinn/training/menus/configuration.py](<../airfoil_pinn/training/menus/configuration.py>) | [4](REPOSITORY_STUDY_PLAN.md#stage-4) | 209 | 0 | 5 | 0 |
| [airfoil_pinn/training/menus/loss.py](<../airfoil_pinn/training/menus/loss.py>) | [4](REPOSITORY_STUDY_PLAN.md#stage-4) | 236 | 0 | 2 | 0 |
| [airfoil_pinn/training/menus/process.py](<../airfoil_pinn/training/menus/process.py>) | [4](REPOSITORY_STUDY_PLAN.md#stage-4) | 84 | 0 | 1 | 0 |
| [airfoil_pinn/training/menus/results.py](<../airfoil_pinn/training/menus/results.py>) | [4](REPOSITORY_STUDY_PLAN.md#stage-4) | 165 | 0 | 3 | 0 |
| [airfoil_pinn/training/menus/strategy.py](<../airfoil_pinn/training/menus/strategy.py>) | [4](REPOSITORY_STUDY_PLAN.md#stage-4) | 247 | 0 | 1 | 0 |
| [airfoil_pinn/training/models.py](<../airfoil_pinn/training/models.py>) | [13](REPOSITORY_STUDY_PLAN.md#stage-13) | 321 | 3 | 11 | 0 |
| [airfoil_pinn/training/network_diagram.py](<../airfoil_pinn/training/network_diagram.py>) | [13](REPOSITORY_STUDY_PLAN.md#stage-13) | 167 | 0 | 2 | 0 |
| [airfoil_pinn/training/numerics.py](<../airfoil_pinn/training/numerics.py>) | [18](REPOSITORY_STUDY_PLAN.md#stage-18) | 131 | 2 | 4 | 0 |
| [airfoil_pinn/training/progress.py](<../airfoil_pinn/training/progress.py>) | [18](REPOSITORY_STUDY_PLAN.md#stage-18) | 43 | 0 | 1 | 0 |
| [airfoil_pinn/training/reference.py](<../airfoil_pinn/training/reference.py>) | [22](REPOSITORY_STUDY_PLAN.md#stage-22) | 214 | 0 | 5 | 0 |
| [airfoil_pinn/training/reproducibility.py](<../airfoil_pinn/training/reproducibility.py>) | [21](REPOSITORY_STUDY_PLAN.md#stage-21) | 65 | 0 | 2 | 0 |
| [airfoil_pinn/training/runner.py](<../airfoil_pinn/training/runner.py>) | [18](REPOSITORY_STUDY_PLAN.md#stage-18) | 108 | 0 | 2 | 0 |
| [airfoil_pinn/training/solver.py](<../airfoil_pinn/training/solver.py>) | [18](REPOSITORY_STUDY_PLAN.md#stage-18) | 1623 | 1 | 32 | 0 |
| [airfoil_pinn/training/telemetry.py](<../airfoil_pinn/training/telemetry.py>) | [24](REPOSITORY_STUDY_PLAN.md#stage-24) | 79 | 1 | 4 | 0 |
| [airfoil_pinn/training/visualizer.py](<../airfoil_pinn/training/visualizer.py>) | [24](REPOSITORY_STUDY_PLAN.md#stage-24) | 1132 | 1 | 27 | 0 |
| [airfoil_pinn/utils/__init__.py](<../airfoil_pinn/utils/__init__.py>) | [3](REPOSITORY_STUDY_PLAN.md#stage-3) | 56 | 0 | 0 | 0 |
| [airfoil_pinn/utils/auxiliary_utils.py](<../airfoil_pinn/utils/auxiliary_utils.py>) | [3](REPOSITORY_STUDY_PLAN.md#stage-3) | 137 | 0 | 3 | 0 |
| [airfoil_pinn/utils/io_utils.py](<../airfoil_pinn/utils/io_utils.py>) | [3](REPOSITORY_STUDY_PLAN.md#stage-3) | 320 | 0 | 9 | 0 |
| [airfoil_pinn/utils/ui_utils.py](<../airfoil_pinn/utils/ui_utils.py>) | [3](REPOSITORY_STUDY_PLAN.md#stage-3) | 448 | 2 | 17 | 0 |
| [ui/__init__.py](<../ui/__init__.py>) | [3](REPOSITORY_STUDY_PLAN.md#stage-3) | 1 | 0 | 0 | 0 |
| [ui/app.py](<../ui/app.py>) | [27](REPOSITORY_STUDY_PLAN.md#stage-27) | 30 | 0 | 1 | 0 |
| [ui/geometry_frontend.py](<../ui/geometry_frontend.py>) | [26](REPOSITORY_STUDY_PLAN.md#stage-26) | 506 | 5 | 33 | 1 |
| [ui/main_window.py](<../ui/main_window.py>) | [27](REPOSITORY_STUDY_PLAN.md#stage-27) | 1998 | 1 | 117 | 3 |
| [ui/mock_backend.py](<../ui/mock_backend.py>) | [27](REPOSITORY_STUDY_PLAN.md#stage-27) | 155 | 0 | 6 | 0 |
| [ui/mock_data.py](<../ui/mock_data.py>) | [26](REPOSITORY_STUDY_PLAN.md#stage-26) | 106 | 2 | 5 | 0 |
| [ui/smoke_tests.py](<../ui/smoke_tests.py>) | [28](REPOSITORY_STUDY_PLAN.md#stage-28) | 109 | 0 | 6 | 1 |
| [ui/theme.py](<../ui/theme.py>) | [26](REPOSITORY_STUDY_PLAN.md#stage-26) | 191 | 1 | 1 | 0 |
| [ui/widgets/__init__.py](<../ui/widgets/__init__.py>) | [3](REPOSITORY_STUDY_PLAN.md#stage-3) | 1 | 0 | 0 | 0 |
| [ui/widgets/geometry_canvas.py](<../ui/widgets/geometry_canvas.py>) | [26](REPOSITORY_STUDY_PLAN.md#stage-26) | 411 | 1 | 34 | 0 |
| [ui/widgets/log_panel.py](<../ui/widgets/log_panel.py>) | [26](REPOSITORY_STUDY_PLAN.md#stage-26) | 27 | 1 | 4 | 0 |
| [ui/widgets/metric_plot.py](<../ui/widgets/metric_plot.py>) | [26](REPOSITORY_STUDY_PLAN.md#stage-26) | 78 | 1 | 4 | 0 |
| [ui/widgets/result_canvas.py](<../ui/widgets/result_canvas.py>) | [26](REPOSITORY_STUDY_PLAN.md#stage-26) | 72 | 1 | 3 | 0 |
| [ui/workers.py](<../ui/workers.py>) | [27](REPOSITORY_STUDY_PLAN.md#stage-27) | 343 | 2 | 17 | 0 |
| [tests/boundary_layer_diagnostics_smoke.py](<../tests/boundary_layer_diagnostics_smoke.py>) | [28](REPOSITORY_STUDY_PLAN.md#stage-28) | 135 | 3 | 6 | 0 |
| [tests/boundary_normal_bc_smoke.py](<../tests/boundary_normal_bc_smoke.py>) | [28](REPOSITORY_STUDY_PLAN.md#stage-28) | 358 | 3 | 12 | 0 |
| [tests/config_contract_smoke.py](<../tests/config_contract_smoke.py>) | [28](REPOSITORY_STUDY_PLAN.md#stage-28) | 312 | 3 | 18 | 0 |
| [tests/differentiable_sdf_smoke.py](<../tests/differentiable_sdf_smoke.py>) | [28](REPOSITORY_STUDY_PLAN.md#stage-28) | 465 | 1 | 14 | 0 |
| [tests/end_to_end_pipeline_smoke.py](<../tests/end_to_end_pipeline_smoke.py>) | [28](REPOSITORY_STUDY_PLAN.md#stage-28) | 261 | 1 | 5 | 0 |
| [tests/euler_wall_bc_smoke.py](<../tests/euler_wall_bc_smoke.py>) | [28](REPOSITORY_STUDY_PLAN.md#stage-28) | 211 | 2 | 9 | 0 |
| [tests/fvm_config.py](<../tests/fvm_config.py>) | [28](REPOSITORY_STUDY_PLAN.md#stage-28) | 118 | 1 | 2 | 0 |
| [tests/geometry_normalization_smoke.py](<../tests/geometry_normalization_smoke.py>) | [28](REPOSITORY_STUDY_PLAN.md#stage-28) | 210 | 1 | 10 | 0 |
| [tests/hard_boundary_transform_smoke.py](<../tests/hard_boundary_transform_smoke.py>) | [28](REPOSITORY_STUDY_PLAN.md#stage-28) | 318 | 3 | 14 | 0 |
| [tests/nondimensionalization_smoke.py](<../tests/nondimensionalization_smoke.py>) | [28](REPOSITORY_STUDY_PLAN.md#stage-28) | 160 | 1 | 7 | 0 |
| [tests/normal_layer_smoke.py](<../tests/normal_layer_smoke.py>) | [28](REPOSITORY_STUDY_PLAN.md#stage-28) | 78 | 0 | 5 | 0 |
| [tests/parametric_fvm_smoke.py](<../tests/parametric_fvm_smoke.py>) | [28](REPOSITORY_STUDY_PLAN.md#stage-28) | 215 | 3 | 11 | 0 |
| [tests/parametric_inputs_smoke.py](<../tests/parametric_inputs_smoke.py>) | [28](REPOSITORY_STUDY_PLAN.md#stage-28) | 233 | 3 | 9 | 0 |
| [tests/parametric_physics_smoke.py](<../tests/parametric_physics_smoke.py>) | [28](REPOSITORY_STUDY_PLAN.md#stage-28) | 357 | 3 | 12 | 0 |
| [tests/profile_section_metrics_smoke.py](<../tests/profile_section_metrics_smoke.py>) | [28](REPOSITORY_STUDY_PLAN.md#stage-28) | 281 | 2 | 9 | 0 |
| [tests/refactor_smoke.py](<../tests/refactor_smoke.py>) | [28](REPOSITORY_STUDY_PLAN.md#stage-28) | 399 | 0 | 10 | 0 |
| [tests/test_fvm_objective.py](<../tests/test_fvm_objective.py>) | [28](REPOSITORY_STUDY_PLAN.md#stage-28) | 407 | 3 | 17 | 0 |
| [tests/test_loss_metric_logging.py](<../tests/test_loss_metric_logging.py>) | [28](REPOSITORY_STUDY_PLAN.md#stage-28) | 219 | 1 | 6 | 2 |
| [tests/training_robustness_smoke.py](<../tests/training_robustness_smoke.py>) | [28](REPOSITORY_STUDY_PLAN.md#stage-28) | 248 | 4 | 14 | 0 |

**Airfoil_PINN.py**

Primary stage: [2. Точки входа, окружение, пути и консольное приложение](REPOSITORY_STUDY_PLAN.md#stage-2).
Source: [open file](<../Airfoil_PINN.py>). Lines: 33. SHA-256: `0e20dcc8f176a187157cb86768ff17c190cb5e93b66f99bf7fe3d511bdac3916`.

Module body, imports, exports and class attributes: planned for line-by-line review.

| Lines | Kind | Qualified name | Decorators |
|---|---|---|---|
| 16-29 | function | `main` |  |

**airfoil_pinn/__init__.py**

Primary stage: [3. Служебные типы, импорты и общие утилиты](REPOSITORY_STUDY_PLAN.md#stage-3).
Source: [open file](<../airfoil_pinn/__init__.py>). Lines: 14. SHA-256: `56548020ae7b10d2c348e32f1373fedf7f244dc7cede46edb220261e3125b69e`.

Module body, imports, exports and class attributes: planned for line-by-line review.

| Lines | Kind | Qualified name | Decorators |
|---|---|---|---|
| whole file | module | no local definitions; review imports, exports and module body | |

**airfoil_pinn/__main__.py**

Primary stage: [2. Точки входа, окружение, пути и консольное приложение](REPOSITORY_STUDY_PLAN.md#stage-2).
Source: [open file](<../airfoil_pinn/__main__.py>). Lines: 28. SHA-256: `5c7ce05ce54197c0a777782fed70d21276914461d13e6f14003a8884c2687151`.

Module body, imports, exports and class attributes: planned for line-by-line review.

| Lines | Kind | Qualified name | Decorators |
|---|---|---|---|
| 10-24 | function | `run` |  |

**airfoil_pinn/app/__init__.py**

Primary stage: [3. Служебные типы, импорты и общие утилиты](REPOSITORY_STUDY_PLAN.md#stage-3).
Source: [open file](<../airfoil_pinn/app/__init__.py>). Lines: 5. SHA-256: `f16ca287a117efe4f5bff265c15ad63783964e4184b920f0f48c65d2c06c45a2`.

Module body, imports, exports and class attributes: planned for line-by-line review.

| Lines | Kind | Qualified name | Decorators |
|---|---|---|---|
| whole file | module | no local definitions; review imports, exports and module body | |

**airfoil_pinn/app/cli.py**

Primary stage: [2. Точки входа, окружение, пути и консольное приложение](REPOSITORY_STUDY_PLAN.md#stage-2).
Source: [open file](<../airfoil_pinn/app/cli.py>). Lines: 507. SHA-256: `370390d8181e542c4a825e6aab27b8ee00910951c86cd74ac44072f892bdbee4`.

Module body, imports, exports and class attributes: planned for line-by-line review.

| Lines | Kind | Qualified name | Decorators |
|---|---|---|---|
| 34-48 | function | `print_banner` |  |
| 51-53 | function | `print_menu_header` |  |
| 56-90 | function | `print_module_info` |  |
| 93-117 | function | `print_main_menu` |  |
| 120-155 | function | `print_settings_menu` |  |
| 158-183 | function | `print_help` |  |
| 186-209 | function | `load_module` |  |
| 212-248 | function | `run_module` |  |
| 251-285 | function | `run_modules_menu` |  |
| 288-349 | function | `run_settings_menu` |  |
| 352-405 | function | `run_interactive_menu` |  |
| 408-433 | function | `_build_parser` |  |
| 436-453 | function | `_run_training_command` |  |
| 456-466 | function | `run_command` |  |
| 469-482 | function | `main` |  |
| 485-503 | function | `guarded_main` |  |

**airfoil_pinn/app/module_registry.py**

Primary stage: [2. Точки входа, окружение, пути и консольное приложение](REPOSITORY_STUDY_PLAN.md#stage-2).
Source: [open file](<../airfoil_pinn/app/module_registry.py>). Lines: 123. SHA-256: `5344033caec3e0677bb7d1b2a02f4901d8ba7635a14599d589a722b8ae352ecc`.

Module body, imports, exports and class attributes: planned for line-by-line review.

| Lines | Kind | Qualified name | Decorators |
|---|---|---|---|
| 17-49 | class | `ModuleDescriptor` | dataclass(frozen=True) |
| 34-49 | method | `ModuleDescriptor.as_legacy_dict` |  |
| 52-109 | function | `default_modules` |  |
| 112-114 | function | `modules_as_legacy_dicts` |  |
| 117-123 | function | `find_by_command` |  |

**airfoil_pinn/app/paths.py**

Primary stage: [2. Точки входа, окружение, пути и консольное приложение](REPOSITORY_STUDY_PLAN.md#stage-2).
Source: [open file](<../airfoil_pinn/app/paths.py>). Lines: 145. SHA-256: `5c8fdddea534411cce5dedcd416f6336063a4e238e13aabc886aecab82874b3b`.

Module body, imports, exports and class attributes: planned for line-by-line review.

| Lines | Kind | Qualified name | Decorators |
|---|---|---|---|
| 21-49 | class | `WorkspacePaths` | dataclass(frozen=True) |
| 52-84 | function | `for_project_root` |  |
| 90-109 | function | `ensure_workspace_dirs` |  |
| 112-122 | function | `resolve_project_path` |  |
| 125-128 | function | `project_relative` |  |

**airfoil_pinn/app/runtime.py**

Primary stage: [2. Точки входа, окружение, пути и консольное приложение](REPOSITORY_STUDY_PLAN.md#stage-2).
Source: [open file](<../airfoil_pinn/app/runtime.py>). Lines: 268. SHA-256: `85e1759649367d816eda6327a0ad4677a8b9fa35120e0ad95da6681a83b1dfd9`.

Module body, imports, exports and class attributes: planned for line-by-line review.

| Lines | Kind | Qualified name | Decorators |
|---|---|---|---|
| 40-57 | function | `configure_text_output` |  |
| 60-70 | function | `ensure_project_root_on_path` |  |
| 73-77 | function | `_venv_python` |  |
| 80-84 | function | `_venv_pip` |  |
| 87-92 | function | `_ask` |  |
| 95-125 | function | `_create_venv` |  |
| 128-147 | function | `_pip_environment` |  |
| 150-162 | function | `_ensure_pip` |  |
| 165-182 | function | `_install_packages` |  |
| 185-205 | function | `_restart_inside_venv` |  |
| 208-252 | function | `ensure_virtual_environment` |  |
| 255-268 | function | `bootstrap_environment` |  |

**airfoil_pinn/config/__init__.py**

Primary stage: [3. Служебные типы, импорты и общие утилиты](REPOSITORY_STUDY_PLAN.md#stage-3).
Source: [open file](<../airfoil_pinn/config/__init__.py>). Lines: 13. SHA-256: `68e8c3dfddf1edef8ca056fc6ef421ac66ac5143b31374f305b6df47ba7844dd`.

Module body, imports, exports and class attributes: planned for line-by-line review.

| Lines | Kind | Qualified name | Decorators |
|---|---|---|---|
| whole file | module | no local definitions; review imports, exports and module body | |

**airfoil_pinn/config/manager.py**

Primary stage: [4. Конфигурация и все меню настройки обучения](REPOSITORY_STUDY_PLAN.md#stage-4).
Source: [open file](<../airfoil_pinn/config/manager.py>). Lines: 769. SHA-256: `07496f56ef80488412168b3bae68c7cd503a52a922072cf89f00c89d6fc74b47`.

Module body, imports, exports and class attributes: planned for line-by-line review.

| Lines | Kind | Qualified name | Decorators |
|---|---|---|---|
| 28-619 | class | `ConfigManager` |  |
| 36-40 | method | `ConfigManager.__init__` |  |
| 42-299 | method | `ConfigManager._default_config` |  |
| 301-318 | method | `ConfigManager.load` |  |
| 320-352 | method | `ConfigManager.load_from_file` |  |
| 354-356 | method | `ConfigManager.validate` |  |
| 358-374 | method | `ConfigManager.save` |  |
| 376-393 | method | `ConfigManager._prepare_for_save` |  |
| 395-572 | method | `ConfigManager.show` |  |
| 574-587 | method | `ConfigManager._deep_update` |  |
| 589-613 | method | `ConfigManager._remove_deprecated_boundary_input` |  |
| 615-619 | method | `ConfigManager.reset_to_defaults` |  |
| 623-639 | function | `yes_no` |  |
| 690-769 | function | `warn_activation` |  |

**airfoil_pinn/config/validation.py**

Primary stage: [4. Конфигурация и все меню настройки обучения](REPOSITORY_STUDY_PLAN.md#stage-4).
Source: [open file](<../airfoil_pinn/config/validation.py>). Lines: 641. SHA-256: `0ece4d5806efbf5f0b73e3bd4356d12155f942daaf29d84641114f88e56a44a7`.

Module body, imports, exports and class attributes: planned for line-by-line review.

| Lines | Kind | Qualified name | Decorators |
|---|---|---|---|
| 24-25 | class | `ConfigurationError` |  |
| 28-37 | function | `_mapping` |  |
| 40-46 | function | `_positive_integer` |  |
| 49-60 | function | `_finite_nonnegative` |  |
| 63-68 | function | `_raise_errors` |  |
| 71-487 | function | `validate_static_config` |  |
| 490-641 | function | `validate_runtime_contract` |  |

**airfoil_pinn/core/__init__.py**

Primary stage: [3. Служебные типы, импорты и общие утилиты](REPOSITORY_STUDY_PLAN.md#stage-3).
Source: [open file](<../airfoil_pinn/core/__init__.py>). Lines: 12. SHA-256: `cfbe14aa91c090b34ca858976c0e48e9bf86a442c773aa151633a2688a3c6556`.

Module body, imports, exports and class attributes: planned for line-by-line review.

| Lines | Kind | Qualified name | Decorators |
|---|---|---|---|
| whole file | module | no local definitions; review imports, exports and module body | |

**airfoil_pinn/core/constants.py**

Primary stage: [3. Служебные типы, импорты и общие утилиты](REPOSITORY_STUDY_PLAN.md#stage-3).
Source: [open file](<../airfoil_pinn/core/constants.py>). Lines: 249. SHA-256: `192cde0eb113ca4dd5d1826b86872997bb8f254736fa8105a5cdcba09705da37`.

Module body, imports, exports and class attributes: planned for line-by-line review.

| Lines | Kind | Qualified name | Decorators |
|---|---|---|---|
| whole file | module | no local definitions; review imports, exports and module body | |

**airfoil_pinn/core/type_hints.py**

Primary stage: [3. Служебные типы, импорты и общие утилиты](REPOSITORY_STUDY_PLAN.md#stage-3).
Source: [open file](<../airfoil_pinn/core/type_hints.py>). Lines: 713. SHA-256: `5cdb64f61aaf9755d4733d0b0b404f55251acfe69526a833fcb34fe20be72929`.

Module body, imports, exports and class attributes: planned for line-by-line review.

| Lines | Kind | Qualified name | Decorators |
|---|---|---|---|
| 28-45 | class | `GeometryConfigDict` |  |
| 48-61 | class | `AirfoilDomainParamsDict` |  |
| 64-87 | class | `AirfoilParamsDict` |  |
| 90-99 | class | `DomainParamsDict` |  |
| 102-115 | class | `ClosedDomainParamsDict` |  |
| 118-131 | class | `GeometryMetadataDict` |  |
| 138-160 | class | `CollocationPointDict` |  |
| 163-228 | class | `SamplerConfigDict` |  |
| 235-260 | class | `BoundaryConditionsDict` |  |
| 267-282 | class | `FluidDict` |  |
| 289-312 | class | `PointDtypeDict` |  |
| 315-366 | class | `InitializerMetadataDict` |  |
| 373-380 | class | `InputParamsDict` |  |
| 383-385 | class | `FeatureToggleDict` |  |
| 388-390 | class | `ReynoldsFeatureDict` |  |
| 393-395 | class | `GeometryFeaturesDict` |  |
| 398-405 | class | `InputFeaturesDict` |  |
| 408-416 | class | `ArchitectureDict` |  |
| 419-424 | class | `CoordinateScalingDict` |  |
| 427-431 | class | `PDEConfigDict` |  |
| 434-446 | class | `BoundaryConditionsConfigDict` |  |
| 449-453 | class | `BoundaryRuleConfigDict` |  |
| 456-465 | class | `LossWeightsConfigDict` |  |
| 468-474 | class | `FVMLossConfigDict` |  |
| 477-502 | class | `TrainingConfigDict` |  |
| 505-510 | class | `EarlyStoppingDict` |  |
| 513-520 | class | `OutputConfigDict` |  |
| 523-527 | class | `GraphicalWindowDict` |  |
| 530-541 | class | `CheckpointConfigDict` |  |
| 544-555 | class | `ProcessConfigDict` |  |
| 558-565 | class | `DataConfigDict` |  |
| 568-570 | class | `ReferenceConfigDict` |  |
| 573-589 | class | `DiagnosticsConfigDict` |  |
| 592-613 | class | `FullConfigDict` |  |
| 620-624 | class | `WorkerProgressDict` |  |
| 631-639 | class | `PlotConfigDict` |  |
| 646-651 | class | `OptimizerParamsDict` |  |
| 654-659 | class | `L2ErrorDict` |  |

**airfoil_pinn/data/__init__.py**

Primary stage: [3. Служебные типы, импорты и общие утилиты](REPOSITORY_STUDY_PLAN.md#stage-3).
Source: [open file](<../airfoil_pinn/data/__init__.py>). Lines: 71. SHA-256: `33d52f7ab43ca92f3e43f150be816579f97f4ad8f3b625fd9e0303d0c0d2c457`.

Module body, imports, exports and class attributes: planned for line-by-line review.

| Lines | Kind | Qualified name | Decorators |
|---|---|---|---|
| 24-71 | function | `__getattr__` |  |

**airfoil_pinn/data/batch.py**

Primary stage: [11. Режимы течения, Dataset, батчи и состояние данных](REPOSITORY_STUDY_PLAN.md#stage-11).
Source: [open file](<../airfoil_pinn/data/batch.py>). Lines: 89. SHA-256: `1534ea151f40ac76990499406228a7c24272bcd53010be15df0e402f56ce542a`.

Module body, imports, exports and class attributes: planned for line-by-line review.

| Lines | Kind | Qualified name | Decorators |
|---|---|---|---|
| 12-70 | class | `PINNBatch` | dataclass(frozen=True) |
| 35-49 | method | `PINNBatch.legacy_tuple` |  |
| 51-53 | method | `PINNBatch.__iter__` |  |
| 55-57 | method | `PINNBatch.__len__` |  |
| 59-61 | method | `PINNBatch.__getitem__` |  |
| 63-70 | method | `PINNBatch.pin_memory` |  |
| 73-89 | function | `collate_pinn_batches` |  |

**airfoil_pinn/data/flow.py**

Primary stage: [11. Режимы течения, Dataset, батчи и состояние данных](REPOSITORY_STUDY_PLAN.md#stage-11).
Source: [open file](<../airfoil_pinn/data/flow.py>). Lines: 224. SHA-256: `aa7ec2c97fb6e96c763b44bafdcb331989d9283d069fe533347891ed41751202`.

Module body, imports, exports and class attributes: planned for line-by-line review.

| Lines | Kind | Qualified name | Decorators |
|---|---|---|---|
| 23-174 | class | `FlowCase` | dataclass(frozen=True) |
| 44-77 | method | `FlowCase.__post_init__` |  |
| 80-82 | method | `FlowCase.angle_radians` | property |
| 85-87 | method | `FlowCase.cos_alpha` | property |
| 90-92 | method | `FlowCase.sin_alpha` | property |
| 94-106 | method | `FlowCase.to_metadata` |  |
| 108-123 | method | `FlowCase.fingerprint` |  |
| 126-144 | method | `FlowCase.from_metadata` | classmethod |
| 147-174 | method | `FlowCase.from_freestream_components` | classmethod |
| 178-191 | class | `FlowCaseBatch` | dataclass(frozen=True) |
| 189-191 | method | `FlowCaseBatch.batch_size` | property |
| 194-224 | function | `flow_cases_from_metadata` |  |

**airfoil_pinn/data/handler.py**

Primary stage: [11. Режимы течения, Dataset, батчи и состояние данных](REPOSITORY_STUDY_PLAN.md#stage-11).
Source: [open file](<../airfoil_pinn/data/handler.py>). Lines: 2320. SHA-256: `70a56e4bff6441453dabb569adf8d546ba2f619a6aef8b4a57365727cff43a79`.

Module body, imports, exports and class attributes: planned for line-by-line review.

| Lines | Kind | Qualified name | Decorators |
|---|---|---|---|
| 68-77 | function | `_seed_dataloader_worker` |  |
| 81-89 | class | `MeshPointAdapter` | dataclass |
| 93-164 | class | `MeshData` | dataclass |
| 128-129 | method | `MeshData.n_nodes` | property |
| 132-133 | method | `MeshData.n_triangles` | property |
| 136-137 | method | `MeshData.n_int_faces` | property |
| 140-141 | method | `MeshData.n_bnd_faces` | property |
| 143-164 | method | `MeshData.to_torch` |  |
| 167-504 | class | `PINNDataset` |  |
| 194-362 | method | `PINNDataset.__init__` |  |
| 219-223 | nested function | `PINNDataset.__init__._field_copy` |  |
| 365-367 | method | `PINNDataset.p_target` | property |
| 370-371 | method | `PINNDataset.p_target` | p_target.setter |
| 374-376 | method | `PINNDataset.u_target` | property |
| 379-380 | method | `PINNDataset.u_target` | u_target.setter |
| 383-385 | method | `PINNDataset.v_target` | property |
| 388-389 | method | `PINNDataset.v_target` | v_target.setter |
| 391-424 | method | `PINNDataset.set_inlet_values` |  |
| 426-433 | method | `PINNDataset.__len__` |  |
| 435-465 | method | `PINNDataset.__getitem__` |  |
| 467-504 | method | `PINNDataset.extend` |  |
| 507-2320 | class | `DataHandler` |  |
| 519-606 | method | `DataHandler.__init__` |  |
| 608-627 | method | `DataHandler._make_dataloader` |  |
| 629-733 | method | `DataHandler.load_metadata` |  |
| 735-744 | method | `DataHandler._log_flow_cases` |  |
| 746-818 | method | `DataHandler.get_flow_case_batch` |  |
| 820-909 | method | `DataHandler._load_sampler_point_metadata` |  |
| 911-1026 | method | `DataHandler._reconcile_geometry_normalization` |  |
| 1028-1068 | method | `DataHandler.load_mesh` |  |
| 1070-1090 | method | `DataHandler.set_mesh_from_mesh2d` |  |
| 1092-1098 | method | `DataHandler._dataset_boundary_type_strings` |  |
| 1100-1125 | method | `DataHandler.mesh_points_from_dataset` |  |
| 1127-1144 | method | `DataHandler.build_mesh_from_current_points` |  |
| 1146-1240 | method | `DataHandler._prepare_fvm_tensors` |  |
| 1242-1249 | method | `DataHandler.get_dataloader` |  |
| 1251-1262 | method | `DataHandler.get_eval_dataloader` |  |
| 1264-1293 | method | `DataHandler.get_boundary_eval_dataloader` |  |
| 1296-1303 | method | `DataHandler.n_points` | property |
| 1306-1313 | method | `DataHandler.boundary_type_names` | property |
| 1315-1324 | method | `DataHandler.add_points` |  |
| 1326-1453 | method | `DataHandler.add_interior_points` |  |
| 1455-1506 | method | `DataHandler.get_state` |  |
| 1508-1573 | method | `DataHandler.load_state` |  |
| 1578-1609 | method | `DataHandler.get_points_by_type` |  |
| 1611-1640 | method | `DataHandler.get_points_by_type_and_coordinate` |  |
| 1642-1697 | method | `DataHandler.interpolate_profile` |  |
| 1699-1767 | method | `DataHandler.interpolate_fvm_profile` |  |
| 1769-1823 | method | `DataHandler.integrate_fvm_vertical_flux` |  |
| 1825-1859 | method | `DataHandler._get_profile_triangulation` |  |
| 1861-1947 | method | `DataHandler._get_profile_stencil` |  |
| 1949-2103 | method | `DataHandler._get_vertical_integration_stencil` |  |
| 2106-2151 | method | `DataHandler._barycentric_weights` | staticmethod |
| 2153-2248 | method | `DataHandler._dataset_values_on_mesh_nodes` |  |
| 2250-2320 | method | `DataHandler._interpolate_profile_without_mesh` |  |

**airfoil_pinn/data/initializer.py**

Primary stage: [10. Физическая нормировка и подготовка начальных артефактов](REPOSITORY_STUDY_PLAN.md#stage-10).
Source: [open file](<../airfoil_pinn/data/initializer.py>). Lines: 1406. SHA-256: `08ec174e0acf4eefff1ad285cfc0642a0ec8577f7404559dd6a3b57132071b91`.

Module body, imports, exports and class attributes: planned for line-by-line review.

| Lines | Kind | Qualified name | Decorators |
|---|---|---|---|
| 130-175 | function | `_compute_sdf_chunk` |  |
| 178-1285 | class | `Initializer` |  |
| 202-248 | method | `Initializer.__init__` |  |
| 250-299 | method | `Initializer.load_sampler_points` |  |
| 301-324 | method | `Initializer.extract_wall_points` |  |
| 327-345 | method | `Initializer._load_coords` | staticmethod |
| 347-365 | method | `Initializer._sampler_metadata_matches_points` |  |
| 367-395 | method | `Initializer._sampler_geom_name` |  |
| 397-415 | method | `Initializer._load_geometry_coords` |  |
| 417-443 | method | `Initializer._domain_coords_from_sampler_bounds` |  |
| 445-456 | method | `Initializer.load_domain_coords` |  |
| 458-467 | method | `Initializer.load_foil_coords` |  |
| 469-573 | method | `Initializer._load_chord_from_metadata` |  |
| 575-590 | method | `Initializer.select_fluid` |  |
| 592-621 | method | `Initializer.select_conditions` |  |
| 623-691 | method | `Initializer._compute_normalization_params` |  |
| 693-759 | method | `Initializer.print_sdf_diagnostics_after_conditions` |  |
| 761-986 | method | `Initializer.initialize_fields` |  |
| 988-1056 | method | `Initializer.save` |  |
| 1059-1069 | method | `Initializer._deduplicate_field_points` | staticmethod |
| 1072-1083 | method | `Initializer._clean_polyline` | staticmethod |
| 1086-1092 | method | `Initializer._clip_polygon` | classmethod |
| 1095-1101 | method | `Initializer._closed_mask_polygon` | classmethod |
| 1104-1105 | method | `Initializer._polyline` | classmethod |
| 1108-1114 | method | `Initializer._points_inside_polygon` | staticmethod |
| 1117-1130 | method | `Initializer._apply_grid_masks` | classmethod |
| 1133-1218 | method | `Initializer._plot_smooth_field` | staticmethod |
| 1220-1268 | method | `Initializer.plot_field` |  |
| 1270-1285 | method | `Initializer.plot_all_fields` |  |
| 1288-1402 | function | `initialize_menu` |  |

**airfoil_pinn/data/initializer_cli.py**

Primary stage: [10. Физическая нормировка и подготовка начальных артефактов](REPOSITORY_STUDY_PLAN.md#stage-10).
Source: [open file](<../airfoil_pinn/data/initializer_cli.py>). Lines: 7. SHA-256: `82d9c58c666c68ded0bc587847e40c69fba96933600dde4ecd199d8d0cdb4358`.

Module body, imports, exports and class attributes: planned for line-by-line review.

| Lines | Kind | Qualified name | Decorators |
|---|---|---|---|
| whole file | module | no local definitions; review imports, exports and module body | |

**airfoil_pinn/data/normalization.py**

Primary stage: [10. Физическая нормировка и подготовка начальных артефактов](REPOSITORY_STUDY_PLAN.md#stage-10).
Source: [open file](<../airfoil_pinn/data/normalization.py>). Lines: 358. SHA-256: `f66d042bd6a75e0ea35922b7c9780560d187ac6d8261986d5c518f96fdd88040`.

Module body, imports, exports and class attributes: planned for line-by-line review.

| Lines | Kind | Qualified name | Decorators |
|---|---|---|---|
| 35-209 | class | `ReferenceScales` | dataclass(frozen=True) |
| 56-79 | method | `ReferenceScales.__post_init__` |  |
| 82-84 | method | `ReferenceScales.dynamic_pressure` | property |
| 87-94 | method | `ReferenceScales.reynolds` | property |
| 96-98 | method | `ReferenceScales.length_to_dimensionless` |  |
| 100-102 | method | `ReferenceScales.length_to_dimensional` |  |
| 104-109 | method | `ReferenceScales.velocity_to_dimensionless` |  |
| 111-116 | method | `ReferenceScales.velocity_to_dimensional` |  |
| 118-123 | method | `ReferenceScales.pressure_to_coefficient` |  |
| 125-131 | method | `ReferenceScales.coefficient_to_pressure` |  |
| 133-135 | method | `ReferenceScales.with_chord` |  |
| 137-155 | method | `ReferenceScales.to_metadata` |  |
| 157-174 | method | `ReferenceScales.fingerprint` |  |
| 177-209 | method | `ReferenceScales.from_metadata` | classmethod |
| 212-233 | function | `_required_number` |  |
| 236-240 | function | `_restore_scalar` |  |
| 243-274 | function | `build_geometry_reference` |  |
| 277-314 | function | `extract_geometry_reference` |  |
| 317-334 | function | `load_companion_metadata` |  |
| 337-349 | function | `compute_reynolds_number` |  |
| 352-358 | function | `_positive_float` |  |

**airfoil_pinn/data/sdf_diagnostics.py**

Primary stage: [8. Пристенное размещение точек и предварительная SDF-диагностика](REPOSITORY_STUDY_PLAN.md#stage-8).
Source: [open file](<../airfoil_pinn/data/sdf_diagnostics.py>). Lines: 290. SHA-256: `f4ad7bc3f0d7ab29d979e5e85aedd51d8f20acd917e253313ca99a1afa2b824f`.

Module body, imports, exports and class attributes: planned for line-by-line review.

| Lines | Kind | Qualified name | Decorators |
|---|---|---|---|
| 62-84 | function | `estimate_boundary_layer_delta` |  |
| 87-92 | function | `resolve_boundary_layer_model` |  |
| 95-120 | function | `compute_nearest_wall_distances` |  |
| 123-159 | function | `compute_sdf_summary` |  |
| 162-200 | function | `format_sdf_summary` |  |
| 203-208 | function | `_as_xy` |  |
| 211-217 | function | `_positive_float` |  |
| 220-231 | function | `_stats` |  |
| 234-248 | function | `_threshold_counts` |  |
| 251-270 | function | `_format_stats` |  |
| 273-284 | function | `_format_counts` |  |
| 287-290 | function | `_fmt` |  |

**airfoil_pinn/geometry/__init__.py**

Primary stage: [3. Служебные типы, импорты и общие утилиты](REPOSITORY_STUDY_PLAN.md#stage-3).
Source: [open file](<../airfoil_pinn/geometry/__init__.py>). Lines: 32. SHA-256: `661887ac03b4f751871c6bc823de29c772960009cce802e4833e7effb97711bb`.

Module body, imports, exports and class attributes: planned for line-by-line review.

| Lines | Kind | Qualified name | Decorators |
|---|---|---|---|
| whole file | module | no local definitions; review imports, exports and module body | |

**airfoil_pinn/geometry/cli.py**

Primary stage: [6. Геометрия вычислительного пакета](REPOSITORY_STUDY_PLAN.md#stage-6).
Source: [open file](<../airfoil_pinn/geometry/cli.py>). Lines: 13. SHA-256: `887b99fa6e8a9392eed0603177a6a8c8f3129a7c9d486990eb696ac8aae28cc5`.

Module body, imports, exports and class attributes: planned for line-by-line review.

| Lines | Kind | Qualified name | Decorators |
|---|---|---|---|
| whole file | module | no local definitions; review imports, exports and module body | |

**airfoil_pinn/geometry/core.py**

Primary stage: [6. Геометрия вычислительного пакета](REPOSITORY_STUDY_PLAN.md#stage-6).
Source: [open file](<../airfoil_pinn/geometry/core.py>). Lines: 828. SHA-256: `42f47ff6a7c118cd70cd791716c5faec281fcb3f7330627ec295a8c46bccc408`.

Module body, imports, exports and class attributes: planned for line-by-line review.

| Lines | Kind | Qualified name | Decorators |
|---|---|---|---|
| 57-85 | class | `GeometryConfig` | dataclass |
| 76-85 | method | `GeometryConfig.to_dict` |  |
| 88-527 | class | `AirfoilGeometry` |  |
| 102-114 | method | `AirfoilGeometry.__init__` |  |
| 116-160 | method | `AirfoilGeometry.load_airfoil` |  |
| 162-186 | method | `AirfoilGeometry._load_text_file` |  |
| 188-202 | method | `AirfoilGeometry._check_closed` |  |
| 204-234 | method | `AirfoilGeometry.create_domain` |  |
| 236-299 | method | `AirfoilGeometry._create_unclosed_domain` |  |
| 301-336 | method | `AirfoilGeometry._create_closed_domain` |  |
| 338-361 | method | `AirfoilGeometry._get_y_at_x` |  |
| 363-415 | method | `AirfoilGeometry.plot` |  |
| 417-465 | method | `AirfoilGeometry.save` |  |
| 467-495 | method | `AirfoilGeometry._save_coords` |  |
| 497-527 | method | `AirfoilGeometry.print_stats` |  |
| 534-700 | class | `Geometry2D` |  |
| 549-572 | method | `Geometry2D.__init__` |  |
| 574-595 | method | `Geometry2D.load_from_file` |  |
| 597-630 | method | `Geometry2D.load_from_yaml` |  |
| 632-681 | method | `Geometry2D._load_airfoil_from_yaml` |  |
| 683-686 | method | `Geometry2D.create_domain` |  |
| 688-690 | method | `Geometry2D.plot` |  |
| 692-696 | method | `Geometry2D.save` |  |
| 698-700 | method | `Geometry2D.print_stats` |  |
| 706-758 | function | `_select_geometry_file` |  |
| 761-824 | function | `g2d` |  |

**airfoil_pinn/geometry/sdf.py**

Primary stage: [12. Дифференцируемое поле расстояний](REPOSITORY_STUDY_PLAN.md#stage-12).
Source: [open file](<../airfoil_pinn/geometry/sdf.py>). Lines: 625. SHA-256: `561cd70abd3920d2b15be79e8fc2cdc74b52e4f3b69b42d177cc5be2f48ecb0d`.

Module body, imports, exports and class attributes: planned for line-by-line review.

| Lines | Kind | Qualified name | Decorators |
|---|---|---|---|
| 29-48 | class | `DistanceFieldDiagnostics` | dataclass(frozen=True) |
| 39-48 | method | `DistanceFieldDiagnostics.to_dict` |  |
| 51-590 | class | `PolylineDistanceField` |  |
| 67-117 | method | `PolylineDistanceField.__init__` |  |
| 120-132 | method | `PolylineDistanceField._remove_adjacent_duplicates` | staticmethod |
| 135-141 | method | `PolylineDistanceField._fingerprint` | staticmethod |
| 144-146 | method | `PolylineDistanceField.segment_count` | property |
| 148-154 | method | `PolylineDistanceField.extra_repr` |  |
| 156-175 | method | `PolylineDistanceField.forward` |  |
| 177-187 | method | `PolylineDistanceField.wall_distance` |  |
| 189-195 | method | `PolylineDistanceField.nearest_two` |  |
| 197-309 | method | `PolylineDistanceField.nearest_two_geometry` |  |
| 311-359 | method | `PolylineDistanceField.surface_normals` |  |
| 361-455 | method | `PolylineDistanceField.tie_mask` |  |
| 457-507 | method | `PolylineDistanceField.diagnostics` |  |
| 510-549 | method | `PolylineDistanceField.evaluate_numpy` | torch.no_grad() |
| 551-564 | method | `PolylineDistanceField._validate_points` |  |
| 566-590 | method | `PolylineDistanceField._squared_distances` |  |
| 593-625 | function | `differentiable_sdf_for_interior` |  |

**airfoil_pinn/physics/__init__.py**

Primary stage: [3. Служебные типы, импорты и общие утилиты](REPOSITORY_STUDY_PLAN.md#stage-3).
Source: [open file](<../airfoil_pinn/physics/__init__.py>). Lines: 12. SHA-256: `529bb78adcf40e95893e2b11cd7102037f2d0c0b45acb6ee598adc3ff0b7beaf`.

Module body, imports, exports and class attributes: planned for line-by-line review.

| Lines | Kind | Qualified name | Decorators |
|---|---|---|---|
| whole file | module | no local definitions; review imports, exports and module body | |

**airfoil_pinn/physics/conditions.py**

Primary stage: [5. Библиотеки жидкостей и условий течения](REPOSITORY_STUDY_PLAN.md#stage-5).
Source: [open file](<../airfoil_pinn/physics/conditions.py>). Lines: 719. SHA-256: `9928e36920ace66dbb0d077f001fcc95de9470b36bacdeb57fe18fb0cffe72ab`.

Module body, imports, exports and class attributes: planned for line-by-line review.

| Lines | Kind | Qualified name | Decorators |
|---|---|---|---|
| 55-177 | class | `BoundaryConditions` | dataclass |
| 82-98 | method | `BoundaryConditions.__post_init__` |  |
| 101-111 | method | `BoundaryConditions.u` | property |
| 114-124 | method | `BoundaryConditions.v` | property |
| 126-139 | method | `BoundaryConditions.to_dict` |  |
| 142-152 | method | `BoundaryConditions.from_dict` | classmethod |
| 154-177 | method | `BoundaryConditions.display` |  |
| 180-327 | class | `ConditionsManager` |  |
| 192-203 | method | `ConditionsManager.__init__` |  |
| 205-230 | method | `ConditionsManager._load_all` |  |
| 232-243 | method | `ConditionsManager.save` |  |
| 245-258 | method | `ConditionsManager.delete` |  |
| 260-267 | method | `ConditionsManager.list_names` |  |
| 269-279 | method | `ConditionsManager.get` |  |
| 281-327 | method | `ConditionsManager.create_defaults` |  |
| 330-359 | function | `select_fluid` |  |
| 362-420 | function | `conditions_menu` |  |
| 423-499 | function | `create_conditions` |  |
| 502-520 | function | `_get_symmetry_selection` |  |
| 523-541 | function | `_get_velocity_input_components` |  |
| 544-562 | function | `_get_velocity_input_magnitude_angle` |  |
| 565-621 | function | `_update_velocity_data` |  |
| 624-685 | function | `edit_conditions` |  |
| 688-715 | function | `delete_conditions` |  |

**airfoil_pinn/physics/conditions_cli.py**

Primary stage: [5. Библиотеки жидкостей и условий течения](REPOSITORY_STUDY_PLAN.md#stage-5).
Source: [open file](<../airfoil_pinn/physics/conditions_cli.py>). Lines: 7. SHA-256: `fc32208304ea1dc405aa5da1eec0fa0ddf0bbf5d32d6aa139cf9d4a69d33a511`.

Module body, imports, exports and class attributes: planned for line-by-line review.

| Lines | Kind | Qualified name | Decorators |
|---|---|---|---|
| whole file | module | no local definitions; review imports, exports and module body | |

**airfoil_pinn/physics/fluids.py**

Primary stage: [5. Библиотеки жидкостей и условий течения](REPOSITORY_STUDY_PLAN.md#stage-5).
Source: [open file](<../airfoil_pinn/physics/fluids.py>). Lines: 397. SHA-256: `1ea76fc46f0cafceb0b5100c06b077aad8e33f3ffc85dc7666f9a794cf32f309`.

Module body, imports, exports and class attributes: planned for line-by-line review.

| Lines | Kind | Qualified name | Decorators |
|---|---|---|---|
| 43-111 | class | `Fluid` | dataclass |
| 60-72 | method | `Fluid.__post_init__` |  |
| 74-87 | method | `Fluid.to_dict` |  |
| 90-100 | method | `Fluid.from_dict` | classmethod |
| 102-111 | method | `Fluid.display` |  |
| 114-251 | class | `FluidLibrary` |  |
| 123-133 | method | `FluidLibrary.__init__` |  |
| 135-154 | method | `FluidLibrary._load_all` |  |
| 156-171 | method | `FluidLibrary.save` |  |
| 173-189 | method | `FluidLibrary.delete` |  |
| 191-198 | method | `FluidLibrary.list_fluids` |  |
| 200-210 | method | `FluidLibrary.get` |  |
| 212-251 | method | `FluidLibrary.create_default_library` |  |
| 254-393 | function | `fluids_menu` |  |

**airfoil_pinn/physics/fluids_cli.py**

Primary stage: [5. Библиотеки жидкостей и условий течения](REPOSITORY_STUDY_PLAN.md#stage-5).
Source: [open file](<../airfoil_pinn/physics/fluids_cli.py>). Lines: 7. SHA-256: `f45565ce9a8108db7f9528376005d5259e15c355e8a931a7ca612fddf38726e5`.

Module body, imports, exports and class attributes: planned for line-by-line review.

| Lines | Kind | Qualified name | Decorators |
|---|---|---|---|
| whole file | module | no local definitions; review imports, exports and module body | |

**airfoil_pinn/physics/fvm.py**

Primary stage: [17. FVM-оператор и расписание глобальной потери](REPOSITORY_STUDY_PLAN.md#stage-17).
Source: [open file](<../airfoil_pinn/physics/fvm.py>). Lines: 870. SHA-256: `e9f4a6e0b086f752e117ac023a1e27dedac21f054ec7341b9639183c95a23998`.

Module body, imports, exports and class attributes: planned for line-by-line review.

| Lines | Kind | Qualified name | Decorators |
|---|---|---|---|
| 69-870 | class | `FVMResiduals` |  |
| 92-98 | method | `FVMResiduals.__init__` |  |
| 102-170 | method | `FVMResiduals._build_from_mesh` |  |
| 173-267 | method | `FVMResiduals._validate_geometry_arrays` | staticmethod |
| 270-294 | method | `FVMResiduals._fingerprint_geometry` | staticmethod |
| 296-315 | method | `FVMResiduals._validate_velocity` |  |
| 317-332 | method | `FVMResiduals._validate_tensor_geometry` |  |
| 334-394 | method | `FVMResiduals._compute_residual` |  |
| 397-421 | method | `FVMResiduals._extract_mesh2d` | staticmethod |
| 424-434 | method | `FVMResiduals._extract_mesh_data` | staticmethod |
| 438-466 | method | `FVMResiduals.compute_residual` |  |
| 468-485 | method | `FVMResiduals.mass_loss` |  |
| 487-542 | method | `FVMResiduals._reduce_residual` |  |
| 544-560 | method | `FVMResiduals.loss_and_divergence_rms` |  |
| 562-573 | method | `FVMResiduals.divergence_rms` |  |
| 577-598 | method | `FVMResiduals.check_geometry_conservativity` |  |
| 600-634 | method | `FVMResiduals.check_conservativity` |  |
| 636-665 | method | `FVMResiduals.summary` |  |
| 668-670 | method | `FVMResiduals.mesh_fingerprint` | property |
| 674-728 | method | `FVMResiduals.set_bc_velocity` |  |
| 730-739 | method | `FVMResiduals.set_inlet_velocity` |  |
| 741-768 | method | `FVMResiduals.reset_prescribed` |  |
| 772-798 | method | `FVMResiduals.to` |  |
| 800-839 | method | `FVMResiduals.nondimensionalize_geometry` |  |
| 841-859 | method | `FVMResiduals.rebuild_from_mesh` |  |
| 864-866 | method | `FVMResiduals.from_mesh_data` | classmethod |
| 868-870 | method | `FVMResiduals.__repr__` |  |

**airfoil_pinn/physics/pde.py**

Primary stage: [14. PDE и автоматическое дифференцирование](REPOSITORY_STUDY_PLAN.md#stage-14).
Source: [open file](<../airfoil_pinn/physics/pde.py>). Lines: 221. SHA-256: `1035e396386b7b735900aa1a0a4bda0af7de65a7254d4dfa32d8c469e82c91ec`.

Module body, imports, exports and class attributes: planned for line-by-line review.

| Lines | Kind | Qualified name | Decorators |
|---|---|---|---|
| 24-221 | class | `PDEResiduals` |  |
| 28-64 | method | `PDEResiduals._inverse_reynolds_like` | staticmethod |
| 67-116 | method | `PDEResiduals.compute_residuals_2d` | staticmethod |
| 119-142 | method | `PDEResiduals.compute_residuals_euler_2d` | staticmethod |
| 145-193 | method | `PDEResiduals.compute_residuals_axisymmetric` | staticmethod |
| 196-221 | method | `PDEResiduals.compute_residuals_euler_axisymmetric` | staticmethod |

**airfoil_pinn/sampling/__init__.py**

Primary stage: [3. Служебные типы, импорты и общие утилиты](REPOSITORY_STUDY_PLAN.md#stage-3).
Source: [open file](<../airfoil_pinn/sampling/__init__.py>). Lines: 37. SHA-256: `f00ffe3f911ff695f3f29e7b3ece69ecf197ff6aa505fea8a5f26630288cca38`.

Module body, imports, exports and class attributes: planned for line-by-line review.

| Lines | Kind | Qualified name | Decorators |
|---|---|---|---|
| whole file | module | no local definitions; review imports, exports and module body | |

**airfoil_pinn/sampling/cli.py**

Primary stage: [7. Коллокационные точки и параллельный sampling](REPOSITORY_STUDY_PLAN.md#stage-7).
Source: [open file](<../airfoil_pinn/sampling/cli.py>). Lines: 12. SHA-256: `6861d32b5721d7650e001dded0a98f35de1939ef8f71d47486bb474bf4bd8584`.

Module body, imports, exports and class attributes: planned for line-by-line review.

| Lines | Kind | Qualified name | Decorators |
|---|---|---|---|
| whole file | module | no local definitions; review imports, exports and module body | |

**airfoil_pinn/sampling/core.py**

Primary stage: [7. Коллокационные точки и параллельный sampling](REPOSITORY_STUDY_PLAN.md#stage-7).
Source: [open file](<../airfoil_pinn/sampling/core.py>). Lines: 2510. SHA-256: `d973a016f57e1ce8a916fd33a6ae3d82b1b1659ecb709391fb2b9141deaf37ed`.

Module body, imports, exports and class attributes: planned for line-by-line review.

| Lines | Kind | Qualified name | Decorators |
|---|---|---|---|
| 104-115 | class | `BoundaryType` |  |
| 119-168 | class | `CollocationPoint` | dataclass |
| 142-159 | method | `CollocationPoint.to_dict` |  |
| 161-168 | method | `CollocationPoint.to_tuple` |  |
| 185-326 | class | `SamplerConfig` | dataclass |
| 257-277 | method | `SamplerConfig.__post_init__` |  |
| 279-326 | method | `SamplerConfig.to_dict` |  |
| 333-367 | function | `_regular_worker` |  |
| 370-413 | function | `_irregular_worker` |  |
| 416-477 | function | `_adaptive_worker` |  |
| 480-2099 | class | `CollocationPoints` |  |
| 496-517 | method | `CollocationPoints.__init__` |  |
| 519-521 | method | `CollocationPoints._get_next_id` |  |
| 523-567 | method | `CollocationPoints.find_geometry_files` |  |
| 569-619 | method | `CollocationPoints.load_geometries` |  |
| 621-637 | method | `CollocationPoints._load_coords` |  |
| 640-660 | method | `CollocationPoints._point_to_polyline_distance` | staticmethod |
| 663-675 | method | `CollocationPoints._polyline_lies_on_polyline` | classmethod |
| 677-700 | method | `CollocationPoints._build_normal_layer_config` |  |
| 702-720 | method | `CollocationPoints._normal_layer_surface_reference` |  |
| 723-767 | method | `CollocationPoints._polyline_normals` | staticmethod |
| 772-881 | method | `CollocationPoints.sample_boundary_external` |  |
| 822-823 | nested function | `CollocationPoints.sample_boundary_external._coord_key` |  |
| 825-851 | nested function | `CollocationPoints.sample_boundary_external._append_point` |  |
| 883-921 | method | `CollocationPoints._classify_external_boundary` |  |
| 926-1008 | method | `CollocationPoints.sample_boundary_foil` |  |
| 1013-1055 | method | `CollocationPoints.sample_bottom_edge` |  |
| 1057-1081 | method | `CollocationPoints.filter_interior_against_closed_foil` |  |
| 1086-1120 | method | `CollocationPoints.sample_interior_regular_sequential` |  |
| 1122-1204 | method | `CollocationPoints.sample_interior_regular_parallel` |  |
| 1157-1181 | nested function | `CollocationPoints.sample_interior_regular_parallel.monitor` |  |
| 1158-1158 | lambda | `CollocationPoints.sample_interior_regular_parallel.monitor.<lambda@1158:40>` |  |
| 1209-1227 | method | `CollocationPoints.sample_interior_irregular` |  |
| 1229-1286 | method | `CollocationPoints._sample_interior_irregular_sequential` |  |
| 1288-1366 | method | `CollocationPoints.sample_interior_irregular_outside_wall_distance` |  |
| 1368-1441 | method | `CollocationPoints._sample_interior_irregular_parallel` |  |
| 1393-1417 | nested function | `CollocationPoints._sample_interior_irregular_parallel.monitor` |  |
| 1394-1394 | lambda | `CollocationPoints._sample_interior_irregular_parallel.monitor.<lambda@1394:40>` |  |
| 1446-1495 | method | `CollocationPoints.sample_interior_adaptive` |  |
| 1497-1573 | method | `CollocationPoints._sample_interior_adaptive_sequential` |  |
| 1575-1651 | method | `CollocationPoints._sample_interior_adaptive_parallel` |  |
| 1603-1627 | nested function | `CollocationPoints._sample_interior_adaptive_parallel.monitor` |  |
| 1604-1604 | lambda | `CollocationPoints._sample_interior_adaptive_parallel.monitor.<lambda@1604:40>` |  |
| 1656-1738 | method | `CollocationPoints.sample_interior_normal_layer` |  |
| 1743-1770 | method | `CollocationPoints._log_sdf_statistics` |  |
| 1772-1963 | method | `CollocationPoints.generate_all_points` |  |
| 1968-2044 | method | `CollocationPoints.plot_points` |  |
| 2046-2087 | method | `CollocationPoints.save_points` |  |
| 2089-2099 | method | `CollocationPoints.get_points_by_type` |  |
| 2102-2162 | function | `_select_geometry_data` |  |
| 2165-2196 | function | `_show_geometry_preview` |  |
| 2199-2506 | function | `S2DR` |  |

**airfoil_pinn/sampling/mesh/__init__.py**

Primary stage: [3. Служебные типы, импорты и общие утилиты](REPOSITORY_STUDY_PLAN.md#stage-3).
Source: [open file](<../airfoil_pinn/sampling/mesh/__init__.py>). Lines: 17. SHA-256: `9464ae522b4297db5f8530f567e48b79b498173cda41a8d74bb607e7380d6292`.

Module body, imports, exports and class attributes: planned for line-by-line review.

| Lines | Kind | Qualified name | Decorators |
|---|---|---|---|
| whole file | module | no local definitions; review imports, exports and module body | |

**airfoil_pinn/sampling/mesh/core.py**

Primary stage: [9. Триангуляция, дуальная сетка и контрольные объёмы](REPOSITORY_STUDY_PLAN.md#stage-9).
Source: [open file](<../airfoil_pinn/sampling/mesh/core.py>). Lines: 2255. SHA-256: `cf419c5911db53bf9d987a8233e1790b732bc552d7fa9759570e9ba46bce1b74`.

Module body, imports, exports and class attributes: planned for line-by-line review.

| Lines | Kind | Qualified name | Decorators |
|---|---|---|---|
| 62-81 | class | `MeshNode` | dataclass |
| 85-101 | class | `InternalFace` | dataclass |
| 100-101 | method | `InternalFace.__post_init__` |  |
| 105-123 | class | `BoundaryFace` | dataclass |
| 122-123 | method | `BoundaryFace.__post_init__` |  |
| 127-135 | class | `_PointAdapter` | dataclass |
| 142-144 | function | `_cross2d` |  |
| 147-158 | function | `_segs_cross_strict` |  |
| 161-173 | function | `_in_circumcircle` |  |
| 176-180 | function | `_ensure_ccw` |  |
| 183-191 | function | `_point_in_tri` |  |
| 198-244 | function | `_triangulate_polygon_delaunay` |  |
| 247-503 | function | `_insert_constrained_edge` |  |
| 462-463 | nested function | `_insert_constrained_edge._param` |  |
| 510-583 | function | `_flip_to_delaunay` |  |
| 590-657 | function | `_flood_fill_remove` |  |
| 664-2255 | class | `Mesh2D` |  |
| 680-700 | method | `Mesh2D.__init__` |  |
| 705-720 | method | `Mesh2D.from_sampler` | classmethod |
| 723-739 | method | `Mesh2D.from_collocation_result` | classmethod |
| 742-762 | method | `Mesh2D.from_npy` | classmethod |
| 766-854 | method | `Mesh2D.build` |  |
| 810-829 | nested function | `Mesh2D.build._step_result` |  |
| 859-860 | method | `Mesh2D.nodes` | property |
| 863-864 | method | `Mesh2D.coords` | property |
| 867-868 | method | `Mesh2D.triangles` | property |
| 871-872 | method | `Mesh2D.cv_volumes` | property |
| 875-876 | method | `Mesh2D.internal_faces` | property |
| 879-880 | method | `Mesh2D.boundary_faces` | property |
| 884-904 | method | `Mesh2D.get_node_faces` |  |
| 906-912 | method | `Mesh2D.get_boundary_nodes` |  |
| 914-937 | method | `Mesh2D.summary` |  |
| 939-990 | method | `Mesh2D.save` |  |
| 992-1071 | method | `Mesh2D.plot` |  |
| 1073-1148 | method | `Mesh2D.plot_cv_volume_map` |  |
| 1150-1216 | method | `Mesh2D.plot_cv_volume_hist` |  |
| 1219-1234 | method | `Mesh2D._boundary_rank` | staticmethod |
| 1237-1240 | method | `Mesh2D._prefer_boundary_type` | classmethod |
| 1243-1256 | method | `Mesh2D._merge_node_metadata` | classmethod |
| 1259-1276 | method | `Mesh2D._clean_reference_chain` | staticmethod |
| 1280-1490 | method | `Mesh2D._prepare_vertices` |  |
| 1292-1294 | nested function | `Mesh2D._prepare_vertices._bucket` |  |
| 1296-1303 | nested function | `Mesh2D._prepare_vertices._find_existing` |  |
| 1305-1306 | nested function | `Mesh2D._prepare_vertices._register` |  |
| 1308-1332 | nested function | `Mesh2D._prepare_vertices._add_point` |  |
| 1351-1366 | nested function | `Mesh2D._prepare_vertices._point_to_chain_distance` |  |
| 1368-1372 | nested function | `Mesh2D._prepare_vertices._point_to_ref_distance` |  |
| 1443-1459 | nested function | `Mesh2D._prepare_vertices._add_geom_point` |  |
| 1492-1497 | method | `Mesh2D._initial_delaunay` |  |
| 1499-1568 | method | `Mesh2D._order_boundary_chain` |  |
| 1570-1577 | method | `Mesh2D._chain_to_edges` |  |
| 1579-1626 | method | `Mesh2D._find_boundary_components` |  |
| 1606-1610 | nested function | `Mesh2D._find_boundary_components._find` |  |
| 1628-1737 | method | `Mesh2D._build_constrained_edge_set` |  |
| 1739-1866 | method | `Mesh2D._insert_all_constrained_edges` |  |
| 1821-1821 | lambda | `Mesh2D._insert_all_constrained_edges.<lambda@1821:24>` |  |
| 1868-1990 | method | `Mesh2D._remove_holes` |  |
| 1945-1945 | lambda | `Mesh2D._remove_holes.<lambda@1945:33>` |  |
| 1957-1957 | lambda | `Mesh2D._remove_holes.<lambda@1957:33>` |  |
| 1992-2005 | method | `Mesh2D._compute_cv_volumes` |  |
| 2007-2067 | method | `Mesh2D._compute_internal_faces` |  |
| 2069-2134 | method | `Mesh2D._compute_boundary_faces` |  |
| 2136-2205 | method | `Mesh2D._validate_topology` |  |
| 2207-2251 | method | `Mesh2D._verify_conservativity` |  |
| 2253-2255 | method | `Mesh2D._assert_built` |  |

**airfoil_pinn/sampling/normal_layer.py**

Primary stage: [8. Пристенное размещение точек и предварительная SDF-диагностика](REPOSITORY_STUDY_PLAN.md#stage-8).
Source: [open file](<../airfoil_pinn/sampling/normal_layer.py>). Lines: 645. SHA-256: `65ac473b45b6cfee3319dfc675f4154a73a8a8119bd8277bb7d8bb5cb96c5af4`.

Module body, imports, exports and class attributes: planned for line-by-line review.

| Lines | Kind | Qualified name | Decorators |
|---|---|---|---|
| 36-54 | class | `NormalLayerConfig` | dataclass |
| 58-64 | class | `NormalLayerResult` | dataclass |
| 61-61 | lambda | `NormalLayerResult.<lambda@61:47>` |  |
| 62-62 | lambda | `NormalLayerResult.<lambda@62:55>` |  |
| 63-63 | lambda | `NormalLayerResult.<lambda@63:48>` |  |
| 68-85 | class | `NormalLayerColumnsResult` | dataclass |
| 76-76 | lambda | `NormalLayerColumnsResult.<lambda@76:47>` |  |
| 77-77 | lambda | `NormalLayerColumnsResult.<lambda@77:55>` |  |
| 78-78 | lambda | `NormalLayerColumnsResult.<lambda@78:49>` |  |
| 79-79 | lambda | `NormalLayerColumnsResult.<lambda@79:48>` |  |
| 80-80 | lambda | `NormalLayerColumnsResult.<lambda@80:48>` |  |
| 81-81 | lambda | `NormalLayerColumnsResult.<lambda@81:57>` |  |
| 82-82 | lambda | `NormalLayerColumnsResult.<lambda@82:51>` |  |
| 83-83 | lambda | `NormalLayerColumnsResult.<lambda@83:51>` |  |
| 84-84 | lambda | `NormalLayerColumnsResult.<lambda@84:44>` |  |
| 88-180 | function | `generate_normal_layer_points` |  |
| 183-296 | function | `generate_normal_layer_columns` |  |
| 299-351 | function | `resample_profile_by_step` |  |
| 354-374 | function | `estimate_first_layer_height_y_plus` |  |
| 377-386 | function | `estimate_skin_friction_coefficient` |  |
| 389-402 | function | `estimate_boundary_layer_thickness` |  |
| 405-468 | function | `_build_offsets` |  |
| 471-478 | function | `_configured_delta` |  |
| 481-524 | function | `_candidate_normals` |  |
| 507-507 | lambda | `_candidate_normals.<lambda@507:33>` |  |
| 527-537 | function | `_valid_flow_points` |  |
| 540-558 | function | `_filter_min_spacing` |  |
| 561-565 | function | `_limit_points` |  |
| 568-572 | function | `_deduplicate` |  |
| 575-582 | function | `_positive_array` |  |
| 585-590 | function | `_remove_adjacent_duplicates` |  |
| 593-596 | function | `_normalize` |  |
| 599-603 | function | `_as_xy` |  |
| 606-607 | function | `_is_closed` |  |
| 610-614 | function | `_chord` |  |
| 617-624 | function | `_orient_tangent_downstream` |  |
| 627-631 | function | `_physical_side_from_normal` |  |

**airfoil_pinn/training/__init__.py**

Primary stage: [3. Служебные типы, импорты и общие утилиты](REPOSITORY_STUDY_PLAN.md#stage-3).
Source: [open file](<../airfoil_pinn/training/__init__.py>). Lines: 25. SHA-256: `5714b5b3cc5952fbdac38f53137c1d414285424e7b8ac9aa67260c72a1575707`.

Module body, imports, exports and class attributes: planned for line-by-line review.

| Lines | Kind | Qualified name | Decorators |
|---|---|---|---|
| whole file | module | no local definitions; review imports, exports and module body | |

**airfoil_pinn/training/batch.py**

Primary stage: [25. Серии экспериментов и перебор конфигураций](REPOSITORY_STUDY_PLAN.md#stage-25).
Source: [open file](<../airfoil_pinn/training/batch.py>). Lines: 297. SHA-256: `9f7a39f853f75ca2c113bed731cf336d49d77b87ba5a3a902f39ba50caa56659`.

Module body, imports, exports and class attributes: planned for line-by-line review.

| Lines | Kind | Qualified name | Decorators |
|---|---|---|---|
| 54-63 | function | `load_base_config` |  |
| 66-84 | function | `load_experiment_list` |  |
| 87-111 | function | `generate_experiments_from_params` |  |
| 114-124 | function | `deep_update` |  |
| 127-134 | function | `create_experiment_dir` |  |
| 137-143 | function | `save_config` |  |
| 146-170 | function | `run_experiment` |  |
| 173-202 | function | `collect_metrics` |  |
| 205-293 | function | `main` |  |

**airfoil_pinn/training/boundary.py**

Primary stage: [15. Граничные условия и преобразование выходов](REPOSITORY_STUDY_PLAN.md#stage-15).
Source: [open file](<../airfoil_pinn/training/boundary.py>). Lines: 167. SHA-256: `bc9179937cac74d1391b1f729774a039694db1f15a056aedc940a669989a5ada`.

Module body, imports, exports and class attributes: planned for line-by-line review.

| Lines | Kind | Qualified name | Decorators |
|---|---|---|---|
| 26-56 | function | `resolve_boundary_conditions` |  |
| 59-68 | function | `distance_mask` |  |
| 71-167 | class | `BoundaryOutputTransform` |  |
| 79-102 | method | `BoundaryOutputTransform.__init__` |  |
| 105-107 | method | `BoundaryOutputTransform.enabled` | property |
| 109-123 | method | `BoundaryOutputTransform.wall_distance` |  |
| 125-152 | method | `BoundaryOutputTransform.forward` |  |
| 154-167 | method | `BoundaryOutputTransform.contract` |  |

**airfoil_pinn/training/boundary_layer_diagnostics.py**

Primary stage: [23. Физическая диагностика пограничного слоя](REPOSITORY_STUDY_PLAN.md#stage-23).
Source: [open file](<../airfoil_pinn/training/boundary_layer_diagnostics.py>). Lines: 888. SHA-256: `7f11aa7701af721c24481c659855a7d86cd8eb1659a54360910af24ff056d67c`.

Module body, imports, exports and class attributes: planned for line-by-line review.

| Lines | Kind | Qualified name | Decorators |
|---|---|---|---|
| 38-136 | class | `BoundaryLayerDiagnosticsConfig` | dataclass |
| 69-107 | method | `BoundaryLayerDiagnosticsConfig.from_dict` | classmethod |
| 109-129 | method | `BoundaryLayerDiagnosticsConfig.normal_layer_config` |  |
| 131-136 | method | `BoundaryLayerDiagnosticsConfig.max_distance` |  |
| 140-145 | class | `BoundaryLayerDiagnosticsResult` | dataclass |
| 148-216 | function | `run_boundary_layer_diagnostics` |  |
| 219-316 | function | `evaluate_wall_normal_profiles` |  |
| 319-383 | function | `compute_boundary_layer_metrics` |  |
| 386-423 | function | `summarize_boundary_layer_metrics` |  |
| 426-469 | function | `save_boundary_layer_outputs` |  |
| 472-508 | function | `plot_wall_normal_profiles` |  |
| 511-543 | function | `plot_surface_coefficient` |  |
| 529-529 | lambda | `plot_surface_coefficient.<lambda@529:32>` |  |
| 546-576 | function | `plot_wall_normal_deficit` |  |
| 579-588 | function | `_load_geometry` |  |
| 591-621 | function | `_filter_columns` |  |
| 624-685 | function | `_predict_model_on_points` |  |
| 688-697 | function | `_boundary_type_matrix` |  |
| 700-732 | function | `_write_metrics_csv` |  |
| 735-754 | function | `_select_section_columns` |  |
| 757-768 | function | `_limit_indices_per_side` |  |
| 771-782 | function | `_edge_velocity` |  |
| 785-804 | function | `_first_threshold_crossing` |  |
| 807-820 | function | `_wall_slope` |  |
| 823-834 | function | `_integral_thicknesses` |  |
| 837-844 | function | `_tuple_float` |  |
| 847-854 | function | `_tuple_str` |  |
| 857-863 | function | `_optional_float` |  |
| 866-870 | function | `_finite_values` |  |
| 873-876 | function | `_safe_mean` |  |
| 879-882 | function | `_safe_min` |  |
| 885-888 | function | `_safe_max` |  |

**airfoil_pinn/training/checkpoint.py**

Primary stage: [21. Checkpoint, контракты и воспроизводимость](REPOSITORY_STUDY_PLAN.md#stage-21).
Source: [open file](<../airfoil_pinn/training/checkpoint.py>). Lines: 128. SHA-256: `397c70c726f79a3226ebf8fba5ddf98db4d79b5078dfe34076df316352e19aed`.

Module body, imports, exports and class attributes: planned for line-by-line review.

| Lines | Kind | Qualified name | Decorators |
|---|---|---|---|
| 26-128 | class | `CheckpointManager` |  |
| 34-40 | method | `CheckpointManager.__init__` |  |
| 43-47 | method | `CheckpointManager._unwrap_model` | staticmethod |
| 49-104 | method | `CheckpointManager.save` |  |
| 106-128 | method | `CheckpointManager.load` |  |

**airfoil_pinn/training/components/__init__.py**

Primary stage: [3. Служебные типы, импорты и общие утилиты](REPOSITORY_STUDY_PLAN.md#stage-3).
Source: [open file](<../airfoil_pinn/training/components/__init__.py>). Lines: 63. SHA-256: `d8ef4342425198b677b358dc1ee358b5825e005ff26b2385c3f7ea867f225474`.

Module body, imports, exports and class attributes: planned for line-by-line review.

| Lines | Kind | Qualified name | Decorators |
|---|---|---|---|
| 22-63 | function | `__getattr__` |  |

**airfoil_pinn/training/components/interactive.py**

Primary stage: [20. Интерактивное управление работающим обучением](REPOSITORY_STUDY_PLAN.md#stage-20).
Source: [open file](<../airfoil_pinn/training/components/interactive.py>). Lines: 327. SHA-256: `c65df5b808a5639b60f423daf306e419f402edc101397501ad0d7a258204d286`.

Module body, imports, exports and class attributes: planned for line-by-line review.

| Lines | Kind | Qualified name | Decorators |
|---|---|---|---|
| 23-327 | class | `InteractiveHandler` |  |
| 31-78 | method | `InteractiveHandler.__init__` |  |
| 80-114 | method | `InteractiveHandler.handle_keys` |  |
| 119-185 | method | `InteractiveHandler._save_current_results` |  |
| 187-245 | method | `InteractiveHandler._edit_weights_interactive` |  |
| 247-266 | method | `InteractiveHandler._reset_optimizer_interactive` |  |
| 268-295 | method | `InteractiveHandler._save_checkpoint_manual` |  |
| 297-309 | method | `InteractiveHandler._adjust_lr` |  |
| 311-327 | method | `InteractiveHandler._set_lr_interactive` |  |

**airfoil_pinn/training/components/loss_computer.py**

Primary stage: [16. Сборка функции потерь, маски и балансировка](REPOSITORY_STUDY_PLAN.md#stage-16).
Source: [open file](<../airfoil_pinn/training/components/loss_computer.py>). Lines: 1320. SHA-256: `3e1856692afb0a9b9c6f944cc8284da78c368ac684d9c6a54e9eb51152de39b4`.

Module body, imports, exports and class attributes: planned for line-by-line review.

| Lines | Kind | Qualified name | Decorators |
|---|---|---|---|
| 39-1320 | class | `LossComputer` |  |
| 45-218 | method | `LossComputer.__init__` |  |
| 220-250 | method | `LossComputer.get_flow_case_batch` |  |
| 252-259 | method | `LossComputer.get_freestream_targets` |  |
| 261-317 | method | `LossComputer._initialize_differentiable_sdf` |  |
| 319-334 | method | `LossComputer.evaluate_sdf_numpy` |  |
| 337-358 | method | `LossComputer.evaluate_sdf_tensor` | torch.no_grad() |
| 360-439 | method | `LossComputer.synchronize_cached_training_sdf` |  |
| 441-524 | method | `LossComputer.prepare_sdf_for_pde` |  |
| 526-578 | method | `LossComputer.compute_pressure_gauge_loss` |  |
| 580-594 | method | `LossComputer._grad` |  |
| 599-618 | method | `LossComputer.compute_fvm_loss` |  |
| 620-644 | method | `LossComputer.compute_fvm_metrics` |  |
| 646-651 | method | `LossComputer._setup_pde_components` |  |
| 656-707 | method | `LossComputer.compute_pde_residuals` |  |
| 712-723 | method | `LossComputer._freestream_velocity_loss` |  |
| 725-757 | method | `LossComputer._boundary_normals` |  |
| 759-771 | method | `LossComputer._normal_derivative` |  |
| 773-793 | method | `LossComputer._normal_velocity_gradient_loss` |  |
| 795-812 | method | `LossComputer._slip_bc_loss` |  |
| 814-850 | method | `LossComputer._symmetry_bc_losses` |  |
| 852-877 | method | `LossComputer.compute_wall_normals` |  |
| 879-899 | method | `LossComputer.compute_wall_point_loss` |  |
| 901-1028 | method | `LossComputer.compute_bc_loss` |  |
| 1033-1080 | method | `LossComputer.compute_mesh_integral_loss` |  |
| 1082-1180 | method | `LossComputer.compute_integral_loss` |  |
| 1139-1170 | nested function | `LossComputer.compute_integral_loss.side_flux` |  |
| 1185-1213 | method | `LossComputer.build_input_tensor` |  |
| 1215-1238 | method | `LossComputer.transform_outputs` |  |
| 1243-1272 | method | `LossComputer.apply_hard_constraints` |  |
| 1274-1302 | method | `LossComputer.apply_velocity_dirichlet_constraint` |  |
| 1307-1320 | method | `LossComputer.update_adaptive_weights` |  |

**airfoil_pinn/training/components/metrics.py**

Primary stage: [22. Метрики, интерполяция сечений и независимый эталон](REPOSITORY_STUDY_PLAN.md#stage-22).
Source: [open file](<../airfoil_pinn/training/components/metrics.py>). Lines: 1714. SHA-256: `5b5e3b42599c93df433353da8eebd4d3755781b9c258145dda0d25b85403c4b9`.

Module body, imports, exports and class attributes: planned for line-by-line review.

| Lines | Kind | Qualified name | Decorators |
|---|---|---|---|
| 30-1714 | class | `MetricsCollector` |  |
| 36-142 | method | `MetricsCollector.__init__` |  |
| 147-266 | method | `MetricsCollector.update_history` |  |
| 268-341 | method | `MetricsCollector._ensure_history_schema` |  |
| 346-361 | method | `MetricsCollector.get_boundary_avg_velocities` |  |
| 363-368 | method | `MetricsCollector.get_boundary_avg_velocities_physical` |  |
| 370-373 | method | `MetricsCollector.invalidate_boundary_cache` |  |
| 375-390 | method | `MetricsCollector.get_mass_flow_metrics` |  |
| 392-396 | method | `MetricsCollector._compute_mass_flow_metrics` |  |
| 398-445 | method | `MetricsCollector._predict_velocity_on_fvm_nodes` |  |
| 447-479 | method | `MetricsCollector._apply_hard_velocity_constraints` |  |
| 481-684 | method | `MetricsCollector._compute_mass_flow_from_fvm_faces` |  |
| 686-765 | method | `MetricsCollector._compute_section_flow_metrics` |  |
| 768-770 | method | `MetricsCollector._format_section_coordinate` | staticmethod |
| 773-803 | method | `MetricsCollector.format_section_flow_log` | staticmethod |
| 806-817 | method | `MetricsCollector._trapz_integral` | staticmethod |
| 819-984 | method | `MetricsCollector._compute_mass_flow_from_boundary_points` |  |
| 986-1052 | method | `MetricsCollector._update_boundary_avg_velocities` |  |
| 1057-1434 | method | `MetricsCollector.print_epoch_table` |  |
| 1104-1112 | nested function | `MetricsCollector.print_epoch_table.format_duration` |  |
| 1114-1117 | nested function | `MetricsCollector.print_epoch_table.format_number` |  |
| 1119-1124 | nested function | `MetricsCollector.print_epoch_table.format_weight` |  |
| 1126-1135 | nested function | `MetricsCollector.print_epoch_table.history_values` |  |
| 1137-1157 | nested function | `MetricsCollector.print_epoch_table.format_delta` |  |
| 1159-1176 | nested function | `MetricsCollector.print_epoch_table.print_row` |  |
| 1186-1194 | nested function | `MetricsCollector.print_epoch_table.border` |  |
| 1196-1200 | nested function | `MetricsCollector.print_epoch_table.print_group` |  |
| 1436-1494 | method | `MetricsCollector._compute_rate` |  |
| 1499-1511 | method | `MetricsCollector.save_log` |  |
| 1516-1705 | method | `MetricsCollector.compute_final_metrics` |  |
| 1710-1714 | method | `MetricsCollector.close` |  |

**airfoil_pinn/training/components/optimizers.py**

Primary stage: [18. Оркестрация, оптимизаторы, численная устойчивость и ускорение](REPOSITORY_STUDY_PLAN.md#stage-18).
Source: [open file](<../airfoil_pinn/training/components/optimizers.py>). Lines: 102. SHA-256: `8f1219be35c795bc04c5afc098c365a94ee7869c0be6dc15132167e3251735b2`.

Module body, imports, exports and class attributes: planned for line-by-line review.

| Lines | Kind | Qualified name | Decorators |
|---|---|---|---|
| 18-64 | class | `LocalMinimaDetector` |  |
| 24-37 | method | `LocalMinimaDetector.__init__` |  |
| 39-64 | method | `LocalMinimaDetector.update` |  |
| 67-102 | class | `CyclicLR` |  |
| 72-84 | method | `CyclicLR.__init__` |  |
| 86-98 | method | `CyclicLR.step` |  |
| 100-102 | method | `CyclicLR.get_lr` |  |

**airfoil_pinn/training/components/rar.py**

Primary stage: [19. Адаптивное добавление точек: RAR](REPOSITORY_STUDY_PLAN.md#stage-19).
Source: [open file](<../airfoil_pinn/training/components/rar.py>). Lines: 1030. SHA-256: `3110a14214c152c77bef1ebafbea71b1c34f8f4f878a59fe8ba75ad43c661bcc`.

Module body, imports, exports and class attributes: planned for line-by-line review.

| Lines | Kind | Qualified name | Decorators |
|---|---|---|---|
| 52-75 | class | `RARGenerationResult` | dataclass |
| 55-55 | lambda | `RARGenerationResult.<lambda@55:42>` |  |
| 56-56 | lambda | `RARGenerationResult.<lambda@56:42>` |  |
| 57-57 | lambda | `RARGenerationResult.<lambda@57:44>` |  |
| 59-59 | lambda | `RARGenerationResult.<lambda@59:24>` |  |
| 63-71 | method | `RARGenerationResult.__post_init__` |  |
| 74-75 | method | `RARGenerationResult.n_points` | property |
| 78-184 | class | `RARManager` |  |
| 81-85 | method | `RARManager.__init__` |  |
| 87-159 | method | `RARManager.generate` |  |
| 162-184 | method | `RARManager._deduplicate` | staticmethod |
| 187-377 | class | `FlowRAR` |  |
| 190-191 | method | `FlowRAR.__init__` |  |
| 193-367 | method | `FlowRAR.generate` |  |
| 369-377 | method | `FlowRAR._exclude_sdf_delta` |  |
| 380-804 | class | `BoundaryLayerRAR` |  |
| 383-384 | method | `BoundaryLayerRAR.__init__` |  |
| 386-513 | method | `BoundaryLayerRAR.generate` |  |
| 515-529 | method | `BoundaryLayerRAR._delta_norm` |  |
| 531-544 | method | `BoundaryLayerRAR._build_candidates` |  |
| 546-629 | method | `BoundaryLayerRAR._filter_candidates` |  |
| 631-742 | method | `BoundaryLayerRAR._score_candidates` |  |
| 744-804 | method | `BoundaryLayerRAR._sample_candidates` |  |
| 807-814 | function | `_estimate_delta_norm` |  |
| 817-821 | function | `_dataset_xy` |  |
| 824-829 | function | `_interior_xy` |  |
| 832-848 | function | `_domain_norm` |  |
| 851-857 | function | `_foil_norm` |  |
| 860-873 | function | `_canonical_sdf_numpy` |  |
| 876-888 | function | `_interior_boundary_types` |  |
| 891-924 | function | `valid_interior_mask` |  |
| 927-969 | function | `resample_polyline_by_arclength` |  |
| 972-1002 | function | `choose_outward_normals` |  |
| 1005-1010 | function | `_remove_adjacent_duplicates` |  |
| 1013-1018 | function | `_positive_array` |  |
| 1021-1030 | function | `_format_eta_counts` |  |

**airfoil_pinn/training/components/trainer.py**

Primary stage: [18. Оркестрация, оптимизаторы, численная устойчивость и ускорение](REPOSITORY_STUDY_PLAN.md#stage-18).
Source: [open file](<../airfoil_pinn/training/components/trainer.py>). Lines: 2631. SHA-256: `81af6a3ccf66ed4ae6c48de8c63040114ae6cafcb7522d5d3a7f59b5e34f04f1`.

Module body, imports, exports and class attributes: planned for line-by-line review.

| Lines | Kind | Qualified name | Decorators |
|---|---|---|---|
| 53-2631 | class | `Trainer` |  |
| 59-186 | method | `Trainer.__init__` |  |
| 188-202 | method | `Trainer._grad` |  |
| 204-207 | method | `Trainer._autocast_context` |  |
| 212-609 | method | `Trainer.forward_pass` |  |
| 611-627 | method | `Trainer._pde_loss_mask` |  |
| 630-636 | method | `Trainer._masked_mean` | staticmethod |
| 639-645 | method | `Trainer._masked_max` | staticmethod |
| 647-715 | method | `Trainer._compute_sa_bc_loss` |  |
| 717-756 | method | `Trainer._apply_hard_constraints_from_boundary_types` |  |
| 758-819 | method | `Trainer._compute_rar_point_score` |  |
| 821-928 | method | `Trainer._compute_bc_point_score` |  |
| 930-936 | method | `Trainer._update_rar_point_score` |  |
| 938-944 | method | `Trainer._update_pde_point_residual` |  |
| 949-987 | method | `Trainer.compute_grad_norms_separately` |  |
| 962-968 | nested function | `Trainer.compute_grad_norms_separately._flatten_grads` |  |
| 989-1042 | method | `Trainer.compute_ntk_condition` |  |
| 1047-1138 | method | `Trainer.get_predictions_on_dataset` |  |
| 1143-1167 | method | `Trainer.get_profile_at_x` |  |
| 1172-1193 | method | `Trainer.compute_spectrum` |  |
| 1198-1248 | method | `Trainer.compute_wall_gradient` |  |
| 1250-1271 | method | `Trainer.get_state` |  |
| 1273-1337 | method | `Trainer.load_state` |  |
| 1280-1296 | nested function | `Trainer.load_state.restore_tensor` |  |
| 1342-1397 | method | `Trainer.apply_rar` |  |
| 1399-1439 | method | `Trainer._apply_rar_legacy` |  |
| 1441-1468 | method | `Trainer._log_rar_sdf_delta_statistics` |  |
| 1470-1476 | method | `Trainer._extend_sa_weights` |  |
| 1478-1634 | method | `Trainer._generate_adaptive_rar_points` |  |
| 1637-1671 | method | `Trainer._valid_interior_mask` | staticmethod |
| 1673-1727 | method | `Trainer._maybe_rebuild_fvm_after_rar` |  |
| 1729-1753 | method | `Trainer._save_rar_mesh_volume_plot` |  |
| 1755-1771 | method | `Trainer._get_fvm_tensors` |  |
| 1773-1840 | method | `Trainer._forward_fvm_cases` |  |
| 1842-1923 | method | `Trainer._compute_global_integral_loss_from_dataset` |  |
| 1928-1991 | method | `Trainer._run_fvm_step` |  |
| 1993-2028 | method | `Trainer._scheduled_fvm_batches` |  |
| 2033-2148 | method | `Trainer.train_batch_adam` |  |
| 2153-2360 | method | `Trainer.train_epoch_adam` |  |
| 2365-2559 | method | `Trainer.train_epoch_lbfgs` |  |
| 2409-2492 | nested function | `Trainer.train_epoch_lbfgs.closure` |  |
| 2564-2601 | method | `Trainer.update_adaptive_weights` |  |
| 2606-2631 | method | `Trainer.reset_optimizer` |  |

**airfoil_pinn/training/components/utils.py**

Primary stage: [3. Служебные типы, импорты и общие утилиты](REPOSITORY_STUDY_PLAN.md#stage-3).
Source: [open file](<../airfoil_pinn/training/components/utils.py>). Lines: 0. SHA-256: `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.

Module body, imports, exports and class attributes: planned for line-by-line review.

| Lines | Kind | Qualified name | Decorators |
|---|---|---|---|
| whole file | module | no local definitions; review imports, exports and module body | |

**airfoil_pinn/training/contract.py**

Primary stage: [21. Checkpoint, контракты и воспроизводимость](REPOSITORY_STUDY_PLAN.md#stage-21).
Source: [open file](<../airfoil_pinn/training/contract.py>). Lines: 265. SHA-256: `a96776a00646b5e88b0f22f49ec88bc4f31c8ad3355e3d8b018de40b754969f7`.

Module body, imports, exports and class attributes: planned for line-by-line review.

| Lines | Kind | Qualified name | Decorators |
|---|---|---|---|
| 22-72 | class | `TrainingContract` | dataclass(frozen=True) |
| 27-35 | method | `TrainingContract.fingerprint` |  |
| 37-44 | method | `TrainingContract.to_metadata` |  |
| 47-72 | method | `TrainingContract.from_metadata` | classmethod |
| 75-235 | function | `build_training_contract` |  |
| 238-247 | function | `_top_level_differences` |  |
| 250-265 | function | `validate_checkpoint_contract` |  |

**airfoil_pinn/training/input_features.py**

Primary stage: [13. Входные признаки и нейронная сеть](REPOSITORY_STUDY_PLAN.md#stage-13).
Source: [open file](<../airfoil_pinn/training/input_features.py>). Lines: 354. SHA-256: `a7b7ba994fdf376a371dd067dc26bee5b18d3be33cf1a03fb3286d621bdfa83f`.

Module body, imports, exports and class attributes: planned for line-by-line review.

| Lines | Kind | Qualified name | Decorators |
|---|---|---|---|
| 26-47 | class | `FeatureSlice` | dataclass(frozen=True) |
| 35-37 | method | `FeatureSlice.width` | property |
| 39-47 | method | `FeatureSlice.to_dict` |  |
| 51-78 | class | `InputFeatureManifest` | dataclass(frozen=True) |
| 57-59 | method | `InputFeatureManifest.input_dim` | property |
| 61-67 | method | `InputFeatureManifest.to_dict` |  |
| 69-78 | method | `InputFeatureManifest.fingerprint` |  |
| 81-354 | class | `InputFeatureAssembler` |  |
| 95-190 | method | `InputFeatureAssembler.__init__` |  |
| 192-207 | method | `InputFeatureAssembler._warn_deprecated_boundary_input` |  |
| 209-212 | method | `InputFeatureAssembler._feature_mapping` |  |
| 214-229 | method | `InputFeatureAssembler._enabled` |  |
| 231-262 | method | `InputFeatureAssembler._build_manifest` |  |
| 236-246 | nested function | `InputFeatureAssembler._build_manifest.append` |  |
| 264-322 | method | `InputFeatureAssembler.build` |  |
| 325-338 | method | `InputFeatureAssembler._validate_base_shapes` | staticmethod |
| 341-354 | method | `InputFeatureAssembler._require_flow` | staticmethod |

**airfoil_pinn/training/key_listener.py**

Primary stage: [20. Интерактивное управление работающим обучением](REPOSITORY_STUDY_PLAN.md#stage-20).
Source: [open file](<../airfoil_pinn/training/key_listener.py>). Lines: 123. SHA-256: `a2484951af1b6ddd73f3d7cde08b6729f3c70946d5fc9c5d269c1e374176b094`.

Module body, imports, exports and class attributes: planned for line-by-line review.

| Lines | Kind | Qualified name | Decorators |
|---|---|---|---|
| 25-123 | class | `KeyListener` |  |
| 39-48 | method | `KeyListener.__init__` |  |
| 50-69 | method | `KeyListener.start` |  |
| 71-84 | method | `KeyListener._run` |  |
| 86-88 | method | `KeyListener._on_save_key` |  |
| 90-92 | method | `KeyListener._on_edit_key` |  |
| 94-96 | method | `KeyListener._on_reset_key` |  |
| 98-100 | method | `KeyListener._on_checkpoint_key` |  |
| 102-104 | method | `KeyListener._on_lr_increase_key` |  |
| 106-108 | method | `KeyListener._on_lr_decrease_key` |  |
| 110-112 | method | `KeyListener._on_lr_set_key` |  |
| 114-123 | method | `KeyListener.stop` |  |

**airfoil_pinn/training/learning_cli.py**

Primary stage: [2. Точки входа, окружение, пути и консольное приложение](REPOSITORY_STUDY_PLAN.md#stage-2).
Source: [open file](<../airfoil_pinn/training/learning_cli.py>). Lines: 62. SHA-256: `c7bdf2023866df35bdb498dd60c29e81dd9946b1c1ba95e3d2a71a82cf4a6d34`.

Module body, imports, exports and class attributes: planned for line-by-line review.

| Lines | Kind | Qualified name | Decorators |
|---|---|---|---|
| 27-58 | function | `main` |  |

**airfoil_pinn/training/menus/__init__.py**

Primary stage: [3. Служебные типы, импорты и общие утилиты](REPOSITORY_STUDY_PLAN.md#stage-3).
Source: [open file](<../airfoil_pinn/training/menus/__init__.py>). Lines: 98. SHA-256: `195fcbbd3c967170fdbdbf08f0f07e6cb3124167f1462722d4e738449fdfe0ab`.

Module body, imports, exports and class attributes: planned for line-by-line review.

| Lines | Kind | Qualified name | Decorators |
|---|---|---|---|
| 35-98 | function | `__getattr__` |  |

**airfoil_pinn/training/menus/architecture.py**

Primary stage: [4. Конфигурация и все меню настройки обучения](REPOSITORY_STUDY_PLAN.md#stage-4).
Source: [open file](<../airfoil_pinn/training/menus/architecture.py>). Lines: 255. SHA-256: `15871b3451ea3f4e9a3ba3f1dd45cfe6f8554bf28f82fba9b3ec2882a5441839`.

Module body, imports, exports and class attributes: planned for line-by-line review.

| Lines | Kind | Qualified name | Decorators |
|---|---|---|---|
| 31-85 | function | `_configure_input_params` |  |
| 88-93 | function | `_configure_hidden_layers` |  |
| 96-147 | function | `_configure_activation` |  |
| 150-190 | function | `_configure_coordinate_scaling` |  |
| 193-214 | function | `_configure_output_params` |  |
| 221-255 | function | `architecture_menu` |  |

**airfoil_pinn/training/menus/common.py**

Primary stage: [4. Конфигурация и все меню настройки обучения](REPOSITORY_STUDY_PLAN.md#stage-4).
Source: [open file](<../airfoil_pinn/training/menus/common.py>). Lines: 20. SHA-256: `e623948f2a11b9eab267d4a0be673a5c2330681a9cd34ae4e00f2d2011914c6b`.

Module body, imports, exports and class attributes: planned for line-by-line review.

| Lines | Kind | Qualified name | Decorators |
|---|---|---|---|
| 13-15 | function | `print_submenu_header` |  |
| 18-20 | function | `print_submenu_footer` |  |

**airfoil_pinn/training/menus/configuration.py**

Primary stage: [4. Конфигурация и все меню настройки обучения](REPOSITORY_STUDY_PLAN.md#stage-4).
Source: [open file](<../airfoil_pinn/training/menus/configuration.py>). Lines: 209. SHA-256: `26bdf7dfd018767153288ad877727a42c28f80e5e42239f9fb0e58a6841df743`.

Module body, imports, exports and class attributes: planned for line-by-line review.

| Lines | Kind | Qualified name | Decorators |
|---|---|---|---|
| 41-79 | function | `main_menu` |  |
| 82-123 | function | `create_new_config` |  |
| 126-149 | function | `load_config` |  |
| 152-192 | function | `edit_config` |  |
| 195-209 | function | `save_config` |  |

**airfoil_pinn/training/menus/loss.py**

Primary stage: [4. Конфигурация и все меню настройки обучения](REPOSITORY_STUDY_PLAN.md#stage-4).
Source: [open file](<../airfoil_pinn/training/menus/loss.py>). Lines: 236. SHA-256: `3d7d4b86161e90ea19b9aa39570203f7e3895ff2a884e69be18bf4e396c50541`.

Module body, imports, exports and class attributes: planned for line-by-line review.

| Lines | Kind | Qualified name | Decorators |
|---|---|---|---|
| 19-98 | function | `loss_function_menu` |  |
| 101-236 | function | `weights_menu` |  |

**airfoil_pinn/training/menus/process.py**

Primary stage: [4. Конфигурация и все меню настройки обучения](REPOSITORY_STUDY_PLAN.md#stage-4).
Source: [open file](<../airfoil_pinn/training/menus/process.py>). Lines: 84. SHA-256: `4e182cbc623a02cf37322130776142c27063aedec40e306d1f454608ed73c380`.

Module body, imports, exports and class attributes: planned for line-by-line review.

| Lines | Kind | Qualified name | Decorators |
|---|---|---|---|
| 28-84 | function | `process_settings` |  |

**airfoil_pinn/training/menus/results.py**

Primary stage: [4. Конфигурация и все меню настройки обучения](REPOSITORY_STUDY_PLAN.md#stage-4).
Source: [open file](<../airfoil_pinn/training/menus/results.py>). Lines: 165. SHA-256: `f07cd2415e5c0431df1ebaaa61205094a9fc12d94fae937c5620f951564dc645`.

Module body, imports, exports and class attributes: planned for line-by-line review.

| Lines | Kind | Qualified name | Decorators |
|---|---|---|---|
| 21-53 | function | `output_menu` |  |
| 56-83 | function | `checkpoint_menu` |  |
| 86-165 | function | `diagnostics_menu` |  |

**airfoil_pinn/training/menus/strategy.py**

Primary stage: [4. Конфигурация и все меню настройки обучения](REPOSITORY_STUDY_PLAN.md#stage-4).
Source: [open file](<../airfoil_pinn/training/menus/strategy.py>). Lines: 247. SHA-256: `60c52ce7998785d1caee3893e82c58c0d2d084bab3d92db582780b9710af31d0`.

Module body, imports, exports and class attributes: planned for line-by-line review.

| Lines | Kind | Qualified name | Decorators |
|---|---|---|---|
| 22-247 | function | `training_strategy_menu` |  |

**airfoil_pinn/training/models.py**

Primary stage: [13. Входные признаки и нейронная сеть](REPOSITORY_STUDY_PLAN.md#stage-13).
Source: [open file](<../airfoil_pinn/training/models.py>). Lines: 321. SHA-256: `28d6fa47fa3c1fd19dac15a5733b29582b2f7ab47c0583fc15fafcde167f334a`.

Module body, imports, exports and class attributes: planned for line-by-line review.

| Lines | Kind | Qualified name | Decorators |
|---|---|---|---|
| 37-55 | class | `Sine` |  |
| 46-51 | method | `Sine.__init__` |  |
| 53-55 | method | `Sine.forward` |  |
| 58-252 | class | `PINNIncompressible` |  |
| 75-164 | method | `PINNIncompressible.__init__` |  |
| 166-198 | method | `PINNIncompressible._get_activation` |  |
| 200-227 | method | `PINNIncompressible._init_weights` |  |
| 229-252 | method | `PINNIncompressible.forward` |  |
| 255-321 | class | `SelfAdaptiveWeights` |  |
| 264-293 | method | `SelfAdaptiveWeights.__init__` |  |
| 296-298 | method | `SelfAdaptiveWeights._reverse_gradient` | staticmethod |
| 300-302 | method | `SelfAdaptiveWeights.forward` |  |
| 305-307 | method | `SelfAdaptiveWeights.post_step_` | torch.no_grad() |
| 309-321 | method | `SelfAdaptiveWeights.extend` |  |

**airfoil_pinn/training/network_diagram.py**

Primary stage: [13. Входные признаки и нейронная сеть](REPOSITORY_STUDY_PLAN.md#stage-13).
Source: [open file](<../airfoil_pinn/training/network_diagram.py>). Lines: 167. SHA-256: `50a8b6e11c28d6544a6c2c7a47db31881fa3c414ce5c86c426014dc5ce172af8`.

Module body, imports, exports and class attributes: planned for line-by-line review.

| Lines | Kind | Qualified name | Decorators |
|---|---|---|---|
| 23-45 | function | `_input_labels` |  |
| 48-167 | function | `visualize_network` |  |

**airfoil_pinn/training/numerics.py**

Primary stage: [18. Оркестрация, оптимизаторы, численная устойчивость и ускорение](REPOSITORY_STUDY_PLAN.md#stage-18).
Source: [open file](<../airfoil_pinn/training/numerics.py>). Lines: 131. SHA-256: `66494d47193b5a5b0c9b637a3a7485d4eb45450d1077e2ba453f1a27d89e405f`.

Module body, imports, exports and class attributes: planned for line-by-line review.

| Lines | Kind | Qualified name | Decorators |
|---|---|---|---|
| 25-26 | class | `NumericalTrainingError` |  |
| 30-41 | class | `GradientHealth` | dataclass(frozen=True) |
| 39-41 | method | `GradientHealth.is_finite` | property |
| 44-63 | function | `_component_summary` |  |
| 66-81 | function | `ensure_finite_loss` |  |
| 84-131 | function | `check_gradient_health` |  |

**airfoil_pinn/training/progress.py**

Primary stage: [18. Оркестрация, оптимизаторы, численная устойчивость и ускорение](REPOSITORY_STUDY_PLAN.md#stage-18).
Source: [open file](<../airfoil_pinn/training/progress.py>). Lines: 43. SHA-256: `40cf89b052afc14f12c60fdb2b16877789592ac1ce89372a7092ef741105cad3`.

Module body, imports, exports and class attributes: planned for line-by-line review.

| Lines | Kind | Qualified name | Decorators |
|---|---|---|---|
| 14-43 | function | `build_epoch_progress` |  |

**airfoil_pinn/training/reference.py**

Primary stage: [22. Метрики, интерполяция сечений и независимый эталон](REPOSITORY_STUDY_PLAN.md#stage-22).
Source: [open file](<../airfoil_pinn/training/reference.py>). Lines: 214. SHA-256: `04a61c9aac71aa30b028be41423c7ab17da4710cb4e73519e1306b4e6444d6a3`.

Module body, imports, exports and class attributes: planned for line-by-line review.

| Lines | Kind | Qualified name | Decorators |
|---|---|---|---|
| 36-80 | function | `load_reference` |  |
| 83-123 | function | `interpolate_reference` |  |
| 126-183 | function | `compute_l2_errors` |  |
| 186-198 | function | `save_reference_as_npy` |  |
| 201-214 | function | `load_reference_from_npy` |  |

**airfoil_pinn/training/reproducibility.py**

Primary stage: [21. Checkpoint, контракты и воспроизводимость](REPOSITORY_STUDY_PLAN.md#stage-21).
Source: [open file](<../airfoil_pinn/training/reproducibility.py>). Lines: 65. SHA-256: `905604edd51e18435d6420b1540b01f33ddac401b78e123125285b1e5c650ce0`.

Module body, imports, exports and class attributes: planned for line-by-line review.

| Lines | Kind | Qualified name | Decorators |
|---|---|---|---|
| 17-30 | function | `capture_rng_state` |  |
| 33-65 | function | `restore_rng_state` |  |

**airfoil_pinn/training/runner.py**

Primary stage: [18. Оркестрация, оптимизаторы, численная устойчивость и ускорение](REPOSITORY_STUDY_PLAN.md#stage-18).
Source: [open file](<../airfoil_pinn/training/runner.py>). Lines: 108. SHA-256: `2d86bbeecfa889873979d5a3e1cd65da1d9d53ec1b491bf448a36bce6ea728aa`.

Module body, imports, exports and class attributes: planned for line-by-line review.

| Lines | Kind | Qualified name | Decorators |
|---|---|---|---|
| 28-52 | function | `run_experiment` |  |
| 55-108 | function | `start_training` |  |

**airfoil_pinn/training/solver.py**

Primary stage: [18. Оркестрация, оптимизаторы, численная устойчивость и ускорение](REPOSITORY_STUDY_PLAN.md#stage-18).
Source: [open file](<../airfoil_pinn/training/solver.py>). Lines: 1623. SHA-256: `feefb145ca9e04efd345858b5da4f002cd1dab1b01676f50ab7bbbcb7b6368ff`.

Module body, imports, exports and class attributes: planned for line-by-line review.

| Lines | Kind | Qualified name | Decorators |
|---|---|---|---|
| 75-1623 | class | `PINNSolver` |  |
| 81-328 | method | `PINNSolver.__init__` |  |
| 333-347 | method | `PINNSolver._init_device` |  |
| 349-350 | method | `PINNSolver._init_dtype` |  |
| 352-405 | method | `PINNSolver._write_run_manifest` |  |
| 407-453 | method | `PINNSolver._build_model` |  |
| 455-494 | method | `PINNSolver._maybe_compile_model` |  |
| 497-501 | method | `PINNSolver._unwrap_model_for_state` | staticmethod |
| 504-510 | method | `PINNSolver._strip_state_prefix` | staticmethod |
| 512-535 | method | `PINNSolver._validate_physics_metadata` |  |
| 537-612 | method | `PINNSolver._setup_fvm_residuals` |  |
| 614-643 | method | `PINNSolver._init_optimizer` |  |
| 645-665 | method | `PINNSolver._ensure_sa_weights_shape` |  |
| 667-679 | method | `PINNSolver._init_scheduler` |  |
| 681-699 | method | `PINNSolver._restore_checkpoint_data_state` |  |
| 701-861 | method | `PINNSolver._load_checkpoint_if_needed` |  |
| 863-866 | method | `PINNSolver._get_run_dir` |  |
| 868-871 | method | `PINNSolver._get_checkpoint_dir` |  |
| 873-876 | method | `PINNSolver._get_logs_dir` |  |
| 878-881 | method | `PINNSolver._get_plots_dir` |  |
| 883-907 | method | `PINNSolver._save_checkpoint` |  |
| 909-947 | method | `PINNSolver._save_checkpoint_flow_images` |  |
| 920-921 | nested function | `PINNSolver._save_checkpoint_flow_images.as_column_tensor` |  |
| 952-1093 | method | `PINNSolver._run_diagnostics` |  |
| 1095-1171 | method | `PINNSolver._get_predictions_norm` |  |
| 1176-1527 | method | `PINNSolver.train` |  |
| 1529-1536 | method | `PINNSolver._get_pde_multiplier` |  |
| 1538-1544 | method | `PINNSolver._capture_trainable_state` |  |
| 1546-1554 | method | `PINNSolver._restore_trainable_state` |  |
| 1556-1584 | method | `PINNSolver._update_early_stopping` |  |
| 1586-1603 | method | `PINNSolver._switch_to_lbfgs` |  |
| 1605-1614 | method | `PINNSolver._recreate_optimizer` |  |
| 1616-1623 | method | `PINNSolver._finalize` |  |

**airfoil_pinn/training/telemetry.py**

Primary stage: [24. Визуализация, спектры, экспорт и telemetry](REPOSITORY_STUDY_PLAN.md#stage-24).
Source: [open file](<../airfoil_pinn/training/telemetry.py>). Lines: 79. SHA-256: `2aaadbb4c181cc9469652584f4c80db375a9c8d2e69d0a8d56396f72aed6a5ae`.

Module body, imports, exports and class attributes: planned for line-by-line review.

| Lines | Kind | Qualified name | Decorators |
|---|---|---|---|
| 16-36 | function | `_json_value` |  |
| 39-79 | class | `RunTelemetry` |  |
| 42-47 | method | `RunTelemetry.__init__` |  |
| 49-65 | method | `RunTelemetry.write_manifest` |  |
| 67-79 | method | `RunTelemetry.append_epoch` |  |

**airfoil_pinn/training/visualizer.py**

Primary stage: [24. Визуализация, спектры, экспорт и telemetry](REPOSITORY_STUDY_PLAN.md#stage-24).
Source: [open file](<../airfoil_pinn/training/visualizer.py>). Lines: 1132. SHA-256: `b9d30f11257061cdce80236df70b2cb6a4c581015f96d369e2653550a52ffdcb`.

Module body, imports, exports and class attributes: planned for line-by-line review.

| Lines | Kind | Qualified name | Decorators |
|---|---|---|---|
| 48-1132 | class | `Visualizer` |  |
| 55-74 | method | `Visualizer.__init__` |  |
| 76-157 | method | `Visualizer.create_live_plot` |  |
| 159-246 | method | `Visualizer.update_plot` |  |
| 248-271 | method | `Visualizer.save_plots` |  |
| 273-291 | method | `Visualizer.save_final_report` |  |
| 293-310 | method | `Visualizer.append_metrics_to_csv` |  |
| 313-324 | method | `Visualizer._deduplicate_field_points` | staticmethod |
| 326-441 | method | `Visualizer._plot_smooth_field_map` |  |
| 444-455 | method | `Visualizer._clean_polyline` | staticmethod |
| 458-465 | method | `Visualizer._clip_polygon` | classmethod |
| 468-475 | method | `Visualizer._closed_mask_polygon` | classmethod |
| 478-479 | method | `Visualizer._polyline` | classmethod |
| 482-486 | method | `Visualizer._scale_geometry_for_plot` | staticmethod |
| 489-495 | method | `Visualizer._points_inside_polygon` | staticmethod |
| 498-513 | method | `Visualizer._apply_grid_masks` | classmethod |
| 515-543 | method | `Visualizer.plot_residual_map` |  |
| 545-569 | method | `Visualizer.plot_rar_error_map` |  |
| 571-593 | method | `Visualizer.plot_gradient_norm_history` |  |
| 595-675 | method | `Visualizer.save_fields_as_npy` |  |
| 677-823 | method | `Visualizer.plot_final_fields_from_tensors` |  |
| 829-854 | method | `Visualizer.plot_velocity_profiles_multiple` |  |
| 856-897 | method | `Visualizer.plot_1d_spectrum` |  |
| 899-986 | method | `Visualizer.plot_2d_spectrum` |  |
| 988-1031 | method | `Visualizer.plot_rar_histogram` |  |
| 1033-1070 | method | `Visualizer.plot_gradient_norms_separate` |  |
| 1072-1117 | method | `Visualizer.plot_final_fields` |  |
| 1119-1132 | method | `Visualizer.close` |  |

**airfoil_pinn/utils/__init__.py**

Primary stage: [3. Служебные типы, импорты и общие утилиты](REPOSITORY_STUDY_PLAN.md#stage-3).
Source: [open file](<../airfoil_pinn/utils/__init__.py>). Lines: 56. SHA-256: `7100b228a0f4647efda62854bc651dd0e4a88b246079faf6e6d3ffaf24fad6a4`.

Module body, imports, exports and class attributes: planned for line-by-line review.

| Lines | Kind | Qualified name | Decorators |
|---|---|---|---|
| whole file | module | no local definitions; review imports, exports and module body | |

**airfoil_pinn/utils/auxiliary_utils.py**

Primary stage: [3. Служебные типы, импорты и общие утилиты](REPOSITORY_STUDY_PLAN.md#stage-3).
Source: [open file](<../airfoil_pinn/utils/auxiliary_utils.py>). Lines: 137. SHA-256: `99bff13f89d4882bc9278a758808dea82812c4c95aa8d15ef1697c033d8820eb`.

Module body, imports, exports and class attributes: planned for line-by-line review.

| Lines | Kind | Qualified name | Decorators |
|---|---|---|---|
| 31-48 | function | `set_seed` |  |
| 51-88 | function | `check_memory_usage` |  |
| 91-137 | function | `safe_grad` |  |

**airfoil_pinn/utils/io_utils.py**

Primary stage: [3. Служебные типы, импорты и общие утилиты](REPOSITORY_STUDY_PLAN.md#stage-3).
Source: [open file](<../airfoil_pinn/utils/io_utils.py>). Lines: 320. SHA-256: `8c2a5931338b1bbe3273a656d5e636294101022a04fe9c1aaa88f4d9a68016f4`.

Module body, imports, exports and class attributes: planned for line-by-line review.

| Lines | Kind | Qualified name | Decorators |
|---|---|---|---|
| 28-66 | function | `sanitize_filename` |  |
| 69-85 | function | `ensure_directory` |  |
| 88-134 | function | `save_json` |  |
| 137-175 | function | `load_json` |  |
| 178-208 | function | `save_text` |  |
| 211-239 | function | `load_text` |  |
| 242-268 | function | `file_exists` |  |
| 271-287 | function | `get_file_stem` |  |
| 290-320 | function | `list_files` |  |

**airfoil_pinn/utils/ui_utils.py**

Primary stage: [3. Служебные типы, импорты и общие утилиты](REPOSITORY_STUDY_PLAN.md#stage-3).
Source: [open file](<../airfoil_pinn/utils/ui_utils.py>). Lines: 448. SHA-256: `1661371c97f0aa57417b2c6cea30fdf42ee5b650293e2e7c9ed685f83fedb507`.

Module body, imports, exports and class attributes: planned for line-by-line review.

| Lines | Kind | Qualified name | Decorators |
|---|---|---|---|
| 27-34 | function | `clear_screen` |  |
| 37-45 | function | `print_separator` |  |
| 48-97 | function | `get_menu_choice` |  |
| 100-161 | function | `safe_input_number` |  |
| 164-197 | function | `confirm_action` |  |
| 200-234 | function | `input_with_default` |  |
| 237-247 | function | `pause_and_continue` |  |
| 250-260 | function | `print_header` |  |
| 263-297 | function | `print_list` |  |
| 300-431 | class | `BaseMenu` |  |
| 324-344 | method | `BaseMenu.__init__` |  |
| 346-365 | method | `BaseMenu.display` |  |
| 367-376 | method | `BaseMenu.get_choice` |  |
| 379-389 | method | `BaseMenu.handle_choice` | abstractmethod |
| 391-393 | method | `BaseMenu.before_loop` |  |
| 395-397 | method | `BaseMenu.after_loop` |  |
| 399-427 | method | `BaseMenu.run` |  |
| 429-431 | method | `BaseMenu.stop` |  |
| 435-448 | class | `Colors` |  |

**ui/__init__.py**

Primary stage: [3. Служебные типы, импорты и общие утилиты](REPOSITORY_STUDY_PLAN.md#stage-3).
Source: [open file](<../ui/__init__.py>). Lines: 1. SHA-256: `9f5382f4a1558ad4a1694313c60231c7db6a424094c3648fd6a720a103df779a`.

Module body, imports, exports and class attributes: planned for line-by-line review.

| Lines | Kind | Qualified name | Decorators |
|---|---|---|---|
| whole file | module | no local definitions; review imports, exports and module body | |

**ui/app.py**

Primary stage: [27. UI: приложение, обучение, процессы и результаты](REPOSITORY_STUDY_PLAN.md#stage-27).
Source: [open file](<../ui/app.py>). Lines: 30. SHA-256: `ec3ffd8b5fd3128f1088f5fb403cc3a7307328d6432da566341ee0451ed381eb`.

Module body, imports, exports and class attributes: planned for line-by-line review.

| Lines | Kind | Qualified name | Decorators |
|---|---|---|---|
| 18-26 | function | `main` |  |

**ui/geometry_frontend.py**

Primary stage: [26. UI: геометрия, редактор, виджеты и демонстрационные элементы](REPOSITORY_STUDY_PLAN.md#stage-26).
Source: [open file](<../ui/geometry_frontend.py>). Lines: 506. SHA-256: `9a1819708e29baba9603504ba444683b10aba94bcf729933744c4b540d1d9f72`.

Module body, imports, exports and class attributes: planned for line-by-line review.

| Lines | Kind | Qualified name | Decorators |
|---|---|---|---|
| 21-23 | class | `Point` | dataclass(frozen=True) |
| 27-42 | class | `GeometryStats` | dataclass(frozen=True) |
| 37-38 | method | `GeometryStats.width` | property |
| 41-42 | method | `GeometryStats.height` | property |
| 46-49 | class | `GeometryArtifacts` | dataclass(frozen=True) |
| 52-53 | class | `GeometryReadError` |  |
| 56-506 | class | `GeometryFrontendSession` |  |
| 59-80 | method | `GeometryFrontendSession.__init__` |  |
| 83-84 | method | `GeometryFrontendSession.is_loaded` | property |
| 87-88 | method | `GeometryFrontendSession.is_closed` | property |
| 90-94 | method | `GeometryFrontendSession.available_foil_files` |  |
| 96-100 | method | `GeometryFrontendSession.available_library_names` |  |
| 102-119 | method | `GeometryFrontendSession.load_sample` |  |
| 121-141 | method | `GeometryFrontendSession.load_file` |  |
| 143-153 | method | `GeometryFrontendSession.load_from_library` |  |
| 155-170 | method | `GeometryFrontendSession.load_inline_xy` |  |
| 172-181 | method | `GeometryFrontendSession.load_text_block` |  |
| 183-208 | method | `GeometryFrontendSession.replace_foil_points` |  |
| 210-249 | method | `GeometryFrontendSession.create_domain` |  |
| 251-279 | method | `GeometryFrontendSession.y_at_x` |  |
| 252-252 | lambda | `GeometryFrontendSession.y_at_x.<lambda@252:46>` |  |
| 281-308 | method | `GeometryFrontendSession.save` |  |
| 310-316 | method | `GeometryFrontendSession.canvas_bounds` |  |
| 318-321 | method | `GeometryFrontendSession._load_npy` |  |
| 323-328 | method | `GeometryFrontendSession._load_csv` |  |
| 330-331 | method | `GeometryFrontendSession._load_text` |  |
| 333-341 | method | `GeometryFrontendSession._parse_coordinate_text` |  |
| 343-356 | method | `GeometryFrontendSession._load_from_library_item` |  |
| 358-367 | method | `GeometryFrontendSession._points_from_rows` |  |
| 369-385 | method | `GeometryFrontendSession._compute_stats` |  |
| 387-398 | method | `GeometryFrontendSession._save_points` |  |
| 400-408 | method | `GeometryFrontendSession._save_metadata` |  |
| 410-411 | method | `GeometryFrontendSession._make_naca0012` |  |
| 413-451 | method | `GeometryFrontendSession._make_naca4` |  |
| 453-464 | method | `GeometryFrontendSession._make_rae2822` |  |
| 466-473 | method | `GeometryFrontendSession._clean_name` |  |
| 475-488 | method | `GeometryFrontendSession._resolve_path` |  |
| 490-499 | method | `GeometryFrontendSession._load_yaml_file` |  |
| 501-506 | method | `GeometryFrontendSession._numpy` |  |

**ui/main_window.py**

Primary stage: [27. UI: приложение, обучение, процессы и результаты](REPOSITORY_STUDY_PLAN.md#stage-27).
Source: [open file](<../ui/main_window.py>). Lines: 1998. SHA-256: `e0ee94fd304eefaa70f4475d907d36f0f6d25dd3418cde87ae061650214f57d3`.

Module body, imports, exports and class attributes: planned for line-by-line review.

| Lines | Kind | Qualified name | Decorators |
|---|---|---|---|
| 64-1998 | class | `MainWindow` |  |
| 78-125 | method | `MainWindow.__init__` |  |
| 127-131 | method | `MainWindow.dragEnterEvent` |  |
| 133-137 | method | `MainWindow.dropEvent` |  |
| 139-148 | method | `MainWindow.eventFilter` |  |
| 150-172 | method | `MainWindow.closeEvent` |  |
| 174-220 | method | `MainWindow._create_actions` |  |
| 222-234 | method | `MainWindow._create_menu` |  |
| 236-249 | method | `MainWindow._create_toolbar` |  |
| 251-288 | method | `MainWindow._build_layout` |  |
| 290-324 | method | `MainWindow._build_module_nav` |  |
| 319-319 | lambda | `MainWindow._build_module_nav.<lambda@319:35>` |  |
| 326-331 | method | `MainWindow._set_active_module` |  |
| 333-341 | method | `MainWindow._build_geometry_center` |  |
| 343-354 | method | `MainWindow._new_side_panel` |  |
| 356-413 | method | `MainWindow._build_geometry_left_panel` |  |
| 415-445 | method | `MainWindow._build_geometry_right_panel` |  |
| 447-460 | method | `MainWindow._build_form_side_panel` |  |
| 462-473 | method | `MainWindow._build_status_side_panel` |  |
| 475-484 | method | `MainWindow._build_collocation_left_panel` |  |
| 486-490 | method | `MainWindow._build_collocation_right_panel` |  |
| 492-501 | method | `MainWindow._build_fluid_left_panel` |  |
| 503-504 | method | `MainWindow._build_fluid_right_panel` |  |
| 506-516 | method | `MainWindow._build_boundary_left_panel` |  |
| 518-519 | method | `MainWindow._build_boundary_right_panel` |  |
| 521-531 | method | `MainWindow._build_initialization_left_panel` |  |
| 533-534 | method | `MainWindow._build_initialization_right_panel` |  |
| 536-625 | method | `MainWindow._build_training_left_panel` |  |
| 627-670 | method | `MainWindow._build_training_right_panel` |  |
| 672-706 | method | `MainWindow._build_results_left_panel` |  |
| 708-717 | method | `MainWindow._build_results_right_panel` |  |
| 719-783 | method | `MainWindow._build_geometry_header` |  |
| 785-795 | method | `MainWindow._metric_card` |  |
| 797-800 | method | `MainWindow._metric_label` |  |
| 802-815 | method | `MainWindow._double_spin` |  |
| 817-822 | method | `MainWindow._spin` |  |
| 824-827 | method | `MainWindow._combo` |  |
| 829-875 | method | `MainWindow._connect_signals` |  |
| 877-888 | method | `MainWindow._apply_widget_hints` |  |
| 890-938 | method | `MainWindow._restore_ui_settings` |  |
| 940-974 | method | `MainWindow._save_ui_settings` |  |
| 976-980 | method | `MainWindow._settings_int` |  |
| 982-986 | method | `MainWindow._settings_float` |  |
| 988-994 | method | `MainWindow._settings_bool` |  |
| 996-1002 | method | `MainWindow._setup_compact_table` |  |
| 1004-1008 | method | `MainWindow._wrap_group` |  |
| 1010-1017 | method | `MainWindow._refresh_foil_files` |  |
| 1019-1030 | method | `MainWindow._refresh_library_names` |  |
| 1032-1051 | method | `MainWindow._update_source_controls` |  |
| 1053-1056 | method | `MainWindow._mime_has_geometry_payload` |  |
| 1058-1072 | method | `MainWindow._handle_geometry_drop` |  |
| 1074-1081 | method | `MainWindow._first_geometry_file_from_urls` |  |
| 1083-1100 | method | `MainWindow._looks_like_coordinate_text` |  |
| 1102-1138 | method | `MainWindow._load_geometry_source` |  |
| 1140-1163 | method | `MainWindow._open_manual_coordinates_dialog` |  |
| 1165-1177 | method | `MainWindow._load_manual_coordinates` |  |
| 1179-1187 | method | `MainWindow._browse_geometry_file` |  |
| 1189-1192 | method | `MainWindow._select_external_geometry_path` |  |
| 1194-1203 | method | `MainWindow._load_geometry_file_path` |  |
| 1205-1211 | method | `MainWindow._create_domain_from_controls` |  |
| 1213-1218 | method | `MainWindow._apply_domain_params_to_controls` |  |
| 1220-1224 | method | `MainWindow._apply_save_format_to_controls` |  |
| 1226-1249 | method | `MainWindow._sync_geometry_ui` |  |
| 1251-1270 | method | `MainWindow._update_stats_table` |  |
| 1272-1289 | method | `MainWindow._update_vertices_table` |  |
| 1291-1303 | method | `MainWindow._export_geometry` |  |
| 1305-1312 | method | `MainWindow._short_path` |  |
| 1314-1315 | method | `MainWindow._update_cursor_label` |  |
| 1317-1329 | method | `MainWindow._handle_canvas_geometry_edited` |  |
| 1331-1338 | method | `MainWindow._rebuild_geometry` |  |
| 1340-1342 | method | `MainWindow._sample_points` |  |
| 1344-1348 | method | `MainWindow._mock_open` |  |
| 1350-1351 | method | `MainWindow._mock_save` |  |
| 1353-1360 | method | `MainWindow._show_about` |  |
| 1362-1407 | method | `MainWindow._start_training` |  |
| 1409-1416 | method | `MainWindow._stop_training` |  |
| 1418-1430 | method | `MainWindow._reset_training_monitor` |  |
| 1432-1439 | method | `MainWindow._set_training_running` |  |
| 1441-1466 | method | `MainWindow._set_training_inputs_enabled` |  |
| 1468-1471 | method | `MainWindow._handle_backend_log` |  |
| 1473-1474 | method | `MainWindow._handle_backend_progress` |  |
| 1476-1478 | method | `MainWindow._handle_backend_metrics` |  |
| 1480-1485 | method | `MainWindow._flush_backend_logs` |  |
| 1487-1492 | method | `MainWindow._flush_backend_metrics` |  |
| 1494-1513 | method | `MainWindow._apply_backend_metrics` |  |
| 1515-1517 | method | `MainWindow._handle_backend_error` |  |
| 1519-1533 | method | `MainWindow._handle_training_finished` |  |
| 1535-1537 | method | `MainWindow._handle_training_thread_finished` |  |
| 1539-1570 | method | `MainWindow._collect_training_config` |  |
| 1572-1598 | method | `MainWindow._apply_training_config` |  |
| 1600-1635 | method | `MainWindow._build_backend_command` |  |
| 1637-1695 | method | `MainWindow._write_backend_yaml_config` |  |
| 1697-1716 | method | `MainWindow._validate_backend_data_paths` |  |
| 1718-1722 | method | `MainWindow._resolve_backend_path` |  |
| 1724-1740 | method | `MainWindow._load_base_backend_config` |  |
| 1742-1753 | method | `MainWindow._save_training_config` |  |
| 1755-1771 | method | `MainWindow._load_training_config` |  |
| 1773-1775 | method | `MainWindow._write_json` |  |
| 1777-1805 | method | `MainWindow._refresh_results` |  |
| 1794-1796 | lambda | `MainWindow._refresh_results.<lambda@1794:12>` |  |
| 1802-1802 | lambda | `MainWindow._refresh_results.<lambda@1802:32>` |  |
| 1807-1833 | method | `MainWindow._handle_results_scan_finished` |  |
| 1835-1837 | method | `MainWindow._handle_results_scan_error` |  |
| 1839-1842 | method | `MainWindow._forget_results_scan` |  |
| 1844-1849 | method | `MainWindow._show_selected_result_image` |  |
| 1851-1856 | method | `MainWindow._open_results_folder` |  |
| 1858-1864 | method | `MainWindow._choose_results_folder` |  |
| 1866-1877 | method | `MainWindow._browse_training_profile_file` |  |
| 1879-1887 | method | `MainWindow._browse_backend_script` |  |
| 1889-1892 | method | `MainWindow._browse_results_dir` |  |
| 1894-1898 | method | `MainWindow._sync_backend_controls` |  |
| 1900-1904 | method | `MainWindow._sync_training_profile_controls` |  |
| 1906-1922 | method | `MainWindow._load_selected_preset_geometry` |  |
| 1924-1928 | method | `MainWindow._available_backend_configs` |  |
| 1930-1953 | method | `MainWindow._update_final_metrics_table` |  |
| 1955-1960 | method | `MainWindow._set_combo_text` |  |
| 1962-1969 | method | `MainWindow._set_combo_data` |  |
| 1971-1972 | method | `MainWindow._short_project_path` |  |
| 1974-1978 | method | `MainWindow._short_path_from` |  |
| 1980-1989 | method | `MainWindow._format_metric` |  |
| 1991-1998 | method | `MainWindow._sanitize_name` |  |

**ui/mock_backend.py**

Primary stage: [27. UI: приложение, обучение, процессы и результаты](REPOSITORY_STUDY_PLAN.md#stage-27).
Source: [open file](<../ui/mock_backend.py>). Lines: 155. SHA-256: `733a6d22b844de4b0e6f9a99621bd6021e3853924c44dcbba149c408381f7e2a`.

Module body, imports, exports and class attributes: planned for line-by-line review.

| Lines | Kind | Qualified name | Decorators |
|---|---|---|---|
| 14-20 | function | `parse_args` |  |
| 23-76 | function | `save_mock_plots` |  |
| 79-115 | function | `_write_simple_png` |  |
| 80-91 | nested function | `_write_simple_png.pixel` |  |
| 101-107 | nested function | `_write_simple_png.chunk` |  |
| 118-151 | function | `main` |  |

**ui/mock_data.py**

Primary stage: [26. UI: геометрия, редактор, виджеты и демонстрационные элементы](REPOSITORY_STUDY_PLAN.md#stage-26).
Source: [open file](<../ui/mock_data.py>). Lines: 106. SHA-256: `d72a852a5925bb2af7e361bec5f150a6f802909e12d65d7f4120bc1dcbfbd7a6`.

Module body, imports, exports and class attributes: planned for line-by-line review.

| Lines | Kind | Qualified name | Decorators |
|---|---|---|---|
| 14-22 | class | `Domain` | dataclass(frozen=True) |
| 21-22 | method | `Domain.rect` | property |
| 26-29 | class | `CollocationPoint` | dataclass(frozen=True) |
| 32-53 | function | `naca_0012_points` |  |
| 56-65 | function | `rotate_points` |  |
| 68-100 | function | `generate_collocation_points` |  |
| 103-106 | function | `next_metric_value` |  |

**ui/smoke_tests.py**

Primary stage: [28. Все тестовые сценарии и подтверждение поведения](REPOSITORY_STUDY_PLAN.md#stage-28).
Source: [open file](<../ui/smoke_tests.py>). Lines: 109. SHA-256: `820fb7cb42048a0a6120c5f32aea98f01f78e26343bc501044b9df00b82c5d10`.

Module body, imports, exports and class attributes: planned for line-by-line review.

| Lines | Kind | Qualified name | Decorators |
|---|---|---|---|
| 25-38 | function | `test_worker_metric_parser` |  |
| 28-28 | lambda | `test_worker_metric_parser.<lambda@28:35>` |  |
| 41-54 | function | `test_geometry_save` |  |
| 57-96 | function | `test_main_window_mock_backend` |  |
| 70-74 | nested function | `test_main_window_mock_backend.poll_done` |  |
| 76-80 | nested function | `test_main_window_mock_backend.timeout` |  |
| 99-105 | function | `main` |  |

**ui/theme.py**

Primary stage: [26. UI: геометрия, редактор, виджеты и демонстрационные элементы](REPOSITORY_STUDY_PLAN.md#stage-26).
Source: [open file](<../ui/theme.py>). Lines: 191. SHA-256: `1f674b38461065a10c11b2441b11803c4ce1e57b074f850651326454d3f0e6b0`.

Module body, imports, exports and class attributes: planned for line-by-line review.

| Lines | Kind | Qualified name | Decorators |
|---|---|---|---|
| 9-25 | class | `Palette` | dataclass(frozen=True) |
| 31-191 | function | `app_stylesheet` |  |

**ui/widgets/__init__.py**

Primary stage: [3. Служебные типы, импорты и общие утилиты](REPOSITORY_STUDY_PLAN.md#stage-3).
Source: [open file](<../ui/widgets/__init__.py>). Lines: 1. SHA-256: `25245065cee4b3eb835f469e85f70c86e35663f085ecd8302ca04de96bee8bc3`.

Module body, imports, exports and class attributes: planned for line-by-line review.

| Lines | Kind | Qualified name | Decorators |
|---|---|---|---|
| whole file | module | no local definitions; review imports, exports and module body | |

**ui/widgets/geometry_canvas.py**

Primary stage: [26. UI: геометрия, редактор, виджеты и демонстрационные элементы](REPOSITORY_STUDY_PLAN.md#stage-26).
Source: [open file](<../ui/widgets/geometry_canvas.py>). Lines: 411. SHA-256: `629ee173096a73eabd98c9845361e210830dae1246d84f2f0bce06194602d52c`.

Module body, imports, exports and class attributes: planned for line-by-line review.

| Lines | Kind | Qualified name | Decorators |
|---|---|---|---|
| 16-411 | class | `GeometryCanvas` |  |
| 23-51 | method | `GeometryCanvas.__init__` |  |
| 53-62 | method | `GeometryCanvas.rebuild_geometry` |  |
| 64-78 | method | `GeometryCanvas.set_geometry` |  |
| 80-82 | method | `GeometryCanvas.regenerate_points` |  |
| 84-87 | method | `GeometryCanvas.reset_view` |  |
| 89-91 | method | `GeometryCanvas.set_points_visible` |  |
| 93-96 | method | `GeometryCanvas.set_streamlines_visible` |  |
| 98-100 | method | `GeometryCanvas.set_grid_visible` |  |
| 102-118 | method | `GeometryCanvas.paintEvent` |  |
| 120-123 | method | `GeometryCanvas.resizeEvent` |  |
| 125-127 | method | `GeometryCanvas.showEvent` |  |
| 129-131 | method | `GeometryCanvas.hideEvent` |  |
| 133-142 | method | `GeometryCanvas.mousePressEvent` |  |
| 144-159 | method | `GeometryCanvas.mouseMoveEvent` |  |
| 161-168 | method | `GeometryCanvas.mouseReleaseEvent` |  |
| 170-177 | method | `GeometryCanvas.wheelEvent` |  |
| 179-182 | method | `GeometryCanvas._tick` |  |
| 184-188 | method | `GeometryCanvas._fit_scale` |  |
| 190-196 | method | `GeometryCanvas._world_to_screen` |  |
| 198-204 | method | `GeometryCanvas._screen_to_world` |  |
| 206-223 | method | `GeometryCanvas._draw_grid` |  |
| 225-249 | method | `GeometryCanvas._draw_domain` |  |
| 251-280 | method | `GeometryCanvas._draw_streamlines` |  |
| 282-302 | method | `GeometryCanvas._draw_points` |  |
| 304-321 | method | `GeometryCanvas._draw_airfoil` |  |
| 323-333 | method | `GeometryCanvas._draw_overlay` |  |
| 335-344 | method | `GeometryCanvas._nearest_vertex` |  |
| 346-352 | method | `GeometryCanvas._set_foil_vertex` |  |
| 354-362 | method | `GeometryCanvas._domain_polygon_from_rect` |  |
| 364-376 | method | `GeometryCanvas._domain_from_points` |  |
| 378-379 | method | `GeometryCanvas._view_rect` |  |
| 381-395 | method | `GeometryCanvas._compute_view_rect` |  |
| 397-402 | method | `GeometryCanvas._sync_animation_timer` |  |
| 404-411 | method | `GeometryCanvas._decimated_points` |  |

**ui/widgets/log_panel.py**

Primary stage: [26. UI: геометрия, редактор, виджеты и демонстрационные элементы](REPOSITORY_STUDY_PLAN.md#stage-26).
Source: [open file](<../ui/widgets/log_panel.py>). Lines: 27. SHA-256: `842adf0d2390fd656e40199d262b3bb0bf122c2abf5a203940b99335c3d63654`.

Module body, imports, exports and class attributes: planned for line-by-line review.

| Lines | Kind | Qualified name | Decorators |
|---|---|---|---|
| 11-27 | class | `LogPanel` |  |
| 12-15 | method | `LogPanel.__init__` |  |
| 17-19 | method | `LogPanel.append_event` |  |
| 21-22 | method | `LogPanel.append_raw` |  |
| 24-27 | method | `LogPanel.append_raw_lines` |  |

**ui/widgets/metric_plot.py**

Primary stage: [26. UI: геометрия, редактор, виджеты и демонстрационные элементы](REPOSITORY_STUDY_PLAN.md#stage-26).
Source: [open file](<../ui/widgets/metric_plot.py>). Lines: 78. SHA-256: `3d5be401eb1ab4ed40077b9e5fe43cd3341c6b7344f5570627bf9525ce479f2a`.

Module body, imports, exports and class attributes: planned for line-by-line review.

| Lines | Kind | Qualified name | Decorators |
|---|---|---|---|
| 15-77 | class | `MetricPlot` |  |
| 18-23 | method | `MetricPlot.__init__` |  |
| 25-27 | method | `MetricPlot.set_values` |  |
| 29-33 | method | `MetricPlot.append_value` |  |
| 35-77 | method | `MetricPlot.paintEvent` |  |

**ui/widgets/result_canvas.py**

Primary stage: [26. UI: геометрия, редактор, виджеты и демонстрационные элементы](REPOSITORY_STUDY_PLAN.md#stage-26).
Source: [open file](<../ui/widgets/result_canvas.py>). Lines: 72. SHA-256: `954bb7cd1758661e6966a6a344177c942a02b66d7d95a76f75b4fa9091922fb1`.

Module body, imports, exports and class attributes: planned for line-by-line review.

| Lines | Kind | Qualified name | Decorators |
|---|---|---|---|
| 13-72 | class | `ResultImageCanvas` |  |
| 16-25 | method | `ResultImageCanvas.__init__` |  |
| 27-32 | method | `ResultImageCanvas.show_message` |  |
| 34-72 | method | `ResultImageCanvas.load_image` |  |

**ui/workers.py**

Primary stage: [27. UI: приложение, обучение, процессы и результаты](REPOSITORY_STUDY_PLAN.md#stage-27).
Source: [open file](<../ui/workers.py>). Lines: 343. SHA-256: `c2cab69efbfd3e6250fc688e27b24287341f3f5fae745cfb0204929a2d4c7fea`.

Module body, imports, exports and class attributes: planned for line-by-line review.

| Lines | Kind | Qualified name | Decorators |
|---|---|---|---|
| 18-296 | class | `BackendWorker` |  |
| 38-54 | method | `BackendWorker.__init__` |  |
| 57-129 | method | `BackendWorker.run` | Slot() |
| 132-146 | method | `BackendWorker.request_stop` | Slot() |
| 148-159 | method | `BackendWorker._terminate_process` |  |
| 161-179 | method | `BackendWorker._read_stream` |  |
| 181-229 | method | `BackendWorker._parse_metrics` |  |
| 232-239 | method | `BackendWorker._extract_float` | staticmethod |
| 242-254 | method | `BackendWorker._extract_table_metric` | staticmethod |
| 256-258 | method | `BackendWorker._emit_failure` |  |
| 260-262 | method | `BackendWorker._is_stop_requested` |  |
| 265-268 | method | `BackendWorker._format_command` | staticmethod |
| 271-280 | method | `BackendWorker._terminate_windows_process_tree` | staticmethod |
| 283-296 | method | `BackendWorker._terminate_posix_process_group` | staticmethod |
| 299-343 | class | `ResultsScanWorker` |  |
| 305-307 | method | `ResultsScanWorker.__init__` |  |
| 310-318 | method | `ResultsScanWorker.run` | Slot() |
| 321-325 | method | `ResultsScanWorker._result_image_paths` | staticmethod |
| 328-343 | method | `ResultsScanWorker._read_final_report` | staticmethod |

**tests/boundary_layer_diagnostics_smoke.py**

Primary stage: [28. Все тестовые сценарии и подтверждение поведения](REPOSITORY_STUDY_PLAN.md#stage-28).
Source: [open file](<../tests/boundary_layer_diagnostics_smoke.py>). Lines: 135. SHA-256: `7984a8fe17efdb98e85e2b96bf1a7ddb0eebfbee7f137a25fd1a48cec971a7d9`.

Module body, imports, exports and class attributes: planned for line-by-line review.

| Lines | Kind | Qualified name | Decorators |
|---|---|---|---|
| 18-29 | class | `DummyBoundaryLayerModel` |  |
| 21-29 | method | `DummyBoundaryLayerModel.forward` |  |
| 32-63 | class | `DummyLossComputer` |  |
| 36-43 | method | `DummyLossComputer.evaluate_sdf_tensor` | staticmethod |
| 45-53 | method | `DummyLossComputer.build_input_tensor` |  |
| 55-63 | method | `DummyLossComputer.apply_velocity_dirichlet_constraint` |  |
| 66-89 | class | `DummyDataHandler` |  |
| 69-89 | method | `DummyDataHandler.__init__` |  |
| 92-130 | function | `test_boundary_layer_diagnostics_outputs_files` |  |

**tests/boundary_normal_bc_smoke.py**

Primary stage: [28. Все тестовые сценарии и подтверждение поведения](REPOSITORY_STUDY_PLAN.md#stage-28).
Source: [open file](<../tests/boundary_normal_bc_smoke.py>). Lines: 358. SHA-256: `44988f13761750eb8b1e53da5fff75cc35d2d82f68f562138c8e29f30f8886ba`.

Module body, imports, exports and class attributes: planned for line-by-line review.

| Lines | Kind | Qualified name | Decorators |
|---|---|---|---|
| 24-89 | function | `_loss_computer` |  |
| 92-99 | function | `_rotation` |  |
| 102-171 | class | `SlipWallNormalTests` |  |
| 103-133 | method | `SlipWallNormalTests.test_horizontal_vertical_and_inclined_boundaries` |  |
| 135-159 | method | `SlipWallNormalTests.test_curved_boundary_uses_pointwise_normals` |  |
| 161-171 | method | `SlipWallNormalTests.test_missing_normals_fail_instead_of_guessing_a_side` |  |
| 174-285 | class | `SymmetryNormalTests` |  |
| 176-182 | method | `SymmetryNormalTests._linear_velocity` | staticmethod |
| 184-206 | method | `SymmetryNormalTests._evaluate` |  |
| 208-256 | method | `SymmetryNormalTests.test_symmetry_enforces_normal_velocity_and_tangent_normal_gradient` |  |
| 258-285 | method | `SymmetryNormalTests.test_boundary_loss_is_invariant_under_rotation` |  |
| 288-354 | class | `BoundaryNormalDataTests` |  |
| 289-296 | method | `BoundaryNormalDataTests.test_default_config_separates_boundary_physics` |  |
| 298-319 | method | `BoundaryNormalDataTests.test_sampler_computes_unit_normals_for_curved_polyline` |  |
| 321-354 | method | `BoundaryNormalDataTests.test_dataset_preserves_boundary_normals_in_named_batch` |  |

**tests/config_contract_smoke.py**

Primary stage: [28. Все тестовые сценарии и подтверждение поведения](REPOSITORY_STUDY_PLAN.md#stage-28).
Source: [open file](<../tests/config_contract_smoke.py>). Lines: 312. SHA-256: `56e47bd6b47d3008910a2dc561a39bc56c78ba3ee3891c56279111d9e1cc43d2`.

Module body, imports, exports and class attributes: planned for line-by-line review.

| Lines | Kind | Qualified name | Decorators |
|---|---|---|---|
| 30-32 | function | `_default_config` |  |
| 35-43 | function | `_reference_scales` |  |
| 46-115 | class | `StaticValidationTests` |  |
| 49-52 | method | `StaticValidationTests.test_default_configuration_is_valid` |  |
| 54-69 | method | `StaticValidationTests.test_legacy_boundary_feature_flags_are_removed_on_load` |  |
| 71-75 | method | `StaticValidationTests.test_mandatory_sdf_cannot_be_disabled` |  |
| 77-86 | method | `StaticValidationTests.test_invalid_reynolds_range_is_rejected` |  |
| 88-97 | method | `StaticValidationTests.test_duplicate_geometry_names_are_rejected` |  |
| 99-103 | method | `StaticValidationTests.test_checkpoint_load_mode_is_validated` |  |
| 105-109 | method | `StaticValidationTests.test_raw_pressure_output_bias_must_be_zero` |  |
| 111-115 | method | `StaticValidationTests.test_pressure_gauge_cannot_be_disabled` |  |
| 118-191 | class | `RuntimeValidationTests` |  |
| 121-142 | method | `RuntimeValidationTests.test_multiple_angles_require_angle_features` |  |
| 144-167 | method | `RuntimeValidationTests.test_complete_parametric_manifest_is_valid` |  |
| 169-191 | method | `RuntimeValidationTests.test_case_reynolds_must_be_inside_feature_range` |  |
| 194-228 | function | `_training_contract` |  |
| 231-308 | class | `CheckpointContractTests` |  |
| 234-255 | method | `CheckpointContractTests.test_contract_roundtrip_and_mismatch` |  |
| 257-261 | method | `CheckpointContractTests.test_corrupted_contract_is_rejected` |  |
| 263-277 | method | `CheckpointContractTests.test_boundary_transform_changes_checkpoint_fingerprint` |  |
| 279-308 | method | `CheckpointContractTests.test_checkpoint_save_is_atomic_and_versioned` |  |

**tests/differentiable_sdf_smoke.py**

Primary stage: [28. Все тестовые сценарии и подтверждение поведения](REPOSITORY_STUDY_PLAN.md#stage-28).
Source: [open file](<../tests/differentiable_sdf_smoke.py>). Lines: 465. SHA-256: `b0b7075d3a6876f2966c4f1e36131c9ac38b07d90d35bcf05b36ccf67607e1d1`.

Module body, imports, exports and class attributes: planned for line-by-line review.

| Lines | Kind | Qualified name | Decorators |
|---|---|---|---|
| 20-34 | function | `_real_airfoil_contour` |  |
| 37-461 | class | `PolylineDistanceFieldTests` |  |
| 40-51 | method | `PolylineDistanceFieldTests.test_distance_is_to_segment_not_nearest_vertex` |  |
| 53-69 | method | `PolylineDistanceFieldTests.test_numpy_cache_matches_online_engine` |  |
| 71-97 | method | `PolylineDistanceFieldTests.test_real_airfoil_training_and_inference_float64_match` |  |
| 99-150 | method | `PolylineDistanceFieldTests.test_stale_cached_sdf_and_fingerprint_are_rebuilt` |  |
| 152-174 | method | `PolylineDistanceFieldTests.test_rar_uses_canonical_segment_distance` |  |
| 176-220 | method | `PolylineDistanceFieldTests.test_line_gradient_and_hessian` |  |
| 222-251 | method | `PolylineDistanceFieldTests.test_autograd_gradcheck` |  |
| 253-273 | method | `PolylineDistanceFieldTests.test_manufactured_chain_rule` |  |
| 275-313 | method | `PolylineDistanceFieldTests.test_boundary_cache_avoids_zero_distance_derivative` |  |
| 315-334 | method | `PolylineDistanceFieldTests.test_tie_detection_and_eikonal_diagnostics` |  |
| 336-416 | method | `PolylineDistanceFieldTests.test_real_airfoil_ties_preserve_near_wall_pde_coverage` |  |
| 360-365 | nested function | `PolylineDistanceFieldTests.test_real_airfoil_ties_preserve_near_wall_pde_coverage.outward_normals` |  |
| 418-461 | method | `PolylineDistanceFieldTests.test_reduced_tolerances_preserve_real_fvm_mesh_coverage` |  |

**tests/end_to_end_pipeline_smoke.py**

Primary stage: [28. Все тестовые сценарии и подтверждение поведения](REPOSITORY_STUDY_PLAN.md#stage-28).
Source: [open file](<../tests/end_to_end_pipeline_smoke.py>). Lines: 261. SHA-256: `3f20cc9baed85f865b82176f9b24cd3c9256cbec45f1b6724138ee720d5142dd`.

Module body, imports, exports and class attributes: planned for line-by-line review.

| Lines | Kind | Qualified name | Decorators |
|---|---|---|---|
| 19-45 | function | `_small_solver_config` |  |
| 48-257 | class | `EndToEndResumeTests` |  |
| 51-72 | method | `EndToEndResumeTests.test_raw_pressure_shift_changes_total_loss` |  |
| 74-178 | method | `EndToEndResumeTests.test_checkpoint_restores_model_optimizer_data_and_rng` |  |
| 180-226 | method | `EndToEndResumeTests.test_hard_wall_trains_and_generalizes_to_new_surface_points` |  |
| 228-257 | method | `EndToEndResumeTests.test_boundary_transform_change_requires_finetune_mode` |  |

**tests/euler_wall_bc_smoke.py**

Primary stage: [28. Все тестовые сценарии и подтверждение поведения](REPOSITORY_STUDY_PLAN.md#stage-28).
Source: [open file](<../tests/euler_wall_bc_smoke.py>). Lines: 211. SHA-256: `f105dba13fb631bfd5c6ef4e322240d39b431b38c7b4d2e0627799666d6878ed`.

Module body, imports, exports and class attributes: planned for line-by-line review.

| Lines | Kind | Qualified name | Decorators |
|---|---|---|---|
| 25-29 | function | `_loss_config` |  |
| 32-65 | function | `_wall_loss_computer` |  |
| 68-154 | class | `EulerWallLossTests` |  |
| 71-110 | method | `EulerWallLossTests.test_euler_leaves_tangential_velocity_unconstrained` |  |
| 112-132 | method | `EulerWallLossTests.test_navier_stokes_retains_no_slip_wall` |  |
| 134-154 | method | `EulerWallLossTests.test_surface_normal_is_sdf_segment_normal` |  |
| 157-207 | class | `EulerHardBoundaryTests` |  |
| 161-175 | method | `EulerHardBoundaryTests._write_points` | staticmethod |
| 177-184 | method | `EulerHardBoundaryTests.test_euler_wall_does_not_fix_velocity_components` |  |
| 186-196 | method | `EulerHardBoundaryTests.test_navier_stokes_wall_still_fixes_both_components` |  |
| 198-207 | method | `EulerHardBoundaryTests.test_euler_hard_bc_configuration_is_rejected` |  |

**tests/fvm_config.py**

Primary stage: [28. Все тестовые сценарии и подтверждение поведения](REPOSITORY_STUDY_PLAN.md#stage-28).
Source: [open file](<../tests/fvm_config.py>). Lines: 118. SHA-256: `0546212afe97f2038a617ee775c3c7002188c908bd50101d6fca62d406111fa1`.

Module body, imports, exports and class attributes: planned for line-by-line review.

| Lines | Kind | Qualified name | Decorators |
|---|---|---|---|
| 50-63 | function | `make_lw` |  |
| 68-71 | class | `_FakeDataHandler` |  |
| 74-90 | function | `_make_lc` |  |

**tests/geometry_normalization_smoke.py**

Primary stage: [28. Все тестовые сценарии и подтверждение поведения](REPOSITORY_STUDY_PLAN.md#stage-28).
Source: [open file](<../tests/geometry_normalization_smoke.py>). Lines: 210. SHA-256: `9c377c4353cf5ed9e86d948c7db323dcf9204948fe1858dcee4c09fd1cfc0d4e`.

Module body, imports, exports and class attributes: planned for line-by-line review.

| Lines | Kind | Qualified name | Decorators |
|---|---|---|---|
| 24-206 | class | `GeometryNormalizationTests` |  |
| 27-34 | method | `GeometryNormalizationTests.test_geometry_reference_round_trip` |  |
| 36-46 | method | `GeometryNormalizationTests.test_initializer_prefers_matching_sampler_metadata` |  |
| 48-83 | method | `GeometryNormalizationTests.test_initializer_loads_only_sampler_linked_geometry` |  |
| 85-106 | method | `GeometryNormalizationTests.test_initializer_reconstructs_domain_from_sampler_bounds` |  |
| 108-117 | method | `GeometryNormalizationTests.test_sampler_requires_explicit_geometry_name` |  |
| 119-133 | method | `GeometryNormalizationTests.test_geometry2d_loads_an_explicit_file` |  |
| 135-148 | method | `GeometryNormalizationTests.test_delta_reference_uses_one_model` |  |
| 150-164 | method | `GeometryNormalizationTests.test_data_handler_repairs_legacy_scales` |  |
| 167-187 | method | `GeometryNormalizationTests._write_legacy_initial` | staticmethod |
| 190-206 | method | `GeometryNormalizationTests._write_sampler` | staticmethod |

**tests/hard_boundary_transform_smoke.py**

Primary stage: [28. Все тестовые сценарии и подтверждение поведения](REPOSITORY_STUDY_PLAN.md#stage-28).
Source: [open file](<../tests/hard_boundary_transform_smoke.py>). Lines: 318. SHA-256: `607d3a949ad62d2ec65147435837fb9c46a96851b53be5e0b9e63aad3aa420b5`.

Module body, imports, exports and class attributes: planned for line-by-line review.

| Lines | Kind | Qualified name | Decorators |
|---|---|---|---|
| 25-29 | function | `_line_field` |  |
| 32-42 | function | `_hard_transform` |  |
| 45-218 | class | `BoundaryOutputTransformTests` |  |
| 48-60 | method | `BoundaryOutputTransformTests.test_unsampled_wall_points_ignore_raw_velocity` |  |
| 62-74 | method | `BoundaryOutputTransformTests.test_constraint_needs_no_boundary_labels_or_dataset_masks` |  |
| 76-96 | method | `BoundaryOutputTransformTests.test_near_wall_trace_tends_continuously_to_zero` |  |
| 98-143 | method | `BoundaryOutputTransformTests.test_manufactured_first_and_second_derivatives` |  |
| 145-199 | method | `BoundaryOutputTransformTests.test_piecewise_sdf_derivatives_are_finite_away_from_ties` |  |
| 201-214 | method | `BoundaryOutputTransformTests.test_soft_transform_is_identity` |  |
| 216-218 | method | `BoundaryOutputTransformTests.test_distance_scale_must_be_positive` |  |
| 221-248 | class | `BoundaryConfigurationTests` |  |
| 224-227 | method | `BoundaryConfigurationTests.test_supported_hard_wall_configuration_is_valid` |  |
| 229-239 | method | `BoundaryConfigurationTests.test_euler_and_non_no_slip_hard_wall_are_rejected` |  |
| 241-248 | method | `BoundaryConfigurationTests.test_global_hard_configuration_is_rejected` |  |
| 251-314 | class | `HybridBoundaryLossTests` |  |
| 254-314 | method | `HybridBoundaryLossTests.test_hard_wall_is_skipped_but_outlet_pressure_remains_soft` |  |
| 262-267 | nested function | `HybridBoundaryLossTests.test_hard_wall_is_skipped_but_outlet_pressure_remains_soft.get_flow_case_batch` |  |

**tests/nondimensionalization_smoke.py**

Primary stage: [28. Все тестовые сценарии и подтверждение поведения](REPOSITORY_STUDY_PLAN.md#stage-28).
Source: [open file](<../tests/nondimensionalization_smoke.py>). Lines: 160. SHA-256: `fd517c099172eddfe4bed7cbf8e4087490847f25dcff665590a826464978c4d4`.

Module body, imports, exports and class attributes: planned for line-by-line review.

| Lines | Kind | Qualified name | Decorators |
|---|---|---|---|
| 16-156 | class | `ReferenceScalesTests` |  |
| 19-27 | method | `ReferenceScalesTests.setUp` |  |
| 29-58 | method | `ReferenceScalesTests.test_dimensional_round_trips` |  |
| 60-72 | method | `ReferenceScalesTests.test_pressure_coefficient_reference` |  |
| 74-84 | method | `ReferenceScalesTests.test_versioned_metadata_round_trip` |  |
| 86-100 | method | `ReferenceScalesTests.test_legacy_metadata_migration` |  |
| 102-137 | method | `ReferenceScalesTests.test_buckingham_pi_invariance` |  |
| 139-156 | method | `ReferenceScalesTests.test_invalid_reference_values_fail_fast` |  |

**tests/normal_layer_smoke.py**

Primary stage: [28. Все тестовые сценарии и подтверждение поведения](REPOSITORY_STUDY_PLAN.md#stage-28).
Source: [open file](<../tests/normal_layer_smoke.py>). Lines: 78. SHA-256: `2f673e418c235972dd159cf5ba21877d7e958b2cebb57a1c2668267b4411ca0f`.

Module body, imports, exports and class attributes: planned for line-by-line review.

| Lines | Kind | Qualified name | Decorators |
|---|---|---|---|
| 18-28 | function | `_rect_domain` |  |
| 31-33 | function | `_closed_ellipse` |  |
| 36-38 | function | `_open_plate` |  |
| 41-54 | function | `test_closed_profile_points_are_in_flow` |  |
| 57-72 | function | `test_open_profile_uses_both_valid_sides` |  |

**tests/parametric_fvm_smoke.py**

Primary stage: [28. Все тестовые сценарии и подтверждение поведения](REPOSITORY_STUDY_PLAN.md#stage-28).
Source: [open file](<../tests/parametric_fvm_smoke.py>). Lines: 215. SHA-256: `61b7f8cb33caa0309214d2003d5cb6d65d5ef21792c602bdb93b7a5e1ff5288c`.

Module body, imports, exports and class attributes: planned for line-by-line review.

| Lines | Kind | Qualified name | Decorators |
|---|---|---|---|
| 16-29 | function | `_one_cell_mesh` |  |
| 32-211 | class | `ParametricFVMTests` |  |
| 35-41 | method | `ParametricFVMTests.setUp` |  |
| 43-78 | method | `ParametricFVMTests.test_case_local_boundary_does_not_mutate_operator` |  |
| 80-101 | method | `ParametricFVMTests.test_fvm_boundary_remains_differentiable_in_alpha` |  |
| 103-112 | method | `ParametricFVMTests.test_invalid_boundary_shape_is_rejected` |  |
| 114-135 | method | `ParametricFVMTests.test_geometry_normalization_survives_rebuild` |  |
| 137-211 | method | `ParametricFVMTests.test_trainer_evaluates_every_flow_case_separately` |  |
| 140-145 | class | `ParametricFVMTests.test_trainer_evaluates_every_flow_case_separately.CaseModel` |  |
| 143-145 | method | `ParametricFVMTests.test_trainer_evaluates_every_flow_case_separately.CaseModel.forward` |  |
| 147-181 | class | `ParametricFVMTests.test_trainer_evaluates_every_flow_case_separately.CaseLoss` |  |
| 153-161 | method | `ParametricFVMTests.test_trainer_evaluates_every_flow_case_separately.CaseLoss.build_input_tensor` | staticmethod |
| 164-171 | method | `ParametricFVMTests.test_trainer_evaluates_every_flow_case_separately.CaseLoss.apply_velocity_dirichlet_constraint` | staticmethod |
| 174-181 | method | `ParametricFVMTests.test_trainer_evaluates_every_flow_case_separately.CaseLoss.get_freestream_targets` | staticmethod |

**tests/parametric_inputs_smoke.py**

Primary stage: [28. Все тестовые сценарии и подтверждение поведения](REPOSITORY_STUDY_PLAN.md#stage-28).
Source: [open file](<../tests/parametric_inputs_smoke.py>). Lines: 233. SHA-256: `0669dfe47dd199a11b52df420c382407f649fa6b0bd0127b571c7454b9e632ca`.

Module body, imports, exports and class attributes: planned for line-by-line review.

| Lines | Kind | Qualified name | Decorators |
|---|---|---|---|
| 16-81 | class | `FlowCaseTests` |  |
| 19-34 | method | `FlowCaseTests.test_freestream_direction_and_metadata` |  |
| 36-44 | method | `FlowCaseTests.test_non_unit_freestream_is_rejected` |  |
| 46-81 | method | `FlowCaseTests.test_data_handler_materializes_multiple_cases` |  |
| 84-117 | class | `TypedBatchTests` |  |
| 88-104 | method | `TypedBatchTests._sample` | staticmethod |
| 106-117 | method | `TypedBatchTests.test_collation_preserves_named_case_ids` |  |
| 120-229 | class | `InputFeatureAssemblerTests` |  |
| 123-146 | method | `InputFeatureAssemblerTests.setUp` |  |
| 148-186 | method | `InputFeatureAssemblerTests.test_manifest_and_encoded_values` |  |
| 188-208 | method | `InputFeatureAssemblerTests.test_legacy_pi_migration` |  |
| 210-229 | method | `InputFeatureAssemblerTests.test_mandatory_features_cannot_be_disabled` |  |

**tests/parametric_physics_smoke.py**

Primary stage: [28. Все тестовые сценарии и подтверждение поведения](REPOSITORY_STUDY_PLAN.md#stage-28).
Source: [open file](<../tests/parametric_physics_smoke.py>). Lines: 357. SHA-256: `950c86c2827dc89d3721a1449c569a167ac137f696482252b7df15dad4f6554c`.

Module body, imports, exports and class attributes: planned for line-by-line review.

| Lines | Kind | Qualified name | Decorators |
|---|---|---|---|
| 20-28 | function | `_grad` |  |
| 31-99 | class | `DimensionlessPDETests` |  |
| 34-67 | method | `DimensionlessPDETests.test_pointwise_reynolds_controls_viscous_residual` |  |
| 69-90 | method | `DimensionlessPDETests.test_cp_pressure_gradient_has_half_coefficient` |  |
| 92-99 | method | `DimensionlessPDETests.test_inverse_reynolds_vector_cannot_broadcast_to_matrix` |  |
| 102-134 | function | `_loss_config` |  |
| 137-176 | function | `_loss_computer` |  |
| 179-327 | class | `CaseAwareLossTests` |  |
| 182-255 | method | `CaseAwareLossTests.test_pressure_gauge_breaks_casewise_additive_nullspace` |  |
| 257-289 | method | `CaseAwareLossTests.test_soft_boundary_targets_use_sin_cos_alpha` |  |
| 291-314 | method | `CaseAwareLossTests.test_hard_velocity_constraint_is_case_specific` |  |
| 316-327 | method | `CaseAwareLossTests.test_rar_deduplication_keeps_distinct_case_associations` |  |
| 330-353 | class | `SelfAdaptiveWeightTests` |  |
| 333-342 | method | `SelfAdaptiveWeightTests.test_attention_increases_for_positive_loss` |  |
| 344-353 | method | `SelfAdaptiveWeightTests.test_rar_extension_preserves_gradient_reversal` |  |

**tests/profile_section_metrics_smoke.py**

Primary stage: [28. Все тестовые сценарии и подтверждение поведения](REPOSITORY_STUDY_PLAN.md#stage-28).
Source: [open file](<../tests/profile_section_metrics_smoke.py>). Lines: 281. SHA-256: `263e4b25fb513d3cffa2453bba25a87928988d8a37e5e761d4101f05b8451113`.

Module body, imports, exports and class attributes: planned for line-by-line review.

| Lines | Kind | Qualified name | Decorators |
|---|---|---|---|
| 20-63 | function | `_ring_mesh` |  |
| 66-82 | function | `_handler` |  |
| 85-155 | class | `ExactProfileTests` |  |
| 86-104 | method | `ExactProfileTests.test_barycentric_profile_uses_exact_x` |  |
| 106-127 | method | `ExactProfileTests.test_solid_hole_and_outside_domain_are_not_extrapolated` |  |
| 129-155 | method | `ExactProfileTests.test_dataset_values_are_mapped_to_mesh_order` |  |
| 158-277 | class | `SectionFlowMetricTests` |  |
| 159-205 | method | `SectionFlowMetricTests.test_section_flux_does_not_bridge_solid_gap` |  |
| 207-231 | method | `SectionFlowMetricTests.test_unavailable_section_and_text_formatter_are_explicit` |  |
| 233-277 | method | `SectionFlowMetricTests.test_section_metrics_survive_cache_and_strict_jsonl` |  |
| 248-250 | nested function | `SectionFlowMetricTests.test_section_metrics_survive_cache_and_strict_jsonl.compute` |  |

**tests/refactor_smoke.py**

Primary stage: [28. Все тестовые сценарии и подтверждение поведения](REPOSITORY_STUDY_PLAN.md#stage-28).
Source: [open file](<../tests/refactor_smoke.py>). Lines: 399. SHA-256: `4eb3f51a9cb3563b44e2f4d32599042cb7f244b00cbdfdfff59108cc66930020`.

Module body, imports, exports and class attributes: planned for line-by-line review.

| Lines | Kind | Qualified name | Decorators |
|---|---|---|---|
| 25-33 | function | `_load_config` |  |
| 36-79 | function | `_make_loss_computer_for_input_flag` |  |
| 82-97 | function | `test_boundary_types_never_enter_model_input` |  |
| 100-157 | function | `test_fvm_checks` |  |
| 160-178 | function | `test_fvm_schedule` |  |
| 181-205 | function | `test_sdf_chunk` |  |
| 208-224 | function | `test_safe_grad_strict_mode` |  |
| 227-275 | function | `test_rar_score_respects_pde_mask` |  |
| 278-374 | function | `test_one_train_batch` |  |
| 377-395 | function | `main` |  |

**tests/test_fvm_objective.py**

Primary stage: [28. Все тестовые сценарии и подтверждение поведения](REPOSITORY_STUDY_PLAN.md#stage-28).
Source: [open file](<../tests/test_fvm_objective.py>). Lines: 407. SHA-256: `1f57ab3445d25e6ac7f91066697a2a48891de26a6df9572a6270d4ae508ebbfe`.

Module body, imports, exports and class attributes: planned for line-by-line review.

| Lines | Kind | Qualified name | Decorators |
|---|---|---|---|
| 30-84 | function | `_structured_cell_mesh` |  |
| 48-49 | nested function | `_structured_cell_mesh.node` |  |
| 87-99 | function | `_one_cell_mesh` |  |
| 102-248 | class | `FVMObjectiveTests` |  |
| 105-125 | method | `FVMObjectiveTests.test_constant_field_is_discretely_divergence_free` |  |
| 127-157 | method | `FVMObjectiveTests.test_known_divergence_converges_to_four_without_h2_collapse` |  |
| 159-174 | method | `FVMObjectiveTests.test_fvm_loss_preserves_nonzero_finite_autograd` |  |
| 176-195 | method | `FVMObjectiveTests.test_raw_weighted_and_rms_metrics_are_separate` |  |
| 197-206 | method | `FVMObjectiveTests.test_invalid_volumes_are_rejected_instead_of_clamped` |  |
| 208-215 | method | `FVMObjectiveTests.test_rebuild_revalidates_geometry` |  |
| 217-236 | method | `FVMObjectiveTests.test_invalid_faces_closure_and_boundary_types_are_rejected` |  |
| 238-248 | method | `FVMObjectiveTests.test_velocity_node_count_and_mesh_fingerprint_are_strict` |  |
| 251-300 | class | `FVMScheduleTests` |  |
| 255-265 | method | `FVMScheduleTests._trainer` | staticmethod |
| 267-276 | method | `FVMScheduleTests.test_full_schedule_runs_once_per_optimizer_step` |  |
| 278-300 | method | `FVMScheduleTests.test_stochastic_gradient_matches_full_schedule_in_expectation` |  |
| 303-403 | class | `FVMConfigurationContractTests` |  |
| 306-318 | method | `FVMConfigurationContractTests.test_active_fvm_configs_use_the_physical_baseline` |  |
| 320-334 | method | `FVMConfigurationContractTests.test_enabled_fvm_without_mesh_fails_fast` |  |
| 336-403 | method | `FVMConfigurationContractTests.test_checkpoint_fingerprint_changes_with_fvm_contract` |  |

**tests/test_loss_metric_logging.py**

Primary stage: [28. Все тестовые сценарии и подтверждение поведения](REPOSITORY_STUDY_PLAN.md#stage-28).
Source: [open file](<../tests/test_loss_metric_logging.py>). Lines: 219. SHA-256: `db32b631cc50336dad74ee7620c222c3c354104036cabab28784a3c090837627`.

Module body, imports, exports and class attributes: planned for line-by-line review.

| Lines | Kind | Qualified name | Decorators |
|---|---|---|---|
| 18-32 | function | `_loss_computer` |  |
| 35-36 | function | `_data_handler` |  |
| 39-215 | class | `LossMetricLoggingTests` |  |
| 40-66 | method | `LossMetricLoggingTests.test_history_stores_mse_rms_and_max_abs_separately` |  |
| 68-129 | method | `LossMetricLoggingTests.test_epoch_table_has_distinct_mse_and_rms_columns` |  |
| 131-215 | method | `LossMetricLoggingTests.test_epoch_mse_is_weighted_by_active_point_count` |  |
| 136-136 | lambda | `LossMetricLoggingTests.test_epoch_mse_is_weighted_by_active_point_count.<lambda@136:27>` |  |
| 160-190 | nested function | `LossMetricLoggingTests.test_epoch_mse_is_weighted_by_active_point_count.batch` |  |
| 198-198 | lambda | `LossMetricLoggingTests.test_epoch_mse_is_weighted_by_active_point_count.<lambda@198:35>` |  |

**tests/training_robustness_smoke.py**

Primary stage: [28. Все тестовые сценарии и подтверждение поведения](REPOSITORY_STUDY_PLAN.md#stage-28).
Source: [open file](<../tests/training_robustness_smoke.py>). Lines: 248. SHA-256: `a9e84a0a19f46d07fdfe0df7a1dfc51704d7bde10ec57874571a33063750a3bc`.

Module body, imports, exports and class attributes: planned for line-by-line review.

| Lines | Kind | Qualified name | Decorators |
|---|---|---|---|
| 35-94 | class | `SirenContractTests` |  |
| 38-43 | method | `SirenContractTests.test_sine_applies_configured_frequency` |  |
| 45-68 | method | `SirenContractTests.test_siren_layer_bounds_match_paper_initialization` |  |
| 70-94 | method | `SirenContractTests.test_siren_has_finite_second_coordinate_derivatives` |  |
| 97-128 | class | `StaticRobustnessValidationTests` |  |
| 100-107 | method | `StaticRobustnessValidationTests.test_relu_is_rejected_for_viscous_navier_stokes` |  |
| 109-113 | method | `StaticRobustnessValidationTests.test_relu_remains_available_for_first_order_euler` |  |
| 115-119 | method | `StaticRobustnessValidationTests.test_direct_float16_is_rejected` |  |
| 121-128 | method | `StaticRobustnessValidationTests.test_siren_frequencies_must_be_positive` |  |
| 131-161 | class | `NumericalGuardTests` |  |
| 134-143 | method | `NumericalGuardTests.test_nonfinite_loss_reports_components` |  |
| 145-161 | method | `NumericalGuardTests.test_gradient_health_reports_norm_and_nan_parameter` |  |
| 164-244 | class | `OperationalRobustnessTests` |  |
| 167-184 | method | `OperationalRobustnessTests.test_telemetry_is_strict_json_and_atomic` |  |
| 186-191 | method | `OperationalRobustnessTests.test_progress_has_plain_iterable_fallback` |  |
| 193-220 | method | `OperationalRobustnessTests.test_network_diagram_uses_modern_manifest_labels` |  |
| 222-229 | method | `OperationalRobustnessTests.test_worker_seeding_is_reproducible` |  |
| 231-244 | method | `OperationalRobustnessTests.test_spawn_safe_dataloader_worker` |  |

**How to use this index**

For every module, review its executable body and every listed symbol. Add explanation/test status only after the corresponding work. Recompute the inventory after source changes; file hashes identify this snapshot.
The AST enumeration was checked against ast.walk totals for classes, function definitions and lambdas in every file. All source files in the declared scope have a stage; no unmapped source file is accepted.

Non-Python artifacts (run.bat, requirements, READMEs, YAML, JSON libraries and workspace NPY/NPZ formats) are covered explicitly by the study plan rather than by AST rows.
