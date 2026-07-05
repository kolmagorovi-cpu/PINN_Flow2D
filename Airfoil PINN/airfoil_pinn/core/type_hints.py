#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Типизированные словари (TypedDict) для проекта Airfoil PINN.

Этот модуль содержит определения структур данных для конфигураций,
используемых в различных модулях проекта.

Example:
    >>> from airfoil_pinn.core.type_hints import GeometryConfigDict, SamplerConfigDict
    >>> config: GeometryConfigDict = {
    ...     "save_format": "npy",
    ...     "plot_enabled": True,
    ...     "plot_show": True,
    ...     "plot_dpi": 300,
    ...     "output_dir": "workspace/geometry/domains"
    ... }
"""

from typing import TypedDict, List, Optional, Union, Literal, Tuple, Any
from typing_extensions import NotRequired


# =============================================================================
# ГЕОМЕТРИЯ (Geometry2D.py, Geometry.py)
# =============================================================================

class GeometryConfigDict(TypedDict):
    """
    Конфигурация для геометрических параметров (универсальная).

    Attributes:
        save_format: Формат сохранения ('npy', 'csv', 'txt')
        plot_enabled: Включена ли визуализация
        plot_show: Показывать ли график сразу
        plot_dpi: DPI для сохранения графиков
        output_dir: Директория для выходных файлов
        verbose: Подробный вывод в консоль
    """
    save_format: str
    plot_enabled: bool
    plot_show: bool
    plot_dpi: int
    output_dir: str
    verbose: bool


class GeomConfigFileParamsDict(TypedDict, total=False):
    """
    Параметры для загрузки геометрии из файла.

    Attributes:
        source_type: Тип источника ('file' или 'inline' для профиля, 'config' или 'file' для тройника)
        file_path: Путь к файлу с геометрией
        file_format: Формат файла ('csv', 'txt', 'dat', 'npy', 'step', 'iges')
    """
    source_type: Literal['file', 'inline', 'config']
    file_path: str
    file_format: str


class AirfoilDomainParamsDict(TypedDict, total=False):
    """
    Параметры расчётной области для профиля.

    Attributes:
        front: Длина домена перед профилем [м]
        rear: Длина домена за профилем [м] (для замкнутого профиля)
        top: Высота домена над профилем [м]
        bottom: Высота домена под профилем [м] (для замкнутого профиля)
    """
    front: float
    rear: float
    top: float
    bottom: float


class AirfoilParamsDict(TypedDict, total=False):
    """
    Параметры профиля (airfoil).

    Attributes:
        source_type: Тип источника ('file' или 'inline')
        file_path: Путь к файлу профиля
        file_format: Формат файла ('csv', 'txt', 'dat', 'npy')
        contour_coef: Коэффициент масштабирования контура
        x_domain_coef: Коэффициент масштабирования длины области
        y_domain_coef: Коэффициент масштабирования высоты области
        domain_params: Параметры расчётной области
        auto_domain: Автоматический расчёт размеров области
        normalize: Нормализация координат к хорде
    """
    source_type: Literal['file', 'inline']
    file_path: str
    file_format: Literal['csv', 'txt', 'dat', 'npy']
    contour_coef: float
    x_domain_coef: float
    y_domain_coef: float
    domain_params: AirfoilDomainParamsDict
    auto_domain: bool
    normalize: bool


class DomainParamsDict(TypedDict, total=False):
    """
    Параметры расчётной области (для незамкнутого профиля).

    Attributes:
        front: Длина домена перед профилем [м]
        top: Высота домена над профилем [м]
    """
    front: float
    top: float


class ClosedDomainParamsDict(TypedDict, total=False):
    """
    Параметры замкнутой расчётной области (для замкнутого профиля).

    Attributes:
        front: Длина домена перед профилем [м]
        rear: Длина домена за профилем [м]
        top: Высота домена над профилем [м]
        bottom: Высота домена под профилем [м]
    """
    front: float
    rear: float
    top: float
    bottom: float


class GeometryMetadataDict(TypedDict):
    """
    Метаданные геометрии профиля.

    Attributes:
        chord: Хорда профиля [м]
        is_closed: Замкнут ли профиль
        num_foil_points: Количество точек профиля
        num_domain_points: Количество точек области
    """
    chord: float
    is_closed: bool
    num_foil_points: int
    num_domain_points: int


# =============================================================================
# ДИСКРЕТИЗАЦИЯ (Collocation.py)
# =============================================================================

class CollocationPointDict(TypedDict):
    """
    Словарное представление точки коллокации.

    Attributes:
        id: Уникальный идентификатор точки
        x: X-координата [м]
        y: Y-координата [м]
        boundary_type: Тип границы ('inlet', 'outlet', 'wall', 'symmetry', 'axis_symmetry', 'interior')
        on_airfoil: Находится ли точка на профиле
        on_boundary: Находится ли точка на границе
        segment_id: ID сегмента границы
    """
    id: int
    x: float
    y: float
    boundary_type: str
    on_airfoil: bool
    on_boundary: bool
    segment_id: int


class SamplerConfigDict(TypedDict, total=False):
    """
    Конфигурация для дискретизации расчётной области.

    Attributes:
        num_processes: Количество процессов для параллельных вычислений
        step_size: Шаг дискретизации [м]
        step_units: Единицы шага ('m' или 'mm')
        use_multiprocessing: Использовать ли многопроцессорность
        show_progress: Показывать ли прогресс-бар
        save_format: Формат сохранения ('npy', 'csv')
        plot_results: Построить ли результаты
        plot_dpi: DPI для графиков
        geometry_dir: Директория с геометрией
        output_dir: Директория для выходных файлов
        boundary_point_density: Плотность точек на границах
        interior_point_density: Плотность внутренних точек
        left_boundary_type: Тип левой границы (x=x_min)
        right_boundary_type: Тип правой границы (x=x_max)
        top_boundary_type: Тип верхней границы (y=y_max)
        bottom_boundary_type: Тип нижней границы (y=y_min)
        sampling_method: Метод генерации ('regular', 'irregular', 'adaptive_foil', 'adaptive_all')
        adaptive_alpha: Степень сгущения (для адаптивных методов)
        adaptive_epsilon_factor: Масштаб сгущения (для адаптивных методов)
    """
    num_processes: int
    step_size: float
    step_units: str
    use_multiprocessing: bool
    show_progress: bool
    save_format: str
    plot_results: bool
    plot_dpi: int
    geometry_dir: str
    output_dir: str
    boundary_point_density: float
    interior_point_density: float
    left_boundary_type: str
    right_boundary_type: str
    top_boundary_type: str
    bottom_boundary_type: str
    sampling_method: Literal['regular', 'irregular', 'adaptive_foil', 'adaptive_all']
    adaptive_alpha: float
    adaptive_epsilon_factor: float


# =============================================================================
# ГРАНИЧНЫЕ УСЛОВИЯ (Conditions.py)
# =============================================================================

class BoundaryConditionsDict(TypedDict, total=False):
    """
    Словарное представление граничных условий.

    Attributes:
        fluid_name: Название жидкости из библиотеки
        inlet_pressure: Давление на входе [Па]
        outlet_pressure: Давление на выходе [Па]
        inlet_velocity_mode: Режим ввода скорости ('components' или 'magnitude_angle')
        inlet_u: Компонента U скорости на входе [м/с]
        inlet_v: Компонента V скорости на входе [м/с]
        inlet_speed: Модуль скорости на входе [м/с]
        inlet_angle: Угол атаки [градусы]
        symmetry_type: Тип симметрии ('symmetry', 'axis_symmetry' или None)
        description: Описание набора условий
    """
    fluid_name: str
    inlet_pressure: float
    outlet_pressure: float
    inlet_velocity_mode: Literal['components', 'magnitude_angle']
    inlet_u: float
    inlet_v: float
    inlet_speed: float
    inlet_angle: float
    symmetry_type: Optional[str]
    description: str


# =============================================================================
# ЖИДКОСТИ (Fluids.py)
# =============================================================================

class FluidDict(TypedDict, total=False):
    """
    Словарное представление жидкости.

    Attributes:
        name: Название жидкости
        density: Плотность [кг/м³]
        kinematic_viscosity: Кинематическая вязкость [м²/с]
        dynamic_viscosity: Динамическая вязкость [Па·с]
        description: Краткое описание
    """
    name: str
    density: float
    kinematic_viscosity: float
    dynamic_viscosity: Optional[float]
    description: str


# =============================================================================
# ИНИЦИАЛИЗАЦИЯ (Initializer.py)
# =============================================================================

class PointDtypeDict(TypedDict):
    """
    Структура нормализованной точки для PINN.

    Attributes:
        id: Идентификатор точки
        x_norm: Нормированная X-координата
        y_norm: Нормированная Y-координата
        sdf_norm: Нормированное расстояние до стенки (SDF)
        boundary_type: Тип границы
        on_boundary: Находится ли на границе
        p_norm: Нормированное давление
        u_norm: Нормированная скорость U
        v_norm: Нормированная скорость V
    """
    id: int
    x_norm: float
    y_norm: float
    sdf_norm: float
    boundary_type: str
    on_boundary: bool
    p_norm: float
    u_norm: float
    v_norm: float


class InitializerMetadataDict(TypedDict, total=False):
    """
    Метаданные инициализации расчётной области.

    Attributes:
        description: Описание инициализации
        fluid_name: Название жидкости
        conditions_name: Название граничных условий
        temperature: Температура [K]
        pressure_mode: Режим давления ('constant' или 'linear_x')
        velocity_initial_mode: Режим инициализации скорости
        velocity_perturbation_amplitude: Амплитуда возмущений скорости
        pressure_perturbation_amplitude: Амплитуда возмущений давления
        num_points: Количество точек
        chord: Хорда профиля [м]
        U_inf: Характерная скорость [м/с]
        p_ref: Опорное давление [Па]
        rho: Плотность [кг/м³]
        mu: Динамическая вязкость [Па·с]
        Re: Число Рейнольдса
        normalized: Флаг нормализации данных
        sdf_norm_factor: Фактор нормализации SDF
        inlet_u_norm: Нормированная скорость U на входе
        inlet_v_norm: Нормированная скорость V на входе
    """
    description: str
    fluid_name: str
    conditions_name: str
    temperature: float
    pressure_mode: str
    velocity_initial_mode: str
    velocity_perturbation_amplitude: float
    pressure_perturbation_amplitude: float
    num_points: int
    chord: float
    U_inf: float
    p_ref: float
    rho: float
    mu: float
    Re: float
    normalized: bool
    sdf_norm_factor: float
    inlet_u_norm: float
    inlet_v_norm: float


# =============================================================================
# КОНФИГУРАЦИЯ (config_manager.py)
# =============================================================================

class InputParamsDict(TypedDict, total=False):
    """Параметры входа модели."""
    coordinates: bool
    sdf: bool
    inlet_velocity: bool
    viscosity: bool
    density: bool
    fluid_file: Optional[str]


class ArchitectureDict(TypedDict, total=False):
    """Архитектура нейронной сети."""
    hidden_layers: List[int]
    activation: str
    output_dim: int
    output_bias: Union[List[float], Tuple[float, ...]]
    coordinate_scaling: 'CoordinateScalingDict'


class CoordinateScalingDict(TypedDict, total=False):
    """Параметры масштабирования координат."""
    enabled: bool
    scale_factor: float
    input_min: float
    input_max: float


class PDEConfigDict(TypedDict, total=False):
    """Конфигурация уравнений в частных производных."""
    system: str
    flow_type: str
    pde_on_boundary: bool


class BoundaryConditionsConfigDict(TypedDict, total=False):
    """Конфигурация граничных условий."""
    type: str


class LossWeightsConfigDict(TypedDict, total=False):
    """Веса функций потерь."""
    pde_weights: Dict[str, float]
    bc_weights: Dict[str, float]
    adaptive_bc: bool
    use_sa_weights: bool
    sa_apply_to_bc: bool
    integral_condition: bool
    integral_weight: float


class TrainingConfigDict(TypedDict, total=False):
    """Конфигурация обучения."""
    two_stage: bool
    stage1_epochs: int
    stage2_pde_weight_start: float
    stage2_pde_weight_end: float
    stage2_ramp_epochs: int
    lr_schedule: str
    lr_cyclic_period: int
    lr_cyclic_amplitude: float
    lr_detector_factor: float
    lr_detector_patience: int
    lr_detector_recovery: int
    reset_interval: int
    reset_noise: float
    grad_clip: Optional[float]
    rar_enable: bool
    rar_interval: int
    rar_num_points: int
    switch_to_lbfgs_epoch: int
    lbfgs_max_iter: int
    lbfgs_history_size: int
    batch_size: int
    gradient_accumulation_steps: int
    epochs: int
    early_stopping: 'EarlyStoppingDict'


class EarlyStoppingDict(TypedDict, total=False):
    """Конфигурация ранней остановки."""
    enabled: bool
    patience: int
    min_delta: float
    restore_best: bool


class OutputConfigDict(TypedDict, total=False):
    """Конфигурация вывода."""
    console_only: bool
    graphical_window: 'GraphicalWindowDict'
    plot_frequency: int


class GraphicalWindowDict(TypedDict, total=False):
    """Настройки графического окна."""
    loss: bool
    pde: bool
    bc: bool


class CheckpointConfigDict(TypedDict, total=False):
    """Конфигурация контрольных точек."""
    interval: int
    name_pattern: str
    model_name_pattern: str


class ProcessConfigDict(TypedDict, total=False):
    """Конфигурация процесса."""
    load_model_path: Optional[str]
    device: str
    num_gpus: int
    dtype: str
    use_amp: bool
    torch_compile: bool
    torch_compile_mode: Optional[str]
    torch_compile_fullgraph: bool
    torch_compile_dynamic: Optional[bool]
    seed: int


class DataConfigDict(TypedDict, total=False):
    """Конфигурация данных."""
    points_file: Optional[str]
    mesh_file: Optional[str]
    num_workers: int
    pin_memory: Optional[bool]
    persistent_workers: Optional[bool]
    prefetch_factor: int


class ReferenceConfigDict(TypedDict, total=False):
    """Конфигурация эталонного решения."""
    path: Optional[str]


class DiagnosticsConfigDict(TypedDict, total=False):
    """Конфигурация диагностики."""
    interval: int
    plot_residual_map: bool
    plot_rar_error_map: bool
    plot_profiles: bool
    profile_sections: List[float]
    plot_spectrum: bool
    spectrum_section_x: float
    plot_2d_spectrum: bool
    spectrum_2d_resolution: int
    compute_grad_cosine: bool
    compute_ntk_condition: bool
    strict_autograd: bool
    save_rar_history: bool
    mass_metrics_interval: int


class FullConfigDict(TypedDict, total=False):
    """
    Полная конфигурация проекта.

    Объединяет все разделы конфигурации.
    """
    project_name: str
    input_params: InputParamsDict
    architecture: ArchitectureDict
    pde: PDEConfigDict
    boundary_conditions: BoundaryConditionsConfigDict
    loss_weights: LossWeightsConfigDict
    training: TrainingConfigDict
    output: OutputConfigDict
    checkpoint: CheckpointConfigDict
    process: ProcessConfigDict
    data: DataConfigDict
    reference: ReferenceConfigDict
    diagnostics: DiagnosticsConfigDict
    enable_key_listener: bool
    checkpoint_dir: str


# =============================================================================
# МНОГОПРОЦЕССОРНОСТЬ
# =============================================================================

class WorkerProgressDict(TypedDict):
    """Прогресс выполнения worker-процесса."""
    proc_id: int
    current: int
    total: int


# =============================================================================
# ВИЗУАЛИЗАЦИЯ
# =============================================================================

class PlotConfigDict(TypedDict, total=False):
    """Конфигурация для построения графиков."""
    figsize: Tuple[int, int]
    dpi: int
    colormap: str
    alpha: float
    point_size: int
    save_path: Optional[str]
    show: bool


# =============================================================================
# ОБУЧЕНИЕ (training.components)
# =============================================================================

class OptimizerParamsDict(TypedDict, total=False):
    """Параметры оптимизатора."""
    lr: float
    weight_decay: float
    betas: Tuple[float, float]
    eps: float


class L2ErrorDict(TypedDict):
    """Результаты вычисления L2-ошибок."""
    u_error: float
    v_error: float
    p_error: float
    total_error: float


# =============================================================================
# ЭКСПОРТ
# =============================================================================

__all__ = [
    # Геометрия
    'GeometryConfigDict',
    'GeomConfigFileParamsDict',
    'AirfoilDomainParamsDict',
    'AirfoilParamsDict',
    'DomainParamsDict',
    'ClosedDomainParamsDict',
    'GeometryMetadataDict',
    # Дискретизация
    'CollocationPointDict',
    'SamplerConfigDict',
    # Граничные условия
    'BoundaryConditionsDict',
    # Жидкости
    'FluidDict',
    # Инициализация
    'PointDtypeDict',
    'InitializerMetadataDict',
    # Конфигурация
    'InputParamsDict',
    'ArchitectureDict',
    'CoordinateScalingDict',
    'PDEConfigDict',
    'BoundaryConditionsConfigDict',
    'LossWeightsConfigDict',
    'TrainingConfigDict',
    'EarlyStoppingDict',
    'OutputConfigDict',
    'GraphicalWindowDict',
    'CheckpointConfigDict',
    'ProcessConfigDict',
    'DataConfigDict',
    'ReferenceConfigDict',
    'DiagnosticsConfigDict',
    'FullConfigDict',
    # Многопроцессорность
    'WorkerProgressDict',
    # Визуализация
    'PlotConfigDict',
    # Обучение
    'OptimizerParamsDict',
    'L2ErrorDict',
]
