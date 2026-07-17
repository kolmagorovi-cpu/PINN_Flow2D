"""
Типизированные словари (TypedDict) для проекта Pipe-Bend PINN.

Автор: Адаптировано из Airfoil PINN
Версия: 1.0
"""

from typing import TypedDict, List, Optional, Union, Literal, Tuple, Any, Dict
from typing_extensions import NotRequired


class GeometryConfigDict(TypedDict):
    save_format: str
    plot_enabled: bool
    plot_show: bool
    plot_dpi: int
    output_dir: str
    verbose: bool


class PipeBendParamsDict(TypedDict, total=False):
    name: str
    source_type: Literal['config', 'file']
    description: str
    inlet_length: float
    outlet_length: float
    bend_radius: float
    pipe_width: float
    bend_angle: float
    auto_domain: bool
    origin_at_bend_start: bool


class DomainParamsDict(TypedDict, total=False):
    front: float
    top: float


class GeometryMetadataDict(TypedDict):
    geom_type: str
    pipe_bend_params: Dict[str, Any]
    num_domain_points: int


class CollocationPointDict(TypedDict):
    id: int
    x: float
    y: float
    boundary_type: str
    on_airfoil: bool
    on_boundary: bool
    segment_id: int
    channel_id: str


class SamplerConfigDict(TypedDict, total=False):
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
    symmetry_type: str
    top_boundary_type: Literal['inlet', 'outlet']
    sampling_method: Literal['regular', 'irregular', 'adaptive_foil', 'adaptive_all']
    adaptive_alpha: float
    adaptive_epsilon_factor: float


class BoundaryConditionsDict(TypedDict, total=False):
    fluid_name: str
    inlet_pressure: float
    outlet_pressure: float
    inlet_velocity_mode: Literal['components', 'magnitude_angle', 'poiseuille']
    inlet_u: float
    inlet_v: float
    inlet_speed: float
    inlet_angle: float
    velocity_profile_mode: Literal['uniform', 'parabolic']
    symmetry_type: Optional[str]
    description: str


class FluidDict(TypedDict, total=False):
    name: str
    density: float
    kinematic_viscosity: float
    dynamic_viscosity: Optional[float]
    description: str


class PointDtypeDict(TypedDict):
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
    description: str
    fluid_name: str
    conditions_name: str
    temperature: float
    pressure_mode: str
    velocity_initial_mode: str
    velocity_perturbation_amplitude: float
    pressure_perturbation_amplitude: float
    num_points: int
    pipe_width: float
    U_inf: float
    p_ref: float
    rho: float
    mu: float
    Re: float
    normalized: bool
    sdf_norm_factor: float
    inlet_u_norm: float
    inlet_v_norm: float


class InputParamsDict(TypedDict, total=False):
    coordinates: bool
    sdf: bool
    inlet_velocity: bool
    viscosity: bool
    density: bool
    fluid_file: Optional[str]


class ArchitectureDict(TypedDict, total=False):
    hidden_layers: List[int]
    activation: str
    output_dim: int
    output_bias: Union[List[float], Tuple[float, ...]]
    coordinate_scaling: 'CoordinateScalingDict'


class CoordinateScalingDict(TypedDict, total=False):
    enabled: bool
    scale_factor: float
    input_min: float
    input_max: float


class PDEConfigDict(TypedDict, total=False):
    system: str
    flow_type: str


class BoundaryConditionsConfigDict(TypedDict, total=False):
    type: str


class LossWeightsConfigDict(TypedDict, total=False):
    pde_weights: Dict[str, float]
    bc_weights: Dict[str, float]
    adaptive_bc: bool
    use_sa_weights: bool
    sa_apply_to_bc: bool
    integral_condition: bool
    integral_weight: float


class TrainingConfigDict(TypedDict, total=False):
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
    enabled: bool
    patience: int
    min_delta: float
    restore_best: bool


class OutputConfigDict(TypedDict, total=False):
    console_only: bool
    graphical_window: 'GraphicalWindowDict'
    plot_frequency: int


class GraphicalWindowDict(TypedDict, total=False):
    loss: bool
    pde: bool
    bc: bool


class CheckpointConfigDict(TypedDict, total=False):
    interval: int
    name_pattern: str
    model_name_pattern: str


class ProcessConfigDict(TypedDict, total=False):
    load_model_path: Optional[str]
    device: str
    num_gpus: int
    dtype: str
    use_amp: bool
    seed: int


class DataConfigDict(TypedDict, total=False):
    points_file: Optional[str]


class ReferenceConfigDict(TypedDict, total=False):
    path: Optional[str]


class DiagnosticsConfigDict(TypedDict, total=False):
    interval: int
    plot_residual_map: bool
    plot_profiles: bool
    profile_sections: List[float]
    plot_spectrum: bool
    spectrum_section_x: float
    plot_2d_spectrum: bool
    spectrum_2d_resolution: int
    compute_grad_cosine: bool
    compute_ntk_condition: bool
    save_rar_history: bool


class FullConfigDict(TypedDict, total=False):
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


class WorkerProgressDict(TypedDict):
    proc_id: int
    current: int
    total: int


class PlotConfigDict(TypedDict, total=False):
    figsize: Tuple[int, int]
    dpi: int
    colormap: str
    alpha: float
    point_size: int
    save_path: Optional[str]
    show: bool


class OptimizerParamsDict(TypedDict, total=False):
    lr: float
    weight_decay: float
    betas: Tuple[float, float]
    eps: float


class L2ErrorDict(TypedDict):
    u_error: float
    v_error: float
    p_error: float
    total_error: float


__all__ = [
    'GeometryConfigDict',
    'PipeBendParamsDict',
    'DomainParamsDict',
    'GeometryMetadataDict',
    'CollocationPointDict',
    'SamplerConfigDict',
    'BoundaryConditionsDict',
    'FluidDict',
    'PointDtypeDict',
    'InitializerMetadataDict',
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
    'WorkerProgressDict',
    'PlotConfigDict',
    'OptimizerParamsDict',
    'L2ErrorDict',
]