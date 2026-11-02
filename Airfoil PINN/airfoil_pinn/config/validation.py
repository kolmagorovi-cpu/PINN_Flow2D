"""Fail-fast validation for PINN configuration and runtime contracts.

Static validation checks the YAML-level schema without touching the
filesystem.  Runtime validation additionally checks the loaded collocation
artifact, flow-case table and manifest-driven network inputs.  Keeping these
checks in one module prevents individual training components from accepting
mutually incompatible assumptions.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

__all__ = [
    "ConfigurationError",
    "validate_runtime_contract",
    "validate_static_config",
]


class ConfigurationError(ValueError):
    """One or more configuration invariants are violated."""


def _mapping(
    config: Mapping[str, Any],
    key: str,
    errors: list[str],
) -> Mapping[str, Any]:
    value = config.get(key)
    if not isinstance(value, Mapping):
        errors.append(f"{key} must be a mapping")
        return {}
    return value


def _positive_integer(
    value: Any,
    name: str,
    errors: list[str],
) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        errors.append(f"{name} must be a positive integer")


def _finite_nonnegative(
    value: Any,
    name: str,
    errors: list[str],
) -> None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        errors.append(f"{name} must be numeric")
        return
    if not math.isfinite(numeric) or numeric < 0.0:
        errors.append(f"{name} must be finite and non-negative")


def _raise_errors(errors: Sequence[str]) -> None:
    if errors:
        details = "\n".join(f"- {message}" for message in errors)
        raise ConfigurationError(
            "Invalid PINN configuration/runtime contract:\n" + details
        )


def validate_static_config(config: Mapping[str, Any]) -> None:
    """Validate configuration values that do not require loaded data."""
    errors: list[str] = []
    input_params = _mapping(config, "input_params", errors)
    architecture = _mapping(config, "architecture", errors)
    pde = _mapping(config, "pde", errors)
    boundary = _mapping(config, "boundary_conditions", errors)
    loss = _mapping(config, "loss_weights", errors)
    training = _mapping(config, "training", errors)
    process = _mapping(config, "process", errors)
    data = _mapping(config, "data", errors)
    output = _mapping(config, "output", errors)
    checkpoint = _mapping(config, "checkpoint", errors)
    sdf_config = config.get("sdf", {})
    if not isinstance(sdf_config, Mapping):
        errors.append("sdf must be a mapping")
        sdf_config = {}

    features = config.get("input_features", {})
    if features is not None and not isinstance(features, Mapping):
        errors.append("input_features must be a mapping when provided")
        features = {}

    coordinates = features.get(
        "coordinates",
        input_params.get("coordinates", True),
    )
    sdf = features.get("sdf", input_params.get("sdf", True))
    if isinstance(coordinates, Mapping):
        coordinates = coordinates.get("enabled", True)
    if isinstance(sdf, Mapping):
        sdf = sdf.get("enabled", True)
    if not bool(coordinates):
        errors.append("coordinates x/c and y/c are mandatory")
    if not bool(sdf):
        errors.append("SDF/c is a mandatory network input")

    hidden = architecture.get("hidden_layers")
    if (
        not isinstance(hidden, Sequence)
        or isinstance(hidden, (str, bytes))
        or not hidden
        or any(
            isinstance(width, bool)
            or not isinstance(width, int)
            or width <= 0
            for width in hidden
        )
    ):
        errors.append(
            "architecture.hidden_layers must contain positive integers"
        )
    if architecture.get("activation") not in {
        "tanh",
        "relu",
        "sigmoid",
        "leaky_relu",
        "elu",
        "swish",
        "siren",
    }:
        errors.append("architecture.activation is unsupported")
    if (
        pde.get("system") == "incompressible_navier_stokes"
        and architecture.get("activation")
        in {"relu", "leaky_relu", "elu"}
    ):
        errors.append(
            "viscous Navier-Stokes requires a twice-differentiable "
            "activation; use tanh, sigmoid, swish or siren"
        )
    for key in ("siren_first_omega_0", "siren_hidden_omega_0"):
        try:
            omega = float(architecture.get(key, 30.0))
        except (TypeError, ValueError):
            errors.append(f"architecture.{key} must be numeric")
        else:
            if not math.isfinite(omega) or omega <= 0.0:
                errors.append(
                    f"architecture.{key} must be finite and positive"
                )
    coordinate_scaling = architecture.get("coordinate_scaling", {})
    if not isinstance(coordinate_scaling, Mapping):
        errors.append("architecture.coordinate_scaling must be a mapping")
    else:
        scaling_mode = coordinate_scaling.get("mode", "multiply")
        if scaling_mode not in {"multiply", "dataset_minmax"}:
            errors.append(
                "architecture.coordinate_scaling.mode must be one of: "
                "multiply, dataset_minmax"
            )
        if (
            bool(coordinate_scaling.get("enabled", False))
            and scaling_mode == "multiply"
        ):
            try:
                scale_factor = float(
                    coordinate_scaling.get("scale_factor", 1.0)
                )
            except (TypeError, ValueError):
                errors.append(
                    "architecture.coordinate_scaling.scale_factor must be numeric"
                )
            else:
                if not math.isfinite(scale_factor) or scale_factor <= 0.0:
                    errors.append(
                        "architecture.coordinate_scaling.scale_factor must be "
                        "finite and positive"
                    )
    if architecture.get("output_dim") != 3:
        errors.append(
            "architecture.output_dim must be 3 for [u/Uinf, v/Uinf, Cp]"
        )
    _finite_nonnegative(
        sdf_config.get("float64_consistency_tolerance", 1e-12),
        "sdf.float64_consistency_tolerance",
        errors,
    )
    output_bias = architecture.get("output_bias")
    if (
        not isinstance(output_bias, Sequence)
        or isinstance(output_bias, (str, bytes))
        or len(output_bias) != 3
        or not all(np.isfinite(float(value)) for value in output_bias)
    ):
        errors.append(
            "architecture.output_bias must contain three finite values"
        )
    elif float(output_bias[2]) != 0.0:
        errors.append(
            "architecture.output_bias[2] must be zero for the raw pressure "
            "output"
        )

    if pde.get("system") not in {
        "incompressible_navier_stokes",
        "euler",
    }:
        errors.append("pde.system must be incompressible_navier_stokes or euler")
    if pde.get("flow_type") not in {"2d", "axisymmetric"}:
        errors.append("pde.flow_type must be 2d or axisymmetric")
    legacy_boundary_type = boundary.get("type")
    if legacy_boundary_type == "hard":
        errors.append(
            "Global boundary_conditions.type=hard is deprecated. Configure "
            "enforcement separately for each boundary."
        )
    elif legacy_boundary_type not in {None, "soft"}:
        errors.append(
            "boundary_conditions.type is deprecated and, when present for "
            "legacy files, must be soft"
        )

    for boundary_name, boundary_value in boundary.items():
        if boundary_name == "type":
            continue
        if not isinstance(boundary_value, Mapping):
            errors.append(
                f"boundary_conditions.{boundary_name} must be a mapping"
            )
            continue
        enforcement = boundary_value.get("enforcement", "soft")
        if enforcement not in {"soft", "hard"}:
            errors.append(
                f"boundary_conditions.{boundary_name}.enforcement must be "
                "soft or hard"
            )
        elif enforcement == "hard" and boundary_name != "wall":
            errors.append(
                f"Hard enforcement is not yet supported for boundary "
                f"{boundary_name!r}; use soft enforcement."
            )

    wall = boundary.get("wall", {})
    if not isinstance(wall, Mapping):
        wall = {}
    wall_enforcement = wall.get("enforcement", "soft")
    wall_kind = wall.get("kind", "no_slip")
    if wall_enforcement == "hard":
        if pde.get("system") != "incompressible_navier_stokes":
            errors.append(
                "Hard wall is currently supported only for incompressible "
                "Navier-Stokes."
            )
        if wall_kind != "no_slip":
            errors.append(
                "Hard wall currently supports only the no-slip condition."
            )
        if not bool(sdf):
            errors.append("Hard wall requires differentiable SDF.")
        try:
            distance_scale = float(wall.get("distance_scale", 0.02))
        except (TypeError, ValueError):
            errors.append(
                "boundary_conditions.wall.distance_scale must be numeric"
            )
        else:
            if not math.isfinite(distance_scale) or distance_scale <= 0.0:
                errors.append(
                    "boundary_conditions.wall.distance_scale must be finite "
                    "and positive"
                )
    if (
        pde.get("system") == "euler"
        and legacy_boundary_type == "hard"
    ):
        errors.append(
            "Euler wall impermeability couples u and v through the surface "
            "normal and currently requires boundary_conditions.type=soft"
        )
    if checkpoint.get("load_mode", "resume") not in {
        "resume",
        "finetune",
    }:
        errors.append("checkpoint.load_mode must be resume or finetune")

    pde_weights = loss.get("pde_weights")
    if not isinstance(pde_weights, Mapping):
        errors.append("loss_weights.pde_weights must be a mapping")
    else:
        components = (
            ("cont", "mom_r", "mom_z")
            if pde.get("flow_type") == "axisymmetric"
            else ("cont", "mom_x", "mom_y")
        )
        for component in components:
            if component not in pde_weights:
                errors.append(
                    f"loss_weights.pde_weights.{component} is required"
                )
            else:
                _finite_nonnegative(
                    pde_weights[component],
                    f"loss_weights.pde_weights.{component}",
                    errors,
                )

    try:
        pressure_gauge_weight = float(
            loss.get("pressure_gauge_weight", 1.0)
        )
    except (TypeError, ValueError):
        errors.append("loss_weights.pressure_gauge_weight must be numeric")
    else:
        if (
            not math.isfinite(pressure_gauge_weight)
            or pressure_gauge_weight <= 0.0
        ):
            errors.append(
                "loss_weights.pressure_gauge_weight must be finite and "
                "strictly positive"
            )

    fvm = loss.get("fvm", {})
    if isinstance(fvm, Mapping):
        _finite_nonnegative(
            fvm.get("weight", 1.0),
            "loss_weights.fvm.weight",
            errors,
        )
        _positive_integer(
            fvm.get("batch_interval", 1),
            "loss_weights.fvm.batch_interval",
            errors,
        )
        _positive_integer(
            fvm.get("forward_chunk_size", 32768),
            "loss_weights.fvm.forward_chunk_size",
            errors,
        )
        if not isinstance(fvm.get("activation_checkpoint", True), bool):
            errors.append(
                "loss_weights.fvm.activation_checkpoint must be boolean"
            )
        reduction = str(
            fvm.get(
                "reduction",
                fvm.get("loss_reduction", "divergence_l2_volume"),
            )
        ).lower().replace("-", "_")
        if reduction not in {
            "divergence_l2_volume",
            "legacy",
            "mean",
            "node_mean",
            "unweighted",
            "volume",
            "volume_weighted",
            "cell_volume",
            "flux",
            "flux_volume",
            "flux_weighted",
            "balance",
            "flux_balance_l2",
            "flux_mean",
            "mean_flux",
            "divergence_volume",
            "divergence_volume_weighted",
            "old_volume_weighted",
        }:
            errors.append("loss_weights.fvm.reduction is unsupported")
        schedule = str(
            fvm.get("schedule", "every_optimizer_step")
        ).lower().replace("-", "_")
        if schedule not in {
            "every_optimizer_step",
            "every_k_optimizer_steps",
            "stochastic_optimizer_step",
        }:
            errors.append("loss_weights.fvm.schedule is unsupported")
        if (
            schedule == "every_optimizer_step"
            and fvm.get("batch_interval", 1) != 1
        ):
            errors.append(
                "loss_weights.fvm.batch_interval must be 1 when schedule is "
                "every_optimizer_step"
            )
        if (
            bool(fvm.get("enabled", False))
            and pde.get("flow_type") == "axisymmetric"
        ):
            errors.append(
                "axisymmetric FVM loss is unavailable for the planar "
                "dual-cell operator"
            )
    else:
        _finite_nonnegative(fvm, "loss_weights.fvm", errors)

    for key in ("batch_size", "epochs", "gradient_accumulation_steps"):
        _positive_integer(
            training.get(key),
            f"training.{key}",
            errors,
        )
    if not isinstance(training.get("full_epoch_optimizer_step", False), bool):
        errors.append("training.full_epoch_optimizer_step must be boolean")
    if training.get("switch_to_lbfgs_epoch", 0) < 0:
        errors.append("training.switch_to_lbfgs_epoch must be non-negative")
    if process.get("dtype") not in {"float32", "float64"}:
        errors.append(
            "process.dtype must be float32 or float64; direct float16 is "
            "unsafe for second-order PINN derivatives (use AMP instead)"
        )
    _positive_integer(
        process.get("num_gpus", 1),
        "process.num_gpus",
        errors,
    )
    grad_clip = training.get("grad_clip")
    if grad_clip is not None:
        try:
            grad_clip_value = float(grad_clip)
        except (TypeError, ValueError):
            errors.append("training.grad_clip must be numeric or null")
        else:
            if not math.isfinite(grad_clip_value) or grad_clip_value <= 0.0:
                errors.append(
                    "training.grad_clip must be finite and positive"
                )
    num_workers = data.get("num_workers", 0)
    if (
        isinstance(num_workers, bool)
        or not isinstance(num_workers, int)
        or num_workers < 0
    ):
        errors.append("data.num_workers must be a non-negative integer")
    _positive_integer(
        data.get("prefetch_factor", 2),
        "data.prefetch_factor",
        errors,
    )
    try:
        progress_mininterval = float(
            output.get("progress_mininterval", 0.5)
        )
    except (TypeError, ValueError):
        errors.append("output.progress_mininterval must be numeric")
    else:
        if (
            not math.isfinite(progress_mininterval)
            or progress_mininterval <= 0.0
        ):
            errors.append(
                "output.progress_mininterval must be finite and positive"
            )

    reynolds_cfg = (
        features.get("reynolds", {})
        if isinstance(features, Mapping)
        else {}
    )
    if isinstance(reynolds_cfg, Mapping) and "range" in reynolds_cfg:
        value_range = reynolds_cfg["range"]
        if (
            not isinstance(value_range, Sequence)
            or isinstance(value_range, (str, bytes))
            or len(value_range) != 2
        ):
            errors.append("input_features.reynolds.range must be [min, max]")
        else:
            lower, upper = (float(value) for value in value_range)
            if (
                not math.isfinite(lower)
                or not math.isfinite(upper)
                or lower <= 0.0
                or upper <= lower
            ):
                errors.append(
                    "input_features.reynolds.range must be finite, positive "
                    "and strictly ordered"
                )

    geometry_cfg = (
        features.get("geometry_parameters", {})
        if isinstance(features, Mapping)
        else {}
    )
    if isinstance(geometry_cfg, Mapping):
        names = geometry_cfg.get("names", ())
        enabled = bool(geometry_cfg.get("enabled", bool(names)))
        if enabled:
            if (
                not isinstance(names, Sequence)
                or isinstance(names, (str, bytes))
                or not names
                or any(not str(name).strip() for name in names)
            ):
                errors.append(
                    "enabled input_features.geometry_parameters requires "
                    "non-empty names"
                )
            elif len({str(name) for name in names}) != len(names):
                errors.append(
                    "input_features.geometry_parameters.names must be unique"
                )

    _raise_errors(errors)


def validate_runtime_contract(
    config: Mapping[str, Any],
    data_handler: Any,
    feature_assembler: Any,
    *,
    sdf_geometry_fingerprint: Any = None,
) -> None:
    """Validate loaded data against the manifest and parametric physics."""
    validate_static_config(config)
    errors: list[str] = []

    cases = list(getattr(data_handler, "flow_cases", {}).values())
    if not cases:
        errors.append("the loaded dataset has no flow cases")
        _raise_errors(errors)

    dataset_ids = set(
        int(value)
        for value in data_handler.dataset.case_id.detach().cpu().tolist()
    )
    table_ids = {int(case.case_id) for case in cases}
    missing_definitions = dataset_ids - table_ids
    missing_points = table_ids - dataset_ids
    if missing_definitions:
        errors.append(
            f"dataset uses undefined flow case IDs {sorted(missing_definitions)}"
        )
    if missing_points:
        errors.append(
            f"flow cases have no collocation points {sorted(missing_points)}"
        )

    if len(cases) > 1:
        reynolds = np.asarray([case.reynolds for case in cases])
        directions = np.asarray(
            [[case.cos_alpha, case.sin_alpha] for case in cases]
        )
        geometry = {tuple(case.geometry_parameters) for case in cases}
        pressure_drop = {
            case.pressure_drop_coefficient for case in cases
        }
        if (
            not np.allclose(reynolds, reynolds[0])
            and not feature_assembler.reynolds_enabled
        ):
            errors.append(
                "multiple Reynolds numbers require the reynolds input feature"
            )
        if (
            not np.allclose(directions, directions[0])
            and not feature_assembler.angle_enabled
        ):
            errors.append(
                "multiple angles of attack require cos(alpha), sin(alpha) "
                "input features"
            )
        if len(geometry) > 1 and not feature_assembler.geometry_enabled:
            errors.append(
                "multiple geometries require geometry_parameters inputs"
            )
        if (
            len(pressure_drop) > 1
            and not feature_assembler.pressure_drop_enabled
        ):
            errors.append(
                "multiple pressure-drop coefficients require their input "
                "feature"
            )

    if feature_assembler.reynolds_enabled:
        outside = [
            case.case_id
            for case in cases
            if not (
                feature_assembler.reynolds_min
                <= case.reynolds
                <= feature_assembler.reynolds_max
            )
        ]
        if outside:
            errors.append(
                "flow-case Reynolds numbers must lie inside "
                "input_features.reynolds.range; outside case IDs "
                f"{outside}"
            )
    if feature_assembler.pressure_drop_enabled:
        missing_pressure = [
            case.case_id
            for case in cases
            if case.pressure_drop_coefficient is None
        ]
        if missing_pressure:
            errors.append(
                "pressure_drop_coefficient input is enabled but missing for "
                f"case IDs {missing_pressure}"
            )
    if feature_assembler.geometry_enabled:
        expected_geometry = len(
            feature_assembler.geometry_parameter_names
        )
        wrong_geometry = [
            case.case_id
            for case in cases
            if len(case.geometry_parameters) != expected_geometry
        ]
        if wrong_geometry:
            errors.append(
                "geometry parameter count does not match configured names "
                f"for case IDs {wrong_geometry}"
            )

    reference_scales = getattr(data_handler, "reference_scales", None)
    if reference_scales is None:
        errors.append("validated ReferenceScales metadata is required")
    if feature_assembler.sdf_enabled and not sdf_geometry_fingerprint:
        errors.append(
            "mandatory differentiable SDF geometry is unavailable"
        )
    if hasattr(data_handler, "cached_sdf_geometry_fingerprint"):
        cached_fingerprint = getattr(
            data_handler,
            "cached_sdf_geometry_fingerprint",
            None,
        )
        cached_method = getattr(data_handler, "cached_sdf_method", None)
        if cached_fingerprint != sdf_geometry_fingerprint:
            errors.append(
                "cached training SDF geometry fingerprint does not match "
                "the active canonical geometry"
            )
        if cached_method != "exact_polyline_segments":
            errors.append(
                "cached training SDF method must be "
                "exact_polyline_segments"
            )
    fvm_fingerprint = getattr(
        data_handler,
        "fvm_sdf_geometry_fingerprint",
        None,
    )
    if (
        fvm_fingerprint
        and sdf_geometry_fingerprint
        and fvm_fingerprint != sdf_geometry_fingerprint
    ):
        errors.append(
            "training SDF and FVM SDF use different geometry fingerprints"
        )
    if feature_assembler.manifest.input_dim < 3:
        errors.append("network input must contain x/c, y/c and SDF/c")

    _raise_errors(errors)
