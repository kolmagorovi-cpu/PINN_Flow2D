"""Validation and checkpoint reproducibility contract tests."""

from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import torch

from airfoil_pinn.config.manager import ConfigManager
from airfoil_pinn.config.validation import (
    ConfigurationError,
    validate_runtime_contract,
    validate_static_config,
)
from airfoil_pinn.data.flow import FlowCase
from airfoil_pinn.data.normalization import ReferenceScales
from airfoil_pinn.training.checkpoint import CheckpointManager
from airfoil_pinn.training.contract import (
    TrainingContract,
    build_training_contract,
    validate_checkpoint_contract,
)
from airfoil_pinn.training.input_features import InputFeatureAssembler


def _default_config() -> dict:
    """Return an isolated valid default configuration."""
    return copy.deepcopy(ConfigManager().config)


def _reference_scales() -> ReferenceScales:
    """Return deterministic physical scales with Re=1000."""
    return ReferenceScales(
        chord=1.0,
        velocity=1.0,
        density=1.0,
        dynamic_viscosity=1e-3,
        static_pressure=101_325.0,
    )


class StaticValidationTests(unittest.TestCase):
    """Reject malformed graph and physical configuration before I/O."""

    def test_default_configuration_is_valid(self) -> None:
        config = _default_config()
        validate_static_config(config)
        self.assertNotIn("boundary_types", config["input_params"])
        self.assertFalse(config["training"]["full_epoch_optimizer_step"])

    def test_full_epoch_optimizer_step_must_be_boolean(self) -> None:
        config = _default_config()
        config["training"]["full_epoch_optimizer_step"] = "yes"
        with self.assertRaisesRegex(
            ConfigurationError,
            "full_epoch_optimizer_step",
        ):
            validate_static_config(config)

    def test_coordinate_scaling_mode_is_validated(self) -> None:
        config = _default_config()
        config["architecture"]["coordinate_scaling"].update(
            {"enabled": True, "mode": "unknown"}
        )
        with self.assertRaisesRegex(ConfigurationError, "coordinate_scaling.mode"):
            validate_static_config(config)

    def test_legacy_boundary_feature_flags_are_removed_on_load(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "legacy.yaml")
            path.write_text(
                "input_params:\n"
                "  boundary_types: true\n"
                "input_features:\n"
                "  boundary_types:\n"
                "    enabled: true\n",
                encoding="utf-8",
            )
            manager = ConfigManager()
            manager.load_from_file(str(path))

        self.assertNotIn("boundary_types", manager.config["input_params"])
        self.assertNotIn("boundary_types", manager.config["input_features"])

    def test_mandatory_sdf_cannot_be_disabled(self) -> None:
        config = _default_config()
        config["input_params"]["sdf"] = False
        with self.assertRaisesRegex(ConfigurationError, "SDF"):
            validate_static_config(config)

    def test_invalid_reynolds_range_is_rejected(self) -> None:
        config = _default_config()
        config["input_features"] = {
            "reynolds": {
                "enabled": True,
                "range": [1e6, 1e4],
            }
        }
        with self.assertRaisesRegex(ConfigurationError, "reynolds"):
            validate_static_config(config)

    def test_duplicate_geometry_names_are_rejected(self) -> None:
        config = _default_config()
        config["input_features"] = {
            "geometry_parameters": {
                "enabled": True,
                "names": ["thickness", "thickness"],
            }
        }
        with self.assertRaisesRegex(ConfigurationError, "unique"):
            validate_static_config(config)

    def test_checkpoint_load_mode_is_validated(self) -> None:
        config = _default_config()
        config["checkpoint"]["load_mode"] = "continue_somehow"
        with self.assertRaisesRegex(ConfigurationError, "load_mode"):
            validate_static_config(config)

    def test_raw_pressure_output_bias_must_be_zero(self) -> None:
        config = _default_config()
        config["architecture"]["output_bias"] = [0.1, 0.0, 1.0]
        with self.assertRaisesRegex(ConfigurationError, r"output_bias\[2\]"):
            validate_static_config(config)

    def test_pressure_gauge_cannot_be_disabled(self) -> None:
        config = _default_config()
        config["loss_weights"]["pressure_gauge_weight"] = 0.0
        with self.assertRaisesRegex(ConfigurationError, "pressure_gauge_weight"):
            validate_static_config(config)


class RuntimeValidationTests(unittest.TestCase):
    """Ensure varying case parameters are visible to the neural network."""

    def test_multiple_angles_require_angle_features(self) -> None:
        config = _default_config()
        handler = SimpleNamespace(
            flow_cases={
                0: FlowCase(0, reynolds=1e5, angle_degrees=0.0),
                1: FlowCase(1, reynolds=1e5, angle_degrees=10.0),
            },
            dataset=SimpleNamespace(case_id=torch.tensor([0, 1])),
            reference_scales=_reference_scales(),
            fvm_sdf_geometry_fingerprint="geometry",
        )
        assembler = InputFeatureAssembler(config)
        with self.assertRaisesRegex(
            ConfigurationError,
            r"cos\(alpha\)",
        ):
            validate_runtime_contract(
                config,
                handler,
                assembler,
                sdf_geometry_fingerprint="geometry",
            )

    def test_complete_parametric_manifest_is_valid(self) -> None:
        config = _default_config()
        config["input_features"] = {
            "coordinates": {"enabled": True},
            "sdf": {"enabled": True},
            "reynolds": {"enabled": True, "range": [1e4, 1e6]},
            "angle_of_attack": {"enabled": True},
        }
        handler = SimpleNamespace(
            flow_cases={
                0: FlowCase(0, reynolds=1e5, angle_degrees=0.0),
                1: FlowCase(1, reynolds=2e5, angle_degrees=10.0),
            },
            dataset=SimpleNamespace(case_id=torch.tensor([0, 1])),
            reference_scales=_reference_scales(),
            fvm_sdf_geometry_fingerprint="geometry",
        )
        assembler = InputFeatureAssembler(config)
        validate_runtime_contract(
            config,
            handler,
            assembler,
            sdf_geometry_fingerprint="geometry",
        )

    def test_case_reynolds_must_be_inside_feature_range(self) -> None:
        config = _default_config()
        config["input_features"] = {
            "coordinates": {"enabled": True},
            "sdf": {"enabled": True},
            "reynolds": {"enabled": True, "range": [1e4, 1e5]},
        }
        handler = SimpleNamespace(
            flow_cases={
                0: FlowCase(0, reynolds=2e5),
            },
            dataset=SimpleNamespace(case_id=torch.tensor([0])),
            reference_scales=_reference_scales(),
            fvm_sdf_geometry_fingerprint="geometry",
        )
        assembler = InputFeatureAssembler(config)
        with self.assertRaisesRegex(ConfigurationError, "outside case IDs"):
            validate_runtime_contract(
                config,
                handler,
                assembler,
                sdf_geometry_fingerprint="geometry",
            )


def _training_contract(
    angle: float = 0.0,
    *,
    wall_enforcement: str = "soft",
    distance_scale: float = 0.02,
) -> TrainingContract:
    """Build one realistic contract with a selectable flow angle."""
    config = _default_config()
    config["boundary_conditions"]["wall"].update(
        {
            "enforcement": wall_enforcement,
            "distance_scale": distance_scale,
        }
    )
    assembler = InputFeatureAssembler(config)
    handler = SimpleNamespace(
        reference_scales=_reference_scales(),
        flow_cases={
            0: FlowCase(
                0,
                reynolds=1e5,
                angle_degrees=angle,
            )
        },
        boundary_type_names=["interior"],
        fvm_sdf_geometry_fingerprint="geometry",
        cached_sdf_geometry_fingerprint="geometry",
        cached_sdf_method="exact_polyline_segments",
    )
    loss = SimpleNamespace(
        feature_manifest=assembler.manifest,
        sdf_geometry_fingerprint="geometry",
        sdf_float64_consistency_tolerance=1e-12,
    )
    return build_training_contract(config, handler, loss)


class CheckpointContractTests(unittest.TestCase):
    """Check contract integrity, incompatibility detection and atomic saves."""

    def test_contract_roundtrip_and_mismatch(self) -> None:
        current = _training_contract(angle=0.0)
        self.assertEqual(
            current.payload["sdf_value_contract"],
            {
                "method": "exact_polyline_segments",
                "evaluation_dtype": "float64",
                "consistency_tolerance": 1e-12,
                "cached_geometry_fingerprint": "geometry",
                "cached_method": "exact_polyline_segments",
            },
        )
        restored = TrainingContract.from_metadata(current.to_metadata())
        self.assertEqual(restored.fingerprint(), current.fingerprint())
        validate_checkpoint_contract(current.to_metadata(), current)

        changed = _training_contract(angle=5.0)
        with self.assertRaisesRegex(RuntimeError, "flow_cases"):
            validate_checkpoint_contract(
                current.to_metadata(),
                changed,
            )

    def test_corrupted_contract_is_rejected(self) -> None:
        metadata = _training_contract().to_metadata()
        metadata["fingerprint"] = "not-a-valid-fingerprint"
        with self.assertRaisesRegex(ValueError, "corrupted"):
            TrainingContract.from_metadata(metadata)

    def test_boundary_transform_changes_checkpoint_fingerprint(self) -> None:
        soft = _training_contract()
        hard = _training_contract(wall_enforcement="hard")
        other_scale = _training_contract(
            wall_enforcement="hard",
            distance_scale=0.05,
        )

        self.assertNotEqual(soft.fingerprint(), hard.fingerprint())
        self.assertNotEqual(hard.fingerprint(), other_scale.fingerprint())
        with self.assertRaisesRegex(RuntimeError, "boundary_enforcement"):
            validate_checkpoint_contract(
                hard.to_metadata(),
                other_scale,
            )

    def test_checkpoint_save_is_atomic_and_versioned(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manager = CheckpointManager(directory)
            model = torch.nn.Linear(2, 1)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            contract = _training_contract()
            manager.save(
                filename="state.pt",
                model=model,
                optimizer=optimizer,
                scheduler=None,
                loss_weights={},
                history={},
                extra_state={
                    "training_contract": contract.to_metadata(),
                },
            )
            checkpoint = manager.load(
                f"{directory}/state.pt",
                map_location=torch.device("cpu"),
            )
            self.assertEqual(
                checkpoint["checkpoint_schema"]["version"],
                2,
            )
            self.assertFalse(Path(directory, "state.pt.tmp").exists())
            validate_checkpoint_contract(
                checkpoint["training_contract"],
                contract,
            )


if __name__ == "__main__":
    unittest.main()
