"""Tests for FlowCase, typed batches and manifest-driven PINN inputs."""

from __future__ import annotations

import math
import unittest

import torch

from airfoil_pinn.data.batch import PINNBatch, collate_pinn_batches
from airfoil_pinn.data.flow import FlowCase, FlowCaseBatch
from airfoil_pinn.data.handler import DataHandler
from airfoil_pinn.training.input_features import InputFeatureAssembler


class FlowCaseTests(unittest.TestCase):
    """Verify angular representation and dimensionless case validation."""

    def test_freestream_direction_and_metadata(self) -> None:
        """Components map to a stable sin/cos angle representation."""
        case = FlowCase.from_freestream_components(
            case_id=3,
            reynolds=250_000.0,
            u_normalized=math.cos(math.radians(10.0)),
            v_normalized=math.sin(math.radians(10.0)),
            name="alpha-10",
        )
        restored = FlowCase.from_metadata(case.to_metadata())

        self.assertAlmostEqual(case.angle_degrees, 10.0)
        self.assertAlmostEqual(case.cos_alpha, math.cos(math.radians(10.0)))
        self.assertAlmostEqual(case.sin_alpha, math.sin(math.radians(10.0)))
        self.assertEqual(restored, case)
        self.assertEqual(restored.fingerprint(), case.fingerprint())

    def test_non_unit_freestream_is_rejected(self) -> None:
        """Normalized components must encode direction, not dimensional speed."""
        with self.assertRaisesRegex(ValueError, "unit magnitude"):
            FlowCase.from_freestream_components(
                case_id=0,
                reynolds=100_000.0,
                u_normalized=2.0,
                v_normalized=0.0,
            )

    def test_data_handler_materializes_multiple_cases(self) -> None:
        """Point case IDs select their own Re and freestream direction."""
        handler = DataHandler.__new__(DataHandler)
        handler.flow_cases = {
            2: FlowCase(
                case_id=2,
                reynolds=1e5,
                angle_degrees=0.0,
            ),
            7: FlowCase(
                case_id=7,
                reynolds=5e5,
                angle_degrees=30.0,
            ),
        }
        batch = handler.get_flow_case_batch(
            torch.tensor([7, 2, 7]),
            device=torch.device("cpu"),
            dtype=torch.float64,
        )

        torch.testing.assert_close(
            batch.reynolds,
            torch.tensor([[5e5], [1e5], [5e5]], dtype=torch.float64),
        )
        torch.testing.assert_close(
            batch.cos_alpha,
            torch.tensor(
                [
                    [math.cos(math.radians(30.0))],
                    [1.0],
                    [math.cos(math.radians(30.0))],
                ],
                dtype=torch.float64,
            ),
        )


class TypedBatchTests(unittest.TestCase):
    """Verify custom collation and historical tuple unpacking."""

    @staticmethod
    def _sample(index: int, case_id: int) -> PINNBatch:
        return PINNBatch(
            index=torch.tensor(index),
            x=torch.tensor([float(index)]),
            y=torch.tensor([0.0]),
            sdf=torch.tensor([0.1]),
            boundary_types=torch.tensor([1.0, 0.0]),
            mask_u_fixed=torch.tensor([False]),
            mask_v_fixed=torch.tensor([False]),
            mask_p_fixed=torch.tensor([False]),
            u_bc=torch.tensor([0.0]),
            v_bc=torch.tensor([0.0]),
            p_bc=torch.tensor([0.0]),
            case_id=torch.tensor(case_id),
        )

    def test_collation_preserves_named_case_ids(self) -> None:
        """The new field is batched without breaking eleven-value unpacking."""
        batch = collate_pinn_batches(
            [self._sample(0, 2), self._sample(1, 4)]
        )
        legacy = tuple(batch)

        self.assertEqual(len(batch), 11)
        self.assertEqual(len(legacy), 11)
        torch.testing.assert_close(batch.index, torch.tensor([0, 1]))
        torch.testing.assert_close(batch.case_id, torch.tensor([2, 4]))
        torch.testing.assert_close(legacy[1], batch.x)


class InputFeatureAssemblerTests(unittest.TestCase):
    """Verify exact feature order, encoding, gradients and validation."""

    def setUp(self) -> None:
        """Create a configuration exercising all supported Pi features."""
        self.config = {
            "input_params": {
                "coordinates": True,
                "sdf": True,
                "boundary_types": True,
            },
            "input_features": {
                "coordinates": {"enabled": True},
                "sdf": {"enabled": True},
                "reynolds": {
                    "enabled": True,
                    "range": [1e4, 1e6],
                },
                "angle_of_attack": {"enabled": True},
                "boundary_types": {"enabled": True},
                "pressure_drop_coefficient": {"enabled": True},
                "geometry_parameters": {
                    "enabled": True,
                    "names": ["thickness_ratio", "camber_ratio"],
                },
            },
        }

    def test_manifest_and_encoded_values(self) -> None:
        """Columns follow the manifest and Reynolds uses log-minmax encoding."""
        assembler = InputFeatureAssembler(
            self.config,
            ["interior", "wall"],
        )
        x = torch.tensor([[0.1], [0.2]], requires_grad=True)
        y = torch.tensor([[0.3], [0.4]], requires_grad=True)
        sdf = (x + y).requires_grad_(True)
        boundary = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        flow = FlowCaseBatch(
            case_id=torch.tensor([[0], [1]]),
            reynolds=torch.tensor([[1e4], [1e6]]),
            cos_alpha=torch.tensor([[1.0], [0.0]]),
            sin_alpha=torch.tensor([[0.0], [1.0]]),
            pressure_drop_coefficient=torch.tensor([[0.2], [0.4]]),
            geometry_parameters=torch.tensor([[0.12, 0.0], [0.15, 0.02]]),
        )

        result = assembler.build(x, y, sdf, boundary, flow)
        names = [feature.name for feature in assembler.manifest.features]

        self.assertEqual(
            names,
            [
                "coordinates",
                "sdf",
                "reynolds",
                "angle_of_attack",
                "boundary_types",
                "pressure_drop_coefficient",
                "geometry_parameters",
            ],
        )
        self.assertEqual(result.shape, (2, 11))
        torch.testing.assert_close(result[:, 3], torch.tensor([-1.0, 1.0]))
        torch.testing.assert_close(result[:, 4:6], torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
        self.assertEqual(
            assembler.manifest.fingerprint(),
            InputFeatureAssembler(
                self.config,
                ["interior", "wall"],
            ).manifest.fingerprint(),
        )

        derivative = torch.autograd.grad(result.sum(), x)[0]
        torch.testing.assert_close(derivative, torch.full_like(x, 2.0))

    def test_legacy_pi_migration(self) -> None:
        """Legacy viscosity/inlet flags become Re and angle, density vanishes."""
        assembler = InputFeatureAssembler(
            {
                "input_params": {
                    "coordinates": True,
                    "sdf": True,
                    "boundary_types": False,
                    "inlet_velocity": True,
                    "viscosity": True,
                    "density": True,
                }
            },
            [],
        )
        names = [feature.name for feature in assembler.manifest.features]

        self.assertEqual(
            names,
            ["coordinates", "sdf", "reynolds", "angle_of_attack"],
        )
        self.assertEqual(assembler.manifest.input_dim, 6)

    def test_mandatory_features_cannot_be_disabled(self) -> None:
        """Differential coordinates and the requested SDF are fail-fast."""
        with self.assertRaisesRegex(ValueError, "Coordinates"):
            InputFeatureAssembler(
                {
                    "input_params": {
                        "coordinates": False,
                        "sdf": True,
                    }
                },
                [],
            )
        with self.assertRaisesRegex(ValueError, "mandatory"):
            InputFeatureAssembler(
                {
                    "input_params": {
                        "coordinates": True,
                        "sdf": False,
                    }
                },
                [],
            )


if __name__ == "__main__":
    unittest.main()
