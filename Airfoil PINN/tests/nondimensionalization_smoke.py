"""Numerical tests for the incompressible CFD normalization contract."""

from __future__ import annotations

import unittest

import numpy as np

from airfoil_pinn.data.normalization import (
    NORMALIZATION_SCHEMA,
    NORMALIZATION_VERSION,
    ReferenceScales,
)


class ReferenceScalesTests(unittest.TestCase):
    """Verify dimensional round trips, metadata migration and Pi similarity."""

    def setUp(self) -> None:
        """Create a representative air-at-low-speed reference state."""
        self.scales = ReferenceScales(
            chord=1.5,
            velocity=12.0,
            density=1.204,
            dynamic_viscosity=1.825e-5,
            static_pressure=101_325.0,
        )

    def test_dimensional_round_trips(self) -> None:
        """Every supported physical quantity must round-trip exactly enough."""
        length = np.array([-2.0, 0.0, 0.75, 3.0])
        velocity = np.array([-1.0, 0.0, 6.0, 12.0])
        pressure = np.array([100_000.0, 101_325.0, 102_500.0])

        np.testing.assert_allclose(
            self.scales.length_to_dimensional(
                self.scales.length_to_dimensionless(length)
            ),
            length,
            rtol=1e-14,
            atol=1e-14,
        )
        np.testing.assert_allclose(
            self.scales.velocity_to_dimensional(
                self.scales.velocity_to_dimensionless(velocity)
            ),
            velocity,
            rtol=1e-14,
            atol=1e-14,
        )
        np.testing.assert_allclose(
            self.scales.coefficient_to_pressure(
                self.scales.pressure_to_coefficient(pressure)
            ),
            pressure,
            rtol=1e-14,
            atol=1e-10,
        )

    def test_pressure_coefficient_reference(self) -> None:
        """Static reference pressure maps to zero and q_inf maps to Cp=1."""
        self.assertEqual(
            self.scales.pressure_to_coefficient(
                self.scales.static_pressure
            ),
            0.0,
        )
        pressure = self.scales.static_pressure + self.scales.dynamic_pressure
        self.assertAlmostEqual(
            self.scales.pressure_to_coefficient(pressure),
            1.0,
        )

    def test_versioned_metadata_round_trip(self) -> None:
        """Current metadata preserves values and its stable identity."""
        metadata = self.scales.to_metadata()
        restored = ReferenceScales.from_metadata(
            {"reference_scales": metadata}
        )

        self.assertEqual(metadata["schema"], NORMALIZATION_SCHEMA)
        self.assertEqual(metadata["version"], NORMALIZATION_VERSION)
        self.assertEqual(restored, self.scales)
        self.assertEqual(restored.fingerprint(), self.scales.fingerprint())

    def test_legacy_metadata_migration(self) -> None:
        """Flat metadata produced by older project versions stays readable."""
        restored = ReferenceScales.from_metadata(
            {
                "chord": 2.0,
                "U_inf": 5.0,
                "rho": 1.2,
                "mu": 1.8e-5,
                "p_ref": 100_000.0,
            }
        )

        self.assertEqual(restored.chord, 2.0)
        self.assertEqual(restored.velocity, 5.0)
        self.assertAlmostEqual(restored.reynolds, 1.2 * 5.0 * 2.0 / 1.8e-5)

    def test_buckingham_pi_invariance(self) -> None:
        """Different dimensional scales with equal Re create equal Pi data."""
        first = ReferenceScales(
            chord=1.0,
            velocity=10.0,
            density=1.0,
            dynamic_viscosity=1e-4,
            static_pressure=100_000.0,
        )
        second = ReferenceScales(
            chord=2.0,
            velocity=5.0,
            density=2.0,
            dynamic_viscosity=2e-4,
            static_pressure=90_000.0,
        )

        self.assertAlmostEqual(first.reynolds, second.reynolds)
        np.testing.assert_allclose(
            first.length_to_dimensionless(np.array([0.25, 0.75])),
            second.length_to_dimensionless(np.array([0.5, 1.5])),
        )
        np.testing.assert_allclose(
            first.velocity_to_dimensionless(np.array([5.0, 10.0])),
            second.velocity_to_dimensionless(np.array([2.5, 5.0])),
        )
        np.testing.assert_allclose(
            first.pressure_to_coefficient(
                first.static_pressure
                + np.array([0.0, 0.5]) * first.dynamic_pressure
            ),
            second.pressure_to_coefficient(
                second.static_pressure
                + np.array([0.0, 0.5]) * second.dynamic_pressure
            ),
        )

    def test_invalid_reference_values_fail_fast(self) -> None:
        """Undefined normalization must never be silently substituted."""
        with self.assertRaisesRegex(ValueError, "velocity"):
            ReferenceScales(
                chord=1.0,
                velocity=0.0,
                density=1.0,
                dynamic_viscosity=1e-5,
                static_pressure=0.0,
            )
        with self.assertRaisesRegex(ValueError, "dynamic_viscosity"):
            ReferenceScales(
                chord=1.0,
                velocity=1.0,
                density=1.0,
                dynamic_viscosity=float("nan"),
                static_pressure=0.0,
            )


if __name__ == "__main__":
    unittest.main()
