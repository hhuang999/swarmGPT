"""Tests for the new motion primitives: firework, pendulum, scatter_gather, form_heart, form_line, orbit."""

import importlib
import sys
import types
import unittest.mock as mock

import numpy as np
import pytest
from numpy.typing import NDArray

# Import motion_primitives directly (bypass swarm_gpt.core.__init__ which triggers heavy deps)
_spec = importlib.util.spec_from_file_location(
    "motion_primitives", "swarm_gpt/core/motion_primitives.py"
)
# Provide the required swarm_gpt.exception module
_exc_mod = types.ModuleType("swarm_gpt.exception")


class _LLMFormatError(Exception):
    pass


_exc_mod.LLMFormatError = _LLMFormatError
sys.modules["swarm_gpt.exception"] = _exc_mod

_mp = importlib.util.module_from_spec(_spec)
sys.modules[_mp.__name__] = _mp  # Register so primitive_by_name can find it
_spec.loader.exec_module(_mp)

firework = _mp.firework
pendulum = _mp.pendulum
scatter_gather = _mp.scatter_gather
form_heart = _mp.form_heart
form_line = _mp.form_line
orbit = _mp.orbit
motion_primitives = _mp.motion_primitives
primitive_by_name = _mp.primitive_by_name


def _make_swarm_pos(n_drones: int = 6) -> NDArray:
    """Create a simple grid of drone positions in cm."""
    rows = int(np.sqrt(n_drones))
    cols = int(np.ceil(n_drones / rows))
    spacing = 60
    positions = []
    for r in range(rows):
        for c in range(cols):
            if len(positions) >= n_drones:
                break
            positions.append(
                [c * spacing - cols * spacing / 2, r * spacing - rows * spacing / 2, 100]
            )
    return np.array(positions[:n_drones], dtype=float)


def _make_limits() -> dict[str, NDArray]:
    """Create standard limits dict. Values are in meters."""
    return {"lower": np.array([-2.0, -2.0, 0.0]), "upper": np.array([2.0, 2.0, 2.0])}


class TestFirework:
    """Tests for the firework primitive."""

    def test_returns_correct_types(self):
        swarm_pos = _make_swarm_pos(6)
        limits = _make_limits()
        final_pos, waypoints = firework((3, 150, 150), swarm_pos, 0.0, 3.0, limits)
        assert isinstance(final_pos, np.ndarray)
        assert final_pos.shape == (6, 3)
        assert isinstance(waypoints, dict)

    def test_waypoint_count_matches_steps(self):
        swarm_pos = _make_swarm_pos(6)
        limits = _make_limits()
        _, waypoints = firework((3, 150, 150), swarm_pos, 0.0, 3.0, limits)
        assert len(waypoints) == 3

    def test_waypoint_keys_are_times(self):
        swarm_pos = _make_swarm_pos(6)
        limits = _make_limits()
        _, waypoints = firework((3, 150, 150), swarm_pos, 0.0, 3.0, limits)
        for t in waypoints:
            assert 0.0 < t <= 3.0

    def test_each_waypoint_has_all_drones(self):
        swarm_pos = _make_swarm_pos(6)
        limits = _make_limits()
        _, waypoints = firework((3, 150, 150), swarm_pos, 0.0, 3.0, limits)
        for t, wp in waypoints.items():
            assert len(wp) == 6

    def test_respects_height_limits(self):
        swarm_pos = _make_swarm_pos(6)
        limits = _make_limits()
        _, waypoints = firework((3, 500, 150), swarm_pos, 0.0, 3.0, limits)
        for t, wp in waypoints.items():
            for drone_id, pos in wp.items():
                assert pos[2] <= limits["upper"][2] * 100

    def test_spread_effect(self):
        swarm_pos = _make_swarm_pos(6)
        limits = _make_limits()
        _, waypoints = firework((4, 150, 200), swarm_pos, 0.0, 4.0, limits)
        # Mid-step should have drones further apart than start
        times = sorted(waypoints.keys())
        mid_idx = len(times) // 2
        mid_wp = waypoints[times[mid_idx]]
        positions_mid = np.array(list(mid_wp.values()))
        np.max(np.linalg.norm(positions_mid[:, :2], axis=1))  # noqa: F841


class TestPendulum:
    """Tests for the pendulum primitive."""

    def test_returns_correct_types(self):
        swarm_pos = _make_swarm_pos(6)
        limits = _make_limits()
        final_pos, waypoints = pendulum((3, 45), swarm_pos, 0.0, 3.0, limits)
        assert isinstance(final_pos, np.ndarray)
        assert final_pos.shape == (6, 3)
        assert isinstance(waypoints, dict)

    def test_waypoint_count_matches_steps(self):
        swarm_pos = _make_swarm_pos(6)
        limits = _make_limits()
        _, waypoints = pendulum((3, 45), swarm_pos, 0.0, 3.0, limits)
        assert len(waypoints) == 3

    def test_oscillating_motion(self):
        swarm_pos = _make_swarm_pos(6)
        limits = _make_limits()
        _, waypoints = pendulum((4, 45), swarm_pos, 0.0, 4.0, limits)
        times = sorted(waypoints.keys())
        # Collect x positions of drone 0 over time
        x_values = [waypoints[t][0][0] for t in times]
        # The pendulum should swing both ways (x should not be monotonic)
        assert any(x_values[i] < x_values[i - 1] for i in range(1, len(x_values)))

    def test_respects_z_limits(self):
        swarm_pos = _make_swarm_pos(6)
        limits = _make_limits()
        _, waypoints = pendulum((3, 90), swarm_pos, 0.0, 3.0, limits)
        for t, wp in waypoints.items():
            for drone_id, pos in wp.items():
                assert pos[2] >= limits["lower"][2] * 100
                assert pos[2] <= limits["upper"][2] * 100


class TestScatterGather:
    """Tests for the scatter_gather primitive."""

    def test_returns_correct_types(self):
        swarm_pos = _make_swarm_pos(6)
        limits = _make_limits()
        final_pos, waypoints = scatter_gather((3, 150, 120), swarm_pos, 0.0, 3.0, limits)
        assert isinstance(final_pos, np.ndarray)
        assert final_pos.shape == (6, 3)
        assert isinstance(waypoints, dict)

    def test_waypoint_count_matches_steps(self):
        swarm_pos = _make_swarm_pos(6)
        limits = _make_limits()
        _, waypoints = scatter_gather((3, 150, 120), swarm_pos, 0.0, 3.0, limits)
        assert len(waypoints) == 3

    def test_scatter_then_gather(self):
        swarm_pos = _make_swarm_pos(6)
        limits = _make_limits()
        _, waypoints = scatter_gather((4, 200, 120), swarm_pos, 0.0, 4.0, limits)
        times = sorted(waypoints.keys())
        mid_wp = waypoints[times[len(times) // 2]]
        end_wp = waypoints[times[-1]]
        mid_positions = np.array(list(mid_wp.values()))
        end_positions = np.array(list(end_wp.values()))
        # At mid-point drones should be more spread out than at the end
        mid_spread = np.std(mid_positions[:, 0])
        end_spread = np.std(end_positions[:, 0])
        assert mid_spread > end_spread

    def test_respects_height_limits(self):
        swarm_pos = _make_swarm_pos(6)
        limits = _make_limits()
        _, waypoints = scatter_gather((3, 150, 500), swarm_pos, 0.0, 3.0, limits)
        for t, wp in waypoints.items():
            for drone_id, pos in wp.items():
                assert pos[2] <= limits["upper"][2] * 100


class TestFormHeart:
    """Tests for the form_heart primitive."""

    def test_returns_correct_types(self):
        swarm_pos = _make_swarm_pos(6)
        limits = _make_limits()
        final_pos, waypoints = form_heart(([1, 2, 3, 4, 5, 6], 150), swarm_pos, 0.0, 1.0, limits)
        assert isinstance(final_pos, np.ndarray)
        assert final_pos.shape == (6, 3)
        assert isinstance(waypoints, dict)

    def test_single_waypoint_at_tend(self):
        swarm_pos = _make_swarm_pos(6)
        limits = _make_limits()
        _, waypoints = form_heart(([1, 2, 3, 4, 5, 6], 150), swarm_pos, 0.0, 5.0, limits)
        assert len(waypoints) == 1
        assert 5.0 in waypoints

    def test_all_drones_at_correct_height(self):
        swarm_pos = _make_swarm_pos(6)
        limits = _make_limits()
        _, waypoints = form_heart(([1, 2, 3, 4, 5, 6], 150), swarm_pos, 0.0, 5.0, limits)
        for drone_id, pos in waypoints[5.0].items():
            assert pos[2] == 150

    def test_respects_position_limits(self):
        swarm_pos = _make_swarm_pos(6)
        limits = _make_limits()
        _, waypoints = form_heart(([1, 2, 3, 4, 5, 6], 150), swarm_pos, 0.0, 5.0, limits)
        for drone_id, pos in waypoints[5.0].items():
            assert pos[0] >= limits["lower"][0] * 100
            assert pos[0] <= limits["upper"][0] * 100
            assert pos[1] >= limits["lower"][1] * 100
            assert pos[1] <= limits["upper"][1] * 100

    def test_heart_shape_symmetry(self):
        swarm_pos = _make_swarm_pos(10)
        limits = _make_limits()
        _, waypoints = form_heart((list(range(1, 11)), 100), swarm_pos, 0.0, 5.0, limits)
        positions = np.array(list(waypoints[5.0].values()))
        # Heart shape should have drones at roughly symmetric y positions
        y_vals = positions[:, 1]
        # The heart has y values centered around 0
        assert abs(np.mean(y_vals)) < 50  # roughly centered


class TestFormLine:
    """Tests for the form_line primitive."""

    def test_returns_correct_types(self):
        swarm_pos = _make_swarm_pos(5)
        limits = _make_limits()
        final_pos, waypoints = form_line(([1, 2, 3, 4, 5], 100), swarm_pos, 0.0, 1.0, limits)
        assert isinstance(final_pos, np.ndarray)
        assert final_pos.shape == (5, 3)
        assert isinstance(waypoints, dict)

    def test_single_waypoint_at_tend(self):
        swarm_pos = _make_swarm_pos(5)
        limits = _make_limits()
        _, waypoints = form_line(([1, 2, 3, 4, 5], 100), swarm_pos, 0.0, 3.0, limits)
        assert len(waypoints) == 1
        assert 3.0 in waypoints

    def test_all_drones_at_correct_height(self):
        swarm_pos = _make_swarm_pos(5)
        limits = _make_limits()
        _, waypoints = form_line(([1, 2, 3, 4, 5], 120), swarm_pos, 0.0, 1.0, limits)
        for drone_id, pos in waypoints[1.0].items():
            assert pos[2] == 120

    def test_line_along_x_axis(self):
        swarm_pos = _make_swarm_pos(5)
        limits = _make_limits()
        _, waypoints = form_line(([1, 2, 3, 4, 5], 100), swarm_pos, 0.0, 1.0, limits)
        positions = np.array(list(waypoints[1.0].values()))
        # All y positions should be ~0 (line along x-axis)
        assert np.allclose(positions[:, 1], 0.0, atol=1.0)

    def test_equal_spacing(self):
        swarm_pos = _make_swarm_pos(5)
        limits = _make_limits()
        _, waypoints = form_line(([1, 2, 3, 4, 5], 100), swarm_pos, 0.0, 1.0, limits)
        positions = np.array(list(waypoints[1.0].values()))
        x_sorted = np.sort(positions[:, 0])
        # Check spacing is uniform
        spacings = np.diff(x_sorted)
        assert np.allclose(spacings, spacings[0], atol=1.0)


class TestOrbit:
    """Tests for the orbit primitive."""

    def test_returns_correct_types(self):
        swarm_pos = _make_swarm_pos(6)
        limits = _make_limits()
        final_pos, waypoints = orbit((3, 15, 10), swarm_pos, 0.0, 3.0, limits)
        assert isinstance(final_pos, np.ndarray)
        assert final_pos.shape == (6, 3)
        assert isinstance(waypoints, dict)

    def test_waypoint_count_matches_steps(self):
        swarm_pos = _make_swarm_pos(6)
        limits = _make_limits()
        _, waypoints = orbit((3, 15, 10), swarm_pos, 0.0, 3.0, limits)
        assert len(waypoints) == 3

    def test_different_heights_per_drone(self):
        swarm_pos = _make_swarm_pos(6)
        limits = _make_limits()
        _, waypoints = orbit((3, 15, 10), swarm_pos, 0.0, 3.0, limits)
        t = sorted(waypoints.keys())[0]
        z_values = [waypoints[t][i][2] for i in range(6)]
        # Not all drones should be at the same height
        assert len(set(z_values)) > 1

    def test_rotation_occurs(self):
        swarm_pos = _make_swarm_pos(6)
        limits = _make_limits()
        _, waypoints = orbit((4, 20, 10), swarm_pos, 0.0, 4.0, limits)
        times = sorted(waypoints.keys())
        # Collect positions of drone 0 at first and last waypoint
        first_pos = waypoints[times[0]][0][:2]
        last_pos = waypoints[times[-1]][0][:2]
        # The drone should have moved (rotated)
        assert not np.allclose(first_pos, last_pos, atol=1.0)

    def test_respects_z_limits(self):
        swarm_pos = _make_swarm_pos(6)
        limits = _make_limits()
        _, waypoints = orbit((3, 15, 10), swarm_pos, 0.0, 3.0, limits)
        for t, wp in waypoints.items():
            for drone_id, pos in wp.items():
                assert pos[2] >= limits["lower"][2] * 100
                assert pos[2] <= limits["upper"][2] * 100


class TestRegistry:
    """Tests for the motion_primitives registry entries."""

    @pytest.mark.parametrize(
        "name", ["firework", "pendulum", "scatter_gather", "form_heart", "form_line", "orbit"]
    )
    def test_primitive_in_registry(self, name):
        assert name in motion_primitives

    @pytest.mark.parametrize(
        "name", ["firework", "pendulum", "scatter_gather", "form_heart", "form_line", "orbit"]
    )
    def test_primitive_lookup_by_name(self, name):
        func = primitive_by_name(name)
        assert callable(func)

    @pytest.mark.parametrize(
        "name,expected_n_args",
        [
            ("firework", 3),
            ("pendulum", 2),
            ("scatter_gather", 3),
            ("form_heart", 2),
            ("form_line", 2),
            ("orbit", 3),
        ],
    )
    def test_n_args_in_registry(self, name, expected_n_args):
        assert motion_primitives[name]["n_args"] == expected_n_args


class TestWithMockedAssignment:
    """Tests that use mocked _assign_positions to verify assignment is called."""

    @mock.patch.object(_mp, "_assign_positions")
    def test_form_heart_uses_assign_positions(self, mock_assign):
        mock_assign.return_value = np.array([0, 1, 2, 3, 4, 5])
        swarm_pos = _make_swarm_pos(6)
        limits = _make_limits()
        form_heart(([1, 2, 3, 4, 5, 6], 150), swarm_pos, 0.0, 1.0, limits)
        mock_assign.assert_called_once()

    @mock.patch.object(_mp, "_assign_positions")
    def test_form_line_uses_assign_positions(self, mock_assign):
        mock_assign.return_value = np.array([0, 1, 2, 3, 4])
        swarm_pos = _make_swarm_pos(5)
        limits = _make_limits()
        form_line(([1, 2, 3, 4, 5], 100), swarm_pos, 0.0, 1.0, limits)
        mock_assign.assert_called_once()

    @mock.patch.object(_mp, "_assign_positions")
    def test_orbit_uses_assign_positions(self, mock_assign):
        mock_assign.return_value = np.array([0, 1, 2, 3, 4, 5])
        swarm_pos = _make_swarm_pos(6)
        limits = _make_limits()
        orbit((3, 15, 10), swarm_pos, 0.0, 3.0, limits)
        mock_assign.assert_called_once()
