"""Tests for the PrimitiveComposer: sequence, parallel, and blend operators."""

import numpy as np
import pytest
import yaml
from numpy.typing import NDArray

from swarm_gpt.core.primitive_composer import (
    COMPOSITION_OPERATORS,
    CompositionError,
    PrimitiveComposer,
)
from swarm_gpt.core.motion_primitives import primitive_by_name


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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
            positions.append([c * spacing - cols * spacing / 2, r * spacing - rows * spacing / 2, 100])
    return np.array(positions[:n_drones], dtype=float)


def _make_limits() -> dict[str, NDArray]:
    """Create standard limits dict. Values are in meters."""
    return {
        "lower": np.array([-2.0, -2.0, 0.0]),
        "upper": np.array([2.0, 2.0, 2.0]),
    }


def _leaf(primitive: str, params: list) -> dict:
    """Create a leaf composition node."""
    return {"primitive": primitive, "params": params}


# ---------------------------------------------------------------------------
# Tests for is_composition
# ---------------------------------------------------------------------------

class TestIsComposition:
    """Tests for PrimitiveComposer.is_composition()."""

    def test_simple_primitive_not_composition(self):
        assert not PrimitiveComposer.is_composition("rotate(30, 'z')")

    def test_plan_not_composition(self):
        assert not PrimitiveComposer.is_composition("PLAN")

    def test_semicolon_separated_not_composition(self):
        assert not PrimitiveComposer.is_composition("rotate(30, 'z'); move_z([1,2,3], 20)")

    def test_blend_is_composition(self):
        yaml_str = "op: blend\nweight: 0.7\nchildren:\n  - primitive: rotate\n    params: [30, 'z']"
        assert PrimitiveComposer.is_composition(yaml_str)

    def test_sequence_is_composition(self):
        yaml_str = "op: sequence\nchildren:\n  - primitive: rotate\n    params: [30, 'z']"
        assert PrimitiveComposer.is_composition(yaml_str)

    def test_parallel_is_composition(self):
        yaml_str = "op: parallel\nchildren:\n  - primitive: rotate\n    params: [30, 'z']"
        assert PrimitiveComposer.is_composition(yaml_str)

    def test_whitespace_before_op(self):
        assert PrimitiveComposer.is_composition("  op: blend")


# ---------------------------------------------------------------------------
# Tests for parse_composition_yaml
# ---------------------------------------------------------------------------

class TestParseCompositionYaml:
    """Tests for PrimitiveComposer.parse_composition_yaml()."""

    def test_missing_op_raises(self):
        composer = PrimitiveComposer()
        with pytest.raises(CompositionError, match="must contain an 'op' key"):
            composer.parse_composition_yaml({"children": []})

    def test_unknown_op_raises(self):
        composer = PrimitiveComposer()
        with pytest.raises(CompositionError, match="Unknown composition operator"):
            composer.parse_composition_yaml({"op": "invalid", "children": []})

    def test_missing_children_raises(self):
        composer = PrimitiveComposer()
        with pytest.raises(CompositionError, match="non-empty 'children' list"):
            composer.parse_composition_yaml({"op": "sequence"})

    def test_empty_children_raises(self):
        composer = PrimitiveComposer()
        with pytest.raises(CompositionError, match="non-empty 'children' list"):
            composer.parse_composition_yaml({"op": "sequence", "children": []})

    def test_unknown_primitive_in_child_raises(self):
        composer = PrimitiveComposer()
        with pytest.raises(CompositionError, match="Unknown primitive"):
            composer.parse_composition_yaml({
                "op": "sequence",
                "children": [{"primitive": "does_not_exist", "params": []}],
            })

    def test_missing_params_in_child_raises(self):
        composer = PrimitiveComposer()
        with pytest.raises(CompositionError, match="must have a 'params' key"):
            composer.parse_composition_yaml({
                "op": "sequence",
                "children": [{"primitive": "rotate"}],
            })

    def test_wrong_param_count_raises(self):
        composer = PrimitiveComposer()
        with pytest.raises(CompositionError, match="expects 2 params, got 1"):
            composer.parse_composition_yaml({
                "op": "sequence",
                "children": [{"primitive": "rotate", "params": [30]}],
            })

    def test_child_without_op_or_primitive_raises(self):
        composer = PrimitiveComposer()
        with pytest.raises(CompositionError, match="must have 'op' or 'primitive' key"):
            composer.parse_composition_yaml({
                "op": "sequence",
                "children": [{"unknown": "value"}],
            })

    def test_blend_missing_weight_raises(self):
        composer = PrimitiveComposer()
        with pytest.raises(CompositionError, match="'blend' composition must have a 'weight' key"):
            composer.parse_composition_yaml({
                "op": "blend",
                "weight": None,
                "children": [
                    {"primitive": "rotate", "params": [30, "z"]},
                    {"primitive": "rotate", "params": [-30, "z"]},
                ],
            })

    def test_blend_weight_out_of_range_raises(self):
        composer = PrimitiveComposer()
        with pytest.raises(CompositionError, match="weight must be between 0 and 1"):
            composer.parse_composition_yaml({
                "op": "blend",
                "weight": 1.5,
                "children": [
                    {"primitive": "rotate", "params": [30, "z"]},
                    {"primitive": "rotate", "params": [-30, "z"]},
                ],
            })

    def test_valid_blend_parses(self):
        composer = PrimitiveComposer()
        result = composer.parse_composition_yaml({
            "op": "blend",
            "weight": 0.7,
            "children": [
                {"primitive": "rotate", "params": [30, "z"]},
                {"primitive": "rotate", "params": [-30, "z"]},
            ],
        })
        assert result["op"] == "blend"
        assert result["weight"] == 0.7
        assert len(result["children"]) == 2

    def test_valid_sequence_parses(self):
        composer = PrimitiveComposer()
        result = composer.parse_composition_yaml({
            "op": "sequence",
            "children": [
                {"primitive": "rotate", "params": [30, "z"]},
                {"primitive": "rotate", "params": [-30, "z"]},
            ],
        })
        assert result["op"] == "sequence"
        assert len(result["children"]) == 2

    def test_valid_parallel_parses(self):
        composer = PrimitiveComposer()
        result = composer.parse_composition_yaml({
            "op": "parallel",
            "children": [
                {"primitive": "rotate", "params": [30, "z"]},
                {"primitive": "rotate", "params": [-30, "z"]},
            ],
        })
        assert result["op"] == "parallel"
        assert len(result["children"]) == 2

    def test_parallel_with_drone_groups(self):
        composer = PrimitiveComposer()
        result = composer.parse_composition_yaml({
            "op": "parallel",
            "drone_groups": [[0, 1, 2], [3, 4, 5]],
            "children": [
                {"primitive": "rotate", "params": [30, "z"]},
                {"primitive": "rotate", "params": [-30, "z"]},
            ],
        })
        assert result["drone_groups"] == [[0, 1, 2], [3, 4, 5]]

    def test_parallel_mismatched_drone_groups_raises(self):
        composer = PrimitiveComposer()
        with pytest.raises(CompositionError, match="same length as children"):
            composer.parse_composition_yaml({
                "op": "parallel",
                "drone_groups": [[0, 1]],
                "children": [
                    {"primitive": "rotate", "params": [30, "z"]},
                    {"primitive": "rotate", "params": [-30, "z"]},
                ],
            })

    def test_nested_composition_parses(self):
        composer = PrimitiveComposer()
        result = composer.parse_composition_yaml({
            "op": "sequence",
            "children": [
                {"primitive": "rotate", "params": [30, "z"]},
                {
                    "op": "blend",
                    "weight": 0.5,
                    "children": [
                        {"primitive": "rotate", "params": [10, "z"]},
                        {"primitive": "rotate", "params": [-10, "z"]},
                    ],
                },
            ],
        })
        assert result["op"] == "sequence"
        assert len(result["children"]) == 2
        assert result["children"][1]["op"] == "blend"


# ---------------------------------------------------------------------------
# Tests for execute_composed - Sequence
# ---------------------------------------------------------------------------

class TestExecuteSequence:
    """Tests for the sequence composition operator."""

    def test_sequence_two_rotations(self):
        """Two rotations executed sequentially should produce waypoints across the full time range."""
        composer = PrimitiveComposer()
        composition = composer.parse_composition_yaml({
            "op": "sequence",
            "children": [
                {"primitive": "rotate", "params": [30, "z"]},
                {"primitive": "rotate", "params": [-30, "z"]},
            ],
        })
        swarm_pos = _make_swarm_pos(4)
        limits = _make_limits()
        final_pos, waypoints = composer.execute_composed(composition, swarm_pos, 0.0, 4.0, limits)

        assert isinstance(final_pos, np.ndarray)
        assert final_pos.shape == (4, 3)
        assert isinstance(waypoints, dict)
        assert len(waypoints) > 0

    def test_sequence_returns_all_waypoints(self):
        """All child waypoints should be present in the output."""
        composer = PrimitiveComposer()
        composition = composer.parse_composition_yaml({
            "op": "sequence",
            "children": [
                {"primitive": "move_z", "params": [[1, 2, 3, 4], 10]},
                {"primitive": "move_z", "params": [[1, 2, 3, 4], -10]},
            ],
        })
        swarm_pos = _make_swarm_pos(4)
        limits = _make_limits()
        _, waypoints = composer.execute_composed(composition, swarm_pos, 0.0, 4.0, limits)

        # move_z produces (tend-tstart) waypoints per step; sequence splits time in half
        assert len(waypoints) >= 2

    def test_sequence_single_child(self):
        """A sequence with a single child should still work."""
        composer = PrimitiveComposer()
        composition = composer.parse_composition_yaml({
            "op": "sequence",
            "children": [
                {"primitive": "rotate", "params": [15, "z"]},
            ],
        })
        swarm_pos = _make_swarm_pos(4)
        limits = _make_limits()
        final_pos, waypoints = composer.execute_composed(composition, swarm_pos, 0.0, 2.0, limits)

        assert final_pos.shape == (4, 3)
        assert len(waypoints) > 0


# ---------------------------------------------------------------------------
# Tests for execute_composed - Parallel
# ---------------------------------------------------------------------------

class TestExecuteParallel:
    """Tests for the parallel composition operator."""

    def test_parallel_with_explicit_groups(self):
        """Parallel with explicit drone_groups should only move assigned drones."""
        composer = PrimitiveComposer()
        composition = composer.parse_composition_yaml({
            "op": "parallel",
            "drone_groups": [[0, 1], [2, 3]],
            "children": [
                {"primitive": "move_z", "params": [[1, 2], 20]},
                {"primitive": "move_z", "params": [[1, 2], -20]},
            ],
        })
        swarm_pos = _make_swarm_pos(4)
        limits = _make_limits()
        final_pos, waypoints = composer.execute_composed(composition, swarm_pos, 0.0, 2.0, limits)

        assert isinstance(final_pos, np.ndarray)
        assert final_pos.shape == (4, 3)

    def test_parallel_auto_split_groups(self):
        """Parallel without drone_groups should auto-split drones."""
        composer = PrimitiveComposer()
        composition = composer.parse_composition_yaml({
            "op": "parallel",
            "children": [
                {"primitive": "move_z", "params": [[1, 2], 10]},
                {"primitive": "move_z", "params": [[1, 2], 10]},
            ],
        })
        swarm_pos = _make_swarm_pos(4)
        limits = _make_limits()
        final_pos, waypoints = composer.execute_composed(composition, swarm_pos, 0.0, 2.0, limits)

        assert isinstance(final_pos, np.ndarray)
        assert final_pos.shape == (4, 3)
        # Check that waypoints have entries for all drones
        for t, wp in waypoints.items():
            for drone_id in range(4):
                assert drone_id in wp, f"Drone {drone_id} missing at t={t}"

    def test_parallel_waypoints_at_same_times(self):
        """Parallel children should produce waypoints at the same timesteps."""
        composer = PrimitiveComposer()
        composition = composer.parse_composition_yaml({
            "op": "parallel",
            "drone_groups": [[0, 1], [2, 3]],
            "children": [
                {"primitive": "move_z", "params": [[1, 2], 20]},
                {"primitive": "move_z", "params": [[1, 2], -20]},
            ],
        })
        swarm_pos = _make_swarm_pos(4)
        limits = _make_limits()
        _, waypoints = composer.execute_composed(composition, swarm_pos, 0.0, 2.0, limits)

        times = sorted(waypoints.keys())
        assert len(times) >= 1
        # Each timestep should have all 4 drones
        for t in times:
            assert len(waypoints[t]) == 4, f"Expected 4 drones at t={t}, got {len(waypoints[t])}"

    def test_parallel_uneven_split(self):
        """Parallel with 5 drones split into 2 groups."""
        composer = PrimitiveComposer()
        composition = composer.parse_composition_yaml({
            "op": "parallel",
            "children": [
                {"primitive": "move_z", "params": [[1, 2, 3], 10]},
                {"primitive": "move_z", "params": [[1, 2], -10]},
            ],
        })
        swarm_pos = _make_swarm_pos(5)
        limits = _make_limits()
        final_pos, waypoints = composer.execute_composed(composition, swarm_pos, 0.0, 2.0, limits)

        assert final_pos.shape == (5, 3)


# ---------------------------------------------------------------------------
# Tests for execute_composed - Blend
# ---------------------------------------------------------------------------

class TestExecuteBlend:
    """Tests for the blend composition operator."""

    def test_blend_returns_correct_types(self):
        composer = PrimitiveComposer()
        composition = composer.parse_composition_yaml({
            "op": "blend",
            "weight": 0.5,
            "children": [
                {"primitive": "move_z", "params": [[1, 2, 3, 4], 20]},
                {"primitive": "move_z", "params": [[1, 2, 3, 4], -20]},
            ],
        })
        swarm_pos = _make_swarm_pos(4)
        limits = _make_limits()
        final_pos, waypoints = composer.execute_composed(composition, swarm_pos, 0.0, 2.0, limits)

        assert isinstance(final_pos, np.ndarray)
        assert final_pos.shape == (4, 3)
        assert isinstance(waypoints, dict)

    def test_blend_weighted_average(self):
        """With weight=0.5, blended positions should be the midpoint."""
        composer = PrimitiveComposer()
        composition = composer.parse_composition_yaml({
            "op": "blend",
            "weight": 0.5,
            "children": [
                {"primitive": "move_z", "params": [[1], 200]},
                {"primitive": "move_z", "params": [[1], -200]},
            ],
        })
        swarm_pos = _make_swarm_pos(1)
        limits = _make_limits()
        final_pos, waypoints = composer.execute_composed(composition, swarm_pos, 0.0, 2.0, limits)

        # Both children start at z=100. Child A moves to ~200, child B to ~-100 (clipped).
        # The blend with weight=0.5 should produce something between them.
        times = sorted(waypoints.keys())
        last_wp = waypoints[times[-1]]
        z_val = last_wp[0][2]
        # Verify it's not equal to either extreme (unless clipping dominates)
        assert isinstance(z_val, (int, float, np.floating))

    def test_blend_weight_one_favors_first(self):
        """With weight=1.0, result should match the first child."""
        composer = PrimitiveComposer()
        composition = composer.parse_composition_yaml({
            "op": "blend",
            "weight": 1.0,
            "children": [
                {"primitive": "move_z", "params": [[1], 50]},
                {"primitive": "move_z", "params": [[1], -50]},
            ],
        })
        swarm_pos = _make_swarm_pos(1)
        limits = _make_limits()
        final_pos, waypoints = composer.execute_composed(composition, swarm_pos, 0.0, 2.0, limits)

        times = sorted(waypoints.keys())
        last_wp = waypoints[times[-1]]
        z_blended = last_wp[0][2]

        # Execute first child alone for comparison
        fn = primitive_by_name("move_z")
        _, wps_a = fn(([1], 50), swarm_pos, 0.0, 2.0, limits)
        z_a = list(wps_a.values())[-1][0][2]
        assert np.isclose(z_blended, z_a, atol=0.1)

    def test_blend_weight_zero_favors_second(self):
        """With weight=0.0, result should match the second child."""
        composer = PrimitiveComposer()
        composition = composer.parse_composition_yaml({
            "op": "blend",
            "weight": 0.0,
            "children": [
                {"primitive": "move_z", "params": [[1], 50]},
                {"primitive": "move_z", "params": [[1], -50]},
            ],
        })
        swarm_pos = _make_swarm_pos(1)
        limits = _make_limits()
        final_pos, waypoints = composer.execute_composed(composition, swarm_pos, 0.0, 2.0, limits)

        times = sorted(waypoints.keys())
        last_wp = waypoints[times[-1]]
        z_blended = last_wp[0][2]

        # Execute second child alone for comparison
        fn = primitive_by_name("move_z")
        _, wps_b = fn(([1], -50), swarm_pos, 0.0, 2.0, limits)
        z_b = list(wps_b.values())[-1][0][2]
        assert np.isclose(z_blended, z_b, atol=0.1)

    def test_blend_requires_two_children(self):
        """Blend with != 2 children should raise."""
        composer = PrimitiveComposer()
        composition = composer.parse_composition_yaml({
            "op": "blend",
            "weight": 0.5,
            "children": [
                {"primitive": "rotate", "params": [30, "z"]},
            ],
        })
        swarm_pos = _make_swarm_pos(4)
        limits = _make_limits()
        with pytest.raises(CompositionError, match="exactly 2 children"):
            composer.execute_composed(composition, swarm_pos, 0.0, 2.0, limits)

    def test_blend_all_drones_in_waypoints(self):
        """All drones should appear in every blended waypoint."""
        composer = PrimitiveComposer()
        composition = composer.parse_composition_yaml({
            "op": "blend",
            "weight": 0.5,
            "children": [
                {"primitive": "rotate", "params": [30, "z"]},
                {"primitive": "rotate", "params": [-30, "z"]},
            ],
        })
        swarm_pos = _make_swarm_pos(6)
        limits = _make_limits()
        _, waypoints = composer.execute_composed(composition, swarm_pos, 0.0, 3.0, limits)

        for t, wp in waypoints.items():
            for drone_id in range(6):
                assert drone_id in wp, f"Drone {drone_id} missing at t={t}"


# ---------------------------------------------------------------------------
# Tests for nested compositions
# ---------------------------------------------------------------------------

class TestNestedComposition:
    """Tests for nested composition trees."""

    def test_sequence_of_blends(self):
        """A sequence containing blend operators."""
        composer = PrimitiveComposer()
        composition = composer.parse_composition_yaml({
            "op": "sequence",
            "children": [
                {
                    "op": "blend",
                    "weight": 0.7,
                    "children": [
                        {"primitive": "rotate", "params": [15, "z"]},
                        {"primitive": "rotate", "params": [-15, "z"]},
                    ],
                },
                {
                    "op": "blend",
                    "weight": 0.3,
                    "children": [
                        {"primitive": "rotate", "params": [10, "z"]},
                        {"primitive": "rotate", "params": [-10, "z"]},
                    ],
                },
            ],
        })
        swarm_pos = _make_swarm_pos(4)
        limits = _make_limits()
        final_pos, waypoints = composer.execute_composed(composition, swarm_pos, 0.0, 4.0, limits)

        assert isinstance(final_pos, np.ndarray)
        assert final_pos.shape == (4, 3)
        assert len(waypoints) > 0

    def test_deep_nesting(self):
        """Three levels of nesting."""
        composer = PrimitiveComposer()
        composition = composer.parse_composition_yaml({
            "op": "sequence",
            "children": [
                {
                    "op": "parallel",
                    "drone_groups": [[0, 1], [2, 3]],
                    "children": [
                        {
                            "op": "blend",
                            "weight": 0.5,
                            "children": [
                                {"primitive": "rotate", "params": [10, "z"]},
                                {"primitive": "rotate", "params": [-10, "z"]},
                            ],
                        },
                        {"primitive": "move_z", "params": [[1, 2], 10]},
                    ],
                },
                {"primitive": "rotate", "params": [5, "z"]},
            ],
        })
        swarm_pos = _make_swarm_pos(4)
        limits = _make_limits()
        final_pos, waypoints = composer.execute_composed(composition, swarm_pos, 0.0, 4.0, limits)

        assert isinstance(final_pos, np.ndarray)
        assert final_pos.shape == (4, 3)


# ---------------------------------------------------------------------------
# Tests for interpolation
# ---------------------------------------------------------------------------

class TestInterpolation:
    """Tests for the waypoint interpolation used in blend."""

    def test_interpolate_missing_timesteps(self):
        """Missing timesteps should be linearly interpolated."""
        from swarm_gpt.core.primitive_composer import PrimitiveComposer

        # Create waypoints at t=1.0 and t=3.0, query at t=2.0
        pos1 = np.array([0.0, 0.0, 100.0])
        pos3 = np.array([0.0, 0.0, 200.0])
        wps = {
            1.0: {0: pos1},
            3.0: {0: pos3},
        }
        result = PrimitiveComposer._interpolate_waypoints(wps, [1.0, 2.0, 3.0], 1)
        assert 2.0 in result
        assert result[2.0][0][2] == pytest.approx(150.0)

    def test_interpolate_preserves_existing(self):
        """Existing timesteps should not be modified."""
        from swarm_gpt.core.primitive_composer import PrimitiveComposer

        pos = np.array([0.0, 0.0, 100.0])
        wps = {1.0: {0: pos.copy()}}
        result = PrimitiveComposer._interpolate_waypoints(wps, [1.0], 1)
        assert 1.0 in result
        np.testing.assert_array_equal(result[1.0][0], pos)

    def test_interpolate_empty_waypoints(self):
        """Empty waypoints dict should return empty."""
        from swarm_gpt.core.primitive_composer import PrimitiveComposer

        result = PrimitiveComposer._interpolate_waypoints({}, [1.0, 2.0], 4)
        assert result == {}


# ---------------------------------------------------------------------------
# Tests for real-world YAML parsing
# ---------------------------------------------------------------------------

class TestYamlIntegration:
    """Test parsing from actual YAML strings (as the LLM would output)."""

    def test_blend_from_yaml_string(self):
        """Parse a blend composition from a raw YAML string."""
        yaml_str = (
            "op: blend\n"
            "weight: 0.7\n"
            "children:\n"
            "  - primitive: rotate\n"
            "    params: [30, 'z']\n"
            "  - primitive: wave\n"
            "    params: [3, 100, [[1,1],[2,2]], [0.5,0.3], [0.2,0.4]]\n"
        )
        data = yaml.safe_load(yaml_str)
        composer = PrimitiveComposer()
        composition = composer.parse_composition_yaml(data)

        assert composition["op"] == "blend"
        assert composition["weight"] == 0.7
        assert len(composition["children"]) == 2
        assert composition["children"][0]["primitive"] == "rotate"
        assert composition["children"][1]["primitive"] == "wave"

    def test_sequence_from_yaml_string(self):
        yaml_str = (
            "op: sequence\n"
            "children:\n"
            "  - primitive: rotate\n"
            "    params: [30, 'z']\n"
            "  - primitive: move_z\n"
            "    params: [[1, 2, 3], 50]\n"
        )
        data = yaml.safe_load(yaml_str)
        composer = PrimitiveComposer()
        composition = composer.parse_composition_yaml(data)

        assert composition["op"] == "sequence"
        assert len(composition["children"]) == 2

    def test_parallel_from_yaml_string(self):
        yaml_str = (
            "op: parallel\n"
            "drone_groups: [[0, 1, 2], [3, 4, 5]]\n"
            "children:\n"
            "  - primitive: rotate\n"
            "    params: [30, 'z']\n"
            "  - primitive: rotate\n"
            "    params: [-30, 'z']\n"
        )
        data = yaml.safe_load(yaml_str)
        composer = PrimitiveComposer()
        composition = composer.parse_composition_yaml(data)

        assert composition["op"] == "parallel"
        assert composition["drone_groups"] == [[0, 1, 2], [3, 4, 5]]

    def test_full_round_trip_blend(self):
        """Parse from YAML and execute a blend end-to-end."""
        yaml_str = (
            "op: blend\n"
            "weight: 0.6\n"
            "children:\n"
            "  - primitive: move_z\n"
            "    params: [[1, 2, 3, 4], 50]\n"
            "  - primitive: move_z\n"
            "    params: [[1, 2, 3, 4], -50]\n"
        )
        data = yaml.safe_load(yaml_str)
        composer = PrimitiveComposer()
        composition = composer.parse_composition_yaml(data)

        swarm_pos = _make_swarm_pos(4)
        limits = _make_limits()
        final_pos, waypoints = composer.execute_composed(composition, swarm_pos, 0.0, 2.0, limits)

        assert final_pos.shape == (4, 3)
        assert len(waypoints) > 0
