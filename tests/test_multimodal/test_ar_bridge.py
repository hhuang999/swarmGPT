"""Tests for ARBridge – set_formation, drone colours, command processing.

All tests run without FastAPI installed so that heavy web dependencies are not
required for the core logic unit-tests.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from swarm_gpt.core.multimodal.ar_bridge import ARBridge

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_positions(n_drones: int = 6) -> NDArray:
    """Return a simple (n_drones, 3) grid of positions in cm."""
    spacing = 60
    cols = int(np.ceil(np.sqrt(n_drones)))
    positions = []
    for i in range(n_drones):
        r, c = divmod(i, cols)
        positions.append([c * spacing, r * spacing, 100])
    return np.array(positions, dtype=float)


# ---------------------------------------------------------------------------
# Tests – set_formation
# ---------------------------------------------------------------------------


class TestSetFormation:
    """Tests for ARBridge.set_formation()."""

    def test_returns_correct_structure(self) -> None:
        bridge = ARBridge()
        positions = _make_positions(4)
        result = bridge.set_formation(positions, {"name": "grid"})

        assert result["n_drones"] == 4
        assert len(result["positions"]) == 4
        assert result["metadata"] == {"name": "grid"}

    def test_each_position_has_required_keys(self) -> None:
        bridge = ARBridge()
        positions = _make_positions(3)
        result = bridge.set_formation(positions)

        for entry in result["positions"]:
            assert "id" in entry
            assert "pos" in entry
            assert "color" in entry
            assert isinstance(entry["pos"], list)
            assert len(entry["pos"]) == 3

    def test_default_metadata_is_empty_dict(self) -> None:
        bridge = ARBridge()
        result = bridge.set_formation(_make_positions(2))
        assert result["metadata"] == {}

    def test_invalid_shape_raises(self) -> None:
        bridge = ARBridge()
        with pytest.raises(ValueError, match="must be \\(n_drones, 3\\)"):
            bridge.set_formation(np.zeros((5, 2)))

    def test_stores_copy_of_positions(self) -> None:
        bridge = ARBridge()
        positions = _make_positions(2)
        bridge.set_formation(positions)
        # Mutating original should not affect stored data
        positions[0] = [9999, 9999, 9999]
        np.testing.assert_array_less(bridge._positions[0], [9999, 9999, 9999])


# ---------------------------------------------------------------------------
# Tests – _get_drone_color
# ---------------------------------------------------------------------------


class TestGetDroneColor:
    """Tests for ARBridge._get_drone_color()."""

    def test_returns_list_of_three_ints(self) -> None:
        color = ARBridge._get_drone_color(0, 4)
        assert isinstance(color, list)
        assert len(color) == 3
        for ch in color:
            assert isinstance(ch, int)

    def test_values_in_0_255_range(self) -> None:
        for idx in range(20):
            color = ARBridge._get_drone_color(idx, 20)
            for ch in color:
                assert 0 <= ch <= 255

    def test_different_indices_give_different_hues(self) -> None:
        c0 = ARBridge._get_drone_color(0, 4)
        c1 = ARBridge._get_drone_color(1, 4)
        c2 = ARBridge._get_drone_color(2, 4)
        # Very unlikely to be all equal
        assert c0 != c1 or c1 != c2

    def test_total_zero_treated_as_one(self) -> None:
        """Division by zero should be guarded."""
        color = ARBridge._get_drone_color(0, 0)
        assert len(color) == 3

    def test_single_drone(self) -> None:
        color = ARBridge._get_drone_color(0, 1)
        assert len(color) == 3
        # All channels within valid range
        for ch in color:
            assert 0 <= ch <= 255


# ---------------------------------------------------------------------------
# Tests – _process_command
# ---------------------------------------------------------------------------


class TestProcessCommand:
    """Tests for ARBridge._process_command()."""

    def test_move_drone_success(self) -> None:
        bridge = ARBridge()
        positions = _make_positions(4)
        bridge.set_formation(positions)

        result = bridge._process_command(
            {"type": "move_drone", "drone_id": 1, "new_position": [10, 20, 30]}
        )

        assert result["status"] == "ok"
        assert result["drone_id"] == 1
        assert result["new_position"] == [10.0, 20.0, 30.0]

    def test_move_drone_updates_internal_state(self) -> None:
        bridge = ARBridge()
        positions = _make_positions(3)
        bridge.set_formation(positions)

        bridge._process_command({"type": "move_drone", "drone_id": 2, "new_position": [50, 60, 70]})
        np.testing.assert_array_almost_equal(bridge._positions[2], [50, 60, 70])

    def test_move_drone_missing_fields(self) -> None:
        bridge = ARBridge()
        bridge.set_formation(_make_positions(2))

        result = bridge._process_command({"type": "move_drone", "drone_id": 0})
        assert result["status"] == "error"
        assert "new_position" in result["detail"]

    def test_move_drone_no_formation_set(self) -> None:
        bridge = ARBridge()

        result = bridge._process_command(
            {"type": "move_drone", "drone_id": 0, "new_position": [1, 2, 3]}
        )
        assert result["status"] == "error"
        assert "No formation" in result["detail"]

    def test_move_drone_out_of_range(self) -> None:
        bridge = ARBridge()
        bridge.set_formation(_make_positions(3))

        result = bridge._process_command(
            {"type": "move_drone", "drone_id": 10, "new_position": [1, 2, 3]}
        )
        assert result["status"] == "error"
        assert "out of range" in result["detail"]

    def test_move_drone_negative_id(self) -> None:
        bridge = ARBridge()
        bridge.set_formation(_make_positions(3))

        result = bridge._process_command(
            {"type": "move_drone", "drone_id": -1, "new_position": [1, 2, 3]}
        )
        assert result["status"] == "error"
        assert "out of range" in result["detail"]

    def test_unknown_command_type(self) -> None:
        bridge = ARBridge()
        result = bridge._process_command({"type": "explode"})
        assert result["status"] == "error"
        assert "Unknown command" in result["detail"]

    def test_missing_type(self) -> None:
        bridge = ARBridge()
        result = bridge._process_command({})
        assert result["status"] == "error"
        assert "Unknown command" in result["detail"]


# ---------------------------------------------------------------------------
# Tests – push_trajectory_point
# ---------------------------------------------------------------------------


class TestPushTrajectoryPoint:
    """Tests for ARBridge.push_trajectory_point() payload generation."""

    def test_payload_structure(self) -> None:
        bridge = ARBridge()
        positions = _make_positions(3)
        payload = bridge.push_trajectory_point(1.5, positions)

        assert "timestamp" in payload
        assert payload["time"] == 1.5
        assert len(payload["drones"]) == 3
        for drone in payload["drones"]:
            assert "id" in drone
            assert "pos" in drone
            assert "color" in drone

    def test_timestamp_is_iso_format(self) -> None:
        bridge = ARBridge()
        payload = bridge.push_trajectory_point(0.0, _make_positions(2))
        # Should parse without error
        from datetime import datetime

        datetime.fromisoformat(payload["timestamp"])

    def test_positions_values_match(self) -> None:
        bridge = ARBridge()
        positions = np.array([[10, 20, 30], [40, 50, 60]], dtype=float)
        payload = bridge.push_trajectory_point(2.0, positions)

        assert payload["drones"][0]["pos"] == [10.0, 20.0, 30.0]
        assert payload["drones"][1]["pos"] == [40.0, 50.0, 60.0]


# ---------------------------------------------------------------------------
# Tests – constructor / server lifecycle
# ---------------------------------------------------------------------------


class TestConstructor:
    """Tests for ARBridge initialisation."""

    def test_default_host_port(self) -> None:
        bridge = ARBridge()
        assert bridge.host == "0.0.0.0"
        assert bridge.port == 8765

    def test_custom_host_port(self) -> None:
        bridge = ARBridge(host="127.0.0.1", port=9999)
        assert bridge.host == "127.0.0.1"
        assert bridge.port == 9999

    def test_no_positions_initially(self) -> None:
        bridge = ARBridge()
        assert bridge._positions is None

    def test_no_ws_clients_initially(self) -> None:
        bridge = ARBridge()
        assert bridge._ws_clients == []
