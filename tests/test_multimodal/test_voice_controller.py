"""Tests for VoiceController – Chinese natural-language command parsing."""

import numpy as np
import pytest

from swarm_gpt.core.multimodal.voice_controller import (
    ACTION_KEYWORDS,
    SPATIAL_KEYWORDS,
    VoiceController,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_positions() -> np.ndarray:
    """Create a 6-drone position array in cm (n_drones, 3).

    Layout (top-down, x right, y forward):

        drone 0: (-90, -60, 100)   drone 1: (-30, -60, 100)   drone 2: ( 30, -60, 100)
        drone 3: (-90,  60, 100)   drone 4: (-30,  60, 100)   drone 5: ( 30,  60, 100)

    Left  (x<0): drones 0, 1, 3, 4
    Right (x>0): drones 2, 5
    Front (y>0): drones 3, 4, 5
    Back  (y<0): drones 0, 1, 2
    """
    return np.array([
        [-90, -60, 100],
        [-30, -60, 100],
        [ 30, -60, 100],
        [-90,  60, 100],
        [-30,  60, 100],
        [ 30,  60, 100],
    ], dtype=float)


def _make_asymmetric_positions() -> np.ndarray:
    """Create a 7-drone position array with a clear centre cluster.

    Centre drones (indices 3, 4) are near the origin.
    Perimeter drones (indices 0, 1, 2, 5, 6) are farther away.
    """
    return np.array([
        [-120, -80, 100],
        [ -40, -90, 100],
        [ 100, -70, 100],
        [ -10,   5, 100],
        [  15, -10, 100],
        [ 110,  80, 100],
        [ -90, 100, 100],
    ], dtype=float)


# ---------------------------------------------------------------------------
# Tests – spatial keyword detection
# ---------------------------------------------------------------------------

class TestSpatialKeywords:
    """Verify that spatial filters select the correct drones."""

    def test_left_filter(self):
        pos = _make_positions()
        mask = SPATIAL_KEYWORDS["左边"](pos)
        assert list(mask.nonzero()[0]) == [0, 1, 3, 4]

    def test_right_filter(self):
        pos = _make_positions()
        mask = SPATIAL_KEYWORDS["右边"](pos)
        assert list(mask.nonzero()[0]) == [2, 5]

    def test_front_filter(self):
        pos = _make_positions()
        mask = SPATIAL_KEYWORDS["前面"](pos)
        assert list(mask.nonzero()[0]) == [3, 4, 5]

    def test_back_filter(self):
        pos = _make_positions()
        mask = SPATIAL_KEYWORDS["后面"](pos)
        assert list(mask.nonzero()[0]) == [0, 1, 2]

    def test_all_filter(self):
        pos = _make_positions()
        mask = SPATIAL_KEYWORDS["全部"](pos)
        assert mask.all()

    def test_center_filter(self):
        pos = _make_asymmetric_positions()
        mask = SPATIAL_KEYWORDS["中间"](pos)
        # Centre drones are closest to centroid; at least one selected
        assert mask.any()
        # Centre drones must be a strict subset of all drones
        assert mask.sum() < len(pos)

    def test_perimeter_filter(self):
        pos = _make_asymmetric_positions()
        mask = SPATIAL_KEYWORDS["外围"](pos)
        # Perimeter is the complement of centre
        centre_mask = SPATIAL_KEYWORDS["中间"](pos)
        np.testing.assert_array_equal(mask, ~centre_mask)


# ---------------------------------------------------------------------------
# Tests – full command parsing
# ---------------------------------------------------------------------------

class TestParseCommand:
    """Test parse_command with various Chinese transcripts."""

    def setup_method(self):
        self.vc = VoiceController()
        self.positions = _make_positions()

    def test_raise_on_wrong_positions(self):
        with pytest.raises(ValueError, match="must be \\(n_drones, 3\\)"):
            self.vc.parse_command("提高30", np.array([1, 2, 3]))

    # -- Specific command tests from the task requirements --

    def test_right_drones_raise_30(self):
        """Parse '把右边的无人机提高30' -> move_z positive on right drones."""
        result = self.vc.parse_command("把右边的无人机提高30", self.positions)
        assert result["action"] == "提高"
        assert result["fallback"] is False
        # Right drones (x>0): indices 2, 5
        assert sorted(result["target_drones"]) == [2, 5]
        # Primitive call should reference 1-indexed IDs and positive magnitude
        call = result["primitive_call"]
        assert "move_z" in call
        # IDs in the call string are 1-indexed: [3, 6]
        assert "3" in call
        assert "6" in call
        assert "30" in call

    def test_all_scatter(self):
        """Parse '全部散开' -> scatter_gather on all drones."""
        result = self.vc.parse_command("全部散开", self.positions)
        assert result["action"] == "散开"
        assert result["fallback"] is False
        # All drones targeted
        assert sorted(result["target_drones"]) == [0, 1, 2, 3, 4, 5]
        assert "scatter_gather" in result["primitive_call"]

    def test_center_drones_lower(self):
        """Parse '中间的降低' -> move_z negative on centre drones."""
        positions = _make_asymmetric_positions()
        result = self.vc.parse_command("中间的降低", positions)
        assert result["action"] == "降低"
        assert result["fallback"] is False
        # Centre drones should be selected (at least one, fewer than all)
        assert len(result["target_drones"]) >= 1
        assert len(result["target_drones"]) < len(positions)
        call = result["primitive_call"]
        assert "move_z" in call
        # Magnitude should be negative (降低 = lower)
        # Default magnitude is 30, sign is -1 => -30
        assert "-30" in call

    def test_left_drones_move_left(self):
        """Parse '左边左移' -> left-shift on left-side drones."""
        result = self.vc.parse_command("左边左移", self.positions)
        assert result["action"] == "左移"
        assert result["fallback"] is False
        # Left drones (x<0): indices 0, 1, 3, 4
        assert sorted(result["target_drones"]) == [0, 1, 3, 4]

    def test_front_drones_rotate(self):
        """Parse '前面旋转45' -> rotate on front drones."""
        result = self.vc.parse_command("前面旋转45", self.positions)
        assert result["action"] == "旋转"
        assert result["fallback"] is False
        assert sorted(result["target_drones"]) == [3, 4, 5]
        assert "rotate" in result["primitive_call"]
        assert "45" in result["primitive_call"]

    def test_perimeter_gather(self):
        """Parse '外围聚拢' -> center on perimeter drones."""
        result = self.vc.parse_command("外围聚拢", self.positions)
        assert result["action"] == "聚拢"
        assert result["fallback"] is False
        assert "center" in result["primitive_call"]

    def test_back_drones_lower_with_magnitude(self):
        """Parse '后面的降低50' -> move_z negative on back drones, magnitude 50."""
        result = self.vc.parse_command("后面的降低50", self.positions)
        assert result["action"] == "降低"
        assert sorted(result["target_drones"]) == [0, 1, 2]
        assert "-50" in result["primitive_call"]

    # -- Default magnitude when none specified --

    def test_default_magnitude(self):
        """When no magnitude is specified, the default (30) should be used."""
        result = self.vc.parse_command("全部提高", self.positions)
        assert result["action"] == "提高"
        assert "30" in result["primitive_call"]

    # -- Fallback for unknown commands --

    def test_fallback_unknown_action(self):
        """An unknown action keyword should trigger fallback."""
        result = self.vc.parse_command("全部跳舞", self.positions)
        assert result["fallback"] is True
        assert result["action"] == "reprompt"
        assert result["primitive_call"] is None
        assert result["raw_transcript"] == "全部跳舞"

    def test_fallback_gibberish(self):
        """Completely unrecognisable text should trigger fallback."""
        result = self.vc.parse_command("asdfghjkl", self.positions)
        assert result["fallback"] is True
        assert result["action"] == "reprompt"

    def test_fallback_unknown_spatial_with_known_action(self):
        """If the spatial keyword is missing but action is known, parse with all drones."""
        result = self.vc.parse_command("提高20", self.positions)
        assert result["action"] == "提高"
        assert result["fallback"] is False
        # No spatial filter => all drones
        assert sorted(result["target_drones"]) == [0, 1, 2, 3, 4, 5]


# ---------------------------------------------------------------------------
# Tests – _make_call
# ---------------------------------------------------------------------------

class TestMakeCall:
    """Unit tests for the _make_call helper."""

    def test_move_z_positive(self):
        call = VoiceController._make_call("提高", [0, 2], 30)
        assert call == "move_z([1, 3], 30)"

    def test_move_z_negative(self):
        call = VoiceController._make_call("降低", [1, 3], 40)
        assert call == "move_z([2, 4], -40)"

    def test_rotate(self):
        call = VoiceController._make_call("旋转", [0, 1, 2], 90)
        assert call == "rotate(90, 'z')"

    def test_scatter_gather(self):
        call = VoiceController._make_call("散开", [0, 1], 50)
        assert "scatter_gather" in call

    def test_center(self):
        call = VoiceController._make_call("聚拢", [0, 1], 30)
        assert "center" in call
        # IDs should be 1-indexed in the call
        assert "[1, 2]" in call

    def test_ids_are_1_indexed_in_call(self):
        """Drone IDs in primitive calls must be 1-indexed."""
        call = VoiceController._make_call("提高", [0, 4], 20)
        # 0-indexed [0,4] -> 1-indexed [1,5]
        assert "[1, 5]" in call
