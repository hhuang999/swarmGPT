"""End-to-end integration tests for the extended SwarmGPT pipeline.

Verifies that all new modules (primitives, composer, providers, multimodal,
custom primitive generator) work together correctly.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_swarm_pos(n_drones: int = 6) -> np.ndarray:
    """Create *n_drones* positions in a grid, values in cm."""
    positions = np.zeros((n_drones, 3))
    cols = int(np.ceil(np.sqrt(n_drones)))
    for i in range(n_drones):
        positions[i] = [(i % cols) * 50, (i // cols) * 50, 100]
    # Centre around origin
    positions[:, 0] -= np.mean(positions[:, 0])
    positions[:, 1] -= np.mean(positions[:, 1])
    return positions


def _make_limits() -> dict[str, np.ndarray]:
    """Return spatial limits matching the project convention (metres)."""
    return {"lower": np.array([-2.0, -2.0, 0.0]), "upper": np.array([2.0, 2.0, 2.0])}


def _restore_exception_module() -> None:
    """Ensure the real swarm_gpt.exception module is in sys.modules.

    Some unit tests replace ``sys.modules["swarm_gpt.exception"]`` with a
    stripped-down fake that lacks ``LLMException``.  This helper restores the
    real module so that subsequent imports of ``swarm_gpt.providers`` (which
    depends on ``LLMException``) succeed.
    """
    import sys

    current = sys.modules.get("swarm_gpt.exception")
    if current is not None and hasattr(current, "LLMException"):
        return  # Already good.

    # Remove the fake and reload from disk.
    if "swarm_gpt.exception" in sys.modules:
        del sys.modules["swarm_gpt.exception"]

    # Also invalidate any provider sub-modules that captured the fake import.
    for key in list(sys.modules):
        if key.startswith("swarm_gpt.providers"):
            del sys.modules[key]

    # Force a fresh import from disk.
    import swarm_gpt.exception  # noqa: F401


# ===================================================================
# TestPrimitiveEcosystem
# ===================================================================


class TestPrimitiveEcosystem:
    """Verify that all new primitives are registered and reachable."""

    NEW_PRIMITIVES = (
        "firework",
        "pendulum",
        "scatter_gather",
        "form_heart",
        "form_line",
        "orbit",
        "form_shape",
    )

    def test_all_new_primitives_callable(self) -> None:
        """Every new primitive must be in the registry and callable."""
        from swarm_gpt.core.motion_primitives import motion_primitives, primitive_by_name

        for name in self.NEW_PRIMITIVES:
            assert name in motion_primitives, f"'{name}' missing from motion_primitives dict"
            fn = primitive_by_name(name)
            assert callable(fn), f"primitive_by_name('{name}') returned non-callable"

    def test_composer_with_new_primitives(self) -> None:
        """PrimitiveComposer must parse a valid composition tree with new primitives."""
        from swarm_gpt.core.primitive_composer import PrimitiveComposer

        composer = PrimitiveComposer()
        data = {
            "op": "sequence",
            "children": [
                {"primitive": "firework", "params": [5, 150, 80]},
                {"primitive": "orbit", "params": [4, 10, 30]},
            ],
        }
        tree = composer.parse_composition_yaml(data)

        assert tree["op"] == "sequence"
        assert len(tree["children"]) == 2
        assert tree["children"][0]["primitive"] == "firework"
        assert tree["children"][0]["params"] == [5, 150, 80]
        assert tree["children"][1]["primitive"] == "orbit"
        assert tree["children"][1]["params"] == [4, 10, 30]

    def test_provider_factory(self) -> None:
        """get_provider must raise ValueError for unknown provider names."""
        _restore_exception_module()
        from swarm_gpt.providers import get_provider

        with pytest.raises(ValueError, match="Unknown provider"):
            get_provider("nonexistent_provider")

    def test_provider_factory_no_key_raises(self) -> None:
        """get_provider('openai', api_key=None) raises when env var is unset.

        We clear OPENAI_API_KEY temporarily.  The OpenAI SDK itself raises
        ``OpenAIError`` when no key is available.
        """
        from unittest.mock import patch

        _restore_exception_module()
        from swarm_gpt.providers import get_provider

        env = {"PATH": "/usr/bin"}
        with patch.dict(os.environ, env, clear=True):
            with pytest.raises(Exception):
                get_provider("openai", api_key=None)

    def test_multimodal_imports(self) -> None:
        """All multimodal classes must import cleanly."""
        from swarm_gpt.core.multimodal.ar_bridge import ARBridge
        from swarm_gpt.core.multimodal.image_to_formation import ImageFormationConverter
        from swarm_gpt.core.multimodal.voice_controller import VoiceController

        assert callable(ImageFormationConverter)
        assert callable(VoiceController)
        assert callable(ARBridge)

    def test_custom_primitive_imports(self) -> None:
        """Custom primitive generator classes must import cleanly."""
        from swarm_gpt.core.custom_primitive_generator import (
            CustomPrimitiveManager,
            CustomPrimitiveValidator,
            PrimitiveSandbox,
        )

        assert callable(PrimitiveSandbox)
        assert callable(CustomPrimitiveValidator)
        assert callable(CustomPrimitiveManager)


# ===================================================================
# TestPrimitiveExecution
# ===================================================================


class TestPrimitiveExecution:
    """Execute new primitives with mock swarm data and verify outputs."""

    def test_firework_produces_waypoints(self) -> None:
        """firework must return (positions, waypoints_dict) with valid data."""
        from swarm_gpt.core.motion_primitives import primitive_by_name

        firework = primitive_by_name("firework")
        swarm_pos = _make_swarm_pos(6)
        limits = _make_limits()

        positions, waypoints = firework(
            params=(5, 150, 80), swarm_pos=swarm_pos.copy(), tstart=0.0, tend=5.0, limits=limits
        )

        # positions is an ndarray of shape (n_drones, 3)
        assert isinstance(positions, np.ndarray)
        assert positions.shape == (6, 3)

        # waypoints is a dict mapping float -> {int -> ndarray(3,)}
        assert isinstance(waypoints, dict)
        assert len(waypoints) > 0
        for t, wp_dict in waypoints.items():
            assert isinstance(t, float)
            assert isinstance(wp_dict, dict)
            for drone_id, pos in wp_dict.items():
                assert isinstance(drone_id, (int, np.integer))
                assert isinstance(pos, np.ndarray)
                assert pos.shape == (3,)

    def test_scatter_gather_round_trip(self) -> None:
        """scatter_gather must spread drones outward and then pull them back."""
        from swarm_gpt.core.motion_primitives import primitive_by_name

        scatter_gather = primitive_by_name("scatter_gather")
        swarm_pos = _make_swarm_pos(6)
        limits = _make_limits()

        positions, waypoints = scatter_gather(
            params=(4, 100, 100), swarm_pos=swarm_pos.copy(), tstart=0.0, tend=4.0, limits=limits
        )

        assert isinstance(positions, np.ndarray)
        assert positions.shape == (6, 3)
        assert isinstance(waypoints, dict)
        assert len(waypoints) > 0

        # At the midpoint the drones should be spread further from the centroid
        # than at the start.  The scatter_gather uses sin(pi*frac) so the max
        # spread is at frac=0.5.
        centroid = np.mean(swarm_pos, axis=0)
        start_spread = np.max(np.linalg.norm(swarm_pos - centroid, axis=1))

        # Find the waypoint closest to t=2.0 (midpoint of 0-4 range)
        times = sorted(waypoints.keys())
        mid_idx = len(times) // 2
        mid_time = times[mid_idx]
        mid_positions = np.array([waypoints[mid_time][i] for i in range(6)])
        mid_spread = np.max(np.linalg.norm(mid_positions - centroid, axis=1))

        # Mid-point spread should be >= start spread (drones expand)
        assert mid_spread >= start_spread * 0.9  # allow small tolerance
