"""Tests for image-to-formation conversion."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from swarm_gpt.core.multimodal.image_to_formation import (
    ConversionMetadata,
    FlightBounds,
    ImageFormationConverter,
    _to_greyscale,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _circle_image(size: int = 200, radius: int = 70) -> np.ndarray:
    """Create a white circle on a black background (BGR)."""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    center = (size // 2, size // 2)
    cv2 = pytest.importorskip("cv2")
    cv2.circle(img, center, radius, (255, 255, 255), thickness=2)
    return img


def _square_image(size: int = 200, side: int = 120) -> np.ndarray:
    """Create a white square outline on a black background (BGR)."""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    cv2 = pytest.importorskip("cv2")
    top_left = ((size - side) // 2, (size - side) // 2)
    bottom_right = (top_left[0] + side, top_left[1] + side)
    cv2.rectangle(img, top_left, bottom_right, (255, 255, 255), thickness=2)
    return img


def _blank_image(size: int = 200) -> np.ndarray:
    """Create a blank (all-black) image."""
    return np.zeros((size, size, 3), dtype=np.uint8)


def _noisy_image(size: int = 200) -> np.ndarray:
    """Create a noisy image with very low edge density."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 50, (size, size, 3), dtype=np.uint8)


def _make_bounds() -> FlightBounds:
    return FlightBounds(
        lower=np.array([-2.0, -2.0, 0.0]),
        upper=np.array([2.0, 2.0, 2.0]),
    )


def _make_bounds_dict() -> dict[str, np.ndarray]:
    return {
        "lower": np.array([-2.0, -2.0, 0.0]),
        "upper": np.array([2.0, 2.0, 2.0]),
    }


# ---------------------------------------------------------------------------
# Tests: Flight-space mapping
# ---------------------------------------------------------------------------


class TestFlightSpaceMapping:
    """Tests for _map_to_flight_space."""

    def test_output_shape(self) -> None:
        converter = ImageFormationConverter()
        pts = np.array([[0.5, 0.5], [0.0, 0.0], [1.0, 1.0]])
        bounds = _make_bounds()
        # Access the private method directly for unit testing
        result = converter._map_to_flight_space(pts, bounds)
        assert result.shape == (3, 3)

    def test_z_is_flight_height(self) -> None:
        converter = ImageFormationConverter(flight_height=150.0)
        pts = np.array([[0.5, 0.5], [0.2, 0.8]])
        result = converter._map_to_flight_space(pts, _make_bounds())
        assert np.allclose(result[:, 2], 150.0)

    def test_z_clipped_to_bounds(self) -> None:
        bounds = FlightBounds(
            lower=np.array([-2.0, -2.0, 0.5]),
            upper=np.array([2.0, 2.0, 1.5]),
        )
        # flight_height=200 cm is above upper[2]*100 = 150 cm
        converter = ImageFormationConverter(flight_height=200.0)
        pts = np.array([[0.5, 0.5]])
        result = converter._map_to_flight_space(pts, bounds)
        assert result[0, 2] == pytest.approx(150.0)

    def test_normalised_origin_maps_near_lower(self) -> None:
        converter = ImageFormationConverter(flight_height=100.0)
        pts = np.array([[0.0, 0.0]])
        result = converter._map_to_flight_space(pts, _make_bounds())
        # With 10% margin, x_lo = -200 + 40 = -160
        assert result[0, 0] == pytest.approx(-160.0)
        assert result[0, 1] == pytest.approx(-160.0)

    def test_normalised_one_maps_near_upper(self) -> None:
        converter = ImageFormationConverter(flight_height=100.0)
        pts = np.array([[1.0, 1.0]])
        result = converter._map_to_flight_space(pts, _make_bounds())
        # x_hi = 200 - 40 = 160
        assert result[0, 0] == pytest.approx(160.0)
        assert result[0, 1] == pytest.approx(160.0)

    def test_dict_bounds_accepted(self) -> None:
        converter = ImageFormationConverter()
        pts = np.array([[0.5, 0.5]])
        result = converter._map_to_flight_space(pts, FlightBounds(**_make_bounds_dict()))
        assert result.shape == (1, 3)


# ---------------------------------------------------------------------------
# Tests: CV extraction
# ---------------------------------------------------------------------------


class TestCVExtraction:
    """Tests for _cv_extract with simple drawn shapes."""

    def test_circle_produces_correct_count(self) -> None:
        converter = ImageFormationConverter()
        img = _circle_image()
        pts = converter._cv_extract(img, 8)
        assert pts.shape == (8, 2)

    def test_square_produces_correct_count(self) -> None:
        converter = ImageFormationConverter()
        img = _square_image()
        pts = converter._cv_extract(img, 12)
        assert pts.shape == (12, 2)

    def test_points_normalised_to_unit_range(self) -> None:
        converter = ImageFormationConverter()
        img = _circle_image()
        pts = converter._cv_extract(img, 10)
        assert np.all(pts >= 0.0)
        assert np.all(pts <= 1.0)

    def test_blank_image_fallback_circle(self) -> None:
        """When no contours are detected, fallback to a uniform circle."""
        converter = ImageFormationConverter()
        img = _blank_image()
        pts = converter._cv_extract(img, 6)
        assert pts.shape == (6, 2)
        # Points should be roughly on a circle centred at (0.5, 0.5)
        centre = np.array([0.5, 0.5])
        radii = np.linalg.norm(pts - centre, axis=1)
        # All radii should be approximately 0.4
        assert np.allclose(radii, radii[0], atol=0.05)

    def test_single_drone(self) -> None:
        converter = ImageFormationConverter()
        img = _circle_image()
        pts = converter._cv_extract(img, 1)
        assert pts.shape == (1, 2)


# ---------------------------------------------------------------------------
# Tests: Strategy selection
# ---------------------------------------------------------------------------


class TestStrategySelection:
    """Tests for _choose_strategy and auto-mode."""

    def test_circle_image_selects_cv(self) -> None:
        converter = ImageFormationConverter()
        img = _circle_image()
        strategy = converter._choose_strategy(img)
        assert strategy == "cv"

    def test_blank_image_without_provider_selects_cv(self) -> None:
        """Low edge density without VLM provider falls back to cv."""
        converter = ImageFormationConverter(provider=None)
        img = _noisy_image()
        strategy = converter._choose_strategy(img)
        assert strategy == "cv"

    def test_blank_image_with_provider_may_select_vlm(self) -> None:
        """Low edge density with VLM provider available should try vlm."""
        mock_provider = MagicMock()
        converter = ImageFormationConverter(provider=mock_provider)
        # Use a truly blank (uniform) image for zero edge density
        img = _blank_image()
        strategy = converter._choose_strategy(img)
        assert strategy == "vlm"

    def test_explicit_cv_strategy(self) -> None:
        converter = ImageFormationConverter()
        img = _circle_image()
        _, _, meta = converter.convert(img, 5, strategy="cv")
        assert meta.strategy == "cv"

    def test_explicit_vlm_strategy_with_mock(self) -> None:
        mock_provider = MagicMock()
        mock_provider.analyze_image.return_value = MagicMock(
            content=json.dumps([[0.5, 0.5], [0.3, 0.7], [0.7, 0.3], [0.2, 0.5]])
        )
        converter = ImageFormationConverter(provider=mock_provider)
        img = _circle_image()
        positions, name, meta = converter.convert(img, 4, strategy="vlm")
        assert meta.strategy == "vlm"
        assert positions.shape == (4, 3)
        mock_provider.analyze_image.assert_called_once()


# ---------------------------------------------------------------------------
# Tests: VLM extraction (mocked)
# ---------------------------------------------------------------------------


class TestVLMExtraction:
    """Tests for _vlm_extract with mocked provider."""

    def test_parse_valid_response(self) -> None:
        content = json.dumps([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        pts = ImageFormationConverter._parse_vlm_points(content, 3)
        assert pts.shape == (3, 2)
        assert np.allclose(pts, [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])

    def test_parse_truncates_excess_points(self) -> None:
        content = json.dumps([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]])
        pts = ImageFormationConverter._parse_vlm_points(content, 2)
        assert pts.shape == (2, 2)

    def test_parse_pads_insufficient_points(self) -> None:
        content = json.dumps([[0.1, 0.2]])
        pts = ImageFormationConverter._parse_vlm_points(content, 3)
        assert pts.shape == (3, 2)
        # Last two points should be copies of the first
        assert np.allclose(pts[1], pts[0])
        assert np.allclose(pts[2], pts[0])

    def test_parse_clips_to_unit_range(self) -> None:
        content = json.dumps([[-0.5, 1.5], [2.0, -1.0]])
        pts = ImageFormationConverter._parse_vlm_points(content, 2)
        assert np.all(pts >= 0.0)
        assert np.all(pts <= 1.0)

    def test_parse_no_json_array_raises(self) -> None:
        with pytest.raises(ValueError, match="does not contain a JSON array"):
            ImageFormationConverter._parse_vlm_points("not json at all", 3)

    def test_parse_malformed_json_raises(self) -> None:
        with pytest.raises(ValueError, match="not valid JSON"):
            ImageFormationConverter._parse_vlm_points("[[1, 2], [3, 4, broken]", 3)

    def test_parse_no_array_raises(self) -> None:
        with pytest.raises(ValueError, match="does not contain a JSON array"):
            ImageFormationConverter._parse_vlm_points("hello world", 3)

    def test_parse_wraps_json_in_text(self) -> None:
        content = "Here are the points: [[0.1, 0.2], [0.5, 0.5]] done."
        pts = ImageFormationConverter._parse_vlm_points(content, 2)
        assert pts.shape == (2, 2)
        assert np.allclose(pts[0], [0.1, 0.2])


# ---------------------------------------------------------------------------
# Tests: Full convert pipeline
# ---------------------------------------------------------------------------


class TestConvertPipeline:
    """End-to-end tests for the convert method."""

    def test_circle_image_cv_pipeline(self) -> None:
        converter = ImageFormationConverter(flight_height=120.0)
        img = _circle_image()
        positions, shape_name, meta = converter.convert(img, 8)
        assert positions.shape == (8, 3)
        assert meta.strategy == "cv"
        assert meta.n_drones == 8
        assert meta.image_shape == (200, 200)
        # All z at flight height
        assert np.allclose(positions[:, 2], 120.0)
        # x, y within flight bounds (cm)
        assert np.all(positions[:, 0] >= -200)
        assert np.all(positions[:, 0] <= 200)
        assert np.all(positions[:, 1] >= -200)
        assert np.all(positions[:, 1] <= 200)

    def test_dict_bounds_in_convert(self) -> None:
        converter = ImageFormationConverter()
        img = _circle_image()
        positions, _, _ = converter.convert(img, 5, flight_bounds=_make_bounds_dict())
        assert positions.shape == (5, 3)

    def test_none_bounds_uses_defaults(self) -> None:
        converter = ImageFormationConverter()
        img = _circle_image()
        positions, _, meta = converter.convert(img, 5, flight_bounds=None)
        assert positions.shape == (5, 3)

    def test_flight_bounds_object(self) -> None:
        converter = ImageFormationConverter()
        img = _circle_image()
        bounds = _make_bounds()
        positions, _, _ = converter.convert(img, 5, flight_bounds=bounds)
        assert positions.shape == (5, 3)

    def test_invalid_n_drones_raises(self) -> None:
        converter = ImageFormationConverter()
        img = _circle_image()
        with pytest.raises(ValueError, match="n_drones must be >= 1"):
            converter.convert(img, 0)

    def test_invalid_strategy_raises(self) -> None:
        converter = ImageFormationConverter()
        img = _circle_image()
        with pytest.raises(ValueError, match="Unknown strategy"):
            converter.convert(img, 5, strategy="invalid")

    def test_vlm_without_provider_raises(self) -> None:
        converter = ImageFormationConverter(provider=None)
        img = _circle_image()
        with pytest.raises(RuntimeError, match="VLM extraction requires"):
            converter.convert(img, 4, strategy="vlm")

    def test_metadata_edge_density_populated(self) -> None:
        converter = ImageFormationConverter()
        img = _circle_image()
        _, _, meta = converter.convert(img, 5, strategy="cv")
        assert meta.edge_density > 0


# ---------------------------------------------------------------------------
# Tests: _to_greyscale helper
# ---------------------------------------------------------------------------


class TestToGreyscale:
    def test_bgr_converted(self) -> None:
        bgr = np.zeros((10, 10, 3), dtype=np.uint8)
        grey = _to_greyscale(bgr)
        assert grey.ndim == 2
        assert grey.shape == (10, 10)

    def test_greyscale_passthrough(self) -> None:
        grey_in = np.zeros((10, 10), dtype=np.uint8)
        grey_out = _to_greyscale(grey_in)
        assert grey_out is grey_in

    def test_bgra_converted(self) -> None:
        bgra = np.zeros((10, 10, 4), dtype=np.uint8)
        grey = _to_greyscale(bgra)
        assert grey.ndim == 2

    def test_single_channel_3d(self) -> None:
        single = np.zeros((10, 10, 1), dtype=np.uint8)
        grey = _to_greyscale(single)
        assert grey.ndim == 2

    def test_invalid_shape_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported image shape"):
            _to_greyscale(np.zeros((10, 10, 5), dtype=np.uint8))
