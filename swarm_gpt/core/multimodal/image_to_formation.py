"""Convert images/sketches into drone formation positions.

This module provides the ``ImageFormationConverter`` class which analyses an
input image (photo, sketch, or diagram) and produces a set of 3-D positions
suitable for a drone swarm to form the detected shape.

Two extraction strategies are supported:

* **CV** -- Uses OpenCV contour detection + uniform sampling.  Works well for
  clean sketches and high-contrast silhouettes.
* **VLM** -- Delegates to a vision-language model (via the provider interface)
  for semantic understanding of complex images.

The ``auto`` strategy (default) selects CV when the image has strong edges and
falls back to VLM otherwise.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from swarm_gpt.providers.base import LLMProvider

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Edge-density thresholds for strategy selection
# ---------------------------------------------------------------------------
_EDGE_DENSITY_LOW: float = 0.02
_EDGE_DENSITY_HIGH: float = 0.15


@dataclass
class FlightBounds:
    """Axis-aligned bounding box for the flight volume.

    All values are in **metres** to match the conventions in
    :pymod:`swarm_gpt.core.motion_primitives`.
    """

    lower: NDArray  # shape (3,) -- x, y, z minimum
    upper: NDArray  # shape (3,) -- x, y, z maximum


def _default_flight_bounds() -> FlightBounds:
    """Return the same default limits used throughout the project."""
    return FlightBounds(lower=np.array([-2.0, -2.0, 0.0]), upper=np.array([2.0, 2.0, 2.0]))


@dataclass
class ConversionMetadata:
    """Metadata returned alongside a conversion result."""

    strategy: str
    n_drones: int
    image_shape: tuple[int, int]
    edge_density: float = 0.0
    shape_name: str = "unknown"
    extra: dict[str, Any] = field(default_factory=dict)


class ImageFormationConverter:
    """Analyse an image and produce drone formation positions.

    Args:
        provider: An ``LLMProvider`` instance with ``analyze_image`` support.
            Only required when the *vlm* strategy is used.
        flight_height: Default z-coordinate for the formation in **cm**.
    """

    def __init__(self, provider: LLMProvider | None = None, flight_height: float = 100.0) -> None:
        self._provider = provider
        self._flight_height = flight_height

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def convert(
        self,
        image: NDArray,
        n_drones: int,
        flight_bounds: FlightBounds | dict[str, NDArray] | None = None,
        strategy: str = "auto",
    ) -> tuple[NDArray, str, ConversionMetadata]:
        """Convert *image* into *n_drones* 3-D formation positions.

        Args:
            image: BGR or greyscale image as a NumPy array.
            n_drones: Number of drones in the swarm.
            flight_bounds: Flight volume limits.  Accepts a ``FlightBounds``
                instance, the ``dict(lower=..., upper=...)`` format used by
                :pymod:`motion_primitives`, or ``None`` for defaults.
            strategy: ``"auto"``, ``"cv"``, or ``"vlm"``.

        Returns:
            A 3-tuple ``(positions, shape_name, metadata)`` where
            *positions* is shape ``(n_drones, 3)`` in **cm**.
        """
        if n_drones < 1:
            raise ValueError("n_drones must be >= 1")
        bounds = self._resolve_bounds(flight_bounds)
        strategy = self._resolve_strategy(image, strategy)

        if strategy == "cv":
            points_2d = self._cv_extract(image, n_drones)
            shape_name = "cv_contour"
        elif strategy == "vlm":
            points_2d = self._vlm_extract_sync(image, n_drones)
            shape_name = "vlm_shape"
        else:
            raise ValueError(f"Unknown strategy '{strategy}'")

        positions = self._map_to_flight_space(points_2d, bounds)

        metadata = ConversionMetadata(
            strategy=strategy,
            n_drones=n_drones,
            image_shape=tuple(image.shape[:2]),
            edge_density=self._edge_density(image),
            shape_name=shape_name,
        )
        return positions, shape_name, metadata

    # ------------------------------------------------------------------
    # Strategy selection
    # ------------------------------------------------------------------

    def _resolve_strategy(self, image: NDArray, strategy: str) -> str:
        """Normalise and resolve the *auto* strategy."""
        if strategy not in ("auto", "cv", "vlm"):
            raise ValueError(f"Unknown strategy '{strategy}'")
        if strategy != "auto":
            return strategy
        return self._choose_strategy(image)

    def _choose_strategy(self, image: NDArray) -> str:
        """Use edge detection to decide between CV and VLM extraction.

        Images with strong, well-defined edges (e.g. line sketches) are
        handled well by OpenCV contour extraction.  Images with low edge
        density (photographs, complex scenes) are delegated to a VLM.
        """
        density = self._edge_density(image)
        if _EDGE_DENSITY_LOW <= density <= _EDGE_DENSITY_HIGH:
            return "cv"
        if density > _EDGE_DENSITY_HIGH:
            return "cv"
        # Very low edge density -- likely a photograph or complex image
        if self._provider is not None:
            return "vlm"
        logger.warning(
            "Image has low edge density (%.4f) but no VLM provider configured; "
            "falling back to CV extraction",
            density,
        )
        return "cv"

    # ------------------------------------------------------------------
    # Edge density helper
    # ------------------------------------------------------------------

    @staticmethod
    def _edge_density(image: NDArray) -> float:
        """Return the fraction of pixels that are edge pixels."""
        grey = _to_greyscale(image)
        edges = cv2.Canny(grey, 50, 150)
        return float(np.count_nonzero(edges)) / float(edges.size)

    # ------------------------------------------------------------------
    # CV extraction
    # ------------------------------------------------------------------

    def _cv_extract(self, image: NDArray, n_points: int) -> NDArray:
        """Extract *n_points* from image contours using uniform sampling.

        Returns:
            Array of shape ``(n_points, 2)`` with values in ``[0, 1]``.
        """
        grey = _to_greyscale(image)
        blurred = cv2.GaussianBlur(grey, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        if not contours:
            # No contours found -- fall back to a uniform circle
            logger.warning("No contours detected; falling back to circle formation")
            angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
            return np.column_stack([0.5 + 0.4 * np.cos(angles), 0.5 + 0.4 * np.sin(angles)])

        # Merge all contour points and sample uniformly along the combined path
        all_pts = np.vstack(contours).squeeze()  # shape (N, 2)
        if all_pts.ndim == 1:
            all_pts = all_pts.reshape(1, 2)

        # Normalise to [0, 1] based on image dimensions
        h, w = grey.shape[:2]
        all_pts_norm = all_pts.astype(np.float64)
        all_pts_norm[:, 0] /= max(w, 1)
        all_pts_norm[:, 1] /= max(h, 1)

        # Uniform sampling along the point sequence
        n_available = all_pts_norm.shape[0]
        if n_available <= n_points:
            # Not enough points -- interpolate
            indices = np.linspace(0, n_available - 1, n_points)
            sampled = all_pts_norm[np.clip(indices.astype(int), 0, n_available - 1)]
        else:
            indices = np.linspace(0, n_available - 1, n_points, dtype=int)
            sampled = all_pts_norm[indices]

        return sampled

    # ------------------------------------------------------------------
    # VLM extraction
    # ------------------------------------------------------------------

    def _vlm_extract(self, image: NDArray, n_points: int) -> NDArray:
        """Async: use a VLM to extract *n_points* from the image.

        Returns:
            Array of shape ``(n_points, 2)`` with values in ``[0, 1]``.
        """
        if self._provider is None:
            raise RuntimeError("VLM extraction requires a configured LLMProvider")

        prompt = (
            f"Analyze this image and identify the main shape or figure. "
            f"Return exactly {n_points} (x, y) coordinate pairs that trace "
            f"the outline of the shape. Coordinates should be normalised to "
            f"the range [0, 1] where (0,0) is the top-left and (1,1) is the "
            f"bottom-right of the image. "
            f"Respond ONLY with a JSON array of {n_points} [x, y] pairs, "
            f"no other text."
        )

        # Encode image as base64 for the provider
        import base64

        success, encoded = cv2.imencode(".png", image)
        if not success:
            raise RuntimeError("Failed to encode image for VLM")
        b64 = base64.b64encode(encoded.tobytes()).decode("ascii")
        image_source = f"data:image/png;base64,{b64}"

        result = self._provider.analyze_image(prompt=prompt, image_source=image_source)

        return self._parse_vlm_points(result.content, n_points)

    def _vlm_extract_sync(self, image: NDArray, n_points: int) -> NDArray:
        """Synchronous wrapper around :meth:`_vlm_extract`."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None and loop.is_running():
            # We are inside an existing event loop -- run in a thread
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(self._vlm_extract, image, n_points)
                return future.result()
        return self._vlm_extract(image, n_points)

    @staticmethod
    def _parse_vlm_points(content: str, n_points: int) -> NDArray:
        """Parse the VLM text response into an array of 2-D points."""
        import json
        import re

        # Try to extract JSON array from the response
        match = re.search(r"\[.*\]", content, re.DOTALL)
        if match is None:
            raise ValueError(f"VLM response does not contain a JSON array: {content[:200]}")

        try:
            pairs = json.loads(match.group())
        except json.JSONDecodeError as exc:
            raise ValueError(f"VLM response is not valid JSON: {exc}") from exc

        pairs = pairs[:n_points]
        if len(pairs) < n_points:
            # Pad by repeating the last point
            pairs = pairs + [pairs[-1]] * (n_points - len(pairs))

        pts = np.array(pairs, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] != 2:
            raise ValueError(f"VLM returned unexpected point shape: {pts.shape}")
        return np.clip(pts, 0.0, 1.0)

    # ------------------------------------------------------------------
    # Flight-space mapping
    # ------------------------------------------------------------------

    def _map_to_flight_space(self, points_2d: NDArray, bounds: FlightBounds) -> NDArray:
        """Map normalised [0, 1] 2-D points into 3-D flight coordinates.

        The output is in **cm** to match the rest of the codebase.

        Args:
            points_2d: Shape ``(N, 2)`` with values in ``[0, 1]``.
            bounds: Flight volume limits in metres.

        Returns:
            Shape ``(N, 3)`` array of positions in cm.
        """
        n = points_2d.shape[0]
        # Convert bounds to cm
        lower_cm = bounds.lower * 100.0
        upper_cm = bounds.upper * 100.0

        # Use 80% of the available x-y volume to keep drones away from walls
        margin = 0.10
        x_lo = lower_cm[0] + (upper_cm[0] - lower_cm[0]) * margin
        x_hi = upper_cm[0] - (upper_cm[0] - lower_cm[0]) * margin
        y_lo = lower_cm[1] + (upper_cm[1] - lower_cm[1]) * margin
        y_hi = upper_cm[1] - (upper_cm[1] - lower_cm[1]) * margin

        positions = np.zeros((n, 3), dtype=np.float64)
        positions[:, 0] = x_lo + points_2d[:, 0] * (x_hi - x_lo)
        positions[:, 1] = y_lo + points_2d[:, 1] * (y_hi - y_lo)
        positions[:, 2] = self._flight_height

        # Clip z to valid range
        positions[:, 2] = np.clip(positions[:, 2], lower_cm[2], upper_cm[2])
        return positions

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_bounds(bounds: FlightBounds | dict[str, NDArray] | None) -> FlightBounds:
        """Normalise *bounds* to a ``FlightBounds`` instance."""
        if bounds is None:
            return _default_flight_bounds()
        if isinstance(bounds, FlightBounds):
            return bounds
        if isinstance(bounds, dict):
            return FlightBounds(
                lower=np.asarray(bounds["lower"], dtype=np.float64),
                upper=np.asarray(bounds["upper"], dtype=np.float64),
            )
        raise TypeError(f"Unsupported bounds type: {type(bounds)}")


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _to_greyscale(image: NDArray) -> NDArray:
    """Ensure *image* is single-channel greyscale."""
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    if image.ndim == 3 and image.shape[2] == 1:
        return image.squeeze(axis=2)
    raise ValueError(f"Unsupported image shape: {image.shape}")
