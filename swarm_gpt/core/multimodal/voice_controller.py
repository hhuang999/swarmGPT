"""Voice controller for drone choreography via Chinese natural-language commands.

Parses spoken Chinese instructions into motion-primitive calls that can be
applied to selected subsets of drones based on their current spatial positions.
"""

from __future__ import annotations

import io
import logging
import re
import wave
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# Spatial keyword  ->  filter function(positions) -> boolean mask
# positions is an (n_drones, 3) array in cm with origin at swarm centre.
SPATIAL_KEYWORDS: dict[str, Callable[[NDArray], NDArray]] = {
    "左边": lambda pos: pos[:, 0] < 0,  # left  (x < 0)
    "右边": lambda pos: pos[:, 0] > 0,  # right (x > 0)
    "前面": lambda pos: pos[:, 1] > 0,  # front (y > 0)
    "后面": lambda pos: pos[:, 1] < 0,  # back  (y < 0)
    "中间": lambda pos: _center_mask(pos),  # centre
    "外围": lambda pos: ~_center_mask(pos),  # perimeter (not centre)
    "全部": lambda pos: np.ones(len(pos), dtype=bool),  # all
    "上面": lambda pos: np.ones(len(pos), dtype=bool),  # above (all drones, z-axis context)
    "下面": lambda pos: np.ones(len(pos), dtype=bool),  # below (all drones, z-axis context)
    "所有": lambda pos: np.ones(len(pos), dtype=bool),  # synonym for 全部
}

# Action keyword  ->  (primitive_name, sign_for_move_z)
# sign indicates direction: +1 for positive z, -1 for negative z, etc.
ACTION_KEYWORDS: dict[str, dict[str, Any]] = {
    "提高": {"primitive": "move_z", "axis": "z", "sign": +1},
    "降低": {"primitive": "move_z", "axis": "z", "sign": -1},
    "左移": {"primitive": "move", "axis": "x", "sign": -1},
    "右移": {"primitive": "move", "axis": "x", "sign": +1},
    "旋转": {"primitive": "rotate", "axis": "z", "sign": +1},
    "散开": {"primitive": "scatter_gather", "axis": None, "sign": +1},
    "聚拢": {"primitive": "center", "axis": None, "sign": +1},
    "升高": {"primitive": "move_z", "axis": "z", "sign": +1},
    "下降": {"primitive": "move_z", "axis": "z", "sign": -1},
    "聚集": {"primitive": "center", "axis": None, "sign": +1},
    "暂停": {"primitive": "stop", "axis": None, "sign": 0},
}

# Default magnitude (cm) when the user does not specify one.
_DEFAULT_MAGNITUDE = 30


def _center_mask(pos: NDArray, fraction: float = 0.4) -> NDArray:
    """Return a boolean mask selecting drones closest to the centroid.

    The *fraction* of drones closest to the centroid are considered "centre".
    When all drones are equidistant (e.g. symmetric layouts) the median
    distance is used as threshold so that roughly half the drones qualify.
    """
    centroid = pos.mean(axis=0)
    dists = np.linalg.norm(pos - centroid, axis=1)
    threshold = np.percentile(dists, fraction * 100)
    # If the threshold covers all drones (all equidistant), fall back to median
    if np.all(dists <= threshold) and len(pos) > 1:
        threshold = np.median(dists)
    return dists <= threshold


class VoiceController:
    """Parse Chinese voice commands into motion-primitive calls.

    Usage::

        vc = VoiceController()
        result = vc.parse_command("把右边的无人机提高30", positions)
        # result = {"action": "提高", "target_drones": [1, 3],
        #           "primitive_call": "move_z([2, 4], 30)"}
    """

    def __init__(self, client: Any = None, language: str = "zh") -> None:
        self.client = client
        self.language = language

    # Magnitude pattern: captures a trailing integer (e.g. "30" in "提高30")
    _MAGNITUDE_RE = re.compile(r"(\d+)\s*(?:厘米|cm)?\s*$")

    def parse_command(self, transcript: str, current_positions: NDArray) -> dict[str, Any]:
        """Parse a Chinese voice transcript into a structured command.

        Args:
            transcript: Raw Chinese text, e.g. "把右边的无人机提高30".
            current_positions: (n_drones, 3) array of drone positions in cm.

        Returns:
            A dict with keys:
              - ``action``: the matched action keyword (str) or ``None``.
              - ``target_drones``: list of **0-indexed** drone ids.
              - ``primitive_call``: a string primitive call that can be fed to
                the choreographer, or ``None`` if parsing failed.
              - ``fallback``: ``True`` when the parser could not fully
                understand the command (the raw transcript is returned so the
                caller can re-prompt an LLM).
        """
        if current_positions.ndim != 2 or current_positions.shape[1] != 3:
            raise ValueError(
                f"current_positions must be (n_drones, 3), got {current_positions.shape}"
            )

        n_drones = len(current_positions)
        all_ids = list(range(n_drones))

        # --- 1. Detect spatial filter ---
        spatial_key = self._match_keyword(transcript, SPATIAL_KEYWORDS)
        if spatial_key is not None:
            mask = SPATIAL_KEYWORDS[spatial_key](current_positions)
            target_ids = [i for i in all_ids if mask[i]]
        else:
            target_ids = list(all_ids)  # default to all if no spatial keyword

        # --- 2. Detect action ---
        action_key = self._match_keyword(transcript, ACTION_KEYWORDS)
        if action_key is None:
            return self._fallback_parse(transcript, current_positions, target_ids)

        # --- 3. Extract magnitude ---
        magnitude = self._extract_magnitude(transcript)

        # --- 4. Build primitive call ---
        primitive_call = self._make_call(action_key, target_ids, magnitude)

        return {
            "action": action_key,
            "target_drones": target_ids,
            "primitive_call": primitive_call,
            "fallback": False,
        }

    def transcribe(self, audio_data: NDArray, sample_rate: int = 16000) -> str:
        """Transcribe audio data to text using the OpenAI Whisper API.

        Args:
            audio_data: 1-D numpy array of mono audio samples.
            sample_rate: Sample rate of the audio in Hz (default 16000).

        Returns:
            The transcribed text (stripped of leading/trailing whitespace).

        Raises:
            RuntimeError: If no OpenAI client is configured on this instance.
        """
        if self.client is None:
            raise RuntimeError(
                "No OpenAI client configured. Pass an OpenAI client to the "
                "VoiceController constructor to enable transcription."
            )

        # Convert mono audio to 16-bit PCM WAV in memory
        buffer = io.BytesIO()
        # Ensure int16 range
        audio_int16 = np.clip(audio_data * 32767, -32768, 32767).astype(np.int16)
        with wave.open(buffer, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit = 2 bytes
            wf.setframerate(sample_rate)
            wf.writeframes(audio_int16.tobytes())
        buffer.seek(0)

        response = self.client.audio.transcriptions.create(
            model="whisper-1",
            file=("audio.wav", buffer),
            language=self.language,
            response_format="text",
        )
        return response.strip()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _match_keyword(transcript: str, keyword_map: dict[str, Any]) -> str | None:
        """Return the first keyword from *keyword_map* found in *transcript*.

        Longer keywords are tried first so that "左边" is not shadowed by "左".
        """
        for kw in sorted(keyword_map, key=len, reverse=True):
            if kw in transcript:
                return kw
        return None

    @staticmethod
    def _extract_magnitude(transcript: str) -> int:
        """Extract a trailing integer magnitude from the transcript.

        Falls back to ``_DEFAULT_MAGNITUDE`` if no number is found.
        """
        m = VoiceController._MAGNITUDE_RE.search(transcript)
        if m:
            return int(m.group(1))
        return _DEFAULT_MAGNITUDE

    @staticmethod
    def _make_call(action: str, drone_ids: list[int], magnitude: int) -> str:
        """Build a primitive-call string for the given action and drones.

        Drone IDs in the call are **1-indexed** (matching LLM / primitive
        convention); *drone_ids* are expected to be 0-indexed internally.
        """
        action_spec = ACTION_KEYWORDS[action]
        primitive = action_spec["primitive"]
        # Convert internal 0-indexed IDs to 1-indexed for the primitive call
        ids_1indexed = [i + 1 for i in drone_ids]

        if primitive == "move_z":
            signed_mag = action_spec["sign"] * magnitude
            return f"move_z({ids_1indexed}, {signed_mag})"

        if primitive == "move":
            # For lateral moves we express a translation along x.
            signed_mag = action_spec["sign"] * magnitude
            return f"move({signed_mag}, 0, 0, {ids_1indexed[0] if len(ids_1indexed) == 1 else ids_1indexed})"

        if primitive == "rotate":
            angle = action_spec["sign"] * magnitude
            return f"rotate({angle}, 'z')"

        if primitive == "scatter_gather":
            return f"scatter_gather(4, {magnitude}, 100)"

        if primitive == "center":
            return f"center({ids_1indexed})"

        if primitive == "stop":
            return f"stop({ids_1indexed})"

        # Fallback for unknown primitives
        return f"{primitive}({ids_1indexed}, {magnitude})"

    @staticmethod
    def _fallback_parse(
        transcript: str, positions: NDArray, drone_ids: list[int]
    ) -> dict[str, Any]:
        """Return a fallback result indicating the command was not understood.

        The caller can use the raw transcript to re-prompt an LLM for a more
        sophisticated parse.
        """
        return {
            "action": "reprompt",
            "target_drones": drone_ids,
            "primitive_call": None,
            "fallback": True,
            "raw_transcript": transcript,
        }
