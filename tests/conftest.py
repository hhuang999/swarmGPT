import os
import sys
from unittest.mock import MagicMock

# Set dummy API keys before any module imports that create clients at module level
os.environ.setdefault("OPENAI_API_KEY", "test-key-dummy")

# Mock heavy/missing optional dependencies before any test collection
for _mod in (
    "rospy", "rospkg", "rosgraph", "roslib", "rosgraph_msgs",
    "crazyflow", "crazyflow.control", "crazyflow.sim",
    "axswarm", "vlc",
    "pycrazyswarm", "libfmp", "libfmp.c5", "libfmp.c6",
    "librosa", "soundfile", "resampy", "soxr",
    "pydub", "madmom", "midiutil",
    "mutagen", "mutagen.mp3", "mutagen.id3",
    "mujoco", "einops", "jax", "jax.numpy",
    "anthropic", "ollama",
):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

from pathlib import Path

import numpy as np
import yaml


def virtual_crazyswarm_config(n_drones: int) -> Path:
    """Create a virtual crazyswarm config file for testing."""
    # Create grid positions like in llm_success_rates.py
    n_cols = int(np.ceil(np.sqrt(n_drones)))
    n_rows = int(np.ceil(n_drones / n_cols))
    spacing = 1.0

    # Create meshgrid
    x = np.linspace(0, (n_cols - 1) * spacing, n_cols)
    y = np.linspace(0, (n_rows - 1) * spacing, n_rows)
    X, Y = np.meshgrid(x, y)

    # Flatten and combine into positions array
    positions = np.zeros((n_drones, 3))
    positions[:, 0] = X.flatten()[:n_drones]
    positions[:, 1] = Y.flatten()[:n_drones]

    # Center the grid around origin
    positions[:, 0] -= np.mean(positions[:, 0])
    positions[:, 1] -= np.mean(positions[:, 1])

    # Round to integers
    positions = np.round(positions).astype(int)

    config = {
        "crazyflies": [
            {"channel": 80, "id": i, "initialPosition": pos.tolist(), "type": "LSYMultiMarker"}
            for i, pos in enumerate(positions)
        ]
    }

    # Create temporary config file
    tmp_dir = Path("/tmp/swarm_gpt_test")
    tmp_dir.mkdir(exist_ok=True)
    config_path = tmp_dir / "crazyflies.yaml"

    with open(config_path, "w") as f:
        yaml.dump(config, f)

    return config_path
