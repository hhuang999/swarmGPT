"""SwarmGPT is a framework to generate choreographies for drones using the power of LLMs."""

# SwarmGPT Windows Compatibility Shim
# Mock ROS and other Linux-only dependencies before importing anything else

import sys
from unittest.mock import MagicMock

# Mock heavy/missing optional dependencies before any imports
for _mod in (
    "rospy",
    "rospkg",
    "rosgraph",
    "roslib",
    "rosgraph_msgs",
    "crazyflow",
    "crazyflow.control",
    "crazyflow.sim",
    "axswarm",
    "vlc",
    "pycrazyswarm",
    "libfmp",
    "libfmp.c5",
    "libfmp.c6",
    "librosa",
    "soundfile",
    "resampy",
    "soxr",
    "pydub",
    "madmom",
    "midiutil",
    "mutagen",
    "mutagen.mp3",
    "mutagen.id3",
    "mujoco",
    "einops",
    "jax",
    "jax.numpy",
):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

# Set dummy API key if not set
import os

os.environ.setdefault("OPENAI_API_KEY", "test-key-dummy")
