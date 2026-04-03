#!/usr/bin/env python
"""Test SwarmGPT environment setup."""
import sys

print("=" * 50)
print("SwarmGPT Environment Test")
print("=" * 50)
print(f"Python: {sys.version}")
print()

# Test core imports
errors = []

try:
    from swarm_gpt.core.motion_primitives import motion_primitives, primitive_by_name
    print(f"[OK] Motion primitives: {len(motion_primitives)} registered")
except Exception as e:
    errors.append(f"motion_primitives: {e}")
    print(f"[FAIL] motion_primitives: {e}")

try:
    from swarm_gpt.core.primitive_composer import PrimitiveComposer
    print("[OK] PrimitiveComposer")
except Exception as e:
    errors.append(f"PrimitiveComposer: {e}")
    print(f"[FAIL] PrimitiveComposer: {e}")

try:
    from swarm_gpt.providers import get_provider
    print("[OK] Provider factory")
except Exception as e:
    errors.append(f"providers: {e}")
    print(f"[FAIL] providers: {e}")

try:
    from swarm_gpt.core.multimodal.image_to_formation import ImageFormationConverter, FlightBounds
    print("[OK] ImageFormationConverter, FlightBounds")
except Exception as e:
    errors.append(f"multimodal.image: {e}")
    print(f"[FAIL] multimodal.image: {e}")

try:
    from swarm_gpt.core.multimodal.voice_controller import VoiceController
    print("[OK] VoiceController")
except Exception as e:
    errors.append(f"VoiceController: {e}")
    print(f"[FAIL] VoiceController: {e}")

try:
    from swarm_gpt.core.multimodal.ar_bridge import ARBridge
    print("[OK] ARBridge")
except Exception as e:
    errors.append(f"ARBridge: {e}")
    print(f"[FAIL] ARBridge: {e}")

try:
    from swarm_gpt.core.custom_primitive_generator import (
        PrimitiveSandbox,
        CustomPrimitiveValidator,
        CustomPrimitiveManager,
    )
    print("[OK] Custom primitive modules")
except Exception as e:
    errors.append(f"custom_primitive_generator: {e}")
    print(f"[FAIL] custom_primitive_generator: {e}")

try:
    from swarm_gpt.ui.ui import create_ui
    print("[OK] UI module")
except Exception as e:
    errors.append(f"ui: {e}")
    print(f"[FAIL] ui: {e}")

print()
print("=" * 50)
if errors:
    print(f"FAILED: {len(errors)} errors")
    sys.exit(1)
else:
    print("SUCCESS: All modules imported correctly!")
    sys.exit(0)
