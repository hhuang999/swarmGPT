"""Tests for the custom primitive generator.

Covers PrimitiveSandbox, CustomPrimitiveValidator, and CustomPrimitiveManager.
Mocks RestrictedPython when the package is not installed so tests remain
portable in CI environments where it may not be available.
"""

from __future__ import annotations

import importlib
import json
import sys
import tempfile
import types
import unittest.mock as mock
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Import the module under test via spec to avoid pulling in heavy deps
# through swarm_gpt.core.__init__.
# ---------------------------------------------------------------------------

# Ensure swarm_gpt.exception is available
if "swarm_gpt.exception" not in sys.modules:
    _exc = types.ModuleType("swarm_gpt.exception")

    class _LLMFormatError(Exception):
        pass

    _exc.LLMFormatError = _LLMFormatError
    sys.modules["swarm_gpt.exception"] = _exc

# Ensure swarm_gpt.core is importable as a package
_pkg = types.ModuleType("swarm_gpt.core")
_pkg.__path__ = [str(Path(__file__).resolve().parent.parent.parent / "swarm_gpt" / "core")]
_pkg.__package__ = "swarm_gpt.core"
sys.modules.setdefault("swarm_gpt.core", _pkg)

# Load motion_primitives module -- reuse existing sys.modules entry if present
# (e.g. from test_composer.py), otherwise load via spec.
_mp_modname = "swarm_gpt.core.motion_primitives"
if _mp_modname in sys.modules:
    _mp = sys.modules[_mp_modname]
else:
    _mp_spec = importlib.util.spec_from_file_location(
        _mp_modname,
        str(
            Path(__file__).resolve().parent.parent.parent
            / "swarm_gpt" / "core" / "motion_primitives.py"
        ),
    )
    _mp = importlib.util.module_from_spec(_mp_spec)
    sys.modules[_mp_modname] = _mp
    _mp_spec_loader = _mp_spec.loader
    assert _mp_spec_loader is not None
    _mp_spec_loader.exec_module(_mp)

# Load the custom_primitive_generator module
_cpg_modname = "swarm_gpt.core.custom_primitive_generator"
if _cpg_modname in sys.modules:
    _cpg = sys.modules[_cpg_modname]
else:
    _cpg_spec = importlib.util.spec_from_file_location(
        _cpg_modname,
        str(
            Path(__file__).resolve().parent.parent.parent
            / "swarm_gpt" / "core" / "custom_primitive_generator.py"
        ),
    )
    _cpg = importlib.util.module_from_spec(_cpg_spec)
    sys.modules[_cpg_modname] = _cpg
    _cpg_spec_loader = _cpg_spec.loader
    assert _cpg_spec_loader is not None
    _cpg_spec_loader.exec_module(_cpg)

# Pull out classes for use in tests
PrimitiveSandbox = _cpg.PrimitiveSandbox
CustomPrimitiveValidator = _cpg.CustomPrimitiveValidator
CustomPrimitiveManager = _cpg.CustomPrimitiveManager

# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------

VALID_CODE = """\
def my_custom_spiral(params, swarm_pos, tstart, tend, limits):
    steps, radius = params
    n_drones = swarm_pos.shape[0]
    angles = np.linspace(0, 2 * np.pi, n_drones, endpoint=False)
    dt = (tend - tstart) / steps
    waypoints = {}
    pos = swarm_pos.copy()
    for t_idx, t in enumerate(np.linspace(tstart, tend, steps + 1)[1:]):
        angles += 0.5 * dt
        x = radius * np.cos(angles)
        y = radius * np.sin(angles)
        z = np.full(n_drones, 100.0)
        pos = np.column_stack([x, y, z])
        waypoints[t] = {i: pos[i].copy() for i in range(n_drones)}
    return pos, waypoints
"""

# Unsafe code that tries to import os (should be blocked by _safe_import)
UNSAFE_CODE = """\
import os
def unsafe_primitive(params, swarm_pos, tstart, tend, limits):
    os.system("echo pwned")
    return swarm_pos, {}
"""

# Code with wrong signature (missing 'limits' parameter)
BAD_SIGNATURE_CODE = """\
def bad_sig(params, swarm_pos, tstart, tend):
    return swarm_pos, {}
"""

# Code that produces boundary violations
BOUNDARY_VIOLATION_CODE = """\
def out_of_bounds(params, swarm_pos, tstart, tend, limits):
    pos = swarm_pos.copy()
    pos[:, 0] = 9999.0
    waypoints = {tend: {i: pos[i].copy() for i in range(pos.shape[0])}}
    return pos, waypoints
"""


def _make_swarm_pos(n_drones: int = 4) -> NDArray:
    """Create simple grid positions in cm."""
    spacing = 60.0
    positions = []
    for i in range(n_drones):
        row = i // 2
        col = i % 2
        positions.append(
            [col * spacing - spacing / 2, row * spacing - spacing, 100.0]
        )
    return np.array(positions, dtype=float)


def _make_limits() -> dict[str, NDArray]:
    """Standard limits (values in meters)."""
    return {
        "lower": np.array([-2.0, -2.0, 0.0]),
        "upper": np.array([2.0, 2.0, 2.0]),
    }


# ---------------------------------------------------------------------------
# Helper: mock RestrictedPython compile_restricted
# ---------------------------------------------------------------------------

def _mock_compile_restricted(code, **kwargs):
    """Mimic compile_restricted for testing without RestrictedPython installed."""
    result = MagicMock()
    result.errors = ()
    try:
        result.code = compile(code, "<custom_primitive>", "exec")
    except SyntaxError:
        result.errors = ("syntax error",)
        result.code = None
    return result


def _rp_mock():
    """Return a context manager that patches RestrictedPython modules."""
    return mock.patch.dict(sys.modules, {
        "RestrictedPython": MagicMock(compile_restricted=_mock_compile_restricted),
        "RestrictedPython.Eval": MagicMock(),
        "RestrictedPython.Guards": MagicMock(),
    })


# ---------------------------------------------------------------------------
# Tests: PrimitiveSandbox
# ---------------------------------------------------------------------------


class TestPrimitiveSandbox:
    """Tests for PrimitiveSandbox.compile_code and .execute."""

    def test_compile_valid_code(self):
        sandbox = PrimitiveSandbox()
        with _rp_mock():
            compiled = sandbox.compile_code(VALID_CODE)
            assert compiled is not None

    def test_execute_valid_code(self):
        sandbox = PrimitiveSandbox()
        with _rp_mock():
            func = sandbox.execute(VALID_CODE, "my_custom_spiral")
            assert callable(func)
            # Call it to verify it works
            swarm_pos = _make_swarm_pos(4)
            limits = _make_limits()
            final, wps = func((3, 80), swarm_pos, 0.0, 3.0, limits)
            assert isinstance(final, np.ndarray)
            assert isinstance(wps, dict)

    def test_execute_missing_function_raises(self):
        sandbox = PrimitiveSandbox()
        code = "x = 42\n"
        with _rp_mock():
            with pytest.raises(NameError, match="not found"):
                sandbox.execute(code, "nonexistent")

    def test_execute_unsafe_import_blocked(self):
        """Code that tries to import os should fail via _safe_import."""
        sandbox = PrimitiveSandbox()
        with _rp_mock():
            with pytest.raises(ImportError, match="not allowed"):
                sandbox.execute(UNSAFE_CODE, "unsafe_primitive")

    def test_safe_import_allows_numpy(self):
        result = _cpg._safe_import("numpy")
        assert result is np

    def test_safe_import_blocks_os(self):
        with pytest.raises(ImportError, match="not allowed"):
            _cpg._safe_import("os")


# ---------------------------------------------------------------------------
# Tests: CustomPrimitiveValidator
# ---------------------------------------------------------------------------


class TestCustomPrimitiveValidator:
    """Tests for CustomPrimitiveValidator.validate."""

    @staticmethod
    def _valid_func(params, swarm_pos, tstart, tend, limits):
        """A minimal valid primitive function."""
        n_drones = swarm_pos.shape[0]
        pos = swarm_pos.copy()
        waypoints = {
            tend: {i: pos[i].copy() for i in range(n_drones)}
        }
        return pos, waypoints

    @staticmethod
    def _bad_sig_func(params, swarm_pos, tstart, tend):
        return swarm_pos, {}

    def test_valid_function_passes(self):
        v = CustomPrimitiveValidator()
        errors = v.validate(self._valid_func)
        assert errors == []

    def test_bad_signature_fails(self):
        v = CustomPrimitiveValidator()
        errors = v.validate(self._bad_sig_func)
        assert len(errors) > 0
        assert "signature" in errors[0].lower()

    def test_boundary_violation_detected(self):
        v = CustomPrimitiveValidator()

        def oob_func(params, swarm_pos, tstart, tend, limits):
            pos = swarm_pos.copy()
            pos[:, 0] = 9999.0  # far outside limits
            waypoints = {tend: {i: pos[i].copy() for i in range(pos.shape[0])}}
            return pos, waypoints

        errors = v.validate(oob_func)
        assert any("boundary" in e.lower() for e in errors)

    def test_velocity_violation_detected(self):
        v = CustomPrimitiveValidator(max_velocity_cm_s=50.0)

        def fast_func(params, swarm_pos, tstart, tend, limits):
            n = swarm_pos.shape[0]
            pos = swarm_pos.copy()
            pos2 = pos.copy()
            # Move drone 0 a huge amount between two waypoints
            pos2[0] += 300.0  # 300cm in 1s = 300 cm/s
            waypoints = {
                tstart + 1.0: {i: pos[i].copy() for i in range(n)},
                tstart + 2.0: {i: pos2[i].copy() for i in range(n)},
            }
            return pos2, waypoints

        errors = v.validate(fast_func, tend=3.0)
        assert any("velocity" in e.lower() for e in errors)

    def test_collision_detected(self):
        v = CustomPrimitiveValidator(min_separation_cm=30.0)

        def collision_func(params, swarm_pos, tstart, tend, limits):
            pos = swarm_pos.copy()
            # Place all drones at the same position -> collision
            for i in range(pos.shape[0]):
                pos[i] = pos[0]
            waypoints = {tend: {i: pos[i].copy() for i in range(pos.shape[0])}}
            return pos, waypoints

        errors = v.validate(collision_func)
        assert any("collision" in e.lower() for e in errors)

    def test_execution_error_returns_error(self):
        v = CustomPrimitiveValidator()

        def crash_func(params, swarm_pos, tstart, tend, limits):
            raise RuntimeError("boom")

        errors = v.validate(crash_func)
        assert any("execution error" in e.lower() for e in errors)

    def test_bad_return_type(self):
        v = CustomPrimitiveValidator()

        def bad_return(params, swarm_pos, tstart, tend, limits):
            return "not a tuple"

        errors = v.validate(bad_return)
        assert any("must return a tuple" in e for e in errors)


# ---------------------------------------------------------------------------
# Tests: CustomPrimitiveManager
# ---------------------------------------------------------------------------


class TestCustomPrimitiveManager:
    """Tests for CustomPrimitiveManager."""

    @pytest.fixture()
    def tmp_dir(self, tmp_path):
        return tmp_path / "custom_primitives"

    @pytest.fixture()
    def manager(self, tmp_dir):
        sandbox = PrimitiveSandbox()
        validator = CustomPrimitiveValidator()
        return CustomPrimitiveManager(
            sandbox=sandbox,
            validator=validator,
            data_dir=tmp_dir,
        )

    def _make_valid_func(self):
        """Return a standalone valid primitive function."""
        def my_test_prim(params, swarm_pos, tstart, tend, limits):
            n_drones = swarm_pos.shape[0]
            pos = swarm_pos.copy()
            waypoints = {tend: {i: pos[i].copy() for i in range(n_drones)}}
            return pos, waypoints
        return my_test_prim

    def test_register_with_func(self, manager, tmp_dir):
        func = self._make_valid_func()
        manager.register(
            name="my_test_prim",
            func=func,
            description="A test primitive",
            params_desc="(int, int)",
            n_args=2,
        )
        assert "my_test_prim" in manager.list_primitives()
        # Verify persisted to disk
        json_path = tmp_dir / "my_test_prim.json"
        assert json_path.exists()
        data = json.loads(json_path.read_text())
        assert data["name"] == "my_test_prim"
        assert data["description"] == "A test primitive"

    def test_register_with_code(self, manager):
        with _rp_mock():
            # Name must match the function name in VALID_CODE: "my_custom_spiral"
            manager.register(
                name="my_custom_spiral",
                code=VALID_CODE,
                description="Custom spiral",
                params_desc="(steps, radius)",
                n_args=2,
            )
        assert "my_custom_spiral" in manager.list_primitives()

    def test_register_motion_primitives_dict_updated(self, manager):
        """After registration the module-level motion_primitives dict has the entry."""
        func = self._make_valid_func()
        manager.register("dict_test_prim", func=func, n_args=1)
        assert "dict_test_prim" in _mp.motion_primitives
        assert _mp.motion_primitives["dict_test_prim"]["n_args"] == 1

    def test_register_sets_module_attr(self, manager):
        """The function should be set as an attribute on the motion_primitives module."""
        func = self._make_valid_func()
        manager.register("attr_test_prim", func=func, n_args=1)
        assert hasattr(_mp, "attr_test_prim")
        assert _mp.attr_test_prim is func

    def test_unregister(self, manager):
        func = self._make_valid_func()
        manager.register("to_remove", func=func, n_args=1)
        assert "to_remove" in manager.list_primitives()
        manager.unregister("to_remove")
        assert "to_remove" not in manager.list_primitives()

    def test_unregister_removes_from_motion_primitives(self, manager):
        func = self._make_valid_func()
        manager.register("rm_test", func=func, n_args=1)
        manager.unregister("rm_test")
        assert "rm_test" not in _mp.motion_primitives

    def test_unregister_unknown_raises(self, manager):
        with pytest.raises(KeyError, match="not found"):
            manager.unregister("nonexistent")

    def test_register_neither_func_nor_code_raises(self, manager):
        with pytest.raises(ValueError, match="One of"):
            manager.register("bad", n_args=0)

    def test_register_both_func_and_code_raises(self, manager):
        with pytest.raises(ValueError, match="either"):
            manager.register("bad", func=lambda: None, code="x=1", n_args=0)

    def test_register_invalid_code_raises(self, manager):
        with _rp_mock():
            with pytest.raises(RuntimeError, match="validation failed"):
                # Name must match the function name in BAD_SIGNATURE_CODE: "bad_sig"
                manager.register(
                    "bad_sig",
                    code=BAD_SIGNATURE_CODE,
                    n_args=0,
                )

    def test_get_returns_metadata(self, manager):
        func = self._make_valid_func()
        manager.register("meta_test", func=func, description="desc", n_args=1)
        meta = manager.get("meta_test")
        assert meta is not None
        assert meta["description"] == "desc"

    def test_get_unknown_returns_none(self, manager):
        assert manager.get("unknown") is None

    def test_persist_and_reload(self, tmp_dir):
        """Primitives should be reloaded from disk when a new manager is created."""
        sandbox1 = PrimitiveSandbox()
        validator = CustomPrimitiveValidator()
        with _rp_mock():
            mgr1 = CustomPrimitiveManager(
                sandbox=sandbox1, validator=validator, data_dir=tmp_dir
            )
            # Register with code so it can be persisted and reloaded
            mgr1.register(
                "persist_test",
                code=VALID_CODE.replace("my_custom_spiral", "persist_test"),
                n_args=2,
            )

        # Create a second manager pointing to the same dir; must mock RP for reload
        with _rp_mock():
            mgr2 = CustomPrimitiveManager(
                sandbox=sandbox1, validator=validator, data_dir=tmp_dir
            )
        assert "persist_test" in mgr2.list_primitives()


# ---------------------------------------------------------------------------
# Tests: Integration with motion_primitives registry
# ---------------------------------------------------------------------------


class TestIntegration:
    """Verify that registered custom primitives work with primitive_by_name."""

    def test_primitive_by_name_finds_custom(self):
        func = lambda params, swarm_pos, tstart, tend, limits: (swarm_pos, {tend: {}})
        tmp = tempfile.mkdtemp()
        manager = CustomPrimitiveManager(data_dir=Path(tmp))
        manager.register("integ_test", func=func, n_args=0)

        # The motion_primitives module should now have it
        assert "integ_test" in _mp.motion_primitives
        # And primitive_by_name should find it
        result = _mp.primitive_by_name("integ_test")
        assert result is func

        # Cleanup
        manager.unregister("integ_test")
