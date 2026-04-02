"""Custom primitive generator for user-defined motion primitives.

Compiles LLM-generated Python code in a RestrictedPython sandbox,
validates it for safety, and registers it at runtime into the
motion_primitives registry.
"""

from __future__ import annotations

import inspect
import json
import logging
import types
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# Directory used for persisting custom primitive JSON files.
_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "custom_primitives"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GENERATE_PRIMITIVE_PROMPT = """
You are generating a motion primitive function for a drone swarm system.

Function signature MUST be exactly:
def {func_name}(params, swarm_pos, tstart, tend, limits):
    pass

Parameters:
- params: tuple of parameters the LLM will specify
- swarm_pos: (n_drones, 3) numpy array, current positions in cm
- tstart: float, start time in seconds
- tend: float, end time in seconds
- limits: {{"lower": array, "upper": array}} boundary constraints in meters (multiply by 100 for cm)

Returns: (final_positions, {{time: {{drone_id: position_array}}}})

Constraints:
- Use ONLY numpy (import numpy as np)
- All positions must stay within limits["lower"]*100 and limits["upper"]*100
- Maximum velocity: 50 cm/s per axis
- Use np.clip() to enforce boundaries

User description: {user_description}
Suggested function name: {suggested_name}
Suggested parameters: {suggested_params}

Generate the complete function implementation. Return ONLY the Python code, no markdown.
"""

SAFE_BUILTINS: dict[str, Any] = {
    # Safe builtins (no file I/O, no exec/eval, no imports)
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "filter": filter,
    "float": float,
    "hasattr": hasattr,
    "int": int,
    "isinstance": isinstance,
    "len": len,
    "list": list,
    "map": map,
    "max": max,
    "min": min,
    "print": print,
    "range": range,
    "repr": repr,
    "round": round,
    "set": set,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "type": type,
    "zip": zip,
    "True": True,
    "False": False,
    "None": None,
    # Expose numpy directly (no import needed)
    "np": np,
    "numpy": np,
}

MAX_VELOCITY_CM_S: float = 50.0  # cm/s
MIN_SEPARATION_CM: float = 30.0  # cm

REQUIRED_PARAMS = ("params", "swarm_pos", "tstart", "tend", "limits")


# ---------------------------------------------------------------------------
# PrimitiveSandbox
# ---------------------------------------------------------------------------


class PrimitiveSandbox:
    """Compile LLM-generated code using RestrictedPython with safe builtins.

    Only ``numpy`` is exposed as an importable module.  All other imports
    and unsafe builtins are blocked.
    """

    def __init__(self) -> None:
        self._compiled: Optional[types.CodeType] = None

    def compile_code(self, code: str) -> types.CodeType:
        """Compile *code* inside the RestrictedPython sandbox.

        Args:
            code: Python source string produced by the LLM.

        Returns:
            A compiled code object that can be executed with :meth:`execute`.

        Raises:
            SyntaxError: If the code fails to compile.
        """
        try:
            from RestrictedPython import compile_restricted, safe_globals
            from RestrictedPython.Eval import (
                default_guarded_getattr,
                default_guarded_getitem,
                default_guarded_getiter,
            )
            from RestrictedPython.Guards import guarded_unpack_sequence, safer_getattr
        except ImportError as exc:
            raise ImportError(
                "RestrictedPython is required for the custom primitive sandbox. "
                "Install it with: pip install RestrictedPython"
            ) from exc

        result = compile_restricted(code, filename="<custom_primitive>", mode="exec")
        if result.errors:
            raise SyntaxError("RestrictedPython compilation errors: " + "\n".join(result.errors))

        self._compiled = result.code  # type: ignore[assignment]
        return self._compiled  # type: ignore[return-value]

    def execute(self, code: str, func_name: str) -> Callable:
        """Compile and execute *code*, returning the function named *func_name*.

        Args:
            code: Python source string produced by the LLM.
            func_name: The name of the top-level function to extract.

        Returns:
            The callable extracted from the executed code.

        Raises:
            SyntaxError: If compilation fails.
            NameError: If *func_name* is not defined after execution.
        """
        compiled = self.compile_code(code)

        # Build a restricted global namespace.
        restricted_globals: dict[str, Any] = {}
        # Provide a controlled __builtins__ that includes our safe import
        safe_bi = dict(SAFE_BUILTINS)
        safe_bi["__import__"] = _safe_import
        restricted_globals["__builtins__"] = safe_bi
        # Also expose numpy at top-level so ``import numpy as np`` works
        restricted_globals["np"] = np
        restricted_globals["numpy"] = np

        local_ns: dict[str, Any] = {}
        exec(compiled, restricted_globals, local_ns)  # noqa: S102

        if func_name not in local_ns:
            raise NameError(
                f"Function '{func_name}' not found in the compiled code. "
                f"Available names: {list(local_ns.keys())}"
            )
        return local_ns[func_name]


def _safe_import(name: str, *args: Any, **kwargs: Any) -> Any:
    """Only allow ``numpy`` imports; block everything else."""
    if name == "numpy" or name.startswith("numpy."):
        return __import__(name, *args, **kwargs)
    raise ImportError(f"Import of '{name}' is not allowed in the custom primitive sandbox")


# ---------------------------------------------------------------------------
# CustomPrimitiveValidator
# ---------------------------------------------------------------------------


class CustomPrimitiveValidator:
    """Multi-stage validator for custom motion primitives.

    Stages:
        1. **Signature check** -- the function must accept exactly the five
           standard parameters (``params, swarm_pos, tstart, tend, limits``).
        2. **Return-type check** -- calling the function must produce a
           ``(final_pos, waypoints)`` tuple.
        3. **Boundary check** -- every position must lie within *limits*.
        4. **Velocity check** -- no drone may exceed :data:`MAX_VELOCITY_CM_S`.
        5. **Collision check** -- every pair of drones must stay at least
           :data:`MIN_SEPARATION_CM` apart.
    """

    def __init__(
        self,
        max_velocity_cm_s: float = MAX_VELOCITY_CM_S,
        min_separation_cm: float = MIN_SEPARATION_CM,
    ) -> None:
        self.max_velocity_cm_s = max_velocity_cm_s
        self.min_separation_cm = min_separation_cm

    # -- public API ----------------------------------------------------------

    def validate(
        self,
        func: Callable,
        n_drones: int = 4,
        tstart: float = 0.0,
        tend: float = 3.0,
        limits: Optional[dict[str, NDArray]] = None,
    ) -> List[str]:
        """Run all validation stages and return a list of error messages.

        An empty list means the function passed all checks.
        """
        if limits is None:
            limits = {"lower": np.array([-2.0, -2.0, 0.0]), "upper": np.array([2.0, 2.0, 2.0])}

        errors: List[str] = []

        # Stage 1 -- signature
        sig_errors = self._check_signature(func)
        errors.extend(sig_errors)
        if sig_errors:
            # Cannot continue if signature is wrong.
            return errors

        # Stage 2 -- call the function
        try:
            swarm_pos = self._make_swarm_pos(n_drones)
            params = (1, 100)  # generic placeholder params
            result = func(params, swarm_pos.copy(), tstart, tend, limits)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"Execution error: {exc}")
            return errors

        if not isinstance(result, tuple) or len(result) != 2:
            errors.append(
                f"Function must return a tuple (final_pos, waypoints), got {type(result).__name__}"
            )
            return errors

        final_pos, waypoints = result

        # Stage 3 -- boundary
        errors.extend(self._check_boundaries(final_pos, waypoints, limits))

        # Stage 4 -- velocity
        errors.extend(self._check_velocity(waypoints, tstart, tend))

        # Stage 5 -- collision
        errors.extend(self._check_collisions(waypoints))

        return errors

    # -- private helpers -----------------------------------------------------

    @staticmethod
    def _make_swarm_pos(n_drones: int) -> NDArray:
        """Create a grid of *n_drones* positions in cm."""
        rows = max(1, int(np.sqrt(n_drones)))
        cols = int(np.ceil(n_drones / rows))
        spacing = 60.0
        positions: List[List[float]] = []
        for r in range(rows):
            for c in range(cols):
                if len(positions) >= n_drones:
                    break
                positions.append(
                    [c * spacing - cols * spacing / 2, r * spacing - rows * spacing / 2, 100.0]
                )
        return np.array(positions[:n_drones], dtype=float)

    @staticmethod
    def _check_signature(func: Callable) -> List[str]:
        """Verify the function accepts the required five parameters."""
        try:
            sig = inspect.signature(func)
        except (ValueError, TypeError) as exc:
            return [f"Cannot inspect signature: {exc}"]

        params = list(sig.parameters.keys())
        expected = list(REQUIRED_PARAMS)
        if params != expected:
            return [
                f"Function signature must be ({', '.join(expected)}), got ({', '.join(params)})"
            ]
        return []

    def _check_boundaries(
        self,
        final_pos: NDArray,
        waypoints: Dict[float, Dict[int, NDArray]],
        limits: dict[str, NDArray],
    ) -> List[str]:
        """Check that all positions lie within the spatial limits."""
        errors: List[str] = []
        lower = limits["lower"] * 100  # convert m -> cm
        upper = limits["upper"] * 100

        for t, wp in waypoints.items():
            for drone_id, pos in wp.items():
                for axis in range(3):
                    if pos[axis] < lower[axis] or pos[axis] > upper[axis]:
                        errors.append(
                            f"Boundary violation at t={t:.2f}, drone={drone_id}, "
                            f"axis={axis}: pos={pos[axis]:.1f} cm, "
                            f"limits=[{lower[axis]:.1f}, {upper[axis]:.1f}] cm"
                        )
        return errors

    def _check_velocity(
        self, waypoints: Dict[float, Dict[int, NDArray]], tstart: float, tend: float
    ) -> List[str]:
        """Check that no drone exceeds the maximum velocity."""
        errors: List[str] = []
        times = sorted(waypoints.keys())
        if len(times) < 2:
            return errors

        prev_time = times[0]
        for curr_time in times[1:]:
            dt = curr_time - prev_time
            if dt <= 0:
                continue
            for drone_id in waypoints[curr_time]:
                if drone_id not in waypoints[prev_time]:
                    continue
                dist = np.linalg.norm(
                    waypoints[curr_time][drone_id] - waypoints[prev_time][drone_id]
                )
                velocity = dist / dt  # cm/s
                if velocity > self.max_velocity_cm_s:
                    errors.append(
                        f"Velocity violation: drone={drone_id}, "
                        f"t=[{prev_time:.2f}, {curr_time:.2f}], "
                        f"v={velocity:.1f} cm/s > {self.max_velocity_cm_s:.1f} cm/s"
                    )
            prev_time = curr_time
        return errors

    def _check_collisions(self, waypoints: Dict[float, Dict[int, NDArray]]) -> List[str]:
        """Check that no two drones are closer than the minimum separation."""
        errors: List[str] = []
        for t, wp in waypoints.items():
            positions = np.array([wp[i] for i in sorted(wp.keys())])
            n = len(positions)
            for i in range(n):
                for j in range(i + 1, n):
                    dist = np.linalg.norm(positions[i] - positions[j])
                    if dist < self.min_separation_cm:
                        errors.append(
                            f"Collision risk at t={t:.2f}: "
                            f"drones {i} and {j} are {dist:.1f} cm apart "
                            f"(min {self.min_separation_cm:.1f} cm)"
                        )
        return errors


# ---------------------------------------------------------------------------
# CustomPrimitiveManager
# ---------------------------------------------------------------------------


class CustomPrimitiveManager:
    """Manages the lifecycle of custom motion primitives.

    * Compiles and validates user-supplied code via :class:`PrimitiveSandbox`
      and :class:`CustomPrimitiveValidator`.
    * Registers valid primitives into the module-level ``motion_primitives``
      dict at runtime.
    * Persists primitives as JSON files for reload across sessions.
    """

    def __init__(
        self,
        sandbox: Optional[PrimitiveSandbox] = None,
        validator: Optional[CustomPrimitiveValidator] = None,
        data_dir: Optional[Path] = None,
    ) -> None:
        self._sandbox = sandbox or PrimitiveSandbox()
        self._validator = validator or CustomPrimitiveValidator()
        self._data_dir = data_dir or _DATA_DIR
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._primitives: Dict[str, dict[str, Any]] = {}
        self._load_existing()

    # -- public API ----------------------------------------------------------

    def register(
        self,
        name: str,
        func: Optional[Callable] = None,
        code: Optional[str] = None,
        description: str = "",
        params_desc: str = "",
        n_args: int = 0,
    ) -> None:
        """Register a custom primitive.

        Exactly one of *func* or *code* must be provided.  When *code* is
        provided it is compiled in the sandbox, validated, and then the
        resulting function is used.

        Args:
            name: Identifier for the primitive (used as the function name).
            func: Pre-compiled callable.  If ``None``, *code* must be given.
            code: Python source string to compile in the sandbox.
            description: Human-readable description.
            params_desc: Description of the parameters the function expects.
            n_args: Number of arguments (stored in the registry).

        Raises:
            ValueError: If neither or both of *func* / *code* are given.
            SyntaxError: If sandbox compilation fails.
            RuntimeError: If validation fails.
        """
        if func is not None and code is not None:
            raise ValueError("Provide either 'func' or 'code', not both")
        if func is None and code is None:
            raise ValueError("One of 'func' or 'code' must be provided")

        if code is not None:
            func = self._sandbox.execute(code, name)

        assert func is not None  # for type-checker

        # Validate
        errors = self._validator.validate(func)
        if errors:
            raise RuntimeError(
                "Custom primitive validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
            )

        # Register into the module-level motion_primitives dict
        import swarm_gpt.core.motion_primitives as mp_mod

        mp_mod.motion_primitives[name] = {"n_args": n_args}
        setattr(mp_mod, name, func)

        # Persist metadata
        stored_code = code
        if stored_code is None:
            try:
                stored_code = inspect.getsource(func)
            except (OSError, TypeError):
                stored_code = ""
        entry: Dict[str, Any] = {
            "name": name,
            "description": description,
            "params_desc": params_desc,
            "n_args": n_args,
            "code": stored_code,
        }
        self._primitives[name] = entry
        self._persist(name, entry)

        logger.info("Registered custom primitive '%s'", name)

    def unregister(self, name: str) -> None:
        """Remove a custom primitive from the registry and delete its file."""
        if name not in self._primitives:
            raise KeyError(f"Custom primitive '{name}' not found")

        import swarm_gpt.core.motion_primitives as mp_mod

        mp_mod.motion_primitives.pop(name, None)
        if hasattr(mp_mod, name):
            delattr(mp_mod, name)

        del self._primitives[name]
        json_path = self._data_dir / f"{name}.json"
        if json_path.exists():
            json_path.unlink()

        logger.info("Unregistered custom primitive '%s'", name)

    def list_primitives(self) -> List[str]:
        """Return the names of all registered custom primitives."""
        return list(self._primitives.keys())

    def get(self, name: str) -> Optional[dict[str, Any]]:
        """Return metadata for a registered custom primitive, or ``None``."""
        return self._primitives.get(name)

    # -- private helpers -----------------------------------------------------

    def _persist(self, name: str, entry: dict[str, Any]) -> None:
        """Write *entry* to a JSON file."""
        path = self._data_dir / f"{name}.json"
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(entry, fh, indent=2)

    def _load_existing(self) -> None:
        """Load previously persisted primitives from the data directory."""
        if not self._data_dir.exists():
            return
        for json_path in self._data_dir.glob("*.json"):
            try:
                with open(json_path, encoding="utf-8") as fh:
                    entry = json.load(fh)
                name = entry["name"]
                code = entry.get("code", "")
                if not code:
                    logger.warning("Skipping %s: no code stored", json_path)
                    continue
                # Compile but skip validation on reload (already validated)
                func = self._sandbox.execute(code, name)
                import swarm_gpt.core.motion_primitives as mp_mod

                mp_mod.motion_primitives[name] = {"n_args": entry.get("n_args", 0)}
                setattr(mp_mod, name, func)
                self._primitives[name] = entry
                logger.info("Loaded custom primitive '%s' from disk", name)
            except Exception:  # noqa: BLE001
                logger.warning("Failed to load custom primitive from %s", json_path, exc_info=True)
