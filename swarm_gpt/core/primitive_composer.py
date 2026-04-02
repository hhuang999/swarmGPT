"""Composable primitive operators for combining motion primitives.

Provides three operators:
- sequence: Execute children serially, splitting the time interval evenly.
- parallel: Execute children on different drone groups simultaneously.
- blend: Weighted average of child trajectories.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from swarm_gpt.core.motion_primitives import motion_primitives as motion_primitives_collection
from swarm_gpt.core.motion_primitives import primitive_by_name
from swarm_gpt.exception import LLMFormatError

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# Composition operators that are valid in choreography YAML
COMPOSITION_OPERATORS = {"sequence", "parallel", "blend"}


class CompositionError(LLMFormatError):
    """Raised when a composition expression is malformed."""


class PrimitiveComposer:
    """Compose motion primitives using sequence, parallel, or blend operators.

    The composition tree is built from a parsed YAML dict. Each node in the tree
    is a dict with an ``op`` key (one of ``sequence``, ``parallel``, ``blend``)
    and a ``children`` list of child nodes. Leaf nodes are dicts with a
    ``primitive`` key naming the motion primitive and a ``params`` list of
    arguments.

    Example composition dict (from YAML)::

        op: blend
        weight: 0.7
        children:
          - primitive: rotate
            params: [30, 'z']
          - primitive: wave
            params: [3, 100, [[1,1],[2,2]], [0.5,0.3], [0.2,0.4]]
    """

    def __init__(self) -> None:
        self._validate_cache: dict[int, bool] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @staticmethod
    def is_composition(step_value: str) -> bool:
        """Return True if *step_value* looks like a YAML composition dict.

        A composition step starts with ``op:`` after optional leading
        whitespace, distinguishing it from a simple primitive call string
        such as ``rotate(30, 'z')``.
        """
        stripped = step_value.lstrip()
        return stripped.startswith("op:")

    def parse_composition_yaml(self, data: dict) -> dict:
        """Validate and normalise a composition tree parsed from YAML.

        Args:
            data: The raw dict produced by ``yaml.safe_load`` on a single
                choreography step that uses composition syntax.

        Returns:
            A normalised composition tree dict with validated keys.

        Raises:
            CompositionError: If the tree is structurally invalid or refers to
                unknown primitives/operators.
        """
        if "op" not in data:
            raise CompositionError("Composition dict must contain an 'op' key")
        op = data["op"]
        if op not in COMPOSITION_OPERATORS:
            raise CompositionError(
                f"Unknown composition operator '{op}'. Valid operators: {COMPOSITION_OPERATORS}"
            )
        if "children" not in data or not isinstance(data["children"], list) or not data["children"]:
            raise CompositionError(f"'{op}' composition must have a non-empty 'children' list")

        children: list[dict] = []
        for idx, child in enumerate(data["children"]):
            if not isinstance(child, dict):
                raise CompositionError(
                    f"Child {idx} of '{op}' must be a dict, got {type(child).__name__}"
                )
            if "op" in child:
                # Recursively validate nested composition
                children.append(self.parse_composition_yaml(child))
            elif "primitive" in child:
                # Leaf node - validate primitive name
                prim_name = child["primitive"]
                if prim_name not in motion_primitives_collection:
                    raise CompositionError(
                        f"Unknown primitive '{prim_name}' in child {idx} of '{op}'"
                    )
                if "params" not in child:
                    raise CompositionError(f"Child {idx} ('{prim_name}') must have a 'params' key")
                # Validate number of params
                expected = motion_primitives_collection[prim_name]["n_args"]
                actual = len(child["params"])
                if actual != expected:
                    raise CompositionError(
                        f"Primitive '{prim_name}' in child {idx} expects "
                        f"{expected} params, got {actual}"
                    )
                children.append(child)
            else:
                raise CompositionError(f"Child {idx} of '{op}' must have 'op' or 'primitive' key")

        result: dict = {"op": op, "children": children}

        # Blend-specific: extract and validate weight
        if op == "blend":
            if "weight" not in data or data["weight"] is None:
                raise CompositionError("'blend' composition must have a 'weight' key")
            try:
                weight = float(data["weight"])
            except (TypeError, ValueError) as e:
                raise CompositionError(
                    f"'blend' weight must be a number, got {data['weight']}"
                ) from e
            if not (0.0 <= weight <= 1.0):
                raise CompositionError(f"'blend' weight must be between 0 and 1, got {weight}")
            result["weight"] = weight

        # Parallel-specific: extract and validate drone_groups (optional)
        if op == "parallel":
            if "drone_groups" in data:
                groups = data["drone_groups"]
                if not isinstance(groups, list) or len(groups) != len(children):
                    raise CompositionError(
                        "'parallel' drone_groups must be a list with the same length as children"
                    )
                result["drone_groups"] = groups

        return result

    def execute_composed(
        self,
        composition: dict,
        swarm_pos: NDArray,
        tstart: float,
        tend: float,
        limits: dict[str, NDArray],
    ) -> tuple[NDArray, dict[float, dict[int, NDArray]]]:
        """Execute a composition tree and return the resulting waypoints.

        Args:
            composition: A validated composition tree (output of
                ``parse_composition_yaml``).
            swarm_pos: Current drone positions, shape ``(n_drones, 3)``.
                Values are in centimetres.
            tstart: Start time of this step (seconds).
            tend: End time of this step (seconds).
            limits: Spatial limits dict with ``lower`` and ``upper`` arrays
                (metres).

        Returns:
            A tuple of ``(final_swarm_pos, waypoints)`` where *final_swarm_pos*
            is the updated ``(n_drones, 3)`` array and *waypoints* is a dict
            mapping time -> {drone_id -> position}.

        Raises:
            CompositionError: If execution fails due to incompatible shapes.
        """
        op = composition["op"]
        if op == "sequence":
            return self._execute_sequence(composition, swarm_pos, tstart, tend, limits)
        elif op == "parallel":
            return self._execute_parallel(composition, swarm_pos, tstart, tend, limits)
        elif op == "blend":
            return self._execute_blend(composition, swarm_pos, tstart, tend, limits)
        else:
            raise CompositionError(f"Unknown operator '{op}'")

    # ------------------------------------------------------------------
    # Internal execution helpers
    # ------------------------------------------------------------------

    def _execute_sequence(
        self,
        composition: dict,
        swarm_pos: NDArray,
        tstart: float,
        tend: float,
        limits: dict[str, NDArray],
    ) -> tuple[NDArray, dict[float, dict[int, NDArray]]]:
        """Execute children serially, splitting time evenly."""
        children = composition["children"]
        n = len(children)
        if n == 0:
            return swarm_pos, {}

        all_waypoints: dict[float, dict[int, NDArray]] = {}
        current_pos = swarm_pos.copy()

        for i, child in enumerate(children):
            child_tstart = tstart + i * (tend - tstart) / n
            child_tend = tstart + (i + 1) * (tend - tstart) / n

            if "op" in child:
                current_pos, wps = self.execute_composed(
                    child, current_pos, child_tstart, child_tend, limits
                )
            else:
                current_pos, wps = self._execute_leaf(
                    child, current_pos, child_tstart, child_tend, limits
                )

            all_waypoints.update(wps)

        return current_pos, all_waypoints

    def _execute_parallel(
        self,
        composition: dict,
        swarm_pos: NDArray,
        tstart: float,
        tend: float,
        limits: dict[str, NDArray],
    ) -> tuple[NDArray, dict[float, dict[int, NDArray]]]:
        """Execute children on different drone groups simultaneously.

        If ``drone_groups`` is specified, each child operates on its assigned
        subset of drones. Otherwise, drones are split evenly among children.
        """
        children = composition["children"]
        n_children = len(children)
        n_drones = swarm_pos.shape[0]

        # Determine drone groups
        if "drone_groups" in composition:
            drone_groups = composition["drone_groups"]
        else:
            # Split drones evenly among children
            base = n_drones // n_children
            remainder = n_drones % n_children
            drone_groups = []
            start = 0
            for i in range(n_children):
                count = base + (1 if i < remainder else 0)
                drone_groups.append(list(range(start, start + count)))
                start += count

        all_waypoints: dict[float, dict[int, NDArray]] = {}
        final_pos = swarm_pos.copy()

        for child, group in zip(children, drone_groups):
            if "op" in child:
                child_pos, wps = self.execute_composed(
                    child, swarm_pos[group], tstart, tend, limits
                )
            else:
                child_pos, wps = self._execute_leaf(child, swarm_pos[group], tstart, tend, limits)

            # Map child drone indices back to global indices
            for t, wp_dict in wps.items():
                global_wp: dict[int, NDArray] = {}
                for local_idx, pos in wp_dict.items():
                    global_idx = group[local_idx]
                    global_wp[global_idx] = pos
                if t in all_waypoints:
                    all_waypoints[t].update(global_wp)
                else:
                    all_waypoints[t] = global_wp

            # Update final positions for the group
            for local_idx in range(len(group)):
                final_pos[group[local_idx]] = child_pos[local_idx]

        return final_pos, all_waypoints

    def _execute_blend(
        self,
        composition: dict,
        swarm_pos: NDArray,
        tstart: float,
        tend: float,
        limits: dict[str, NDArray],
    ) -> tuple[NDArray, dict[float, dict[int, NDArray]]]:
        """Execute children independently and blend their trajectories.

        The first child gets weight *w*, the second child gets weight *(1-w)*.
        The blended position at each timestep is the weighted average.
        """
        children = composition["children"]
        weight = composition["weight"]

        if len(children) != 2:
            raise CompositionError("'blend' requires exactly 2 children")

        # Execute both children independently from the same starting position
        child_a_pos, wps_a = self._execute_node(children[0], swarm_pos.copy(), tstart, tend, limits)
        _, wps_b = self._execute_node(children[1], swarm_pos.copy(), tstart, tend, limits)

        # Collect all timesteps from both children
        all_times = sorted(set(list(wps_a.keys()) + list(wps_b.keys())))

        # Interpolate missing timesteps for each child
        interpolated_a = self._interpolate_waypoints(wps_a, all_times, swarm_pos.shape[0])
        interpolated_b = self._interpolate_waypoints(wps_b, all_times, swarm_pos.shape[0])

        # Blend the trajectories
        blended_waypoints: dict[float, dict[int, NDArray]] = {}
        for t in all_times:
            wp: dict[int, NDArray] = {}
            for drone_id in range(swarm_pos.shape[0]):
                pos_a = interpolated_a[t].get(drone_id, swarm_pos[drone_id].copy())
                pos_b = interpolated_b[t].get(drone_id, swarm_pos[drone_id].copy())
                wp[drone_id] = weight * pos_a + (1 - weight) * pos_b
            blended_waypoints[t] = wp

        # Final position is the blend at the last timestep
        final_pos = np.stack(
            [blended_waypoints[all_times[-1]][i] for i in range(swarm_pos.shape[0])]
        )

        return final_pos, blended_waypoints

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _execute_node(
        self, node: dict, swarm_pos: NDArray, tstart: float, tend: float, limits: dict[str, NDArray]
    ) -> tuple[NDArray, dict[float, dict[int, NDArray]]]:
        """Execute a composition node (either a subtree or a leaf)."""
        if "op" in node:
            return self.execute_composed(node, swarm_pos, tstart, tend, limits)
        return self._execute_leaf(node, swarm_pos, tstart, tend, limits)

    @staticmethod
    def _execute_leaf(
        child: dict, swarm_pos: NDArray, tstart: float, tend: float, limits: dict[str, NDArray]
    ) -> tuple[NDArray, dict[float, dict[int, NDArray]]]:
        """Execute a single primitive leaf node."""
        prim_name = child["primitive"]
        params = tuple(child["params"])

        fn = primitive_by_name(prim_name)
        return fn(params, swarm_pos, tstart, tend, limits)

    @staticmethod
    def _interpolate_waypoints(
        waypoints: dict[float, dict[int, NDArray]], target_times: list[float], n_drones: int
    ) -> dict[float, dict[int, NDArray]]:
        """Linearly interpolate waypoint positions for missing timesteps.

        For timesteps present in *waypoints*, the original positions are used.
        For missing timesteps, positions are linearly interpolated between the
        nearest available timesteps. Drones not present in a waypoint keep
        their previous known position.
        """
        if not waypoints:
            return {}

        available_times = sorted(waypoints.keys())
        result: dict[float, dict[int, NDArray]] = {}

        # Track the last known position per drone
        last_known: dict[int, NDArray] = {}

        for t in target_times:
            if t in waypoints:
                result[t] = {d: p.copy() for d, p in waypoints[t].items()}
                for d, p in waypoints[t].items():
                    last_known[d] = p.copy()
                continue

            # Find bracketing timesteps
            prev_t = None
            next_t = None
            for at in available_times:
                if at < t:
                    prev_t = at
                elif at > t:
                    next_t = at
                    break

            wp: dict[int, NDArray] = {}
            for drone_id in range(n_drones):
                if drone_id in last_known:
                    if prev_t is not None and next_t is not None:
                        # Both brackets available - interpolate
                        pos_prev = waypoints[prev_t].get(drone_id)
                        pos_next = waypoints[next_t].get(drone_id)
                        if pos_prev is not None and pos_next is not None:
                            alpha = (t - prev_t) / (next_t - prev_t) if next_t != prev_t else 0.5
                            wp[drone_id] = alpha * pos_next + (1 - alpha) * pos_prev
                            last_known[drone_id] = wp[drone_id].copy()
                        else:
                            wp[drone_id] = last_known[drone_id].copy()
                    elif prev_t is not None:
                        # Only previous available - hold position
                        pos_prev = waypoints[prev_t].get(drone_id)
                        if pos_prev is not None:
                            wp[drone_id] = pos_prev.copy()
                            last_known[drone_id] = wp[drone_id].copy()
                        else:
                            wp[drone_id] = last_known[drone_id].copy()
                    else:
                        # Only next available - hold position
                        wp[drone_id] = last_known[drone_id].copy()
                # Drones with no known position are omitted
            result[t] = wp

        return result
