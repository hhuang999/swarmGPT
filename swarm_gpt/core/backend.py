"""Backend module for the swarm_gpt web app."""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Literal, ParamSpec, TypeVar

import numpy as np
import yaml
from scipy.interpolate import make_smoothing_spline

from swarm_gpt.core import Choreographer, DroneController
from swarm_gpt.core.custom_primitive_generator import (
    GENERATE_PRIMITIVE_PROMPT,
    CustomPrimitiveManager,
    CustomPrimitiveValidator,
    PrimitiveSandbox,
)
from swarm_gpt.core.sim import simulate_axswarm, simulate_spline
from swarm_gpt.exception import LLMException
from swarm_gpt.utils import MusicManager

logger = logging.getLogger(__name__)

colors = [
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [1.0, 0.7, 0.0],
    [1.0, 0.0, 1.0],
    [0.0, 1.0, 0.5],
]

P = ParamSpec("P")  # Represents arbitrary parameters
R = TypeVar("R")  # Represents the return type


def self_correct(n_retries: int) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Create a decorator that retries a function n times if it fails.

    Args:
        n_retries: Number of times to retry the function
    """

    def decorator(fn: Callable[P, R]) -> Callable[P, R]:
        """Decorator that retries a function n times if it fails."""

        @wraps(fn)
        def wrapper(self: AppBackend, *args: P.args, **kwargs: P.kwargs) -> R:
            assert isinstance(self, AppBackend), "self_correct decorator must be used on AppBackend"
            try:
                return fn(self, *args, **kwargs)
            except LLMException as e:
                error_message = str(e)
                for i in range(n_retries):
                    try:
                        logger.info("Reprompting due to LLM error")
                        message = "The provided response failed with the following error:"
                        message += f"\n{error_message}\n\n"
                        message += "Analyze the error, re-read the instructions and try again."
                        # Use the underlying, undecorated reprompt function to avoid infinite
                        # recursion.
                        return self.reprompt.__wrapped__(self, message)
                    except LLMException as inner_e:
                        if i == n_retries - 1:
                            raise inner_e
                        error_message = str(inner_e)
                        continue
                raise e

        return wrapper

    return decorator


class AppBackend:
    """Backend class for the swarm_gpt gradio web app."""

    def __init__(
        self,
        config_file: Path,
        *,
        music_dir: Path = Path(__file__).parents[2] / "music",
        strict_processing: bool = True,
        strict_drone_match: bool = True,
        model_id: str = "gpt-4o-2024-05-13",
        use_motion_primitives: bool = True,
    ):
        """Initialize the backend by loading the music files and initializing the choreographer.

        Args:
            config_file: Path to the config file.
            music_dir: Path to the music directory.
            strict_processing: Flag to raise an error on waypoint collisions.
            strict_drone_match: Flag to raise an error when preset drones do not match the current
                swarm.
            model_id: The OpenAI GPT model ID.
            use_motion_primitives: If we want LLM to use motion primitives for choreography
        """
        self.root_path = Path(__file__).resolve().parents[2]
        with open(self.root_path / "swarm_gpt/data/settings.yaml", "r") as f:
            self.settings = yaml.safe_load(f)
        # Initialize drone control elements
        self.waypoints = None  # High-level LLM commands
        self.splines = {}  # Low-level optimized commands from axswarm
        self.drone_controller = DroneController(20)  # Controller for the Crazyflie drones
        # Initialize chat elements
        self.choreographer = Choreographer(
            config_file=config_file, model_id=model_id, use_motion_primitives=use_motion_primitives
        )
        self.music_manager = MusicManager(music_dir)
        self.mode: Literal["preset", "real"] = "real"
        self._preset: None | str = None
        self._strict_processing = strict_processing
        self._strict_drone_match = strict_drone_match
        if set(self.songs) & set(self.presets):
            raise ValueError("Songs and presets must have unique names")

    @property
    def songs(self) -> list[str]:
        """List of available songs."""
        return self.music_manager.songs

    @property
    def presets(self) -> list[str]:
        """List of available presets."""
        return [s.name for s in (self.root_path / "swarm_gpt/data/presets").glob("*")]

    @self_correct(n_retries=2)
    def initial_prompt(self, song: str, *, response: str | None = None) -> list[dict[str, str]]:
        """Set the song and generate the choreography.

        Args:
            song: Name of the song or preset to use.
            response: Optional, predefined response. Used for testing.

        Returns:
            The chat history as a list of dictionaries with the role and content.
        """
        logger.info(f"Generating initial choreography for song: {song}")
        song_name = self._load_song(song)
        music_info = self.music_manager.extract_song_info()
        self.choreographer.reset_history()
        prompt = self.choreographer.format_initial_prompt(song_name, music_info)

        fixed_response = response is not None
        if preset := song in self.presets:  # Preset was provided
            logger.debug(f"Loading preset: {song}")
            response = self.load_preset(song)
        elif fixed_response:  # Response was provided, do not use LLM
            logger.debug(f"Using predefined response: {response}")
            self.choreographer.messages.append({"role": "assistant", "content": response})
        else:  # Use LLM to generate the choreography
            logger.debug(f"Using LLM to generate choreography for song: {song_name}")
            response = self.choreographer.generate_choreography(prompt)

        try:
            self.waypoints = self.choreographer.response2waypoints(
                response, music_info=music_info, strict=self._strict_processing
            )
        except LLMException as e:
            # We do not want to retry if we are using a preset or a fixed response. This
            # would use the LLM. We raise an error type that is not caught by
            # self_correct to exit immediately.
            if preset or fixed_response:
                raise RuntimeError("Initial prompt failed") from e
            raise e
        logger.info("Successfully generated choreography")
        return self.choreographer.messages

    @self_correct(n_retries=3)
    def reprompt(self, message: str) -> list[dict[str, str]]:
        """Reprompt the LLM to generate new waypoints based on the previous choreography.

        Args:
            message: The reprompt.

        Returns:
            The chat history as a list of dictionaries with the role and content.
        """
        logger.info(f"Reprompting with message: {message}")
        if message == "":
            logger.warning("No message provided, returning current history")
            return self.choreographer.messages
        prompt = self.choreographer.format_reprompt(message)
        music_info = self.music_manager.extract_song_info()
        response = self.choreographer.generate_choreography(prompt)
        self.waypoints = self.choreographer.response2waypoints(
            response, music_info=music_info, strict=self._strict_processing
        )
        logger.info("Successfully generated choreography")
        return self.choreographer.messages

    def simulate(self: Any, gui: bool = True) -> dict[str, Any]:
        """Run the simulation with waypoints generated by the choreographer.

        Before the simulation is run, the waypoints are interpolated by axswarm to ensure that the
        trajectories are collision-free.

        Args:
            gui: Whether to show the GUI.

        Returns:
            A collection of data from the simulation.
        """
        logger.info("Simulating trajectories with axswarm")
        assert self.waypoints is not None, "Please generate a choreography first"

        for key, data, total in simulate_axswarm(self.waypoints, self.settings, gui=False):
            if key == "progress":
                yield key, data, total
            else:
                sim_data = data
                break
        t = sim_data["timestamps"][::10]
        lam = 0.1  # TODO: Adjust the smoothing parameters
        self.splines.clear()
        for i, drone in self.choreographer.agents.items():
            controls = sim_data["controls"][:, i, :3]
            self.splines[drone] = [
                make_smoothing_spline(t, controls[:, j], lam=lam) for j in range(3)
            ]
        if gui:  # Rerun the simulation of the resulting trajectories with GUI
            simulate_spline(self.splines, self.settings, t[-1], self.music_manager, gui)
        logger.info("Simulation successful")
        return sim_data

    def deploy(self, drone_ids: list[int] | None = None):
        """Run the Crazyflie drones with waypoints generated by the choreographer.

        We call the waypoint_helpers.py script from the Crazyflie ROS package to run the drones.

        Returns:
            The chat history as a list of prompts and answers.
        """
        logger.info("Deploying drones")
        assert self.splines, "Please run the simulation first!"
        # If a deploy version of the song is present, play it
        if not self.drone_controller._ros_running:
            raise RuntimeError("ROS is not running. Please start ROS before deploying.")

        if drone_ids is not None:
            cfs = self.drone_controller.swarm.allcfs.crazyfliesById
            cfs = {k: v for k, v in cfs.items() if k in drone_ids}
            self.drone_controller.swarm.allcfs.crazyfliesById = cfs

        for i, drone in enumerate(self.drone_controller.swarm.allcfs.crazyfliesById.values()):
            drone.setLEDColor(*colors[i % len(colors)])
        original_song = self.music_manager.song
        duration = next(iter(self.waypoints.values()))[-1, 0]
        try:
            self.music_manager.song = original_song + "[deploy]"
        except AssertionError:
            ...
        self.drone_controller.takeoff(target_height=1.0, duration=3.0)
        self.music_manager.play()
        self.drone_controller.run_spline_trajectories(self.splines, duration=duration)
        self.drone_controller.land()
        self.music_manager.song = original_song
        logger.info("Deployment successful")

    def load_preset(self, preset_id: str) -> list[dict[str, str]]:
        """Load a preset response.

        Args:
            preset_id: Name of the preset.
        """
        assert preset_id, "Please select a valid preset"
        assert preset_id in self.presets, "No preset for this song"
        preset_path = self.root_path / "swarm_gpt/data/presets" / preset_id
        n_drones = self.choreographer.num_drones
        preset_n_drones = int(preset_id.split("|")[1].strip())
        if preset_n_drones != n_drones and self._strict_drone_match:
            raise ValueError(
                f"Preset n_drones ({preset_n_drones}) do not match current swarm ({n_drones})"
            )
        with open(preset_path / "history.json", "r") as f:
            history = json.load(f)
        with open(preset_path / "meta.json", "r") as f:
            meta = json.load(f)
        if meta["use_motion_primitives"] != self.choreographer.use_motion_primitives:
            raise ValueError("Preset was generated with a different use_motion_primitives setting")
        assert history[-1]["role"] == "assistant", "Last message in history is not a response"
        self.choreographer.messages = history
        return history[-1]["content"]

    def create_custom_primitive(
        self,
        description: str,
        name: str | None = None,
        params_desc: str = "()",
        n_args: int = 0,
    ) -> dict[str, Any]:
        """Generate, validate, and register a custom motion primitive using the LLM.

        Args:
            description: Natural-language description of the desired primitive.
            name: Optional function name.  Auto-generated from the description if
                not provided.
            params_desc: Human-readable description of the parameters.
            n_args: Number of positional arguments the function expects.

        Returns:
            A dict with ``"success"`` (bool) and either ``"name"`` on success
            or ``"error"`` on failure.
        """
        logger.info("Creating custom primitive: %s", description)

        # Derive a function name from the description if not supplied.
        if name is None:
            name = "_".join(description.lower().split()[:3])
            # Sanitise to valid Python identifier
            name = "".join(c if c.isalnum() or c == "_" else "_" for c in name)
            if not name.isidentifier():
                name = f"custom_{name}"

        sandbox = PrimitiveSandbox()
        validator = CustomPrimitiveValidator()

        # Format the LLM prompt.
        prompt = GENERATE_PRIMITIVE_PROMPT.format(
            func_name=name,
            user_description=description,
            suggested_name=name,
            suggested_params=params_desc,
        )

        # Call the LLM provider to generate code.
        try:
            messages = [
                {"role": "system", "content": "You are a Python code generator for drone motion primitives."},
                {"role": "user", "content": prompt},
            ]
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop and loop.is_running():
                # We are inside an existing event loop (e.g. Gradio) --
                # schedule the coroutine and block until it completes.
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor() as pool:
                    code = pool.submit(
                        asyncio.run, self._call_llm_for_primitive(messages)
                    ).result()
            else:
                code = asyncio.run(self._call_llm_for_primitive(messages))
        except Exception as exc:
            logger.error("LLM code generation failed: %s", exc)
            return {"success": False, "error": str(exc)}

        # Strip markdown fences if the LLM wrapped the code.
        if "```" in code:
            code = code.split("```")[1]
            if code.startswith("python"):
                code = code[len("python"):]
            code = code.strip()

        # Compile in sandbox.
        try:
            func = sandbox.execute(code, name)
        except (SyntaxError, NameError, ImportError) as exc:
            logger.error("Sandbox compilation failed: %s", exc)
            return {"success": False, "error": f"Compilation error: {exc}"}

        # Validate.
        errors = validator.validate(func)
        if errors:
            logger.error("Validation failed: %s", "; ".join(errors))
            return {"success": False, "error": "Validation failed: " + "; ".join(errors)}

        # Register.
        try:
            manager = CustomPrimitiveManager(sandbox=sandbox, validator=validator)
            manager.register(
                name=name,
                func=func,
                code=code,
                description=description,
                params_desc=params_desc,
                n_args=n_args,
            )
        except Exception as exc:
            logger.error("Registration failed: %s", exc)
            return {"success": False, "error": f"Registration error: {exc}"}

        logger.info("Custom primitive '%s' created successfully", name)
        return {"success": True, "name": name}

    async def _call_llm_for_primitive(self, messages: list[dict[str, str]]) -> str:
        """Invoke the choreographer's LLM provider asynchronously.

        Delegates to the synchronous OpenAI / ollama client used by the
        :class:`Choreographer` so that custom primitive generation shares the
        same model and API key configuration.
        """
        return await asyncio.to_thread(
            self.choreographer.generate_choreography, messages
        )

    def save_preset(self):
        """Save the preset."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        preset_name = self.music_manager.song + f" | {self.choreographer.num_drones} | {timestamp}"
        path = self.root_path / "swarm_gpt/data/presets" / preset_name
        path.mkdir(parents=True, exist_ok=True)
        if not self.choreographer.messages:
            raise ValueError("No preset to save. Run Simulation first")
        with open(path / "history.json", "w") as f:
            json.dump(self.choreographer.messages, f)
        meta = {"n_drones": self.choreographer.num_drones, "song": self.music_manager.song}
        meta["use_motion_primitives"] = self.choreographer.use_motion_primitives
        with open(path / "meta.json", "w") as f:
            json.dump(meta, f)
        if self.waypoints is not None:
            np.save(path / "waypoints.npy", self.waypoints)

    def _load_song(self, song: str) -> tuple[str, str]:
        """Load the song on the music manager."""
        if song in self.presets:
            song = song.split("|")[0].strip()
        self.music_manager.song = song
        return song
