"""GUI module for the gradio web app."""

from __future__ import annotations

import json
import warnings
from typing import TYPE_CHECKING, Callable, List

import gradio as gr
import numpy as np

if TYPE_CHECKING:
    from swarm_gpt.core import AppBackend


def padding_column():
    """Create a column with a hidden textbox to add padding to the UI."""
    with gr.Column():
        gr.Textbox(visible=False)


def centered_markdown(text: str) -> gr.Markdown:
    """Create a centered markdown element.

    Args:
        text: The text to display.

    Returns:
        A markdown element formatted to be centered.
    """
    md = f'<div align="center"> <font size = "10"> <span style="color:grey">{text}</span>'
    return gr.Markdown(md, visible=False)


def update_visibility(visible_flags: List[bool]) -> Callable:
    """Update the visibility of the UI elements.

    We return a function that returns the gradio updates since gradio expects a function instead of
    plain update values.

    Args:
        visible_flags: A list of booleans indicating whether the UI elements should be visible.

    Returns:
        A function that returns the list of gradio updates for the UI elements.
    """

    def gradio_ui_update() -> List[dict]:
        return [gr.update(visible=x) for x in visible_flags]

    return gradio_ui_update


def run_with_bar(backend: AppBackend, progress: gr.Progress = gr.Progress(track_tqdm=True)) -> str:
    """Run the simulation with a progress bar."""
    # Get the generator from your simulation code
    for key, data, total in backend.simulate():
        if key == "progress":
            if data != total:
                percent = int(data / total)
                progress(percent, desc="Simulation Loading...", total=100)
            else:
                progress(100, desc="Simulation Playing", total=100)
        else:
            return "Simulation Playing!"


def create_ui(backend: AppBackend) -> gr.Blocks:
    """Create the gradio web app.

    Args:
        backend: The app backend. This is used to connect the UI to the simulator, AMSwarm and the
            ROS nodes that execute the choreography.

    Returns:
        The UI.
    """
    # Ignore gradio renaming warnings
    warnings.filterwarnings("ignore", category=UserWarning, message="api_name")
    # Define the UI
    with gr.Blocks(theme=gr.themes.Monochrome()) as ui:
        gr.Markdown(
            """ <div align="center"> <font size = "50"> SwarmGPT-Primitive""", elem_id="swarmgpt"
        )
        # Initial window with song selection
        with gr.Row():
            padding_column()
            with gr.Column():
                song_input = gr.Dropdown(
                    choices=backend.songs + backend.presets, label="Select song"
                )
            with gr.Column():
                prompt_choices = list(backend.choreographer.prompts.keys())
                gr.Dropdown(
                    choices=prompt_choices,
                    label="Enter prompt type:",
                    visible=False,
                    interactive=True,
                )
        # Interface during data processing and simulation
        with gr.Row():
            with gr.Column():
                replay_msg = centered_markdown("Replaying simulation")
                sim_msg = centered_markdown("Simulating safe choreography")
                choreo_msg = centered_markdown("LLM is generating choreography")
        # Chatbot and message display
        chatbot = gr.Chatbot(visible=False, type="messages")
        message = gr.Textbox(label="Enter prompt:", visible=False)

        with gr.Row():
            with gr.Column():
                progress_bar = gr.Textbox("Progress", visible=False)

        with gr.Row():
            padding_column()
            with gr.Column():
                replay_sim_button = gr.Button("Replay simulation", visible=False)
                sim_button = gr.Button("Simulate", visible=False)
            with gr.Column():
                alter_button = gr.Button("Refine/Modify the choreography", visible=False)
            padding_column()

        with gr.Row():
            padding_column()
            with gr.Column():
                select_song_button = gr.Button("Choose another song", visible=False)
            padding_column()

        with gr.Row():
            padding_column()
            with gr.Column():
                start_button = gr.Button("Send selections to LLM", visible=False)
                deploy_button = gr.Button("Let the Crazyflies dance", visible=False)
                save_preset_button = gr.Button("Save preset", visible=False)
                show_output = gr.Checkbox(
                    label="Display conversation with LLM",
                    visible=False,
                    value=False,
                    container=True,
                    interactive=True,
                )
            padding_column()

        # Define the UI control flow when the user interacts with the UI elements
        # Song selection flow. On select, the start button and the show output checkbox appear.
        song_input.select(update_visibility([True, True]), [], [start_button, show_output])
        # Start button flow. On click, the song input and start button disappear
        # The choreo message appears
        start_button_flow = start_button.click(
            update_visibility([False, False, True]), [], [song_input, start_button, choreo_msg]
        )
        # The song is handed to the backend start function, and the output of `start` is piped into
        # the chatbot.
        start_button_flow = start_button_flow.success(backend.initial_prompt, song_input, chatbot)
        # The choreo message disappears and the simulate, modify and select song buttons appear
        start_button_flow = start_button_flow.success(
            update_visibility([False, True, True, True, True, True]),
            [],
            [
                choreo_msg,
                sim_button,
                alter_button,
                select_song_button,
                deploy_button,
                save_preset_button,
            ],
        )

        # Alter waypoints flow
        alter_button_flow = alter_button.click(
            lambda: gr.update(visible=True, value=None), [], [message]
        )
        alter_button_flow = alter_button_flow.success(
            update_visibility([False, False, False, False, True]),
            [],
            [alter_button, deploy_button, replay_sim_button, sim_button, chatbot],
        )

        # Show output of the LLM if the checkbox is checked
        def on_select(evt: gr.SelectData) -> dict:
            return gr.update(visible=evt.value)

        show_output.select(on_select, [], [chatbot])  # Toggle chatbot visibility

        # Message flow
        message_flow = message.submit(
            update_visibility([False, False, True]), [], [sim_msg, replay_msg, choreo_msg]
        )
        message_flow = message_flow.success(backend.reprompt, [message], [chatbot])
        message_flow = message_flow.success(
            update_visibility([False, False, True, True, False]),
            [],
            outputs=[alter_button, choreo_msg, sim_button, deploy_button, replay_sim_button],
        )
        message_flow = message_flow.success(
            lambda: gr.update(visible=True, value=None), [], message
        )

        # Sim button flow. On click, the sim message appears and all other messages disappear.
        sim_button_flow = sim_button.click(
            update_visibility([False, False, True, True]),
            [],
            [replay_msg, choreo_msg, sim_msg, progress_bar],
        )
        # AMSwarm is launched and the resulting trajectories are simulated
        sim_button_flow = sim_button_flow.success(
            lambda: run_with_bar(backend), outputs=progress_bar
        )

        # The buttons reappear and the sim message disappears
        sim_button_flow = sim_button_flow.success(
            update_visibility([False, False, True, True, True, True, False]),
            [],
            [
                sim_msg,
                sim_button,
                replay_sim_button,
                alter_button,
                deploy_button,
                select_song_button,
                progress_bar,
            ],
        )
        # Deploy button flow
        deploy_button.click(backend.deploy, [], chatbot)

        # Save preset button flow
        save_preset_button.click(backend.save_preset, [], [])

        # Replay sim button flow
        replay_sim_flow = replay_sim_button.click(
            update_visibility([False, True]), [], [sim_msg, replay_msg]
        )

        replay_sim_flow = replay_sim_flow.success(
            lambda: run_with_bar(backend), outputs=progress_bar
        )

        replay_sim_flow = replay_sim_flow.success(
            update_visibility([False, True]), [], [replay_msg, select_song_button]
        )
        select_song_button.click(None, js="window.location.reload()")

        # ------------------------------------------------------------------
        # Custom Primitive Accordion
        # ------------------------------------------------------------------
        with gr.Accordion("Custom Primitive", open=False):
            cp_name = gr.Textbox(label="Primitive Name", placeholder="e.g. zigzag")
            cp_desc = gr.Textbox(
                label="Description",
                placeholder="Describe the motion primitive in natural language",
                lines=3,
            )
            cp_params = gr.Textbox(
                label="JSON Parameters (optional)",
                placeholder='{"n_args": 0, "params_desc": "()"}',
                lines=2,
            )
            cp_generate_btn = gr.Button("Generate Primitive")
            cp_status = gr.Textbox(label="Status", interactive=False)

            def _on_generate_primitive(
                name: str, desc: str, params_json: str
            ) -> str:
                """Handle the Generate Primitive button click."""
                if not desc.strip():
                    return "Error: description is required."
                kwargs = {}
                if name.strip():
                    kwargs["name"] = name.strip()
                if params_json.strip():
                    try:
                        params = json.loads(params_json)
                    except json.JSONDecodeError as exc:
                        return f"Error: invalid JSON parameters: {exc}"
                    if "n_args" in params:
                        kwargs["n_args"] = int(params["n_args"])
                    if "params_desc" in params:
                        kwargs["params_desc"] = str(params["params_desc"])
                result = backend.create_custom_primitive(desc.strip(), **kwargs)
                if result.get("success"):
                    return f"Primitive '{result.get('name', name)}' created successfully."
                return f"Error: {result.get('error', 'unknown error')}"

            cp_generate_btn.click(
                _on_generate_primitive,
                inputs=[cp_name, cp_desc, cp_params],
                outputs=[cp_status],
            )

        # ------------------------------------------------------------------
        # Image / Sketch Input Accordion
        # ------------------------------------------------------------------
        with gr.Accordion("Image / Sketch Input", open=False):
            img_upload = gr.Image(label="Upload Image", type="numpy")
            img_sketch = gr.Sketchpad(
                label="Sketchpad", type="numpy", brush=gr.Brush(colors=["#000000"])
            )
            img_n_drones = gr.Slider(
                minimum=3, maximum=20, value=6, step=1, label="Number of Drones"
            )
            img_detect_btn = gr.Button("Detect Shape")
            img_preview = gr.Plot(label="Formation Preview")
            img_shape_name = gr.Textbox(label="Detected Shape", interactive=False)

            def _on_detect_shape(
                image: object, sketch: object, n_drones: int
            ) -> tuple:
                """Handle the Detect Shape button click."""
                from swarm_gpt.core.multimodal import FlightBounds, ImageFormationConverter

                # Prefer the uploaded image; fall back to the sketch
                src = image
                if src is None and sketch is not None:
                    # Sketchpad returns a dict with 'composite' or 'background'
                    if isinstance(sketch, dict):
                        src = sketch.get("composite") or sketch.get("background")
                    else:
                        src = sketch
                if src is None:
                    import plotly.graph_objects as go

                    fig = go.Figure()
                    fig.update_layout(title="No image provided")
                    return fig, "No image provided"

                converter = ImageFormationConverter()
                # Build flight bounds from backend settings when available
                try:
                    ax_cfg = backend.settings.get("axswarm", {})
                    lower = np.array(ax_cfg.get("pos_min", [-2.0, -2.0, 0.0]))
                    upper = np.array(ax_cfg.get("pos_max", [2.0, 2.0, 2.0]))
                    bounds = FlightBounds(lower=lower, upper=upper)
                except Exception:
                    bounds = None

                positions, shape_name, _meta = converter.convert(
                    src, int(n_drones), flight_bounds=bounds
                )

                import plotly.graph_objects as go

                fig = go.Figure(
                    data=[
                        go.Scatter3d(
                            x=positions[:, 0],
                            y=positions[:, 1],
                            z=positions[:, 2],
                            mode="markers+text",
                            marker=dict(size=6, color="blue"),
                            text=[str(i) for i in range(len(positions))],
                            textposition="top center",
                        )
                    ]
                )
                fig.update_layout(title=f"Formation: {shape_name}")
                return fig, shape_name

            img_detect_btn.click(
                _on_detect_shape,
                inputs=[img_upload, img_sketch, img_n_drones],
                outputs=[img_preview, img_shape_name],
            )

        # ------------------------------------------------------------------
        # Voice Control Accordion
        # ------------------------------------------------------------------
        with gr.Accordion("Voice Control", open=False):
            voice_audio = gr.Audio(
                label="Record Voice Command", sources=["microphone"], type="numpy"
            )
            voice_transcript = gr.Textbox(label="Transcript", interactive=False)
            voice_parsed = gr.Textbox(label="Parsed Command (JSON)", interactive=False)
            voice_apply_btn = gr.Button("Apply Command")

            def _on_audio_record(audio_data: object) -> tuple:
                """Handle audio input: transcribe and display raw transcript."""
                if audio_data is None:
                    return "", "{}"
                from swarm_gpt.core.multimodal.voice_controller import VoiceController

                vc = VoiceController()
                sample_rate, samples = audio_data
                # Ensure mono float64 for the transcriber
                if samples.ndim > 1:
                    samples = samples.mean(axis=1)
                samples = samples.astype(np.float64)
                try:
                    transcript = vc.transcribe(samples, int(sample_rate))
                except Exception as exc:
                    return f"Transcription error: {exc}", "{}"
                return transcript, "{}"

            voice_audio.change(
                _on_audio_record,
                inputs=[voice_audio],
                outputs=[voice_transcript, voice_parsed],
            )

            def _on_apply_voice_command(transcript: str) -> str:
                """Parse the transcript against current drone positions."""
                from swarm_gpt.core.multimodal.voice_controller import VoiceController

                if not transcript.strip():
                    return json.dumps({"error": "No transcript to parse"})
                vc = VoiceController()
                try:
                    positions = np.array(
                        list(backend.choreographer.starting_pos.values())
                    )
                    positions_cm = positions * 100
                except Exception:
                    return json.dumps({"error": "Could not retrieve drone positions"})
                result = vc.parse_command(transcript.strip(), positions_cm)
                return json.dumps(result, indent=2, default=str)

            voice_apply_btn.click(
                _on_apply_voice_command,
                inputs=[voice_transcript],
                outputs=[voice_parsed],
            )

    return ui
