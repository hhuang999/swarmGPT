"""WebSocket / REST bridge for AR/MR preview of drone positions.

Provides a lightweight FastAPI server that broadcasts drone positions in
real-time over WebSocket and exposes REST endpoints for formation queries,
health checks, and commanding individual drones.
"""

from __future__ import annotations

import asyncio
import colorsys
import json
import logging
import threading
from datetime import datetime, timezone
from typing import Any

import numpy as np
from numpy.typing import NDArray

try:
    import uvicorn
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import JSONResponse

    _FASTAPI_AVAILABLE = True
except ImportError:  # pragma: no cover
    _FASTAPI_AVAILABLE = False
    FastAPI = None  # type: ignore[assignment, misc]
    WebSocket = None  # type: ignore[assignment]
    uvicorn = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


class ARBridge:
    """Bridge that serves drone formation data over WebSocket and REST.

    Parameters
    ----------
    host:
        Interface to bind the server on.
    port:
        TCP port for the server.
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 8765) -> None:
        self.host = host
        self.port = port

        # Current formation state
        self._positions: NDArray | None = None
        self._metadata: dict[str, Any] = {}

        # Connected WebSocket clients
        self._ws_clients: list[Any] = []

        # FastAPI app (None when fastapi is not installed)
        self._app: Any = None
        self._server_thread: threading.Thread | None = None

        if _FASTAPI_AVAILABLE:
            self._app = FastAPI(title="SwarmGPT AR Bridge")
            self._setup_routes()
        else:  # pragma: no cover
            logger.warning(
                "FastAPI/uvicorn not installed – ARBridge REST and WebSocket "
                "endpoints are unavailable.  Install with:  pip install fastapi uvicorn websockets"
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_formation(
        self,
        positions: NDArray,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Store the current formation and return a serialisable summary.

        Parameters
        ----------
        positions:
            ``(n_drones, 3)`` numpy array of drone positions in cm.
        metadata:
            Optional dict with extra formation information (formation name,
            music file, etc.).

        Returns
        -------
        dict
            A summary ``{"n_drones": int, "positions": list, "metadata": dict}``.
        """
        positions = np.asarray(positions, dtype=float)
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError(
                f"positions must be (n_drones, 3), got shape {positions.shape}"
            )
        self._positions = positions.copy()
        self._metadata = metadata if metadata is not None else {}

        n_drones = positions.shape[0]
        return {
            "n_drones": n_drones,
            "positions": [
                {
                    "id": i,
                    "pos": positions[i].tolist(),
                    "color": self._get_drone_color(i, n_drones),
                }
                for i in range(n_drones)
            ],
            "metadata": self._metadata,
        }

    def push_trajectory_point(
        self,
        time: float,
        positions: NDArray,
    ) -> dict[str, Any]:
        """Broadcast a single trajectory snapshot to all WebSocket clients.

        Parameters
        ----------
        time:
            Simulation / trajectory time in seconds.
        positions:
            ``(n_drones, 3)`` array of drone positions in cm.

        Returns
        -------
        dict
            The message payload that was broadcast (for testing).
        """
        positions = np.asarray(positions, dtype=float)
        n_drones = positions.shape[0]

        payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "time": float(time),
            "drones": [
                {
                    "id": i,
                    "pos": positions[i].tolist(),
                    "color": self._get_drone_color(i, n_drones),
                }
                for i in range(n_drones)
            ],
        }

        if self._ws_clients and _FASTAPI_AVAILABLE:
            msg = json.dumps(payload)
            for client in list(self._ws_clients):
                try:
                    asyncio.get_event_loop().run_until_complete(
                        client.send_text(msg)
                    )
                except Exception:  # pragma: no cover
                    logger.debug("Failed to send to a WS client", exc_info=True)

        return payload

    def _process_command(self, command: dict[str, Any]) -> dict[str, Any]:
        """Handle a REST command dict and return a response.

        Supported commands
        ------------------
        ``move_drone``
            Requires ``drone_id`` (int) and ``new_position`` ([x, y, z]).

        Returns
        -------
        dict
            ``{"status": "ok", ...}`` on success or
            ``{"status": "error", "detail": ...}`` on failure.
        """
        cmd_type = command.get("type")

        if cmd_type == "move_drone":
            drone_id = command.get("drone_id")
            new_position = command.get("new_position")

            if drone_id is None or new_position is None:
                return {
                    "status": "error",
                    "detail": "move_drone requires drone_id and new_position",
                }

            if self._positions is None:
                return {
                    "status": "error",
                    "detail": "No formation set – call set_formation first",
                }

            drone_id = int(drone_id)
            if drone_id < 0 or drone_id >= self._positions.shape[0]:
                return {
                    "status": "error",
                    "detail": f"drone_id {drone_id} out of range "
                    f"[0, {self._positions.shape[0]})",
                }

            self._positions[drone_id] = np.array(new_position, dtype=float)
            return {
                "status": "ok",
                "drone_id": drone_id,
                "new_position": self._positions[drone_id].tolist(),
            }

        return {"status": "error", "detail": f"Unknown command type: {cmd_type}"}

    # ------------------------------------------------------------------
    # Server lifecycle
    # ------------------------------------------------------------------

    def run_background(self) -> None:
        """Start the FastAPI server in a daemon thread.

        No-op when FastAPI is not installed.
        """
        if not _FASTAPI_AVAILABLE or self._app is None:  # pragma: no cover
            logger.warning("Cannot start server – FastAPI not available")
            return

        if self._server_thread is not None and self._server_thread.is_alive():
            logger.info("AR Bridge server already running")
            return

        def _serve() -> None:
            uvicorn.run(self._app, host=self.host, port=self.port, log_level="warning")

        self._server_thread = threading.Thread(target=_serve, daemon=True)
        self._server_thread.start()
        logger.info("AR Bridge server started on %s:%s", self.host, self.port)

    # ------------------------------------------------------------------
    # Route setup (FastAPI only)
    # ------------------------------------------------------------------

    def _setup_routes(self) -> None:  # type: ignore[union-attr]
        """Register WebSocket and REST endpoints on the FastAPI app."""
        if self._app is None:
            return

        @self._app.websocket("/ws/positions")
        async def ws_positions(websocket: WebSocket) -> None:  # type: ignore[misc]
            await websocket.accept()
            self._ws_clients.append(websocket)
            try:
                while True:
                    # Keep connection alive; client messages are ignored
                    await websocket.receive_text()
            except WebSocketDisconnect:
                pass
            finally:
                if websocket in self._ws_clients:
                    self._ws_clients.remove(websocket)

        @self._app.get("/api/formation")
        async def get_formation() -> JSONResponse:  # type: ignore[misc]
            if self._positions is None:
                return JSONResponse(
                    {"status": "error", "detail": "No formation set"},
                    status_code=404,
                )
            n_drones = self._positions.shape[0]
            return JSONResponse(
                {
                    "status": "ok",
                    "n_drones": n_drones,
                    "positions": [
                        {
                            "id": i,
                            "pos": self._positions[i].tolist(),
                            "color": self._get_drone_color(i, n_drones),
                        }
                        for i in range(n_drones)
                    ],
                    "metadata": self._metadata,
                }
            )

        @self._app.get("/api/health")
        async def health() -> JSONResponse:  # type: ignore[misc]
            return JSONResponse({"status": "ok"})

        @self._app.post("/api/command")
        async def command_endpoint(payload: dict[str, Any]) -> JSONResponse:  # type: ignore[misc]
            result = self._process_command(payload)
            status_code = 200 if result.get("status") == "ok" else 400
            return JSONResponse(result, status_code=status_code)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_drone_color(idx: int, total: int) -> list[int]:
        """Generate an RGB colour for drone *idx* out of *total* using HSV.

        Returns
        -------
        list[int]
            ``[r, g, b]`` where each channel is in ``[0, 255]``.
        """
        if total <= 0:
            total = 1
        hue = (idx / total) % 1.0
        r, g, b = colorsys.hsv_to_rgb(hue, 0.9, 0.95)
        return [int(r * 255), int(g * 255), int(b * 255)]
