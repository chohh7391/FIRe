from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation
from lerobot.processor import RobotAction
from lerobot.teleoperators.teleoperator import Teleoperator

from .config_inverse3 import Inverse3TeleopConfig
from .inverse3_bridge import Inverse3Server


def _quat_wxyz_to_rotation(q: np.ndarray) -> Rotation:
    w, x, y, z = q
    return Rotation.from_quat([x, y, z, w])


def _rotation_to_quat_wxyz(r: Rotation) -> np.ndarray:
    x, y, z, w = r.as_quat()
    return np.array([w, x, y, z], dtype=np.float32)


_IDENTITY_QUAT = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)


class Inverse3Teleop(Teleoperator):
    """
    Haption Inverse3 + VerseGrip teleoperator.

    get_action() returns calibrated device-space displacement from the home pose
    set at calibrate() (and re-anchored on the rising edge of the enable button).
    The caller maps this to robot task-space:
        robot_target_pos = robot_home_pos + action["inv3.pos"]
        robot_target_rot = Rotation(action["inv3.rot"]) * robot_home_rot
    """

    config_class = Inverse3TeleopConfig
    name = "inverse3"

    def __init__(self, config: Inverse3TeleopConfig) -> None:
        super().__init__(config)
        self.config = config
        self._server: Inverse3Server | None = None
        self._is_connected = False
        self._inv3_home_pos: np.ndarray = np.zeros(3, dtype=np.float32)
        self._inv3_home_rot: Rotation = Rotation.identity()
        self._enabled: bool = False

    # ── LeRobot abstract properties ──────────────────────────────────────────

    @property
    def action_features(self) -> dict[str, Any]:
        return {
            "inv3.pos": {"dtype": "float32", "shape": (3,), "names": ["x", "y", "z"]},
            "inv3.rot": {"dtype": "float32", "shape": (4,), "names": ["qw", "qx", "qy", "qz"]},
            "inv3.buttons": {"dtype": "int32", "shape": (1,)},
            "inv3.enabled": {"dtype": "bool", "shape": (1,)},
        }

    @property
    def feedback_features(self) -> dict[str, Any]:
        return {
            "force": {"dtype": "float32", "shape": (3,), "names": ["fx", "fy", "fz"]},
        }

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @property
    def is_calibrated(self) -> bool:
        return True

    # ── Connection lifecycle ──────────────────────────────────────────────────

    def connect(self, calibrate: bool = True) -> None:
        if self._is_connected:
            return
        self._server = Inverse3Server(self.config.inverse3_port, self.config.versegrip_port)
        self._is_connected = True
        if calibrate:
            self.calibrate()

    def calibrate(self) -> None:
        """Capture current device pose as home. Enable button re-anchors position on rising edge."""
        state = self._server.get_state()
        self._inv3_home_pos = state.position.copy()
        self._inv3_home_rot = _quat_wxyz_to_rotation(state.quaternion)
        self._enabled = False

    def configure(self) -> None:
        pass

    def disconnect(self) -> None:
        if self._server is not None:
            self._server.close()
            self._server = None
        self._is_connected = False

    # ── Action reading ────────────────────────────────────────────────────────

    def get_action(self) -> RobotAction:
        state = self._server.get_state()

        enabled = bool((state.buttons >> self.config.enable_button) & 1)

        # Rising edge: re-anchor so robot doesn't jump when teleop re-engages
        if enabled and not self._enabled:
            self._inv3_home_pos = state.position.copy()

        self._enabled = enabled

        if not enabled:
            return {
                "inv3.pos": np.zeros(3, dtype=np.float32),
                "inv3.rot": _IDENTITY_QUAT.copy(),
                "inv3.buttons": np.array([state.buttons], dtype=np.int32),
                "inv3.enabled": np.array([False]),
            }

        delta_pos = (state.position - self._inv3_home_pos) * self.config.position_scale

        cur_rot = _quat_wxyz_to_rotation(state.quaternion)
        delta_rot = cur_rot * self._inv3_home_rot.inv()
        if self.config.rotation_scale != 1.0:
            delta_rot = Rotation.from_rotvec(delta_rot.as_rotvec() * self.config.rotation_scale)

        return {
            "inv3.pos": delta_pos.astype(np.float32),
            "inv3.rot": _rotation_to_quat_wxyz(delta_rot),
            "inv3.buttons": np.array([state.buttons], dtype=np.int32),
            "inv3.enabled": np.array([True]),
        }

    # ── Haptic feedback ───────────────────────────────────────────────────────

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        force = np.asarray(
            feedback.get("force", [0.0, 0.0, 0.0]), dtype=np.float32
        )
        self._server.send_force(force)
