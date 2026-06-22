from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation
from lerobot.processor import RobotAction
from lerobot.teleoperators.teleoperator import Teleoperator

from .config_inverse3 import Inverse3TeleopConfig
from .inverse3_bridge import Inverse3Server


_AXIS_TO_INDEX: dict[str, int] = {"x": 0, "y": 1, "z": 2}


def _axis_mapping_matrix(
    axes: tuple[str, str, str], *, require_proper_rotation: bool
) -> np.ndarray:
    if len(axes) != 3:
        raise ValueError(f"expected 3 axes, got {axes!r}")

    mat = np.zeros((3, 3), dtype=np.float64)
    used: set[int] = set()
    for row, axis in enumerate(axes):
        if len(axis) != 2 or axis[0] not in ("+", "-") or axis[1] not in _AXIS_TO_INDEX:
            raise ValueError(
                f"invalid axis spec {axis!r}; expected values like '+x', '-y', '+z'"
            )
        col = _AXIS_TO_INDEX[axis[1]]
        if col in used:
            raise ValueError(f"axis specs must use x, y, z exactly once, got {axes!r}")
        used.add(col)
        mat[row, col] = 1.0 if axis[0] == "+" else -1.0

    det = float(np.linalg.det(mat))
    if require_proper_rotation and not np.isclose(det, 1.0):
        raise ValueError(
            "rotation_axes must define a right-handed frame mapping with determinant +1; "
            f"got {axes!r} with determinant {det:+.0f}"
        )
    return mat


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
    set at calibrate() (and optionally re-anchored on the rising edge of the enable button).
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
        self._is_calibrated_by_button: bool = not self.config.require_calibration
        self._calib_pressed: bool = False
        self._position_map: np.ndarray = _axis_mapping_matrix(
            self.config.position_axes,
            require_proper_rotation=False,
        )
        self._rotation_map_rot: Rotation = Rotation.from_matrix(
            _axis_mapping_matrix(
                self.config.rotation_axes,
                require_proper_rotation=True,
            )
        )

    # ── LeRobot abstract properties ──────────────────────────────────────────

    @property
    def action_features(self) -> dict[str, Any]:
        return {
            "inv3.pos": {"dtype": "float32", "shape": (3,), "names": ["x", "y", "z"]},
            "inv3.rot": {"dtype": "float32", "shape": (4,), "names": ["qw", "qx", "qy", "qz"]},
            "inv3.buttons": {"dtype": "int32", "shape": (1,)},
            "inv3.enabled": {"dtype": "bool", "shape": (1,)},
            "inv3.calibrated": {"dtype": "bool", "shape": (1,)},
            "inv3.gripper": {"dtype": "float32", "shape": (1,)},
            "inv3.end_episode": {"dtype": "bool", "shape": (1,)},
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
        self._server = Inverse3Server(
            self.config.inverse3_port,
            self.config.versegrip_port,
        )
        self._is_connected = True
        self._server.send_force(np.zeros(3, dtype=np.float32))
        if calibrate:
            self.calibrate()

    def calibrate(self) -> None:
        """Capture current device pose as home. Enable button re-anchors position on rising edge."""
        state = self._server.get_state()
        self._inv3_home_pos = state.position.copy()
        self._inv3_home_rot = _quat_wxyz_to_rotation(state.quaternion)
        self._enabled = False
        self._is_calibrated_by_button = not self.config.require_calibration
        self._calib_pressed = False

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
        cur_rot = _quat_wxyz_to_rotation(state.quaternion)

        # Calibration button (rising edge): re-capture the home pose, including
        # the rotation reference, so the current device orientation becomes the
        # new zero for the rotation delta. Works regardless of enable state.
        calib = bool((state.buttons >> self.config.calibration_button) & 1)
        calibrated = calib and not self._calib_pressed
        if calibrated:
            self._inv3_home_pos = state.position.copy()
            self._inv3_home_rot = cur_rot
            self._is_calibrated_by_button = True
        self._calib_pressed = calib

        if self.config.enable_button < 0:
            enabled = True
        else:
            enabled = bool((state.buttons >> self.config.enable_button) & 1)
        enabled = enabled and self._is_calibrated_by_button

        gripper_action = self._get_gripper_action(state.buttons)
        end_episode = bool((state.buttons >> self.config.end_episode_button) & 1)

        # Rising edge: re-anchor position so robot doesn't jump when teleop re-engages.
        # Disable this for absolute teleop, where the calibration pose remains
        # the fixed device origin for the whole episode.
        if self.config.reanchor_on_enable and enabled and not self._enabled:
            self._inv3_home_pos = state.position.copy()

        self._enabled = enabled

        if not enabled:
            return {
                "inv3.pos": np.zeros(3, dtype=np.float32),
                "inv3.rot": _IDENTITY_QUAT.copy(),
                "inv3.buttons": np.array([state.buttons], dtype=np.int32),
                "inv3.enabled": np.array([False]),
                "inv3.calibrated": np.array([calibrated]),
                "inv3.gripper": np.array([gripper_action], dtype=np.float32),
                "inv3.end_episode": np.array([end_episode]),
            }

        delta_pos = self._position_map @ (state.position - self._inv3_home_pos)
        delta_pos = delta_pos * self.config.position_scale

        delta_rot = cur_rot * self._inv3_home_rot.inv()
        delta_rot = self._rotation_map_rot * delta_rot * self._rotation_map_rot.inv()
        if self.config.rotation_scale != 1.0:
            delta_rot = Rotation.from_rotvec(delta_rot.as_rotvec() * self.config.rotation_scale)

        return {
            "inv3.pos": delta_pos.astype(np.float32),
            "inv3.rot": _rotation_to_quat_wxyz(delta_rot),
            "inv3.buttons": np.array([state.buttons], dtype=np.int32),
            "inv3.enabled": np.array([True]),
            "inv3.calibrated": np.array([calibrated]),
            "inv3.gripper": np.array([gripper_action], dtype=np.float32),
            "inv3.end_episode": np.array([end_episode]),
        }

    def _get_gripper_action(self, buttons: int) -> float:
        grasp_pressed = bool((buttons >> self.config.grasp_button) & 1)
        if grasp_pressed:
            return float(self.config.gripper_close_value)
        return float(self.config.gripper_open_value)

    # ── Haptic feedback ───────────────────────────────────────────────────────

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        if not self.config.haptic_feedback_enabled:
            if self._server is not None:
                self._server.send_force(np.zeros(3, dtype=np.float32))
            return
        force = np.asarray(
            feedback.get("force", [0.0, 0.0, 0.0]), dtype=np.float32
        )
        self._server.send_force(force)
