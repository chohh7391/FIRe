from __future__ import annotations

from dataclasses import dataclass

from lerobot.teleoperators.config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("inverse3")
@dataclass(kw_only=True)
class Inverse3TeleopConfig(TeleoperatorConfig):
    inverse3_port: str = "/dev/inverse3_left"
    versegrip_port: str = "/dev/versegrip_left"
    position_scale: float = 3.0
    position_axes: tuple[str, str, str] = ("-y", "+x", "+z")
    rotation_axes: tuple[str, str, str] = ("-y", "+x", "+z")
    # The VerseGrip reports absolute orientation whose horizontal heading (yaw about
    # gravity) is set by its magnetometer / power-on direction and so differs per
    # machine/location. Left uncorrected, that heading offset cross-couples roll and
    # pitch (yaw stays fine). When True, calibration captures the grip's heading and
    # cancels it, so the calibration orientation defines "forward" regardless of the
    # device's absolute heading. Hold the stylus in the canonical pose when calibrating.
    align_heading_on_calibration: bool = True
    reanchor_on_enable: bool = False
    require_calibration: bool = True
    enable_button: int = -1  # VerseGrip button bit; -1 means always enabled
    grasp_button: int = 0
    end_episode_button: int = 1
    gripper_open_value: float = 1.0
    gripper_close_value: float = -1.0
    haptic_feedback_enabled: bool = False
    calibration_button: int = 2  # VerseGrip button bit; press to re-capture home pose (position + rotation)
