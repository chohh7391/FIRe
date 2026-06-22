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
    reanchor_on_enable: bool = True
    require_calibration: bool = False
    enable_button: int = 0  # VerseGrip button bit; set to -1 for always enabled
    grasp_button: int = 0
    end_episode_button: int = 1
    gripper_open_value: float = 1.0
    gripper_close_value: float = -1.0
    haptic_feedback_enabled: bool = False
    calibration_button: int = 2  # VerseGrip button bit; press to re-capture home pose (position + rotation)
