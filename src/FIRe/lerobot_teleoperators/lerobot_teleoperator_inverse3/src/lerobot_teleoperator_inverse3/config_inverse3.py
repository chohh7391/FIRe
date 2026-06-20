from __future__ import annotations

from dataclasses import dataclass

from lerobot.teleoperators.config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("inverse3")
@dataclass(kw_only=True)
class Inverse3TeleopConfig(TeleoperatorConfig):
    inverse3_port: str = "/dev/inverse3_left"
    versegrip_port: str = "/dev/versegrip_left"
    position_scale: float = 1.0
    rotation_scale: float = 1.0
    enable_button: int = 0  # VerseGrip button bit; hold to enable teleop, re-anchors on press
