"""
Quick smoke-test for Inverse3Teleop using a mock Inverse3Server.
Run from the repo root:
    python src/FIRe/lerobot_teleoperators/lerobot_teleoperator_inverse3/test_teleop_mock.py
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

# Allow running without installing the package
sys.path.insert(0, str(Path(__file__).parent / "src"))

from lerobot_teleoperator_inverse3 import Inverse3Teleop, Inverse3TeleopConfig
from lerobot_teleoperator_inverse3.inverse3_bridge import DeviceState


def make_state(
    pos=(0.0, 0.0, 0.0),
    vel=(0.0, 0.0, 0.0),
    quat=(1.0, 0.0, 0.0, 0.0),  # WXYZ identity
    buttons=0,
    battery=80.0,
) -> DeviceState:
    return DeviceState(
        position=np.array(pos, dtype=np.float32),
        velocity=np.array(vel, dtype=np.float32),
        quaternion=np.array(quat, dtype=np.float32),
        buttons=buttons,
        battery=battery,
    )


def print_action(label: str, action: dict) -> None:
    print(f"\n--- {label} ---")
    print(f"  enabled : {action['inv3.enabled']}")
    print(f"  buttons : {action['inv3.buttons']}")
    print(f"  pos     : {action['inv3.pos']}")
    print(f"  rot(wxyz): {action['inv3.rot']}")


def run() -> None:
    cfg = Inverse3TeleopConfig(
        inverse3_port="/dev/inverse3_left",
        versegrip_port="/dev/versegrip_left",
        position_scale=2.0,   # 2× workspace scaling
        rotation_scale=1.0,
        enable_button=0,      # bit-0 of buttons bitmask
    )

    teleop = Inverse3Teleop(cfg)

    # ── Patch Inverse3Server so no real USB device is needed ──────────────
    mock_server = MagicMock()
    mock_server.get_state.return_value = make_state()  # at-home, button off

    with patch(
        "lerobot_teleoperator_inverse3.inverse3.Inverse3Server",
        return_value=mock_server,
    ):
        teleop.connect(calibrate=True)
        print("Connected and calibrated (home = device origin)")

        # 1. Disabled: button bit-0 = 0
        mock_server.get_state.return_value = make_state(buttons=0)
        action = teleop.get_action()
        print_action("Button OFF → should be zeros", action)
        assert not action["inv3.enabled"].item()
        assert np.allclose(action["inv3.pos"], 0.0)

        # 2. Rising edge: button pressed for the first time
        mock_server.get_state.return_value = make_state(
            pos=(0.01, 0.0, 0.0), buttons=1  # bit-0 set, device moved 1 cm
        )
        action = teleop.get_action()
        print_action("Rising edge (re-anchor at 1 cm) → delta should be 0", action)
        assert action["inv3.enabled"].item()
        # re-anchor happened → displacement is 0 at the moment of press
        assert np.allclose(action["inv3.pos"], 0.0, atol=1e-6)

        # 3. Still pressed, device moved further 5 cm in X
        mock_server.get_state.return_value = make_state(
            pos=(0.06, 0.0, 0.0), buttons=1
        )
        action = teleop.get_action()
        print_action("Moved 5 cm in X (scale=2) → expect +0.10 m", action)
        assert action["inv3.enabled"].item()
        expected_x = (0.06 - 0.01) * 2.0   # 5 cm × scale 2 = 0.10 m
        assert abs(action["inv3.pos"][0] - expected_x) < 1e-5, \
            f"Expected {expected_x}, got {action['inv3.pos'][0]}"

        # 4. Button released
        mock_server.get_state.return_value = make_state(
            pos=(0.06, 0.0, 0.0), buttons=0
        )
        action = teleop.get_action()
        print_action("Button released → zeros again", action)
        assert not action["inv3.enabled"].item()

        # 5. Re-pressed at a different position → re-anchors again
        mock_server.get_state.return_value = make_state(
            pos=(0.06, 0.02, 0.0), buttons=1
        )
        action = teleop.get_action()
        print_action("Re-press at (6cm,2cm) → re-anchor, delta = 0", action)
        assert np.allclose(action["inv3.pos"], 0.0, atol=1e-6)

        teleop.disconnect()

    print("\n✓ All assertions passed.")


if __name__ == "__main__":
    run()
