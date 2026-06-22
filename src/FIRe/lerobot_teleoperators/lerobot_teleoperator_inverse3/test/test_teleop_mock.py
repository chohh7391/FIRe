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
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

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
        position_axes=("+x", "+y", "+z"),
        rotation_axes=("+x", "+y", "+z"),
        enable_button=0,      # bit-0 of buttons bitmask
        calibration_button=1,  # bit-1 of buttons bitmask
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
        assert not action["inv3.calibrated"].item()
        assert np.allclose(action["inv3.pos"], 0.0)
        assert np.allclose(action["inv3.gripper"], [1.0])
        assert not action["inv3.end_episode"].item()

        # 2. Rising edge: button pressed for the first time
        mock_server.get_state.return_value = make_state(
            pos=(0.01, 0.0, 0.0), buttons=1  # bit-0 set, device moved 1 cm
        )
        action = teleop.get_action()
        print_action("Rising edge (re-anchor at 1 cm) → delta should be 0", action)
        assert action["inv3.enabled"].item()
        assert np.allclose(action["inv3.gripper"], [-1.0])
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

        # 6. Calibration button (bit-1) re-homes rotation at a tilted pose.
        #    Default home_rot is identity; after calibrating at R, the rotation
        #    delta when enabling at R must be identity (proves rot was re-homed).
        from scipy.spatial.transform import Rotation

        r = Rotation.from_euler("x", 45.0, degrees=True)
        x, y, z, w = r.as_quat()
        q_wxyz = (w, x, y, z)

        # Press calibration button (bit-1) at tilted pose; enable (bit-0) is off.
        mock_server.get_state.return_value = make_state(
            pos=(0.1, 0.2, 0.3), quat=q_wxyz, buttons=0b10
        )
        action = teleop.get_action()
        print_action("Calibration press (bit-1) at 45° tilt → disabled output", action)
        assert not action["inv3.enabled"].item()
        assert action["inv3.calibrated"].item()

        # Now enable at the same pose: position re-anchors, rotation home was set
        # by calibration, so both deltas are ~0.
        mock_server.get_state.return_value = make_state(
            pos=(0.1, 0.2, 0.3), quat=q_wxyz, buttons=0b01
        )
        action = teleop.get_action()
        print_action("Enable at calibrated pose → pos & rot delta = 0", action)
        assert np.allclose(action["inv3.pos"], 0.0, atol=1e-6)
        assert np.allclose(action["inv3.rot"], [1.0, 0.0, 0.0, 0.0], atol=1e-5), \
            f"rotation not re-homed: {action['inv3.rot']}"

        teleop.disconnect()

    cfg = Inverse3TeleopConfig(
        inverse3_port="/dev/inverse3_left",
        versegrip_port="/dev/versegrip_left",
        position_scale=1.0,
        position_axes=("+y", "-x", "+z"),
        rotation_axes=("+y", "-x", "+z"),
        reanchor_on_enable=False,
        enable_button=0,
    )
    teleop = Inverse3Teleop(cfg)
    mock_server = MagicMock()
    mock_server.get_state.return_value = make_state()

    with patch(
        "lerobot_teleoperator_inverse3.inverse3.Inverse3Server",
        return_value=mock_server,
    ):
        teleop.connect(calibrate=True)
        mock_server.get_state.return_value = make_state(
            pos=(0.01, 0.02, 0.03), buttons=1
        )
        action = teleop.get_action()
        print_action("Axis remap (+y,-x,+z), absolute home → expect (0.02,-0.01,0.03)", action)
        assert np.allclose(action["inv3.pos"], [0.02, -0.01, 0.03], atol=1e-6)
        teleop.disconnect()

    cfg = Inverse3TeleopConfig(
        inverse3_port="/dev/inverse3_left",
        versegrip_port="/dev/versegrip_left",
        position_scale=1.0,
        position_axes=("+x", "+y", "+z"),
        rotation_axes=("+x", "+y", "+z"),
        reanchor_on_enable=False,
        require_calibration=True,
        enable_button=-1,
        calibration_button=2,
        grasp_button=0,
        end_episode_button=1,
        gripper_open_value=1.0,
        gripper_close_value=-1.0,
    )
    teleop = Inverse3Teleop(cfg)
    mock_server = MagicMock()
    mock_server.get_state.return_value = make_state()

    with patch(
        "lerobot_teleoperator_inverse3.inverse3.Inverse3Server",
        return_value=mock_server,
    ):
        teleop.connect(calibrate=True)

        mock_server.get_state.return_value = make_state(
            pos=(0.01, 0.0, 0.0), buttons=0b001
        )
        action = teleop.get_action()
        print_action("Require calibration: button0 grasps but motion disabled", action)
        assert not action["inv3.enabled"].item()
        assert np.allclose(action["inv3.gripper"], [-1.0])
        assert not action["inv3.end_episode"].item()

        mock_server.get_state.return_value = make_state(
            pos=(0.01, 0.0, 0.0), buttons=0b100
        )
        action = teleop.get_action()
        print_action("Third button calibrates and enables always-on motion", action)
        assert action["inv3.enabled"].item()
        assert action["inv3.calibrated"].item()
        assert np.allclose(action["inv3.pos"], 0.0, atol=1e-6)

        mock_server.get_state.return_value = make_state(
            pos=(0.02, 0.0, 0.0), buttons=0b010
        )
        action = teleop.get_action()
        print_action("Button1 requests episode end while gripper opens", action)
        assert action["inv3.enabled"].item()
        assert np.allclose(action["inv3.pos"], [0.01, 0.0, 0.0], atol=1e-6)
        assert np.allclose(action["inv3.gripper"], [1.0])
        assert action["inv3.end_episode"].item()
        teleop.disconnect()

    print("\n✓ All assertions passed.")


if __name__ == "__main__":
    run()
