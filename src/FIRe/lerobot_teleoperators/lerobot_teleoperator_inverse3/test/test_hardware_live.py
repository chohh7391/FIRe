"""
Live hardware smoke-test for the Inverse3 + VerseGrip via the C++ bridge.

Connects to the real devices, then polls state continuously and prints it at a
human rate. Move the Inverse3 end-effector and rotate / press buttons on the
VerseGrip handle to confirm position, orientation and buttons update.

Run from the repo root (devices must be powered on; handle awake):

    python src/FIRe/lerobot_teleoperators/lerobot_teleoperator_inverse3/\
test/test_hardware_live.py --inverse3 /dev/ttyACM2 --versegrip /dev/ttyACM1

Defaults use the udev symlinks /dev/inverse3_left and /dev/versegrip_left.
Stop with Ctrl+C (the bridge is closed cleanly so the devices are not wedged).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lerobot_teleoperator_inverse3.inverse3_bridge import Inverse3Server


AXIS_CHOICES: tuple[str, ...] = ("+x", "-x", "+y", "-y", "+z", "-z")


def normalize_axis_args(argv: list[str]) -> list[str]:
    normalized: list[str] = []
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg in {"--position-axes", "--rotation-axes"} and i + 3 < len(argv):
            axes = argv[i + 1:i + 4]
            if all(axis in AXIS_CHOICES for axis in axes):
                normalized.append(f"{arg}={','.join(axes)}")
                i += 4
                continue
        normalized.append(arg)
        i += 1
    return normalized


def parse_axes(value: str) -> tuple[str, str, str]:
    axes = tuple(part.strip() for part in value.replace(" ", ",").split(",") if part.strip())
    if len(axes) != 3 or any(axis not in AXIS_CHOICES for axis in axes):
        raise argparse.ArgumentTypeError(
            "expected 3 signed axes from +x, -x, +y, -y, +z, -z"
        )
    if {axis[1] for axis in axes} != {"x", "y", "z"}:
        raise argparse.ArgumentTypeError("axes must use x, y, z exactly once")
    return axes  # type: ignore[return-value]


def map_buttons(args) -> None:
    """Press each VerseGrip button one at a time; prints the bit that toggled."""
    print(f"[map] opening inverse3={args.inverse3} versegrip={args.versegrip} ...")
    server = Inverse3Server(args.inverse3, args.versegrip)
    dur = args.duration or 30.0
    print(f"[map] press each VerseGrip button one at a time for {dur:.0f}s "
          f"(Ctrl+C to stop). Watching for bitmask changes...\n")
    t0 = time.perf_counter()
    prev = None
    try:
        while time.perf_counter() - t0 < dur:
            b = server.get_state().buttons
            if b != prev:
                bits = [i for i in range(8) if (b >> i) & 1]
                print(f"[{time.perf_counter()-t0:6.1f}s] buttons={b:08b} (0x{b:02X})  "
                      f"set bits={bits}")
                prev = b
    except KeyboardInterrupt:
        print("\n[map] interrupted.")
    finally:
        server.close()
        print("[map] done.")


def teleop_live(args) -> None:
    """Drive Inverse3Teleop: hold enable button to move; press calibration to re-home rotation."""
    from scipy.spatial.transform import Rotation
    from lerobot_teleoperator_inverse3 import Inverse3Teleop, Inverse3TeleopConfig

    cfg = Inverse3TeleopConfig(
        inverse3_port=args.inverse3,
        versegrip_port=args.versegrip,
        position_scale=args.position_scale,
        rotation_scale=args.rotation_scale,
        position_axes=tuple(args.position_axes),
        rotation_axes=tuple(args.rotation_axes),
        reanchor_on_enable=not args.absolute_teleop,
        require_calibration=args.require_calibration,
        enable_button=args.enable_button,
        calibration_button=args.calibration_button,
        grasp_button=args.grasp_button,
        end_episode_button=args.end_episode_button,
        gripper_open_value=args.gripper_open_value,
        gripper_close_value=args.gripper_close_value,
        haptic_feedback_enabled=args.haptic_feedback,
    )
    teleop = Inverse3Teleop(cfg)
    print(f"[teleop] connecting {args.inverse3} / {args.versegrip} ...")
    teleop.connect(calibrate=True)
    print(
        f"[teleop] connected.\n"
        f"         HOLD button bit-{cfg.enable_button} to enable + move (pos/rot become non-zero).\n"
        f"         PRESS button bit-{cfg.calibration_button} to re-home rotation "
        f"(rot delta resets to 0 at current orientation).\n"
        f"         Ctrl+C to stop.\n"
    )
    dur = args.duration or 0.0
    print_period = 1.0 / args.print_hz
    t0 = time.perf_counter()
    last_print = 0.0
    try:
        while True:
            a = teleop.get_action()
            now = time.perf_counter()
            if now - last_print >= print_period:
                pos = a["inv3.pos"]
                gripper = float(a["inv3.gripper"][0])
                end_episode = int(a["inv3.end_episode"].item())
                rpy = Rotation.from_quat(
                    [a["inv3.rot"][1], a["inv3.rot"][2], a["inv3.rot"][3], a["inv3.rot"][0]]
                ).as_euler("xyz", degrees=True)
                print(
                    f"\r[{now - t0:6.1f}s] enabled={int(a['inv3.enabled'].item())} "
                    f"btn={int(a['inv3.buttons'][0]):08b} "
                    f"dpos=({pos[0]:+.3f},{pos[1]:+.3f},{pos[2]:+.3f}) "
                    f"drpy=({rpy[0]:+6.1f},{rpy[1]:+6.1f},{rpy[2]:+6.1f})° "
                    f"grip={gripper:+.1f} end={end_episode}   ",
                    end="", flush=True,
                )
                last_print = now
            if dur and (now - t0) >= dur:
                break
    except KeyboardInterrupt:
        print("\n[teleop] interrupted.")
    finally:
        print("\n[teleop] disconnecting ...")
        teleop.disconnect()
        print("[teleop] done.")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--inverse3", default="/dev/inverse3_left")
    p.add_argument("--versegrip", default="/dev/versegrip_left")
    p.add_argument("--duration", type=float, default=0.0, help="seconds; 0 = until Ctrl+C")
    p.add_argument("--print-hz", type=float, default=10.0)
    p.add_argument("--map-buttons", action="store_true",
                   help="print one line whenever the button bitmask changes (for mapping bits)")
    p.add_argument("--teleop", action="store_true",
                   help="drive Inverse3Teleop and print get_action() output (enable + calibration)")
    p.add_argument("--position-scale", type=float, default=2.0)
    p.add_argument("--rotation-scale", type=float, default=1.0)
    p.add_argument("--position-axes", type=parse_axes, default=("-y", "+x", "+z"))
    p.add_argument("--rotation-axes", type=parse_axes, default=("-y", "+x", "+z"))
    p.add_argument("--absolute-teleop", action="store_true")
    p.add_argument("--require-calibration", action="store_true")
    p.add_argument("--enable-button", type=int, default=0)
    p.add_argument("--calibration-button", type=int, default=2)
    p.add_argument("--grasp-button", type=int, default=0)
    p.add_argument("--end-episode-button", type=int, default=1)
    p.add_argument("--gripper-open-value", type=float, default=1.0)
    p.add_argument("--gripper-close-value", type=float, default=-1.0)
    p.add_argument("--haptic-feedback", action="store_true")
    args = p.parse_args(normalize_axis_args(sys.argv[1:]))

    if args.map_buttons:
        map_buttons(args)
        return
    if args.teleop:
        teleop_live(args)
        return

    print(f"[live] opening inverse3={args.inverse3} versegrip={args.versegrip} ...")
    server = Inverse3Server(args.inverse3, args.versegrip)
    print("[live] connected. Move the device / press buttons. Ctrl+C to stop.\n")

    print_period = 1.0 / args.print_hz
    t0 = time.perf_counter()
    last_print = 0.0
    polls = 0
    grip_seen_valid = False
    try:
        while True:
            st = server.get_state()
            polls += 1
            # A valid grip frame yields a unit quaternion; identity (1,0,0,0)
            # is what the bridge emits when the handle is asleep.
            qnorm = float(np.linalg.norm(st.quaternion))
            if not np.allclose(st.quaternion, [1, 0, 0, 0]):
                grip_seen_valid = True

            now = time.perf_counter()
            if now - last_print >= print_period:
                rate = polls / (now - t0)
                print(
                    f"\r[{now - t0:6.1f}s {rate:5.0f}Hz] "
                    f"pos=({st.position[0]:+.4f},{st.position[1]:+.4f},{st.position[2]:+.4f}) "
                    f"quat=({st.quaternion[0]:+.3f},{st.quaternion[1]:+.3f},"
                    f"{st.quaternion[2]:+.3f},{st.quaternion[3]:+.3f}) |q|={qnorm:.3f} "
                    f"btn={st.buttons:08b} bat={st.battery:.0f}",
                    end="",
                    flush=True,
                )
                last_print = now

            if args.duration and (now - t0) >= args.duration:
                break
    except KeyboardInterrupt:
        print("\n[live] interrupted.")
    finally:
        print("\n[live] closing bridge ...")
        server.close()
        print(f"[live] done. polls={polls}, grip valid frames seen={grip_seen_valid}")


if __name__ == "__main__":
    main()
