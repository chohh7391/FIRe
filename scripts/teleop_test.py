"""teleop_test.py

Robot motion test — circle trajectory in the XY plane.
Rotation is fixed at home. Translation is scaled by --scale.

Usage:
    python scripts/teleop_test.py
    python scripts/teleop_test.py --radius 0.03 --period 6.0 --n_circles 2
"""

from __future__ import annotations

import argparse
import math
import time

import numpy as np

from lerobot_robot_fr3.config_fr3 import FR3RobotConfig
from lerobot_robot_fr3.fr3 import FR3Robot, TeleopAction


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--task",       default="forge-peg_insert")
    p.add_argument("--radius",     type=float, default=0.02,
                   help="Circle radius in metres")
    p.add_argument("--period",     type=float, default=4.0,
                   help="Time for one full circle (s)")
    p.add_argument("--n_circles",  type=int,   default=2,
                   help="Number of circles to execute")
    p.add_argument("--scale",      type=float, default=1.0,
                   help="Translation scale (tune later)")
    p.add_argument("--hold_sec",   type=float, default=2.0,
                   help="Hold at home before/after motion (s)")
    p.add_argument("--hz",         type=float, default=15.0)
    p.add_argument("--use_sim_time", action="store_true")
    return p.parse_args()


def send_absolute(
    robot: FR3Robot, pos: np.ndarray, quat: np.ndarray
) -> None:
    arm = np.concatenate([pos, quat]).astype(np.float32)
    robot.send_teleop_action(TeleopAction(
        action={
            "arm_actions": arm,
            "gripper_actions": np.array([], dtype=np.float32),
        },
        action_space="task_space",
        is_relative=False,
    ))


def main() -> None:
    args = parse_args()
    dt = 1.0 / args.hz

    config = FR3RobotConfig(
        use_sim_time=args.use_sim_time,
        is_relative=False,
        rotation_type="quaternion",
        use_cameras=False,
        use_ft_sensor=False,
    )
    robot = FR3Robot(config, task_name=args.task)

    print("[teleop_test] Connecting to robot...")
    robot.connect()

    home_pos  = robot.robot_state_manager.ee_pos.copy()
    home_quat = robot.robot_state_manager.ee_quat.copy()
    print(f"[teleop_test] Home pos : {home_pos}")
    print(f"[teleop_test] Home quat: {home_quat}")

    radius = args.radius * args.scale

    try:
        # ── Hold home ─────────────────────────────────────────────────────────
        # Mirrors Inverse3 teleop: robot_home is re-captured at enable-button
        # rising edge so the robot stays put when teleop engages.
        print(f"\n[teleop_test] Holding home for {args.hold_sec}s...")
        t_end = time.time() + args.hold_sec
        while time.time() < t_end:
            send_absolute(robot, home_pos, home_quat)
            time.sleep(dt)

        # Re-capture home from actual EEF pose (after settling)
        home_pos  = robot.robot_state_manager.ee_pos.copy()
        home_quat = robot.robot_state_manager.ee_quat.copy()
        print(f"[teleop_test] Settled home pos : {home_pos}")

        # ── Circle (XY plane, rotation fixed) ────────────────────────────────
        total_sec = args.period * args.n_circles
        print(
            f"[teleop_test] Circle: radius={radius*100:.1f}cm  "
            f"period={args.period}s  x{args.n_circles}"
        )
        t_start = time.time()
        t_end   = t_start + total_sec
        while time.time() < t_end:
            elapsed = time.time() - t_start
            angle = 2.0 * math.pi * elapsed / args.period
            target_pos = home_pos.copy()
            target_pos[0] += radius * math.cos(angle)
            target_pos[1] += radius * math.sin(angle)
            send_absolute(robot, target_pos, home_quat)
            time.sleep(dt)

        # ── Return home ───────────────────────────────────────────────────────
        print(f"[teleop_test] Returning to home for {args.hold_sec}s...")
        t_end = time.time() + args.hold_sec
        while time.time() < t_end:
            send_absolute(robot, home_pos, home_quat)
            time.sleep(dt)

        print("[teleop_test] Done.")

    except KeyboardInterrupt:
        print("\n[teleop_test] Interrupted — returning to home...")
        t_end = time.time() + 1.5
        while time.time() < t_end:
            send_absolute(robot, home_pos, home_quat)
            time.sleep(dt)

    finally:
        robot.disconnect()


if __name__ == "__main__":
    main()
