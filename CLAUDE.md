# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**FIRe** (Force Informed Residual) is a system for running RL-trained manipulation policies on a Franka FR3 robot. The RL policy (trained with `rl_games`/PPO) takes a 24-dimensional force-aware observation and outputs 7-dimensional actions. A separate VisionServer streams RGB frames from three cameras (wrist, left, right) via ZMQ and optionally applies SAM3 segmentation masks.

## Environment Setup

Two separate conda environments are required:

**Main environment (`fire`, Python 3.10):**
```bash
conda create -n fire python=3.10
conda activate fire
pip install -r requirements.txt  # installs lerobot_robot_fr3, lerobot_ft_sensor, and FIRe as editable packages
```

**Vision environment (`sam3`, Python 3.12):**
```bash
conda create -n sam3 python=3.12
conda activate sam3
pip install -e src/FIRe/vision_server
```

ROS 2 workspace must be sourced before any robot-facing scripts:
```bash
source ~/ros2_ws/install/setup.bash
```

Export HuggingFace token for SAM3 model download:
```bash
export HUGGING_FACE_HUB_TOKEN="YOUR_TOKEN"
```

## Running

**Bringup robot (ROS 2 controller):**
```bash
ros2 launch cho_franka_bringup bringup_gazebo_robot.launch.py vla:=true control_mode:=torque
```

**Vision server (in `sam3` env):**
```bash
python scripts/run_vision_server.py                          # plain camera stream
python scripts/run_vision_server.py --use_sam3 --target_object wristwatch  # with SAM3 masking
```

**Test robot connection:**
```bash
python test/test_lerobot_robot_fr3.py
```

**Run RL policy:**
```bash
python scripts/play.py \
  --checkpoint checkpoints/VLA_RL-BL-forge-peg_insert/nn/Forge.pth \
  --cfg scripts/configs/rl_games_ppo_cfg.yaml \
  --obs_dim 24 --action_dim 7 --device cuda:0 --control_hz 15.0
```

**Replay CSV actions on robot:**
```bash
python scripts/play_with_csv.py --csv_path scripts/configs/traj_save.csv --hz 15.0 --log
```

**Benchmark RL inference without robot:**
```bash
python scripts/dummy_inference.py \
  --checkpoint checkpoints/VLA_RL-BL-forge-peg_insert/nn/Forge.pth \
  --cfg scripts/configs/rl_games_ppo_cfg.yaml \
  --obs_dim 24 --action_dim 7 --num_steps 1000
```

## Architecture

### Package layout

`requirements.txt` installs three packages as editable from `src/`:
- `lerobot_robot_fr3` — FR3 robot driver
- `lerobot_ft_sensor` — Bota FT sensor driver
- `FIRe` (top-level) — vision server

### FR3Robot (`fr3.py`)

`FR3Robot` extends lerobot's `Robot` base class. It manages three concurrent subsystems in a single ROS 2 node with a `MultiThreadedExecutor` (4 threads) running in a background daemon thread:

| Subsystem | Class | Source |
|---|---|---|
| Robot state (EE pose, velocity, joints) | `RobotStateManager` | subscribes `/ee_state/pose`, `/ee_state/twist`, `/joint_states` |
| Cameras | `CameraSensorManager` | polls ZMQ `ZMQCamera`s, publishes `/camera/<name>/image_raw` at 30 Hz |
| FT sensor | `FTSensorManager` | wraps `FTSensor` (Bota driver), publishes `/ft_sensor/wrench` at 100 Hz |

On `connect()`, `FR3Robot` also sends a `VisionLanguageAction` goal to the ROS 2 action server `/controller_action_server/vla_controller` to claim control of the robot. The action server must accept the goal before any actions are published.

### Observation spaces

`get_observation()` returns a dict assembled into a 24-D tensor for the RL policy:

| Key | Dim | Source |
|---|---|---|
| `fingertip_pos_rel_fixed` | 3 | EE pos minus fixed target point |
| `fingertip_quat` | 4 | EE orientation (wxyz) |
| `ee_linvel` | 3 | `RobotStateManager` |
| `ee_angvel` | 3 | `RobotStateManager` |
| `force_threshold` | 1 | random jitter `[5, 10]` N per step |
| `ft_force` | 3 | `FTSensorManager` (scaled, force only) |
| `prev_actions` | 7 | EMA-smoothed previous action |

`get_vla_observation()` returns camera frames (`video.<name>_view`) plus EE position, quaternion, and gripper position, used for VLA policy inputs.

### Action pipeline (`send_action`)

1. **EMA smoothing** of arm actions with factor randomly drawn from `[0.025, 0.1]` at init.
2. **`_process_action()`** converts raw RL outputs (normalized `[-1, 1]`) to absolute EE pose targets:
   - Position scaled by 0.05 m and clipped to ±0.02 m delta from current EE.
   - Rotation: x/y channels zeroed; z-channel mapped to yaw in `[-180°, 90°]` joint limit range; roll/pitch clipped to ±0.097 rad delta.
   - Output: `[x, y, z, qw, qx, qy, qz]` after wxyz→xyzw conversion.
3. Publishes an `ActionChunk` message to `/vla/action/ee_pose`.

### VisionServer (`vision_server.py`)

Runs in the `sam3` conda env. Uses a two-thread pipeline (capture → infer) with a maxsize-1 queue (always processes the freshest frame). Publishes JPEG-encoded frames as JSON over a ZMQ PUB socket (port 5555). With `--use_sam3`, runs SAM3 segmentation with text prompts before publishing. `ZMQCamera` on the robot side subscribes to this socket.

### RL model loading pattern

Both `play.py` and `dummy_inference.py` use a `DummyEnv` shim to satisfy `rl_games`' environment registration requirement before calling `Runner` + `BasePlayer.restore()`. This pattern must be preserved whenever loading `rl_games` checkpoints.
