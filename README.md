# Installation
- create conda env
```bash
conda create -n fire python=3.10
conda activate fire
git clone --recurse-submodules https://github.com/chohh7391/FIRe.git
cd FIRe
pip install -r requirements.txt
```

> **Submodule:** the Inverse3 teleoperator lives in a git submodule
> (`src/FIRe/lerobot_teleoperators/lerobot_teleoperator_inverse3`) that `requirements.txt`
> installs editable, so clone with `--recurse-submodules` — or, in an existing clone, run
> `git submodule update --init --recursive` **before** `pip install -r requirements.txt`.
> Its C++ bridge additionally needs the proprietary Haply SDK built separately; see the
> submodule's `SETUP.md` and [docs/inverse3_setup.md](docs/inverse3_setup.md).

The vision (camera) server now runs in the same `fire` env — `requirements.txt` installs
`src/FIRe/vision_server` (with `pyrealsense2`), so no separate env is needed.

- install ros2 based controller: (https://github.com/chohh7391/cho_robot_project)

# How to run

- bringup vla controller
```bash
source ~/ros2_ws/install/setup.bash
ros2 launch cho_franka_bringup bringup_gazebo_robot.launch.py vla:=true control_mode:=torque
```

- vision server
```bash
conda activate fire
python scripts/run_vision_server.py
```

- run model
```bash
python scripts/play.py --task <TASK_NAME> --checkpoint <CHECKPOINT_PATH>
```

## VLA backends (`--vla`)

`play.py` (and `record.py`) can drive the
RL action path with different VLA servers via `--vla`. Supported backends:

| `--vla`   | server                                   | protocol        | default port | chunk size |
|-----------|------------------------------------------|-----------------|--------------|------------|
| `gr00t`   | Isaac-GR00T `inference_service.py`       | ZMQ             | 5555         | 16         |
| `pi05`    | openpi `serve_policy.py`                 | WebSocket       | 8000         | 8          |
| `openvla` | openvla-oft `vla-scripts/deploy_batch.py`| HTTP            | 8778         | 8          |

The chunk size is auto-selected per backend (override with `--vla_chunk_size`). FIRe speaks a
single gr00t-canonical observation/action format internally; each client
(`src/FIRe/fire_core/src/fire_core/vla_clients/`) translates to/from its server's format, so the
backends are interchangeable.

```bash
# pi05 (openpi WebSocket server on port 8000)
python scripts/play.py --task forge-peg_insert --checkpoint <CHECKPOINT_PATH> \
  --vla pi05 --host 163.180.160.225 --port 8000

# openvla (openvla-oft HTTP server on port 8778)
python scripts/play.py --task forge-peg_insert --checkpoint <CHECKPOINT_PATH> \
  --vla openvla --host localhost --port 8778
```

> Note: each VLA model must output actions in the task's expected (normalized) action
> convention. If a model outputs raw metric deltas, motion may look very small after the task's
> `process_action` scaling — that is a model/units mismatch, not a client-side scale.

## Play a VLA model only (no RL checkpoint)

Drive the robot purely from the VLA server's action chunk — no RL policy. Pass `--vla` **without**
`--checkpoint`/`--hf_checkpoint` (e.g. the teleop-collected `pick_place` task). The task's action
convention decides how the chunk is applied:

- Absolute EE-pose tasks (e.g. `pick_place`): the server returns `eef_position` + `eef_quaternion`,
  sent to the robot as an absolute task-space pose (no `process_action` delta scaling).
- Relative-delta tasks: the server returns `eef_position_delta` + `eef_rotation_delta`, turned into
  an absolute pose by the task's `process_action`.

Prerequisites (each in its own sourced terminal): robot/controller bringup, vision server, task
manager for the task, and the GR00T inference server (`--host`/`--port` below point to it).

```bash
conda activate fire
source ~/ros2_ws/install/setup.bash
cd /home/home/FIRe

python scripts/play.py \
--task pick_place \
--vla gr00t \
--host 163.180.160.225 --port 7777 \
--use_cameras \
--episode_length 384
```

`--use_cameras` is required (the VLA server needs the camera views). Swap `--vla`/`--host`/`--port`
for `pi05` or `openvla` as in the table above.

- plot data
python scripts/plot/plot_data.py --task <TASK_NAME> --sim <ISAACLAB_DATA> --real <COLLECTED_DATA> --save_path <FIG_SAVE_PATH>

- record data
```bash
python scripts/record.py --task <TASK_NAME> --checkpoint <CHECKPOINT_PATH> --vla gr00t --lerobot_root <PATH_TO_SAVE> --lerobot_task "<TASK_DESCRIPTION>"
```

> **Note:** `record.py` records **GR00T-format** LeRobot datasets only (`--vla gr00t`). To use the
> collected data with **pi05** or **openvla**, first record the GR00T dataset and then convert it to
> the target format separately — there is no built-in pi05/openvla recorder.

- record GR00T data and encode all deferred videos after the episode
```bash
python scripts/record.py --task <TASK_NAME> --checkpoint <CHECKPOINT_PATH> --vla gr00t --lerobot_root <PATH_TO_SAVE> --lerobot_task "<TASK_DESCRIPTION>" --last_episode
```

Use `--resume` to append the next episode to the same dataset root. When `--last_episode` is set, `record.py` first records one episode without video encoding, then encodes every missing video in the dataset root after the robot disconnects. Existing `.mp4` files are skipped.

Add `--obs_save_path <DIR>` to also dump per-step observation info to a CSV (same format as `play.py --save_path`), independent of the LeRobot dataset:
```bash
python scripts/record.py --task <TASK_NAME> --checkpoint <CHECKPOINT_PATH> --vla gr00t --lerobot_root <PATH_TO_SAVE> --lerobot_task "<TASK_DESCRIPTION>" --obs_save_path <OBS_CSV_DIR>
```

- plot recorded LeRobot dataset
```bash
python scripts/plot/plot_gr00t_dataset.py --root <LEROBOT_DATASET_ROOT> --task <TASK_NAME> --save_path <FIG_SAVE_PATH>
```

- Bota force/torque sensor setup

See the standalone module guide at [src/FIRe/lerobot_ft_sensor/README.md](/home/home/FIRe/src/FIRe/lerobot_ft_sensor/README.md).


# TEMPORARY COMMANDS
- bringup robot
```bash
source ~/ros2_ws/install/setup.bash
ros2 launch cho_franka_bringup bringup_gazebo_robot.launch.py control_mode:=torque vla:=true 
```

- run vision server
```bash
conda activate fire
python scripts/run_vision_server.py
```

- run task manager
```bash
source ~/ros2_ws/install/setup.bash
ros2 launch cho_task_manager run_task_manager.launch.py task:=forge
```

- play
```bash
python scripts/play.py \
--task forge-peg_insert \
--checkpoint /home/home/FIRe/checkpoints/FIRe-BL-Forge-PegInsert/nn/Forge.pth
```

- record (local checkpoint)
```bash
python scripts/record.py \
--task forge-peg_insert \
--checkpoint /home/home/FIRe/checkpoints/FIRe-BL-Forge-PegInsert/nn/Forge.pth \
--vla gr00t \
--lerobot_root /home/home/datasets/FIRe/gr00t/peg_insert \
--lerobot_task "Insert peg into the socket" \
--last_episode
```

Add `--resume` to append another episode to the same dataset root.

- record (huggingface checkpoint)
```bash
python scripts/record.py \
--task forge-peg_insert \
--hf_checkpoint bhe1004/VLA_RL-BL-forge-peg_insert/nn/Forge.pth \
--vla gr00t \
--lerobot_root /home/home/datasets/FIRe/gr00t/peg_insert \
--lerobot_task "Insert peg into the socket" \
--last_episode
```

Add `--resume` to append another episode to the same dataset root.

`--last_episode` can also be used without a value after data has already been collected:

```bash
conda activate fire
cd /home/home/FIRe

python scripts/record.py \
--lerobot_root /home/home/datasets/FIRe/gr00t/peg_insert \
--last_episode
```

This encode-only mode does not connect to the robot. To encode only through a specific inclusive episode index, pass the index explicitly, for example `--last_episode 12`.

- plot latest recorded LeRobot episode
```bash
conda activate fire
cd /home/home/FIRe

python scripts/plot/plot_gr00t_dataset.py \
--root /home/home/datasets/FIRe/gr00t/peg_insert \
--task forge-peg_insert \
--save_path /home/home/FIRe/outputs/plots/gr00t_peg_insert_latest.png
```

- plot all recorded LeRobot episodes
```bash
conda activate fire
cd /home/home/FIRe

python scripts/plot/plot_gr00t_dataset.py \
--root /home/home/datasets/FIRe/gr00t/peg_insert \
--task forge-peg_insert \
--episode -2 \
--save_path /home/home/FIRe/outputs/plots/gr00t_peg_insert_all.png
```

- plot one LeRobot episode file directly
```bash
conda activate fire
cd /home/home/FIRe

python scripts/plot/plot_gr00t_dataset.py \
--data /home/home/datasets/FIRe/gr00t/peg_insert/data/chunk-000/episode_000000.parquet \
--task forge-peg_insert \
--save_path /home/home/FIRe/outputs/plots/gr00t_peg_insert_episode_000000.png
```

# Inverse3 Haptic Teleop

Haption Inverse3 + VerseGrip로 demonstration을 직접 수집한다.
최초 세팅(빌드 / udev 시리얼 포트 설정 / 문제 해결)은 [docs/inverse3_setup.md](docs/inverse3_setup.md) 참고.
상세 아키텍처 및 운용 매뉴얼은 [docs/inverse3_teleoperator.md](docs/inverse3_teleoperator.md) 참고.

## Install

```bash
# C++ bridge server 빌드 (최초 1회)
cd src/FIRe/lerobot_teleoperators/lerobot_teleoperator_inverse3/\
src/lerobot_teleoperator_inverse3/inverse3_bridge
make
cd /home/home/FIRe

# Python 패키지 설치
conda activate fire
pip install -e src/FIRe/lerobot_teleoperators/lerobot_teleoperator_inverse3
```

## Record demonstrations via teleop

```bash
conda activate fire
source ~/ros2_ws/install/setup.bash

python scripts/record.py \
--teleop inverse3 \
--task forge-peg_insert \
--vla gr00t \
--lerobot_root /home/home/datasets/FIRe/teleop/peg_insert \
--lerobot_task "Insert peg into the socket" \
--use_cameras \
--inv3_port /dev/inverse3_left \
--versegrip_port /dev/versegrip_left \
--position_scale 3.0 \
--episode_length 200
```

Add `--resume` to append another episode to the same dataset root.

**동작 방법:**
- VerseGrip의 enable button(기본 bit-0)을 **누른 상태**에서 로봇이 따라온다.
- 버튼을 **처음 누른 순간** Inverse3 현재 위치와 로봇 현재 위치가 자동으로 매칭된다 (점프 없음).
- 버튼을 놓으면 일시 정지. 다시 누르면 현재 위치에서 새로 시작.
- `Ctrl+C`로 녹화 중단 → 성공 여부 입력 후 저장.

## Mock test (기기 없이 동작 확인)

```bash
python src/FIRe/lerobot_teleoperators/lerobot_teleoperator_inverse3/test_teleop_mock.py
```
