# Installation
- create conda env
```bash
conda create -n fire python=3.10
conda activate fire
git clone --recursive https://github.com/chohh7391/FIRe.git
cd FIRe
pip install -r requirements.txt
```

- create another conda env for vision
```bash
conda create -n sam3 python=3.12
conda activate sam3
pip install torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -e third_party/sam3
pip install -e src/FIRe/vision_server
```

- export hugging face hub token
```bash
export HUGGING_FACE_HUB_TOKEN="YOUR_TOKEN"
```

- install ros2 based controller: (https://github.com/chohh7391/cho_robot_project)

# How to run

- bringup vla controller
```bash
source ~/ros2_ws/install/setup.bash
ros2 launch cho_franka_bringup bringup_gazebo_robot.launch.py vla:=true control_mode:=torque
```

- vision server
```bash
conda activate sam3
python scripts/run_vision_server.py
```

- run model
```bash
python scripts/play.py --task <TASK_NAME> --checkpoint <CHECKPOINT_PATH>
```

- plot data
python scripts/plot_data.py --task <TASK_NAME> --sim <ISAACLAB_DATA> --real <COLLECTED_DATA> --save_path <FIG_SAVE_PATH>

- record data
```bash
python scripts/record.py --task <TASK_NAME> --checkpoint <CHECKPOINT_PATH> --vla <VLA_NAME> --lerobot_root <PATH_TO_SAVE> --lerobot_task "<TASK_DESCRIPTION>"
```

- record GR00T data and encode all deferred videos after the episode
```bash
python scripts/record.py --task <TASK_NAME> --checkpoint <CHECKPOINT_PATH> --vla gr00t --lerobot_root <PATH_TO_SAVE> --lerobot_task "<TASK_DESCRIPTION>" --last_episode
```

Use `--resume` to append the next episode to the same dataset root. When `--last_episode` is set, `record.py` first records one episode without video encoding, then encodes every missing video in the dataset root after the robot disconnects. Existing `.mp4` files are skipped.

- plot recorded LeRobot dataset
```bash
python scripts/plot_lerobot_dataset.py --root <LEROBOT_DATASET_ROOT> --task <TASK_NAME> --save_path <FIG_SAVE_PATH>
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
conda activate sam3
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
--checkpoint /home/home/FIRe/checkpoints/Forge/VLA_RL-BL-forge-peg_insert/nn/Forge.pth
```

- record (local checkpoint)
```bash
python scripts/record.py \
--task forge-peg_insert \
--checkpoint /home/home/FIRe/checkpoints/Forge/VLA_RL-BL-forge-peg_insert/nn/Forge.pth \
--vla gr00t \
--lerobot_root /home/home/FIRe/test_data \
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

python scripts/plot_lerobot_dataset.py \
--root /home/home/datasets/FIRe/gr00t/peg_insert \
--task forge-peg_insert \
--save_path /home/home/FIRe/outputs/plots/gr00t_peg_insert_latest.png
```

- plot all recorded LeRobot episodes
```bash
conda activate fire
cd /home/home/FIRe

python scripts/plot_lerobot_dataset.py \
--root /home/home/datasets/FIRe/gr00t/peg_insert \
--task forge-peg_insert \
--episode -2 \
--save_path /home/home/FIRe/outputs/plots/gr00t_peg_insert_all.png
```

- plot one LeRobot episode file directly
```bash
conda activate fire
cd /home/home/FIRe

python scripts/plot_lerobot_dataset.py \
--data /home/home/datasets/FIRe/gr00t/peg_insert/data/chunk-000/episode_000000.parquet \
--task forge-peg_insert \
--save_path /home/home/FIRe/outputs/plots/gr00t_peg_insert_episode_000000.png
```

- upload dataset to hugging face
```bash
python scripts/push_to_hub.py \
--root test_data \
--repo_id chohh7391/test_datasets
```

# Residual SAC real robot training

This workflow trains and plays a LeRobot SAC residual policy on top of the FIRe/VLA action path.
It does not replace the RL-Games `scripts/play.py` workflow above.

The residual policy uses the task observation/action definitions in:

```bash
src/FIRe/lerobot_robot_fr3/src/lerobot_robot_fr3/tasks
```

The final action sent to the robot is:

```text
final_action = vla_action + residual_action
```

For test runs, use `--test` to replace the VLA action with zero action.

## Install FIRe keyboard teleoperator

If the keyboard teleoperator package is not already installed in the `fire` environment:

```bash
conda activate fire
cd /home/home/FIRe
pip install -e src/FIRe/lerobot_teleoperators/lerobot_teleoperator_keyboard
```

## Bring up the robot/controller

```bash
conda activate fire
source ~/ros2_ws/install/setup.bash
ros2 launch cho_franka_bringup bringup_gazebo_robot.launch.py vla:=true control_mode:=torque
```

## Start the residual learner

Run this in one terminal:

```bash
conda activate fire
cd /home/home/FIRe

python scripts/train_residual_learner.py \
--task forge-peg_insert \
--fps 15 \
--output_dir outputs/fire_residual \
--device cuda \
--storage_device cpu \
--batch_size 256 \
--online_steps 100000 \
--online_buffer_capacity 100000 \
--online_step_before_learning 100
```

## Start the residual actor without VLA server

Run this in another sourced terminal. `--test` uses zero VLA action. `--teleop_device keyboard`
enables simple keyboard intervention.

```bash
conda activate fire
source ~/ros2_ws/install/setup.bash
cd /home/home/FIRe

python scripts/train_residual_actor.py \
--task forge-peg_insert \
--fps 15 \
--output_dir outputs/fire_residual \
--device cuda \
--storage_device cpu \
--learner_host 127.0.0.1 \
--learner_port 50051 \
--online_steps 10000 \
--use_sim_time \
--test \
--residual_scale 0.05 \
--teleop_device keyboard
```

Keyboard teleop currently maps:

```text
w: +x residual action
s: -x residual action
a: +y residual action
d: -y residual action
```

## Start the residual actor with GR00T VLA

Start the GR00T server first, then run:

```bash
conda activate fire
source ~/ros2_ws/install/setup.bash
cd /home/home/FIRe

python scripts/train_residual_actor.py \
--task forge-peg_insert \
--fps 15 \
--output_dir outputs/fire_residual \
--device cuda \
--storage_device cpu \
--learner_host 127.0.0.1 \
--learner_port 50051 \
--online_steps 10000 \
--use_sim_time \
--use_cameras \
--use_ft_sensor \
--vla gr00t \
--host localhost \
--port 5555 \
--residual_scale 0.05 \
--teleop_device keyboard
```

## Play a trained residual policy

`--policy` can point to the training output directory, a checkpoint directory, or a
`pretrained_model` directory.

Test mode without VLA server:

```bash
conda activate fire
source ~/ros2_ws/install/setup.bash
cd /home/home/FIRe

python scripts/play_real.py \
--task forge-peg_insert \
--policy outputs/fire_residual \
--device cuda \
--control_hz 15 \
--episode_length 300 \
--use_sim_time \
--test \
--residual_scale 0.05 \
--teleop_device keyboard
```

With GR00T VLA:

```bash
conda activate fire
source ~/ros2_ws/install/setup.bash
cd /home/home/FIRe

python scripts/play_real.py \
--task forge-peg_insert \
--policy outputs/fire_residual \
--device cuda \
--control_hz 15 \
--episode_length 300 \
--use_sim_time \
--use_cameras \
--use_ft_sensor \
--vla gr00t \
--host localhost \
--port 5555 \
--residual_scale 0.05 \
--teleop_device keyboard
```

Note: if task rewards are still the default `0.0` and `done/truncated` are always false,
this workflow only verifies the actor-learner pipeline. Meaningful residual learning requires
task-specific reward and termination logic.
