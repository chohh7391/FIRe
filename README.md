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
python scripts/play.py --task <TASK_NAME> --checkpoint <CHECKPOINT_PATH> --cfg <CONFIG_PATH>
```

- plot data
python scripts/plot_data.py --task <TASK_NAME> --sim <ISAACLAB_DATA> --real <COLLECTED_DATA> --save_path <FIG_SAVE_PATH>

- recorde data
```bash
python scripts/record.py --task <TASK_NAME> --checkpoint <CHECKPOINT_PATH> --vla <VLA_NAME> --lerobot_root <PATH_TO_SAVE> --lerobot_task "<TASK_DESCRIPTION>"
```


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
# --resume
```

- record (huggingface checkpoint)
```bash
python scripts/record.py \
--task forge-peg_insert \
--hf_checkpoint bhe1004/VLA_RL-BL-forge-peg_insert/nn/Forge.pth \
--vla gr00t \
--lerobot_root /home/home/datasets/FIRe/gr00t/peg_insert \
--lerobot_task "Insert peg into the socket" \
# --resume
```

- upload dataset to hugging face
```bash
python scripts/push_to_hub.py \
--root test_data \
--repo_id chohh7391/test_datasets
```