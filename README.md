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

- test robot
```bash
source ~/ros2_ws/install/setup.bash
conda activate fire
python test/test_lerobot_robot_fr3.py
```

- run model (test)
```bash
source ~/ros2_ws/install/setup.bash
python scripts/play.py \
--checkpoint checkpoints/Factory/test/nn/Factory.pth \
--cfg scripts/configs/rl_games_ppo_cfg.yaml \
--obs_dim 19 \
--action_dim 6 \
--device cuda:0
```

- for specific task
```bash
# bringup robot
source ~/ros2_ws/install/setup.bash
ros2 launch cho_franka_bringup bringup_gazebo_robot.launch.py vla:=true control_mode:=torque

# run vision server
conda activate sam3
# change target_object for your need
python scripts/run_vision_server.py --use_sam3 --target_object wristwatch

# run task manager for specific task
source ~/ros2_ws/install/setup.bash
ros2 launch cho_task_manager run_task_manager.launch.py task:=forge

# # run model when VLACompletionWaiterBehavior is running
# source ~/ros2_ws/install/setup.bash
# python scripts/play.py \
# --checkpoint checkpoints/Factory/test/nn/Factory.pth \
# --cfg scripts/configs/rl_games_ppo_cfg.yaml \
# --obs_dim 19 \
# --action_dim 6 \
# --device cuda:0

python scripts/play.py --csv_path /home/home/FIRe/scripts/configs/traj_save.csv --hz 15.0

# success feedback using gui
source ~/ros2_ws/install/setup.bash
python3 ~/ros2_ws/src/cho_robot_project/cho_task_manager/python/vla_success_gui.py
```