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

# run model when VLACompletionWaiterBehavior is running
```bash
python scripts/play.py --task forge-peg_insert --checkpoint /home/home/FIRe/checkpoints/Forge/peg_insert/nn/Forge.pth --cfg /home/home/FIRe/src/FIRe/lerobot_robot_fr3/src/lerobot_robot_fr3/tasks/forge/agents/forge-peg_insert.yaml --replay /home/home/FIRe/logs/forge/csv/sim/origin.csv --raw --save_path /home/home/FIRe/logs/forge/csv/real
```
