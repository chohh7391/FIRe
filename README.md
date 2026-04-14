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

# How to run

- bringup vla_controller
```bash
source ~/ros2_ws/install/setup.bash
ros2 launch cho_franka_bringup bringup_mujoco_robot.launch.py vla:=true control_mode:=torque
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
