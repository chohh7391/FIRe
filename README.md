# Installation
- create conda env
```bash
conda create -n fire python=3.10
conda activate fire
```

- install dependencies
```bash
pip install -r requirements.txt
```

# How to run

- bringup vla_controller
```bash
source ~/ros2_ws/install/setup.bash
ros2 launch cho_franka_bringup bringup_mujoco_robot.launch.py vla:=true control_mode:=torque
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