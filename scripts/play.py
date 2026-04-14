import argparse
import time
import torch
import yaml
import numpy as np

# [1] rl_games 라이브러리
from rl_games.common import env_configurations, vecenv
from rl_games.torch_runner import Runner
from rl_games.common.player import BasePlayer

# [2] 우리가 만든 로봇 통신 라이브러리
from lerobot_robot_fr3.config_fr3 import FR3RobotConfig
from lerobot_robot_fr3.fr3 import FR3Robot

def parse_args():
    parser = argparse.ArgumentParser(description="RL-Games Inference to Real Robot (Franka)")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pth)")
    parser.add_argument("--cfg", type=str, required=True, help="Path to agent config yaml")
    parser.add_argument("--obs_dim", type=int, required=True, help="Observation dimension of RL model")
    parser.add_argument("--action_dim", type=int, default=7, help="Action dimension (e.g., 6 arm + 1 gripper)")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to run inference on")
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # =================================================================
    # 1. RL 모델 (rl_games) 초기화 및 로드
    # =================================================================
    with open(args.cfg, "r") as f:
        agent_cfg = yaml.safe_load(f)

    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = args.checkpoint
    agent_cfg["params"]["config"]["device"] = str(device)
    agent_cfg["params"]["config"]["num_actors"] = 1

    class DummyEnv:
        def __init__(self):
            self.observation_space = type("S", (), {"shape": (args.obs_dim,)})()
            self.action_space = type("S", (), {
                "shape": (args.action_dim,),
                "high": torch.ones(args.action_dim).numpy(),
                "low": -torch.ones(args.action_dim).numpy(),
            })()
            self.num_envs = 1
        def reset(self): 
            return torch.zeros(1, args.obs_dim, device=device)
        def step(self, actions): 
            return torch.zeros(1, args.obs_dim, device=device), torch.zeros(1), torch.zeros(1, dtype=torch.bool), {}

    dummy_env = DummyEnv()
    vecenv.register("IsaacRlgWrapper", lambda config_name, num_actors, **kwargs: dummy_env)
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: dummy_env})

    runner = Runner()
    runner.load(agent_cfg)
    agent: BasePlayer = runner.create_player()
    agent.restore(args.checkpoint)
    agent.reset()

    if agent.is_rnn:
        agent.init_rnn()

    print("[INFO] RL Model Loaded Successfully.")

    # =================================================================
    # 2. 로봇 (ROS 2) 초기화
    # =================================================================
    print("[INFO] Connecting to Robot...")
    config = FR3RobotConfig(is_relative=True, rotation_type="axis_angle", arm_action_dim=6)
    robot = FR3Robot(config)
    robot.connect()

    if not robot.is_connected:
        print("[ERROR] Failed to connect to robot.")
        return

    # =================================================================
    # 3. 실시간 추론 및 제어 루프 (Control Loop)
    # =================================================================
    control_rate = 12.0 # Hz
    sleep_time = 1.0 / control_rate

    print("\n🚀 [INFO] Starting Real-time Inference Control! Press Ctrl+C to stop.")
    try:
        while True:
            t_start = time.time()

            # --- A. 로봇으로부터 실제 관측값(Observation) 가져오기 ---
            obs_dict = robot.get_observation()
            print("=========== observation ===========")
            for obs_key, obs_value in obs_dict.items():
                print(f"{obs_key}: {obs_value.shape}")

            vla_obs_dict = robot.get_vla_observation()
            print("=========== vla observation ===========")
            for obs_key, obs_value in vla_obs_dict.items():
                print(f"{obs_key}: {obs_value.shape}")

            # TODO: obs_dict 데이터를 모델 입력 규격에 맞게 매핑
            obs_tensor = torch.randn(1, args.obs_dim, device=device)

            # --- B. RL 모델 추론 (Inference) ---
            with torch.inference_mode():
                obs_t = agent.obs_to_torch(obs_tensor)
                rl_action = agent.get_action(obs_t, is_deterministic=True)
                rl_action_np = rl_action.cpu().numpy().flatten() 

            # --- C. C++ 액션 서버로 전송 (Action Publish) ---
            arm_action = rl_action_np[:6]
            gripper_action = [rl_action_np[6]] if args.action_dim > 6 else [1.0]

            action_dict = {
                "arm_action": np.array([arm_action], dtype=np.float32),
                "gripper_action": np.array(gripper_action, dtype=np.float32)
            }
            robot.send_action(action_dict)

            # --- D. 제어 주기 (Hz) 맞추기 ---
            elapsed = time.time() - t_start
            time.sleep(max(0.0, sleep_time - elapsed))

    except KeyboardInterrupt:
        print("\n[INFO] Stopped by User.")
    finally:
        robot.disconnect()
        print("[INFO] Robot Disconnected Safely.")

if __name__ == "__main__":
    main()