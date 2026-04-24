import argparse
import time
import torch
import yaml
import numpy as np
import cv2
import pandas as pd

from rl_games.common import env_configurations, vecenv
from rl_games.torch_runner import Runner
from rl_games.common.player import BasePlayer

from lerobot_robot_fr3.config_fr3 import FR3RobotConfig
from lerobot_robot_fr3.fr3 import FR3Robot

OBS_KEYS_AND_DIM = [
    ("fingertip_pos_rel_fixed", 3),
    ("fingertip_quat", 4),
    ("ee_linvel", 3),
    ("ee_angvel", 3),
    ("force_threshold", 1),
    ("ft_force", 3),
    ("prev_actions", 7),
]

def build_obs_tensor(obs_dict: dict, obs_dim: int, device: torch.device) -> torch.Tensor:
    parts = []
    for key, expected_dim in OBS_KEYS_AND_DIM:
        if key not in obs_dict:
            raise KeyError(f"[OBS] '{key}' not in obs_dict. Available: {list(obs_dict.keys())}")
        
        val = obs_dict[key]
        if isinstance(val, np.ndarray):
            val = torch.from_numpy(val).float()
        elif isinstance(val, (list, float, int)):
            val = torch.tensor(np.array(val)).float()
            
        val = val.flatten()
        if val.shape[0] != expected_dim:
            raise ValueError(f"[OBS] '{key}': expected {expected_dim}D, got {val.shape[0]}D")
        parts.append(val)

    obs_vec = torch.cat(parts)
    if obs_vec.shape[0] != obs_dim:
        raise ValueError(f"[OBS] Total dim mismatch: built {obs_vec.shape[0]}, expected {obs_dim}")

    return obs_vec.unsqueeze(0).to(device)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--cfg",        type=str, required=True)
    parser.add_argument("--obs_dim",    type=int, default=24)
    parser.add_argument("--action_dim", type=int, default=7)
    parser.add_argument("--device",     type=str, default="cuda:0")
    parser.add_argument("--control_hz", type=float, default=15.0)
    parser.add_argument("--visualize",  action="store_true", default=False)
    parser.add_argument("--use_sim_time", action="store_true")
    # 리플레이 인자
    parser.add_argument("--replay",     type=str, default=None, help="Path to replay CSV file")
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # =================================================================
    # 1. RL 모델 로드
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
                "high": np.ones(args.action_dim, dtype=np.float32),
                "low": -np.ones(args.action_dim, dtype=np.float32),
            })()
            self.num_envs = 1
        def reset(self): return torch.zeros(1, args.obs_dim, device=device)
        def step(self, actions): return torch.zeros(1, args.obs_dim, device=device), torch.zeros(1), torch.zeros(1, dtype=torch.bool), {}

    dummy_env = DummyEnv()
    vecenv.register("IsaacRlgWrapper", lambda config_name, num_actors, **kwargs: dummy_env)
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kwargs: dummy_env})

    runner = Runner()
    runner.load(agent_cfg)
    agent: BasePlayer = runner.create_player()
    agent.restore(args.checkpoint)
    agent.reset()
    if agent.is_rnn: agent.init_rnn()

    print("[INFO] RL Model Loaded Successfully.")

    # =================================================================
    # 2. 로봇 연결 (Replay 모드여도 실제 로봇에 연결)
    # =================================================================
    print("[INFO] Connecting to Robot...")
    config = FR3RobotConfig(use_sim_time=args.use_sim_time, is_relative=False, rotation_type="quaternion", arm_action_dim=6)
    robot = FR3Robot(config)
    
    try:
        robot.connect()
    except Exception as e:
        print(f"[ERROR] Connection failed: {e}")
        return

    if not robot.is_connected:
        print("[ERROR] Failed to connect to robot.")
        return

    # =================================================================
    # 3. CSV 데이터 로드
    # =================================================================
    replay_data = None
    if args.replay:
        print(f"[INFO] Loading CSV for observations: {args.replay}")
        replay_data = pd.read_csv(args.replay)
        print(f"[INFO] Loaded {len(replay_data)} steps from CSV. Robot will be driven by these observations.")

    # =================================================================
    # 4. Control Loop
    # =================================================================
    dt = 1.0 / args.control_hz
    step_idx = 0
    print(f"\n[INFO] Starting control loop at {args.control_hz} Hz. Ctrl+C or 'q' to stop.")

    try:
        while True:
            t_start = time.time()

            # --- A. Observation ---
            if args.replay:
                # CSV가 끝났으면 종료
                if step_idx >= len(replay_data):
                    print("\n[INFO] Replay CSV finished.")
                    break
                
                row = replay_data.iloc[step_idx]
                obs_dict = {}
                for key, dim in OBS_KEYS_AND_DIM:
                    if dim == 1:
                        obs_dict[key] = np.array([row[f"obs/{key}"]], dtype=np.float32)
                    else:
                        obs_dict[key] = np.array([row[f"obs/{key}_{i}"] for i in range(dim)], dtype=np.float32)
                
                print(f"[REPLAY] Sending step {step_idx}/{len(replay_data)} to robot...", end='\r')
                step_idx += 1
            else:
                obs_dict = robot.get_observation()

            # 시각화를 위한 VLA 데이터 (실제 센서 데이터 유지)
            vla_obs_dict = robot.get_vla_observation() if args.visualize else {}

            # --- B. Obs → tensor ---
            obs_tensor = build_obs_tensor(obs_dict, args.obs_dim, device)

            # --- C. RL Inference ---
            with torch.inference_mode():
                obs_t = agent.obs_to_torch(obs_tensor)
                rl_action = agent.get_action(obs_t, is_deterministic=True)
            
            action_np = rl_action.cpu().numpy().flatten()
            arm_action_np = action_np[:6]
            # FT 센서 높이 보정
            arm_action_np[2] += 0.15

            # --- D. Action 전송 (무조건 전송) ---
            action_dict = {
                "arm_actions": arm_action_np,
                "success_prediction": action_np[6:],
                "gripper_actions": np.array([0.0], dtype=np.float32),
            }
            robot.send_action(action_dict)

            # --- E. 카메라 시각화 ---
            if args.visualize:
                panels = []
                for key in ["wrist", "left", "right"]:
                    obs_key = f"video.{key}_view"
                    if obs_key in vla_obs_dict:
                        frame = vla_obs_dict[obs_key]
                        if frame.ndim == 4: frame = frame[0]
                        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        label = key
                    else:
                        bgr = np.zeros((config.cameras[key].height, config.cameras[key].width, 3), dtype=np.uint8)
                        label = f"{key} (no frame)"
                    cv2.putText(bgr, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
                    panels.append(bgr)

                cv2.imshow("FR3 Multi-Camera Viewer", np.hstack(panels))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("\n[INFO] 'q' pressed. Exiting.")
                    break

            # --- F. 주기 유지 ---
            elapsed = time.time() - t_start
            sleep_t = dt - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)

            print(f"time: {time.time() - t_start}")

    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user.")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        raise
    finally:
        cv2.destroyAllWindows()
        if robot:
            robot.disconnect()
        print("[INFO] Robot disconnected safely.")

if __name__ == "__main__":
    main()