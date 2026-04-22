import argparse
import time
import torch
import yaml
import numpy as np
import cv2

from rl_games.common import env_configurations, vecenv
from rl_games.torch_runner import Runner
from rl_games.common.player import BasePlayer

from lerobot_robot_fr3.config_fr3 import FR3RobotConfig
from lerobot_robot_fr3.fr3 import FR3Robot


OBS_KEYS_AND_DIM = [
    # ("fingertip_pos",           3),
    ("fingertip_pos_rel_fixed", 3),
    ("fingertip_quat",          4),
    ("ee_linvel",               3),
    ("ee_angvel",               3),
    ("prev_actions",            6),
]

def build_obs_tensor(obs_dict: dict, obs_dim: int, device: torch.device) -> torch.Tensor:
    """
    robot.get_observation() dict → policy input tensor (1, obs_dim)
    처음 실행 시 key 불일치 에러 메시지로 수정 방향 확인 가능.
    """
    parts = []
    for key, expected_dim in OBS_KEYS_AND_DIM:
        if key not in obs_dict:
            raise KeyError(
                f"[OBS] '{key}' not in obs_dict.\n"
                f"  Available: {list(obs_dict.keys())}\n"
                f"  → OBS_KEYS_AND_DIM 수정 필요"
            )
        val = obs_dict[key]
        if isinstance(val, np.ndarray):
            val = torch.from_numpy(val).float()
        val = val.flatten()

        if val.shape[0] != expected_dim:
            raise ValueError(
                f"[OBS] '{key}': expected {expected_dim}D, got {val.shape[0]}D"
            )
        parts.append(val)

    obs_vec = torch.cat(parts)

    if obs_vec.shape[0] != obs_dim:
        raise ValueError(
            f"[OBS] Total dim mismatch: built {obs_vec.shape[0]}, expected {obs_dim}\n"
            f"  → OBS_KEYS_AND_DIM 항목 추가/수정 필요"
        )

    return obs_vec.unsqueeze(0).to(device)   # (1, obs_dim)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--cfg",        type=str, required=True)
    parser.add_argument("--obs_dim",    type=int, default=19,
                        help="env.yaml: observation_space=21")
    parser.add_argument("--action_dim", type=int, default=6,
                        help="env.yaml: action_space=6 (gripper는 policy 외부 제어)")
    parser.add_argument("--device",     type=str, default="cuda:0")
    parser.add_argument("--control_hz", type=float, default=15.0,
                        help="dt=0.00833 x decimation=8 ≈ 15 Hz")
    parser.add_argument("--visualize",  action="store_true", default=False)
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

    agent_cfg["params"]["load_checkpoint"]      = True
    agent_cfg["params"]["load_path"]            = args.checkpoint
    agent_cfg["params"]["config"]["device"]     = str(device)
    agent_cfg["params"]["config"]["num_actors"] = 1

    class DummyEnv:
        def __init__(self):
            self.observation_space = type("S", (), {"shape": (args.obs_dim,)})()
            self.action_space      = type("S", (), {
                "shape": (args.action_dim,),
                "high":  np.ones(args.action_dim, dtype=np.float32),
                "low":  -np.ones(args.action_dim, dtype=np.float32),
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
        print("[INFO] LSTM states initialized.")

    print("[INFO] RL Model Loaded Successfully.")

    # =================================================================
    # 2. 로봇 연결
    # =================================================================
    print("[INFO] Connecting to Robot...")
    config = FR3RobotConfig(is_relative=False, rotation_type="quaternion", arm_action_dim=6)
    robot  = FR3Robot(config)

    try:
        robot.connect()
    except Exception as e:
        print(f"[ERROR] Connection failed: {e}")
        return

    if not robot.is_connected:
        print("[ERROR] Failed to connect to robot.")
        return

    # 실제 obs key 확인용 (최초 1회)
    print("[DEBUG] Checking robot.get_observation() keys...")
    _obs_dict = robot.get_observation()
    for k, v in _obs_dict.items():
        print(f"  {k}: shape={np.array(v).shape}")

    # =================================================================
    # 3. Control Loop
    # =================================================================
    dt = 1.0 / args.control_hz
    print(f"\n[INFO] Starting control loop at {args.control_hz} Hz. Ctrl+C or 'q' to stop.")

    try:
        while True:
            t_start = time.time()

            # --- A. Observation ---
            obs_dict     = robot.get_observation()
            vla_obs_dict = robot.get_vla_observation() if args.visualize else {}

            # --- B. Obs → tensor ---
            obs_tensor = build_obs_tensor(obs_dict, args.obs_dim, device)

            # --- C. RL Inference ---
            with torch.inference_mode():
                obs_t     = agent.obs_to_torch(obs_tensor)
                rl_action = agent.get_action(obs_t, is_deterministic=True)
                # clip_actions=1.0은 agent 내부에서 처리됨

            arm_action_np = rl_action.cpu().numpy().flatten()[:6]  # (6,) delta EE pose

            # --- D. Action 전송 ---
            # gripper: task logic에 따라 조정 (현재는 열린 상태 고정)
            action_dict = {
                "arm_actions":     arm_action_np,
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
                        if frame.ndim == 4:
                            frame = frame[0]
                        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        label = key
                    else:
                        bgr   = np.zeros((config.cameras[key].height,
                                          config.cameras[key].width, 3), dtype=np.uint8)
                        label = f"{key} (no frame)"
                    cv2.putText(bgr, label, (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
                    panels.append(bgr)

                cv2.imshow("FR3 Multi-Camera Viewer", np.hstack(panels))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("[INFO] 'q' pressed. Exiting.")
                    break

            # --- F. 주기 유지 ---
            elapsed = time.time() - t_start
            sleep_t = dt - elapsed
            if sleep_t > 0:
                time.sleep(sleep_t)
            elif elapsed > dt * 1.1:
                print(f"[WARN] Loop overrun: {elapsed*1000:.1f} ms > {dt*1000:.1f} ms")

    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user.")
    except Exception as e:
        print(f"\n[ERROR] {e}")
        raise
    finally:
        cv2.destroyAllWindows()
        robot.disconnect()
        print("[INFO] Robot disconnected safely.")


if __name__ == "__main__":
    main()