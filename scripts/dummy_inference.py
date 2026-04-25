import argparse
import time
import torch
import yaml
import numpy as np

from rl_games.common import env_configurations, vecenv
from rl_games.torch_runner import Runner
from rl_games.common.player import BasePlayer

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint file")
    parser.add_argument("--cfg",        type=str, required=True, help="Path to config yaml")
    parser.add_argument("--obs_dim",    type=int, default=24)
    parser.add_argument("--action_dim", type=int, default=7)
    parser.add_argument("--device",     type=str, default="cuda:0")
    parser.add_argument("--num_steps",  type=int, default=1000, help="Number of steps to benchmark")
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # =================================================================
    # 1. RL 모델 로드 및 설정
    # =================================================================
    with open(args.cfg, "r") as f:
        agent_cfg = yaml.safe_load(f)

    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = args.checkpoint
    agent_cfg["params"]["config"]["device"] = str(device)
    agent_cfg["params"]["config"]["num_actors"] = 1

    # rl_games 모델 초기화를 위한 Dummy Environment
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
    if agent.is_rnn:
        agent.init_rnn()

    print("[INFO] RL Model Loaded Successfully.")
    
    # 모델 파라미터가 올라가 있는 실제 디바이스 확인
    model_actual_device = next(agent.model.parameters()).device

    # =================================================================
    # 2. Warm-up (초기 GPU 커널 로딩 지연 방지)
    # =================================================================
    print("[INFO] Warming up model...")
    dummy_input = torch.ones((1, args.obs_dim), device=device)
    obs_t = agent.obs_to_torch(dummy_input).to(model_actual_device)

    with torch.inference_mode():
        for _ in range(50):
            _ = agent.get_action(obs_t, is_deterministic=True)
            
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # =================================================================
    # 3. Inference 성능 측정 (Benchmark)
    # =================================================================
    print(f"[INFO] Starting Benchmark for {args.num_steps} steps...\n")
    infer_times = []

    for i in range(args.num_steps):
        t0 = time.perf_counter()
        
        with torch.inference_mode():
            # Dummy tensor 생성 및 변환 시간을 포함하여 실제 루프와 유사하게 세팅
            obs_tensor = torch.zeros((1, args.obs_dim), device=device)
            obs_t = agent.obs_to_torch(obs_tensor).to(model_actual_device)
            
            rl_action = agent.get_action(obs_t, is_deterministic=True)
            
            # GPU 비동기 실행이 끝날 때까지 대기해야 정확한 시간 측정이 가능함
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                
        t1 = time.perf_counter()
        infer_times.append((t1 - t0) * 1000) # 초(s) 단위를 밀리초(ms)로 변환

        if (i + 1) % max(1, (args.num_steps // 10)) == 0:
            print(f"  Progress: {i+1}/{args.num_steps} steps done...")

    # =================================================================
    # 4. 결과 요약
    # =================================================================
    infer_times = np.array(infer_times)
    
    print("\n" + "="*50)
    print("[INFERENCE TIMING SUMMARY] (ms)")
    print(f"Total Steps : {args.num_steps}")
    print("-" * 50)
    print(f"Mean Time   : {infer_times.mean():6.3f} ms")
    print(f"Std Dev     : {infer_times.std():6.3f} ms")
    print(f"Min Time    : {infer_times.min():6.3f} ms")
    print(f"Max Time    : {infer_times.max():6.3f} ms")
    print(f"99th Pct    : {np.percentile(infer_times, 99):6.3f} ms")
    print("="*50)

if __name__ == "__main__":
    main()