from __future__ import annotations

import time
from dataclasses import dataclass
from multiprocessing import RawValue

import numpy as np
import torch
import torch.multiprocessing as mp


# ──────────────────────────────────────────────────────────────────────────────
# Subprocess helpers
# ──────────────────────────────────────────────────────────────────────────────

def _build_dummy_env(obs_dim: int, action_dim: int, device: torch.device, clip_actions: float):
    """rl-games가 요구하는 최소 env 인터페이스."""

    class _DummyEnv:
        observation_space = type("S", (), {"shape": (obs_dim,)})()
        action_space = type("S", (), {
            "shape": (action_dim,),
            "high":  np.full(action_dim, clip_actions, dtype=np.float32),
            "low":   np.full(action_dim, -clip_actions, dtype=np.float32),
        })()
        num_envs = 1

        def reset(self):
            return torch.zeros(1, obs_dim, device=device)

        def step(self, _):
            return (torch.zeros(1, obs_dim, device=device),
                    torch.zeros(1), torch.zeros(1, dtype=torch.bool), {})

    return _DummyEnv()


# ──────────────────────────────────────────────────────────────────────────────
# Inference process entry point
# ──────────────────────────────────────────────────────────────────────────────

def run_inference_process(
    obs_shm: torch.Tensor,
    action_shm: torch.Tensor,
    obs_flag: RawValue,
    action_flag: RawValue,
    stop_event: mp.Event,
    *,
    checkpoint: str,
    cfg_path: str,
    obs_dim: int,
    action_dim: int,
    device_str: str,
) -> None:
    """Inference-only subprocess: ROS 없음, GIL 경합 없음.

    obs_flag 대기는 pure spin-wait(continue). time.sleep() 사용 시
    Linux 타이머 해상도(1-4ms)만큼 불필요한 대기가 붙는다.
    """
    import yaml
    from rl_games.common import env_configurations, vecenv
    from rl_games.torch_runner import Runner

    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    cfg["params"].update({"load_checkpoint": True, "load_path": checkpoint})
    cfg["params"]["config"].update({
        "device": str(device),
        "device_name": str(device),
        "num_actors": 1,
    })

    clip_actions = float(cfg["params"]["env"].get("clip_actions", 1.0))
    dummy_env = _build_dummy_env(obs_dim, action_dim, device, clip_actions)
    vecenv.register("IsaacRlgWrapper", lambda *_, **__: dummy_env)
    env_configurations.register("rlgpu", {
        "vecenv_type": "IsaacRlgWrapper",
        "env_creator": lambda **__: dummy_env,
    })

    runner = Runner()
    runner.load(cfg)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("highest")

    agent = runner.create_player()
    agent.restore(checkpoint)
    agent.reset()
    _ = agent.get_batch_size(torch.zeros(1, obs_dim, device=device), 1)
    if agent.is_rnn:
        agent.init_rnn()

    model_device = next(agent.model.parameters()).device

    print("[INFER] Ready.")
    action_flag.value = 2  # warm-up done 시그널

    # ── Inference loop (spin-wait) ────────────────────────────────────────────
    while not stop_event.is_set():
        if obs_flag.value != 1:
            continue  # ← pure spin: sleep 금지
        obs_flag.value = 0

        with torch.inference_mode():
            obs_t = agent.obs_to_torch(obs_shm.to(device)).to(model_device)
            act = agent.get_action(obs_t, is_deterministic=True)
            if torch.cuda.is_available():
                torch.cuda.synchronize()

        action_shm.copy_(act.cpu())
        action_flag.value = 1

    print("[INFER] Stopped.")


# ──────────────────────────────────────────────────────────────────────────────
# Handle
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class InferenceProcessHandle:
    proc: mp.Process
    stop_event: mp.Event

    def wait_ready(self, action_flag: RawValue) -> None:
        print("[INFO] Waiting for inference warm-up ...")
        while action_flag.value != 2:
            time.sleep(0.05)
        action_flag.value = 0
        print("[INFO] Inference process ready.")

    def shutdown(self) -> None:
        self.stop_event.set()
        self.proc.join(timeout=3)


def start_inference_process(
    obs_shm: torch.Tensor,
    action_shm: torch.Tensor,
    obs_flag: RawValue,
    action_flag: RawValue,
    *,
    checkpoint: str,
    cfg_path: str,
    obs_dim: int,
    action_dim: int,
    device: str,
) -> InferenceProcessHandle:
    stop_event = mp.Event()
    proc = mp.Process(
        target=run_inference_process,
        args=(obs_shm, action_shm, obs_flag, action_flag, stop_event),
        kwargs=dict(
            checkpoint=checkpoint,
            cfg_path=cfg_path,
            obs_dim=obs_dim,
            action_dim=action_dim,
            device_str=device,
        ),
        daemon=True,
    )
    proc.start()
    return InferenceProcessHandle(proc=proc, stop_event=stop_event)