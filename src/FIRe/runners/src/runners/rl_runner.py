"""runners/rl_runner.py

RLRunner
--------
subprocess에서 inference를 돌리되, infer()는 non-blocking.

구조:
  - obs를 obs_shm에 쓰고 obs_flag를 세우면 subprocess가 알아서 action 계산.
  - 별도 _action_poll_thread가 action_flag를 감시해서
    새 action이 오면 SharedState.update_rl_action()으로 넣어줌.
  - rl_inference_loop는 obs가 생길 때마다 trigger만 하고 blocking하지 않음.
"""

from __future__ import annotations

import ctypes
import time
from multiprocessing import RawValue
from typing import Optional

import numpy as np
import torch
import torch.multiprocessing as mp


# ──────────────────────────────────────────────────────────────────────────────
# Subprocess
# ──────────────────────────────────────────────────────────────────────────────

def _build_dummy_env(obs_dim: int, action_dim: int, device: torch.device, clip_actions: float):
    class _DummyEnv:
        observation_space = type("S", (), {"shape": (obs_dim,)})()
        action_space = type("S", (), {
            "shape": (action_dim,),
            "high": np.full(action_dim, clip_actions, dtype=np.float32),
            "low":  np.full(action_dim, -clip_actions, dtype=np.float32),
        })()
        num_envs = 1

        def reset(self):
            return torch.zeros(1, obs_dim, device=device)

        def step(self, _):
            return (torch.zeros(1, obs_dim, device=device),
                    torch.zeros(1), torch.zeros(1, dtype=torch.bool), {})
    return _DummyEnv()


def _run_inference_process(
    obs_shm: torch.Tensor,
    action_shm: torch.Tensor,
    obs_flag: RawValue,    # 0=idle  1=obs_ready
    action_flag: RawValue, # 0=idle  1=action_ready  2=warmup_done
    stop_event: mp.Event,
    *,
    checkpoint: str,
    cfg_path: str,
    obs_dim: int,
    action_dim: int,
    device_str: str,
) -> None:
    import yaml
    from rl_games.common import env_configurations, vecenv
    from rl_games.torch_runner import Runner

    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    cfg["params"].update({"load_checkpoint": True, "load_path": checkpoint})
    cfg["params"]["config"].update({
        "device": str(device), "device_name": str(device), "num_actors": 1,
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

    print("[RL-PROC] Ready.")
    action_flag.value = 2  # warmup done

    while not stop_event.is_set():
        if obs_flag.value != 1:
            continue
        obs_flag.value = 0

        with torch.inference_mode():
            obs_t = agent.obs_to_torch(obs_shm.to(device)).to(model_device)
            act = agent.get_action(obs_t, is_deterministic=True)
            print(f"act: {act.shape}")
            if torch.cuda.is_available():
                torch.cuda.synchronize()

        action_shm.copy_(act.cpu())
        action_flag.value = 1

    print("[RL-PROC] Stopped.")


# ──────────────────────────────────────────────────────────────────────────────
# Public class
# ──────────────────────────────────────────────────────────────────────────────

class RLRunner:
    """RL inference subprocess를 관리.

    rl_inference_loop에서 trigger(obs_np)를 호출하면 obs_flag를 세우고
    즉시 반환(non-blocking). 별도 _action_poll_thread가 새 action을
    감지하면 shared_state.update_rl_action()으로 전달.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        checkpoint: str,
        cfg_path: str,
        device: str = "cuda:0",
    ) -> None:
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self.obs_shm    = torch.zeros(1, obs_dim).share_memory_()
        self.action_shm = torch.zeros(1, action_dim).share_memory_()
        self.obs_flag    = RawValue(ctypes.c_int32, 0)
        self.action_flag = RawValue(ctypes.c_int32, 0)

        self._stop_event = mp.Event()
        self._proc = mp.Process(
            target=_run_inference_process,
            args=(self.obs_shm, self.action_shm,
                  self.obs_flag, self.action_flag, self._stop_event),
            kwargs=dict(checkpoint=checkpoint, cfg_path=cfg_path,
                        obs_dim=obs_dim, action_dim=action_dim, device_str=device),
            daemon=True,
        )
        self._poll_thread: Optional[object] = None
        self._poll_stop = False

    def start(self) -> None:
        self._proc.start()
        print("[RLRunner] Waiting for warm-up ...")
        while self.action_flag.value != 2:
            time.sleep(0.05)
        self.action_flag.value = 0
        print("[RLRunner] Ready.")

    def trigger(self, obs_np: np.ndarray) -> None:
        """obs를 shm에 쓰고 inference trigger — non-blocking."""
        self.obs_shm[0].copy_(torch.from_numpy(obs_np))
        self.obs_flag.value = 1

    def start_action_poll(self, shared_state) -> None:
        """subprocess의 action_flag를 감시하고 새 action을 SharedState에 넣는 스레드 시작."""
        import threading

        self._poll_stop = False

        def _poll():
            action_dim = self.action_dim
            while not self._poll_stop:
                if self.action_flag.value == 1:
                    self.action_flag.value = 0
                    action = self.action_shm.numpy().flatten()[:action_dim].copy()
                    shared_state.update_rl_action(action)
                else:
                    time.sleep(0.001) # 1ms 대기 (필수!)

        self._poll_thread = threading.Thread(target=_poll, daemon=True)
        self._poll_thread.start()

    def stop(self) -> None:
        self._poll_stop = True
        self._stop_event.set()
        self._proc.join(timeout=3)