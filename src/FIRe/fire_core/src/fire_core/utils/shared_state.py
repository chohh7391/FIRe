from __future__ import annotations

import threading
from typing import Optional, Tuple

import numpy as np


class SharedState:
    def __init__(self) -> None:
        self._obs_lock = threading.Lock()
        self._latest_obs_dict: Optional[dict] = None

        self._action_lock = threading.Lock()
        self._latest_rl_action: Optional[np.ndarray] = None
        self._latest_vla_chunk: Optional[np.ndarray] = None

        # VLA loop가 blocking wait하는 Event
        self.vla_request_event = threading.Event()
        self.is_running = True

    # ── Obs ───────────────────────────────────────────────────────────────────

    def update_obs(self, obs_dict: dict) -> None:
        with self._obs_lock:
            self._latest_obs_dict = obs_dict

    def get_obs(self) -> Optional[dict]:
        with self._obs_lock:
            return self._latest_obs_dict

    # ── Actions ───────────────────────────────────────────────────────────────

    def update_rl_action(self, action: np.ndarray) -> None:
        with self._action_lock:
            self._latest_rl_action = action

    def update_vla_chunk(self, chunk: np.ndarray) -> None:
        with self._action_lock:
            self._latest_vla_chunk = chunk

    def get_actions(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """(rl_action, vla_chunk) 를 atomic하게 반환."""
        with self._action_lock:
            return self._latest_rl_action, self._latest_vla_chunk

    def has_first_rl_action(self) -> bool:
        with self._action_lock:
            return self._latest_rl_action is not None