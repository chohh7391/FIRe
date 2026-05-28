"""runners/vla_runner.py

VLARunner
---------
VLA client를 래핑. chunk formatting (raw dict → np.ndarray) 책임을 여기서 담당.
control_loop / vla_inference_loop는 항상 np.ndarray (chunk_size, action_dim)을 받음.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


class VLARunner:
    def __init__(
        self,
        vla_type: str,
        host: str,
        port: int,
        action_dim: int,
    ) -> None:
        self.vla_type = vla_type
        self.host = "127.0.0.1" if host == "localhost" else host
        self.port = port
        self.action_dim = action_dim
        self.client: Optional[object] = None

    def start(self) -> None:
        if self.vla_type == "gr00t":
            from runners.utils.vla_clients import AsyncGr00tInferenceClient
            self.client = AsyncGr00tInferenceClient(self.host, self.port)
        elif self.vla_type == "pi05":
            from runners.utils.vla_clients import AsyncPi05InferenceClient
            self.client = AsyncPi05InferenceClient(self.host, self.port)
        else:
            raise ValueError(f"Unknown VLA type: {self.vla_type}")
        print(f"[VLARunner] Connected to {self.vla_type} at {self.host}:{self.port}")

    def get_action_sync(self, vla_obs: dict) -> np.ndarray:
        """첫 chunk 동기 수신. np.ndarray (chunk_size, action_dim) 반환."""
        raw = self.client.get_action_sync(vla_obs)
        return self._format_chunk(raw)

    def request_action(self, vla_obs: dict) -> None:
        """비동기 추론 요청 — non-blocking."""
        self.client.request_action(vla_obs)

    def get_result(self) -> np.ndarray:
        """비동기 요청 결과 회수. np.ndarray (chunk_size, action_dim) 반환."""
        raw = self.client.get_result()
        return self._format_chunk(raw)

    def stop(self) -> None:
        pass

    # ── Internal ──────────────────────────────────────────────────────────────

    def _format_chunk(self, raw: dict) -> np.ndarray:
        """raw dict → (chunk_size, action_dim) float32 array.

        eef_position_delta (chunk, 3) + eef_rotation_delta (chunk, 3) 을 concat.
        action_dim이 6보다 크면 나머지를 0으로 패딩 (FORGE success-bit 등).
        """
        pos = np.asarray(raw["action.eef_position_delta"], dtype=np.float32)
        rot = np.asarray(raw["action.eef_rotation_delta"], dtype=np.float32)

        # (1, chunk, 3) → (chunk, 3) squeez
        if pos.ndim == 3:
            pos = pos.squeeze(0)
        if rot.ndim == 3:
            rot = rot.squeeze(0)

        chunk = np.concatenate([pos, rot], axis=-1)  # (chunk_size, 6)

        if chunk.shape[-1] < self.action_dim:
            pad = np.zeros((chunk.shape[0], self.action_dim - chunk.shape[-1]), dtype=np.float32)
            chunk = np.concatenate([chunk, pad], axis=-1)

        return chunk  # (chunk_size, action_dim)