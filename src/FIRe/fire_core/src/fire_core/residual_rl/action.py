from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from .features import pad_action


def build_vla_action_chunk(raw_actions: Dict[str, np.ndarray], action_dim: int) -> np.ndarray:
    pos = np.asarray(raw_actions["action.eef_position_delta"], dtype=np.float32)
    rot = np.asarray(raw_actions["action.eef_rotation_delta"], dtype=np.float32)
    if pos.ndim == 3:
        pos = pos.squeeze(0)
    if rot.ndim == 3:
        rot = rot.squeeze(0)
    chunk = np.concatenate([pos, rot], axis=-1)
    if chunk.ndim == 1:
        chunk = chunk[None, :]
    padded = np.zeros((chunk.shape[0], action_dim), dtype=np.float32)
    width = min(chunk.shape[1], action_dim)
    padded[:, :width] = chunk[:, :width]
    return padded


def combine_vla_and_residual(
    *,
    vla_action: Optional[np.ndarray],
    residual_action: np.ndarray,
    action_dim: int,
) -> np.ndarray:
    residual = pad_action(residual_action, action_dim)
    if vla_action is None:
        return residual
    return pad_action(vla_action, action_dim) + residual


def make_robot_action(
    arm_action: np.ndarray,
    gripper_action: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    action: Dict[str, np.ndarray] = {"arm_actions": np.asarray(arm_action, dtype=np.float32)}
    if gripper_action is not None:
        action["gripper_actions"] = np.asarray(gripper_action, dtype=np.float32).reshape(-1)
    return action
