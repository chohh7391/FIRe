# processor_franka.py

from dataclasses import dataclass
from typing import Any
import torch
from lerobot.processor import ProcessorStep, ProcessorStepRegistry
from lerobot.processor.core import EnvTransition, TransitionKey
from lerobot.utils.constants import OBS_STATE, ACTION


def quaternion_to_rotation_6d(quat: torch.Tensor) -> torch.Tensor:
    """
    quaternion (B, 4) [x, y, z, w] → rotation_6d (B, 6)
    rotation matrix의 첫 두 column을 사용
    """
    # quat: (B, 4) → rotation matrix (B, 3, 3)
    x, y, z, w = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]

    R = torch.stack([
        1 - 2*(y*y + z*z),  2*(x*y - w*z),  2*(x*z + w*y),
        2*(x*y + w*z),  1 - 2*(x*x + z*z),  2*(y*z - w*x),
        2*(x*z - w*y),  2*(y*z + w*x),  1 - 2*(x*x + y*y),
    ], dim=-1).reshape(-1, 3, 3)  # (B, 3, 3)

    # 첫 두 column → (B, 6)
    return torch.cat([R[:, :, 0], R[:, :, 1]], dim=-1)


@dataclass
@ProcessorStepRegistry.register(name="franka_state_action_transform_v1")
class FrankaStateActionTransformStep(ProcessorStep):
    """
    Gr00tPackInputsStep 이전에 실행.
    - state: [eef_pos(3), eef_quat(4), gripper_qpos(1)]
              → [eef_pos(3), rotation_6d(6), gripper_qpos(1)] = 10D
    - action: [eef_pos_delta(3), eef_rot_delta(3), gripper_close(1)]
               → eef_rot_delta는 그대로 (axis_angle),
                 gripper_close는 binary clamp
    
    state/action dim 레이아웃은 데이터셋 concat 순서와 일치해야 함.
    """

    # state 내 각 key의 dim (concat 순서 기준)
    state_eef_pos_dim: int = 3
    state_eef_quat_dim: int = 4
    state_gripper_dim: int = 1

    # action 내 각 key의 dim
    action_pos_delta_dim: int = 3
    action_rot_delta_dim: int = 3   # axis_angle: 이미 3D
    action_gripper_dim: int = 1

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        obs = transition.get(TransitionKey.OBSERVATION, {}) or {}

        # --- State 변환 ---
        if OBS_STATE in obs:
            state = obs[OBS_STATE]  # (B, D) or (D,)
            squeeze = state.dim() == 1
            if squeeze:
                state = state.unsqueeze(0)

            # split
            pos   = state[:, :self.state_eef_pos_dim]                         # (B, 3)
            quat  = state[:, self.state_eef_pos_dim:
                              self.state_eef_pos_dim + self.state_eef_quat_dim]  # (B, 4)
            grip  = state[:, self.state_eef_pos_dim + self.state_eef_quat_dim:] # (B, 1)

            rot6d = quaternion_to_rotation_6d(quat)  # (B, 6)

            new_state = torch.cat([pos, rot6d, grip], dim=-1)  # (B, 10)

            if squeeze:
                new_state = new_state.squeeze(0)
            obs[OBS_STATE] = new_state
            transition[TransitionKey.OBSERVATION] = obs

        # --- Action 변환 ---
        action = transition.get(TransitionKey.ACTION)
        if isinstance(action, torch.Tensor):
            squeeze = action.dim() == 1
            if squeeze:
                action = action.unsqueeze(0)

            has_time = action.dim() == 3  # (B, T, D)
            if not has_time:
                action = action.unsqueeze(1)  # (B, 1, D)

            b, t, d = action.shape
            action_flat = action.reshape(b * t, d)

            pos_d  = self.action_pos_delta_dim
            rot_d  = self.action_rot_delta_dim
            grip_d = self.action_gripper_dim

            pos_delta = action_flat[:, :pos_d]               # (BT, 3)
            rot_delta = action_flat[:, pos_d:pos_d+rot_d]    # (BT, 3) axis_angle → 그대로
            gripper   = action_flat[:, pos_d+rot_d:]         # (BT, 1)

            # binary clamp
            gripper = (gripper > 0.5).float()

            new_action = torch.cat([pos_delta, rot_delta, gripper], dim=-1)  # (BT, 7)
            new_action = new_action.reshape(b, t, -1)

            if not has_time:
                new_action = new_action.squeeze(1)
            if squeeze:
                new_action = new_action.squeeze(0)

            transition[TransitionKey.ACTION] = new_action

        return transition

    def transform_features(self, features):
        return features