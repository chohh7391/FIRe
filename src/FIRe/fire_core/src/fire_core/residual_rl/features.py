from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.sac.configuration_sac import SACConfig
from lerobot.utils.constants import ACTION, OBS_STATE

from fire_core.utils import Features, total_dim


OBSERVATION_STATE_KEY = OBS_STATE
ACTION_KEY = ACTION


def make_residual_action_features(action_features: Features) -> Features:
    if "arm_actions" in action_features:
        return {"arm_actions": action_features["arm_actions"]}
    return dict(action_features)


def make_residual_observation_features(
    obs_features: Features,
    action_features: Features,
) -> Features:
    features = dict(obs_features)
    residual_action_dim = total_dim(make_residual_action_features(action_features))
    if "prev_actions" in features:
        prev_dim = int(np.prod(features["prev_actions"]))
        if prev_dim > residual_action_dim:
            features["prev_actions"] = (residual_action_dim,)
    return features


def task_arm_action_dim(task: object) -> int:
    action = getattr(task, "action", None)
    if action is not None:
        return int(np.asarray(action).reshape(-1).shape[0])

    action_features = getattr(task, "action_features", {})
    if "arm_actions" in action_features:
        return int(np.prod(action_features["arm_actions"]))
    return total_dim(action_features)


def flatten_feature_dict(data: Dict[str, np.ndarray], features: Features) -> np.ndarray:
    values: list[np.ndarray] = []
    for key, shape in features.items():
        arr = np.asarray(data[key], dtype=np.float32).reshape(-1)
        dim = int(np.prod(shape))
        values.append(arr[:dim])
    if not values:
        return np.zeros((0,), dtype=np.float32)
    return np.concatenate(values, axis=0).astype(np.float32)


def make_state_batch(
    obs_dict: Dict[str, np.ndarray],
    obs_features: Features,
    *,
    device: str,
) -> Dict[str, torch.Tensor]:
    state = flatten_feature_dict(obs_dict, obs_features)
    return {
        OBSERVATION_STATE_KEY: torch.as_tensor(
            state, dtype=torch.float32, device=device
        ).unsqueeze(0)
    }


def build_sac_config(
    *,
    obs_features: Features,
    action_features: Features,
    device: str,
    storage_device: str = "cpu",
    online_steps: int = 1000000,
    online_buffer_capacity: int = 100000,
    online_step_before_learning: int = 100,
    batch_size: int | None = None,
    use_torch_compile: bool = False,
) -> SACConfig:
    obs_dim = total_dim(obs_features)
    action_dim = total_dim(action_features)
    config = SACConfig(
        input_features={
            OBSERVATION_STATE_KEY: PolicyFeature(
                type=FeatureType.STATE,
                shape=(obs_dim,),
            )
        },
        output_features={
            ACTION_KEY: PolicyFeature(
                type=FeatureType.ACTION,
                shape=(action_dim,),
            )
        },
        device=device,
        storage_device=storage_device,
        online_steps=online_steps,
        online_buffer_capacity=online_buffer_capacity,
        online_step_before_learning=online_step_before_learning,
        vision_encoder_name=None,
        use_torch_compile=use_torch_compile,
    )
    if batch_size is not None:
        # Kept here so callers can store one source of truth near policy creation;
        # TrainRLServerPipelineConfig still owns the actual learner batch size.
        config.batch_size = batch_size  # type: ignore[attr-defined]
    return config


def action_to_numpy(action: torch.Tensor, action_dim: int) -> np.ndarray:
    return action.detach().cpu().numpy().reshape(-1)[:action_dim].astype(np.float32)


def pad_action(action: np.ndarray, action_dim: int) -> np.ndarray:
    arr = np.asarray(action, dtype=np.float32).reshape(-1)
    if arr.shape[0] >= action_dim:
        return arr[:action_dim].copy()
    padded = np.zeros((action_dim,), dtype=np.float32)
    padded[: arr.shape[0]] = arr
    return padded
