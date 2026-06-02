from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.configs.train import TrainRLServerPipelineConfig
from lerobot.envs.configs import EnvConfig
from lerobot.policies.sac.configuration_sac import SACConfig
from lerobot.utils.constants import ACTION, OBS_STATE


@EnvConfig.register_subclass("fire_residual")
@dataclass
class FIReResidualEnvConfig(EnvConfig):
    task: str | None = "forge-peg_insert"
    fps: int = 15

    @property
    def gym_kwargs(self) -> Dict[str, object]:
        return {}


def build_train_config(
    *,
    policy: SACConfig,
    task: str,
    fps: int,
    output_dir: str,
    job_name: str = "fire_residual_sac",
    batch_size: int = 256,
    log_freq: int = 100,
    save_freq: int = 5000,
    save_checkpoint: bool = True,
) -> TrainRLServerPipelineConfig:
    features = {
        "state": PolicyFeature(
            type=FeatureType.STATE,
            shape=policy.input_features[OBS_STATE].shape,
        ),
        "action": PolicyFeature(
            type=FeatureType.ACTION,
            shape=policy.output_features[ACTION].shape,
        ),
    }
    features_map = {
        "state": OBS_STATE,
        "action": ACTION,
    }
    config = TrainRLServerPipelineConfig(
        dataset=None,
        env=FIReResidualEnvConfig(
            task=task,
            fps=fps,
            features=features,
            features_map=features_map,
        ),
        policy=policy,
        output_dir=Path(output_dir),
        job_name=job_name,
        batch_size=batch_size,
        log_freq=log_freq,
        save_freq=save_freq,
        save_checkpoint=save_checkpoint,
    )
    config.optimizer = policy.get_optimizer_preset()
    config.scheduler = policy.get_scheduler_preset()
    return config
