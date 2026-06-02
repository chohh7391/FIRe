from __future__ import annotations

import logging
import os

from lerobot.rl.learner import start_learner_threads
from lerobot.rl.process import ProcessSignalHandler
from lerobot.rl.wandb_utils import WandBLogger
from lerobot.utils.random_utils import set_seed
from lerobot.utils.utils import init_logging

from .config import build_train_config
from .features import build_sac_config


def run_residual_learner(
    *,
    obs_features: dict[str, tuple[int, ...]],
    action_features: dict[str, tuple[int, ...]],
    task: str,
    fps: int,
    output_dir: str,
    device: str,
    storage_device: str,
    batch_size: int,
    online_steps: int,
    online_buffer_capacity: int,
    online_step_before_learning: int,
) -> None:
    policy = build_sac_config(
        obs_features=obs_features,
        action_features=action_features,
        device=device,
        storage_device=storage_device,
        online_steps=online_steps,
        online_buffer_capacity=online_buffer_capacity,
        online_step_before_learning=online_step_before_learning,
    )
    cfg = build_train_config(
        policy=policy,
        task=task,
        fps=fps,
        output_dir=output_dir,
        batch_size=batch_size,
    )

    os.makedirs(os.path.join(output_dir, "logs"), exist_ok=True)
    init_logging(log_file=os.path.join(output_dir, "logs", "residual_learner.log"))
    logging.info("Starting FIRe residual SAC learner")
    set_seed(seed=cfg.seed)

    shutdown_event = ProcessSignalHandler(use_threads=True).shutdown_event
    wandb_logger: WandBLogger | None = None
    if cfg.wandb.enable and cfg.wandb.project:
        wandb_logger = WandBLogger(cfg)

    start_learner_threads(
        cfg=cfg,
        wandb_logger=wandb_logger,
        shutdown_event=shutdown_event,
    )
