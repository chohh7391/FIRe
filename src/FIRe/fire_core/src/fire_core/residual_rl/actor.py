from __future__ import annotations

import logging
import os
import time
from queue import Queue
from threading import Thread
from typing import Any, Optional

import numpy as np
import torch
from lerobot.policies.sac.modeling_sac import SACPolicy
from lerobot.rl.actor import (
    establish_learner_connection,
    learner_service_client,
    push_transitions_to_transport_queue,
    receive_policy,
    send_interactions,
    send_transitions,
    update_policy_parameters,
)
from lerobot.rl.process import ProcessSignalHandler
from lerobot.transport.utils import python_object_to_bytes
from lerobot.utils.random_utils import set_seed
from lerobot.utils.transition import Transition
from lerobot.utils.utils import init_logging

from fire_core.utils import total_dim
from fire_core.vla_observation import (
    VLAObservationNotReady,
    get_ready_vla_observation,
    wait_for_ready_vla_observation,
)

from .action import build_vla_action_chunk, combine_vla_and_residual, make_robot_action
from .config import build_train_config
from .features import (
    action_to_numpy,
    build_sac_config,
    make_residual_action_features,
    make_residual_observation_features,
    make_state_batch,
    pad_action,
    task_arm_action_dim,
)


class ResidualVLAProvider:
    def reset(self, robot: Any, action_dim: int) -> None:
        pass

    def get_action(self, robot: Any, action_dim: int) -> Optional[np.ndarray]:
        return None


class ZeroVLAProvider(ResidualVLAProvider):
    def get_action(self, robot: Any, action_dim: int) -> Optional[np.ndarray]:
        return np.zeros((action_dim,), dtype=np.float32)


class AsyncVLAProvider(ResidualVLAProvider):
    def __init__(self, vla_policy: Any, chunk_size: int) -> None:
        self._vla_policy = vla_policy
        self._chunk_size = int(chunk_size)
        self._chunk: Optional[np.ndarray] = None
        self._requested = False
        self._chunk_step = 0

    def reset(self, robot: Any, action_dim: int) -> None:
        raw = self._vla_policy.get_action_sync(wait_for_ready_vla_observation(robot))
        self._chunk = build_vla_action_chunk(raw, action_dim)
        self._requested = False
        self._chunk_step = 0

    def get_action(self, robot: Any, action_dim: int) -> Optional[np.ndarray]:
        if self._chunk is None:
            self.reset(robot, action_dim)
        assert self._chunk is not None

        chunk_idx = self._chunk_step % self._chunk_size
        if chunk_idx == 0 and self._chunk_step != 0 and self._requested:
            try:
                self._chunk = build_vla_action_chunk(self._vla_policy.get_result(), action_dim)
            except Exception as exc:
                logging.warning("VLA result failed, reusing previous chunk: %s", exc)
            self._requested = False

        if chunk_idx == self._chunk_size // 2 and not self._requested:
            try:
                self._vla_policy.request_action(get_ready_vla_observation(robot))
                self._requested = True
            except VLAObservationNotReady as exc:
                logging.warning("VLA request skipped: %s", exc)

        action = self._chunk[chunk_idx]
        self._chunk_step += 1
        return action


def run_residual_actor(
    *,
    robot: Any,
    task: str,
    fps: int,
    output_dir: str,
    device: str,
    storage_device: str,
    learner_host: str,
    learner_port: int,
    online_steps: int,
    online_buffer_capacity: int,
    online_step_before_learning: int,
    batch_size: int,
    residual_scale: float = 1.0,
    vla_provider: ResidualVLAProvider | None = None,
    teleop: Any | None = None,
) -> None:
    obs_features = make_residual_observation_features(
        robot.task.observation_features,
        robot.task.action_features,
    )
    action_features = make_residual_action_features(robot.task.action_features)
    action_dim = total_dim(action_features)
    robot_arm_dim = task_arm_action_dim(robot.task)

    policy_cfg = build_sac_config(
        obs_features=obs_features,
        action_features=action_features,
        device=device,
        storage_device=storage_device,
        online_steps=online_steps,
        online_buffer_capacity=online_buffer_capacity,
        online_step_before_learning=online_step_before_learning,
    )
    policy_cfg.actor_learner_config.learner_host = learner_host
    policy_cfg.actor_learner_config.learner_port = learner_port
    cfg = build_train_config(
        policy=policy_cfg,
        task=task,
        fps=fps,
        output_dir=output_dir,
        batch_size=batch_size,
    )

    os.makedirs(os.path.join(output_dir, "logs"), exist_ok=True)
    init_logging(log_file=os.path.join(output_dir, "logs", "residual_actor.log"))
    logging.info("Starting FIRe residual actor")

    shutdown_event = ProcessSignalHandler(use_threads=True).shutdown_event
    learner_client, grpc_channel = learner_service_client(host=learner_host, port=learner_port)
    if not establish_learner_connection(learner_client, shutdown_event):
        raise ConnectionError("Failed to connect to residual learner.")

    parameters_queue: Queue[bytes] = Queue()
    transitions_queue: Queue[bytes] = Queue()
    interactions_queue: Queue[bytes] = Queue()

    receive_thread = Thread(
        target=receive_policy,
        args=(cfg, parameters_queue, shutdown_event, grpc_channel),
        daemon=True,
    )
    transition_thread = Thread(
        target=send_transitions,
        args=(cfg, transitions_queue, shutdown_event, grpc_channel),
        daemon=True,
    )
    interaction_thread = Thread(
        target=send_interactions,
        args=(cfg, interactions_queue, shutdown_event, grpc_channel),
        daemon=True,
    )
    receive_thread.start()
    transition_thread.start()
    interaction_thread.start()

    set_seed(seed=cfg.seed)
    policy = SACPolicy(policy_cfg).to(device).eval()
    initial_sync_start = time.time()
    while parameters_queue.empty() and time.time() - initial_sync_start < 10.0:
        time.sleep(0.1)
    if parameters_queue.empty():
        logging.warning(
            "No initial learner parameters received within 10 seconds; "
            "actor will continue with locally initialized policy parameters."
        )
    update_policy_parameters(policy=policy, parameters_queue=parameters_queue, device=device)

    if not robot.is_connected:
        robot.connect()
    if vla_provider is None:
        vla_provider = ResidualVLAProvider()
    vla_provider.reset(robot, action_dim)
    if teleop is not None:
        teleop.connect()

    dt = 1.0 / float(fps)
    obs_dict = robot.get_observation()
    state = make_state_batch(obs_dict, obs_features, device=device)

    episode_reward = 0.0
    episode_steps = 0
    episode_interventions = 0
    pending: list[Transition] = []

    try:
        for step_idx in range(online_steps):
            if shutdown_event.is_set():
                break
            start_time = time.perf_counter()
            update_policy_parameters(policy=policy, parameters_queue=parameters_queue, device=device)

            with torch.no_grad():
                residual_tensor = policy.select_action(batch=state)
            residual_action = action_to_numpy(residual_tensor, action_dim) * float(residual_scale)
            vla_action = vla_provider.get_action(robot, action_dim)
            action_for_learning = residual_action
            is_intervention = 0
            if teleop is not None:
                if hasattr(teleop, "get_intervention_action"):
                    teleop_action, intervention = teleop.get_intervention_action()
                else:
                    raw_teleop_action = teleop.get_action()
                    teleop_action = raw_teleop_action.get(
                        "arm_actions",
                        np.zeros((action_dim,), dtype=np.float32),
                    )
                    intervention_value = raw_teleop_action.get("is_intervention", False)
                    intervention = bool(np.asarray(intervention_value).reshape(-1)[0])
                if intervention:
                    action_for_learning = teleop_action
                    residual_action = teleop_action
                    is_intervention = 1
                    episode_interventions += 1
                    key = getattr(teleop, "last_key", None)
                    print(
                        f"[TELEOP] key={key} residual_action={teleop_action}",
                        flush=True,
                    )
            final_action = combine_vla_and_residual(
                vla_action=vla_action,
                residual_action=residual_action,
                action_dim=action_dim,
            )
            robot_action = make_robot_action(pad_action(final_action, robot_arm_dim))

            if is_intervention and hasattr(robot, "send_teleop_action"):
                from lerobot_robot_fr3.fr3 import TeleopAction

                processed_action = robot.send_teleop_action(
                    TeleopAction(
                        action=robot_action,
                        action_space="task_space",
                        is_relative=True,
                    )
                )
            else:
                processed_action = robot.send_action(robot_action)
            next_obs_dict = robot.get_observation()
            next_state = make_state_batch(next_obs_dict, obs_features, device=device)
            reward = float(robot.task.get_reward())
            done = bool(robot.task.get_done())
            truncated = bool(robot.task.get_truncated())
            info = robot.task.get_info()

            pending.append(
                Transition(
                    state=state,
                    action=torch.as_tensor(
                        action_for_learning,
                        dtype=torch.float32,
                        device=device,
                    ).unsqueeze(0),
                    reward=reward,
                    next_state=next_state,
                    done=done,
                    truncated=truncated,
                    complementary_info={
                        "is_intervention": is_intervention,
                    },
                )
            )
            episode_reward += reward
            episode_steps += 1

            if pending:
                push_transitions_to_transport_queue(pending, transitions_queue)
                pending = []

            if done or truncated:
                interactions_queue.put(
                    python_object_to_bytes(
                        {
                            "Episodic reward": episode_reward,
                            "Interaction step": step_idx,
                            "Episode intervention": episode_interventions,
                            "Intervention rate": episode_interventions / max(episode_steps, 1),
                            **info,
                        }
                    )
                )
                logging.info(
                    "Episode ended: reward=%s steps=%s interventions=%s done=%s truncated=%s",
                    episode_reward,
                    episode_steps,
                    episode_interventions,
                    done,
                    truncated,
                )
                robot.task.reset()
                vla_provider.reset(robot, action_dim)
                next_obs_dict = robot.get_observation()
                next_state = make_state_batch(next_obs_dict, obs_features, device=device)
                episode_reward = 0.0
                episode_steps = 0
                episode_interventions = 0

            state = next_state
            elapsed = time.perf_counter() - start_time
            if elapsed < dt:
                time.sleep(dt - elapsed)

    finally:
        shutdown_event.set()
        if teleop is not None:
            teleop.disconnect()
        robot.disconnect()
        receive_thread.join(timeout=2.0)
        transition_thread.join(timeout=2.0)
        interaction_thread.join(timeout=2.0)
        grpc_channel.close()
