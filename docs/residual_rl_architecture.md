# FIRe Residual RL Architecture

This document describes how the current residual RL path runs in FIRe.

The short version:

FIRe runs a LeRobot SAC learner and a real-robot actor as separate processes. The actor reads the existing FR3 task observation, predicts a residual action with SAC, adds it to a VLA action, sends the combined command to the FR3 ROS client, and streams the resulting transition back to the learner. Human teleoperation can override the residual action during online training.

## Entry Points

| Script | Purpose |
| --- | --- |
| `scripts/train_residual_learner.py` | Starts the LeRobot SAC learner server. |
| `scripts/train_residual_actor.py` | Runs the real-robot actor, collects online transitions, and streams them to the learner. |
| `scripts/play_real.py` | Loads a saved residual SAC policy and executes it on the real robot without learner updates. |

The learner must be started before the actor.

## Process Layout

```text
┌─────────────────────────────┐       gRPC        ┌─────────────────────────────┐
│ Learner process              │ <───────────────> │ Actor process                │
│ scripts/train_residual_...   │                  │ scripts/train_residual_...   │
│                              │                  │                              │
│ - SACPolicy                  │  parameters      │ - SACPolicy copy             │
│ - replay buffer              │ ───────────────> │ - FR3Robot                   │
│ - optimizer updates          │                  │ - VLA provider               │
│ - checkpointing              │  transitions     │ - optional teleop            │
│                              │ <──────────────  │ - control loop               │
└─────────────────────────────┘                  └─────────────────────────────┘
```

The implementation uses LeRobot's actor/learner transport helpers:

- `receive_policy`
- `send_transitions`
- `send_interactions`
- `start_learner_threads`

The FIRe-specific code lives in `src/FIRe/fire_core/src/fire_core/residual_rl`.

## Feature Construction

Residual SAC does not train on every task action feature directly. It builds a residual-specific feature view:

- Observation features come from `robot.task.observation_features`.
- Action features are reduced to controllable `arm_actions`.
- If an observation contains `prev_actions`, it is truncated to the residual action dimension.

This matters for Forge tasks because the original Forge action format contains a non-control head:

```text
Forge.action_features = {
    "arm_actions": (6,),
    "success_pred": (1,),
}
```

Residual SAC learns only the 6 controllable arm dimensions. Before sending to the robot, the 6D residual command is padded back to the task's internal action buffer size when needed, so existing Forge task processing remains compatible.

Relevant helpers:

- `make_residual_action_features()`
- `make_residual_observation_features()`
- `task_arm_action_dim()`
- `make_state_batch()`

## Learner Flow

`scripts/train_residual_learner.py` creates a task only to read its feature definitions, then starts the learner.

Flow:

1. Create task with `create_task(args.task)`.
2. Build residual observation/action features.
3. Build a `SACConfig`.
4. Build a `TrainRLServerPipelineConfig` with `dataset=None`.
5. Start LeRobot learner threads with `start_learner_threads()`.

There is no offline dataset prefill in the current residual RL path. The replay buffer is filled by online actor transitions.

## Actor Flow

`scripts/train_residual_actor.py` creates the real `FR3Robot`, optional VLA provider, optional teleoperator, and starts `run_residual_actor()`.

Per control step:

1. Receive any newer learner policy parameters.
2. Convert the current task observation into `observation.state`.
3. Run SAC policy to get `residual_action`.
4. Apply `residual_scale`.
5. Get the current VLA action.
6. If human intervention is active, replace the residual action with teleop action.
7. Compute:

```python
final_action = vla_action + residual_action
```

8. Pad `final_action` to the task's internal arm action dimension if needed.
9. Send the command to the robot.
10. Read next observation, reward, done, truncated, and info from the task.
11. Create a LeRobot `Transition`.
12. Stream the transition to the learner.

The actor loop runs at `fps`, default 15 Hz.

## VLA Providers

The actor supports three VLA modes.

| Provider | Behavior |
| --- | --- |
| `ResidualVLAProvider` | Returns `None`; the final command is residual-only. |
| `ZeroVLAProvider` | Returns zero action; useful for `--test`. |
| `AsyncVLAProvider` | Uses an async GR00T/PI05 client and action chunks. |

`AsyncVLAProvider` requests a new action chunk halfway through the current chunk and swaps to the result on the next chunk boundary. On reset, its chunk index is reset so each episode starts at chunk index 0.

VLA raw actions are converted with `build_vla_action_chunk()`, which reads:

- `action.eef_position_delta`
- `action.eef_rotation_delta`

and pads/truncates them to the residual action dimension.

## Human Intervention

The actor accepts an optional teleoperator. The currently wired teleop device is:

```text
lerobot_teleoperator_keyboard.FIReKeyboardTeleop
```

Keyboard mapping:

| Key | Residual direction |
| --- | --- |
| `w` | +x |
| `s` | -x |
| `a` | +y |
| `d` | -y |

When intervention is active:

- `residual_action` is replaced by the teleop action.
- `action_for_learning` is also replaced by the teleop action.
- The transition stores `complementary_info["is_intervention"] = 1`.
- Episode intervention count and intervention rate are sent as interaction metrics on episode end.

Important: intervention currently overrides only the residual action. The final command is still `vla_action + teleop_residual_action`, not pure teleop takeover.

## Robot Command Path

The actor builds:

```python
robot_action = {
    "arm_actions": final_action,
    "gripper_actions": np.array([-1.0], dtype=np.float32),
}
```

Normal policy action path:

```text
actor
  -> robot.send_action(robot_action)
  -> task.process_action(...)
  -> robot.apply_action(...)
  -> ROS ActionChunk publish
```

Human intervention path:

```text
actor
  -> robot.send_teleop_action(TeleopAction(..., action_space="task_space", is_relative=True))
  -> task.process_action(...)
  -> robot.apply_action(...)
  -> ROS ActionChunk publish
```

The FR3 package is a ROS client. The low-level control loop is handled by the external ROS controller project.

## Transition Contents

Each actor step sends a LeRobot transition:

```python
Transition(
    state=state,
    action=action_for_learning,
    reward=robot.task.get_reward(),
    next_state=next_state,
    done=robot.task.get_done(),
    truncated=robot.task.get_truncated(),
    complementary_info={"is_intervention": is_intervention},
)
```

The action stored for learning is the residual action. During intervention, it is the teleop residual action.

## Episode Handling

At each step, the actor reads:

- `robot.task.get_reward()`
- `robot.task.get_done()`
- `robot.task.get_truncated()`
- `robot.task.get_info()`

When `done` or `truncated` is true:

1. Episode metrics are sent to the learner interaction stream.
2. `robot.task.reset()` is called.
3. The VLA provider is reset.
4. Episode counters are cleared.

Current default task hooks are stubs, so reward is zero and episodes do not end until task-specific logic is implemented.

## Play Real Flow

`scripts/play_real.py` runs a saved SAC policy without learner communication.

Flow:

1. Resolve a LeRobot `pretrained_model` directory from `--policy`.
2. Load `SACPolicy.from_pretrained(...)`.
3. Create `FR3Robot`.
4. Build residual observation/action feature view.
5. Run the same `vla_action + residual_action` command path.

It supports `--test`, GR00T/PI05 VLA providers, and optional keyboard teleop.

## Typical Commands

Start learner:

```bash
python scripts/train_residual_learner.py \
  --task forge-peg_insert \
  --fps 15 \
  --output_dir outputs/fire_residual \
  --device cuda \
  --storage_device cpu
```

Start actor without VLA server:

```bash
python scripts/train_residual_actor.py \
  --task forge-peg_insert \
  --fps 15 \
  --output_dir outputs/fire_residual \
  --device cuda \
  --storage_device cpu \
  --learner_host 127.0.0.1 \
  --learner_port 50051 \
  --test \
  --residual_scale 0.05 \
  --teleop_device keyboard
```

Play a trained residual policy:

```bash
python scripts/play_real.py \
  --task forge-peg_insert \
  --policy outputs/fire_residual \
  --device cuda \
  --test \
  --residual_scale 0.05
```

