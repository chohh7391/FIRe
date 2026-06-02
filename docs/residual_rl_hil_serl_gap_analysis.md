# FIRe Residual RL HIL-SERL Gap Analysis

This document tracks what is already implemented in the FIRe residual RL path, which HIL-SERL pieces are intentionally omitted, and what still needs to be implemented before treating the system as a complete online RL + human intervention training workflow.

## Current Status

The current implementation supports the core online actor-learner loop:

| Component | Status | Notes |
| --- | --- | --- |
| Separate actor and learner processes | Implemented | `scripts/train_residual_actor.py` and `scripts/train_residual_learner.py`. |
| SAC policy with learner-to-actor parameter sync | Implemented | Uses LeRobot RL transport and SAC policy; warns if initial sync does not arrive within 10 seconds. |
| gRPC transition and interaction transport | Implemented | Actor streams transitions and receives learner parameters. |
| VLA + residual action composition | Implemented | `vla_action + residual_action`; `--test` uses zero VLA action. |
| Residual action feature filtering | Implemented | Residual SAC uses controllable `arm_actions` only; non-control heads such as Forge `success_pred` are excluded from the residual policy action space. |
| Asynchronous VLA action chunk provider | Implemented | GR00T/PI05 async client path is wired through `AsyncVLAProvider`; chunk indexing is reset per provider reset. |
| Fixed-rate control loop | Implemented | Actor sleeps to match `fps`, default 15 Hz. |
| Keyboard human intervention | Implemented, limited | Keyboard override can replace residual action; configured scale is applied to valid key presses. |
| Intervention flag in transition | Implemented | Stored in `Transition.complementary_info["is_intervention"]`. |
| Episode intervention metrics | Implemented | Reports episode intervention count and intervention rate on episode end. |
| Real task observation reuse | Implemented | Uses existing task observation values; residual SAC truncates `prev_actions` to the controllable residual action dimension when the task has non-control action heads. |
| Reward/done/truncated hooks | Stubbed | Default task hooks return `0.0`, `False`, `False`. |
| Offline demo pretraining | Not implemented | Intentionally omitted for now because the VLA is assumed to be finetuned. |
| Reward classifier | Not implemented | User plans to define reward directly. |

The practical summary is:

The system is a working FIRe-style online residual SAC scaffold inspired by HIL-SERL. It is not yet a complete HIL-SERL implementation. The main missing parts are actual task reward/termination logic, safety bounds, and optional demo/replay prefill.

## Remaining Issues

### Gripper action is hardcoded

Current behavior:

```python
"gripper_actions": np.array([-1.0], dtype=np.float32),
```

This is acceptable for current peg insertion smoke tests if gripper control is irrelevant, but it is not general. Tasks requiring open/close control need `make_robot_action()` to accept an optional gripper action.

Suggested signature:

```python
def make_robot_action(
    arm_action: np.ndarray,
    gripper_action: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    ...
```

Keep `-1.0` as the default for current behavior.

### Keyboard intervention is limited to X/Y relative control

The keyboard teleoperator currently maps:

- `w`: +x
- `s`: -x
- `a`: +y
- `d`: -y

All other action dimensions are zero. This is fine for a minimal smoke-test intervention device, but it is not enough for precise correction in contact-rich manipulation. For stronger HIL behavior, add Z and rotation bindings or use a richer teleoperator such as a leader arm.

This is a limitation, not a bug.

### Human intervention currently overrides only the residual action

When intervention is active, the actor replaces `residual_action` and `action_for_learning` with the teleop action. The final command is still:

```python
final_action = vla_action + residual_action
```

So the intervention overrides the residual part, not the full robot command. This matches the current FIRe design if teleop is intended to teach the residual correction. If the desired behavior is full human takeover, the actor needs a separate mode where teleop replaces the final command instead of only the residual.

### Reward and termination are required for learning

The task hooks currently default to:

- `get_reward() -> 0.0`
- `get_done() -> False`
- `get_truncated() -> False`

With these defaults, the actor and learner can run, but the policy cannot learn meaningful task behavior. Implement task-specific reward and termination in `lerobot_robot_fr3/tasks`.

### Episode reset is only task-level

When `done` or `truncated` is true, the actor calls:

```python
robot.task.reset()
```

This does not necessarily reset the physical robot, object, or controller state. For real robot training, define what reset means per task. Options include manual reset, scripted reset pose, or a reset service in the ROS controller.

### Residual action safety bounds are minimal

`residual_scale` limits magnitude globally, but there is no axis-wise clipping or workspace safety layer in the residual actor. Add hard limits before sending commands to the robot.

Recommended checks:

- Per-axis residual delta limit
- Max final end-effector delta
- Optional workspace bounds
- Optional force/torque abort threshold
- Optional action NaN/Inf guard

### Offline demo prefill is omitted

This is intentional for now because the VLA is already finetuned. However, the implementation currently does not include:

- Demo replay buffer prefill
- Behavior cloning warm start
- Reward classifier training
- HIL-SERL-style offline intervention dataset reuse

This is acceptable for the current FIRe assumption, but it should be documented as a design choice rather than described as full HIL-SERL.

### Intervention data is not used as a separate demo buffer

Human intervention actions are stored as online SAC transitions, so they do contribute to learning. They are not currently routed into a separate demonstration buffer or used with a BC auxiliary loss.

If stronger HIL-SERL behavior is desired, add:

- Intervention transition tagging with LeRobot-compatible intervention event keys
- Separate intervention/demo buffer
- Optional supervised loss on intervention actions

### Policy normalization and feature statistics need validation

LeRobot warns when creating a policy from environment features without dataset statistics. Smoke tests have run, but long real-robot training should verify whether normalization modules are active and whether feature/action scaling is appropriate.

Recommended follow-up:

- Inspect saved `pretrained_model/config.json`
- Verify observation magnitudes
- Add explicit feature normalization if SAC uses raw force/pose values with very different scales

### Checkpoint ergonomics need improvement

The learner has default `save_freq=5000` through `build_train_config()`, but the scripts do not expose it. For real training and `scripts/play_real.py`, expose:

- `--save_freq`
- `--log_freq`
- Optional `--save_checkpoint/--no_save_checkpoint`

### VLA action is not included in the residual policy observation

The residual policy currently observes the task observation only. This follows the current request to keep FIRe model observations unchanged. The tradeoff is that the residual policy may be partially blind to the nominal VLA action it is correcting.

This is acceptable as a first version. If residual learning becomes unstable when VLA actions vary, consider adding the current VLA action to the residual policy observation in a future experiment.

## Recommended Implementation Order

1. Make `make_robot_action()` accept optional gripper actions.
2. Implement task-specific reward, done, truncated, and info.
3. Add residual action safety clipping and NaN/Inf guards.
4. Define real reset behavior per task.
5. Expose learner save/log frequency CLI options.
6. Decide whether intervention should override residual only or full final action for each teleoperator.
