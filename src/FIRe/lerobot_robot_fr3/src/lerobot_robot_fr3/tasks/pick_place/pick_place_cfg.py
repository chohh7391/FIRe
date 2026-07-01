"""Config for the pick_place task.

Mirrors the vla_lab `base_line/pick_place` manager-based RL environment so a policy
trained there (rsl_rl PPO, joint-position control) can be deployed on the real FR3.

References (training side):
  - vla_lab/.../base_line/lift/lift_env_cfg.py            (obs/action structure)
  - vla_lab/.../base_line/pick_place/config/franka/joint_pos_env_cfg.py
  - vla_lab/_isaaclab/.../isaaclab_assets/robots/franka.py  (FRANKA_PANDA_CFG defaults)
"""


class PickPlaceCtrlCfg:
    # JointPositionActionCfg(scale=0.5, use_default_offset=True):
    #   joint_target = default_joint_pos + scale * action
    action_scale: float = 0.5


class PickPlaceTask:
    name: str = "pick_place"
    duration_s: float = 8.0  # episode_length_s in training

    # Franka default joint positions (FRANKA_PANDA_CFG.init_state.joint_pos).
    # Order: arm(7) then fingers(2) — matches IsaacLab joint ordering.
    default_arm_joint_pos: list = [0.0, -0.569, 0.0, -2.810, 0.0, 3.037, 0.741]
    default_finger_joint_pos: list = [0.04, 0.04]

    # Fixed object/target poses in the robot root frame (robot root ≈ world; robot at origin).
    # Perception is intentionally NOT wired here — these are constants the user edits to
    # match the real setup. Defaults follow the training scene nominal positions.
    object_position: list = [0.5, 0.0, 0.021]          # white cube (resting center)
    green_cube_position: list = [0.5, 0.15, 0.0203]    # green target cube

    # Command (object_pose): white cube center when stacked on the green cube.
    target_object_position: list = [0.5, 0.15, 0.0616]
    target_object_quat: list = [1.0, 0.0, 0.0, 0.0]    # wxyz, identity

    # BinaryJointPositionActionCfg: action > 0 -> open, else close.
    gripper_open: float = 0.04
    gripper_close: float = 0.0


class PickPlaceEnvCfg:
    action_space: int = 8        # 7 arm joints + 1 binary gripper
    observation_space: int = 39  # 9 + 9 + 3 + 7 + 8 + 3

    # Concatenation order of the policy observation. MUST match the training env's
    # ObservationsCfg.PolicyCfg term order (lift) + green_cube_position (pick_place).
    obs_order: list = [
        "joint_pos",               # joint_pos_rel       (9)
        "joint_vel",               # joint_vel_rel       (9)
        "object_position",         # white cube in root  (3)
        "target_object_position",  # command pose        (7)
        "actions",                 # last_action         (8)
        "green_cube_position",     # green cube in root  (3)
    ]

    task_name: str = "pick_place"
    task: PickPlaceTask = PickPlaceTask()
    ctrl: PickPlaceCtrlCfg = PickPlaceCtrlCfg()


class PickPlaceTaskCfg(PickPlaceEnvCfg):
    task_name = "pick_place"
    task = PickPlaceTask()
