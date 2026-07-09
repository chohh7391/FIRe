from __future__ import annotations

import time
from typing import Optional
import numpy as np

from fire_core.strategies import ControlStrategy
from fire_core.logger import StepLogger
# BaseRecorder가 위치한 정확한 경로로 import를 맞춰주세요.
from fire_core.recorders.base_recorder import BaseRecorder


def run_control_loop(
    robot,
    strategy: ControlStrategy,
    *,
    control_hz: float,
    max_steps: int,
    logger: Optional[StepLogger] = None,
    recorder: Optional[BaseRecorder] = None
) -> None:
    dt = 1.0 / control_hz
    print(f"\n[INFO] Control loop @ {control_hz} Hz  (Ctrl+C to stop)")

    strategy.reset()

    step_idx = 0
    try:
        while True:
            if max_steps > 0 and step_idx >= max_steps:
                print(f"\n[INFO] Reached max steps ({max_steps}). Auto-stopping.")
                break

            t_start = time.time()

            # 1. 전략(Strategy)을 통해 다음 액션 계산
            result = strategy.step(step_idx)
            if result is None:
                break

            # 2. 로봇에게 액션 전달
            #    절대 task-space pose를 내보내는 전략(teleop replay, VLA-only)은
            #    process_action(상대 delta 변환)을 건너뛰고 teleop과 동일한
            #    절대 pose 경로로 전송한다.
            if getattr(strategy, "sends_task_space_pose", False):
                if result.action_dict:
                    result.processed_action = robot.send_processed_action(result.action_dict)
            else:
                result.processed_action = robot.send_action(result.action_dict)

            # 3. CSV 등 기존 Logger 기록
            if logger is not None:
                logger.record(result)

            # 4. LeRobot/GR00T/Pi0 데이터셋 기록 (핵심 부분)
            if recorder is not None:                
                arm_action = result.policy_action

                # 그리퍼 액션 추출 (로봇 태스크 인터페이스 활용)
                gripper_action = robot.task.get_gripper_action(result.action_dict)

                recorder.record(
                    arm_action=arm_action,
                    gripper_action=gripper_action,
                )

            # 5. 후처리 작업
            result = strategy.after_action_sent(step_idx, result)

            # 6. 주파수(Hz) 동기화
            sleep_t = dt - (time.time() - t_start)
            if sleep_t > 0:
                time.sleep(sleep_t)

            step_idx += 1

    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user.")