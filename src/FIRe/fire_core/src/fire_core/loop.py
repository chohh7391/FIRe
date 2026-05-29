from __future__ import annotations

import time
from typing import Optional

from fire_core.strategies import ControlStrategy, ReplayPoseStrategy
from fire_core.io.logger import StepLogger


def run_control_loop(
    robot,
    strategy: ControlStrategy,
    *,
    control_hz: float,
    logger: Optional[StepLogger] = None,
) -> None:
    dt = 1.0 / control_hz
    print(f"\n[INFO] Control loop @ {control_hz} Hz  (Ctrl+C to stop)")

    strategy.reset()

    step_idx = 0
    try:
        while True:
            t_start = time.time()

            result = strategy.step(step_idx)
            if result is None:
                break

            if isinstance(strategy, ReplayPoseStrategy):
                if result.action_dict:
                    result.processed_action = robot.send_processed_action(result.action_dict)
            else:
                result.processed_action = robot.send_action(result.action_dict)

            if logger is not None:
                logger.record(result)

            result = strategy.after_action_sent(step_idx, result)

            sleep_t = dt - (time.time() - t_start)
            if sleep_t > 0:
                time.sleep(sleep_t)

            step_idx += 1

    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user.")