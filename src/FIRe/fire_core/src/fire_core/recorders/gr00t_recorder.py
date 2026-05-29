from __future__ import annotations

import numpy as np
import pandas as pd
import json
import shutil
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

from .base_recorder import BaseRecorder, VALID_TASK
from .utils import vector_feature


class GR00TRecorder(BaseRecorder):
    """Project GR00T 규격에 맞게 데이터를 변환하고 저장하는 레코더입니다."""

    def _build_vla_dataset_features(self) -> dict[str, dict]:
        gr00t_action_names = [f"arm_action_{i}" for i in range(6)] + ["gripper_action"]
        features = {
            "observation.state": vector_feature(self._vla_state_names),
            "action": vector_feature(gr00t_action_names),
        }
        features.update(self._image_features(use_view_suffix=True))
        return features

    def _format_action(self, arm_action: np.ndarray, gripper_action: np.ndarray) -> np.ndarray:
        # 기존 _gr00t_action 로직
        arm = np.asarray(arm_action, dtype=np.float32).reshape(-1)
        gripper = np.asarray(gripper_action, dtype=np.float32).reshape(-1)
        return np.concatenate([arm[:6], gripper[:1]]).astype(np.float32, copy=False)

    def _post_save_processing(self, success: bool) -> None:
        """저장 완료 후 GR00T 규격으로 변환합니다."""
        self._convert_vla_dataset_to_gr00t(success)
        self._remove_images_dir(self._vla_dataset.root) # Base가 아닌 여기서 처리

    # 기존 GR00T 전용 헬퍼 함수들 (내부 구현은 기존과 동일)
    def _convert_vla_dataset_to_gr00t(self, success: bool) -> None:
        root = Path(self._vla_dataset.root)
        meta_dir = root / "meta"
        data_dir = root / "data" / "chunk-000"
        src_data = data_dir / "file-000.parquet"
        dst_data = data_dir / "episode_000000.parquet"

        df = pd.read_parquet(src_data)
        total_frames = len(df)
        reward = np.zeros(total_frames, dtype=np.float64)
        done = np.zeros(total_frames, dtype=bool)
        validity = np.zeros(total_frames, dtype=bool)
        if success and total_frames > 0:
            reward[-1] = 1.0
            done[-1] = True
            validity[-1] = True

        task_index = int(self._task_info["task_index"])
        gr00t_df = pd.DataFrame(
            {
                "observation.state": [
                    np.asarray(v, dtype=np.float64) for v in df["observation.state"].to_list()
                ],
                "action": [np.asarray(v, dtype=np.float64) for v in df["action"].to_list()],
                "timestamp": df["timestamp"].astype(np.float64).to_numpy(),
                "annotation.human.action.task_description": np.full(
                    total_frames, task_index, dtype=np.int64
                ),
                "task_index": np.full(total_frames, task_index, dtype=np.int64),
                "annotation.human.validity": validity,
                "episode_index": np.zeros(total_frames, dtype=np.int64),
                "index": np.arange(total_frames, dtype=np.int64),
                "next.reward": reward,
                "next.done": done,
            }
        )
        gr00t_df.to_parquet(dst_data, index=False)
        if src_data != dst_data and src_data.exists():
            src_data.unlink()

        video_root = root / "videos"
        for name in ("left_view", "right_view", "wrist_view"):
            src_video = video_root / f"observation.images.{name}" / "chunk-000" / "file-000.mp4"
            dst_dir = video_root / "chunk-000" / f"observation.images.{name}"
            dst_dir.mkdir(parents=True, exist_ok=True)
            dst_video = dst_dir / "episode_000000.mp4"
            if src_video.exists():
                if dst_video.exists():
                    dst_video.unlink()
                shutil.move(str(src_video), str(dst_video))

        for stale_dir in video_root.glob("observation.images.*"):
            if stale_dir.is_dir():
                shutil.rmtree(stale_dir)

        tasks_parquet = meta_dir / "tasks.parquet"
        if tasks_parquet.exists():
            tasks_parquet.unlink()
        episodes_dir = meta_dir / "episodes"
        if episodes_dir.exists():
            shutil.rmtree(episodes_dir)

        with (meta_dir / "tasks.jsonl").open("w") as f:
            f.write(json.dumps(self._task_info) + "\n")
            f.write(json.dumps(VALID_TASK) + "\n")
        with (meta_dir / "episodes.jsonl").open("w") as f:
            f.write(
                json.dumps(
                    {
                        "episode_index": 0,
                        "tasks": [self._task_info["task"], VALID_TASK["task"]],
                        "length": total_frames,
                    }
                )
                + "\n"
            )
        with (meta_dir / "modality.json").open("w") as f:
            json.dump(self._gr00t_modality(), f, indent=4)
        with (meta_dir / "info.json").open("w") as f:
            json.dump(self._gr00t_info(total_frames), f, indent=4)
        self._write_gr00t_stats(gr00t_df, meta_dir)
        self._remove_images_dir(root)

    def _gr00t_modality(self) -> dict[str, Any]:
        return {
            "state": {
                "eef_position": {"start": 0, "end": 3},
                "eef_quaternion": {"start": 3, "end": 7, "rotation_type": "quaternion"},
                "gripper_qpos": {"start": 7, "end": 9},
            },
            "action": {
                "eef_position_delta": {"start": 0, "end": 3},
                "eef_rotation_delta": {"start": 3, "end": 6, "rotation_type": "axis_angle"},
                "gripper_close": {"start": 6, "end": 7},
            },
            "video": {
                "left_view": {"original_key": "observation.images.left_view"},
                "right_view": {"original_key": "observation.images.right_view"},
                "wrist_view": {"original_key": "observation.images.wrist_view"},
            },
            "annotation": {
                "human.action.task_description": {},
                "human.validity": {},
            },
        }
    

    def _gr00t_info(self, total_frames: int) -> dict[str, Any]:
        video_info = {
            "video.fps": float(self._fps),
            "video.codec": "h264",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "has_audio": False,
        }
        image_feature = {
            "dtype": "video",
            "shape": [256, 256, 3],
            "names": ["height", "width", "channel"],
            "video_info": video_info,
        }
        return {
            "codebase_version": "v2.0",
            "robot_type": "franka_emika_panda",
            "total_episodes": 1,
            "total_frames": total_frames,
            "total_tasks": 2,
            "total_videos": 3,
            "total_chunks": 1,
            "chunks_size": 64,
            "fps": float(self._fps),
            "splits": {"train": "0:1"},
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            "features": {
                "observation.images.left_view": image_feature,
                "observation.images.right_view": image_feature,
                "observation.images.wrist_view": image_feature,
                "observation.state": {
                    "dtype": "float64",
                    "shape": [9],
                    "names": [
                        "x",
                        "y",
                        "z",
                        "qw",
                        "qx",
                        "qy",
                        "qz",
                        "gripper_qpos1",
                        "gripper_qpos2",
                    ],
                },
                "action": {
                    "dtype": "float64",
                    "shape": [7],
                    "names": ["dx", "dy", "dz", "drx", "dry", "drz", "gripper_close"],
                },
                "timestamp": {"dtype": "float64", "shape": [1]},
                "annotation.human.action.task_description": {"dtype": "int64", "shape": [1]},
                "task_index": {"dtype": "int64", "shape": [1]},
                "annotation.human.validity": {"dtype": "int64", "shape": [1]},
                "episode_index": {"dtype": "int64", "shape": [1]},
                "index": {"dtype": "int64", "shape": [1]},
                "next.reward": {"dtype": "float64", "shape": [1]},
                "next.done": {"dtype": "bool", "shape": [1]},
            },
        }
    
    def _write_gr00t_stats(self, df: pd.DataFrame, meta_dir: Path) -> None:
        stats_keys = [
            "observation.state",
            "action",
            "timestamp",
            "annotation.human.action.task_description",
            "task_index",
            "episode_index",
            "index",
            "next.reward",
        ]
        stats = {key: self._stats_for_values(df[key]) for key in stats_keys if key in df}
        with (meta_dir / "stats.json").open("w") as f:
            json.dump(stats, f, indent=4)
    
    def _stats_for_values(self, values: pd.Series) -> dict[str, Any]:
        first = values.iloc[0]
        if isinstance(first, np.ndarray):
            arr = np.stack(values.to_numpy()).astype(np.float64)
        elif isinstance(first, (list, tuple)):
            arr = np.asarray(values.to_list(), dtype=np.float64)
        else:
            arr = values.to_numpy(dtype=np.float64)

        def clean(value: np.ndarray | np.number) -> Any:
            if np.ndim(value) == 0:
                return float(value)
            return np.asarray(value).tolist()

        return {
            "mean": clean(np.mean(arr, axis=0)),
            "std": clean(np.std(arr, axis=0)),
            "min": clean(np.min(arr, axis=0)),
            "max": clean(np.max(arr, axis=0)),
            "q01": clean(np.quantile(arr, 0.01, axis=0)),
            "q99": clean(np.quantile(arr, 0.99, axis=0)),
        }
