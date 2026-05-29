from __future__ import annotations

import datetime
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import cv2
import pandas as pd

from lerobot.datasets.lerobot_dataset import LeRobotDataset


TASK_REGISTRY = {
    "forge-peg_insert": {
        "task_index": 0,
        "task": "Insert peg into the socket",
    },
    "peg_insert": {
        "task_index": 0,
        "task": "Insert peg into the socket",
    },
}

VALID_TASK = {"task_index": 1, "task": "valid"}


def _flat_feature_names(prefix: str, features: dict[str, tuple[int, ...]]) -> list[str]:
    names: list[str] = []
    for key, shape in features.items():
        dim = int(np.prod(shape))
        names.extend(f"{prefix}{key}_{i}" for i in range(dim))
    return names


def _total_dim(features: dict[str, tuple[int, ...]]) -> int:
    return sum(int(np.prod(shape)) for shape in features.values())


def _vector_feature(names: list[str]) -> dict[str, Any]:
    return {
        "dtype": "float32",
        "shape": (len(names),),
        "names": names,
    }


def _flatten_feature_values(data: dict[str, Any], features: dict[str, tuple[int, ...]]) -> np.ndarray:
    values: list[np.ndarray] = []
    for key, shape in features.items():
        dim = int(np.prod(shape))
        arr = np.asarray(data[key], dtype=np.float32).reshape(-1)
        values.append(arr[:dim])
    if not values:
        return np.zeros((0,), dtype=np.float32)
    return np.concatenate(values).astype(np.float32, copy=False)


def _squeeze_image(value: Any, shape: tuple[int, int, int]) -> np.ndarray:
    image = np.asarray(value)
    while image.ndim > 3 and image.shape[0] == 1:
        image = image[0]
    if image.ndim != 3:
        raise ValueError(f"Expected image with 3 dims after squeeze, got shape {image.shape}.")

    height, width, _ = shape
    if image.shape[:2] != (height, width):
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

    return np.ascontiguousarray(image, dtype=np.uint8)


def _camera_shapes(robot) -> dict[str, tuple[int, int, int]]:
    if getattr(robot, "camera_sensor_manager", None) is not None:
        return dict(robot.camera_sensor_manager.shapes)
    return {
        name: (cfg.height, cfg.width, 3)
        for name, cfg in getattr(robot.config, "cameras", {}).items()
    }


class LeRobotStepRecorder:
    """Append play.py samples to separate VLA-ready and full LeRobot episodes."""

    def __init__(
        self,
        robot,
        repo_id: str,
        task: str,
        fps: int,
        root: str | Path | None = None,
        full_repo_id: str | None = None,
        full_root: str | Path | None = None,
        task_text: str | None = None,
        use_videos: bool = True,
        image_writer_processes: int = 0,
        image_writer_threads_per_camera: int = 4,
        vcodec: str = "h264",
        streaming_encoding: bool = False,
        encoder_threads: int | None = None,
    ) -> None:
        self._robot = robot
        self._task_key = task
        self._task_info = self._resolve_task_info(task, task_text)
        self._task = self._task_info["task"]
        self._fps = fps
        self._frames = 0
        self._closed = False
        self._use_videos = use_videos
        self._image_writer_processes = image_writer_processes
        self._image_writer_threads_per_camera = image_writer_threads_per_camera
        self._vcodec = vcodec
        self._streaming_encoding = streaming_encoding
        self._encoder_threads = encoder_threads

        self._obs_features = robot.task.observation_features
        self._action_features = robot.task.action_features
        self._log_features = robot.task.log_features
        self._camera_shapes = _camera_shapes(robot)
        self._image_dtype = "video" if use_videos else "image"
        self._rl_state_names = _flat_feature_names("", self._obs_features)
        self._log_state_names = _flat_feature_names("log_", self._log_features)
        self._vla_state_names = [
            "eef_position_0",
            "eef_position_1",
            "eef_position_2",
            "eef_quaternion_0",
            "eef_quaternion_1",
            "eef_quaternion_2",
            "eef_quaternion_3",
            "gripper_qpos_0",
            "gripper_qpos_1",
        ]
        self._raw_action_dim = _total_dim(self._action_features)

        self._vla_dataset = self._create_dataset(
            repo_id=repo_id,
            root=self._resolve_root(root),
            features=self._build_vla_dataset_features(),
        )
        self._full_dataset = self._create_dataset(
            repo_id=full_repo_id or self._default_full_repo_id(repo_id),
            root=self._resolve_root(full_root if full_root is not None else self._default_full_root(root)),
            features=self._build_full_dataset_features(),
        )

    def _resolve_root(self, root: str | Path | None) -> Path | None:
        if root is None:
            return None

        path = Path(root)
        if not path.exists():
            return path

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        candidate = path.with_name(f"{path.name}_{timestamp}")
        suffix = 1
        while candidate.exists():
            candidate = path.with_name(f"{path.name}_{timestamp}_{suffix}")
            suffix += 1
        print(f"[INFO] LeRobot root exists, using new root: {candidate}")
        return candidate

    def _resolve_task_info(self, task_key: str, task_text: str | None) -> dict[str, Any]:
        info = dict(TASK_REGISTRY.get(task_key, {"task_index": 0, "task": task_key}))
        if task_text is not None:
            info["task"] = task_text
        return info

    def _create_dataset(
        self,
        repo_id: str,
        root: str | Path | None,
        features: dict[str, dict],
    ) -> LeRobotDataset:
        return LeRobotDataset.create(
            repo_id=repo_id,
            fps=self._fps,
            root=root,
            robot_type=self._robot.name,
            features=features,
            use_videos=self._use_videos,
            image_writer_processes=self._image_writer_processes,
            image_writer_threads=self._image_writer_threads_per_camera * len(self._camera_shapes),
            vcodec=self._vcodec,
            streaming_encoding=self._streaming_encoding,
            encoder_threads=self._encoder_threads,
        )

    def _default_full_repo_id(self, repo_id: str) -> str:
        namespace, sep, name = repo_id.rpartition("/")
        full_name = f"{name or namespace}_full"
        return f"{namespace}/{full_name}" if sep else full_name

    def _default_full_root(self, root: str | Path | None) -> Path | None:
        if root is None:
            return None
        path = Path(root)
        return path.with_name(f"{path.name}_full")

    def _image_features(self, use_view_suffix: bool) -> dict[str, dict]:
        features: dict[str, dict] = {}
        for name, shape in self._camera_shapes.items():
            feature_name = f"{name}_view" if use_view_suffix else name
            features[f"observation.images.{feature_name}"] = {
                "dtype": self._image_dtype,
                "shape": shape,
                "names": ["height", "width", "channels"],
            }
        return features

    def _build_vla_dataset_features(self) -> dict[str, dict]:
        gr00t_action_names = [f"arm_action_{i}" for i in range(6)] + ["gripper_action"]
        features: dict[str, dict] = {
            # Direct GR00T/VLA fine-tuning dataset. This intentionally mirrors
            # the simulation demo format: cameras + 9D state -> 7D RL action.
            "observation.state": _vector_feature(self._vla_state_names),
            "action": _vector_feature(gr00t_action_names),
        }
        features.update(self._image_features(use_view_suffix=True))
        return features

    def _build_full_dataset_features(self) -> dict[str, dict]:
        raw_action_names = _flat_feature_names("", self._action_features)
        processed_action_names = [f"processed_arm_action_{i}" for i in range(7)] + [
            "processed_gripper_action_0"
        ]
        gr00t_action_names = [f"arm_action_{i}" for i in range(6)] + ["gripper_action"]

        features: dict[str, dict] = {
            "observation.state": _vector_feature(self._vla_state_names),
            "observation.rl_state": _vector_feature(self._rl_state_names),
            "observation.log_state": _vector_feature(self._log_state_names),
            "action": _vector_feature(gr00t_action_names),
            "action.rl_action": _vector_feature(raw_action_names),
            "action.vla_action": _vector_feature(raw_action_names),
            "action.combined_action": _vector_feature(raw_action_names),
            "action.processed_action": _vector_feature(processed_action_names),
        }
        features.update(self._image_features(use_view_suffix=False))
        return features

    def _vla_state(self, vla_obs: dict[str, Any]) -> np.ndarray:
        parts = [
            np.asarray(vla_obs["state.eef_position"], dtype=np.float32).reshape(-1)[:3],
            np.asarray(vla_obs["state.eef_quaternion"], dtype=np.float32).reshape(-1)[:4],
            np.asarray(vla_obs["state.gripper_qpos"], dtype=np.float32).reshape(-1)[:2],
        ]
        return np.concatenate(parts).astype(np.float32, copy=False)

    def _gr00t_action(self, arm_action: np.ndarray, gripper_action: np.ndarray) -> np.ndarray:
        arm = np.asarray(arm_action, dtype=np.float32).reshape(-1)
        gripper = np.asarray(gripper_action, dtype=np.float32).reshape(-1)
        if arm.size < 6:
            raise ValueError(f"Expected at least 6 arm action values, got {arm.size}.")
        if gripper.size < 1:
            raise ValueError("Expected at least 1 gripper action value.")
        return np.concatenate([arm[:6], gripper[:1]]).astype(np.float32, copy=False)

    def _raw_action(self, action: np.ndarray) -> np.ndarray:
        raw = np.asarray(action, dtype=np.float32).reshape(-1)
        out = np.zeros((self._raw_action_dim,), dtype=np.float32)
        out[: min(out.size, raw.size)] = raw[: min(out.size, raw.size)]
        return out

    def _processed_action(self, processed_action: dict[str, Any] | None) -> np.ndarray:
        out = np.zeros((8,), dtype=np.float32)
        if processed_action is None:
            return out

        arm = processed_action.get("processed_arm_action")
        gripper = processed_action.get("processed_gripper_action")
        if arm is not None:
            arm_arr = np.asarray(arm, dtype=np.float32).reshape(-1)
            out[: min(7, arm_arr.size)] = arm_arr[: min(7, arm_arr.size)]
        if gripper is not None:
            grip_arr = np.asarray(gripper, dtype=np.float32).reshape(-1)
            if grip_arr.size > 0:
                out[7] = grip_arr[0]
        return out

    def record(
        self,
        obs_dict: dict[str, Any],
        rl_action: np.ndarray,
        vla_action: np.ndarray,
        combined_action: np.ndarray,
        gripper_action: np.ndarray,
        processed_action: dict[str, Any] | None = None,
    ) -> None:
        vla_obs = self._robot.get_vla_observation()

        vla_image_frame = {
            f"observation.images.{name}_view": _squeeze_image(
                vla_obs[f"video.{name}_view"],
                shape,
            )
            for name, shape in self._camera_shapes.items()
        }
        full_image_frame = {
            f"observation.images.{name}": image
            for name, image in (
                (name, vla_image_frame[f"observation.images.{name}_view"])
                for name in self._camera_shapes
            )
        }
        vla_frame: dict[str, Any] = {
            "observation.state": self._vla_state(vla_obs),
            "action": self._gr00t_action(rl_action, gripper_action),
            "task": self._task,
        }
        vla_frame.update(vla_image_frame)

        full_frame: dict[str, Any] = {
            "observation.state": self._vla_state(vla_obs),
            "observation.rl_state": _flatten_feature_values(obs_dict, self._obs_features),
            "observation.log_state": _flatten_feature_values(self._robot.task.get_log(), self._log_features),
            "action": self._gr00t_action(rl_action, gripper_action),
            "action.rl_action": self._raw_action(rl_action),
            "action.vla_action": self._raw_action(vla_action),
            "action.combined_action": self._raw_action(combined_action),
            "action.processed_action": self._processed_action(processed_action),
            "task": self._task,
        }
        full_frame.update(full_image_frame)

        self._vla_dataset.add_frame(vla_frame)
        self._full_dataset.add_frame(full_frame)
        self._frames += 1

    def _ask_episode_success(self) -> bool:
        while True:
            value = input("[INPUT] Episode success? 0 = fail, 1 = success: ").strip()
            if value in {"0", "1"}:
                return value == "1"
            print("[WARN] Please enter 0 or 1.")

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

    def _remove_images_dir(self, root: str | Path) -> None:
        images_dir = Path(root) / "images"
        if images_dir.exists():
            shutil.rmtree(images_dir)

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

    def save(self) -> None:
        if self._closed:
            return
        self._closed = True

        if self._frames == 0:
            print("[INFO] LeRobot recorder buffer empty - nothing to save.")
            self._vla_dataset.finalize()
            self._full_dataset.finalize()
            return

        success = self._ask_episode_success()
        self._vla_dataset.save_episode()
        self._full_dataset.save_episode()
        self._vla_dataset.finalize()
        self._full_dataset.finalize()
        self._convert_vla_dataset_to_gr00t(success)
        self._remove_images_dir(self._full_dataset.root)
        print(f"[INFO] Saved {self._frames} VLA frames to LeRobot dataset: {self._vla_dataset.root}")
        print(f"[INFO] Saved {self._frames} full frames to LeRobot dataset: {self._full_dataset.root}")
