from __future__ import annotations

import datetime
from pathlib import Path

import numpy as np

from .base_recorder import BaseRecorder
from .gr00t_exporter import GR00TExporter
from .utils import vector_feature


class GR00TRecorder(BaseRecorder):
    """Record one episode through LeRobot, then export it to the GR00T v2 layout."""

    def _resolve_recording_root(self, dataset_root: Path, resume: bool) -> Path:
        self._resume_target_root = dataset_root
        if resume and not dataset_root.exists():
            raise FileNotFoundError(f"Cannot resume GR00T dataset because root does not exist: {dataset_root}")

        self._uses_staging_root = bool(resume)
        if not self._uses_staging_root:
            return dataset_root

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        staging_root = dataset_root.with_name(f"{dataset_root.name}_staging_{timestamp}")
        suffix = 1
        while staging_root.exists():
            staging_root = dataset_root.with_name(f"{dataset_root.name}_staging_{timestamp}_{suffix}")
            suffix += 1
        print(f"[INFO] Resuming GR00T dataset. Recording new episode in staging root: {staging_root}")
        return staging_root

    def _build_vla_dataset_features(self) -> dict[str, dict]:
        gr00t_action_names = [f"arm_action_{i}" for i in range(6)] + ["gripper_action"]
        features = {
            "observation.state": vector_feature(self._vla_state_names),
            "action": vector_feature(gr00t_action_names),
        }
        features.update(self._image_features(use_view_suffix=True))
        return features

    def _format_action(self, arm_action: np.ndarray, gripper_action: np.ndarray) -> np.ndarray:
        arm = np.asarray(arm_action, dtype=np.float32).reshape(-1)
        gripper = np.asarray(gripper_action, dtype=np.float32).reshape(-1)
        return np.concatenate([arm[:6], gripper[:1]]).astype(np.float32, copy=False)

    def _post_save_processing(self, success: bool) -> None:
        exporter = GR00TExporter(
            task_info=self._task_info,
            fps=self._fps,
            camera_shapes=self.camera_shapes,
        )
        exporter.convert_episode(Path(self._vla_dataset.root), success)

        if self._uses_staging_root and self._resume_target_root is not None:
            episode_index = exporter.append_staging_episode(
                staging_root=Path(self._vla_dataset.root),
                target_root=self._resume_target_root,
            )
            print(f"[INFO] Appended GR00T episode {episode_index:06d} to dataset: {self._resume_target_root}")

        self._remove_images_dir(self._dataset_root or self._vla_dataset.root)
