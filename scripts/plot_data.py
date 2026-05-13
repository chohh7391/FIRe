from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

from lerobot_robot_fr3.config_fr3 import FR3RobotConfig
from lerobot_robot_fr3.fr3 import FR3Robot


# ─── task feature loading ─────────────────────────────────────────────────────

def load_task_features(task_name: str) -> Tuple[Dict, Dict]:
    """Factory task를 인스턴스화해서 observation_features, log_features를 반환.
    robot.connect() 없이 __init__만으로 접근 가능."""
    try:
        from lerobot_robot_fr3.tasks.factory.factory import Factory
    except ImportError as e:
        print(f"[ERROR] Cannot import Factory task: {e}")
        sys.exit(1)
    
    config = FR3RobotConfig()
    robot = FR3Robot(config, task_name=task_name)
    task = robot.task
    return task.observation_features, task.log_features


# ─── sim CSV 컬럼명 정규화 ────────────────────────────────────────────────────
# sim CSV: x/y/z, w/x/y/z 네이밍 → _0/_1/_2, _0/_1/_2/_3 인덱스 네이밍

_AXIS_TO_IDX = {"x": 0, "y": 1, "z": 2}
_QUAT_TO_IDX = {"w": 0, "x": 1, "y": 2, "z": 3}


def _normalize_sim_df(df: pd.DataFrame) -> pd.DataFrame:
    """sim CSV의 x/y/z, w/x/y/z 컬럼 suffix를 _0/_1/... 인덱스로 rename."""
    import re
    rename_map: Dict[str, str] = {}
    for col in df.columns:
        m = re.match(r"^(.+_quat)_([wxyz])$", col)
        if m:
            rename_map[col] = f"{m.group(1)}_{_QUAT_TO_IDX[m.group(2)]}"
            continue
        m = re.match(r"^(.+)_([xyz])$", col)
        if m:
            rename_map[col] = f"{m.group(1)}_{_AXIS_TO_IDX[m.group(2)]}"
    return df.rename(columns=rename_map)


# ─── feature → 컬럼명 / y축 라벨 헬퍼 ───────────────────────────────────────

def features_total_dim(features: Dict[str, Tuple[int, ...]]) -> int:
    return sum(int(np.prod(s)) for s in features.values())


def feature_flat_cols(key: str, shape: Tuple[int, ...]) -> List[str]:
    """단일 feature 키를 평탄화한 컬럼명 리스트.  예) ee_pos → [ee_pos_0, ee_pos_1, ee_pos_2]"""
    return [f"{key}_{i}" for i in range(int(np.prod(shape)))]


def feature_ylabels(key: str, shape: Tuple[int, ...]) -> List[str]:
    """dim에 따라 x/y/z 또는 w/x/y/z suffix를 붙인 y축 라벨."""
    dim = int(np.prod(shape))
    if dim == 3:
        suffixes = ["x", "y", "z"]
    elif dim == 4:
        suffixes = ["w", "x", "y", "z"]
    else:
        suffixes = [str(i) for i in range(dim)]
    return [f"{key}_{s}" for s in suffixes]


def make_obs_groups(obs_features: Dict[str, Tuple[int, ...]]) -> List[dict]:
    """observation_features로부터 obs_N 기반 서브그룹 정의를 생성."""
    groups: List[dict] = []
    idx = 0
    for key, shape in obs_features.items():
        dim = int(np.prod(shape))
        start, end = idx, idx + dim - 1
        groups.append({
            "title":    f"{key}  (obs {start}–{end})",
            "obs_cols": [f"obs_{idx + i}" for i in range(dim)],
            "ylabels":  feature_ylabels(key, shape),
        })
        idx += dim
    return groups


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="sim2real comparison driven by Factory task features."
    )
    p.add_argument("--task", type=str, required=True,
                   help="Factory task name: peg_insert | gear_mesh | nut_thread")
    p.add_argument("--sim",  type=str, required=True,
                   help="Sim CSV path (forge_env.py logging)")
    p.add_argument("--real", type=str, required=True,
                   help="Real CSV path (play.py --save_path)")
    p.add_argument("--save_path", type=str, default=None,
                   help="Output PNG path. Omit for interactive plt.show().")
    return p.parse_args()


# ─── data loading ─────────────────────────────────────────────────────────────

def load_sim(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "episode" in df.columns and "step" in df.columns:
        if df["episode"].nunique() == len(df):
            df = df.sort_values("episode").reset_index(drop=True)
            print(f"[sim]  episode col treated as step index → {len(df)} rows")
        else:
            first_ep = df["episode"].iloc[0]
            df = df[df["episode"] == first_ep].sort_values("step").reset_index(drop=True)
            print(f"[sim]  episode={first_ep} → {len(df)} steps")
    else:
        print(f"[sim]  no episode/step cols → {len(df)} rows")
    df = _normalize_sim_df(df)
    print(f"[sim]  loaded {len(df)} steps ← {os.path.basename(path)}")
    return df


def load_real(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"[real] {len(df)} steps ← {os.path.basename(path)}")
    return df


def safe_extract(df: pd.DataFrame, cols: List[str], label: str) -> np.ndarray:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"[{label}] missing columns: {missing}")
    return df[cols].to_numpy(dtype=np.float64)


# ─── drawing primitives ───────────────────────────────────────────────────────

def _plot_pair(ax, sim_vals, real_vals, ylabel, show_legend=False):
    """ax 하나에 sim/real 한 채널을 오버레이."""
    n_min = min(len(sim_vals), len(real_vals))
    sim_x  = np.arange(len(sim_vals))
    real_x = np.arange(len(real_vals))

    ax.plot(sim_x[:n_min],  sim_vals[:n_min],  color="C0", lw=1.0,
            label="sim" if show_legend else None)
    if len(sim_vals) > n_min:
        ax.plot(sim_x[n_min:], sim_vals[n_min:], color="C0", lw=1.0,
                linestyle=":", alpha=0.5)

    ax.plot(real_x[:n_min], real_vals[:n_min], color="C1", lw=1.0,
            label="real" if show_legend else None)
    if len(real_vals) > n_min:
        ax.plot(real_x[n_min:], real_vals[n_min:], color="C1", lw=1.0,
                linestyle=":", alpha=0.5)

    ax.set_ylabel(ylabel, fontsize=7, labelpad=2)
    ax.yaxis.set_tick_params(labelsize=6)
    ax.xaxis.set_tick_params(labelsize=6)
    ax.grid(True, alpha=0.25)
    if show_legend:
        ax.legend(loc="upper right", fontsize=7)


def _draw_feature_panel(
    fig,
    gs_cell,
    groups: List[dict],        # [{title, cols, ylabels}, ...]
    sim_df: pd.DataFrame,
    real_df: pd.DataFrame,
    n_total_rows: int,
):
    """공통 패널 렌더러: groups 정의를 받아 서브그룹별로 그린다."""
    inner_gs = gridspec.GridSpecFromSubplotSpec(
        n_total_rows, 1,
        subplot_spec=gs_cell,
        hspace=0.10,
    )

    row_cursor  = 0
    first_group = True

    for grp in groups:
        cols    = grp["cols"]
        labels  = grp["ylabels"]
        dim     = len(cols)

        sim_data  = safe_extract(sim_df,  cols, f"sim/{grp['title']}")
        real_data = safe_extract(real_df, cols, f"real/{grp['title']}")

        axes = [fig.add_subplot(inner_gs[row_cursor + i, 0]) for i in range(dim)]
        for ax in axes[1:]:
            ax.sharex(axes[0])
        for ax in axes[:-1]:
            plt.setp(ax.get_xticklabels(), visible=False)

        axes[0].set_title(grp["title"], fontsize=8, pad=3,
                          loc="left", color="dimgray", style="italic")

        for i, (ax, ylabel) in enumerate(zip(axes, labels)):
            _plot_pair(
                ax,
                sim_data[:, i], real_data[:, i],
                ylabel=ylabel,
                show_legend=(first_group and i == 0),
            )
        if first_group:
            first_group = False

        axes[-1].set_xlabel("step", fontsize=7)
        row_cursor += dim


# ─── panel builders ───────────────────────────────────────────────────────────

def draw_log_panel(fig, gs_cell,
                   log_features: Dict[str, Tuple[int, ...]],
                   sim_df: pd.DataFrame, real_df: pd.DataFrame):
    """왼쪽 패널: log_features 각 키를 서브그룹으로 그린다."""
    groups = [
        {
            "title":   key,
            "cols":    feature_flat_cols(key, shape),
            "ylabels": feature_ylabels(key, shape),
        }
        for key, shape in log_features.items()
    ]
    _draw_feature_panel(fig, gs_cell, groups, sim_df, real_df,
                        n_total_rows=features_total_dim(log_features))


def draw_obs_panel(fig, gs_cell,
                   obs_features: Dict[str, Tuple[int, ...]],
                   sim_df: pd.DataFrame, real_df: pd.DataFrame):
    """오른쪽 패널: obs_N 컬럼 기반으로 observation_features 서브그룹을 그린다."""
    groups = [
        {
            "title":   grp["title"],
            "cols":    grp["obs_cols"],
            "ylabels": grp["ylabels"],
        }
        for grp in make_obs_groups(obs_features)
    ]
    _draw_feature_panel(fig, gs_cell, groups, sim_df, real_df,
                        n_total_rows=features_total_dim(obs_features))


# ─── entry point ──────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    obs_features, log_features = load_task_features(args.task)

    n_log_rows = features_total_dim(log_features)
    n_obs_rows = features_total_dim(obs_features)
    n_rows     = max(n_log_rows, n_obs_rows)

    print(f"[info] task={args.task}  log_rows={n_log_rows}  obs_rows={n_obs_rows}")

    df_sim  = load_sim(args.sim)
    df_real = load_real(args.real)

    # ── figure ────────────────────────────────────────────────────────────
    fig_h = max(n_rows * 0.85, 12)
    fig = plt.figure(figsize=(18, fig_h))
    fig.suptitle(args.task, fontsize=13)

    gs = gridspec.GridSpec(
        n_rows, 2,
        figure=fig,
        hspace=0.08, wspace=0.38,
        left=0.07, right=0.97,
        top=0.95,   bottom=0.04,
    )

    fig.text(0.26, 0.972, "Current EE Pose / Target EE Pose",
             ha="center", fontsize=12, fontweight="bold")
    fig.text(0.76, 0.972, "Observations",
             ha="center", fontsize=12, fontweight="bold")

    draw_log_panel(fig, gs[:n_log_rows, 0], log_features, df_sim, df_real)
    draw_obs_panel(fig, gs[:n_obs_rows, 1], obs_features, df_sim, df_real)

    print(f"[plot] sim={len(df_sim)} steps  real={len(df_real)} steps")

    if args.save_path:
        os.makedirs(os.path.dirname(os.path.abspath(args.save_path)) or ".", exist_ok=True)
        fig.savefig(args.save_path, dpi=150, bbox_inches="tight")
        print(f"[plot] saved → {args.save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    main()