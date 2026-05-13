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
    try:
        from lerobot_robot_fr3.tasks.factory.factory import Factory
    except ImportError as e:
        print(f"[ERROR] Cannot import Factory task: {e}")
        sys.exit(1)

    config = FR3RobotConfig()
    robot  = FR3Robot(config, task_name=task_name)
    task   = robot.task
    return task.observation_features, task.log_features


# ─── feature 헬퍼 ────────────────────────────────────────────────────────────

def features_total_dim(features: Dict[str, Tuple[int, ...]]) -> int:
    return sum(int(np.prod(s)) for s in features.values())


def feature_ylabels(key: str, shape: Tuple[int, ...]) -> List[str]:
    dim = int(np.prod(shape))
    if dim == 3:
        suffixes = ["x", "y", "z"]
    elif dim == 4:
        suffixes = ["w", "x", "y", "z"]
    else:
        suffixes = [str(i) for i in range(dim)]
    return [f"{key}_{s}" for s in suffixes]


# ─── log 패널 그룹 정의 (sim/real 공통 컬럼) ─────────────────────────────────

LOG_GROUPS = [
    {
        "title":   "ee_pos (current)",
        "cols":    ["ee_pos_0",   "ee_pos_1",   "ee_pos_2"],
        "ylabels": ["x", "y", "z"],
    },
    {
        "title":   "ee_quat (current)",
        "cols":    ["ee_quat_0",  "ee_quat_1",  "ee_quat_2",  "ee_quat_3"],
        "ylabels": ["w", "x", "y", "z"],
    },
    {
        "title":   "target_pos",
        "cols":    ["target_pos_0",  "target_pos_1",  "target_pos_2"],
        "ylabels": ["x", "y", "z"],
    },
    {
        "title":   "target_quat",
        "cols":    ["target_quat_0", "target_quat_1", "target_quat_2", "target_quat_3"],
        "ylabels": ["w", "x", "y", "z"],
    },
]
N_LOG_ROWS = sum(len(g["cols"]) for g in LOG_GROUPS)   # 3+4+3+4 = 14


# ─── obs 패널 그룹 정의 ───────────────────────────────────────────────────────

def make_obs_groups(obs_features: Dict[str, Tuple[int, ...]]) -> List[dict]:
    groups: List[dict] = []
    idx = 0
    for key, shape in obs_features.items():
        dim = int(np.prod(shape))
        groups.append({
            "title":   f"{key}  (obs {idx}–{idx + dim - 1})",
            "cols":    [f"obs_{idx + i}" for i in range(dim)],
            "ylabels": feature_ylabels(key, shape),
        })
        idx += dim
    return groups


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--task",      required=True)
    p.add_argument("--sim",       required=True)
    p.add_argument("--real",      required=True)
    p.add_argument("--save_path", default=None)
    return p.parse_args()


# ─── data loading ─────────────────────────────────────────────────────────────

def load_csv(path: str, label: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "episode" in df.columns and "step" in df.columns:
        ep = df["episode"].value_counts().idxmax()
        df = df[df["episode"] == ep].sort_values("step").reset_index(drop=True)
        print(f"[{label}] episode={ep} -> {len(df)} steps  <- {os.path.basename(path)}")
    else:
        df = df.reset_index(drop=True)
        print(f"[{label}] {len(df)} rows  <- {os.path.basename(path)}")
    return df


def safe_extract(df: pd.DataFrame, cols: List[str], label: str) -> np.ndarray:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"[{label}] missing columns: {missing}")
    return df[cols].to_numpy(dtype=np.float64)


# ─── drawing ──────────────────────────────────────────────────────────────────

def _plot_pair(ax, sim_vals, real_vals, ylabel, show_legend=False):
    n  = min(len(sim_vals), len(real_vals))
    xs = np.arange(len(sim_vals))
    xr = np.arange(len(real_vals))

    ax.plot(xs[:n], sim_vals[:n],  color="C0", lw=1.0, label="sim"  if show_legend else None)
    ax.plot(xr[:n], real_vals[:n], color="C1", lw=1.0, label="real" if show_legend else None)

    if len(sim_vals) > n:
        ax.plot(xs[n:], sim_vals[n:],  color="C0", lw=1.0, ls=":", alpha=0.5)
    if len(real_vals) > n:
        ax.plot(xr[n:], real_vals[n:], color="C1", lw=1.0, ls=":", alpha=0.5)

    ax.set_ylabel(ylabel, fontsize=7, labelpad=2)
    ax.yaxis.set_tick_params(labelsize=6)
    ax.xaxis.set_tick_params(labelsize=6)
    ax.grid(True, alpha=0.25)
    if show_legend:
        ax.legend(loc="upper right", fontsize=7)


def _draw_panel(fig, gs_cell, groups, sim_df, real_df, n_rows):
    inner = gridspec.GridSpecFromSubplotSpec(n_rows, 1,
                                             subplot_spec=gs_cell, hspace=0.10)
    row         = 0
    first_group = True

    for grp in groups:
        cols   = grp["cols"]
        labels = grp["ylabels"]
        dim    = len(cols)

        sim_data  = safe_extract(sim_df,  cols, f"sim/{grp['title']}")
        real_data = safe_extract(real_df, cols, f"real/{grp['title']}")

        axes = [fig.add_subplot(inner[row + i, 0]) for i in range(dim)]
        for ax in axes[1:]:
            ax.sharex(axes[0])
        for ax in axes[:-1]:
            plt.setp(ax.get_xticklabels(), visible=False)

        axes[0].set_title(grp["title"], fontsize=8, pad=3,
                          loc="left", color="dimgray", style="italic")
        for i, (ax, lbl) in enumerate(zip(axes, labels)):
            _plot_pair(ax, sim_data[:, i], real_data[:, i],
                       ylabel=lbl,
                       show_legend=(first_group and i == 0))
        axes[-1].set_xlabel("step", fontsize=7)

        first_group = False
        row += dim


# ─── entry point ──────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    obs_features, log_features = load_task_features(args.task)
    n_obs_rows = features_total_dim(obs_features)
    n_rows     = max(N_LOG_ROWS, n_obs_rows)

    print(f"[info] task={args.task}  log_rows={N_LOG_ROWS}  obs_rows={n_obs_rows}")

    df_sim  = load_csv(args.sim,  "sim")
    df_real = load_csv(args.real, "real")

    fig_h = max(n_rows * 0.85, 12)
    fig   = plt.figure(figsize=(18, fig_h))
    fig.suptitle(args.task, fontsize=13)

    gs = gridspec.GridSpec(
        n_rows, 2, figure=fig,
        hspace=0.08, wspace=0.38,
        left=0.07, right=0.97, top=0.95, bottom=0.04,
    )
    fig.text(0.26, 0.972, "Current EE Pose / Target EE Pose",
             ha="center", fontsize=12, fontweight="bold")
    fig.text(0.76, 0.972, "Observations",
             ha="center", fontsize=12, fontweight="bold")

    _draw_panel(fig, gs[:N_LOG_ROWS, 0], LOG_GROUPS,
                df_sim, df_real, N_LOG_ROWS)

    _draw_panel(fig, gs[:n_obs_rows, 1], make_obs_groups(obs_features),
                df_sim, df_real, n_obs_rows)

    print(f"[plot] sim={len(df_sim)} steps  real={len(df_real)} steps")

    if args.save_path:
        os.makedirs(os.path.dirname(os.path.abspath(args.save_path)) or ".", exist_ok=True)
        fig.savefig(args.save_path, dpi=150, bbox_inches="tight")
        print(f"[plot] saved -> {args.save_path}")
    else:
        plt.show()


if __name__ == "__main__":
    main()