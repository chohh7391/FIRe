"""sim2real comparison of ee_pos/quat, target_pos/quat, and action trajectories.

Both CSVs share the same column names for ee and target pose, so no coordinate
transform is needed — we load and plot directly.

Sim CSV (from forge_env.py logging):
  episode, step,
  ee_pos_x/y/z, ee_quat_w/x/y/z,
  target_pos_x/y/z, target_quat_w/x/y/z,
  raw_action_0~6, ...

Real CSV (from play.py with --save_path):
  ee_pos_x/y/z, ee_quat_w/x/y/z,
  target_pos_x/y/z, target_quat_w/x/y/z,
  prev_actions_0~6, ...  (one episode per file)

Layout: 3 column grid
  Left   column : ee_pos_x/y/z + ee_quat_w/x/y/z        (7 rows)
  Center column : target_pos_x/y/z + target_quat_w/x/y/z (7 rows)
  Right  column : action_0~6                              (7 rows)

Usage:
  python plot_compare.py --sim forge_data.csv --real collected_data.csv
  python plot_compare.py --sim forge_data.csv --real collected_data.csv \\
      --save_path compare.png --title "hole task sim2real"
"""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd


EE_POS_COLS   = ["ee_pos_x",     "ee_pos_y",     "ee_pos_z"]
EE_QUAT_COLS  = ["ee_quat_w",    "ee_quat_x",    "ee_quat_y",    "ee_quat_z"]
TGT_POS_COLS  = ["target_pos_x", "target_pos_y", "target_pos_z"]
TGT_QUAT_COLS = ["target_quat_w","target_quat_x","target_quat_y","target_quat_z"]

SIM_ACTION_COLS  = [f"raw_action_{i}"   for i in range(7)]
REAL_ACTION_COLS = [f"prev_actions_{i}" for i in range(7)]

POS_YLABELS    = ["pos_x [m]", "pos_y [m]", "pos_z [m]"]
QUAT_YLABELS   = ["quat_w",    "quat_x",    "quat_y",    "quat_z"]
ACTION_YLABELS = [f"action_{i}" for i in range(7)]


def parse_args():
    p = argparse.ArgumentParser(
        description="sim2real comparison: ee_pos/quat, target_pos/quat, and actions."
    )
    p.add_argument("--sim",  type=str, required=True, help="forge simulation CSV path")
    p.add_argument("--real", type=str, required=True, help="real robot CSV path (from play.py)")
    p.add_argument(
        "--save_path",
        type=str,
        default=None,
        help="Output figure path (e.g. compare.png). "
             "If omitted, plt.show() is called interactively.",
    )
    p.add_argument("--title", type=str, default=None, help="Optional figure suptitle.")
    return p.parse_args()


def load_sim_first_episode(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "episode" not in df.columns:
        raise KeyError(f"'episode' column not found in {path}")
    if "step" not in df.columns:
        raise KeyError(f"'step' column not found in {path}")

    first_ep = df["episode"].iloc[0]
    df_ep = df[df["episode"] == first_ep].sort_values("step").reset_index(drop=True)
    print(f"[sim]  episode={first_ep}  → {len(df_ep)} steps")
    return df_ep


def load_real(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"[real] {len(df)} steps  ← {os.path.basename(path)}")
    return df


def extract_cols(df: pd.DataFrame, cols: list[str], label: str) -> np.ndarray:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"[{label}] missing columns: {missing}")
    return df[cols].to_numpy(dtype=np.float64)


def draw_column(
    axes: list,
    sim_data: np.ndarray,
    real_data: np.ndarray,
    ylabels: list[str],
    col_title: str,
    show_legend: bool,
):
    """Fill one figure column with subplots.
    sim_data and real_data must have shape (T, len(ylabels)).
    """
    n_sim  = len(sim_data)
    n_real = len(real_data)
    n_min  = min(n_sim, n_real)
    sim_x  = np.arange(n_sim)
    real_x = np.arange(n_real)

    axes[0].set_title(col_title, fontsize=11, pad=6)

    for i, (ax, label) in enumerate(zip(axes, ylabels)):
        # Overlap region: solid; extra tail: dashed, same color.
        ax.plot(sim_x[:n_min],  sim_data[:n_min, i],
                color="C0", linestyle="-",
                label="sim (forge)" if (show_legend and i == 0) else None)
        if n_sim > n_min:
            ax.plot(sim_x[n_min:], sim_data[n_min:, i],
                    color="C0", linestyle=":", alpha=0.6)

        ax.plot(real_x[:n_min], real_data[:n_min, i],
                color="C1", linestyle="-",
                label="real (FR3)" if (show_legend and i == 0) else None)
        if n_real > n_min:
            ax.plot(real_x[n_min:], real_data[n_min:, i],
                    color="C1", linestyle=":", alpha=0.6)

        ax.set_ylabel(label, fontsize=9)
        ax.grid(True, alpha=0.3)
        if show_legend and i == 0:
            ax.legend(loc="upper right", fontsize=8)


def plot_compare(
    sim_ee_pos: np.ndarray,    sim_ee_quat: np.ndarray,
    sim_tgt_pos: np.ndarray,   sim_tgt_quat: np.ndarray,
    sim_actions: np.ndarray,
    real_ee_pos: np.ndarray,   real_ee_quat: np.ndarray,
    real_tgt_pos: np.ndarray,  real_tgt_quat: np.ndarray,
    real_actions: np.ndarray,
    save_path: str | None,
    title: str | None,
):
    n_rows = 7  # 3 pos + 4 quat  (action column also has 7 dims)
    fig = plt.figure(figsize=(21, 16))
    if title:
        fig.suptitle(title, fontsize=13)

    gs = gridspec.GridSpec(n_rows, 3, figure=fig, hspace=0.08, wspace=0.35)

    ee_axes  = [fig.add_subplot(gs[r, 0]) for r in range(n_rows)]
    tgt_axes = [fig.add_subplot(gs[r, 1]) for r in range(n_rows)]
    act_axes = [fig.add_subplot(gs[r, 2]) for r in range(n_rows)]

    # Share x-axis within each column.
    for ax in ee_axes[1:]:
        ax.sharex(ee_axes[0])
    for ax in tgt_axes[1:]:
        ax.sharex(tgt_axes[0])
    for ax in act_axes[1:]:
        ax.sharex(act_axes[0])

    # Hide x tick labels except on the bottom row.
    for col_axes in (ee_axes, tgt_axes, act_axes):
        for ax in col_axes[:-1]:
            plt.setp(ax.get_xticklabels(), visible=False)

    ylabels_pose = POS_YLABELS + QUAT_YLABELS  # 7 labels

    draw_column(
        axes=ee_axes,
        sim_data=np.hstack([sim_ee_pos,  sim_ee_quat]),
        real_data=np.hstack([real_ee_pos, real_ee_quat]),
        ylabels=ylabels_pose,
        col_title="EE pose (actual)",
        show_legend=True,
    )
    draw_column(
        axes=tgt_axes,
        sim_data=np.hstack([sim_tgt_pos,  sim_tgt_quat]),
        real_data=np.hstack([real_tgt_pos, real_tgt_quat]),
        ylabels=ylabels_pose,
        col_title="Target pose (commanded)",
        show_legend=True,
    )
    draw_column(
        axes=act_axes,
        sim_data=sim_actions,
        real_data=real_actions,
        ylabels=ACTION_YLABELS,
        col_title="Actions (sim: raw_action / real: prev_actions)",
        show_legend=True,
    )

    for col_axes in (ee_axes, tgt_axes, act_axes):
        col_axes[-1].set_xlabel("step", fontsize=9)

    print(
        f"[plot]  ee     — sim={len(sim_ee_pos)} steps, real={len(real_ee_pos)} steps\n"
        f"[plot]  tgt    — sim={len(sim_tgt_pos)} steps, real={len(real_tgt_pos)} steps\n"
        f"[plot]  action — sim={len(sim_actions)} steps, real={len(real_actions)} steps"
    )

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[plot]  saved → {save_path}")
    else:
        plt.show()


def main():
    args = parse_args()

    df_sim  = load_sim_first_episode(args.sim)
    df_real = load_real(args.real)

    sim_ee_pos   = extract_cols(df_sim,  EE_POS_COLS,      "sim/ee_pos")
    sim_ee_quat  = extract_cols(df_sim,  EE_QUAT_COLS,     "sim/ee_quat")
    sim_tgt_pos  = extract_cols(df_sim,  TGT_POS_COLS,     "sim/target_pos")
    sim_tgt_quat = extract_cols(df_sim,  TGT_QUAT_COLS,    "sim/target_quat")
    sim_actions  = extract_cols(df_sim,  SIM_ACTION_COLS,  "sim/raw_action")

    real_ee_pos   = extract_cols(df_real, EE_POS_COLS,      "real/ee_pos")
    real_ee_quat  = extract_cols(df_real, EE_QUAT_COLS,     "real/ee_quat")
    real_tgt_pos  = extract_cols(df_real, TGT_POS_COLS,     "real/target_pos")
    real_tgt_quat = extract_cols(df_real, TGT_QUAT_COLS,    "real/target_quat")
    real_actions  = extract_cols(df_real, REAL_ACTION_COLS, "real/prev_actions")

    plot_compare(
        sim_ee_pos=sim_ee_pos,    sim_ee_quat=sim_ee_quat,
        sim_tgt_pos=sim_tgt_pos,  sim_tgt_quat=sim_tgt_quat,
        sim_actions=sim_actions,
        real_ee_pos=real_ee_pos,  real_ee_quat=real_ee_quat,
        real_tgt_pos=real_tgt_pos, real_tgt_quat=real_tgt_quat,
        real_actions=real_actions,
        save_path=args.save_path,
        title=args.title,
    )


if __name__ == "__main__":
    main()