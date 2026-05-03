"""sim2real comparison of ee_pos/quat, target_pos/quat, action, and obs trajectories.

Sim CSV (from forge_env.py logging):
  episode, step,
  raw_action_0~5,
  target_pos_x/y/z, target_quat_w/x/y/z,
  ee_pos_x/y/z, ee_quat_w/x/y/z,
  fixed_pos/quat, held_pos/quat,
  obs_0~18

Real CSV (from play.py with --save_path):
  fingertip_pos_rel_fixed_x/y/z,  (→ obs_0~2)
  fingertip_quat_w/x/y/z,         (→ obs_3~6)
  ee_linvel_x/y/z,                 (→ obs_7~9)
  ee_angvel_x/y/z,                 (→ obs_10~12)
  prev_actions_0~5,                (→ obs_13~18)
  ee_pos_x/y/z, ee_quat_w/x/y/z,
  target_pos_x/y/z, target_quat_w/x/y/z

Obs mapping (19 signals):
  obs_0~2  : fingertip_pos_rel_fixed
  obs_3~6  : fingertip_quat
  obs_7~9  : ee_linvel
  obs_10~12: ee_angvel
  obs_13~18: prev_actions (6 DoF)

Layout: 4 column grid
  Col 0 : EE pose        — ee_pos_x/y/z + ee_quat_w/x/y/z         (7 rows)
  Col 1 : Target pose    — target_pos_x/y/z + target_quat_w/x/y/z  (7 rows)
  Col 2 : Actions 0~5                                               (6 rows)
  Col 3 : Observations   — 5 sub-groups, 19 rows total

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


# ─── column definitions ───────────────────────────────────────────────────────
EE_POS_COLS   = ["ee_pos_x",     "ee_pos_y",     "ee_pos_z"]
EE_QUAT_COLS  = ["ee_quat_w",    "ee_quat_x",    "ee_quat_y",    "ee_quat_z"]
TGT_POS_COLS  = ["target_pos_x", "target_pos_y", "target_pos_z"]
TGT_QUAT_COLS = ["target_quat_w","target_quat_x","target_quat_y","target_quat_z"]

SIM_ACTION_COLS  = [f"raw_action_{i}"   for i in range(6)]
REAL_ACTION_COLS = [f"prev_actions_{i}" for i in range(6)]

POS_YLABELS    = ["pos_x [m]", "pos_y [m]", "pos_z [m]"]
QUAT_YLABELS   = ["quat_w",    "quat_x",    "quat_y",    "quat_z"]
ACTION_YLABELS = [f"action_{i}" for i in range(6)]

N_POSE_ROWS   = 7   # 3 pos + 4 quat
N_ACTION_ROWS = 6   # 6 actions

# ─── obs sub-group definitions ────────────────────────────────────────────────
# Each entry:
#   sim_idx   : list of obs_N indices to pull from sim CSV
#   real_cols : real CSV column names (same order as sim_idx)
#   ylabels   : y-axis labels per signal
#   title     : group header shown on the first subplot

OBS_GROUPS = [
    {
        "title":     "fingertip_pos_rel_fixed  (obs 0–2)",
        "sim_idx":   [0, 1, 2],
        "real_cols": ["fingertip_pos_rel_fixed_x",
                      "fingertip_pos_rel_fixed_y",
                      "fingertip_pos_rel_fixed_z"],
        "ylabels":   ["fp_rel_x [m]", "fp_rel_y [m]", "fp_rel_z [m]"],
    },
    {
        "title":     "fingertip_quat  (obs 3–6)",
        "sim_idx":   [3, 4, 5, 6],
        "real_cols": ["fingertip_quat_w", "fingertip_quat_x",
                      "fingertip_quat_y", "fingertip_quat_z"],
        "ylabels":   ["fq_w", "fq_x", "fq_y", "fq_z"],
    },
    {
        "title":     "ee_linvel  (obs 7–9)",
        "sim_idx":   [7, 8, 9],
        "real_cols": ["ee_linvel_x", "ee_linvel_y", "ee_linvel_z"],
        "ylabels":   ["lv_x [m/s]", "lv_y [m/s]", "lv_z [m/s]"],
    },
    {
        "title":     "ee_angvel  (obs 10–12)",
        "sim_idx":   [10, 11, 12],
        "real_cols": ["ee_angvel_x", "ee_angvel_y", "ee_angvel_z"],
        "ylabels":   ["av_x [r/s]", "av_y [r/s]", "av_z [r/s]"],
    },
    {
        "title":     "prev_actions  (obs 13–18)",
        "sim_idx":   [13, 14, 15, 16, 17, 18],
        "real_cols": [f"prev_actions_{i}" for i in range(6)],
        "ylabels":   [f"prev_a{i}" for i in range(6)],
    },
]

N_OBS_ROWS = sum(len(g["ylabels"]) for g in OBS_GROUPS)   # 3+4+3+3+6 = 19
N_GRID_ROWS = max(N_POSE_ROWS, N_ACTION_ROWS, N_OBS_ROWS)  # 19


# ─── CLI ──────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="sim2real comparison: ee_pos/quat, target_pos/quat, actions, and obs."
    )
    p.add_argument("--sim",  type=str, required=True, help="factory simulation CSV path")
    p.add_argument("--real", type=str, required=True, help="real robot CSV path (from play.py)")
    p.add_argument("--save_path", type=str, default=None,
                   help="Output figure path (e.g. compare.png). "
                        "If omitted, plt.show() is called interactively.")
    p.add_argument("--title", type=str, default=None, help="Optional figure suptitle.")
    return p.parse_args()


# ─── data loading ─────────────────────────────────────────────────────────────
def load_sim(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "episode" in df.columns and "step" in df.columns:
        if df["episode"].nunique() == len(df):
            print(f"[sim]  'episode' col is actually step index → using all {len(df)} rows as-is")
            df = df.sort_values("episode").reset_index(drop=True)
        else:
            first_ep = df["episode"].iloc[0]
            df = df[df["episode"] == first_ep].sort_values("step").reset_index(drop=True)
            print(f"[sim]  episode={first_ep}  → {len(df)} steps")
    else:
        print(f"[sim]  no episode/step columns → using all {len(df)} rows")
    print(f"[sim]  loaded {len(df)} steps  ← {os.path.basename(path)}")
    return df


def load_real(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"[real] {len(df)} steps  ← {os.path.basename(path)}")
    return df


def extract_cols(df: pd.DataFrame, cols: list[str], label: str) -> np.ndarray:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"[{label}] missing columns: {missing}")
    return df[cols].to_numpy(dtype=np.float64)


# ─── drawing helpers ──────────────────────────────────────────────────────────
def draw_column(
    axes: list,
    sim_data: np.ndarray,
    real_data: np.ndarray,
    ylabels: list[str],
    col_title: str,
    show_legend: bool,
):
    """Fill one figure column with overlaid sim/real lines."""
    n_sim  = len(sim_data)
    n_real = len(real_data)
    n_min  = min(n_sim, n_real)
    sim_x  = np.arange(n_sim)
    real_x = np.arange(n_real)

    axes[0].set_title(col_title, fontsize=11, pad=6)

    for i, (ax, label) in enumerate(zip(axes, ylabels)):
        ax.plot(sim_x[:n_min], sim_data[:n_min, i],
                color="C0", linestyle="-",
                label="sim (factory)" if (show_legend and i == 0) else None)
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


def draw_obs_column(
    fig,
    outer_gs_cell,
    sim_df: pd.DataFrame,
    real_df: pd.DataFrame,
):
    """Draw all obs sub-groups inside a single outer GridSpec cell."""
    n_sim  = len(sim_df)
    n_real = len(real_df)

    inner_gs = gridspec.GridSpecFromSubplotSpec(
        N_OBS_ROWS, 1,
        subplot_spec=outer_gs_cell,
        hspace=0.12,
    )

    row_cursor  = 0
    first_group = True

    for grp in OBS_GROUPS:
        sim_cols  = [f"obs_{i}" for i in grp["sim_idx"]]
        real_cols = grp["real_cols"]
        ylabels   = grp["ylabels"]
        n_ch      = len(ylabels)

        sim_data  = sim_df[sim_cols].to_numpy(dtype=np.float64)
        real_data = real_df[real_cols].to_numpy(dtype=np.float64)

        n_min  = min(n_sim, n_real)
        sim_x  = np.arange(n_sim)
        real_x = np.arange(n_real)

        axes = [fig.add_subplot(inner_gs[row_cursor + r, 0]) for r in range(n_ch)]

        # share x within group, hide inner tick labels
        for ax in axes[1:]:
            ax.sharex(axes[0])
        for ax in axes[:-1]:
            plt.setp(ax.get_xticklabels(), visible=False)

        # italic group title on the first subplot of each group
        axes[0].set_title(grp["title"], fontsize=8, pad=3,
                          loc="left", color="dimgray", style="italic")

        for i, (ax, ylabel) in enumerate(zip(axes, ylabels)):
            ax.plot(sim_x[:n_min], sim_data[:n_min, i],
                    color="C0", lw=0.9,
                    label="sim (factory)" if (first_group and i == 0) else None)
            if n_sim > n_min:
                ax.plot(sim_x[n_min:], sim_data[n_min:, i],
                        color="C0", lw=0.9, linestyle=":", alpha=0.6)

            ax.plot(real_x[:n_min], real_data[:n_min, i],
                    color="C1", lw=0.9,
                    label="real (FR3)" if (first_group and i == 0) else None)
            if n_real > n_min:
                ax.plot(real_x[n_min:], real_data[n_min:, i],
                        color="C1", lw=0.9, linestyle=":", alpha=0.6)

            ax.set_ylabel(ylabel, fontsize=7, labelpad=2)
            ax.yaxis.set_tick_params(labelsize=6)
            ax.xaxis.set_tick_params(labelsize=6)
            ax.grid(True, alpha=0.25)

            if first_group and i == 0:
                ax.legend(loc="upper right", fontsize=7)
                first_group = False

        axes[-1].set_xlabel("step", fontsize=7)
        row_cursor += n_ch


# ─── main plotting ────────────────────────────────────────────────────────────
def plot_compare(
    sim_ee_pos: np.ndarray,   sim_ee_quat: np.ndarray,
    sim_tgt_pos: np.ndarray,  sim_tgt_quat: np.ndarray,
    sim_actions: np.ndarray,
    real_ee_pos: np.ndarray,  real_ee_quat: np.ndarray,
    real_tgt_pos: np.ndarray, real_tgt_quat: np.ndarray,
    real_actions: np.ndarray,
    sim_df: pd.DataFrame,
    real_df: pd.DataFrame,
    save_path: str | None,
    title: str | None,
):
    fig = plt.figure(figsize=(26, 18))
    if title:
        fig.suptitle(title, fontsize=13, y=0.998)

    # 4 columns; row count driven by obs (19)
    gs = gridspec.GridSpec(
        N_GRID_ROWS, 4,
        figure=fig,
        hspace=0.08, wspace=0.40,
        left=0.06, right=0.97,
        top=0.97,   bottom=0.04,
    )

    # ── Col 0: EE pose ────────────────────────────────────────────────────
    ee_axes = [fig.add_subplot(gs[r, 0]) for r in range(N_POSE_ROWS)]
    for ax in ee_axes[1:]:  ax.sharex(ee_axes[0])
    for ax in ee_axes[:-1]: plt.setp(ax.get_xticklabels(), visible=False)
    draw_column(
        axes=ee_axes,
        sim_data=np.hstack([sim_ee_pos,  sim_ee_quat]),
        real_data=np.hstack([real_ee_pos, real_ee_quat]),
        ylabels=POS_YLABELS + QUAT_YLABELS,
        col_title="EE pose (actual)",
        show_legend=True,
    )
    ee_axes[-1].set_xlabel("step", fontsize=9)

    # ── Col 1: Target pose ────────────────────────────────────────────────
    tgt_axes = [fig.add_subplot(gs[r, 1]) for r in range(N_POSE_ROWS)]
    for ax in tgt_axes[1:]:  ax.sharex(tgt_axes[0])
    for ax in tgt_axes[:-1]: plt.setp(ax.get_xticklabels(), visible=False)
    draw_column(
        axes=tgt_axes,
        sim_data=np.hstack([sim_tgt_pos,  sim_tgt_quat]),
        real_data=np.hstack([real_tgt_pos, real_tgt_quat]),
        ylabels=POS_YLABELS + QUAT_YLABELS,
        col_title="Target pose (commanded)",
        show_legend=True,
    )
    tgt_axes[-1].set_xlabel("step", fontsize=9)

    # ── Col 2: Actions ────────────────────────────────────────────────────
    act_axes = [fig.add_subplot(gs[r, 2]) for r in range(N_ACTION_ROWS)]
    for ax in act_axes[1:]:  ax.sharex(act_axes[0])
    for ax in act_axes[:-1]: plt.setp(ax.get_xticklabels(), visible=False)
    draw_column(
        axes=act_axes,
        sim_data=sim_actions,
        real_data=real_actions,
        ylabels=ACTION_YLABELS,
        col_title="Actions (sim: raw_action / real: prev_actions)",
        show_legend=True,
    )
    act_axes[-1].set_xlabel("step", fontsize=9)

    # ── Col 3: Observations ───────────────────────────────────────────────
    draw_obs_column(fig, gs[:N_OBS_ROWS, 3], sim_df, real_df)

    # column header for obs (sits above the nested GridSpec)
    obs_header = fig.add_axes([0.0, 0.0, 0.0, 0.0])  # invisible placeholder
    fig.text(
        x=0.967, y=0.985,
        s="Observations (obs 0–18)",
        ha="right", va="top",
        fontsize=11, fontweight="bold",
    )

    print(
        f"[plot]  ee     — sim={len(sim_ee_pos)} steps, real={len(real_ee_pos)} steps\n"
        f"[plot]  tgt    — sim={len(sim_tgt_pos)} steps, real={len(real_tgt_pos)} steps\n"
        f"[plot]  action — sim={len(sim_actions)} steps, real={len(real_actions)} steps\n"
        f"[plot]  obs    — sim={len(sim_df)} steps, real={len(real_df)} steps"
    )

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[plot]  saved → {save_path}")
    else:
        plt.show()


# ─── entry point ──────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    df_sim  = load_sim(args.sim)
    df_real = load_real(args.real)

    sim_ee_pos   = extract_cols(df_sim,  EE_POS_COLS,      "sim/ee_pos")
    sim_ee_quat  = extract_cols(df_sim,  EE_QUAT_COLS,     "sim/ee_quat")
    sim_tgt_pos  = extract_cols(df_sim,  TGT_POS_COLS,     "sim/target_pos")
    sim_tgt_quat = extract_cols(df_sim,  TGT_QUAT_COLS,    "sim/target_quat")
    sim_actions  = extract_cols(df_sim,  SIM_ACTION_COLS,  "sim/raw_action")

    real_ee_pos   = extract_cols(df_real, EE_POS_COLS,       "real/ee_pos")
    real_ee_quat  = extract_cols(df_real, EE_QUAT_COLS,      "real/ee_quat")
    real_tgt_pos  = extract_cols(df_real, TGT_POS_COLS,      "real/target_pos")
    real_tgt_quat = extract_cols(df_real, TGT_QUAT_COLS,     "real/target_quat")
    real_actions  = extract_cols(df_real, REAL_ACTION_COLS,  "real/prev_actions")

    plot_compare(
        sim_ee_pos=sim_ee_pos,    sim_ee_quat=sim_ee_quat,
        sim_tgt_pos=sim_tgt_pos,  sim_tgt_quat=sim_tgt_quat,
        sim_actions=sim_actions,
        real_ee_pos=real_ee_pos,  real_ee_quat=real_ee_quat,
        real_tgt_pos=real_tgt_pos, real_tgt_quat=real_tgt_quat,
        real_actions=real_actions,
        sim_df=df_sim,
        real_df=df_real,
        save_path=args.save_path,
        title=args.title,
    )


if __name__ == "__main__":
    main()