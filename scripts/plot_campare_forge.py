"""sim2real comparison of ee_pos/quat, target_pos/quat, action, and obs trajectories.

Both CSVs share the same column names for ee and target pose, so no coordinate
transform is needed — we load and plot directly.

Sim CSV (from forge_env.py logging):
  episode, step,
  ee_pos_x/y/z, ee_quat_w/x/y/z,
  target_pos_x/y/z, target_quat_w/x/y/z,
  raw_action_0~6, ...
  obs_0~23

Real CSV (from play.py with --save_path):
  fingertip_pos_rel_fixed_x/y/z,
  fingertip_quat_w/x/y/z,
  ee_linvel_x/y/z,
  ee_angvel_x/y/z,
  force_threshold,
  prev_actions_0~6,
  ee_pos_x/y/z, ee_quat_w/x/y/z,
  target_pos_x/y/z, target_quat_w/x/y/z
  (ft_force is absent → padded with 0)

Layout: 4 column grid
  Col 0 : EE pose        — ee_pos_x/y/z + ee_quat_w/x/y/z        (7 rows)
  Col 1 : Target pose    — target_pos_x/y/z + target_quat_w/x/y/z (7 rows)
  Col 2 : Actions 0~6                                              (7 rows)
  Col 3 : Observations   — fingertip_pos_rel(3) + fingertip_quat(4)
                           + ee_linvel(3) + ee_angvel(3)
                           + force_threshold(1) + ft_force(3)
                           + prev_actions sub-set(6) → 18 rows total,
           split into two sub-columns inside col 3 via nested GridSpec

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


# ─── column names ────────────────────────────────────────────────────────────
EE_POS_COLS   = ["ee_pos_x",     "ee_pos_y",     "ee_pos_z"]
EE_QUAT_COLS  = ["ee_quat_w",    "ee_quat_x",    "ee_quat_y",    "ee_quat_z"]
TGT_POS_COLS  = ["target_pos_x", "target_pos_y", "target_pos_z"]
TGT_QUAT_COLS = ["target_quat_w","target_quat_x","target_quat_y","target_quat_z"]

SIM_ACTION_COLS  = [f"raw_action_{i}"   for i in range(7)]
REAL_ACTION_COLS = [f"prev_actions_{i}" for i in range(7)]

# sim obs columns
SIM_OBS_COLS = [f"obs_{i}" for i in range(24)]

# real obs — explicit column names
REAL_FT_POS_COLS   = ["fingertip_pos_rel_fixed_x", "fingertip_pos_rel_fixed_y", "fingertip_pos_rel_fixed_z"]
REAL_FT_QUAT_COLS  = ["fingertip_quat_w", "fingertip_quat_x", "fingertip_quat_y", "fingertip_quat_z"]
REAL_LINVEL_COLS   = ["ee_linvel_x", "ee_linvel_y", "ee_linvel_z"]
REAL_ANGVEL_COLS   = ["ee_angvel_x", "ee_angvel_y", "ee_angvel_z"]
REAL_FTHRESH_COLS  = ["force_threshold"]
# ft_force absent in real → will be zeros
REAL_PREV_ACT_COLS = [f"prev_actions_{i}" for i in range(7)]

# ─── obs sub-group definitions ──────────────────────────────────────────────
#  Each entry: (group_label, sim_slice, real_col_list_or_None, ylabels)
#  sim_slice: slice into obs_0~23
#  real_col:  list of real CSV column names (or None → zeros)

OBS_GROUPS = [
    {
        "title":    "fingertip_pos_rel_fixed",
        "sim_idx":  list(range(0, 3)),           # obs_0~2
        "real_cols": REAL_FT_POS_COLS,
        "ylabels":  ["fp_rel_x [m]", "fp_rel_y [m]", "fp_rel_z [m]"],
    },
    {
        "title":    "fingertip_quat",
        "sim_idx":  list(range(3, 7)),            # obs_3~6
        "real_cols": REAL_FT_QUAT_COLS,
        "ylabels":  ["fq_w", "fq_x", "fq_y", "fq_z"],
    },
    {
        "title":    "ee_linvel",
        "sim_idx":  list(range(7, 10)),           # obs_7~9
        "real_cols": REAL_LINVEL_COLS,
        "ylabels":  ["lv_x [m/s]", "lv_y [m/s]", "lv_z [m/s]"],
    },
    {
        "title":    "ee_angvel",
        "sim_idx":  list(range(10, 13)),          # obs_10~12
        "real_cols": REAL_ANGVEL_COLS,
        "ylabels":  ["av_x [r/s]", "av_y [r/s]", "av_z [r/s]"],
    },
    {
        "title":    "force_threshold",
        "sim_idx":  [13],                         # obs_13
        "real_cols": REAL_FTHRESH_COLS,
        "ylabels":  ["f_thresh"],
    },
    {
        "title":    "ft_force  (sim only)",
        "sim_idx":  list(range(14, 17)),          # obs_14~16
        "real_cols": None,                        # absent in real → zeros
        "ylabels":  ["ft_fx [N]", "ft_fy [N]", "ft_fz [N]"],
    },
    {
        "title":    "prev_actions (obs_17~23)",
        "sim_idx":  list(range(17, 24)),          # obs_17~23
        "real_cols": REAL_PREV_ACT_COLS,
        "ylabels":  [f"prev_a{i}" for i in range(7)],
    },
]

# total obs rows = sum of lengths
N_OBS_ROWS = sum(len(g["ylabels"]) for g in OBS_GROUPS)   # 3+4+3+3+1+3+7 = 24

N_POSE_ROWS   = 7   # 3 pos + 4 quat
N_ACTION_ROWS = 7   # actions 0~6
N_GRID_ROWS   = max(N_POSE_ROWS, N_ACTION_ROWS, N_OBS_ROWS)  # 24

POS_YLABELS    = ["pos_x [m]", "pos_y [m]", "pos_z [m]"]
QUAT_YLABELS   = ["quat_w",    "quat_x",    "quat_y",    "quat_z"]
ACTION_YLABELS = [f"action_{i}" for i in range(7)]


# ─── argument parsing ────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="sim2real comparison: ee_pos/quat, target_pos/quat, actions, and obs."
    )
    p.add_argument("--sim",  type=str, required=True, help="factory simulation CSV path")
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


# ─── data loading ────────────────────────────────────────────────────────────
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


def extract_cols_or_zeros(
    df: pd.DataFrame, cols: list[str] | None, n_rows: int, label: str
) -> np.ndarray:
    """Return data for `cols` from df, or a zero array if cols is None/absent."""
    if cols is None:
        print(f"[warn] {label}: no real columns defined → zeros")
        return np.zeros((n_rows, 0 if cols is None else len(cols)), dtype=np.float64).reshape(n_rows, -1) if cols is None else np.zeros((n_rows, len(cols)))
    missing = [c for c in cols if c not in df.columns]
    if missing:
        print(f"[warn] {label}: columns {missing} absent in real CSV → zeros")
        out = np.zeros((n_rows, len(cols)), dtype=np.float64)
        for j, c in enumerate(cols):
            if c not in missing:
                out[:, j] = df[c].to_numpy(dtype=np.float64)
        return out
    return df[cols].to_numpy(dtype=np.float64)


# ─── drawing helpers ─────────────────────────────────────────────────────────
def draw_column(
    axes: list,
    sim_data: np.ndarray,
    real_data: np.ndarray,
    ylabels: list[str],
    col_title: str,
    show_legend: bool,
    legend_labels: tuple[str, str] = ("sim (factory)", "real (FR3)"),
):
    """Fill one figure column (list of axes) with overlaid sim/real lines."""
    n_sim  = len(sim_data)
    n_real = len(real_data)
    n_min  = min(n_sim, n_real)
    sim_x  = np.arange(n_sim)
    real_x = np.arange(n_real)

    axes[0].set_title(col_title, fontsize=10, pad=5)

    for i, (ax, label) in enumerate(zip(axes, ylabels)):
        ax.plot(sim_x[:n_min],  sim_data[:n_min, i],
                color="C0", linestyle="-",
                label=legend_labels[0] if (show_legend and i == 0) else None)
        if n_sim > n_min:
            ax.plot(sim_x[n_min:], sim_data[n_min:, i],
                    color="C0", linestyle=":", alpha=0.6)

        ax.plot(real_x[:n_min], real_data[:n_min, i],
                color="C1", linestyle="-",
                label=legend_labels[1] if (show_legend and i == 0) else None)
        if n_real > n_min:
            ax.plot(real_x[n_min:], real_data[n_min:, i],
                    color="C1", linestyle=":", alpha=0.6)

        ax.set_ylabel(label, fontsize=8)
        ax.grid(True, alpha=0.3)
        if show_legend and i == 0:
            ax.legend(loc="upper right", fontsize=7)


def draw_obs_column(
    fig,
    outer_gs_cell,
    sim_df: pd.DataFrame,
    real_df: pd.DataFrame,
):
    """
    Draw all obs sub-groups inside a single outer GridSpec cell using a nested
    GridSpec.  Groups are stacked vertically; within each group the individual
    signal rows share x-axis.
    """
    n_sim  = len(sim_df)
    n_real = len(real_df)

    # nested GridSpec — one row per obs signal
    inner_gs = gridspec.GridSpecFromSubplotSpec(
        N_OBS_ROWS, 1,
        subplot_spec=outer_gs_cell,
        hspace=0.10,
    )

    row_cursor = 0
    first_group = True

    for g_idx, grp in enumerate(OBS_GROUPS):
        sim_idx   = grp["sim_idx"]
        real_cols = grp["real_cols"]
        ylabels   = grp["ylabels"]
        n_ch      = len(ylabels)

        # --- extract sim data from obs columns ---
        sim_data = sim_df[[f"obs_{i}" for i in sim_idx]].to_numpy(dtype=np.float64)

        # --- extract real data (or zeros) ---
        real_data = extract_cols_or_zeros(real_df, real_cols, n_real, f"real/{grp['title']}")

        n_min  = min(n_sim, n_real)
        sim_x  = np.arange(n_sim)
        real_x = np.arange(n_real)

        axes = [fig.add_subplot(inner_gs[row_cursor + r, 0]) for r in range(n_ch)]

        # share x within group
        for ax in axes[1:]:
            ax.sharex(axes[0])
        for ax in axes[:-1]:
            plt.setp(ax.get_xticklabels(), visible=False)

        # group title as text annotation on first axis
        axes[0].set_title(grp["title"], fontsize=8, pad=3, loc="left",
                          color="dimgray", style="italic")

        for i, (ax, ylabel) in enumerate(zip(axes, ylabels)):
            ax.plot(sim_x[:n_min],  sim_data[:n_min, i],  color="C0", lw=0.9,
                    label="sim" if (first_group and i == 0) else None)
            if n_sim > n_min:
                ax.plot(sim_x[n_min:], sim_data[n_min:, i], color="C0", lw=0.9,
                        linestyle=":", alpha=0.6)

            if real_data.shape[1] == 0:
                # ft_force absent — shade region to indicate sim-only
                ax.fill_between(sim_x, sim_data[:, i], alpha=0.15, color="C0")
            else:
                ax.plot(real_x[:n_min],  real_data[:n_min, i],  color="C1", lw=0.9,
                        label="real" if (first_group and i == 0) else None)
                if n_real > n_min:
                    ax.plot(real_x[n_min:], real_data[n_min:, i], color="C1", lw=0.9,
                            linestyle=":", alpha=0.6)

            ax.set_ylabel(ylabel, fontsize=7, labelpad=2)
            ax.yaxis.set_tick_params(labelsize=6)
            ax.xaxis.set_tick_params(labelsize=6)
            ax.grid(True, alpha=0.25)

            if first_group and i == 0:
                ax.legend(loc="upper right", fontsize=7)
                first_group = False

        # x-label on last axis of each group
        axes[-1].set_xlabel("step", fontsize=7)

        row_cursor += n_ch


# ─── main plotting ───────────────────────────────────────────────────────────
def plot_compare(
    sim_df: pd.DataFrame,
    real_df: pd.DataFrame,
    save_path: str | None,
    title: str | None,
):
    sim_ee_pos   = extract_cols(sim_df,  EE_POS_COLS,      "sim/ee_pos")
    sim_ee_quat  = extract_cols(sim_df,  EE_QUAT_COLS,     "sim/ee_quat")
    sim_tgt_pos  = extract_cols(sim_df,  TGT_POS_COLS,     "sim/target_pos")
    sim_tgt_quat = extract_cols(sim_df,  TGT_QUAT_COLS,    "sim/target_quat")
    sim_actions  = extract_cols(sim_df,  SIM_ACTION_COLS,  "sim/raw_action")

    real_ee_pos   = extract_cols(real_df, EE_POS_COLS,       "real/ee_pos")
    real_ee_quat  = extract_cols(real_df, EE_QUAT_COLS,      "real/ee_quat")
    real_tgt_pos  = extract_cols(real_df, TGT_POS_COLS,      "real/target_pos")
    real_tgt_quat = extract_cols(real_df, TGT_QUAT_COLS,     "real/target_quat")
    real_actions  = extract_cols(real_df, REAL_ACTION_COLS,  "real/prev_actions")

    # ── figure layout ──────────────────────────────────────────────────────
    # 4 outer columns: ee | target | action | obs
    # Row count driven by obs (24); other columns use top-aligned subplots.
    fig = plt.figure(figsize=(28, 20))
    if title:
        fig.suptitle(title, fontsize=13, y=0.995)

    outer_gs = gridspec.GridSpec(
        N_GRID_ROWS, 4,
        figure=fig,
        hspace=0.08,
        wspace=0.40,
        left=0.06, right=0.97,
        top=0.97,  bottom=0.04,
    )

    # ── Col 0: EE pose (7 rows) ───────────────────────────────────────────
    ee_axes  = [fig.add_subplot(outer_gs[r, 0]) for r in range(N_POSE_ROWS)]
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

    # ── Col 1: Target pose (7 rows) ───────────────────────────────────────
    tgt_axes = [fig.add_subplot(outer_gs[r, 1]) for r in range(N_POSE_ROWS)]
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

    # ── Col 2: Actions (7 rows) ───────────────────────────────────────────
    act_axes = [fig.add_subplot(outer_gs[r, 2]) for r in range(N_ACTION_ROWS)]
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

    # ── Col 3: Obs (all 24 rows, nested) ─────────────────────────────────
    # Pass the entire column-3 span of the outer GridSpec as one cell.
    obs_cell = outer_gs[:N_OBS_ROWS, 3]
    draw_obs_column(fig, obs_cell, sim_df, real_df)

    # Add a column header label for obs
    obs_header_ax = fig.add_subplot(outer_gs[0, 3])
    obs_header_ax.set_title("Observations (obs_0 ~ obs_23)", fontsize=10, pad=5)
    obs_header_ax.axis("off")

    # ── summary ───────────────────────────────────────────────────────────
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


# ─── entry point ─────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    df_sim  = load_sim(args.sim)
    df_real = load_real(args.real)
    plot_compare(
        sim_df=df_sim,
        real_df=df_real,
        save_path=args.save_path,
        title=args.title,
    )


if __name__ == "__main__":
    main()