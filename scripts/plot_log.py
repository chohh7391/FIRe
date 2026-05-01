import pandas as pd
import matplotlib.pyplot as plt

# ── Config ────────────────────────────────────────────────────────────────────
SIM_CSV   = "/home/home/FIRe/scripts/configs/traj_save.csv"
SIM_EPISODE = 2   # traj_save_1.csv에서 사용할 episode
REAL_CSV  = "/home/home/FIRe/logs/20260427/collected_data_20260427_142257.csv"
OUT_PNG   = "/home/home/FIRe/logs/20260427/fig/obs_comparison.png"
MAX_STEP  = None  # None = all steps

# ── Observation groups to plot ─────────────────────────────────────────────
OBS_GROUPS = [
    {
        "title": "fingertip_pos_rel_fixed",
        "cols":  ["obs/fingertip_pos_rel_fixed_0", "obs/fingertip_pos_rel_fixed_1", "obs/fingertip_pos_rel_fixed_2"],
        "labels": ["X", "Y", "Z"],
        "ylabel": "m",
    },
    {
        "title": "ft_force",
        "cols":  ["obs/ft_force_0", "obs/ft_force_1", "obs/ft_force_2"],
        "labels": ["Fx", "Fy", "Fz"],
        "ylabel": "N",
    },
    {
        "title": "ee_linvel",
        "cols":  ["obs/ee_linvel_0", "obs/ee_linvel_1", "obs/ee_linvel_2"],
        "labels": ["Vx", "Vy", "Vz"],
        "ylabel": "m/s",
    },
    {
        "title": "ee_angvel",
        "cols":  ["obs/ee_angvel_0", "obs/ee_angvel_1", "obs/ee_angvel_2"],
        "labels": ["ωx", "ωy", "ωz"],
        "ylabel": "rad/s",
    },
    {
        "title": "fingertip_quat",
        "cols":  ["obs/fingertip_quat_0", "obs/fingertip_quat_1", "obs/fingertip_quat_2", "obs/fingertip_quat_3"],
        "labels": ["w", "x", "y", "z"],
        "ylabel": "",
    },
    {
        "title": "prev_actions",
        "cols":  [f"obs/prev_actions_{i}" for i in range(7)],
        "labels": [f"a{i}" for i in range(7)],
        "ylabel": "",
    },
    {
        "title": "action (ctrl_target)",
        "cols":  ["action_0", "action_1", "action_2", "action_3", "action_4", "action_5"],
        "labels": ["x", "y", "z", "rx", "ry", "rz"],
        "ylabel": "",
    },
]

COLORS = ["steelblue", "tomato", "seagreen", "darkorange", "mediumpurple", "saddlebrown", "deeppink"]


# traj_save_1.csv obs_order:
#   obs_0..2  = fingertip_pos_rel_fixed
#   obs_3..6  = fingertip_quat
#   obs_7..9  = ee_linvel
#   obs_10..12 = ee_angvel
#   obs_13..15 = ft_force
#   obs_16    = force_threshold
#   obs_17..23 = prev_actions
SIM_COL_MAP = {
    "obs/fingertip_pos_rel_fixed_0": "obs_0",
    "obs/fingertip_pos_rel_fixed_1": "obs_1",
    "obs/fingertip_pos_rel_fixed_2": "obs_2",
    "obs/fingertip_quat_0":          "obs_3",
    "obs/fingertip_quat_1":          "obs_4",
    "obs/fingertip_quat_2":          "obs_5",
    "obs/fingertip_quat_3":          "obs_6",
    "obs/ee_linvel_0":               "obs_7",
    "obs/ee_linvel_1":               "obs_8",
    "obs/ee_linvel_2":               "obs_9",
    "obs/ee_angvel_0":               "obs_10",
    "obs/ee_angvel_1":               "obs_11",
    "obs/ee_angvel_2":               "obs_12",
    "obs/ft_force_0":                "obs_13",
    "obs/ft_force_1":                "obs_14",
    "obs/ft_force_2":                "obs_15",
    "obs/force_threshold":           "obs_16",
    **{f"obs/prev_actions_{i}": f"obs_{17 + i}" for i in range(7)},
}


def load_real(path, max_step):
    df = pd.read_csv(path)
    if max_step is not None:
        df = df[df["step"] <= max_step]
    return df.reset_index(drop=True)


def load_sim(path, episode, max_step):
    df = pd.read_csv(path)
    df = df[df["episode"] == episode].reset_index(drop=True)
    if max_step is not None:
        df = df[df["step"] <= max_step].reset_index(drop=True)
    # reverse-map: add obs/... columns from obs_N columns
    for named_col, obs_col in SIM_COL_MAP.items():
        if obs_col in df.columns:
            df[named_col] = df[obs_col]
    return df


df_sim  = load_sim(SIM_CSV, SIM_EPISODE, MAX_STEP)
df_real = load_real(REAL_CSV, MAX_STEP)

n_rows = len(OBS_GROUPS)
fig, axes = plt.subplots(n_rows, 2, figsize=(16, 3.2 * n_rows), sharex=False)
fig.suptitle("Observation Comparison: Sim (left) vs Real (right)", fontsize=14, fontweight="bold")

for row_idx, grp in enumerate(OBS_GROUPS):
    for col_idx, (df, label_prefix) in enumerate([(df_sim, "Sim"), (df_real, "Real")]):
        ax = axes[row_idx, col_idx]
        steps = df["step"]
        for i, (col, lbl) in enumerate(zip(grp["cols"], grp["labels"])):
            if col in df.columns:
                ax.plot(steps, df[col], color=COLORS[i % len(COLORS)], lw=1.5, label=lbl)
        ax.set_title(f"{label_prefix} — {grp['title']}", fontsize=9)
        ax.set_xlabel("step", fontsize=8)
        ax.set_ylabel(grp["ylabel"], fontsize=8)
        ax.legend(fontsize=7, ncol=4, loc="upper right")
        ax.grid(True, alpha=0.4)
        ax.tick_params(labelsize=7)

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
plt.show()
print(f"[INFO] Saved: {OUT_PNG}")
