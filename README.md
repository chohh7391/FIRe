# FIRe — Force-Informed Residual Policy for Contact-Rich Manipulation with VLA Models

This repository is the top-level workspace for **FIRe (Force-Informed Residual Policy)**,
the system described in the paper *"FIRe: Force-Informed Residual Policy for Contact-Rich
Manipulation with Vision-Language-Action Models"* (submitted to IEEE RA-L, 2026).

FIRe augments a **frozen, action-head-fine-tuned Vision-Language-Action (VLA) model** with a
**force-aware residual reinforcement-learning policy**. The VLA supplies long-horizon action
chunks from vision and language; the residual policy runs at every control step, observes
contact force, and adds bounded corrections. The two are combined by simple addition in a shared
relative-pose representation:

```
a_t = a_t^VLA + a_t^res
```

Trained entirely in simulation with only a **sparse task-success reward** and a **task-agnostic
force-aware dense reward**, FIRe transfers **zero-shot** to a real Franka FR3. It improves success
over the VLA base policy by an average of **64.5 percentage points** (up to 90.0) across three
contact-rich assembly tasks, matches an RL controller trained with task-specific dense rewards
(no per-task reward engineering), and ports across the **GR00T, π₀.₅, and OpenVLA** backbones
without architectural changes.

---

## Why FIRe

VLA models generalize well but are brittle in contact-rich assembly for three reasons the paper
identifies: (1) force/tactile signals are underused, and chunked open-loop VLA actions cannot
react at the control rate; (2) the end-effector and manipulated object occlude task-critical
state from vision; (3) collecting contact-rich demonstrations is costly. RL can use force feedback
for reactive control but needs hand-engineered per-task dense rewards and is sample-hungry.

FIRe combines the two so each covers the other's weakness:

- The **VLA base action warm-starts** the residual RL agent near the goal, removing the need for a
  task-specific dense reward — the goal-directed drive comes from the VLA chunk plus a sparse
  success reward.
- The **residual policy** observes force and reacts every control step, compensating for the VLA's
  limited tactile understanding and its susceptibility to visual occlusion.

Because the VLA stays frozen and only a shared 6-DoF relative-pose action representation must
match, the same residual policy plugs into any action-chunking VLA backbone.

### Slow-fast asynchronous control loop

- The **VLA** is queried once per chunk of length `T = 16`; a single inference exceeds one control
  step, so the next chunk is **prefetched asynchronously** at the chunk midpoint and swapped in
  when the current chunk is exhausted.
- The **residual policy** runs at **every** control step (~15 Hz), and each combined command is
  interpolated to the **1 kHz** task-space impedance controller.

### Observations

- **Residual (deployment-accessible):** `o^res = { p_ee^rel, v_ee, F_t, a_{t-1} }` — EEF pose relative
  to the fixed object, EEF velocity, 3-axis contact force, previous command.
- **VLA:** `o^VLA = { {I_k}, p_ee^abs, ℓ }` — multi-camera RGB, absolute EEF pose, language instruction;
  read once per chunk.

The residual policy is trained with **PPO (asymmetric actor-critic)** in **Isaac Lab**: the critic
uses privileged simulator state, the actor uses only `o^res`.

---

## Repository layout

This is a git **superproject** that pulls together two independently versioned repositories as
submodules, plus the paper sources (kept locally, not tracked).

```
FIRe/
├── fire_lab/     → chohh7391/fire_lab   Simulation & training side
├── fire_deploy/  → chohh7391/FIRe       Real-robot deployment side
└── _2026__RAL___VLA_RL/                 Paper sources (git-ignored, reference only)
```

### `fire_lab/` — simulation & training

An Isaac Lab (Isaac Sim 5.1 / Isaac Lab 2.3) extension. It provides:

- The contact-rich task environments (Factory / Forge families: **Peg Insert, Gear Mesh, Nut Thread**)
  with a Franka FR3 + wrist force/torque sensor.
- **Residual RL training** (PPO via rl_games) where the RL action is added to a VLA action chunk,
  with the force-aware reward.
- **VLA integration as external inference servers** (GR00T over ZMQ, π₀.₅ over WebSocket, OpenVLA
  over HTTP) and the slow-fast chunk scheduling.
- **Demo generation** for action-head fine-tuning of the VLA (GR00T / LeRobot-format datasets).

Output: trained residual-policy checkpoints, consumed by `fire_deploy`.

### `fire_deploy/` — real-robot deployment

The ROS 2 execution stack that runs on the physical **Franka FR3**. It:

- Loads the residual checkpoints trained in `fire_lab` and runs them in a dedicated inference process.
- Executes the **slow-fast control loop**, adding the residual output on top of the live VLA chunk.
- Talks to the same VLA backbones (GR00T / π₀.₅ / OpenVLA) as inference servers.
- Reads the wrist force/torque sensor and RealSense cameras, and can record teleoperation datasets.

No training, reward, or simulator lives here — it is execution-only, and its observation/action
pipeline mirrors the simulator's so that policies transfer zero-shot.

### How the two connect (the sim→real contract)

`fire_lab` trains and `fire_deploy` deploys, across the simulation-to-reality boundary. They must
agree on: the residual observation `o^res`, the relative-pose action representation and the residual
sum `a = a^VLA + a^res`, the VLA client protocol and per-backbone chunk sizes, the slow-fast schedule,
and the checkpoint format. Keep these consistent when changing either side.

---

## Results (real robot, 30 trials per method)

| Method | Peg Insert | Gear Mesh | Nut Thread |
| --- | --- | --- | --- |
| VLA (GR00T base) | 23.3% | 10.0% | 3.3% |
| RL dense (FORGE, task-specific reward) | 86.7% | 100.0% | 33.3% |
| **FIRe (ours)** | **96.7%** | **100.0%** | **33.3%** |
| Improvement over VLA base | +73.4 | +90.0 | +30.0 |

FIRe matches the dense-reward RL baseline **without any task-specific reward engineering**, and an
ablation shows the force **observation** (enables alignment) and the force-aware **reward** (removes
jamming/entry failures) each contribute significantly. The same residual policy improves the GR00T,
π₀.₅, and OpenVLA backbones on every task.

---

## Getting started

Clone the workspace with its submodules:

```bash
git clone --recurse-submodules <this-repo-url> FIRe
cd FIRe
# or, if already cloned without submodules:
git submodule update --init --recursive
```

The two components have **separate, incompatible environments** — set up each in its own conda env,
following its own README:

- **Simulation / training:** see [`fire_lab/README.md`](fire_lab/README.md)
  (Isaac Sim 5.1, Python 3.11; `pip install -e source/fire_lab`).
- **Real-robot deployment:** see [`fire_deploy/README.md`](fire_deploy/README.md)
  (Python 3.10, ROS 2, LeRobot; `pip install -r requirements.txt`).

A typical flow: generate demos and fine-tune a VLA → train the residual policy in `fire_lab` →
copy the checkpoint → run it on the real robot from `fire_deploy`.

---

## Notes

- The paper sources under `_2026__RAL___VLA_RL/` are intentionally **git-ignored**; they are kept
  locally as a reference for understanding the codebase and are not part of the tracked project.
- `fire_lab` and `fire_deploy` retain their own git histories and remotes; commit changes inside the
  respective submodule, then update the submodule pointer here.

## Citation

```bibtex
@article{fire2026,
  title   = {FIRe: Force-Informed Residual Policy for Contact-Rich Manipulation
             with Vision-Language-Action Models},
  year    = {2026},
  note    = {Submitted to IEEE Robotics and Automation Letters (RA-L)}
}
```
