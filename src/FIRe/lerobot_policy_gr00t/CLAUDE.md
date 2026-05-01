# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Package Overview

`lerobot_policy_gr00t` is a LeRobot-compatible wrapper around NVIDIA's **GR00T N1.5** (3B) vision-language-action model. It integrates the GR00T model into the LeRobot policy interface so it can be trained and deployed alongside other LeRobot policies.

Install as editable (already included via the root `requirements.txt`):
```bash
pip install -e src/FIRe/lerobot_policy_gr00t
```

## Architecture

### Class hierarchy

```
Gr00tPolicy (PreTrainedPolicy)        # modeling_gr00t.py — LeRobot interface
  └── _gr00t_model: GR00TN15          # gr00t_n1.py — HuggingFace PreTrainedModel
        ├── backbone: EagleBackbone   # Eagle VLM (vision-language encoder)
        └── action_head: FlowmatchingActionHead  # action_head/flow_matching_action_head.py
```

### Configuration (`configuration_gr00t.py`)

`Gr00tConfig` is registered as a LeRobot `PreTrainedConfig` subclass under the tag `"gr00t"`. Key constraints to be aware of:

- **State** is zero-padded to `max_state_dim=64`; **action** to `max_action_dim=32`.
- `chunk_size` may be set freely, but the pretrained model architecture caps the effective action horizon at **16** — `processor_gr00t.py` enforces `action_horizon = min(chunk_size, 16)`.
- Default `base_model_path = "nvidia/GR00T-N1.5-3B"` (downloaded via HuggingFace Hub on first use).
- Default fine-tuning knobs: `tune_llm=False`, `tune_visual=False`, `tune_projector=True`, `tune_diffusion_model=True`.

### GR00TN15 / EagleBackbone (`gr00t_n1.py`)

`EagleBackbone` wraps the Eagle VLM (vendored in `eagle2_hg_model/`). It reads model config from the local vendor directory copied into the LeRobot cache (`HF_LEROBOT_HOME / tokenizer_assets_repo`). This copy is done by `ensure_eagle_cache_ready()` in `utils.py` and must succeed before any Eagle processor or backbone is built.

`GR00TN15.from_pretrained()` calls `snapshot_download` and falls back to a local path if the repo is not found on the Hub.

### Preprocessing pipeline (`processor_gr00t.py`)

`make_gr00t_pre_post_processors(config, dataset_stats)` returns a `(preprocessor, postprocessor)` pair built from these steps:

| Step | Class | What it does |
|---|---|---|
| 1 | `RenameObservationsProcessorStep` | Optional key remapping |
| 2 | `AddBatchDimensionProcessorStep` | Unbatched → batched |
| 3 | `Gr00tPackInputsStep` | Packs video/state/action/language/embodiment; applies min-max norm before zero-padding to `(max_state_dim, max_action_dim)` |
| 4 | `Gr00tEagleEncodeStep` | Runs Eagle chat template + `process_vision_info` → stores `eagle_content` |
| 5 | `Gr00tEagleCollateStep` | Collates `eagle_content` → `eagle_*` tensors via the Eagle processor |
| 6 | `DeviceProcessorStep` | Moves tensors to GPU |

Postprocessor (`Gr00tActionUnpackUnnormalizeStep`) takes the last timestep of the model's `action_pred` output, slices to `env_action_dim`, and inverts the min-max normalization back to environment scale.

Normalization statistics (`stats`) are persisted via `state_dict()` / `load_state_dict()` on both `Gr00tPackInputsStep` and `Gr00tActionUnpackUnnormalizeStep`, saved alongside model weights in safetensors format.

### Action head (`action_head/`)

`FlowmatchingActionHead` is a cross-attention DiT (`action_head/cross_attention_dit.py`). It encodes state and noisy actions with `MultiEmbodimentActionEncoder` (category-specific linear layers per embodiment ID), cross-attends against Eagle backbone features, and decodes predicted velocity for flow-matching.

**Embodiment mapping** (used as the integer `embodiment_id`):
- `"new_embodiment"` → 31 (default for novel robots)
- `"franka"` → 25
- `"so100"` → 2

Training loss: MSE on predicted velocity vs. ground-truth velocity (flow-matching target = `action - noise`), masked by `action_mask`.

Inference: Euler integration over `num_inference_timesteps` steps starting from Gaussian noise.

### Eagle processor cache

The Eagle processor (`eagle2_hg_model/`) is vendored locally and must be copied into:
```
$HF_LEROBOT_HOME/lerobot/eagle2hg-processor-groot-n1p5/
```
This happens automatically in `EagleBackbone.__init__` and `_build_eagle_processor()` via `ensure_eagle_cache_ready()`. If you see `FileNotFoundError` about the Eagle processor cache, create the policy/model first (which triggers the copy), or call `ensure_eagle_cache_ready()` directly.

## Loading a policy

```python
from lerobot_policy_gr00t.modeling_gr00t import Gr00tPolicy
from lerobot_policy_gr00t.processor_gr00t import make_gr00t_pre_post_processors

# Base model (fresh from HuggingFace)
policy = Gr00tPolicy.from_pretrained("nvidia/GR00T-N1.5-3B")

# Fine-tuned LeRobot checkpoint (directory with model.safetensors)
policy = Gr00tPolicy.from_pretrained("/path/to/checkpoint")

# Build processors (requires dataset stats for normalization)
preprocessor, postprocessor = make_gr00t_pre_post_processors(policy.config, dataset_stats)
```

`from_pretrained` auto-detects whether the path is a base HuggingFace model or a fine-tuned LeRobot checkpoint by checking for `model.safetensors`.
