# Cosmos-H-Surgical baseline package (WhatIf-Surg)

Runs NVIDIA Cosmos-H-Surgical (v0.3.1, image2video mode) on 32 jobs and zips the videos.

**Run:** `bash run.sh`. When it finishes, send back `cosmos_results.zip` (created next to `run.sh`).

**Requirements**
- One GPU with at least 48 GB (NVIDIA validated 80 GB A100/H100). The model has 15.8 B bf16 parameters (31.5 GB of weights).
- A driver for CUDA 12.8, or set `CUDA_GROUP=cu130` for CUDA 13.
- `git` (git-lfs not needed), `uv`, about 70 GB of disk, and internet access to Hugging Face.

**What it does**
1. Clones Cosmos-H-Surgical v0.3.1 and installs it with `uv sync`.
2. Replaces the release's checkpoint download helper with a `huggingface_hub` version. The original calls `uvx hf download --format=json`, which failed for us. Set `NO_PATCH=1` to keep the original.
3. Downloads the base checkpoint (`nvidia/Cosmos-H-Surgical`, about 31.5 GB). The Wan2.2 VAE is fetched by the model on first use.
4. Builds the input jsonl from `inputs/jobs.json`:
   - 20 factual windows: the real frame 0 plus the real tool's action;
   - 12 counterfactual swaps: our composed counterfactual frame 0 plus the target tool's action.
   
   The settings are 480p, 16 fps, 65 frames (4 s, same as our clips), seed 1.
5. Runs inference on 1 GPU and zips `out/`, `infer.log` and the jsonl.

**Settings:** `WORK` (default `./work`), `HF_HOME` (default `$WORK/hf_home`), `CUDA_GROUP`, `NO_PATCH`.
