# Cosmos-H-Surgical baseline package (WhatIf-Surg)

> **New (2026-11-14):** [`video2video/`](video2video/) — the transfer (video2video) baseline with the source video as context, three versions (full GT / masked / masked + CAD), tested on an RTX 4090 in low-VRAM mode.

Runs NVIDIA Cosmos-H-Surgical (v0.3.1, image2video mode) on 32 jobs and zips the videos.

**Tested end to end on 2026-11-13** on Google Colab (A100-SXM4-40GB, driver 580.82, no system CUDA toolkit used). `env.sh` passed its self-test, and `SMOKE=1 bash run.sh` wrote one 832×480, 65-frame, 16 fps video (35 sampling steps at 1.7 s/step, about 1 min per video after the model loads).

## Quick start

```bash
bash env.sh              # 1. build + self-test the environment (~10 min, once)
SMOKE=1 bash run.sh      # 2. one video end to end (checks weights, inputs, inference)
bash run.sh              # 3. all 32 jobs -> cosmos_results.zip
```

Send back `cosmos_results.zip` (written next to `run.sh`). Every step is safe to re-run: finished steps are skipped.

`env.sh` ends with a self-test (`check_env.py`). If it prints `[check_env] all good`, the environment is fine. If a line says `FAIL`, send us that line.

## Requirements

- **GPU:** one GPU with 40 GB or more. Tested on an A100 40 GB; H100/H200 are fine. The model has 15.8 B bf16 parameters (31.5 GB of weights).
- **Driver:** version 570 or newer for CUDA 12.8, or set `CUDA_GROUP=cu130` (driver 580 or newer).
- **No CUDA toolkit needed:** CUDA comes from pip wheels inside the venv.
- **Tools:** `git` (git-lfs not needed). `uv` is installed automatically if missing.
- **Disk:** about 70 GB.
- **Internet:** GitHub, PyPI, download.pytorch.org and Hugging Face.

## Environment fixes applied by `env.sh`

| Error | Cause | Fix |
|---|---|---|
| `git-lfs ... not found` (clone or `uv sync`) | A global git-lfs filter is configured but the binary is missing | The LFS filter is disabled for every git call via `GIT_CONFIG_*` env vars. The LFS files are only NVIDIA example media. |
| `RuntimeError: cudart shared object not found` (transformer_engine import) | On a machine without a system CUDA toolkit, TE 2.12 looks for the pip CUDA runtime in `nvidia/cudart`, but the wheel installs it as `nvidia/cuda_runtime` | Symlink `nvidia/cudart -> cuda_runtime`. `activate.sh` also puts all pip CUDA libs on `LD_LIBRARY_PATH` and points `NVRTC_HOME`/`CURAND_HOME`/`CUDNN_HOME` at them, so a system CUDA of another version is never mixed in. |
| `uvx hf download --format=json` rejected | The release's checkpoint helper uses a CLI option that newer `hf` versions don't have | `patch/checkpoint_db_patched.py` uses the `huggingface_hub` API instead. Set `NO_PATCH=1` to keep the original. |
| `failed to symlink ... Operation not permitted` (`uv sync`) | The work dir is on a filesystem without symlinks (exFAT, some NFS) | Set `VENV_DIR=/path/on/ext4/cosmos_venv`. |

To use the env by hand: `source work/activate.sh`.

## What `run.sh` does

1. **Environment:** `bash env.sh` (skipped if it already passed).
2. **Weights:** downloads the base checkpoint `nvidia/Cosmos-H-Surgical` (about 31.5 GB). The Wan2.2 VAE is fetched by the model on first use.
3. **Inputs:** builds the input jsonl from `inputs/jobs.json`.
   - 20 factual windows: the real frame 0 plus the real tool's action.
   - 12 counterfactual swaps: our composed counterfactual frame 0 plus the target tool's action.
   - Settings: 480p, 16 fps, 65 frames (4 s, same as our clips).
   - `SMOKE=N` keeps only the first N jobs.
4. **Inference:** runs on 1 GPU with `--no-guardrails`. Log: `work/infer.log`.
5. **Pack:** zips `out/`, `infer.log` and the jsonl into `cosmos_results.zip`.

## Settings

| Variable | Default | Meaning |
|---|---|---|
| `WORK` | `./work` | Work directory |
| `HF_HOME` | `$WORK/hf_home` | Hugging Face cache and weights |
| `VENV_DIR` | `$WORK/Cosmos-H-Surgical/.venv` | Where the venv goes |
| `CUDA_GROUP` | `cu128` | `cu128` or `cu130` |
| `NO_PATCH` | unset | Keep the original checkpoint helper |
| `SMOKE` | unset | Run only the first N jobs |
