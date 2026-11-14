# Cosmos-H-Surgical video2video baseline (WhatIf-Surg)

Runs NVIDIA Cosmos-H-Surgical v0.3.1 in **video2video (transfer)** mode, using a blur control on 24 jobs (12 factual windows and 12 counterfactual swaps) in three versions. That is 72 videos, all zipped at the end.

The image2video package (frame 0 and text only) is in the repository root. This folder is separate.

**Tested end to end on 2026-11-14 on an RTX 4090 (24 GB)**, in low-VRAM mode (see below):
- `env.sh` was built from scratch and passed its self-test.
- `SMOKE=cf_A54__33125__to_scissors bash run.sh` ran versions A, B and C back to back in one process.
- Each version wrote an 832×480, 65-frame, 16 fps video. The run took 50 steps at 8.5 s per step, about 7 min per video.

On a ≥40 GB GPU (A100, H100, H200), the standard path is used, which is much faster.

## Quick start

```bash
bash env.sh                                         # 1. build + self-test the environment (~10 min, once)
SMOKE=cf_A54__33125__to_scissors bash run.sh        # 2. one swap, all three versions (A, B, C) end to end
bash run.sh                                         # 3. all 72 jobs -> cosmos_v2v_results.zip
```

- Send back `cosmos_v2v_results.zip`, written next to `run.sh`.
- Every step is safe to re-run; finished steps are skipped.
- To run one version only: `VERSIONS=A bash run.sh`. Any subset works, e.g. `VERSIONS=BC`.
- The input videos (72 mp4, 197 MB) are downloaded automatically from the release asset `v2v-inputs-v1` on the first run.

## The three versions

All three use the same prompt and the same frame 0. For a counterfactual swap, frame 0 is our composed counterfactual frame 0 (the same frame 0 every baseline gets). Only the control video differs.

| Version | Control video (`vision_path`; Cosmos computes the blur control from it) |
|---|---|
| **A — full GT** | The full real clip, with the source tool. |
| **B — masked** | The real clip with the source tool (instrument segmentation, dilated 7 px) blacked out on frames 1–64. No CAD information is used. |
| **C — masked + CAD** | Our model's own control video: the real clip outside the edit mask, with our coloured CAD guide inside it. |

The prompt is the requested tool's description and action, as a JSON caption. For factual windows that is the real tool; for swaps it is the target tool.

## Settings

| | |
|---|---|
| Mode | `video2video`, blur control (`preset_blur_strength: medium`) |
| Frames / fps / resolution | 65 / 16 / 480p 16:9 (832×480), one chunk |
| Conditioning | frame 0 of the control video (`num_first_chunk_conditional_frames: 1`) |
| Sampler | 50 steps, guidance 1.5, control guidance 1.0, shift 5 (the release's transfer settings), seed 1 |
| Checkpoint | base `Cosmos-H-Surgical` (the DMD2 4-step student is not used) |

## Requirements

- **GPU:** either one GPU with 40 GB or more, or a 24 GB GPU plus about 32 GB of free RAM in low-VRAM mode.
- **Driver:** version 570 or newer (CUDA 12.8).
- **Tools:** `git`.
- **Disk:** about 70 GB.
- **Internet:** GitHub, PyPI and Hugging Face.

## Low-VRAM mode (GPUs under 39 GiB)

The 15.8 B model has 31.5 GB of bf16 weights, which do not fit on a 24 GB card. On such a GPU, `run.sh` automatically starts inference through `patch/lowvram_launch.py`:
- The first decoder blocks stay on the GPU (budget `COSMOS_LOWVRAM_GPU_GB`, default 8).
- The remaining blocks stay in CPU RAM and are copied to the GPU just before each call.
- The weights and kernels are the same as in the standard path; only where the weights are stored changes.
- `LOWVRAM=1` forces this mode and `LOWVRAM=0` disables it.

## Environment fixes applied by `env.sh`

These are the same fixes as the image2video package (see the root README):
- git-lfs is disabled.
- A `nvidia/cudart` symlink is added, which fixes "cudart shared object not found".
- The `hf download` checkpoint helper is patched.
- The venv can be placed on ext4 with `VENV_DIR=...` when the work directory's filesystem does not support symlinks.

All upstream dependencies are pinned: the v0.3.1 tag, cosmos-framework `ed8287f`, Megatron-LM `de56227`, and `uv.lock`.

## Files

- `run.sh`: env, weights, inputs, inference, zip.
- `env.sh`: builds the environment and runs `check_env.py`.
- `inputs/make_jsonl.py`: builds the transfer jsonl.
- `inputs/build_inputs.py`: how the control videos were made. It runs on our machine only, for provenance.
- `patch/lowvram_launch.py`: the low-VRAM launcher.
- `patch/checkpoint_db_patched.py`: the patched checkpoint helper.
