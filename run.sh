#!/usr/bin/env bash
# Cosmos-H-Surgical baseline (image + text -> video) for the WhatIf-Surg comparison. One command: bash run.sh
# Needs: 1 GPU with >= 48 GB (validated by NVIDIA on 80 GB A100/H100), CUDA 12.8 driver (>= 570), git, uv (https://docs.astral.sh/uv/), ~70 GB disk, internet for the HF download.
set -euo pipefail
# Git LFS: this machine's git config calls git-lfs, which is not installed. The repos' LFS files are only example media we don't use.
# Disable the LFS filter for EVERY git call in this script, including the ones uv makes when it fetches cosmos-framework.
export GIT_LFS_SKIP_SMUDGE=1 GIT_CONFIG_COUNT=3 GIT_CONFIG_KEY_0=filter.lfs.process GIT_CONFIG_VALUE_0= GIT_CONFIG_KEY_1=filter.lfs.smudge GIT_CONFIG_VALUE_1= GIT_CONFIG_KEY_2=filter.lfs.required GIT_CONFIG_VALUE_2=false
HERE="$(cd "$(dirname "$0")" && pwd)"; WORK=${WORK:-$HERE/work}; mkdir -p "$WORK"; cd "$WORK"
export HF_HOME=${HF_HOME:-$WORK/hf_home}
# 1. code (pinned release) + environment
# Git LFS holds only NVIDIA's example media (not needed); skip it so the clone works without git-lfs installed
[ -f Cosmos-H-Surgical/pyproject.toml ] || { rm -rf Cosmos-H-Surgical; git clone --depth 1 --branch v0.3.1 https://github.com/NVIDIA-Medtech/Cosmos-H-Surgical.git; }
cd Cosmos-H-Surgical; uv cache clean cosmos-framework >/dev/null 2>&1 || true; uv sync --group ${CUDA_GROUP:-cu128}; source .venv/bin/activate
# 2. the release's checkpoint helper calls `uvx hf download --format=json`, which failed on our machine (option rejected) -> replace with the huggingface_hub Python API (patch/checkpoint_db_patched.py; set NO_PATCH=1 to skip)
if [ -z "${NO_PATCH:-}" ]; then T=$(python -c "import cosmos_framework.utils.checkpoint_db as c; print(c.__file__)"); cp "$T" "$T.orig"; cp "$HERE/patch/checkpoint_db_patched.py" "$T"; fi
# 3. weights: base checkpoint only (the DMD2 transfer student is not needed for image2video)
python - <<'P'
from huggingface_hub import snapshot_download
import os
p = snapshot_download("nvidia/Cosmos-H-Surgical", local_dir=os.environ["HF_HOME"] + "/Cosmos-H-Surgical", allow_patterns=["*.json", "*.md", "*.txt", "*.sha256", "LICENSE*", "NOTICE*", "model-*.safetensors", "*.jinja", "tokenizer*", "vocab*", "merges*"], max_workers=8)
print("weights at", p)
P
# 4. inputs: 32 jobs (20 factual windows + 12 counterfactual swaps), frame 0 + action text
python "$HERE/inputs/make_jsonl.py" "$WORK/cosmos_in.jsonl"
# 5. inference (one GPU)
mkdir -p "$WORK/out"
torchrun --nproc-per-node=1 -m cosmos_h_surgical infer --checkpoint-path "$HF_HOME/Cosmos-H-Surgical" --parallelism-preset=latency --dp-shard-size=1 --no-use-torch-compile -i "$WORK/cosmos_in.jsonl" -o "$WORK/out" --no-guardrails --seed=0 2>&1 | tee "$WORK/infer.log"
# 6. pack results to send back
cd "$WORK"; find out -name '*.mp4' | wc -l; zip -qr "$HERE/cosmos_results.zip" out infer.log cosmos_in.jsonl; echo "RESULTS: $HERE/cosmos_results.zip"
