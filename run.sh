#!/usr/bin/env bash
# Cosmos-H-Surgical baseline (image + text -> video) for the WhatIf-Surg comparison.
#   bash run.sh            -> env (env.sh) + weights + all 32 jobs + cosmos_results.zip
#   SMOKE=1 bash run.sh    -> same, but only the first job (use this first: ~10 min after the env is built)
# Needs: 1 GPU with >= 40 GB (tested on Colab A100-40GB, 2026-11-13), driver >= 570, git, ~70 GB disk, internet (GitHub, PyPI, Hugging Face).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; export WORK=${WORK:-$HERE/work}; mkdir -p "$WORK"; WORK="$(cd "$WORK" && pwd)"
export HF_HOME=${HF_HOME:-$WORK/hf_home}

# 1. environment (idempotent; see env.sh for the fixes it applies)
bash "$HERE/env.sh"
source "$WORK/activate.sh"
cd "$WORK/Cosmos-H-Surgical"

# 2. weights: base checkpoint only (the DMD2 transfer student is not needed for image2video)
python - <<'P'
from huggingface_hub import snapshot_download
import os
p = snapshot_download("nvidia/Cosmos-H-Surgical", local_dir=os.environ["HF_HOME"] + "/Cosmos-H-Surgical", allow_patterns=["*.json", "*.md", "*.txt", "*.sha256", "LICENSE*", "NOTICE*", "model-*.safetensors", "*.jinja", "tokenizer*", "vocab*", "merges*"], max_workers=8)
print("weights at", p)
P

# 3. inputs: 32 jobs (20 factual windows + 12 counterfactual swaps), frame 0 + action text. SMOKE=N keeps the first N.
python "$HERE/inputs/make_jsonl.py" "$WORK/cosmos_in.jsonl"
if [ -n "${SMOKE:-}" ]; then head -n "$SMOKE" "$WORK/cosmos_in.jsonl" > "$WORK/cosmos_in.smoke.jsonl"; IN="$WORK/cosmos_in.smoke.jsonl"; echo "SMOKE: $SMOKE job(s)"; else IN="$WORK/cosmos_in.jsonl"; fi

# 4. inference (one GPU)
mkdir -p "$WORK/out"
torchrun --nproc-per-node=1 -m cosmos_h_surgical infer --checkpoint-path "$HF_HOME/Cosmos-H-Surgical" --parallelism-preset=latency --dp-shard-size=1 --no-use-torch-compile -i "$IN" -o "$WORK/out" --no-guardrails --seed=0 2>&1 | tee "$WORK/infer.log"

# 5. pack results to send back
cd "$WORK"; NV=$(find out -name '*.mp4' | wc -l); echo "videos: $NV"
[ "$NV" -gt 0 ] || { echo "no videos were written; see $WORK/infer.log"; exit 1; }
python - "$HERE/cosmos_results.zip" <<'P'
import sys, zipfile, pathlib
with zipfile.ZipFile(sys.argv[1], "w", zipfile.ZIP_DEFLATED) as z:
    for p in [*pathlib.Path("out").rglob("*"), pathlib.Path("infer.log"), *pathlib.Path(".").glob("cosmos_in*.jsonl")]:
        if p.is_file(): z.write(p)
P
echo "RESULTS: $HERE/cosmos_results.zip"
