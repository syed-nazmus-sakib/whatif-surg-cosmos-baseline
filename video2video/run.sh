#!/usr/bin/env bash
# Cosmos-H-Surgical baseline, video2video (transfer, blur control) with the source video as context, for the WhatIf-Surg comparison.
#   bash run.sh                 -> env (env.sh) + weights + all 72 jobs (24 x versions A, B, C) + cosmos_v2v_results.zip
#   VERSIONS=B bash run.sh      -> one version only (A = full real video, B = source tool blacked out, C = our control video: outside the edit mask + our CAD guide)
#   SMOKE=id1,id2 bash run.sh   -> only these job ids (see inputs/jobs.json)
# Needs: 1 GPU with >= 40 GB, or a 24 GB GPU + ~32 GB free RAM in low-VRAM mode (tested on RTX 4090 24 GB, 2026-11-14), driver >= 570, git, ~70 GB disk, internet (GitHub, PyPI, Hugging Face).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; export WORK=${WORK:-$HERE/work}; mkdir -p "$WORK"; WORK="$(cd "$WORK" && pwd)"
export HF_HOME=${HF_HOME:-$WORK/hf_home}

# 1. environment (idempotent; see env.sh for the fixes it applies)
bash "$HERE/env.sh"
source "$WORK/activate.sh"
cd "$WORK/Cosmos-H-Surgical"

# 2. weights: base checkpoint only (50-step transfer; the DMD2 4-step student is not used)
python - <<'P'
from huggingface_hub import snapshot_download
import os
p = snapshot_download("nvidia/Cosmos-H-Surgical", local_dir=os.environ["HF_HOME"] + "/Cosmos-H-Surgical", allow_patterns=["*.json", "*.md", "*.txt", "*.sha256", "LICENSE*", "NOTICE*", "model-*.safetensors", "*.jinja", "tokenizer*", "vocab*", "merges*"], max_workers=8)
print("weights at", p)
P

# 3a. input videos (72 mp4, 197 MB) come from the GitHub release asset when not already present
if [ ! -f "$HERE/inputs/videos/C/cf_A54__33125__to_scissors.mp4" ]; then
  python - "$HERE" <<'Q'
import sys, urllib.request, zipfile, io
url = "https://github.com/syed-nazmus-sakib/whatif-surg-cosmos-baseline/releases/download/v2v-inputs-v1/v2v_inputs.zip"
print("downloading", url); zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(url).read())).extractall(sys.argv[1])
Q
fi
# 3. inputs: 24 jobs (12 factual windows + 12 counterfactual swaps) x versions; vision video + action text
IN="$WORK/cosmos_v2v_in.jsonl"; python "$HERE/inputs/make_jsonl.py" "$IN" "${VERSIONS:-ABC}" "${SMOKE:-}"

# 4. inference (one GPU). GPUs with < 39 GiB (e.g. RTX 4090 24 GB) use patch/lowvram_launch.py: decoder blocks that don't fit stay in CPU RAM
#    (needs ~32 GB free RAM) and are streamed to the GPU per call; same weights and kernels, slower. LOWVRAM=1 forces it, LOWVRAM=0 disables it.
mkdir -p "$WORK/out"
GMEM=$(python -c "import torch; print(int(torch.cuda.get_device_properties(0).total_memory / 2**30))")
if [ "${LOWVRAM:-auto}" = 1 ] || { [ "${LOWVRAM:-auto}" = auto ] && [ "$GMEM" -lt 39 ]; }; then LAUNCH=("$HERE/patch/lowvram_launch.py"); echo "GPU ${GMEM} GiB -> low-VRAM mode"; else LAUNCH=(-m cosmos_h_surgical); fi
torchrun --nproc-per-node=1 "${LAUNCH[@]}" infer --checkpoint-path "$HF_HOME/Cosmos-H-Surgical" --parallelism-preset=latency --dp-shard-size=1 --no-use-torch-compile -i "$IN" -o "$WORK/out" --no-guardrails --seed=0 2>&1 | tee "$WORK/infer.log"

# 5. pack results to send back
cd "$WORK"; NV=$(find out -name '*.mp4' | wc -l); echo "videos: $NV"
[ "$NV" -gt 0 ] || { echo "no videos were written; see $WORK/infer.log"; exit 1; }
python - "$HERE/cosmos_v2v_results.zip" <<'P'
import sys, zipfile, pathlib
with zipfile.ZipFile(sys.argv[1], "w", zipfile.ZIP_DEFLATED) as z:
    for p in [*pathlib.Path("out").rglob("*"), pathlib.Path("infer.log"), *pathlib.Path(".").glob("cosmos_v2v_in*.jsonl")]:
        if p.is_file(): z.write(p)
P
echo "RESULTS: $HERE/cosmos_v2v_results.zip"
