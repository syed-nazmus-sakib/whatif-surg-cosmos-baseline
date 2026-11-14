#!/usr/bin/env bash
# Prepare the Cosmos-H-Surgical environment. Safe to re-run: finished steps are skipped.
#   bash env.sh        -> builds $WORK/Cosmos-H-Surgical/.venv, writes $WORK/activate.sh, runs a GPU self-test
# Settings (all optional): WORK (default ./work), VENV_DIR (default $WORK/Cosmos-H-Surgical/.venv), CUDA_GROUP (cu128 | cu130)
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; WORK=${WORK:-$HERE/work}; mkdir -p "$WORK"; WORK="$(cd "$WORK" && pwd)"
CUDA_GROUP=${CUDA_GROUP:-cu128}
say() { echo "[env.sh] $*"; }

# 0. prerequisites
command -v git >/dev/null || { echo "git is required"; exit 1; }
command -v nvidia-smi >/dev/null || { echo "nvidia-smi not found: no NVIDIA driver on this machine"; exit 1; }
DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1); say "driver $DRV, GPU $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)"
if [ "$CUDA_GROUP" = cu128 ] && [ "${DRV%%.*}" -lt 570 ]; then say "WARNING: driver $DRV < 570 (CUDA 12.8 wants >= 570); the GPU self-test below decides"; fi
if [ "$CUDA_GROUP" = cu130 ] && [ "${DRV%%.*}" -lt 580 ]; then say "WARNING: driver $DRV < 580 (CUDA 13.0 wants >= 580); the GPU self-test below decides"; fi
if ! command -v uv >/dev/null; then
  [ -x "$HOME/.local/bin/uv" ] || { say "installing uv"; curl -LsSf https://astral.sh/uv/install.sh | sh; }
  export PATH="$HOME/.local/bin:$PATH"
fi
command -v zip >/dev/null || say "WARNING: 'zip' not found; run.sh will fall back to python's zipfile"

# 1. git-lfs is not needed (the repos' LFS files are NVIDIA's example media). Disable the LFS filter for every git call,
#    including the ones uv makes when it fetches cosmos-framework / Megatron-LM, so a global lfs config without git-lfs can't break them.
export GIT_LFS_SKIP_SMUDGE=1 GIT_CONFIG_COUNT=3 GIT_CONFIG_KEY_0=filter.lfs.process GIT_CONFIG_VALUE_0= GIT_CONFIG_KEY_1=filter.lfs.smudge GIT_CONFIG_VALUE_1= GIT_CONFIG_KEY_2=filter.lfs.required GIT_CONFIG_VALUE_2=false

# 2. code (pinned release)
cd "$WORK"
[ -f Cosmos-H-Surgical/pyproject.toml ] || { rm -rf Cosmos-H-Surgical; git clone --depth 1 --branch v0.3.1 https://github.com/NVIDIA-Medtech/Cosmos-H-Surgical.git; }
cd Cosmos-H-Surgical
VENV_DIR=${VENV_DIR:-$PWD/.venv}; export UV_PROJECT_ENVIRONMENT="$VENV_DIR" UV_LINK_MODE=copy

# 3. python env (uv picks the pinned torch 2.10 / transformer-engine 2.12 / flash-attn wheels)
if [ ! -f "$VENV_DIR/.whatif_env_ok" ]; then
  uv cache clean cosmos-framework >/dev/null 2>&1 || true
  uv sync --group "$CUDA_GROUP"
fi
PY="$VENV_DIR/bin/python"; SP=$("$PY" -c "import sysconfig; print(sysconfig.get_path('purelib'))")

# 4. FIX for "RuntimeError: cudart shared object not found" (transformer_engine/common/__init__.py).
#    On a machine without a system CUDA toolkit TE looks for the pip CUDA runtime in nvidia/cudart or nvidia/cuda_cudart,
#    but the nvidia-cuda-runtime-cu12 wheel installs it as nvidia/cuda_runtime. (a) symlink the name TE expects;
#    (b) put every pip CUDA lib dir on LD_LIBRARY_PATH and point TE's NVRTC/CURAND/CUDNN lookups at them, so a different
#    system CUDA (e.g. a cluster module with CUDA 12.2) is never mixed with the venv's CUDA 12.8 libraries.
N="$SP/nvidia"
if [ -d "$N/cuda_runtime" ] && [ ! -e "$N/cudart" ]; then ln -s cuda_runtime "$N/cudart" 2>/dev/null || cp -r "$N/cuda_runtime" "$N/cudart"; fi
cat > "$WORK/activate.sh" <<EOF
# source this before running anything in the Cosmos env (written by env.sh)
source "$VENV_DIR/bin/activate"
export LD_LIBRARY_PATH="$(ls -d "$N"/*/lib "$N"/cu13/lib 2>/dev/null | tr '\n' ':')\${LD_LIBRARY_PATH:-}"
export NVRTC_HOME="$N/cuda_nvrtc" CURAND_HOME="$N/curand" CUDNN_HOME="$N/cudnn"
export GIT_LFS_SKIP_SMUDGE=1 GIT_CONFIG_COUNT=3 GIT_CONFIG_KEY_0=filter.lfs.process GIT_CONFIG_VALUE_0= GIT_CONFIG_KEY_1=filter.lfs.smudge GIT_CONFIG_VALUE_1= GIT_CONFIG_KEY_2=filter.lfs.required GIT_CONFIG_VALUE_2=false
export PYTORCH_CUDA_ALLOC_CONF=\${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
EOF
source "$WORK/activate.sh"

# 5. the release's checkpoint helper calls `uvx hf download --format=json`, which fails (option rejected) -> huggingface_hub API version
if [ -z "${NO_PATCH:-}" ]; then
  T=$("$PY" -c "import importlib.util as u; print(u.find_spec('cosmos_framework.utils.checkpoint_db').origin)")
  [ -f "$T.orig" ] || cp "$T" "$T.orig"; cp "$HERE/patch/checkpoint_db_patched.py" "$T"
fi

# 6. self-test: imports + a real transformer-engine kernel on the GPU
"$PY" "$HERE/check_env.py"
touch "$VENV_DIR/.whatif_env_ok"
say "OK. Activate with: source $WORK/activate.sh"
