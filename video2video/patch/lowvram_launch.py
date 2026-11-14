"""Low-VRAM launcher for Cosmos-H-Surgical inference on GPUs with < 40 GB (tested: RTX 4090, 24 GB).
The 15.8 B model (31.5 GB bf16) does not fit a 24 GB card. This keeps the first decoder blocks resident on the GPU (budget
COSMOS_LOWVRAM_GPU_GB, default 8 GB) and leaves the rest in CPU RAM; each offloaded block is copied to the GPU right before
its forward and dropped right after (weights are read-only at inference, so nothing is copied back). Numerics are unchanged:
the same bf16 weights run the same kernels, only their storage location differs. Slower (PCIe traffic per denoiser call).
usage (same arguments as `python -m cosmos_h_surgical`):  torchrun --nproc-per-node=1 patch/lowvram_launch.py infer ...
Not needed on >= 40 GB GPUs (run.sh uses it only when the GPU has < 39 GiB, or LOWVRAM=1)."""
import os
import sys

import torch

GPU_GB = float(os.environ.get("COSMOS_LOWVRAM_GPU_GB", "8"))


def _nbytes(m):   # sizes at bf16 (blocks are still fp32 on the meta device when this runs; the model is cast to bf16 before loading)
    return sum(t.numel() * (2 if t.is_floating_point() else t.element_size()) for t in list(m.parameters()) + list(m.buffers()))


def _install(net):
    layers = net.language_model.model.layers
    blocks = list(layers.children()) if not isinstance(layers, (list, tuple)) else list(layers)
    keep, used = 0, 0.0
    for b in blocks:
        s = _nbytes(b) / 2**30
        if used + s > GPU_GB:
            break
        used += s; keep += 1
    off = blocks[keep:]
    print(f"[lowvram] {len(blocks)} decoder blocks: {keep} resident on GPU ({used:.1f} GB), {len(off)} streamed from CPU "
          f"({sum(_nbytes(b) for b in off) / 2**30:.1f} GB)", flush=True)

    def to_empty(*, device, recurse=True):
        for b in off:   # offloaded blocks: real (uninitialised) CPU storage; the checkpoint load fills it in place
            b.to_empty(device="cpu", recurse=recurse)
        net._apply(lambda t: torch.empty_like(t, device=device) if t.is_meta else t)
        return net

    net.to_empty = to_empty

    def pre(mod, args, kwargs):
        for t in list(mod.parameters()) + list(mod.buffers()):   # decided at call time: to_empty / the checkpoint load replace tensor objects
            if not hasattr(t, "_lv_cpu"):
                if t.device.type != "cpu":   # e.g. buffers initialised on the GPU: leave them there
                    continue
                try:
                    t._lv_cpu = t.data.pin_memory()
                except RuntimeError:
                    t._lv_cpu = t.data
            t.data = t._lv_cpu.to("cuda", non_blocking=True)
        return None

    def post(mod, args, kwargs, out):
        for t in list(mod.parameters()) + list(mod.buffers()):
            if hasattr(t, "_lv_cpu"):
                t.data = t._lv_cpu
        return out

    for b in off:
        b.register_forward_pre_hook(pre, with_kwargs=True)
        b.register_forward_hook(post, with_kwargs=True)
        if hasattr(b, "reasoner_forward"):   # AR text path (prompt upsampling); same treatment
            f = b.reasoner_forward

            def rf(*a, _f=f, _b=b, **k):
                pre(_b, a, k); o = _f(*a, **k); return post(_b, a, k, o)

            b.reasoner_forward = rf
    return net


def patch():
    import cosmos_framework.model.generator.omni_mot_model as omm
    orig = omm.parallelize_vfm_network

    def parallelize_vfm_network(net, *a, **k):
        return _install(orig(net, *a, **k))

    omm.parallelize_vfm_network = parallelize_vfm_network


if __name__ == "__main__":
    patch()
    from cosmos_h_surgical.cli import main
    raise SystemExit(main(sys.argv[1:]))
