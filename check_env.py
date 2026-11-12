"""Self-test for the Cosmos-H-Surgical env: every import the inference path needs + one transformer-engine GPU kernel."""
import sys

def step(name, fn):
    try:
        r = fn(); print(f"  ok   {name}{': ' + str(r) if r is not None else ''}")
    except Exception as e:  # noqa: BLE001
        print(f"  FAIL {name}: {type(e).__name__}: {e}"); sys.exit(1)

print("[check_env]")
import torch  # noqa: E402
step("torch", lambda: f"{torch.__version__}, CUDA {torch.version.cuda}")
step("GPU visible", lambda: (_ for _ in ()).throw(RuntimeError("torch.cuda.is_available() is False")) if not torch.cuda.is_available() else torch.cuda.get_device_name(0))
mem = torch.cuda.get_device_properties(0).total_memory / 2**30
step("GPU memory", lambda: f"{mem:.0f} GiB" + ("  (WARNING: < 40 GiB, the 15.8 B model will likely run out of memory)" if mem < 39 else ""))
step("transformer_engine (cudart lookup)", lambda: __import__("transformer_engine").__version__)
import transformer_engine.pytorch as te  # noqa: E402

def te_kernel():
    lin = te.Linear(256, 256, params_dtype=torch.bfloat16).cuda()
    y = lin(torch.randn(32, 256, device="cuda", dtype=torch.bfloat16)); torch.cuda.synchronize()
    return tuple(y.shape)

step("transformer_engine GPU kernel", te_kernel)
step("flash_attn", lambda: __import__("flash_attn").__version__)
step("natten", lambda: __import__("natten").__version__)
step("megatron.core", lambda: __import__("megatron.core", fromlist=["x"]).__name__)
step("cosmos_framework inference args", lambda: (__import__("cosmos_framework.inference.args", fromlist=["x"]), None)[1])
step("cosmos_h_surgical", lambda: (__import__("cosmos_h_surgical.checkpoints", fromlist=["x"]), None)[1])
print("[check_env] all good")
