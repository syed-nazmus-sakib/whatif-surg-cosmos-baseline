"""2026-11-14 user: Cosmos-H-Surgical video2video (transfer, blur control) baselines with the source video as context, three versions:
  A (full):    vision video = the full real clip WITH the source tool (CF: frame 0 = our composed CF frame 0 ctx[0], the frame 0 every baseline gets).
  B (outside): vision video = real clip with the SOURCE tool (instrument seg `tm`, dilated 7 px) blacked out on frames 1..64. Frame 0 = the same frame 0 as A.
               Our edit mask is NOT used: it is source U target CAD silhouette, so its outline would hand Cosmos the CAD shape (checked: blades visible in the hole).
  C (outside + CAD): vision video = OUR model's own control video `ctx` (real clip outside the edit mask, our coloured CAD guide inside it; frame 0 = the same frame 0).
               This hands Cosmos our CAD fit/guide (= our contribution): it isolates the generator from the CAD pipeline.
Factual: the 12 benchmark windows (real tool in the prompt); CF: the 12 common swaps (target tool in the prompt). Same JSON prompt as the image2video run. Run on rmedu-4090 (CPU), output inputs/{videos,jobs.json}."""
import sys, os, json, subprocess, numpy as np, cv2
sys.path.insert(0, "/media/rmedu-4090/Storage_4/counterfactual/cf_surgvid/local_edit/eval_v1/bench"); import bench_common as C
D = os.path.dirname(os.path.abspath(__file__)); B = C.B
DESC = {j["tool"]: (j["tool_desc"], j["action"]) for j in json.load(open(f"{B}/cosmos_pkg/inputs/jobs.json"))}
def write(path, F):
    p = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", "832x480", "-r", "16", "-i", "-", "-c:v", "libx264", "-crf", "12", "-pix_fmt", "yuv420p", path], stdin=subprocess.PIPE)
    p.stdin.write(np.ascontiguousarray(F).tobytes()); p.stdin.close(); assert p.wait() == 0
def hole(real, f0, tm):
    out = real.copy(); out[0] = f0; k = np.ones((15, 15), np.uint8)
    for t in range(1, len(out)):
        h = cv2.dilate(tm[min(t, len(tm) - 1)].astype(np.uint8), k) > 0; out[t][h] = 0
    return out
jobs = []
for j in C.factual_jobs():
    z = np.load(f"{j['src']}/in/{j['w']}.npz"); real = z["real"][:65]; tm = np.load(C.segs()[j["w"]])["tm"] > 0
    jobs.append(dict(id=j["id"], set="factual", tool=j["tool"], f0=real[0], A=real, B=hole(real, real[0], tm), C=z["ctx"][:65]))
for x in C.cf_jobs(common_only=True):
    z = np.load(f"{x['dir']}/in/{x['key']}.npz"); real = z["real"][:65].copy(); f0 = z["ctx"][0]; tm = np.load(C.segs()[x["w"]])["tm"] > 0
    A = real.copy(); A[0] = f0; jobs.append(dict(id=f"cf_{x['key']}", set="cf", tool=x["tgt"], f0=f0, A=A, B=hole(real, f0, tm), C=z["ctx"][:65]))
for v in "ABC": os.makedirs(f"{D}/videos/{v}", exist_ok=True)
meta = []
for j in jobs:
    assert j["A"].shape == (65, 480, 832, 3), j["A"].shape
    if j["set"] == "cf": assert (j["C"][0] == j["f0"]).all(), j["id"]   # factual ctx[0] = our model's re-encoded frame 0 (1.7-6 grey from real[0]); C keeps it (= our input)
    for v in "ABC": write(f"{D}/videos/{v}/{j['id']}.mp4", j[v])
    d, a = DESC[j["tool"]]; meta.append(dict(id=j["id"], set=j["set"], tool=j["tool"], tool_desc=d, action=a, seed=1)); print(j["id"], flush=True)
json.dump(meta, open(f"{D}/jobs.json", "w"), indent=0); print("jobs", len(meta))
