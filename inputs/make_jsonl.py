"""jobs.json -> Cosmos-H-Surgical image2video input jsonl (absolute image paths). Same frame 0 and the same action text as all other baselines; 65 frames at 16 fps (= our 4 s windows), 480p."""
import json, os, sys
D = os.path.dirname(os.path.abspath(__file__)); jobs = json.load(open(f"{D}/jobs.json")); out = open(sys.argv[1] if len(sys.argv) > 1 else f"{D}/cosmos_in.jsonl", "w")
for j in jobs:
    pr = json.dumps({"subjects": [{"description": j["tool_desc"], "action": j["action"], "location": "contact with the tissue in the centre of the view"}], "scene": "laparoscopic cholecystectomy, endoscopic view of the gallbladder and liver"})
    out.write(json.dumps({"name": j["id"], "model_mode": "image2video", "prompt": pr, "vision_path": f"{D}/{j['image']}", "resolution": "480", "aspect_ratio": "16,9", "fps": 16, "num_frames": 65, "guidance": 6.0, "shift": 5.0, "seed": j["seed"]}) + "\n")
print("jobs", len(jobs))
