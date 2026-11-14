"""jobs.json + videos/{A,B} -> Cosmos-H-Surgical video2video (transfer) jsonl. Blur control (preset medium) computed by Cosmos from the vision video;
frame 0 of the vision video is the conditioning frame (num_first_chunk_conditional_frames = 1); 65 frames, 16 fps, 480p; base checkpoint, 50 steps,
guidance 1.5, control guidance 1.0 (the release's validated transfer settings), seed 1. Same JSON prompt as the image2video run.
usage: make_jsonl.py <out.jsonl> <versions, e.g. ABC> [comma-separated job ids]"""
import json, os, sys
D = os.path.dirname(os.path.abspath(__file__)); jobs = json.load(open(f"{D}/jobs.json")); out = open(sys.argv[1], "w"); vers = sys.argv[2] if len(sys.argv) > 2 else "ABC"
keep = set(sys.argv[3].split(",")) if len(sys.argv) > 3 and sys.argv[3] else None; n = 0
for v in vers:
    for j in jobs:
        if keep and j["id"] not in keep: continue
        pr = json.dumps({"subjects": [{"description": j["tool_desc"], "action": j["action"], "location": "contact with the tissue in the centre of the view"}], "scene": "laparoscopic cholecystectomy, endoscopic view of the gallbladder and liver"})
        out.write(json.dumps({"name": f"{v}_{j['id']}", "model_mode": "video2video", "prompt": pr, "vision_path": f"{D}/videos/{v}/{j['id']}.mp4", "resolution": "480", "aspect_ratio": "16,9",
            "fps": 16, "num_frames": 65, "num_video_frames_per_chunk": 65, "num_conditional_frames": 1, "num_first_chunk_conditional_frames": 1, "share_vision_temporal_positions": True,
            "resize_mode": "stretch", "num_steps": 50, "guidance": 1.5, "control_guidance": 1.0, "shift": 5.0, "seed": j["seed"], "blur": {"preset_blur_strength": "medium"}}) + "\n"); n += 1
print("jobs", n)
