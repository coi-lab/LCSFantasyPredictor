import json
import os
import subprocess
import hashlib

def file_hash(path):
    if not os.path.exists(path): return None
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()

def run_cmd(cmd):
    try:
        subprocess.check_call(cmd, shell=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running {cmd}: {e}")

inventory_path = "/home/raymondw/Documents/RWorkspace/LCSFantasy/.agent-runs/player-model-v2-stage-6r-runtime-integration-20260807/stage-6r-repository-file-inventory.json"
plan_path = "/home/raymondw/Documents/RWorkspace/LCSFantasy/.agent-runs/player-model-v2-stage-6r-runtime-integration-20260807/stage-6r-file-relocation-plan.json"
results_path = "/home/raymondw/Documents/RWorkspace/LCSFantasy/.agent-runs/player-model-v2-stage-6r-runtime-integration-20260807/stage-6r-file-relocation-results.json"

with open(inventory_path, "r") as f:
    inventory = json.load(f)

plan = []
for item in inventory:
    src = item["path"]
    dest_dir = item["safe_destination"]
    if dest_dir == "UNKNOWN":
        continue
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, os.path.basename(src))
    plan.append({
        "old_path": src,
        "new_path": dest,
        "classification": item["classification"],
        "reason": item["recommended_action"],
        "tracked": item["git_tracking_status"] == "tracked"
    })

with open(plan_path, "w") as f:
    json.dump(plan, f, indent=2)

results = []
for p in plan:
    src = p["old_path"]
    dest = p["new_path"]
    h_before = file_hash(src)
    
    if p["tracked"]:
        run_cmd(f"git mv {src} {dest}")
    else:
        run_cmd(f"mv {src} {dest}")
    
    h_after = file_hash(dest)
    
    results.append({
        "old_path": src,
        "new_path": dest,
        "classification": p["classification"],
        "reason": p["reason"],
        "hash_before": h_before,
        "hash_after": h_after
    })

with open(results_path, "w") as f:
    json.dump(results, f, indent=2)

print("Files relocated.")
