import os
import glob
import subprocess
import json
import datetime

root_dir = "/home/raymondw/Documents/RWorkspace/LCSFantasy"
inventory_path = os.path.join(root_dir, ".agent-runs", "player-model-v2-stage-6r-runtime-integration-20260807", "stage-6r-repository-file-inventory.json")

def run_cmd(cmd):
    try:
        return subprocess.check_output(cmd, shell=True, text=True, stderr=subprocess.STDOUT).strip()
    except subprocess.CalledProcessError as e:
        return ""

def check_references(filename):
    refs = {"tests": False, "docs": False, "skills": False}
    if run_cmd(f"git grep -l {filename} tests/"): refs["tests"] = True
    if run_cmd(f"git grep -l {filename} docs/"): refs["docs"] = True
    if run_cmd(f"git grep -l {filename} .agents/skills/"): refs["skills"] = True
    if run_cmd(f"git grep -l {filename} config/"): refs["config"] = True
    return refs

inventory = []

for py_file in glob.glob("*.py"):
    # get imports
    with open(py_file, "r") as f:
        content = f.read()
    imports = [line.strip() for line in content.split("\n") if line.startswith("import ") or line.startswith("from ")]
    
    tracked = py_file in run_cmd("git ls-files")
    status = run_cmd(f"git status --short {py_file}")
    last_mod = datetime.datetime.fromtimestamp(os.path.getmtime(py_file)).isoformat()
    
    refs = check_references(py_file)
    
    # Simple classification heuristic
    classification = "UNKNOWN_REQUIRES_REVIEW"
    if "stage6" in py_file:
        classification = "STAGE_EVIDENCE_RUNNER"
    elif "scratch" in py_file:
        classification = "SCRATCH_OR_TRANSIENT"
    elif "make_" in py_file or "generate_" in py_file:
        classification = "REUSABLE_TOOLING"
        
    rec_action = "MOVE_TO_EVIDENCE_ARCHIVE"
    if classification == "REUSABLE_TOOLING":
        rec_action = "MOVE_TO_TOOLS"

    inventory.append({
        "path": py_file,
        "classification": classification,
        "imports": imports,
        "callers": refs,
        "git_tracking_status": "tracked" if tracked else "untracked",
        "git_status": status,
        "last_modified": last_mod,
        "referenced_by_tests": refs["tests"],
        "referenced_by_docs": refs["docs"],
        "referenced_by_skills": refs["skills"],
        "safe_destination": ".agent-runs/player-model-v2-stage-6r-runtime-integration-20260807/tools/" if classification == "STAGE_EVIDENCE_RUNNER" or classification == "SCRATCH_OR_TRANSIENT" else "tools/" if classification == "REUSABLE_TOOLING" else "UNKNOWN",
        "recommended_action": rec_action
    })

with open(inventory_path, "w") as f:
    json.dump(inventory, f, indent=2)

print("Inventory generated.")
