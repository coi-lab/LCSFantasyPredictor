import unittest
import re
from pathlib import Path

class TestRepositoryRootHygiene(unittest.TestCase):
    def test_root_directory_hygiene(self):
        # Resolve repository root conceptually from this file's location
        root = Path(__file__).resolve().parents[1]
        
        # Enforce a small explicit allowlist of legitimate root files/folders
        allowlist = {
            ".env",
            ".env.example",
            ".gitignore",
            "AGENTS.md",
            "IDEAS.md",
            "README.md",
            "project-skills.md",
            "requirements.txt",
            ".git",
            ".agent-runs",
            ".agents",
            ".codex",
            ".venv",
            "champion_prediction",
            "data_pipeline",
            "fantasy_prediction",
            "learning",
            "rag",
            "tests",
            "tools",
            "docs",
            "config",
            "dashboard",
            "data",
            "scripts",
            "skills",
            "reports",
            "prompts",
            "analysis",
            "LCSFantasyImages",
            "__pycache__"
        }
        
        prohibited_patterns = [
            r"^stage.*\.py$",
            r"^scratch.*\.py$",
            r"^make_.*\.py$",
            r"^generate_.*\.py$",
            r"^.*_build\.py$",
            r"^.*_fit\.py$",
            r"^.*_handoff\.py$",
            r"^.*_reproduce\.py$",
            r"^.*_audit\.py$",
            r"^.*_recovery\.py$",
            r"^debug.*\.py$",
            r"^temp.*\.py$",
            r"^tmp.*\.py$",
            r"^.*\.json$",
            r"^.*\.csv$",
            r"^.*\.log$",
            r"^.*\.txt$"
        ]
        
        offending_paths = []
        
        # List all entries in the root
        for item in root.iterdir():
            name = item.name
            
            # Check if name is in the allowlist
            if name in allowlist:
                continue
                
            # Check for ignored files like python cache or similar
            if name.startswith("."):
                # Hidden files (e.g. .DS_Store, .project) not in allowlist
                offending_paths.append(str(item.relative_to(root)))
                continue
                
            # If item is a file, check for prohibited patterns
            if item.is_file():
                is_prohibited = False
                for pattern in prohibited_patterns:
                    if re.match(pattern, name, re.IGNORECASE):
                        is_prohibited = True
                        break
                if is_prohibited:
                    offending_paths.append(str(item.relative_to(root)))
                else:
                    # Not matching a prohibited pattern, but not in allowlist
                    offending_paths.append(str(item.relative_to(root)))
            else:
                # Directory not in allowlist
                offending_paths.append(str(item.relative_to(root)))
                
        if offending_paths:
            print("\n!!! Offending root level entries found !!!")
            for path in sorted(offending_paths):
                print(f"  - {path}")
            self.fail(f"Unexpected root level files or directories found: {sorted(offending_paths)}")

if __name__ == "__main__":
    unittest.main()
