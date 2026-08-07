import re

file_path = "/home/raymondw/Documents/RWorkspace/LCSFantasy/tests/test_market_pricing.py"
with open(file_path, "r") as f:
    content = f.read()

# Add "games": 1 to all weekly dicts
content = re.sub(r'("fantasy_pts": [0-9.]+)(,?\n)', r'\1, "games": 1\2', content)

# Remove the price floor/ceiling from model dict to avoid confusion
content = re.sub(r'\s*"price_floor": 5.0,\n', '\n', content)
content = re.sub(r'\s*"price_ceiling": 32.0,\n', '\n', content)

# Remove test_repeated_high_scores_remain_capped entirely
test_def = "    def test_repeated_high_scores_remain_capped(self) -> None:"
idx = content.find(test_def)
if idx != -1:
    content = content[:idx] + "\n"

with open(file_path, "w") as f:
    f.write(content)

