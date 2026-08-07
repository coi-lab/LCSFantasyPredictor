import os

file_path = "/home/raymondw/Documents/RWorkspace/LCSFantasy/data_pipeline/official_prices.py"
with open(file_path, "r") as f:
    content = f.read()

new_budget_func = """
def calculate_next_budget(current_unspent: float, held_roster_next_value: float) -> float:
    \"\"\"Calculate the next round's budget deterministically.
    
    Equivalent to:
    next_budget = current_unspent + sum(next_round_price(asset) for asset in currently_held_roster)
    \"\"\"
    return round(current_unspent + held_roster_next_value, 2)

"""

# Insert right after resolve_price
content = content.replace("return None, PriceProvenance.UNAVAILABLE", "return None, PriceProvenance.UNAVAILABLE\n" + new_budget_func)

with open(file_path, "w") as f:
    f.write(content)
