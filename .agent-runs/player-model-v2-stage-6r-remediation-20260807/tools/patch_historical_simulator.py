import os

file_path = "/home/raymondw/Documents/RWorkspace/LCSFantasy/fantasy_prediction/historical_simulator.py"
with open(file_path, "r") as f:
    content = f.read()

# Replace SyntheticPriceModel
old_class = """@dataclass(frozen=True)
class SyntheticPriceModel:
    \"\"\"Predeclared score/price transition used only for scenario analysis.\"\"\"

    starting_price: float = 15.0
    previous_price_weight: float = 1.0
    score_weight: float = 0.0
    intercept: float = 0.0
    floor: float = 5.0
    ceiling: float = 32.0
    decimals: int = 1

    def update(self, previous_price: float, actual_points: float) -> float:
        value = (
            self.previous_price_weight * previous_price
            + self.score_weight * actual_points
            + self.intercept
        )
        return round(min(self.ceiling, max(self.floor, value)), self.decimals)"""

new_class = """@dataclass(frozen=True)
class SyntheticPriceModel:
    \"\"\"Predeclared score/price transition used only for scenario analysis.\"\"\"

    starting_price: float = 15.0
    decimals: int = 1

    def update(self, previous_price: float, actual_points: float, did_participate: bool = True) -> float:
        from data_pipeline.official_prices import reconstruct_price
        return reconstruct_price(previous_price, actual_points, did_participate)"""

content = content.replace(old_class, new_class)

# Replace budget calculation
old_budget = """        held_asset_change = round(
            sum(next_prices[player_id] - current_prices[player_id] for player_id in chosen),
            2,
        )
        next_budget = round(budget + held_asset_change, 2)"""

new_budget = """        from data_pipeline.official_prices import calculate_next_budget
        held_asset_change = round(
            sum(next_prices[player_id] - current_prices[player_id] for player_id in chosen),
            2,
        )
        next_budget = calculate_next_budget(
            round(budget - roster_cost, 2),
            round(sum(next_prices[player_id] for player_id in chosen), 2)
        )"""

content = content.replace(old_budget, new_budget)

with open(file_path, "w") as f:
    f.write(content)
