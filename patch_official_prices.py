import os

file_path = "/home/raymondw/Documents/RWorkspace/LCSFantasy/data_pipeline/official_prices.py"
with open(file_path, "r") as f:
    content = f.read()

new_imports_and_constants = """
from enum import Enum

class PriceProvenance(Enum):
    OFFICIAL_SNAPSHOT = "OFFICIAL_SNAPSHOT"
    OFFICIAL_EMBEDDED_PREVIOUS_PRICE = "OFFICIAL_EMBEDDED_PREVIOUS_PRICE"
    RECONSTRUCTED_STAGE_6F = "RECONSTRUCTED_STAGE_6F"
    UNAVAILABLE = "UNAVAILABLE"

RECONSTRUCTED_PRICE_WEIGHT = 0.747528
RECONSTRUCTED_SCORE_WEIGHT = 0.239998
RECONSTRUCTED_INTERCEPT = 0.015874

def reconstruct_price(previous_price: float, last_round_score: float, did_participate: bool) -> float:
    \"\"\"Calculate Stage 6F piecewise reconstructed simulation price.\"\"\"
    if not did_participate:
        return previous_price
    return round(
        RECONSTRUCTED_PRICE_WEIGHT * previous_price
        + RECONSTRUCTED_SCORE_WEIGHT * last_round_score
        + RECONSTRUCTED_INTERCEPT,
        1
    )

def resolve_price(
    official_snapshot_price: float | None = None,
    official_embedded_previous_price: float | None = None,
    reconstructed_price: float | None = None,
) -> tuple[float | None, PriceProvenance]:
    \"\"\"Resolve price using exact precedence rules.\"\"\"
    if official_snapshot_price is not None:
        return official_snapshot_price, PriceProvenance.OFFICIAL_SNAPSHOT
    if official_embedded_previous_price is not None:
        return official_embedded_previous_price, PriceProvenance.OFFICIAL_EMBEDDED_PREVIOUS_PRICE
    if reconstructed_price is not None:
        return reconstructed_price, PriceProvenance.RECONSTRUCTED_STAGE_6F
    return None, PriceProvenance.UNAVAILABLE
"""

# Insert right before load_official_price_history
content = content.replace("def load_official_price_history(", new_imports_and_constants + "\ndef load_official_price_history(")

with open(file_path, "w") as f:
    f.write(content)
