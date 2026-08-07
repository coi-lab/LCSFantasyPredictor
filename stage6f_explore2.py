import pandas as pd
import numpy as np

cw = pd.read_csv(".agent-runs/player-model-v2-stage-6e-pricing-budget-audit-20260807/stage-6e-player-price-crosswalk.csv")
r23 = pd.read_csv(".agent-runs/player-model-v2-stage-6e-pricing-budget-audit-20260807/stage-6e-r2-r3-official-transition.csv")
r23 = r23.merge(cw, on='pro_player_id', how='inner')

print(r23[r23['round3_name'].isin(['Inspired', 'Zven', 'APA'])][['round3_name', 'price_r2', 'price_r3', 'last_round_score_r3', 'official_price_change']])

