import pandas as pd
import numpy as np

cw = pd.read_csv(".agent-runs/player-model-v2-stage-6e-pricing-budget-audit-20260807/stage-6e-player-price-crosswalk.csv")
r23 = pd.read_csv(".agent-runs/player-model-v2-stage-6e-pricing-budget-audit-20260807/stage-6e-r2-r3-official-transition.csv")
r23 = r23.merge(cw, on='pro_player_id', how='inner')

# Filter out the 3 missing players
played = r23[r23['last_round_score_r3'] > 0.0].copy()

played['delta'] = played['price_r3'] - played['price_r2']

a, b, c = 0.747528, 0.239998, 0.015874

# Method A: round(a*P + b*S + c, 1)
played['A'] = np.round(a * played['price_r2'] + b * played['last_round_score_r3'] + c, 1)

# Method B: round price change
# Wait, original formula: P_next = 0.75 P + 0.24 S
# which means delta_P = -0.25 P + 0.24 S
played['B'] = played['price_r2'] + np.round(-0.252472 * played['price_r2'] + b * played['last_round_score_r3'] + c, 1)

# Let's see residuals
played['res_A'] = played['price_r3'] - played['A']
played['res_B'] = played['price_r3'] - played['B']

print(played[['round3_name', 'price_r2', 'last_round_score_r3', 'price_r3', 'A', 'res_A', 'B', 'res_B']].sort_values('res_A', ascending=False).head(15))

