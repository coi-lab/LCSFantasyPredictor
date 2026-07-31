# Weekly operations runbook

Updating the points:
python data_pipeline/export_dashboard_data.py

Run included server:
python dashboard/server.py

http://localhost:8050


Every week we need to run this:
1. Re-download current year's Oracle's Elixir match file into data/raw/oracles_elixir/:
   2026_LoL_esports_match_data_from_OraclesElixir.csv
   (Do NOT delete historical 2023-2025 files!)

2. Refresh 24-Hour Riot API Key in '.env':
   - Copy key from https://developer.riotgames.com/
   - Set RIOT_API_KEY=RGAPI-... in .env (kept secure & git-ignored)

3. Activate the virtual environment, then run market snapshot & champion predictor:
   - PowerShell: `.venv\Scripts\Activate.ps1`
   - bash/zsh: `source .venv/bin/activate`
   python data_pipeline/snapshot_official_market.py
   python -m champion_prediction.simple_predictor
   python data_pipeline/export_dashboard_data.py

4. Run included server:
   python dashboard/server.py
   (http://localhost:8050)

Notes:
- Player pricing in gold: verify formula for high values (hard to reach above 30).
- Champion predictor outputs current_champion_rankings.csv and current_champion_portfolio.csv (1.3x floor, 1.5x adoption, 1.7x wildcard picks).
