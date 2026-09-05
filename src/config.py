"""Central configuration. Everything tunable lives here."""

# --- Data source -------------------------------------------------------------
# Public AirNow embassy archive (no API key). dosairnowdata.org is dead; this
# S3 bucket is the live mirror the EPA embassy map itself reads from.
AIRNOW_BASE = (
    "https://s3-us-west-1.amazonaws.com/files.airnowtech.org"
    "/airnow/EmbassyHistorical/Dhaka"
)
YEARS = list(range(2016, 2026))  # 2025 stops at 2025-03-24

RAW_DIR = "data/raw/airnow_ytd"
COMBINED_CSV = "data/raw/dhaka_pm25_airnow_2016_2025_raw.csv"

# --- Time split (README 6.6 — never randomSplit) -----------------------------
# Train on everything before this instant, test on everything from it onward.
SPLIT_TS = "2024-01-01 00:00:00"

# --- Feature definition ------------------------------------------------------
LAG_HOURS = [1, 2, 3, 24, 168]        # 1-3h recent, 24h same hour yesterday, 168h same hour last week
ROLL_WINDOWS = [3, 24]                # rolling mean/std windows, all ending at t-1

FEATURE_COLS = (
    [f"lag_{h}" for h in LAG_HOURS]
    + [f"roll_mean_{w}" for w in ROLL_WINDOWS]
    + [f"roll_std_{w}" for w in ROLL_WINDOWS]
    + ["nowcast_lag_1", "hour", "month", "dow",
       "hour_sin", "hour_cos", "month_sin", "month_cos"]
)

LABEL_COL = "pm25"

# --- Model -------------------------------------------------------------------
GBT_PARAMS = dict(maxIter=100, maxDepth=6, stepSize=0.1, seed=42)

MODEL_DIR = "models/pm25_v1"
