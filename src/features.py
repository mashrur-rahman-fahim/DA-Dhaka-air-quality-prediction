"""Lag / rolling feature engineering.

Imported by BOTH phases so the two never drift apart. Phase 2 calls the exact
same `build_features` inside `foreachBatch`, where the micro-batch is an
ordinary batch DataFrame and Window functions work normally (README 8.1).
"""

import math

from pyspark.sql import DataFrame, Window, functions as F

from .config import FEATURE_COLS, LABEL_COL, LAG_HOURS, ROLL_WINDOWS


def build_features(df: DataFrame) -> DataFrame:
    """Turn a gap-free hourly spine into model-ready features.

    Every feature is built strictly from hours BEFORE t, so predicting `pm25`
    at t is a genuine one-hour-ahead forecast.

    Two leakage rules enforced here (README 6.5):
      * rolling windows end at -1, never 0. `rowsBetween(-3, -1)` is the three
        hours before now; `rowsBetween(-2, 0)` would include the answer.
      * NowCast is a smoothed function of recent readings INCLUDING the current
        hour, so only its lagged value is ever used.

    Requires `df` to already sit on a complete hourly spine -- see
    data.hourly_spine. Without that these row-counting windows silently cross
    gaps.
    """
    w = Window.orderBy("ts")

    for h in LAG_HOURS:
        df = df.withColumn(f"lag_{h}", F.lag("pm25", h).over(w))

    for n in ROLL_WINDOWS:
        win = w.rowsBetween(-n, -1)          # ends at -1: no leakage
        df = (df
              .withColumn(f"roll_mean_{n}", F.avg("pm25").over(win))
              .withColumn(f"roll_std_{n}", F.stddev("pm25").over(win)))

    # NowCast at time t contains the answer. Only the lagged value is safe.
    df = df.withColumn("nowcast_lag_1", F.lag("nowcast", 1).over(w))

    # Calendar features come from `ts`, not from the Year/Month/Day/Hour
    # columns -- those are null on spine rows that had no source reading.
    hour = F.hour("ts")
    month = F.month("ts")
    tau = 2.0 * math.pi
    df = (df
          .withColumn("hour", hour.cast("double"))
          .withColumn("month", month.cast("double"))
          .withColumn("hour_sin", F.sin(hour * (tau / 24)))
          .withColumn("hour_cos", F.cos(hour * (tau / 24)))
          .withColumn("month_sin", F.sin((month - 1) * (tau / 12)))
          .withColumn("month_cos", F.cos((month - 1) * (tau / 12)))
          .withColumn("dow", F.dayofweek("ts").cast("double")))

    return df


def drop_incomplete(df: DataFrame) -> DataFrame:
    """Remove rows whose label or any feature is null.

    This is what actually deletes the spine holes and every row whose lag
    window reached back across a gap.
    """
    return df.dropna(subset=FEATURE_COLS + [LABEL_COL])
