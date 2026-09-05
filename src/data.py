"""Download, load and clean the Dhaka AirNow PM2.5 series."""

import os
import urllib.request

from pyspark.sql import DataFrame, SparkSession, functions as F

from .config import AIRNOW_BASE, RAW_DIR, YEARS
from .schema import AIRNOW_SCHEMA, DROP_COLS, RENAME, SENTINEL


def download_raw(dest: str = RAW_DIR, years=YEARS, force: bool = False) -> str:
    """Fetch one CSV per year from the AirNow embassy archive.

    Public S3, no API key. Skips files already on disk unless force=True.
    """
    os.makedirs(dest, exist_ok=True)
    for year in years:
        name = f"Dhaka_PM2.5_{year}_YTD.csv"
        path = os.path.join(dest, name)
        if os.path.exists(path) and not force:
            continue
        url = f"{AIRNOW_BASE}/{year}/{name}"
        urllib.request.urlretrieve(url, path)
        print(f"  downloaded {name} ({os.path.getsize(path):,} bytes)")
    print(f"{len(os.listdir(dest))} files in {dest}")
    return dest


def load_raw(spark: SparkSession, path: str = RAW_DIR) -> DataFrame:
    """Read the yearly CSVs with the EXPLICIT schema (never inferSchema).

    Same schema object Phase 2 hands to `readStream`, which cannot infer.
    """
    df = (spark.read
          .option("header", True)
          .schema(AIRNOW_SCHEMA)
          .csv(path))
    for old, new in RENAME.items():
        df = df.withColumnRenamed(old, new)
    return df


def clean(df: DataFrame) -> DataFrame:
    """Apply every cleaning rule the profiling turned up.

    1. Build the timestamp from Year/Month/Day/Hour rather than parsing the
       12-hour `Date (LT)` string. Verified identical on all 77,710 rows, and
       it sidesteps AM/PM locale parsing entirely.
    2. Keep only qc == 'Valid'. NOT a no-op on the raw download: 2,138 rows are
       'Missing', 197 'Invalid', 7 'Suspect', all carrying the -999 sentinel.
       (The README says this filter is a no-op -- that was true only of a
       teammate's pre-cleaned file, not of the real archive.)
    3. Null out the -999 sentinel wherever it survives a 'Valid' flag: 94 rows
       have a valid PM2.5 but a sentinel NowCast.
    4. Clip physically impossible negative concentrations to 0. 24 rows read as
       low as -4.0 ug/m3 -- sensor noise near zero.
    5. Drop duplicate timestamps. Year files overlap slightly at the seams.
    """
    df = (df
          .withColumn("ts", F.expr(
              "make_timestamp(Year, Month, Day, Hour, 0, 0)"))
          .filter(F.col("qc") == "Valid")
          .filter(F.col("pm25") != SENTINEL))

    # NowCast keeps its sentinel on a handful of otherwise-valid rows.
    df = df.withColumn(
        "nowcast",
        F.when(F.col("nowcast") == SENTINEL, None).otherwise(F.col("nowcast")))

    # Negative PM2.5 is physically impossible -> clip, do not drop, so the
    # hourly spine keeps its row.
    df = df.withColumn("pm25", F.greatest(F.col("pm25"), F.lit(0.0)))

    # AQI Category 'N/A' (96 rows) is not a real class -- null it so the
    # classification target stays clean. Regression is unaffected.
    df = df.withColumn(
        "aqi_category",
        F.when(F.col("aqi_category") == "N/A", None).otherwise(F.col("aqi_category")))

    df = df.drop(*[c for c in DROP_COLS if c in df.columns])
    return df.dropDuplicates(["ts"])


def hourly_spine(spark: SparkSession, df: DataFrame) -> DataFrame:
    """Reindex onto a gap-free hourly spine (README 6.1).

    THIS IS THE STEP THAT MAKES LAG FEATURES CORRECT.

    `F.lag(...)` counts ROWS, not hours. The series is missing ~3,200 hours
    spread over hundreds of gaps, so lagging the raw rows would hand the model
    a reading from days earlier and label it `lag_1`. Generating every hour in
    the range and left-joining puts an explicit null in each hole, which
    `.na.drop()` later removes along with any feature that spanned a gap.
    """
    lo, hi = df.agg(F.min("ts"), F.max("ts")).first()

    spine = (spark
             .createDataFrame([(lo, hi)], "lo timestamp, hi timestamp")
             .select(F.explode(
                 F.sequence("lo", "hi", F.expr("INTERVAL 1 HOUR"))).alias("ts")))

    return spine.join(df, on="ts", how="left")
