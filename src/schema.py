"""Explicit schema for the AirNow embassy CSV.

Written once, used by BOTH phases. Phase 1 could infer it; Phase 2 cannot --
`spark.readStream` refuses to infer schema (README 8.4), so it is defined here
and imported by both.
"""

from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType, DoubleType,
)

# Column order exactly as it appears in the AirNow CSV header:
# Site,Parameter,Date (LT),Year,Month,Day,Hour,NowCast Conc.,AQI,AQI Category,
# Raw Conc.,Conc. Unit,Duration,QC Name
AIRNOW_SCHEMA = StructType([
    StructField("Site",           StringType(),  True),
    StructField("Parameter",      StringType(),  True),
    StructField("Date (LT)",      StringType(),  True),
    StructField("Year",           IntegerType(), True),
    StructField("Month",          IntegerType(), True),
    StructField("Day",            IntegerType(), True),
    StructField("Hour",           IntegerType(), True),
    StructField("NowCast Conc.",  DoubleType(),  True),
    StructField("AQI",            IntegerType(), True),
    StructField("AQI Category",   StringType(),  True),
    StructField("Raw Conc.",      DoubleType(),  True),
    StructField("Conc. Unit",     StringType(),  True),
    StructField("Duration",       StringType(),  True),
    StructField("QC Name",        StringType(),  True),
])

# Original name -> clean name. The source names carry spaces, dots and
# parentheses, which need backticks everywhere in Spark SQL. Rename on load.
RENAME = {
    "Date (LT)":     "date_lt",
    "NowCast Conc.": "nowcast",
    "AQI Category":  "aqi_category",
    "Raw Conc.":     "pm25",
    "QC Name":       "qc",
}

# Constant across every row in this dataset -- verified, not assumed:
# Site='Dhaka', Parameter='PM2.5 - Principal', Conc. Unit='UG/M3', Duration='1 Hr'.
# Dropped on load; keep `Site` in mind if the project ever scales to more stations.
DROP_COLS = ["Site", "Parameter", "Conc. Unit", "Duration", "date_lt"]

# The AirNow sentinel for "no reading". Appears in NowCast Conc., Raw Conc. and
# AQI. Rows carrying it are normally flagged qc != 'Valid', but 94 rows are
# flagged Valid and still carry -999 in NowCast -- so filter on the value too.
SENTINEL = -999.0
