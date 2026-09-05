# Dhaka Air-Quality Forecasting with PySpark

**Short-Term PM2.5 Prediction using Big-Data Analytics (PySpark Structured Streaming + MLlib)**

> **New to this project? Read Sections 1–4. That is enough to understand what we are building and why.**
> Sections 5–9 are the build plan. Section 10 is a glossary if any term is unfamiliar, and Section 11 answers the questions people actually ask about this design.

---

## 1. TL;DR

We are building an **early-warning system for air pollution in Dhaka**.

Given the last few hours of PM2.5 readings, predict **how polluted the air will be next**. Today the response to bad air is purely reactive — people find out it is dangerous only after it already is. We want to see the spike coming *before* it arrives.

The work splits into **two completely separate phases**:

| Phase | Runs | Does |
|---|---|---|
| **Phase 1 — Trainer** | Once, offline, batch | Reads the full historical CSV, engineers features, trains a model, **saves it to disk** |
| **Phase 2 — Predictor** | Continuously, streaming | **Loads the saved model**, receives one new hour at a time, predicts, writes the forecast out |

**There is no model training in Phase 2.** This is the single most important thing to understand about the design, and Section 4 explains why.

**Current status:** proposal deck complete and presented. **No code written yet.** Phase 1 has not been started.

---

## 1b. Quick Start — Get Running

**1. Get the data.** It is *not* in this repo (public government data, linked rather than vendored).

Download from **AirNow** — US Dept. of State / EPA embassy monitoring program:

> **https://gispub.epa.gov/airnowembassy/** → **Archive** tab → select **Dhaka** → download CSV

Place it at:
```
data/raw/dhaka_air_quality_clean.csv
```

Full source details, citation, and a warning about the pre-cleaned file are in **Section 5.2**.

**2. Set up Spark.** We use **Google Colab** (free, no local install):
```python
!pip install -q pyspark findspark
```

**3. Read before writing code.** In this order:
- **Section 4** — the two-phase architecture. Non-negotiable; the design will not make sense without it.
- **Section 6** — data gotchas. Every one of these silently produces a wrong model if ignored.
- **Section 7** — Phase 1 build steps. **Start here.** Do not start Phase 2 first.

**4. Confused by a term?** Section 10 is a glossary. Section 11 is an FAQ.

---

## 2. The Problem

Dhaka has some of the worst air quality in the world. The existing situation:

- **No visibility ahead** — nobody can reliably know how bad the air will be later today or tomorrow.
- **Purely reactive** — we learn the air is dangerous only once it already is.
- **No way to prepare** — without a forecast, nobody can plan, protect themselves, or act early.

**The goal:** turn a reactive problem into a proactive, predictable one.

**Why it matters:**

| Area | Benefit of a forecast |
|---|---|
| Health | People can wear masks, stay indoors, protect children *before* spikes hit |
| Planning | Schools, hospitals, outdoor workers can prepare ahead |
| Policy | Authorities get data-driven evidence for pollution-control decisions |
| Awareness | Makes an invisible daily danger visible |

---

## 3. What We Predict

Two targets, from the same pipeline.

### Primary target — Regression
**Next-hour / next-day PM2.5 concentration**, in µg/m³.
A continuous number. Evaluated with **RMSE** (root mean squared error — lower is better, 0 is perfect, same unit as the thing predicted).

### Secondary target — Classification
**AQI Category** — `Good → Moderate → Unhealthy for Sensitive Groups → Unhealthy → Very Unhealthy → Hazardous`.
A label. Easier for the public to read than a raw number. Evaluated with **AUC / per-class F1** — *not* plain accuracy, for reasons in Section 6.4.

**In one line:** given the recent hours of readings, predict how polluted the air will be next.

---

## 4. Architecture — The Two-Phase Design

This is the core of the project. Read this section carefully.

```
┌─ PHASE 1 · BATCH TRAINER ────────┐     ┌─ PHASE 2 · STREAMING PREDICTOR ──┐
│  runs occasionally (offline)     │     │  runs continuously (live)         │
│                                  │     │                                   │
│  historical CSV                  │     │  new hour arrives                 │
│      ↓ clean                     │     │      ↓                            │
│      ↓ fill hourly gaps          │     │  build lag features               │
│      ↓ lag / rolling features    │     │      ↓                            │
│      ↓ time-based train/test     │     │  PipelineModel.load()             │
│      ↓ GBTRegressor.fit()        │     │      ↓                            │
│      ↓                           │     │  .transform()  ← PREDICT ONLY     │
│  model.save("models/pm25_v1") ───┼────▶│      ↓                            │
└──────────────────────────────────┘     │  write forecast + alert           │
                 ▲                       └───────────────┬───────────────────┘
                 │                                       │
                 └───── accumulated predictions ─────────┘
                        + actuals → retrain → v2
```

### 4.1 Why there is no training in Phase 2

**Spark MLlib has no incremental `fit()` for tree ensembles.** Calling `pipeline.fit()` on a *streaming* DataFrame throws an error — Structured Streaming supports only a subset of operations, and model training is not one of them.

What **is** supported on a stream is `.transform()` — inference. So:

- **Train** = batch operation on accumulated history
- **Predict** = streaming operation on arriving data

This is not a compromise or a workaround. **It is how essentially every production ML system works.** Training on a single hourly record would be statistically meaningless anyway.

### 4.2 But the system still improves over time

Learning happens on a **separate loop**, not inside the stream:

1. The streaming job writes its inputs and predictions out to storage.
2. Later, the batch trainer reads that accumulated history — now including hours the model has never seen — and trains **v2**.
3. The stream picks up v2 and keeps predicting.

This is called **scheduled retraining**. The streaming query still never calls `fit()`.

### 4.3 Options we deliberately rejected

| Option | Why not |
|---|---|
| `StreamingLinearRegressionWithSGD`, `StreamingKMeans` | Legacy RDD-based `pyspark.mllib` API driven by **DStreams** (deprecated), not Structured Streaming. Only linear models and k-means — **no GBT, no Random Forest**. |
| Calling `.fit()` inside `foreachBatch` | Technically legal, but one hour = one row. Training on that is meaningless. Refitting on accumulated history is just scheduled retraining with extra steps. |

### 4.4 If asked to defend the design

> *"Training is a batch operation on accumulated history; inference is a streaming operation on arriving data. Separating them is deliberate — it is the standard production pattern. Retraining on a single hourly record would be statistically meaningless, and MLlib's tree ensembles have no incremental fit. The pipeline still runs end-to-end on Spark, and the model updates on a scheduled retrain loop."*

That answer is stronger than claiming online training, because anyone knowledgeable already knows GBT cannot do it.

---

## 5. The Data

### 5.1 Location

```
~/Downloads/dhaka_air_quality_clean.csv     3.9 MB
```

> **TODO:** move this into the project directory (`data/raw/`) and out of `~/Downloads`.

Companion proposal deck: `~/Downloads/01-Dhaka_AirQuality_Proposal_Redesigned.pptx` (12 slides)

### 5.2 Where the data came from

**AirNow** — the US Department of State / EPA embassy air-quality monitoring program. The CSV's column set (`NowCast Conc.`, `Raw Conc.`, `QC Name`, `AQI Category`, `Conc. Unit`, `Duration`) is that program's standard export format.

| | |
|---|---|
| Download portal | **https://gispub.epa.gov/airnowembassy/** → *Archive* tab → select **Dhaka** → CSV |
| Embassy page | https://bd.usembassy.gov/air-quality-data/ |

**Citation for the report:**

```
U.S. Department of State and U.S. Environmental Protection Agency,
"AirNow Department of State — Dhaka, Bangladesh (PM2.5)."
Available: https://gispub.epa.gov/airnowembassy/
Data range used: 2016-03-01 to 2020-09-05 (36,561 hourly records).
```

> ⚠️ **The file on disk is not the raw download.** The `_clean` suffix means it was post-processed by someone on the team — confirmed by profiling (`QC Name` is 100% `Valid`, so non-valid rows were already stripped). A fresh download from AirNow will contain **more rows** and **non-`Valid` QC entries you must filter yourself**.
>
> **TODO:** track down the original raw file and the cleaning script from whoever downloaded it. For a publication the cleaning step must be reproducible.

### 5.3 Source and shape

| Property | Value |
|---|---|
| Source | **U.S. Embassy Dhaka monitor (AirNow)**, pre-cleaned |
| Rows | **36,561** hourly records |
| Range | **2016-03-01 03:00** → **2020-09-05 00:00** |
| Site | `Dhaka` — single station |
| Parameter | `PM2.5 - Principal` — single pollutant |

Rows per year: 2016 → 7,321 · 2017 → 8,476 · 2018 → 6,477 · 2019 → 8,597 · 2020 → 5,690

### 5.4 Schema

| Column | Type | What it is | How it is used |
|---|---|---|---|
| *(unnamed first column)* | int | Pandas index artifact | **Drop it** |
| `Site` | string | Station location | Constant (`Dhaka`) — drop, but keep in mind for scaling to more stations |
| `Parameter` | string | Pollutant name | Constant (`PM2.5 - Principal`) — drop |
| `Year`, `Month`, `Day`, `Hour` | int | Split time components | `Hour` and `Month` become features (daily peaks, winter seasonality) |
| `NowCast Conc.` | double | AirNow's smoothed concentration | Optional feature — **see leakage warning in 6.5** |
| `AQI` | int | Standard 0–500 index | Alternative target, or a feature |
| `AQI Category` | string | `Good` … `Hazardous` | **The classification target** |
| `Raw Conc.` | double | Measured PM2.5 in µg/m³ | **The regression target** — the value we predict |
| `Conc. Unit` | string | Always `UG/M3` | Drop |
| `Duration` | string | Always `1 Hr` | Drop |
| `QC Name` | string | Quality flag | **All 36,561 rows are `Valid`** — filtering on this is a no-op |
| `Date` | string | `YYYY-MM-DD HH:MM:SS` | **Parse to timestamp** — orders the series, drives all lag features |

### 5.5 Value distribution

`Raw Conc.` — min **−4.0**, max **985.0**, mean **82.5** µg/m³

`AQI Category` counts:

| Category | Count | Share |
|---|---|---|
| Unhealthy | 12,367 | 33.8% |
| Moderate | 9,899 | 27.1% |
| Unhealthy for Sensitive Groups | 7,279 | 19.9% |
| Very Unhealthy | 4,690 | 12.8% |
| Hazardous | 1,309 | 3.6% |
| **Good** | **1,017** | **2.8%** |

---

## 6. Data Gotchas — Read Before Writing Any Code

These were found by profiling the actual file. Each one will silently produce a wrong model if ignored.

### 6.1 ~7.6% of hours are missing — and one gap is 76 days

| Metric | Value |
|---|---|
| Expected hourly slots in range | 39,574 |
| Actual rows | 36,561 |
| **Missing hours** | **3,013 (7.6%)** |
| Number of separate gaps | **227** |
| Gaps of 2–3 hours | 165 (mostly harmless) |
| Gaps longer than 24 hours | **13** |
| **Largest single gap** | **1,824 hours ≈ 76 days** |

**Why this breaks things.** `F.lag("Raw Conc.", 1).over(Window.orderBy("Date"))` counts **rows, not hours**. Across a gap it will happily hand you a reading from 76 days ago while labelling it `lag_1`. The model then learns from garbage.

**Fix — one of:**
- Reindex onto a **complete hourly spine** (generate every hour in the range, left-join the data, leaving explicit nulls), then lag over that; or
- Use a **time-range window** (`rangeBetween` on a timestamp cast to seconds) instead of `rowsBetween`.

Then drop rows whose lag features fall inside a gap.

### 6.2 13 rows have negative PM2.5

`Raw Conc.` goes as low as **−4.0**. A negative particulate concentration is physically impossible — this is sensor noise near zero.

**Fix:** clip to 0, or drop those 13 rows. Either is defensible; document which you chose.

### 6.3 `QC Name` filtering is unnecessary

All 36,561 rows are already `Valid`. The file is pre-cleaned. Filtering on it does nothing — do not waste a stage on it, and do not claim it as a cleaning step in the report.

### 6.4 The classification target is badly imbalanced

`Good` is **2.8%** of rows; `Unhealthy` is **33.8%**. A model that *never once* predicts `Good` still scores well on plain accuracy.

**Fix:** report **AUC and per-class precision/recall/F1**, never bare accuracy. Consider class weights.

### 6.5 Do not leak the answer into the features

Two specific traps:

- **`NowCast Conc.`** is a smoothed function of recent readings *including the current hour*. If you predict `Raw Conc.` at time *t* using `NowCast` at time *t*, you are feeding the model the answer. Use only **lagged** NowCast, or drop it.
- **Rolling windows must end at `-1`, not `0`.** `rowsBetween(-3, -1)` looks at the previous three hours. `rowsBetween(-2, 0)` includes the current hour — which holds the value you are trying to predict.

### 6.6 Never split this data randomly

`randomSplit([0.8, 0.2])` shuffles rows, which lets the model train on the future and test on the past. For time series that inflates every metric and means nothing.

**Split by time.** For example: train on 2016-03 → 2019-12, test on 2020. State the cut-off explicitly in the report.

---

## 7. Phase 1 — Batch Trainer (build this first)

**Do not start Phase 2 until Phase 1 produces an honest RMSE.** Streaming is the delivery mechanism, not the model. Debugging both at once means you will not know whether a bad number is a broken model or a broken stream. Phase 1 also produces the saved artifact that Phase 2 requires in order to exist at all.

### Steps

1. **Load** the CSV with an **explicit schema** (do not rely on inference — Phase 2 will require an explicit schema anyway, so write it once and reuse it).
2. **Clean** — drop the unnamed index / `Site` / `Parameter` / `Conc. Unit` / `Duration` / `QC Name`; parse `Date` to timestamp; handle the 13 negative values.
3. **Gap-fill** — build the complete hourly spine (Section 6.1).
4. **Feature-engineer** — `lag_1`, `lag_2`, `lag_3`, `lag_24` (same hour yesterday), `rolling_avg_3`, `rolling_avg_24`, plus `Hour` and `Month` for daily and seasonal cycles. Then `.na.drop()`.
5. **Split by time** (Section 6.6).
6. **Assemble + train** — `VectorAssembler` → `GBTRegressor` inside a `Pipeline`.
7. **Evaluate** — `RegressionEvaluator` with `rmse` on the held-out *later* period. Compare against a **persistence baseline** ("tomorrow = today"); if the model cannot beat that, it has learned nothing.
8. **Save** — `pipeline_model.write().overwrite().save("models/pm25_v1")`.

### Sketch

```python
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.window import Window
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator

# 4. features  (window must be gap-safe — see 6.1)
w = Window.orderBy("ts")
df = (df
      .withColumn("lag_1",  F.lag("pm25", 1).over(w))
      .withColumn("lag_2",  F.lag("pm25", 2).over(w))
      .withColumn("lag_24", F.lag("pm25", 24).over(w))
      .withColumn("roll_3", F.avg("pm25").over(w.rowsBetween(-3, -1)))
      .na.drop())

# 5. TIME split — never randomSplit
train = df.filter(F.col("ts") <  "2020-01-01")
test  = df.filter(F.col("ts") >= "2020-01-01")

# 6. train
assembler = VectorAssembler(
    inputCols=["lag_1", "lag_2", "lag_24", "roll_3", "Hour", "Month"],
    outputCol="features")
gbt = GBTRegressor(featuresCol="features", labelCol="pm25", maxIter=50, seed=42)
model = Pipeline(stages=[assembler, gbt]).fit(train)

# 7. evaluate on the LATER period
rmse = RegressionEvaluator(labelCol="pm25", metricName="rmse").evaluate(
        model.transform(test))
print("RMSE:", rmse)

# 8. save
model.write().overwrite().save("models/pm25_v1")
```

---

## 8. Phase 2 — Streaming Predictor

### 8.0 Which streaming API — read this before Googling anything

Spark has **two** streaming APIs. Only one is alive.

| API | Entry point | Status | Use it? |
|---|---|---|---|
| **Structured Streaming** | `spark.readStream` / `spark.writeStream` — built on DataFrames | Current | ✅ **This one** |
| **Spark Streaming (DStreams)** | `pyspark.streaming.StreamingContext` | **Deprecated** since Spark 3.4 | ❌ Never |

⚠️ **Searching *"PySpark streaming tutorial"* returns mostly DStream tutorials.** They are the old RDD-based API. They will not integrate with MLlib pipelines, and they are the same dead branch as `StreamingLinearRegressionWithSGD` (Section 4.3). **If a code sample uses `StreamingContext`, close the tab.**

**Phase split, to be unambiguous:**

- **Phase 1 is batch only** — plain `spark.read.csv`, no streaming anywhere in it.
- **Phase 2 is Structured Streaming** — `readStream` → `foreachBatch` → load model → `transform` → write.

Both are PySpark. Only Phase 2 streams.

### 8.1 The constraint that shapes everything

**`F.lag().over(Window.orderBy(...))` does not work on a streaming DataFrame.** Non-time-based window functions are unsupported in Structured Streaming — and lags are our entire feature set.

Three ways out, cheapest first:

1. **`foreachBatch`** — each micro-batch arrives as an ordinary *batch* DataFrame, so `Window` + `lag` work normally inside it. **This is the right answer for this project.**
2. **Time-window aggregations** — `groupBy(window(col("ts"), "3 hours", "1 hour"))` *is* supported natively. Good for rolling averages, useless for an exact `lag_1`.
3. **`applyInPandasWithState`** (Spark 3.4+) — real per-key state. Correct, but heavy for this scope.

### 8.2 The follow-on problem

With `foreachBatch`, `lag_1` for the newest hour needs the *previous* hour — which arrived in a *previous* batch. Solve it by keeping a small rolling **"recent readings" table** (parquet, last ~48 rows) that each batch appends to and reads back before building features.

### 8.3 Sketch

```python
from pyspark.ml import PipelineModel

_model, _model_version = None, None

def process_batch(batch_df, batch_id):
    global _model, _model_version
    latest = read_version_marker()                 # tiny text/json file
    if latest != _model_version:                   # hot-swap, no restart
        _model = PipelineModel.load(f"models/pm25_{latest}")
        _model_version = latest

    append_to_recent(batch_df)                     # rolling history table
    feats = build_lag_features(read_recent())      # Window/lag OK here — batch DF
    preds = _model.transform(feats)
    preds.write.mode("append").parquet("output/predictions")

(spark.readStream
      .schema(SCHEMA)                              # REQUIRED — see 8.4
      .option("maxFilesPerTrigger", 1)             # replays hour by hour
      .csv("data/stream_in/")
      .writeStream
      .foreachBatch(process_batch)
      .start())
```

The version-marker check gives a nice demo property: retrain in another notebook, save `v2`, and the running stream swaps to it **mid-flight**. Do not reload every batch — that is a full model deserialise per hour for nothing.

### 8.4 Two things that trip everyone on first run

- **`readStream` cannot infer schema.** You must pass one explicitly. Batch can infer; streaming cannot. (This is why Phase 1 step 1 says write the schema once.)
- **`maxFilesPerTrigger=1`** is what makes the demo *look* like live hourly arrival. Without it Spark consumes every file at once.

---

## 9. Tech Stack & Environment

| Component | Choice |
|---|---|
| Platform | **Google Colab** (free, no local setup) |
| Engine | **PySpark** |
| Streaming | **Spark Structured Streaming** |
| ML | **Spark MLlib** |
| Models | `GBTRegressor` (regression) · classifier TBD for AQI Category |

**Scalability claim:** the same pipeline extends from one station to all of Bangladesh. The `Site` column already exists for exactly that — partition by it and the design carries over unchanged.

---

## 10. Glossary

| Term | Meaning |
|---|---|
| **PM2.5** | Fine particulate matter under 2.5 µm. The pollutant we forecast. Measured in µg/m³. |
| **AQI** | Air Quality Index, 0–500. A standardised human-readable scale derived from PM2.5. |
| **Transformer** | A Spark ML stage that needs to learn nothing — call `.transform()` directly. e.g. `VectorAssembler`. |
| **Estimator** | A Spark ML stage that must study the data first — call `.fit()`, which returns a Model (which is a Transformer). e.g. `GBTRegressor`. |
| **Pipeline** | An ordered list of stages. `Pipeline.fit()` returns a `PipelineModel`. **Save the PipelineModel, load with `PipelineModel.load`.** |
| **features / label** | Every Spark ML model eats exactly two columns: `features` (one vector) and `label` (one number). Everything before the model exists to build those two. |
| **Lag feature** | A past value used as an input, e.g. `lag_1` = the reading one hour ago. |
| **Data leakage** | Accidentally letting the answer into the inputs. Produces models that score beautifully and fail completely in reality. |
| **RMSE** | Root Mean Squared Error. "On average, how far off were we?" Same units as the target. Lower is better. |
| **AUC** | Area Under ROC Curve. Classification quality. 1.0 perfect, 0.5 coin-flip. Higher is better. |
| **`foreachBatch`** | Structured Streaming escape hatch — hands you each micro-batch as a normal batch DataFrame, where all batch operations work. |
| **Persistence baseline** | The trivial forecast "next value = current value". Any real model must beat it. |

---

## 11. FAQ — Questions That Actually Came Up

Kept verbatim in spirit, because the next person will ask the same ones.

### "Are we using PySpark Streaming?"

Yes — **Spark Structured Streaming**, in **Phase 2 only**. Phase 1 is pure batch. See Section 8.0 for the DStreams warning; it is the easiest mistake to make.

### "Is there really no training in Phase 2?"

Correct. The streaming query **only loads a saved model and predicts**. It never calls `fit()`.

Not a limitation we chose — MLlib physically cannot do it. `pipeline.fit()` on a streaming DataFrame throws. Structured Streaming supports `.transform()` (predict) but not `fit()` (train), and GBT / Random Forest have no incremental learning in any Spark API.

### "So we train first, and then it predicts live?"

Exactly.

1. **Train once**, offline, on the historical CSV → save the model to disk.
2. **Streaming job loads that saved model** → each new hour arrives → predict → write forecast.

The model improves over time via **scheduled retraining** (Section 4.2), never inside the stream.

### "Then how does the system ever learn from new data?"

Separate loop:

```
stream writes predictions + actuals  →  accumulates
        →  batch trainer reads that history later
        →  produces v2  →  stream hot-swaps to v2  (Section 8.3)
```

The demo trick: retrain in another notebook, save `v2`, and the running stream picks it up **mid-flight** via the version marker. Looks like live learning; stays architecturally honest.

### "Which do we build first?"

**Phase 1, completely, until it produces an honest RMSE that beats the persistence baseline.**

Streaming is the delivery mechanism, not the model. If you debug both at once you will not know whether a bad number means a broken model or a broken stream. Phase 1 also produces the saved artifact that Phase 2 needs in order to run at all.

### "Where did the dataset come from?"

**AirNow** (US Dept. of State / EPA embassy monitoring). Portal and citation in Section 5.2. Note the on-disk file is **pre-cleaned**, not the raw download.

---

## 12. Team

| Name | Student ID |
|---|---|
| Mashrur Rahman | 20220104108 |
| Chowdhury Ajmayeen Adil | 20220104121 |
| Ahnuf Karim Chowdhury | 20220104122 |
| Nahid Asef | 20220104128 |

---

## 13. Status & Next Actions

**Done**
- Problem framing, targets, and dataset selected
- Proposal deck complete (12 slides)
- Dataset profiled; all gotchas in Section 6 identified

**Not done**
- Everything else. **No code exists.**

**Next actions, in order**

1. Move `dhaka_air_quality_clean.csv` into `data/raw/` in this directory
2. Write the explicit schema (used by both phases)
3. Phase 1 steps 2–4: clean, gap-fill, features
4. Phase 1 steps 5–7: time-split, train, evaluate **against the persistence baseline**
5. Phase 1 step 8: save `models/pm25_v1`
6. Only then start Phase 2

**Expected outcome (from the proposal):** accurate short-term PM2.5 forecasts for Dhaka; a scalable pipeline extendable to more cities and pollutants; public-health impact through earlier warnings; a foundation for a research publication.
