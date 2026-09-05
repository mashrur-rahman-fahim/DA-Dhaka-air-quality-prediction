# Dhaka Air-Quality Forecasting with PySpark

**Short-term PM2.5 prediction using big-data analytics — PySpark Structured Streaming + MLlib**

Predict how polluted Dhaka's air will be in the next hour, from the last few
hours of readings. Today the response to bad air is purely reactive: people find
out it is dangerous only once it already is. The goal is to see the spike coming
before it arrives.

---

## Quick start

Everything lives in **one Colab notebook**. Nothing to install, nothing to
upload — it downloads its own data.

**[Open `dhaka_pm25.ipynb` in Colab](https://colab.research.google.com/github/mashrur-rahman-fahim/DA-Dhaka-air-quality-prediction/blob/main/dhaka_pm25.ipynb)**

Use a **CPU** runtime. Spark MLlib has no GPU support, so a GPU runtime costs
compute units and changes nothing.

New work is **appended to the bottom of that same notebook**. There is one
notebook and one URL for the whole project, so the Spark session and every
computed variable stay alive as the work grows.

---

## 1. The problem

Dhaka has some of the worst air quality in the world.

- **No visibility ahead** — nobody can reliably know how bad the air will be later today.
- **Purely reactive** — we learn the air is dangerous only once it already is.
- **No way to prepare** — without a forecast, nobody can plan or protect themselves.

| Area | What a forecast enables |
|---|---|
| Health | Masks, staying indoors, protecting children *before* a spike |
| Planning | Schools, hospitals and outdoor workers prepare ahead |
| Policy | Evidence for pollution-control decisions |
| Awareness | Makes an invisible daily danger visible |

## 2. What we predict

**Primary — regression.** Next-hour PM2.5 concentration in µg/m³. Scored with
**RMSE** (same units as the thing predicted; lower is better).

**Secondary — classification.** AQI category, `Good` through `Hazardous`.
Easier for the public to read. Scored with **per-class F1 and AUC**, never plain
accuracy — see the imbalance in Q13 below.

Every model must beat the **persistence baseline**: *"the next hour will be the
same as this hour."* Anything that cannot beat that has learned nothing.

## 3. The data

Hourly PM2.5 from the **US Embassy monitor in Dhaka**, published through the
AirNow programme (US Dept. of State / EPA). One station, one pollutant.

| | |
|---|---|
| Range | 2016-03-01 to 2025-03-24 |
| Rows in the files | 77,716 |
| Usable rows | 75,374 (`qc = Valid`), of which 6 are duplicate hours |
| Size | ~8 MB across ten yearly CSVs |

**Where it comes from.** Most tutorials point at `dosairnowdata.org`. That
domain no longer resolves. The live source, which the EPA's own embassy map
reads from, is public and needs no API key:

```
https://s3-us-west-1.amazonaws.com/files.airnowtech.org/airnow/EmbassyHistorical/Dhaka/<YEAR>/Dhaka_PM2.5_<YEAR>_YTD.csv
```

The notebook downloads these itself. Nothing is committed to the repo.

**Citation for the report:**

```
U.S. Department of State and U.S. Environmental Protection Agency,
"AirNow Department of State — Dhaka, Bangladesh (PM2.5)."
https://gispub.epa.gov/airnowembassy/
Data range used: 2016-03-01 to 2025-03-24 (75,374 valid hourly records).
```

### Columns

| Column | What it is | Verdict |
|---|---|---|
| `site`, `parameter`, `unit`, `duration` | Station, pollutant, µg/m³, 1 Hr | Constant. Drop. |
| `date_text` | `2016-01-01 01:00 AM`, 12-hour format | Superseded by a real timestamp |
| `year`, `month`, `day`, `hour` | Split time components | Timestamp is built from these |
| `nowcast` | AirNow's smoothed concentration | **Leaks — see Q12** |
| `aqi` | Standard 0–500 index | **Leaks — see Q12** |
| `aqi_category` | `Good` … `Hazardous` | Classification target; **leaks as an input** |
| `pm25` (`Raw Conc.`) | Measured PM2.5 | **The regression target** |
| `qc` | Quality flag | Filter on it. Not a no-op. |

## 4. What the exploration found

Produced by Q1–Q15 in the notebook. Every number is from the real file.

| # | Question | Finding |
|---|---|---|
| Q1 | Useful columns? | Four never vary. Dead weight. |
| Q2 | Quality flag? | 2,342 of 77,716 rows are not `Valid` and hold `-999`. |
| — | Duplicate hours? | **6 hours appear twice**, from the seams between yearly files. |
| — | Impossible values? | **24 readings are negative**, as low as −4.0 µg/m³. |
| Q4 | Missing hours? | 5.1% missing across **517 gaps**. 419 are a single hour; the worst is **1,822 hours (76 days)**. |
| Q5 | Distribution? | Median 65, mean 93, max 985. Skew **+1.86** raw, **−0.25** under `log(1+x)`. |
| Q6 | Season? | January averages **200**, July **31** — a **6.4× swing**. Strongest pattern in the data. |
| Q7 | Time of day? | Peaks near midnight (112), dips mid-afternoon (63). |
| Q8 | Day of week? | Spans under 7 µg/m³. Effectively no signal. |
| Q9 | Yearly trend? | Drifts upward, but the final year is partial and dry-season only. |
| Q10 | Volatility? | Hour-to-hour SD **33.2**. 7.1% of hours move more than 50. |
| Q11 | Memory? | Correlation **0.92** at 1h, **0.55** at 12h, back up to **0.72** at 24h. |
| Q12 | `nowcast` / `aqi`? | Correlate **0.97** / **0.94** with the same-hour target. |
| Q13 | Health? | Only **2%** of hours are `Good`. A third are `Unhealthy` or worse. |

### Three findings that constrain every later decision

**Q12 — `nowcast`, `aqi` and `aqi_category` contain the answer.** `nowcast` is a
weighted average of roughly the last twelve hours *including the current one*,
and `aqi` is computed from it — the EPA breakpoint formula reproduces 89.6% of
published AQI values from `nowcast`. The visible consequence is that
`aqi_category` bands overlap heavily in raw PM2.5: `Good` spans 0–19,
`Unhealthy` spans 13–272. Use any of the three for the current hour and the
model scores beautifully in testing and fails completely in reality.

Also: **94 rows are flagged `Valid` yet still hold `-999`** in `nowcast`. A
quality flag is not a guarantee.

**Q4 — `lag()` counts rows, not hours.** With 5.1% of hours missing, "one row
back" and "one hour back" are different things. Lagging raw rows would hand the
model a reading from 76 days earlier while labelling it `lag_1`.

**Q11 — 24 hours ago beats 12 hours ago.** The correlation curve falls and then
*rises again* at 24h. The daily cycle is real, and this is the evidence for
which lags are worth building.

## 5. Open decisions

The exploration deliberately settles nothing. These are the calls to make:

1. **Constant columns** — Q1 settles it: drop.
2. **`-999` rows** — Q2 settles it: drop. This leaves holes in time, which feeds into 3.
2b. **Duplicate hours and negative readings** — the 6 duplicates must go before anything is sorted by time. The 24 negatives can be nudged to 0 or dropped; nudging keeps the hour in the series, dropping punches another hole in it.
3. **Missing hours** — build a complete hourly timeline, or fill small gaps and cut around large ones. The 76-day gap needs handling either way.
4. **`nowcast` / `aqi`** — drop entirely, or keep only their *past* values. Their high correlation makes lagged versions potentially useful.
5. **Which lags** — 1, 2, 3 and 24 are supported by Q11. Whether 168 (one week) earns its place is open; it costs rows.
6. **Calendar inputs** — Q6 favours month, Q7 favours hour, Q8 argues against day of week.
7. **Transform the target?** — Q5 says `log(1+x)` is far more symmetric. Test both.
8. **Train/test cut-off** — must be by time, never `randomSplit`. Q3 and Q9 warn that the final partial year is dry-season only, so testing on it alone is seasonally biased.
9. **Which model** — gradient-boosted trees handle skew and interactions without scaling and report feature importance. A linear model makes a good sanity check.
10. **PCA?** — probably not. After dropping constant and leaky columns there are about three real inputs. Trees are untroubled by correlated features, and PCA would destroy the *"the reading 24 hours ago mattered most"* result. Revisit only if weather data or more stations are added.

## 6. Architecture

Two phases, deliberately separate.

```
┌─ TRAINER (batch, offline) ───────┐     ┌─ PREDICTOR (streaming, live) ────┐
│  historical CSV                  │     │  new hour arrives                 │
│      ↓ clean                     │     │      ↓ build lag features         │
│      ↓ fill hourly gaps          │     │      ↓ PipelineModel.load()       │
│      ↓ lag / rolling features    │     │      ↓ .transform()  PREDICT ONLY │
│      ↓ time-based train/test     │     │      ↓                            │
│      ↓ GBTRegressor.fit()        │     │  write forecast + alert           │
│  model.save(...) ────────────────┼────▶│                                   │
└──────────────────────────────────┘     └───────────────┬───────────────────┘
                 ▲                                       │
                 └───── accumulated predictions ─────────┘
                        + actuals → retrain → v2
```

**There is no training in the streaming phase.** This is the single most
important thing about the design.

Calling `pipeline.fit()` on a *streaming* DataFrame throws — Structured
Streaming supports only a subset of operations, and training is not one of them.
`.transform()` (inference) *is* supported. So training is a batch operation on
accumulated history, and prediction is a streaming operation on arriving data.

This is not a workaround. It is how essentially every production ML system
works, and training on a single hourly record would be statistically meaningless
anyway. MLlib's tree ensembles have no incremental `fit` in any Spark API.

The system still improves, on a **separate loop**: the streaming job writes its
predictions out, the batch trainer later reads that accumulated history and
produces `v2`, and the stream picks `v2` up mid-flight via a version marker.

### Rejected alternatives

| Option | Why not |
|---|---|
| `StreamingLinearRegressionWithSGD`, `StreamingKMeans` | Legacy RDD API driven by **DStreams**, deprecated since Spark 3.4. Linear models and k-means only — no GBT, no Random Forest. |
| `.fit()` inside `foreachBatch` | Legal, but one hour is one row. Refitting on accumulated history is just scheduled retraining with extra steps. |

### Two things that trip everyone

- **`readStream` cannot infer a schema.** Batch can; streaming cannot. This is why the notebook writes the schema out by hand in Step 7.
- **`F.lag().over(Window.orderBy(...))` does not work on a streaming DataFrame.** Non-time-based window functions are unsupported, and lags are the entire feature set. The way out is `foreachBatch`, where each micro-batch arrives as an ordinary batch DataFrame and `Window` works normally.

## 7. Conventions

- **One notebook.** Work is appended to `dhaka_pm25.ipynb`, never split into new files.
- **Seed everything.** A single `SEED` is fixed in Step 3 and passed to every sample, split and estimator, so a metric that moves between runs moves because something changed — not because the dice landed differently.
- **Never `randomSplit` on this data.** Shuffling lets the model train on the future and be tested on the past.
- **Rolling windows end at `-1`, never `0`.** `rowsBetween(-3, -1)` is the three hours before now; `rowsBetween(-2, 0)` includes the value being predicted.

## 8. Glossary

| Term | Meaning |
|---|---|
| **PM2.5** | Fine particulate matter under 2.5 µm, in µg/m³. What we forecast. |
| **AQI** | Air Quality Index, 0–500. A human-readable scale derived from PM2.5. |
| **NowCast** | AirNow's smoothed concentration, weighted over ~12 recent hours. |
| **Lag feature** | A past value used as an input. `lag_1` is the reading one hour ago. |
| **Data leakage** | Letting the answer into the inputs. Produces models that score well and fail in reality. |
| **RMSE** | Root mean squared error. Same units as the target. Lower is better. |
| **Persistence baseline** | The trivial forecast "next value = current value". Any real model must beat it. |
| **Estimator / Transformer** | An estimator must study the data first (`.fit()`); a transformer does not (`.transform()`). |
| **Pipeline** | An ordered list of stages. `Pipeline.fit()` returns a `PipelineModel` — save that, not the bare model. |
| **`foreachBatch`** | Structured Streaming escape hatch: hands you each micro-batch as a normal batch DataFrame. |

## 9. Team

| Name | Student ID |
|---|---|
| Mashrur Rahman | 20220104108 |
| Chowdhury Ajmayeen Adil | 20220104121 |
| Ahnuf Karim Chowdhury | 20220104122 |
| Nahid Asef | 20220104128 |

## 10. Status

**Done**
- Problem framing, targets and dataset selected
- Proposal deck complete (12 slides)
- Live data source located and verified after `dosairnowdata.org` went dead
- Part 1 written: full exploration of the data, Q1–Q15

**Next**
1. Run the notebook end to end
2. Settle the ten open decisions in section 5
3. Append **Part 2 — prepare the data**: act on decisions 1–6 and build the model inputs
4. Append **Part 3 — train**: time split, train, and check against the persistence baseline
5. Only then, the streaming predictor
