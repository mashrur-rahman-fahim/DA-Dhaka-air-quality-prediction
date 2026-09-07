# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A university big-data project: short-term PM2.5 forecasting for Dhaka using PySpark.
The entire deliverable is **one Jupyter notebook**, `dhaka_pm25.ipynb`, which the user
runs in **Google Colab**. There is no application, no test suite, no build step, and no
Python package. `README.md` is the project write-up and is kept in sync with the
notebook's actual measured output.

The notebook is organised as Part 1 (explore) and Part 2 (prepare). Part 3 (train and
evaluate) is not written yet.

## The one rule that shapes everything

**Never create a second notebook.** New work is appended as cells to the end of
`dhaka_pm25.ipynb`. A new notebook file means a new Colab URL, a fresh kernel, and
re-running pip plus the Spark startup plus the data download. Read the `.ipynb`, append
to its `cells` list, write it back, push, then tell the user which cell number to start
running from.

## There is no local runtime

The dev machine has **no JVM**, so PySpark cannot be executed here. Do not try to
install Java or run the notebook locally — the user has explicitly asked to keep load off
their machine.

Two consequences:

**Verify logic without Spark.** Before pushing a change to feature engineering, splits
or baselines, mirror the same steps in plain Python against the local CSV and check the
numbers. `data/raw/dhaka_pm25_airnow_2016_2025_raw.csv` (gitignored) is a
deduplicated concatenation of the ten yearly files, kept for exactly this. `pandas` and
`numpy` are not installed; use `csv`, `datetime`, `statistics` and `math`. If the file is
missing, rebuild it by downloading the yearly CSVs listed in "Data source" below and
concatenating them, deduplicating on `Date (LT)`.

**Read results from git.** The user runs the notebook in Colab and saves it back to
GitHub with outputs embedded. `git pull`, then parse `dhaka_pm25.ipynb` for
`output_type == "error"` and for the printed values. That is the only way to see what the
code actually did.

## Editing the notebook

Edit the `.ipynb` JSON directly with a Python script. Two things that have already caused
shipped bugs:

**Cell source must keep its newlines.** Build every `source` list with
`text.splitlines(keepends=True)`, never `text.split("\n")`. The latter strips the newline
from each element; renderers join with no separator and the whole cell collapses onto one
line — headings run into body text and code becomes a `SyntaxError`. After writing,
assert that no element except the last carries no trailing `\n`.

**Clear outputs on cells you edit.** Set `outputs = []` and `execution_count = None` on
any cell whose source changes, so stale output does not sit under new code.

Keep `nbformat: 4`, `nbformat_minor: 0`. Cells need no `id` field at that version.

## Working with Colab and GitHub

The user and this session both push to the same branch, and **Colab's "Save in GitHub"
force-pushes against whatever revision the browser tab loaded**. It has already silently
discarded a commit. The protocol that avoids it:

1. Push, then give the user a **commit-pinned** Colab URL:
   `https://colab.research.google.com/github/mashrur-rahman-fahim/DA-Dhaka-air-quality-prediction/blob/<sha>/dhaka_pm25.ipynb`
2. Do not push again while they are running.
3. They save from Colab; `git pull --rebase` and read the outputs.

Always hand out the pinned form. The `blob/main/` form is cached by Colab and has served
a stale notebook more than once. Before giving a URL, confirm `origin/main` matches local
HEAD, and tell the user whether saving is currently safe.

For a small fix mid-run, paste the corrected cell body into the chat instead of pushing —
that preserves their Spark session and computed variables.

## Data source

The address in most tutorials, `dosairnowdata.org`, no longer resolves. The live public
mirror, no API key:

```
https://s3-us-west-1.amazonaws.com/files.airnowtech.org/airnow/EmbassyHistorical/Dhaka/<YEAR>/Dhaka_PM2.5_<YEAR>_YTD.csv
```

US Embassy Dhaka, station `DK1010001`, hourly PM2.5, 2016-03-01 to **2025-03-24**. It
stops there because the State Department defunded the embassy monitoring programme in
March 2025; Dhaka's monitor was never restored, so there is no 2026 data and there will
not be. Some cities (New Delhi, Lima) did resume — Dhaka did not.

Raw-file facts that a pre-cleaned copy would hide: `QC Name` is **not** all `Valid`
(2,138 `Missing`, 197 `Invalid`, 7 `Suspect`, all carrying `-999`); 94 rows are flagged
`Valid` yet still hold `-999` in `NowCast Conc.`; six timestamps appear twice with
**different** readings, mostly on US daylight-saving switch dates.

## Domain invariants

Violating any of these produces a model that scores well and is worthless. They are
enforced in the notebook and each is justified by a numbered question in Part 1.

**Lags must mean hours, not rows.** 5.1% of hours are missing across 517 gaps, the worst
being 1,822 hours (76 days, Aug–Nov 2018). `F.lag()` counts rows, so Step 11 builds a
complete hourly timeline first — generate every hour with `sequence` + `explode`, then
left-join the readings onto it. Any new lag or window must be computed on that complete
frame, before the coverage gate filters rows out of it.

**Rolling windows end at `-1`, never `0`.** `rowsBetween(-2, 0)` includes the value being
predicted.

**`nowcast`, `aqi` and `aqi_category` leak.** NowCast is a ~12-hour weighted average
including the current hour, and AQI derives from it (the EPA breakpoint formula
reproduces 89.6% of published AQI values from NowCast). Same-hour correlations with the
target are 0.97 and 0.94. Only `nowcast` lagged is admissible.

**The classification target is built, not taken.** `aqi_category` labels NowCast, so
copying the previous hour's value is already right 83.4% of the time. The target is
instead the EPA category of the raw reading, where the same naive forecast scores 71.9%.
`epa_category()` is defined in the notebook; reuse it rather than restating thresholds.

**Never `randomSplit`.** The split is three-way and by date: train `< 2023-01-01`,
validation to `< 2024-01-01`, test onward. All model-selection decisions belong on
validation; the test set is opened once.

**Seed everything.** One `SEED = 42` set in Step 3 and passed to every sample, split and
estimator. Deterministic aggregation matters too: the six duplicate hours are collapsed by
**averaging**, not `dropDuplicates`, which would return different rows on different runs.

**Session timezone is UTC, deliberately.** The readings are Dhaka wall-clock labels and
nothing converts between zones. Setting `Asia/Dhaka` makes Spark print one time while
Python's `.first()` reports another six hours apart, because Colab's own clock is UTC —
this shipped once and made a correct split look like it leaked. Render any timestamp you
print with `F.date_format` rather than pulling it into Python.

## Numbers the notebook produces

Useful for spotting a regression without a full re-run. All verified against real output.

| | |
|---|---|
| Rows in the ten files | 77,716 |
| `Valid` rows | 75,374, of which 6 are duplicate hours |
| Timeline hours / missing | 79,457 / 4,089 (5.1%) |
| After coverage gate and `dropna` | 72,722 |
| Inputs / targets | 14 / `pm25` and `category` |
| train / valid / test | 53,445 / 8,622 / 10,655 |
| Persistence baseline, validation | RMSE 34.11, label 73.5% |
| Persistence baseline, test | RMSE 35.58, label 75.0% |

## Environment pins

`pyspark==4.0.4`. Colab runs Python 3.13, which PySpark 3.5 predates, and Colab
pre-installs a package requiring `pyspark ~= 4.0.0`. Also on Python 3.13 the `imp` module
is gone, so IPython's `autoreload` extension raises `ModuleNotFoundError` — do not use
`%autoreload` in this notebook.

## Writing style for the notebook

The user is learning as the project is built and has asked repeatedly for beginner-level
explanation. Every code cell gets a markdown cell above it saying what it does, why it
matters, and what to look for in the output. Prefer showing evidence over asserting a
conclusion: measure the thing, print the number, then state the verdict. Keep code inline
and readable rather than hidden behind helper modules — an earlier `src/` layout was
removed for exactly this reason.
