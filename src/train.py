"""Phase 1: split, baseline, train, evaluate, save."""

from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import GBTRegressor
from pyspark.sql import DataFrame, functions as F

from .config import FEATURE_COLS, GBT_PARAMS, LABEL_COL, MODEL_DIR, SPLIT_TS


def time_split(df: DataFrame, split_ts: str = SPLIT_TS):
    """Split by time, never randomly (README 6.6).

    `randomSplit` shuffles rows, letting the model train on the future and test
    on the past. On a time series that inflates every metric into meaninglessness.
    """
    train = df.filter(F.col("ts") < F.lit(split_ts))
    test = df.filter(F.col("ts") >= F.lit(split_ts))
    return train, test


def persistence_rmse(test: DataFrame) -> float:
    """Baseline: 'next hour = this hour'. Prediction is simply lag_1.

    Any model that cannot beat this has learned nothing about the series.
    """
    evaluator = RegressionEvaluator(
        labelCol=LABEL_COL, predictionCol="lag_1", metricName="rmse")
    return evaluator.evaluate(test)


def build_pipeline() -> Pipeline:
    assembler = VectorAssembler(inputCols=FEATURE_COLS, outputCol="features")
    gbt = GBTRegressor(featuresCol="features", labelCol=LABEL_COL, **GBT_PARAMS)
    return Pipeline(stages=[assembler, gbt])


def evaluate(model: PipelineModel, test: DataFrame) -> dict:
    preds = model.transform(test).cache()
    out = {}
    for metric in ("rmse", "mae", "r2"):
        out[metric] = RegressionEvaluator(
            labelCol=LABEL_COL, predictionCol="prediction",
            metricName=metric).evaluate(preds)
    preds.unpersist()
    return out


def save(model: PipelineModel, path: str = MODEL_DIR) -> str:
    """Write the PipelineModel Phase 2 will load.

    Save the whole PipelineModel, not the bare GBT -- the assembler stage has
    to travel with it or the streaming job cannot build a `features` vector.
    """
    model.write().overwrite().save(path)
    return path


def feature_importances(model: PipelineModel) -> list:
    """(feature, importance) pairs, most important first."""
    gbt = model.stages[-1]
    pairs = list(zip(FEATURE_COLS, gbt.featureImportances.toArray()))
    return sorted(pairs, key=lambda p: p[1], reverse=True)
