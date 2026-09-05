"""SparkSession factory."""

from pyspark.sql import SparkSession


def get_spark(app_name: str = "dhaka-pm25-phase1",
              driver_memory: str = "4g") -> SparkSession:
    """Local SparkSession sized for Colab's standard runtime.

    Note on hardware: Spark MLlib is CPU-only. GBTRegressor has no CUDA path,
    so a Colab GPU runtime buys this job nothing -- pick a CPU runtime.
    """
    return (SparkSession.builder
            .appName(app_name)
            .master("local[*]")
            .config("spark.driver.memory", driver_memory)
            .config("spark.sql.session.timeZone", "Asia/Dhaka")
            .config("spark.sql.shuffle.partitions", "8")
            .getOrCreate())
