from pyspark.sql import SparkSession


def get_spark(app="d3-arm", cores="*", driver_mem="8g", shuffle_partitions=None):
    """Local-mode Spark session.

    cores: "*" for all cores, or a number (we change this for the core-scaling runs).
    driver_mem: in local mode everything runs inside the driver JVM, so this is
    basically the memory Spark gets. 8g is fine on a 16 GB laptop.
    """
    builder = (
        SparkSession.builder.appName(app)
        .master(f"local[{cores}]")
        .config("spark.driver.memory", driver_mem)
        .config("spark.sql.session.timeZone", "UTC")
    )
    if shuffle_partitions:
        builder = builder.config("spark.sql.shuffle.partitions", str(shuffle_partitions))
    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark
