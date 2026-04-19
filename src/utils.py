import os
# os.environ["JAVA_HOME"] = "/opt/homebrew/opt/openjdk@17"

import logging
from pyspark.sql import SparkSession


def get_spark_session(app_name="FraudDetectionProject"):
    """
    Create and return a SparkSession.
    """

    spark = (
        SparkSession.builder
        .appName(app_name)
        .config("spark.driver.memory", "4g")
        .config("spark.executor.memory", "4g")
        .config("spark.memory.fraction", "0.8")
        .config("spark.memory.offHeap.enabled", "true")
        .config("spark.memory.offHeap.size", "2g")
        .getOrCreate()
    )

    return spark


def get_logger(logger_name="FraudProjectLogger"):
    """
    Create and return a reusable logger.
    Prevents duplicate handlers when imported across files.
    """

    logger = logging.getLogger(logger_name)

    if not logger.handlers:
        logger.setLevel(logging.INFO)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        console_handler.setFormatter(formatter)

        logger.addHandler(console_handler)

    logger.propagate = False

    return logger
