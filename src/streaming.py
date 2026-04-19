from pyspark.sql.functions import (
    col, when, hour, sum, count,
    radians, sin, cos, sqrt, atan2, avg, dayofweek
)
from pyspark.sql.window import Window
from utils import get_spark_session
import os
import shutil


def run_streaming():

    spark = get_spark_session("Fraud-Streaming")

    # -----------------------------
    # Load test dataset
    # -----------------------------
    test_df = spark.read.csv(
        "data/CreditCard_Fraud_Test.csv",
        header=True,
        inferSchema=True
    )

    # Remove unwanted index column if present
    if "_c0" in test_df.columns:
        test_df = test_df.drop("_c0")

    print("Test dataset loaded")
    total_count = test_df.count()
    print("Total rows:", total_count)

    # -----------------------------
    # Split into batch & streaming
    # -----------------------------
    batch_df = test_df.limit(total_count // 2)
    stream_df = test_df.subtract(batch_df)

    print("Batch rows:", batch_df.count())
    print("Streaming rows:", stream_df.count())

    # -----------------------------
    # Prepare streaming folder
    # -----------------------------
    stream_path = "data/stream_input"

    if os.path.exists(stream_path):
        shutil.rmtree(stream_path)

    os.makedirs(stream_path)

    # -----------------------------
    # Apply Feature Engineering
    # -----------------------------

    # Weekend indicator
    stream_df = stream_df.withColumn(
        "is_weekend",
        when(dayofweek(col("trans_date_trans_time")).isin([1, 7]), 1).otherwise(0)
    )

    # Distance between customer and merchant (Haversine formula)
    stream_df = stream_df \
        .withColumn("lat1", radians(col("lat"))) \
        .withColumn("lon1", radians(col("long"))) \
        .withColumn("lat2", radians(col("merch_lat"))) \
        .withColumn("lon2", radians(col("merch_long")))

    stream_df = stream_df \
        .withColumn("dlat", col("lat2") - col("lat1")) \
        .withColumn("dlon", col("lon2") - col("lon1"))

    stream_df = stream_df.withColumn("a",
        sin(col("dlat") / 2) ** 2 +
        cos(col("lat1")) * cos(col("lat2")) * sin(col("dlon") / 2) ** 2
    )

    stream_df = stream_df \
        .withColumn("c", 2 * atan2(sqrt(col("a")), sqrt(1 - col("a")))) \
        .withColumn("distance_km", col("c") * 6371) \
        .drop("lat1", "lon1", "lat2", "lon2", "dlat", "dlon", "a", "c")

    # Average recent transaction amount per card
    window_spec = Window.partitionBy("cc_num").orderBy("trans_date_trans_time")
    stream_df = stream_df.withColumn(
        "avg_amt_recent",
        avg("amt").over(window_spec.rowsBetween(-10, 0))
    )

    print("Feature engineering applied to streaming data!")
    print("Features added: is_weekend, distance_km, avg_amt_recent")

    # -----------------------------
    # Save streaming data WITH features
    # -----------------------------
    stream_df = stream_df.repartition(5)

    stream_df.write \
        .option("header", True) \
        .mode("overwrite") \
        .csv(stream_path)

    print("Streaming input files created with features!")

    # -----------------------------
    # Define schema
    # -----------------------------
    schema = stream_df.schema

    # -----------------------------
    # Read as streaming
    # -----------------------------
    streaming_df = spark.readStream \
        .option("header", True) \
        .schema(schema) \
        .csv(stream_path)

    print("Streaming started...")

    # -----------------------------
    # Transformations
    # -----------------------------
    enriched_df = streaming_df \
        .withColumn("txn_hour", hour(col("trans_date_trans_time"))) \
        .withColumn("high_value_flag", when(col("amt") > 500, 1).otherwise(0))

    # -----------------------------
    # Streaming Queries
    # -----------------------------

    # Fraud by category
    fraud_by_category = enriched_df.groupBy("category").agg(
        sum("is_fraud").alias("fraud_txn"),
        count("*").alias("total_txn")
    )

    # Fraud trend by hour
    fraud_by_hour = enriched_df.groupBy("txn_hour").agg(
        sum("is_fraud").alias("fraud_txn"),
        count("*").alias("total_txn")
    )

    # High-value fraud detection
    high_value_fraud = enriched_df.groupBy("high_value_flag").agg(
        sum("is_fraud").alias("fraud_txn"),
        count("*").alias("total_txn")
    )

    # -----------------------------
    # Output streams
    # -----------------------------
    query1 = fraud_by_category.writeStream \
        .outputMode("complete") \
        .format("console") \
        .option("truncate", False) \
        .start()

    query2 = fraud_by_hour.writeStream \
        .outputMode("complete") \
        .format("console") \
        .option("truncate", False) \
        .start()

    query3 = high_value_fraud.writeStream \
        .outputMode("complete") \
        .format("console") \
        .option("truncate", False) \
        .start()

    # -----------------------------
    # Keep alive
    # -----------------------------
    spark.streams.awaitAnyTermination()


if __name__ == "__main__":
    run_streaming()