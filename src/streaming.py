from pyspark.sql.functions import col, when, hour, sum, count
from utils import get_spark_session
import os
import shutil


def run_streaming():

    spark = get_spark_session("Fraud-Streaming")

    # -----------------------------
    # STEP 1: Load test dataset
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
    # STEP 2: Split into batch & streaming
    # -----------------------------
    batch_df = test_df.limit(total_count // 2)
    stream_df = test_df.subtract(batch_df)

    print("Batch rows:", batch_df.count())
    print("Streaming rows:", stream_df.count())

    # -----------------------------
    # STEP 3: Prepare streaming folder
    # -----------------------------
    stream_path = "data/stream_input"

    if os.path.exists(stream_path):
        shutil.rmtree(stream_path)

    os.makedirs(stream_path)

    # -----------------------------
    # STEP 4: Create streaming input files
    # -----------------------------
    # repartition ensures multiple files (simulate micro-batches)
    stream_df = stream_df.repartition(5)

    stream_df.write \
        .option("header", True) \
        .mode("overwrite") \
        .csv(stream_path)

    print("Streaming input files created")

    # -----------------------------
    # STEP 5: Define schema
    # -----------------------------
    schema = test_df.schema

    # -----------------------------
    # STEP 6: Read as streaming
    # -----------------------------
    streaming_df = spark.readStream \
        .option("header", True) \
        .schema(schema) \
        .csv(stream_path)

    print("Streaming started...")

    # -----------------------------
    # STEP 7: Transformations
    # -----------------------------
    enriched_df = streaming_df \
        .withColumn("txn_hour", hour(col("trans_date_trans_time"))) \
        .withColumn("high_value_flag", when(col("amt") > 500, 1).otherwise(0))

    # -----------------------------
    # STEP 8: Streaming Queries
    # -----------------------------

    # Query 1: Fraud by category
    fraud_by_category = enriched_df.groupBy("category").agg(
        sum("is_fraud").alias("fraud_txn"),
        count("*").alias("total_txn")
    )

    # Query 2: Fraud trend by hour
    fraud_by_hour = enriched_df.groupBy("txn_hour").agg(
        sum("is_fraud").alias("fraud_txn"),
        count("*").alias("total_txn")
    )

    # Query 3: High-value fraud detection
    high_value_fraud = enriched_df.groupBy("high_value_flag").agg(
        sum("is_fraud").alias("fraud_txn"),
        count("*").alias("total_txn")
    )

    # -----------------------------
    # STEP 9: Output streams
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
    # STEP 10: Keep alive
    # -----------------------------
    spark.streams.awaitAnyTermination()


if __name__ == "__main__":
    run_streaming()