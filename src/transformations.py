# Data transformations script placeholder
from ingestion import load_and_process_data
from pyspark.sql.functions import col, count, avg, sum, when, hour, rank
from pyspark.sql.window import Window


def run_transformations(df):

    print("\n================ COMPLEX QUERY 1 =================")
    print("Fraud rate by transaction category")

    category_fraud = df.groupBy("category").agg(
        count("*").alias("total_txn"),
        sum("is_fraud").alias("fraud_txn")
    ).withColumn(
        "fraud_rate", col("fraud_txn") / col("total_txn")
    ).orderBy(col("fraud_rate").desc())

    category_fraud.show(10, False)


    print("\n================ COMPLEX QUERY 2 =================")
    print("Fraud pattern by hour of day")

    df_with_hour = df.withColumn("txn_hour", hour(col("trans_date_trans_time")))

    hourly_fraud = df_with_hour.groupBy("txn_hour").agg(
        count("*").alias("total_txn"),
        sum("is_fraud").alias("fraud_txn")
    ).withColumn(
        "fraud_rate", col("fraud_txn") / col("total_txn")
    ).orderBy("txn_hour")

    hourly_fraud.show(24, False)


    print("\n================ COMPLEX QUERY 3 =================")
    print("Top high-risk customers based on fraud rate")

    customer_risk = df.groupBy("cc_num").agg(
        count("*").alias("total_txn"),
        sum("is_fraud").alias("fraud_txn"),
        avg("amt").alias("avg_amt")
    ).withColumn(
        "fraud_rate", col("fraud_txn") / col("total_txn")
    ).filter(col("total_txn") > 50) \
     .orderBy(col("fraud_rate").desc())

    customer_risk.show(10, False)


    print("\n================ COMPLEX QUERY 4 =================")
    print("Distance-based fraud analysis (bucketed)")

    distance_bucket = df.withColumn(
        "distance_bucket",
        when(col("distance_km") < 50, "Near")
        .when(col("distance_km") < 200, "Medium")
        .otherwise("Far")
    )

    distance_analysis = distance_bucket.groupBy("distance_bucket").agg(
        count("*").alias("total_txn"),
        sum("is_fraud").alias("fraud_txn")
    ).withColumn(
        "fraud_rate", col("fraud_txn") / col("total_txn")
    ).orderBy(col("fraud_rate").desc())

    distance_analysis.show(truncate=False)


    print("\n================ COMPLEX QUERY 5 =================")
    print("High-value transactions fraud impact")

    high_value = df.withColumn(
        "high_value_flag",
        when(col("amt") > 500, 1).otherwise(0)
    )

    high_value_analysis = high_value.groupBy("high_value_flag").agg(
        count("*").alias("total_txn"),
        sum("is_fraud").alias("fraud_txn")
    ).withColumn(
        "fraud_rate", col("fraud_txn") / col("total_txn")
    )

    high_value_analysis.show(truncate=False)


    print("\n================ COMPLEX QUERY 6 =================")
    print("Top fraud-prone categories ranked within each state")

    window_spec = Window.partitionBy("state").orderBy(col("fraud_rate").desc())

    state_category = df.groupBy("state", "category").agg(
        count("*").alias("total_txn"),
        sum("is_fraud").alias("fraud_txn")
    ).withColumn(
        "fraud_rate", col("fraud_txn") / col("total_txn")
    )

    ranked = state_category.withColumn(
        "rank", rank().over(window_spec)
    ).filter(col("rank") <= 3)

    ranked.show(50, truncate=False)


if __name__ == "__main__":

    df = load_and_process_data()
    run_transformations(df)
