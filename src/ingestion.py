from utils import get_spark_session, get_logger
from pyspark.sql.functions import col, to_timestamp, when, dayofweek
from pyspark.sql.functions import radians, sin, cos, sqrt, atan2, avg, trim, mean
from pyspark.sql.window import Window


def load_and_process_data():

    logger = get_logger("Ingestion")
    spark = get_spark_session("Fraud-Ingestion")

    logger.info("Starting ingestion process")

    # -----------------------------
    # Load datasets
    # -----------------------------
    train_path = "data/CreditCard_Fraud_Train.csv"
    test_path = "data/CreditCard_Fraud_Test.csv"

    train_df = spark.read.csv(train_path, header=True, inferSchema=True).drop("_c0")
    test_df = spark.read.csv(test_path, header=True, inferSchema=True).drop("_c0")

    # Add identifier before combining
    train_df = train_df.withColumn("dataset_type", col("is_fraud") * 0 + 1)
    test_df = test_df.withColumn("dataset_type", col("is_fraud") * 0 + 2)

    df = train_df.unionByName(test_df)

    logger.info("Datasets loaded and combined")

    # -----------------------------
    # Basic cleaning (strict)
    # -----------------------------

    # Remove duplicates
    df = df.dropDuplicates()

    # Convert timestamp column properly
    df = df.withColumn(
        "trans_date_trans_time",
        to_timestamp(col("trans_date_trans_time"))
    )

    # Handle missing values (empty strings -> null)
    df = df.replace("", None)

    # Trim string columns (removes spaces like "   ")
    string_columns = ["category", "merchant", "city", "state"]
    for column in string_columns:
        df = df.withColumn(column, trim(col(column)))

    logger.info("Basic cleaning completed")

    # -----------------------------
    # Handle Missing Values
    # -----------------------------
 
    # Calculate mean for numeric columns
    amt_mean      = df.select(mean(col("amt"))).first()[0]
    lat_mean      = df.select(mean(col("lat"))).first()[0]
    long_mean     = df.select(mean(col("long"))).first()[0]
    merch_lat_mean = df.select(mean(col("merch_lat"))).first()[0]
    merch_long_mean = df.select(mean(col("merch_long"))).first()[0]
    city_pop_mean  = df.select(mean(col("city_pop"))).first()[0]
 
    # Fill missing numeric values with mean
    df = df.fillna({
        "amt"       : amt_mean,
        "lat"       : lat_mean,
        "long"      : long_mean,
        "merch_lat" : merch_lat_mean,
        "merch_long": merch_long_mean,
        "city_pop"  : city_pop_mean
    })
 
    # Fill missing string values with "Unknown"
    df = df.fillna({
        "category" : "Unknown",
        "merchant" : "Unknown",
        "city"     : "Unknown",
        "state"    : "Unknown",
        "gender"   : "Unknown"
    })
 
    logger.info("Missing values filled with mean (numeric) and Unknown (string)")

    # -----------------------------
    # Feature Engineering
    # -----------------------------

    # Weekend indicator
    df = df.withColumn(
        "is_weekend",
        when(dayofweek(col("trans_date_trans_time")).isin([1, 7]), 1).otherwise(0)
    )

    # Distance calculation (Haversine)
    df = df.withColumn("lat1", radians(col("lat"))) \
           .withColumn("lon1", radians(col("long"))) \
           .withColumn("lat2", radians(col("merch_lat"))) \
           .withColumn("lon2", radians(col("merch_long")))

    df = df.withColumn("dlat", col("lat2") - col("lat1")) \
           .withColumn("dlon", col("lon2") - col("lon1"))

    df = df.withColumn("a",
        sin(col("dlat")/2)**2 +
        cos(col("lat1")) * cos(col("lat2")) * sin(col("dlon")/2)**2
    )

    df = df.withColumn("c", 2 * atan2(sqrt(col("a")), sqrt(1-col("a"))))

    df = df.withColumn("distance_km", col("c") * 6371)

    # Drop intermediate columns
    df = df.drop("lat1", "lon1", "lat2", "lon2", "dlat", "dlon", "a", "c")

    # Average recent transaction amount per user
    window_spec = Window.partitionBy("cc_num").orderBy("trans_date_trans_time")

    df = df.withColumn(
        "avg_amt_recent",
        avg("amt").over(window_spec.rowsBetween(-10, 0))
    )

    # 2: Amount buckets
    df = df.withColumn("amt_bucket",
        when(col("amt") < 500,   "0-500")
        .when(col("amt") < 1000,  "500-1000")
        .when(col("amt") < 5000,  "1000-5000")
        .when(col("amt") < 10000, "5000-10000")
        .otherwise("10000+")
    )
 
    logger.info("Feature engineering completed — is_weekend, distance_km, avg_amt_recent, amt_bucket")

    # -----------------------------
    # EDA SECTION
    # -----------------------------

    print("\n===== Sample Data =====")
    df.show(5)

    print("\n===== Schema =====")
    df.printSchema()

    print("\n===== Fraud Distribution =====")
    df.groupBy("is_fraud").count().show()

    print("\n===== Amount Statistics =====")
    df.select("amt").describe().show()

    print("\n===== Category Distribution =====")
    df.groupBy("category").count().orderBy("count", ascending=False).show(10)

    print("\n===== Weekend vs Fraud =====")
    df.groupBy("is_weekend").agg({"is_fraud": "avg"}).show()

    print("\n===== Distance vs Fraud =====")
    df.groupBy("is_fraud").agg({"distance_km": "avg"}).show()

    print("\n===== High Value Transactions =====")
    df.filter(col("amt") > 1000).groupBy("is_fraud").count().show()

    logger.info("EDA completed")

    return df


# -----------------------------
# Split datasets back
# -----------------------------

def split_train_test(df):

    train_df = df.filter(col("dataset_type") == 1)
    test_df = df.filter(col("dataset_type") == 2)

    # Remove helper column
    train_df = train_df.drop("dataset_type")
    test_df = test_df.drop("dataset_type")

    return train_df, test_df


# -----------------------------
# Main execution
# -----------------------------

if __name__ == "__main__":

    df = load_and_process_data()

    train_df, test_df = split_train_test(df)

    print("\nTrain Count:", train_df.count())
    print("Test Count:", test_df.count())
    print("\n===== Amount Bucket Sample =====")
    train_df.groupBy("amt_bucket").count().orderBy("amt_bucket").show()
 