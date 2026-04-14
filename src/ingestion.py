# Data ingestion script placeholder
from utils import get_spark_session, get_logger
from pyspark.sql.functions import col, to_timestamp, when, dayofweek, year, month, day, hour, minute
from pyspark.sql.functions import radians, sin, cos, sqrt, atan2, avg, trim, pi
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
        to_timestamp(col("trans_date_trans_time"), "M/d/yyyy H:mm")
    )

    # Handle missing values (empty strings -> null)
    df = df.replace("", None)

    # Trim string columns (removes spaces like "   ")
    string_columns = ["category", "merchant", "city", "state"]
    for column in string_columns:
        df = df.withColumn(column, trim(col(column)))

    # Drop rows with nulls in any important column
    df = df.dropna(subset=[
        "trans_date_trans_time",
        "amt",
        "category",
        "lat",
        "long",
        "merch_lat",
        "merch_long",
        "cc_num",
        "is_fraud"
    ])

    logger.info("Null and missing values handled strictly")

    # -----------------------------
    # Feature Engineering
    # -----------------------------

    # Weekend indicator
    df = df.withColumn(
        "is_weekend",
        when(dayofweek(col("trans_date_trans_time")).isin([1, 7]), 1).otherwise(0)
    )

    # Break transaction timestamp up
    df = df.withColumn("trans_year", year(col("trans_date_trans_time"))) \
        .withColumn("trans_month", month(col("trans_date_trans_time"))) \
        .withColumn("trans_day", day(col("trans_date_trans_time"))) \
        .withColumn("trans_hour", hour(col("trans_date_trans_time"))) \
        .withColumn("trans_minute", minute(col("trans_date_trans_time")))
    
    # Make time cyclical
    df = df.withColumn("sin_hour", sin((col("trans_hour") * 2 * pi()) / 24)) \
        .withColumn("cos_hour", cos((col("trans_hour") * 2 * pi()) / 24)) \
        .withColumn("sin_minute", sin((col("trans_minute") * 2 * pi()) / 60)) \
        .withColumn("cos_minute", cos((col("trans_minute") * 2 * pi()) / 60)) \
        .drop("trans_hour").drop("trans_minute")

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

    # Average recent transaction amount per user
    window_spec = Window.partitionBy("cc_num").orderBy("trans_date_trans_time")

    df = df.withColumn(
        "avg_amt_recent",
        avg("amt").over(window_spec.rowsBetween(-10, 0))
    )

    logger.info("Feature engineering completed")

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
# Create fraud weights to penalize missing fraud
# -----------------------------
def create_fraud_weight(train_df):

    fraud_count = train_df.filter(col("is_fraud") == 1).count()
    nonfraud_count = train_df.filter(col("is_fraud") == 0).count()

    ratio = nonfraud_count / fraud_count

    return train_df.withColumn(
        "fraud_weight",
        when(col("is_fraud") == 1, ratio).otherwise(1.0)
    )


# -----------------------------
# Main execution
# -----------------------------

if __name__ == "__main__":

    df = load_and_process_data()

    train_df, test_df = split_train_test(df)

    print("\nTrain Count:", train_df.count())
    print("Test Count:", test_df.count())