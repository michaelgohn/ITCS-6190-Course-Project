from utils import get_spark_session, get_logger
from ingestion import load_and_process_data, split_train_test

from pyspark.sql.functions import col as spark_col,when
from pyspark.sql.types import (
    StructType, StructField, StringType, DoubleType,
    IntegerType, LongType, TimestampType, DateType
)
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import VectorAssembler, StringIndexer, StandardScaler
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator, MulticlassClassificationEvaluator


# -----------------------------
# Helper: Evaluate any model
# -----------------------------
def evaluate_model(predictions, model_name):

    print(f"\n===== {model_name} — Evaluation Metrics =====")

    auc = BinaryClassificationEvaluator(
        labelCol="is_fraud",
        rawPredictionCol="rawPrediction",
        metricName="areaUnderROC"
    ).evaluate(predictions)

    accuracy = MulticlassClassificationEvaluator(
        labelCol="is_fraud",
        predictionCol="prediction",
        metricName="accuracy"
    ).evaluate(predictions)

    precision = MulticlassClassificationEvaluator(
        labelCol="is_fraud",
        predictionCol="prediction",
        metricName="weightedPrecision"
    ).evaluate(predictions)

    recall = MulticlassClassificationEvaluator(
        labelCol="is_fraud",
        predictionCol="prediction",
        metricName="weightedRecall"
    ).evaluate(predictions)

    f1 = MulticlassClassificationEvaluator(
        labelCol="is_fraud",
        predictionCol="prediction",
        metricName="f1"
    ).evaluate(predictions)

    print(f"AUC-ROC   : {round(auc, 4)}")
    print(f"Accuracy  : {round(accuracy, 4)}")
    print(f"Precision : {round(precision, 4)}")
    print(f"Recall    : {round(recall, 4)}")
    print(f"F1 Score  : {round(f1, 4)}")

    print(f"\n===== {model_name} — Sample Predictions =====")
    predictions.select("amt", "category", "is_fraud", "prediction", "probability") \
        .show(10, truncate=False)

    print(f"\n===== {model_name} — Confusion Matrix =====")
    predictions.groupBy("is_fraud", "prediction") \
        .count() \
        .orderBy("is_fraud", "prediction") \
        .show()

    return {
        "model"    : model_name,
        "auc"      : round(auc, 4),
        "accuracy" : round(accuracy, 4),
        "precision": round(precision, 4),
        "recall"   : round(recall, 4),
        "f1"       : round(f1, 4)
    }


# -----------------------------
# Main ML Pipeline
# -----------------------------
def run_ml_pipeline():

    logger = get_logger("MLPipeline")
    spark  = get_spark_session("Fraud-MLPipeline")
    logger.info("Starting ML Pipeline")

    # Load and Process Data
    df = load_and_process_data()
    train_df, test_df = split_train_test(df)

    train_df = train_df.withColumn("is_fraud", spark_col("is_fraud").cast("double"))
    test_df  = test_df.withColumn("is_fraud",  spark_col("is_fraud").cast("double"))

    print(f"\nTrain rows: {train_df.count()}")
    print(f"Test rows : {test_df.count()}")
    logger.info("Data loaded and split into train/test")

    # StringIndexers
    category_indexer = StringIndexer(
        inputCol="category",
        outputCol="category_index",
        handleInvalid="keep"
    )

    gender_indexer = StringIndexer(
        inputCol="gender",
        outputCol="gender_index",
        handleInvalid="keep"
    )

    logger.info("StringIndexers defined for category and gender")

    # VectorAssembler
    feature_cols = [
        "amt",
        "city_pop",
        "is_weekend",
        "distance_km",
        "avg_amt_recent",
        "category_index",
        "gender_index"
    ]

    assembler = VectorAssembler(
        inputCols=feature_cols,
        outputCol="features",
        handleInvalid="keep"
    )

    logger.info(f"VectorAssembler defined with features: {feature_cols}")

    # StandardScaler
    scaler = StandardScaler(
        inputCol="features",
        outputCol="scaled_features",
        withStd=True,
        withMean=False
    )

    logger.info("StandardScaler defined")

    # Model 1 — Logistic Regression (Baseline)
    lr = LogisticRegression(
        featuresCol="scaled_features",
        labelCol="is_fraud",
        maxIter=10,
        regParam=0.01
    )

    lr_pipeline = Pipeline(stages=[
        category_indexer,
        gender_indexer,
        assembler,
        scaler,
        lr
    ])

    print("\nTraining Logistic Regression model...")
    lr_model       = lr_pipeline.fit(train_df)
    lr_predictions = lr_model.transform(test_df)
    lr_metrics     = evaluate_model(lr_predictions, "Logistic Regression")

    lr_model.write().overwrite().save("models/fraud_lr_pipeline")
    print("\nLogistic Regression model saved to: models/fraud_lr_pipeline")
    logger.info(f"LR — AUC: {lr_metrics['auc']}, F1: {lr_metrics['f1']}")

    # Add class weights to handle imbalance
    train_df = train_df.withColumn(
        "weight",
        when(spark_col("is_fraud") == 1.0, 200.0)
        .otherwise(1.0)
    )
    # Model 2 — Random Forest (Improved)
    rf = RandomForestClassifier(
        featuresCol="scaled_features",
        labelCol="is_fraud",
        weightCol="weight",
        numTrees=100,
        maxDepth=10,
        seed=42
    )

    rf_pipeline = Pipeline(stages=[
        category_indexer,
        gender_indexer,
        assembler,
        scaler,
        rf
    ])

    print("\nTraining Random Forest model...")
    rf_model       = rf_pipeline.fit(train_df)
    rf_predictions = rf_model.transform(test_df)
    rf_metrics     = evaluate_model(rf_predictions, "Random Forest")

    rf_model.write().overwrite().save("models/fraud_rf_pipeline")
    print("\nRandom Forest model saved to: models/fraud_rf_pipeline")
    logger.info(f"RF — AUC: {rf_metrics['auc']}, F1: {rf_metrics['f1']}")

    # Compare Both Models
    print("\n===== Model Comparison =====")
    print(f"{'Metric':<12} {'Logistic Regression':<22} {'Random Forest':<22}")
    print("-" * 56)
    print(f"{'AUC-ROC':<12} {lr_metrics['auc']:<22} {rf_metrics['auc']:<22}")
    print(f"{'Accuracy':<12} {lr_metrics['accuracy']:<22} {rf_metrics['accuracy']:<22}")
    print(f"{'Precision':<12} {lr_metrics['precision']:<22} {rf_metrics['precision']:<22}")
    print(f"{'Recall':<12} {lr_metrics['recall']:<22} {rf_metrics['recall']:<22}")
    print(f"{'F1 Score':<12} {lr_metrics['f1']:<22} {rf_metrics['f1']:<22}")

    best = "Random Forest" if rf_metrics['f1'] > lr_metrics['f1'] else "Logistic Regression"
    print(f"\nBest Model: {best}")
    logger.info(f"Best model: {best}")

    return lr_model, rf_model


# -----------------------------
# Streaming Real-Time Predictions
# -----------------------------
def run_streaming_predictions():

    logger = get_logger("MLPipeline-Streaming")
    spark  = get_spark_session("Fraud-StreamingPredictions")
    logger.info("Starting Real-Time Streaming Predictions")

    # Schema matches stream_input/ columns
    # includes engineered features saved by streaming.py
    schema = StructType([
        StructField("trans_date_trans_time", TimestampType(), True),
        StructField("cc_num",                LongType(),      True),
        StructField("merchant",              StringType(),    True),
        StructField("category",              StringType(),    True),
        StructField("amt",                   DoubleType(),    True),
        StructField("first",                 StringType(),    True),
        StructField("last",                  StringType(),    True),
        StructField("gender",                StringType(),    True),
        StructField("street",                StringType(),    True),
        StructField("city",                  StringType(),    True),
        StructField("state",                 StringType(),    True),
        StructField("zip",                   IntegerType(),   True),
        StructField("lat",                   DoubleType(),    True),
        StructField("long",                  DoubleType(),    True),
        StructField("city_pop",              IntegerType(),   True),
        StructField("job",                   StringType(),    True),
        StructField("dob",                   DateType(),      True),
        StructField("trans_num",             StringType(),    True),
        StructField("unix_time",             IntegerType(),   True),
        StructField("merch_lat",             DoubleType(),    True),
        StructField("merch_long",            DoubleType(),    True),
        StructField("is_fraud",              DoubleType(),    True),
        StructField("is_weekend",            IntegerType(),   True),
        StructField("distance_km",           DoubleType(),    True),
        StructField("avg_amt_recent",        DoubleType(),    True)
    ])

    # Load saved Random Forest model
    print("\nLoading saved Random Forest model...")
    rf_model = PipelineModel.load("models/fraud_rf_pipeline")
    logger.info("RF model loaded successfully")

    # Read stream_input/ as readStream
    print("Reading streaming data from data/stream_input/...")
    streaming_df = spark.readStream \
        .option("header", True) \
        .schema(schema) \
        .csv("data/stream_input/")

    print(f"Is Streaming: {streaming_df.isStreaming}")

    # Directly apply RF model — no feature engineering needed here!
    stream_predictions = rf_model.transform(streaming_df)

    # Query 1: All predictions
    query1 = stream_predictions \
        .select("trans_num", "amt", "category",
                "distance_km", "is_fraud", "prediction") \
        .writeStream \
        .outputMode("append") \
        .format("console") \
        .option("numRows", 20) \
        .option("truncate", False) \
        .start()

    print("\nStreaming Query 1: All Predictions running...")
    query1.awaitTermination(timeout=15)
    print("Query 1 complete!")

    # Query 2: Fraud alerts only
    fraud_alerts = stream_predictions \
        .filter(spark_col("prediction") == 1.0) \
        .select("trans_num", "amt", "category",
                "merchant", "distance_km", "is_fraud", "prediction")

    query2 = fraud_alerts \
        .writeStream \
        .outputMode("append") \
        .format("console") \
        .option("numRows", 20) \
        .option("truncate", False) \
        .start()

    print("\nStreaming Query 2: Fraud Alerts Only running...")
    query2.awaitTermination(timeout=15)
    print("Query 2 complete!")

    logger.info("Streaming predictions completed successfully")


# -----------------------------
# Load saved model
# -----------------------------
def load_model(model_path):
    logger = get_logger("MLPipeline")
    saved_model = PipelineModel.load(model_path)
    logger.info(f"Model loaded from {model_path}")
    return saved_model


# -----------------------------
# Main execution
# -----------------------------
if __name__ == "__main__":

    # Step 1: Train + Evaluate on static data
    lr_model, rf_model = run_ml_pipeline()
    print("\nML Pipeline completed successfully!")

    # Step 2: Real-time predictions on streaming data
    print("\n" + "=" * 56)
    print("Starting Real-Time Streaming Predictions...")
    print("=" * 56)
    run_streaming_predictions()
    print("\nStreaming Predictions completed!")