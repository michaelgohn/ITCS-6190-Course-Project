# ML pipeline placeholder
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.sql import DataFrame
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator

from ingestion import load_and_process_data, split_train_test, create_fraud_weight

feature_names = [
    "trans_year", "trans_month", "trans_day", "sin_hour", "cos_hour", "sin_minute", "cos_minute", \
    "amt", "lat", "long", "merch_lat", "merch_long", "is_weekend", "distance_km", "avg_amt_recent"
]

def transform_data(df: DataFrame) -> tuple:

    vectAssembler = VectorAssembler(
        inputCols=feature_names,
        outputCol="features"
    )

    vectDf = vectAssembler.transform(df)
    # vectDf.select("features", "is_fraud").show(10, truncate=False)

    scaler = StandardScaler(
        inputCol="features",
        outputCol="scaled_features",
        withMean=True,
        withStd=True
    )

    scalerModel = scaler.fit(vectDf)
    scaledDf = scalerModel.transform(vectDf)
    return scaledDf, vectAssembler

def run_estimation(df: DataFrame) -> LogisticRegression:

    lr = LogisticRegression(featuresCol="scaled_features", labelCol="is_fraud")
    lr_model = lr.fit(df)
    # rf = RandomForestClassifier(featuresCol="scaled_features", labelCol="is_fraud")
    # rf_model = rf.fit(df)
    # lr = LogisticRegression(featuresCol="scaled_features", labelCol="is_fraud")
    # lrModel = lr.fit(df)

    # Analysis
    b = round(lr_model.intercept, 2)

    for name, coef in zip(feature_names, lr_model.coefficients):
        print(f"{name}: {coef}")

    print(f"b score: {b}")

    # Maybe change return to lr_model?
    return lr

def create_and_run_pipeline(train_df: DataFrame, test_df: DataFrame) -> tuple:

    vectAssembler = VectorAssembler(
        inputCols=feature_names,
        outputCol="features"
    )

    scaler = StandardScaler(
        inputCol="features",
        outputCol="scaled_features",
        withMean=True,
        withStd=True
    )

    rf = RandomForestClassifier(featuresCol="scaled_features", labelCol="is_fraud", weightCol="fraud_weight")
    # lr = LogisticRegression(featuresCol="scaled_features", labelCol="is_fraud", weightCol="fraud_weight")
    # rf = RandomForestClassifier(featuresCol="scaled_features", labelCol="is_fraud")
    # lr = LogisticRegression(featuresCol="scaled_features", labelCol="is_fraud")
    
    pipeline = Pipeline(stages=[vectAssembler, scaler, rf])
    pipeline_model = pipeline.fit(train_df)

    pred_df = pipeline_model.transform(test_df)

    return pipeline_model, pred_df

if __name__ == "__main__":

    df = load_and_process_data()

    train_df, test_df = split_train_test(df)

    print("\nTrain Count:", train_df.count())
    print("Test Count:", test_df.count())

    # train_df, vect_assembler = transform_data(train_df)
    # lr = run_estimation(train_df)

    train_df = create_fraud_weight(train_df)

    pipeline_model, pred_df = create_and_run_pipeline(train_df, test_df)

    # pred_df.select("cos_hour", "sin_hour", "amt", "scaled_features", "is_fraud", "prediction").show(10)

    # pred_df.groupBy("is_fraud", "prediction").count().show()

    evaluator = BinaryClassificationEvaluator(
        labelCol="is_fraud",
        metricName="areaUnderPR"
    )

    auc = evaluator.evaluate(pred_df)
    print("AUC:", auc)

    pred_df.groupBy("is_fraud", "prediction").count().show()
