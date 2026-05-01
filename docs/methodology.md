# Methodology
1. Overview

This project follows an end-to-end big data analytics methodology for detecting fraudulent credit card transactions using Apache Spark. The methodology is designed to cover the complete data processing lifecycle, starting from raw data ingestion and ending with machine learning-based fraud prediction and dashboard visualization.
The project pipeline includes the following major stages:
Data ingestion
Data cleaning and preprocessing
Feature engineering
Exploratory data analysis
Complex Spark SQL transformations
Structured Streaming simulation
Machine learning pipeline development
Model evaluation and comparison
Real-time fraud prediction
Streamlit dashboard visualization
The overall goal of this methodology is to build a scalable fraud detection system that can process large transaction datasets, identify fraud-related patterns, and classify transactions as legitimate or fraudulent.
---
2. Dataset Ingestion
The project uses the Credit Card Fraud Detection dataset from Kaggle. The dataset contains transaction-level information such as transaction timestamp, card number, merchant, category, amount, customer location, merchant location, and fraud label.
Two CSV files are used:
```text
CreditCard_Fraud_Train.csv
CreditCard_Fraud_Test.csv
```
The ingestion process is implemented in `ingestion.py`. Both training and testing datasets are loaded using PySpark with header detection and schema inference.
```python
train_df = spark.read.csv(train_path, header=True, inferSchema=True).drop("_c0")
test_df = spark.read.csv(test_path, header=True, inferSchema=True).drop("_c0")
```
The `_c0` column is dropped because it represents an unnecessary index column from the original CSV file. Since this column does not provide meaningful fraud detection information, it is removed before further processing.
Before combining the datasets, a helper column named `dataset_type` is added:
```python
train_df = train_df.withColumn("dataset_type", col("is_fraud") * 0 + 1)
test_df = test_df.withColumn("dataset_type", col("is_fraud") * 0 + 2)
```
This column helps identify whether each row originally came from the training file or the testing file. After that, both datasets are combined using `unionByName()`.
```python
df = train_df.unionByName(test_df)
```
This approach allows the project to perform common preprocessing and feature engineering on the complete dataset while still being able to separate the data back into training and testing sets later.
---
3. Spark Session and Logging Setup
The project uses a reusable Spark session utility defined in `utils.py`. The Spark session is configured with driver memory, executor memory, memory fraction, and off-heap memory settings.
```python
SparkSession.builder \
    .appName(app_name) \
    .config("spark.driver.memory", "4g") \
    .config("spark.executor.memory", "4g") \
    .config("spark.memory.fraction", "0.8") \
    .config("spark.memory.offHeap.enabled", "true") \
    .config("spark.memory.offHeap.size", "2g") \
    .getOrCreate()
```
This setup improves consistency across different project modules such as ingestion, transformations, streaming, and machine learning.
A reusable logger is also created to display clean execution messages. This helps track the progress of each pipeline stage and makes debugging easier.
---
4. Data Cleaning and Preprocessing
After loading the data, several cleaning steps are applied to improve data quality.
4.1 Duplicate Removal
Duplicate rows are removed using:
```python
df = df.dropDuplicates()
```
This ensures that repeated transactions do not bias the analysis or machine learning model.
4.2 Timestamp Conversion
The transaction timestamp column is converted into a proper timestamp type:
```python
df = df.withColumn(
    "trans_date_trans_time",
    to_timestamp(col("trans_date_trans_time"))
)
```
This conversion is necessary because time-based features and fraud trend analysis depend on accurate timestamp processing.
4.3 Empty String Handling
Empty string values are replaced with null values:
```python
df = df.replace("", None)
```
This allows Spark to properly recognize missing values and handle them consistently.
4.4 String Column Trimming
Extra spaces are removed from important string columns:
```python
string_columns = ["category", "merchant", "city", "state"]
for column in string_columns:
    df = df.withColumn(column, trim(col(column)))
```
This prevents duplicate category or merchant values caused only by unwanted whitespace.
---
5. Missing Value Handling
The project handles missing values separately for numeric and categorical columns.
5.1 Numeric Missing Values
For numeric columns, the mean value of each column is calculated and used to fill missing values.
Numeric columns handled include:
```text
amt, lat, long, merch_lat, merch_long, city_pop
```
Example:
```python
amt_mean = df.select(mean(col("amt"))).first()[0]
```
Then the missing values are filled:
```python
df = df.fillna({
    "amt": amt_mean,
    "lat": lat_mean,
    "long": long_mean,
    "merch_lat": merch_lat_mean,
    "merch_long": merch_long_mean,
    "city_pop": city_pop_mean
})
```
This prevents rows from being lost due to missing numeric data.
5.2 Categorical Missing Values
For categorical columns, missing values are filled with the value `Unknown`.
```python
df = df.fillna({
    "category": "Unknown",
    "merchant": "Unknown",
    "city": "Unknown",
    "state": "Unknown",
    "gender": "Unknown"
})
```
This allows the model and analysis queries to retain rows even when some categorical values are missing.
---
6. Feature Engineering
Feature engineering is a major part of this project. Several new features are created to improve fraud pattern detection.
6.1 Weekend Indicator: `is_weekend`
A new feature named `is_weekend` is created using the transaction date.
```python
df = df.withColumn(
    "is_weekend",
    when(dayofweek(col("trans_date_trans_time")).isin([1, 7]), 1).otherwise(0)
)
```
This feature identifies whether a transaction occurred on a weekend.
```text
1 = Weekend transaction
0 = Weekday transaction
```
This is useful because fraud behavior may vary depending on the day of the week.
---
6.2 Geographic Distance Feature: `distance_km`
The project calculates the distance between the customer location and merchant location using the Haversine formula.
Customer location columns:
```text
lat, long
```
Merchant location columns:
```text
merch_lat, merch_long
```
The latitude and longitude values are first converted into radians:
```python
df = df.withColumn("lat1", radians(col("lat"))) \
       .withColumn("lon1", radians(col("long"))) \
       .withColumn("lat2", radians(col("merch_lat"))) \
       .withColumn("lon2", radians(col("merch_long")))
```
Then the Haversine calculation is applied:
```python
df = df.withColumn("a",
    sin(col("dlat")/2)**2 +
    cos(col("lat1")) * cos(col("lat2")) * sin(col("dlon")/2)**2
)

df = df.withColumn("c", 2 * atan2(sqrt(col("a")), sqrt(1-col("a"))))
df = df.withColumn("distance_km", col("c") * 6371)
```
The value `6371` represents the approximate radius of the Earth in kilometers.
This feature is useful because transactions occurring far away from the customer’s usual location may indicate suspicious behavior.
After calculating `distance_km`, temporary intermediate columns are dropped.
---
6.3 Recent Average Transaction Amount: `avg_amt_recent`
A window function is used to calculate the recent average transaction amount for each credit card.
```python
window_spec = Window.partitionBy("cc_num").orderBy("trans_date_trans_time")

df = df.withColumn(
    "avg_amt_recent",
    avg("amt").over(window_spec.rowsBetween(-10, 0))
)
```
This calculates the average transaction amount over the current transaction and the previous 10 transactions for the same credit card number.
This feature helps identify sudden spending spikes. For example, if a customer usually spends small amounts but suddenly makes a very large transaction, that transaction may be suspicious.
---
6.4 Amount Bucket: `amt_bucket`
The transaction amount is grouped into ranges using conditional logic.
```python
df = df.withColumn("amt_bucket",
    when(col("amt") < 500, "0-500")
    .when(col("amt") < 1000, "500-1000")
    .when(col("amt") < 5000, "1000-5000")
    .when(col("amt") < 10000, "5000-10000")
    .otherwise("10000+")
)
```
This feature is useful for analyzing fraud patterns across different transaction amount ranges.
---
7. Exploratory Data Analysis
After cleaning and feature engineering, basic exploratory data analysis is performed in `ingestion.py`.
The EDA section displays:
Sample data
Dataset schema
Fraud distribution
Amount statistics
Category distribution
Weekend vs fraud analysis
Distance vs fraud analysis
High-value transaction analysis
Examples:
```python
df.groupBy("is_fraud").count().show()
df.select("amt").describe().show()
df.groupBy("category").count().orderBy("count", ascending=False).show(10)
df.groupBy("is_weekend").agg({"is_fraud": "avg"}).show()
```
This step helps understand the dataset before performing complex transformations and machine learning.
---
8. Train-Test Separation
After the combined dataset is processed, it is split back into training and testing datasets using the `dataset_type` column.
```python
train_df = df.filter(col("dataset_type") == 1)
test_df = df.filter(col("dataset_type") == 2)
```
Then the helper column is removed:
```python
train_df = train_df.drop("dataset_type")
test_df = test_df.drop("dataset_type")
```
This ensures that model training and evaluation are performed on the correct original train and test datasets.
---
9. Complex Spark SQL Transformations
The file `transformations.py` performs six major analytical queries using Spark DataFrame operations. These queries are designed to identify fraud patterns from different perspectives.
---
9.1 Fraud Rate by Transaction Category
The first query calculates fraud rate for each transaction category.
```python
category_fraud = df.groupBy("category").agg(
    count("*").alias("total_txn"),
    sum("is_fraud").alias("fraud_txn")
).withColumn(
    "fraud_rate", col("fraud_txn") / col("total_txn")
)
```
This helps identify which transaction categories have higher fraud risk.
---
9.2 Fraud Pattern by Hour of Day
The second query extracts the transaction hour and calculates fraud rate by hour.
```python
df_with_hour = df.withColumn("txn_hour", hour(col("trans_date_trans_time")))
```
Then the data is grouped by `txn_hour`.
This helps identify whether fraud is more common during certain hours, such as late night or early morning.
---
9.3 High-Risk Customers Based on Fraud Rate
The third query groups transactions by credit card number.
```python
customer_risk = df.groupBy("cc_num").agg(
    count("*").alias("total_txn"),
    sum("is_fraud").alias("fraud_txn"),
    avg("amt").alias("avg_amt")
)
```
A fraud rate is calculated for each customer, and only customers with more than 50 transactions are considered.
This helps identify accounts with repeated suspicious activity.
---
9.4 Distance-Based Fraud Analysis
The fourth query groups transactions into distance buckets:
```text
Near: less than 50 km
Medium: 50 km to 200 km
Far: greater than or equal to 200 km
```
```python
distance_bucket = df.withColumn(
    "distance_bucket",
    when(col("distance_km") < 50, "Near")
    .when(col("distance_km") < 200, "Medium")
    .otherwise("Far")
)
```
This helps analyze whether transactions farther from the customer location have higher fraud rates.
---
9.5 High-Value Transaction Fraud Impact
The fifth query creates a flag for high-value transactions.
```python
high_value = df.withColumn(
    "high_value_flag",
    when(col("amt") > 500, 1).otherwise(0)
)
```
Transactions above `$500` are treated as high-value transactions.
This query compares fraud rates between high-value and normal transactions.
---
9.6 Top Fraud-Prone Categories Ranked Within Each State
The sixth query uses a Spark window function to rank fraud-prone categories within each state.
```python
window_spec = Window.partitionBy("state").orderBy(col("fraud_rate").desc())
```
After calculating fraud rates by state and category, the categories are ranked within each state.
```python
ranked = state_category.withColumn(
    "rank", rank().over(window_spec)
).filter(col("rank") <= 3)
```
This helps identify the top three risky transaction categories for each state.
---
10. Structured Streaming Methodology
The streaming component is implemented in `streaming.py`. Since the project does not use a real-time message broker like Kafka, file-based Spark Structured Streaming is used to simulate real-time transaction processing.
---
10.1 Streaming Data Preparation
The test dataset is loaded first:
```python
test_df = spark.read.csv(
    "data/CreditCard_Fraud_Test.csv",
    header=True,
    inferSchema=True
)
```
Then the dataset is divided into two parts:
```python
batch_df = test_df.limit(total_count // 2)
stream_df = test_df.subtract(batch_df)
```
The first half is treated as batch data, and the second half is used for streaming simulation.
---
10.2 Streaming Folder Creation
A folder named `data/stream_input` is created to store streaming input files.
```python
stream_path = "data/stream_input"
```
If the folder already exists, it is removed and recreated. This ensures a clean streaming environment each time the pipeline runs.
---
10.3 Feature Engineering on Streaming Data
The streaming data is enriched with the same important features used in batch processing:
`is_weekend`
`distance_km`
`avg_amt_recent`
This ensures consistency between batch training data and streaming prediction data.
---
10.4 Micro-Batch Simulation
The streaming data is repartitioned into five files:
```python
stream_df = stream_df.repartition(5)
```
Then it is written into the streaming input folder:
```python
stream_df.write \
    .option("header", True) \
    .mode("overwrite") \
    .csv(stream_path)
```
This simulates multiple micro-batches arriving into the streaming folder.
---
10.5 Reading Data as a Stream
The schema of the prepared streaming DataFrame is reused:
```python
schema = stream_df.schema
```
Then Spark reads the folder as a streaming source:
```python
streaming_df = spark.readStream \
    .option("header", True) \
    .schema(schema) \
    .csv(stream_path)
```
This creates a streaming DataFrame.
---
10.6 Streaming Queries
Three streaming queries are implemented.
Query 1: Fraud by Category
```python
fraud_by_category = enriched_df.groupBy("category").agg(
    sum("is_fraud").alias("fraud_txn"),
    count("*").alias("total_txn")
)
```
Query 2: Fraud by Hour
```python
fraud_by_hour = enriched_df.groupBy("txn_hour").agg(
    sum("is_fraud").alias("fraud_txn"),
    count("*").alias("total_txn")
)
```
Query 3: High-Value Fraud Detection
```python
high_value_fraud = enriched_df.groupBy("high_value_flag").agg(
    sum("is_fraud").alias("fraud_txn"),
    count("*").alias("total_txn")
)
```
All three queries use console output in complete mode:
```python
.writeStream \
.outputMode("complete") \
.format("console")
```
This allows the project to display updated fraud summaries as streaming data is processed.
---
11. Machine Learning Pipeline Methodology
The machine learning pipeline is implemented in `ml_pipeline.py`. The goal is to train fraud classification models and compare their performance.
---
11.1 Target Variable
The target column is:
```text
is_fraud
```
The values represent:
```text
0 = Legitimate transaction
1 = Fraudulent transaction
```
The target column is cast to double before training:
```python
train_df = train_df.withColumn("is_fraud", spark_col("is_fraud").cast("double"))
test_df = test_df.withColumn("is_fraud", spark_col("is_fraud").cast("double"))
```
---
11.2 Feature Columns
The model uses seven features:
```text
amt, city_pop, is_weekend, distance_km,
avg_amt_recent, category_index, gender_index
```
These features include transaction amount, customer city population, weekend indicator, distance between customer and merchant, recent spending behavior, transaction category, and gender.
---
11.3 Categorical Encoding
The categorical columns `category` and `gender` are converted into numeric form using `StringIndexer`.
```python
category_indexer = StringIndexer(
    inputCol="category",
    outputCol="category_index",
    handleInvalid="keep"
)
```
```python
gender_indexer = StringIndexer(
    inputCol="gender",
    outputCol="gender_index",
    handleInvalid="keep"
)
```
The `handleInvalid="keep"` option ensures that unseen or invalid category values do not break the pipeline.
---
11.4 Feature Vector Creation
The selected features are combined into a single feature vector using `VectorAssembler`.
```python
assembler = VectorAssembler(
    inputCols=feature_cols,
    outputCol="features",
    handleInvalid="keep"
)
```
This step is required because Spark ML models expect all input features to be present in one vector column.
---
11.5 Feature Scaling
The project applies `StandardScaler` to scale the assembled features.
```python
scaler = StandardScaler(
    inputCol="features",
    outputCol="scaled_features",
    withStd=True,
    withMean=False
)
```
Scaling helps normalize feature ranges and improves the stability of models such as Logistic Regression.
---
12. Model Training
Two machine learning models are trained and compared.
---
12.1 Logistic Regression Baseline Model
Logistic Regression is used as the baseline model.
```python
lr = LogisticRegression(
    featuresCol="scaled_features",
    labelCol="is_fraud",
    maxIter=10,
    regParam=0.01
)
```
The Logistic Regression pipeline includes:
Category indexing
Gender indexing
Feature assembling
Feature scaling
Logistic Regression model training
```python
lr_pipeline = Pipeline(stages=[
    category_indexer,
    gender_indexer,
    assembler,
    scaler,
    lr
])
```
This model provides a simple baseline for comparison.
---
12.2 Random Forest Improved Model
Random Forest is used as the improved model.
Before training Random Forest, class weights are added to handle class imbalance.
```python
train_df = train_df.withColumn(
    "weight",
    when(spark_col("is_fraud") == 1.0, 200.0)
    .otherwise(1.0)
)
```
Fraud cases receive a higher weight because fraud transactions are much less frequent than legitimate transactions.
The Random Forest model is defined as:
```python
rf = RandomForestClassifier(
    featuresCol="scaled_features",
    labelCol="is_fraud",
    weightCol="weight",
    numTrees=100,
    maxDepth=10,
    seed=42
)
```
The Random Forest pipeline includes:
Category indexing
Gender indexing
Feature assembling
Feature scaling
Random Forest model training
Random Forest is selected as the improved model because it can capture non-linear relationships and interactions between features better than Logistic Regression.
---
13. Model Evaluation
The project evaluates each model using multiple metrics.
The evaluation function calculates:
```text
AUC-ROC
Accuracy
Precision
Recall
F1 Score
Confusion Matrix
```
13.1 AUC-ROC
AUC-ROC measures how well the model separates fraud and legitimate transactions across different thresholds.
```python
BinaryClassificationEvaluator(
    labelCol="is_fraud",
    rawPredictionCol="rawPrediction",
    metricName="areaUnderROC"
)
```
13.2 Accuracy
Accuracy measures the overall percentage of correct predictions. However, because fraud detection is highly imbalanced, accuracy alone is not enough.
13.3 Precision
Precision measures how many predicted fraud transactions were actually fraud.
13.4 Recall
Recall measures how many actual fraud transactions were successfully detected.
Recall is especially important in fraud detection because missing fraudulent transactions can cause financial loss.
13.5 F1 Score
F1 score balances precision and recall.
13.6 Confusion Matrix
The confusion matrix shows:
True legitimate transactions
False fraud alerts
Missed fraud transactions
Correctly detected fraud transactions
```python
predictions.groupBy("is_fraud", "prediction") \
    .count() \
    .orderBy("is_fraud", "prediction") \
    .show()
```
This helps understand the real fraud detection capability of each model.
---
14. Model Saving
After training, both models are saved for reuse.
```python
lr_model.write().overwrite().save("models/fraud_lr_pipeline")
rf_model.write().overwrite().save("models/fraud_rf_pipeline")
```
Saving the models allows the dashboard and streaming prediction pipeline to load the trained model without retraining every time.
---
15. Real-Time Streaming Prediction Methodology
After training and saving the Random Forest model, the project performs real-time prediction on streaming data.
The saved Random Forest model is loaded using:
```python
rf_model = PipelineModel.load("models/fraud_rf_pipeline")
```
The streaming data is read from:
```text
data/stream_input/
```
Then the trained Random Forest model is directly applied to the streaming DataFrame:
```python
stream_predictions = rf_model.transform(streaming_df)
```
Two streaming prediction outputs are created.
15.1 All Predictions
The first streaming output displays all predictions:
```python
stream_predictions.select(
    "trans_num", "amt", "category", "distance_km", "is_fraud", "prediction"
)
```
15.2 Fraud Alerts Only
The second streaming output filters only predicted fraud transactions:
```python
fraud_alerts = stream_predictions \
    .filter(spark_col("prediction") == 1.0)
```
This simulates a real fraud alert system where only suspicious transactions are highlighted.
---
16. Dashboard Methodology
The dashboard is implemented using Streamlit in `app.py`. It provides a user-friendly interface for viewing project results, fraud patterns, and live predictions.
The dashboard contains five main pages:
```text
Data Overview
Fraud Analysis
Distance Analysis
Live Prediction
Model Comparison
```
---
16.1 Data Overview Page
This page displays:
Total number of transactions
Fraud transactions
Legitimate transactions
Fraud rate
Sample transaction data
Features used in the machine learning model
This gives users a quick summary of the dataset and model input features.
---
16.2 Fraud Analysis Page
This page visualizes fraud patterns such as:
Fraud vs legitimate distribution
Fraud rate by category
Fraud rate by amount bucket
Fraud pattern by hour of day
Distance-based fraud analysis
High-value transaction fraud impact
Gender-based fraud analysis
Weekend vs weekday fraud comparison
These visualizations connect the Spark analysis results to business insights.
---
16.3 Distance Analysis Page
This page explains the importance of geographic distance in fraud detection.
It compares customer location and merchant location and explains how unusual transaction locations can be suspicious. The dashboard also uses a Gmail login alert analogy to make the concept easier to understand.
---
16.4 Live Prediction Page
This page allows the user to manually enter transaction details such as:
Transaction amount
Category
Gender
Weekend indicator
Distance from home to merchant
City population
Average recent transaction amount
The saved Random Forest model is loaded and used to predict whether the entered transaction is legitimate or fraudulent.
---
16.5 Model Comparison Page
This page compares Logistic Regression and Random Forest using performance metrics such as:
AUC-ROC
Accuracy
Precision
Recall
F1 Score
It also compares fraud detection capability using:
Fraud caught
Fraud missed
False alarms
This page clearly shows why Random Forest with class weights is the stronger model for this project.
---
17. Project Execution Methodology
The project can be executed using the `run.sh` file.
```bash
bash run.sh
```
This command runs the full pipeline in the following order:
```text
1. Data ingestion and preprocessing
2. Spark SQL transformations
3. Structured Streaming simulation
4. Machine learning training and prediction
5. Streamlit dashboard launch
```
The project also includes a `Makefile`, which allows each component to be executed separately.
```bash
make ingestion
make transformations
make streaming
make pipeline
make dashboard
```
This modular execution approach makes the project easier to test, debug, and explain.
---
18. Methodology Summary
This methodology demonstrates a complete fraud detection workflow using Apache Spark. The project begins by ingesting raw CSV data and applying cleaning, preprocessing, and feature engineering. It then performs fraud pattern analysis using Spark transformations and window functions. A simulated streaming pipeline is created using Spark Structured Streaming to demonstrate real-time fraud analysis.
For machine learning, the project builds a Spark MLlib pipeline using categorical indexing, feature assembling, scaling, and classification models. Logistic Regression is used as the baseline model, while Random Forest with class weights is used as the improved model to address the highly imbalanced fraud dataset. The trained model is then applied to streaming data for real-time fraud prediction.
Finally, the Streamlit dashboard presents the results through interactive pages, visualizations, model comparison, and live transaction prediction. Overall, the methodology provides a scalable and structured approach for detecting credit card fraud using big data tools and machine learning.
