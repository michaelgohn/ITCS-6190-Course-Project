# Reproduction Guide

## Project: Credit Card Fraud Detection Using Apache Spark

This guide explains how to reproduce the complete credit card fraud detection project from setup to execution. It includes the required software, dataset preparation, dependency installation, project structure, execution commands, expected outputs, and troubleshooting notes.

The project implements an end-to-end fraud detection pipeline using:

- Apache Spark Structured APIs
- Spark SQL transformations
- Spark Structured Streaming
- Spark MLlib
- Streamlit dashboard

---

## 1. System Requirements

Before running the project, make sure the following tools are installed.

### Required Software

| Software | Purpose |
|---|---|
| Python 3.9 or higher | Runs the PySpark project and dashboard |
| Java 17 | Required by Apache Spark |
| Apache Spark / PySpark | Distributed data processing and ML pipeline |
| Streamlit | Interactive dashboard |
| Git / Terminal | Running commands and managing files |

---

## 2. Python Dependencies

The project dependencies are listed in `requirements.txt`.

```text
pyspark
pandas
matplotlib
pytest
streamlit
python-dotenv
```

Install the dependencies using one of the following methods.

### Install manually

```bash
pip install -r requirements.txt
```

If a virtual environment is preferred, use:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

For Windows PowerShell:

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

---

## 3. Java Setup

Apache Spark requires Java. This project was developed using Java 17.

### macOS Setup

Install Java 17 using Homebrew:

```bash
brew install openjdk@17
```

Set `JAVA_HOME`:

```bash
export JAVA_HOME=/opt/homebrew/opt/openjdk@17
```

Verify Java installation:

```bash
java -version
```

Expected output should show Java 17.

### Windows Setup

1. Download Java 17 from Adoptium or Oracle.
2. Install Java 17.
3. Set the `JAVA_HOME` environment variable to the Java installation folder.
4. Download winutils.exe and hadoop.dll for Hadoop (required on Windows)
5. Place winutils.exe and hadoop.dll in: C:\hadoop\bin\
6. Set HADOOP_HOME in Environment Variables
7. Add Java `bin` folder to the system `PATH`.
8. Verify installation:

```powershell
java -version
```

### Create .env File

Run the command  - ```cp .env-example .env```

change JAVA_HOME to Java installation folder

change HADOOP_HOME to hadoop folder if necessary

---

## 4. Dataset Preparation

The dataset used in this project is the Kaggle credit card fraud detection dataset.

### Dataset Source

```text
https://www.kaggle.com/datasets/kartik2112/fraud-detection
```

Download the dataset files and rename them as follows:

| Original File | Required File Name |
|---|---|
| `fraudTrain.csv` | `CreditCard_Fraud_Train.csv` |
| `fraudTest.csv` | `CreditCard_Fraud_Test.csv` |

Place both files inside the `data/` folder.

Expected structure:

```text
data/
├── CreditCard_Fraud_Train.csv
└── CreditCard_Fraud_Test.csv
```

These files are required because the ingestion, streaming, and machine learning scripts read from these exact paths:

```text
data/CreditCard_Fraud_Train.csv
data/CreditCard_Fraud_Test.csv
```

---

## 5. Expected Project Structure

The project should be organized as follows:

```text
spark_project_template/
├── app.py
├── Makefile
├── README.md
├── requirements.txt
├── run.sh
├── data/
│   ├── CreditCard_Fraud_Train.csv
│   ├── CreditCard_Fraud_Test.csv
│   └── stream_input/
├── models/
│   ├── fraud_lr_pipeline/
│   └── fraud_rf_pipeline/
├── src/
│   ├── ingestion.py
│   ├── transformations.py
│   ├── streaming.py
│   ├── ml_pipeline.py
│   └── utils.py
└── docs/
    ├── dataset_overview.md
    ├── methodology.md
    ├── results.md
    ├── limitations.md
    └── reproduction_guide.md
```

The `data/stream_input/` folder and `models/` folders may not exist before execution. They are generated during the pipeline run.

---

## 6. Verify Setup Before Running

Before running the full pipeline, verify that Python, Java, and PySpark are installed correctly.

### Check Python

```bash
python3 --version
```

### Check Java

```bash
java -version
```

### Check PySpark

```bash
python3 -c "import pyspark; print('PySpark version:', pyspark.__version__)"
```

If all commands run successfully, the environment is ready.

---

## 7. Running the Full Pipeline

The complete project can be executed with one command:

```bash
bash run.sh
```

The `run.sh` file executes the following scripts in order:

```bash
export PYTHONPATH=src

python3 src/ingestion.py
python3 src/transformations.py
python3 src/streaming.py
python3 src/ml_pipeline.py
streamlit run app.py
```

This runs the entire workflow:

1. Data ingestion and cleaning
2. Feature engineering
3. Spark SQL transformations
4. Structured Streaming simulation
5. ML model training and evaluation
6. Real-time streaming predictions
7. Streamlit dashboard launch

---

## 8. Running Individual Components

Each component can also be executed separately using the Makefile.

### 8.1 Install Dependencies

```bash
make install
```

### 8.2 Run Data Ingestion

```bash
make ingestion
```

Equivalent command:

```bash
PYTHONPATH=src python3 src/ingestion.py
```

This step performs:

- Loading train and test CSV files
- Removing unwanted index column `_c0`
- Combining train and test data temporarily
- Removing duplicate records
- Converting transaction timestamp into timestamp format
- Handling empty strings and missing values
- Trimming text columns
- Creating engineered features
- Displaying EDA outputs

Expected outputs include:

- Sample data
- Schema
- Fraud distribution
- Amount statistics
- Category distribution
- Weekend vs fraud summary
- Distance vs fraud summary
- High-value transaction summary
- Train and test row counts

---

### 8.3 Run Spark SQL Transformations

```bash
make transformations
```

Equivalent command:

```bash
PYTHONPATH=src python3 src/transformations.py
```

This step runs six major analytical queries:

1. Fraud rate by transaction category
2. Fraud pattern by hour of day
3. Top high-risk customers based on fraud rate
4. Distance-based fraud analysis
5. High-value transaction fraud impact
6. Top fraud-prone categories ranked within each state

Expected output is printed in the terminal using Spark DataFrame `.show()`.

---

### 8.4 Run Spark Structured Streaming

```bash
make streaming
```

Equivalent command:

```bash
PYTHONPATH=src timeout 60 python3 src/streaming.py
```

This step performs:

- Loading the test dataset
- Splitting test data into batch and streaming portions
- Creating the `data/stream_input/` folder
- Repartitioning streaming data into 5 files to simulate micro-batches
- Applying streaming feature engineering
- Reading the folder using `readStream`
- Running three streaming aggregation queries

Streaming queries include:

1. Fraud by category
2. Fraud trend by hour
3. High-value fraud detection

Expected output appears in the terminal as Spark streaming console output.

---

### 8.5 Run Machine Learning Pipeline

```bash
make pipeline
```

Equivalent command:

```bash
PYTHONPATH=src python3 src/ml_pipeline.py
```

This step performs:

- Loading processed data
- Splitting train and test data
- Indexing categorical columns
- Assembling features
- Scaling features
- Training Logistic Regression
- Training Random Forest with class weights
- Evaluating both models
- Saving trained pipelines
- Running streaming predictions using the saved Random Forest model

The machine learning pipeline uses the following features:

```text
amt
city_pop
is_weekend
distance_km
avg_amt_recent
category_index
gender_index
```

Expected model output includes:

- AUC-ROC
- Accuracy
- Precision
- Recall
- F1 Score
- Sample predictions
- Confusion matrix
- Model comparison table

Expected saved model folders:

```text
models/fraud_lr_pipeline/
models/fraud_rf_pipeline/
```

---

### 8.6 Run Streamlit Dashboard

```bash
make dashboard
```

Equivalent command:

```bash
streamlit run app.py
```

The dashboard opens in a browser, usually at:

```text
http://localhost:8501
```

The dashboard contains five pages:

1. Data Overview
2. Fraud Analysis
3. Distance Analysis
4. Live Prediction
5. Model Comparison

The dashboard requires the saved Random Forest model at:

```text
models/fraud_rf_pipeline/
```

If the model does not exist, run the ML pipeline first:

```bash
make pipeline
```

---

## 9. Data Processing Details

### 9.1 Data Loading

The train and test CSV files are loaded using Spark:

```python
spark.read.csv(path, header=True, inferSchema=True)
```

The unwanted index column `_c0` is dropped if present.

---

### 9.2 Data Cleaning

The ingestion script performs the following cleaning steps:

- Removes duplicate rows
- Converts `trans_date_trans_time` to timestamp format
- Replaces empty strings with null values
- Trims spaces from selected string columns
- Fills missing numeric columns with mean values
- Fills missing string columns with `Unknown`

Numeric columns handled include:

```text
amt, lat, long, merch_lat, merch_long, city_pop
```

String columns handled include:

```text
category, merchant, city, state, gender
```

---

### 9.3 Feature Engineering

The project creates four engineered features.

| Feature | Description |
|---|---|
| `is_weekend` | Indicates whether the transaction occurred on Saturday or Sunday |
| `distance_km` | Distance between customer and merchant using the Haversine formula |
| `avg_amt_recent` | Average amount of the latest 10 transactions per card |
| `amt_bucket` | Groups transaction amount into predefined ranges |

### Haversine Distance

The `distance_km` feature is calculated using customer coordinates and merchant coordinates:

```text
Customer location: lat, long
Merchant location: merch_lat, merch_long
```

This helps detect transactions that occur far away from the customer's usual location.

### Recent Average Amount

The `avg_amt_recent` feature is calculated using a Spark window function partitioned by credit card number:

```text
partitionBy(cc_num).orderBy(trans_date_trans_time).rowsBetween(-10, 0)
```

This captures recent spending behavior for each card.

---

## 10. Machine Learning Methodology

### 10.1 Categorical Encoding

The categorical columns are converted into numeric indexes using `StringIndexer`:

```text
category → category_index
gender → gender_index
```

The parameter `handleInvalid="keep"` is used so unseen or invalid values do not break the pipeline.

---

### 10.2 Feature Vector Creation

The selected input features are combined using `VectorAssembler`.

```text
amt, city_pop, is_weekend, distance_km,
avg_amt_recent, category_index, gender_index
```

The output column is:

```text
features
```

---

### 10.3 Feature Scaling

The project uses `StandardScaler` to create:

```text
scaled_features
```

This is used as the model input column for both Logistic Regression and Random Forest.

---

### 10.4 Models Trained

Two models are trained and evaluated.

| Model | Purpose |
|---|---|
| Logistic Regression | Baseline model |
| Random Forest | Improved model with class weights |

The Random Forest model uses class weights to handle the severe fraud class imbalance:

```text
Fraud transactions: weight = 200.0
Legitimate transactions: weight = 1.0
```

---

### 10.5 Evaluation Metrics

The following metrics are used:

- AUC-ROC
- Accuracy
- Weighted Precision
- Weighted Recall
- F1 Score
- Confusion Matrix

These metrics provide a more complete evaluation than accuracy alone, especially because fraud detection is a highly imbalanced classification problem.

---

## 11. Expected Results

The project is expected to produce model comparison results similar to the following:

| Metric | Logistic Regression | Random Forest |
|---|---:|---:|
| AUC-ROC | 0.9514 | 0.9946 |
| Accuracy | 0.9956 | 0.9875 |
| Precision | 0.9932 | 0.9968 |
| Recall | 0.9956 | 0.9875 |
| F1 Score | 0.9942 | 0.9912 |

The Random Forest model is selected as the stronger fraud detection model because it catches significantly more fraud cases after applying class weights.

Expected fraud detection comparison:

| Model | Fraud Caught | Fraud Missed | False Alarms |
|---|---:|---:|---:|
| Logistic Regression | 101 | 2,044 | 411 |
| Random Forest | 2,028 | 117 | 6,857 |

---

## 12. Dashboard Reproduction

After the model is trained, run:

```bash
streamlit run app.py
```

The dashboard provides:

- Dataset summary metrics
- Fraud vs legitimate distribution
- Fraud rate by category
- Fraud by amount bucket
- Fraud by hour of day
- Distance-based fraud analysis
- High-value transaction fraud analysis
- Gender-based fraud analysis
- Weekend vs weekday fraud comparison
- Live single-transaction prediction
- Logistic Regression vs Random Forest comparison

For live prediction, the dashboard loads the saved model from:

```text
models/fraud_rf_pipeline/
```

Then it predicts fraud using user-provided transaction inputs.

---

## 13. Cleaning Generated Files

To remove generated streaming data and saved models, run:

```bash
make clean
```

This removes:

```text
data/stream_input/
models/fraud_lr_pipeline/
models/fraud_rf_pipeline/
```

After cleaning, run the pipeline again to regenerate these outputs.

---

## 14. Troubleshooting

### Issue 1: Java Error or Spark Cannot Start

Possible cause:

- Java is not installed
- `JAVA_HOME` is incorrect
- Java version is incompatible

Solution:

```bash
java -version
```

Make sure Java 17 is installed and `JAVA_HOME` is correctly set.

---

### Issue 2: Dataset File Not Found

Possible cause:

Dataset files are missing or incorrectly named.

Solution:

Make sure the following files exist:

```text
data/CreditCard_Fraud_Train.csv
data/CreditCard_Fraud_Test.csv
```

---

### Issue 3: Streamlit Dashboard Cannot Load Model

Possible cause:

The Random Forest model has not been trained yet.

Solution:

Run:

```bash
make pipeline
```

Then launch the dashboard again:

```bash
streamlit run app.py
```

---

### Issue 4: Streaming Folder Missing

Possible cause:

The streaming script has not been executed.

Solution:

Run:

```bash
make streaming
```

This creates:

```text
data/stream_input/
```

---

### Issue 5: Python Cannot Find Project Modules

Possible cause:

`PYTHONPATH` is not set to `src`.

Solution:

Run scripts using:

```bash
PYTHONPATH=src python3 src/ingestion.py
```

or use the Makefile commands.

---

## 15. Reproduction Checklist

Use this checklist before submission.

- [ ] Java 17 installed and verified
- [ ] Python dependencies installed
- [ ] Dataset downloaded from Kaggle
- [ ] Dataset files renamed correctly
- [ ] Dataset files placed in `data/`
- [ ] `PYTHONPATH=src` configured when running scripts manually
- [ ] `ingestion.py` runs successfully
- [ ] `transformations.py` runs successfully
- [ ] `streaming.py` creates `data/stream_input/`
- [ ] `ml_pipeline.py` saves both trained models
- [ ] `app.py` launches Streamlit dashboard
- [ ] Dashboard live prediction works
- [ ] Generated model folders exist
- [ ] Final documentation files are saved under `docs/`

---

## 16. Full Reproduction Command Summary

```bash
# 1. Install dependencies
make install

# 2. Run ingestion and EDA
make ingestion

# 3. Run Spark SQL transformations
make transformations

# 4. Run streaming simulation
make streaming

# 5. Train and evaluate ML models
make pipeline

# 6. Launch dashboard
make dashboard
```

Or run the entire pipeline using:

```bash
bash run.sh
```

---

## 17. Conclusion

Following this guide allows the complete fraud detection pipeline to be reproduced from raw dataset files to final dashboard output. The project demonstrates data ingestion, data cleaning, feature engineering, Spark SQL analytics, Structured Streaming, Spark MLlib model training, streaming predictions, and dashboard visualization in a single end-to-end workflow.

