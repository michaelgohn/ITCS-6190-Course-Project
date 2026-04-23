# Credit Card Fraud Detection

## Team 4

- **Course:** ITCS 6190 — Cloud Computing for Data Analytics
- **Professor:** Prof. Marco Vieira
- **University:** University of North Carolina at Charlotte
- **Semester:** Spring 2026

## Team Members

| Name           |
| -------------- |
| Yamini Buyya   |
| Rishitha Adapa |
| Lakshmi Deepak |
| Michael Gohn   |

## Project Description

An end-to-end Big Data analytics pipeline for credit card fraud detection built with Apache Spark. The pipeline integrates all four Spark components:

- **Structured APIs** — data ingestion, cleaning, and feature engineering on 1.8 million transactions
- **Spark SQL** — complex fraud pattern analysis with 6 non-trivial queries
- **Spark Structured Streaming** — real-time transaction processing with 3 streaming queries
- **MLlib** — Logistic Regression and Random Forest classification with evaluation metrics

Processes 1,852,394 transactions achieving **99.46% AUC-ROC** with Random Forest, catching **94% of all fraud cases** (2,028 out of 2,145).

---

## Overview

Credit card fraud is a critical challenge for financial institutions worldwide. This project implements a **complete Big Data pipeline** for real-time fraud detection using Apache Spark.

### Problem Statement

Traditional fraud detection systems are slow, rule-based, and cannot scale to handle millions of transactions in real-time. There is a need for a distributed, intelligent system that detects fraud patterns automatically.

### Solution

We built an end-to-end pipeline that:

- Ingests and cleans **1,852,394 transactions** using PySpark
- Performs **complex SQL analysis** to discover fraud patterns
- Simulates **real-time transaction streaming** using Spark Structured Streaming
- Trains **Logistic Regression and Random Forest** models using PySpark MLlib
- Delivers **real-time streaming predictions** using the trained model
- Provides an **interactive Streamlit dashboard** for business insights

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    DATA PIPELINE                        │
│                                                         │
│  CSV Data → Ingestion → SQL Analysis → Streaming        │
│                              │                          │
│                              ▼                          │
│                    ML Pipeline                          │
│              (Train → Evaluate → Predict)               │
│                              │                          │
│                              ▼                          │
│                  Streamlit Dashboard                    │
└─────────────────────────────────────────────────────────┘
```

### Detailed Flow

```
CreditCard_Fraud_Train.csv (80%)
CreditCard_Fraud_Test.csv  (20%)
            │
            ▼
    ┌───────────────┐
    │  ingestion.py │  Clean + Feature Engineering
    └───────────────┘
            │
            ▼
    ┌───────────────────┐
    │ transformations.py│  6 Complex SparkSQL Queries
    └───────────────────┘
            │
            ▼
    ┌───────────────┐
    │  streaming.py │  5 Micro-batches + 3 Streaming Queries
    └───────────────┘
            │
            ▼
    ┌───────────────────┐
    │  ml_pipeline.py   │  Train LR + RF → Streaming Predictions
    └───────────────────┘
            │
            ▼
    ┌───────────────┐
    │    app.py     │  Interactive Streamlit Dashboard
    └───────────────┘
```

---

## Dataset

| Property                | Value                                                      |
| ----------------------- | ---------------------------------------------------------- |
| Name                    | Credit Card Transactions Fraud Detection                   |
| Source                  | Kaggle                                                     |
| Download                | https://www.kaggle.com/datasets/kartik2112/fraud-detection |
| Total Rows              | 1,852,394                                                  |
| Total Columns           | 23                                                         |
| Fraud Transactions      | 9,651 (0.52%)                                              |
| Legitimate Transactions | 1,842,743 (99.48%)                                         |
| Training Split          | 80% - 1,574,534 rows                                       |
| Test Split              | 20% - 555,719 rows                                         |

### Key Columns

| Column                | Description                             |
| --------------------- | --------------------------------------- |
| trans_date_trans_time | Transaction timestamp                   |
| cc_num                | Credit card number                      |
| merchant              | Merchant name                           |
| category              | Transaction category                    |
| amt                   | Transaction amount                      |
| lat, long             | Customer location                       |
| merch_lat, merch_long | Merchant location                       |
| city_pop              | Customer city population                |
| is_fraud              | Fraud label (0 = legitimate, 1 = fraud) |

---

## Feature Engineering

Four new features were engineered from raw data:

| Feature          | Description                              | Formula              |
| ---------------- | ---------------------------------------- | -------------------- |
| `is_weekend`     | Weekend transaction indicator            | `dayofweek ∈ {1, 7}` |
| `distance_km`    | Distance between customer and merchant   | Haversine formula    |
| `avg_amt_recent` | Average of last 10 transactions per card | Window function      |
| `amt_bucket`     | Transaction amount group                 | CASE WHEN bucketing  |

### Haversine Distance Formula

```python
# Calculates real geographic distance on Earth's surface
# Similar to how Gmail detects login from unusual location
distance_km = 2 * R * arcsin(sqrt(
    sin²(Δlat/2) + cos(lat1) * cos(lat2) * sin²(Δlon/2)
))
```

### ML Features Used (7 total)

```
amt, city_pop, is_weekend, distance_km,
avg_amt_recent, category_index, gender_index
```

---

## Installation

### Step 1 — Install Java 17

**Mac:**

```bash
# Install Java 17 using Homebrew
brew install openjdk@17

# Set JAVA_HOME
export JAVA_HOME=/opt/homebrew/opt/openjdk@17

# Verify
java -version
```

**Windows:**

```
1. Download Java 17 from:
   https://adoptium.net/temurin/releases/?version=17

2. Set JAVA_HOME in Environment Variables:
   JAVA_HOME = C:\Program Files\Path to Java Folder\jdk-17.x.x

3. Download winutils.exe and hadoop.dll for Hadoop (required on Windows):
   https://github.com/cdarlint/winutils/tree/master/hadoop-3.3.5/bin
   Place winutils.exe and hadoop.dll in: C:\hadoop\bin\

4. Set HADOOP_HOME in Environment Variables:
   HADOOP_HOME = C:\hadoop

5. Verify:
   java -version
```

---

### Step 2 — Install Python Dependencies

```bash
make install
```

Or manually:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

### Step 3 — Download Dataset

```
1. Download from Kaggle:
   https://www.kaggle.com/datasets/kartik2112/fraud-detection

2. Rename the files:
   fraudTrain.csv → CreditCard_Fraud_Train.csv
   fraudTest.csv  → CreditCard_Fraud_Test.csv

3. Place both files in:
   spark_project_template/data/
```

---

### Step 4 — Verify Setup

```bash
python3 -c "import pyspark; print('PySpark version:', pyspark.__version__)"
java -version
```

---

## How to Run

### Run Full Pipeline (One Command)

```bash
bash run.sh
```

Executes all steps in order:

```
Step 1 → Data Ingestion + EDA
Step 2 → Complex SQL Transformations
Step 3 → Spark Structured Streaming
Step 4 → ML Training + Streaming Predictions
Step 5 → Streamlit Dashboard
```

### Run Individual Components

```bash
make ingestion        # Data ingestion and EDA
make transformations  # Complex SQL queries
make streaming        # Streaming pipeline
make pipeline         # ML training and predictions
make dashboard        # Streamlit dashboard
make clean            # Reset models and data
```

### Launch Dashboard Only

```bash
streamlit run app.py
# Opens at http://localhost:8501
```

---

## Results

### Model Performance

| Metric    | Logistic Regression | Random Forest |
| --------- | ------------------- | ------------- |
| AUC-ROC   | 0.9514              | **0.9946**    |
| Accuracy  | 0.9956              | 0.9875        |
| Precision | 0.9932              | **0.9968**    |
| Recall    | 0.9956              | 0.9875        |
| F1 Score  | 0.9942              | 0.9912        |

### Fraud Detection Capability

| Model               | Fraud Caught | Fraud Missed | False Alarms |
| ------------------- | ------------ | ------------ | ------------ |
| Logistic Regression | 101          | 2,044        | 411          |
| **Random Forest**   | **2,028**    | **117**      | 6,857        |

**Best Model: Random Forest with Class Weights (200x)**

Random Forest catches **20x more fraud** than Logistic Regression, reducing missed fraud by **94%** (from 2,044 to 117 cases).

### Key Business Findings

**1. Card Testing Attack Pattern**
Fraudsters make small test transactions ($5-$10) in small cities on weekends to verify stolen card details. Our model identifies this pattern with high precision.

**2. Late Night Fraud Spike**
Hours 22-23 show a **2.6% fraud rate** — over 25x higher than daytime (0.1%). Real-time monitoring during these hours is critical.

**3. High-Value Transaction Risk**
Transactions above $500 carry a **21.5% fraud rate** compared to only 0.27% for normal transactions — making them 79x more risky.

**4. Top Fraud Categories**

| Category     | Fraud Rate |
| ------------ | ---------- |
| shopping_net | 1.59%      |
| misc_net     | 1.30%      |
| grocery_pos  | 1.26%      |

### Class Imbalance Handling

```
Before class weights:
Fraud caught → 1,207 out of 2,145 (56%)

After class weights (200x for fraud):
Fraud caught → 2,028 out of 2,145 (94%)
Missed fraud reduced from 938 → 117
```

---

## Project Structure

```
spark_project_template/
├── .github/                            # GitHub templates
├── data/
│   ├── stream_input/                   # 5 streaming micro-batch files
│   ├── streaming_data/                 # Notebook streaming data
│   ├── CreditCard_Fraud_Train.csv      # Training data (80%) — NOT in GitHub
│   ├── CreditCard_Fraud_Test.csv       # Test data (20%) — NOT in GitHub
│   └── external.txt                    # External data references
│
├── docs/
│   ├── slides/                         # Presentation slides
│   ├── dataset_overview.md             # Dataset source, schema, stats
│   ├── methodology.md                  # Pipeline and ML methodology
│   ├── results.md                      # Key findings and metrics
│   ├── limitations.md                  # Known constraints and future work
│   └── reproduction_guide.md           # Step-by-step reproduction guide
│
├── models/
│   ├── fraud_lr_pipeline/              # Saved Logistic Regression model
│   └── fraud_rf_pipeline/              # Saved Random Forest model
│
├── notebooks/
│   ├── eda.ipynb                       # Exploratory data analysis
│   ├── ingestion.ipynb                 # Data ingestion walkthrough
│   ├── ml_pipeline.ipynb               # ML pipeline walkthrough
│   ├── sql_queries.ipynb               # 9 complex SparkSQL queries
│   └── streaming_demo.ipynb            # Streaming demonstration
│
├── outputs/
│   ├── avg_transaction_amount.png      # Average transaction amount chart
│   ├── eda_summary.png                 # EDA summary visualization
│   ├── fraud_by_category.png           # Fraud rate by category
│   ├── fraud_by_gender.png             # Fraud rate by gender
│   ├── fraud_by_state.png              # Fraud rate by state
│   └── fraud_vs_nonfraud.png           # Fraud vs legitimate distribution
│
├── src/
│   ├── __init__.py
│   ├── utils.py                        # SparkSession and logger utilities
│   ├── ingestion.py                    # Data ingestion + feature engineering
│   ├── transformations.py              # 6 complex SQL transformations
│   ├── streaming.py                    # Spark Structured Streaming
│   └── ml_pipeline.py                  # ML training + streaming predictions
│
├── tests/
│   ├── test_ingestion.py               # Ingestion unit tests
│   ├── test_ml.py                      # ML pipeline unit tests
│   ├── test_sql.py                     # SQL query unit tests
│   └── test_streaming.py               # Streaming unit tests
│
├── .gitignore
├── app.py                              # Streamlit interactive dashboard
├── LICENSE
├── Makefile                            # Build automation commands
├── README.md                           # Project documentation
├── requirements.txt                    # Python dependencies
└── run.sh                              # One-command pipeline execution
```

---

## Technologies

| Technology                 | Version | Purpose                     |
| -------------------------- | ------- | --------------------------- |
| Apache Spark               | 3.x     | Distributed data processing |
| PySpark MLlib              | 3.x     | Machine learning pipeline   |
| Spark Structured Streaming | 3.x     | Real-time data processing   |
| SparkSQL                   | 3.x     | Complex query analysis      |
| Streamlit                  | Latest  | Interactive dashboard       |
| Matplotlib                 | Latest  | Data visualization          |
| Python                     | 3.9+    | Programming language        |
| Java                       | 17      | Spark runtime               |

---

## Limitations and Future Work

| Current Limitation                         | Proposed Solution                         |
| ------------------------------------------ | ----------------------------------------- |
| Synthetic dataset — limited fraud patterns | Train on real bank transaction data       |
| Class imbalance (0.52% fraud)              | Implement SMOTE oversampling              |
| Static CSV streaming source                | Integrate with Apache Kafka               |
| No hyperparameter tuning                   | Apply CrossValidator with ParamGrid       |
| Model slightly overfits synthetic patterns | Reduce class weight, apply regularization |

---
