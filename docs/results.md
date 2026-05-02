# Results
## 1. Overview

This section presents the results of the end-to-end credit card fraud detection pipeline implemented using Apache Spark. The results include Exploratory Data Analysis (EDA), Spark SQL-based transformations, streaming analytics, machine learning model evaluation, and dashboard insights.

The dataset consists of **1,852,394 transactions**, with a fraud rate of approximately **0.52%**, making it a highly imbalanced classification problem.

---

## 2. Exploratory Data Analysis (EDA)

### Key Observations

- Total transactions: **1,852,394**
- Fraud transactions: **9,651**
- Legitimate transactions: **1,842,743**
- Fraud rate: **0.52%**
- Dataset is highly imbalanced

### Amount Statistics

- Mean transaction amount: **70.06**
- Standard deviation: **159.25**
- Minimum: **1.0**
- Maximum: **28,948.9**

### Category Distribution (Top Categories)

- gas_transport: 188,029
- grocery_pos: 176,191
- home: 175,460
- shopping_pos: 166,463

### Behavioral Insights

- Weekend fraud rate: **0.00508**
- Weekday fraud rate: **0.00527**
  - Fraud is slightly higher on weekdays

### Distance-Based Insight

- Fraud average distance: **76.25 km**
- Legitimate average distance: **76.11 km**

➡️ Distance alone is not a strong discriminator of fraud.

---

## 3. Spark SQL Transformations

### 3.1 Fraud Rate by Category

Top fraud-prone categories:

- shopping_net → **1.59%**
- misc_net → **1.30%**
- grocery_pos → **1.26%**

➡️ Online and miscellaneous categories show higher fraud risk.

---

### 3.2 Fraud Pattern by Hour

Fraud is significantly higher during:

- Hour 22 → **2.60%**
- Hour 23 → **2.54%**
- Hours 0–3 → ~**1.3%**

Daytime fraud rates are much lower (~0.1%).

➡️ Fraud peaks during late-night and early-morning hours.

---

### 3.3 High-Risk Customers

Top customers show fraud rates around:

- ~**2.0% – 2.15%**

➡️ Indicates repeated fraud behavior on certain cards.

---

### 3.4 Distance-Based Fraud Analysis

- Medium distance fraud rate: **0.00524**
- Near distance fraud rate: **0.00506**

➡️ Fraud distribution is similar across distances.

---

### 3.5 High-Value Transaction Fraud

- High-value transactions (> $500):
  - Fraud rate: **21.57%**
- Normal transactions:
  - Fraud rate: **0.27%**

➡️ High-value transactions are **~79x more likely to be fraud**.

---

### 3.6 State-Level Fraud Analysis

- Fraud-prone categories vary across states
- Example:
  - AK → shopping_net highest fraud
  - CA → grocery_pos highest fraud

➡️ Fraud behavior varies geographically.

---

## 4. Streaming Analytics

The system successfully simulates real-time streaming using micro-batch processing.

### Observations

- Streaming queries correctly aggregate fraud trends
- Real-time fraud patterns are continuously updated
- Supports:
  - Fraud by category
  - Fraud by hour
  - High-value fraud detection

➡️ Demonstrates capability for real-time fraud monitoring.

---

## 5. Machine Learning Model Performance

### 5.1 Logistic Regression

| Metric    | Value  |
|----------|--------|
| AUC-ROC  | 0.9514 |
| Accuracy | 0.9956 |
| Precision| 0.9932 |
| Recall   | 0.9956 |
| F1 Score | 0.9942 |

#### Confusion Matrix

- True Negatives: **553,163**
- False Positives: **411**
- False Negatives: **2,044**
- True Positives: **101**

➡️ Model has high accuracy but **misses many fraud cases**.

---

### 5.2 Random Forest

| Metric    | Value  |
|----------|--------|
| AUC-ROC  | 0.9954 |
| Accuracy | 0.9871 |
| Precision| 0.9968 |
| Recall   | 0.9871 |
| F1 Score | 0.991  |

#### Confusion Matrix

- True Negatives: **546,505**
- False Positives: **7,069**
- False Negatives: **112**
- True Positives: **2,033**

➡️ Model detects significantly more fraud cases with higher recall.

---

### 5.3 Model Comparison

| Metric    | Logistic Regression | Random Forest |
|----------|-------------------|--------------|
| AUC-ROC  | 0.9514            | 0.9954       |
| Accuracy | 0.9956            | 0.9871       |
| Precision| 0.9932            | 0.9968       |
| Recall   | 0.9956            | 0.9871       |
| F1 Score | 0.9942            | 0.991        |

### Key Findings

- Random Forest achieves **higher AUC-ROC (0.9954)**
- Logistic Regression has slightly higher accuracy due to imbalance
- Random Forest detects **20x more fraud cases (2033 vs 101)**
- Random Forest significantly reduces missed fraud

---

## 6. Fraud Detection Capability

| Model               | Fraud Caught | Fraud Missed | False Alarms |
|--------------------|------------|-------------|-------------|
| Logistic Regression | 101        | 2,044       | 411         |
| Random Forest       | 2,033      | 112         | 7,069       |

### Observations

- Random Forest improves fraud detection dramatically
- Trade-off: increase in false positives
- Critical improvement in reducing missed fraud cases

---

## 7. Key Business Insights

- Fraud is concentrated in **online transaction categories**
- Fraud peaks during **late-night hours**
- **High-value transactions are extremely risky**
- Behavioral features such as:
  - recent transaction patterns
  - transaction distance
  - spending spikes  
  significantly improve fraud detection

---

## 8. Summary

The project successfully demonstrates:

- Scalable fraud detection using Apache Spark
- Real-time streaming simulation
- Effective feature engineering
- Strong machine learning performance

The Random Forest model, enhanced with class weights, provides **high fraud detection capability in an imbalanced dataset**, making it suitable for real-world fraud detection scenarios.

---

> The results confirm that combining distributed data processing, behavioral feature engineering, and machine learning enables accurate and scalable fraud detection systems.
