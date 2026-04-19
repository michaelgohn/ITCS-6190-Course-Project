import os
os.environ["JAVA_HOME"] = "/opt/homebrew/opt/openjdk@17"

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Page Config
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="",
    layout="wide"
)

st.title("Credit Card Fraud Detection Dashboard")
st.markdown("**ITCS 6190 Cloud Computing for Data Analytics — Team 4**")
st.markdown("---")


# Initialize Spark + Load Model
@st.cache_resource
def get_spark_and_model():
    from pyspark.sql import SparkSession
    from pyspark.ml import PipelineModel

    spark = SparkSession.builder \
        .appName("FraudDashboard") \
        .config("spark.driver.memory", "2g") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    rf_model = PipelineModel.load("models/fraud_rf_pipeline")
    return spark, rf_model


# Load Sample Data
@st.cache_data
def load_sample_data():
    train = pd.read_csv("data/CreditCard_Fraud_Train.csv", nrows=50000)
    test  = pd.read_csv("data/CreditCard_Fraud_Test.csv",  nrows=10000)
    df = pd.concat([train, test], ignore_index=True)
    return df


# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", [
    "Data Overview",
    "Fraud Analysis",
    "Distance Analysis",
    "Live Prediction",
    "Model Comparison"
])

st.sidebar.markdown("---")
st.sidebar.markdown("### Dataset Info")
st.sidebar.markdown("- **Source:** Kaggle (kartik2112)")
st.sidebar.markdown("- **Total Rows:** 1,852,394")
st.sidebar.markdown("- **Fraud Rate:** 0.52%")
st.sidebar.markdown("- **Best Model:** Random Forest")
st.sidebar.markdown("- **AUC-ROC:** 0.9908")

with st.spinner("Loading data..."):
    df = load_sample_data()


# PAGE 1: Data Overview
if page == "Data Overview":

    st.header("Data Overview")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Transactions", "1,852,394")
    col2.metric("Fraud Transactions", "9,651")
    col3.metric("Legitimate", "1,842,743")
    col4.metric("Fraud Rate", "0.52%")

    st.markdown("---")
    st.subheader("Sample Data")
    st.dataframe(df[["trans_date_trans_time", "merchant", "category",
                      "amt", "gender", "state", "is_fraud"]].head(10))

    st.markdown("---")
    st.subheader("Features Used in ML Model")
    features = pd.DataFrame({
        "Feature": ["amt", "city_pop", "is_weekend", "distance_km",
                    "avg_amt_recent", "category_index", "gender_index"],
        "Description": [
            "Transaction amount ($)",
            "Population of customer city",
            "Weekend transaction? (0/1)",
            "Distance between customer and merchant (km)",
            "Average of last 10 transactions per card",
            "Encoded transaction category",
            "Encoded gender"
        ],
        "Type": ["Original", "Original", "Engineered",
                 "Engineered", "Engineered", "Encoded", "Encoded"]
    })
    st.dataframe(features)


# PAGE 2: Fraud Analysis
elif page == "Fraud Analysis":

    st.header("Fraud Analysis")

    st.subheader("1. Fraud vs Legitimate")
    col1, col2 = st.columns(2)

    with col1:
        fraud_counts = df["is_fraud"].value_counts()
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.bar(["Legitimate", "Fraud"], fraud_counts.values,
               color=["#2ecc71", "#e74c3c"])
        ax.set_title("Transaction Count")
        ax.set_ylabel("Count")
        for i, v in enumerate(fraud_counts.values):
            ax.text(i, v + 10, f"{v:,}", ha="center", fontweight="bold")
        st.pyplot(fig)
        plt.close()

    with col2:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.pie(fraud_counts.values, labels=["Legitimate", "Fraud"],
               colors=["#2ecc71", "#e74c3c"], autopct="%1.2f%%")
        ax.set_title("Transaction Percentage")
        st.pyplot(fig)
        plt.close()

    st.markdown("---")

    st.subheader("2. Fraud Rate by Category")
    cat = df.groupby("category").agg(
        total=("is_fraud", "count"), fraud=("is_fraud", "sum")
    ).reset_index()
    cat["fraud_rate"] = (cat["fraud"] / cat["total"] * 100).round(2)
    cat = cat.sort_values("fraud_rate", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(cat["category"], cat["fraud_rate"], color="#e74c3c")
    ax.set_xlabel("Fraud Rate (%)")
    ax.set_title("Fraud Rate by Category")
    for bar, val in zip(bars, cat["fraud_rate"]):
        ax.text(val + 0.05, bar.get_y() + bar.get_height()/2,
                f"{val}%", va="center", fontsize=8)
    st.pyplot(fig)
    plt.close()

    st.markdown("---")

    st.subheader("3. Fraud by Amount Bucket")
    df["amt_bucket"] = pd.cut(df["amt"],
        bins=[0, 100, 500, 1000, 5000, float("inf")],
        labels=["$0-100", "$100-500", "$500-1K", "$1K-5K", "$5K+"]
    )
    bucket = df.groupby("amt_bucket", observed=True).agg(
        total=("is_fraud", "count"), fraud=("is_fraud", "sum")
    ).reset_index()
    bucket["fraud_rate"] = (bucket["fraud"] / bucket["total"] * 100).round(2)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(bucket["amt_bucket"].astype(str), bucket["fraud_rate"], color="#3498db")
    ax.set_xlabel("Amount Bucket")
    ax.set_ylabel("Fraud Rate (%)")
    ax.set_title("Fraud Rate by Transaction Amount")
    for i, v in enumerate(bucket["fraud_rate"]):
        ax.text(i, v + 0.05, f"{v}%", ha="center", fontweight="bold")
    st.pyplot(fig)
    plt.close()


# PAGE 3: Distance Analysis
elif page == "Distance Analysis":

    st.header("Distance-Based Fraud Analysis")

    st.info("""
    Just like Gmail alerts you when someone logs in from an unusual location,
    our system flags transactions where the merchant is unusually far 
    from the customer's home location.
    """)

    st.markdown("---")

    st.subheader("1. Gmail-Like Alert System")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Gmail Login Alert")
        st.markdown("""
        ```
        New login detected!
        Location: Tokyo, Japan
        Device: Unknown laptop
        Time: 3:00 AM
        
        Was this you?
        [YES]  [NO - Secure Account]
        ```
        """)

    with col2:
        st.markdown("### Our Fraud Alert System")
        st.markdown("""
        ```
        Suspicious Transaction Detected!
        Merchant: Tokyo Electronics
        Distance: 13,000 km from home
        Amount: $1,500
        Time: 3:00 AM (Weekend)
        
        ML Model: FRAUD DETECTED
        Confidence: 99.08%
        Action: Transaction BLOCKED
        ```
        """)

    st.markdown("---")

    st.subheader("2. Distance Risk Levels")
    distance_table = pd.DataFrame({
        "Scenario": [
            "Local store purchase",
            "Nearby city purchase",
            "Online order (same state)",
            "Cross-country purchase",
            "International purchase"
        ],
        "Customer": ["Charlotte, NC"] * 5,
        "Merchant Location": [
            "Charlotte, NC",
            "Raleigh, NC",
            "New York, NY",
            "Los Angeles, CA",
            "Tokyo, Japan"
        ],
        "Distance": ["~5 km", "~250 km", "~900 km", "~3,500 km", "~13,000 km"],
        "Risk Level": ["Low", "Low", "Medium", "High", "Very High"]
    })
    st.dataframe(distance_table)

    st.markdown("---")

    st.subheader("3. How We Calculate Distance")
    st.code("""
# Haversine Formula - calculates real distance on Earth's surface
df = df.withColumn("distance_km", col("c") * 6371)

# Uses customer lat/long vs merchant lat/long
# Example:
# Customer: (35.2271, -80.8431) -> Charlotte, NC
# Merchant: (35.6762, 139.6503) -> Tokyo, Japan
# Distance = 13,247 km -> FLAGGED
    """, language="python")

    st.markdown("---")

    st.subheader("4. Fraud vs Legitimate - Average Distance")
    dist_data = pd.DataFrame({
        "Type": ["Legitimate", "Fraud"],
        "Avg Distance (km)": [76.11, 76.25]
    })

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(dist_data["Type"], dist_data["Avg Distance (km)"],
           color=["#2ecc71", "#e74c3c"])
    ax.set_ylabel("Average Distance (km)")
    ax.set_title("Average Distance: Fraud vs Legitimate")
    ax.set_ylim(75.5, 76.5)
    for i, v in enumerate(dist_data["Avg Distance (km)"]):
        ax.text(i, v + 0.01, f"{v} km", ha="center", fontweight="bold")
    st.pyplot(fig)
    plt.close()

    st.success("""
    How Our Project Handles This:
    1. Haversine formula calculates exact distance for every transaction
    2. distance_km is used as a key feature in Random Forest model
    3. Combined with amount, category, and time patterns
    4. Streaming pipeline processes transactions in real-time
    5. Model flags suspicious transactions instantly
    """)


# PAGE 4: Live Prediction
elif page == "Live Prediction":

    st.header("Live Fraud Prediction")
    st.markdown("Enter transaction details below to get an instant fraud prediction.")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Transaction Details")
        amt = st.number_input("Transaction Amount ($)",
                              min_value=1.0, max_value=30000.0, value=100.0, step=10.0)
        category = st.selectbox("Category", [
            "gas_transport", "grocery_pos", "home", "shopping_pos",
            "kids_pets", "shopping_net", "entertainment", "food_dining",
            "personal_care", "health_fitness", "misc_net", "misc_pos",
            "grocery_net", "travel"
        ])
        gender = st.selectbox("Gender", ["M", "F"])
        is_weekend = st.selectbox("Weekend Transaction?", [0, 1],
                                  format_func=lambda x: "Yes" if x == 1 else "No")

    with col2:
        st.subheader("Location Details")
        distance_km = st.slider(
            "Distance from Home to Merchant (km)",
            min_value=0, max_value=15000, value=50, step=10
        )
        city_pop = st.number_input("City Population",
                                   min_value=100, max_value=3000000, value=50000)
        avg_amt_recent = st.number_input(
            "Your Average Recent Transaction ($)",
            min_value=1.0, max_value=5000.0, value=70.0
        )

        st.markdown("### Distance Alert:")
        if distance_km > 5000:
            st.error(f"{distance_km:,} km — Very suspicious location")
        elif distance_km > 1000:
            st.warning(f"{distance_km:,} km — Unusual distance")
        elif distance_km > 200:
            st.info(f"{distance_km} km — Moderate distance")
        else:
            st.success(f"{distance_km} km — Normal distance")

    st.markdown("---")

    if st.button("Predict", type="primary", use_container_width=True):
        with st.spinner("Analyzing transaction with Random Forest model..."):
            try:
                spark, rf_model = get_spark_and_model()

                input_data = spark.createDataFrame([{
                    "amt"           : float(amt),
                    "city_pop"      : int(city_pop),
                    "is_weekend"    : int(is_weekend),
                    "distance_km"   : float(distance_km),
                    "avg_amt_recent": float(avg_amt_recent),
                    "category"      : str(category),
                    "gender"        : str(gender),
                    "is_fraud"      : 0.0
                }])

                result = rf_model.transform(input_data).select(
                    "prediction", "probability"
                ).collect()[0]

                pred = int(result["prediction"])
                prob = result["probability"]
                fraud_prob = round(float(prob[1]) * 100, 2)
                legit_prob = round(float(prob[0]) * 100, 2)

                st.markdown("---")
                st.subheader("Prediction Result")

                if pred == 1:
                    st.error(f"""
                    FRAUD DETECTED

                    This transaction has been flagged as FRAUDULENT.

                    | Detail | Value |
                    |---|---|
                    | Fraud Probability | {fraud_prob}% |
                    | Legitimate Probability | {legit_prob}% |
                    | Distance from Home | {distance_km:,} km |
                    | Amount | ${amt:,.2f} |
                    | Category | {category} |

                    Recommended Action: BLOCK this transaction and alert customer.
                    """)
                else:
                    st.success(f"""
                    LEGITIMATE TRANSACTION

                    This transaction appears to be legitimate.

                    | Detail | Value |
                    |---|---|
                    | Legitimate Probability | {legit_prob}% |
                    | Fraud Probability | {fraud_prob}% |
                    | Distance from Home | {distance_km} km |
                    | Amount | ${amt:,.2f} |
                    | Category | {category} |

                    Recommended Action: APPROVE this transaction.
                    """)

            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.info("Make sure models are saved in models/ folder.")

    st.markdown("---")
    st.markdown("### Sample Test Cases:")
    test_cases = pd.DataFrame({
        "Test Case": ["Normal purchase", "Suspicious purchase", "Fraud attempt"],
        "Amount": ["$50", "$200", "$1,500"],
        "Distance": ["5 km", "500 km", "13,000 km"],
        "Category": ["grocery_pos", "shopping_net", "shopping_net"],
        "Expected Result": ["Legitimate", "Check", "Fraud"]
    })
    st.dataframe(test_cases)


# PAGE 5: Model Comparison
elif page == "Model Comparison":

    st.header("Model Comparison")

    st.subheader("Performance Metrics")
    comparison = pd.DataFrame({
        "Metric": ["AUC-ROC", "Accuracy", "Precision", "Recall", "F1 Score"],
        "Logistic Regression": [0.9514, 0.9956, 0.9932, 0.9956, 0.9942],
        "Random Forest":       [0.9908, 0.9975, 0.9973, 0.9975, 0.9974]
    })
    st.dataframe(comparison)

    st.markdown("---")

    st.subheader("Visual Comparison")
    fig, ax = plt.subplots(figsize=(12, 5))
    x = range(len(comparison["Metric"]))
    width = 0.35
    bars1 = ax.bar([i - width/2 for i in x],
                   comparison["Logistic Regression"],
                   width, label="Logistic Regression", color="#3498db")
    bars2 = ax.bar([i + width/2 for i in x],
                   comparison["Random Forest"],
                   width, label="Random Forest", color="#e74c3c")
    ax.set_xticks(list(x))
    ax.set_xticklabels(comparison["Metric"])
    ax.set_ylim(0.94, 1.01)
    ax.set_ylabel("Score")
    ax.set_title("Model Performance Comparison")
    ax.legend()
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f"{bar.get_height():.4f}", ha="center", fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                f"{bar.get_height():.4f}", ha="center", fontsize=8)
    st.pyplot(fig)
    plt.close()

    st.markdown("---")

    st.subheader("Fraud Detection Capability")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Logistic Regression (Baseline)")
        st.metric("Fraud Caught", "101")
        st.metric("Fraud Missed", "2,044")
        st.metric("False Alarms", "411")

    with col2:
        st.markdown("### Random Forest (Best Model)")
        st.metric("Fraud Caught", "1,207", "+1,106")
        st.metric("Fraud Missed", "938", "-1,106")
        st.metric("False Alarms", "427")

    st.markdown("---")

    st.success("""
    Why Random Forest is Better:
    - Catches 12x more fraud than Logistic Regression (1,207 vs 101)
    - AUC improved from 0.9514 to 0.9908
    - Handles non-linear relationships in fraud patterns
    - Uses distance, amount, category, and time features together
    """)

    st.info("""
    Next Steps for Improvement:
    - Handle class imbalance using SMOTE oversampling
    - Hyperparameter tuning for better recall
    - Integrate with streaming for real-time alerts
    - Add more features based on user behavior patterns
    """)