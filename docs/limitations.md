# Limitations
## 1. Synthetic Dataset and Limited Real-World Complexity
This project uses a publicly available Kaggle dataset for credit card fraud detection. While the dataset is large (1.85M transactions), it is synthetic in nature and does not fully capture real-world fraud behavior.
-	Fraud patterns in real banking systems are more dynamic and evolve continuously.
-	The dataset does not include adaptive fraud strategies, such as coordinated attacks or multi-account fraud.
-	Some behavioral features (e.g., device fingerprinting, IP address, transaction velocity across platforms) are not present.
As a result, the model may perform well on this dataset but may not generalize directly to real-world banking environments.

---


## 2. Severe Class Imbalance
The dataset contains only ~0.52% fraud transactions, which creates a highly imbalanced classification problem.
-	Without handling imbalance, models tend to predict most transactions as legitimate.
-	This project addresses imbalance using class weights (200x for fraud) in Random Forest.
-	However, this approach has trade-offs:
  1.	Improves fraud detection (recall)
  2.	Increases false positives (false alarms)
More advanced imbalance techniques such as SMOTE, ADASYN, or ensemble balancing methods were not implemented.

---


## 4. Limited Feature Set
The model relies on a relatively small set of engineered features:
amt, city_pop, is_weekend, distance_km,
avg_amt_recent, category_index, gender_index

Limitations include:
-	No temporal velocity features (e.g., transactions per minute/hour)
-	No user behavior profiling over longer time windows
-	No device, IP, or authentication-related features
-	No merchant risk scoring or historical fraud reputation

This restricts the model’s ability to capture complex fraud patterns.

---


## 4. Static Batch-Based Training
The machine learning models are trained on static historical data.
-	The model does not automatically update with new incoming data.
-	Fraud patterns can change over time (concept drift), but the model does not adapt dynamically.
-	Retraining must be done manually.
In real-world systems, models are often updated using continuous or incremental learning pipelines.

---


## 5. Simulated Streaming Instead of True Real-Time Data
The streaming pipeline uses file-based micro-batch simulation:
CSV files → written to folder → read using readStream
Limitations:
-	Not truly real-time (depends on file arrival)
-	No integration with real streaming systems like:
    - Apache Kafka
    - AWS Kinesis
-	No event-time watermarking or late data handling
This limits the realism of the streaming architecture.

---


## 6. Limited Model Diversity
The project implements only two models:
-	Logistic Regression (baseline)
-	Random Forest (final model)
Limitations:
-	No advanced models such as:
    -Gradient Boosting (XGBoost, LightGBM)
    -Neural Networks
    -Deep learning-based anomaly detection
-	No ensemble stacking or hybrid models
Although Random Forest performs well, other models could potentially improve performance further.

---


## 7. No Hyperparameter Tuning
Model parameters are manually defined:
Random Forest:
- numTrees = 100
- maxDepth = 10
Limitations:
-	No automated tuning using:
    - Grid Search
    - Random Search
    - CrossValidator (Spark ML)
-	Model performance may not be fully optimized

---

## 8. Limited Evaluation for Business Impact
The project evaluates models using standard metrics:
AUC-ROC, Accuracy, Precision, Recall, F1 Score
However:
-	No cost-based evaluation (financial loss due to fraud vs false alarms)
-	No threshold tuning for business trade-offs
-	No precision-recall curve analysis
In real systems, fraud detection is evaluated based on monetary impact, not just accuracy metrics.

---


## 9. Simplified Distance-Based Feature
Distance is calculated using the Haversine formula, which is accurate geographically, but:
-	Assumes transactions occur at exact coordinates
-	Does not consider:
    - Travel patterns
    - Frequent user locations
    - Online transactions (where location may not be meaningful)
Thus, distance alone is not always a reliable fraud indicator.

---


## 10. Limited Scalability Testing

Although Spark is used, the project does not include:

-	Cluster-based deployment testing
-	Performance benchmarking at scale
-	Resource optimization (e.g., partition tuning, caching strategies)

All experiments are performed in a local or limited environment.

---


## 11. Dashboard Constraints

The Streamlit dashboard is designed for visualization and demonstration:

-	Uses sampled data instead of full dataset
-	Predictions are single-transaction based (not batch predictions)
-	No authentication or user-level access control
-	Not optimized for production-scale deployment

---


## 12. No External Data Integration

The system relies only on the provided dataset.

Missing integrations include:

-	Blacklisted cards or merchants
-	External fraud intelligence feeds
-	Geo-risk scoring APIs
-	Historical banking transaction systems
These are commonly used in real fraud detection systems.

---


## Summary

While this project successfully demonstrates an end-to-end fraud detection pipeline using Apache Spark, it is designed primarily for academic and learning purposes.

The key limitations arise from:

-	Synthetic dataset constraints
-	Simplified feature engineering
-	Simulated streaming
-	Limited model complexity

Despite these limitations, the project provides a strong foundation for building scalable, real-time fraud detection systems and can be extended further for production-level applications.

---


# Future Work

## 1. Use Real-World Transaction Data

To overcome the limitation of synthetic datasets, future work can involve:

-	Training the model on real banking transaction datasets
-	Incorporating live fraud cases and evolving fraud patterns
-	Using industry datasets with richer attributes such as device IDs, IP addresses, and authentication logs

This would significantly improve the model’s ability to generalize to real-world scenarios.

---


## 2. Advanced Handling of Class Imbalance

Instead of relying only on class weights, future improvements include:

-	Implementing SMOTE (Synthetic Minority Over-sampling Technique)
-	Using ADASYN or hybrid resampling techniques
-	Applying ensemble methods for imbalanced learning (e.g., Balanced Random Forest)

This would help improve fraud detection while reducing false positives.

---


## 3. Enhanced Feature Engineering

Future work can focus on building more powerful features:
-	Transaction velocity features (e.g., number of transactions in last 5 minutes)
-	User behavioral profiling (spending patterns over weeks/months)
-	Merchant risk scoring based on historical fraud
-	Time-based features (time since last transaction, session activity)
-	Geographical patterns (frequent vs unusual locations)

These features can significantly improve model accuracy.

---


## 4. Online Learning and Model Updating
To handle changing fraud patterns (concept drift):
-	Implement incremental learning models
-	Build pipelines for automatic retraining with new data
-	Use sliding window retraining strategies
This will allow the system to adapt dynamically to new fraud behaviors.

---


## 5. Integration with Real-Time Streaming Systems
Replace file-based streaming with production-grade systems:
-	Integrate Apache Kafka for real-time event ingestion
-	Use Spark Structured Streaming with Kafka source
-	Implement event-time processing and watermarking
This will make the pipeline closer to real-world fraud detection systems.

---


## 6. Advanced Machine Learning Models
Explore more powerful models such as:
-	Gradient Boosting Models (XGBoost, LightGBM)
-	Deep Learning Models (Neural Networks, LSTMs)
-	Autoencoders for anomaly detection
-	Graph-based fraud detection (fraud networks)
These models can capture complex patterns beyond traditional methods.

---


## 7. Hyperparameter Optimization
Improve model performance by:
-	Using CrossValidator in Spark ML
-	Applying Grid Search or Random Search
-	Optimizing parameters like:
    -number of trees
    -depth of trees
    -learning rate (for boosting models)
This ensures the model is tuned for best performance.

---


## 8. Business-Oriented Evaluation Metrics
Future work should include:
-	Cost-sensitive evaluation
    -Cost of missed fraud vs false alarms
-	Threshold tuning based on business needs
-	Precision-Recall curves for imbalanced data
-	ROI-based fraud detection metrics
This aligns the system with real-world financial impact.

---


## 9. Improved Distance and Location Modeling
Enhance the current distance feature by:
-	Tracking frequent user locations
-	Identifying unusual travel patterns
-	Handling online transactions separately
-	Using geo-risk scoring models
This makes location-based fraud detection more reliable.

---


## 10. Scalable Distributed Deployment
Future improvements include:
-	Deploying on Spark clusters (AWS EMR, Databricks)
-	Optimizing:
    -partitioning
    -caching strategies
    -memory usage
-	Benchmarking performance at large scale
This ensures the system can handle production workloads.

---


## 11. Production-Ready Dashboard
Enhance the Streamlit dashboard by:
-	Supporting real-time streaming predictions
-	Adding user authentication and role-based access
-	Enabling batch transaction monitoring
-	Deploying using Docker or cloud platforms
This transforms the dashboard into a real-world monitoring system.

---


## 12. Integration with External Systems
Future work can include integration with:
-	Fraud intelligence APIs
-	Blacklisted cards/merchant databases
-	Banking transaction systems
-	Real-time alerting systems (SMS/Email notifications)
This will make the system more practical and deployable.

---


> This project establishes a strong foundation for scalable fraud detection using Apache Spark. By addressing the above limitations and incorporating advanced techniques such as real-time streaming, adaptive learning, and richer feature engineering, the system can be extended into a production-grade fraud detection solution suitable for real-world financial systems.
