# Dataset Overview

The dataset used in this project is the Credit Card Fraud Detection dataset from Kaggle:
https://www.kaggle.com/datasets/kartik2112/fraud-detection
The dataset contains credit card transaction records with customer details, merchant details, transaction amount, location information, and the fraud label.
There are 23 original columns with headings in the dataset. In addition to these, the CSV/Excel file also contains one unnamed index column at the beginning, which stores the row number of each transaction. This unnamed column is not an actual feature, so it was removed during ingestion.
So technically:
Unnamed index column: 1
Actual dataset columns: 23
Independent features: 22
Target column: is_fraud
________________________________________
## Unnamed Index Column
### Unnamed first column
This is the first column in the Excel file without a column name.
It contains row numbers such as:
0, 1, 2, 3, ...
This column only represents the transaction row index. It does not provide meaningful information for fraud detection. In Spark, this column appeared as _c0, so it was removed during ingestion.
________________________________________
## Feature Descriptions
### 1. trans_date_trans_time
This column stores the date and time when the transaction occurred.

Example:
1/1/2019 0:00

This feature is useful for extracting time-based patterns such as transaction hour, day of week, and weekend indicator.
________________________________________
### 2. cc_num
This column represents the customer’s credit card number or unique card identifier.
It is useful for grouping transactions by customer and analyzing spending behavior over time.
________________________________________
### 3. merchant
This column contains the merchant name where the transaction happened.

Example:
fraud_Rippin, Kub and Mann

It helps identify merchant-level transaction patterns.
________________________________________
### 4. category
This column represents the transaction category.

Examples:
misc_net
grocery_pos
gas_transport
shopping_net

This is useful for finding which transaction categories have higher fraud rates.
________________________________________
### 5. amt
This column represents the transaction amount.

Example:
4.97

This is one of the most important features because fraud patterns can differ between low-value and high-value transactions.
________________________________________
### 6. first
This column contains the customer’s first name.

Example:
Jennifer

This is mainly customer identity information and is not usually useful directly for machine learning.
________________________________________
### 7. last
This column contains the customer’s last name.

Example:
Banks

Like the first name, this is mainly identification information and is not a strong predictive feature by itself.
________________________________________
### 8. gender
This column contains the gender of the cardholder.

Values:
M
F

It can be used during exploratory analysis to compare fraud distribution across gender groups.
________________________________________
### 9. street
This column contains the customer’s street address.

Example:
561 Perry Cove

It gives customer location information, but for modeling, latitude and longitude are usually more useful than the raw street address.
________________________________________
### 10. city
This column contains the city where the customer lives.

Example:
Moravian Falls

It can be used for location-based analysis.
________________________________________
### 11. state
This column contains the state where the customer lives.

Example:
NC

This is useful for state-wise fraud analysis and geographical grouping.
________________________________________
### 12. zip
This column contains the ZIP code of the customer location.

Example:
28654

It provides more detailed geographical information.
________________________________________
### 13. lat
This column represents the latitude of the customer’s location.

Example:
36.0788

It is useful for calculating the distance between the customer and merchant locations.
________________________________________
### 14. long
This column represents the longitude of the customer’s location.

Example:
-81.1781

Together with lat, it helps in location-based fraud analysis.
________________________________________
### 15. city_pop
This column represents the population of the customer’s city.

Example:
3495

It can help compare transaction behavior between smaller and larger population areas.
________________________________________
### 16. job
This column contains the customer’s occupation.

Example:
Psychologist, counselling

It may help understand spending behavior across different occupation groups.
________________________________________
### 17. dob
This column contains the customer’s date of birth.

Example:
3/9/1988

It can be used to derive the customer’s age, which may be useful for demographic analysis.
________________________________________
### 18. trans_num
This column is the unique transaction identifier.

Example:
0b242abb623afc578575680df30655b9

It is useful for uniquely tracking each transaction, but it is not usually used as a predictive feature.
________________________________________
### 19. unix_time
This column stores the transaction time in Unix timestamp format.

Example:
1325376018

It is useful for time-based ordering and time difference calculations.
________________________________________
### 20. merch_lat
This column represents the latitude of the merchant location.

Example:
36.011293

It is used along with customer latitude to calculate the customer-to-merchant distance.
________________________________________
### 21. merch_long
This column represents the longitude of the merchant location.

Example:
-82.048315

It is used along with customer longitude to calculate geographical distance.
________________________________________
### 22. is_fraud
This is the target column.

Values:
0 = Legitimate transaction
1 = Fraudulent transaction

This is the column that the machine learning model will try to predict.
________________________________________
## Target and Independent Features
### Target Feature

is_fraud

This is the output label used for classification.
### Independent Features
All other meaningful columns are treated as independent features or used for feature engineering.
Examples include:
trans_date_trans_time

cc_num

merchant

category

amt

gender

city

state

lat

long

city_pop

job

dob

unix_time

merch_lat

merch_long

The unnamed first column is not treated as an independent feature because it is only a row index.

