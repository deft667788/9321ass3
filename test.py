import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score, r2_score
from scipy.stats import pearsonr

# Load data
df_train = pd.read_csv("train.tsv", sep="\t")

# Part I: Regression
# Define the target variable
b_train_part1 = df_train["revenue"]

# Data preprocessing
num_pipeline = Pipeline([
    ("std_scaler", StandardScaler()),
])

full_pipeline = ColumnTransformer([
    ("num", num_pipeline, ["Number_of_Shops_Around_ATM", "No_of_Other_ATMs_in_1_KM_radius",
                            "Estimated_Number_of_Houses_in_1_KM_Radius", "Average_Wait_Time", "rating"]),
    ("cat", OneHotEncoder(), ["ATM_Zone", "ATM_Placement", "ATM_TYPE", "ATM_looks", "ATM_Attached_to", "Day_Type"]),
])

# Data transformation
atm_prepared_train = full_pipeline.fit_transform(df_train)

# Random forest
reg_RF = RandomForestRegressor(n_estimators=100, random_state=42)

# Cross-validation predictions
y_pred_regression = cross_val_predict(reg_RF, atm_prepared_train, b_train_part1, cv=5)

# Correlation coefficient
correlation_coefficient, _ = pearsonr(b_train_part1, y_pred_regression)
print("Correlation Coefficient:", correlation_coefficient)

# Validate the condition for the correlation coefficient
if correlation_coefficient >= 0.40:
    print("The correlation coefficient requirement is met.")
else:
    print("The correlation coefficient requirement is not met.")

# Part II: Classification
# Define the target variable
b_train = df_train["rating"]

# Data preprocessing
num_pipeline = Pipeline([
    ("std_scaler", StandardScaler()),
])

full_pipeline = ColumnTransformer([
    ("num", num_pipeline, ["Number_of_Shops_Around_ATM", "No_of_Other_ATMs_in_1_KM_radius",
                            "Estimated_Number_of_Houses_in_1_KM_Radius", "Average_Wait_Time", "revenue"]),
    ("cat", OneHotEncoder(), ["ATM_Zone", "ATM_Placement", "ATM_TYPE", "ATM_looks", "ATM_Attached_to", "Day_Type"]),
])

# Data transformation
atm_prepared_train = full_pipeline.fit_transform(df_train)

# Random forest
clf_RF = RandomForestClassifier(n_estimators=100, random_state=42)

# Cross-validation predictions
y_pred_classification = cross_val_predict(clf_RF, atm_prepared_train, b_train, cv=5)

# Accuracy score
accuracy = accuracy_score(b_train, y_pred_classification)
print("Accuracy:", accuracy)

# Validate the condition for the accuracy
if accuracy >= 0.98:
    print("The accuracy requirement is met.")
else:
    print("The accuracy requirement is not met.")


