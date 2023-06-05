import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight

# Load train and test datasets
train_data = pd.read_csv('train.tsv', sep='\t')
test_data = pd.read_csv('test.tsv', sep='\t')

# Part I: Regression
# Define the target variable
y_train_part1 = train_data['revenue']

# Data preprocessing
num_pipeline = Pipeline([
    ('std_scaler', StandardScaler()),  # Standardizing numeric features
])
full_pipeline = ColumnTransformer([
    ("num", num_pipeline,  # Numerical feature preprocessing
    ['Number_of_Shops_Around_ATM', 'No_of_Other_ATMs_in_1_KM_radius', 'Estimated_Number_of_Houses_in_1_KM_Radius',
    'Average_Wait_Time', 'rating']),
    ("cat", OneHotEncoder(),  # Categorical feature preprocessing
    ['ATM_Zone', 'ATM_Placement', 'ATM_TYPE', 'ATM_looks', 'ATM_Attached_to', 'Day_Type']),
])

# Data transformation
X_train = full_pipeline.fit_transform(train_data)
X_test = full_pipeline.transform(test_data)

# Train-test split for validation
X_train_part1, X_val_part1, y_train_part1, y_val_part1 = train_test_split(X_train, y_train_part1, test_size=0.2, random_state=42)

# Random forest
reg_RF = RandomForestRegressor(n_estimators=100, random_state=42)

# Training the model
reg_RF_model = reg_RF.fit(X_train_part1, y_train_part1)

# Predict on validation set
y_pred_part1 = reg_RF_model.predict(X_val_part1)

# Calculate R2 score
regression_accuracy = r2_score(y_val_part1, y_pred_part1)

# Part II: Classification
# Define the target variable
y_train = train_data['rating']

# Train-test split for validation
X_train_part2, X_val_part2, y_train_part2, y_val_part2 = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Calculate the class and sample weights
class_weight = compute_class_weight('balanced',
                                    classes=np.unique(y_train),
                                    y=y_train)
sample_weights = compute_sample_weight('balanced', y_train_part2)

# Create a dictionary of class weights
dict_w = {}
index = 2
for i in class_weight:
    dict_w.update({index: i})
    index += 1

# Random forest
clf_RF = RandomForestClassifier(n_estimators=100, class_weight=dict_w, random_state=42)

# Training the model
clf_RF = clf_RF.fit(X_train_part2, y_train_part2, sample_weight=sample_weights)

# Predict on validation set
y_pred_part2 = clf_RF.predict(X_val_part2)

# Calculate accuracy
classification_accuracy = accuracy_score(y_val_part2, y_pred_part2)

# Check if the model meets the minimum requirements
if regression_accuracy >= 0.40 and classification_accuracy >= 0.98:
    print("Your model meets the minimum requirements.")
else:
    print("Your model does not meet the minimum requirements.")

