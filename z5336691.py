import numpy as np
import pandas as pd
import argparse
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import sys
import os
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight

name = os.path.basename(__file__)
id = name.split('.')[0]

parser = argparse.ArgumentParser(description='Part1 Regression')
parser.add_argument('--train_tsv', default="train.tsv", type=str, help='input training tsv file')
parser.add_argument('--test_tsv', default="test.tsv", type=str, help='input testing tsv file')
args = parser.parse_args(['--train_tsv', sys.argv[1], '--test_tsv', sys.argv[2]])

# Reading data
df_train = pd.read_csv(args.train_tsv, sep='\t')
df_test = pd.read_csv(args.test_tsv, sep='\t')

# Part I: Regression
# Define the target variable
b_train_part1 = df_train['revenue']

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
atm_prepared_train = full_pipeline.fit_transform(df_train)
atm_prepared_test = full_pipeline.transform(df_test)

# Random forest
# Selection model
reg_RF = RandomForestRegressor(n_estimators=100, random_state=42)

# Training the model
reg_RF_model = reg_RF.fit(atm_prepared_train, b_train_part1.ravel())

# Forecast
y_pred_RF = reg_RF_model.predict(atm_prepared_test)

# Result processing
y_pred_RF1 = np.int64(np.rint(y_pred_RF))
y_pred_pd_RF = pd.Series(y_pred_RF1)

# Output results
y_pred_pd_RF.to_csv('{}.PART1.output.csv'.format(id), sep='\t', index=False, header=['predicted_revenue'])


# Part II: Classification
# Define the target variable
b_train = df_train['rating']

# Data preprocessing
num_pipeline = Pipeline([
    ('std_scaler', StandardScaler()),
])
full_pipeline = ColumnTransformer([
    ("num", num_pipeline,   # Numerical feature preprocessing
    ['Number_of_Shops_Around_ATM', 'No_of_Other_ATMs_in_1_KM_radius', 'Estimated_Number_of_Houses_in_1_KM_Radius',
    'Average_Wait_Time', 'revenue']),
    ("cat", OneHotEncoder(),   # Categorical feature preprocessing
    ['ATM_Zone', 'ATM_Placement', 'ATM_TYPE', 'ATM_looks', 'ATM_Attached_to', 'Day_Type']),
])

# Data transformation
atm_prepared_train = full_pipeline.fit_transform(df_train)
atm_prepared_test = full_pipeline.transform(df_test)

# Calculate the class and sample weights
class_weight = compute_class_weight('balanced',
                                    classes=np.unique(b_train),
                                    y=b_train)
sample_weights = compute_sample_weight('balanced', b_train)

# Create a dictionary of class weights
dict_w = {}
index = 2
for i in class_weight:
    dict_w.update({index: i})
    index += 1

# Random forest
# Selection model
clf_RF = RandomForestClassifier(n_estimators=100, class_weight=dict_w, random_state=42)

# Training the model
r_RF = clf_RF.fit(atm_prepared_train, b_train, sample_weight=sample_weights)

# Forecast
y_pred_RF = r_RF.predict(atm_prepared_test)

# Result processing
y_pred_RF1 = np.int64(np.rint(y_pred_RF))
y_pred_pd_RF = pd.Series(y_pred_RF1)

# Output results
y_pred_pd_RF.to_csv('{}.PART2.output.csv'.format(id), sep='\t', index=False, header=['predicted_rating'])
