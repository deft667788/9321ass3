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
from sklearn.model_selection import cross_val_score

name = os.path.basename(__file__)
id = name.split('.')[0]

parser = argparse.ArgumentParser(description='Part2 Classification')
parser.add_argument('--train_tsv', default="train.tsv", type=str, help='input training tsv file')
args = parser.parse_args(['--train_tsv', sys.argv[1]])

# Reading data
df_train = pd.read_csv(args.train_tsv, sep='\t')

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

# Calculate cross-validation scores
scores = cross_val_score(clf_RF, atm_prepared_train, b_train, cv=5)

# Output cross-validation scores
print("Classification cross-validation scores: ", scores)
