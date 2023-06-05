import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import numpy as np
import pandas as pd
import argparse
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import sys
import os

name = os.path.basename(__file__)
id = name.split('.')[0]

parser = argparse.ArgumentParser(description='Part1 Regression')
parser.add_argument('--train_tsv', default="train.tsv", type=str, help='input training tsv file')
args = parser.parse_args(['--train_tsv', sys.argv[1]])

# Reading data
df_train = pd.read_csv(args.train_tsv, sep='\t')

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

# Random forest
# Selection model
reg_RF = RandomForestRegressor(n_estimators=100, random_state=42)

# Cross-validation
scores = cross_val_score(reg_RF, atm_prepared_train, b_train_part1.ravel(), cv=5)

# Output results
print('Regression cross-validation scores: ', scores)
