!pip install pandas scikit-learn httpx
# Importing the necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
# Read the Auto MPG dataset
cols = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight', 'Acceleration', 'Model Year', 'Origin']
df = pd.read_csv('./auto-mpg.data', names=cols, na_values="?", comment='\t', sep=" ", skipinitialspace=True)
data = df.copy()
# Split the dataset using StratifiedShuffleSplit to ensure representative training and test sets
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(data, data["Cylinders"]):
    strat_train_set = data.loc[train_index]
    strat_test_set = data.loc[test_index]
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
# Preprocess the 'Origin' column
def preprocess_origin_cols(df):
    df["Origin"] = df["Origin"].map({1: "India", 2: "USA", 3: "Germany"})
    return df
# Custom attribute adder class
class CustomAttrAdder(BaseEstimator, TransformerMixin):
    def init(self, acc_on_power=True):
        self.acc_on_power = acc_on_power
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        acc_on_cyl = X[:, acc_ix] / X[:, cyl_ix]
        if self.acc_on_power:
            acc_on_power = X[:, acc_ix] / X[:, hpower_ix]
            return np.c_[X, acc_on_power, acc_on_cyl]
        
        return np.c_[X, acc_on_cyl]
# Numeric pipeline transformer
def num_pipeline_transformer(data):
    numerics = ['float64', 'int64']
    num_attrs = data.select_dtypes(include=numerics)
    
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('attrs_adder', CustomAttrAdder()),
        ('std_scaler', StandardScaler()),
    ])
    
    return num_attrs, num_pipeline
# Complete transformation pipeline
def pipeline_transformer(data):
    cat_attrs = ["Origin"]
    num_attrs, num_pipeline = num_pipeline_transformer(data)
    
    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, list(num_attrs)),
        ("cat", OneHotEncoder(), cat_attrs),
    ])
    
    prepared_data = full_pipeline.fit_transform(data)
    return prepared_data
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
# Segregate feature and target variables
data = strat_train_set.drop("MPG", axis=1)
data_labels = strat_train_set["MPG"].copy()
# Preprocess the data
preprocessed_df = preprocess_origin_cols(data)
prepared_data = pipeline_transformer(preprocessed_df)
# Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(prepared_data, data_labels)
# Decision Tree Regression
tree_reg = DecisionTreeRegressor()
tree_reg.fit(prepared_data, data_labels)
# Random Forest Regression
forest_reg = RandomForestRegressor()
forest_reg.fit(prepared_data, data_labels)
# Support Vector Machine (SVM) Regression
svm_reg = SVR(kernel='linear')
svm_reg.fit(prepared_data, data_labels)
import httpx
# Define the API endpoint URL
url = "http://localhost:9696/predict"
# Define vehicle data for prediction
vehicle_data = {
    'Cylinders': 4,
    'Displacement': 155.0,
    'Horsepower': 93.0,
    'Weight': 2500.0,
    'Acceleration': 15.0,
    'Model Year': 81,
    'Origin': 3
}
try:
    # Make a POST request to the API endpoint
    response = httpx.post(url, json=vehicle_data)
    response_json = response.json()
    print(response_json)
except Exception as e:
    print(f"Error: {e}")
