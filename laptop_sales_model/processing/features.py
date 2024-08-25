from typing import List
import sys
import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from laptop_sales_model.config.core import config

# Define categorical columns

categorical_features = [config.model_config.make_var, config.model_config.color_var]

# Create categorical transformer (imputes missing values, then encodes them)
categorical_transformer = Pipeline(steps=[
  ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
  ('onehot', OneHotEncoder(handle_unknown='ignore'))                                         
])

# Define port feature
port_feature = [config.model_config.port_var]
# Create port transformer (fills all door missing values with 3)
port_transformer = Pipeline(steps=[
  ('imputer', SimpleImputer(strategy='constant', fill_value=3)),
])

# Define numeric features
numeric_features = [config.model_config.usage_var]
# Create a transformer for filling all missing numeric values with the mean
numeric_transformer = Pipeline(steps=[
  ('imputer', SimpleImputer(strategy='mean'))  
])

