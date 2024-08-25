import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer

from laptop_sales_model.config.core import config

from laptop_sales_model.processing.features import categorical_features, categorical_transformer
from laptop_sales_model.processing.features import port_feature, port_transformer
from laptop_sales_model.processing.features import numeric_features, numeric_transformer






# Create a column transformer which combines all of the other transformers 
preprocessor = ColumnTransformer(
    transformers=[
      # (name, transformer_to_use, features_to_use transform)
      ('categorical', categorical_transformer, categorical_features),
      ('port', port_transformer, port_feature),
      ('numerical', numeric_transformer, numeric_features)
])

laptop_sales_pipe = Pipeline(steps=[('preprocessor', preprocessor), # fill our missing data and will make sure it's all number  
    # Regressor
    ('model_rf', RandomForestRegressor(n_estimators = config.model_config.n_estimators, 
                                       max_depth = config.model_config.max_depth,
                                      random_state = config.model_config.random_state))
    
    ])
