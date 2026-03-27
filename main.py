import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import cross_val_score
import joblib

MODEL_FILE = "model.pkl"
PIPELINE_FILE = "pipeline.pkl"

housing = pd.read_csv("housing.csv")
housing['income_cat'] = pd.cut(housing['median_income'],
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                                 labels=[1, 2, 3, 4, 5])    

def build_pipeline(num_attributes, cat_attributes):
    num_attributes = housing.drop("ocean_proximity", axis=1).columns.tolist()
    cat_attributes = ["ocean_proximity"]

    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('std_scaler', StandardScaler()),
    ])

    cat_pipeline = Pipeline([
        ('one_hot', OneHotEncoder(handle_unknown="ignore")),
    ])

    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attributes),
        ("cat", cat_pipeline, cat_attributes),
    ])

    return full_pipeline


if not os.path.exists(MODEL_FILE):
    housing = pd.read_csv("housing.csv")

    housing['income_cat'] = pd.cut(housing['median_income'],
                                   bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                                     labels=[1, 2, 3, 4, 5])    
    
    split = StratifiedShuffleSplit(n_splits= 1, random_state=42, test_size=0.2) 

    for train_index, test_index in split.split(housing, housing["income_cat"]):
       housing.loc[train_index].drop("income_cat", axis=1).to_csv("train_data.csv", index=False)
       housing.loc[test_index].drop("income_cat", axis=1).to_csv("test_data.csv", index=False)
    housing_labels = housing["median_house_value"].copy()
    
    housing_labels = housing["median_house_value"].copy()
    housing_features = housing.drop("median_house_value", axis=1)

    num_attributes = housing.drop("ocean_proximity", axis=1).columns.tolist()
    cat_attributes = ["ocean_proximity"]

    pipeline = build_pipeline(num_attributes, cat_attributes)
    housing_prepared = pipeline.fit_transform(housing)

    print(housing_prepared)

    model = RandomForestRegressor()
    model.fit(housing_prepared, housing_labels)

    joblib.dump(model, MODEL_FILE)
    joblib.dump(pipeline, PIPELINE_FILE)
else:

    print("Model already exists. Loading the model and pipeline...")
    model = joblib.load(MODEL_FILE)
    pipeline = joblib.load(PIPELINE_FILE)

    input_data = pd.read_csv("input_data.csv")  
    input_prepared = pipeline.transform(input_data)
    predictions = model.predict(input_prepared)
    input_data["median_house_value"] = predictions
    input_data.to_csv("predictions_output.csv", index=False)
    print("Results saved to predictions_output.csv")