from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor  
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import cross_val_score


housing = pd.read_csv("housing.csv")
housing['income_cat'] = pd.cut(housing['median_income'],
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                                 labels=[1, 2, 3, 4, 5])    


split = StratifiedShuffleSplit(n_splits= 1, random_state=42, test_size=0.2)

for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index].drop("income_cat", axis=1)
    strat_test_set = housing.loc[test_index].drop("income_cat", axis=1)

housing = strat_train_set.copy()
# separete the predictors and the labels

housing_labels = housing["median_house_value"].copy()
housing = housing.drop("median_house_value", axis=1)  # features

print(housing, housing_labels)

housing_num = housing.drop("ocean_proximity", axis=1)

#4. separate the numerical and categorical columns
num_attributes = housing.drop("ocean_proximity", axis=1).columns.tolist()
cat_attributes = ["ocean_proximity"]

#5. create pipelines for numerical 
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('std_scaler', StandardScaler()),
])

#6. create a categorical attributes
cat_pipeline = Pipeline([
    ('one_hot', OneHotEncoder(handle_unknown="ignore")),
])

#7. create a full pipeline 
full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attributes),
    ("cat", cat_pipeline, cat_attributes),
])

#6. tarnsform the data using the full pipeline

housing_prepared = full_pipeline.fit_transform(housing)
print(housing_prepared.shape)

# Train a Linear Regression model

lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)
lin_preds = lin_reg.predict(housing_prepared)
#lin_rmse = root_mean_squared_error(housing_labels, lin_preds)
lin_rmse = -cross_val_score(lin_reg, housing_prepared, housing_labels, scoring="neg_root_mean_squared_error", cv=10)
#print(f"The root mean squared error (RMSE) for Linear Regression: {lin_rmse}")
print(f"results for linear regression:\n{pd.Series(lin_rmse).describe()}")

# Train a Decision Tree Regressor
tree_reg = DecisionTreeRegressor(random_state=42)
tree_reg.fit(housing_prepared, housing_labels)
tree_preds = tree_reg.predict(housing_prepared)
#tree_rmse = root_mean_squared_error(housing_labels, tree_preds)
tree_rmse = -cross_val_score(tree_reg, housing_prepared, housing_labels, scoring="neg_root_mean_squared_error", cv=10)
#print(f"The root mean squared error (RMSE) for Decision Tree Regressor: {tree_rmse}")
print(f"results for decision tree regressor:\n{pd.Series(tree_rmse).describe()}")


# Train a Random Forest Regressor
forest_reg = RandomForestRegressor(random_state=42)
forest_reg.fit(housing_prepared, housing_labels)
forest_preds = forest_reg.predict(housing_prepared)
forest_rmse = -cross_val_score(forest_reg, housing_prepared, housing_labels, scoring="neg_root_mean_squared_error", cv=10)
#forest_rmse = root_mean_squared_error(housing_labels, forest_preds)
#print(f"The root mean squared error (RMSE) for Random Forest Regressor: {forest_rmse}")
print(f"results for random forest regressor:\n{pd.Series(forest_rmse).describe()}")



