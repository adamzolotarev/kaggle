import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


main_file_path = '~/.kaggle/competitions/house-prices-advanced-regression-techniques/train.csv'
dataset = pd.read_csv(main_file_path)
dataset = dataset.dropna(axis=1)


# Delete Outliers:
dataset.sort_values(by='GrLivArea', ascending=False)[:2]
# print(dataset.sort_values(by='GrLivArea', ascending=False)[:2])
dataset = dataset.drop(dataset[dataset['Id'] == 1299].index)
dataset = dataset.drop(dataset[dataset['Id'] == 524].index)

y = dataset.SalePrice

predictors = ['OverallQual',
              'GrLivArea',
              'GarageCars',
              'TotalBsmtSF',
              'FullBath',
              'YearBuilt'
              ]

X = dataset[predictors]

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_utilities = LabelEncoder()
utilities_column_index = X.columns.get_loc('OverallQual')
X.loc[:, 'OverallQual'] = labelencoder_utilities.fit_transform(X.OverallQual)

# encoder = LabelEncoder()
# X.loc[:, 'KitchenQual'] = encoder.fit_transform(X.KitchenQual)

# onehotencoder = OneHotEncoder(categorical_features=[utilities_column_index])
# X = onehotencoder.fit_transform(X).toarray()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)


from xgboost import XGBRegressor
# classifier = XGBRegressor()
classifier = XGBRegressor(colsample_bytree=0.7,
                          learning_rate=0.03,
                          max_depth=5,
                          min_child_weight=4,
                          n_estimators=500,
                          nthread=4,
                          objective='reg:linear',
                          silent=1,
                          subsample=0.7)
classifier.fit(X_train, y_train)

# for submitting:
# from xgboost import XGBRegressor
# classifier = XGBRegressor()
# classifier.fit(X, y)
from sklearn.model_selection import GridSearchCV


# brute force scan for all parameters, here are the tricks
# usually max_depth is 6,7,8
# learning rate is around 0.05, but small changes may make big diff
# tuning min_child_weight subsample colsample_bytree can have
# much fun of fighting against overfit
# n_estimators is how many round of boosting
# finally, ensemble xgboost with multiple seeds may reduce variance
parameters = {'nthread': [4],  # when use hyperthread, xgboost may become slower
              'objective': ['reg:linear'],
              'learning_rate': [.03, 0.05, .07],  # so called `eta` value
              'max_depth': [5, 6, 7],
              'min_child_weight': [4],
              'silent': [1],
              'subsample': [0.7],
              'colsample_bytree': [0.7],
              'n_estimators': [500]}

# grid_search = GridSearchCV(classifier,
#                            parameters,
#                            cv=2,
#                            n_jobs=5,
#                            verbose=True)
# grid_search = grid_search.fit(X_train, y_train)
# best_accuracy = grid_search.best_score_
# best_parameters = grid_search.best_params_
# print(best_accuracy)
# print(best_parameters)


from sklearn.metrics import mean_absolute_error
y_pred = classifier.predict(X_test)
print(mean_absolute_error(y_test, y_pred))

# best so far: 22689.104532320205
# with overall quality hot encoded: 21240.258963
# with overall quality hot encoded: 19229.0547945
# with YearRemodAdd: 18921.7343349

# After refactor and removing outliers 18530
# After grid search: 17721.7215994
